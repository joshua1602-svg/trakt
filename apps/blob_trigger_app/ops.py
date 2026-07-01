"""apps.blob_trigger_app.ops — the operator feedback loop (CLI-first).

A managed-service console for closing the loop on halted blob-trigger onboarding:
see WHY a run halted, review the agent's mapping recommendations, approve / reject
/ edit them, promote the approved mapping, and re-fire the same pack. Everything
is JSON in the state store (``trakt-state/runs`` + ``trakt-state/approvals``); a
React UI can wrap the same operations later.

    python -m apps.blob_trigger_app.ops list-halted
    python -m apps.blob_trigger_app.ops show <run_id|pack_key>
    python -m apps.blob_trigger_app.ops show-recommendations <run_id|pack_key>
    python -m apps.blob_trigger_app.ops show-handoff <run_id|pack_key>
    python -m apps.blob_trigger_app.ops show-transform <run_id|pack_key>
    python -m apps.blob_trigger_app.ops show-validation <run_id|pack_key>
    python -m apps.blob_trigger_app.ops show-gates <pack_key>
    python -m apps.blob_trigger_app.ops show-gate <pack_key> <gate_name>
    python -m apps.blob_trigger_app.ops show-llm <run_id|pack_key>
    python -m apps.blob_trigger_app.ops debug-storage [pack_key]
    python -m apps.blob_trigger_app.ops approve <approval_id> --mapping-id <id> \
        --mapping-config-path <path>
    python -m apps.blob_trigger_app.ops reject <approval_id> --reason <why>
    python -m apps.blob_trigger_app.ops edit <approval_id> --set key=value ...
    python -m apps.blob_trigger_app.ops approve-recommendations <pack_key>
    python -m apps.blob_trigger_app.ops promote <approval_id|pack_key>
    python -m apps.blob_trigger_app.ops rerun <pack_key> [--force-publish]

``rerun`` re-fires the SAME pack (force_reprocess). ``--force-publish`` is the
explicit break-glass override that publishes despite validation exceptions — it
is never implied.

The CLI-parity loop for a NEW source (replicates the Codespaces flow of
approving LLM recs and rerunning onboarding before promoting):

    ops approve-recommendations <pack_key>   # accept advised LLM recs → 34_ decisions (persisted, NOT applied)
    ops rerun <pack_key>                     # rerun onboarding APPLYING the accepted decisions
    ops promote <pack_key>                   # persist the mapping active → future packs deterministic (no LLM)
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Any, Callable, Dict, List, Optional

from . import approvals as APP
from . import run_records as RR
from .layout import Layout
from .persistence import ProductionPersistence
from .source_registry import SourceRegistry
from .storage import Storage, open_storage

# A reprocessor re-fires a pack and returns the fresh event manifest.
Reprocessor = Callable[..., Dict[str, Any]]


# --------------------------------------------------------------------------- #
# Read-side operations (list / show)
# --------------------------------------------------------------------------- #

def list_halted(storage: Storage, layout: Layout) -> List[Dict[str, Any]]:
    return RR.list_halted(storage, layout)


def show(storage: Storage, layout: Layout, ref: str) -> Optional[Dict[str, Any]]:
    return RR.resolve(storage, layout, ref)


def recommendations(storage: Storage, layout: Layout, ref: str) -> Dict[str, Any]:
    rec = RR.resolve(storage, layout, ref)
    if rec is None:
        return {}
    return {
        "pack_key": rec.get("pack_key"),
        "run_id": rec.get("run_id"),
        "source_portfolio_id": rec.get("source_portfolio_id"),
        "approval_id": rec.get("approval_id"),
        "mapping_recommendations": rec.get("mapping_recommendations") or [],
        "validation_issues": rec.get("validation_issues") or [],
        "diagnostics": rec.get("diagnostics") or {},
        "next_action": rec.get("next_action") or {},
    }


def handoff(storage: Storage, layout: Layout, ref: str) -> Dict[str, Any]:
    """The full onboarding handoff readiness for a run: which readiness gate
    failed, the blocking decisions / unresolved+missing fields, and the durable
    URIs of the persisted handoff manifest + target coverage matrix."""
    rec = RR.resolve(storage, layout, ref)
    if rec is None:
        return {}
    diag = rec.get("diagnostics") or {}
    hr = rec.get("handoff_readiness") or diag.get("handoff_readiness") or {}
    return {
        "pack_key": rec.get("pack_key"),
        "run_id": rec.get("run_id"),
        "source_portfolio_id": rec.get("source_portfolio_id"),
        "issue_count": rec.get("issue_count", diag.get("issue_count", 0)),
        "handoff_readiness": hr,
        "handoff_artifacts": rec.get("handoff_artifacts") or {},
        "next_action": rec.get("next_action") or {},
    }


def transform(storage: Storage, layout: Layout, ref: str) -> Dict[str, Any]:
    """The Gate 2 transform readiness for a run: readiness flags, the issue tally,
    the first 20 issues (field/type/severity), the affected fields, and the durable
    URIs of the persisted transformation manifest + issues."""
    rec = RR.resolve(storage, layout, ref)
    if rec is None:
        return {}
    diag = rec.get("diagnostics") or {}
    tr = rec.get("transform_readiness") or diag.get("transform_readiness") or {}
    return {
        "pack_key": rec.get("pack_key"),
        "run_id": rec.get("run_id"),
        "source_portfolio_id": rec.get("source_portfolio_id"),
        "transform_readiness": tr,
        "transform_artifacts": rec.get("transform_artifacts") or {},
        "next_action": rec.get("next_action") or {},
    }


def validation(storage: Storage, layout: Layout, ref: str) -> Dict[str, Any]:
    """The Gate 3 validation readiness for a run: mandatory/type/numeric/date
    failures, first 20 issues, ready_for_publish flag, and artefact URIs."""
    rec = RR.resolve(storage, layout, ref)
    if rec is None:
        return {}
    diag = rec.get("diagnostics") or {}
    vr = rec.get("validation_readiness") or diag.get("validation_readiness") or {}
    return {
        "pack_key": rec.get("pack_key"),
        "run_id": rec.get("run_id"),
        "source_portfolio_id": rec.get("source_portfolio_id"),
        "validation_readiness": vr,
        "next_action": rec.get("next_action") or {},
    }


def _pack_key_of(storage: Storage, layout: Layout, ref: str) -> str:
    rec = RR.resolve(storage, layout, ref)
    return (rec or {}).get("pack_key") or ref


def gates(persistence: ProductionPersistence, ref: str) -> Dict[str, Any]:
    """All persisted per-gate diagnostics for a run (falls back to the run
    record's embedded gates)."""
    pk = _pack_key_of(persistence.storage, persistence.layout, ref)
    names = persistence.list_gate_names(pk)
    gates_map = {n: persistence.load_gate_diagnostics(pk, n) for n in names}
    rec = persistence.load_run_record(pk) or {}
    if not gates_map:
        gates_map = {g.get("gate_name"): g for g in (rec.get("gates") or [])}
    if not gates_map and rec == {}:
        return {}
    return {"pack_key": pk, "failed_gate": rec.get("failed_gate"),
            "gate_status": rec.get("gate_status") or {}, "gates": gates_map}


def gate(persistence: ProductionPersistence, pack_key: str,
         gate_name: str) -> Optional[Dict[str, Any]]:
    pk = _pack_key_of(persistence.storage, persistence.layout, pack_key)
    g = persistence.load_gate_diagnostics(pk, gate_name)
    if g is not None:
        return g
    rec = persistence.load_run_record(pk) or {}
    for gg in (rec.get("gates") or []):
        if gg.get("gate_name") == gate_name:
            return gg
    return None


def show_llm(persistence: ProductionPersistence, ref: str) -> Dict[str, Any]:
    """LLM advisory status + persisted recommendations for a run."""
    pk = _pack_key_of(persistence.storage, persistence.layout, ref)
    rec = persistence.load_run_record(pk) or {}
    doc = persistence.load_llm_recommendations(pk)
    if rec == {} and doc is None:
        return {}
    return {"pack_key": pk, "llm": rec.get("llm") or {},
            "recommendations_doc": doc,
            "recommendations": (doc or {}).get("recommendations", [])}


def debug_storage(persistence: ProductionPersistence,
                  pack_key: Optional[str] = None) -> Dict[str, Any]:
    """Report the storage backend selection + (optionally) a pack's run-record and
    gates-folder existence — the first thing to check when nothing persists."""
    from .storage import decide_backend
    storage, layout = persistence.storage, persistence.layout
    d = decide_backend()
    info: Dict[str, Any] = {
        "selected_backend": d["backend"],
        "backend_reason": d["reason"],
        "in_azure": d["in_azure"],
        "TRAKT_BLOB_CONNECTION_present": bool(os.environ.get("TRAKT_BLOB_CONNECTION")),
        "state_container": layout.state_container,
        "processed_container": layout.processed_container,
        "registry_uri": layout.registry_uri,
    }
    if pack_key:
        run_uri = layout.run_uri(pack_key)
        info["pack_key"] = pack_key
        info["run_record_uri"] = run_uri
        info["run_record_exists"] = storage.exists(run_uri)
        info["gates_folder_exists"] = persistence.gates_folder_exists(pack_key)
    return info


# --------------------------------------------------------------------------- #
# Write-side operations (edit / rerun; approve/reject/promote reuse approvals)
# --------------------------------------------------------------------------- #

def edit_approval(storage: Storage, layout: Layout, approval_id: str,
                  updates: Dict[str, Any]) -> Dict[str, Any]:
    """Edit an operator-editable field on a pending/approved approval artifact
    (e.g. correct a suggested mapping before approving)."""
    art = APP.show(storage, layout, approval_id)
    if art is None:
        raise KeyError(f"no such approval: {approval_id}")
    editable = {"suggested_mapping_id", "suggested_mapping_config_path",
                "mapping_id", "mapping_config_path", "source_metadata", "notes"}
    rejected = [k for k in updates if k not in editable]
    if rejected:
        raise ValueError(f"non-editable fields: {rejected}; editable: {sorted(editable)}")
    art.update(updates)
    storage.write_text(layout.approval_uri(approval_id),
                       json.dumps(art, indent=2, default=str))
    return art


def rerun(persistence: ProductionPersistence, registry: SourceRegistry,
          pack_key: str, *, force_publish: bool = False,
          reprocessor: Optional[Reprocessor] = None) -> Dict[str, Any]:
    """Re-fire the same pack with ``force_reprocess`` (and optional break-glass
    ``force_publish``). Records a rerun request, then invokes the reprocessor."""
    record = persistence.load_run_record(pack_key)
    if record is None:
        raise KeyError(f"no run record for pack_key {pack_key!r}")
    meta = {"force_reprocess": True}
    if force_publish:
        meta["force_publish"] = True
    # Apply accepted decisions if the operator ran approve-recommendations — a
    # deterministic-apply rerun, exactly like the CLI's `--target-first-decisions`.
    if persistence.has_approved_decisions(pack_key):
        local = persistence.localise_approved_decisions(
            pack_key, str(persistence.storage._local_path(persistence.layout.runs_prefix())))
        if local:
            meta["applied_decisions_path"] = local
    reproc = reprocessor or _default_reprocessor(persistence, registry)
    manifest = reproc(
        record.get("blob_path"),
        container=record.get("container") or "raw-v2",
        input_dir=record.get("input_dir"),
        marker_metadata=meta)
    return manifest or {}


def approve_recommendations(persistence: ProductionPersistence, pack_key: str, *,
                            approved_by: str = "", min_confidence: float = 0.0) -> Dict[str, Any]:
    """Accept the advised LLM recommendations into an APPROVED decisions file
    (never auto-applied). Records the approved-decisions URI on the run record."""
    summary = persistence.approve_recommendations(
        pack_key, approved_by=approved_by, min_confidence=min_confidence)
    if not summary.get("error"):
        rec = persistence.load_run_record(pack_key)
        if rec is not None:
            rec["approved_decisions_uri"] = summary.get("approved_decisions_uri")
            rec["decisions_accepted"] = {"approved": summary.get("approved"),
                                         "pending": summary.get("pending"),
                                         "skipped": len(summary.get("skipped") or [])}
            RR.write_run_record(persistence.storage, persistence.layout, rec)
    return summary


def promote_pack(persistence: ProductionPersistence, registry: SourceRegistry,
                 pack_key: str) -> "SourceRecord":
    """Promote a pack's accepted decisions into an ACTIVE registry mapping, so
    future monthly packs run deterministically with no LLM. (For approval-artifact
    promotion use ``ops promote <approval_id>``.)"""
    from .source_registry import SourceRecord, STATUS_ACTIVE
    rec = persistence.load_run_record(pack_key)
    if rec is None:
        raise KeyError(f"no run record for pack_key {pack_key!r}")
    approved_uri = rec.get("approved_decisions_uri") or (
        persistence.approved_decisions_uri(pack_key)
        if persistence.has_approved_decisions(pack_key) else None)
    if not approved_uri:
        raise ValueError(f"no accepted decisions to promote for {pack_key} — run "
                         f"approve-recommendations first")
    src = registry.lookup(rec["client_id"], rec["source_portfolio_id"],
                          rec["dataset"], rec["frequency"]) or SourceRecord(
        client_id=rec["client_id"], source_portfolio_id=rec["source_portfolio_id"],
        dataset=rec["dataset"], frequency=rec["frequency"])
    src.source_portfolio_type = rec.get("source_portfolio_id", "").split("_")[0] or src.source_portfolio_type
    src.approved_mapping_id = f"{rec['source_portfolio_id']}_accepted_v{int(getattr(src, 'mapping_version', 0) or 0) + 1}"
    src.mapping_config_path = approved_uri
    src.expected_schema_fingerprint = rec.get("schema_fingerprint") or src.expected_schema_fingerprint
    src.mapping_version = int(getattr(src, "mapping_version", 0) or 0) + 1
    src.status = STATUS_ACTIVE
    registry.upsert(src)
    return src


def _default_reprocessor(persistence: ProductionPersistence,
                         registry: SourceRegistry) -> Reprocessor:
    """Build a reprocessor that re-invokes the router for the stored blob path.

    The router re-fetches/re-fingerprints as configured; in Azure the Function
    layer supplies the downloaded pack. Kept thin + injectable so the CLI stays
    testable without a live orchestrator."""
    from . import router as R

    def _reprocess(blob_path, *, container, input_dir, marker_metadata):
        if not blob_path:
            raise ValueError("run record has no blob_path to re-fire")
        return R.handle_blob_event(
            blob_path, registry=registry,
            out_dir=str(persistence.storage._local_path(persistence.layout.runs_prefix())),
            container=container, input_dir_override=input_dir,
            marker_metadata=marker_metadata, persistence=persistence)

    return _reprocess


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def _context(args) -> "tuple[Storage, Layout, ProductionPersistence]":
    layout = Layout.from_env()
    storage = open_storage(local_root=args.local_root)
    return storage, layout, ProductionPersistence(storage, layout)


def _registry(storage: Storage, layout: Layout, args) -> SourceRegistry:
    return SourceRegistry(args.registry or layout.registry_uri, storage=storage)


def _print_halted(rows: List[Dict[str, Any]]) -> None:
    if not rows:
        print("No halted runs. 🎉")
        return
    print(f"{len(rows)} halted run(s) awaiting an operator:\n")
    for r in rows:
        na = r.get("next_action") or {}
        print(f"● {r.get('status','?').upper():<14} {r.get('pack_key')}")
        print(f"    run_id={r.get('run_id')}  approval_id={r.get('approval_id')}  "
              f"source={r.get('source_portfolio_id')}")
        print(f"    why: {r.get('event_decision')}  →  next: {na.get('action')}")
        # Surface the failure reason inline so a failed/errored run explains itself
        # here — no second command needed.
        reason = (r.get("error")
                  or (r.get("run_summary") or {}).get("central_canonical_unavailable_reason")
                  or na.get("summary"))
        if reason:
            print(f"    reason: {reason}")
        if na.get("command"):
            print(f"    $ {na['command']}")
        print()


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(
        prog="python -m apps.blob_trigger_app.ops",
        description="Trakt operator feedback loop for blob-trigger onboarding.")
    p.add_argument("--local-root", default=None,
                   help="Local root emulating blob containers (local/dev).")
    p.add_argument("--registry", default=None,
                   help="Registry URI for promote/rerun (default: layout.registry_uri).")
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("list-halted", help="List runs awaiting an operator.")
    sp = sub.add_parser("show", help="Show a run's diagnostics + next action.")
    sp.add_argument("ref", help="run_id or pack_key")
    sp = sub.add_parser("show-recommendations",
                        help="Show mapping recommendations + validation issues.")
    sp.add_argument("ref", help="run_id or pack_key")
    sp = sub.add_parser("show-handoff",
                        help="Show the onboarding handoff readiness (which gate "
                             "failed, blocking decisions, missing/unresolved fields).")
    sp.add_argument("ref", help="run_id or pack_key")
    sp = sub.add_parser("show-transform",
                        help="Show the Gate 2 transform readiness (issue count, "
                             "blocking issues, affected fields, first 20 issues).")
    sp.add_argument("ref", help="run_id or pack_key")
    sp = sub.add_parser("show-validation",
                        help="Show the Gate 3 validation readiness (mandatory/type/"
                             "numeric/date failures, first 20 issues).")
    sp.add_argument("ref", help="run_id or pack_key")
    sp = sub.add_parser("show-gates", help="Show all persisted per-gate diagnostics.")
    sp.add_argument("pack_key", help="pack_key (or run_id)")
    sp = sub.add_parser("show-gate", help="Show one gate's persisted diagnostics.")
    sp.add_argument("pack_key", help="pack_key (or run_id)")
    sp.add_argument("gate_name",
                    help="onboarding|transform|validation|stamp|assembler|projection")
    sp = sub.add_parser("show-llm", help="Show LLM advisory status + recommendations.")
    sp.add_argument("ref", help="run_id or pack_key")
    sp = sub.add_parser("debug-storage",
                        help="Report storage backend + (optional) run-record existence.")
    sp.add_argument("pack_key", nargs="?", default=None, help="optional pack_key")
    sp = sub.add_parser("approve", help="Approve a source with a mapping.")
    sp.add_argument("approval_id"); sp.add_argument("--mapping-id", required=True)
    sp.add_argument("--mapping-config-path", default=None); sp.add_argument("--by", default=None)
    sp = sub.add_parser("reject", help="Reject a source.")
    sp.add_argument("approval_id"); sp.add_argument("--reason", required=True)
    sp.add_argument("--by", default=None)
    sp = sub.add_parser("edit", help="Edit an approval's mapping recommendation.")
    sp.add_argument("approval_id")
    sp.add_argument("--set", dest="sets", action="append", default=[],
                    metavar="KEY=VALUE", help="Field to update (repeatable).")
    sp = sub.add_parser("approve-recommendations",
                        help="Accept the advised LLM recommendations into an approved "
                             "34_ decisions file (never auto-applied).")
    sp.add_argument("pack_key")
    sp.add_argument("--min-confidence", type=float, default=0.0)
    sp.add_argument("--by", default=None)
    sp = sub.add_parser("promote", help="Promote an approved mapping to active "
                                        "(<approval_id>, or <pack_key> for accepted decisions).")
    sp.add_argument("ref")
    sp = sub.add_parser("rerun", help="Re-fire the same pack (force_reprocess); applies "
                                      "accepted decisions if approve-recommendations was run.")
    sp.add_argument("pack_key")
    sp.add_argument("--force-publish", action="store_true",
                    help="Break-glass: publish despite validation exceptions.")

    args = p.parse_args(argv)
    storage, layout, persistence = _context(args)

    if args.cmd == "list-halted":
        _print_halted(list_halted(storage, layout))
        return 0

    if args.cmd == "show":
        rec = show(storage, layout, args.ref)
        print(json.dumps(rec, indent=2) if rec else f"No run record for: {args.ref}")
        return 0 if rec else 1

    if args.cmd == "show-recommendations":
        recs = recommendations(storage, layout, args.ref)
        print(json.dumps(recs, indent=2) if recs else f"No run record for: {args.ref}")
        return 0 if recs else 1

    if args.cmd == "show-handoff":
        h = handoff(storage, layout, args.ref)
        print(json.dumps(h, indent=2) if h else f"No run record for: {args.ref}")
        return 0 if h else 1

    if args.cmd == "show-transform":
        t = transform(storage, layout, args.ref)
        print(json.dumps(t, indent=2) if t else f"No run record for: {args.ref}")
        return 0 if t else 1

    if args.cmd == "show-validation":
        v = validation(storage, layout, args.ref)
        print(json.dumps(v, indent=2) if v else f"No run record for: {args.ref}")
        return 0 if v else 1

    if args.cmd == "show-gates":
        g = gates(persistence, args.pack_key)
        print(json.dumps(g, indent=2) if g else f"No run record for: {args.pack_key}")
        return 0 if g else 1

    if args.cmd == "show-gate":
        g = gate(persistence, args.pack_key, args.gate_name)
        print(json.dumps(g, indent=2) if g
              else f"No {args.gate_name!r} gate diagnostics for: {args.pack_key}")
        return 0 if g else 1

    if args.cmd == "show-llm":
        l = show_llm(persistence, args.ref)
        print(json.dumps(l, indent=2) if l else f"No run record for: {args.ref}")
        return 0 if l else 1

    if args.cmd == "debug-storage":
        print(json.dumps(debug_storage(persistence, args.pack_key), indent=2))
        return 0

    if args.cmd == "approve":
        art = APP.approve(storage, layout, args.approval_id, mapping_id=args.mapping_id,
                          mapping_config_path=args.mapping_config_path, decided_by=args.by)
        print(f"approved: {art['approval_id']} (mapping_id={art['mapping_id']})")
        print(f"next: python -m apps.blob_trigger_app.ops promote {art['approval_id']}")
        return 0

    if args.cmd == "reject":
        art = APP.reject(storage, layout, args.approval_id, reason=args.reason, decided_by=args.by)
        print(f"rejected: {art['approval_id']} ({art['reject_reason']})")
        return 0

    if args.cmd == "edit":
        updates: Dict[str, Any] = {}
        for kv in args.sets:
            if "=" not in kv:
                print(f"bad --set {kv!r} (expected KEY=VALUE)"); return 2
            k, v = kv.split("=", 1)
            updates[k.strip()] = v.strip()
        art = edit_approval(storage, layout, args.approval_id, updates)
        print(f"edited: {art['approval_id']} ({', '.join(updates)})")
        return 0

    if args.cmd == "approve-recommendations":
        summary = approve_recommendations(persistence, args.pack_key,
                                          approved_by=args.by or "",
                                          min_confidence=args.min_confidence)
        if summary.get("error"):
            print(f"approve-recommendations: {summary['error']}")
            return 1
        print(f"accepted {summary['approved']} decision(s) "
              f"({summary['pending']} pending, {len(summary.get('skipped') or [])} skipped) → "
              f"{summary.get('approved_decisions_uri')}")
        print(f"next: python -m apps.blob_trigger_app.ops rerun {args.pack_key}")
        print(f"then: python -m apps.blob_trigger_app.ops promote {args.pack_key}")
        return 0

    if args.cmd == "promote":
        registry = _registry(storage, layout, args)
        # <pack_key> → promote accepted decisions; <approval_id> → promote approval.
        if persistence.load_run_record(args.ref) is not None or \
                persistence.has_approved_decisions(args.ref):
            rec = promote_pack(persistence, registry, args.ref)
            print(f"promoted: {rec.key} → active (accepted decisions, "
                  f"mapping_id={rec.approved_mapping_id}, version={rec.mapping_version})")
            print("future monthly packs for this source now run DETERMINISTICALLY (no LLM).")
            return 0
        rec = APP.promote(storage, layout, registry, args.ref)
        print(f"promoted: {rec.key} → active "
              f"(mapping_id={rec.approved_mapping_id}, version={rec.mapping_version})")
        print(f"next: python -m apps.blob_trigger_app.ops rerun "
              f"{rec.client_id}_{rec.source_portfolio_id}_{rec.dataset}_{rec.frequency}_<period>")
        return 0

    if args.cmd == "rerun":
        registry = _registry(storage, layout, args)
        manifest = rerun(persistence, registry, args.pack_key,
                         force_publish=args.force_publish)
        print(f"re-fired {args.pack_key}: status={manifest.get('status')}")
        na = manifest.get("next_action") or {}
        if na.get("command"):
            print(f"next: {na['command']}")
        return 0

    return 2


if __name__ == "__main__":
    raise SystemExit(main())
