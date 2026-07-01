"""apps.blob_trigger_app.ops — the operator feedback loop (CLI-first).

A managed-service console for closing the loop on halted blob-trigger onboarding:
see WHY a run halted, review the agent's mapping recommendations, approve / reject
/ edit them, promote the approved mapping, and re-fire the same pack. Everything
is JSON in the state store (``trakt-state/runs`` + ``trakt-state/approvals``); a
React UI can wrap the same operations later.

    python -m apps.blob_trigger_app.ops list-halted
    python -m apps.blob_trigger_app.ops show <run_id|pack_key>
    python -m apps.blob_trigger_app.ops show-recommendations <run_id|pack_key>
    python -m apps.blob_trigger_app.ops approve <approval_id> --mapping-id <id> \
        --mapping-config-path <path>
    python -m apps.blob_trigger_app.ops reject <approval_id> --reason <why>
    python -m apps.blob_trigger_app.ops edit <approval_id> --set key=value ...
    python -m apps.blob_trigger_app.ops promote <approval_id>
    python -m apps.blob_trigger_app.ops rerun <pack_key> [--force-publish]

``rerun`` re-fires the SAME pack (force_reprocess). ``--force-publish`` is the
explicit break-glass override that publishes despite validation exceptions — it
is never implied.
"""

from __future__ import annotations

import argparse
import json
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
    reproc = reprocessor or _default_reprocessor(persistence, registry)
    manifest = reproc(
        record.get("blob_path"),
        container=record.get("container") or "raw-v2",
        input_dir=record.get("input_dir"),
        marker_metadata=meta)
    return manifest or {}


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
    sp = sub.add_parser("promote", help="Promote an approved mapping to active.")
    sp.add_argument("approval_id")
    sp = sub.add_parser("rerun", help="Re-fire the same pack (force_reprocess).")
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

    if args.cmd == "promote":
        registry = _registry(storage, layout, args)
        rec = APP.promote(storage, layout, registry, args.approval_id)
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
