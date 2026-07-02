"""apps.blob_trigger_app.approvals — file/Blob-based source approval workflow.

New sources and schema changes are **human-gated**. The router writes a pending
approval artifact; an operator lists/inspects it and approves (with the mapping
to use) or rejects. Approval **promotes** the source to an ``active`` registry
entry so subsequent packs route deterministically. No DB, no UI — JSON artifacts
in the state store, driven by a small CLI.

Artifact: ``{approvals_prefix}/{approval_id}.json``
    approval_id, kind (new_source|schema_drift), status
    (pending|approved|rejected|promoted), client_id, source_book_type, dataset,
    frequency, source_portfolio_id, period, schema_fingerprint,
    prior_schema_fingerprint (drift), detected_files, suggested_mapping_id,
    suggested_mapping_config_path, source_metadata, created_at, decided_at,
    decided_by, mapping_id, mapping_config_path, reject_reason.
"""

from __future__ import annotations

import argparse
import json
import re
from typing import Any, Dict, List, Optional

from .layout import Layout
from .source_registry import STATUS_ACTIVE, SourceRecord, SourceRegistry
from .storage import Storage, open_storage

KIND_NEW_SOURCE = "new_source"
KIND_SCHEMA_DRIFT = "schema_drift"

STATUS_PENDING = "pending"
STATUS_APPROVED = "approved"
STATUS_REJECTED = "rejected"
STATUS_PROMOTED = "promoted"


def make_approval_id(client_id: str, source_portfolio_id: str, dataset: str,
                     frequency: str, period: str, fingerprint: str) -> str:
    fp = (fingerprint or "").replace("sha256:", "")[:10]
    raw = f"{client_id}_{source_portfolio_id}_{dataset}_{frequency}_{period}_{fp}"
    return re.sub(r"[^A-Za-z0-9._-]+", "_", raw)


def write_pending(
    storage: Storage, layout: Layout, *,
    kind: str, client_id: str, source_book_type: Optional[str], dataset: str,
    frequency: str, source_portfolio_id: str, period: str,
    schema_fingerprint: str, detected_files: Optional[List[str]] = None,
    suggested_mapping_id: Optional[str] = None,
    suggested_mapping_config_path: Optional[str] = None,
    prior_schema_fingerprint: Optional[str] = None,
    source_metadata: Optional[Dict[str, Any]] = None,
    role_schemas: Optional[Dict[str, List[str]]] = None,
    role_aliases: Optional[Dict[str, List[str]]] = None,
    created_at: Optional[str] = None,
) -> Dict[str, Any]:
    """Write (idempotently, keyed on the fingerprint) a pending approval artifact."""
    approval_id = make_approval_id(client_id, source_portfolio_id, dataset,
                                   frequency, period, schema_fingerprint)
    artifact: Dict[str, Any] = {
        "approval_id": approval_id,
        "kind": kind,
        "status": STATUS_PENDING,
        "client_id": client_id,
        "source_book_type": source_book_type,
        "dataset": dataset,
        "frequency": frequency,
        "source_portfolio_id": source_portfolio_id,
        "period": period,
        "schema_fingerprint": schema_fingerprint,
        "prior_schema_fingerprint": prior_schema_fingerprint,
        # Approved header-first role signatures (role -> columns) + filename alias
        # fallbacks, captured from THIS pack's classification and promoted so future
        # months are recognised by header regardless of filename.
        "role_schemas": dict(role_schemas or {}),
        "role_aliases": dict(role_aliases or {}),
        "detected_files": list(detected_files or []),
        "suggested_mapping_id": suggested_mapping_id,
        "suggested_mapping_config_path": suggested_mapping_config_path,
        "source_metadata": dict(source_metadata or {}),
        "created_at": created_at,
        "decided_at": None,
        "decided_by": None,
        "mapping_id": None,
        "mapping_config_path": None,
        "reject_reason": None,
    }
    storage.write_text(layout.approval_uri(approval_id),
                       json.dumps(artifact, indent=2, default=str))
    return artifact


def list_pending(storage: Storage, layout: Layout,
                 *, include_decided: bool = False) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for uri in storage.list(layout.approvals_prefix()):
        if not uri.endswith(".json"):
            continue
        try:
            art = json.loads(storage.read_text(uri))
        except Exception:  # noqa: BLE001
            continue
        if include_decided or art.get("status") == STATUS_PENDING:
            out.append(art)
    return sorted(out, key=lambda a: a.get("approval_id", ""))


def show(storage: Storage, layout: Layout, approval_id: str) -> Optional[Dict[str, Any]]:
    uri = layout.approval_uri(approval_id)
    if not storage.exists(uri):
        return None
    return json.loads(storage.read_text(uri))


def _save(storage: Storage, layout: Layout, art: Dict[str, Any]) -> Dict[str, Any]:
    storage.write_text(layout.approval_uri(art["approval_id"]),
                       json.dumps(art, indent=2, default=str))
    return art


def approve(storage: Storage, layout: Layout, approval_id: str, *,
            mapping_id: str, mapping_config_path: Optional[str] = None,
            decided_by: Optional[str] = None,
            decided_at: Optional[str] = None) -> Dict[str, Any]:
    art = show(storage, layout, approval_id)
    if art is None:
        raise KeyError(f"no such approval: {approval_id}")
    art.update(status=STATUS_APPROVED, mapping_id=mapping_id,
               mapping_config_path=mapping_config_path,
               decided_by=decided_by, decided_at=decided_at)
    return _save(storage, layout, art)


def reject(storage: Storage, layout: Layout, approval_id: str, *,
           reason: str, decided_by: Optional[str] = None,
           decided_at: Optional[str] = None) -> Dict[str, Any]:
    art = show(storage, layout, approval_id)
    if art is None:
        raise KeyError(f"no such approval: {approval_id}")
    art.update(status=STATUS_REJECTED, reject_reason=reason,
               decided_by=decided_by, decided_at=decided_at)
    return _save(storage, layout, art)


def promote(storage: Storage, layout: Layout, registry: SourceRegistry,
            approval_id: str) -> SourceRecord:
    """Promote an approved approval into an ``active`` source registry entry."""
    art = show(storage, layout, approval_id)
    if art is None:
        raise KeyError(f"no such approval: {approval_id}")
    if art.get("status") != STATUS_APPROVED:
        raise ValueError(
            f"approval {approval_id} is {art.get('status')!r}, must be 'approved' to promote")
    meta = art.get("source_metadata") or {}
    rec = registry.lookup(art["client_id"], art["source_portfolio_id"],
                          art["dataset"], art["frequency"]) or SourceRecord(
        client_id=art["client_id"], source_portfolio_id=art["source_portfolio_id"],
        dataset=art["dataset"], frequency=art["frequency"])
    rec.source_portfolio_type = (art.get("source_book_type")
                                 or meta.get("source_portfolio_type")
                                 or rec.source_portfolio_type)
    rec.approved_mapping_id = art.get("mapping_id") or art.get("suggested_mapping_id")
    rec.mapping_config_path = (art.get("mapping_config_path")
                               or art.get("suggested_mapping_config_path"))
    rec.expected_schema_fingerprint = art["schema_fingerprint"]
    # Persist the approved role -> header signature (and alias) mapping so future
    # months are classified header-first, regardless of filename.
    if art.get("role_schemas"):
        rec.file_role_schemas = dict(art["role_schemas"])
    if art.get("role_aliases"):
        rec.file_role_aliases = dict(art["role_aliases"])
    rec.regime_required = bool(meta.get("regime_required", rec.regime_required))
    rec.mapping_version = int(getattr(rec, "mapping_version", 0) or 0) + 1
    rec.status = STATUS_ACTIVE
    registry.upsert(rec)
    art["status"] = STATUS_PROMOTED
    art["promoted_mapping"] = {"mapping_id": rec.approved_mapping_id,
                               "mapping_config_path": rec.mapping_config_path,
                               "mapping_version": rec.mapping_version,
                               "expected_schema_fingerprint": rec.expected_schema_fingerprint}
    _save(storage, layout, art)
    return rec


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def _ctx(args) -> "tuple[Storage, Layout]":
    layout = Layout.from_env()
    storage = open_storage(local_root=args.local_root)
    return storage, layout


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(
        prog="python -m apps.blob_trigger_app.approvals",
        description="Trakt source approval workflow (file/Blob based).")
    p.add_argument("--local-root", default=None,
                   help="Local root that emulates blob containers (local/dev).")
    p.add_argument("--registry", default=None,
                   help="Registry path/URI for 'promote' (default: layout.registry_uri).")
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("list", help="List pending approvals.")
    sp = sub.add_parser("show", help="Show one approval."); sp.add_argument("approval_id")
    sp = sub.add_parser("approve", help="Approve a source with a mapping.")
    sp.add_argument("approval_id"); sp.add_argument("--mapping-id", required=True)
    sp.add_argument("--mapping-config-path", default=None); sp.add_argument("--by", default=None)
    sp = sub.add_parser("reject", help="Reject a source.")
    sp.add_argument("approval_id"); sp.add_argument("--reason", required=True)
    sp.add_argument("--by", default=None)
    sp = sub.add_parser("promote", help="Promote an approved source to active registry.")
    sp.add_argument("approval_id")

    args = p.parse_args(argv)
    storage, layout = _ctx(args)

    if args.cmd == "list":
        rows = list_pending(storage, layout)
        if not rows:
            print("No pending approvals.")
        for a in rows:
            print(f"{a['approval_id']}  [{a['kind']}]  {a['client_id']}/"
                  f"{a['source_portfolio_id']}/{a['dataset']}/{a['frequency']}/{a['period']}"
                  f"  fp={a['schema_fingerprint']}")
        return 0
    if args.cmd == "show":
        art = show(storage, layout, args.approval_id)
        print(json.dumps(art, indent=2) if art else f"No such approval: {args.approval_id}")
        return 0 if art else 1
    if args.cmd == "approve":
        art = approve(storage, layout, args.approval_id, mapping_id=args.mapping_id,
                      mapping_config_path=args.mapping_config_path, decided_by=args.by)
        print(f"approved: {art['approval_id']} (mapping_id={art['mapping_id']})")
        return 0
    if args.cmd == "reject":
        art = reject(storage, layout, args.approval_id, reason=args.reason, decided_by=args.by)
        print(f"rejected: {art['approval_id']} ({art['reject_reason']})")
        return 0
    if args.cmd == "promote":
        reg_uri = args.registry or layout.registry_uri
        registry = SourceRegistry(reg_uri, storage=storage)
        rec = promote(storage, layout, registry, args.approval_id)
        print(f"promoted: {rec.key} → active (mapping_id={rec.approved_mapping_id})")
        return 0
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
