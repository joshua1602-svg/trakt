"""apps.blob_trigger_app.run_records — durable operator-facing run ledger.

One JSON record per reporting pack (keyed on ``pack_key``) in the state store,
carrying everything an operator needs to close a halted run WITHOUT reading logs:

    run_id, pack_key, blob_path, client_id, source_portfolio_id, dataset,
    frequency, reporting_period, status, decision, event_decision, approval_id,
    diagnostics (halt stage/reason, blocking decisions, registry gaps),
    validation_issues, mapping_recommendations, next_action (approve/promote/
    rerun/fix_data_supply + exact command), promoted_mapping, created_at.

This is the query surface for ``ops list-halted`` / ``ops show`` — a small,
inspectable ledger that migrates to a table/Cosmos later with the same shape.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from .layout import Layout
from .ops_advice import next_operator_action
from .storage import Storage

# Statuses that represent a terminal, operator-relevant outcome (i.e. worth a
# durable run record). Transient acks (awaiting_pack, already_processed) are not.
OPERATOR_STATUSES = ("halted", "failed", "pending_review", "processed")
#: The subset an operator still has to act on (drives ``list-halted``).
ACTIONABLE_STATUSES = ("halted", "failed", "pending_review")


def build_run_record(manifest: Dict[str, Any]) -> Dict[str, Any]:
    """Project a terminal event manifest into an operator run record."""
    diag = manifest.get("orchestrator_diagnostics") or {}
    next_action = manifest.get("next_action") or next_operator_action(manifest)
    return {
        "pack_key": manifest.get("pack_key"),
        "run_id": manifest.get("orchestrator_run_id"),
        "event_id": manifest.get("event_id"),
        "blob_path": manifest.get("blob_path"),
        "input_dir": manifest.get("input_dir"),
        "container": manifest.get("container"),
        "client_id": manifest.get("client_id"),
        "source_portfolio_id": manifest.get("source_portfolio_id"),
        "dataset": manifest.get("dataset"),
        "frequency": manifest.get("frequency"),
        "reporting_period": manifest.get("reporting_period"),
        "status": manifest.get("status"),
        "decision": manifest.get("decision"),
        "event_decision": manifest.get("event_decision"),
        "target": manifest.get("target"),
        "approval_id": manifest.get("approval_id"),
        "central_canonical_path": manifest.get("central_canonical_path"),
        "diagnostics": diag,
        "validation_issues": diag.get("validation_errors") or [],
        "mapping_recommendations": diag.get("mapping_recommendations") or [],
        "handoff_readiness": diag.get("handoff_readiness") or {},
        "transform_readiness": diag.get("transform_readiness") or {},
        "validation_readiness": diag.get("validation_readiness") or {},
        # Generic gate observability + run-level summary.
        "gates": diag.get("gates") or [],
        "run_summary": diag.get("run_summary") or {},
        "failed_gate": (diag.get("run_summary") or {}).get("failed_gate"),
        "gate_status": (diag.get("run_summary") or {}).get("gate_status") or {},
        "central_canonical_unavailable_reason": (
            (diag.get("run_summary") or {}).get("central_canonical_unavailable_reason")),
        "issue_count": diag.get("issue_count", 0),
        # LLM advisory status (deterministic remains the source of truth).
        "llm": manifest.get("llm") or {},
        "next_action": next_action,
        "error": manifest.get("error"),
        "created_at": manifest.get("created_at"),
    }


def write_run_record(storage: Storage, layout: Layout,
                     record: Dict[str, Any]) -> Optional[str]:
    pack_key = record.get("pack_key")
    if not pack_key:
        return None
    uri = layout.run_uri(pack_key)
    storage.write_text(uri, json.dumps(record, indent=2, default=str))
    return uri


def persist_from_manifest(storage: Storage, layout: Layout,
                          manifest: Dict[str, Any]) -> Optional[str]:
    """Build + write a run record from a manifest, if the outcome warrants one."""
    if not manifest.get("is_pack_marker"):
        return None
    if manifest.get("status") not in OPERATOR_STATUSES:
        return None
    return write_run_record(storage, layout, build_run_record(manifest))


def load_run_record(storage: Storage, layout: Layout,
                    pack_key: str) -> Optional[Dict[str, Any]]:
    uri = layout.run_uri(pack_key)
    if not storage.exists(uri):
        return None
    try:
        return json.loads(storage.read_text(uri))
    except Exception:  # noqa: BLE001
        return None


def list_run_records(storage: Storage, layout: Layout,
                     *, statuses: Optional[List[str]] = None) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for uri in storage.list(layout.runs_prefix()):
        if not uri.endswith(".json"):
            continue
        try:
            rec = json.loads(storage.read_text(uri))
        except Exception:  # noqa: BLE001
            continue
        if statuses is None or rec.get("status") in statuses:
            out.append(rec)
    return sorted(out, key=lambda r: (r.get("created_at") or "", r.get("pack_key") or ""))


def list_halted(storage: Storage, layout: Layout) -> List[Dict[str, Any]]:
    """Runs still awaiting an operator (halted / failed / pending_review)."""
    return list_run_records(storage, layout, statuses=list(ACTIONABLE_STATUSES))


def find_by_run_id(storage: Storage, layout: Layout,
                   run_id: str) -> Optional[Dict[str, Any]]:
    for rec in list_run_records(storage, layout):
        if rec.get("run_id") == run_id:
            return rec
    return None


def resolve(storage: Storage, layout: Layout,
            ref: str) -> Optional[Dict[str, Any]]:
    """Resolve a record by run_id first, then by pack_key (``show`` takes either)."""
    return find_by_run_id(storage, layout, ref) or load_run_record(storage, layout, ref)
