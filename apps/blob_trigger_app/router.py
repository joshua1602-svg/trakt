"""apps.blob_trigger_app.router — the trigger's decision core (no Azure deps).

Pure, testable routing: parse path → fingerprint → registry inference →
source-onboarding vs deterministic decision → invoke the Orchestrator Agent →
write an event manifest. No business logic beyond routing/inference lives here.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

from .event_log import make_event_id, write_event_manifest
from .orchestrator_invoke import OrchestratorInvoker, default_orchestrator_invoker
from .path_parser import ParsedPath, PathParseError, parse_blob_path
from .schema_fingerprint import SchemaInfo, compute_schema_fingerprint
from .source_registry import (
    STATUS_ACTIVE, STATUS_PENDING_REVIEW, SourceRecord, SourceRegistry,
)
from .target_selection import select_target

# Final event statuses.
STATUS_PROCESSED = "processed"
STATUS_HALTED = "halted"
STATUS_PENDING_REVIEW = "pending_review"
STATUS_FAILED = "failed"

# Routing decisions.
DECISION_DETERMINISTIC = "deterministic"
DECISION_SOURCE_ONBOARDING = "source_onboarding"
DECISION_SCHEMA_DRIFT = "schema_drift"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def handle_blob_event(
    blob_path: str,
    *,
    registry: SourceRegistry,
    out_dir: str | Path,
    local_input_path: Optional[str] = None,
    schema_info: Optional[SchemaInfo] = None,
    orchestrator_invoker: OrchestratorInvoker = default_orchestrator_invoker,
    now: Optional[str] = None,
    run_id_for_registry: Optional[str] = None,
) -> Dict[str, Any]:
    """Route one uploaded blob and return its event manifest (also written to
    ``out_dir``). Fails closed on unparseable paths and un-fingerprintable files.
    """
    created_at = now or _now()
    manifest: Dict[str, Any] = {
        "event_id": make_event_id(blob_path, created_at),
        "blob_path": blob_path,
        "client_id": None, "dataset": None, "frequency": None,
        "source_portfolio_id": None, "reporting_period": None,
        "schema_fingerprint": None,
        "registry_match": False,
        "requires_source_onboarding": None,
        "selected_target": None,
        "orchestrator_invocation": None,
        "status": None,
        "error": None,
        "created_at": created_at,
    }

    # 1) Parse path (fail closed) -----------------------------------------
    try:
        parsed: ParsedPath = parse_blob_path(blob_path)
    except PathParseError as exc:
        manifest["status"] = STATUS_FAILED
        manifest["error"] = f"path_parse_error: {exc}"
        write_event_manifest(manifest, out_dir)
        return manifest
    manifest.update(
        client_id=parsed.client_id, dataset=parsed.dataset,
        frequency=parsed.frequency, source_portfolio_id=parsed.source_portfolio_id,
        reporting_period=parsed.reporting_period)

    # 2) Schema fingerprint (fail closed) ---------------------------------
    try:
        if schema_info is None:
            if not local_input_path:
                raise ValueError("no local_input_path to fingerprint")
            schema_info = compute_schema_fingerprint(local_input_path)
    except Exception as exc:  # noqa: BLE001
        manifest["status"] = STATUS_FAILED
        manifest["error"] = f"fingerprint_error: {exc}"
        write_event_manifest(manifest, out_dir)
        return manifest
    manifest["schema_fingerprint"] = schema_info.fingerprint

    # 3) Registry inference ------------------------------------------------
    rec = registry.lookup(parsed.client_id, parsed.source_portfolio_id,
                          parsed.dataset, parsed.frequency)
    manifest["registry_match"] = rec is not None
    regime_required = bool(rec.regime_required) if rec else False
    sel = select_target(parsed.dataset, parsed.frequency, regime_required=regime_required)
    manifest["selected_target"] = {"target": sel.target, "run_regime": sel.run_regime,
                                   "reason": sel.reason}

    if rec is None or not rec.has_approved_mapping:
        decision = DECISION_SOURCE_ONBOARDING
    elif rec.expected_schema_fingerprint == schema_info.fingerprint:
        decision = DECISION_DETERMINISTIC
    else:
        decision = DECISION_SCHEMA_DRIFT
    manifest["requires_source_onboarding"] = decision != DECISION_DETERMINISTIC
    manifest["decision"] = decision

    input_for_orch = (str(Path(local_input_path).parent)
                      if local_input_path else parsed.blob_path)

    # 4) Route -------------------------------------------------------------
    if decision == DECISION_SCHEMA_DRIFT:
        # Fail closed — never process with a stale mapping.
        manifest["status"] = STATUS_PENDING_REVIEW
        manifest["error"] = (
            f"schema_drift: incoming fingerprint {schema_info.fingerprint} != "
            f"saved {rec.expected_schema_fingerprint}")
        manifest["orchestrator_invocation"] = {
            "invoked": False, "reason": "schema_drift_fail_closed"}
        rec.status = STATUS_PENDING_REVIEW
        registry.upsert(rec)
        write_event_manifest(manifest, out_dir)
        return manifest

    if decision == DECISION_SOURCE_ONBOARDING:
        result = orchestrator_invoker(
            processing_mode=DECISION_SOURCE_ONBOARDING,
            client_id=parsed.client_id, source_portfolio_id=parsed.source_portfolio_id,
            source_portfolio_type=_derive_type(parsed.source_portfolio_id),
            dataset=parsed.dataset, frequency=parsed.frequency,
            reporting_period=parsed.reporting_period, input_path=input_for_orch,
            target=sel.target, run_regime=sel.run_regime,
            mapping_config_path=None, out_dir=str(out_dir))
        # Approval is human-gated: a new/changed source stops at pending_review.
        manifest["orchestrator_invocation"] = {
            "invoked": True, "mode": DECISION_SOURCE_ONBOARDING,
            "target": sel.target, "run_regime": sel.run_regime, **_inv(result)}
        manifest["status"] = STATUS_PENDING_REVIEW
        _upsert_source(registry, parsed, schema_info, status=STATUS_PENDING_REVIEW,
                       regime_required=regime_required)
        write_event_manifest(manifest, out_dir)
        return manifest

    # decision == deterministic ------------------------------------------
    result = orchestrator_invoker(
        processing_mode=DECISION_DETERMINISTIC,
        client_id=parsed.client_id, source_portfolio_id=parsed.source_portfolio_id,
        source_portfolio_type=rec.source_portfolio_type,
        dataset=parsed.dataset, frequency=parsed.frequency,
        reporting_period=parsed.reporting_period, input_path=input_for_orch,
        target=sel.target, run_regime=sel.run_regime,
        mapping_config_path=rec.mapping_config_path, out_dir=str(out_dir))
    manifest["orchestrator_invocation"] = {
        "invoked": True, "mode": DECISION_DETERMINISTIC,
        "target": sel.target, "run_regime": sel.run_regime, **_inv(result)}
    orch_status = (result or {}).get("status")
    if orch_status == "done":
        manifest["status"] = STATUS_PROCESSED
        rec.last_successful_run_id = (result.get("run_id") or run_id_for_registry)
        rec.last_successful_reporting_period = parsed.reporting_period
        registry.upsert(rec)
    elif orch_status == "halted":
        manifest["status"] = STATUS_HALTED
    else:
        manifest["status"] = STATUS_FAILED
        manifest["error"] = "; ".join((result or {}).get("blockers") or []) or "orchestrator_failed"
    write_event_manifest(manifest, out_dir)
    return manifest


def _inv(result: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    r = result or {}
    return {"run_id": r.get("run_id"), "orchestrator_status": r.get("status"),
            "central_canonical_path": r.get("central_canonical_path")}


def _derive_type(source_portfolio_id: str) -> Optional[str]:
    from engine.provenance import derive_portfolio_type
    return derive_portfolio_type(source_portfolio_id)


def _upsert_source(registry: SourceRegistry, parsed: ParsedPath, schema: SchemaInfo,
                   *, status: str, regime_required: bool) -> None:
    existing = registry.lookup(parsed.client_id, parsed.source_portfolio_id,
                               parsed.dataset, parsed.frequency)
    rec = existing or SourceRecord(
        client_id=parsed.client_id, source_portfolio_id=parsed.source_portfolio_id,
        dataset=parsed.dataset, frequency=parsed.frequency,
        source_portfolio_type=_derive_type(parsed.source_portfolio_id))
    rec.status = status
    rec.regime_required = regime_required
    # Record the observed schema so the eventual approval can confirm it.
    rec.expected_schema_fingerprint = rec.expected_schema_fingerprint or schema.fingerprint
    rec.expected_columns = rec.expected_columns or schema.columns
    registry.upsert(rec)
