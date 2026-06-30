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
STATUS_AWAITING_PACK = "awaiting_pack"        # a data file arrived; waiting for the READY marker
STATUS_ALREADY_PROCESSED = "already_processed"  # idempotency: this pack already ran

DEFAULT_PACK_MARKER = "_READY"

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
    container: str = "raw",
    pack_marker: str = DEFAULT_PACK_MARKER,
    local_input_path: Optional[str] = None,
    input_dir_override: Optional[str] = None,
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
        parsed: ParsedPath = parse_blob_path(blob_path, container)
    except PathParseError as exc:
        manifest["status"] = STATUS_FAILED
        manifest["error"] = f"path_parse_error: {exc}"
        write_event_manifest(manifest, out_dir)
        return manifest
    manifest.update(
        client_id=parsed.client_id, dataset=parsed.dataset,
        frequency=parsed.frequency, source_portfolio_id=parsed.source_portfolio_id,
        reporting_period=parsed.reporting_period)
    manifest["pack_key"] = _pack_key(parsed)
    manifest["is_pack_marker"] = parsed.filename == pack_marker

    # 1b) Completion gate (READY sentinel) --------------------------------
    # Only the marker file starts processing. A data-file upload is acknowledged
    # and logged as a pack member, but never starts the Orchestrator — the pack
    # runs once, when the uploader writes the marker last.
    if parsed.filename != pack_marker:
        manifest["status"] = STATUS_AWAITING_PACK
        manifest["decision"] = "pack_member"
        manifest["orchestrator_invocation"] = {
            "invoked": False, "reason": f"awaiting completion marker {pack_marker!r}"}
        write_event_manifest(manifest, out_dir)
        return manifest

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

    # 2b) Idempotency — a pack that already ran this exact schema is skipped,
    # so a duplicate/re-fired marker event never double-runs the Orchestrator.
    prior = _read_processed(out_dir, manifest["pack_key"])
    if prior and prior.get("schema_fingerprint") == schema_info.fingerprint:
        manifest["status"] = STATUS_ALREADY_PROCESSED
        manifest["decision"] = "idempotent_skip"
        manifest["orchestrator_invocation"] = {
            "invoked": False, "reason": "pack already processed",
            "prior_run_id": prior.get("run_id"), "prior_status": prior.get("status")}
        write_event_manifest(manifest, out_dir)
        return manifest

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

    input_for_orch = (input_dir_override
                      or (str(Path(local_input_path).parent) if local_input_path
                          else parsed.blob_path))

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
        _write_processed(out_dir, manifest, schema_info, result)
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
    # Mark the pack processed (idempotency) on any orchestrator-invoking outcome.
    _write_processed(out_dir, manifest, schema_info, result)
    write_event_manifest(manifest, out_dir)
    return manifest


# --------------------------------------------------------------------------- #
# Pack idempotency helpers (folder-level, keyed on the reporting pack)
# --------------------------------------------------------------------------- #

def _pack_key(parsed: ParsedPath) -> str:
    import re
    raw = "/".join([parsed.client_id, parsed.source_portfolio_id,
                    parsed.dataset, parsed.frequency, parsed.reporting_period])
    return re.sub(r"[^A-Za-z0-9._-]+", "_", raw)


def _processed_path(out_dir: str | Path, pack_key: str) -> Path:
    return Path(out_dir) / "_packs" / f"{pack_key}.json"


def _read_processed(out_dir: str | Path, pack_key: str) -> Optional[Dict[str, Any]]:
    import json
    p = _processed_path(out_dir, pack_key)
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return None


def _write_processed(out_dir: str | Path, manifest: Dict[str, Any],
                     schema: SchemaInfo, result: Optional[Dict[str, Any]]) -> None:
    import json
    p = _processed_path(out_dir, manifest["pack_key"])
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({
        "pack_key": manifest["pack_key"],
        "schema_fingerprint": schema.fingerprint,
        "status": manifest["status"],
        "event_id": manifest["event_id"],
        "run_id": (result or {}).get("run_id"),
        "created_at": manifest["created_at"],
    }, indent=2), encoding="utf-8")


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
