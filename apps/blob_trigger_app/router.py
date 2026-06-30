"""apps.blob_trigger_app.router — the trigger's decision core (no Azure deps).

Pure, testable routing: parse path → completion gate → pack fingerprint →
registry inference → source-onboarding vs deterministic decision → invoke the
Orchestrator Agent → refresh the central platform canonical via the Assembler
Agent → write an event manifest. No business logic beyond routing/inference
lives here; the Azure layer (root ``function_app.py`` Event Grid handler) only
fetches blobs and calls in here.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from .assembler_refresh import AssemblerRefresher, default_assembler_refresher
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
STATUS_AWAITING_PACK = "awaiting_pack"          # a data file arrived; waiting for the marker
STATUS_ALREADY_PROCESSED = "already_processed"  # idempotency: this pack already ran

# Completion marker (Option A). Production uploads ``_READY.json`` last; its
# JSON body may carry expected_files / target / regime_required / seller_name /
# acquisition_date / force_reprocess.
DEFAULT_PACK_MARKER = "_READY.json"

# Routing decisions (back-compatible: existing values unchanged).
DECISION_DETERMINISTIC = "deterministic"
DECISION_SOURCE_ONBOARDING = "source_onboarding"
DECISION_SCHEMA_DRIFT = "schema_drift"
DECISION_PACK_MEMBER = "pack_member"
DECISION_IDEMPOTENT_SKIP = "idempotent_skip"
DECISION_INCOMPLETE_PACK = "incomplete_pack"

# Audit decision vocabulary (manifest ``event_decision`` — the canonical audit
# label requested by the spec; ``decision`` above stays for back-compat).
EVT_IGNORED_DATA_FILE = "ignored_data_file_waiting_for_ready"
EVT_INVALID_PATH = "invalid_path"
EVT_NEW_SOURCE_PENDING = "new_source_pending_review"
EVT_KNOWN_SOURCE_PROCESSED = "known_source_processed"
EVT_KNOWN_SOURCE_HALTED = "known_source_halted"
EVT_SCHEMA_DRIFT_PENDING = "schema_drift_pending_review"
EVT_DUPLICATE_READY_IGNORED = "duplicate_ready_ignored"
EVT_INCOMPLETE_PACK_PENDING = "incomplete_pack_pending_review"
EVT_FAILED = "failed"


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
    marker_metadata: Optional[Dict[str, Any]] = None,
    pack_files: Optional[List[str]] = None,
    orchestrator_invoker: OrchestratorInvoker = default_orchestrator_invoker,
    assembler_refresher: AssemblerRefresher = default_assembler_refresher,
    accepted_root: Optional[str] = None,
    platform_out_dir: Optional[str] = None,
    now: Optional[str] = None,
    run_id_for_registry: Optional[str] = None,
) -> Dict[str, Any]:
    """Route one uploaded blob and return its event manifest (also written to
    ``out_dir``). Fails closed on unparseable paths and un-fingerprintable files.

    ``marker_metadata`` is the parsed ``_READY.json`` body (expected_files,
    target, regime_required, seller_name, acquisition_date, force_reprocess).
    ``pack_files`` is the list of data filenames present in the folder (for the
    expected-files completeness check).
    """
    created_at = now or _now()
    meta = dict(marker_metadata or {})
    manifest: Dict[str, Any] = {
        "event_id": make_event_id(blob_path, created_at),
        "blob_path": blob_path,
        "container": container,
        "pack_folder": blob_path.rsplit("/", 1)[0] if "/" in blob_path else None,
        "event_type": None,                 # data_file | ready_marker
        "event_decision": None,             # audit vocabulary (see EVT_*)
        "client_id": None, "dataset": None, "dataset_type": None, "frequency": None,
        "source_portfolio_id": None, "reporting_period": None, "reporting_date": None,
        "schema_fingerprint": None,
        "registry_match": False,
        "requires_source_onboarding": None,
        "selected_target": None,
        "target": None,
        "orchestrator_invocation": None,
        "orchestrator_run_id": None,
        "central_canonical_path": None,
        "assembler_refresh": None,
        "status": None,
        "error": None,
        "created_at": created_at,
    }

    # 1) Parse path (fail closed) -----------------------------------------
    try:
        parsed: ParsedPath = parse_blob_path(blob_path, container)
    except PathParseError as exc:
        manifest["status"] = STATUS_FAILED
        manifest["event_decision"] = EVT_INVALID_PATH
        manifest["error"] = f"path_parse_error: {exc}"
        write_event_manifest(manifest, out_dir)
        return manifest
    manifest.update(
        client_id=parsed.client_id, dataset=parsed.dataset, dataset_type=parsed.dataset,
        frequency=parsed.frequency, source_portfolio_id=parsed.source_portfolio_id,
        reporting_period=parsed.reporting_period, reporting_date=parsed.reporting_period)
    manifest["pack_key"] = _pack_key(parsed)
    is_marker = parsed.filename == pack_marker
    manifest["is_pack_marker"] = is_marker
    manifest["event_type"] = "ready_marker" if is_marker else "data_file"

    # 1b) Completion gate (READY sentinel) --------------------------------
    # Only the marker file starts processing. A data-file upload is acknowledged
    # and logged as a pack member, but never starts the Orchestrator.
    if not is_marker:
        manifest["status"] = STATUS_AWAITING_PACK
        manifest["decision"] = DECISION_PACK_MEMBER
        manifest["event_decision"] = EVT_IGNORED_DATA_FILE
        manifest["orchestrator_invocation"] = {
            "invoked": False, "reason": f"awaiting completion marker {pack_marker!r}"}
        write_event_manifest(manifest, out_dir)
        return manifest

    # 2) Schema fingerprint (fail closed) — over the PACK's data files -----
    try:
        if schema_info is None:
            if not local_input_path:
                raise ValueError("no local_input_path to fingerprint")
            schema_info = compute_schema_fingerprint(local_input_path)
    except Exception as exc:  # noqa: BLE001
        manifest["status"] = STATUS_FAILED
        manifest["event_decision"] = EVT_FAILED
        manifest["error"] = f"fingerprint_error: {exc}"
        write_event_manifest(manifest, out_dir)
        return manifest
    manifest["schema_fingerprint"] = schema_info.fingerprint

    # 2b) Pack completeness — expected files declared in the marker -------
    expected = list(meta.get("expected_files") or [])
    if expected and pack_files is not None:
        missing = [f for f in expected if f not in set(pack_files)]
        if missing:
            manifest["status"] = STATUS_PENDING_REVIEW
            manifest["decision"] = DECISION_INCOMPLETE_PACK
            manifest["event_decision"] = EVT_INCOMPLETE_PACK_PENDING
            manifest["error"] = f"incomplete_pack: missing {missing}"
            manifest["orchestrator_invocation"] = {
                "invoked": False, "reason": "incomplete_pack_fail_closed"}
            write_event_manifest(manifest, out_dir)
            return manifest

    # 2c) Idempotency — a pack that already ran this exact schema is skipped,
    # unless the marker explicitly carries force_reprocess=true.
    force = bool(meta.get("force_reprocess"))
    prior = _read_processed(out_dir, manifest["pack_key"])
    if (not force) and prior and prior.get("schema_fingerprint") == schema_info.fingerprint:
        manifest["status"] = STATUS_ALREADY_PROCESSED
        manifest["decision"] = DECISION_IDEMPOTENT_SKIP
        manifest["event_decision"] = EVT_DUPLICATE_READY_IGNORED
        manifest["orchestrator_invocation"] = {
            "invoked": False, "reason": "pack already processed",
            "prior_run_id": prior.get("run_id"), "prior_status": prior.get("status")}
        write_event_manifest(manifest, out_dir)
        return manifest

    # 3) Registry inference + target selection -----------------------------
    rec = registry.lookup(parsed.client_id, parsed.source_portfolio_id,
                          parsed.dataset, parsed.frequency)
    manifest["registry_match"] = rec is not None
    # regime_required: marker override wins, else the registry record, else off.
    if "regime_required" in meta:
        regime_required = bool(meta.get("regime_required"))
    else:
        regime_required = bool(rec.regime_required) if rec else False
    sel = select_target(parsed.dataset, parsed.frequency, regime_required=regime_required)
    sel_target, sel_run_regime, sel_reason = sel.target, sel.run_regime, sel.reason

    # Optional explicit target override from the marker — honoured ONLY for
    # funded (pipeline/forecast are MI-only and never route to Regime).
    override = (meta.get("target") or "").strip().lower()
    if override in ("mi", "all", "regime"):
        if parsed.dataset == "funded":
            sel_target = "all" if override in ("all", "regime") else "mi"
            sel_run_regime = override in ("all", "regime")
            sel_reason = f"marker target override → {override}"
        else:
            sel_reason = (f"{sel_reason}; marker target override {override!r} "
                          f"ignored ({parsed.dataset} is MI-only)")
    manifest["selected_target"] = {"target": sel_target, "run_regime": sel_run_regime,
                                   "reason": sel_reason}
    manifest["target"] = sel_target

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

    # acquired-portfolio metadata (marker wins; registry record may also carry it).
    acquisition_date = meta.get("acquisition_date") or getattr(rec, "acquisition_date", None)
    seller_name = meta.get("seller_name") or getattr(rec, "seller_name", None)
    portfolio_type = (meta.get("source_portfolio_type")
                      or (rec.source_portfolio_type if rec else None)
                      or _derive_type(parsed.source_portfolio_id))

    # 4) Route -------------------------------------------------------------
    if decision == DECISION_SCHEMA_DRIFT:
        # Fail closed — never process with a stale mapping.
        manifest["status"] = STATUS_PENDING_REVIEW
        manifest["event_decision"] = EVT_SCHEMA_DRIFT_PENDING
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
            source_portfolio_type=portfolio_type,
            dataset=parsed.dataset, frequency=parsed.frequency,
            reporting_period=parsed.reporting_period, input_path=input_for_orch,
            target=sel_target, run_regime=sel_run_regime,
            acquisition_date=acquisition_date, seller_name=seller_name,
            mapping_config_path=None, out_dir=str(out_dir))
        # Approval is human-gated: a new/changed source stops at pending_review.
        manifest["orchestrator_invocation"] = {
            "invoked": True, "mode": DECISION_SOURCE_ONBOARDING,
            "target": sel_target, "run_regime": sel_run_regime, **_inv(result)}
        manifest["orchestrator_run_id"] = (result or {}).get("run_id")
        manifest["status"] = STATUS_PENDING_REVIEW
        manifest["event_decision"] = EVT_NEW_SOURCE_PENDING
        _upsert_source(registry, parsed, schema_info, status=STATUS_PENDING_REVIEW,
                       regime_required=regime_required, portfolio_type=portfolio_type)
        _write_processed(out_dir, manifest, schema_info, result)
        write_event_manifest(manifest, out_dir)
        return manifest

    # decision == deterministic ------------------------------------------
    result = orchestrator_invoker(
        processing_mode=DECISION_DETERMINISTIC,
        client_id=parsed.client_id, source_portfolio_id=parsed.source_portfolio_id,
        source_portfolio_type=portfolio_type,
        dataset=parsed.dataset, frequency=parsed.frequency,
        reporting_period=parsed.reporting_period, input_path=input_for_orch,
        target=sel_target, run_regime=sel_run_regime,
        acquisition_date=acquisition_date, seller_name=seller_name,
        mapping_config_path=rec.mapping_config_path, out_dir=str(out_dir))
    manifest["orchestrator_invocation"] = {
        "invoked": True, "mode": DECISION_DETERMINISTIC,
        "target": sel_target, "run_regime": sel_run_regime, **_inv(result)}
    manifest["orchestrator_run_id"] = (result or {}).get("run_id")
    manifest["central_canonical_path"] = (result or {}).get("central_canonical_path")
    orch_status = (result or {}).get("status")
    if orch_status == "done":
        manifest["status"] = STATUS_PROCESSED
        manifest["event_decision"] = EVT_KNOWN_SOURCE_PROCESSED
        rec.last_successful_run_id = (result.get("run_id") or run_id_for_registry)
        rec.last_successful_reporting_period = parsed.reporting_period
        registry.upsert(rec)
        # 5) Assembler refresh — rebuild the central platform canonical across
        # portfolios from accepted canonical outputs (funded packs only).
        if parsed.dataset == "funded":
            _refresh_platform_canonical(
                manifest, parsed, result, sel_target, sel_run_regime,
                assembler_refresher, out_dir, accepted_root, platform_out_dir)
    elif orch_status == "halted":
        manifest["status"] = STATUS_HALTED
        manifest["event_decision"] = EVT_KNOWN_SOURCE_HALTED
    else:
        manifest["status"] = STATUS_FAILED
        manifest["event_decision"] = EVT_FAILED
        manifest["error"] = "; ".join((result or {}).get("blockers") or []) or "orchestrator_failed"
    # Mark the pack processed (idempotency) on any orchestrator-invoking outcome.
    _write_processed(out_dir, manifest, schema_info, result)
    write_event_manifest(manifest, out_dir)
    return manifest


# --------------------------------------------------------------------------- #
# Assembler refresh
# --------------------------------------------------------------------------- #

def _refresh_platform_canonical(manifest, parsed, result, target, run_regime,
                                assembler_refresher, out_dir, accepted_root,
                                platform_out_dir) -> None:
    accepted_root = accepted_root or str(Path(out_dir) / "_accepted")
    platform_out_dir = platform_out_dir or str(Path(out_dir) / "_platform")
    try:
        refresh = assembler_refresher(
            client_id=parsed.client_id,
            source_portfolio_id=parsed.source_portfolio_id,
            canonical_path=(result or {}).get("central_canonical_path"),
            accepted_root=accepted_root, platform_out_dir=platform_out_dir,
            target=target, run_regime=run_regime,
            regime=("ESMA_Annex2" if run_regime else None))
        manifest["assembler_refresh"] = refresh
        if refresh and refresh.get("central_canonical_path"):
            manifest["central_canonical_path"] = refresh["central_canonical_path"]
    except Exception as exc:  # noqa: BLE001 — pack DID process; record refresh failure
        manifest["assembler_refresh"] = {"error": f"{type(exc).__name__}: {exc}"}


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
                   *, status: str, regime_required: bool,
                   portfolio_type: Optional[str] = None) -> None:
    existing = registry.lookup(parsed.client_id, parsed.source_portfolio_id,
                               parsed.dataset, parsed.frequency)
    rec = existing or SourceRecord(
        client_id=parsed.client_id, source_portfolio_id=parsed.source_portfolio_id,
        dataset=parsed.dataset, frequency=parsed.frequency,
        source_portfolio_type=portfolio_type or _derive_type(parsed.source_portfolio_id))
    rec.status = status
    rec.regime_required = regime_required
    if portfolio_type:
        rec.source_portfolio_type = portfolio_type
    # Record the observed schema so the eventual approval can confirm it.
    rec.expected_schema_fingerprint = rec.expected_schema_fingerprint or schema.fingerprint
    rec.expected_columns = rec.expected_columns or schema.columns
    registry.upsert(rec)
