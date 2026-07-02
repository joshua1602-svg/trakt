"""apps.blob_trigger_app.router — the trigger's decision core (no Azure deps).

Pure, testable routing: parse path → completion gate → pack fingerprint →
registry inference → source-onboarding vs deterministic decision → invoke the
Orchestrator Agent → refresh the central platform canonical via the Assembler
Agent → write an event manifest. No business logic beyond routing/inference
lives here; the Azure layer (root ``function_app.py`` Event Grid handler) only
fetches blobs and calls in here.
"""

from __future__ import annotations

import functools
import logging
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, TYPE_CHECKING

from .assembler_refresh import AssemblerRefresher, default_assembler_refresher
from .event_log import make_event_id, write_event_manifest
from .ops_advice import next_operator_action
from .orchestrator_invoke import OrchestratorInvoker, default_orchestrator_invoker
from .path_parser import ParsedPath, PathParseError, parse_blob_path
from .schema_fingerprint import SchemaInfo, compute_schema_fingerprint
from .source_registry import (
    STATUS_ACTIVE, STATUS_PENDING_REVIEW, SourceRecord, SourceRegistry,
)
from .target_selection import select_target

if TYPE_CHECKING:  # avoid import cost at module load; persistence is optional
    from .persistence import ProductionPersistence

def _pack_match_record(registry: SourceRegistry, blob_path: str, container: str):
    try:
        parsed = parse_blob_path(blob_path, container)
        return registry.lookup(parsed.client_id, parsed.source_portfolio_id,
                               parsed.dataset, parsed.frequency)
    except Exception:  # noqa: BLE001 — resolution is best-effort; defaults apply
        return None


def aliases_for_pack(registry: SourceRegistry, blob_path: str,
                     container: str = "raw") -> Optional[Dict[str, List[str]]]:
    """Resolve the approved filename-alias fallback hints for a pack's source, so
    the caller can pass them to ``fingerprint_pack(..., aliases=)``. Returns
    ``None`` when the source/aliases are unknown."""
    rec = _pack_match_record(registry, blob_path, container)
    return (getattr(rec, "file_role_aliases", None) or None) if rec else None


def role_schemas_for_pack(registry: SourceRegistry, blob_path: str,
                          container: str = "raw") -> Optional[Dict[str, List[str]]]:
    """Resolve the approved per-role header signatures (``role -> [columns]``) for
    a pack's source, so ``fingerprint_pack(..., role_schemas=)`` can classify
    HEADER-FIRST (assign a role by matching headers, regardless of filename).
    Returns ``None`` before the source has been promoted."""
    rec = _pack_match_record(registry, blob_path, container)
    return (getattr(rec, "file_role_schemas", None) or None) if rec else None


# Final event statuses.
STATUS_PROCESSED = "processed"
STATUS_HALTED = "halted"
STATUS_PENDING_REVIEW = "pending_review"
STATUS_FAILED = "failed"
STATUS_AWAITING_PACK = "awaiting_pack"          # a data file arrived; waiting for the marker
STATUS_ALREADY_PROCESSED = "already_processed"  # idempotency: this pack already ran

# Terminal pack outcomes that warrant an operator advisory + durable run record.
_OPERATOR_STATUSES = (STATUS_PROCESSED, STATUS_HALTED, STATUS_PENDING_REVIEW, STATUS_FAILED)

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
EVT_BOOK_TYPE_MISMATCH = "book_type_mismatch"
EVT_AUTO_APPROVED = "recurring_auto_approved_non_material"
EVT_MATERIAL_CHANGE_PENDING = "material_change_pending_review"
EVT_FAILED = "failed"


logger = logging.getLogger("trakt.blob_trigger.router")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _log_router_failures(fn):
    """Wrap the whole router so ANY uncaught exception is logged with its full
    traceback and the blob path before propagating — turns a silent Azure
    'Executed (Failed)' into a diagnosable error."""
    @functools.wraps(fn)
    def wrapper(blob_path, **kwargs):
        try:
            return fn(blob_path, **kwargs)
        except Exception:  # noqa: BLE001 — log then re-raise (preserve Failed status)
            logger.error("BLOB-TRIGGER ROUTER FAILED blob_path=%s\n%s",
                         blob_path, traceback.format_exc())
            raise
    return wrapper


@_log_router_failures
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
    persistence: Optional["ProductionPersistence"] = None,
    regime_runner: Optional[Callable[..., Dict[str, Any]]] = None,
    llm_generator: Optional[Callable[..., Any]] = None,
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
        "approval_id": None,
        "persisted": None,
        "created_at": created_at,
    }

    def _emit() -> Dict[str, Any]:
        """Write the event manifest locally and (if configured) durably.

        For terminal pack outcomes, attach the actionable ``next_action``
        advisory (approve / promote / rerun / fix_data_supply + exact command)
        and persist an operator run record so ``ops list-halted`` / ``ops show``
        can drive the feedback loop.
        """
        if manifest.get("is_pack_marker") and manifest.get("status") in _OPERATOR_STATUSES:
            # Advisory LLM recommendations (deterministic stays the source of truth).
            try:
                _attach_llm(manifest, persistence, created_at, llm_generator)
            except Exception as exc:  # noqa: BLE001 — advisory must never fail the event
                manifest.setdefault("persist_errors", []).append(f"llm: {exc}")
            try:
                manifest["next_action"] = next_operator_action(manifest)
            except Exception as exc:  # noqa: BLE001 — advisory is best effort
                manifest.setdefault("persist_errors", []).append(f"next_action: {exc}")
        write_event_manifest(manifest, out_dir)
        if persistence is not None:
            try:
                manifest["persisted_event_uri"] = persistence.persist_event_manifest(manifest)
            except Exception as exc:  # noqa: BLE001 — never fail the event on audit write
                manifest.setdefault("persist_errors", []).append(f"event_manifest: {exc}")
            try:
                uri = persistence.persist_run_record(manifest)
                if uri:
                    manifest["persisted_run_uri"] = uri
            except Exception as exc:  # noqa: BLE001 — never fail the event on ledger write
                manifest.setdefault("persist_errors", []).append(f"run_record: {exc}")
        return manifest

    # 1) Parse path (fail closed) -----------------------------------------
    try:
        parsed: ParsedPath = parse_blob_path(blob_path, container)
    except PathParseError as exc:
        manifest["status"] = STATUS_FAILED
        manifest["event_decision"] = EVT_INVALID_PATH
        manifest["error"] = f"path_parse_error: {exc}"
        return _emit()
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
        return _emit()

    # 2) Schema fingerprint (fail closed) — over the PACK's data files -----
    try:
        if schema_info is None:
            if not local_input_path:
                raise ValueError(
                    "no data files found in the pack folder to fingerprint — the "
                    "reporting folder appears to contain only the completion marker. "
                    "Upload the data file(s) listed in _READY.json alongside it, then "
                    "re-fire the marker.")
            schema_info = compute_schema_fingerprint(local_input_path)
    except Exception as exc:  # noqa: BLE001
        manifest["status"] = STATUS_FAILED
        manifest["event_decision"] = EVT_FAILED
        manifest["error"] = f"fingerprint_error: {exc}"
        return _emit()
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
            return _emit()

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
        return _emit()

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

    # Header-first role classification signals (from fingerprint_pack). Surface the
    # per-file diagnostics regardless of outcome so the operator can see WHY each
    # file got its role (header_signature / registry_alias / filename_keyword /
    # fallback_unknown), with confidence + matched/unmatched column counts.
    role_conflict = bool(getattr(schema_info, "ambiguous_role_conflict", False))
    drift_suspected = bool(getattr(schema_info, "drift_suspected", False))
    if getattr(schema_info, "role_diagnostics", None):
        manifest["file_role_diagnostics"] = schema_info.role_diagnostics
    if role_conflict:
        manifest["ambiguous_role_conflict"] = True
        manifest["conflicting_roles"] = list(getattr(schema_info, "conflicting_roles", []))
    if drift_suspected:
        manifest["drift_files"] = list(getattr(schema_info, "drift_files", []))

    # APPROVAL POLICY state (threaded into the deterministic success branch).
    auto_approved = False
    materiality = None
    if rec is None or not rec.has_approved_mapping:
        # New client / new source_portfolio (no ACTIVE record) → always one-click.
        decision = DECISION_SOURCE_ONBOARDING
    elif role_conflict:
        # Two files resolved to the same logical role — fail closed, never process
        # with an ambiguous pack.
        decision = DECISION_SCHEMA_DRIFT
    elif rec.expected_schema_fingerprint == schema_info.fingerprint and not drift_suspected:
        # Exact schema match → deterministic as today (no LLM, no approval).
        decision = DECISION_DETERMINISTIC
    elif drift_suspected:
        # A file matched NO approved role header signature (only weak filename
        # evidence) — header evidence absent, cannot trust classification → drift.
        decision = DECISION_SCHEMA_DRIFT
    elif not (rec.file_role_schemas and getattr(schema_info, "sheet_columns", None)):
        # Recurring approved source with a fingerprint change but NO pinned header
        # signatures to compare against (legacy/placeholder pin, or a pack with no
        # role columns) → cannot prove "significantly the same"; fail closed to the
        # one-click review path, exactly as before the approval policy existed.
        decision = DECISION_SCHEMA_DRIFT
    else:
        # Recurring APPROVED source, fingerprint CHANGED, but every file still
        # header-matched an approved role → run the APPROVAL POLICY materiality
        # classifier over the deterministic (+ any LLM) evidence.
        from . import approval_policy as _ap
        materiality = _ap.classify(
            old_role_schemas=rec.file_role_schemas or {},
            new_role_schemas=dict(getattr(schema_info, "sheet_columns", {}) or {}),
            old_fingerprint=rec.expected_schema_fingerprint,
            new_fingerprint=schema_info.fingerprint,
            # No mandatory-column set is threaded here → the classifier treats ANY
            # removed previously-mapped column as material (conservative default).
            mandatory_columns=None)
        manifest["materiality"] = materiality.to_dict()
        if materiality.auto_approvable:
            # NON-MATERIAL "significantly the same" recurring change → AUTO-APPROVE:
            # process deterministically with the saved mapping, then re-pin + write
            # the governance evidence (handled in the deterministic success branch).
            decision = DECISION_DETERMINISTIC
            auto_approved = True
            manifest["auto_approved"] = True
        else:
            # MATERIAL change to a known source → re-run discovery + the LLM mapping
            # resolver to PRE-FILL the mapping, then halt at pending_review (one-click).
            decision = DECISION_SOURCE_ONBOARDING
            manifest["material_change"] = True
    manifest["requires_source_onboarding"] = decision != DECISION_DETERMINISTIC
    manifest["decision"] = decision

    # Operator "apply my accepted decisions" rerun (after ops approve-recommendations,
    # before promote): force a DETERMINISTIC-apply run against the accepted 34_
    # decisions file — exactly the CLI's `rerun onboarding --target-first-decisions`.
    applied_decisions_path = meta.get("applied_decisions_path")
    if applied_decisions_path:
        decision = DECISION_DETERMINISTIC
        manifest["decision"] = decision
        manifest["requires_source_onboarding"] = False
        manifest["applied_decisions_path"] = applied_decisions_path

    input_for_orch = (input_dir_override
                      or (str(Path(local_input_path).parent) if local_input_path
                          else parsed.blob_path))
    manifest["input_dir"] = input_for_orch

    # acquired-portfolio metadata (marker wins; registry record may also carry it).
    acquisition_date = meta.get("acquisition_date") or getattr(rec, "acquisition_date", None)
    seller_name = meta.get("seller_name") or getattr(rec, "seller_name", None)
    # source_portfolio_type: marker → path book type → registry → pid-derived.
    marker_type = (meta.get("source_portfolio_type") or "").strip().lower() or None
    portfolio_type = (marker_type or parsed.source_book_type
                      or (rec.source_portfolio_type if rec else None)
                      or _derive_type(parsed.source_portfolio_id))
    # Reject a marker that contradicts the path's source_book_type (fail closed).
    if marker_type and parsed.source_book_type and marker_type != parsed.source_book_type:
        manifest["status"] = STATUS_FAILED
        manifest["event_decision"] = EVT_BOOK_TYPE_MISMATCH
        manifest["error"] = (
            f"book_type_mismatch: _READY.json source_portfolio_type={marker_type!r} "
            f"contradicts path source_book_type={parsed.source_book_type!r}")
        manifest["orchestrator_invocation"] = {"invoked": False,
                                               "reason": "book_type_mismatch"}
        return _emit()

    # Funded MI runs the FULL production pipeline (onboard→transform→validate→
    # stamp) — the same path the Codespaces CLI uses — so Gate 2 typing and Gate 3
    # validation are applied before the platform canonical is published.
    # Pipeline/forecast keep the lean MI path. force_publish (from _READY.json)
    # publishes despite validation exceptions.
    full_pipeline = parsed.dataset == "funded"
    force_publish = bool(meta.get("force_publish"))
    manifest["full_pipeline"] = full_pipeline
    manifest["force_publish"] = force_publish

    # 4) Route -------------------------------------------------------------
    if decision == DECISION_SCHEMA_DRIFT:
        # Fail closed — never process with a stale mapping or an ambiguous pack.
        manifest["status"] = STATUS_PENDING_REVIEW
        manifest["event_decision"] = EVT_SCHEMA_DRIFT_PENDING
        if role_conflict:
            manifest["error"] = (
                "ambiguous_role_conflict: two files resolved to the same logical "
                f"role(s) {manifest.get('conflicting_roles')} — see file_role_diagnostics")
            reason = "ambiguous_role_conflict_fail_closed"
        elif drift_suspected:
            manifest["error"] = (
                "schema_drift: file(s) do not match any approved role header "
                f"signature (only weak filename evidence): {manifest.get('drift_files')}")
            reason = "header_signature_mismatch_fail_closed"
        else:
            manifest["error"] = (
                f"schema_drift: incoming fingerprint {schema_info.fingerprint} != "
                f"saved {rec.expected_schema_fingerprint}")
            reason = "schema_drift_fail_closed"
        manifest["orchestrator_invocation"] = {"invoked": False, "reason": reason}
        rec.status = STATUS_PENDING_REVIEW
        registry.upsert(rec)
        _write_pending_approval(
            persistence, manifest, parsed, schema_info, created_at,
            kind="schema_drift", pack_files=pack_files,
            prior_fingerprint=rec.expected_schema_fingerprint,
            suggested_mapping_id=rec.approved_mapping_id,
            suggested_mapping_config_path=rec.mapping_config_path,
            portfolio_type=portfolio_type, regime_required=regime_required,
            acquisition_date=acquisition_date, seller_name=seller_name)
        return _emit()

    if decision == DECISION_SOURCE_ONBOARDING:
        result = orchestrator_invoker(
            processing_mode=DECISION_SOURCE_ONBOARDING,
            client_id=parsed.client_id, source_portfolio_id=parsed.source_portfolio_id,
            source_portfolio_type=portfolio_type,
            dataset=parsed.dataset, frequency=parsed.frequency,
            reporting_period=parsed.reporting_period, input_path=input_for_orch,
            target=sel_target, run_regime=sel_run_regime,
            acquisition_date=acquisition_date, seller_name=seller_name,
            full_pipeline=full_pipeline, force_publish=force_publish,
            mapping_config_path=None, out_dir=str(out_dir))
        # Approval is human-gated: a new/changed source stops at pending_review.
        manifest["orchestrator_invocation"] = {
            "invoked": True, "mode": DECISION_SOURCE_ONBOARDING,
            "target": sel_target, "run_regime": sel_run_regime, **_inv(result)}
        manifest["orchestrator_run_id"] = (result or {}).get("run_id")
        # Surface any onboarding recommendations/diagnostics so the operator can
        # review them before approving the mapping.
        if (result or {}).get("status") != "done":
            manifest["orchestrator_diagnostics"] = _halt_diagnostics(result)
        manifest["status"] = STATUS_PENDING_REVIEW
        # A MATERIAL change to a KNOWN source is audited distinctly from a brand-new
        # source, though both take the same one-click path (mapping pre-filled).
        is_material_change = bool(manifest.get("material_change"))
        manifest["event_decision"] = (EVT_MATERIAL_CHANGE_PENDING if is_material_change
                                      else EVT_NEW_SOURCE_PENDING)
        _upsert_source(registry, parsed, schema_info, status=STATUS_PENDING_REVIEW,
                       regime_required=regime_required, portfolio_type=portfolio_type)
        _write_pending_approval(
            persistence, manifest, parsed, schema_info, created_at,
            kind=("schema_drift" if is_material_change else "new_source"),
            pack_files=pack_files,
            suggested_mapping_id=None, suggested_mapping_config_path=None,
            portfolio_type=portfolio_type, regime_required=regime_required,
            acquisition_date=acquisition_date, seller_name=seller_name)
        _write_processed(out_dir, manifest, schema_info, result)
        return _emit()

    # decision == deterministic ------------------------------------------
    # The accepted-decisions override wins over any saved mapping (apply-my-decisions
    # rerun); otherwise use the promoted registry mapping. rec may be None when
    # applying accepted decisions on a not-yet-promoted source.
    mapping_cfg = applied_decisions_path or (rec.mapping_config_path if rec else None)
    # A promoted accepted-decisions mapping is stored as a blob:// URI — localise it
    # so the onboarding agent can load it as a target-first decisions file.
    if (mapping_cfg and str(mapping_cfg).startswith("blob://") and persistence is not None):
        try:
            dest = Path(out_dir) / "_mapping" / Path(mapping_cfg).name
            mapping_cfg = str(persistence.storage.download_file(mapping_cfg, dest))
        except Exception as exc:  # noqa: BLE001 — record, fall through with the URI
            manifest.setdefault("persist_errors", []).append(f"mapping_localise: {exc}")
    result = orchestrator_invoker(
        processing_mode=DECISION_DETERMINISTIC,
        client_id=parsed.client_id, source_portfolio_id=parsed.source_portfolio_id,
        source_portfolio_type=portfolio_type,
        dataset=parsed.dataset, frequency=parsed.frequency,
        reporting_period=parsed.reporting_period, input_path=input_for_orch,
        target=sel_target, run_regime=sel_run_regime,
        acquisition_date=acquisition_date, seller_name=seller_name,
        full_pipeline=full_pipeline, force_publish=force_publish,
        mapping_config_path=mapping_cfg, out_dir=str(out_dir))
    manifest["orchestrator_invocation"] = {
        "invoked": True, "mode": DECISION_DETERMINISTIC,
        "target": sel_target, "run_regime": sel_run_regime, **_inv(result)}
    manifest["orchestrator_run_id"] = (result or {}).get("run_id")
    manifest["central_canonical_path"] = (result or {}).get("central_canonical_path")
    orch_status = (result or {}).get("status")
    if orch_status == "done":
        manifest["status"] = STATUS_PROCESSED
        manifest["event_decision"] = EVT_KNOWN_SOURCE_PROCESSED
        if rec is not None:
            rec.last_successful_run_id = (result.get("run_id") or run_id_for_registry)
            rec.last_successful_reporting_period = parsed.reporting_period
            # AUTO-APPROVE (non-material recurring change): re-pin the registry
            # expected_schema_fingerprint + file_role_schemas to THIS pack so the
            # next identical upload is a clean `deterministic` (exact-match) run,
            # and write the governance audit trail. Deterministic mapping stays the
            # source of truth; canonical-only nulling is enforced downstream.
            if auto_approved:
                _auto_approve_repin(registry, rec, parsed, schema_info, materiality,
                                    persistence, manifest, created_at)
                manifest["event_decision"] = EVT_AUTO_APPROVED
            registry.upsert(rec)
        # 5) Assembler refresh — rebuild the central platform canonical across
        # portfolios from accepted canonical outputs (funded packs only).
        if parsed.dataset == "funded":
            _refresh_platform_canonical(
                manifest, parsed, result, sel_target, sel_run_regime,
                assembler_refresher, out_dir, accepted_root, platform_out_dir)
            # 6) Durable persistence — upload accepted + platform canonicals (and
            # regime outputs) to the persistent store (Azure Blob / filesystem).
            if persistence is not None:
                _persist_funded_outputs(
                    persistence, manifest, parsed, out_dir, accepted_root,
                    sel_run_regime, regime_runner)
    elif orch_status == "halted":
        manifest["status"] = STATUS_HALTED
        manifest["event_decision"] = EVT_KNOWN_SOURCE_HALTED
        # Explain the halt (stage/reason/blocking decisions/run_state.json path)
        # so the manifest says WHY central_canonical_path is null.
        manifest["orchestrator_diagnostics"] = _halt_diagnostics(result)
    else:
        manifest["status"] = STATUS_FAILED
        manifest["event_decision"] = EVT_FAILED
        manifest["orchestrator_diagnostics"] = _halt_diagnostics(result)
        manifest["error"] = "; ".join((result or {}).get("blockers") or []) or "orchestrator_failed"
    # Mark the pack processed (idempotency) on any orchestrator-invoking outcome.
    _write_processed(out_dir, manifest, schema_info, result)
    return _emit()


# --------------------------------------------------------------------------- #
# Approval policy — auto-approve re-pin + governance
# --------------------------------------------------------------------------- #

def _auto_approve_repin(registry, rec, parsed, schema_info, materiality,
                        persistence, manifest, created_at) -> None:
    """Re-pin the registry to a NON-MATERIAL recurring change and write the
    governance audit trail. ``rec`` is upserted by the caller.

    The evidence (deterministic conf, value-match rate, LLM conf, role-set diff,
    old→new fingerprint) lands in BOTH the run record (via ``manifest``) and a
    durable governance artifact, per the approval policy.
    """
    old_fingerprint = rec.expected_schema_fingerprint
    new_role_schemas = dict(getattr(schema_info, "sheet_columns", {}) or {})
    rec.expected_schema_fingerprint = schema_info.fingerprint
    rec.expected_columns = list(schema_info.columns)
    if new_role_schemas:
        rec.file_role_schemas = {r: list(c) for r, c in new_role_schemas.items()}
    rec.mapping_version = int(getattr(rec, "mapping_version", 0) or 0) + 1
    evidence = materiality.to_dict() if materiality is not None else {}
    manifest["auto_approval"] = {
        "old_fingerprint": old_fingerprint,
        "new_fingerprint": schema_info.fingerprint,
        "repinned_roles": sorted(new_role_schemas.keys()),
        "mapping_version": rec.mapping_version,
        "materiality": evidence,
    }
    if persistence is not None:
        try:
            doc = {
                "pack_key": manifest.get("pack_key"),
                "event_id": manifest.get("event_id"),
                "client_id": parsed.client_id,
                "source_portfolio_id": parsed.source_portfolio_id,
                "dataset": parsed.dataset, "frequency": parsed.frequency,
                "reporting_period": parsed.reporting_period,
                "decision": "auto_approved_non_material",
                "old_fingerprint": old_fingerprint,
                "new_fingerprint": schema_info.fingerprint,
                "repinned_role_schemas": rec.file_role_schemas,
                "mapping_version": rec.mapping_version,
                "materiality_evidence": evidence,
                "approved_by": "approval_policy:auto",
                "created_at": created_at,
            }
            manifest["governance_artifact_uri"] = (
                persistence.persist_governance_artifact(manifest.get("pack_key") or "unknown", doc))
        except Exception as exc:  # noqa: BLE001 — never fail the event on audit write
            manifest.setdefault("persist_errors", []).append(f"governance: {exc}")


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
# Production persistence + approvals (only when a persistence facade is given)
# --------------------------------------------------------------------------- #

def _write_pending_approval(persistence, manifest, parsed, schema_info, created_at, *,
                            kind, pack_files, portfolio_type, regime_required,
                            acquisition_date=None, seller_name=None,
                            prior_fingerprint=None, suggested_mapping_id=None,
                            suggested_mapping_config_path=None) -> None:
    if persistence is None:
        return
    try:
        art = persistence.write_pending_approval(
            kind=kind, client_id=parsed.client_id,
            source_book_type=parsed.source_book_type or portfolio_type,
            dataset=parsed.dataset, frequency=parsed.frequency,
            source_portfolio_id=parsed.source_portfolio_id,
            period=parsed.reporting_period, schema_fingerprint=schema_info.fingerprint,
            detected_files=pack_files, prior_schema_fingerprint=prior_fingerprint,
            suggested_mapping_id=suggested_mapping_id,
            suggested_mapping_config_path=suggested_mapping_config_path,
            # Capture THIS pack's role -> header signature (role -> columns) so
            # promotion pins the approved schemas for header-first matching next
            # month. sheet_columns is the {role: columns} map from classification.
            role_schemas=dict(getattr(schema_info, "sheet_columns", {}) or {}),
            source_metadata={
                "source_portfolio_type": portfolio_type,
                "regime_required": regime_required,
                "acquisition_date": acquisition_date, "seller_name": seller_name},
            created_at=created_at)
        manifest["approval_id"] = art.get("approval_id")
    except Exception as exc:  # noqa: BLE001 — never fail the event on approval write
        manifest.setdefault("persist_errors", []).append(f"approval: {exc}")


def _persist_funded_outputs(persistence, manifest, parsed, out_dir, accepted_root,
                            run_regime, regime_runner) -> None:
    accepted_root = accepted_root or str(Path(out_dir) / "_accepted")
    accepted_local = (Path(accepted_root) / parsed.client_id
                      / f"{parsed.source_portfolio_id}_canonical_typed.csv")
    central_local = manifest.get("central_canonical_path")
    persisted: Dict[str, Any] = {}
    try:
        persisted["accepted_uri"] = persistence.persist_accepted(
            parsed.client_id, parsed.source_portfolio_id, str(accepted_local))
        if central_local:
            plat = persistence.persist_platform(
                parsed.client_id, parsed.reporting_period, str(central_local))
            persisted["platform_latest_uri"] = plat.get("latest")
            persisted["platform_period_uri"] = plat.get("period")
            # Make the durably-persisted latest the manifest's central pointer.
            if plat.get("latest"):
                manifest["central_canonical_uri"] = plat["latest"]
        ref = manifest.get("assembler_refresh") or {}
        persisted["portfolios"] = ref.get("portfolios")
    except Exception as exc:  # noqa: BLE001
        manifest.setdefault("persist_errors", []).append(f"funded_outputs: {exc}")

    # Regime projection from the persisted central canonical (run-clean ESMA +
    # provenance companion), uploaded under the regime prefix for this period.
    if run_regime and regime_runner is not None and central_local:
        try:
            rr = regime_runner(
                central_canonical_path=str(central_local),
                client_id=parsed.client_id, period=parsed.reporting_period,
                regime="ESMA_Annex2", out_dir=str(Path(out_dir) / "_regime"))
            local_dir = (rr or {}).get("output_dir")
            if local_dir:
                persisted["regime_uris"] = persistence.persist_regime_dir(
                    parsed.client_id, parsed.reporting_period, local_dir)
            persisted["regime_ok"] = (rr or {}).get("ok")
        except Exception as exc:  # noqa: BLE001
            manifest.setdefault("persist_errors", []).append(f"regime: {exc}")
    manifest["persisted"] = persisted


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
    out = {"run_id": r.get("run_id"), "orchestrator_status": r.get("status"),
           "central_canonical_path": r.get("central_canonical_path")}
    # Surface the final investor-pack artifact (when the run produced one) so the
    # event manifest advertises the downloadable deck alongside the run outputs.
    if r.get("investor_pack_pptx") is not None:
        out["investor_pack_pptx"] = r.get("investor_pack_pptx")
    return out


def _attach_llm(manifest: Dict[str, Any], persistence, now: str, generator) -> None:
    """Attach ADVISORY LLM recommendation status to the manifest (and persist the
    recommendations artefact). Deterministic mapping/registry stays the production
    source of truth — the LLM never auto-applies and never fails the run. A CLEAN
    recurring known source triggers no LLM call."""
    from . import llm_recommendations as _llm

    policy = _llm.resolve_llm_policy()
    diag = manifest.get("orchestrator_diagnostics") or {}
    gates = diag.get("gates") or []
    gate_failed = bool((diag.get("run_summary") or {}).get("failed_gate")) or (
        manifest.get("status") in (STATUS_HALTED, STATUS_FAILED, STATUS_PENDING_REVIEW))
    pack_key = manifest.get("pack_key") or "unknown"
    recs, meta = _llm.generate_recommendations(
        pack_key=pack_key, decision=manifest.get("decision"), gates=gates,
        gate_failed=gate_failed, generator=generator, policy=policy, now=now)
    llm_status = {
        "llm_enabled": meta["llm_enabled"], "llm_invoked": meta["llm_invoked"],
        "llm_available": meta["llm_available"], "llm_reason": meta["llm_reason"],
        "llm_error": meta["llm_error"],
        "recommendations_present": meta["recommendations_present"],
        "deterministic_fallback_used": meta["deterministic_fallback_used"],
        "recommendations_artifact_uri": None,
    }
    if recs and persistence is not None:
        try:
            llm_status["recommendations_artifact_uri"] = (
                persistence.persist_llm_recommendations(pack_key, recs, meta, now))
        except Exception as exc:  # noqa: BLE001
            manifest.setdefault("persist_errors", []).append(f"llm_persist: {exc}")
    manifest["llm"] = llm_status


def _halt_diagnostics(result: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Manifest diagnostics for a non-done orchestrator outcome.

    Surfaces the halt stage/reason, blocking decisions, registry-gap count,
    validation errors and the path to the resumable ``run_state.json`` so the
    event manifest EXPLAINS why ``central_canonical_path`` is null instead of
    silently reporting ``orchestrator_status=halted``. Best effort: falls back
    to run-level blockers / state_path when the invoker predates the
    ``diagnostics`` block (e.g. a recording stub).
    """
    r = result or {}
    diag = r.get("diagnostics") or {}
    halt_stage = diag.get("halt_stage")
    halt_reason = (diag.get("halt_reason")
                   or "; ".join(r.get("blockers") or []) or None)
    out: Dict[str, Any] = {
        "halt_stage": halt_stage,
        "halt_reason": halt_reason,
        "blocking_decisions": diag.get("blocking_decisions") or (r.get("blockers") or []),
        "registry_gap_count": diag.get("registry_gap_count", 0),
        "issue_count": diag.get("issue_count", 0),
        "validation_errors": diag.get("validation_errors") or [],
        "mapping_recommendations": diag.get("mapping_recommendations") or [],
        "handoff_readiness": diag.get("handoff_readiness") or {},
        "transform_readiness": diag.get("transform_readiness") or {},
        "validation_readiness": diag.get("validation_readiness") or {},
        "gates": diag.get("gates") or [],
        "run_summary": diag.get("run_summary") or {},
        "run_state_path": diag.get("run_state_path") or r.get("state_path"),
    }
    # Say plainly why there is no central canonical.
    if r.get("central_canonical_path"):
        out["central_canonical_unavailable_reason"] = None
    else:
        where = f" at stage {halt_stage}" if halt_stage else ""
        because = f": {halt_reason}" if halt_reason else ""
        out["central_canonical_unavailable_reason"] = (
            f"orchestrator {r.get('status') or 'did not complete'}{where} before "
            f"the central canonical was assembled — no platform canonical was "
            f"published{because}")
    return out


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
