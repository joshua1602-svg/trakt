"""apps.blob_trigger_app.orchestrator_invoke — the Orchestrator boundary.

The trigger decides WHAT arrived; the Orchestrator Agent decides WHAT TO DO. This
is the single seam the trigger calls. The default implementation wraps
``engine.orchestrator_agent.run_orchestration``; tests inject a recording stub.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from . import gate_diagnostics as _gd

# An orchestrator invoker takes the routing decision and returns a small dict:
#   {"run_id", "status", "central_canonical_path", ...}
OrchestratorInvoker = Callable[..., Dict[str, Any]]

# Terminal step statuses that halt / fail a run (kept local to avoid importing
# engine state constants at module import time).
_HALTED = "halted"
_FAILED = "failed"

# Manifest / result key for the final investor-PPTX artifact.
ARTIFACT_KEY = "investor_pack_pptx"


def _run_diagnostics(state: Any) -> Dict[str, Any]:
    """Explain WHY a run did not publish a central canonical.

    Walks the run's steps in execution order — each portfolio's
    onboard → transform → validate → stamp, then assemble → route → project —
    and pins the FIRST step that halted or failed. That step is the reason the
    central canonical is null. Returns a JSON-safe dict; every field is best
    effort (``None``/``0``/``[]`` when the underlying agent did not surface it).
    """
    halt_stage: Optional[str] = None
    halt_reason: Optional[str] = None
    blocking_decisions: List[str] = []
    registry_gap_count = 0
    validation_errors: List[str] = []

    def _reason(step: Any) -> str:
        return "; ".join(getattr(step, "blockers", None) or []) or getattr(step, "message", "") or ""

    mapping_recommendations = _mapping_recommendations(state)

    # Pull registry-gap / validation signal from wherever it surfaced, even when
    # the halt was at a later stage (diligence wants the full picture).
    for p in getattr(state, "portfolios", []) or []:
        try:
            onboard = p.step("onboard")
            gap = (onboard.readiness or {}).get("registry_gap_count")
            if gap is None:
                gap = (onboard.readiness or {}).get("issue_count")
            if isinstance(gap, (int, float)):
                registry_gap_count = max(registry_gap_count, int(gap))
        except Exception:  # noqa: BLE001 — diagnostics must never raise
            pass
        try:
            vstep = p.step("validate")
            if getattr(vstep, "status", None) in (_HALTED, _FAILED):
                validation_errors.extend(getattr(vstep, "blockers", None) or [])
        except Exception:  # noqa: BLE001
            pass

    # First halted/failed step in execution order → the blocking stage.
    def _scan(label: str, step: Any) -> bool:
        nonlocal halt_stage, halt_reason, blocking_decisions
        if getattr(step, "status", None) in (_HALTED, _FAILED):
            halt_stage = label
            halt_reason = _reason(step)
            blocking_decisions = list(getattr(step, "blockers", None) or [])
            return True
        return False

    found = False
    for p in getattr(state, "portfolios", []) or []:
        for name in ("onboard", "transform", "validate", "stamp"):
            try:
                if _scan(f"{p.source_portfolio_id}/{name}", p.step(name)):
                    found = True
                    break
            except Exception:  # noqa: BLE001
                continue
        if found:
            break
    if not found:
        for label in ("assemble", "route", "project"):
            step = getattr(state, label, None)
            if step is not None and _scan(label, step):
                break

    # Fall back to the run-level blockers when no single step pinned a reason.
    if not halt_reason:
        halt_reason = "; ".join(getattr(state, "blockers", None) or []) or None

    # Generic per-gate observability (onboarding/transform/validation/stamp/
    # assembler/projection) + the run-level summary.
    gates = _gd.collect_gates(state)
    central = getattr(state, "central_canonical_path", None)
    run_summary = _gd.build_run_summary(state, gates, central)

    def _payload(name: str) -> Dict[str, Any]:
        for g in gates:
            if g["gate_name"] == name:
                return g.get("payload") or {}
        return {}

    handoff_readiness = _payload("onboarding")
    transform_readiness = _payload("transform")
    validation_readiness = _payload("validation")
    # issue_count: prefer the handoff's own tally, else derive from gaps + blockers.
    issue_count = handoff_readiness.get("issue_count")
    if issue_count is None:
        issue_count = registry_gap_count + len(handoff_readiness.get("blocking_decisions") or [])

    return {
        "halt_stage": halt_stage,
        "halt_reason": halt_reason,
        "blocking_decisions": blocking_decisions,
        "registry_gap_count": registry_gap_count,
        "issue_count": issue_count,
        "validation_errors": validation_errors,
        "mapping_recommendations": mapping_recommendations,
        "handoff_readiness": handoff_readiness,
        "transform_readiness": transform_readiness,
        "validation_readiness": validation_readiness,
        "gates": gates,
        "run_summary": run_summary,
        "run_state_path": str(state.state_path()),
    }


def _mapping_recommendations(state: Any) -> List[Dict[str, Any]]:
    """Best-effort: surface the onboarding agent's mapping recommendations /
    unresolved decisions from each portfolio's handoff manifest, so an operator
    can review + approve them from the CLI. Never raises."""
    recs: List[Dict[str, Any]] = []
    for p in getattr(state, "portfolios", []) or []:
        try:
            manifest_path = p.step("onboard").manifest_path
        except Exception:  # noqa: BLE001
            continue
        if not manifest_path:
            continue
        try:
            data = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            continue
        # The handoff exposes recommendations under a few possible keys across
        # onboarding modes — normalise whatever is present into a flat list.
        found = None
        for key in ("mapping_recommendations", "pending_decisions", "decisions",
                    "unmapped_fields", "mapping_review", "recommendations"):
            val = data.get(key)
            if val:
                found = val
                break
        if found is None:
            continue
        items = found if isinstance(found, list) else [found]
        for item in items:
            rec: Dict[str, Any] = {"source_portfolio_id": p.source_portfolio_id,
                                   "handoff_manifest": manifest_path}
            if isinstance(item, dict):
                rec.update(item)
            else:
                rec["field"] = item
            recs.append(rec)
    return recs


def default_orchestrator_invoker(
    *,
    processing_mode: str,          # "deterministic" | "source_onboarding"
    client_id: str,
    source_portfolio_id: str,
    source_portfolio_type: Optional[str],
    dataset: str,
    frequency: str,
    reporting_period: str,
    input_path: str,
    target: str,                   # "mi" | "all"
    run_regime: bool,
    mapping_config_path: Optional[str],
    out_dir: str,
    acquisition_date: Optional[str] = None,   # acquired portfolios (from _READY.json/registry)
    seller_name: Optional[str] = None,        # acquired portfolios (from _READY.json/registry)
    full_pipeline: bool = False,   # funded MI: run the full onboard→transform→validate→stamp
    force_publish: bool = False,   # publish the platform canonical despite validation exceptions
    regime: str = "ESMA_Annex2",
) -> Dict[str, Any]:
    """Invoke the real Orchestrator Agent.

    ``full_pipeline`` controls EXECUTION DEPTH only (whether Gate 2/Gate 3 run) —
    it does NOT change the contract. The onboarding CONTRACT follows the TARGET:
    ``mi`` → ``mi_only`` (MI contract; no Annex 2-only mandatory fields), ``all`` →
    ``regulatory_mi`` (combined). So funded MI runs the full pipeline against the
    MI contract, exactly like the Codespaces CLI. ``processing_mode`` still selects
    discovery (new/changed source) vs saved-mapping deterministic processing
    (recurring approved packs — no LLM)."""
    from engine.orchestrator_agent import run_orchestration
    from engine.orchestrator_agent.orchestrator import onboarding_mode_for_target
    from engine.orchestrator_agent.adapters import RealAgentAdapters, PortfolioSpec

    # LLM target advisor: only for a NEW/changed source (source_onboarding) and only
    # when the LLM policy is enabled. Recurring approved packs (deterministic) never
    # invoke it — they apply the promoted mapping.
    from . import llm_recommendations as _llm
    _policy = _llm.resolve_llm_policy()
    enable_llm_advisor = (processing_mode == "source_onboarding"
                          and _policy.get("enabled", False))
    # Agentic mapping RESOLVER: wired into the automated path only for a new/changed
    # source (source_onboarding) when TRAKT_LLM_MODE=resolving. A recurring approved
    # pack (deterministic) applies the saved mapping — never the resolver.
    enable_llm_mapping_review = (processing_mode == "source_onboarding"
                                 and _policy.get("resolve_mapping", False))
    adapters = RealAgentAdapters(
        client_name=client_id,
        onboarding_mode=onboarding_mode_for_target(target),   # contract by target
        processing_mode=processing_mode,
        mapping_config_path=mapping_config_path,
        full_pipeline=full_pipeline,   # Gate 2 will run → onboarding must emit the handoff
        reporting_period=reporting_period,   # derive reporting_date from the folder period
        enable_llm_advisor=enable_llm_advisor,
        enable_llm_mapping_review=enable_llm_mapping_review,
        llm_mapping_profile="low",
        managed_service=True,   # headless: run context from blob/folder only, no cli_fallback
    )
    spec = PortfolioSpec(
        source_portfolio_id=source_portfolio_id, input=input_path,
        source_portfolio_type=source_portfolio_type,
        acquisition_date=acquisition_date,
        seller_name=seller_name,
        # If the acquisition_date is supplied (acquired packs), require it; only
        # tolerate "unknown" when none was provided (direct books).
        allow_unknown_acquisition_date=(acquisition_date is None),
    )
    state = run_orchestration(
        client_id, [spec], target=target, regime=(regime if run_regime else None),
        out_root=out_dir, adapters=adapters,
        full_pipeline=full_pipeline, force_publish=force_publish,
        created_at=datetime.now(timezone.utc).isoformat())
    result: Dict[str, Any] = {
        "run_id": state.run_id,
        "status": state.status,                       # done | halted | failed
        "central_canonical_path": state.central_canonical_path,
        "blockers": state.blockers,
        "state_path": str(state.state_path()),
    }
    # Non-done runs get a diagnostics block explaining why the central canonical
    # is null (halt stage/reason, blocking decisions, registry gaps, validation
    # errors, and the path to the resumable run_state.json).
    if state.status != "done":
        result["diagnostics"] = _run_diagnostics(state)
        return result

    # ---- final orchestration stage: Investor PPTX ------------------------
    # A successful pipeline run automatically produces the MI Agent-native
    # investor PowerPoint as its final artifact, consuming the completed
    # canonical / analytics / risk / manifest outputs. The deck failure is
    # recorded in the manifest but does NOT fail an otherwise-successful run
    # unless PPTX has been explicitly configured mandatory.
    _generate_investor_pptx(state, client_id=client_id,
                            reporting_period=reporting_period, result=result)
    return result


def _generate_investor_pptx(state: Any, *, client_id: str,
                            reporting_period: str, result: Dict[str, Any]) -> None:
    """Run the investor-PPTX stage for a completed run (guarded).

    Client name / as-of date are resolved from run metadata already produced by
    the orchestration (never by reopening the raw uploaded files). The stage is
    idempotent — a replay overwrites the existing deck and refreshes the
    manifest timestamp.
    """
    from .pptx_stage import (generate_investor_pptx, pptx_enabled,
                             pptx_mandatory)

    if not pptx_enabled():
        return

    # Derive the run directory from the run manifest the orchestration already
    # saved (out_root/run_id/run_state.json → run_dir). Gate on the manifest
    # existing on disk: a real completed run always persisted it, so the deck
    # consumes completed artifacts (and never runs against a bare stub state).
    try:
        manifest_path = Path(state.state_path())
    except Exception:  # noqa: BLE001 — malformed state, skip silently
        return
    if not manifest_path.exists():
        return
    run_dir = manifest_path.parent

    mandatory = pptx_mandatory()
    # Client name priority: run metadata (client_id) → resolved client id → "Client".
    client_name = getattr(state, "client_id", None) or client_id or "Client"
    # As-of date priority: run metadata reporting period (already resolved);
    # otherwise the generator infers the tape's data cut-off date.
    as_of_date = reporting_period or ""
    try:
        result[ARTIFACT_KEY] = generate_investor_pptx(
            run_dir, client_name=client_name, as_of_date=as_of_date,
            mandatory=mandatory)
    except Exception as exc:  # mandatory failure → fail the run
        logging.exception("Investor PPTX stage failed (mandatory=%s)", mandatory)
        if mandatory:
            raise
        result[ARTIFACT_KEY] = {
            "type": "pptx", "status": "failed", "error": str(exc),
            "generator": "mi_agent_pptx"}
