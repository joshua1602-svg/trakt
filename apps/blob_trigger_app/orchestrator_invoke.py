"""apps.blob_trigger_app.orchestrator_invoke — the Orchestrator boundary.

The trigger decides WHAT arrived; the Orchestrator Agent decides WHAT TO DO. This
is the single seam the trigger calls. The default implementation wraps
``engine.orchestrator_agent.run_orchestration``; tests inject a recording stub.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

# An orchestrator invoker takes the routing decision and returns a small dict:
#   {"run_id", "status", "central_canonical_path", ...}
OrchestratorInvoker = Callable[..., Dict[str, Any]]

# Terminal step statuses that halt / fail a run (kept local to avoid importing
# engine state constants at module import time).
_HALTED = "halted"
_FAILED = "failed"


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

    handoff_readiness = _handoff_readiness(state)
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
        "run_state_path": str(state.state_path()),
    }


def _read_json_maybe(path: Optional[str]) -> Optional[Dict[str, Any]]:
    """Read a JSON artefact; if given a ``.csv`` path, try the ``.json`` sibling."""
    if not path:
        return None
    p = Path(path)
    for cand in (p, p.with_suffix(".json")):
        try:
            if cand.exists():
                return json.loads(cand.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001 — diagnostics must never raise
            continue
    return None


def _decision_queue(dq_path: Optional[str]) -> "tuple[List[Dict[str, Any]], List[Dict[str, Any]]]":
    """Return (blocking_decisions, unresolved_decisions) from the 28c queue."""
    doc = _read_json_maybe(dq_path) or {}
    blocking: List[Dict[str, Any]] = []
    unresolved: List[Dict[str, Any]] = []
    for r in (doc.get("rows", []) or []):
        entry = {
            "target_field": r.get("target_field", ""),
            "esma_code": r.get("esma_code", ""),
            "reason": (r.get("operator_question") or r.get("question")
                       or r.get("decision_reason") or r.get("coverage_status") or ""),
            "decision_status": r.get("decision_status"),
        }
        if r.get("blocking"):
            blocking.append(entry)
        elif r.get("requires_user_decision") or not r.get("decision_status"):
            unresolved.append(entry)
    return blocking, unresolved


def _coverage_gaps(cov_path: Optional[str]) -> "tuple[List[str], List[str]]":
    """Return (missing_target_fields, unresolved_fields) from the 28a matrix."""
    doc = _read_json_maybe(cov_path) or {}
    missing: List[str] = []
    unresolved: List[str] = []
    for r in (doc.get("rows", []) or []):
        tf = r.get("target_field", "")
        if not tf:
            continue
        if r.get("coverage_status") == "missing_required":
            missing.append(tf)
        if r.get("requires_user_decision") or r.get("blocking"):
            unresolved.append(tf)
    return missing, unresolved


def _handoff_readiness(state: Any) -> Dict[str, Any]:
    """Full onboarding-handoff readiness payload — explains EXACTLY which readiness
    gate failed and surfaces the actual blocking decisions / unresolved fields.

    Reads the first portfolio's handoff manifest (24_) + readiness (25_) + decision
    queue (28c) + coverage matrix (28a) at run time (Azure scratch still present)
    and EMBEDS the manifest so ops can inspect it after scratch is reclaimed. Never
    raises."""
    for p in getattr(state, "portfolios", []) or []:
        try:
            mpath = p.step("onboard").manifest_path
        except Exception:  # noqa: BLE001
            continue
        m = _read_json_maybe(mpath)
        if not m:
            continue
        rj = _read_json_maybe(m.get("readiness_path")) or {}
        blocking_count = int(m.get("blocking_decision_count", rj.get("blocking_decision_count", 0)) or 0)
        gap = int(m.get("registry_gap_count", rj.get("registry_gap_count", 0)) or 0)
        central_present = bool(rj.get("central_tape_present",
                                      int(m.get("central_tape_row_count", 0) or 0) > 0))
        coverage_present = bool(rj.get("coverage_matrix_present", True))
        universe_loaded = bool(rj.get("target_universe_loaded", True))

        # WHICH readiness gate failed (mirrors onboarding compute_readiness).
        failed_gates: List[str] = []
        if not central_present:
            failed_gates.append("central_tape_present=false")
        if not coverage_present:
            failed_gates.append("coverage_matrix_present=false")
        if not universe_loaded:
            failed_gates.append("target_universe_loaded=false")
        if gap > 0:
            failed_gates.append(f"registry_gap_count={gap}")
        if blocking_count > 0:
            failed_gates.append(f"blocking_decision_count={blocking_count}")

        blocking_decisions, unresolved_decisions = _decision_queue(m.get("decision_queue_path"))
        missing_fields, unresolved_fields = _coverage_gaps(m.get("target_coverage_matrix_path"))
        issue_count = blocking_count + gap + len(missing_fields)

        return {
            "source_portfolio_id": p.source_portfolio_id,
            "ready_for_transformation_validation": bool(m.get("ready_for_transformation_validation")),
            "ready_for_projection": bool(m.get("ready_for_projection")),
            "ready_for_xml_delivery": bool(m.get("ready_for_xml_delivery")),
            "failed_readiness_gates": failed_gates,
            "blocking_decision_count": blocking_count,
            "non_blocking_decision_count": int(m.get("non_blocking_decision_count", 0) or 0),
            "operator_decision_pending_count": int(m.get("operator_decision_pending_count", 0) or 0),
            "registry_gap_count": gap,
            "issue_count": issue_count,
            "blocking_decisions": blocking_decisions,
            "unresolved_decisions": unresolved_decisions,
            "unresolved_fields": sorted(set(unresolved_fields)),
            "missing_target_fields": sorted(set(missing_fields)),
            "source_absent_count": int(m.get("source_absent_count", 0) or 0),
            "target_field_count": int(m.get("target_field_count", 0) or 0),
            "source_mapped_count": int(m.get("source_mapped_count", 0) or 0),
            # scratch paths (ephemeral) + embedded manifest (durable once persisted).
            "handoff_manifest_path": str(mpath),
            "readiness_path": m.get("readiness_path"),
            "target_coverage_matrix_path": m.get("target_coverage_matrix_path"),
            "decision_queue_path": m.get("decision_queue_path"),
            "field_contract_path": m.get("field_contract_path"),
            "handoff_manifest": m,
        }
    return {}


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

    adapters = RealAgentAdapters(
        client_name=client_id,
        onboarding_mode=onboarding_mode_for_target(target),   # contract by target
        processing_mode=processing_mode,
        mapping_config_path=mapping_config_path,
        full_pipeline=full_pipeline,   # Gate 2 will run → onboarding must emit the handoff
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
