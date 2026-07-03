"""engine.orchestrator_agent.orchestrator — the governed conductor.

Two target-aware per-portfolio pipelines (the MI and regulatory canonicals
genuinely differ — regime carries the fuller ESMA Annex 2 field set with
mandatory fields, MI uses the lean central tape):

  * target = mi      : Onboarding(mi_only) ─▶ central tape ─▶ stamp
  * target = regime  : Onboarding(regulatory_mi) ─▶ Transformation ─▶ Validation ─▶ stamp
  * target = all     : the regulatory pipeline (its canonical is a superset that
                       also serves MI)

then, across portfolios: Assembler ─▶ central canonical ─▶ MI route and/or
Projection (ESMA Annex 2 + provenance companion).

Governed auto-halt: non-blocking steps proceed; the first step whose agent
reports a False readiness flag / blocking exception / mapping review HALTS the
run with resumable state. Resume continues from the halted step.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Sequence

from .adapters import AgentAdapters, PortfolioSpec, StepResult
from .state import (
    PortfolioState,
    RunState,
    STEP_DONE,
    STEP_FAILED,
    STEP_HALTED,
    STEP_RUNNING,
    StepState,
)

VALID_TARGETS = ("mi", "regime", "all")

# Per-portfolio step sequence by target. MI uses the central tape directly;
# regime/all run the regulatory Transformation + Validation agents.
_STEPS_MI = ("onboard", "stamp")
_STEPS_REG = ("onboard", "transform", "validate", "stamp")


def steps_for_target(target: str, full_pipeline: bool = False) -> Sequence[str]:
    """Per-portfolio EXECUTION DEPTH — independent of the target contract.

    ``full_pipeline`` runs the full agentic path (onboard → transform → validate
    → stamp) for ANY target; without it an MI-target run may take the lean
    central-tape shortcut (onboard → stamp). Regime/all always run the full path.
    Depth does NOT change the contract (see ``onboarding_mode_for_target``).
    """
    if target == "mi" and not full_pipeline:
        return _STEPS_MI
    return _STEPS_REG


def onboarding_mode_for_target(target: str) -> str:
    """Onboarding CONTRACT — determined by the TARGET alone, independent of
    pipeline depth:

      * mi        → ``mi_only``       (MI contract; no Annex 2-only mandatory fields)
      * regime    → ``regulatory_mi`` (ESMA Annex 2 contract)
      * all       → ``regulatory_mi`` (combined superset that also serves MI)
    """
    return "mi_only" if target == "mi" else "regulatory_mi"


def new_run_id(client_id: str, stamp: str) -> str:
    safe = "".join(c for c in client_id.lower() if c.isalnum() or c in "-_") or "client"
    return f"orun_{safe}_{stamp}"


def _apply(step: StepState, r: StepResult) -> None:
    step.output_path = r.output_path or step.output_path
    step.manifest_path = r.manifest_path or step.manifest_path
    step.readiness = r.readiness or step.readiness
    step.blockers = r.blockers or []
    step.message = r.message
    if r.blocking:
        step.status = STEP_HALTED
    elif r.ok:
        step.status = STEP_DONE
    else:
        step.status = STEP_FAILED


def _spec_from_portfolio(p: PortfolioState) -> PortfolioSpec:
    return PortfolioSpec(
        source_portfolio_id=p.source_portfolio_id, input=p.input,
        source_portfolio_type=p.source_portfolio_type,
        source_portfolio_label=p.source_portfolio_label,
        acquisition_date=p.acquisition_date, seller_name=p.seller_name,
        allow_unknown_acquisition_date=p.allow_unknown_acquisition_date,
    )


def _safe_derive_type(pid: str) -> Optional[str]:
    from engine.provenance import derive_portfolio_type
    return derive_portfolio_type(pid)


def _init_state(client_id, specs, target, out_root, run_id, created_at) -> RunState:
    state = RunState(run_id=run_id, client_id=client_id, target=target,
                     out_root=out_root, created_at=created_at)
    for s in specs:
        state.portfolios.append(PortfolioState(
            source_portfolio_id=s.source_portfolio_id,
            source_portfolio_type=(s.source_portfolio_type
                                   or _safe_derive_type(s.source_portfolio_id) or ""),
            source_portfolio_label=s.source_portfolio_label,
            acquisition_date=s.acquisition_date, seller_name=s.seller_name,
            allow_unknown_acquisition_date=s.allow_unknown_acquisition_date,
            input=s.input))
    return state


def _canonical_for_stamp(p: PortfolioState) -> Optional[str]:
    """The canonical fed to provenance stamping: the validated regulatory tape
    when present, else the MI central tape from onboarding."""
    v = p.step("validate")
    if v.done and v.output_path:
        return v.output_path
    return p.step("onboard").output_path


def _run_portfolio_step(adapters, p, step_name, work_dir) -> StepResult:
    spec = _spec_from_portfolio(p)
    if step_name == "onboard":
        return adapters.onboard(spec, work_dir)
    if step_name == "transform":
        handoff = p.step("onboard").manifest_path
        if not handoff:
            return StepResult(
                ok=False, blocking=True,
                blockers=["full pipeline requested but onboarding produced no "
                          "handoff manifest (Gate 2 has no contract to transform)"],
                message="missing onboarding handoff")
        # Governed HALT (pending review, not a hard error) when the handoff is not
        # ready for transformation/validation — unresolved mapping gaps / blocking
        # decisions. A clean approved mapping clears this and Gate 2/3 proceed.
        import json as _json
        try:
            ready = bool(_json.loads(Path(handoff).read_text(encoding="utf-8"))
                         .get("ready_for_transformation_validation"))
        except Exception:  # noqa: BLE001
            ready = False
        if not ready:
            return StepResult(
                ok=False, blocking=True,
                blockers=["onboarding handoff not ready_for_transformation_validation "
                          "(unresolved mapping gaps / blocking decisions) — pending review"],
                message="handoff not ready for Gate 2")
        return adapters.transform(spec, handoff, work_dir)
    if step_name == "validate":
        return adapters.validate(spec, p.step("transform").manifest_path, work_dir)
    if step_name == "stamp":
        return adapters.stamp_provenance(spec, _canonical_for_stamp(p), work_dir / "stamped")
    raise ValueError(f"unknown step {step_name!r}")  # pragma: no cover


def run_orchestration(
    client_id: str,
    portfolios: Sequence[PortfolioSpec],
    *,
    target: str = "mi",
    out_root: str,
    adapters: AgentAdapters,
    created_at: str,
    run_id: Optional[str] = None,
    regime: Optional[str] = None,
    resume_state: Optional[RunState] = None,
    full_pipeline: bool = False,
    force_publish: bool = False,
    dataset: str = "",
) -> RunState:
    """Run (or resume) the governed orchestration. Returns the final RunState.

    ``full_pipeline`` runs the production onboard→transform→validate→stamp path
    for an MI-target run (funded MI), so Gate 2 typing and Gate 3 validation are
    applied before the canonical is stamped/assembled. ``force_publish`` proceeds
    past validation exceptions (the tape is still typed) so the platform canonical
    is published anyway; without it a validation halt stops before publishing.
    """
    if target not in VALID_TARGETS:
        raise ValueError(f"target must be one of {VALID_TARGETS}")
    if target in ("regime", "all") and not regime:
        regime = "ESMA_Annex2"

    state = resume_state or _init_state(
        client_id, portfolios, target, out_root,
        run_id or new_run_id(client_id, created_at.replace(":", "").replace("-", "")[:15]),
        created_at)
    if resume_state is None:
        state.full_pipeline = full_pipeline
        state.force_publish = force_publish
        state.dataset = dataset
    state.status = STEP_RUNNING
    state.blockers = []
    state.save()

    run_dir = Path(state.out_root) / state.run_id
    steps = steps_for_target(state.target, full_pipeline=state.full_pipeline)

    # ---- per-portfolio fan-out -------------------------------------------
    for p in state.portfolios:
        if p.status == STEP_DONE:
            continue
        work_dir = run_dir / "portfolios" / p.source_portfolio_id
        for step_name in steps:
            step = p.step(step_name)
            if step.done:
                continue
            step.status = STEP_RUNNING
            state.save()
            try:
                r = _run_portfolio_step(adapters, p, step_name, work_dir)
            except Exception as exc:
                step.status = STEP_FAILED
                step.message = f"{type(exc).__name__}: {exc}"
                step.blockers = [step.message]
                p.status = STEP_FAILED
                state.status = STEP_FAILED
                state.blockers.append(f"{p.source_portfolio_id}/{step_name}: {step.message}")
                state.save()
                return state
            _apply(step, r)
            state.save()
            if step.status != STEP_DONE:
                # force_publish: proceed past a VALIDATION halt (the transformed
                # tape is still typed) so the platform canonical is published;
                # every other halt/failure still stops the run.
                if (state.force_publish and step_name == "validate"
                        and step.status == STEP_HALTED):
                    step.status = STEP_DONE
                    state.blockers.append(
                        f"{p.source_portfolio_id}/validate: FORCE-PUBLISHED past "
                        "validation exceptions: "
                        + ("; ".join(step.blockers) or step.message or ""))
                    state.save()
                else:
                    p.status = step.status
                    state.status = STEP_HALTED if step.status == STEP_HALTED else STEP_FAILED
                    state.blockers.append(
                        f"{p.source_portfolio_id}/{step_name}: "
                        + ("; ".join(step.blockers) or step.message))
                    state.save()
                    return state
        p.status = STEP_DONE
        state.save()

    # ---- consolidate (Assembler) -----------------------------------------
    if not state.assemble.done:
        if state.dataset == "pipeline":
            # A pipeline dataset is NOT a funded loan canonical: its deliverable is
            # the central PIPELINE tape (18a, stamped), and the funded platform
            # assembler's loan-identity requirement (loan_identifier/unique_identifier)
            # does not apply. Use the stamped pipeline tape as the central canonical
            # and skip the funded assemble — funded runs are unaffected.
            stamped = next((p.step("stamp").output_path or p.step("onboard").output_path
                            for p in state.portfolios), None)
            state.assemble.status = STEP_DONE
            state.assemble.output_path = stamped
            state.assemble.readiness = {"pipeline_dataset": True,
                                        "central_pipeline_tape": stamped}
            state.assemble.message = ("pipeline dataset — central pipeline tape used "
                                      "as the canonical (funded platform assembler skipped)")
            state.central_canonical_path = stamped
            state.save()
        else:
            stamped = [p.step("stamp").output_path for p in state.portfolios]
            state.assemble.status = STEP_RUNNING
            state.save()
            r = adapters.assemble(stamped, run_dir / "out_platform",
                                  state.client_id, state.target, regime=regime)
            _apply(state.assemble, r)
            state.save()
            if state.assemble.status != STEP_DONE:
                state.status = STEP_HALTED if state.assemble.status == STEP_HALTED else STEP_FAILED
                state.blockers.append("assemble: " + ("; ".join(state.assemble.blockers)
                                                      or state.assemble.message))
                state.save()
                return state
            state.central_canonical_path = state.assemble.output_path
            state.save()

    # ---- route to MI ------------------------------------------------------
    if state.target in ("mi", "all") and not state.route.done:
        state.route.status = STEP_RUNNING
        state.save()
        _apply(state.route, adapters.route_mi(state.central_canonical_path))
        state.save()

    # ---- project to Regime (ESMA Annex 2 + companion) --------------------
    if state.target in ("regime", "all") and not state.project.done:
        state.project.status = STEP_RUNNING
        state.save()
        r = adapters.project(state.central_canonical_path, run_dir / "out_regime", regime)
        _apply(state.project, r)
        state.save()
        if state.project.status != STEP_DONE:
            state.status = STEP_HALTED if state.project.status == STEP_HALTED else STEP_FAILED
            state.blockers.append("project: " + ("; ".join(state.project.blockers)
                                                 or state.project.message))
            state.save()
            return state

    state.status = STEP_DONE
    state.save()
    return state
