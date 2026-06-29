"""engine.orchestrator_agent.orchestrator — the governed conductor.

Drives the existing agents in a fixed DAG with per-portfolio fan-out and a
governed auto-halt gate policy: non-blocking steps proceed automatically; a step
whose agent reports a False readiness flag / blocking exception / mapping review
HALTS the run with a resumable state + blocker report. After the operator
resolves it out-of-band (the existing approve/accept CLIs), the run is resumed
and continues from the halted step.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Sequence

from .adapters import AgentAdapters, PortfolioSpec, StepResult
from .state import (
    PORTFOLIO_STEPS,
    PortfolioState,
    RunState,
    STEP_DONE,
    STEP_FAILED,
    STEP_HALTED,
    STEP_RUNNING,
    StepState,
)

VALID_TARGETS = ("mi", "regime", "all")


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
        source_portfolio_id=p.source_portfolio_id,
        input=p.input,
        source_portfolio_type=p.source_portfolio_type,
        source_portfolio_label=p.source_portfolio_label,
        acquisition_date=p.acquisition_date,
        seller_name=p.seller_name,
        allow_unknown_acquisition_date=p.allow_unknown_acquisition_date,
    )


def _init_state(client_id: str, specs: Sequence[PortfolioSpec], target: str,
                out_root: str, run_id: str, created_at: str) -> RunState:
    state = RunState(run_id=run_id, client_id=client_id, target=target,
                     out_root=out_root, created_at=created_at)
    for s in specs:
        ptype = (s.source_portfolio_type
                 or _safe_derive_type(s.source_portfolio_id))
        state.portfolios.append(PortfolioState(
            source_portfolio_id=s.source_portfolio_id,
            source_portfolio_type=ptype or "",
            source_portfolio_label=s.source_portfolio_label,
            acquisition_date=s.acquisition_date,
            seller_name=s.seller_name,
            allow_unknown_acquisition_date=s.allow_unknown_acquisition_date,
            input=s.input,
        ))
    return state


def _safe_derive_type(pid: str) -> Optional[str]:
    from engine.provenance import derive_portfolio_type
    return derive_portfolio_type(pid)


# Per-portfolio step → adapter call. Each returns a StepResult.
def _run_portfolio_step(adapters: AgentAdapters, p: PortfolioState, step_name: str,
                        work_dir: Path) -> StepResult:
    spec = _spec_from_portfolio(p)
    if step_name == "onboard":
        return adapters.onboard(spec, work_dir)
    if step_name == "transform":
        handoff = p.step("onboard").manifest_path
        return adapters.transform(spec, handoff, work_dir)
    if step_name == "validate":
        tx_manifest = p.step("transform").manifest_path
        return adapters.validate(spec, tx_manifest, work_dir)
    if step_name == "stamp":
        validated = p.step("validate").output_path
        return adapters.stamp_provenance(spec, validated, work_dir / "stamped")
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
) -> RunState:
    """Run (or resume) the governed orchestration. Returns the final RunState.

    Governed auto-halt: the first blocking step halts the run (state saved);
    re-invoke with ``resume_state`` after the operator resolves the blocker.
    """
    if target not in VALID_TARGETS:
        raise ValueError(f"target must be one of {VALID_TARGETS}")

    state = resume_state or _init_state(
        client_id, portfolios, target, out_root,
        run_id or new_run_id(client_id, created_at.replace(":", "").replace("-", "")[:15]),
        created_at)
    state.status = STEP_RUNNING
    state.blockers = []
    state.save()

    run_dir = Path(state.out_root) / state.run_id

    # ---- per-portfolio fan-out (onboard → transform → validate → stamp) ----
    for p in state.portfolios:
        if p.status == STEP_DONE:
            continue
        work_dir = run_dir / "portfolios" / p.source_portfolio_id
        for step_name in PORTFOLIO_STEPS:
            step = p.step(step_name)
            if step.done:
                continue
            step.status = STEP_RUNNING
            state.save()
            try:
                r = _run_portfolio_step(adapters, p, step_name, work_dir)
            except Exception as exc:  # hard failure — record + halt
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
                # Blocking gate (or failure) — halt the whole run, resumable.
                p.status = step.status
                state.status = STEP_HALTED if step.status == STEP_HALTED else STEP_FAILED
                state.blockers.append(
                    f"{p.source_portfolio_id}/{step_name}: "
                    + ("; ".join(step.blockers) or step.message))
                state.save()
                return state
        p.status = STEP_DONE
        state.save()

    # ---- consolidate (Assembler) ------------------------------------------
    if not state.assemble.done:
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
        r = adapters.route_mi(state.central_canonical_path)
        _apply(state.route, r)
        state.save()

    state.status = STEP_DONE
    state.save()
    return state
