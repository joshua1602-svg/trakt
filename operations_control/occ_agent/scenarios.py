"""operations_control.occ_agent.scenarios — driving a fixture end to end.

One function, :func:`run_scenario`, walks a fixture through the whole operating
process by calling the SAME service methods the UI calls, in the order a human
would. It exists so a fixture is exercised through the real path rather than
through a shortcut: the tests, the "load a scenario" button in the tab and any
future demonstration all take this route.

It never forces a state. Where a control blocks, the drive stops there and
returns the case as it stands — which is how scenarios C, D and E prove their
blockers.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List

from . import states as _states
from .case import SyntheticCase
from .fixtures import Scenario, by_id
from .service import OccAgentService


@dataclass
class ScenarioRun:
    """The case, and the state it passed through on the way."""

    case: SyntheticCase
    progression: List[str] = field(default_factory=list)
    stopped_because: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {"case_id": self.case.case_id, "state": self.case.state,
                "progression": self.progression,
                "stopped_because": self.stopped_because,
                "blockers": self.case.blockers}


def run_scenario(service: OccAgentService, scenario_or_id, *, tenant: str,
                 actor: str,
                 resolve_decisions: bool = True) -> ScenarioRun:
    """Drive one fixture as far as its controls allow.

    ``resolve_decisions`` answers any mapping decision the run raises by
    accepting the agent's recommendation — the human action scenario B is about.
    Set it False to observe the halt itself.
    """
    scenario: Scenario = (scenario_or_id if isinstance(scenario_or_id, Scenario)
                          else by_id(str(scenario_or_id)))
    run = ScenarioRun(case=service.create_case(
        tenant=tenant, initiating_user=actor,
        instruction=scenario.instruction, fixture_id=scenario.fixture_id))
    record(run)

    # (label, the lifecycle action the step needs, whether it applies, the call).
    always = lambda _case: True                                   # noqa: E731
    steps = (
        ("confirm the interpretation", _states.ACTION_CONFIRM_REQUIREMENTS,
         always, lambda c: service.confirm_requirements(c, actor=actor)),
        ("generate the onboarding pack", _states.ACTION_GENERATE_PACK,
         always, lambda c: service.generate_onboarding_pack(c, actor=actor)),
        ("approve and synthetically issue the pack",
         _states.ACTION_APPROVE_PACK, always,
         lambda c: service.approve_onboarding_pack(c, actor=actor)),
        ("provide the synthetic client response",
         _states.ACTION_REGISTER_ARTEFACT, always,
         lambda c: _register_files(service, c, scenario, actor)),
        ("classify the artefacts", _states.ACTION_CLASSIFY_ARTEFACTS, always,
         lambda c: service.classify_artefacts(c, actor=actor)),
        ("draft the configuration", _states.ACTION_DRAFT_CONFIG, always,
         lambda c: service.draft_client_config(c, actor=actor)),
        ("approve the configuration", _states.ACTION_APPROVE_CONFIG, always,
         lambda c: service.approve_client_config(c, actor=actor)),
        ("run the synthetic onboarding", _states.ACTION_RUN_ONBOARDING, always,
         lambda c: service.run_synthetic_onboarding(c, actor=actor)),
        # Only when the run actually raised something, and only when the caller
        # asked for decisions to be settled.
        ("settle any open decisions", _states.ACTION_RESOLVE_DECISION,
         lambda c: resolve_decisions and bool(_open(c)),
         lambda c: _settle(service, c, actor)),
        ("generate the orchestration plan", _states.ACTION_GENERATE_PLAN,
         always, lambda c: service.generate_orchestration_plan(c, actor=actor)),
        ("approve readiness", _states.ACTION_APPROVE_EXECUTION, always,
         lambda c: service.approve_execution_readiness(c, actor=actor)),
    )

    for label, action, applies, step in steps:
        if run.case.state in (_states.BLOCKED, _states.CANCELLED):
            run.stopped_because = (
                f"the case was blocked before it could {label}")
            return run
        if not applies(run.case):
            continue
        if not _states.action_allowed(run.case.state, action):
            # A control is holding the case where it is. That is the drive
            # finishing correctly, not the driver failing: pushing past it is
            # exactly what this feature must never do.
            run.stopped_because = (
                f"the case is waiting at "
                f"{_states.spec_label(run.case.state)}, so it could not "
                f"{label}")
            return run
        run.case = step(run.case)
        record(run)
    if run.case.state != _states.READY_FOR_EXECUTION:
        run.stopped_because = "the case did not satisfy every readiness " \
                              "criterion"
    return run


def _open(case: SyntheticCase) -> List[Dict[str, Any]]:
    return [d for d in case.open_decisions if d.get("status", "open") == "open"]


def record(run: ScenarioRun) -> None:
    if not run.progression or run.progression[-1] != run.case.state:
        run.progression.append(run.case.state)


def _register_files(service: OccAgentService, case: SyntheticCase,
                    scenario: Scenario, actor: str) -> SyntheticCase:
    for spec in scenario.files:
        case = service.register_synthetic_artefact(
            case, filename=spec.filename,
            data=spec.content.encode("utf-8"), actor=actor,
            fixture_id=scenario.fixture_id, declared_type=spec.declared_type)
    return case


def _settle(service: OccAgentService, case: SyntheticCase,
            actor: str) -> SyntheticCase:
    """Answer every open decision by accepting the recommendation.

    Bounded: each pass must reduce the open set, so a decision that reappears
    after a rerun cannot loop. A decision the run keeps raising is left open and
    the case stays where the control put it.
    """
    for _ in range(5):
        open_ids = [d["decision_id"] for d in case.open_decisions
                    if d.get("status", "open") == "open"]
        if not open_ids:
            return case
        for decision_id in open_ids:
            still_open = next(
                (d for d in case.open_decisions
                 if d["decision_id"] == decision_id
                 and d.get("status", "open") == "open"), None)
            if still_open is None:
                continue
            case = service.resolve_decision(
                case, decision_id=decision_id, action="approve",
                value=str(still_open.get("recommendation") or ""),
                actor=actor,
                reason="accepted the agent's recommendation")
            if case.state in (_states.BLOCKED, _states.CANCELLED):
                return case
        remaining = [d["decision_id"] for d in case.open_decisions
                     if d.get("status", "open") == "open"]
        if set(remaining) == set(open_ids):
            return case
    return case
