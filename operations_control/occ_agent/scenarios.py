"""operations_control.occ_agent.scenarios — driving a fixture end to end.

One function, :func:`run_scenario`, walks a fixture through the whole operating
process by calling the SAME service methods the UI calls, in the order a human
would: answer what the instruction says, provide the client's response, ask for
what is still outstanding, record what comes back, approve the onboarding, then
run the practice execution and take it to readiness.

It exists so a fixture is exercised through the real path rather than through a
shortcut: the tests, the "load a scenario" button in the tab and any future
demonstration all take this route.

It never forces a state. Where a control blocks, the drive stops there and
returns the case as it stands — which is how scenarios C, D and E prove their
blockers. Notably, it never calls activation: a scenario reaches readiness with
an *approved* onboarding case and no configuration created.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Tuple

from ..engine import OpsError
from ..onboarding.case import APPROVED, OnboardingCase
from . import states as _states
from .fixtures import Scenario, by_id
from .service import AgentCase, OccAgentService


@dataclass
class ScenarioRun:
    """The case, and the states it passed through on the way."""

    case: AgentCase
    progression: List[str] = field(default_factory=list)
    onboarding_progression: List[str] = field(default_factory=list)
    stopped_because: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {"case_ref": self.case.case_ref,
                "state": self.case.run.state,
                "onboarding_status": self.case.case.status,
                "progression": self.progression,
                "onboarding_progression": self.onboarding_progression,
                "stopped_because": self.stopped_because,
                "blockers": self.case.run.blockers}


def run_scenario(service: OccAgentService, scenario_or_id, *, tenant: str,
                 actor: str, resolve_decisions: bool = True) -> ScenarioRun:
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
    if scenario.reporting_period:
        run.case.run.reporting_period = scenario.reporting_period
        service.store.save(run.case.run)
    record(run)

    # (label, the lifecycle action the step needs, whether it applies, the call)
    always: Callable[[AgentCase], bool] = lambda _c: True          # noqa: E731
    steps: Tuple[Tuple[str, str, Callable[[AgentCase], bool],
                       Callable[[AgentCase], AgentCase]], ...] = (
        ("provide the practice client response",
         _states.ACTION_REGISTER_ARTEFACT, always,
         lambda c: _register_files(service, c, scenario, actor)),
        ("ask the client for what is outstanding",
         _states.ACTION_REQUEST_INFORMATION,
         lambda c: bool(service.onboarding.client_checklist(c.case)),
         lambda c: service.request_client_information(c, actor=actor)),
        ("record the client's response", _states.ACTION_RECORD_RESPONSE,
         lambda c: bool(c.case.outstanding_requests),
         lambda c: _respond(service, c, scenario, actor)),
        ("submit the onboarding for approval",
         _states.ACTION_SUBMIT_FOR_APPROVAL,
         lambda c: service.onboarding_readiness(c)["ready"],
         lambda c: service.submit_for_approval(c, actor=actor)),
        ("approve the onboarding", _states.ACTION_APPROVE_ONBOARDING,
         lambda c: service.onboarding_readiness(c)["ready"],
         lambda c: service.approve_onboarding(c, actor=actor)),
        ("run the practice onboarding", _states.ACTION_RUN_ONBOARDING,
         lambda c: c.case.status == APPROVED,
         lambda c: service.run_synthetic_onboarding(c, actor=actor)),
        # Only when the run actually raised something, and only when the caller
        # asked for decisions to be settled.
        ("settle any open decisions", _states.ACTION_RESOLVE_DECISION,
         lambda c: resolve_decisions and bool(_open(c)),
         lambda c: _settle(service, c, actor)),
        ("generate the orchestration plan", _states.ACTION_GENERATE_PLAN,
         always,
         lambda c: service.generate_orchestration_plan(c, actor=actor)),
        ("approve readiness", _states.ACTION_APPROVE_EXECUTION, always,
         lambda c: service.approve_execution_readiness(c, actor=actor)),
    )

    for label, action, applies, step in steps:
        if run.case.run.state in (_states.BLOCKED, _states.CANCELLED):
            run.stopped_because = (
                f"the practice run was blocked before it could {label}")
            return run
        if not applies(run.case):
            if action in (_states.ACTION_SUBMIT_FOR_APPROVAL,
                          _states.ACTION_APPROVE_ONBOARDING,
                          _states.ACTION_RUN_ONBOARDING):
                # The onboarding itself is holding the case. That IS the answer
                # for a scenario about an unanswered question, so it is reported
                # in the onboarding's own words rather than as a missing step.
                run.stopped_because = _onboarding_holds(service, run.case,
                                                        label)
                return run
            continue
        if not _states.action_allowed(run.case.run.state, action):
            # A control is holding the run where it is. That is the drive
            # finishing correctly, not the driver failing: pushing past it is
            # exactly what this feature must never do.
            run.stopped_because = (
                f"the practice run is waiting at "
                f"{_states.spec_label(run.case.run.state)}, so it could not "
                f"{label}")
            return run
        run.case = step(run.case)
        record(run)
    if run.case.run.state != _states.READY_FOR_EXECUTION:
        run.stopped_because = "the case did not satisfy every readiness " \
                              "criterion"
    return run


def _onboarding_holds(service: OccAgentService, agent_case: AgentCase,
                      label: str) -> str:
    """Why the onboarding is not ready, in Client Onboarding's own words."""
    readiness = service.onboarding_readiness(agent_case)
    problems = [str(p.get("message") or "") for p in readiness["blocking"]]
    outstanding = readiness["outstanding_requests"]
    if problems:
        return ("the onboarding is not ready, so it could not "
                f"{label}: " + "; ".join(problems[:3]))
    if outstanding:
        return (f"{len(outstanding)} information request(s) are still open, so "
                f"it could not {label}")
    return f"there was nothing to {label}"


def _open(agent_case: AgentCase) -> List[Dict[str, Any]]:
    return [d for d in agent_case.run.open_decisions
            if d.get("status", "open") == "open"]


def record(run: ScenarioRun) -> None:
    if not run.progression or run.progression[-1] != run.case.run.state:
        run.progression.append(run.case.run.state)
    if not run.onboarding_progression or \
            run.onboarding_progression[-1] != run.case.case.status:
        run.onboarding_progression.append(run.case.case.status)


def _register_files(service: OccAgentService, agent_case: AgentCase,
                    scenario: Scenario, actor: str) -> AgentCase:
    for spec in scenario.files:
        agent_case = service.register_synthetic_artefact(
            agent_case, filename=spec.filename,
            data=spec.content.encode("utf-8"), actor=actor,
            fixture_id=scenario.fixture_id, declared_type=spec.declared_type)
    return service.classify_artefacts(agent_case, actor=actor)


def _respond(service: OccAgentService, agent_case: AgentCase,
             scenario: Scenario, actor: str) -> AgentCase:
    """Answer the outstanding request with what the scenario says came back."""
    outstanding = agent_case.case.outstanding_requests
    if not outstanding:
        return agent_case
    request = outstanding[0]
    answers = client_answers(agent_case.case, request.items,
                             scenario.client_response)
    return service.record_client_response(
        agent_case, request_id=request.request_id, actor=actor,
        answers=answers, note="Practice client response.")


def client_answers(case: OnboardingCase, items: List[Dict[str, Any]],
                   response: Dict[str, Any]) -> Dict[str, Any]:
    """Turn ``{"section.field": value}`` into a ``record_response`` payload.

    Only the fields the request actually asked for are answered — a scenario
    that carries an answer to a question the catalogue no longer asks simply
    never applies it. Repeatable sections are rebuilt from the case's current
    items, so identifiers Trakt minted survive the response.
    """
    out: Dict[str, Any] = {}
    for item in items:
        section = str(item.get("section") or "")
        field_key = str(item.get("field") or "")
        key = f"{section}.{field_key}"
        if key not in response:
            continue
        value = response[key]
        index = item.get("index")
        if index is None:
            out.setdefault(section, {})[field_key] = value
            continue
        rows = out.get(section)
        if not isinstance(rows, list):
            rows = [copy.deepcopy(r) for r in case.items(section)]
            out[section] = rows
        if 0 <= int(index) < len(rows):
            rows[int(index)][field_key] = value
    return out


def _settle(service: OccAgentService, agent_case: AgentCase,
            actor: str) -> AgentCase:
    """Answer every open decision by accepting the recommendation.

    Bounded: each pass must reduce the open set, so a decision that reappears
    after a rerun cannot loop. A decision the run keeps raising is left open and
    the run stays where the control put it.
    """
    for _ in range(5):
        open_ids = [d["decision_id"] for d in agent_case.run.open_decisions
                    if d.get("status", "open") == "open"]
        if not open_ids:
            return agent_case
        for decision_id in open_ids:
            still_open = next(
                (d for d in agent_case.run.open_decisions
                 if d["decision_id"] == decision_id
                 and d.get("status", "open") == "open"), None)
            if still_open is None:
                continue
            try:
                agent_case = service.resolve_decision(
                    agent_case, decision_id=decision_id, action="approve",
                    value=str(still_open.get("recommendation") or ""),
                    actor=actor, reason="accepted the agent's recommendation")
            except OpsError:
                return agent_case
            if agent_case.run.state in (_states.BLOCKED, _states.CANCELLED):
                return agent_case
        remaining = [d["decision_id"] for d in agent_case.run.open_decisions
                     if d.get("status", "open") == "open"]
        if set(remaining) == set(open_ids):
            return agent_case
    return agent_case
