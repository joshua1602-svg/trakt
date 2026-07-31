"""The two lifecycle tables, and the fact that there are exactly two.

The OCC Agent adds an EXECUTION lifecycle. It does not restate the onboarding
lifecycle — that belongs to
:mod:`operations_control.onboarding.case`, and these tests assert the OCC Agent
neither duplicates it nor bypasses it.
"""

from __future__ import annotations

import pytest

from operations_control.occ_agent import states as _states
from operations_control.occ_agent.service import ActionNotAllowed
from operations_control.onboarding import case as _onboarding_case

from .conftest import ACTOR, TENANT_A

#: Every execution state the feature promises. Kept as a literal list so a state
#: silently dropped from the table fails here.
REQUIRED_STATES = (
    "AWAITING_ONBOARDING", "READY_TO_RUN", "SYNTHETIC_ONBOARDING_RUNNING",
    "EXCEPTIONS_REQUIRE_INPUT", "SYNTHETIC_ONBOARDING_PASSED",
    "ORCHESTRATION_PLAN_GENERATED", "EXECUTION_APPROVAL_REQUIRED",
    "READY_FOR_EXECUTION", "BLOCKED", "CANCELLED",
)


# --------------------------------------------------------------------------- #
# There is one onboarding lifecycle, and it is not this module's
# --------------------------------------------------------------------------- #

def test_the_execution_table_does_not_restate_the_onboarding_table():
    """No onboarding status may appear as an execution state.

    Two tables describing the same thing is the duplication this refactor
    exists to avoid; a name in both would be the first symptom.
    """
    onboarding = {s.upper() for s in _onboarding_case.STATUSES}
    assert onboarding.isdisjoint(set(_states.STATE_SPECS))


def test_onboarding_actions_are_not_gated_by_the_execution_table():
    """Client Onboarding's own transition table decides its actions.

    The execution table must not second-guess it — otherwise there would be two
    answers to "can this case be approved?", and they would eventually differ.
    """
    for state in _states.STATE_SPECS:
        for action in _states.ONBOARDING_ACTIONS:
            assert _states.action_allowed(state, action) is True, (state, action)


def test_activation_is_not_a_state_this_feature_can_reach():
    assert "ACTIVATED" not in _states.STATE_SPECS
    assert not any("activat" in state.lower() for state in _states.STATE_SPECS)


# --------------------------------------------------------------------------- #
# The execution table
# --------------------------------------------------------------------------- #

def test_every_required_state_exists():
    assert set(REQUIRED_STATES) == set(_states.STATE_SPECS)


def test_every_state_defines_its_full_contract():
    for state, spec in _states.STATE_SPECS.items():
        described = _states.describe(state)
        for key in ("permitted_prior", "required_inputs", "automatic_actions",
                    "deterministic_controls", "required_approvals",
                    "allowed_human_actions", "next_states",
                    "blocking_conditions"):
            assert key in described, f"{state} is missing {key}"
        assert spec.label, state


def test_every_live_state_can_be_blocked_or_cancelled():
    for state in _states.STATE_SPECS:
        if state in (_states.READY_FOR_EXECUTION, _states.CANCELLED):
            continue        # terminal
        if state != _states.BLOCKED:
            assert _states.BLOCKED in _states.permitted_next(state), state
        # A blocked run can always still be cancelled.
        assert _states.CANCELLED in _states.permitted_next(state), state


def test_terminal_states_go_nowhere():
    for state in _states.TERMINAL_STATES:
        assert _states.permitted_next(state) == ()


def test_every_declared_next_state_exists():
    for state, spec in _states.STATE_SPECS.items():
        for nxt in spec.next_states:
            assert nxt in _states.STATE_SPECS, f"{state} -> {nxt}"
        for prior in spec.permitted_prior:
            assert prior in _states.STATE_SPECS, f"{prior} -> {state}"


def test_the_two_directions_of_the_table_agree():
    """A state listed as a successor must accept its predecessor.

    This is the property that catches a state added to one half of the table
    and not the other — which would otherwise permit a transition nobody
    intended.
    """
    for state, spec in _states.STATE_SPECS.items():
        for nxt in spec.next_states:
            if nxt in (_states.BLOCKED, _states.CANCELLED):
                continue
            if state == _states.BLOCKED:
                continue
            assert state in _states.STATE_SPECS[nxt].permitted_prior, (
                f"{state} -> {nxt} is declared forward but not backward")


def test_the_happy_path_is_walkable_end_to_end():
    path = [
        _states.AWAITING_ONBOARDING, _states.READY_TO_RUN,
        _states.SYNTHETIC_ONBOARDING_RUNNING,
        _states.SYNTHETIC_ONBOARDING_PASSED,
        _states.ORCHESTRATION_PLAN_GENERATED,
        _states.EXECUTION_APPROVAL_REQUIRED, _states.READY_FOR_EXECUTION,
    ]
    for current, nxt in zip(path, path[1:]):
        _states.assert_transition(current, nxt)


@pytest.mark.parametrize("current,target", [
    ("AWAITING_ONBOARDING", "READY_FOR_EXECUTION"),
    ("AWAITING_ONBOARDING", "SYNTHETIC_ONBOARDING_PASSED"),
    ("READY_TO_RUN", "READY_FOR_EXECUTION"),
    ("SYNTHETIC_ONBOARDING_PASSED", "READY_FOR_EXECUTION"),
    ("READY_FOR_EXECUTION", "AWAITING_ONBOARDING"),
    ("CANCELLED", "READY_TO_RUN"),
])
def test_invalid_transitions_are_refused(current, target):
    assert _states.is_transition_allowed(current, target) is False
    with pytest.raises(_states.IllegalRunTransition):
        _states.assert_transition(current, target)


def test_a_state_cannot_transition_to_itself():
    for state in _states.STATE_SPECS:
        assert _states.is_transition_allowed(state, state) is False


def test_an_unknown_state_is_refused():
    assert _states.is_transition_allowed("MADE_UP", "READY_TO_RUN") is False
    with pytest.raises(_states.IllegalRunTransition):
        _states.spec("MADE_UP")


def test_a_terminal_run_offers_no_execution_actions():
    for state in _states.TERMINAL_STATES:
        assert _states.STATE_SPECS[state].allowed_human_actions == ()


def test_asking_is_allowed_everywhere():
    for state in _states.STATE_SPECS:
        assert _states.action_allowed(state, _states.ACTION_ASK) is True


def test_the_lifecycle_is_exposed_for_the_ui():
    lifecycle = _states.lifecycle()
    assert [entry["state"] for entry in lifecycle] == list(_states.STATE_ORDER)
    assert lifecycle[0]["state"] == _states.AWAITING_ONBOARDING
    assert any(entry["terminal"] for entry in lifecycle)


# --------------------------------------------------------------------------- #
# Enforcement through the service
# --------------------------------------------------------------------------- #

def test_the_service_refuses_an_execution_action_the_state_does_not_allow(
        service):
    agent_case = service.create_case(tenant=TENANT_A, initiating_user=ACTOR,
                                     instruction="Onboard Northstar Lending.")
    for call in (
        lambda: service.run_synthetic_onboarding(agent_case, actor=ACTOR),
        lambda: service.generate_orchestration_plan(agent_case, actor=ACTOR),
        lambda: service.approve_execution_readiness(agent_case, actor=ACTOR),
    ):
        with pytest.raises(ActionNotAllowed):
            call()


def test_the_practice_run_cannot_start_before_the_onboarding_is_approved(
        service):
    """The gate between the two halves. Approval is the pipeline's input."""
    from operations_control.engine import OpsError
    agent_case = service.create_case(
        tenant=TENANT_A, initiating_user=ACTOR,
        instruction="Onboard Northstar Lending. UK equity release. Monthly "
                    "portfolio MI. Portfolio id direct_101.")
    # Force the run into a state that permits the action, so the refusal under
    # test is the approval check rather than the lifecycle table.
    agent_case.run.state = _states.READY_TO_RUN
    service.store.save(agent_case.run)
    with pytest.raises(OpsError) as excinfo:
        service.run_synthetic_onboarding(agent_case, actor=ACTOR)
    assert excinfo.value.code == "OCC_AGENT_ONBOARDING_NOT_APPROVED"


def test_the_onboarding_refuses_its_own_illegal_transition(service):
    """Approval is refused by Client Onboarding, not by anything written here."""
    from operations_control.engine import OpsError
    agent_case = service.create_case(tenant=TENANT_A, initiating_user=ACTOR,
                                     instruction="Onboard Northstar Lending.")
    with pytest.raises(OpsError) as excinfo:
        service.approve_onboarding(agent_case, actor=ACTOR, reason="because")
    assert excinfo.value.code == "OPS_ONBOARDING_INCOMPLETE"
