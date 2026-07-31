"""The lifecycle table and its enforcement.

The state model is the contract the API, the service and the UI all read, so it
is tested as a table in its own right — not only through the scenarios that
happen to walk it.
"""

from __future__ import annotations

import pytest

from operations_control.occ_agent import states as _states
from operations_control.occ_agent.case import SyntheticCase
from operations_control.occ_agent.service import ActionNotAllowed

from .conftest import ACTOR, TENANT_A

#: Every state named in the specification. Kept as a literal list so a state
#: silently dropped from the table fails here.
REQUIRED_STATES = (
    "CASE_CREATED", "REQUIREMENTS_EXTRACTED",
    "REQUIREMENTS_CONFIRMATION_REQUIRED", "REQUIREMENTS_CONFIRMED",
    "ONBOARDING_PACK_GENERATED", "ONBOARDING_PACK_APPROVAL_REQUIRED",
    "ONBOARDING_PACK_ISSUED_SYNTHETICALLY", "AWAITING_CLIENT_INPUT",
    "ARTEFACTS_RECEIVED", "ARTEFACTS_CLASSIFIED", "CONFIG_DRAFTED",
    "CONFIG_APPROVAL_REQUIRED", "CONFIG_APPROVED",
    "SYNTHETIC_ONBOARDING_RUNNING", "EXCEPTIONS_REQUIRE_INPUT",
    "SYNTHETIC_ONBOARDING_PASSED", "ORCHESTRATION_PLAN_GENERATED",
    "EXECUTION_APPROVAL_REQUIRED", "READY_FOR_EXECUTION", "BLOCKED",
    "CANCELLED",
)


def test_every_required_state_exists():
    assert set(REQUIRED_STATES) <= set(_states.STATE_SPECS)


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
        # A blocked case can always still be cancelled.
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
        _states.CASE_CREATED, _states.REQUIREMENTS_EXTRACTED,
        _states.REQUIREMENTS_CONFIRMATION_REQUIRED,
        _states.REQUIREMENTS_CONFIRMED, _states.ONBOARDING_PACK_GENERATED,
        _states.ONBOARDING_PACK_APPROVAL_REQUIRED,
        _states.ONBOARDING_PACK_ISSUED_SYNTHETICALLY,
        _states.AWAITING_CLIENT_INPUT, _states.ARTEFACTS_RECEIVED,
        _states.ARTEFACTS_CLASSIFIED, _states.CONFIG_DRAFTED,
        _states.CONFIG_APPROVAL_REQUIRED, _states.CONFIG_APPROVED,
        _states.SYNTHETIC_ONBOARDING_RUNNING,
        _states.SYNTHETIC_ONBOARDING_PASSED,
        _states.ORCHESTRATION_PLAN_GENERATED,
        _states.EXECUTION_APPROVAL_REQUIRED, _states.READY_FOR_EXECUTION,
    ]
    for current, nxt in zip(path, path[1:]):
        _states.assert_transition(current, nxt)


@pytest.mark.parametrize("current,target", [
    ("CASE_CREATED", "READY_FOR_EXECUTION"),
    ("CASE_CREATED", "CONFIG_APPROVED"),
    ("REQUIREMENTS_EXTRACTED", "SYNTHETIC_ONBOARDING_PASSED"),
    ("ARTEFACTS_RECEIVED", "READY_FOR_EXECUTION"),
    ("CONFIG_DRAFTED", "READY_FOR_EXECUTION"),
    ("SYNTHETIC_ONBOARDING_PASSED", "READY_FOR_EXECUTION"),
    ("READY_FOR_EXECUTION", "CASE_CREATED"),
    ("CANCELLED", "REQUIREMENTS_EXTRACTED"),
])
def test_invalid_transitions_are_refused(current, target):
    assert _states.is_transition_allowed(current, target) is False
    with pytest.raises(_states.IllegalCaseTransition):
        _states.assert_transition(current, target)


def test_a_state_cannot_transition_to_itself():
    for state in _states.STATE_SPECS:
        assert _states.is_transition_allowed(state, state) is False


def test_an_unknown_state_is_refused():
    assert _states.is_transition_allowed("MADE_UP", "CASE_CREATED") is False
    with pytest.raises(_states.IllegalCaseTransition):
        _states.spec("MADE_UP")


def test_a_terminal_case_offers_no_human_actions():
    assert _states.STATE_SPECS[_states.READY_FOR_EXECUTION].allowed_human_actions == ()
    assert _states.STATE_SPECS[_states.CANCELLED].allowed_human_actions == ()


def test_asking_is_allowed_everywhere():
    for state in _states.STATE_SPECS:
        assert _states.action_allowed(state, _states.ACTION_ASK) is True


def test_returnable_states_never_skip_a_control():
    """Returning re-opens a stage; it can never jump past one."""
    downstream = {_states.CONFIG_APPROVED, _states.SYNTHETIC_ONBOARDING_PASSED,
                  _states.ORCHESTRATION_PLAN_GENERATED,
                  _states.EXECUTION_APPROVAL_REQUIRED,
                  _states.READY_FOR_EXECUTION}
    assert set(_states.RETURNABLE_STATES).isdisjoint(downstream)


# --------------------------------------------------------------------------- #
# Enforcement through the service
# --------------------------------------------------------------------------- #

def test_the_service_refuses_an_action_the_state_does_not_allow(service):
    case = service.create_case(tenant=TENANT_A, initiating_user=ACTOR,
                               instruction="Onboard Northstar Lending.")
    for call in (
        lambda: service.generate_onboarding_pack(case, actor=ACTOR),
        lambda: service.classify_artefacts(case, actor=ACTOR),
        lambda: service.draft_client_config(case, actor=ACTOR),
        lambda: service.run_synthetic_onboarding(case, actor=ACTOR),
        lambda: service.generate_orchestration_plan(case, actor=ACTOR),
        lambda: service.approve_execution_readiness(case, actor=ACTOR),
    ):
        with pytest.raises(ActionNotAllowed):
            call()


def test_the_service_refuses_to_confirm_requirements_that_are_incomplete(service):
    case = service.create_case(tenant=TENANT_A, initiating_user=ACTOR,
                               instruction="Onboard someone or other.")
    case = service.confirm_requirements(case, actor=ACTOR)
    assert case.state == _states.BLOCKED
    assert case.blockers


def test_the_lifecycle_is_exposed_for_the_ui():
    lifecycle = _states.lifecycle()
    assert [entry["state"] for entry in lifecycle] == list(_states.STATE_ORDER)
    assert lifecycle[0]["state"] == _states.CASE_CREATED
    assert any(entry["terminal"] for entry in lifecycle)


# --------------------------------------------------------------------------- #
# Identity
# --------------------------------------------------------------------------- #

def test_client_identity_cannot_change_implicitly():
    from operations_control.occ_agent.case import IdentityChangeRefused
    case = SyntheticCase(case_id="CASE-AAAA0001", tenant="client_a",
                         initiating_user="alice", client_id="northstar",
                         portfolio_id="direct_101")
    with pytest.raises(IdentityChangeRefused):
        case.rebind_identity(client_id="someone_else")
    with pytest.raises(IdentityChangeRefused):
        case.rebind_identity(portfolio_id="direct_999")
    # An explicit rebinding is permitted, and only that.
    case.rebind_identity(client_id="someone_else", explicit=True)
    assert case.client_id == "someone_else"


def test_a_correction_cannot_move_a_case_to_another_client(service):
    from operations_control.occ_agent.case import IdentityChangeRefused
    case = service.create_case(
        tenant=TENANT_A, initiating_user=ACTOR,
        instruction="Onboard Northstar Lending. UK equity release. Monthly "
                    "portfolio MI. Portfolio id direct_101. First reporting "
                    "date 2026-06-30.")
    case = service.confirm_requirements(case, actor=ACTOR)
    assert case.client_id == "northstar_lending"
    with pytest.raises(IdentityChangeRefused):
        service.apply_requirement_change(
            case, updates={"proposed_client_id": "someone_else"}, actor=ACTOR)


def test_setting_identity_for_the_first_time_is_allowed():
    case = SyntheticCase(case_id="CASE-AAAA0001", tenant="client_a",
                         initiating_user="alice")
    case.rebind_identity(client_id="northstar", portfolio_id="direct_101")
    assert case.client_id == "northstar"
    assert case.portfolio_id == "direct_101"
