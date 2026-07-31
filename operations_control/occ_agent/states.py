"""operations_control.occ_agent.states — the synthetic EXECUTION lifecycle.

Client Onboarding already owns the onboarding lifecycle: ``draft →
information_requested → awaiting_client → in_review → ready_for_approval →
approved → activated``, with its own transition table in
:mod:`operations_control.onboarding.case`. This module does **not** restate it.

What it covers is the part that has no counterpart there. Client Onboarding
stops at activation — it creates the configuration and knows nothing about
running a pipeline. The OCC Agent continues from an *approved* case into a
synthetic execution and on to ``READY_FOR_EXECUTION``, and these are that
continuation's states:

    AWAITING_ONBOARDING        the case is not approved yet; nothing to run
    READY_TO_RUN               approved, artefacts present, run not started
    SYNTHETIC_ONBOARDING_RUNNING
    EXCEPTIONS_REQUIRE_INPUT   a control needs a human
    SYNTHETIC_ONBOARDING_PASSED
    ORCHESTRATION_PLAN_GENERATED
    EXECUTION_APPROVAL_REQUIRED
    READY_FOR_EXECUTION
    BLOCKED
    CANCELLED

Two properties hold, as before: the transition check lives here rather than in
the caller, so an invalid move cannot be reached from the API, the UI or a
natural-language instruction; and the interpreter never decides a transition —
it proposes an action, controls run, and the resulting state is asserted here.

Activation is deliberately absent. In synthetic mode the policy refuses
``OnboardingService.activate()`` — the one call that writes configuration — so
a practice case reaches readiness having created nothing.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

from ..contracts import STAGE_ASSEMBLY, STAGE_MAPPING, STAGE_VALIDATION
from ..engine import OpsError

# --------------------------------------------------------------------------- #
# The states
# --------------------------------------------------------------------------- #

AWAITING_ONBOARDING = "AWAITING_ONBOARDING"
READY_TO_RUN = "READY_TO_RUN"
SYNTHETIC_ONBOARDING_RUNNING = "SYNTHETIC_ONBOARDING_RUNNING"
EXCEPTIONS_REQUIRE_INPUT = "EXCEPTIONS_REQUIRE_INPUT"
SYNTHETIC_ONBOARDING_PASSED = "SYNTHETIC_ONBOARDING_PASSED"
ORCHESTRATION_PLAN_GENERATED = "ORCHESTRATION_PLAN_GENERATED"
EXECUTION_APPROVAL_REQUIRED = "EXECUTION_APPROVAL_REQUIRED"
READY_FOR_EXECUTION = "READY_FOR_EXECUTION"
BLOCKED = "BLOCKED"
CANCELLED = "CANCELLED"

#: Terminal for this feature. Handing the package to the live pipeline is
#: deferred functionality.
TERMINAL_STATES = (READY_FOR_EXECUTION, CANCELLED)

# --------------------------------------------------------------------------- #
# Human actions the interpreter maps natural language onto
# --------------------------------------------------------------------------- #

# -- onboarding half: these delegate straight to OnboardingService ---------- #
ACTION_ANSWER = "answer_onboarding_question"
ACTION_REQUEST_INFORMATION = "request_client_information"
ACTION_RECORD_RESPONSE = "record_client_response"
ACTION_SUBMIT_FOR_APPROVAL = "submit_for_approval"
ACTION_APPROVE_ONBOARDING = "approve_onboarding"
ACTION_REQUEST_CHANGES = "request_changes"
ACTION_WITHDRAW = "withdraw_case"

# -- execution half: these are this module's own ---------------------------- #
ACTION_REGISTER_ARTEFACT = "register_synthetic_artefact"
ACTION_RUN_ONBOARDING = "run_synthetic_onboarding"
ACTION_RESOLVE_DECISION = "resolve_decision"
ACTION_ACKNOWLEDGE_EXCEPTION = "acknowledge_exception"
ACTION_GENERATE_PLAN = "generate_orchestration_plan"
ACTION_APPROVE_EXECUTION = "approve_execution_readiness"
ACTION_CANCEL = "cancel_run"
ACTION_ASK = "ask"                       # a question; changes no state

#: Available wherever the run is (they never move it).
UNIVERSAL_ACTIONS = (ACTION_ASK,)

#: Actions handled by Client Onboarding rather than by the execution state
#: machine. They are legal whenever the CASE allows them, which the onboarding
#: service decides — so the execution table does not gate them.
ONBOARDING_ACTIONS = (
    ACTION_ANSWER, ACTION_REQUEST_INFORMATION, ACTION_RECORD_RESPONSE,
    ACTION_SUBMIT_FOR_APPROVAL, ACTION_APPROVE_ONBOARDING,
    ACTION_REQUEST_CHANGES, ACTION_WITHDRAW,
)


# --------------------------------------------------------------------------- #
# The table
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class StateSpec:
    """Everything that governs one execution state."""

    state: str
    label: str
    permitted_prior: Tuple[str, ...]
    required_inputs: Tuple[str, ...] = ()
    automatic_actions: Tuple[str, ...] = ()
    deterministic_controls: Tuple[str, ...] = ()
    required_approvals: Tuple[str, ...] = ()
    allowed_human_actions: Tuple[str, ...] = ()
    next_states: Tuple[str, ...] = ()
    blocking_conditions: Tuple[str, ...] = ()
    #: The OCC operator stage this sits over, when there is one — used to
    #: deep-link to the existing OCC views rather than reproducing them.
    occ_stage: str = ""


_ALWAYS_NEXT = (BLOCKED, CANCELLED)

STATE_SPECS: Dict[str, StateSpec] = {
    AWAITING_ONBOARDING: StateSpec(
        state=AWAITING_ONBOARDING,
        label="Working through onboarding",
        permitted_prior=(),
        required_inputs=("an onboarding case",),
        automatic_actions=("follow the onboarding case's own status",),
        allowed_human_actions=(ACTION_REGISTER_ARTEFACT, ACTION_CANCEL),
        next_states=(READY_TO_RUN,) + _ALWAYS_NEXT,
        blocking_conditions=("The onboarding has not been approved yet.",),
    ),
    READY_TO_RUN: StateSpec(
        state=READY_TO_RUN,
        label="Ready to run",
        permitted_prior=(AWAITING_ONBOARDING, EXCEPTIONS_REQUIRE_INPUT),
        required_inputs=("an approved onboarding case",
                         "at least one synthetic artefact"),
        deterministic_controls=("the onboarding case reports no blocking "
                               "problems",),
        allowed_human_actions=(ACTION_RUN_ONBOARDING, ACTION_REGISTER_ARTEFACT,
                               ACTION_CANCEL),
        next_states=(SYNTHETIC_ONBOARDING_RUNNING,) + _ALWAYS_NEXT,
    ),
    SYNTHETIC_ONBOARDING_RUNNING: StateSpec(
        state=SYNTHETIC_ONBOARDING_RUNNING,
        label="Synthetic onboarding running",
        permitted_prior=(READY_TO_RUN, EXCEPTIONS_REQUIRE_INPUT),
        required_inputs=("an approved configuration", "classified artefacts"),
        automatic_actions=("run the existing orchestration conductor over the "
                           "synthetic execution adapter",),
        deterministic_controls=("source profiling", "header mapping",
                               "canonical transformation",
                               "canonical + business-rule validation",
                               "materiality assessment",
                               "provenance stamping", "assembly"),
        allowed_human_actions=(ACTION_CANCEL,),
        next_states=(SYNTHETIC_ONBOARDING_PASSED, EXCEPTIONS_REQUIRE_INPUT)
                    + _ALWAYS_NEXT,
        occ_stage=STAGE_MAPPING,
    ),
    EXCEPTIONS_REQUIRE_INPUT: StateSpec(
        state=EXCEPTIONS_REQUIRE_INPUT,
        label="Exceptions need your input",
        permitted_prior=(SYNTHETIC_ONBOARDING_RUNNING,),
        required_inputs=("open decisions from the synthetic run",),
        automatic_actions=("present each open decision as a decision card",),
        required_approvals=("each blocking decision",),
        allowed_human_actions=(ACTION_RESOLVE_DECISION,
                               ACTION_ACKNOWLEDGE_EXCEPTION,
                               ACTION_RUN_ONBOARDING, ACTION_ANSWER,
                               ACTION_CANCEL),
        next_states=(SYNTHETIC_ONBOARDING_RUNNING, READY_TO_RUN)
                    + _ALWAYS_NEXT,
        blocking_conditions=("A blocking decision is unresolved.",),
        occ_stage=STAGE_VALIDATION,
    ),
    SYNTHETIC_ONBOARDING_PASSED: StateSpec(
        state=SYNTHETIC_ONBOARDING_PASSED,
        label="Synthetic onboarding passed",
        permitted_prior=(SYNTHETIC_ONBOARDING_RUNNING,),
        required_inputs=("a completed synthetic run with no blocking "
                         "exceptions",),
        deterministic_controls=("no blocking control failures remain",),
        allowed_human_actions=(ACTION_GENERATE_PLAN, ACTION_CANCEL),
        next_states=(ORCHESTRATION_PLAN_GENERATED,) + _ALWAYS_NEXT,
        occ_stage=STAGE_ASSEMBLY,
    ),
    ORCHESTRATION_PLAN_GENERATED: StateSpec(
        state=ORCHESTRATION_PLAN_GENERATED,
        label="Execution plan prepared",
        permitted_prior=(SYNTHETIC_ONBOARDING_PASSED,),
        required_inputs=("a passed synthetic run",),
        automatic_actions=("generate the orchestration execution plan",
                           "validate the assembler prerequisites",
                           "generate the intended run manifest"),
        deterministic_controls=("orchestration sequencing valid",
                               "assembler prerequisites satisfied",
                               "intended blob paths valid"),
        allowed_human_actions=(ACTION_APPROVE_EXECUTION, ACTION_CANCEL),
        next_states=(EXECUTION_APPROVAL_REQUIRED,) + _ALWAYS_NEXT,
        occ_stage=STAGE_ASSEMBLY,
    ),
    EXECUTION_APPROVAL_REQUIRED: StateSpec(
        state=EXECUTION_APPROVAL_REQUIRED,
        label="Readiness needs your approval",
        permitted_prior=(ORCHESTRATION_PLAN_GENERATED,),
        required_inputs=("orchestration execution plan",),
        required_approvals=("execution_readiness",),
        allowed_human_actions=(ACTION_APPROVE_EXECUTION, ACTION_CANCEL),
        next_states=(READY_FOR_EXECUTION,) + _ALWAYS_NEXT,
        blocking_conditions=("Readiness has not been approved.",),
    ),
    READY_FOR_EXECUTION: StateSpec(
        state=READY_FOR_EXECUTION,
        label="READY_FOR_EXECUTION",
        permitted_prior=(EXECUTION_APPROVAL_REQUIRED,),
        required_inputs=("every readiness criterion satisfied",),
        automatic_actions=("generate the readiness package",),
        deterministic_controls=("all readiness criteria evaluated "
                               "deterministically",),
        allowed_human_actions=(),
        next_states=(),
    ),
    BLOCKED: StateSpec(
        state=BLOCKED,
        label="Blocked",
        # Reachable from anywhere; what may FOLLOW a block is what a human can
        # legitimately return the run to.
        permitted_prior=(),
        automatic_actions=("record why the run is blocked",),
        allowed_human_actions=(ACTION_RESOLVE_DECISION,
                               ACTION_REGISTER_ARTEFACT, ACTION_ANSWER,
                               ACTION_RUN_ONBOARDING, ACTION_CANCEL),
        next_states=(AWAITING_ONBOARDING, READY_TO_RUN,
                     SYNTHETIC_ONBOARDING_RUNNING, EXCEPTIONS_REQUIRE_INPUT,
                     CANCELLED),
        blocking_conditions=("See the recorded blockers on the run.",),
    ),
    CANCELLED: StateSpec(
        state=CANCELLED,
        label="Cancelled",
        permitted_prior=(),
        allowed_human_actions=(),
        next_states=(),
    ),
}

#: Declaration order, which is also operator display order.
STATE_ORDER: Tuple[str, ...] = tuple(STATE_SPECS.keys())


class IllegalRunTransition(OpsError):
    """An execution-state transition the lifecycle does not permit."""

    def __init__(self, from_state: str, to_state: str):
        self.from_state, self.to_state = from_state, to_state
        super().__init__(
            "OCC_AGENT_ILLEGAL_TRANSITION",
            f"This practice case cannot move from {spec_label(from_state)} to "
            f"{spec_label(to_state)}.",
            http_status=409)


def spec_label(state: str) -> str:
    spec_ = STATE_SPECS.get(state)
    return spec_.label if spec_ else str(state)


def spec(state: str) -> StateSpec:
    s = STATE_SPECS.get(state)
    if s is None:
        raise IllegalRunTransition(state, state)
    return s


def permitted_next(state: str) -> Tuple[str, ...]:
    return spec(state).next_states


def is_transition_allowed(from_state: str, to_state: str) -> bool:
    """Both directions must agree.

    ``next_states`` says where a state may go; ``permitted_prior`` says what may
    precede it. Requiring both means a state added to one table but not the
    other is refused rather than silently permitted.
    """
    if from_state == to_state:
        return False
    target = STATE_SPECS.get(to_state)
    source = STATE_SPECS.get(from_state)
    if target is None or source is None:
        return False
    if to_state not in source.next_states:
        return False
    if to_state in (BLOCKED, CANCELLED):
        return True
    if from_state == BLOCKED:
        # Returning from BLOCKED is governed by BLOCKED.next_states alone.
        return True
    return from_state in target.permitted_prior


def assert_transition(from_state: str, to_state: str) -> None:
    if not is_transition_allowed(from_state, to_state):
        raise IllegalRunTransition(from_state, to_state)


def action_allowed(state: str, action: str) -> bool:
    if action in UNIVERSAL_ACTIONS:
        return True
    if action in ONBOARDING_ACTIONS:
        # Client Onboarding's own transition table decides these.
        return True
    return action in spec(state).allowed_human_actions


def describe(state: str) -> Dict[str, object]:
    """The state's full contract, for the API and the UI."""
    s = spec(state)
    return {
        "state": s.state,
        "label": s.label,
        "permitted_prior": list(s.permitted_prior),
        "required_inputs": list(s.required_inputs),
        "automatic_actions": list(s.automatic_actions),
        "deterministic_controls": list(s.deterministic_controls),
        "required_approvals": list(s.required_approvals),
        "allowed_human_actions": list(s.allowed_human_actions),
        "next_states": list(s.next_states),
        "blocking_conditions": list(s.blocking_conditions),
        "occ_stage": s.occ_stage,
        "terminal": s.state in TERMINAL_STATES,
    }


def lifecycle() -> list:
    """The whole table, in display order. The UI renders progress from this."""
    return [describe(s) for s in STATE_ORDER]
