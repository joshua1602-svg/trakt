"""operations_control.occ_agent.states — the OCC Agent's own lifecycle.

Client Onboarding owns the *case* lifecycle: ``draft → information_requested →
awaiting_client → in_review → ready_for_approval → approved → activated``, with
its own transition table in :mod:`operations_control.onboarding.case`. This
module does **not** restate it, and a test asserts no onboarding status appears
here.

What it covers is everything Client Onboarding has no concept of, because it
stops at activation and knows nothing about running a pipeline: preparing and
issuing the client's pack, rehearsing the delivery, reviewing the result, and —
after an explicit human decision — activating and starting ingestion.

    AWAITING_ONBOARDING           the case is not approved yet
    PACK_DRAFTED                  the client-facing pack exists in draft
    PACK_REVIEW_REQUIRED          a human must read it before it goes out
    PACK_APPROVED_TO_SEND         approved; not yet issued
    PACK_SENT                     recorded as issued to the client
    READY_TO_RUN                  approved case, artefacts present
    SYNTHETIC_ONBOARDING_RUNNING
    EXCEPTIONS_REQUIRE_INPUT      a control needs a human
    SYNTHETIC_ONBOARDING_PASSED
    ORCHESTRATION_PLAN_GENERATED
    EXECUTION_APPROVAL_REQUIRED
    READY_FOR_EXECUTION           the rehearsal passed — a WAYPOINT
    READY_FOR_REVIEW              the review package is complete
    APPROVED_FOR_ACTIVATION       a human approved activation
    ACTIVATION_CONFIRMATION_REQUIRED   the last, explicit confirmation
    ACTIVATING                    the governed activation is running
    INGESTION_STARTED             the existing Onboarding Agent has the work
    ACTIVATION_FAILED
    BLOCKED
    CANCELLED

``READY_FOR_EXECUTION`` used to be terminal. It is not the end of the operating
process — it is the point at which a rehearsal has passed and a human can be
asked to approve the real thing. The terminal states are ``INGESTION_STARTED``
and ``CANCELLED``.

Two properties hold. The transition check lives here rather than in the caller,
so an invalid move cannot be reached from the API, the UI or a sentence; and
the interpreter never decides a transition — it proposes an action, controls
run, and the resulting state is asserted here.
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
PACK_DRAFTED = "PACK_DRAFTED"
PACK_REVIEW_REQUIRED = "PACK_REVIEW_REQUIRED"
PACK_APPROVED_TO_SEND = "PACK_APPROVED_TO_SEND"
PACK_SENT = "PACK_SENT"
READY_TO_RUN = "READY_TO_RUN"
SYNTHETIC_ONBOARDING_RUNNING = "SYNTHETIC_ONBOARDING_RUNNING"
EXCEPTIONS_REQUIRE_INPUT = "EXCEPTIONS_REQUIRE_INPUT"
SYNTHETIC_ONBOARDING_PASSED = "SYNTHETIC_ONBOARDING_PASSED"
ORCHESTRATION_PLAN_GENERATED = "ORCHESTRATION_PLAN_GENERATED"
EXECUTION_APPROVAL_REQUIRED = "EXECUTION_APPROVAL_REQUIRED"
READY_FOR_EXECUTION = "READY_FOR_EXECUTION"
READY_FOR_REVIEW = "READY_FOR_REVIEW"
APPROVED_FOR_ACTIVATION = "APPROVED_FOR_ACTIVATION"
ACTIVATION_CONFIRMATION_REQUIRED = "ACTIVATION_CONFIRMATION_REQUIRED"
ACTIVATING = "ACTIVATING"
INGESTION_STARTED = "INGESTION_STARTED"
ACTIVATION_FAILED = "ACTIVATION_FAILED"
BLOCKED = "BLOCKED"
CANCELLED = "CANCELLED"

#: The end of the operating process. Readiness is a waypoint, not a finish.
TERMINAL_STATES = (INGESTION_STARTED, CANCELLED)

#: States in which the client's pack is being prepared or issued.
PACK_STATES = (PACK_DRAFTED, PACK_REVIEW_REQUIRED, PACK_APPROVED_TO_SEND,
               PACK_SENT)

#: States after the rehearsal, where activation is the subject.
ACTIVATION_STATES = (READY_FOR_REVIEW, APPROVED_FOR_ACTIVATION,
                     ACTIVATION_CONFIRMATION_REQUIRED, ACTIVATING,
                     INGESTION_STARTED, ACTIVATION_FAILED)

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

# -- the client pack --------------------------------------------------------- #
ACTION_DRAFT_PACK = "draft_onboarding_pack"
ACTION_APPROVE_PACK = "approve_pack_to_send"
ACTION_SEND_PACK = "send_onboarding_pack"

# -- the rehearsal ----------------------------------------------------------- #
ACTION_REGISTER_ARTEFACT = "register_synthetic_artefact"
ACTION_RUN_ONBOARDING = "run_synthetic_onboarding"
ACTION_RESOLVE_DECISION = "resolve_decision"
ACTION_ACKNOWLEDGE_EXCEPTION = "acknowledge_exception"
ACTION_GENERATE_PLAN = "generate_orchestration_plan"
ACTION_APPROVE_EXECUTION = "approve_execution_readiness"

# -- activation -------------------------------------------------------------- #
ACTION_REQUEST_ACTIVATION = "request_activation"
ACTION_APPROVE_ACTIVATION = "approve_activation"
ACTION_CONFIRM_ACTIVATION = "confirm_activation"

ACTION_CANCEL = "cancel_run"
ACTION_ASK = "ask"                       # a question; changes no state

#: Available wherever the run is (they never move it).
UNIVERSAL_ACTIONS = (ACTION_ASK,)

#: Actions handled by Client Onboarding rather than by this table. They are
#: legal whenever the CASE allows them, which the onboarding service decides —
#: so the execution table does not second-guess it.
ONBOARDING_ACTIONS = (
    ACTION_ANSWER, ACTION_REQUEST_INFORMATION, ACTION_RECORD_RESPONSE,
    ACTION_SUBMIT_FOR_APPROVAL, ACTION_APPROVE_ONBOARDING,
    ACTION_REQUEST_CHANGES, ACTION_WITHDRAW,
)

#: Everything this table governs.
EXECUTION_ACTIONS = (
    ACTION_DRAFT_PACK, ACTION_APPROVE_PACK, ACTION_SEND_PACK,
    ACTION_REGISTER_ARTEFACT, ACTION_RUN_ONBOARDING, ACTION_RESOLVE_DECISION,
    ACTION_ACKNOWLEDGE_EXCEPTION, ACTION_GENERATE_PLAN,
    ACTION_APPROVE_EXECUTION, ACTION_REQUEST_ACTIVATION,
    ACTION_APPROVE_ACTIVATION, ACTION_CONFIRM_ACTIVATION,
    ACTION_CANCEL, ACTION_ASK,
)


# --------------------------------------------------------------------------- #
# The table
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class StateSpec:
    """Everything that governs one state."""

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
    #: The OCC operator stage this sits over, when there is one — used to deep
    #: link to the existing OCC views rather than reproducing them.
    occ_stage: str = ""


_ALWAYS_NEXT = (BLOCKED, CANCELLED)

#: Preparing the pack never stops an operator providing artefacts.
_PACK_ACTIONS = (ACTION_DRAFT_PACK, ACTION_REGISTER_ARTEFACT, ACTION_CANCEL)

STATE_SPECS: Dict[str, StateSpec] = {
    AWAITING_ONBOARDING: StateSpec(
        state=AWAITING_ONBOARDING,
        label="Working through onboarding",
        permitted_prior=(),
        required_inputs=("an onboarding case",),
        automatic_actions=("follow the onboarding case's own status",),
        allowed_human_actions=_PACK_ACTIONS,
        next_states=(PACK_DRAFTED, READY_TO_RUN) + _ALWAYS_NEXT,
        blocking_conditions=("The onboarding has not been approved yet.",),
    ),
    PACK_DRAFTED: StateSpec(
        state=PACK_DRAFTED,
        label="Onboarding pack drafted",
        permitted_prior=(AWAITING_ONBOARDING, PACK_SENT,
                         PACK_REVIEW_REQUIRED),
        required_inputs=("the catalogue's outstanding client questions",),
        automatic_actions=("project the governed catalogue into a "
                           "client-facing pack", "draft the covering email"),
        allowed_human_actions=(ACTION_DRAFT_PACK, ACTION_APPROVE_PACK,
                               ACTION_REGISTER_ARTEFACT, ACTION_CANCEL),
        # READY_TO_RUN is reachable from every pack state: an operator who
        # already holds the client's answers must not be made to issue a pack
        # they do not need.
        next_states=(PACK_REVIEW_REQUIRED, READY_TO_RUN) + _ALWAYS_NEXT,
    ),
    PACK_REVIEW_REQUIRED: StateSpec(
        state=PACK_REVIEW_REQUIRED,
        label="Pack needs your review",
        permitted_prior=(PACK_DRAFTED,),
        required_inputs=("a drafted pack and covering email",),
        required_approvals=("the pack, before anything leaves Trakt",),
        allowed_human_actions=(ACTION_APPROVE_PACK, ACTION_DRAFT_PACK,
                               ACTION_REGISTER_ARTEFACT, ACTION_CANCEL),
        next_states=(PACK_APPROVED_TO_SEND, PACK_DRAFTED, READY_TO_RUN)
                    + _ALWAYS_NEXT,
        blocking_conditions=("Nothing is sent until a human approves it.",),
    ),
    PACK_APPROVED_TO_SEND: StateSpec(
        state=PACK_APPROVED_TO_SEND,
        label="Pack approved to send",
        permitted_prior=(PACK_REVIEW_REQUIRED,),
        required_approvals=("recorded, with who approved it and when",),
        allowed_human_actions=(ACTION_SEND_PACK, ACTION_REGISTER_ARTEFACT,
                               ACTION_CANCEL),
        next_states=(PACK_SENT, READY_TO_RUN) + _ALWAYS_NEXT,
    ),
    PACK_SENT: StateSpec(
        state=PACK_SENT,
        label="Pack issued to the client",
        permitted_prior=(PACK_APPROVED_TO_SEND,),
        automatic_actions=("record what was issued, to whom, and by which "
                           "channel",),
        allowed_human_actions=(ACTION_REGISTER_ARTEFACT, ACTION_DRAFT_PACK,
                               ACTION_CANCEL),
        next_states=(READY_TO_RUN, PACK_DRAFTED) + _ALWAYS_NEXT,
    ),
    READY_TO_RUN: StateSpec(
        state=READY_TO_RUN,
        label="Ready to run",
        permitted_prior=(AWAITING_ONBOARDING, PACK_DRAFTED,
                         PACK_REVIEW_REQUIRED, PACK_APPROVED_TO_SEND,
                         PACK_SENT, EXCEPTIONS_REQUIRE_INPUT),
        required_inputs=("an approved onboarding case",
                         "at least one artefact"),
        deterministic_controls=("the onboarding case reports no blocking "
                               "problems",),
        allowed_human_actions=(ACTION_RUN_ONBOARDING, ACTION_REGISTER_ARTEFACT,
                               ACTION_CANCEL),
        next_states=(SYNTHETIC_ONBOARDING_RUNNING,) + _ALWAYS_NEXT,
    ),
    SYNTHETIC_ONBOARDING_RUNNING: StateSpec(
        state=SYNTHETIC_ONBOARDING_RUNNING,
        label="Rehearsal running",
        permitted_prior=(READY_TO_RUN, EXCEPTIONS_REQUIRE_INPUT),
        required_inputs=("an approved configuration", "classified artefacts"),
        automatic_actions=("run the existing orchestration conductor over the "
                           "execution adapter",),
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
        required_inputs=("open decisions from the run",),
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
        label="Rehearsal passed",
        permitted_prior=(SYNTHETIC_ONBOARDING_RUNNING,),
        required_inputs=("a completed run with no blocking exceptions",),
        deterministic_controls=("no blocking control failures remain",),
        allowed_human_actions=(ACTION_GENERATE_PLAN, ACTION_CANCEL),
        next_states=(ORCHESTRATION_PLAN_GENERATED,) + _ALWAYS_NEXT,
        occ_stage=STAGE_ASSEMBLY,
    ),
    ORCHESTRATION_PLAN_GENERATED: StateSpec(
        state=ORCHESTRATION_PLAN_GENERATED,
        label="Execution plan prepared",
        permitted_prior=(SYNTHETIC_ONBOARDING_PASSED,),
        required_inputs=("a passed run",),
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
        automatic_actions=("generate the readiness package",
                           "assemble the human review package"),
        deterministic_controls=("all readiness criteria evaluated "
                               "deterministically",),
        # A WAYPOINT: the rehearsal passed. What follows is a human decision
        # about the real thing.
        allowed_human_actions=(ACTION_REQUEST_ACTIVATION, ACTION_CANCEL),
        next_states=(READY_FOR_REVIEW,) + _ALWAYS_NEXT,
    ),
    READY_FOR_REVIEW: StateSpec(
        state=READY_FOR_REVIEW,
        label="Ready for review and approval",
        permitted_prior=(READY_FOR_EXECUTION,),
        required_inputs=("the complete human review package",),
        automatic_actions=("present the review package",),
        required_approvals=("activation, by a human",),
        allowed_human_actions=(ACTION_APPROVE_ACTIVATION, ACTION_CANCEL),
        next_states=(APPROVED_FOR_ACTIVATION,) + _ALWAYS_NEXT,
        blocking_conditions=("Activation has not been approved.",),
    ),
    APPROVED_FOR_ACTIVATION: StateSpec(
        state=APPROVED_FOR_ACTIVATION,
        label="Approved for activation",
        permitted_prior=(READY_FOR_REVIEW,),
        required_approvals=("recorded, attributed and audited",),
        automatic_actions=("prepare the activation confirmation",),
        # Approval is not the trigger. A separate, explicit confirmation is.
        allowed_human_actions=(ACTION_CONFIRM_ACTIVATION, ACTION_CANCEL),
        next_states=(ACTIVATION_CONFIRMATION_REQUIRED,) + _ALWAYS_NEXT,
    ),
    ACTIVATION_CONFIRMATION_REQUIRED: StateSpec(
        state=ACTIVATION_CONFIRMATION_REQUIRED,
        label="Confirm activation",
        permitted_prior=(APPROVED_FOR_ACTIVATION, ACTIVATION_FAILED),
        required_inputs=("the client, portfolio, files, target locations and "
                         "the actions that will occur",),
        required_approvals=("an explicit confirmation naming what will "
                            "happen",),
        allowed_human_actions=(ACTION_CONFIRM_ACTIVATION, ACTION_CANCEL),
        next_states=(ACTIVATING,) + _ALWAYS_NEXT,
        blocking_conditions=("Production is not started by an approval "
                             "alone.",),
    ),
    ACTIVATING: StateSpec(
        state=ACTIVATING,
        label="Activating",
        permitted_prior=(ACTIVATION_CONFIRMATION_REQUIRED,),
        automatic_actions=("write the approved configuration",
                           "register the source artefacts",
                           "place the files in the approved location",
                           "start the existing Onboarding Agent"),
        deterministic_controls=("every activation precondition, checked "
                               "together",),
        allowed_human_actions=(),
        next_states=(INGESTION_STARTED, ACTIVATION_FAILED) + _ALWAYS_NEXT,
    ),
    INGESTION_STARTED: StateSpec(
        state=INGESTION_STARTED,
        label="Ingestion started",
        permitted_prior=(ACTIVATING,),
        automatic_actions=("hand off to the existing orchestration",),
        allowed_human_actions=(),
        next_states=(),
        occ_stage=STAGE_MAPPING,
    ),
    ACTIVATION_FAILED: StateSpec(
        state=ACTIVATION_FAILED,
        label="Activation failed",
        permitted_prior=(ACTIVATING,),
        automatic_actions=("record what failed, and what had already "
                           "happened",),
        allowed_human_actions=(ACTION_CONFIRM_ACTIVATION, ACTION_CANCEL),
        next_states=(ACTIVATION_CONFIRMATION_REQUIRED,) + _ALWAYS_NEXT,
        blocking_conditions=("See the recorded failure on the run.",),
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
                               ACTION_RUN_ONBOARDING, ACTION_DRAFT_PACK,
                               ACTION_CANCEL),
        next_states=(AWAITING_ONBOARDING, PACK_DRAFTED, READY_TO_RUN,
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
    """A transition the lifecycle does not permit."""

    def __init__(self, from_state: str, to_state: str):
        self.from_state, self.to_state = from_state, to_state
        super().__init__(
            "OCC_AGENT_ILLEGAL_TRANSITION",
            f"This case cannot move from {spec_label(from_state)} to "
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

    ``next_states`` says where a state may go; ``permitted_prior`` says what
    may precede it. Requiring both means a state added to one table but not the
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
