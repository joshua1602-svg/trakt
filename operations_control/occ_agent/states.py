"""operations_control.occ_agent.states — the synthetic onboarding lifecycle.

One table, :data:`STATE_SPECS`, defines every state: which states may precede
it, what it needs, what happens automatically, which deterministic controls
apply, which approvals it requires, what a human may do there, where it can go
next, and what blocks it. The table is the contract — the API, the service and
the UI all read it, so there is one place to change and one place to test.

Two properties matter more than the list itself:

* **The transition check is here, not in the caller.** :func:`assert_transition`
  refuses an illegal move with a typed error, so an invalid transition cannot be
  reached from the API, the UI, or a natural-language instruction.
* **The model never decides a transition.** The interpreter proposes an
  *action*; the service applies deterministic controls and then asks this module
  whether the resulting state is legal. There is deliberately no function here
  that takes free text.

Terminology follows the OCC where the OCC has a word for something: gates are
named after the existing operator stages (``STAGE_ORDER`` in
:mod:`operations_control.contracts`), and approvals reuse the OCC decision
vocabulary. The case states themselves are new because a synthetic *case*
(instruction → readiness) spans more than one live workflow run does.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

from ..contracts import (
    STAGE_ASSEMBLY,
    STAGE_MAPPING,
    STAGE_RECEIVED,
    STAGE_UNDERSTANDING,
    STAGE_VALIDATION,
)
from ..engine import OpsError

# --------------------------------------------------------------------------- #
# The states
# --------------------------------------------------------------------------- #

CASE_CREATED = "CASE_CREATED"
REQUIREMENTS_EXTRACTED = "REQUIREMENTS_EXTRACTED"
REQUIREMENTS_CONFIRMATION_REQUIRED = "REQUIREMENTS_CONFIRMATION_REQUIRED"
REQUIREMENTS_CONFIRMED = "REQUIREMENTS_CONFIRMED"
ONBOARDING_PACK_GENERATED = "ONBOARDING_PACK_GENERATED"
ONBOARDING_PACK_APPROVAL_REQUIRED = "ONBOARDING_PACK_APPROVAL_REQUIRED"
ONBOARDING_PACK_ISSUED_SYNTHETICALLY = "ONBOARDING_PACK_ISSUED_SYNTHETICALLY"
AWAITING_CLIENT_INPUT = "AWAITING_CLIENT_INPUT"
ARTEFACTS_RECEIVED = "ARTEFACTS_RECEIVED"
ARTEFACTS_CLASSIFIED = "ARTEFACTS_CLASSIFIED"
CONFIG_DRAFTED = "CONFIG_DRAFTED"
CONFIG_APPROVAL_REQUIRED = "CONFIG_APPROVAL_REQUIRED"
CONFIG_APPROVED = "CONFIG_APPROVED"
SYNTHETIC_ONBOARDING_RUNNING = "SYNTHETIC_ONBOARDING_RUNNING"
EXCEPTIONS_REQUIRE_INPUT = "EXCEPTIONS_REQUIRE_INPUT"
SYNTHETIC_ONBOARDING_PASSED = "SYNTHETIC_ONBOARDING_PASSED"
ORCHESTRATION_PLAN_GENERATED = "ORCHESTRATION_PLAN_GENERATED"
EXECUTION_APPROVAL_REQUIRED = "EXECUTION_APPROVAL_REQUIRED"
READY_FOR_EXECUTION = "READY_FOR_EXECUTION"
BLOCKED = "BLOCKED"
CANCELLED = "CANCELLED"

#: The terminal states. ``READY_FOR_EXECUTION`` is terminal for this feature:
#: handing the package to the live pipeline is deferred functionality.
TERMINAL_STATES = (READY_FOR_EXECUTION, CANCELLED)

# --------------------------------------------------------------------------- #
# Human actions (the vocabulary the interpreter maps natural language onto)
# --------------------------------------------------------------------------- #

ACTION_CONFIRM_REQUIREMENTS = "confirm_requirements"
ACTION_CORRECT_REQUIREMENTS = "correct_requirements"
ACTION_GENERATE_PACK = "generate_onboarding_pack"
ACTION_APPROVE_PACK = "approve_onboarding_pack"
ACTION_ISSUE_PACK = "issue_onboarding_pack_synthetically"
ACTION_REGISTER_ARTEFACT = "register_synthetic_artefact"
ACTION_CLASSIFY_ARTEFACTS = "classify_artefacts"
ACTION_DRAFT_CONFIG = "draft_client_config"
ACTION_APPROVE_CONFIG = "approve_client_config"
ACTION_CORRECT_CONFIG = "correct_client_config"
ACTION_RUN_ONBOARDING = "run_synthetic_onboarding"
ACTION_RESOLVE_DECISION = "resolve_decision"
ACTION_ACKNOWLEDGE_EXCEPTION = "acknowledge_exception"
ACTION_GENERATE_PLAN = "generate_orchestration_plan"
ACTION_APPROVE_EXECUTION = "approve_execution_readiness"
ACTION_RETURN_TO_STAGE = "return_to_stage"
ACTION_CANCEL = "cancel_case"
ACTION_ASK = "ask"                       # a question; changes no state

#: Actions that are always available wherever the case is (they never move it).
UNIVERSAL_ACTIONS = (ACTION_ASK,)


# --------------------------------------------------------------------------- #
# The table
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class StateSpec:
    """Everything that governs one state.

    ``deterministic_controls`` names the controls that must have run and passed
    for the case to be IN this state; ``blocking_conditions`` describes, in
    operator language, what holds a case out of its next state.
    """

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
    #: The OCC operator stage this case state sits over, when there is one. Used
    #: to deep-link the conversation to the existing OCC views rather than
    #: reproducing them.
    occ_stage: str = ""


#: Every state may be cancelled, and every state may be blocked. Declared once
#: and appended to each spec's ``next_states`` so the table stays readable.
_ALWAYS_NEXT = (BLOCKED, CANCELLED)

STATE_SPECS: Dict[str, StateSpec] = {
    CASE_CREATED: StateSpec(
        state=CASE_CREATED,
        label="Case created",
        permitted_prior=(),
        required_inputs=("tenant", "initiating_user"),
        automatic_actions=("allocate immutable case id",
                           "record synthetic runtime policy"),
        allowed_human_actions=(ACTION_CORRECT_REQUIREMENTS, ACTION_CANCEL),
        next_states=(REQUIREMENTS_EXTRACTED,) + _ALWAYS_NEXT,
        blocking_conditions=("No instruction has been given yet.",),
    ),
    REQUIREMENTS_EXTRACTED: StateSpec(
        state=REQUIREMENTS_EXTRACTED,
        label="Interpretation proposed",
        permitted_prior=(CASE_CREATED, REQUIREMENTS_CONFIRMED,
                         REQUIREMENTS_CONFIRMATION_REQUIRED),
        required_inputs=("instruction",),
        automatic_actions=("extract structured facts from the instruction",
                           "list unresolved questions"),
        deterministic_controls=("extracted facts validate against the case "
                                "schema",),
        allowed_human_actions=(ACTION_CORRECT_REQUIREMENTS,
                               ACTION_CONFIRM_REQUIREMENTS, ACTION_CANCEL),
        next_states=(REQUIREMENTS_CONFIRMATION_REQUIRED,) + _ALWAYS_NEXT,
        blocking_conditions=("The interpretation has not been shown for "
                             "confirmation yet.",),
    ),
    REQUIREMENTS_CONFIRMATION_REQUIRED: StateSpec(
        state=REQUIREMENTS_CONFIRMATION_REQUIRED,
        label="Interpretation needs your confirmation",
        permitted_prior=(REQUIREMENTS_EXTRACTED,),
        required_inputs=("extracted_requirements",),
        automatic_actions=("present the proposed interpretation",),
        required_approvals=("requirements",),
        allowed_human_actions=(ACTION_CORRECT_REQUIREMENTS,
                               ACTION_CONFIRM_REQUIREMENTS, ACTION_CANCEL),
        next_states=(REQUIREMENTS_CONFIRMED, REQUIREMENTS_EXTRACTED)
                    + _ALWAYS_NEXT,
        blocking_conditions=("The proposed interpretation has not been "
                             "confirmed.",),
    ),
    REQUIREMENTS_CONFIRMED: StateSpec(
        state=REQUIREMENTS_CONFIRMED,
        label="Requirements confirmed",
        permitted_prior=(REQUIREMENTS_CONFIRMATION_REQUIRED,),
        required_inputs=("confirmed_requirements",),
        deterministic_controls=("mandatory facts present (client, portfolio, "
                                "asset type, products)",),
        allowed_human_actions=(ACTION_GENERATE_PACK,
                               ACTION_CORRECT_REQUIREMENTS,
                               ACTION_RETURN_TO_STAGE, ACTION_CANCEL),
        next_states=(ONBOARDING_PACK_GENERATED, REQUIREMENTS_EXTRACTED)
                    + _ALWAYS_NEXT,
        blocking_conditions=("A mandatory fact is still missing.",),
    ),
    ONBOARDING_PACK_GENERATED: StateSpec(
        state=ONBOARDING_PACK_GENERATED,
        label="Onboarding pack drafted",
        permitted_prior=(REQUIREMENTS_CONFIRMED,
                         ONBOARDING_PACK_APPROVAL_REQUIRED),
        required_inputs=("confirmed_requirements",),
        automatic_actions=("determine required questions from the onboarding "
                           "configuration framework",
                           "draft the client email and artefact requests"),
        deterministic_controls=("required input roles resolved from "
                                "workflow_input_requirements.yaml",),
        allowed_human_actions=(ACTION_APPROVE_PACK,
                               ACTION_CORRECT_REQUIREMENTS,
                               ACTION_GENERATE_PACK, ACTION_CANCEL),
        next_states=(ONBOARDING_PACK_APPROVAL_REQUIRED,) + _ALWAYS_NEXT,
    ),
    ONBOARDING_PACK_APPROVAL_REQUIRED: StateSpec(
        state=ONBOARDING_PACK_APPROVAL_REQUIRED,
        label="Onboarding pack needs your approval",
        permitted_prior=(ONBOARDING_PACK_GENERATED,),
        required_inputs=("onboarding_pack",),
        required_approvals=("onboarding_pack",),
        allowed_human_actions=(ACTION_APPROVE_PACK, ACTION_GENERATE_PACK,
                               ACTION_CORRECT_REQUIREMENTS, ACTION_CANCEL),
        next_states=(ONBOARDING_PACK_ISSUED_SYNTHETICALLY,
                     ONBOARDING_PACK_GENERATED) + _ALWAYS_NEXT,
        blocking_conditions=("The onboarding pack has not been approved.",),
    ),
    ONBOARDING_PACK_ISSUED_SYNTHETICALLY: StateSpec(
        state=ONBOARDING_PACK_ISSUED_SYNTHETICALLY,
        label="Pack recorded as issued (synthetically)",
        permitted_prior=(ONBOARDING_PACK_APPROVAL_REQUIRED,),
        required_inputs=("approved onboarding pack",),
        automatic_actions=("record the pack as synthetically issued — no email "
                           "is sent",),
        allowed_human_actions=(ACTION_REGISTER_ARTEFACT, ACTION_CANCEL),
        next_states=(AWAITING_CLIENT_INPUT,) + _ALWAYS_NEXT,
    ),
    AWAITING_CLIENT_INPUT: StateSpec(
        state=AWAITING_CLIENT_INPUT,
        label="Waiting for the client response",
        permitted_prior=(ONBOARDING_PACK_ISSUED_SYNTHETICALLY,
                         ARTEFACTS_RECEIVED),
        automatic_actions=("list the artefacts the pack asked for",),
        allowed_human_actions=(ACTION_REGISTER_ARTEFACT,
                               ACTION_RETURN_TO_STAGE, ACTION_CANCEL),
        next_states=(ARTEFACTS_RECEIVED,) + _ALWAYS_NEXT,
        blocking_conditions=("No synthetic artefacts have been provided yet.",),
        occ_stage=STAGE_RECEIVED,
    ),
    ARTEFACTS_RECEIVED: StateSpec(
        state=ARTEFACTS_RECEIVED,
        label="Synthetic artefacts received",
        permitted_prior=(AWAITING_CLIENT_INPUT, ARTEFACTS_CLASSIFIED),
        required_inputs=("at least one synthetic artefact",),
        automatic_actions=("store the artefact inside the case sandbox",
                           "derive the intended live storage path"),
        deterministic_controls=("file name sanitised",
                               "intended blob path accepted by the production "
                               "path parser"),
        allowed_human_actions=(ACTION_REGISTER_ARTEFACT,
                               ACTION_CLASSIFY_ARTEFACTS, ACTION_CANCEL),
        next_states=(ARTEFACTS_CLASSIFIED,) + _ALWAYS_NEXT,
        occ_stage=STAGE_RECEIVED,
    ),
    ARTEFACTS_CLASSIFIED: StateSpec(
        state=ARTEFACTS_CLASSIFIED,
        label="Artefacts recognised",
        permitted_prior=(ARTEFACTS_RECEIVED,),
        required_inputs=("received artefacts",),
        automatic_actions=("classify each artefact into a semantic input role",),
        deterministic_controls=("role classification (apps.blob_trigger_app."
                                "file_roles)",
                               "required roles satisfied "
                               "(workflow_input_requirements.yaml)"),
        allowed_human_actions=(ACTION_DRAFT_CONFIG, ACTION_REGISTER_ARTEFACT,
                               ACTION_RESOLVE_DECISION, ACTION_CANCEL),
        next_states=(CONFIG_DRAFTED, ARTEFACTS_RECEIVED) + _ALWAYS_NEXT,
        blocking_conditions=("A required input role has not been provided.",),
        occ_stage=STAGE_UNDERSTANDING,
    ),
    CONFIG_DRAFTED: StateSpec(
        state=CONFIG_DRAFTED,
        label="Configuration proposed",
        # EXCEPTIONS_REQUIRE_INPUT is here because an exception is often
        # answered by correcting the configuration rather than the mapping —
        # which sends the case back to a fresh draft.
        permitted_prior=(ARTEFACTS_CLASSIFIED, CONFIG_APPROVAL_REQUIRED,
                         CONFIG_APPROVED, EXCEPTIONS_REQUIRE_INPUT),
        required_inputs=("classified artefacts", "confirmed requirements"),
        automatic_actions=("resolve the configuration layers with the existing "
                           "resolver", "record provenance for every value"),
        deterministic_controls=("effective configuration resolves",
                               "schema validation of the client config "
                               "candidate"),
        allowed_human_actions=(ACTION_APPROVE_CONFIG, ACTION_CORRECT_CONFIG,
                               ACTION_DRAFT_CONFIG, ACTION_CANCEL),
        next_states=(CONFIG_APPROVAL_REQUIRED,) + _ALWAYS_NEXT,
    ),
    CONFIG_APPROVAL_REQUIRED: StateSpec(
        state=CONFIG_APPROVAL_REQUIRED,
        label="Configuration needs your approval",
        permitted_prior=(CONFIG_DRAFTED,),
        required_inputs=("proposed configuration",),
        required_approvals=("client_configuration",),
        allowed_human_actions=(ACTION_APPROVE_CONFIG, ACTION_CORRECT_CONFIG,
                               ACTION_CANCEL),
        next_states=(CONFIG_APPROVED, CONFIG_DRAFTED) + _ALWAYS_NEXT,
        blocking_conditions=("The proposed configuration has not been "
                             "approved.",
                             "A material inferred value is unconfirmed."),
    ),
    CONFIG_APPROVED: StateSpec(
        state=CONFIG_APPROVED,
        label="Configuration approved",
        permitted_prior=(CONFIG_APPROVAL_REQUIRED,),
        required_inputs=("approved configuration",),
        allowed_human_actions=(ACTION_RUN_ONBOARDING, ACTION_CORRECT_CONFIG,
                               ACTION_RETURN_TO_STAGE, ACTION_CANCEL),
        next_states=(SYNTHETIC_ONBOARDING_RUNNING, CONFIG_DRAFTED)
                    + _ALWAYS_NEXT,
    ),
    SYNTHETIC_ONBOARDING_RUNNING: StateSpec(
        state=SYNTHETIC_ONBOARDING_RUNNING,
        label="Synthetic onboarding running",
        permitted_prior=(CONFIG_APPROVED, EXCEPTIONS_REQUIRE_INPUT),
        required_inputs=("approved configuration", "classified artefacts"),
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
                               ACTION_RUN_ONBOARDING, ACTION_CORRECT_CONFIG,
                               ACTION_RETURN_TO_STAGE, ACTION_CANCEL),
        next_states=(SYNTHETIC_ONBOARDING_RUNNING, CONFIG_DRAFTED)
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
        allowed_human_actions=(ACTION_GENERATE_PLAN, ACTION_RETURN_TO_STAGE,
                               ACTION_CANCEL),
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
        allowed_human_actions=(ACTION_APPROVE_EXECUTION,
                               ACTION_RETURN_TO_STAGE, ACTION_CANCEL),
        next_states=(EXECUTION_APPROVAL_REQUIRED,) + _ALWAYS_NEXT,
        occ_stage=STAGE_ASSEMBLY,
    ),
    EXECUTION_APPROVAL_REQUIRED: StateSpec(
        state=EXECUTION_APPROVAL_REQUIRED,
        label="Readiness needs your approval",
        permitted_prior=(ORCHESTRATION_PLAN_GENERATED,),
        required_inputs=("orchestration execution plan",),
        required_approvals=("execution_readiness",),
        allowed_human_actions=(ACTION_APPROVE_EXECUTION,
                               ACTION_RETURN_TO_STAGE, ACTION_CANCEL),
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
        # A case can be blocked from anywhere; the states that may FOLLOW a
        # block are the ones a human can legitimately return it to.
        permitted_prior=tuple(),
        automatic_actions=("record why the case is blocked",),
        allowed_human_actions=(ACTION_RESOLVE_DECISION,
                               ACTION_REGISTER_ARTEFACT,
                               ACTION_CORRECT_REQUIREMENTS,
                               ACTION_CORRECT_CONFIG,
                               ACTION_RETURN_TO_STAGE, ACTION_CANCEL),
        next_states=(REQUIREMENTS_EXTRACTED, AWAITING_CLIENT_INPUT,
                     ARTEFACTS_RECEIVED, ARTEFACTS_CLASSIFIED, CONFIG_DRAFTED,
                     SYNTHETIC_ONBOARDING_RUNNING, EXCEPTIONS_REQUIRE_INPUT,
                     CANCELLED),
        blocking_conditions=("See the recorded blockers on the case.",),
    ),
    CANCELLED: StateSpec(
        state=CANCELLED,
        label="Cancelled",
        permitted_prior=tuple(),
        allowed_human_actions=(),
        next_states=(),
    ),
}

#: Declaration order, which is also operator display order.
STATE_ORDER: Tuple[str, ...] = tuple(STATE_SPECS.keys())

#: The states a human may explicitly send a case BACK to (§14 "return the case
#: to an earlier permitted stage"). Deliberately excludes anything downstream of
#: a control: you may re-open a stage, never skip one.
RETURNABLE_STATES: Tuple[str, ...] = (
    REQUIREMENTS_EXTRACTED,
    ONBOARDING_PACK_GENERATED,
    AWAITING_CLIENT_INPUT,
    ARTEFACTS_RECEIVED,
    CONFIG_DRAFTED,
)


class IllegalCaseTransition(OpsError):
    """A case-state transition the lifecycle does not permit."""

    def __init__(self, from_state: str, to_state: str):
        self.from_state, self.to_state = from_state, to_state
        super().__init__(
            "OCC_AGENT_ILLEGAL_TRANSITION",
            f"This case cannot move from {spec_label(from_state)} to "
            f"{spec_label(to_state)}.",
            http_status=409)


def spec_label(state: str) -> str:
    spec = STATE_SPECS.get(state)
    return spec.label if spec else str(state)


def spec(state: str) -> StateSpec:
    s = STATE_SPECS.get(state)
    if s is None:
        raise IllegalCaseTransition(state, state)
    return s


def permitted_next(state: str) -> Tuple[str, ...]:
    return spec(state).next_states


def is_transition_allowed(from_state: str, to_state: str) -> bool:
    """Both directions must agree.

    ``next_states`` says where a state may go; ``permitted_prior`` says what may
    precede it. Requiring both means a state added to one table but not the
    other is refused rather than silently permitted — the failure mode that
    matters when the table grows.
    """
    if from_state == to_state:
        return False
    target = STATE_SPECS.get(to_state)
    source = STATE_SPECS.get(from_state)
    if target is None or source is None:
        return False
    if to_state not in source.next_states:
        return False
    # BLOCKED and CANCELLED are reachable from anywhere by declaration; their
    # permitted_prior is empty on purpose.
    if to_state in (BLOCKED, CANCELLED):
        return True
    if from_state == BLOCKED:
        # Returning from BLOCKED is governed by BLOCKED.next_states alone: the
        # target's permitted_prior lists its normal predecessor, not the block.
        return True
    return from_state in target.permitted_prior


def assert_transition(from_state: str, to_state: str) -> None:
    if not is_transition_allowed(from_state, to_state):
        raise IllegalCaseTransition(from_state, to_state)


def action_allowed(state: str, action: str) -> bool:
    if action in UNIVERSAL_ACTIONS:
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
