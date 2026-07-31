"""Natural-language interpretation, and the catalogue it writes into.

The interpreter's contract has four parts, and each is tested here:

* whatever it produces is validated against the Client Onboarding field
  catalogue, so it cannot invent a field or reach a section that does not exist;
* it is never authoritative until a human confirms it;
* it answers only what the sentence says, leaving everything Trakt works out for
  itself to Client Onboarding's own inference;
* its vocabulary comes from configuration, not from a list in the feature.
"""

from __future__ import annotations

import pytest

from operations_control.occ_agent import states as _states
from operations_control.occ_agent.interpretation import (
    PROV_AGENT_INFERENCE,
    PROV_HUMAN_INSTRUCTION,
    DeterministicInterpreter,
    Interpretation,
    InterpretationError,
    ProposedChange,
    asset_tokens,
    cadence_tokens,
    product_tokens,
)
from operations_control.occ_agent.run import SyntheticRun
from operations_control.onboarding.case import (
    APPROVED,
    DRAFT,
    READY_FOR_APPROVAL,
    OnboardingCase,
)
from operations_control.onboarding.catalogue import catalogue

from .conftest import ACTOR, TENANT_A

#: The worked example from the brief.
EXAMPLE = ("Onboard Northstar Lending. It is a UK equity-release portfolio. "
           "They require monthly management information and investor "
           "reporting. The initial monthly files will be a loan tape, IVSR "
           "actuals and executed cashflows.")


@pytest.fixture()
def interpreter() -> DeterministicInterpreter:
    return DeterministicInterpreter()


def _run(state: str = _states.AWAITING_ONBOARDING) -> SyntheticRun:
    return SyntheticRun(case_ref="ONB-2026-0001", tenant="client_a",
                        initiating_user="alice", state=state)


def _case(status: str = DRAFT) -> OnboardingCase:
    return OnboardingCase(case_id="ONB-2026-0001", status=status)


# --------------------------------------------------------------------------- #
# Instruction interpretation
# --------------------------------------------------------------------------- #

def test_the_worked_example_answers_the_catalogue(interpreter):
    result = interpreter.interpret_instruction(EXAMPLE)
    assert result.steps["client"]["client_name"] == "Northstar Lending"
    assert result.steps["client"]["jurisdiction"] == "GB"
    portfolio = result.steps["portfolios"]["portfolios"][0]
    assert portfolio["asset_class"] == "equity_release"
    assert result.steps["reporting"]["products"] == ["mi", "investor_reporting"]
    assert result.cadence == "monthly"
    assert set(result.expected_artefacts) == {"loan_extract",
                                              "property_extract",
                                              "cashflow_extract"}


def test_every_field_it_proposes_exists_in_the_catalogue(interpreter):
    """The validation that stops a model widening the contract."""
    result = interpreter.interpret_instruction(EXAMPLE)
    cat = catalogue()
    for step, payload in result.steps.items():
        section = cat.section(step)
        assert section is not None, step
        rows = ((payload.get(section.key) or []) if section.repeatable
                else [payload])
        for row in rows:
            for key in row:
                assert section.field(key) is not None, f"{step}.{key}"


def test_a_field_the_catalogue_does_not_have_is_refused():
    bad = Interpretation(steps={"client": {"favourite_colour": "blue"}})
    with pytest.raises(InterpretationError):
        bad.validate()


def test_a_step_the_wizard_does_not_have_is_refused():
    bad = Interpretation(steps={"secrets": {"token": "x"}})
    with pytest.raises(InterpretationError):
        bad.validate()


@pytest.mark.parametrize("interpretation", [
    Interpretation(reporting_period="30/06/2026"),
    Interpretation(steps={"client": {"client_name": "x" * 3000}}),
    Interpretation(confidence={"client.client_name": 1.5}),
    Interpretation(steps={"portfolios": {"portfolios": "not a list"}}),
])
def test_an_invalid_interpretation_is_refused(interpretation):
    with pytest.raises(InterpretationError):
        interpretation.validate()


def test_the_client_is_recorded_as_the_originator_entity(interpreter):
    """A legal entity is where the catalogue holds this, so that is where it
    goes — not into a free-text note."""
    result = interpreter.interpret_instruction(EXAMPLE)
    entities = result.steps["entities"]["entities"]
    assert entities[0]["legal_name"] == "Northstar Lending"
    assert entities[0]["roles"] == ["originator"]


def test_a_named_counterparty_takes_its_stated_role(interpreter):
    result = interpreter.interpret_instruction(
        "Onboard Northstar Lending. The servicer is Meridian Servicing.")
    entities = result.steps["entities"]["entities"]
    by_name = {e["legal_name"]: e["roles"] for e in entities}
    assert by_name["Meridian Servicing"] == ["servicer"]
    assert by_name["Northstar Lending"] == ["originator"]


def test_an_inferred_value_is_marked_as_an_inference(interpreter):
    result = interpreter.interpret_instruction(EXAMPLE)
    assert result.provenance["client.client_name"] == PROV_HUMAN_INSTRUCTION
    assert result.provenance["portfolios.display_name"] == PROV_AGENT_INFERENCE
    assert result.confidence["portfolios.display_name"] < 1.0


def test_a_client_name_stops_at_the_sentence_boundary(interpreter):
    result = interpreter.interpret_instruction(
        "Onboard Northstar Lending. UK equity release. Monthly portfolio MI.")
    assert result.steps["client"]["client_name"] == "Northstar Lending"


def test_nothing_is_guessed_when_it_was_not_said(interpreter):
    result = interpreter.interpret_instruction("Onboard Northstar Lending.")
    assert "reporting" not in result.steps
    assert result.reporting_period == ""
    assert result.cadence == ""
    # No identifier is proposed here: Client Onboarding proposes one, with a
    # collision check, and records where it came from.
    assert "client_id" not in result.steps["client"]


def test_a_date_is_only_taken_when_it_is_qualified(interpreter):
    result = interpreter.interpret_instruction(
        "Onboard Northstar Lending. Go-live 2026-09-30.")
    assert result.reporting_period == "", \
        "an unqualified date became the reporting period"
    result = interpreter.interpret_instruction(
        "Onboard Northstar Lending. First reporting date 2026-06-30.")
    assert result.reporting_period == "2026-06-30"


def test_an_empty_instruction_is_refused(interpreter):
    with pytest.raises(InterpretationError):
        interpreter.interpret_instruction("   ")


def test_extraction_is_deterministic(interpreter):
    first = interpreter.interpret_instruction(EXAMPLE).to_dict()
    second = DeterministicInterpreter().interpret_instruction(EXAMPLE).to_dict()
    assert first == second


# --------------------------------------------------------------------------- #
# Vocabulary comes from configuration
# --------------------------------------------------------------------------- #

def test_the_asset_vocabulary_is_the_platforms_own():
    declared = {o["value"] for o in
                (catalogue().field("portfolios", "asset_class").options or [])}
    assert declared
    assert {asset for _token, asset, _confidence in asset_tokens()} <= declared


def test_the_product_vocabulary_is_the_regime_declaration():
    declared = set(catalogue().regime_products)
    assert declared
    assert {product for _token, product in product_tokens()} <= declared


def test_the_cadence_vocabulary_is_the_catalogues_own():
    declared = {o["value"] for o in
                (catalogue().field("sources", "cadence").options or [])}
    assert declared
    assert {value for _token, value in cadence_tokens()} <= declared


def test_a_product_the_catalogue_does_not_declare_is_never_matched(interpreter):
    result = interpreter.interpret_instruction(
        "Onboard Northstar Lending. They need weather forecasting.")
    assert "reporting" not in result.steps


# --------------------------------------------------------------------------- #
# Schema validation of anything an interpreter produces
# --------------------------------------------------------------------------- #

def test_a_model_produced_action_is_schema_validated():
    change = ProposedChange(action="do_whatever_i_say")
    with pytest.raises(InterpretationError):
        change.validate()
    oversized = ProposedChange(action=_states.ACTION_ASK,
                               payload={"question": "x" * 3000})
    with pytest.raises(InterpretationError):
        oversized.validate()


def test_a_swapped_interpreter_still_passes_through_validation(service):
    """An interpreter that returns rubbish cannot corrupt a case."""
    class RogueInterpreter:
        def interpret_instruction(self, text):
            return Interpretation(steps={"client": {"secret_backdoor": "yes"}})

        def interpret_action(self, text, run, case):   # pragma: no cover
            raise AssertionError("not reached")

    service.interpreter = RogueInterpreter()
    with pytest.raises(InterpretationError):
        service.create_case(tenant=TENANT_A, initiating_user=ACTOR,
                            instruction="Onboard Someone.")


def test_answers_always_reach_the_case_through_client_onboarding(service):
    """There is no back door into the answers.

    Every value the interpreter reads is written with ``save_step``, so Client
    Onboarding's validation, inference and event history all apply to it.
    """
    agent_case = service.create_case(
        tenant=TENANT_A, initiating_user=ACTOR,
        instruction="Onboard Northstar Lending. UK equity release.")
    assert any(e["event"] == "answered_client" for e in agent_case.case.events)


# --------------------------------------------------------------------------- #
# Follow-up instructions
# --------------------------------------------------------------------------- #

def test_a_question_is_never_read_as_an_instruction(interpreter):
    for text in ("What is still needed?", "Why is that input needed",
                 "Should I approve the onboarding?"):
        change = interpreter.interpret_action(text, _run(), _case())
        assert change.action == _states.ACTION_ASK, text


def test_a_mapping_instruction_becomes_a_typed_change(interpreter):
    change = interpreter.interpret_action(
        "Map Current Balance to current outstanding balance.", _run(), _case())
    assert change.action == _states.ACTION_RESOLVE_DECISION
    assert change.payload["source_column"] == "Current Balance"
    assert change.payload["canonical_field"] == "current_outstanding_balance"
    assert change.requires_confirmation is True


def test_a_bare_yes_resolves_against_both_lifecycles(interpreter):
    """Onboarding first: until the case is approved there is nothing to run."""
    assert interpreter.interpret_action("yes", _run(), _case(DRAFT)).action == \
        _states.ACTION_SUBMIT_FOR_APPROVAL
    assert interpreter.interpret_action(
        "yes", _run(), _case(READY_FOR_APPROVAL)).action == \
        _states.ACTION_APPROVE_ONBOARDING
    assert interpreter.interpret_action(
        "yes", _run(_states.READY_TO_RUN), _case(APPROVED)).action == \
        _states.ACTION_RUN_ONBOARDING
    assert interpreter.interpret_action(
        "yes", _run(_states.EXECUTION_APPROVAL_REQUIRED),
        _case(APPROVED)).action == _states.ACTION_APPROVE_EXECUTION


def test_a_bare_yes_with_nothing_pending_is_refused(interpreter):
    with pytest.raises(InterpretationError):
        interpreter.interpret_action("yes", _run(_states.READY_FOR_EXECUTION),
                                     _case("activated"))


def test_an_unintelligible_instruction_is_refused(interpreter):
    with pytest.raises(InterpretationError):
        interpreter.interpret_action("asdf qwer zxcv", _run(), _case())


def test_a_bare_name_on_a_follow_up_is_not_read_as_an_answer(interpreter):
    """"Send it to Northstar" must not silently rename the client."""
    with pytest.raises(InterpretationError):
        interpreter.interpret_action("Send it to Northstar", _run(), _case())


def test_a_stated_fact_becomes_an_answer(interpreter):
    change = interpreter.interpret_action(
        "The jurisdiction is the Netherlands.", _run(), _case())
    assert change.action == _states.ACTION_ANSWER
    assert change.requires_confirmation is True
    steps = change.payload["interpretation"]["steps"]
    assert steps["client"]["jurisdiction"] == "NL"


def test_every_supported_human_action_is_reachable_in_language(interpreter):
    expected = {
        "The jurisdiction is the Netherlands.": _states.ACTION_ANSWER,
        "ask the client for the outstanding information":
            _states.ACTION_REQUEST_INFORMATION,
        "record the client's response": _states.ACTION_RECORD_RESPONSE,
        "submit for approval": _states.ACTION_SUBMIT_FOR_APPROVAL,
        "approve the onboarding": _states.ACTION_APPROVE_ONBOARDING,
        "request changes": _states.ACTION_REQUEST_CHANGES,
        "withdraw this onboarding": _states.ACTION_WITHDRAW,
        "run the practice run": _states.ACTION_RUN_ONBOARDING,
        "Map Current Balance to current principal balance.":
            _states.ACTION_RESOLVE_DECISION,
        "acknowledge that": _states.ACTION_ACKNOWLEDGE_EXCEPTION,
        "prepare the orchestration plan": _states.ACTION_GENERATE_PLAN,
        "approve readiness": _states.ACTION_APPROVE_EXECUTION,
        "what is still needed?": _states.ACTION_ASK,
        "cancel this case": _states.ACTION_CANCEL,
    }
    for text, action in expected.items():
        assert interpreter.interpret_action(
            text, _run(), _case()).action == action, text


def test_every_action_the_interpreter_can_name_is_a_lifecycle_action():
    """No action exists in language that the state model does not know."""
    from operations_control.occ_agent.interpretation import _ALL_ACTIONS
    known = set(_states.ONBOARDING_ACTIONS) | {
        _states.ACTION_REGISTER_ARTEFACT, _states.ACTION_RUN_ONBOARDING,
        _states.ACTION_RESOLVE_DECISION, _states.ACTION_ACKNOWLEDGE_EXCEPTION,
        _states.ACTION_GENERATE_PLAN, _states.ACTION_APPROVE_EXECUTION,
        _states.ACTION_CANCEL, _states.ACTION_ASK}
    assert _ALL_ACTIONS == known
