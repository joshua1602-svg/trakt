"""Natural-language interpretation, and the onboarding pack it feeds.

The interpreter's contract is that whatever it produces is schema-validated and
never authoritative until a human confirms it. The pack's contract is that every
question it asks comes from platform configuration.
"""

from __future__ import annotations

import pytest

from operations_control.occ_agent import pack as _pack
from operations_control.occ_agent import states as _states
from operations_control.occ_agent.case import (
    CaseSchemaError,
    ExtractedRequirements,
    PROV_AGENT_INFERENCE,
    PROV_HUMAN_INSTRUCTION,
    SyntheticCase,
)
from operations_control.occ_agent.interpretation import (
    DeterministicInterpreter,
    InterpretationError,
    ProposedChange,
)

from .conftest import ACTOR, TENANT_A

#: The specification's own worked example.
EXAMPLE = ("Onboard Northstar Lending. It is a UK equity-release portfolio. "
           "They require monthly portfolio MI and investor reporting. The "
           "initial monthly files will be a loan tape, IVSR actuals and "
           "executed cashflows.")


@pytest.fixture()
def interpreter() -> DeterministicInterpreter:
    return DeterministicInterpreter()


# --------------------------------------------------------------------------- #
# Instruction interpretation
# --------------------------------------------------------------------------- #

def test_the_worked_example_extracts_the_expected_facts(interpreter):
    req = interpreter.interpret_instruction(EXAMPLE)
    assert req.client_name == "Northstar Lending"
    assert req.asset_type == "equity_release"
    assert req.jurisdiction == "GB"
    assert req.required_products == ["portfolio_mi", "investor_reporting"]
    assert req.reporting_frequency == "monthly"
    assert set(req.expected_artefacts) == {"loan_extract", "property_extract",
                                           "cashflow_extract"}


def test_the_worked_example_leaves_the_right_questions_outstanding(interpreter):
    req = interpreter.interpret_instruction(EXAMPLE)
    outstanding = " ".join(req.unresolved_questions).lower()
    assert "first reporting date" in outstanding
    assert "portfolio identifier" in outstanding
    assert "go-live" in outstanding


def test_a_proposed_identifier_is_marked_as_an_inference(interpreter):
    req = interpreter.interpret_instruction(EXAMPLE)
    assert req.proposed_client_id == "northstar_lending"
    assert req.provenance["proposed_client_id"] == PROV_AGENT_INFERENCE
    assert req.confidence["proposed_client_id"] < 1.0
    # A stated fact is not an inference.
    assert req.provenance["client_name"] == PROV_HUMAN_INSTRUCTION


def test_a_client_name_stops_at_the_sentence_boundary(interpreter):
    req = interpreter.interpret_instruction(
        "Onboard Northstar Lending. UK equity release. Monthly portfolio MI.")
    assert req.client_name == "Northstar Lending"


def test_nothing_is_guessed_when_it_was_not_said(interpreter):
    req = interpreter.interpret_instruction("Onboard Northstar Lending.")
    assert req.reporting_date == ""
    assert req.target_go_live_date == ""
    assert req.required_products == []
    # Everything absent becomes a question rather than a value.
    assert len(req.unresolved_questions) >= 5


def test_a_date_is_only_taken_when_it_is_qualified(interpreter):
    req = interpreter.interpret_instruction(
        "Onboard Northstar Lending. Go-live 2026-09-30.")
    assert req.target_go_live_date == "2026-09-30"
    assert req.reporting_date == "", "an unqualified date became the reporting date"


def test_an_empty_instruction_is_refused(interpreter):
    with pytest.raises(InterpretationError):
        interpreter.interpret_instruction("   ")


def test_extraction_is_deterministic(interpreter):
    first = interpreter.interpret_instruction(EXAMPLE).to_dict()
    second = DeterministicInterpreter().interpret_instruction(EXAMPLE).to_dict()
    assert first == second


# --------------------------------------------------------------------------- #
# Schema validation of anything an interpreter produces
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("field,value", [
    ("client_name", 42),
    ("required_products", "portfolio_mi"),
    ("reporting_date", "30/06/2026"),
    ("proposed_client_id", "not a valid id!"),
    ("client_name", "x" * 300),
])
def test_invalid_extracted_facts_are_refused(field, value):
    req = ExtractedRequirements()
    setattr(req, field, value)
    with pytest.raises(CaseSchemaError):
        req.validate()


def test_an_unknown_provenance_source_is_refused():
    req = ExtractedRequirements(provenance={"client_name": "vibes"})
    with pytest.raises(CaseSchemaError):
        req.validate()


def test_an_out_of_range_confidence_is_refused():
    req = ExtractedRequirements(confidence={"asset_type": 1.5})
    with pytest.raises(CaseSchemaError):
        req.validate()


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
            bad = ExtractedRequirements()
            bad.reporting_date = "whenever"        # not a date
            return bad

        def interpret_action(self, text, case):    # pragma: no cover
            raise AssertionError("not reached")

    service.interpreter = RogueInterpreter()
    with pytest.raises(CaseSchemaError):
        service.create_case(tenant=TENANT_A, initiating_user=ACTOR,
                            instruction="Onboard Someone.")


# --------------------------------------------------------------------------- #
# Follow-up instructions
# --------------------------------------------------------------------------- #

def test_a_question_is_never_read_as_an_instruction(interpreter):
    case = SyntheticCase(case_id="CASE-AAAA0001", tenant="client_a",
                         initiating_user="alice")
    for text in ("What is still needed?", "Why is that input needed",
                 "Should I approve the pack?"):
        change = interpreter.interpret_action(text, case)
        assert change.action == _states.ACTION_ASK, text


def test_a_mapping_instruction_becomes_a_typed_change(interpreter):
    case = SyntheticCase(case_id="CASE-AAAA0001", tenant="client_a",
                         initiating_user="alice")
    change = interpreter.interpret_action(
        "Map Current Balance to current outstanding balance.", case)
    assert change.action == _states.ACTION_RESOLVE_DECISION
    assert change.payload["source_column"] == "Current Balance"
    assert change.payload["canonical_field"] == "current_outstanding_balance"
    assert change.requires_confirmation is True


def test_a_bare_yes_resolves_against_the_current_state(interpreter):
    case = SyntheticCase(case_id="CASE-AAAA0001", tenant="client_a",
                         initiating_user="alice",
                         state=_states.CONFIG_APPROVAL_REQUIRED)
    change = interpreter.interpret_action("yes", case)
    assert change.action == _states.ACTION_APPROVE_CONFIG


def test_a_bare_yes_with_nothing_pending_is_refused(interpreter):
    case = SyntheticCase(case_id="CASE-AAAA0001", tenant="client_a",
                         initiating_user="alice",
                         state=_states.SYNTHETIC_ONBOARDING_PASSED)
    with pytest.raises(InterpretationError):
        interpreter.interpret_action("yes", case)


def test_an_unintelligible_instruction_is_refused(interpreter):
    case = SyntheticCase(case_id="CASE-AAAA0001", tenant="client_a",
                         initiating_user="alice")
    with pytest.raises(InterpretationError):
        interpreter.interpret_action("asdf qwer zxcv", case)


def test_a_return_request_names_the_stage(interpreter):
    case = SyntheticCase(case_id="CASE-AAAA0001", tenant="client_a",
                         initiating_user="alice")
    change = interpreter.interpret_action(
        "Go back to requirements extracted.", case)
    assert change.action == _states.ACTION_RETURN_TO_STAGE
    assert change.payload["target_state"] == _states.REQUIREMENTS_EXTRACTED
    with pytest.raises(InterpretationError):
        interpreter.interpret_action("Go back to somewhere else.", case)


def test_every_supported_human_action_is_reachable_in_language(interpreter):
    """§14's list, each expressible as a sentence."""
    case = SyntheticCase(case_id="CASE-AAAA0001", tenant="client_a",
                         initiating_user="alice")
    expected = {
        "confirm the interpretation": _states.ACTION_CONFIRM_REQUIREMENTS,
        "The reporting date is 2026-06-30.":
            _states.ACTION_CORRECT_REQUIREMENTS,
        "generate the onboarding pack": _states.ACTION_GENERATE_PACK,
        "approve the pack": _states.ACTION_APPROVE_PACK,
        "Map Current Balance to current principal balance.":
            _states.ACTION_RESOLVE_DECISION,
        "approve the configuration": _states.ACTION_APPROVE_CONFIG,
        "acknowledge that": _states.ACTION_ACKNOWLEDGE_EXCEPTION,
        "regenerate the pack": _states.ACTION_GENERATE_PACK,
        "approve readiness": _states.ACTION_APPROVE_EXECUTION,
        "go back to config drafted": _states.ACTION_RETURN_TO_STAGE,
        "what is still needed?": _states.ACTION_ASK,
        "cancel this case": _states.ACTION_CANCEL,
    }
    for text, action in expected.items():
        assert interpreter.interpret_action(text, case).action == action, text


# --------------------------------------------------------------------------- #
# The onboarding pack
# --------------------------------------------------------------------------- #

def test_the_pack_has_every_required_section(interpreter):
    req = interpreter.interpret_instruction(EXAMPLE)
    pack = _pack.build_pack(req)
    keys = {section["key"] for section in pack.sections}
    assert keys == set(_pack.SECTION_KEYS)


def test_the_pack_asks_for_the_configured_required_roles(interpreter):
    from operations_control.occ_agent.vocabulary import artefact_vocabulary
    req = interpreter.interpret_instruction(EXAMPLE)
    pack = _pack.build_pack(req)
    assert pack.required_roles == artefact_vocabulary().required_roles(
        pack.outcome)
    initial = pack.section(_pack.SEC_INITIAL_REQUEST)
    required = [item["role"] for item in initial["items"] if item["required"]]
    assert required == pack.required_roles


def test_the_pack_asks_the_selected_products_own_questions(interpreter):
    from operations_control.occ_agent.vocabulary import product_vocabulary
    req = interpreter.interpret_instruction(EXAMPLE)
    pack = _pack.build_pack(req)
    section = pack.section(_pack.SEC_PRODUCT_QUESTIONS)
    asked = {item["product"] for item in section["items"]}
    assert asked == set(req.required_products)
    products = product_vocabulary()
    for item in section["items"]:
        product = products.by_id(item["product"])
        assert item["question"] in product.questions


def test_a_regulatory_product_adds_its_own_questions(interpreter):
    plain = _pack.build_pack(
        ExtractedRequirements(client_name="X", asset_type="equity_release",
                              required_products=["portfolio_mi"]))
    regulated = _pack.build_pack(
        ExtractedRequirements(client_name="X", asset_type="equity_release",
                              required_products=["portfolio_mi",
                                                 "regulatory_reporting"]))
    plain_qs = {i["question"] for i in
                plain.section(_pack.SEC_PRODUCT_QUESTIONS)["items"]}
    regulated_qs = {i["question"] for i in
                    regulated.section(_pack.SEC_PRODUCT_QUESTIONS)["items"]}
    assert plain_qs < regulated_qs
    assert plain.outcome == "mi"
    assert regulated.outcome == "mi_annex2"


def test_the_asset_questions_follow_the_configured_asset(interpreter):
    req = interpreter.interpret_instruction(EXAMPLE)
    section = _pack.build_pack(req).section(_pack.SEC_ASSET_QUESTIONS)
    text = " ".join(item["question"] for item in section["items"])
    assert "ESMA_Annex2" in text          # the regime this asset supports


def test_the_pack_says_where_a_location_cannot_yet_be_derived(interpreter):
    """A missing fact produces an honest gap, not an invented path."""
    req = interpreter.interpret_instruction("Onboard Northstar Lending.")
    section = _pack.build_pack(req).section(_pack.SEC_DELIVERY)
    assert "blob://" not in section["body"]
    assert "cannot be quoted" in section["body"]


def test_an_unknown_fact_is_shown_as_a_gap_not_omitted(interpreter):
    req = interpreter.interpret_instruction("Onboard Northstar Lending.")
    email = _pack.build_pack(req).section(_pack.SEC_EMAIL)
    assert _pack.UNKNOWN in email["body"]


def test_the_pack_content_hash_is_stable(interpreter):
    req = interpreter.interpret_instruction(EXAMPLE)
    assert _pack.build_pack(req).content_hash == \
        _pack.build_pack(req).content_hash


def test_the_pack_changes_when_the_facts_change(interpreter):
    req = interpreter.interpret_instruction(EXAMPLE)
    before = _pack.build_pack(req)
    req.reporting_date = "2026-06-30"
    assert _pack.build_pack(req).content_hash != before.content_hash


def test_the_expected_stages_come_from_the_lifecycle(interpreter):
    req = interpreter.interpret_instruction(EXAMPLE)
    section = _pack.build_pack(req).section(_pack.SEC_STAGES)
    states = [item["state"] for item in section["items"]]
    assert states == [s for s in _states.STATE_ORDER
                      if s not in (_states.BLOCKED, _states.CANCELLED)]
