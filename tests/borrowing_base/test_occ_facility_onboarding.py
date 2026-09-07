"""OCC onboarding: asking for the facility, and turning it into configuration.

The flow this protects, end to end:

    facility agreement / schedules   (the client sends what they have)
            v
    operator records the structured terms
            v
    generated client configuration   (funding_facility: ...)
            v
    the borrowing-base engine reads it

What it must NOT do is interpret a legal document on its own. The client
supplies the documentation; an OPERATOR reads it and records the terms; nothing
in this path infers a commitment, an advance rate or an eligibility criterion
from prose.
"""

from __future__ import annotations

import pytest
import yaml

from operations_control.occ_agent import classification as _classification
from operations_control.occ_agent import client_form as _client_form
from operations_control.onboarding.catalogue import catalogue
from operations_control.onboarding.validation import (
    SEVERITY_ADVISORY,
    SEVERITY_BLOCKING,
    Validator,
)


@pytest.fixture(scope="module")
def cat():
    return catalogue()


@pytest.fixture(scope="module")
def section(cat):
    s = cat.section("funding_facility")
    assert s is not None, "the funding-facility section must exist"
    return s


# --------------------------------------------------------------------------- #
# The request the client sees
# --------------------------------------------------------------------------- #
class TestTheClientRequest:
    def test_the_documentation_request_is_worded_as_specified(self, section):
        field = section.field("facility_documentation")
        assert field.label == "Warehouse / Funding Facility Documentation"
        help_text = " ".join(field.help.split())
        assert help_text.startswith(
            "If the portfolio is financed through a warehouse, "
            "borrowing-base or other secured facility, please provide the "
            "current facility agreement and relevant schedules/amendments.")
        assert ("If available, also provide the latest borrowing-base "
                "certificate or facility statement showing current drawings."
                in help_text)

    def test_only_the_documentation_itself_is_asked_of_the_client(self, section):
        client_fields = [f.key for f in section.fields if f.asked_of_client]
        assert client_fields == ["facility_documentation"]

    def test_every_structured_term_is_an_OPERATOR_decision(self, section):
        operator_fields = [f.key for f in section.fields
                           if f.source == "operator_supplied"]
        assert "facility_commitment" in operator_fields
        assert "facility_advance_rate" in operator_fields
        assert "facility_eligibility_status" in operator_fields

    def test_the_whole_section_is_optional_so_it_blocks_nobody(self, section):
        assert section.optional_context is True
        assert not [f for f in section.fields if f.required]

    def test_it_is_asked_only_of_clients_who_take_an_MI_surface(self, cat):
        field = cat.field("funding_facility", "facility_documentation")
        assert field.asked_when == "reporting.products contains mi"
        assert cat.is_asked(field, {"reporting": {"products": ["mi"]}})
        assert not cat.is_asked(field, {"reporting": {"products": ["esma_annex2"]}})

    def test_the_client_meets_it_as_its_own_step_in_the_pack(self):
        step = next(s for s in _client_form.STEPS if s["key"] == "funding")
        assert step["sections"] == ("funding_facility",)
        assert "Skip it if there is no facility" in step["help"]

    def test_the_documentation_question_is_client_facing(self, cat):
        rows = _classification.classify_all(
            {"reporting": {"products": ["mi"]}}, cat=cat)
        row = next(r for r in rows
                   if r.section == "funding_facility"
                   and r.field == "facility_documentation")
        assert row.client_facing is True


# --------------------------------------------------------------------------- #
# The structured terms
# --------------------------------------------------------------------------- #
REQUIRED_TERMS = ("facility_id", "facility_type", "facility_currency",
                  "facility_commitment", "facility_advance_rate",
                  "facility_current_drawn_amount", "facility_effective_date",
                  "facility_maturity_date",
                  "facility_concentration_denominator_floor",
                  "facility_eligibility_status", "facility_eligibility_criteria",
                  "facility_breach_treatment")


class TestTheStructuredTerms:
    @pytest.mark.parametrize("key", REQUIRED_TERMS)
    def test_the_term_is_collected(self, section, key):
        assert section.field(key) is not None, key

    def test_the_terms_write_into_the_clients_own_configuration(self, section):
        destinations = {f.key: f.writes_to for f in section.fields}
        assert destinations["facility_id"] == "client_config:funding_facility.facility_id"
        assert destinations["facility_commitment"] == \
            "client_config:funding_facility.commitment"
        assert destinations["facility_advance_rate"] == \
            "client_config:funding_facility.advance_rate"
        assert destinations["facility_concentration_denominator_floor"] == \
            "client_config:funding_facility.concentration_denominator_floor"

    def test_the_source_documents_are_retained_as_provenance(self, section):
        assert section.field("facility_documentation_reference").writes_to == \
            "client_config:funding_facility.governance.source_reference"

    def test_the_drawn_amount_tells_the_operator_to_LEAVE_IT_BLANK(self, section):
        help_text = " ".join(section.field(
            "facility_current_drawn_amount").help.split())
        assert "LEAVE IT BLANK if you do not have one" in help_text
        assert "NOT CALCULABLE" in help_text

    def test_the_breach_treatment_defaults_to_monitor_only(self, section):
        field = section.field("facility_breach_treatment")
        assert field.default == "monitor_only"
        assert {o["value"] for o in field.options} == {"monitor_only",
                                                       "exclude_excess"}

    def test_eligibility_starts_as_awaiting_the_agreement(self, section):
        field = section.field("facility_eligibility_status")
        assert field.default == "awaiting_facility_agreement"
        assert "does not infer eligibility from the concentration limits" in \
            " ".join(field.help.split())


# --------------------------------------------------------------------------- #
# Generation: answers -> client configuration -> the engine
# --------------------------------------------------------------------------- #
ANSWERS = {
    "reporting": {"products": ["mi"]},
    "funding_facility": {
        "facility_documentation": "Facility agreement and Schedule 8 attached.",
        "facility_documentation_status": "supplied",
        "facility_documentation_reference":
            "Warehouse Facility Agreement dated 2025-06-30, Schedule 8.",
        "facility_id": "WAREHOUSE_01",
        "facility_type": "warehouse",
        "facility_currency": "gbp",
        "facility_commitment": "250,000,000",
        "facility_advance_rate": "103",
        "facility_concentration_denominator_floor": "£33,000,000",
        "facility_current_drawn_amount": "180,000,000",
        "facility_current_drawn_amount_as_of": "2025-11-30",
        "facility_effective_date": "2025-06-30",
        "facility_maturity_date": "2028-06-30",
        "facility_eligibility_status": "awaiting_facility_agreement",
        "facility_breach_treatment": "monitor_only",
    },
}


def render(answers, client_id="testco"):
    """The generated client configuration document, as YAML then re-parsed."""
    from operations_control.onboarding.artefacts import render_client_config
    from operations_control.onboarding.case import OnboardingCase

    case = OnboardingCase(case_id="ONB-TEST", client_id=client_id.upper())
    case.answers = dict(answers)
    doc, _notes = render_client_config(case)
    return yaml.safe_load(yaml.safe_dump(doc))


@pytest.fixture(scope="module")
def block():
    return render(ANSWERS)["funding_facility"]


class TestGeneratedConfiguration:
    def test_the_terms_land_under_funding_facility(self, block):
        assert block["facility_id"] == "WAREHOUSE_01"
        assert block["facility_type"] == "warehouse"
        assert block["currency"] == "GBP"

    def test_amounts_arrive_as_NUMBERS_not_as_typed_strings(self, block):
        assert block["commitment"] == 250_000_000
        assert block["concentration_denominator_floor"] == 33_000_000
        assert block["current_drawn_amount"] == 180_000_000

    def test_the_source_document_reference_is_retained(self, block):
        assert "Schedule 8" in block["governance"]["source_reference"]

    def test_the_engine_reads_the_generated_block_back(self, tmp_path,
                                                       monkeypatch):
        from mi_agent.borrowing_base import config as bb_config

        doc = render(ANSWERS, client_id="testco")
        doc.setdefault("client", {})["client_id"] = "testco"
        (tmp_path / "config_client_testco.yaml").write_text(
            yaml.safe_dump(doc), encoding="utf-8")
        monkeypatch.setenv(bb_config.CLIENT_CONFIG_DIR_ENV, str(tmp_path))

        facility = bb_config.load_facility("testco")
        assert facility.facility_id == "WAREHOUSE_01"
        assert facility.commitment == 250_000_000.0
        assert facility.advance_rate == 1.03          # "103" -> 103%
        assert facility.concentration_denominator_floor == 33_000_000.0
        assert facility.current_drawn_amount == 180_000_000.0
        # Onboarding writes no environment, so the facility is PRODUCTION and
        # fails closed: no approved criteria means UNDETERMINED, not eligible.
        assert facility.environment == "production"
        assert facility.eligibility_governed is False
        assert facility.prototype_assumption_active is False


# --------------------------------------------------------------------------- #
# The advisory flag
# --------------------------------------------------------------------------- #
def problems(answers):
    from operations_control.onboarding.case import OnboardingCase
    case = OnboardingCase(case_id="ONB-TEST", client_id="TESTCO")
    case.answers = dict(answers)
    return Validator().validate(case)


class TestTheAdvisoryFlag:
    MESSAGE = "Funding facility rules not fully governed"

    def test_no_documentation_flags_but_never_blocks(self):
        found = problems({"reporting": {"products": ["mi"]},
                          "funding_facility": {
                              "facility_documentation_status":
                                  "pending_client_response"}})
        flagged = [p for p in found if self.MESSAGE in p.message]
        assert flagged, "the facility governance gap must be surfaced"
        assert all(p.severity == SEVERITY_ADVISORY for p in flagged)
        assert not [p for p in found
                    if p.section == "funding_facility"
                    and p.severity == SEVERITY_BLOCKING]

    def test_a_client_with_no_facility_is_not_flagged_at_all(self):
        found = problems({"reporting": {"products": ["mi"]},
                          "funding_facility": {
                              "facility_documentation_status": "no_facility"}})
        assert not [p for p in found if p.section == "funding_facility"]

    def test_supplied_documentation_with_no_terms_taken_off_it_is_flagged(self):
        found = problems({"reporting": {"products": ["mi"]},
                          "funding_facility": {
                              "facility_documentation_status": "supplied"}})
        assert any(self.MESSAGE in p.message and "taken off it yet" in p.message
                   for p in found)

    def test_an_unapproved_eligibility_definition_is_flagged(self):
        found = problems(ANSWERS)
        assert any("Eligible Mortgage Loan definition has not been approved"
                   in p.message for p in found)

    def test_a_missing_drawn_amount_is_noted_without_the_governance_flag(self):
        answers = {"reporting": {"products": ["mi"]},
                   "funding_facility": {
                       **ANSWERS["funding_facility"],
                       "facility_current_drawn_amount": "",
                       "facility_eligibility_status": "approved"}}
        found = problems(answers)
        drawn = [p for p in found if p.field == "facility_current_drawn_amount"]
        assert drawn and "NOT CALCULABLE" in drawn[0].message
        assert all(p.severity == SEVERITY_ADVISORY for p in drawn)

    def test_a_fully_governed_facility_raises_nothing(self):
        answers = {"reporting": {"products": ["mi"]},
                   "funding_facility": {**ANSWERS["funding_facility"],
                                        "facility_eligibility_status": "approved"}}
        assert not [p for p in problems(answers)
                    if p.section == "funding_facility"]

    def test_an_ESMA_only_client_is_never_asked_and_never_flagged(self):
        found = problems({"reporting": {"products": ["esma_annex2"]}})
        assert not [p for p in found if p.section == "funding_facility"]
