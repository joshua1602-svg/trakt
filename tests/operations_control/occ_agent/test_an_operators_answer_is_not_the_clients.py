"""A value an operator typed is not a value the client supplied.

WHAT THE RECORD CLAIMED

Contacts, legal names, LEIs — anything whose catalogue ``source`` is
``client_supplied`` — read back as ``client_supplied`` whoever had typed them.
``origin_provenance`` falls back to the field's DECLARED source when nothing
recorded where a value actually came from, and for these fields that declared
source is the client. So an operator filling in a contact to get an onboarding
moving produced a governed record stating the client had given it.

That is a claim made on the client's behalf, about a conversation that never
happened, in the document an approver reads to decide whether to activate.

WHY THE FALLBACK IS RIGHT FOR EVERY OTHER SOURCE

For ``inferred``, ``derived``, ``trakt_default`` and ``system_generated`` the
field's own source genuinely does say who filled it — Trakt did, and there is
nobody else it could have been. ``client_supplied`` is the one source where the
declared value is an ASSUMPTION rather than a fact, which is precisely why
``_mark_client_supplied`` exists: the client's own submissions are marked so an
approver can tell them apart.

The marking scheme was only ever applied to one side. An operator's write went
unmarked, fell through to the declared source, and came back wearing the
client's name. So the operator's write is marked too, and the client path —
which marks immediately afterwards — still wins where it applies.
"""

from __future__ import annotations

import pytest

from operations_control.occ_agent import classification as _c
from operations_control.onboarding.catalogue import catalogue
from operations_control.occ_agent.interpretation import PROV_CLIENT, PROV_HUMAN

from .conftest import ACTOR, TENANT_A

OPENING = ("Onboard Northstar Lending. UK equity release. Monthly portfolio "
           "MI. Portfolio id direct_101.")


@pytest.fixture()
def opened(service):
    return service.create_case(tenant=TENANT_A, initiating_user=ACTOR,
                               instruction=OPENING)


class TestAnOperatorsAnswerSaysSo:
    def test_the_operator_is_recorded_as_the_source(self, service, opened):
        service.onboarding.save_step(
            case_id=opened.case_ref, step="contacts", by="operator",
            payload={"reporting_contact_name": "A Person",
                     "reporting_contact_email": "a@northstar.example"})
        case = service.load(TENANT_A, opened.case_ref).case
        assert case.provenance_class["contacts.reporting_contact_email"] \
            == PROV_HUMAN

    def test_and_the_classification_an_approver_reads_agrees(self, service,
                                                             opened):
        """The property that was actually wrong — not the map, the read."""
        service.onboarding.save_step(
            case_id=opened.case_ref, step="contacts", by="operator",
            payload={"reporting_contact_email": "a@northstar.example"})
        case = service.load(TENANT_A, opened.case_ref).case
        section = catalogue().section("contacts")
        row = _c.classify(section, section.field("reporting_contact_email"),
                          holder=case.answers.get("contacts") or {},
                          answers=case.answers,
                          provenance=case.provenance_class.get(
                              "contacts.reporting_contact_email", ""))
        assert row.provenance == PROV_HUMAN

    def test_a_value_nobody_has_touched_claims_nothing(self, service, opened):
        case = service.load(TENANT_A, opened.case_ref).case
        assert "contacts.reporting_contact_email" not in case.provenance_class


class TestTheClientStillWinsWhereTheClientAnswered:
    """`submit_client_response` writes through `save_step` and then marks."""

    def test_a_clients_own_answer_is_still_client_supplied(self, service,
                                                           opened):
        service.submit_client_response(
            opened, actor=ACTOR, strict=False,
            response={"contacts.reporting_contact_email":
                      "them@northstar.example"})
        case = service.load(TENANT_A, opened.case_ref).case
        assert case.provenance_class["contacts.reporting_contact_email"] \
            == PROV_CLIENT

    def test_an_operator_correcting_it_afterwards_takes_it_back(self, service,
                                                                opened):
        """Whoever wrote last is who the record names."""
        service.submit_client_response(
            opened, actor=ACTOR, strict=False,
            response={"contacts.reporting_contact_email":
                      "them@northstar.example"})
        service.onboarding.save_step(
            case_id=opened.case_ref, step="contacts", by="operator",
            payload={"reporting_contact_email": "corrected@northstar.example"})
        case = service.load(TENANT_A, opened.case_ref).case
        assert case.provenance_class["contacts.reporting_contact_email"] \
            == PROV_HUMAN


class TestTrakstOwnValuesAreUnaffected:
    """The fallback is right for every source that is not the client's."""

    @pytest.mark.parametrize("section_key,field_key", [
        ("presentation", "report_title"),
        ("presentation", "day_count_convention"),
    ])
    def test_an_inferred_field_still_reads_as_inferred(self, section_key,
                                                       field_key):
        section = catalogue().section(section_key)
        field = section.field(field_key)
        assert field.source == "inferred"
        row = _c.classify(section, field, holder={field_key: "something"},
                          answers={})
        assert row.provenance and row.provenance != PROV_CLIENT
