"""A regulatory deadline that has not arrived must not hold the client's MI.

THE SITUATION THIS IS FOR

A client takes management information AND ESMA Annex 2. The tape exists today;
the return is not due for weeks. The originator's LEI has been asked for and
has not come back — GLEIF evidence takes as long as it takes.

Onboarding refused to complete, because ``entities.lei`` and
``entities.country_of_establishment`` are required once an entity holds the
originator role, and an unanswered required field refused approval. So a client
taking MI alone went live the day their tape arrived, and the same client taking
MI *and* Annex 2 waited for a regulatory identity their MI does not use.

TWO COSTS THAT LOOK ALIKE AND ARE NOT

Naming the originator is a tick: the operator already knows which entity it is.
Obtaining its LEI is a GLEIF lookup that takes as long as it takes. Only the
second is a reason to wait, and conflating them causes damage in either
direction:

* leave both blocking and MI waits weeks for a value it never reads;
* make both advisory and a case completes with NO originator — at which point
  ``entities.lei``, whose ``required_when`` is "roles contains originator", is
  required of nobody. It drops off the client's checklist as optional, the
  client reasonably deprioritises it, and the one value standing between them
  and their first return is never actually asked for.

The second is worse, because it is silent. So the cheap half stays BLOCKING and
the slow half is advisory: name the originator now, send the LEI when it comes.

WHAT MUST NOT WEAKEN

The fields stay REQUIRED where the client reads them. This is not a demotion to
"optional" — a client told the LEI is optional will reasonably not prioritise
it, and then the first Annex 2 return is late for a reason onboarding invited.
They stay on the checklist, stay outstanding, stay chased. Only their power to
refuse ACTIVATION is removed, and the regime's own gate still refuses to build
a return without RREL83/RREL84.
"""

from __future__ import annotations

import pytest

from operations_control.onboarding.service import OnboardingService
from operations_control.onboarding.validation import (
    SEVERITY_ADVISORY,
    SEVERITY_BLOCKING,
)
from operations_control.stores import OpsLayout, OpsStore

CONTACTS = {
    "reporting_contact_name": "A Reporter",
    "reporting_contact_email": "reporting@ere.example",
    "operational_contact_name": "An Operator",
    "operational_contact_email": "ops@ere.example",
}


@pytest.fixture()
def onboarding(tmp_path, monkeypatch):
    monkeypatch.setenv("TRAKT_STORAGE_BACKEND", "file")
    monkeypatch.setenv("TRAKT_LOCAL_BLOB_ROOT", str(tmp_path / "blob"))
    from apps.blob_trigger_app.storage import Storage
    return OnboardingService(
        OpsStore(Storage(tmp_path / "blob"),
                 OpsLayout(container="operations-control-synthetic")))


def _case(onboarding, *, roles, products=("esma_annex2",), contacts=False,
          file_format=True):
    case = onboarding.start_new_client(by="operator")
    case = onboarding.save_step(case_id=case.case_id, step="client", payload={
        "client_name": "ERE Funding Limited", "client_id": "ERE",
        "jurisdiction": "GB"}, by="operator")
    case = onboarding.save_step(case_id=case.case_id, step="entities", payload={
        "entities": [{"legal_name": "ERE Funding Limited",
                      "roles": list(roles)}]}, by="operator")
    case = onboarding.save_step(case_id=case.case_id, step="portfolios", payload={
        "portfolios": [{"portfolio_id": "direct_001",
                        "display_name": "ERE Direct Origination Book",
                        "portfolio_type": "direct",
                        "asset_class": "equity_release"}]}, by="operator")
    case = onboarding.save_step(case_id=case.case_id, step="reporting",
                                payload={"products": list(products)},
                                by="operator")
    if contacts:
        case = onboarding.save_step(case_id=case.case_id, step="contacts",
                                    payload=dict(CONTACTS), by="operator")
    if file_format:
        sources = [{**s, "file_format": "csv"} for s in case.items("sources")]
        case = onboarding.save_step(case_id=case.case_id, step="sources",
                                    payload={"sources": sources}, by="operator")
    return case


def _by_severity(onboarding, case):
    problems = onboarding._validator().validate(case)
    return ({f"{p.section}.{p.field}" for p in problems
             if p.severity == SEVERITY_BLOCKING},
            {f"{p.section}.{p.field}" for p in problems
             if p.severity == SEVERITY_ADVISORY})


class TestTheRegimeIdentityIsAdvisory:
    def test_the_lei_does_not_refuse_approval(self, onboarding):
        case = _case(onboarding, roles=["originator", "reporting_entity"])
        blocking, advisory = _by_severity(onboarding, case)
        assert "entities.lei" in advisory
        assert "entities.lei" not in blocking

    def test_nor_does_the_country_of_establishment(self, onboarding):
        case = _case(onboarding, roles=["originator", "reporting_entity"])
        blocking, advisory = _by_severity(onboarding, case)
        assert "entities.country_of_establishment" in advisory
        assert "entities.country_of_establishment" not in blocking

    def test_but_naming_the_originator_still_does(self, onboarding):
        """The cheap half stays required, and must.

        Naming the originator is a tick the operator can do now; obtaining its
        LEI is a GLEIF lookup that takes weeks. Treating both as advisory was
        briefly tried and did more damage than the delay it avoided — see
        ``TestTheLeiIsAskedForAtAll`` below.
        """
        case = _case(onboarding, roles=["reporting_entity"])
        blocking, advisory = _by_severity(onboarding, case)
        assert "entities.roles" in blocking
        assert "entities.roles" not in advisory

    def test_mi_goes_live_with_the_lei_still_outstanding(self, onboarding):
        """The whole point, stated as the platform states it."""
        case = _case(onboarding, roles=["originator", "reporting_entity"],
                     contacts=True)
        assert onboarding._validator().is_ready(case) is True
        lei = case.items("entities")[0].get("lei")
        assert not lei, "the case is ready precisely BECAUSE this is unanswered"


class TestTheLeiIsAskedForAtAll:
    """The trap in making the originator role advisory too.

    ``entities.lei`` is required when "roles contains originator". Make naming
    the originator advisory as well, and a case can complete with no originator
    at all — at which point the LEI is required of NOBODY. It leaves the
    client's checklist as optional, the client reasonably deprioritises it, and
    the single value standing between them and their first Annex 2 return is
    never actually asked for.

    That is worse than the delay the change was meant to avoid, and it is silent
    — so it is asserted directly rather than left to follow from the severity.
    """

    def test_selecting_annex_2_forces_someone_to_be_the_originator(
            self, onboarding):
        case = _case(onboarding, roles=["reporting_entity"], contacts=True)
        assert onboarding._validator().is_ready(case) is False, (
            "a case with no originator must not complete: its LEI would never "
            "be asked for")

    def test_and_once_named_the_lei_is_asked_as_required(self, onboarding):
        case = _case(onboarding, roles=["originator", "reporting_entity"])
        asked = {r["field"] for r in
                 onboarding.catalogue.outstanding_for_client(case.answers)}
        assert "lei" in asked
        assert "country_of_establishment" in asked


class TestItIsStillAskedAsRequired:
    """Advisory to activation is not the same as optional to the client.

    A client told the LEI is optional will not prioritise it, and the first
    return is then late for a reason onboarding invited.
    """

    @pytest.mark.parametrize("field", ["lei", "country_of_establishment"])
    def test_the_client_is_still_asked(self, onboarding, field):
        case = _case(onboarding, roles=["originator", "reporting_entity"])
        asked = {r["field"] for r in
                 onboarding.catalogue.outstanding_for_client(case.answers)}
        assert field in asked

    @pytest.mark.parametrize("field", ["lei", "country_of_establishment"])
    def test_and_the_catalogue_still_calls_it_required(self, onboarding, field):
        case = _case(onboarding, roles=["originator", "reporting_entity"])
        cat = onboarding.catalogue
        f = cat.section("entities").field(field)
        item = case.items("entities")[0]
        assert cat.is_required(f, case.answers, item) is True
        assert f.blocks_activation is False


class TestTheOrdinaryBlockersAreUntouched:
    def test_contacts_still_refuse_approval(self, onboarding):
        case = _case(onboarding, roles=["originator"], contacts=False)
        blocking, _ = _by_severity(onboarding, case)
        assert "contacts.reporting_contact_email" in blocking
        assert onboarding._validator().is_ready(case) is False

    def test_a_client_taking_no_regime_is_asked_for_no_regime_identity(
            self, onboarding):
        """No regime selected, so the LEI is neither required nor reported.

        Asserted as ABSENCE FROM BOTH lists rather than as readiness: a case
        with no product selected is blocked on ``reporting.products``, which is
        a question about whether management information alone is expressible as
        a product selection and has nothing to do with this change.
        """
        case = _case(onboarding, roles=["reporting_entity"], products=[],
                     contacts=True)
        blocking, advisory = _by_severity(onboarding, case)
        assert "entities.lei" not in blocking | advisory
        assert "entities.roles" not in blocking | advisory


# --------------------------------------------------------------------------- #
# What a client is asked to send
# --------------------------------------------------------------------------- #

class TestFourTapes:
    """The checklist asks for four files, named the way a client says them.

    "Funder principal & interest tape" was asked for BESIDE the cash-flow tape,
    and it is the same file under a more technical name. Asking for both
    invited a client to wonder which of the two they had, or to send one and
    believe the other still outstanding. The generic name wins because more
    people recognise it.
    """

    @staticmethod
    def vocab():
        from operations_control.occ_agent.input_roles import artefact_vocabulary
        return artefact_vocabulary()

    @pytest.mark.parametrize("outcome", ["mi", "mi_annex2"])
    def test_exactly_four_tapes_are_asked_for(self, outcome):
        vocab = self.vocab()
        asked = set(vocab.required_roles(outcome)) | set(
            vocab.optional_roles(outcome))
        assert asked == {"loan_extract", "collateral_extract",
                         "cashflow_extract", "pipeline_report"}

    def test_the_loan_tape_is_the_only_required_one(self):
        vocab = self.vocab()
        assert list(vocab.required_roles("mi_annex2")) == ["loan_extract"]

    @pytest.mark.parametrize("role,label", [
        ("loan_extract", "Loan tape"),
        ("collateral_extract", "Collateral tape"),
        ("cashflow_extract", "Cash-flow tape"),
        ("pipeline_report", "Pipeline tape"),
    ])
    def test_each_is_labelled_as_a_tape(self, role, label):
        assert self.vocab().label(role) == label

    @pytest.mark.parametrize("role", ["funder_pi_extract", "property_extract"])
    def test_a_retired_role_is_still_recognised(self, role):
        """No longer ASKED, still CLASSIFIED — so a delivery that already
        carries one keeps its role rather than becoming an unknown file."""
        vocab = self.vocab()
        assert vocab.label(role), f"{role} lost its label and so its identity"
        for outcome in ("mi", "mi_annex2"):
            assert role not in set(vocab.required_roles(outcome)) | set(
                vocab.optional_roles(outcome))
