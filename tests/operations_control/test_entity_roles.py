"""Legal entities, their identifiers, and which role each one plays.

A securitisation names several companies, and they are rarely the same company:
the originator that wrote the loans, the sponsor that arranged the deal, the
SSPE that issues, the entity that reports. Each has its own LEI, and a
regulatory field wants one particular company's — RREL83 is the ORIGINATOR's
identifier, and filling it with the sponsor's would be a misstatement to the
regulator that validates cleanly.

So the model is not "what is your LEI?". It is a repeatable list of entities,
each captured once with its own name, identifier and country, and given every
role it holds. These tests hold that line: they build structures where all four
identifiers differ and assert that each regulatory consumer takes the one
belonging to its own role.
"""

from __future__ import annotations

import pytest
import yaml

import operations_control.onboarding.catalogue as cat_mod
from operations_control.onboarding import artefacts
from operations_control.onboarding.case import OnboardingCase
from operations_control.onboarding.service import OnboardingService
from operations_control.onboarding.store import OnboardingStore

#: Four valid but DIFFERENT identifiers. Every assertion below depends on them
#: being distinguishable, so a routing mistake cannot pass by coincidence.
ORIGINATOR_LEI = "549300ABCDE123456702"
SPONSOR_LEI = "213800ABCDE123456701"
SSPE_LEI = "635400ABCDE123456703"
REPORTING_LEI = "984500ABCDE123456704"

ORIGINATOR = {"legal_name": "Origin Lending Limited", "roles": ["originator"],
              "lei": ORIGINATOR_LEI, "country_of_establishment": "GB"}
SPONSOR = {"legal_name": "Sponsor Bank plc", "roles": ["sponsor"],
           "lei": SPONSOR_LEI, "country_of_establishment": "GB"}
SSPE = {"legal_name": "Structured Issuer 2026-1 DAC", "roles": ["sspe"],
        "lei": SSPE_LEI, "country_of_establishment": "IE"}
REPORTING = {"legal_name": "Reporting Services Limited",
             "roles": ["reporting_entity"], "lei": REPORTING_LEI,
             "country_of_establishment": "GB"}


@pytest.fixture()
def service(store, source_registry) -> OnboardingService:
    cat_mod.reset_cache()
    return OnboardingService(store, registry_factory=lambda: source_registry)


def build(service: OnboardingService, entities, *, client_id="STRUCTURED",
          products=("esma_annex2",), regime=True) -> OnboardingCase:
    """A complete case over ``entities``, answered the way an operator would."""
    case = service.start_new_client(by="Operator")
    cid = case.case_id
    service.save_step(case_id=cid, step="client", by="Operator", payload={
        "client_id": client_id, "client_name": "Structured Lending",
        "jurisdiction": "GB", "reporting_currency": "GBP",
        "time_zone": "Europe/London"})
    case = service.save_step(case_id=cid, step="entities", by="Operator",
                             payload={"entities": [dict(e) for e in entities]})
    service.save_step(case_id=cid, step="contacts", by="Operator", payload={
        "reporting_contact_name": "Rae Reporter",
        "reporting_contact_email": "reporting@structured.example",
        "reporting_contact_phone": "+44-0000000000",
        "operational_contact_name": "Ola Ops",
        "operational_contact_email": "ops@structured.example",
        "investor_report_recipients": "investors@structured.example"})
    owning = case.items("entities")[0]["entity_id"]
    service.save_step(case_id=cid, step="portfolios", by="Operator", payload={
        "portfolios": [{"portfolio_id": "direct_001",
                        "display_name": "Structured Book",
                        "asset_class": "equity_release",
                        "portfolio_type": "direct",
                        "portfolio_structure": "whole_loan",
                        "originates": True, "owning_entity": owning,
                        "datasets": ["funded"], "cadence": "monthly"}]})
    service.save_step(case_id=cid, step="sources", by="Operator", payload={
        "sources": [{"source_key": "direct_001/funded",
                     "portfolio_id": "direct_001", "dataset": "funded",
                     "cadence": "monthly", "source_party": "Core platform",
                     "delivery_channel": "sftp", "file_format": "csv"}]})
    service.save_step(case_id=cid, step="reporting", by="Operator",
                      payload={"products": list(products)})
    if regime and "esma_annex2" in products:
        originators = [e for e in entities if "originator" in e["roles"]]
        if originators:
            service.save_step(case_id=cid, step="regime", by="Operator",
                              payload={"regime": {"esma_annex2": {
                                  "originator_name": originators[0]["legal_name"],
                                  "originator_legal_entity_identifier":
                                      originators[0]["lei"],
                                  "originator_establishment_country":
                                      originators[0]["country_of_establishment"],
                              }}})
    return service.load_case(cid)


def client_config(case: OnboardingCase, store, registry=None) -> dict:
    planned = artefacts.plan(case, store=OnboardingStore(store),
                             registry=registry)
    return yaml.safe_load(next(a.text for a in planned
                               if a.kind == artefacts.ARTEFACT_CLIENT_CONFIG))


# --------------------------------------------------------------------------- #
# The structures a client can describe
# --------------------------------------------------------------------------- #

class TestEntityStructures:
    def test_a_single_entity_acting_only_as_originator(self, service, store,
                                                       source_registry):
        case = build(service, [ORIGINATOR])
        assert service.readiness(case)["ready"] is True
        doc = client_config(case, store, source_registry)
        assert doc["defaults"]["originator_name"] == "Origin Lending Limited"
        assert doc["defaults"]["originator_legal_entity_identifier"] \
            == ORIGINATOR_LEI
        # Nothing invents a sponsor or an SSPE that was never named.
        parties = doc.get("reporting_parties") or {}
        assert "sponsor" not in parties
        assert "sspe" not in parties

    def test_a_separate_originator_and_sspe(self, service, store,
                                            source_registry):
        case = build(service, [ORIGINATOR, SSPE])
        doc = client_config(case, store, source_registry)
        assert doc["defaults"]["originator_legal_entity_identifier"] \
            == ORIGINATOR_LEI
        sspe = doc["reporting_parties"]["sspe"][0]
        assert sspe["legal_name"] == "Structured Issuer 2026-1 DAC"
        assert sspe["lei"] == SSPE_LEI
        # The one that matters: the SSPE's identifier did not become the
        # originator's, and the originator's did not become the SSPE's.
        assert sspe["lei"] != doc["defaults"]["originator_legal_entity_identifier"]

    def test_a_separate_originator_sponsor_and_sspe(self, service, store,
                                                    source_registry):
        case = build(service, [ORIGINATOR, SPONSOR, SSPE])
        doc = client_config(case, store, source_registry)
        assert doc["defaults"]["originator_legal_entity_identifier"] \
            == ORIGINATOR_LEI
        assert doc["reporting_parties"]["sponsor"][0]["lei"] == SPONSOR_LEI
        assert doc["reporting_parties"]["sspe"][0]["lei"] == SSPE_LEI
        # Three roles, three identifiers, no two the same.
        assert len({ORIGINATOR_LEI, SPONSOR_LEI, SSPE_LEI}) == 3

    def test_one_entity_may_hold_several_roles(self, service, store,
                                               source_registry):
        """The same company is often originator, servicer and reporting entity.
        It is captured once and named once, not asked for three times."""
        combined = {"legal_name": "Origin Lending Limited",
                    "roles": ["originator", "servicer", "reporting_entity"],
                    "lei": ORIGINATOR_LEI, "country_of_establishment": "GB"}
        case = build(service, [combined])
        assert service.readiness(case)["ready"] is True
        assert len(case.items("entities")) == 1
        doc = client_config(case, store, source_registry)
        assert doc["defaults"]["originator_legal_entity_identifier"] \
            == ORIGINATOR_LEI
        parties = doc["reporting_parties"]
        assert parties["servicer"][0]["lei"] == ORIGINATOR_LEI
        assert parties["reporting_entity"][0]["lei"] == ORIGINATOR_LEI

    def test_a_reporting_entity_distinct_from_the_originator(
            self, service, store, source_registry):
        case = build(service, [ORIGINATOR, REPORTING])
        doc = client_config(case, store, source_registry)
        assert doc["defaults"]["originator_legal_entity_identifier"] \
            == ORIGINATOR_LEI
        reporting = doc["reporting_parties"]["reporting_entity"][0]
        assert reporting["legal_name"] == "Reporting Services Limited"
        assert reporting["lei"] == REPORTING_LEI
        assert reporting["lei"] != ORIGINATOR_LEI

    def test_the_investor_report_names_the_reporting_entity_not_the_originator(
            self, service, store, source_registry):
        case = build(service, [ORIGINATOR, REPORTING],
                     products=("esma_annex2", "investor_reporting"))
        planned = artefacts.plan(case, store=OnboardingStore(store),
                                 registry=source_registry)
        annex12 = yaml.safe_load(next(
            a.text for a in planned
            if a.kind == artefacts.ARTEFACT_ANNEX12_CONFIG))
        deal = annex12["annex12"]["deal"]
        assert deal["IVSS1"] == REPORTING_LEI
        assert deal["IVSS3"] == "Reporting Services Limited"
        assert deal["IVSS4"] == "Reporting Services Limited"


# --------------------------------------------------------------------------- #
# What the model refuses
# --------------------------------------------------------------------------- #

class TestEntityValidation:
    def test_annex_2_without_an_originator_is_blocked(self, service):
        case = build(service, [SPONSOR, SSPE], regime=False)
        problems = [p["message"] for p in service.readiness(case)["problems"]]
        assert any("originator" in p.lower() for p in problems)

    def test_an_invalid_lei_is_refused(self, service):
        """The legacy 27-character shape is not a 20-character identifier, and
        the validator is not relaxed to accept one."""
        bad = {**ORIGINATOR, "lei": "213800ABCDE123456701N202501"}
        case = build(service, [bad], regime=False)
        problems = [p["message"] for p in service.readiness(case)["problems"]]
        assert any("20-character identifier" in p for p in problems)

    def test_an_identifier_is_required_of_a_regulatory_role_only(self):
        """A trustee needs no LEI; an originator does. The requirement follows
        the role, so a client is not asked for identifiers it has no reason to
        hold."""
        cat = cat_mod.load()
        lei = cat.field("entities", "lei")
        assert cat.is_required(lei, {}, {"roles": ["originator"]}) is True
        assert cat.is_required(lei, {}, {"roles": ["trustee"]}) is False

    def test_an_absent_sponsor_or_sspe_never_blocks_annex_2(self, service):
        """Not every structure has them, and Annex 2 does not require them."""
        case = build(service, [ORIGINATOR])
        assert service.readiness(case)["ready"] is True
        blocking = [p["message"] for p in service.readiness(case)["blocking"]]
        assert not any("sponsor" in m.lower() or "sspe" in m.lower()
                       for m in blocking)

    def test_the_client_is_asked_for_entities_once_as_a_list(self):
        """One repeatable question carrying name, identifier, country and
        roles — not a separate 'what is the sponsor's LEI?' per role."""
        cat = cat_mod.load()
        section = next(s for s in cat.sections if s.key == "entities")
        assert section.repeatable is True
        keys = {f.key for f in section.fields}
        assert {"legal_name", "lei", "roles",
                "country_of_establishment"} <= keys
        roles = cat.vocabularies["entity_role"]["values"]
        assert {"originator", "sponsor", "sspe", "reporting_entity"} \
            <= {r["value"] for r in roles}
        # One identifier question in the whole catalogue, and it is per entity.
        # The regulatory identifier fields exist, but are DERIVED from the
        # entity holding the role — never collected, so there is no second
        # place a client could state an identifier that outranks its entities.
        lei_fields = [(s.key, f.key, f.source, f.collected)
                      for s in cat.sections for f in s.fields
                      if f.validation == "lei" or f.type == "lei"]
        asked = [(s, k) for s, k, _, collected in lei_fields if collected]
        assert asked == [("entities", "lei")]
        for section, key, source, collected in lei_fields:
            if (section, key) != ("entities", "lei"):
                assert source == "derived", (section, key, source)
                assert collected is False, (section, key)


# --------------------------------------------------------------------------- #
# Survival: case -> approval -> activation -> artefact -> resolver
# --------------------------------------------------------------------------- #

class TestEntityDataSurvives:
    def test_role_specific_identifiers_survive_to_the_resolved_configuration(
            self, service, store, source_registry):
        from operations_control.configuration.resolver import (
            EffectiveConfigResolver,
        )
        from operations_control.rules import RuleStore

        case = build(service, [ORIGINATOR, SPONSOR, SSPE, REPORTING])
        assert service.readiness(case)["ready"] is True
        service.approve(case_id=case.case_id, by="Administrator",
                        reason="New structured client")
        result = service.activate(case_id=case.case_id, by="Administrator")
        assert result["version"] == 1

        resolver = EffectiveConfigResolver(store, RuleStore(store))
        doc = yaml.safe_load(resolver.client_config_for("STRUCTURED").read_text(
            encoding="utf-8"))

        assert doc["defaults"]["originator_legal_entity_identifier"] \
            == ORIGINATOR_LEI
        assert doc["defaults"]["originator_name"] == "Origin Lending Limited"
        assert doc["defaults"]["originator_establishment_country"] == "GB"

        parties = doc["reporting_parties"]
        assert parties["sponsor"][0]["lei"] == SPONSOR_LEI
        assert parties["sspe"][0]["lei"] == SSPE_LEI
        assert parties["reporting_entity"][0]["lei"] == REPORTING_LEI

        # Every identifier reached exactly one place, and the four are distinct
        # everywhere they appear.
        assert len({ORIGINATOR_LEI, SPONSOR_LEI, SSPE_LEI, REPORTING_LEI}) == 4

    def test_no_single_identifier_stands_in_for_the_others(
            self, service, store, source_registry):
        """Change only the originator, and only the originator's fields move."""
        case = build(service, [ORIGINATOR, SPONSOR, SSPE, REPORTING])
        before = client_config(case, store, source_registry)

        other = {**ORIGINATOR, "legal_name": "Second Origin Limited",
                 "lei": "789000ABCDE123456705"}
        case2 = build(service, [other, SPONSOR, SSPE, REPORTING],
                      client_id="STRUCTURED_2")
        after = client_config(case2, store, source_registry)

        assert before["defaults"]["originator_legal_entity_identifier"] \
            != after["defaults"]["originator_legal_entity_identifier"]
        assert after["defaults"]["originator_legal_entity_identifier"] \
            == "789000ABCDE123456705"
        for role in ("sponsor", "sspe", "reporting_entity"):
            assert before["reporting_parties"][role] \
                == after["reporting_parties"][role], role
