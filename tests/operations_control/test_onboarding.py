"""Client Onboarding — the governed capability for a client Trakt has never met.

The primary journey is BLANK: no legacy configuration exists, is read, or is
required. Migration and amendments are secondary paths that reach the same model
and generate the same artefacts.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from apps.blob_trigger_app.source_registry import (
    STATUS_ACTIVE,
    STATUS_PENDING_REVIEW,
    SourceRecord,
)

from operations_control.engine import OpsError
import operations_control.onboarding.catalogue as cat_mod
from operations_control.onboarding import artefacts, migration
from operations_control.onboarding.case import (
    ACTIVATED,
    APPROVED,
    AWAITING_CLIENT,
    DRAFT,
    INFORMATION_REQUESTED,
    IN_REVIEW,
    KIND_AMENDMENT,
    KIND_MIGRATION,
    KIND_NEW_CLIENT,
    READY_FOR_APPROVAL,
    WITHDRAWN,
    CaseError,
    OnboardingCase,
)
from operations_control.onboarding.service import OnboardingService
from operations_control.onboarding.store import OnboardingStore

from .conftest import OP_A, OP_ADMIN, OP_B

VALID_LEI = "549300ABCDE123456702"
OTHER_LEI = "213800ABCDE123456701"


@pytest.fixture()
def service(store, source_registry) -> OnboardingService:
    cat_mod.reset_cache()
    return OnboardingService(store, registry_factory=lambda: source_registry)


# --------------------------------------------------------------------------- #
# Building a complete blank case
# --------------------------------------------------------------------------- #

def complete(service: OnboardingService, *, by: str = "Operator",
             client_id: str = "NORDIC",
             products=("esma_annex2",)) -> OnboardingCase:
    """A brand-new client, answered from nothing."""
    case = service.start_new_client(by=by)
    cid = case.case_id
    service.save_step(case_id=cid, step="client", by=by, payload={
        "client_id": client_id, "client_name": "Nordic Lending",
        "jurisdiction": "SE", "reporting_currency": "SEK",
        "time_zone": "Europe/Stockholm"})
    case = service.save_step(case_id=cid, step="entities", by=by, payload={
        "entities": [
            {"legal_name": "Nordic Lending AB",
             "roles": ["originator", "servicer", "reporting_entity"],
             "lei": VALID_LEI, "country_of_establishment": "SE"},
            {"legal_name": "Nordic Trustee ASA", "roles": ["trustee"]}]})
    entity_id = case.items("entities")[0]["entity_id"]
    service.save_step(case_id=cid, step="contacts", by=by, payload={
        "reporting_contact_name": "Rae Reporter",
        "reporting_contact_email": "reporting@nordic.example",
        "operational_contact_name": "Ola Ops",
        "operational_contact_email": "ops@nordic.example",
        "reporting_contact_phone": "+46-000000000"})
    service.save_step(case_id=cid, step="portfolios", by=by, payload={
        "portfolios": [{
            "portfolio_id": "direct_001", "display_name": "Nordic Direct",
            "portfolio_type": "direct", "asset_class": "equity_release",
            "structure": "spv", "owning_entity": entity_id,
            "period_convention": "calendar_month_end"}]})
    service.save_step(case_id=cid, step="reporting", by=by,
                      payload={"products": list(products)})
    service.save_step(case_id=cid, step="sources", by=by, payload={
        "sources": [{
            "source_key": "direct_001/funded", "portfolio_id": "direct_001",
            "dataset": "funded", "cadence": "monthly",
            "source_party": "Nordic core lending platform",
            "delivery_channel": "sftp", "file_format": "csv"}]})
    regime = {}
    if "esma_annex2" in products:
        regime["esma_annex2"] = {
            "originator_name": "Nordic Lending AB",
            "originator_legal_entity_identifier": VALID_LEI,
            "originator_establishment_country": "SE"}
    if "investor_reporting" in products:
        regime["investor_reporting"] = {
            "IVSS5": "Rae Reporter", "IVSS6": "+46-000000000",
            "IVSS7": "reporting@nordic.example", "IVSS8": "VSLC",
            "IVSS9": "ORIG", "IVSS10": "RMRT"}
        service.save_step(case_id=cid, step="contacts", by=by, payload={
            "investor_report_recipients": "investors@nordic.example"})
    if regime:
        service.save_step(case_id=cid, step="regime", by=by,
                          payload={"regime": regime})
    return service.load_case(cid)


def activate(service: OnboardingService, case: OnboardingCase, *,
             by: str = "Administrator", reason: str = "New client"):
    service.approve(case_id=case.case_id, by=by, reason=reason)
    return service.activate(case_id=case.case_id, by=by)


# --------------------------------------------------------------------------- #
# The catalogue
# --------------------------------------------------------------------------- #

class TestCatalogue:
    def test_it_carries_no_client_values(self):
        """The catalogue is a schema. A client name, LEI or ERE-specific default
        appearing in it would make onboarding client-specific again."""
        text = Path("config/onboarding/field_catalogue.yaml").read_text(
            encoding="utf-8")
        for forbidden in ("ERE", "ERM_UK", "Equity Release Mortgages",
                          "213800", "ere_funding"):
            assert forbidden not in text

    def test_vocabularies_are_read_from_the_modules_that_own_them(self):
        from operations_control.configuration.packages import ASSET_MODEL
        from operations_control.contracts import BATCH_DATASETS
        cat = cat_mod.load()
        assert [v["value"] for v in cat.vocabularies["asset_class"]["values"]] \
            == sorted(ASSET_MODEL)
        assert [v["value"] for v in cat.vocabularies["dataset"]["values"]] \
            == list(BATCH_DATASETS)

    def test_every_field_declares_who_supplies_it(self):
        cat = cat_mod.load()
        allowed = {cat_mod.SOURCE_CLIENT, cat_mod.SOURCE_OPERATOR,
                   cat_mod.SOURCE_INFERRED, cat_mod.SOURCE_DERIVED,
                   cat_mod.SOURCE_DEFAULT, cat_mod.SOURCE_SYSTEM,
                   cat_mod.SOURCE_DELIVERY}
        for section in cat.sections:
            for f in section.fields:
                assert f.source in allowed, f"{section.key}.{f.key}"
                assert f.scope, f"{section.key}.{f.key} has no scope"

    def test_delivery_specific_values_are_declared_and_not_collected(self):
        cat = cat_mod.load()
        excluded = {n["key"] for n in cat.not_collected}
        assert {"reporting_period", "static_reporting_date",
                "expected_schema_fingerprint"} <= excluded
        collected = {f.key for s in cat.sections for f in s.fields}
        assert "reporting_period" not in collected

    def test_regime_fields_arrive_from_the_governed_regime_declaration(self):
        cat = cat_mod.load()
        regime = cat.section("regime")
        assert regime is not None and regime.from_regime
        products = {f.product for f in regime.fields}
        assert "esma_annex2" in products and "investor_reporting" in products
        # Enumerations resolve to the values the projector accepts.
        ivss8 = next(f for f in regime.fields if f.key == "IVSS8")
        assert {"VSLC", "OTHR"} <= {o["value"] for o in ivss8.options}

    def test_a_new_regime_needs_no_code_change(self, tmp_path):
        """A future regime added to the governed declaration appears in the
        catalogue — and therefore in the wizard — without touching Python."""
        spec = tmp_path / "standing.yaml"
        spec.write_text(yaml.safe_dump({"version": 1, "products": {
            "future_regime": {"label": "Future Regime", "fields": [
                {"key": "supervisor", "label": "Supervisor",
                 "classification": "standing_client", "required": True,
                 "format": "text", "scope": "regime",
                 "writes_to": "client_config:defaults.supervisor"}]}}}),
            encoding="utf-8")
        cat = cat_mod.load(standing_fields_path=spec)
        keys = {(f.product, f.key) for f in cat.section("regime").fields}
        assert ("future_regime", "supervisor") in keys


class TestConditionalRequirements:
    def test_a_field_is_required_only_when_its_condition_holds(self):
        cat = cat_mod.load()
        phone = cat.field("contacts", "reporting_contact_phone")
        assert cat.is_required(phone,
                               {"reporting": {"products": ["investor_reporting"]}})
        assert not cat.is_required(phone, {"reporting": {"products": []}})

    def test_an_entity_role_drives_its_own_requirements(self):
        cat = cat_mod.load()
        lei = cat.field("entities", "lei")
        assert cat.is_required(lei, {}, {"roles": ["originator"]})
        assert not cat.is_required(lei, {}, {"roles": ["trustee"]})

    def test_an_unparseable_condition_makes_a_field_optional(self):
        """A malformed catalogue must not make onboarding impossible."""
        assert cat_mod.evaluate("this is not an expression", {}) is False


# --------------------------------------------------------------------------- #
# The blank new-client journey — the product
# --------------------------------------------------------------------------- #

class TestNewClientOnboarding:
    def test_a_case_starts_blank_with_a_generated_reference(self, service):
        case = service.start_new_client(by="Operator")
        assert case.case_id.startswith("ONB-")
        assert case.kind == KIND_NEW_CLIENT
        assert case.status == DRAFT
        assert case.client_id == ""
        # Nothing is pre-filled: no client, no entities, no portfolios.
        assert case.answers.get("client", {}) == {}
        assert case.items("entities") == []
        assert case.items("portfolios") == []
        assert case.items("sources") == []

    def test_no_existing_client_needs_to_be_selected(self, service):
        """Nothing about opening or completing a case names an existing
        client."""
        case = complete(service)
        assert case.client_id == "NORDIC"
        assert service.readiness(case)["ready"] is True

    def test_no_legacy_configuration_is_read(self, service, monkeypatch):
        """Point the legacy paths at nothing and the journey is unaffected."""
        monkeypatch.setattr(migration, "DEFAULT_CLIENT_CONFIG",
                            Path("/nonexistent/client.yaml"))
        monkeypatch.setattr(migration, "DEFAULT_ANNEX12_CONFIG",
                            Path("/nonexistent/annex12.yaml"))
        case = complete(service)
        assert service.readiness(case)["ready"] is True
        assert case.base_documents == {}

    def test_a_client_is_created_only_on_activation(self, service, store):
        case = complete(service)
        profiles = OnboardingStore(store)
        rel = artefacts.client_config_rel("NORDIC")

        service.approve(case_id=case.case_id, by="Administrator",
                        reason="New client")
        assert profiles.read_artefact("NORDIC", rel) is None
        assert profiles.current("NORDIC") is None

        activate_result = service.activate(case_id=case.case_id,
                                           by="Administrator")
        assert activate_result["version"] == 1
        assert profiles.read_artefact("NORDIC", rel) is not None
        assert profiles.current("NORDIC").version == 1

    def test_the_generated_configuration_is_built_from_the_answers(self, service,
                                                                   store):
        case = complete(service)
        activate(service, case)
        text = OnboardingStore(store).read_artefact(
            "NORDIC", artefacts.client_config_rel("NORDIC"))
        doc = yaml.safe_load(text)
        assert doc["client"]["client_id"] == "NORDIC"
        assert doc["client"]["display_name"] == "Nordic Lending"
        assert doc["portfolio"]["country"] == "SE"
        assert doc["portfolio"]["base_currency"] == "SEK"
        assert doc["defaults"]["originator_legal_entity_identifier"] == VALID_LEI
        assert doc["pipeline"]["esma_enabled"] is True
        assert doc["regime"] == "ESMA_Annex2"        # legacy compatibility key

    def test_nothing_is_copied_from_any_other_client(self, service, store):
        case = complete(service)
        activate(service, case)
        text = OnboardingStore(store).read_artefact(
            "NORDIC", artefacts.client_config_rel("NORDIC"))
        for foreign in ("ERE", "Equity Release Mortgages", OTHER_LEI):
            assert foreign not in text
        doc = yaml.safe_load(text)
        assert "enrichment" not in doc
        assert "transformations" not in doc

    def test_several_portfolios_can_be_created_before_any_delivery(self,
                                                                   service,
                                                                   source_registry):
        case = complete(service)
        entity_id = case.items("entities")[0]["entity_id"]
        service.save_step(
            case_id=case.case_id, step="portfolios", by="Operator", payload={
                "portfolios": [
                    {**case.items("portfolios")[0]},
                    {"portfolio_id": "acquired_001",
                     "display_name": "Acquired Book",
                     "portfolio_type": "acquired",
                     "asset_class": "equity_release", "structure": "spv",
                     "owning_entity": entity_id,
                     "period_convention": "calendar_month_end"}]})
        case = service.save_step(
            case_id=case.case_id, step="sources", by="Operator", payload={
                "sources": [
                    {"source_key": "direct_001/funded",
                     "portfolio_id": "direct_001", "dataset": "funded",
                     "cadence": "monthly", "source_party": "Nordic core",
                     "delivery_channel": "sftp", "file_format": "csv"},
                    {"source_key": "acquired_001/funded",
                     "portfolio_id": "acquired_001", "dataset": "funded",
                     "cadence": "ad_hoc", "source_party": "Seller",
                     "delivery_channel": "blob_upload", "file_format": "xlsx"}]})
        activate(service, case)
        keys = {r.key for r in source_registry.records()}
        assert keys == {"NORDIC/direct_001/funded/monthly",
                        "NORDIC/acquired_001/funded/ad_hoc"}

    def test_source_registrations_exist_before_the_first_delivery(
            self, service, source_registry):
        case = complete(service)
        activate(service, case)
        record = source_registry.lookup("NORDIC", "direct_001", "funded",
                                        "monthly")
        assert record is not None
        assert record.status == STATUS_PENDING_REVIEW
        assert record.expected_schema_fingerprint is None
        assert record.last_successful_run_id is None
        assert record.regime_required is True
        assert record.source_system == "Nordic core lending platform"

    def test_a_pipeline_book_is_added_deliberately_not_assumed(self, service):
        case = complete(service)
        assert [s["dataset"] for s in case.items("sources")] == ["funded"]
        case = service.add_pipeline_source(case_id=case.case_id,
                                           portfolio_id="direct_001",
                                           by="Operator")
        assert {s["dataset"] for s in case.items("sources")} == {"funded",
                                                                 "pipeline"}

    def test_regulatory_scope_is_derived_and_never_on_a_pipeline_book(
            self, service):
        case = complete(service)
        case = service.add_pipeline_source(case_id=case.case_id,
                                           portfolio_id="direct_001",
                                           by="Operator")
        scope = {s["dataset"]: s["regime_required"] for s in case.items("sources")}
        assert scope == {"funded": True, "pipeline": False}

    def test_standing_regime_information_is_collected_and_generated(
            self, service, store):
        case = complete(service, products=("esma_annex2", "investor_reporting"))
        activate(service, case)
        profiles = OnboardingStore(store)
        annex12 = yaml.safe_load(profiles.read_artefact(
            "NORDIC", artefacts.annex12_config_rel("NORDIC")))
        deal = annex12["annex12"]["deal"]
        assert deal["IVSS8"] == "VSLC"          # risk retention method
        assert deal["IVSS9"] == "ORIG"          # risk retention holder
        assert deal["IVSS7"] == "reporting@nordic.example"

    def test_delivery_specific_values_are_never_written(self, service, store):
        case = complete(service)
        activate(service, case)
        doc = yaml.safe_load(OnboardingStore(store).read_artefact(
            "NORDIC", artefacts.client_config_rel("NORDIC")))
        assert "static_reporting_date" not in (doc.get("portfolio") or {})
        assert "reporting_date" not in (doc.get("loan_engine") or {})

    def test_the_client_index_is_generated(self, service, store):
        case = complete(service)
        activate(service, case)
        doc = yaml.safe_load(OnboardingStore(store).read_artefact(
            "NORDIC", artefacts.tenancy_rel("NORDIC")))
        tenant = doc["tenants"]["NORDIC"]
        assert tenant["display_name"] == "Nordic Lending"
        assert "direct_001" in tenant["portfolios"]

    def test_activation_is_idempotent(self, service, store, source_registry):
        """Applying an approved case again converges: the same documents, the
        same registrations, and no second version of the client."""
        case = complete(service)
        first = activate(service, case)
        profiles = OnboardingStore(store)
        rel = artefacts.client_config_rel("NORDIC")
        before_text = profiles.read_artefact("NORDIC", rel)
        before_records = {r.key for r in source_registry.records()}

        case = service.load_case(case.case_id)
        again = artefacts.apply(case, store=profiles, version=99,
                                registry=source_registry)
        assert {a["action"] for a in again if a["kind"] != "source_registry"} \
            == {"unchanged"}
        assert profiles.read_artefact("NORDIC", rel) == before_text
        assert {r.key for r in source_registry.records()} == before_records

        # Committing the same answers returns the version already in force
        # rather than forking the client's history.
        repeat = profiles.commit(case=case, changes=[], artefacts=again,
                                 activated_by="Administrator")
        assert repeat.version == first["version"] == 1
        assert len(profiles.history("NORDIC")) == 1

    def test_generation_is_deterministic(self, service, store):
        case = complete(service)
        planned_a = artefacts.plan(case, store=OnboardingStore(store))
        planned_b = artefacts.plan(case, store=OnboardingStore(store))
        assert [a.text for a in planned_a] == [a.text for a in planned_b]


# --------------------------------------------------------------------------- #
# Entities
# --------------------------------------------------------------------------- #

class TestEntities:
    def test_one_entity_can_hold_several_roles_without_duplication(self,
                                                                   service,
                                                                   store):
        case = complete(service)
        entities = case.items("entities")
        assert len(entities) == 2
        assert set(entities[0]["roles"]) == {"originator", "servicer",
                                             "reporting_entity"}
        activate(service, case)
        doc = yaml.safe_load(OnboardingStore(store).read_artefact(
            "NORDIC", artefacts.client_config_rel("NORDIC")))
        # Named once as the originator, once as the servicer — same identity.
        assert doc["defaults"]["originator_name"] == "Nordic Lending AB"
        assert doc["reporting_parties"]["servicer"][0]["legal_name"] \
            == "Nordic Lending AB"

    def test_entity_references_are_generated(self, service):
        case = complete(service)
        assert all(e["entity_id"].startswith("ent_")
                   for e in case.items("entities"))

    def test_regulatory_reporting_requires_an_originator(self, service):
        case = service.start_new_client(by="Operator")
        service.save_step(case_id=case.case_id, step="reporting", by="Operator",
                          payload={"products": ["esma_annex2"]})
        problems = [p.message for p in service.readiness(
            service.load_case(case.case_id))["problems"]
            and service._validator().validate(service.load_case(case.case_id))]
        assert any("originator role" in m for m in problems)

    def test_a_portfolio_cannot_point_at_a_removed_entity(self, service):
        case = complete(service)
        service.save_step(case_id=case.case_id, step="entities", by="Operator",
                          payload={"entities": [
                              {"legal_name": "Someone Else",
                               "roles": ["originator"], "lei": VALID_LEI,
                               "country_of_establishment": "SE"}]})
        case = service.load_case(case.case_id)
        messages = [p.message for p in service._validator().validate(case)]
        assert any("no longer listed" in m for m in messages)


# --------------------------------------------------------------------------- #
# Validation
# --------------------------------------------------------------------------- #

class TestValidation:
    def test_an_invalid_identifier_is_refused(self, service):
        case = complete(service)
        service.save_step(case_id=case.case_id, step="entities", by="Operator",
                          payload={"entities": [
                              {**case.items("entities")[0], "lei": "NOT-A-LEI"}]})
        case = service.load_case(case.case_id)
        messages = [p.message for p in service._validator().validate(case)]
        assert any("20-character identifier" in m for m in messages)

    def test_a_duplicate_client_identifier_is_refused(self, service):
        first = complete(service)
        activate(service, first)
        second = complete(service, client_id="NORDIC")
        messages = [p.message for p in service._validator().validate(second)]
        assert any("already in use" in m for m in messages)

    def test_a_duplicate_portfolio_identifier_is_refused(self, service):
        case = complete(service)
        entity_id = case.items("entities")[0]["entity_id"]
        base = case.items("portfolios")[0]
        service.save_step(case_id=case.case_id, step="portfolios",
                          by="Operator",
                          payload={"portfolios": [base, {**base}]})
        case = service.load_case(case.case_id)
        messages = [p.message for p in service._validator().validate(case)]
        assert any("listed twice" in m for m in messages)

    def test_a_regime_the_asset_cannot_support_is_refused(self, service):
        case = complete(service)
        service.save_step(case_id=case.case_id, step="portfolios",
                          by="Operator", payload={"portfolios": [
                              {**case.items("portfolios")[0],
                               "asset_class": "not_a_real_asset"}]})
        case = service.load_case(case.case_id)
        messages = [p.message for p in service._validator().validate(case)]
        assert any("asset class" in m for m in messages)

    def test_a_regime_that_does_not_apply_asks_for_nothing(self, service):
        """Investor reporting fields must not block a client who does not
        receive investor reporting."""
        case = complete(service, products=("esma_annex2",))
        problems = service._validator().blocking(case)
        assert not any(p.field.startswith("investor_reporting")
                       for p in problems)
        assert service.readiness(case)["ready"] is True

    def test_selecting_a_regime_activates_its_requirements(self, service):
        case = complete(service, products=("esma_annex2",))
        service.save_step(case_id=case.case_id, step="reporting", by="Operator",
                          payload={"products": ["esma_annex2",
                                                "investor_reporting"]})
        case = service.load_case(case.case_id)
        fields = {p.field for p in service._validator().blocking(case)}
        assert any(f.startswith("investor_reporting.IVSS") for f in fields)

    def test_problems_carry_no_file_paths(self, service):
        case = service.start_new_client(by="Operator")
        for problem in service._validator().validate(case):
            assert "config/" not in problem.message
            assert ".yaml" not in problem.message


# --------------------------------------------------------------------------- #
# Information requests
# --------------------------------------------------------------------------- #

class TestInformationRequests:
    def test_the_checklist_asks_only_for_what_the_client_can_answer(self,
                                                                    service):
        case = service.start_new_client(by="Operator")
        service.save_step(case_id=case.case_id, step="client", by="Operator",
                          payload={"client_id": "NEWCO"})
        case = service.load_case(case.case_id)
        checklist = service.client_checklist(case)
        labels = {row["label"] for row in checklist}
        assert "Client name" in labels
        # The identifier is the operator's decision, not the client's.
        assert "Client identifier" not in labels

    def test_a_request_moves_the_case_through_its_lifecycle(self, service):
        case = complete(service)
        case = service.create_request(
            case_id=case.case_id, by="Operator",
            items=[{"section": "entities", "field": "registration_number",
                    "label": "Company registration number", "index": 0}])
        assert case.status == INFORMATION_REQUESTED
        case = service.mark_request_sent(case_id=case.case_id, by="Operator",
                                         request_id=case.requests()[0].request_id)
        assert case.status == AWAITING_CLIENT
        case = service.record_response(
            case_id=case.case_id, request_id=case.requests()[0].request_id,
            by="Operator", note="Received by email",
            evidence=[{"name": "Certificate", "reference": "DOC-1"}])
        assert case.status == IN_REVIEW
        request = case.requests()[0]
        assert request.evidence[0]["name"] == "Certificate"
        assert request.responded_at

    def test_an_outstanding_request_blocks_approval(self, service):
        case = complete(service)
        case = service.create_request(
            case_id=case.case_id, by="Operator",
            items=[{"section": "entities", "field": "registration_number",
                    "label": "Company registration number", "index": 0}])
        assert service.readiness(case)["ready"] is False
        with pytest.raises(OpsError) as exc:
            service.approve(case_id=case.case_id, by="Administrator",
                            reason="too early")
        assert exc.value.code == "OPS_ONBOARDING_INCOMPLETE"

    def test_a_response_can_supply_the_answers(self, service):
        case = complete(service)
        case = service.create_request(
            case_id=case.case_id, by="Operator",
            items=[{"section": "presentation", "field": "report_title",
                    "label": "Report title", "index": None}])
        case = service.record_response(
            case_id=case.case_id, request_id=case.requests()[0].request_id,
            by="Operator", answers={"presentation": {
                "report_title": "Nordic Portfolio MI"}})
        assert case.answers["presentation"]["report_title"] \
            == "Nordic Portfolio MI"

    def test_an_unresolved_question_is_visible(self, service):
        case = complete(service)
        case = service.add_question(case_id=case.case_id, by="Operator",
                                    question="Which entity holds the risk?")
        assert len(case.unresolved_questions) == 1
        case = service.resolve_question(
            case_id=case.case_id, by="Operator",
            question_id=case.open_questions[0]["question_id"],
            resolution="Confirmed as the originator")
        assert case.unresolved_questions == []


# --------------------------------------------------------------------------- #
# Governance
# --------------------------------------------------------------------------- #

class TestGovernance:
    def test_approval_writes_no_configuration(self, service, store,
                                              source_registry):
        case = complete(service)
        service.approve(case_id=case.case_id, by="Administrator",
                        reason="Approved")
        assert OnboardingStore(store).read_artefact(
            "NORDIC", artefacts.client_config_rel("NORDIC")) is None
        assert source_registry.records() == []

    def test_activation_requires_approval(self, service):
        case = complete(service)
        with pytest.raises(OpsError) as exc:
            service.activate(case_id=case.case_id, by="Administrator")
        assert exc.value.code == "OPS_ONBOARDING_NOT_APPROVED"

    def test_approval_requires_a_reason(self, service):
        case = complete(service)
        with pytest.raises(OpsError) as exc:
            service.approve(case_id=case.case_id, by="Administrator", reason="")
        assert exc.value.code == "OPS_REASON_REQUIRED"

    def test_withdrawal_writes_no_configuration_and_keeps_the_record(
            self, service, store, source_registry):
        case = complete(service)
        case = service.withdraw(case_id=case.case_id, by="Operator",
                                reason="Client did not proceed")
        assert case.status == WITHDRAWN
        assert case.withdrawal_reason == "Client did not proceed"
        assert OnboardingStore(store).read_artefact(
            "NORDIC", artefacts.client_config_rel("NORDIC")) is None
        assert source_registry.records() == []
        assert service.load_case(case.case_id).status == WITHDRAWN

    def test_a_withdrawn_case_cannot_be_edited(self, service):
        case = complete(service)
        service.withdraw(case_id=case.case_id, by="Operator", reason="Stopped")
        with pytest.raises(OpsError) as exc:
            service.save_step(case_id=case.case_id, step="client",
                              by="Operator", payload={"client_name": "X"})
        assert exc.value.code == "OPS_ONBOARDING_LOCKED"

    def test_an_illegal_transition_is_refused_in_operator_wording(self, service):
        case = complete(service)
        service.withdraw(case_id=case.case_id, by="Operator", reason="Stopped")
        case = service.load_case(case.case_id)
        with pytest.raises(CaseError) as exc:
            case.transition(APPROVED, actor="Administrator")
        assert "cannot move from withdrawn" in exc.value.message

    def test_the_case_history_records_who_when_what_before_after_and_why(
            self, service):
        case = complete(service)
        service.save_step(case_id=case.case_id, step="client", by="Operator",
                          payload={"client_name": "Nordic Lending Group"})
        case = service.load_case(case.case_id)
        event = [e for e in case.events if e["event"] == "answered_client"][-1]
        assert event["actor"] == "Operator"
        assert event["at"]
        assert event["before"]["client"]["client_name"] == "Nordic Lending"
        assert event["after"]["client"]["client_name"] == "Nordic Lending Group"

    def test_activation_appends_to_the_hash_chained_audit(self, service, store):
        case = complete(service)
        activate(service, case)
        rows = store.list_audit("NORDIC")
        events = [r["event"] for r in rows]
        assert "onboarding_configuration_activated" in events
        entry = next(r for r in rows
                     if r["event"] == "onboarding_configuration_activated")
        assert entry["actor"] == "Administrator"
        assert entry["detail"]["reason"] == "New client"
        assert entry["detail"]["artefacts"]
        assert store.verify_audit_chain("NORDIC") is True

    def test_the_preview_shows_generated_identifiers_and_defaults(self, service):
        case = complete(service)
        preview = service.preview(case)
        values = [g["value"] for g in preview["generated_identifiers"]]
        assert case.case_id in values
        assert any(v.startswith("raw-v2/NORDIC/") for v in values)
        assert any(d["label"] == "Environment"
                   for d in preview["defaults_used"])

    def test_the_preview_reports_what_cannot_be_represented(self, service):
        case = complete(service)
        keys = {u.get("key") for u in service.preview(case)["unrepresented"]}
        assert {"sponsor_name", "sspe_name", "sts_status"} <= keys
        assert "authorised_approver" in keys      # collected, no artefact


# --------------------------------------------------------------------------- #
# Migration — secondary
# --------------------------------------------------------------------------- #

@pytest.fixture()
def legacy(tmp_path, source_registry):
    client_config = tmp_path / "config_client_LEGACY.yaml"
    client_config.write_text(yaml.safe_dump({
        "client": {"client_id": "LEGACY", "display_name": "Legacy Funding",
                   "environment": "production"},
        "portfolio": {"asset_class": "equity_release", "country": "GB",
                      "base_currency": "GBP"},
        "supported_regimes": ["ESMA_Annex2"],
        "pipeline": {"mi_enabled": True, "esma_enabled": True},
        "defaults": {"originator_name": "Legacy Funding Limited",
                     # Deliberately the wrong length — the real ERE file has
                     # a 27-character value where 20 is required.
                     "originator_legal_entity_identifier": "213800ABCDE123456701N202501",
                     "originator_establishment_country": "GB"},
        "transformations": {"copy_if_missing": {"a": "b"}},
        "enrichment": {"enable": {"uk_nuts3": True}},
        "mi": {"branding": {"app_title": "Legacy MI"}},
    }, sort_keys=False), encoding="utf-8")
    annex12 = tmp_path / "annex12.yaml"
    annex12.write_text(yaml.safe_dump({"annex12": {
        "deal": {"IVSS4": "Legacy Funding Limited", "IVSS5": "Ada Reporter",
                 "IVSS7": "reporting@legacy.example", "IVSS8": "VSLC",
                 "IVSR1": "TRIGGER", "IVSF1": "CF_001"}}}, sort_keys=False),
        encoding="utf-8")
    portfolios = tmp_path / "portfolios.yaml"
    portfolios.write_text(yaml.safe_dump({"clients": {"LEGACY": {"portfolios": [
        {"source_portfolio_id": "direct_001",
         "source_portfolio_type": "direct",
         "source_portfolio_label": "Legacy Direct", "originates": True}]}}},
        sort_keys=False), encoding="utf-8")
    source_registry.upsert(SourceRecord(
        client_id="LEGACY", source_portfolio_id="direct_001",
        source_portfolio_type="direct", dataset="funded", frequency="monthly",
        source_system="Legacy core", approved_mapping_id="legacy_v1",
        expected_schema_fingerprint="sha256:proven",
        file_role_schemas={"loan_extract": ["a", "b"]},
        last_successful_reporting_period="2025-11-30",
        regime_required=True, status=STATUS_ACTIVE))
    return {"client_config_path": client_config,
            "annex12_config_path": annex12,
            "portfolio_registry_path": portfolios,
            "tenancy_path": tmp_path / "absent.yaml"}


@pytest.fixture()
def migrating(store, source_registry, legacy) -> OnboardingService:
    cat_mod.reset_cache()
    return OnboardingService(store, registry_factory=lambda: source_registry,
                             adopt_paths=legacy)


class TestMigration:
    def test_migration_is_not_required_for_new_client_onboarding(self, service):
        """The primary journey completes with the migration module's default
        paths pointing at a client that has nothing to do with it."""
        case = complete(service)
        assert case.kind == KIND_NEW_CLIENT
        assert service.readiness(case)["ready"] is True

    def test_it_populates_the_same_generic_model(self, migrating):
        case = migrating.start_migration(client_id="LEGACY", by="Operator")
        assert case.kind == KIND_MIGRATION
        assert case.answers["client"]["client_name"] == "Legacy Funding"
        assert case.answers["client"]["jurisdiction"] == "GB"
        assert [p["portfolio_id"] for p in case.items("portfolios")] \
            == ["direct_001"]
        assert set(case.answers["reporting"]["products"]) == {
            "esma_annex2", "investor_reporting"}

    def test_legacy_values_become_entities_with_roles(self, migrating):
        case = migrating.start_migration(client_id="LEGACY", by="Operator")
        entities = case.items("entities")
        originator = next(e for e in entities if "originator" in e["roles"])
        assert originator["legal_name"] == "Legacy Funding Limited"
        # The same company is the reporting entity: captured once, two roles.
        assert "reporting_entity" in originator["roles"]

    def test_an_invalid_legacy_value_surfaces_as_a_migration_issue(self,
                                                                   migrating):
        case = migrating.start_migration(client_id="LEGACY", by="Operator")
        questions = [q["question"] for q in case.unresolved_questions]
        assert any("20-character identifier" in q for q in questions)
        assert any("carried over from the existing configuration" in q
                   for q in questions)

    def test_migration_changes_no_active_configuration(self, migrating, store,
                                                       legacy, source_registry):
        before = {k: Path(v).read_text(encoding="utf-8")
                  for k, v in legacy.items() if Path(v).exists()}
        record_before = source_registry.lookup("LEGACY", "direct_001", "funded",
                                               "monthly").to_dict()
        migrating.start_migration(client_id="LEGACY", by="Operator")
        after = {k: Path(v).read_text(encoding="utf-8")
                 for k, v in legacy.items() if Path(v).exists()}
        assert before == after
        assert OnboardingStore(store).current("LEGACY") is None
        assert source_registry.lookup("LEGACY", "direct_001", "funded",
                                      "monthly").to_dict() == record_before

    def test_every_adopted_value_carries_its_origin(self, migrating):
        case = migrating.start_migration(client_id="LEGACY", by="Operator")
        assert "RREL82/83/84" in case.provenance["entities.originator"]
        assert case.provenance["sources"] == "the durable source registrations"

    def test_blocks_onboarding_does_not_own_survive(self, migrating, store):
        case = migrating.start_migration(client_id="LEGACY", by="Operator")
        planned = artefacts.plan(case, store=OnboardingStore(store))
        doc = yaml.safe_load(next(a.text for a in planned
                                  if a.kind == artefacts.ARTEFACT_CLIENT_CONFIG))
        assert doc["enrichment"]["enable"]["uk_nuts3"] is True
        assert doc["transformations"]["copy_if_missing"] == {"a": "b"}
        annex12 = yaml.safe_load(next(
            a.text for a in planned
            if a.kind == artefacts.ARTEFACT_ANNEX12_CONFIG))
        assert annex12["annex12"]["deal"]["IVSR1"] == "TRIGGER"

    def test_an_approved_mapping_is_never_reset(self, migrating,
                                                source_registry):
        case = migrating.start_migration(client_id="LEGACY", by="Operator")
        records, _ = artefacts.render_source_records(case,
                                                     registry=source_registry)
        funded = next(r for r in records if r.dataset == "funded")
        assert funded.approved_mapping_id == "legacy_v1"
        assert funded.expected_schema_fingerprint == "sha256:proven"
        assert funded.file_role_schemas == {"loan_extract": ["a", "b"]}
        assert funded.status == STATUS_ACTIVE


# --------------------------------------------------------------------------- #
# Amendments
# --------------------------------------------------------------------------- #

class TestAmendments:
    def test_an_amendment_starts_from_the_active_version(self, service):
        case = complete(service)
        activate(service, case)
        amendment = service.start_amendment(client_id="NORDIC", by="Operator")
        assert amendment.kind == KIND_AMENDMENT
        assert amendment.based_on_version == 1
        assert amendment.answers["client"]["client_name"] == "Nordic Lending"

    def test_a_client_with_no_configuration_cannot_be_amended(self, service):
        with pytest.raises(OpsError) as exc:
            service.start_amendment(client_id="GHOST", by="Operator")
        assert exc.value.code == "OPS_CLIENT_NOT_ONBOARDED"

    def test_approval_creates_a_new_version_and_keeps_the_old_one(self, service,
                                                                  store):
        case = complete(service)
        activate(service, case)
        amendment = service.start_amendment(client_id="NORDIC", by="Operator")
        service.save_step(case_id=amendment.case_id, step="contacts",
                          by="Operator",
                          payload={"reporting_contact_email":
                                   "new@nordic.example"})
        result = activate(service, service.load_case(amendment.case_id),
                          reason="Contact changed")
        assert result["version"] == 2

        profiles = OnboardingStore(store)
        assert profiles.current("NORDIC").version == 2
        first = profiles.version("NORDIC", 1)
        assert first.status == "superseded"
        assert first.answers["contacts"]["reporting_contact_email"] \
            == "reporting@nordic.example"

    def test_history_shows_the_change_with_before_and_after(self, service):
        case = complete(service)
        activate(service, case)
        amendment = service.start_amendment(client_id="NORDIC", by="Operator")
        service.save_step(case_id=amendment.case_id, step="client",
                          by="Operator",
                          payload={"client_name": "Nordic Lending Group"})
        activate(service, service.load_case(amendment.case_id),
                 reason="Rebranded")
        history = service.client_view("NORDIC")["history"]
        assert [h["version"] for h in history] == [2, 1]
        change = next(c for c in history[0]["changes"]
                      if c["path"] == "client.client_name")
        assert change["before"] == "Nordic Lending"
        assert change["after"] == "Nordic Lending Group"
        assert history[0]["reason"] == "Rebranded"

    def test_an_invalid_amendment_cannot_activate(self, service):
        case = complete(service)
        activate(service, case)
        amendment = service.start_amendment(client_id="NORDIC", by="Operator")
        service.save_step(case_id=amendment.case_id, step="client",
                          by="Operator", payload={"reporting_currency": "SEKK"})
        with pytest.raises(OpsError) as exc:
            service.approve(case_id=amendment.case_id, by="Administrator",
                            reason="Bad currency")
        assert exc.value.code == "OPS_ONBOARDING_INCOMPLETE"

    def test_an_amendment_does_not_trip_the_identifier_collision_check(
            self, service):
        """The client already owns its identifier — reusing it is the point."""
        case = complete(service)
        activate(service, case)
        amendment = service.start_amendment(client_id="NORDIC", by="Operator")
        messages = [p.message for p in service._validator().validate(amendment)]
        assert not any("already in use" in m for m in messages)


# --------------------------------------------------------------------------- #
# The home queues
# --------------------------------------------------------------------------- #

class TestHome:
    def test_cases_appear_in_the_queue_matching_their_status(self, service):
        draft = service.start_new_client(by="Operator")
        working = complete(service, client_id="OTHERCO")
        service.create_request(
            case_id=working.case_id, by="Operator",
            items=[{"section": "presentation", "field": "report_title",
                    "label": "Report title", "index": None}])
        home = service.home()
        assert draft.case_id in {r["case_id"] for r in home["drafts"]}
        assert working.case_id in {r["case_id"]
                                   for r in home["awaiting_client"]}

    def test_an_activated_client_appears_as_active(self, service):
        case = complete(service)
        activate(service, case)
        home = service.home()
        row = next(c for c in home["active_clients"]
                   if c["client_id"] == "NORDIC")
        assert row["version"] == 1
        assert row["portfolios"] == 1

    def test_a_legacy_client_is_offered_for_migration_only(self, service,
                                                           source_registry):
        source_registry.upsert(SourceRecord(
            client_id="OLDCO", source_portfolio_id="p1", dataset="funded",
            frequency="monthly"))
        home = service.home()
        assert "OLDCO" in {c["client_id"] for c in home["legacy_clients"]}
        assert "OLDCO" not in {c["client_id"] for c in home["active_clients"]}


# --------------------------------------------------------------------------- #
# The resolver seam
# --------------------------------------------------------------------------- #

class TestResolverSeam:
    def test_a_client_without_configuration_uses_the_repository_file(self,
                                                                     store,
                                                                     tmp_path):
        from operations_control.configuration.resolver import EffectiveConfigResolver
        from operations_control.rules import RuleStore
        repo_file = tmp_path / "repo.yaml"
        repo_file.write_text("client: {client_id: X}\n", encoding="utf-8")
        resolver = EffectiveConfigResolver(store, RuleStore(store),
                                           client_config_path=repo_file)
        assert resolver.client_config_for("UNKNOWN") == repo_file

    def test_an_activated_client_resolves_with_its_generated_configuration(
            self, service, store, tmp_path):
        from operations_control.configuration.resolver import EffectiveConfigResolver
        from operations_control.rules import RuleStore
        case = complete(service)
        activate(service, case)
        repo_file = tmp_path / "repo.yaml"
        repo_file.write_text("client: {client_id: X}\n", encoding="utf-8")
        resolver = EffectiveConfigResolver(store, RuleStore(store),
                                           client_config_path=repo_file)
        path = resolver.client_config_for("NORDIC")
        assert path != repo_file
        doc = yaml.safe_load(path.read_text(encoding="utf-8"))
        assert doc["client"]["client_id"] == "NORDIC"

    def test_the_generated_configuration_satisfies_the_regulatory_preflight(
            self, service, store):
        from operations_control.annex2 import preflight
        from operations_control.configuration.resolver import EffectiveConfigResolver
        from operations_control.rules import RuleStore
        case = complete(service)
        activate(service, case)
        resolver = EffectiveConfigResolver(store, RuleStore(store))
        result = preflight.run_preflight(
            client_config_path=resolver.client_config_for("NORDIC"),
            reporting_period="2026-06-30")
        assert result["blocked"] == []


# --------------------------------------------------------------------------- #
# API + tenancy
# --------------------------------------------------------------------------- #

@pytest.fixture()
def api(store, source_registry, ops_env):
    from fastapi.testclient import TestClient
    from operations_control.api import app as app_module
    from operations_control.engine import OpsEngine
    cat_mod.reset_cache()
    engine = OpsEngine(store, source_registry_factory=lambda: source_registry)
    app_module.set_engine(engine)
    yield TestClient(app_module.app)
    app_module.set_engine(None)


ADMIN = {"X-Operator-Token": OP_ADMIN}
ALICE = {"X-Operator-Token": OP_A}
BOB = {"X-Operator-Token": OP_B}


class TestApi:
    def test_the_information_model_is_served(self, api):
        body = api.get("/ops/onboarding/reference", headers=ADMIN).json()
        assert body["ok"] is True
        sections = {s["key"] for s in body["catalogue"]["sections"]}
        assert {"client", "entities", "portfolios", "sources"} <= sections
        assert {s["key"] for s in body["steps"]} >= {"client", "review"}

    def test_a_new_client_case_starts_blank_over_http(self, api):
        case = api.post("/ops/onboarding/cases", headers=ADMIN).json()["case"]
        assert case["case_id"].startswith("ONB-")
        assert case["client_id"] == ""
        assert case["status"] == DRAFT

    def test_an_unauthenticated_caller_is_refused(self, api):
        assert api.get("/ops/onboarding/home").status_code == 401

    def test_an_ordinary_operator_may_work_a_case_but_not_approve(self, api):
        case_id = api.post("/ops/onboarding/cases",
                           headers=ALICE).json()["case"]["case_id"]
        saved = api.put(f"/ops/onboarding/cases/{case_id}",
                        json={"step": "client",
                              "payload": {"client_id": "client_a",
                                          "client_name": "A"}},
                        headers=ALICE)
        assert saved.status_code == 200
        refused = api.post(f"/ops/onboarding/cases/{case_id}/approve",
                           json={"reason": "trying"}, headers=ALICE)
        assert refused.status_code == 403
        assert refused.json()["detail"]["errorCode"] == "OPS_ADMIN_REQUIRED"

    def test_a_case_cannot_be_named_after_another_tenants_client(self, api):
        case_id = api.post("/ops/onboarding/cases",
                           headers=ALICE).json()["case"]["case_id"]
        refused = api.put(f"/ops/onboarding/cases/{case_id}",
                          json={"step": "client",
                                "payload": {"client_id": "client_b"}},
                          headers=ALICE)
        assert refused.status_code == 404

    def test_another_tenants_case_is_not_found(self, api):
        case_id = api.post("/ops/onboarding/cases",
                           headers=ALICE).json()["case"]["case_id"]
        api.put(f"/ops/onboarding/cases/{case_id}",
                json={"step": "client", "payload": {"client_id": "client_a"}},
                headers=ALICE)
        assert api.get(f"/ops/onboarding/cases/{case_id}",
                       headers=BOB).status_code == 404

    def test_an_illegal_transition_returns_an_operator_safe_message(self, api):
        case_id = api.post("/ops/onboarding/cases",
                           headers=ADMIN).json()["case"]["case_id"]
        api.put(f"/ops/onboarding/cases/{case_id}",
                json={"step": "client", "payload": {"client_id": "client_a"}},
                headers=ADMIN)
        api.post(f"/ops/onboarding/cases/{case_id}/withdraw",
                 json={"reason": "Stopped"}, headers=ADMIN)
        # A withdrawn case cannot be resurrected by approving it.
        refused = api.post(f"/ops/onboarding/cases/{case_id}/approve",
                           json={"reason": "Changed my mind"}, headers=ADMIN)
        assert refused.status_code == 409
        assert "withdrawn" in refused.json()["message"].lower()


# --------------------------------------------------------------------------- #
# Inference — the questions Trakt answers for itself
# --------------------------------------------------------------------------- #

class TestInference:
    """Every rule here removes a question. Each is overridable, and each records
    where the value came from."""

    def test_jurisdiction_supplies_the_currency_and_the_clock(self, service):
        case = service.start_new_client(by="Operator")
        case = service.save_step(case_id=case.case_id, step="client",
                                 by="Operator",
                                 payload={"client_name": "Nordic Lending",
                                          "jurisdiction": "SE"})
        client = case.block("client")
        assert client["reporting_currency"] == "SEK"
        assert client["time_zone"] == "Europe/Stockholm"
        assert "SE" in case.provenance["client.reporting_currency"]

    def test_an_unlisted_jurisdiction_does_not_guess_a_currency(self, service):
        """A country with no rule leaves the currency as a question rather than
        inventing one."""
        case = service.start_new_client(by="Operator")
        case = service.save_step(case_id=case.case_id, step="client",
                                 by="Operator",
                                 payload={"client_name": "Somewhere",
                                          "jurisdiction": "ZZ"})
        assert not case.block("client").get("reporting_currency")
        assert case.block("client")["time_zone"] == "UTC"

    def test_an_answer_the_operator_gave_is_never_overwritten(self, service):
        case = service.start_new_client(by="Operator")
        case = service.save_step(case_id=case.case_id, step="client",
                                 by="Operator",
                                 payload={"client_name": "Nordic Lending",
                                          "jurisdiction": "SE",
                                          "reporting_currency": "EUR"})
        assert case.block("client")["reporting_currency"] == "EUR"

    def test_the_client_identifier_is_proposed_not_demanded(self, service):
        case = service.start_new_client(by="Operator")
        case = service.save_step(case_id=case.case_id, step="client",
                                 by="Operator",
                                 payload={"client_name": "Nordic Lending AB",
                                          "jurisdiction": "SE"})
        # Whole words only — an identifier is permanent.
        assert case.block("client")["client_id"] == "NORDIC"
        assert "already in use" in case.provenance["client.client_id"]

    def test_a_proposed_identifier_avoids_one_already_taken(self, service):
        first = complete(service, client_id="NORDIC")
        activate(service, first)
        case = service.start_new_client(by="Operator")
        case = service.save_step(case_id=case.case_id, step="client",
                                 by="Operator",
                                 payload={"client_name": "Nordic Lending AB",
                                          "jurisdiction": "SE"})
        assert case.block("client")["client_id"] == "NORDIC2"

    def test_portfolio_identifiers_follow_the_platform_convention(self, service):
        case = service.start_new_client(by="Operator")
        service.save_step(case_id=case.case_id, step="client", by="Operator",
                          payload={"client_name": "Nordic", "jurisdiction": "SE"})
        case = service.save_step(
            case_id=case.case_id, step="portfolios", by="Operator", payload={
                "portfolios": [{"display_name": "Direct", "portfolio_type": "direct"},
                               {"display_name": "Bought", "portfolio_type": "acquired"},
                               {"display_name": "More", "portfolio_type": "direct"}]})
        assert [p["portfolio_id"] for p in case.items("portfolios")] == [
            "direct_001", "acquired_001", "direct_002"]

    def test_a_single_entity_owns_every_portfolio_without_asking(self, service):
        case = service.start_new_client(by="Operator")
        service.save_step(case_id=case.case_id, step="client", by="Operator",
                          payload={"client_name": "Nordic", "jurisdiction": "SE"})
        case = service.save_step(case_id=case.case_id, step="entities",
                                 by="Operator", payload={"entities": [
                                     {"legal_name": "Nordic AB",
                                      "roles": ["originator"],
                                      "lei": VALID_LEI,
                                      "country_of_establishment": "SE"}]})
        entity_id = case.items("entities")[0]["entity_id"]
        case = service.save_step(case_id=case.case_id, step="portfolios",
                                 by="Operator", payload={"portfolios": [
                                     {"display_name": "Direct",
                                      "portfolio_type": "direct"}]})
        assert case.items("portfolios")[0]["owning_entity"] == entity_id

    def test_a_sample_pack_answers_format_files_and_asset_class(self, service):
        case = service.start_new_client(by="Operator")
        service.save_step(case_id=case.case_id, step="client", by="Operator",
                          payload={"client_name": "Nordic", "jurisdiction": "SE"})
        service.save_step(case_id=case.case_id, step="portfolios",
                          by="Operator", payload={"portfolios": [
                              {"display_name": "Direct",
                               "portfolio_type": "direct"}]})
        case = service.register_sample(
            case_id=case.case_id, by="Operator",
            files=[{"name": "LoanExtract.csv",
                    "headers": ["loan_id", "no_negative_equity", "roll_up",
                                "drawdown_facility"]}])
        assert case.items("portfolios")[0]["asset_class"] == "equity_release"
        source = case.items("sources")[0]
        assert source["file_format"] == "csv"
        assert source["expected_files"] == ["LoanExtract.csv"]

    def test_an_ambiguous_tape_is_left_as_a_question(self, service):
        """Two asset classes scoring equally is exactly when a human should
        look."""
        from operations_control.onboarding import inference
        assert inference.infer_from_sample(
            {"files": [{"name": "x.csv", "headers": ["loan_id", "balance"]}]}
        ).get("asset_class") is None

    def test_the_annex_2_originator_block_is_never_asked_for(self, service):
        """It is the originator entity, restated in regulator vocabulary."""
        case = complete(service)
        annex2 = case.block("regime")["esma_annex2"]
        assert annex2["originator_name"] == "Nordic Lending AB"
        assert annex2["originator_legal_entity_identifier"] == VALID_LEI
        assert annex2["originator_establishment_country"] == "SE"
        # And it tracks the entity: rename the company, the regime block follows.
        service.save_step(case_id=case.case_id, step="entities", by="Operator",
                          payload={"entities": [
                              {**case.items("entities")[0],
                               "legal_name": "Nordic Lending Group AB"},
                              case.items("entities")[1]]})
        case = service.load_case(case.case_id)
        assert case.block("regime")["esma_annex2"]["originator_name"] \
            == "Nordic Lending Group AB"

    def test_the_investor_report_contacts_come_from_the_contacts_step(self,
                                                                      service):
        case = complete(service, products=("esma_annex2", "investor_reporting"))
        annex12 = case.block("regime")["investor_reporting"]
        assert annex12["IVSS5"] == "Rae Reporter"
        assert annex12["IVSS7"] == "reporting@nordic.example"
        assert annex12["IVSS3"] == "Nordic Lending AB"

    def test_the_risk_retention_holder_code_comes_from_the_role(self, service):
        case = complete(service, products=("esma_annex2", "investor_reporting"))
        service.save_step(case_id=case.case_id, step="entities", by="Operator",
                          payload={"entities": [
                              {**case.items("entities")[0],
                               "roles": ["originator", "reporting_entity",
                                         "risk_retention_holder"]},
                              case.items("entities")[1]]})
        case = service.load_case(case.case_id)
        assert case.block("regime")["investor_reporting"]["IVSS9"] == "ORIG"

    def test_the_exposure_type_comes_from_the_asset_class(self, service):
        case = complete(service, products=("esma_annex2", "investor_reporting"))
        assert case.block("regime")["investor_reporting"]["IVSS10"] == "RMRT"

    def test_a_uk_book_reports_geography_as_gbzzz_without_being_asked(self,
                                                                      service):
        case = service.start_new_client(by="Operator")
        service.save_step(case_id=case.case_id, step="client", by="Operator",
                          payload={"client_name": "British", "jurisdiction": "GB"})
        service.save_step(case_id=case.case_id, step="reporting", by="Operator",
                          payload={"products": ["esma_annex2"]})
        case = service.load_case(case.case_id)
        assert case.block("regime")["esma_annex2"]["uk_geography_override"] is True

    def test_a_declared_default_fills_a_field_nobody_answered(self, service):
        """Including in the regime blocks, which is where most of them live."""
        case = complete(service)
        assert case.block("client")["environment"] == "production"
        assert case.items("sources")[0]["cadence"] == "monthly"
        assert case.block("regime")["esma_annex2"]["nuts_classification_year"] \
            == "2021"

    def test_a_default_never_displaces_something_better_known(self, service):
        """Defaults are applied last. A jurisdiction knows the time zone; the
        catalogue does not."""
        case = service.start_new_client(by="Operator")
        case = service.save_step(case_id=case.case_id, step="client",
                                 by="Operator",
                                 payload={"client_name": "Nordic Lending",
                                          "jurisdiction": "SE",
                                          "environment": "uat"})
        assert case.block("client")["time_zone"] == "Europe/Stockholm"
        assert case.block("client")["environment"] == "uat"

    def test_a_case_nobody_has_touched_holds_no_defaults(self, service):
        """A default belongs to a question that would otherwise be asked, so it
        arrives when the operator reaches that part of the form — not before."""
        case = service.start_new_client(by="Operator")
        assert case.answers.get("client", {}) == {}
        assert case.answers.get("presentation", {}) == {}

    def test_conventions_come_from_the_asset_class(self, service):
        case = complete(service)
        presentation = case.block("presentation")
        assert presentation["day_count_convention"] == "ACT365"
        assert presentation["payment_frequency"] == "AT_REDEMPTION"
        assert presentation["report_title"] == "Nordic Lending Portfolio MI"

    def test_a_minimal_case_reaches_a_complete_configuration(self, service,
                                                             store):
        """The whole point: name the client, its country, its entity, its
        contacts, one portfolio and the products — everything else follows."""
        case = service.start_new_client(by="Operator")
        cid = case.case_id
        service.save_step(case_id=cid, step="client", by="Operator",
                          payload={"client_name": "Nordic Lending",
                                   "jurisdiction": "SE"})
        service.save_step(case_id=cid, step="entities", by="Operator", payload={
            "entities": [{"legal_name": "Nordic Lending AB",
                          "roles": ["originator", "reporting_entity"],
                          "lei": VALID_LEI,
                          "country_of_establishment": "SE"}]})
        service.save_step(case_id=cid, step="contacts", by="Operator", payload={
            "reporting_contact_name": "Rae", "reporting_contact_email": "r@n.example",
            "operational_contact_name": "Ola", "operational_contact_email": "o@n.example"})
        service.register_sample(case_id=cid, by="Operator", files=[
            {"name": "LoanExtract.csv",
             "headers": ["loan_id", "no_negative_equity", "roll_up"]}])
        service.save_step(case_id=cid, step="portfolios", by="Operator", payload={
            "portfolios": [{"display_name": "Nordic Direct",
                            "portfolio_type": "direct"}]})
        service.save_step(case_id=cid, step="reporting", by="Operator",
                          payload={"products": ["esma_annex2"]})

        case = service.load_case(cid)
        assert service.readiness(case)["ready"] is True, [
            p["message"] for p in service.readiness(case)["blocking"]]

        activate(service, case)
        doc = yaml.safe_load(OnboardingStore(store).read_artefact(
            "NORDIC", artefacts.client_config_rel("NORDIC")))
        # A complete governed configuration, from six answers.
        assert doc["client"]["client_id"] == "NORDIC"
        assert doc["portfolio"]["base_currency"] == "SEK"
        assert doc["defaults"]["originator_legal_entity_identifier"] == VALID_LEI
        assert doc["pipeline"]["esma_enabled"] is True
        assert doc["loan_engine"]["day_count_convention"] == "ACT365"
