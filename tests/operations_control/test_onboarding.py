"""Client Onboarding — the governed standing-configuration capability.

Covers the model and its governed vocabularies, adoption of an existing client
from its legacy files, artefact generation (including what generation must NOT
touch), immutable versioning with before/after history, the audit trail, the
resolver seam that makes a generated configuration the one deliveries run with,
and tenancy on every route.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from apps.blob_trigger_app.source_registry import (
    STATUS_ACTIVE,
    STATUS_PENDING_REVIEW,
    SourceRecord,
)

from operations_control.onboarding import generation, migration
from operations_control.onboarding.model import (
    ClientOnboardingProfile,
    PortfolioStanding,
    SourceRegistrationStanding,
    derive_reporting,
    derive_sources,
    governed_vocabularies,
    standing_field_spec,
    validate,
)
from operations_control.onboarding.service import OnboardingService, diff_profiles
from operations_control.onboarding.store import OnboardingStore

from .conftest import OP_A, OP_ADMIN, OP_ALL, OP_B


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #

@pytest.fixture()
def service(store, source_registry) -> OnboardingService:
    return OnboardingService(store, registry_factory=lambda: source_registry)


@pytest.fixture()
def legacy_files(tmp_path) -> dict:
    """A client configured the old way: master config + Annex 12 overlay +
    portfolio registry + tenancy."""
    client_config = tmp_path / "config_client_LEGACY.yaml"
    client_config.write_text(yaml.safe_dump({
        "client": {"client_id": "LEGACY",
                   "display_name": "Legacy Funding Limited",
                   "environment": "production"},
        "portfolio": {"asset_class": "equity_release", "country": "GB",
                      "base_currency": "GBP"},
        "default_regime": "ESMA_Annex2",
        "supported_regimes": ["ESMA_Annex2"],
        "regime": "ESMA_Annex2",
        "pipeline": {"loan_engine_enabled": False, "mi_enabled": True,
                     "esma_enabled": True},
        "defaults": {
            "originator_legal_entity_identifier": "213800ABCDE123456701",
            "originator_name": "Legacy Funding Limited",
            "originator_establishment_country": "GB"},
        "transformations": {"copy_if_missing": {"a": "b"}},
        "enrichment": {"enable": {"uk_nuts3": True}},
        "loan_engine": {"day_count_convention": "ACT365",
                        "interest_payment_frequency": "QUARTERLY"},
        "nuts_classification_year": 2021,
        "mi": {"branding": {"app_title": "Legacy MI",
                            "theme": {"primary_color": "#0B1F3B"}}},
    }, sort_keys=False), encoding="utf-8")

    annex12 = tmp_path / "config_client_LEGACY_annex12.yaml"
    annex12.write_text(yaml.safe_dump({
        "annex12": {
            "meta": {"template": "ESMA Annex 12 (Investor Report)"},
            "deal": {"IVSS5": "Ada Reporter", "IVSS6": "+44-2000000000",
                     "IVSS7": "reporting@legacy.example", "IVSS8": "VSLC",
                     "IVSS9": "SPON", "IVSS10": "RMRT",
                     "IVSR1": "TRIGGER", "IVSF1": "CF_001"},
            "period": {"IVSS2": "2025-10-31"},
        }}, sort_keys=False), encoding="utf-8")

    portfolios = tmp_path / "portfolio_registry.yaml"
    portfolios.write_text(yaml.safe_dump({
        "clients": {"LEGACY": {"portfolios": [
            {"source_portfolio_id": "direct_001",
             "source_portfolio_type": "direct",
             "source_portfolio_label": "Legacy Direct Originations",
             "originates": True}]}}}, sort_keys=False), encoding="utf-8")

    tenancy = tmp_path / "tenancy.yaml"
    tenancy.write_text(yaml.safe_dump({
        "tenants": {"LEGACY": {"display_name": "Legacy Funding",
                               "default_portfolio": "LEGACY"}}},
        sort_keys=False), encoding="utf-8")

    return {"client_config_path": client_config,
            "annex12_config_path": annex12,
            "portfolio_registry_path": portfolios,
            "tenancy_path": tenancy}


@pytest.fixture()
def legacy_registry(source_registry):
    source_registry.upsert(SourceRecord(
        client_id="LEGACY", source_portfolio_id="direct_001",
        source_portfolio_type="direct", dataset="funded", frequency="monthly",
        source_system="Legacy core lending platform",
        approved_mapping_id="legacy_direct_funded_monthly_v1",
        mapping_config_path="config/client/mappings/legacy.yaml",
        expected_schema_fingerprint="sha256:proven",
        file_role_schemas={"loan_extract": ["a", "b"]},
        last_successful_reporting_period="2025-11-30",
        regime_required=True, status=STATUS_ACTIVE))
    source_registry.upsert(SourceRecord(
        client_id="LEGACY", source_portfolio_id="direct_001",
        source_portfolio_type="direct", dataset="pipeline", frequency="weekly",
        source_system="Legacy origination pipeline",
        regime_required=False, status=STATUS_ACTIVE))
    return source_registry


def complete_profile(client_id: str = "NEWCO") -> ClientOnboardingProfile:
    profile = ClientOnboardingProfile(client_id=client_id)
    profile.client.client_id = client_id
    profile.client.client_name = "Newco Lending"
    profile.client.legal_entity_name = "Newco Lending Limited"
    profile.client.lei = "213800ABCDE123456702"
    profile.client.jurisdiction = "GB"
    profile.client.reporting_currency = "GBP"
    profile.client.time_zone = "Europe/London"
    profile.client.primary_reporting_contact = "Rae Reporter"
    profile.client.reporting_email = "reporting@newco.example"
    profile.client.operational_contact = "Ola Ops"
    profile.client.operational_email = "ops@newco.example"
    profile.portfolios = [PortfolioStanding(
        portfolio_id="direct_001", display_name="Newco Direct",
        portfolio_type="direct", asset_class="equity_release",
        structure="warehouse", funded_cadence="monthly",
        pipeline_cadence="weekly", source_system="Newco core")]
    profile.reporting.products = ["esma_annex2"]
    profile.regime["esma_annex2"] = {
        "originator_name": "Newco Lending Limited",
        "originator_legal_entity_identifier": "213800ABCDE123456702",
        "originator_establishment_country": "GB"}
    profile.sources = derive_sources(profile)
    return profile


# --------------------------------------------------------------------------- #
# Governed vocabularies
# --------------------------------------------------------------------------- #

class TestGovernedVocabularies:
    def test_option_lists_come_from_existing_configuration(self):
        """Onboarding restates no vocabulary of its own: every option list is
        owned somewhere else and named as such."""
        from operations_control.configuration.packages import ASSET_MODEL
        from operations_control.contracts import BATCH_DATASETS
        from apps.blob_trigger_app.path_parser import VALID_FREQUENCIES

        vocab = governed_vocabularies()
        assert [a["value"] for a in vocab["asset_classes"]] == sorted(ASSET_MODEL)
        assert vocab["datasets"] == list(BATCH_DATASETS)
        assert set(vocab["cadences"]) == set(VALID_FREQUENCIES)
        assert vocab["sources"]["asset_classes"].endswith("ASSET_MODEL")

    def test_a_regime_the_asset_does_not_support_is_not_offered(self):
        """The asset support model — not onboarding — decides eligibility."""
        profile = complete_profile()
        profile.portfolios[0].asset_class = "equity_release"
        derived = derive_reporting(profile)
        assert derived["esma_annex2"]["eligible"] is True
        assert "direct_001" in derived["esma_annex2"]["reason"]

    def test_mi_is_derived_never_chosen(self):
        derived = derive_reporting(complete_profile())
        assert derived["mi"]["derived"] is True
        assert derived["mi"]["applies"] is True

    def test_standing_field_spec_resolves_regime_enumerations(self):
        """Annex 12 enum values are read from the regime template, so the
        wizard cannot offer a value the projector would refuse."""
        spec = standing_field_spec()
        fields = {f["key"]: f for f in
                  spec["products"]["investor_reporting"]["fields"]}
        options = {o["value"] for o in fields["IVSS8"]["options"]}
        assert {"VSLC", "SLLS", "OTHR"} <= options

    def test_products_declare_what_they_cannot_represent(self):
        spec = standing_field_spec()
        unrepresented = {u["key"] for u in
                         spec["products"]["esma_annex2"]["unrepresented"]}
        assert {"sponsor_name", "sspe_name", "sts_status",
                "national_competent_authority"} <= unrepresented


# --------------------------------------------------------------------------- #
# Validation
# --------------------------------------------------------------------------- #

class TestValidation:
    def test_a_complete_profile_has_no_problems(self):
        assert validate(complete_profile()) == []

    def test_an_invalid_lei_is_refused_in_operator_wording(self):
        profile = complete_profile()
        profile.client.lei = "NOT-A-LEI"
        problems = validate(profile)
        assert any("20-character LEI" in p.message for p in problems)
        assert all("config/" not in p.message for p in problems)

    def test_regime_scope_cannot_be_put_on_a_pipeline_book(self):
        """The same rule the intake enforces: a pipeline view never carries a
        regime delivery."""
        profile = complete_profile()
        profile.sources.append(SourceRegistrationStanding(
            portfolio_id="direct_001", dataset="pipeline", frequency="weekly",
            regime_required=True))
        problems = validate(profile)
        assert any("funded book" in p.message for p in problems)

    def test_a_source_for_an_unknown_portfolio_is_refused(self):
        profile = complete_profile()
        profile.sources.append(SourceRegistrationStanding(
            portfolio_id="ghost_001", dataset="funded", frequency="monthly"))
        assert any("not one of this client's portfolios" in p.message
                   for p in validate(profile))

    def test_a_required_regime_field_blocks_the_product(self):
        profile = complete_profile()
        profile.regime["esma_annex2"].pop("originator_legal_entity_identifier")
        problems = validate(profile)
        assert any("Originator LEI" in p.message for p in problems)

    def test_duplicate_portfolios_are_refused(self):
        profile = complete_profile()
        profile.portfolios.append(profile.portfolios[0])
        assert any("listed twice" in p.message for p in validate(profile))


# --------------------------------------------------------------------------- #
# Derivation
# --------------------------------------------------------------------------- #

class TestDerivation:
    def test_sources_follow_from_portfolios_and_reporting(self):
        profile = complete_profile()
        sources = derive_sources(profile)
        assert [(s.portfolio_id, s.dataset, s.frequency) for s in sources] == [
            ("direct_001", "funded", "monthly"),
            ("direct_001", "pipeline", "weekly")]
        assert sources[0].regime_required is True     # funded + Annex 2
        assert sources[1].regime_required is False    # pipeline is MI-only

    def test_no_pipeline_cadence_means_no_pipeline_registration(self):
        profile = complete_profile()
        profile.portfolios[0].pipeline_cadence = ""
        assert [s.dataset for s in derive_sources(profile)] == ["funded"]

    def test_regime_scope_is_not_set_when_the_asset_cannot_support_it(self):
        profile = complete_profile()
        profile.portfolios[0].asset_class = "not_a_real_asset"
        assert all(not s.regime_required for s in derive_sources(profile))


# --------------------------------------------------------------------------- #
# Adoption of an existing client
# --------------------------------------------------------------------------- #

class TestAdoption:
    def test_an_existing_client_arrives_with_its_current_values(
            self, legacy_registry, legacy_files):
        adoption = migration.adopt("LEGACY", registry=legacy_registry,
                                   **legacy_files)
        p = adoption.profile
        assert p.client.client_name == "Legacy Funding Limited"
        assert p.client.lei == "213800ABCDE123456701"
        assert p.client.jurisdiction == "GB"
        assert p.client.reporting_currency == "GBP"
        assert p.static_reporting.app_title == "Legacy MI"
        assert p.static_reporting.day_count_convention == "ACT365"

    def test_portfolios_and_cadences_come_from_the_source_registry(
            self, legacy_registry, legacy_files):
        p = migration.adopt("LEGACY", registry=legacy_registry,
                            **legacy_files).profile
        assert [x.portfolio_id for x in p.portfolios] == ["direct_001"]
        portfolio = p.portfolios[0]
        assert portfolio.display_name == "Legacy Direct Originations"
        assert portfolio.funded_cadence == "monthly"
        assert portfolio.pipeline_cadence == "weekly"
        assert portfolio.portfolio_type == "direct"

    def test_reporting_products_are_read_from_what_is_configured(
            self, legacy_registry, legacy_files):
        p = migration.adopt("LEGACY", registry=legacy_registry,
                            **legacy_files).profile
        assert set(p.reporting.products) == {"esma_annex2",
                                             "investor_reporting"}

    def test_regime_answers_are_read_back_through_the_same_mapping(
            self, legacy_registry, legacy_files):
        p = migration.adopt("LEGACY", registry=legacy_registry,
                            **legacy_files).profile
        annex2 = p.regime["esma_annex2"]
        assert annex2["originator_name"] == "Legacy Funding Limited"
        assert annex2["originator_establishment_country"] == "GB"
        investor = p.regime["investor_reporting"]
        assert investor["IVSS5"] == "Ada Reporter"
        assert investor["IVSS8"] == "VSLC"

    def test_what_the_legacy_files_cannot_answer_is_reported_not_guessed(
            self, legacy_registry, legacy_files):
        adoption = migration.adopt("LEGACY", registry=legacy_registry,
                                   **legacy_files)
        gaps = {g["field"] for g in adoption.gaps}
        assert "reporting_contact" in gaps
        assert "operational_contact" in gaps
        assert adoption.profile.client.operational_email == ""

    def test_every_adopted_value_carries_its_origin(
            self, legacy_registry, legacy_files):
        adoption = migration.adopt("LEGACY", registry=legacy_registry,
                                   **legacy_files)
        assert "RREL83" in adoption.provenance["client.lei"]
        assert adoption.provenance["sources"] == "the durable source registry"

    def test_adoption_never_writes_to_the_legacy_files(
            self, legacy_registry, legacy_files):
        before = {k: Path(v).read_text(encoding="utf-8")
                  for k, v in legacy_files.items()}
        migration.adopt("LEGACY", registry=legacy_registry, **legacy_files)
        after = {k: Path(v).read_text(encoding="utf-8")
                 for k, v in legacy_files.items()}
        assert before == after


# --------------------------------------------------------------------------- #
# Generation
# --------------------------------------------------------------------------- #

class TestGeneration:
    def test_the_client_configuration_keeps_its_existing_shape(self, store):
        artefacts = generation.plan(complete_profile(),
                                    store=OnboardingStore(store))
        client_cfg = next(a for a in artefacts
                          if a.kind == generation.ARTEFACT_CLIENT_CONFIG)
        doc = yaml.safe_load(client_cfg.text)
        assert doc["client"]["client_id"] == "NEWCO"
        assert doc["portfolio"]["country"] == "GB"
        assert doc["portfolio"]["base_currency"] == "GBP"
        assert doc["supported_regimes"] == ["ESMA_Annex2"]
        # The legacy compatibility key is still written.
        assert doc["regime"] == "ESMA_Annex2"
        assert doc["pipeline"]["esma_enabled"] is True
        assert doc["defaults"]["originator_legal_entity_identifier"] \
            == "213800ABCDE123456702"

    def test_generated_files_say_they_are_generated(self, store):
        artefacts = generation.plan(complete_profile(),
                                    store=OnboardingStore(store))
        text = next(a.text for a in artefacts
                    if a.kind == generation.ARTEFACT_CLIENT_CONFIG)
        assert text.startswith("# GENERATED by the Trakt Operations Control")

    def test_blocks_onboarding_does_not_own_survive_adoption(
            self, store, legacy_registry, legacy_files):
        """Enrichment, transformations and deal-structure triggers are carried
        through untouched — onboarding owns the standing business facts, not
        the whole file."""
        adoption = migration.adopt("LEGACY", registry=legacy_registry,
                                   **legacy_files)
        profile = adoption.profile
        profile.client.primary_reporting_contact = "Ada Reporter"
        profile.client.reporting_email = "reporting@legacy.example"
        profile.client.operational_contact = "Ola Ops"
        profile.client.operational_email = "ops@legacy.example"
        artefacts = generation.plan(profile, store=OnboardingStore(store),
                                    base_documents=adoption.base_documents)
        doc = yaml.safe_load(next(
            a.text for a in artefacts
            if a.kind == generation.ARTEFACT_CLIENT_CONFIG))
        assert doc["enrichment"]["enable"]["uk_nuts3"] is True
        assert doc["transformations"]["copy_if_missing"] == {"a": "b"}
        annex12 = yaml.safe_load(next(
            a.text for a in artefacts
            if a.kind == generation.ARTEFACT_ANNEX12_CONFIG))
        assert annex12["annex12"]["deal"]["IVSR1"] == "TRIGGER"
        assert annex12["annex12"]["deal"]["IVSF1"] == "CF_001"

    def test_a_new_client_never_inherits_another_clients_configuration(
            self, store):
        """No base documents means a clean document — inheriting ERE's
        enrichment rules is exactly the coupling this capability removes."""
        artefacts = generation.plan(complete_profile(),
                                    store=OnboardingStore(store))
        doc = yaml.safe_load(next(
            a.text for a in artefacts
            if a.kind == generation.ARTEFACT_CLIENT_CONFIG))
        assert "enrichment" not in doc
        assert "transformations" not in doc

    def test_the_uk_geography_override_is_written_in_its_governed_shape(
            self, store):
        profile = complete_profile()
        profile.regime["esma_annex2"]["uk_geography_override"] = True
        artefacts = generation.plan(profile, store=OnboardingStore(store))
        doc = yaml.safe_load(next(
            a.text for a in artefacts
            if a.kind == generation.ARTEFACT_CLIENT_CONFIG))
        uk = doc["regime_overrides"]["ESMA_Annex2"]["uk_geography"]
        assert uk["enabled"] is True
        assert uk["override_value"] == "GBZZZ"
        assert uk["target_fields"] == ["geographic_region_obligor",
                                       "geographic_region_collateral"]

    def test_portfolio_metadata_uses_the_per_client_shape(self, store):
        artefacts = generation.plan(complete_profile(),
                                    store=OnboardingStore(store))
        doc = yaml.safe_load(next(
            a.text for a in artefacts
            if a.kind == generation.ARTEFACT_PORTFOLIO_REGISTRY))
        entry = doc["clients"]["NEWCO"]["portfolios"][0]
        assert entry["source_portfolio_id"] == "direct_001"
        assert entry["originates"] is True
        assert entry["pipeline_data_available"] is True

    def test_an_acquired_book_without_a_curve_gets_no_forecast_treatment(
            self, store):
        """Trakt consumes a runoff curve; it never generates one."""
        profile = complete_profile()
        profile.portfolios[0].portfolio_type = "acquired"
        artefacts = generation.plan(profile, store=OnboardingStore(store))
        doc = yaml.safe_load(next(
            a.text for a in artefacts
            if a.kind == generation.ARTEFACT_PORTFOLIO_REGISTRY))
        entry = doc["clients"]["NEWCO"]["portfolios"][0]
        assert entry["originates"] is False
        assert "forecast_treatment" not in entry

    def test_source_registrations_are_generated_not_typed(self, store):
        artefacts = generation.plan(complete_profile(),
                                    store=OnboardingStore(store))
        registry_plan = next(a for a in artefacts
                             if a.kind == generation.ARTEFACT_SOURCE_REGISTRY)
        keys = {(r["dataset"], r["frequency"], r["regime_required"])
                for r in registry_plan.records}
        assert keys == {("funded", "monthly", True),
                        ("pipeline", "weekly", False)}
        location = registry_plan.records[0]["expected_location"]
        assert location.startswith("raw-v2/NEWCO/direct/funded/monthly/")

    def test_an_approved_mapping_is_never_reset_by_onboarding(
            self, store, legacy_registry, legacy_files):
        """A source proven by a real delivery keeps its fingerprint, mapping and
        active status — otherwise a recognised pack would be sent back through
        source onboarding."""
        adoption = migration.adopt("LEGACY", registry=legacy_registry,
                                   **legacy_files)
        profile = adoption.profile
        profile.sources[0].source_system = "Legacy core lending platform v2"
        records, _ = generation.render_source_records(profile,
                                                      registry=legacy_registry)
        funded = next(r for r in records if r.dataset == "funded")
        assert funded.approved_mapping_id == "legacy_direct_funded_monthly_v1"
        assert funded.expected_schema_fingerprint == "sha256:proven"
        assert funded.file_role_schemas == {"loan_extract": ["a", "b"]}
        assert funded.last_successful_reporting_period == "2025-11-30"
        assert funded.status == STATUS_ACTIVE
        assert funded.source_system == "Legacy core lending platform v2"

    def test_a_new_source_waits_for_its_first_pack(self, store):
        records, _ = generation.render_source_records(complete_profile())
        assert all(r.status == STATUS_PENDING_REVIEW for r in records)
        assert all(r.expected_schema_fingerprint is None for r in records)

    def test_planning_writes_nothing(self, store, source_registry):
        profile = complete_profile()
        generation.plan(profile, store=OnboardingStore(store),
                        registry=source_registry)
        assert OnboardingStore(store).read_artefact(
            "NEWCO", generation.client_config_rel("NEWCO")) is None
        assert source_registry.records() == []


# --------------------------------------------------------------------------- #
# The workflow
# --------------------------------------------------------------------------- #

class TestWizardWorkflow:
    def test_a_new_client_can_be_created_end_to_end(self, service,
                                                    source_registry, store):
        draft = service.start_draft(by="Operator")
        did = draft["draft_id"]
        assert draft["ready"] is False

        profile = complete_profile()
        service.save_step(client_id="", draft_id=did, step="client",
                          payload=profile.client.to_dict(), by="Operator")
        service.save_step(client_id="NEWCO", draft_id=did, step="portfolio",
                          payload={"portfolios": [p.to_dict() for p in
                                                  profile.portfolios]},
                          by="Operator")
        service.save_step(client_id="NEWCO", draft_id=did, step="reporting",
                          payload={"products": ["esma_annex2"]}, by="Operator")
        state = service.save_step(
            client_id="NEWCO", draft_id=did, step="regime",
            payload={"regime": {"esma_annex2": profile.regime["esma_annex2"]}},
            by="Operator")
        assert state["ready"] is True

        review = service.review(client_id="NEWCO", draft_id=did)
        assert review["ready"] is True
        assert {a["kind"] for a in review["artefacts"]} == {
            "client_config", "portfolio_registry", "source_registry"}
        assert all(a["action"] == "create" for a in review["artefacts"])

        result = service.approve(client_id="NEWCO", draft_id=did,
                                 by="Administrator",
                                 reason="New client onboarded")
        assert result["version"] == 1
        text = OnboardingStore(store).read_artefact(
            "NEWCO", generation.client_config_rel("NEWCO"))
        assert "213800ABCDE123456702" in text
        assert {r.key for r in source_registry.records()} == {
            "NEWCO/direct_001/funded/monthly",
            "NEWCO/direct_001/pipeline/weekly"}

    def test_source_registrations_are_rebuilt_when_portfolios_change(
            self, service):
        draft = service.start_draft(by="Operator")
        did = draft["draft_id"]
        profile = complete_profile()
        service.save_step(client_id="", draft_id=did, step="client",
                          payload=profile.client.to_dict(), by="Operator")
        state = service.save_step(
            client_id="NEWCO", draft_id=did, step="portfolio",
            payload={"portfolios": [p.to_dict() for p in profile.portfolios]},
            by="Operator")
        assert len(state["profile"]["sources"]) == 2
        state = service.save_step(
            client_id="NEWCO", draft_id=did, step="portfolio",
            payload={"portfolios": [{**profile.portfolios[0].to_dict(),
                                     "pipeline_cadence": ""}]},
            by="Operator")
        assert [s["dataset"] for s in state["profile"]["sources"]] == ["funded"]

    def test_an_incomplete_profile_cannot_be_approved(self, service):
        draft = service.start_draft(by="Operator")
        service.save_step(client_id="", draft_id=draft["draft_id"],
                          step="client", payload={"client_id": "PART"},
                          by="Operator")
        from operations_control.engine import OpsError
        with pytest.raises(OpsError) as exc:
            service.approve(client_id="PART", draft_id=draft["draft_id"],
                            by="Administrator", reason="too early")
        assert exc.value.code == "OPS_ONBOARDING_INCOMPLETE"

    def test_approval_requires_a_reason(self, service):
        draft = service.start_draft(by="Operator")
        from operations_control.engine import OpsError
        with pytest.raises(OpsError) as exc:
            service.approve(client_id="", draft_id=draft["draft_id"],
                            by="Administrator", reason="")
        assert exc.value.code == "OPS_REASON_REQUIRED"

    def test_review_reports_what_cannot_be_represented(self, service):
        draft = service.start_draft(by="Operator")
        did = draft["draft_id"]
        profile = complete_profile()
        service.save_step(client_id="", draft_id=did, step="client",
                          payload=profile.client.to_dict(), by="Operator")
        service.save_step(client_id="NEWCO", draft_id=did, step="portfolio",
                          payload={"portfolios": [p.to_dict() for p in
                                                  profile.portfolios]},
                          by="Operator")
        service.save_step(client_id="NEWCO", draft_id=did, step="reporting",
                          payload={"products": ["esma_annex2"]}, by="Operator")
        review = service.review(client_id="NEWCO", draft_id=did)
        keys = {u["key"] for u in review["unrepresented"]}
        assert {"sponsor_name", "sspe_name", "servicer_name"} <= keys


# --------------------------------------------------------------------------- #
# Versioning, history and audit
# --------------------------------------------------------------------------- #

def _onboard(service, profile, *, by="Administrator", reason="Initial"):
    draft = service.start_draft(client_id=profile.client_id, by=by)
    did = draft["draft_id"]
    service.save_step(client_id=profile.client_id, draft_id=did, step="client",
                      payload=profile.client.to_dict(), by=by)
    service.save_step(client_id=profile.client_id, draft_id=did,
                      step="portfolio",
                      payload={"portfolios": [p.to_dict()
                                              for p in profile.portfolios]},
                      by=by)
    service.save_step(client_id=profile.client_id, draft_id=did,
                      step="reporting",
                      payload={"products": list(profile.reporting.products)},
                      by=by)
    service.save_step(client_id=profile.client_id, draft_id=did, step="regime",
                      payload={"regime": profile.regime}, by=by)
    return service.approve(client_id=profile.client_id, draft_id=did, by=by,
                           reason=reason)


class TestVersioningAndAudit:
    def test_a_change_creates_a_new_version_and_keeps_the_old_one(
            self, service, store):
        _onboard(service, complete_profile())
        draft = service.start_draft(client_id="NEWCO", by="Administrator")
        service.save_step(client_id="NEWCO", draft_id=draft["draft_id"],
                          step="client",
                          payload={"reporting_email": "new@newco.example"},
                          by="Administrator")
        service.approve(client_id="NEWCO", draft_id=draft["draft_id"],
                        by="Administrator", reason="Contact changed")

        profiles = OnboardingStore(store)
        assert profiles.current("NEWCO").version == 2
        v1 = profiles.version("NEWCO", 1)
        assert v1.status == "superseded"
        assert v1.profile["client"]["reporting_email"] \
            == "reporting@newco.example"
        assert profiles.version("NEWCO", 2).profile["client"]["reporting_email"] \
            == "new@newco.example"

    def test_history_records_who_when_what_before_after_and_why(self, service):
        _onboard(service, complete_profile())
        draft = service.start_draft(client_id="NEWCO", by="Administrator")
        service.save_step(client_id="NEWCO", draft_id=draft["draft_id"],
                          step="client", payload={"client_name": "Newco Group"},
                          by="Administrator")
        service.approve(client_id="NEWCO", draft_id=draft["draft_id"],
                        by="Administrator", reason="Rebrand")

        history = service.client_view("NEWCO")["history"]
        assert [h["version"] for h in history] == [2, 1]
        latest = history[0]
        assert latest["approved_by"] == "Administrator"
        assert latest["approved_at"]
        assert latest["reason"] == "Rebrand"
        change = next(c for c in latest["changes"]
                      if c["path"] == "client.client_name")
        assert change["before"] == "Newco Lending"
        assert change["after"] == "Newco Group"
        assert change["label"] == "Client name"

    def test_the_audit_chain_stays_intact_across_approvals(self, service,
                                                           store):
        _onboard(service, complete_profile())
        rows = store.list_audit("NEWCO")
        events = [r["event"] for r in rows]
        assert "onboarding_configuration_approved" in events
        approved = next(r for r in rows
                        if r["event"] == "onboarding_configuration_approved")
        assert approved["actor"] == "Administrator"
        assert approved["detail"]["reason"] == "Initial"
        assert approved["detail"]["artefacts"]
        assert store.verify_audit_chain("NEWCO") is True

    def test_an_unchanged_artefact_is_reported_as_unchanged(self, service):
        _onboard(service, complete_profile())
        draft = service.start_draft(client_id="NEWCO", by="Administrator")
        review = service.review(client_id="NEWCO",
                                draft_id=draft["draft_id"])
        actions = {a["kind"]: a["action"] for a in review["artefacts"]}
        assert actions["client_config"] == "unchanged"
        assert actions["portfolio_registry"] == "unchanged"

    def test_versions_are_addressable_after_the_fact(self, service):
        _onboard(service, complete_profile())
        version = service.version_view("NEWCO", 1)
        assert version["version"] == 1
        assert version["profile"]["client"]["client_name"] == "Newco Lending"

    def test_diff_of_a_first_version_reports_additions(self):
        changes = diff_profiles(None, complete_profile())
        assert all(c["action"] == "added" for c in changes)
        assert any(c["path"] == "client.lei" for c in changes)


# --------------------------------------------------------------------------- #
# The home screen and the client view
# --------------------------------------------------------------------------- #

class TestClientViews:
    def test_a_client_that_predates_onboarding_is_listed_as_legacy(
            self, service, legacy_registry):
        rows = {c["client_id"]: c for c in service.list_clients()}
        assert rows["LEGACY"]["status"] == "legacy"
        assert "adopted" in rows["LEGACY"]["summary"]

    def test_an_onboarded_client_shows_its_current_version(self, service):
        _onboard(service, complete_profile())
        rows = {c["client_id"]: c for c in service.list_clients()}
        assert rows["NEWCO"]["status"] == "onboarded"
        assert rows["NEWCO"]["version"] == 1
        assert rows["NEWCO"]["portfolios"] == 1
        assert rows["NEWCO"]["products"] == ["esma_annex2"]

    def test_the_client_view_carries_every_tab(self, service):
        _onboard(service, complete_profile())
        view = service.client_view("NEWCO")
        assert view["profile"]["client"]["client_name"] == "Newco Lending"
        assert view["profile"]["portfolios"]
        assert view["derived_reporting"]["mi"]["applies"] is True
        assert view["profile"]["regime"]["esma_annex2"]
        assert view["history"]
        assert view["artefacts"]

    def test_the_client_view_shows_the_live_registry_not_a_copy(
            self, service, legacy_registry):
        _onboard(service, complete_profile())
        view = service.client_view("NEWCO")
        keys = {(r["source_portfolio_id"], r["dataset"])
                for r in view["live_source_registrations"]}
        assert keys == {("direct_001", "funded"), ("direct_001", "pipeline")}

    def test_adoption_is_offered_for_a_legacy_client(self, service,
                                                    legacy_registry):
        view = service.client_view("LEGACY")
        assert view["status"] == "legacy"
        assert view["profile"] is None
        assert "Adopt" in view["summary"]


# --------------------------------------------------------------------------- #
# The resolver seam: generated configuration is what deliveries run with
# --------------------------------------------------------------------------- #

class TestResolverSeam:
    def test_a_client_without_a_profile_still_uses_the_repository_file(
            self, store, tmp_path):
        from operations_control.configuration.resolver import EffectiveConfigResolver
        from operations_control.rules import RuleStore
        repo_file = tmp_path / "repo_client.yaml"
        repo_file.write_text("client: {client_id: X}\n", encoding="utf-8")
        resolver = EffectiveConfigResolver(store, RuleStore(store),
                                           client_config_path=repo_file)
        assert resolver.client_config_for("UNKNOWN") == repo_file

    def test_an_onboarded_client_resolves_with_its_generated_configuration(
            self, service, store, tmp_path):
        from operations_control.configuration.resolver import EffectiveConfigResolver
        from operations_control.rules import RuleStore
        _onboard(service, complete_profile())
        repo_file = tmp_path / "repo_client.yaml"
        repo_file.write_text("client: {client_id: X}\n", encoding="utf-8")
        resolver = EffectiveConfigResolver(store, RuleStore(store),
                                           client_config_path=repo_file)
        path = resolver.client_config_for("NEWCO")
        assert path != repo_file
        doc = yaml.safe_load(path.read_text(encoding="utf-8"))
        assert doc["client"]["client_id"] == "NEWCO"
        assert doc["defaults"]["originator_legal_entity_identifier"] \
            == "213800ABCDE123456702"

    def test_the_generated_configuration_satisfies_the_annex2_preflight(
            self, service, store):
        """What onboarding asks the operator for is exactly what the regulatory
        preflight demands — so an onboarded client is not blocked at delivery
        time for configuration it already supplied."""
        from operations_control.annex2 import preflight
        from operations_control.configuration.resolver import EffectiveConfigResolver
        from operations_control.rules import RuleStore
        _onboard(service, complete_profile())
        resolver = EffectiveConfigResolver(store, RuleStore(store))
        path = resolver.client_config_for("NEWCO")
        result = preflight.run_preflight(client_config_path=path,
                                         reporting_period="2026-06-30")
        assert result["blocked"] == []
        # The only item left needing review is the optional client no-data
        # policy, which the governed delivery rules default anyway — nothing
        # onboarding asked the operator for is missing.
        assert [c["key"] for c in result["needs_review"]] == ["nd_defaults"]


# --------------------------------------------------------------------------- #
# API + tenancy
# --------------------------------------------------------------------------- #

@pytest.fixture()
def api(store, source_registry, ops_env, monkeypatch):
    from fastapi.testclient import TestClient
    from operations_control.api import app as app_module
    from operations_control.engine import OpsEngine
    engine = OpsEngine(store, source_registry_factory=lambda: source_registry)
    app_module.set_engine(engine)
    yield TestClient(app_module.app), engine
    app_module.set_engine(None)


ADMIN = {"X-Operator-Token": OP_ADMIN}
ALICE = {"X-Operator-Token": OP_A}
BOB = {"X-Operator-Token": OP_B}


class TestApi:
    def test_vocabularies_are_served_with_their_owners(self, api):
        client, _ = api
        body = client.get("/ops/onboarding/vocabularies",
                          headers=ADMIN).json()
        assert body["ok"] is True
        assert body["vocabularies"]["datasets"] == ["funded", "pipeline"]
        assert "esma_annex2" in body["standing_fields"]["products"]

    def test_an_unauthenticated_caller_is_refused(self, api):
        client, _ = api
        assert client.get("/ops/onboarding/clients").status_code == 401

    def test_a_wizard_runs_end_to_end_over_http(self, api, source_registry):
        client, _ = api
        profile = complete_profile()
        did = client.post("/ops/onboarding/drafts", json={},
                          headers=ADMIN).json()["draft"]["draft_id"]
        client.put(f"/ops/onboarding/drafts/{did}",
                   json={"step": "client", "payload": profile.client.to_dict()},
                   headers=ADMIN)
        client.put(f"/ops/onboarding/drafts/{did}?client=NEWCO",
                   json={"step": "portfolio",
                         "payload": {"portfolios": [p.to_dict() for p in
                                                    profile.portfolios]}},
                   headers=ADMIN)
        client.put(f"/ops/onboarding/drafts/{did}?client=NEWCO",
                   json={"step": "reporting",
                         "payload": {"products": ["esma_annex2"]}},
                   headers=ADMIN)
        client.put(f"/ops/onboarding/drafts/{did}?client=NEWCO",
                   json={"step": "regime",
                         "payload": {"regime": {"esma_annex2":
                                                profile.regime["esma_annex2"]}}},
                   headers=ADMIN)
        review = client.get(f"/ops/onboarding/drafts/{did}/review?client=NEWCO",
                            headers=ADMIN).json()["review"]
        assert review["ready"] is True
        approved = client.post(
            f"/ops/onboarding/drafts/{did}/approve?client=NEWCO",
            json={"reason": "New client"}, headers=ADMIN)
        assert approved.status_code == 200
        assert approved.json()["version"] == 1
        detail = client.get("/ops/onboarding/clients/NEWCO",
                            headers=ADMIN).json()["client"]
        assert detail["status"] == "onboarded"

    def test_an_ordinary_operator_may_draft_but_not_approve(self, api):
        client, _ = api
        did = client.post("/ops/onboarding/drafts",
                          json={"client_id": "client_a"},
                          headers=ALICE).json()["draft"]["draft_id"]
        refused = client.post(
            f"/ops/onboarding/drafts/{did}/approve?client=client_a",
            json={"reason": "trying"}, headers=ALICE)
        assert refused.status_code == 403
        assert refused.json()["detail"]["errorCode"] == "OPS_ADMIN_REQUIRED"

    def test_another_tenants_client_is_not_found(self, api):
        client, _ = api
        assert client.get("/ops/onboarding/clients/client_a",
                          headers=BOB).status_code == 404
        assert client.post("/ops/onboarding/drafts",
                           json={"client_id": "client_a"},
                           headers=BOB).status_code == 404

    def test_the_client_list_is_scoped_to_the_operator(self, api,
                                                       source_registry):
        client, _ = api
        source_registry.upsert(SourceRecord(
            client_id="client_a", source_portfolio_id="p1", dataset="funded",
            frequency="monthly"))
        source_registry.upsert(SourceRecord(
            client_id="client_b", source_portfolio_id="p1", dataset="funded",
            frequency="monthly"))
        rows = client.get("/ops/onboarding/clients",
                          headers=ALICE).json()["clients"]
        assert {r["client_id"] for r in rows} == {"client_a"}

    def test_a_draft_cannot_be_named_after_another_tenants_client(self, api,
                                                                  store):
        """The binding is checked BEFORE the write, so a refused draft never
        lands in the other tenant's prefix."""
        client, _ = api
        did = client.post("/ops/onboarding/drafts", json={},
                          headers=ALICE).json()["draft"]["draft_id"]
        refused = client.put(
            f"/ops/onboarding/drafts/{did}",
            json={"step": "client", "payload": {"client_id": "client_b",
                                                "client_name": "Not mine"}},
            headers=ALICE)
        assert refused.status_code == 404
        assert OnboardingStore(store).list_drafts("client_b") == []

    def test_another_tenants_draft_cannot_be_read(self, api):
        client, _ = api
        did = client.post("/ops/onboarding/drafts",
                          json={"client_id": "client_a"},
                          headers=ALICE).json()["draft"]["draft_id"]
        assert client.get(f"/ops/onboarding/drafts/{did}?client=client_a",
                          headers=BOB).status_code == 404
