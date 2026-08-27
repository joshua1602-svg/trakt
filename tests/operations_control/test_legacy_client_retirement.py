"""Retiring the legacy client document, and pinning what a run actually reads.

Two controls, proved here rather than asserted in a comment.

**A client is never governed by another client's configuration.** The legacy
``config_client_ERM_UK.yaml`` was the universal fallback: any client whose own
configuration could not be read was silently delivered under ERE Funding's
identity, LEI and reporting date. The resolver now fails closed, and this module
holds the cases that prove it — including the ones a broad ``except`` used to
swallow.

**What a run pins is what a run reads.** The effective configuration already
recorded a version and a content hash per governed layer; the Annex 2 chain
still read the repository working tree. These tests execute a run, mutate the
working-tree copy underneath it, and assert the pinned bytes are what reach the
projector.

The legacy document itself is kept as a fixture (``tests/fixtures/legacy_client``)
because the adoption path genuinely reads one. It is no longer reachable from
any production default.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

import operations_control.onboarding.catalogue as cat_mod
from operations_control.configuration.packages import ASSET_MODEL
from operations_control.configuration.resolver import (
    ClientConfigurationUnavailable,
    EffectiveConfigResolver,
)
from operations_control.onboarding import artefacts
from operations_control.onboarding.service import OnboardingService
from operations_control.onboarding.store import OnboardingStore
from operations_control.rules import RuleStore

#: The legacy documents, as a fixture. Production reads neither.
LEGACY_DIR = Path(__file__).resolve().parents[1] / "fixtures" / "legacy_client"
LEGACY_CLIENT_CONFIG = LEGACY_DIR / "config_client_ERM_UK.yaml"
LEGACY_ANNEX12_CONFIG = LEGACY_DIR / "config_client_annex12.yaml"

#: The identity the legacy document carries, and the corrected LEI an operator
#: supplies during adoption. The legacy value is 27 characters where ISO 17442
#: requires 20 — today's validator refuses it, and adoption surfaces that rather
#: than carrying it across.
LEGACY_CLIENT_ID = "ere_funding_uk"
LEGACY_INVALID_LEI = "213800ABCDE123456701N202501"
CORRECTED_LEI = "213800ABCDE123456701"

#: Blocks of the legacy document the onboarding catalogue does not own. They
#: survive activation only through the ``base_documents`` mechanism.
UNOWNED_BLOCKS = ("pipeline_persistence", "transformations", "enrichment")


@pytest.fixture()
def legacy_document() -> dict:
    return yaml.safe_load(LEGACY_CLIENT_CONFIG.read_text(encoding="utf-8"))


@pytest.fixture()
def adopting(store, source_registry, tmp_path) -> OnboardingService:
    """A service whose adoption path reads the legacy fixture explicitly."""
    cat_mod.reset_cache()
    return OnboardingService(
        store, registry_factory=lambda: source_registry,
        adopt_paths={"client_config_path": LEGACY_CLIENT_CONFIG,
                     "annex12_config_path": LEGACY_ANNEX12_CONFIG,
                     "portfolio_registry_path": tmp_path / "absent.yaml",
                     "tenancy_path": tmp_path / "absent.yaml"})


def adopt_and_activate(service: OnboardingService) -> dict:
    """The legacy client, taken through the SAME path as any other client."""
    case = service.start_migration(client_id=LEGACY_CLIENT_ID, by="Operator")
    cid = case.case_id
    existing = service.load_case(cid).items("entities")[0]
    service.save_step(case_id=cid, step="entities", by="Operator", payload={
        "entities": [{**existing, "lei": CORRECTED_LEI,
                      "roles": ["originator", "servicer", "reporting_entity"],
                      "country_of_establishment": "GB"}]})
    service.save_step(case_id=cid, step="contacts", by="Operator", payload={
        "reporting_contact_name": "Andrew Fenlon",
        "reporting_contact_email": "reporting@example.com",
        "reporting_contact_phone": "+44-0000000000",
        "operational_contact_name": "Ops Desk",
        "operational_contact_email": "ops@example.com",
        "investor_report_recipients": "investors@example.com"})
    service.save_step(case_id=cid, step="portfolios", by="Operator", payload={
        "portfolios": [{"portfolio_id": "direct_001",
                        "display_name": "ERE Direct",
                        "asset_class": "equity_release",
                        "portfolio_type": "direct",
                        "portfolio_structure": "whole_loan",
                        "originates": True, "datasets": ["funded"],
                        "cadence": "monthly"}]})
    service.save_step(case_id=cid, step="sources", by="Operator", payload={
        "sources": [{"source_key": "direct_001/funded",
                     "portfolio_id": "direct_001", "dataset": "funded",
                     "cadence": "monthly", "source_party": "ERE core",
                     "delivery_channel": "sftp", "file_format": "csv"}]})
    service.save_step(case_id=cid, step="regime", by="Operator", payload={
        "regime": {"esma_annex2": {
            "originator_name": "ERE Funding Limited",
            "originator_legal_entity_identifier": CORRECTED_LEI,
            "originator_establishment_country": "GB"}}})
    for question in list(service.load_case(cid).open_questions):
        service.resolve_question(
            case_id=cid, question_id=question["question_id"],
            resolution="The registered LEI was supplied by the operator.",
            by="Operator")
    case = service.load_case(cid)
    assert service.readiness(case)["ready"] is True
    service.submit_for_approval(case_id=cid, by="Operator")
    service.approve(case_id=cid, by="Administrator",
                    reason="Adoption of an existing client")
    return service.activate(case_id=cid, by="Administrator")


def generated_config(store, client_id: str) -> dict:
    text = OnboardingStore(store).read_artefact(
        client_id, artefacts.client_config_rel(client_id))
    assert text, f"no generated configuration for {client_id}"
    return yaml.safe_load(text)


# --------------------------------------------------------------------------- #
# Stage 1 — adoption preserves what the legacy document uniquely carried
# --------------------------------------------------------------------------- #

class TestLegacyAdoption:
    def test_the_legacy_document_is_not_production_configuration(self):
        """It lives under the test fixtures, and nowhere a run can reach."""
        assert LEGACY_CLIENT_CONFIG.exists()
        repo = Path(__file__).resolve().parents[2]
        assert not (repo / "config/client/config_client_ERM_UK.yaml").exists()

    def test_an_invalid_legacy_value_is_refused_not_carried(self, adopting):
        """The legacy LEI is 27 characters. Adoption surfaces it; nothing
        silently rewrites it, and the case cannot be approved while it stands."""
        case = adopting.start_migration(client_id=LEGACY_CLIENT_ID,
                                        by="Operator")
        assert case.items("entities")[0]["lei"] == LEGACY_INVALID_LEI
        problems = [p["message"] for p in adopting.readiness(case)["problems"]]
        assert any("20-character identifier" in p for p in problems)

    def test_blocks_the_catalogue_does_not_own_survive_activation(
            self, adopting, store, legacy_document):
        adopt_and_activate(adopting)
        generated = generated_config(store, LEGACY_CLIENT_ID)
        for block in UNOWNED_BLOCKS:
            assert legacy_document[block], f"{block} missing from the fixture"
            assert generated[block] == legacy_document[block], block

    def test_the_nuts_vintage_survives_as_the_governed_enum(
            self, adopting, store, legacy_document):
        """The catalogue owns this one, and declares it an enum of strings. The
        transform coerces with ``str()``, so ``2021`` and ``"2021"`` are the
        same classification year — a representation change, not a behaviour one.
        """
        adopt_and_activate(adopting)
        generated = generated_config(store, LEGACY_CLIENT_ID)
        assert str(generated["nuts_classification_year"]) \
            == str(legacy_document["nuts_classification_year"])

    @pytest.mark.parametrize("block", [
        "portfolio", "regime_overrides", "pipeline", "loan_engine", "mi",
        "supported_regimes", "default_regime", "regime"])
    def test_every_other_block_is_regenerated_identically(
            self, adopting, store, legacy_document, block):
        adopt_and_activate(adopting)
        assert generated_config(store, LEGACY_CLIENT_ID)[block] \
            == legacy_document[block]

    def test_the_client_identity_is_carried_and_extended(
            self, adopting, store, legacy_document):
        adopt_and_activate(adopting)
        generated = generated_config(store, LEGACY_CLIENT_ID)
        legacy_client = legacy_document["client"]
        assert generated["client"]["client_id"] == legacy_client["client_id"]
        assert generated["client"]["display_name"] \
            == legacy_client["display_name"]
        assert generated["client"]["environment"] == legacy_client["environment"]

    def test_the_originator_defaults_carry_the_valid_identifier(
            self, adopting, store, legacy_document):
        """Name and country are unchanged; the identifier is the one the
        operator supplied, because the legacy value fails ISO 17442."""
        adopt_and_activate(adopting)
        defaults = generated_config(store, LEGACY_CLIENT_ID)["defaults"]
        legacy_defaults = legacy_document["defaults"]
        assert defaults["originator_name"] == legacy_defaults["originator_name"]
        assert defaults["originator_establishment_country"] \
            == legacy_defaults["originator_establishment_country"]
        assert defaults["originator_legal_entity_identifier"] == CORRECTED_LEI

    def test_activation_is_versioned_hashed_and_convergent(self, adopting,
                                                           store):
        result = adopt_and_activate(adopting)
        assert result["version"] == 1
        current = OnboardingStore(store).current(LEGACY_CLIENT_ID)
        assert current.version == 1
        assert current.content_hash
        # An amendment that changes no answer converges on the version already
        # in force rather than minting an identical second one.
        amendment = adopting.start_amendment(client_id=LEGACY_CLIENT_ID,
                                             by="Operator")
        adopting.submit_for_approval(case_id=amendment.case_id, by="Operator")
        adopting.approve(case_id=amendment.case_id, by="Administrator",
                         reason="No change")
        again = adopting.activate(case_id=amendment.case_id, by="Administrator")
        assert again["version"] == 1
        assert OnboardingStore(store).current(
            LEGACY_CLIENT_ID).content_hash == current.content_hash


# --------------------------------------------------------------------------- #
# Stage 2 — client isolation, and a resolver that fails closed
# --------------------------------------------------------------------------- #

class _BrokenStorage:
    """Storage whose reads raise — a transient fault, not an absent client."""

    def __init__(self, inner, failing_client: str):
        self._inner = inner
        self._failing = failing_client

    def __getattr__(self, name):
        return getattr(self._inner, name)

    def read_text(self, uri: str, *a, **kw):
        if self._failing in str(uri):
            raise OSError("storage is unavailable")
        return self._inner.read_text(uri, *a, **kw)

    def exists(self, uri: str, *a, **kw):
        if self._failing in str(uri):
            raise OSError("storage is unavailable")
        return self._inner.exists(uri, *a, **kw)


@pytest.fixture()
def resolver(store) -> EffectiveConfigResolver:
    return EffectiveConfigResolver(store, RuleStore(store))


class TestClientIsolation:
    def test_an_onboarded_client_resolves_its_own_configuration(
            self, adopting, store, resolver):
        adopt_and_activate(adopting)
        doc = yaml.safe_load(
            resolver.client_config_for(LEGACY_CLIENT_ID).read_text(
                encoding="utf-8"))
        assert doc["client"]["client_id"] == LEGACY_CLIENT_ID

    def test_two_clients_never_share_a_configuration(self, adopting, store,
                                                     source_registry,
                                                     resolver):
        """Client B is onboarded from nothing; nothing of ERE's reaches it."""
        from .test_onboarding import activate as activate_case, complete
        adopt_and_activate(adopting)
        second = OnboardingService(store,
                                   registry_factory=lambda: source_registry)
        activate_case(second, complete(second, client_id="CLIENT_B"))

        ere = yaml.safe_load(resolver.client_config_for(
            LEGACY_CLIENT_ID).read_text(encoding="utf-8"))
        other = yaml.safe_load(resolver.client_config_for(
            "CLIENT_B").read_text(encoding="utf-8"))
        assert ere["client"]["client_id"] == LEGACY_CLIENT_ID
        assert other["client"]["client_id"] == "CLIENT_B"
        assert other["defaults"]["originator_name"] \
            != ere["defaults"]["originator_name"]
        assert LEGACY_CLIENT_ID not in yaml.safe_dump(other)
        assert "ERE" not in yaml.safe_dump(other)

    def test_a_client_with_no_activated_onboarding_does_not_resolve_ere(
            self, adopting, resolver):
        adopt_and_activate(adopting)
        with pytest.raises(ClientConfigurationUnavailable) as raised:
            resolver.client_config_for("NEVER_ONBOARDED")
        assert raised.value.reason == "not_onboarded"

    def test_a_storage_failure_does_not_resolve_ere(self, adopting, store,
                                                    resolver):
        """A transient fault is not an absent client, and never a fallback."""
        adopt_and_activate(adopting)
        store.storage = _BrokenStorage(store.storage, "CLIENT_B")
        with pytest.raises(ClientConfigurationUnavailable) as raised:
            resolver.client_config_for("CLIENT_B")
        assert raised.value.reason == "unreadable"

    def test_a_corrupt_activated_artefact_blocks(self, adopting, store,
                                                 resolver):
        adopt_and_activate(adopting)
        onboarding = OnboardingStore(store)
        store.storage.write_text(
            onboarding._artefact_current_uri(
                LEGACY_CLIENT_ID,
                artefacts.client_config_rel(LEGACY_CLIENT_ID)),
            "client: [unclosed\n")
        with pytest.raises(ClientConfigurationUnavailable) as raised:
            resolver.client_config_for(LEGACY_CLIENT_ID)
        assert raised.value.reason == "invalid"

    def test_a_missing_client_id_is_refused(self, resolver):
        with pytest.raises(ClientConfigurationUnavailable) as raised:
            resolver.client_config_for("")
        assert raised.value.reason == "client_required"

    def test_a_failed_resolution_cannot_produce_a_ready_configuration(
            self, resolver):
        outcome = resolver.resolve(client_id="NEVER_ONBOARDED",
                                   portfolio_id="direct_001")
        assert outcome.status == "BLOCKED"
        assert outcome.effective is None
        assert any("onboard" in b.lower() for b in outcome.blockers)


# --------------------------------------------------------------------------- #
# Stage 6 — the reporting products equity release actually supports
# --------------------------------------------------------------------------- #

class TestEquityReleaseReportingProducts:
    def test_both_reporting_products_are_declared(self):
        """Annex 2 reports the underlying exposures; Annex 12 reports to
        investors. They are complementary, and equity release supports both."""
        supported = ASSET_MODEL["equity_release"]["supports_regimes"]
        assert "ESMA_Annex2" in supported
        assert "ESMA_Annex12" in supported

    def test_annex_2_remains_the_default_regime(self, adopting, store):
        """Declaring Annex 12 must not change which regime a delivery projects."""
        adopt_and_activate(adopting)
        generated = generated_config(store, LEGACY_CLIENT_ID)
        assert generated["default_regime"] == "ESMA_Annex2"
        assert generated["regime"] == "ESMA_Annex2"
        assert generated["supported_regimes"][0] == "ESMA_Annex2"


# --------------------------------------------------------------------------- #
# Stage 5 — a run reads the configuration it pinned
# --------------------------------------------------------------------------- #

class TestRuntimePinning:
    def _annex2_run(self, engine, delivery_dir, client_id="client_a"):
        from operations_control.contracts import RUN_AWAITING_PUBLICATION
        from .conftest import register_and_create, start_and_wait
        run = register_and_create(engine, delivery_dir, outcome="mi_annex2",
                                  client_id=client_id)
        final = start_and_wait(engine, run)
        assert final.status == RUN_AWAITING_PUBLICATION, final.blockers
        return final

    def test_the_projector_is_given_this_run_s_own_client_configuration(
            self, store, source_registry, delivery_dir, tmp_path):
        """Not the repository, and not whichever client owned a default."""
        from .conftest import StubAnnex2Stages, make_client_config, make_engine
        stages = StubAnnex2Stages()
        config = make_client_config(tmp_path, valid=True)
        engine = make_engine(store, source_registry, annex2_stages=stages,
                             client_config=config)
        self._annex2_run(engine, delivery_dir)
        projections = [c for c in stages.calls
                       if isinstance(c, tuple) and c[0] == "projection"]
        assert projections, "the projection stage did not run"
        supplied = Path(projections[0][1])
        assert supplied.exists()
        assert yaml.safe_load(supplied.read_text(encoding="utf-8")) \
            == yaml.safe_load(config.read_text(encoding="utf-8"))

    def test_editing_the_activated_configuration_does_not_reach_a_pinned_run(
            self, store, source_registry, delivery_dir, tmp_path):
        """A → mutate underneath → A. The run keeps the bytes it pinned.

        The client configuration is edited in the store after the run pinned
        it. Re-materialising the run's pinned layer refuses the changed bytes
        rather than quietly delivering them, because they are not what the
        effective configuration hashed.
        """
        from .conftest import (StubAnnex2Stages, make_client_config,
                               make_engine, onboard_client)
        stages = StubAnnex2Stages()
        original = make_client_config(tmp_path, valid=True)
        engine = make_engine(store, source_registry, annex2_stages=stages,
                             client_config=original)
        final = self._annex2_run(engine, delivery_dir)
        pinned = engine._pinned_client_config(final)
        pinned_text = pinned.read_text(encoding="utf-8")

        edited = tmp_path / "edited.yaml"
        edited.write_text(
            original.read_text(encoding="utf-8").replace("GB", "IE"),
            encoding="utf-8")
        onboard_client(store, final.client_id, edited)

        # The already-materialised snapshot still holds the pinned bytes...
        assert engine._pinned_client_config(final).read_text(
            encoding="utf-8") == pinned_text
        # ...and re-materialising from scratch refuses the changed document.
        pinned.unlink()
        with pytest.raises(ClientConfigurationUnavailable) as raised:
            engine._pinned_client_config(final)
        assert raised.value.reason == "invalid"

    def test_a_later_run_pins_the_new_configuration(
            self, store, source_registry, delivery_dir, tmp_path):
        """B → B. A new run resolves and pins the configuration now in force."""
        from .conftest import (StubAnnex2Stages, make_client_config,
                               make_engine, onboard_client)
        stages = StubAnnex2Stages()
        engine = make_engine(store, source_registry, annex2_stages=stages,
                             client_config=make_client_config(tmp_path,
                                                              valid=True))
        self._annex2_run(engine, delivery_dir)

        second = tmp_path / "b"
        second.mkdir()
        changed = tmp_path / "changed.yaml"
        changed.write_text(
            make_client_config(second, valid=True).read_text(
                encoding="utf-8").replace("GBP", "EUR"), encoding="utf-8")
        onboard_client(store, "client_a", changed)

        stages.calls.clear()
        self._annex2_run(engine, delivery_dir)
        supplied = Path([c for c in stages.calls
                         if isinstance(c, tuple) and c[0] == "projection"][0][1])
        assert "EUR" in supplied.read_text(encoding="utf-8")

    def test_a_governed_package_file_edited_in_the_working_tree_is_ignored(
            self, store, tmp_path):
        """The system package pins its own bytes; the working tree is not read.

        ``materialise`` re-checks every file against the hash the package
        recorded, so a snapshot can only ever contain what was pinned.
        """
        from operations_control.configuration.packages import (
            ConfigPackageStore, LAYER_SYSTEM,
        )
        packages = ConfigPackageStore(store)
        seeded = packages.ensure_seeded(LAYER_SYSTEM)
        rel = "config/system/onboarding_modes.yaml"
        pinned_text = seeded["files"][rel]["content"]

        repo_copy = Path(__file__).resolve().parents[2] / rel
        before = repo_copy.read_text(encoding="utf-8")
        try:
            repo_copy.write_text(before + "\n# edited underneath the run\n",
                                 encoding="utf-8")
            written = packages.materialise(LAYER_SYSTEM, seeded["version"],
                                           tmp_path / "snap")
            assert written[rel].read_text(encoding="utf-8") == pinned_text
            assert "edited underneath the run" not in pinned_text
        finally:
            repo_copy.write_text(before, encoding="utf-8")
