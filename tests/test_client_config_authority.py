"""One client configuration, read by both MI and regime — or by neither.

The cutover's premise is that after onboarding, a client's standing
configuration is the artefact OCC activated for it, and that BOTH consumers
read that same artefact:

* MI, for reporting currency, asset class and the rest of the client layer;
* Regime / Annex 2, for the originator identity a regulatory return carries.

Two readers of one fact that can disagree is a defect waiting for a deadline,
so these tests prove they resolve the same artefact — and, just as important,
that a client with no activated configuration gets NOTHING from either.

Why "nothing" is the point. The old behaviour substituted the repository file
for an unonboarded client, and that file is the incumbent lender's. A
regulatory return built that way carries someone else's LEI, originator name
and establishment country: not a degraded answer, a wrong one that looks right.
An unonboarded client is a VALID platform state — precisely the state between a
client wipe and its fresh onboarding — and must fail closed.

Everything here is file-backed in a tmp directory, with a deliberately non-ERE
synthetic client, so nothing depends on the repository's own client files.
"""

from __future__ import annotations

import pytest
import yaml

from operations_control.configuration.resolver import BLOCKED
from operations_control.configuration.client_config import (
    DEV_OVERRIDE_ENV,
    SOURCE_DEV_OVERRIDE,
    SOURCE_OCC_ACTIVATED,
    get_active_client_config,
    is_configured,
)
from operations_control.onboarding.artefacts import client_config_rel
from operations_control.onboarding.store import OnboardingStore
from operations_control.stores import OpsLayout, OpsStore

TEST_CLIENT = "TESTCLIENT"
UNKNOWN_CLIENT = "UNKNOWNCLIENT"

#: A distinctive, harmless standing value. EUR is not what any repository
#: client file says, so seeing it proves the OCC artefact was read and not a
#: repo file that happened to be lying around.
DISTINCTIVE = {
    "client": {"client_id": TEST_CLIENT, "display_name": "Test Client",
               "reporting_currency": "EUR"},
    "portfolio": {"country": "IE", "base_currency": "EUR",
                  "asset_class": "equity_release"},
    "defaults": {"originator_name": "Test Client Ltd",
                 "originator_legal_entity_identifier": "894500TESTCLIENT001",
                 "originator_establishment_country": "IE"},
}


@pytest.fixture()
def ops(tmp_path, monkeypatch):
    """A file-backed operations-control container with nothing in it."""
    monkeypatch.setenv("TRAKT_STORAGE_BACKEND", "file")
    monkeypatch.setenv("TRAKT_LOCAL_BLOB_ROOT", str(tmp_path / "blob"))
    monkeypatch.delenv(DEV_OVERRIDE_ENV, raising=False)
    monkeypatch.delenv("TRAKT_MI_CLIENT_CONFIG", raising=False)
    monkeypatch.delenv("TRAKT_OPS_CLIENT_CONFIG", raising=False)
    # Production runtime: the development overrides must be inert.
    monkeypatch.setenv("TRAKT_RUNTIME_MODE", "production")
    from apps.blob_trigger_app.storage import Storage
    return OpsStore(Storage(tmp_path / "blob"),
                    OpsLayout(container="operations-control"))


def _activate(ops, client_id: str, document: dict) -> str:
    """Write an activated client configuration exactly where OCC writes one."""
    onboarding = OnboardingStore(ops)
    return onboarding.write_artefact(
        client_id, 1, client_config_rel(client_id),
        yaml.safe_dump(document, sort_keys=False))


# --------------------------------------------------------------------------- #
# The authority itself
# --------------------------------------------------------------------------- #

class TestTheAuthority:
    def test_an_activated_client_resolves(self, ops):
        _activate(ops, TEST_CLIENT, DISTINCTIVE)
        active = get_active_client_config(TEST_CLIENT, store=ops)
        assert active is not None
        assert active.client_id == TEST_CLIENT
        assert active.document["client"]["reporting_currency"] == "EUR"
        assert active.source == SOURCE_OCC_ACTIVATED
        assert active.is_governed is True
        assert active.content_hash

    def test_an_unonboarded_client_resolves_to_nothing(self, ops):
        assert get_active_client_config(UNKNOWN_CLIENT, store=ops) is None
        assert is_configured(UNKNOWN_CLIENT, store=ops) is False

    def test_no_client_resolves_to_nothing(self, ops):
        """Reading 'the' configuration with no client named is how one tenant's
        decision reaches another."""
        assert get_active_client_config(None, store=ops) is None
        assert get_active_client_config("", store=ops) is None

    def test_one_clients_config_is_never_served_for_another(self, ops):
        _activate(ops, TEST_CLIENT, DISTINCTIVE)
        assert get_active_client_config(UNKNOWN_CLIENT, store=ops) is None

    def test_the_uri_names_the_governed_artefact(self, ops):
        _activate(ops, TEST_CLIENT, DISTINCTIVE)
        active = get_active_client_config(TEST_CLIENT, store=ops)
        assert active.uri.startswith("blob://operations-control/")
        assert TEST_CLIENT in active.uri
        assert active.uri.endswith(f"config_client_{TEST_CLIENT}.yaml")

    def test_the_development_override_is_inert_in_production(
            self, ops, tmp_path, monkeypatch):
        """A stray app setting must not outrank an activated configuration."""
        stray = tmp_path / "someone_elses.yaml"
        stray.write_text(yaml.safe_dump(
            {"client": {"client_id": "OTHER", "reporting_currency": "JPY"}}),
            encoding="utf-8")
        monkeypatch.setenv(DEV_OVERRIDE_ENV, str(stray))
        monkeypatch.setenv("TRAKT_RUNTIME_MODE", "production")
        assert get_active_client_config(UNKNOWN_CLIENT, store=ops) is None

    def test_the_development_override_works_outside_production(
            self, ops, tmp_path, monkeypatch):
        override = tmp_path / "dev.yaml"
        override.write_text(yaml.safe_dump(DISTINCTIVE), encoding="utf-8")
        monkeypatch.setenv(DEV_OVERRIDE_ENV, str(override))
        monkeypatch.setenv("TRAKT_RUNTIME_MODE", "development")
        active = get_active_client_config(TEST_CLIENT, store=ops)
        assert active is not None
        assert active.source == SOURCE_DEV_OVERRIDE
        # And it is never mistaken for a governed activation.
        assert active.is_governed is False


# --------------------------------------------------------------------------- #
# MI and regime read the SAME artefact
# --------------------------------------------------------------------------- #

class TestMiAndRegimeParity:
    def test_mi_resolves_the_activated_artefact(self, ops, monkeypatch):
        _activate(ops, TEST_CLIENT, DISTINCTIVE)
        from mi_agent_api import currency

        location = currency.client_config_path(TEST_CLIENT)
        assert location is not None, "MI resolved no configuration"
        assert location.startswith("blob://operations-control/")
        assert TEST_CLIENT in location

    def test_mi_reads_the_distinctive_value(self, ops, monkeypatch):
        """Not just the path — the value MI actually uses."""
        _activate(ops, TEST_CLIENT, DISTINCTIVE)
        from mi_agent_api import currency
        currency._load_client_config.cache_clear()

        code = currency.governed_currency_code(client_id=TEST_CLIENT)
        assert code == "EUR"

    def test_regime_resolves_the_activated_artefact(self, ops):
        _activate(ops, TEST_CLIENT, DISTINCTIVE)
        from operations_control.configuration.resolver import (
            EffectiveConfigResolver,
        )
        from operations_control.rules import RuleStore

        resolver = EffectiveConfigResolver(ops, RuleStore(ops))
        path = resolver.client_config_for(TEST_CLIENT)
        assert path is not None, "regime resolved no configuration"
        doc = yaml.safe_load(path.read_text(encoding="utf-8"))
        assert doc["client"]["reporting_currency"] == "EUR"

    def test_both_resolve_the_same_content(self, ops):
        """The parity the cutover turns on: one artefact, one hash."""
        _activate(ops, TEST_CLIENT, DISTINCTIVE)
        from mi_agent_api import currency
        from operations_control.configuration.resolver import (
            EffectiveConfigResolver,
        )
        from operations_control.rules import RuleStore

        currency._load_client_config.cache_clear()
        mi_doc = currency._load_client_config(
            currency.client_config_path(TEST_CLIENT))
        regime_path = EffectiveConfigResolver(
            ops, RuleStore(ops)).client_config_for(TEST_CLIENT)
        regime_doc = yaml.safe_load(regime_path.read_text(encoding="utf-8"))

        assert mi_doc == regime_doc
        assert mi_doc["client"]["reporting_currency"] == "EUR"
        assert (mi_doc["defaults"]["originator_legal_entity_identifier"]
                == "894500TESTCLIENT001")


# --------------------------------------------------------------------------- #
# The pre-client empty state — what production looks like after the wipe
# --------------------------------------------------------------------------- #

class TestEmptyStateIsSafe:
    def test_mi_fails_closed_for_an_unonboarded_client(self, ops):
        from mi_agent_api import currency
        assert currency.client_config_path(UNKNOWN_CLIENT) is None

    def test_mi_does_not_reach_a_repository_lender_config(self, ops):
        """The repo still HAS config_client_ERE.yaml. MI must not find it."""
        from mi_agent_api import currency
        for client in (UNKNOWN_CLIENT, "ERE"):
            location = currency.client_config_path(client)
            assert location is None or location.startswith("blob://"), (
                f"MI resolved a repository file for {client}: {location}")

    def test_regime_fails_closed_for_an_unonboarded_client(self, ops):
        from operations_control.configuration.resolver import (
            EffectiveConfigResolver,
        )
        from operations_control.rules import RuleStore

        resolver = EffectiveConfigResolver(ops, RuleStore(ops))
        assert resolver.client_config_for(UNKNOWN_CLIENT) is None

    def test_regime_resolution_blocks_rather_than_substituting(self, ops):
        from operations_control.configuration.resolver import (
            EffectiveConfigResolver,
        )
        from operations_control.rules import RuleStore

        outcome = EffectiveConfigResolver(ops, RuleStore(ops)).resolve(
            client_id=UNKNOWN_CLIENT, portfolio_id="p1",
            outcome="mi_annex2", reporting_period="2026-06-30")
        assert outcome.status == BLOCKED
        assert any("activated client configuration" in b.lower()
                   for b in outcome.blockers), outcome.blockers

    def test_an_empty_platform_is_a_valid_state_not_a_crash(self, ops):
        """No clients at all: every answer is 'not configured', nothing raises."""
        from mi_agent_api import currency
        from operations_control.configuration.resolver import (
            EffectiveConfigResolver,
        )
        from operations_control.rules import RuleStore

        assert OnboardingStore(ops).onboarded_clients() == []
        assert currency.client_config_path("ANYONE") is None
        assert EffectiveConfigResolver(
            ops, RuleStore(ops)).client_config_for("ANYONE") is None
