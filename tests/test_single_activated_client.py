"""Owner decision D19 (2026-09-30): one activated client, and MI reads it.

"There should only be one single activated client in trakt - it's already
plugged into the mi dashboard. Any clients that are not active are just
test / dummy runs."

The MI deployment knows its tenant by a deployment identifier (``client_001``);
OCC keys the activation by the client's own (``ERE``). Before D19 that meant MI
read none of the activated configuration — no portfolio stage, so every "scale"
question was declined as not configured. These tests pin the resolution AND
every fail-closed rule it must not loosen:

* only the ONE tenant a single-tenant deployment serves is resolved this way;
* any other client still gets its own activation or nothing;
* zero activated clients, or more than one, resolve to nothing;
* a client that never activated (a test / dummy run) is not counted.

File-backed, in a tmp directory, with synthetic clients: nothing here depends
on the repository's own client files.
"""
from __future__ import annotations

import pytest
import yaml

from operations_control.configuration.client_config import (
    DEV_OVERRIDE_ENV,
    activated_clients,
    get_single_activated_client_config,
)
from operations_control.onboarding.artefacts import client_config_rel
from operations_control.onboarding.case import OnboardingCase
from operations_control.onboarding.store import OnboardingStore
from operations_control.stores import OpsLayout, OpsStore

SERVED = "client_001"          # what the MI deployment calls its tenant
ACTIVATED = "LENDERX"          # what OCC activated the client as


def _document(client_id: str, *, stage: str = "pre_securitisation_spv",
              currency: str = "EUR") -> dict:
    return {"client": {"client_id": client_id.lower(),
                       "display_name": f"{client_id} Ltd"},
            "portfolio": {"base_currency": currency, "stage": stage,
                          "asset_class": "equity_release"}}


@pytest.fixture()
def ops(tmp_path, monkeypatch):
    """An empty file-backed operations-control container, as the default
    store, in a single-tenant MI deployment serving ``client_001``."""
    monkeypatch.setenv("TRAKT_STORAGE_BACKEND", "file")
    monkeypatch.setenv("TRAKT_LOCAL_BLOB_ROOT", str(tmp_path / "blob"))
    monkeypatch.setenv("TRAKT_OPS_CONTAINER", "operations-control")
    monkeypatch.delenv(DEV_OVERRIDE_ENV, raising=False)
    monkeypatch.delenv("TRAKT_MI_CLIENT_CONFIG", raising=False)
    monkeypatch.setenv("TRAKT_RUNTIME_MODE", "production")
    monkeypatch.setenv("MI_AGENT_CLIENT_ID", SERVED)
    monkeypatch.setenv("TRAKT_TENANCY_CONFIG", str(tmp_path / "no-tenancy.yaml"))
    from apps.blob_trigger_app.storage import Storage
    from mi_agent_api import currency
    currency._load_client_config.cache_clear()
    store = OpsStore(Storage(tmp_path / "blob"),
                     OpsLayout(container="operations-control"))
    monkeypatch.setattr(OpsStore, "from_env", classmethod(lambda cls: store))
    yield store
    currency._load_client_config.cache_clear()


def _activate(ops, client_id: str, document: dict) -> None:
    """Activate a configuration exactly as OCC does: an immutable version,
    the current pointer, the client index, and the generated artefact."""
    onboarding = OnboardingStore(ops)
    case = OnboardingCase(case_id=f"ONB-2026-{abs(hash(client_id)) % 9999:04d}",
                          client_id=client_id, answers={"client": client_id})
    version = onboarding.commit(case=case, changes=[], artefacts=[],
                                activated_by="test")
    onboarding.write_artefact(client_id, version.version,
                              client_config_rel(client_id),
                              yaml.safe_dump(document, sort_keys=False))


def _open_only(ops, client_id: str) -> None:
    """A test / dummy run: a case opened for a client, never activated."""
    OnboardingStore(ops).save_case(OnboardingCase(
        case_id="ONB-2026-0999", client_id=client_id))


# --------------------------------------------------------------------------- #
# OCC: which client is THE activated one
# --------------------------------------------------------------------------- #
class TestTheSingleActivatedClient:
    def test_the_one_activated_client_is_resolved(self, ops):
        _activate(ops, ACTIVATED, _document(ACTIVATED))
        active = get_single_activated_client_config(store=ops)
        assert active is not None
        assert active.client_id == ACTIVATED
        assert active.is_governed
        assert active.document["portfolio"]["stage"] == "pre_securitisation_spv"

    def test_a_dummy_run_that_never_activated_is_not_counted(self, ops):
        _activate(ops, ACTIVATED, _document(ACTIVATED))
        _open_only(ops, "DUMMY")
        assert activated_clients(store=ops) == [ACTIVATED]
        assert get_single_activated_client_config(store=ops).client_id == ACTIVATED

    def test_no_activated_client_is_not_configured(self, ops):
        _open_only(ops, "DUMMY")
        assert get_single_activated_client_config(store=ops) is None

    def test_two_activated_clients_are_never_chosen_between(self, ops):
        _activate(ops, ACTIVATED, _document(ACTIVATED))
        _activate(ops, "OTHER", _document("OTHER", currency="USD"))
        assert get_single_activated_client_config(store=ops) is None


# --------------------------------------------------------------------------- #
# MI: the served tenant reads it; nobody else does
# --------------------------------------------------------------------------- #
class TestMiReadsTheActivatedClient:
    def test_the_served_tenant_resolves_to_the_activated_artefact(self, ops):
        _activate(ops, ACTIVATED, _document(ACTIVATED))
        from mi_agent_api import currency
        location = currency.client_config_path(SERVED)
        assert location is not None
        assert location.startswith("blob://operations-control/")
        assert ACTIVATED in location

    def test_scale_has_a_threshold_for_the_served_tenant(self, ops):
        """The finding D19 closes: 'scale' was declined as not configured."""
        _activate(ops, ACTIVATED, _document(ACTIVATED))
        from mi_agent_api import scale_policy
        threshold, reason, _ = scale_policy.resolve(SERVED)
        assert reason == ""
        assert threshold is not None
        assert threshold.stage == "pre_securitisation_spv"

    def test_the_served_tenant_reads_the_activated_currency(self, ops):
        _activate(ops, ACTIVATED, _document(ACTIVATED, currency="EUR"))
        from mi_agent_api import currency
        assert currency.governed_currency_code(SERVED) == "EUR"

    def test_another_client_never_reads_it(self, ops):
        _activate(ops, ACTIVATED, _document(ACTIVATED))
        from mi_agent_api import currency
        assert currency.client_config_path("some_other_lender") is None
        assert currency.governed_currency_code("some_other_lender") is None

    def test_the_activated_client_still_reads_its_own(self, ops):
        _activate(ops, ACTIVATED, _document(ACTIVATED))
        from mi_agent_api import currency
        assert ACTIVATED in currency.client_config_path(ACTIVATED)

    def test_two_activated_clients_leave_the_served_tenant_unconfigured(self, ops):
        _activate(ops, ACTIVATED, _document(ACTIVATED))
        _activate(ops, "OTHER", _document("OTHER"))
        from mi_agent_api import currency
        assert currency.client_config_path(SERVED) is None

    def test_a_multi_tenant_deployment_infers_nothing(self, ops, tmp_path,
                                                      monkeypatch):
        tenancy = tmp_path / "tenancy.yaml"
        tenancy.write_text(yaml.safe_dump(
            {"tenants": {SERVED: {"display_name": "Served"},
                         "client_002": {"display_name": "Another"}}}))
        monkeypatch.setenv("TRAKT_TENANCY_CONFIG", str(tenancy))
        _activate(ops, ACTIVATED, _document(ACTIVATED))
        from mi_agent_api import currency
        assert currency.client_config_path(SERVED) is None

    def test_an_unidentified_client_still_reads_nothing(self, ops):
        _activate(ops, ACTIVATED, _document(ACTIVATED))
        from mi_agent_api import currency
        assert currency.client_config_path(None) is None
        assert currency.client_config_path("") is None

    def test_the_resolution_is_made_once_per_request(self, ops, monkeypatch):
        """Currency, geography, stage and the cache fingerprint all ask."""
        _activate(ops, ACTIVATED, _document(ACTIVATED))
        from mi_agent_api import currency, request_scope
        from operations_control.configuration import client_config
        calls = []
        real = client_config.get_single_activated_client_config

        def counted(**kwargs):
            calls.append(1)
            return real(**kwargs)

        monkeypatch.setattr(client_config, "get_single_activated_client_config",
                            counted)
        with request_scope.scope():
            first = currency.client_config_path(SERVED)
            for _ in range(4):
                assert currency.client_config_path(SERVED) == first
        assert len(calls) == 1
