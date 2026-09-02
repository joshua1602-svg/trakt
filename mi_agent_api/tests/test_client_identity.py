"""The dashboard header names the client, and never names the wrong one.

Reported defect: the client control read "Platform". Two causes, both fixed
here. The blob-triggered discovery index took the raw ``MI_AGENT_CLIENT_ID`` and
fell through to the literal string ``"platform"`` when it was unset, while every
other surface resolved a real client id; and even with a client id in hand, the
label was ``client_id.upper()`` — an internal identifier on the surface a client
looks at first.

The name is a GOVERNED fact. These tests pin where it may come from, and — the
point of the module — where it may not: no source that could attribute one
client's name to another book is allowed to answer.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import yaml

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from mi_agent_api import client_identity as identity  # noqa: E402
from mi_agent_api import currency as currency_mod  # noqa: E402


@pytest.fixture(autouse=True)
def _clean(monkeypatch, tmp_path):
    """Every test starts with no governed source in force and no memo."""
    identity.cache_clear()
    currency_mod._load_client_config.cache_clear()
    monkeypatch.delenv(currency_mod.ENV_CLIENT_CONFIG, raising=False)
    monkeypatch.setattr(identity, "_CLIENTS_ROOT", tmp_path / "clients")
    # Point the tenancy loader at nothing, so the default is "single tenant".
    monkeypatch.setenv("TRAKT_TENANCY_CONFIG", str(tmp_path / "absent-tenancy.yaml"))
    monkeypatch.setenv("MI_AGENT_CLIENT_ID", "acme_bank")
    yield
    identity.cache_clear()
    currency_mod._load_client_config.cache_clear()


def _client_file(root: Path, client_id: str, doc: dict) -> None:
    d = root / client_id
    d.mkdir(parents=True, exist_ok=True)
    (d / "client.yaml").write_text(yaml.safe_dump(doc), encoding="utf-8")


def _config(tmp_path: Path, doc: dict, name: str = "cfg.yaml") -> str:
    p = tmp_path / name
    p.write_text(yaml.safe_dump(doc), encoding="utf-8")
    return str(p)


def _tenancy(tmp_path: Path, tenants: dict) -> str:
    p = tmp_path / "tenancy.yaml"
    p.write_text(yaml.safe_dump({"tenants": tenants}), encoding="utf-8")
    return str(p)


# --------------------------------------------------------------------------- #
# The name is read from governed sources, most-specific first
# --------------------------------------------------------------------------- #
def test_the_per_client_directory_names_the_client(tmp_path):
    _client_file(tmp_path / "clients", "acme_bank",
                 {"client": {"display_name": "Acme Bank plc"}})
    assert identity.governed_client_name("acme_bank") == "Acme Bank plc"


def test_the_tenant_registry_names_the_client(tmp_path, monkeypatch):
    monkeypatch.setenv("TRAKT_TENANCY_CONFIG",
                       _tenancy(tmp_path, {"acme_bank": {"display_name": "Acme Bank plc"}}))
    assert identity.governed_client_name("acme_bank") == "Acme Bank plc"


def test_the_client_configuration_names_the_client(tmp_path, monkeypatch):
    monkeypatch.setenv(currency_mod.ENV_CLIENT_CONFIG, _config(tmp_path, {
        "client": {"client_id": "acme_bank", "display_name": "Acme Bank plc"}}))
    assert identity.governed_client_name("acme_bank") == "Acme Bank plc"


def test_the_per_client_directory_outranks_the_registry(tmp_path, monkeypatch):
    _client_file(tmp_path / "clients", "acme_bank",
                 {"client": {"display_name": "Acme Bank plc"}})
    monkeypatch.setenv("TRAKT_TENANCY_CONFIG",
                       _tenancy(tmp_path, {"acme_bank": {"display_name": "Stale Name Ltd"}}))
    assert identity.governed_client_name("acme_bank") == "Acme Bank plc"


@pytest.mark.parametrize("doc", [
    {"client": {"client_name": "Acme Bank plc"}},
    {"client": {"business_name": "Acme Bank plc"}},
    {"display_name": "Acme Bank plc"},
])
def test_the_conventional_name_keys_are_all_read(tmp_path, doc):
    _client_file(tmp_path / "clients", "acme_bank", doc)
    assert identity.governed_client_name("acme_bank") == "Acme Bank plc"


# --------------------------------------------------------------------------- #
# No source may attribute one client's name to another
# --------------------------------------------------------------------------- #
def test_a_multi_tenant_deployment_never_borrows_an_unaddressed_config(tmp_path, monkeypatch):
    """The whole reason this module exists. With several tenants served, a
    client config that does not say whose it is names nobody."""
    monkeypatch.setenv("TRAKT_TENANCY_CONFIG", _tenancy(tmp_path, {
        "acme_bank": {"default_portfolio": "acme_bank"},
        "beta_bank": {"default_portfolio": "beta_bank"}}))
    monkeypatch.setenv(currency_mod.ENV_CLIENT_CONFIG,
                       _config(tmp_path, {"client": {"display_name": "Acme Bank plc"}}))
    assert identity.governed_client_name("beta_bank") is None
    assert identity.portfolio_label("beta_bank") == "BETA_BANK"


def test_a_config_addressed_to_another_client_names_nobody(tmp_path, monkeypatch):
    monkeypatch.setenv(currency_mod.ENV_CLIENT_CONFIG, _config(tmp_path, {
        "client": {"client_id": "acme_bank", "display_name": "Acme Bank plc"}}))
    assert identity.governed_client_name("beta_bank") is None


def test_a_client_this_deployment_does_not_serve_is_not_named(tmp_path, monkeypatch):
    """Single-tenant inference is limited to the tenant actually served."""
    monkeypatch.setenv("MI_AGENT_CLIENT_ID", "acme_bank")
    monkeypatch.setenv(currency_mod.ENV_CLIENT_CONFIG,
                       _config(tmp_path, {"client": {"display_name": "Acme Bank plc"}}))
    assert identity.governed_client_name("acme_bank") == "Acme Bank plc"
    assert identity.governed_client_name("beta_bank") is None


def test_the_registry_only_answers_for_its_own_tenant(tmp_path, monkeypatch):
    monkeypatch.setenv("TRAKT_TENANCY_CONFIG", _tenancy(tmp_path, {
        "acme_bank": {"display_name": "Acme Bank plc"},
        "beta_bank": {"display_name": "Beta Bank NV"}}))
    assert identity.governed_client_name("acme_bank") == "Acme Bank plc"
    assert identity.governed_client_name("beta_bank") == "Beta Bank NV"


def test_an_unidentified_caller_is_never_named():
    """A client must be identified before anything can name it."""
    assert identity.governed_client_name(None) is None
    assert identity.governed_client_name("") is None
    assert identity.governed_client_name("   ") is None


# --------------------------------------------------------------------------- #
# The fallback is exactly the previous behaviour
# --------------------------------------------------------------------------- #
def test_with_no_governed_name_the_label_is_unchanged():
    assert identity.portfolio_label("client_001") == "CLIENT_001"


def test_a_governed_name_becomes_the_label(tmp_path):
    _client_file(tmp_path / "clients", "acme_bank",
                 {"client": {"display_name": "Acme Bank plc"}})
    assert identity.portfolio_label("acme_bank") == "Acme Bank plc"


@pytest.mark.parametrize("body", ["", "   ", "{{{ not yaml", "- a\n- list\n"])
def test_a_malformed_client_file_falls_back_rather_than_raising(tmp_path, body):
    d = tmp_path / "clients" / "acme_bank"
    d.mkdir(parents=True)
    (d / "client.yaml").write_text(body, encoding="utf-8")
    identity.cache_clear()
    assert identity.portfolio_label("acme_bank") == "ACME_BANK"


def test_a_blank_governed_name_is_not_a_name(tmp_path):
    _client_file(tmp_path / "clients", "acme_bank", {"client": {"display_name": "   "}})
    assert identity.governed_client_name("acme_bank") is None
    assert identity.portfolio_label("acme_bank") == "ACME_BANK"


# --------------------------------------------------------------------------- #
# The blob discovery index no longer invents "platform"
# --------------------------------------------------------------------------- #
def test_the_blob_index_resolves_the_served_tenant(monkeypatch):
    from mi_agent_api import platform_snapshots_blob as blob
    monkeypatch.setenv("MI_AGENT_CLIENT_ID", "acme_bank")
    assert blob._served_client_id() == "acme_bank"


def test_the_blob_index_falls_back_to_the_same_default_as_every_other_surface(monkeypatch):
    """It used to read the raw env var and, unset, label the book "platform"
    while the rest of the API resolved a real client."""
    from mi_agent_api import dependencies
    from mi_agent_api import platform_snapshots_blob as blob
    monkeypatch.delenv("MI_AGENT_CLIENT_ID", raising=False)
    assert blob._served_client_id() == dependencies.default_tenant_id()
    assert blob._served_client_id() != "platform"


def test_a_client_file_in_the_wrong_directory_is_refused(tmp_path):
    """The directory keys the file, so a declared id that disagrees with it is a
    copy-paste error — and serving it would name one client from another's
    governed directory."""
    _client_file(tmp_path / "clients", "beta_bank",
                 {"client": {"client_id": "acme_bank", "display_name": "Acme Bank plc"}})
    assert identity.governed_client_name("beta_bank") is None
    assert identity.portfolio_label("beta_bank") == "BETA_BANK"


# --------------------------------------------------------------------------- #
# The committed estate
# --------------------------------------------------------------------------- #
def test_the_served_client_is_named_in_this_repository(monkeypatch):
    """The header defect in one assertion: client_001 renders as a name.

    Unlike every test above, this one reads the REAL config tree, so it fails if
    the governed identity file is removed and the header silently reverts to an
    identifier.
    """
    monkeypatch.setattr(identity, "_CLIENTS_ROOT",
                        _REPO_ROOT / "config" / "clients")
    identity.cache_clear()
    label = identity.portfolio_label("client_001")
    assert label == "ERE Funding - Equity Release Mortgages"
    assert label != "CLIENT_001" and label.lower() != "platform"
