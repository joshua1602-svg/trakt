#!/usr/bin/env python3
"""The MI geography basis resolves down the governed configuration hierarchy.

    explicit query
        >  explicit client / portfolio override
        >  asset-class default
        >  unresolved

WHAT THIS REPLACED, and why it matters more than it looks. The basis used to be
established by the portfolio registry ALONE, keyed by ``source_portfolio_id``.
A deployment whose client configuration declared ``portfolio.asset_class:
equity_release`` — a complete, governed statement of what kind of book it runs —
still reported ``basisSource: unconfigured``, because a second file listing its
portfolios by id happened not to exist. Generic "region" then fell back to a
fixed field order that no layer had chosen, which is the thing this whole
architecture exists to stop.

The asset layer already says what region means for a kind of book; the client
layer already says what kind of book this client runs. Joining them needs no new
configuration and no per-portfolio entry. The registry keeps what it genuinely
owns — per-portfolio metadata, and a per-portfolio EXCEPTION — and loses only the
burden of being mandatory.

The layers, and who owns what:

    config/asset/mi_geography.yaml     asset_class -> default basis
    config/client/config_client_*.yaml which asset_class this client is
    portfolio registry                 per-portfolio exception (optional)
    the question itself                a basis stated outright
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import yaml

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from mi_agent import mi_geography as geo
from mi_agent.portfolio_metadata import ENV_REGISTRY_PATH
from mi_agent_api.currency import ENV_CLIENT_CONFIG


@pytest.fixture()
def layers(tmp_path, monkeypatch):
    """Write the governed layers a deployment may have, and point MI at them.

    Returns a callable taking ``client=<dict|None>`` and ``registry=<list|None>``
    so each test states exactly which layers exist — the absence of a layer is
    as much a case under test as its contents.
    """
    def _install(client=None, registry=None):
        monkeypatch.delenv(ENV_CLIENT_CONFIG, raising=False)
        monkeypatch.delenv(ENV_REGISTRY_PATH, raising=False)
        if client is not None:
            path = tmp_path / "config_client_TEST.yaml"
            path.write_text(yaml.safe_dump(client), encoding="utf-8")
            monkeypatch.setenv(ENV_CLIENT_CONFIG, str(path))
        if registry is not None:
            path = tmp_path / "portfolio_registry.yaml"
            path.write_text(yaml.safe_dump({"portfolios": registry}),
                            encoding="utf-8")
            monkeypatch.setenv(ENV_REGISTRY_PATH, str(path))
        return geo.contract_for_scope(client_id="TEST")
    return _install


# =========================================================================== #
# A. The asset default, from the client layer, with no registry at all
# =========================================================================== #
def test_a_client_declaring_its_asset_class_needs_no_registry(layers):
    """THE CASE THE OLD RESOLVER FAILED. This is the ordinary deployment."""
    contract = layers(client={"portfolio": {"asset_class": "equity_release"}})
    assert contract.primary_basis == geo.BASIS_COLLATERAL
    assert contract.source == geo.SOURCE_ASSET_DEFAULT
    assert contract.asset_class == "equity_release"


def test_the_onboarding_dialect_resolves_the_same_way(layers):
    """Onboarding writes `equity_release_mortgage`; the boundary normalises."""
    contract = layers(client={"portfolio": {"asset_class": "equity_release_mortgage"}})
    assert contract.primary_basis == geo.BASIS_COLLATERAL
    assert contract.source == geo.SOURCE_ASSET_DEFAULT


def test_a_flat_client_declaration_is_read_too(layers):
    contract = layers(client={"asset_class": "residential_mortgage"})
    assert contract.primary_basis == geo.BASIS_COLLATERAL
    assert contract.source == geo.SOURCE_ASSET_DEFAULT


# =========================================================================== #
# B. A different asset class means a different basis
# =========================================================================== #
def test_an_auto_book_reports_on_its_borrowers(layers):
    contract = layers(client={"portfolio": {"asset_class": "auto_finance"}})
    assert contract.primary_basis == geo.BASIS_BORROWER
    assert contract.source == geo.SOURCE_ASSET_DEFAULT


def test_the_same_configuration_shape_gives_two_answers(layers):
    """The architecture in one assertion: identical config shape, different
    asset class, correctly different basis."""
    collateral = layers(client={"portfolio": {"asset_class": "commercial_real_estate"}})
    borrower = layers(client={"portfolio": {"asset_class": "equipment_leasing"}})
    assert collateral.primary_basis == geo.BASIS_COLLATERAL
    assert borrower.primary_basis == geo.BASIS_BORROWER


# =========================================================================== #
# C. An explicit query basis outranks every configured layer
# =========================================================================== #
@pytest.mark.parametrize("question,expected", [
    ("Total balance by borrower region", geo.BASIS_BORROWER),
    ("Total balance by obligor region", geo.BASIS_BORROWER),
    ("Total balance by property region", geo.BASIS_COLLATERAL),
    ("Total balance by collateral region", geo.BASIS_COLLATERAL),
])
def test_a_stated_basis_wins_over_the_asset_default(layers, question, expected):
    contract = layers(client={"portfolio": {"asset_class": "equity_release"}})
    effective = contract.effective_for(question)
    assert effective["primaryBasis"] == expected
    assert effective["basisSource"] == geo.SOURCE_QUESTION
    # And the configured contract stays visible beside it, so a reader can see
    # both that the answer was steered and what it would otherwise have used.
    assert effective["configuredBasis"] == geo.BASIS_COLLATERAL
    assert effective["configuredBasisSource"] == geo.SOURCE_ASSET_DEFAULT


def test_a_question_naming_no_basis_reports_the_configured_one(layers):
    contract = layers(client={"portfolio": {"asset_class": "equity_release"}})
    effective = contract.effective_for("Total balance by region")
    assert effective["primaryBasis"] == geo.BASIS_COLLATERAL
    assert effective["basisSource"] == geo.SOURCE_ASSET_DEFAULT
    assert "configuredBasis" not in effective


# =========================================================================== #
# D. An explicit CLIENT override outranks the asset default
# =========================================================================== #
def test_a_client_may_override_its_asset_default(layers):
    """A buy-to-let book run as a credit exposure rather than a property
    portfolio. The exception is declared where the client is described."""
    contract = layers(client={"portfolio": {
        "asset_class": "residential_mortgage",
        "mi_geography": {"primary_basis": "borrower"}}})
    assert contract.primary_basis == geo.BASIS_BORROWER
    assert contract.source == geo.SOURCE_CLIENT
    assert contract.asset_class == "residential_mortgage"


# =========================================================================== #
# E. An asset class with no governed default is not given one
# =========================================================================== #
@pytest.mark.parametrize("asset_class", ["sme", "bridge", "unsecured_personal"])
def test_an_unsettled_asset_class_stays_unresolved(layers, asset_class):
    contract = layers(client={"portfolio": {"asset_class": asset_class}})
    assert contract.primary_basis is None
    assert contract.source == geo.SOURCE_NONE
    # The class IS reported, so an operator can see what was not settled.
    assert contract.asset_class == asset_class


def test_a_client_declaring_nothing_stays_unresolved(layers):
    contract = layers(client={"portfolio": {"country": "GB"}})
    assert contract.primary_basis is None
    assert contract.source == geo.SOURCE_NONE


def test_no_configuration_at_all_stays_unresolved(layers):
    contract = layers()
    assert contract.primary_basis is None
    assert contract.source == geo.SOURCE_NONE


# =========================================================================== #
# F + G. The registry is an OPTIONAL override, never a precondition
# =========================================================================== #
def test_the_registry_being_absent_does_not_disable_the_asset_default(layers):
    """Stated as its own case because it is the regression that mattered: the
    file's absence used to be the whole reason a governed deployment reported
    `unconfigured`."""
    contract = layers(client={"portfolio": {"asset_class": "equity_release"}},
                      registry=None)
    assert contract.source == geo.SOURCE_ASSET_DEFAULT
    assert contract.primary_basis == geo.BASIS_COLLATERAL


def test_a_registry_override_wins_over_the_asset_default(layers):
    contract = layers(
        client={"portfolio": {"asset_class": "equity_release"}},
        registry=[{"source_portfolio_id": "book_a", "asset_class": "equity_release",
                   geo.REGISTRY_KEY: {geo.REGISTRY_BASIS_KEY: "borrower"}}])
    assert contract.primary_basis == geo.BASIS_BORROWER
    assert contract.source == geo.SOURCE_PORTFOLIO


def test_a_registry_override_wins_over_a_client_override(layers):
    """Most specific wins: the book's own exception beats the client's."""
    contract = layers(
        client={"portfolio": {"asset_class": "equity_release",
                              "mi_geography": {"primary_basis": "borrower"}}},
        registry=[{"source_portfolio_id": "book_a",
                   geo.REGISTRY_KEY: {geo.REGISTRY_BASIS_KEY: "collateral"}}])
    assert contract.primary_basis == geo.BASIS_COLLATERAL
    assert contract.source == geo.SOURCE_PORTFOLIO


def test_a_registry_carrying_only_an_asset_class_still_gives_the_default(layers):
    """A registry entry that names the book's class but declares no exception
    is not an override; it is the asset default arrived at more specifically."""
    contract = layers(
        client=None,
        registry=[{"source_portfolio_id": "book_a", "asset_class": "auto_finance"}])
    assert contract.primary_basis == geo.BASIS_BORROWER
    assert contract.source == geo.SOURCE_ASSET_DEFAULT


def test_books_declaring_different_bases_resolve_to_none(layers):
    """A mixed scope has no single meaning for "balance by region", and
    inventing one would put two different facts in one column."""
    contract = layers(
        client={"portfolio": {"asset_class": "equity_release"}},
        registry=[{"source_portfolio_id": "houses",
                   geo.REGISTRY_KEY: {geo.REGISTRY_BASIS_KEY: "collateral"}},
                  {"source_portfolio_id": "cars",
                   geo.REGISTRY_KEY: {geo.REGISTRY_BASIS_KEY: "borrower"}}])
    assert contract.primary_basis is None
    assert contract.source == geo.SOURCE_NONE


def test_books_of_different_asset_classes_resolve_to_none(layers):
    contract = layers(
        client=None,
        registry=[{"source_portfolio_id": "houses", "asset_class": "equity_release"},
                  {"source_portfolio_id": "cars", "asset_class": "auto_finance"}])
    assert contract.primary_basis is None
    assert contract.source == geo.SOURCE_NONE


# =========================================================================== #
# The shipped ERE configuration, end to end
# =========================================================================== #
def test_the_shipped_ere_client_configuration_resolves_to_collateral(monkeypatch):
    """Not a fixture: the file this estate actually ships for ERE."""
    monkeypatch.delenv(ENV_REGISTRY_PATH, raising=False)
    monkeypatch.setenv(ENV_CLIENT_CONFIG,
                       str(_REPO_ROOT / "config/client/config_client_ERM_UK.yaml"))
    contract = geo.contract_for_scope(client_id="ERE")
    assert contract.asset_class == "equity_release"
    assert contract.primary_basis == geo.BASIS_COLLATERAL
    assert contract.source == geo.SOURCE_ASSET_DEFAULT


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))


# =========================================================================== #
# OCC alignment: onboarding already writes the key MI reads
# =========================================================================== #
def test_occ_writes_the_asset_class_where_mi_reads_it():
    """PHASE 5, asserted rather than assumed.

    OCC captures the asset class as a REQUIRED, operator-confirmed portfolio
    field and routes it to ``client_config:portfolio.asset_class``. That is the
    same key ``portfolio_metadata.client_asset_class`` reads, in the same file
    ``operations_control.onboarding.artefacts.client_config_rel`` writes. So a
    client that has been through onboarding gets its geography default with no
    second artefact and nothing repeating `equity_release -> collateral`.

    If this ever drifts, the symptom is silent: geography quietly falls back to
    unresolved for every newly onboarded client while every test using a
    hand-written config keeps passing.
    """
    from mi_agent.portfolio_metadata import _CLIENT_ASSET_CLASS_PATHS
    from operations_control.onboarding.artefacts import client_config_rel

    catalogue = yaml.safe_load(
        (_REPO_ROOT / "config/onboarding/field_catalogue.yaml").read_text())
    fields = [f for section in catalogue.get("sections", [])
              if section.get("key") == "portfolios"
              for f in (section.get("fields") or [])]
    asset = next((f for f in fields if f.get("key") == "asset_class"), None)
    assert asset is not None, "OCC no longer captures the asset class"
    assert asset["required"] is True
    assert asset["writes_to"] == "client_config:portfolio.asset_class"

    # The target OCC writes is the first place MI looks.
    target = tuple(asset["writes_to"].split(":", 1)[1].split("."))
    assert _CLIENT_ASSET_CLASS_PATHS[0] == target

    # And into the file MI's locator resolves for that client.
    assert client_config_rel("ERE") == "config/client/config_client_ERE.yaml"


def test_occ_does_not_need_a_second_geography_artefact():
    """The asset default is derived, never restated. OCC persists WHICH asset
    class a client is; `config/asset/mi_geography.yaml` owns what that means.
    A per-client copy of that mapping would be a second source able to drift."""
    catalogue = (_REPO_ROOT / "config/onboarding/field_catalogue.yaml").read_text()
    assert "mi_geography" not in catalogue
