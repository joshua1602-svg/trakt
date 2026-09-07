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
                       str(_REPO_ROOT / "config/client/config_client_ERE.yaml"))
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


# =========================================================================== #
# Config OWNERSHIP: ERE is the client, ERM is the asset
# =========================================================================== #
#
# These two were conflated in a filename and it cost a live book its
# configuration. `config/client/config_client_ERM_UK.yaml` named the CLIENT
# layer after the ASSET, so `mi_agent_api.currency.client_config_path`, which
# resolves `config/client/config_client_{client_id}.yaml`, found nothing for the
# client the platform actually runs — ERE — and every client-layer fact came
# back unconfigured. The fix is ownership, not an alias: the client file is
# named for the client, the asset pack owns the asset's behaviour, and the
# effective configuration composes one under the other.


def test_the_client_layer_is_named_for_the_client():
    """The locator's own arithmetic, against the shipped file.

    `client_config_path` builds the path from the client id. Asserting the file
    exists AT THAT PATH is the whole defect: nothing about the lookup was ever
    wrong, and an alias from ERM_UK to ERE would have left two names for one
    client and no rule about which is right.
    """
    from mi_agent_api.currency import client_config_path

    location = client_config_path("ERE")
    assert location is not None, "ERE has no governed client configuration"
    assert Path(location).name == "config_client_ERE.yaml"
    assert Path(location).exists()


def test_ere_resolves_to_collateral_with_no_registry_and_no_environment(
        monkeypatch):
    """The live case, with every escape hatch closed.

    No ``TRAKT_MI_CLIENT_CONFIG`` pointing at the file by hand, no portfolio
    registry entry, no explicit asset class passed in: just the client id the
    live portfolio ``ERE/2026-06-30`` splits to. This is the assertion that was
    impossible before the rename, and it is the one that matters — the
    deployment sets neither variable.
    """
    monkeypatch.delenv(ENV_CLIENT_CONFIG, raising=False)
    monkeypatch.delenv(ENV_REGISTRY_PATH, raising=False)
    contract = geo.contract_for_scope(client_id="ERE")
    assert contract.asset_class == "equity_release"
    assert contract.primary_basis == geo.BASIS_COLLATERAL
    assert contract.source == geo.SOURCE_ASSET_DEFAULT


def test_the_erm_pack_owns_the_equity_release_geography():
    """The asset's behaviour is declared in the asset's own configuration.

    Which geography an equity-release book reports on is a fact about equity
    release, so it sits beside the rest of what equity release does rather than
    in a shared table keyed by asset class. The pack is discovered by the class
    IT declares, so nothing hard-codes `ERM -> equity_release` twice.
    """
    pack = yaml.safe_load(
        (_REPO_ROOT / "config/asset/product_defaults_ERM.yaml").read_text())
    assert pack["asset_class"] == "equity_release"
    assert pack["mi_geography"]["primary_basis"] == geo.BASIS_COLLATERAL
    assert geo.declaring_sources("equity_release") == ("asset_pack",)


def test_no_asset_class_declares_its_basis_in_two_places():
    """Two sources for one decision are two sources able to drift.

    A class with a pack declares its basis there; a class without one is
    declared in the shared table. Never both — and this is the check that keeps
    it that way, because the runtime preference (pack first) would otherwise
    hide the duplicate until the two disagreed.
    """
    packs = geo._load_pack_defaults(geo.asset_pack_dir())
    table = geo._load_defaults()
    assert packs, "no asset pack declares a geography basis"
    overlap = sorted(set(packs) & set(table))
    assert overlap == [], (
        f"declared in both the asset pack and config/asset/mi_geography.yaml: "
        f"{overlap}")
    for asset_class in sorted(set(packs) | set(table)):
        assert len(geo.declaring_sources(asset_class)) == 1


def test_the_pack_the_orchestrator_runs_is_the_pack_geography_reads():
    """One class-to-pack mapping, agreed by every module that needs one.

    The orchestrator picks an asset pack to RUN with; OCC picks one to COMPOSE
    under the client; MI reads one to learn what region means. If those three
    ever named different files for the same asset class, MI would answer from a
    pack the pipeline never used.
    """
    from engine.orchestrator.trakt_run import ASSET_PACKS
    from operations_control.configuration.packages import ASSET_MODEL

    for asset_class, pack in ASSET_PACKS.items():
        model = ASSET_MODEL.get(asset_class)
        assert model is not None, f"{asset_class} is not a configured asset"
        assert Path(model["pack"]).name == Path(pack).name
        declared = yaml.safe_load(Path(pack).read_text())["asset_class"]
        assert declared == asset_class, (
            f"{Path(pack).name} declares {declared!r}, but the orchestrator "
            f"runs it for {asset_class!r}")


def test_the_client_declares_its_asset_exactly_once():
    """ERE says which asset it runs, in the one key OCC writes.

    The class is what selects the pack, so restating it — a second key here, a
    pack path spelled out by hand — would be a second place to change when a
    book's asset class changes, and a second place to get it wrong.
    """
    text = (_REPO_ROOT / "config/client/config_client_ERE.yaml").read_text()
    doc = yaml.safe_load(text)
    assert doc["portfolio"]["asset_class"] == "equity_release"

    from operations_control.configuration.packages import ASSET_MODEL
    pack = ASSET_MODEL[doc["portfolio"]["asset_class"]]["pack"]
    assert Path(_REPO_ROOT / pack).exists()

    # The pack is named by the model, not by the client file.
    body = "\n".join(line for line in text.splitlines()
                     if not line.lstrip().startswith("#"))
    assert "product_defaults" not in body
    assert body.count("asset_class") == 1

    # And the client states no geography of its own: it has no reason to
    # override what its asset already decided.
    assert "mi_geography" not in doc
