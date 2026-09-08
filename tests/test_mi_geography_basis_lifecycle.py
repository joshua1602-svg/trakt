#!/usr/bin/env python3
"""tests/test_mi_geography_basis_lifecycle.py — the governed geography handoff.

The architecture this pins, deliberately mirroring
``tests/test_asset_class_lifecycle.py``, because it is the same handoff carrying
one more established fact:

    asset class (onboarding decides)
      -> governed default basis          (config/asset/mi_geography.yaml)
      -> portfolio_registry.yaml         (engine.onboarding_agent.portfolio_registry_writer)
      -> portfolio metadata overlay      (mi_agent.portfolio_metadata)
      -> the effective geography contract (mi_agent.mi_geography)

WHY IT HAS TO BE A CONFIGURED FACT
----------------------------------
A book carries a borrower geography and a collateral geography, and they are
different facts. "Balance by region" has no answer until somebody decides which
one this book reports on. Before this chain existed, three separate places each
answered by taking whichever geography column happened to be populated first —
in the preparation layer, in the harmonisation layer, and in the parser's fixed
preference order — which is not a semantics but the absence of one, and which
produced a borrower region column filled from the property's region.

WHAT IS DELIBERATELY *NOT* HERE
-------------------------------
No place name, postcode, ITL or NUTS code. This chain names a BASIS. Resolving a
region TERM onto a region VALUE stays with ``mi_agent.region_resolution``, and
harmonising region vocabularies stays with ``engine.region_taxonomy``.

And nothing regulatory. The Annex geography fields are projected from their own
sources under their own contract; an ND code in a regulatory geography field is
a declaration ("not collected"), never an absence for MI to fill. See
``tests/test_mi_geography_leaves_the_regime_alone.py``.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import yaml

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from engine.onboarding_agent import portfolio_registry_writer as writer
from mi_agent import mi_geography as geo


# =========================================================================== #
# 1. The governed default table
# =========================================================================== #
@pytest.mark.parametrize("asset_class,expected", [
    # Secured on real property — the security is what the book is about.
    ("equity_release", geo.BASIS_COLLATERAL),
    ("equity_release_mortgage", geo.BASIS_COLLATERAL),   # onboarding's dialect
    ("Lifetime Mortgage", geo.BASIS_COLLATERAL),
    ("residential_mortgage", geo.BASIS_COLLATERAL),
    ("residential_real_estate", geo.BASIS_COLLATERAL),
    ("rmbs", geo.BASIS_COLLATERAL),                      # via the synonym table
    ("commercial_real_estate", geo.BASIS_COLLATERAL),
    ("cre", geo.BASIS_COLLATERAL),
    # Movable or unsecured — the obligor is what the book is about.
    ("auto_finance", geo.BASIS_BORROWER),
    ("leasing", geo.BASIS_BORROWER),
    ("equipment_leasing", geo.BASIS_BORROWER),
    ("asset_finance", geo.BASIS_BORROWER),               # via the synonym table
])
def test_the_governed_default_for_an_asset_class(asset_class, expected):
    assert geo.default_primary_basis(asset_class) == expected


@pytest.mark.parametrize("asset_class", ["sme", "bridge", "unsecured_personal",
                                         "student_loan", "", None])
def test_an_asset_class_with_no_governed_default_is_not_given_one(asset_class):
    """MI would rather say it does not know than assume. An invented default is
    a silent wrong answer for every question that reaches it."""
    assert geo.default_primary_basis(asset_class) is None


def test_the_table_carries_no_geography_values():
    """Not a second mapping owner: this file names bases, never places."""
    text = (_REPO_ROOT / "config" / "asset" / "mi_geography.yaml").read_text()
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        assert ":" in stripped
        value = stripped.split(":", 1)[1].strip()
        assert value in ("", "1") or geo.normalise_basis(value), stripped


# =========================================================================== #
# 2. Onboarding publishes the basis it derived
# =========================================================================== #
def _read(path: Path):
    return yaml.safe_load(path.read_text(encoding="utf-8"))["portfolios"]


def test_onboarding_publishes_the_basis_for_the_class_it_decided(tmp_path):
    target = tmp_path / "portfolio_registry.yaml"
    writer.publish(target, [{"source_portfolio_id": "alderbridge"}],
                   asset_class="equity_release_mortgage")
    entry = _read(target)[0]
    assert entry["asset_class"] == "equity_release"
    assert entry[geo.REGISTRY_KEY] == {geo.REGISTRY_BASIS_KEY: geo.BASIS_COLLATERAL}


def test_each_portfolio_gets_the_basis_of_its_own_class(tmp_path):
    target = tmp_path / "portfolio_registry.yaml"
    writer.publish(target, [
        {"source_portfolio_id": "houses", "asset_class": "residential_mortgage"},
        {"source_portfolio_id": "cars", "asset_class": "auto_finance"},
    ], asset_class="equity_release")
    by_id = {e["source_portfolio_id"]: e for e in _read(target)}
    assert by_id["houses"][geo.REGISTRY_KEY][geo.REGISTRY_BASIS_KEY] == geo.BASIS_COLLATERAL
    assert by_id["cars"][geo.REGISTRY_KEY][geo.REGISTRY_BASIS_KEY] == geo.BASIS_BORROWER


def test_an_unknown_asset_class_publishes_no_basis(tmp_path):
    """The existing operator-review path, reused. A portfolio onboarding could
    not classify is written without an asset class today; it is now written
    without a geography basis for the same reason and in the same way."""
    target = tmp_path / "portfolio_registry.yaml"
    writer.publish(target, [{"source_portfolio_id": "mystery"}])
    entry = _read(target)[0]
    assert "asset_class" not in entry
    assert geo.REGISTRY_KEY not in entry


def test_a_classified_book_with_no_governed_default_publishes_no_basis(tmp_path):
    """An asset class Trakt knows but has not settled a geography basis for is
    NOT given one silently."""
    target = tmp_path / "portfolio_registry.yaml"
    writer.publish(target, [{"source_portfolio_id": "smebook"}], asset_class="sme_loan")
    entry = _read(target)[0]
    assert entry["asset_class"] == "sme"
    assert geo.REGISTRY_KEY not in entry


def test_publishing_preserves_a_hand_authored_override(tmp_path):
    """An operator who overrode the asset default keeps their decision through a
    re-run of onboarding. The writer owns the key, so it must not clobber a
    deliberate override with the class default — it merges onto what is there."""
    target = tmp_path / "portfolio_registry.yaml"
    target.write_text(yaml.safe_dump({"portfolios": [{
        "source_portfolio_id": "btl",
        "asset_class": "residential_mortgage",
        geo.REGISTRY_KEY: {geo.REGISTRY_BASIS_KEY: geo.BASIS_BORROWER,
                           "set_by": "operator"},
        "runoff_profile_id": "seller_a_2026",
    }]}), encoding="utf-8")
    writer.publish(target, [{"source_portfolio_id": "btl"}],
                   asset_class="residential_mortgage")
    entry = _read(target)[0]
    assert entry["runoff_profile_id"] == "seller_a_2026"
    assert entry[geo.REGISTRY_KEY][geo.REGISTRY_BASIS_KEY] == geo.BASIS_BORROWER
    assert entry[geo.REGISTRY_KEY]["set_by"] == "operator"


# =========================================================================== #
# 3. The approved onboarding config states the same basis
# =========================================================================== #
@pytest.mark.parametrize("asset_class,expected", [
    ("equity_release_mortgage", geo.BASIS_COLLATERAL),
    ("residential_mortgage", geo.BASIS_COLLATERAL),
    ("auto_finance", geo.BASIS_BORROWER),
])
def test_the_approved_config_records_the_basis_onboarding_established(
        asset_class, expected):
    """It used to record a hard-coded display column and no basis at all, so the
    decision existed nowhere an operator reviewing the pack could see it."""
    from engine.onboarding_agent.answer_ingestion import _mi_geography_policy

    policy = _mi_geography_policy(asset_class)
    assert policy["primary_basis"] == expected
    # The display column is a presentation choice, not the basis, and is kept.
    assert policy["region_display_field"] == "collateral_geography"


def test_the_approved_config_states_no_basis_it_could_not_establish():
    from engine.onboarding_agent.answer_ingestion import _mi_geography_policy

    assert "primary_basis" not in _mi_geography_policy("sme_loan")
    assert "primary_basis" not in _mi_geography_policy(None)


def test_the_config_and_the_registry_cannot_state_different_bases(tmp_path):
    """Two writers, one table. The approved pack an operator signs off and the
    registry the runtime reads are derived from the same governed default."""
    from engine.onboarding_agent.answer_ingestion import _mi_geography_policy

    for asset_class in ("equity_release", "residential_mortgage", "auto_finance",
                        "equipment_leasing", "commercial_real_estate"):
        target = tmp_path / f"{asset_class}.yaml"
        writer.publish(target, [{"source_portfolio_id": "p"}],
                       asset_class=asset_class)
        registry_basis = _read(target)[0][geo.REGISTRY_KEY][geo.REGISTRY_BASIS_KEY]
        assert registry_basis == _mi_geography_policy(asset_class)["primary_basis"]


# =========================================================================== #
# 4. MI reads it back
# =========================================================================== #
def test_the_overlay_carries_the_basis_to_mi(tmp_path, monkeypatch):
    from mi_agent.portfolio_metadata import ENV_REGISTRY_PATH, load_portfolio_metadata

    target = tmp_path / "portfolio_registry.yaml"
    writer.publish(target, [{"source_portfolio_id": "cars"}], asset_class="auto_finance")
    monkeypatch.setenv(ENV_REGISTRY_PATH, str(target))
    overlay = load_portfolio_metadata()
    assert geo.configured_basis(overlay["cars"]) == geo.BASIS_BORROWER


def test_the_portfolio_declaration_beats_the_asset_class_default():
    entry = {"asset_class": "equity_release",
             geo.REGISTRY_KEY: {geo.REGISTRY_BASIS_KEY: "obligor"}}
    contract = geo.resolve_contract(asset_class="equity_release",
                                    registry_entry=entry)
    assert contract.primary_basis == geo.BASIS_BORROWER
    assert contract.source == geo.SOURCE_PORTFOLIO


def test_a_flat_declaration_is_read_too():
    """A human hand-authoring the registry writes the short form."""
    assert geo.configured_basis({geo.REGISTRY_KEY: "collateral"}) == geo.BASIS_COLLATERAL


def test_the_asset_default_is_used_when_the_portfolio_declares_nothing():
    contract = geo.resolve_contract(asset_class="lifetime_mortgage")
    assert contract.primary_basis == geo.BASIS_COLLATERAL
    assert contract.source == geo.SOURCE_ASSET_DEFAULT
    assert contract.asset_class == "equity_release"


def test_an_unconfigured_book_has_no_primary_basis():
    contract = geo.resolve_contract(asset_class="sme")
    assert contract.primary_basis is None
    assert contract.source == geo.SOURCE_NONE


def test_the_contract_is_publishable_as_receipt_provenance():
    contract = geo.resolve_contract(asset_class="equity_release")
    payload = contract.to_dict()
    assert payload["primaryBasis"] == geo.BASIS_COLLATERAL
    assert payload["basisSource"] == geo.SOURCE_ASSET_DEFAULT
    assert payload["assetClass"] == "equity_release"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
