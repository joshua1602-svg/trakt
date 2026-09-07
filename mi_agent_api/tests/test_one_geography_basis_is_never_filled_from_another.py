#!/usr/bin/env python3
"""A borrower geography is not a collateral geography, and neither fills the other.

THE DEFECT THIS PINS
--------------------
``funded_prep._coalesce_group_dimensions`` treated the region columns as
interchangeable deliveries of one concept::

    "geographic_region_obligor": {
        "kind": "group", "primary": "geographic_region_obligor",
        "sources": ["geographic_region_obligor", "geographic_region_collateral",
                    "collateral_geography"]},

so a row whose OBLIGOR geography was never collected had it gap-filled, row by
row, from the COLLATERAL geography. Measured on a three-row frame carrying a
Welsh property and no obligor region, the borrower column came back "Wales".

That is a silent wrong answer of the worst kind. "Balance by borrower region"
then answers with where the houses are, labelled as where the borrowers are, and
nothing in the answer tells the reader which they got. Where the obligor column
holds an ESMA no-data code the error is sharper still: ND1 means "not
collected", a regulatory DECLARATION, and overwriting it with the collateral
region converts a declaration into a fabricated fact.

The two geographies are different facts about the loan. Which one a book reports
on generically is a decision about the ASSET — see :mod:`mi_agent.mi_geography`
— not something to settle by taking whichever column was populated first.

WHAT STAYS
----------
Coalescing WITHIN one basis is still right and still happens: a book delivering
its collateral region as ``geographic_region_collateral`` and another delivering
it as ``collateral_geography`` are describing the same thing at different
granularities, and a combined tape must read as one book. Only the crossing of
the borrower/collateral boundary is retired.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from mi_agent import mi_geography as geo
from mi_agent_api.funded_prep import (
    augment_platform_canonical_dimensions, prepare_funded_mi_dataset)


def _frame(**columns) -> pd.DataFrame:
    n = len(next(iter(columns.values())))
    base = {
        "loan_identifier": [f"L{i:03d}" for i in range(n)],
        "current_outstanding_balance": [100000.0 + i * 1000 for i in range(n)],
        "current_valuation_amount": [250000.0 + i * 1000 for i in range(n)],
        "origination_date": ["2020-06-15"] * n,
        "reporting_date": ["2026-06-30"] * n,
    }
    base.update(columns)
    return pd.DataFrame(base)


# --------------------------------------------------------------------------- #
# The defect itself, on both preparation paths.
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("prepare", [
    pytest.param(lambda d: augment_platform_canonical_dimensions(d)[0], id="serving"),
    pytest.param(lambda d: prepare_funded_mi_dataset(d)[0], id="historical"),
])
def test_an_uncollected_borrower_region_is_not_filled_from_the_property(prepare):
    """The exact falsification: property in Wales, obligor region never collected."""
    prepared = prepare(_frame(
        geographic_region_obligor=["TLC31", "", ""],
        collateral_geography=["Wales", "Wales", "Scotland"]))
    obligor = prepared["geographic_region_obligor"].astype(str).str.strip()
    # Row 0 keeps the value the book supplied; rows 1 and 2 stay empty. They do
    # NOT become "Wales" and "Scotland".
    assert obligor.iloc[0] == "TLC31"
    assert obligor.iloc[1] in ("", "nan", "None", "<NA>")
    assert obligor.iloc[2] in ("", "nan", "None", "<NA>")
    # And the collateral column is untouched either way.
    assert list(prepared["collateral_geography"]) == ["Wales", "Wales", "Scotland"]


@pytest.mark.parametrize("prepare", [
    pytest.param(lambda d: augment_platform_canonical_dimensions(d)[0], id="serving"),
    pytest.param(lambda d: prepare_funded_mi_dataset(d)[0], id="historical"),
])
def test_a_regulatory_no_data_declaration_is_not_overwritten(prepare):
    """ND1 means "not collected". It is a declaration, not a gap to fill."""
    prepared = prepare(_frame(
        geographic_region_obligor=["ND1", "ND1", "TLC31"],
        geographic_region_collateral=["TLI35", "TLI35", "TLC31"],
        collateral_geography=["Wales", "Wales", "Wales"]))
    assert list(prepared["geographic_region_obligor"].astype(str)) == [
        "ND1", "ND1", "TLC31"]


@pytest.mark.parametrize("prepare", [
    pytest.param(lambda d: augment_platform_canonical_dimensions(d)[0], id="serving"),
    pytest.param(lambda d: prepare_funded_mi_dataset(d)[0], id="historical"),
])
def test_a_collateral_region_is_still_coalesced_within_its_own_tier(prepare):
    """The capability that was right is kept: same basis, same granularity,
    different delivery name."""
    prepared = prepare(_frame(
        collateral_geography=["Wales", "", ""],
        property_region=["Wales", "Scotland", ""]))
    filled = prepared["collateral_geography"].astype(str).str.strip()
    assert filled.iloc[0] == "Wales"
    assert filled.iloc[1] == "Scotland"      # from the other reporting-tier name
    assert filled.iloc[2] in ("", "nan", "None", "<NA>")


@pytest.mark.parametrize("prepare", [
    pytest.param(lambda d: augment_platform_canonical_dimensions(d)[0], id="serving"),
    pytest.param(lambda d: prepare_funded_mi_dataset(d)[0], id="historical"),
])
def test_a_column_of_region_names_is_not_filled_with_region_codes(prepare):
    """Granularity does not gap-fill across itself.

    ``collateral_geography`` holds readable names; ``geographic_region_collateral``
    holds ITL3 codes. Both are the collateral basis, so it is tempting to let one
    fill the other — and the result is a single column speaking two vocabularies,
    "London" beside "TLC31", which splits a regional breakdown into categories no
    reader can reconcile. The granularity is chosen once per book instead, by
    ``mi_geography.field_for_basis``.
    """
    prepared = prepare(_frame(
        collateral_geography=["Wales", "", ""],
        geographic_region_collateral=["TLC31", "TLI35", ""],
        geographic_region_collateral_itl3=["TLC31", "TLI35", "TLH37"]))
    names = prepared["collateral_geography"].astype(str).str.strip()
    assert names.iloc[0] == "Wales"
    assert names.iloc[1] in ("", "nan", "None", "<NA>")
    # The code tier coalesces within itself, so the codes column is complete.
    codes = prepared["geographic_region_collateral"].astype(str).str.strip()
    assert list(codes) == ["TLC31", "TLI35", "TLH37"]


def test_the_borrower_family_coalesces_within_itself_too():
    prepared, _ = augment_platform_canonical_dimensions(_frame(
        geographic_region_obligor=["TLC31", ""],
        geographic_region_obligor_itl3=["TLC31", "TLH37"]))
    obligor = prepared["geographic_region_obligor"].astype(str).str.strip()
    assert list(obligor) == ["TLC31", "TLH37"]


def test_the_book_chooses_the_most_readable_tier_it_populates():
    """A book carrying both gets names; a book carrying only codes gets codes —
    and neither gets a mixture."""
    both = _frame(collateral_geography=["Wales", "London"],
                  geographic_region_collateral=["TLC31", "TLH37"])
    codes_only = _frame(collateral_geography=["", ""],
                        geographic_region_collateral=["TLC31", "TLH37"])
    assert geo.field_for_basis("collateral", frame=both) == "collateral_geography"
    assert geo.field_for_basis(
        "collateral", frame=codes_only) == "geographic_region_collateral"


# --------------------------------------------------------------------------- #
# The topology this depends on: every region column belongs to one basis or to
# none, and the two families never share a member.
# --------------------------------------------------------------------------- #
def test_no_column_carries_both_bases():
    borrower = set(geo.BASIS_FIELDS[geo.BASIS_BORROWER])
    collateral = set(geo.BASIS_FIELDS[geo.BASIS_COLLATERAL])
    assert borrower and collateral
    assert not (borrower & collateral)


def test_the_harmonised_columns_belong_to_no_basis():
    """They are derived from whichever source was populated first, so they
    cannot answer a question that is about one basis."""
    for field in ("canonical_region_reporting", "canonical_region_detail"):
        assert geo.basis_of_field(field) is None


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
