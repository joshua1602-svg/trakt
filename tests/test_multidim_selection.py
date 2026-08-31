#!/usr/bin/env python3
"""tests/test_multidim_selection.py

Which crossings a book actually supports.

``snapshots.cross_tab`` has always been generic over the eleven governed
stratification dimensions, but the pack drew a fixed three — LTV against
borrower age, borrower type and region — on every book, whatever that book
could support. Selection is now a property of the book.

The rule is deliberately about SHAPE, never about asset class: a pair whose
dimensions a tape cannot supply does not resolve, and a crossing with one
category on a side, or too few populated cells, or nothing new to say, is
recorded with the reason it lost.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

pd = pytest.importorskip("pandas")

from mi_agent_api import snapshots as S  # noqa: E402

_REGIONS = ("London", "South East", "Wales", "Scotland")
_PRODUCTS = ("Lifetime Mortgage", "Retirement Interest Only")


def book(n=400, *, one_region=False, one_product=False):
    """A funded frame carrying the columns the governed dimensions read."""
    rows = []
    for i in range(n):
        ltv = 22.0 + (i * 7) % 60
        rows.append({
            "unique_identifier": f"L{i:05d}",
            "current_outstanding_balance": 100_000.0 + (i * 3_700) % 400_000,
            "current_valuation_amount": 900_000.0,
            "current_loan_to_value": ltv,
            "current_interest_rate": 5.0 + (i % 5) * 0.7,
            "youngest_borrower_age": 58 + (i * 3) % 30,
            "geographic_region_collateral": ("London" if one_region
                                             else _REGIONS[i % len(_REGIONS)]),
            "collateral_geography": ("London" if one_region
                                     else _REGIONS[i % len(_REGIONS)]),
            "product_type": (_PRODUCTS[0] if one_product
                             else _PRODUCTS[i % len(_PRODUCTS)]),
            "origination_date": f"202{2 + i % 4}-06-01",
            "data_cut_off_date": "2026-06-30",
        })
    df = pd.DataFrame(rows)
    from mi_agent_api.funded_prep import prepare_funded_mi_dataset
    out, _report = prepare_funded_mi_dataset(df)
    return out


@pytest.fixture(scope="module")
def funded():
    return book()


# --------------------------------------------------------------------------- #
# The library is wider than the three that used to be drawn.
# --------------------------------------------------------------------------- #

def test_more_than_one_valid_pair_can_be_selected(funded):
    """Catches: a pack that draws LTV × age and LTV × region and stops."""
    chosen = S.select_multidim_pairs(funded, want=4)
    assert len(chosen["selected"]) >= 3, chosen["selected"].keys()


def test_the_candidate_library_spans_more_than_the_historical_three():
    keys = {k for k, _x, _y in S.MULTIDIM_CANDIDATE_PAIRS}
    assert len(keys) >= 8
    # And it is built from the generic dimension set, not a private list.
    dims = {d for _k, x, y in S.MULTIDIM_CANDIDATE_PAIRS for d in (x, y)}
    assert dims <= set(S.DIMENSION_NAMES)


def test_a_pair_label_comes_from_the_shared_dimension_names():
    assert S.pair_label("ltv", "ticket") == "Balance by LTV x ticket size"


# --------------------------------------------------------------------------- #
# Exclusion.
# --------------------------------------------------------------------------- #

def test_a_single_category_side_is_not_a_crossing():
    """Catches: a "crossing" that is a stratification drawn as a grid."""
    chosen = S.select_multidim_pairs(book(one_region=True), want=6)
    reasons = {r["key"]: r["reason"] for r in chosen["rejected"]}
    assert "ltv_region" not in chosen["selected"]
    assert "single" in reasons.get("ltv_region", ""), reasons


def test_a_crossing_that_repeats_one_above_it_is_dropped(funded):
    """Both dimensions already crossed on the page means it tells that story
    again."""
    # want is set past the candidate list so the cap cannot mask the rule:
    # every pair is considered and only redundancy can reject these.
    chosen = S.select_multidim_pairs(
        funded, want=len(S.MULTIDIM_CANDIDATE_PAIRS))
    reasons = [r["reason"] for r in chosen["rejected"]]
    assert any("repeats a story" in r for r in reasons), reasons
    # ticket × age is the case: both sides are already crossed against LTV.
    assert "ticket_age" not in chosen["selected"]


def test_nothing_is_dropped_without_a_reason(funded):
    chosen = S.select_multidim_pairs(funded, want=2)
    assert chosen["rejected"]
    assert all(r.get("reason") for r in chosen["rejected"])
    assert all(r.get("label") for r in chosen["rejected"])


def test_selection_is_deterministic(funded):
    first = list(S.select_multidim_pairs(funded, want=4)["selected"])
    for _ in range(4):
        assert list(S.select_multidim_pairs(funded, want=4)["selected"]) == first


def test_the_page_is_never_asked_to_draw_more_than_it_holds(funded):
    assert len(S.select_multidim_pairs(funded, want=4)["selected"]) <= 4
    assert len(S.select_multidim_pairs(funded, want=2)["selected"]) <= 2


# --------------------------------------------------------------------------- #
# Reconciliation.
# --------------------------------------------------------------------------- #

def test_every_selected_crossing_reconciles_to_its_own_total(funded):
    """A cell grid that does not sum to the total printed with it is a chart
    a reader cannot trust."""
    for key, table in S.select_multidim_pairs(funded, want=4)["selected"].items():
        cells = sum(v for row in table["matrix"] for v in row)
        assert cells == pytest.approx(table["total"], rel=1e-6), key


def test_a_crossing_never_claims_more_than_the_funded_book(funded):
    """Rows the tape cannot band on both axes are excluded, so a crossing
    covers at most the book — never more."""
    total = float(funded["current_outstanding_balance"].sum())
    for key, table in S.select_multidim_pairs(funded, want=4)["selected"].items():
        assert table["total"] <= total * 1.000001, key
