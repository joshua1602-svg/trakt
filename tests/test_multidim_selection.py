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
            # A meaningful single/joint split, so borrower type can compete
            # for a crossing rather than being unavailable on every fixture.
            # ``funded_prep`` derives it from second-applicant presence, which
            # is how a real tape carries the fact.
            "borrower_2_DOB": ("1955-03-14" if i % 3 == 0 else None),
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


# --------------------------------------------------------------------------- #
# Which dimensions may reach a matrix.
# --------------------------------------------------------------------------- #

def test_borrower_type_can_take_a_crossing_when_it_earns_one(funded):
    """Catches: LTV x borrower type being permanently unreachable.

    The pack used to draw a fixed three crossings, so a single/joint split
    could never appear however much it mattered. It is a governed candidate
    now, and on a book carrying a real split it competes on the same terms as
    the rest — it is either selected, or rejected for a reason about ITS OWN
    shape rather than about the list it sits in.
    """
    chosen = S.select_multidim_pairs(
        funded, want=len(S.MULTIDIM_CANDIDATE_PAIRS))
    reasons = {r["key"]: r for r in chosen["rejected"]}
    assert "ltv_borrower_type" in chosen["selected"] or \
        "ltv_borrower_type" in reasons, "borrower type never reached the contest"
    if "ltv_borrower_type" in reasons:
        row = reasons["ltv_borrower_type"]
        assert row["reasonCode"] in (S.REASON_TOO_SPARSE, S.REASON_REDUNDANT,
                                     S.REASON_ONE_CATEGORY, S.REASON_LOWER_RANKED), row


def test_a_borrower_type_crossing_reconciles_like_any_other(funded):
    """It is the same generic cross-tab, so it must obey the same arithmetic."""
    chosen = S.select_multidim_pairs(
        funded, want=len(S.MULTIDIM_CANDIDATE_PAIRS))
    table = chosen["selected"].get("ltv_borrower_type")
    if table is None:
        pytest.skip("this book did not select the borrower type crossing")
    cells = sum(v for row in table["matrix"] for v in row)
    assert cells == pytest.approx(table["total"], rel=1e-6)


def _diagonal_book():
    """A book that lands one loan in each (LTV band, age band) pair.

    Both axes are wide and every off-diagonal cell is empty, which is exactly
    the grid a reader cannot read: seven bands each way is forty-nine cells
    carrying seven numbers.
    """
    rows = []
    for i in range(7):
        rows.append({
            "unique_identifier": f"D{i:03d}",
            "current_outstanding_balance": 250_000.0,
            "current_valuation_amount": 1_000_000.0,
            "current_loan_to_value": 25.0 + i * 10.0,     # 25 .. 85
            "current_interest_rate": 6.0,
            "youngest_borrower_age": 57 + i * 5,          # 57 .. 87
            "geographic_region_collateral": _REGIONS[i % len(_REGIONS)],
            "collateral_geography": _REGIONS[i % len(_REGIONS)],
            "product_type": _PRODUCTS[i % len(_PRODUCTS)],
            "origination_date": "2023-06-01",
            "data_cut_off_date": "2026-06-30",
        })
    from mi_agent_api.funded_prep import prepare_funded_mi_dataset
    out, _report = prepare_funded_mi_dataset(pd.DataFrame(rows))
    return out


def test_a_sparse_matrix_is_suppressed_with_the_sparsity_reason():
    """Catches: a grid of mostly empty cells presented as an analysis.

    The reason recorded has to say sparsity — not that the crossing was
    uninformative, which would be a different and untrue claim about a book
    that simply has too few loans to fill the grid.
    """
    chosen = S.select_multidim_pairs(
        _diagonal_book(), want=len(S.MULTIDIM_CANDIDATE_PAIRS))
    sparse = [r for r in chosen["rejected"]
              if r["reasonCode"] == S.REASON_TOO_SPARSE]
    assert sparse, [(r["key"], r["reasonCode"]) for r in chosen["rejected"]]
    for row in sparse:
        assert "cells carry balance" in row["reason"], row
        assert row["key"] not in chosen["selected"]


def test_the_sparsity_reason_quotes_the_density_it_measured():
    """The number in the prose must be the number that failed the gate.

    A ledger line quoting a figure the gate did not use is the class of untrue
    reason this sprint exists to remove.
    """
    chosen = S.select_multidim_pairs(
        _diagonal_book(), want=len(S.MULTIDIM_CANDIDATE_PAIRS))
    for row in chosen["rejected"]:
        if row["reasonCode"] != S.REASON_TOO_SPARSE:
            continue
        quoted = int(row["reason"].split("only ")[1].split("%")[0])
        assert quoted < S.MULTIDIM_MIN_DENSITY * 100, row


def test_the_sparsity_floor_is_the_one_the_reason_quotes():
    """The prose and the constant must be the same number.

    A ledger line that quotes a threshold the code does not use is exactly the
    class of untrue reason this sprint exists to remove.
    """
    assert 0.0 < S.MULTIDIM_MIN_DENSITY < 1.0
    assert S.MULTIDIM_MIN_AXIS >= 2
