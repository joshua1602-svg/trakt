"""Pipeline movement attributes to product, and the attribution reconciles.

Product is a governed pipeline dimension the preparation layer already
materialises, but it was not one of the movement contributor dimensions — so the
weekly review could say a broker and a region drove the pipeline and could not
say which PRODUCT did, which is the first thing a reader asks about a book that
grew.

Nothing here computes an expected answer with the code under test. The fixture
states what each case is worth and which product it belongs to; the expected
per-product movement is arithmetic done in the test, in the open.
"""

from __future__ import annotations

import pandas as pd
import pytest

from mi_agent_api import movement_detail as md
from mi_agent_api.insight_generators import pipeline_movement

CASE = md.CASE_KEY
MEASURE = md.MEASURE
STAGE = md.STAGE


def _frame(rows) -> pd.DataFrame:
    """A prepared weekly pipeline frame: the columns the movement layer reads."""
    return pd.DataFrame(
        [{CASE: c, MEASURE: v, STAGE: s,
          "broker_channel": b, "geographic_region_obligor": r,
          "product_type": p}
         for c, v, s, b, r, p in rows])


# Two weeks of one book. Product movement is deliberately NOT the same shape as
# broker or region movement, so a test that passed by reading the wrong
# dimension would give the wrong number rather than the right one by luck.
#
#   C1  Lump Sum   1.0m -> 1.4m   (+0.4m)   grows
#   C2  Drawdown   2.0m -> 1.5m   (-0.5m)   shrinks
#   C3  Lump Sum      — -> 3.0m   (+3.0m)   new
#   C4  Drawdown   0.8m ->    —   (-0.8m)   removed
#
#   Lump Sum net  +3.4m ;  Drawdown net  -1.3m ;  total  +2.1m
PRIOR = _frame([
    ("C1", 1_000_000.0, "KFI", "Broker A", "London", "Lump Sum"),
    ("C2", 2_000_000.0, "KFI", "Broker B", "South East", "Drawdown"),
    ("C4",   800_000.0, "OFFER", "Broker B", "London", "Drawdown"),
])
CURRENT = _frame([
    ("C1", 1_400_000.0, "OFFER", "Broker A", "London", "Lump Sum"),
    ("C2", 1_500_000.0, "KFI", "Broker B", "South East", "Drawdown"),
    ("C3", 3_000_000.0, "KFI", "Broker A", "London", "Lump Sum"),
])

EXPECTED_TOTAL = 2_100_000.0
EXPECTED_BY_PRODUCT = {"Lump Sum": 3_400_000.0, "Drawdown": -1_300_000.0}


@pytest.fixture
def detail() -> dict:
    return md.build_movement_detail(
        md.DETAIL_PIPELINE, CURRENT, PRIOR,
        as_of_date="2026-08-07", comparison_date="2026-07-31",
        portfolio_id="portfolio_alpha", run_id="run-1")


# --------------------------------------------------------------------------- #
# The binding
# --------------------------------------------------------------------------- #
def test_product_is_a_governed_contributor_dimension():
    assert ("products", "product_type") in md.DIMENSIONS


def test_pipeline_movement_attributes_to_product(detail):
    products = detail["contributors"]["products"]
    assert {p["name"] for p in products} == set(EXPECTED_BY_PRODUCT)

    by_name = {p["name"]: p["amount"] for p in products}
    assert by_name == pytest.approx(EXPECTED_BY_PRODUCT)

    # The largest mover leads, by absolute movement.
    assert products[0]["name"] == "Lump Sum"


# --------------------------------------------------------------------------- #
# Reconciliation — the property that makes the attribution quotable
# --------------------------------------------------------------------------- #
def test_product_contributions_reconcile_to_the_governed_movement(detail):
    """Every product's delta, summed, IS the headline movement.

    ``top_n`` defaults to 3 and this fixture has two products, so the returned
    set is the whole decomposition here. A book with more products would return
    the top slice; the reconciliation property belongs to the full grouping,
    which is what the second assertion checks directly.
    """
    assert detail["headline_metric"]["change"] == pytest.approx(EXPECTED_TOTAL)

    products = detail["contributors"]["products"]
    assert sum(p["amount"] for p in products) == pytest.approx(EXPECTED_TOTAL)

    components = md.movement_components(
        CURRENT, PRIOR, dims=[c for _k, c in md.DIMENSIONS])
    full = md.rank_contributors(components, "product_type",
                                total=EXPECTED_TOTAL, top_n=100)
    assert sum(p["amount"] for p in full) == pytest.approx(EXPECTED_TOTAL)


def test_each_dimension_reconciles_independently(detail):
    """Broker, region and product are three decompositions of ONE movement.

    Each sums to the total on its own. They are not additive with each other,
    which is why the card names one lead per dimension rather than adding them.
    """
    for key in ("brokers", "regions", "products"):
        rows = detail["contributors"][key]
        assert sum(r["amount"] for r in rows) == pytest.approx(EXPECTED_TOTAL), key


def test_shares_are_stated_against_the_headline_change(detail):
    by_name = {p["name"]: p for p in detail["contributors"]["products"]}
    assert by_name["Lump Sum"]["share_of_change_pct"] == pytest.approx(161.9, abs=0.1)
    assert by_name["Drawdown"]["share_of_change_pct"] == pytest.approx(-61.9, abs=0.1)


# --------------------------------------------------------------------------- #
# The dimension reaches the governed insight and the card wording
# --------------------------------------------------------------------------- #
def test_the_insight_carries_and_names_the_product_driver(detail):
    ctx = {"tenant_id": "T", "portfolio_id": "portfolio_alpha",
           "portfolio_context": "total", "run_id": "run-1",
           "as_of_date": "2026-08-07", "comparison_date": "2026-07-31"}
    insights, omissions = pipeline_movement(ctx, detail)

    assert insights and not omissions
    insight = insights[0]
    assert [p["name"] for p in insight.contributors["products"]] == \
        ["Lump Sum", "Drawdown"]
    assert "Largest product movement: Lump Sum" in insight.summary


def test_a_book_with_no_product_column_names_no_product_driver():
    """An absent dimension must not become a fabricated driver.

    The attribution still reconciles — the whole movement lands in the governed
    ``Unknown`` bucket, which this module never drops or merges — but the
    narrative must not report "growth was led by Unknown", because that reads as
    an answer when it is the absence of one.
    """
    bare_current = CURRENT.drop(columns=["product_type"])
    bare_prior = PRIOR.drop(columns=["product_type"])
    detail = md.build_movement_detail(
        md.DETAIL_PIPELINE, bare_current, bare_prior,
        as_of_date="2026-08-07", comparison_date="2026-07-31",
        portfolio_id="portfolio_alpha")

    products = detail["contributors"]["products"]
    # Reconciliation survives: no case is lost by the dimension being absent.
    assert sum(p["amount"] for p in products) == pytest.approx(EXPECTED_TOTAL)
    # And no product name is invented.
    assert {p["name"] for p in products} <= {md.UNKNOWN}

    ctx = {"tenant_id": "T", "portfolio_id": "portfolio_alpha",
           "portfolio_context": "total", "run_id": "run-1",
           "as_of_date": "2026-08-07", "comparison_date": "2026-07-31"}
    insight = pipeline_movement(ctx, detail)[0][0]
    assert "Largest product movement" not in insight.summary
    assert md.UNKNOWN not in insight.summary
