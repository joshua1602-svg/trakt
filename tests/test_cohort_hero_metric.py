#!/usr/bin/env python3
"""tests/test_cohort_hero_metric.py

Which cohort measure answers "how are vintages behaving as they season?".

Funded balance was the hero unconditionally. On a stable book that drew four
nearly flat lines — arithmetically true, and an answer to no question a reader
has. A measure now earns the curve by being available for these cohorts AND by
moving as they age, with the governed preference order breaking ties.

Nothing branches on what kind of book this is: availability is what the
governed service actually emitted, and NNEG appears for a lifetime book because
the tape carries valuations, not because the renderer knows what it is looking
at.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from mi_agent_pptx import cohorts as CO  # noqa: E402


def series(vintage: str, points):
    """A cohort whose live points carry the given metric bags."""
    return CO.adapt_progression(
        {"available": True,
         "periods": [{"period": f"P{i}", "loanCount": 10,
                      "loanRetention": p.pop("loan_retention", None),
                      "metrics": p}
                     for i, p in enumerate(points)]},
        vintage)


def test_a_flat_balance_is_not_made_the_hero():
    """Catches: four nearly flat lines presented as a seasoning story."""
    flat = [series("2022", [{"funded_balance": 10_000_000.0},
                            {"funded_balance": 10_010_000.0},
                            {"funded_balance": 10_005_000.0}])]
    assert CO.hero_metric(flat) is None


def test_a_measure_that_moves_earns_the_curve():
    moving = [series("2022", [{"funded_balance": 10_000_000.0},
                              {"funded_balance": 8_000_000.0},
                              {"funded_balance": 6_000_000.0}])]
    hero = CO.hero_metric(moving)
    assert hero is not None
    assert hero[0] == "funded_balance"


def test_a_risk_measure_outranks_a_balance_curve_when_both_move():
    """"How are vintages behaving" is a question about credit. A balance curve
    answers it only when the balance is what moved."""
    both = [series("2022", [
        {"funded_balance": 10_000_000.0, "wa_ltv": 50.0},
        {"funded_balance": 8_000_000.0, "wa_ltv": 55.0},
        {"funded_balance": 6_000_000.0, "wa_ltv": 61.0}])]
    metric, title, fmt = CO.hero_metric(both)
    assert metric == "wa_ltv"
    assert fmt == "pct"
    assert "LTV" in title


def test_the_asset_specific_measure_is_chosen_by_availability_not_branching():
    """NNEG leads where the governed service emitted it, and is simply absent
    where it did not — no PPTX branch decides that."""
    with_nneg = [series("2022", [
        {"wa_ltv": 50.0, "nneg_headroom_pct": 44.0},
        {"wa_ltv": 52.0, "nneg_headroom_pct": 47.0},
        {"wa_ltv": 54.0, "nneg_headroom_pct": 51.0}])]
    assert CO.hero_metric(with_nneg)[0] == "nneg_headroom_pct"

    without = [series("2022", [
        {"wa_ltv": 50.0}, {"wa_ltv": 52.0}, {"wa_ltv": 54.0}])]
    assert CO.hero_metric(without)[0] == "wa_ltv"


def test_loan_survival_is_read_from_the_pool_not_the_metric_bag():
    """Survival is a property of the pool, not a measure of the loans in it."""
    survival = [series("2022", [
        {"loan_retention": 100.0, "funded_balance": 10_000_000.0},
        {"loan_retention": 94.0, "funded_balance": 10_010_000.0},
        {"loan_retention": 88.0, "funded_balance": 10_005_000.0}])]
    assert CO.hero_metric(survival)[0] == "loan_retention"


def test_a_single_period_cohort_has_no_curve():
    """One point is a formation snapshot; joining it into a line would be the
    misleading trend the sufficiency rules exist to prevent."""
    assert CO.hero_metric([series("2026", [{"funded_balance": 1_000_000.0}])]) is None


def test_selection_is_deterministic():
    book = [series("2022", [{"wa_ltv": 50.0}, {"wa_ltv": 55.0}, {"wa_ltv": 61.0}]),
            series("2024", [{"wa_ltv": 48.0}, {"wa_ltv": 51.0}, {"wa_ltv": 54.0}])]
    first = CO.hero_metric(book)
    for _ in range(5):
        assert CO.hero_metric(book) == first
        assert CO.hero_metric(list(reversed(book))) == first


def test_travel_is_judged_against_the_measures_own_level():
    """A 50% LTV moving one point and a £10m balance moving £200k are the same
    size of movement, and must be scored the same way."""
    assert CO._travel([50.0, 51.0]) == pytest.approx(1.0 / 50.5, rel=1e-3)
    assert CO._travel([10_000_000.0, 10_200_000.0]) == pytest.approx(
        200_000.0 / 10_100_000.0, rel=1e-3)
    assert CO._travel([1.0]) is None
    assert CO._travel([]) is None
