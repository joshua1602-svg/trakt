#!/usr/bin/env python3
"""Do the metrics mean what their labels claim?

A deterministic number is only valuable if its definition is also correct. This
suite checks the definitions, on tiny portfolios where the right answer can be
worked out by inspection rather than by running the code under test.

The two defects that motivated it, both found by audit rather than by a failing
test:

* **"Prepayment Speed (CPR)"** in the static-pools chart config was
  ``metric: prepayment_amount, agg: sum`` — an amount in £, labelled as an
  annualised rate. Found in Sprint 2.5B.
* **"30+ DPD" meant 31+.** ``_eval_arrears_share`` masked on ``> min_days``
  while the Sprint 2.5B history measure masked on ``>= minimum``, so the same
  nominal metric reported 25% and 50% on the same book. Two definitions of one
  metric, in one system.

Neither was caught by the existing tests, which is the point: passing tests do
not establish that a metric means what it says.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from analytics_lib.history import prepayment_rate
from mi_agent.concentration_tests.library import load_library
from mi_agent.concentration_tests.metrics import evaluate_metric
from trakt_tools.handlers.history import _measure_functions


def _book(days, balances=None) -> pd.DataFrame:
    """A tiny book with explicit days-in-arrears, for boundary work."""
    balances = balances or [100_000.0] * len(days)
    return pd.DataFrame({
        "loan_identifier": [f"L{i}" for i in range(len(days))],
        "current_principal_balance": balances,
        "current_outstanding_balance": balances,
        "number_of_days_in_arrears": days,
        "arrears_balance": [0.0 if d == 0 else 100.0 for d in days],
    })


def _arrears(df, min_days: int, **params):
    library = load_library()
    metric = library.get("perf_arrears_share")
    return evaluate_metric(df, library, metric,
                           {"min_days": min_days, **params}).value


# =========================================================================== #
# The DPD boundary
# =========================================================================== #
def test_30_plus_dpd_includes_a_loan_at_exactly_30_days():
    """"30+ DPD" means 30 or more. Before Sprint 2.5C the mask was ``>``, so a
    loan at exactly 30 days was excluded from a metric labelled 30+."""
    book = _book([0, 29, 30, 31])
    # Two of four loans, equally sized, qualify: the one at 30 and the one at 31.
    assert _arrears(book, 30) == pytest.approx(50.0)


def test_the_exclusive_boundary_is_still_reachable_but_must_be_asked_for():
    """A contract that genuinely says "more than 30 days" can have it — the
    change is that it is now requested rather than inherited from a ``>``."""
    book = _book([0, 29, 30, 31])
    assert _arrears(book, 30, dpd_boundary="exclusive") == pytest.approx(25.0)


def test_the_history_measure_and_the_library_metric_cannot_disagree():
    """The structural fix. The history tool CALLS the library rather than
    agreeing with it by hand, because agreement by hand is what drifts."""
    book = _book([0, 15, 30, 45, 60, 89, 90, 120])
    measures = _measure_functions(["arrears_30_plus_pct", "arrears_60_plus_pct",
                                   "arrears_90_plus_pct"])
    for minimum, name in ((30, "arrears_30_plus_pct"),
                          (60, "arrears_60_plus_pct"),
                          (90, "arrears_90_plus_pct")):
        assert measures[name](book) == pytest.approx(_arrears(book, minimum)), (
            f"{name} disagrees with the library at min_days={minimum}")


def test_the_dpd_bands_are_cumulative_not_mutually_exclusive():
    """"30+" and "60+" overlap by construction. A reader who assumed exclusive
    buckets would double-count, so the cumulative nature is asserted rather than
    left to be inferred from a label."""
    book = _book([0, 30, 60, 90])
    thirty, sixty, ninety = (_arrears(book, 30), _arrears(book, 60),
                             _arrears(book, 90))
    assert thirty > sixty > ninety
    # Three of four loans are 30+, two are 60+, one is 90+.
    assert thirty == pytest.approx(75.0)
    assert sixty == pytest.approx(50.0)
    assert ninety == pytest.approx(25.0)


def test_arrears_share_is_balance_weighted_not_a_loan_count():
    """A large delinquent loan must move the metric more than a small one. A
    count-based figure on this book would be 50%; the balance figure is 90%."""
    book = _book([0, 90], balances=[100_000.0, 900_000.0])
    assert _arrears(book, 30) == pytest.approx(90.0)


# =========================================================================== #
# SMM and CPR
# =========================================================================== #
def _two_period_book(opening_balance: float, unscheduled: float):
    """Opening and closing snapshots with a stated unscheduled prepayment."""
    opening = pd.DataFrame({
        "loan_identifier": ["L0"],
        "current_principal_balance": [opening_balance],
        "unscheduled_principal_collections": [0.0],
    })
    closing = pd.DataFrame({
        "loan_identifier": ["L0"],
        "current_principal_balance": [opening_balance - unscheduled],
        "unscheduled_principal_collections": [unscheduled],
    })
    return {"2026-01-31": opening, "2026-02-28": closing}


def test_smm_and_cpr_match_hand_arithmetic():
    """The brief's own worked example: £100m exposed, £500k qualifying
    unscheduled prepayment, SMM 0.5%, CPR = 1 - (1 - 0.005)^12.

    Computed here by hand, not by calling the function under test.
    """
    snapshots = _two_period_book(100_000_000.0, 500_000.0)
    result = prepayment_rate(snapshots, periods=list(snapshots))

    assert result.smm_pct == pytest.approx(0.5, abs=1e-9)
    # 1 - (1 - 0.005)^12 = 0.0583772 -> 5.8377%. Worked by hand; the published
    # value is rounded to six decimal places, so the tolerance allows for that
    # and nothing more.
    expected_cpr = (1.0 - (1.0 - 0.005) ** 12) * 100.0
    assert expected_cpr == pytest.approx(5.8377, abs=1e-4)
    assert result.cpr_pct == pytest.approx(expected_cpr, abs=1e-6)


def test_cpr_is_not_smm_and_is_labelled_as_the_annualised_form():
    """Confusing the two is a factor-of-twelve error dressed as a percentage."""
    snapshots = _two_period_book(100_000_000.0, 500_000.0)
    result = prepayment_rate(snapshots, periods=list(snapshots))
    assert result.cpr_pct > result.smm_pct * 10
    assert result.method == "OBSERVED_CPR@v1"
    assert any("annualisation" in note.lower() for note in result.notes)


def test_scheduled_amortisation_alone_produces_no_prepayment():
    """The defect class the whole methodology exists to avoid: a balance that
    falls purely through scheduled principal has prepaid nothing, and a rate
    inferred from balance movement would report otherwise."""
    opening = pd.DataFrame({
        "loan_identifier": ["L0"],
        "current_principal_balance": [100_000_000.0],
        "unscheduled_principal_collections": [0.0],
    })
    closing = pd.DataFrame({
        "loan_identifier": ["L0"],
        "current_principal_balance": [99_000_000.0],   # £1m of SCHEDULED principal
        "unscheduled_principal_collections": [0.0],
    })
    result = prepayment_rate({"2026-01-31": opening, "2026-02-28": closing},
                             periods=["2026-01-31", "2026-02-28"])
    assert result.smm_pct == pytest.approx(0.0, abs=1e-9)
    assert result.cpr_pct == pytest.approx(0.0, abs=1e-9)


def test_an_amount_is_never_reported_as_a_rate():
    """The original CPR defect in one assertion: the result carries the £ inputs
    AND the rate, and they are different fields with different units."""
    snapshots = _two_period_book(100_000_000.0, 500_000.0)
    result = prepayment_rate(snapshots, periods=list(snapshots))
    payload = result.to_dict()
    assert payload["unscheduled_principal"] == pytest.approx(500_000.0)
    assert payload["smm_pct"] == pytest.approx(0.5)
    # An amount and a rate must never be the same number.
    assert payload["unscheduled_principal"] != payload["smm_pct"]


# =========================================================================== #
# Weighted averages
# =========================================================================== #
def test_wa_ltv_is_balance_weighted_and_differs_from_the_arithmetic_mean():
    """A tiny book built so the two answers are materially apart. A metric
    labelled "WA LTV" must not be an arithmetic mean of loan LTVs."""
    book = pd.DataFrame({
        "loan_identifier": ["L0", "L1"],
        "current_principal_balance": [900_000.0, 100_000.0],
        "current_outstanding_balance": [900_000.0, 100_000.0],
        "current_loan_to_value": [90.0, 10.0],
    })
    # Arithmetic mean = 50.0. Balance-weighted = (900k*90 + 100k*10) / 1m = 82.0.
    arithmetic_mean = 50.0
    balance_weighted = 82.0

    measures = _measure_functions(["wa_ltv"])
    result = measures["wa_ltv"](book)
    assert result == pytest.approx(balance_weighted)
    assert result != pytest.approx(arithmetic_mean)


def test_the_library_weighted_average_defaults_to_balance_weighting():
    library = load_library()
    metric = library.get("ltv_weighted_average")
    assert metric is not None
    weighting = (metric.parameters or {}).get("weighting") or {}
    assert weighting.get("default", "current_balance") == "current_balance"


# =========================================================================== #
# The label register itself
# =========================================================================== #
def test_the_static_pool_cpr_label_no_longer_claims_to_be_cpr():
    """The known WRONG case: a chart titled "Prepayment Speed (CPR)" whose spec
    was `metric: prepayment_amount, agg: sum` — an amount in £ presented as an
    annualised rate."""
    import yaml

    config = yaml.safe_load(
        (_REPO_ROOT / "config/asset/static_pools_config_erm.yaml").read_text())
    charts = (config.get("static_pools") or {}).get("charts") or []
    for chart in charts:
        title = str(chart.get("title") or "")
        if chart.get("agg") == "sum" and chart.get("metric") == "prepayment_amount":
            assert "CPR" not in title.upper(), (
                f"{title!r} sums an amount but claims to be CPR")
            assert "RATE" not in title.upper() and "SPEED" not in title.upper(), (
                f"{title!r} sums an amount but its label implies a rate")


def test_percentage_points_and_percent_change_are_distinguishable():
    """1.0% -> 1.5% is +0.5 percentage points, not +50%. The series reports the
    absolute difference and names it `change`, so a consumer cannot read a
    relative increase from it by accident."""
    from analytics_lib.history import portfolio_series

    frames = {
        "2026-01-31": pd.DataFrame({"current_principal_balance": [100.0],
                                    "loan_identifier": ["L0"]}),
        "2026-02-28": pd.DataFrame({"current_principal_balance": [100.0],
                                    "loan_identifier": ["L0"]}),
    }
    values = iter([1.0, 1.5])
    outcome = portfolio_series(frames, {"m": lambda _df: next(values)},
                               periods=list(frames))
    series = outcome["series"]["m"]
    assert series["change"] == pytest.approx(0.5), (
        "change is the percentage-POINT difference")
    assert series["change"] != pytest.approx(50.0)
