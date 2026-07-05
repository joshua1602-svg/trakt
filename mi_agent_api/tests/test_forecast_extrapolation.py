"""tests/test_forecast_extrapolation.py

Securitisation scale-up forecast (Part 6): run-rate calculation, scenario bands,
milestone date calculation and the insufficient-history caveat.
"""
from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from mi_agent_api import forecast_extrapolation as fx


def test_add_months():
    assert fx._add_months("2025-11", 1) == "2025-12"
    assert fx._add_months("2025-11", 2) == "2026-01"
    assert fx._add_months("2025-01-31", 12) == "2026-01"


def test_completion_history_from_funded():
    periods = [
        {"period": "2025-09", "metrics": {"funded_balance": 8_000_000.0}},
        {"period": "2025-10", "metrics": {"funded_balance": 10_000_000.0}},
        {"period": "2025-11", "metrics": {"funded_balance": 12_500_000.0}},
    ]
    comp = fx.completion_history(periods)
    assert [c["completion_amount"] for c in comp] == [2_000_000.0, 2_500_000.0]


def test_run_rate_model_with_history():
    # 4 monthly completions averaging ~2.5m.
    out = fx.run_rate_model(12_500_000.0, [2_000_000, 2_500_000, 3_000_000, 2_500_000],
                            reporting_period="2025-11")
    assert out["available"] is True and out["status"] == "ok"
    assert out["observedMonths"] == 4
    assert out["baseMonthlyRunRate"] > 0
    assert out["annualisedRunRate"] == round(out["baseMonthlyRunRate"] * 12, 2)
    sc = out["scenarioMonthlyRunRate"]
    assert sc["downside"] <= sc["base"] <= sc["upside"]
    # 4w lookback present.
    assert "4w" in out["lookbackAverages"]


def test_run_rate_projection_and_milestones():
    out = fx.run_rate_model(50_000_000.0, [2_500_000, 2_500_000, 2_500_000, 2_500_000],
                            reporting_period="2025-11")
    # Base run-rate 2.5m/mo: £100m is +£50m -> 20 months under base.
    base_ms = next(m for m in out["milestones"] if m["threshold"] == 100_000_000)
    assert base_ms["reached"] is False
    assert base_ms["baseMonths"] == 20
    assert base_ms["baseDate"] == fx._add_months("2025-11", 20)
    # Already-reached threshold.
    reached = next(m for m in out["milestones"] if m["threshold"] == 25_000_000)
    assert reached["reached"] is True
    assert reached["baseDate"] == "reached"
    # Projected series spans the horizon and downside <= base <= upside.
    last = out["projectedBalances"][-1]
    assert last["downside"] <= last["base"] <= last["upside"]


def test_run_rate_insufficient_history_caveat():
    out = fx.run_rate_model(10_000_000.0, [2_000_000], reporting_period="2025-11")
    assert out["available"] is True
    assert out["status"] == "limited_history"
    # 75%/125% fallback bands.
    sc = out["scenarioMonthlyRunRate"]
    assert sc["downside"] == round(sc["base"] * 0.75, 2)
    assert sc["upside"] == round(sc["base"] * 1.25, 2)
    assert any("indicative" in c for c in out["caveats"])


def test_run_rate_no_history():
    out = fx.run_rate_model(10_000_000.0, [])
    assert out["available"] is False
    assert out["status"] == "insufficient_data"


def test_kfi_conversion_model():
    # Recent weekly rate 5%/week on a £20m KFI book, £1m/week inflow, ~6w lag.
    out = fx.kfi_conversion_model(12_500_000.0, kfi_stock_now=20_000_000.0,
                                  weekly_inflow=1_000_000.0,
                                  weekly_conversion_rate=0.05, lag_weeks=6,
                                  reporting_period="2025-11")
    assert out["available"] is True
    assert out["weeklyConversionRate"] == 0.05
    assert out["expectedMonthlyCompletion"] > 0
    assert out["milestones"]
    # The stock-flow model must NOT multiply the whole KFI book by an
    # annualisation factor: one month's completion is a fraction of the book.
    assert out["expectedMonthlyCompletion"] < 20_000_000.0
    # First projected point is the current balance; the series then grows.
    proj = out["projectedBalances"]
    assert proj[0]["base"] == 12_500_000.0
    assert proj[-1]["base"] > proj[0]["base"]


def test_kfi_conversion_unavailable_without_rate():
    # No conversion rate -> unavailable.
    out = fx.kfi_conversion_model(12_500_000.0, 20_000_000.0, 1_000_000.0, None)
    assert out["available"] is False
    assert out["status"] == "insufficient_data"
    # No KFI stock -> unavailable too.
    out2 = fx.kfi_conversion_model(12_500_000.0, 0.0, 1_000_000.0, 0.05)
    assert out2["available"] is False


def test_kfi_conversion_withheld_when_rate_too_few_weeks():
    # A rate built on only 2 weeks is too volatile to forecast off.
    out = fx.kfi_conversion_model(12_500_000.0, 20_000_000.0, 1_000_000.0, 0.05,
                                  rate_weeks=2, min_rate_weeks=3)
    assert out["available"] is False
    assert out["status"] == "limited_history"
    assert out["rateWeeks"] == 2
    # Enough weeks -> it projects.
    ok = fx.kfi_conversion_model(12_500_000.0, 20_000_000.0, 1_000_000.0, 0.05,
                                 rate_weeks=4, min_rate_weeks=3)
    assert ok["available"] is True
    assert ok["rateWeeks"] == 4


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-q"]))
