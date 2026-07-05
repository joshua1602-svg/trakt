"""tests/test_funnel_evolution.py

Weekly origination funnel trends (Part 2): weekly KFI / Application / Offer /
Completion value + count, 5-week average, latest week value/count, delta vs prior
week, and the single-period state. Uses the governed client_001 pipeline fixture.
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from mi_agent_api import evolution as evo

_PIPELINE_FIXTURE = _REPO_ROOT / "tests" / "fixtures" / "client_001_mi_pack"


def test_five_week_average_helper():
    assert evo._trailing_avg([10.0, 20.0, 30.0], 5) == 20.0
    assert evo._trailing_avg([1, 2, 3, 4, 5, 6, 7], 5) == 5.0  # last 5: 3..7
    assert evo._trailing_avg([], 5) is None
    assert evo._trailing_avg([None, 10.0], 5) == 10.0


def test_trend_helper():
    assert evo._trend([1.0, 2.0]) == "up"
    assert evo._trend([2.0, 1.0]) == "down"
    assert evo._trend([1.0]) == "flat"


def test_weekly_flow_from_stock_levels():
    # Stock levels week-on-week -> weekly flow (net change). First week has no
    # prior extract so its flow is None (never fabricated as the level).
    assert evo.weekly_flow([100.0, 130.0, 120.0, 160.0]) == [None, 30.0, -10.0, 40.0]
    assert evo.weekly_flow([]) == []
    assert evo.weekly_flow([50.0]) == [None]
    # A missing level (and the week after it) yields None flow, no exception.
    assert evo.weekly_flow([100.0, None, 140.0]) == [None, None, None]


def test_five_week_average_is_flow_not_stock():
    """Acceptance (Task 4): the 5-week average must be the trailing mean of the
    WEEKLY FLOW, not the average stock level. A cumulative/stock series of
    250,255,260,265,280,312.7 (MM) has a large stock average but a small weekly
    flow — the two must not be conflated."""
    stock = [250.0, 255.0, 260.0, 265.0, 280.0, 312.7]  # £MM, cumulative-style
    flow = evo.weekly_flow(stock)                         # [None,5,5,5,15,32.7]
    # The 5-week trailing average of the WEEKLY FLOW (last 5 non-null flows).
    flow_avg = evo._trailing_avg([f for f in flow if f is not None], 5)
    assert flow_avg == round((5 + 5 + 5 + 15 + 32.7) / 5, 2)   # 12.54, NOT ~272
    # A naive stock average would be ~270 — proving the bug the fix removes.
    stock_avg = evo._trailing_avg(stock, 5)
    assert stock_avg > 250 and abs(stock_avg - flow_avg) > 100
    # Δ vs prior week reconciles with the flow basis: 32.7 − 15 = 17.7.
    assert round(flow[-1] - flow[-2], 2) == 17.7


def test_funnel_evolution_builds_stage_series():
    warnings.simplefilter("ignore")
    out = evo.pipeline_funnel_evolution(_PIPELINE_FIXTURE, "client_001", None)
    assert out["dataset"] == "pipeline_funnel"
    assert out["stages"] == ["KFI", "APPLICATION", "OFFER", "COMPLETED"]
    # Every stage has a per-week series with value + count keys.
    for stage in out["stages"]:
        assert stage in out["series"]
        for pt in out["series"][stage]:
            assert "week" in pt and "value" in pt and "count" in pt


def test_funnel_summary_metrics():
    warnings.simplefilter("ignore")
    out = evo.pipeline_funnel_evolution(_PIPELINE_FIXTURE, "client_001", None)
    summ = out["summary"]
    for stage in out["stages"]:
        s = summ[stage]
        # Weekly-flow basis (the default) and the stock level, clearly separated.
        assert "latestFlowValue" in s and "latestFlowCount" in s
        assert "fiveWeekAvgFlowValue" in s and "fiveWeekAvgFlowCount" in s
        assert "deltaFlowValue" in s and "deltaFlowCount" in s
        assert "latestStockValue" in s and "latestStockCount" in s
        assert s["trend"] in ("up", "down", "flat")
    # Completions stage has at least one completed case in the fixture (stock).
    assert summ["COMPLETED"]["latestStockCount"] >= 1
    assert summ["KFI"]["label"] == "KFIs"
    # KFI is the funnel denominator: no conversion; downstream stages carry one.
    assert summ["KFI"]["conversion"] is None
    for stage in ("APPLICATION", "OFFER", "COMPLETED"):
        conv = summ[stage]["conversion"]
        assert conv is not None
        for k in ("basis", "lagWeeks", "lagApplied", "denominatorWeek",
                  "avgWeeklyFlowCount", "avgWeeklyFlowValue",
                  "kfiStockCount", "kfiStockValue",
                  "weeklyRateCount", "weeklyRateValue"):
            assert k in conv


def test_funnel_conversion_is_flow_over_lagged_kfi_stock():
    """The forward conversion rate = avg weekly flow into a stage (last 5 weeks)
    over the KFI STOCK as it stood `lag_weeks` earlier — never a sum of stock
    across weeks (the old bug that could exceed 100%)."""
    warnings.simplefilter("ignore")
    lag = 3
    out = evo.pipeline_funnel_evolution(_PIPELINE_FIXTURE, "client_001", None,
                                        lag_weeks=lag)
    assert out["conversionLagWeeks"] == lag
    weeks = out["weeks"]
    kfi_counts = [float(p["count"]) for p in out["series"]["KFI"]]
    kfi_values = [p["value"] for p in out["series"]["KFI"]]
    denom_idx = max(0, len(kfi_counts) - 1 - lag)
    for stage in ("APPLICATION", "OFFER", "COMPLETED"):
        conv = out["summary"][stage]["conversion"]
        assert conv["lagWeeks"] == lag and conv["lagApplied"] is True
        # Denominator is the lagged KFI stock, not a sum across weeks.
        assert conv["kfiStockCount"] == int(kfi_counts[denom_idx])
        assert conv["kfiStockValue"] == kfi_values[denom_idx]
        assert conv["denominatorWeek"] == weeks[denom_idx]
        # Numerator is the trailing-5 average weekly flow into the stage.
        flows = [f["flowValue"] for f in out["flowSeries"][stage]]
        assert conv["avgWeeklyFlowValue"] == evo._trailing_avg(flows, 5)
        # Rate reconciles: numerator / denominator (divide-by-zero safe).
        if conv["kfiStockValue"]:
            assert conv["weeklyRateValue"] == round(
                conv["avgWeeklyFlowValue"] / conv["kfiStockValue"] * 100.0, 2)


def test_funnel_conversion_unlagged_when_lag_unknown():
    warnings.simplefilter("ignore")
    out = evo.pipeline_funnel_evolution(_PIPELINE_FIXTURE, "client_001", None)
    assert out["conversionLagWeeks"] is None
    conv = out["summary"]["COMPLETED"]["conversion"]
    assert conv["lagWeeks"] is None and conv["lagApplied"] is False
    # Unlagged denominator is the latest KFI stock.
    assert conv["kfiStockCount"] == out["summary"]["KFI"]["latestStockCount"]


def test_funnel_flow_series_reconciles_with_stock():
    """The flow series is the week-on-week change in the stock series, and the
    5-week average flow is the trailing mean of that flow (Task 4)."""
    warnings.simplefilter("ignore")
    out = evo.pipeline_funnel_evolution(_PIPELINE_FIXTURE, "client_001", None)
    for stage in out["stages"]:
        stock = [p["value"] for p in out["series"][stage]]
        flows = [f["flowValue"] for f in out["flowSeries"][stage]]
        assert flows == evo.weekly_flow(stock)
        assert out["summary"][stage]["fiveWeekAvgFlowValue"] == \
            evo._trailing_avg([f for f in flows if f is not None], 5)


def test_funnel_has_source_and_weeks():
    warnings.simplefilter("ignore")
    out = evo.pipeline_funnel_evolution(_PIPELINE_FIXTURE, "client_001", None)
    assert out["weeks"]
    assert out["sourceFiles"]
    assert out["lineage"]["source"].startswith("governed weekly pipeline")


def test_funnel_single_period_state(tmp_path):
    # No pipeline data -> empty, single-period.
    out = evo.pipeline_funnel_evolution(tmp_path, "client_001", None)
    assert out["singlePeriod"] is True
    assert out["weeks"] == []


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-q"]))
