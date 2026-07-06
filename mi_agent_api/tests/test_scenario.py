"""mi_agent_api/tests/test_scenario.py

The pure what-if / sensitivity engine: recomputing the milestone-to-target under
an adjusted completion run-rate.
"""
from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from mi_agent_api import scenario as scen


def test_conversion_delta_maps_to_run_rate_multiplier():
    assert scen.multiplier_from_conversion_delta(10) == 1.10
    assert scen.multiplier_from_conversion_delta(-20) == 0.80
    assert scen.multiplier_from_conversion_delta(-500) == 0.0  # floored, never negative


def test_higher_conversion_reaches_target_sooner():
    # £10m now, £1m/month base. To reach £22m needs 12 months at base; a +10%
    # run-rate (£1.1m/month) needs ceil(12/1.1)=11 -> one month sooner.
    out = scen.apply_scenario(
        current_balance=10_000_000, base_monthly_run_rate=1_000_000,
        reporting_period="2025-11", run_rate_multiplier=1.10, target_value=22_000_000)
    assert out["available"] is True
    assert out["baseMonthlyRunRate"] == 1_000_000
    assert out["scenarioMonthlyRunRate"] == 1_100_000
    assert out["baseMonthsToTarget"] == 12
    assert out["scenarioMonthsToTarget"] == 11
    assert out["monthsSaved"] == 1
    assert out["baseTargetDate"] == "2026-11"    # 2025-11 + 12
    assert out["scenarioTargetDate"] == "2026-10"  # 2025-11 + 11


def test_lower_conversion_pushes_target_out():
    out = scen.apply_scenario(
        current_balance=10_000_000, base_monthly_run_rate=1_000_000,
        reporting_period="2025-11", run_rate_multiplier=0.5, target_value=22_000_000)
    assert out["scenarioMonthsToTarget"] == 24  # ceil(12 / 0.5)
    assert out["monthsSaved"] == 12 - 24        # negative -> slower


def test_already_reached_target_is_zero_months():
    out = scen.apply_scenario(
        current_balance=30_000_000, base_monthly_run_rate=1_000_000,
        reporting_period="2025-11", run_rate_multiplier=1.2, target_value=22_000_000)
    assert out["baseMonthsToTarget"] == 0 and out["scenarioMonthsToTarget"] == 0
    assert out["baseTargetDate"] == "reached"


def test_non_positive_rate_has_no_milestone():
    out = scen.apply_scenario(
        current_balance=10_000_000, base_monthly_run_rate=0.0,
        reporting_period="2025-11", run_rate_multiplier=2.0, target_value=22_000_000)
    assert out["available"] is False
    assert out["baseMonthsToTarget"] is None
    assert out["scenarioMonthsToTarget"] is None
    assert out["monthsSaved"] is None


def test_projected_series_diverges_by_the_multiplier():
    out = scen.apply_scenario(
        current_balance=10_000_000, base_monthly_run_rate=1_000_000,
        reporting_period="2025-11", run_rate_multiplier=1.10, target_value=None)
    assert out["projectedSeries"][0]["base"] == out["projectedSeries"][0]["scenario"] == 10_000_000
    last = out["projectedSeries"][-1]
    # At the horizon the scenario is 10% further above the starting balance.
    assert last["scenario"] - 10_000_000 == round((last["base"] - 10_000_000) * 1.10, 2)
