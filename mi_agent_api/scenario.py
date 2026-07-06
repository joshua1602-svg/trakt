"""mi_agent_api/scenario.py

Deterministic what-if / sensitivity engine for the securitisation scale-up.

Answers questions of the form "if our completion conversion (run-rate) changed by
X%, what happens to the time to reach £T?". It is PURE and side-effect free: it
takes an already-derived base (current balance, base monthly run-rate, reporting
period) plus a small set of typed overrides and recomputes the projected balance
series and the milestone date to a target under the adjusted rate, alongside the
unchanged base for comparison.

Design notes:
  * No data access, no LLM, no NL parsing — the caller (the /mi/query router or a
    future scenario UI) resolves the base from ``forecast_extrapolation`` and the
    overrides from the question, then calls ``apply_scenario``. This keeps the
    math reproducible and unit-testable, and lets the same engine serve chat, a
    dashboard panel and deck generation.
  * Completions scale with conversion: holding KFI inflow constant, a +10%
    completion-conversion rate lifts the monthly completion run-rate ~10%, so a
    conversion delta maps to a run-rate multiplier (stated as a caveat).
  * Date math reuses ``forecast_extrapolation._add_months`` so a scenario date is
    on the exact same calendar basis as the base forecast (no ``Date.now``).
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional

from . import forecast_extrapolation as fx_mod

_HORIZON_MONTHS = 18


def multiplier_from_conversion_delta(pct_delta: float) -> float:
    """A conversion-rate change of ``pct_delta`` percent → a run-rate multiplier.
    +10 (%) → 1.10; -20 (%) → 0.80. Floored at 0 (a rate can't go negative)."""
    return max(0.0, 1.0 + float(pct_delta) / 100.0)


def apply_scenario(*, current_balance: float, base_monthly_run_rate: Optional[float],
                   reporting_period: Optional[str], run_rate_multiplier: float = 1.0,
                   target_value: Optional[float] = None,
                   horizon_months: int = _HORIZON_MONTHS) -> Dict[str, Any]:
    """Recompute the projection and milestone-to-target under an adjusted run-rate.

    Returns the base and scenario monthly run-rates, the months/date each takes to
    reach ``target_value`` (``None`` when a rate is non-positive, ``0``/"reached"
    when already at/above target), the months saved (positive = faster), and a
    per-month base-vs-scenario balance series for charting.
    """
    cur = float(current_balance or 0.0)
    base_rate = max(float(base_monthly_run_rate or 0.0), 0.0)
    mult = max(float(run_rate_multiplier), 0.0)
    adj_rate = round(base_rate * mult, 2)
    period = reporting_period or "2025-01"

    def _months_to(rate: float) -> Optional[int]:
        if target_value is None:
            return None
        if cur >= float(target_value):
            return 0
        if rate <= 0:
            return None
        return math.ceil((float(target_value) - cur) / rate)

    def _date_for(months: Optional[int]) -> Optional[str]:
        if months is None:
            return None
        return "reached" if months == 0 else fx_mod._add_months(period, months)

    base_months = _months_to(base_rate)
    scen_months = _months_to(adj_rate)
    months_saved = (base_months - scen_months
                    if base_months is not None and scen_months is not None else None)

    series: List[Dict[str, Any]] = []
    for m in range(0, horizon_months + 1):
        series.append({
            "month": fx_mod._add_months(period, m),
            "offset": m,
            "base": round(cur + base_rate * m, 2),
            "scenario": round(cur + adj_rate * m, 2),
        })

    return {
        "available": base_rate > 0,
        "currentBalance": round(cur, 2),
        "reportingPeriod": period,
        "runRateMultiplier": round(mult, 4),
        "baseMonthlyRunRate": round(base_rate, 2),
        "scenarioMonthlyRunRate": adj_rate,
        "targetValue": (float(target_value) if target_value is not None else None),
        "baseMonthsToTarget": base_months,
        "scenarioMonthsToTarget": scen_months,
        "baseTargetDate": _date_for(base_months),
        "scenarioTargetDate": _date_for(scen_months),
        "monthsSaved": months_saved,
        "projectedSeries": series,
        "horizonMonths": horizon_months,
    }
