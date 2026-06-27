"""mi_agent_api/forecast_extrapolation.py

Securitisation scale-up forecast — answers "when does the book reach scale?".

Three models (the existing point-in-time weighted pipeline is kept and clearly
labelled, NOT presented as the scale-up forecast):

  Model A — completion run-rate extrapolation. Recent completion amounts
    (month-on-month funded growth) → a monthly run-rate, annualised, projected
    forward with downside / base / upside scenario BANDS (not statistical CIs)
    and milestone dates to £25m / £50m / £75m / £100m / £150m.

  Model B — KFI run-rate × completion-rate. Recent KFI inflow × the historical
    KFI→completion conversion rate (with a lag assumption) → projected
    completions and balance. Marked unavailable with a caveat when the history
    is insufficient.

  Model C — current weighted pipeline forecast (the existing forecast bridge),
    labelled "Current weighted pipeline forecast".

All series reuse the governed evolution time-series; nothing is re-discovered.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

from . import evolution as evolution_mod

_THRESHOLDS = [25_000_000, 50_000_000, 75_000_000, 100_000_000, 150_000_000]
_LOOKBACKS = (4, 5, 8, 12)
_HORIZON_MONTHS = 18


# --------------------------------------------------------------------------- #
# Date helpers (no Date.now — work off the governed reporting date)
# --------------------------------------------------------------------------- #
def _add_months(ym: str, months: int) -> str:
    """``YYYY-MM`` (or ``YYYY-MM-DD``) + N months -> ``YYYY-MM``."""
    try:
        year, mon = int(ym[:4]), int(ym[5:7])
    except (ValueError, IndexError):
        return ym
    idx = (year * 12 + (mon - 1)) + months
    return f"{idx // 12:04d}-{idx % 12 + 1:02d}"


def _percentile(values: List[float], pct: float) -> Optional[float]:
    if not values:
        return None
    s = sorted(values)
    if len(s) == 1:
        return s[0]
    k = (len(s) - 1) * pct
    lo = math.floor(k)
    hi = math.ceil(k)
    if lo == hi:
        return s[int(k)]
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


# --------------------------------------------------------------------------- #
# Completion history (month-on-month funded growth = net completions)
# --------------------------------------------------------------------------- #
def completion_history(funded_periods: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Net monthly completion amounts from consecutive funded balances."""
    out: List[Dict[str, Any]] = []
    prev = None
    for p in funded_periods:
        bal = (p.get("metrics") or {}).get("funded_balance")
        if bal is None:
            continue
        if prev is not None:
            out.append({"period": p.get("period"), "completion_amount": round(float(bal) - prev, 2)})
        prev = float(bal)
    return out


# --------------------------------------------------------------------------- #
# Model A — completion run-rate extrapolation
# --------------------------------------------------------------------------- #
def run_rate_model(current_balance: float, completions: List[float], *,
                   reporting_period: Optional[str] = None,
                   thresholds: Optional[List[float]] = None) -> Dict[str, Any]:
    """Completion run-rate forecast with downside / base / upside scenario bands."""
    thresholds = thresholds or _THRESHOLDS
    obs = [c for c in completions if c is not None]
    n = len(obs)
    if n == 0:
        return {"model": "completion_run_rate", "available": False,
                "status": "insufficient_data",
                "caveat": "No completion history (need at least two funded runs).",
                "observedMonths": 0}

    # Lookback averages where enough history exists.
    lookbacks: Dict[str, float] = {}
    for w in _LOOKBACKS:
        if n >= w:
            window = obs[-w:]
            lookbacks[f"{w}w"] = round(sum(window) / w, 2)

    base = round(sum(obs[-min(5, n):]) / min(5, n), 2)  # recent average
    sufficient = n >= 3
    if sufficient:
        downside = round(_percentile(obs, 0.25), 2)
        upside = round(_percentile(obs, 0.75), 2)
        scenario_basis = "empirical 25th/75th percentile of recent monthly completions"
    else:
        downside = round(base * 0.75, 2)
        upside = round(base * 1.25, 2)
        scenario_basis = "75% / 125% of the base run-rate (insufficient history for percentiles)"
    # Guard against a non-positive base making the projection meaningless.
    scenarios = {"downside": max(downside, 0.0), "base": max(base, 0.0),
                 "upside": max(upside, 0.0)}

    projected = _project_series(current_balance, scenarios, reporting_period)
    milestones = _milestones(current_balance, scenarios, reporting_period, thresholds)

    caveats: List[str] = []
    if not sufficient:
        caveats.append("Fewer than 3 monthly completion observations — scenario bands "
                       "are indicative (75%/125% of base), not statistically validated.")
    if base <= 0:
        caveats.append("Recent net completions are flat or negative; milestone dates "
                       "are unavailable until the run-rate turns positive.")

    return {
        "model": "completion_run_rate",
        "available": True,
        "status": "ok" if sufficient else "limited_history",
        "observedMonths": n,
        "lookbackAverages": lookbacks,
        "baseMonthlyRunRate": base,
        "annualisedRunRate": round(base * 12, 2),
        "scenarioMonthlyRunRate": scenarios,
        "scenarioBasis": scenario_basis,
        "projectedBalances": projected,
        "milestones": milestones,
        "assumptions": {
            "lookbackWindowsMonths": list(_LOOKBACKS),
            "observedMonths": n,
            "horizonMonths": _HORIZON_MONTHS,
            "currentFundedBalance": round(current_balance, 2),
            "completionSignal": "month-on-month funded balance growth",
        },
        "caveats": caveats,
    }


def _project_series(current: float, scenarios: Dict[str, float],
                    reporting_period: Optional[str]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for m in range(0, _HORIZON_MONTHS + 1):
        row = {"month": _add_months(reporting_period or "2025-01", m), "offset": m}
        for name, rate in scenarios.items():
            row[name] = round(current + rate * m, 2)
        rows.append(row)
    return rows


def _milestones(current: float, scenarios: Dict[str, float],
                reporting_period: Optional[str], thresholds: List[float]
                ) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for thr in thresholds:
        row: Dict[str, Any] = {"threshold": thr, "thresholdLabel": f"£{int(thr/1_000_000)}m"}
        if current >= thr:
            row["reached"] = True
            for name in scenarios:
                row[f"{name}Date"] = "reached"
            rows.append(row)
            continue
        row["reached"] = False
        for name, rate in scenarios.items():
            if rate and rate > 0:
                months = math.ceil((thr - current) / rate)
                row[f"{name}Date"] = _add_months(reporting_period or "2025-01", months)
                row[f"{name}Months"] = months
            else:
                row[f"{name}Date"] = None
        rows.append(row)
    return rows


# --------------------------------------------------------------------------- #
# Model B — KFI run-rate × completion-rate
# --------------------------------------------------------------------------- #
def kfi_conversion_model(current_balance: float, kfi_amounts: List[float],
                         conversion_rate: Optional[float], *,
                         lag_months: Optional[int] = None,
                         reporting_period: Optional[str] = None,
                         thresholds: Optional[List[float]] = None) -> Dict[str, Any]:
    """KFI inflow × KFI→completion conversion rate → projected completions/balance."""
    thresholds = thresholds or _THRESHOLDS
    obs = [k for k in kfi_amounts if k is not None]
    if not obs or conversion_rate is None or conversion_rate <= 0:
        return {"model": "kfi_conversion", "available": False,
                "status": "insufficient_data",
                "caveat": ("Insufficient KFI history or no trackable KFI→completion "
                           "conversion rate; KFI-based projection unavailable."),
                "observedWeeks": len(obs)}
    avg_kfi = sum(obs[-min(8, len(obs)):]) / min(8, len(obs))
    # Convert weekly KFI inflow to an expected monthly completion value.
    monthly_completion = round(avg_kfi * conversion_rate * (52 / 12), 2)
    scenarios = {"downside": round(monthly_completion * 0.75, 2),
                 "base": monthly_completion,
                 "upside": round(monthly_completion * 1.25, 2)}
    return {
        "model": "kfi_conversion",
        "available": True,
        "status": "ok",
        "observedWeeks": len(obs),
        "avgWeeklyKfiInflow": round(avg_kfi, 2),
        "conversionRate": round(conversion_rate, 4),
        "lagMonths": lag_months,
        "expectedMonthlyCompletion": monthly_completion,
        "scenarioMonthlyRunRate": scenarios,
        "projectedBalances": _project_series(current_balance, scenarios, reporting_period),
        "milestones": _milestones(current_balance, scenarios, reporting_period, thresholds),
        "assumptions": {
            "conversionRate": round(conversion_rate, 4),
            "lagMonths": lag_months,
            "kfiLookbackWeeks": min(8, len(obs)),
            "currentFundedBalance": round(current_balance, 2),
        },
        "caveats": ["KFI→completion conversion and lag are empirical estimates; "
                    "scenario bands are indicative (75%/125% of base)."],
    }


# --------------------------------------------------------------------------- #
# Entry point — wires the governed evolution series into the three models
# --------------------------------------------------------------------------- #
def _kfi_weekly_amounts(pipeline_evo: Dict[str, Any]) -> List[float]:
    out: List[float] = []
    for s in pipeline_evo.get("byStage", []):
        if str(s.get("stage", "")).upper() == "KFI":
            out.append(float(s.get("value", 0.0)))
    return out


def build_extrapolation(output_root, pipeline_root, client_id: str,
                        to_run_id: Optional[str], *,
                        history_model: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Compose Models A/B/C from the governed funded + pipeline + forecast series."""
    funded = evolution_mod.funded_evolution(output_root, client_id, to_run_id)
    forecast = evolution_mod.forecast_evolution(output_root, pipeline_root, client_id, to_run_id)
    try:
        pipeline = evolution_mod.pipeline_evolution(pipeline_root, client_id, to_run_id)
    except Exception:  # noqa: BLE001
        pipeline = {"periods": [], "byStage": []}

    funded_periods = funded.get("periods", [])
    latest = funded_periods[-1] if funded_periods else None
    current_balance = float((latest or {}).get("metrics", {}).get("funded_balance") or 0.0)
    reporting_period = (latest or {}).get("period")

    # Model C — current weighted pipeline forecast (existing bridge, latest period).
    fc_latest = (forecast.get("periods") or [{}])[-1] if forecast.get("periods") else {}
    weighted_pipeline = (fc_latest.get("metrics") or {}).get("weighted_expected_pipeline")
    current_weighted = {
        "model": "current_weighted_pipeline",
        "label": "Current weighted pipeline forecast",
        "available": weighted_pipeline is not None,
        "fundedBalance": round(current_balance, 2),
        "weightedExpectedPipeline": weighted_pipeline,
        "forecastFundedBalance": (fc_latest.get("metrics") or {}).get("forecast_funded_balance"),
        "note": ("Point-in-time bridge (funded balance + weighted expected pipeline). "
                 "NOT the full scale-up forecast."),
    }

    # Model A — completion run-rate.
    comp = completion_history(funded_periods)
    model_a = run_rate_model(current_balance, [c["completion_amount"] for c in comp],
                             reporting_period=reporting_period)
    model_a["completionHistory"] = comp

    # Model B — KFI conversion (needs a trackable conversion rate from history).
    conv = None
    lag = None
    if history_model and history_model.get("available"):
        rates = (history_model.get("stage_rates") or {})
        conv = rates.get("KFI") or rates.get("kfi")
        timing = (history_model.get("historicalCompletionTimingByStage") or {}).get("KFI", {})
        median_days = timing.get("medianDays")
        lag = round(median_days / 30) if median_days else None
    model_b = kfi_conversion_model(current_balance, _kfi_weekly_amounts(pipeline),
                                   conv, lag_months=lag, reporting_period=reporting_period)

    sufficiency = ("ok" if model_a.get("status") == "ok"
                   else ("limited_history" if model_a.get("available") else "insufficient_data"))

    return {
        "portfolioId": client_id,
        "toRunId": to_run_id,
        "reportingPeriod": reporting_period,
        "currentFundedBalance": round(current_balance, 2),
        "currentWeightedPipelineForecast": current_weighted,
        "completionRunRateForecast": model_a,
        "kfiConversionForecast": model_b,
        "thresholds": _THRESHOLDS,
        "dataSufficiency": sufficiency,
        "sourcePeriods": [p.get("period") for p in funded_periods],
        "sourceFiles": funded.get("sourceFiles", []),
        "lineage": {
            "source": "governed funded central tapes + weekly pipeline extracts",
            "completionSignal": "month-on-month funded balance growth",
            "scenarioNote": "Scenario bands are indicative ranges, not statistical confidence intervals.",
            "weightedPipelineFormula": "funded balance + Σ(weighted expected pipeline)",
        },
    }
