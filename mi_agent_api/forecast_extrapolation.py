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
from typing import Any, Dict, List, Optional, Tuple, Sequence

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
#: Balance growth is only a PROXY for completions: it also carries interest
#: roll-up and, decisively, any portfolio onboarded in one step. Distinguishing
#: a trend from a step needs at least three observations, so below this the
#: proxy publishes no run-rate at all.
_MIN_PROXY_OBSERVATIONS = 3

#: A proxy observation this many times the median of the others is a step
#: change (a book arriving), not a month of lending, and is excluded.
_STEP_CHANGE_MULTIPLE = 3.0


def _screen_proxy_observations(obs: List[float]) -> Tuple[List[float], List[int]]:
    """Drop balance-growth observations that cannot be organic lending.

    A portfolio onboarded in a single period shows up as one enormous
    month-on-month balance jump. Averaged into a run-rate it produces a growth
    rate the book has never achieved, so it is excluded and disclosed rather
    than smoothed. Needs enough observations to have a median to judge against.
    """
    if len(obs) < _MIN_PROXY_OBSERVATIONS:
        return obs, []
    kept, excluded = [], []
    for i, value in enumerate(obs):
        others = [o for j, o in enumerate(obs) if j != i and o > 0]
        if not others:
            kept.append(value)
            continue
        median = sorted(others)[len(others) // 2]
        if median > 0 and value > median * _STEP_CHANGE_MULTIPLE:
            excluded.append(i)
        else:
            kept.append(value)
    return (kept or obs), excluded


def run_rate_model(current_balance: float, completions: List[float], *,
                   reporting_period: Optional[str] = None,
                   thresholds: Optional[List[float]] = None,
                   observed_monthly_completions: Optional[float] = None,
                   completion_basis: Optional[str] = None) -> Dict[str, Any]:
    """Completion run-rate forecast with downside / base / upside scenario bands.

    ``observed_monthly_completions`` is the GOVERNED completion run-rate, taken
    from the pipeline's own COMPLETED weekly flow. Where it exists it is the
    signal, because it measures loans that actually completed. ``completions``
    (month-on-month funded balance growth) is only a proxy for the same thing
    and is used when no funnel is available — screened, and never from fewer
    than three observations, because a single onboarded portfolio is
    indistinguishable from a month of lending in a two-point series.
    """
    thresholds = thresholds or _THRESHOLDS
    obs = [c for c in completions if c is not None]
    n = len(obs)

    # --- Preferred signal: completions the pipeline actually observed. ------ #
    if observed_monthly_completions is not None and observed_monthly_completions > 0:
        base = round(float(observed_monthly_completions), 2)
        return _assemble_run_rate(
            current_balance, base, thresholds, reporting_period,
            observed_months=n, sufficient=True, lookbacks={},
            downside=round(base * 0.75, 2), upside=round(base * 1.25, 2),
            scenario_basis="75% / 125% of the observed completion run-rate",
            completion_signal=(completion_basis
                               or "governed weekly completion flow (pipeline COMPLETED stage)"),
            caveats=[], excluded=[])

    # --- Fallback: month-on-month funded balance growth, screened. ---------- #
    if n == 0:
        return {"model": "completion_run_rate", "available": False,
                "status": "insufficient_data",
                "caveat": "No completion history (need at least two funded runs).",
                "observedMonths": 0}
    if n < _MIN_PROXY_OBSERVATIONS:
        return {
            "model": "completion_run_rate", "available": False,
            "status": "insufficient_data",
            "observedMonths": n,
            "completionSignal": "month-on-month funded balance growth (proxy)",
            "caveat": (
                f"Only {n} month-on-month observation(s) of funded balance growth, and no "
                "observed completion flow to measure against. Balance growth also carries "
                "interest roll-up and any portfolio onboarded in one step, so a run-rate "
                f"built on fewer than {_MIN_PROXY_OBSERVATIONS} observations could report "
                "growth the book has never achieved. No run-rate is published."),
            "caveats": [
                "A scale-up run-rate needs either observed completion flow from the "
                "pipeline, or at least three months of funded history to tell a trend "
                "from a one-off portfolio arrival."],
        }

    kept, excluded = _screen_proxy_observations(obs)
    obs = kept
    n_kept = len(obs)

    # Lookback averages where enough history exists.
    lookbacks: Dict[str, float] = {}
    for w in _LOOKBACKS:
        if n_kept >= w:
            window = obs[-w:]
            lookbacks[f"{w}w"] = round(sum(window) / w, 2)

    base = round(sum(obs[-min(5, n_kept):]) / min(5, n_kept), 2)  # recent average
    sufficient = n_kept >= 3
    if sufficient:
        downside = round(_percentile(obs, 0.25), 2)
        upside = round(_percentile(obs, 0.75), 2)
        scenario_basis = "empirical 25th/75th percentile of recent monthly completions"
    else:
        downside = round(base * 0.75, 2)
        upside = round(base * 1.25, 2)
        scenario_basis = "75% / 125% of the base run-rate (insufficient history for percentiles)"

    caveats: List[str] = []
    if not sufficient:
        caveats.append("Fewer than 3 monthly completion observations — scenario bands "
                       "are indicative (75%/125% of base), not statistically validated.")
    return _assemble_run_rate(
        current_balance, base, thresholds, reporting_period,
        observed_months=n, sufficient=sufficient, lookbacks=lookbacks,
        downside=downside, upside=upside, scenario_basis=scenario_basis,
        completion_signal="month-on-month funded balance growth (proxy)",
        caveats=caveats, excluded=excluded)


def _assemble_run_rate(current_balance: float, base: float, thresholds: List[float],
                       reporting_period: Optional[str], *, observed_months: int,
                       sufficient: bool, lookbacks: Dict[str, float],
                       downside: float, upside: float, scenario_basis: str,
                       completion_signal: str, caveats: List[str],
                       excluded: List[int]) -> Dict[str, Any]:
    """Project, find milestones and package — shared by both completion signals."""
    # Guard against a non-positive base making the projection meaningless.
    scenarios = {"downside": max(downside, 0.0), "base": max(base, 0.0),
                 "upside": max(upside, 0.0)}
    projected = _project_series(current_balance, scenarios, reporting_period)
    milestones = _milestones(current_balance, scenarios, reporting_period, thresholds)

    caveats = list(caveats)
    if base <= 0:
        caveats.append("Recent net completions are flat or negative; milestone dates "
                       "are unavailable until the run-rate turns positive.")
    if excluded:
        caveats.append(
            f"{len(excluded)} period(s) of funded balance growth were excluded from the "
            "run-rate as step changes — growth of that size is a portfolio arriving, not "
            "a month of lending, and averaging it in would report growth the book has "
            "never achieved.")
    if all(m.get("reached") for m in milestones):
        caveats.append("The book already exceeds every configured funding threshold, so "
                       "the milestone table has no dates left to project.")

    return {
        "model": "completion_run_rate",
        "available": True,
        "status": "ok" if sufficient else "limited_history",
        "observedMonths": observed_months,
        "lookbackAverages": lookbacks,
        "baseMonthlyRunRate": base,
        "annualisedRunRate": round(base * 12, 2),
        "scenarioMonthlyRunRate": scenarios,
        "scenarioBasis": scenario_basis,
        "projectedBalances": projected,
        "milestones": milestones,
        "excludedObservations": excluded,
        "assumptions": {
            "lookbackWindowsMonths": list(_LOOKBACKS),
            "observedMonths": observed_months,
            "horizonMonths": _HORIZON_MONTHS,
            "currentFundedBalance": round(current_balance, 2),
            "completionSignal": completion_signal,
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
_MONTHS_PER_WEEK = 52 / 12  # weeks per month (≈4.33)


def _project_kfi_flow(current: float, kfi0: float, monthly_inflow: float,
                      w_monthly: float, scenarios: Dict[str, float],
                      reporting_period: Optional[str]) -> List[Dict[str, Any]]:
    """Monthly stock-flow projection of the funded balance from the KFI book.

    Each month the KFI STOCK converts to completions at ``w_monthly`` (a monthly
    fraction, capped so it can't complete more than the book holds), the funded
    balance grows by those completions, and the book is REPLENISHED by the KFI
    inflow run-rate. So the existing stock drains into completions while new KFIs
    sustain the run-rate — the number can never multiply the whole book by an
    annualisation factor (the old bug)."""
    state = {name: {"funded": current, "kfi": kfi0} for name in scenarios}
    rows: List[Dict[str, Any]] = [
        {"month": _add_months(reporting_period or "2025-01", 0), "offset": 0,
         **{name: round(current, 2) for name in scenarios}}]
    for m in range(1, _HORIZON_MONTHS + 1):
        row: Dict[str, Any] = {"month": _add_months(reporting_period or "2025-01", m),
                               "offset": m}
        for name, mult in scenarios.items():
            s = state[name]
            completion = min(s["kfi"], s["kfi"] * w_monthly * mult)
            s["funded"] += completion
            s["kfi"] = max(0.0, s["kfi"] + monthly_inflow - completion)
            row[name] = round(s["funded"], 2)
        rows.append(row)
    return rows


def _milestones_from_series(rows: List[Dict[str, Any]], current: float,
                            thresholds: List[float]) -> List[Dict[str, Any]]:
    """First month each scenario's projected balance crosses each threshold."""
    scen_names = [k for k in rows[0] if k not in ("month", "offset")]
    out: List[Dict[str, Any]] = []
    for thr in thresholds:
        row: Dict[str, Any] = {"threshold": thr, "thresholdLabel": f"£{int(thr / 1_000_000)}m"}
        if current >= thr:
            row["reached"] = True
            for n in scen_names:
                row[f"{n}Date"] = "reached"
            out.append(row)
            continue
        row["reached"] = False
        for n in scen_names:
            hit = next((r for r in rows if isinstance(r[n], (int, float)) and r[n] >= thr), None)
            if hit:
                row[f"{n}Date"] = hit["month"]
                row[f"{n}Months"] = hit["offset"]
            else:
                row[f"{n}Date"] = None
        out.append(row)
    return out


def kfi_conversion_model(current_balance: float,
                         kfi_stock_now: Optional[float],
                         weekly_inflow: Optional[float],
                         weekly_conversion_rate: Optional[float], *,
                         lag_weeks: Optional[int] = None,
                         rate_weeks: Optional[int] = None,
                         min_rate_weeks: int = 3,
                         reporting_period: Optional[str] = None,
                         thresholds: Optional[List[float]] = None) -> Dict[str, Any]:
    """Project completions from the KFI book (Model B).

    The current KFI STOCK converts at the recent weekly conversion rate and the
    book is replenished by the weekly KFI inflow run-rate — a monthly stock-flow
    loop. This replaces the prior ``standing_stock × rate × 52/12`` which
    multiplied the whole book by ~4.33 every month.

    ``weekly_conversion_rate`` is the recent (5-week) completion-vs-KFI rate as a
    fraction (avg weekly completions ÷ the KFI stock ``lag_weeks`` earlier).
    ``rate_weeks`` is how many weeks that average is built on; the projection is
    withheld until at least ``min_rate_weeks`` are observed, because a 1-2 week
    rate is too volatile to forecast off.
    """
    thresholds = thresholds or _THRESHOLDS
    if (kfi_stock_now is None or kfi_stock_now <= 0
            or not weekly_conversion_rate or weekly_conversion_rate <= 0):
        return {"model": "kfi_conversion", "available": False,
                "status": "insufficient_data",
                "caveat": ("No current KFI stock or no trackable recent KFI→completion "
                           "conversion rate; KFI-based projection unavailable."),
                "kfiStockNow": round(kfi_stock_now, 2) if kfi_stock_now else 0.0}
    if rate_weeks is not None and rate_weeks < min_rate_weeks:
        return {"model": "kfi_conversion", "available": False,
                "status": "limited_history",
                "caveat": (f"Recent conversion rate is based on only {rate_weeks} week(s); "
                           f"at least {min_rate_weeks} are needed before projecting off it "
                           "to avoid week-to-week volatility."),
                "kfiStockNow": round(float(kfi_stock_now), 2),
                "weeklyConversionRate": round(float(weekly_conversion_rate), 4),
                "rateWeeks": rate_weeks, "minRateWeeks": min_rate_weeks}

    monthly_inflow = max(0.0, float(weekly_inflow or 0.0)) * _MONTHS_PER_WEEK
    # Monthly fraction of the KFI book that completes, from the weekly rate.
    w_monthly = float(weekly_conversion_rate) * _MONTHS_PER_WEEK
    scen = {"downside": 0.75, "base": 1.0, "upside": 1.25}

    projected = _project_kfi_flow(current_balance, float(kfi_stock_now),
                                  monthly_inflow, w_monthly, scen, reporting_period)
    # First-month completion per scenario (indicative monthly run-rate).
    first_completion = {
        name: round(min(kfi_stock_now, kfi_stock_now * w_monthly * mult), 2)
        for name, mult in scen.items()}
    lag_months = round(lag_weeks / _MONTHS_PER_WEEK, 1) if lag_weeks else None

    return {
        "model": "kfi_conversion",
        "available": True,
        "status": "ok",
        "kfiStockNow": round(float(kfi_stock_now), 2),
        "avgWeeklyKfiInflow": round(float(weekly_inflow or 0.0), 2),
        "monthlyKfiInflow": round(monthly_inflow, 2),
        "weeklyConversionRate": round(float(weekly_conversion_rate), 4),
        "conversionRate": round(float(weekly_conversion_rate), 4),  # back-compat key
        "rateWeeks": rate_weeks,
        "lagWeeks": lag_weeks,
        "lagMonths": lag_months,
        "expectedMonthlyCompletion": first_completion["base"],
        "scenarioMonthlyRunRate": first_completion,
        "projectedBalances": projected,
        "milestones": _milestones_from_series(projected, current_balance, thresholds),
        "assumptions": {
            "weeklyConversionRate": round(float(weekly_conversion_rate), 4),
            "conversionBasis": "recent 5-week completion-vs-KFI rate (responsive)",
            "kfiStockNow": round(float(kfi_stock_now), 2),
            "weeklyKfiInflow": round(float(weekly_inflow or 0.0), 2),
            "lagWeeks": lag_weeks,
            "currentFundedBalance": round(current_balance, 2),
            "model": "KFI stock converts at the recent weekly rate; the inflow "
                     "run-rate replenishes the book each month",
        },
        "caveats": ["Recent conversion rate and KFI→completion lag are empirical "
                    "estimates; scenario bands are indicative (75%/125% of the rate)."],
    }


# --------------------------------------------------------------------------- #
# Entry point — wires the governed evolution series into the three models
# --------------------------------------------------------------------------- #
def _withdraw_kfi_model(model_b: Dict[str, Any]) -> Dict[str, Any]:
    """Withdraw the KFI run-rate projection from presentation.

    The KFI stock-flow (``_project_kfi_flow``) has no KFI→completion ATTRITION
    (withdrawal) term, so the self-replenishing stock ultimately funds ~100% of
    gross KFI throughput and the projection over-states scale — it tracks gross
    KFI origination value, not net funded growth (e.g. ~£400m vs a ~£65m
    completion run-rate). Until a governed attrition/conversion basis is
    calibrated the projection is withdrawn: the diagnostic inputs are retained
    for transparency but no balance path or milestones are presented, and the UI
    is told to use the completion run-rate model instead."""
    if not model_b.get("available"):
        return model_b
    model_b["available"] = False
    model_b["status"] = "withdrawn_pending_calibration"
    model_b["reliability"] = "low"
    model_b["caveat"] = (
        "Withdrawn: this KFI run-rate projection omits KFI→completion attrition, "
        "so it over-states scale (it tracks gross KFI origination value, not net "
        "funded growth). Use the completion run-rate forecast until a governed "
        "KFI attrition/conversion basis is calibrated.")
    model_b.pop("projectedBalances", None)
    model_b.pop("milestones", None)
    return model_b


def build_extrapolation(output_root, pipeline_root, client_id: str,
                        to_run_id: Optional[str], *,
                        history_model: Optional[Dict[str, Any]] = None,
                        scope=None,
                        extra_thresholds: Sequence[float] = ()) -> Dict[str, Any]:
    """``extra_thresholds`` — targets the READER named, added to the governed
    ladder for this request only.

    The ladder is a fixed list of round numbers. A question naming a target the
    ladder does not carry had no milestone to answer from, and the caller then
    reported the nearest one it did carry as though it were the answer. Asking
    the projector about the number actually requested is what removes that
    substitution at the source; the ladder itself is unchanged.
    """
    """Compose Models A/B/C from the governed funded + pipeline + forecast series,
    with the funded side narrowed to the governed portfolio ``scope``."""
    funded = evolution_mod.funded_evolution(output_root, client_id, to_run_id,
                                            scope=scope)
    # Weight the pipeline by the SAME governed historical stage rates as the
    # point-in-time bridge, so Model C's 'weighted expected pipeline' matches the
    # Forecast tab instead of silently using the config-only fallback.
    forecast = evolution_mod.forecast_evolution(
        output_root, pipeline_root, client_id, to_run_id, historical_model=history_model,
        scope=scope)
    # NOTE: a third traversal of the weekly pipeline series used to sit here —
    # ``pipeline = evolution_mod.pipeline_evolution(...)`` — whose result was
    # never read by anything below (verified by AST: the name had zero Load
    # references and this function uses no locals()/eval indirection). It
    # prepared every retained weekly extract a second time, which on a 26-week
    # root was ~6.4s of the cold request for a value that was discarded.
    # ``forecast_evolution`` above already traverses the same series, with the
    # governed historical model applied, and Model C reads its output.

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

    # Model B — KFI conversion. Uses the RECENT weekly completion-vs-KFI rate
    # (from the governed funnel, lagged by the KFI→completion timeline) applied
    # to the current KFI stock, replenished by the KFI inflow run-rate.
    lag_weeks = None
    if history_model and history_model.get("available"):
        timing = (history_model.get("historicalCompletionTimingByStage") or {}).get("KFI", {})
        median_days = timing.get("medianDays")
        lag_weeks = max(1, round(median_days / 7)) if median_days else None
    try:
        funnel = evolution_mod.pipeline_funnel_evolution(
            pipeline_root, client_id, to_run_id, lag_weeks=lag_weeks,
            # Reuses the frames forecast_evolution already prepared above.
            historical_model=history_model)
    except Exception:  # noqa: BLE001 - forecast must not 500 on a funnel error
        funnel = {"summary": {}}
    fsum = funnel.get("summary", {}) or {}
    kfi_summary = fsum.get("KFI", {}) or {}
    completed_conv = (fsum.get("COMPLETED", {}) or {}).get("conversion") or {}
    kfi_stock_now = kfi_summary.get("latestStockValue")
    weekly_inflow = kfi_summary.get("fiveWeekAvgFlowValue")
    weekly_rate_pct = completed_conv.get("weeklyRateValue")
    weekly_conv = (weekly_rate_pct / 100.0) if weekly_rate_pct is not None else None

    # Model A — completion run-rate. The signal is the completions the pipeline
    # actually OBSERVED (its COMPLETED weekly flow, the same series the Pipeline
    # → Evolution tab charts), so the two surfaces reconcile by construction.
    # Month-on-month funded balance growth is only a proxy for that, and carries
    # interest roll-up and any portfolio onboarded in one step, so it is the
    # fallback rather than the signal.
    completed_summary = (fsum.get("COMPLETED", {}) or {})
    weekly_completions = completed_summary.get("fiveWeekAvgFlowValue")
    observed_monthly = (float(weekly_completions) * _MONTHS_PER_WEEK
                        if weekly_completions is not None and weekly_completions > 0
                        else None)
    comp = completion_history(funded_periods)
    model_a = run_rate_model(
        current_balance, [c["completion_amount"] for c in comp],
        reporting_period=reporting_period,
        thresholds=sorted({float(t) for t in _THRESHOLDS}
                          | {float(t) for t in (extra_thresholds or ())}),
        observed_monthly_completions=observed_monthly,
        completion_basis=("observed completion flow — 5-week average of the pipeline's "
                          "COMPLETED stage, annualised to a month"))
    model_a["completionHistory"] = comp
    rate_weeks = completed_conv.get("weeksInWindow")
    min_rate_weeks = completed_conv.get("minWeeks", 3)
    model_b = _withdraw_kfi_model(
        kfi_conversion_model(current_balance, kfi_stock_now, weekly_inflow,
                             weekly_conv, lag_weeks=lag_weeks,
                             rate_weeks=rate_weeks, min_rate_weeks=min_rate_weeks,
                             reporting_period=reporting_period))

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
