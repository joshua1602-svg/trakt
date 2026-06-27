"""mi_agent_api/evolution.py

Funded / pipeline / forecast EVOLUTION (time series) across the governed monthly
funded runs and weekly pipeline extracts already produced by onboarding.

This module REUSES the existing loaders — ``snapshots`` for funded central tapes
and ``pipeline_contract`` for the governed weekly pipeline extracts — and never
re-implements raw onboarding discovery. Each period carries its own reconciliation
(records / balance / coverage) and lineage (run id, reporting date, source file),
matching the point-in-time MI standard.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from analytics_lib.numeric import coerce_numeric

from . import snapshots as snap
from . import pipeline_contract as pipeline_mod

_BALANCE = "current_outstanding_balance"
# Funded breakdown dimensions exposed over time (kept small + governed).
_FUNDED_BREAKDOWN_DIMS = {
    "broker": "broker_channel",
    "region": "geographic_region_obligor",
    "ltv_bucket": "ltv_bucket",
}
MISSING_BUCKET = "Unknown / Missing"


# --------------------------------------------------------------------------- #
# Small metric helpers (balance-weighted, missing-aware) — mirror snapshots.
# --------------------------------------------------------------------------- #
def _bal_sum(df: pd.DataFrame, col: str = _BALANCE) -> Optional[float]:
    if col not in df.columns:
        return None
    return float(coerce_numeric(df[col]).sum())


def _weighted_avg(df: pd.DataFrame, value_col: str, weight_col: str = _BALANCE) -> Optional[float]:
    if value_col not in df.columns or weight_col not in df.columns:
        return None
    v = coerce_numeric(df[value_col])
    w = coerce_numeric(df[weight_col])
    mask = v.notna() & w.notna()
    denom = float(w[mask].sum())
    if denom == 0:
        return None
    return round(float((v[mask] * w[mask]).sum() / denom), 4)


def _simple_avg(df: pd.DataFrame, col: str) -> Optional[float]:
    if col not in df.columns:
        return None
    v = coerce_numeric(df[col])
    return round(float(v.mean()), 4) if v.notna().any() else None


def _reconciliation(df: pd.DataFrame, dataset: str, run_id: str,
                    required: List[str]) -> Dict[str, Any]:
    total_n = int(len(df))
    total_bal = _bal_sum(df)
    missing_measure = [c for c in required if c not in df.columns]
    return {
        "dataset": dataset,
        "run_id": run_id,
        "total_records": total_n,
        "total_balance": (round(total_bal, 2) if total_bal is not None else None),
        "records_included": total_n,
        "balance_included": (round(total_bal, 2) if total_bal is not None else None),
        "records_excluded_missing": 0,
        "balance_excluded_missing": 0.0 if total_bal is not None else None,
        "coverage_by_balance_pct": 100.0 if total_bal else None,
        "missing_dimension_fields": [],
        "missing_measure_fields": missing_measure,
        "filters": {},
    }


def _breakdown(df: pd.DataFrame, dim_col: str, value_col: str = _BALANCE
               ) -> List[Dict[str, Any]]:
    """``[{key, value}]`` summing ``value_col`` by ``dim_col``; missing keys go to
    an explicit Unknown / Missing bucket so the breakdown reconciles to the total."""
    if dim_col not in df.columns or value_col not in df.columns:
        return []
    keys = df[dim_col].astype(object)
    blank = ~keys.notna() | keys.astype(str).str.strip().isin(["", "nan", "None", "NaT"])
    keys = keys.where(~blank, MISSING_BUCKET).astype(str)
    grp = coerce_numeric(df[value_col]).groupby(keys).sum()
    return [{"key": str(k), "value": round(float(v), 2)} for k, v in grp.items()]


# --------------------------------------------------------------------------- #
# Funded evolution
# --------------------------------------------------------------------------- #
def _runs_up_to(output_root: str | os.PathLike, client_id: str,
                to_run_id: Optional[str]) -> List[Dict[str, Any]]:
    disc = snap.discover_snapshots(output_root)
    pf = next((p for p in disc.get("portfolios", []) if p.get("client_id") == client_id), None)
    runs = list(pf.get("runs", [])) if pf else []
    if to_run_id:
        cut = next((i for i, r in enumerate(runs) if r["run_id"] == to_run_id), None)
        if cut is not None:
            runs = runs[: cut + 1]
    return runs


def funded_evolution(output_root: str | os.PathLike, client_id: str,
                     to_run_id: Optional[str] = None,
                     breakdowns: Optional[List[str]] = None) -> Dict[str, Any]:
    """Funded time series across monthly runs up to ``to_run_id`` (inclusive)."""
    runs = _runs_up_to(output_root, client_id, to_run_id)
    required = [_BALANCE, "current_loan_to_value", "current_interest_rate",
                "youngest_borrower_age"]
    want_breakdowns = breakdowns or ["broker", "region", "ltv_bucket"]

    periods: List[Dict[str, Any]] = []
    run_ids: List[str] = []
    dates: List[Optional[str]] = []
    sources: List[Optional[str]] = []
    bd_series: Dict[str, List[Dict[str, Any]]] = {b: [] for b in want_breakdowns}

    for run in runs:
        run_id = run["run_id"]
        tape = snap.resolve_tape_path(output_root, client_id, run_id)
        if tape is None:
            continue
        try:
            df, _rep = snap.load_prepared_run(tape)
        except Exception:  # noqa: BLE001 - a bad tape never breaks the series
            continue
        rdate = run.get("reporting_date") or snap.infer_reporting_date(run_id, df)
        run_ids.append(run_id)
        dates.append(rdate)
        sources.append(str(tape))
        periods.append({
            "run_id": run_id,
            "reporting_date": rdate,
            "period": (rdate or run_id)[:7],
            "metrics": {
                "funded_balance": _bal_sum(df),
                "loan_count": int(len(df)),
                "wa_ltv": _weighted_avg(df, "current_loan_to_value"),
                "wa_interest_rate": _weighted_avg(df, "current_interest_rate"),
                "avg_borrower_age": _simple_avg(df, "youngest_borrower_age"),
            },
            "reconciliation": _reconciliation(df, "funded", run_id, required),
            "source_file": str(tape),
        })
        for b in want_breakdowns:
            dim_col = _FUNDED_BREAKDOWN_DIMS.get(b)
            if dim_col:
                for row in _breakdown(df, dim_col):
                    bd_series[b].append({"period": (rdate or run_id)[:7], **row})

    return {
        "dataset": "funded",
        "portfolioId": client_id,
        "toRunId": to_run_id,
        "availableRunIds": run_ids,
        "reportingDates": dates,
        "sourceFiles": sources,
        "periods": periods,
        "breakdowns": bd_series,
        "lineage": {
            "source": "governed monthly central lender tapes (18_central_lender_tape.csv)",
            "metric": "funded book actuals per reporting month",
            "note": "Each period is an independent funded run; no cross-run merge.",
        },
        "singlePeriod": len(periods) <= 1,
    }


# --------------------------------------------------------------------------- #
# Pipeline evolution (governed weekly extracts)
# --------------------------------------------------------------------------- #
def pipeline_evolution(pipeline_root: str | os.PathLike, client_id: str,
                       to_run_id: Optional[str] = None) -> Dict[str, Any]:
    """Pipeline time series across the governed UNIQUE weekly extracts."""
    inv = pipeline_mod.weekly_extract_inventory(pipeline_root, client_id)
    extracts = inv.get("extracts", [])
    cut_ym = pipeline_mod._year_month(str(to_run_id)) if to_run_id else None

    periods: List[Dict[str, Any]] = []
    by_stage: List[Dict[str, Any]] = []
    sources: List[str] = []
    dates: List[Optional[str]] = []

    for ext in extracts:
        edate = ext.get("pipeline_extract_date")
        if cut_ym and edate and edate[:7] > cut_ym:
            continue
        try:
            df, report = pipeline_mod.load_prepared_pipeline(ext)
        except Exception:  # noqa: BLE001
            continue
        amount = report.get("total_pipeline_amount")
        weighted = report.get("weighted_expected_funded_amount")
        sources.append(ext.get("source_file", ""))
        dates.append(edate)
        periods.append({
            "extract_date": edate,
            "period": (edate or "")[:7],
            "week": edate,
            "metrics": {
                "pipeline_amount": (round(float(amount), 2) if amount is not None else None),
                "pipeline_case_count": int(report.get("row_count", len(df))),
                "weighted_expected_funded_amount": (round(float(weighted), 2)
                                                    if weighted is not None else None),
            },
            "reconciliation": {
                "dataset": "pipeline",
                "extract_date": edate,
                "total_records": int(report.get("row_count", len(df))),
                "total_balance": (round(float(amount), 2) if amount is not None else None),
                "coverage_by_balance_pct": 100.0,
                "missing_measure_fields": [],
                "filters": {},
            },
            "source_file": ext.get("source_file", ""),
        })
        # Pipeline amount by stage for this extract (stacked/multi-line over time).
        if "pipeline_stage" in df.columns and _BALANCE in df.columns:
            grp = coerce_numeric(df[_BALANCE]).groupby(
                df["pipeline_stage"].astype(str)).sum()
            for stage, val in grp.items():
                if str(stage).strip() and str(stage) not in ("nan", "None"):
                    by_stage.append({"period": (edate or "")[:7], "stage": str(stage),
                                     "value": round(float(val), 2)})

    return {
        "dataset": "pipeline",
        "portfolioId": client_id,
        "toRunId": to_run_id,
        "availableExtractDates": dates,
        "sourceFiles": sources,
        "sourceFilesScanned": inv.get("sourceFilesScanned"),
        "uniqueWeeklyExtractsUsed": inv.get("uniqueWeeklyExtractsUsed"),
        "periods": periods,
        "byStage": by_stage,
        "lineage": {
            "source": "governed weekly pipeline extracts (deduplicated)",
            "metric": "origination pipeline amount / weighted expected funded per extract",
            "primarySourcePreference": inv.get("primarySourcePreference"),
        },
        "singlePeriod": len(periods) <= 1,
    }


# --------------------------------------------------------------------------- #
# Weekly origination funnel trends (KFI / Application / Offer / Completion)
# --------------------------------------------------------------------------- #
_FUNNEL_STAGES = ("KFI", "APPLICATION", "OFFER", "COMPLETED")
_FUNNEL_LABELS = {"KFI": "KFIs", "APPLICATION": "Applications",
                  "OFFER": "Offers", "COMPLETED": "Completions"}


def _trailing_avg(values: List[Optional[float]], window: int = 5) -> Optional[float]:
    vals = [v for v in values if v is not None]
    if not vals:
        return None
    use = vals[-window:]
    return round(sum(use) / len(use), 2)


def _trend(values: List[Optional[float]]) -> str:
    vals = [v for v in values if v is not None]
    if len(vals) < 2:
        return "flat"
    delta = vals[-1] - vals[-2]
    return "up" if delta > 0 else ("down" if delta < 0 else "flat")


def pipeline_funnel_evolution(pipeline_root: str | os.PathLike, client_id: str,
                              to_run_id: Optional[str] = None) -> Dict[str, Any]:
    """Weekly origination funnel: KFI / Application / Offer / Completion value AND
    count per governed weekly extract, with a 5-week trailing average, latest
    week value/count and the delta vs the prior week. Reuses the governed weekly
    pipeline extracts (same source as ``pipeline_evolution``)."""
    inv = pipeline_mod.weekly_extract_inventory(pipeline_root, client_id)
    extracts = inv.get("extracts", [])
    cut_ym = pipeline_mod._year_month(str(to_run_id)) if to_run_id else None

    weeks: List[Optional[str]] = []
    sources: List[str] = []
    # series[stage] = [{week, value, count}]
    series: Dict[str, List[Dict[str, Any]]] = {s: [] for s in _FUNNEL_STAGES}

    for ext in extracts:
        edate = ext.get("pipeline_extract_date")
        if cut_ym and edate and edate[:7] > cut_ym:
            continue
        try:
            df, _report = pipeline_mod.load_prepared_pipeline(ext)
        except Exception:  # noqa: BLE001
            continue
        weeks.append(edate)
        sources.append(ext.get("source_file", ""))
        stage_col = df["pipeline_stage"].astype(str) if "pipeline_stage" in df.columns else None
        bal = coerce_numeric(df[_BALANCE]) if _BALANCE in df.columns else None
        for stage in _FUNNEL_STAGES:
            if stage_col is None:
                series[stage].append({"week": edate, "value": None, "count": 0})
                continue
            mask = stage_col.str.upper() == stage
            value = round(float(bal[mask].sum()), 2) if bal is not None else None
            series[stage].append({"week": edate, "value": value, "count": int(mask.sum())})

    summary: Dict[str, Any] = {}
    for stage in _FUNNEL_STAGES:
        pts = series[stage]
        values = [p["value"] for p in pts]
        counts = [float(p["count"]) for p in pts]
        latest_value = values[-1] if values else None
        latest_count = pts[-1]["count"] if pts else 0
        prior_value = values[-2] if len(values) >= 2 else None
        prior_count = pts[-2]["count"] if len(pts) >= 2 else None
        summary[stage] = {
            "label": _FUNNEL_LABELS[stage],
            "latestValue": latest_value,
            "latestCount": latest_count,
            "fiveWeekAvgValue": _trailing_avg(values, 5),
            "fiveWeekAvgCount": _trailing_avg(counts, 5),
            "deltaValue": (round(latest_value - prior_value, 2)
                           if latest_value is not None and prior_value is not None else None),
            "deltaCount": (latest_count - prior_count
                           if prior_count is not None else None),
            "trend": _trend(values),
            "weeksObserved": len([v for v in values if v is not None]),
        }

    return {
        "dataset": "pipeline_funnel",
        "portfolioId": client_id,
        "toRunId": to_run_id,
        "stages": list(_FUNNEL_STAGES),
        "stageLabels": _FUNNEL_LABELS,
        "weeks": weeks,
        "sourceFiles": sources,
        "uniqueWeeklyExtractsUsed": inv.get("uniqueWeeklyExtractsUsed"),
        "series": series,
        "summary": summary,
        "lineage": {
            "source": "governed weekly pipeline extracts (deduplicated)",
            "metric": "weekly KFI / Application / Offer / Completion value and count",
            "fiveWeekAverage": "trailing mean of up to the last 5 weekly extracts",
        },
        "singlePeriod": len(weeks) <= 1,
    }


# --------------------------------------------------------------------------- #
# Forecast bridge evolution (funded balance + weighted pipeline, per funded run)
# --------------------------------------------------------------------------- #
def forecast_evolution(output_root: str | os.PathLike,
                       pipeline_root: str | os.PathLike, client_id: str,
                       to_run_id: Optional[str] = None) -> Dict[str, Any]:
    """Forecast bridge over time: funded balance per run + the latest weighted
    pipeline contribution available at/under that run's month."""
    funded = funded_evolution(output_root, client_id, to_run_id)
    pipe = pipeline_evolution(pipeline_root, client_id, to_run_id)
    # Index pipeline weighted-expected by year-month (latest extract per month).
    weighted_by_month: Dict[str, float] = {}
    for p in pipe["periods"]:
        ym = (p.get("period") or "")
        w = p["metrics"].get("weighted_expected_funded_amount")
        if ym and w is not None:
            weighted_by_month[ym] = float(w)  # later extract overwrites -> latest wins

    periods: List[Dict[str, Any]] = []
    for fp in funded["periods"]:
        ym = fp.get("period") or ""
        funded_bal = fp["metrics"].get("funded_balance") or 0.0
        wpipe = weighted_by_month.get(ym)
        periods.append({
            "period": ym,
            "run_id": fp.get("run_id"),
            "reporting_date": fp.get("reporting_date"),
            "metrics": {
                "funded_balance": round(float(funded_bal), 2),
                "weighted_expected_pipeline": (round(wpipe, 2) if wpipe is not None else None),
                "forecast_funded_balance": round(float(funded_bal) + float(wpipe or 0.0), 2),
            },
            "reconciliation": fp.get("reconciliation"),
            "source_file": fp.get("source_file"),
        })
    return {
        "dataset": "forecast",
        "portfolioId": client_id,
        "toRunId": to_run_id,
        "periods": periods,
        "lineage": {
            "source": "funded central tapes + governed weighted pipeline",
            "formula": "forecast = funded balance + Σ(weighted expected pipeline)",
        },
        "singlePeriod": len(periods) <= 1,
    }
