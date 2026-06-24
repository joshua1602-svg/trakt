"""Deterministic funded + pipeline forecast bridge (Pipeline MI — Phase 2).

The forecast bridge is the **aggregate** composition of the funded snapshot and
the pipeline snapshot — it never merges pipeline rows into the funded book. It
uses the same formula already implemented row-level in
``mi_agent.states.assembler.total_forecast_funded``:

    forecast_funded_balance
        = current_funded_balance
        + sum(expected_funded_amount * completion_probability)

Completion probabilities come from the governed pipeline prep/contract (which
sources them from ``config/client/pipeline_expected_funding.yaml``) — never from
the frontend, never invented here. A small deterministic **watchlist** of
business-facing early warnings is derived from the funded + pipeline data.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import pandas as pd

from analytics_lib.numeric import coerce_numeric

from .pipeline_prep import classify_forecast_gaps, diagnostics_by_severity

# Deterministic watchlist thresholds (documented heuristics, not probabilities).
CONCENTRATION_SHARE = 0.40   # top broker / region share of pipeline amount
STALE_CASE_DAYS = 120        # a pipeline case older than this is "stale"
HIGH_LTV_RATIO = 0.80        # pipeline LTV at/above this is high-LTV exposure

COMPLETION_PROBABILITY_BASIS = "stage_config"  # from pipeline_expected_funding.yaml


def _num_sum(df: pd.DataFrame, col: str) -> float:
    return float(coerce_numeric(df[col]).sum()) if col in df.columns else 0.0


# --------------------------------------------------------------------------- #
# Forecast readiness
# --------------------------------------------------------------------------- #
def _forecast_readiness(pipeline_available: bool,
                        pipeline_report: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not pipeline_available or pipeline_report is None:
        return {"status": "blocked", "missingRequiredFields": [],
                "warnings": ["No pipeline data available for this run."]}

    fr = pipeline_report.get("forecast_readiness", {}) or {}
    fields_available = fr.get("fields_available", {}) or {}
    # Required for the base forecast bridge: amount + completion probability.
    required = ("expected_amount", "completion_probability")
    missing = [k for k in required if not fields_available.get(k, False)]

    dq = diagnostics_by_severity(pipeline_report)
    if dq["blocker"] or missing:
        status = "blocked" if dq["blocker"] else "partial"
    elif dq["warning"] or not fields_available.get("expected_completion_date", True):
        status = "partial"
    else:
        status = "ready"

    warnings = [d["detail"] for d in dq["warning"]]
    if missing:
        warnings.append("Forecast missing required field(s): " + ", ".join(missing))
    return {"status": status, "missingRequiredFields": missing, "warnings": warnings}


def _grouped_data_quality(pipeline_available: bool,
                          pipeline_report: Optional[Dict[str, Any]]) -> Dict[str, List[Any]]:
    if not pipeline_available or pipeline_report is None:
        return {"blockers": [], "warnings": [],
                "info": [{"check": "no_pipeline_source", "severity": "info",
                          "detail": "No governed pipeline source found for this run."}]}
    g = diagnostics_by_severity(pipeline_report)
    return {"blockers": g["blocker"], "warnings": g["warning"], "info": g["info"]}


# --------------------------------------------------------------------------- #
# Watchlist (Part 4) — concise business-facing early warnings
# --------------------------------------------------------------------------- #
def _watch(category: str, severity: str, title: str, detail: str,
           **extra: Any) -> Dict[str, Any]:
    item = {"category": category, "severity": severity, "title": title, "detail": detail}
    item.update(extra)
    return item


def _concentration(df: pd.DataFrame, field: str, label: str,
                   amount_col: str) -> Optional[Dict[str, Any]]:
    if field not in df.columns or amount_col not in df.columns:
        return None
    amt = coerce_numeric(df[amount_col])
    total = float(amt.sum())
    if total <= 0:
        return None
    by = amt.groupby(df[field].astype(str)).sum().sort_values(ascending=False)
    if by.empty:
        return None
    top_name, top_amt = str(by.index[0]), float(by.iloc[0])
    share = top_amt / total
    if share < CONCENTRATION_SHARE:
        return None
    return _watch(
        f"{field}_concentration", "warning",
        f"{label} concentration: {top_name} is {share * 100:.0f}% of pipeline",
        f"Top {label.lower()} {top_name!r} accounts for {share * 100:.1f}% "
        f"(£{top_amt:,.0f}) of £{total:,.0f} pipeline amount.",
        share=round(share, 4), top=top_name)


def build_pipeline_watchlist(df: pd.DataFrame, pipeline_report: Dict[str, Any],
                             readiness: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Deterministic early-warning items from funded + pipeline + forecast."""
    items: List[Dict[str, Any]] = []
    n = int(len(df))
    if n == 0:
        return items

    # Forecast readiness escalation.
    if readiness["status"] in ("blocked", "partial"):
        items.append(_watch(
            "forecast_readiness", "blocker" if readiness["status"] == "blocked" else "warning",
            f"Forecast readiness: {readiness['status']}",
            "; ".join(readiness.get("warnings", []))
            or f"Forecast readiness is {readiness['status']}."))

    # Governance gaps (amount / stage / probability / completion date).
    def _missing(col: str) -> int:
        if col not in df.columns:
            return n
        s = df[col]
        if s.dtype.kind in "Mf":
            return int(s.isna().sum())
        return int((s.astype(str).str.strip().isin(["", "nan", "NaT", "None", "UNKNOWN"])).sum())

    amt_missing = _missing("current_outstanding_balance")
    if amt_missing:
        items.append(_watch("missing_pipeline_amount",
                            "blocker" if amt_missing == n else "warning",
                            f"{amt_missing} pipeline case(s) missing an amount",
                            f"{amt_missing}/{n} rows have no parseable economic amount.",
                            count=amt_missing))
    # Stage / completion-probability / expected-date gaps, classified by stage:
    # withdrawn/inactive exclusions read as INFO; active gaps as WARNING.
    stage_blank = int((df["pipeline_stage"].astype(str).isin(["", "nan", "None"])).sum()) \
        if "pipeline_stage" in df.columns else n
    if stage_blank == n and n:
        items.append(_watch("missing_pipeline_stage", "blocker",
                            "All pipeline cases are missing a stage/status",
                            f"{n}/{n} rows have no stage value.", count=n))
    for gap in classify_forecast_gaps(df):
        by_stage = gap.get("by_stage", {})
        stage_txt = ", ".join(f"{k}:{v}" for k, v in by_stage.items()) or "—"
        excluded = gap.get("excluded")
        detail = (f"{gap['detail']}. By stage [{stage_txt}]. "
                  f"{'Intentionally excluded from weighted forecast.' if excluded else 'Affects weighted forecast / timing.'}")
        items.append(_watch(gap["check"], gap["severity"], gap["detail"], detail,
                            count=gap.get("count"), byStage=by_stage,
                            excluded=bool(excluded), weighted=bool(gap.get("weighted"))))

    # Stale cases.
    if "pipeline_case_age_days" in df.columns:
        age = coerce_numeric(df["pipeline_case_age_days"])
        stale = int((age >= STALE_CASE_DAYS).sum())
        if stale:
            items.append(_watch("stale_cases", "warning",
                                f"{stale} stale pipeline case(s) (>{STALE_CASE_DAYS}d)",
                                f"{stale} case(s) have been in pipeline for "
                                f"{STALE_CASE_DAYS}+ days without completing.", count=stale))

    # Concentration (broker / region).
    for fld, label in (("broker_channel", "Broker"),
                       ("geographic_region_obligor", "Region")):
        item = _concentration(df, fld, label, "current_outstanding_balance")
        if item:
            items.append(item)

    # High-LTV pipeline exposure.
    if "current_loan_to_value" in df.columns:
        ltv = coerce_numeric(df["current_loan_to_value"])
        high = int((ltv >= HIGH_LTV_RATIO).sum())
        if high:
            amt = coerce_numeric(df.get("current_outstanding_balance", pd.Series(dtype=float)))
            high_amt = float(amt[ltv >= HIGH_LTV_RATIO].sum())
            items.append(_watch("high_ltv_pipeline", "warning",
                                f"{high} high-LTV pipeline case(s) (≥{HIGH_LTV_RATIO * 100:.0f}%)",
                                f"{high} case(s) (£{high_amt:,.0f}) have LTV at or above "
                                f"{HIGH_LTV_RATIO * 100:.0f}%.", count=high))
    return items


# --------------------------------------------------------------------------- #
# Forecast bridge composition
# --------------------------------------------------------------------------- #
def compute_forecast_bridge(
    *,
    client_id: str,
    run_id: str,
    funded_reporting_date: Optional[str],
    funded_df: Optional[pd.DataFrame],
    pipeline_df: Optional[pd.DataFrame],
    pipeline_report: Optional[Dict[str, Any]],
    pipeline_snapshot: Optional[Dict[str, Any]],
    pipeline_source: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Deterministically compose funded + pipeline into a forecast bridge.

    Date concepts are kept distinct (the pipeline is a continuous weekly view, NOT
    a monthly accounting cut-off): ``fundedReportingDate`` is the funded book's
    cut-off for the run, while ``pipelineAsOfDate`` / ``pipelineExtractDate`` /
    ``pipelineSourceFolderDate`` describe the selected weekly pipeline extract.
    There is deliberately no single ambiguous ``reportingDate``.

    ``funded_df`` is the prepared funded MI dataset (loan-level, never merged with
    pipeline). Returns the full forecast-snapshot envelope (funded headline +
    ``pipelineSnapshot`` + ``forecastBridge`` + ``watchlist``). Never raises for a
    missing pipeline.
    """
    funded_balance = _num_sum(funded_df, "current_outstanding_balance") if funded_df is not None else 0.0
    funded_loan_count = int(len(funded_df)) if funded_df is not None else 0

    pipeline_available = pipeline_df is not None and pipeline_report is not None
    readiness = _forecast_readiness(pipeline_available, pipeline_report)
    src = pipeline_source or {}
    pipeline_dates = {
        "pipelineAsOfDate": src.get("pipeline_as_of_date"),
        "pipelineExtractDate": src.get("pipeline_extract_date"),
        "pipelineSourceFolderDate": src.get("pipeline_source_folder_date"),
        "sourceFile": src.get("source_file"),
    }

    prob_summary: Dict[str, Any] = {}
    prob_basis = "none"
    if pipeline_available:
        pipeline_amount = float(pipeline_report.get("total_pipeline_amount") or 0.0)
        pipeline_case_count = int(pipeline_report.get("row_count") or len(pipeline_df))
        weighted = pipeline_report.get("weighted_expected_funded_amount")
        weighted = float(weighted) if weighted is not None else 0.0
        stage_breakdown = (pipeline_snapshot or {}).get("stageBreakdown", [])
        completion_breakdown = (pipeline_snapshot or {}).get("expectedCompletionBreakdown", [])
        watchlist = build_pipeline_watchlist(pipeline_df, pipeline_report, readiness)
        prob_summary = pipeline_report.get("completion_probability_summary", {}) or {}
        prob_basis = pipeline_report.get("completion_probability_basis", "stage_config")
        if not pipeline_dates["pipelineAsOfDate"]:
            pipeline_dates["pipelineAsOfDate"] = pipeline_report.get("pipeline_as_of_date")
    else:
        pipeline_amount = 0.0
        pipeline_case_count = 0
        weighted = 0.0
        stage_breakdown, completion_breakdown = [], []
        watchlist = [_watch("no_pipeline", "info", "No pipeline data available",
                            "No governed pipeline source was found for this run; "
                            "forecast equals the current funded balance.")]

    forecast_balance = funded_balance + weighted
    forecast_loan_count = funded_loan_count + pipeline_case_count

    bridge = {
        "portfolioId": f"{client_id}/{run_id}",
        "client_id": client_id,
        "runId": run_id,
        "fundedReportingDate": funded_reporting_date,
        **pipeline_dates,
        "fundedBalance": round(funded_balance, 2),
        "fundedLoanCount": funded_loan_count,
        "pipelineAvailable": pipeline_available,
        "pipelineAmount": round(pipeline_amount, 2),
        "pipelineCaseCount": pipeline_case_count,
        "weightedExpectedFundedAmount": round(weighted, 2),
        "forecastFundedBalance": round(forecast_balance, 2),
        "forecastLoanCount": forecast_loan_count,
        "completionProbabilityBasis": prob_basis,
        # Governed probability disclosure (gross / excluded / how weighted).
        "grossPipelineAmount": round(pipeline_amount, 2),
        "excludedFromWeightingAmount": prob_summary.get("excluded_amount", 0.0),
        "excludedCaseCount": prob_summary.get("excluded_count", 0),
        "activeGrossPipelineAmount": prob_summary.get("active_gross_amount"),
        "amountWeightedHistorical": prob_summary.get("amount_weighted_historical"),
        "amountWeightedConfig": prob_summary.get("amount_weighted_config"),
        "blendedWeightedConversion": prob_summary.get("blended_weighted_conversion"),
        "expectedCompletionBreakdown": completion_breakdown,
        "stageBreakdown": stage_breakdown,
        "forecastReadiness": readiness,
        "dataQuality": _grouped_data_quality(pipeline_available, pipeline_report),
    }

    return {
        "ok": True,
        "portfolioId": f"{client_id}/{run_id}",
        "client_id": client_id,
        "runId": run_id,
        "fundedReportingDate": funded_reporting_date,
        **pipeline_dates,
        "fundedBalance": round(funded_balance, 2),
        "fundedLoanCount": funded_loan_count,
        "pipelineAvailable": pipeline_available,
        "pipelineSnapshot": pipeline_snapshot,
        "forecastBridge": bridge,
        "watchlist": watchlist,
    }
