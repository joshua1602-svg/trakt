"""MI workspace orchestration — Funded / Pipeline / Forecast views.

Composes the existing funded snapshot + pipeline snapshot + forecast bridge into
one workspace view-model and supports the tab-aware MI Agent query. It never
merges the funded and pipeline SPINE datasets; the Forecast view frame is a
DERIVED, in-memory projection (funded balance + probability-weighted pipeline,
the deterministic bridge) used only for view breakdowns / queries.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import pandas as pd

from analytics_lib.numeric import coerce_numeric

VIEWS = ("funded", "pipeline", "forecast")
DEFAULT_VIEW = "funded"

# Unqualified "amount"/"balance" resolves to this column per view. The pipeline
# prepared dataset and the forecast frame both carry the view's primary metric
# under ``current_outstanding_balance``, so the SAME query shape works per view.
PRIMARY_METRIC = {
    "funded": "current_outstanding_balance",
    "pipeline": "current_outstanding_balance",
    "forecast": "current_outstanding_balance",
}

# Dimension columns carried onto the derived forecast frame (intersection of the
# funded + pipeline canonical columns that MI queries stratify on).
_SHARED_DIMS = [
    "geographic_region_obligor", "collateral_geography", "ltv_bucket",
    "original_ltv_bucket", "broker_channel", "origination_channel",
    "current_loan_to_value", "current_interest_rate", "interest_rate_bucket",
    "age_bucket", "ticket_bucket", "expected_completion_month",
]


def resolve_active_view(question: str, dataset_context: Optional[str]) -> str:
    """The dataset/view a query runs against. Explicit wording in the question
    overrides the active tab; otherwise the tab (``dataset_context``) wins.

    Priority of explicit wording: forecast > pipeline > funded (so "forecast
    funded balance" routes to forecast, not funded)."""
    q = (question or "").lower()
    if "forecast" in q:
        return "forecast"
    if "pipeline" in q:
        return "pipeline"
    if "funded" in q:
        return "funded"
    ctx = (dataset_context or "").strip().lower()
    return ctx if ctx in VIEWS else DEFAULT_VIEW


def build_forecast_view_frame(funded_df: Optional[pd.DataFrame],
                              pipeline_df: Optional[pd.DataFrame]) -> pd.DataFrame:
    """Derived forecast frame: one row per funded loan (contribution = funded
    balance) and per pipeline case (contribution = weighted expected funded
    amount), carrying shared dimensions. ``current_outstanding_balance`` holds the
    forecast contribution so any "X by dimension" query yields forecast X by
    dimension. NOT persisted; never written to the spine.
    """
    parts: List[pd.DataFrame] = []
    if funded_df is not None and len(funded_df):
        f = pd.DataFrame(index=funded_df.index)
        f["current_outstanding_balance"] = coerce_numeric(
            funded_df.get("current_outstanding_balance", pd.Series(dtype=float)))
        for d in _SHARED_DIMS:
            if d in funded_df.columns:
                f[d] = funded_df[d].values
        f["state_component"] = "funded"
        parts.append(f)
    if pipeline_df is not None and len(pipeline_df):
        p = pd.DataFrame(index=pipeline_df.index)
        # The forecast CONTRIBUTION of a pipeline case is its weighted expected
        # funded amount (amount x completion probability) — not its gross amount.
        p["current_outstanding_balance"] = coerce_numeric(
            pipeline_df.get("weighted_expected_funded_amount", pd.Series(dtype=float)))
        for d in _SHARED_DIMS:
            if d in pipeline_df.columns:
                p[d] = pipeline_df[d].values
        p["state_component"] = "forecast_pipeline"
        # Drop pipeline rows with no weightable contribution (withdrawn/unknown).
        p = p[coerce_numeric(p["current_outstanding_balance"]).notna()]
        parts.append(p)
    if not parts:
        return pd.DataFrame(columns=["current_outstanding_balance", "state_component"])
    return pd.concat(parts, ignore_index=True)


def _dim_sum(df: Optional[pd.DataFrame], dim: str, col: str) -> Dict[str, float]:
    if df is None or dim not in df.columns or col not in df.columns:
        return {}
    amt = coerce_numeric(df[col])
    grp = amt.groupby(df[dim].astype(str)).sum()
    return {str(k): float(v) for k, v in grp.items()
            if str(k).strip() and str(k) not in ("nan", "NaT", "None")}


def forecast_dimension_breakdown(funded_df: Optional[pd.DataFrame],
                                 pipeline_df: Optional[pd.DataFrame],
                                 dim: str) -> List[Dict[str, Any]]:
    """``[{key, fundedAmount, weightedPipelineAmount, forecastAmount}]`` for one
    dimension — funded exposure + weighted expected pipeline = forecast. Derived
    by aggregate composition (no row merge), ordered by forecast amount desc."""
    funded = _dim_sum(funded_df, dim, "current_outstanding_balance")
    pipe = _dim_sum(pipeline_df, dim, "weighted_expected_funded_amount")
    keys = set(funded) | set(pipe)
    rows = []
    for k in keys:
        fa = round(funded.get(k, 0.0), 2)
        wp = round(pipe.get(k, 0.0), 2)
        rows.append({"key": k, "fundedAmount": fa, "weightedPipelineAmount": wp,
                     "forecastAmount": round(fa + wp, 2)})
    rows.sort(key=lambda r: r["forecastAmount"], reverse=True)
    return rows


def forecast_breakdowns(funded_df: Optional[pd.DataFrame],
                        pipeline_df: Optional[pd.DataFrame]) -> Dict[str, Any]:
    """Forecast-by-dimension breakdowns for the Forecast view (region / LTV /
    completion month), capped where long."""
    from .pipeline_contract import cap_breakdown
    region = forecast_dimension_breakdown(funded_df, pipeline_df, "geographic_region_obligor")
    ltv = forecast_dimension_breakdown(funded_df, pipeline_df, "ltv_bucket")
    # Completion-month: pipeline contributes weighted by month; funded is "now".
    month = _dim_sum(pipeline_df, "expected_completion_month", "weighted_expected_funded_amount")
    by_month = [{"month": k, "weightedExpectedFundedAmount": round(v, 2)}
                for k, v in sorted(month.items())]
    # Re-cap region/ltv to top 10 for the visual, keyed on forecastAmount.
    def _cap(rows):
        capped = cap_breakdown(
            [{"key": r["key"], "caseCount": 0, "pipelineAmount": r["forecastAmount"],
              "weightedExpectedFundedAmount": r["weightedPipelineAmount"]} for r in rows], 10)
        return capped
    return {
        "byRegion": region,
        "byLtvBucket": ltv,
        "byCompletionMonth": by_month,
        "byRegionCapped": _cap(region),
        "byLtvBucketCapped": _cap(ltv),
    }


# --------------------------------------------------------------------------- #
# Lineage ("How calculated") per view — from existing metadata.
# --------------------------------------------------------------------------- #
def lineage_for(view: str, *, funded_reporting_date: Optional[str] = None,
                pipeline_as_of_date: Optional[str] = None,
                completion_probability_basis: Optional[str] = None,
                source_file: Optional[str] = None,
                pipeline_source_folder_date: Optional[str] = None,
                current_pipeline_snapshot_date: Optional[str] = None,
                current_pipeline_source_file: Optional[str] = None,
                historical_model_evidence: Optional[Dict[str, Any]] = None
                ) -> Dict[str, Any]:
    """Per-view "How calculated" lineage. The pipeline / forecast views carry the
    historical completion-model evidence and keep distinct concepts separate: the
    funded reporting date, the CURRENT pipeline snapshot (latest weekly extract +
    its file), and the historical probability observation window (start/end)."""
    if view == "funded":
        return {
            "view": "funded",
            "source": "18_central_lender_tape.csv",
            "metric": "current_outstanding_balance",
            "fundedReportingDate": funded_reporting_date,
            "explanation": "Funded book actuals from the governed central lender tape.",
        }
    evidence = historical_model_evidence or {}
    # The current snapshot date is the latest weekly extract; fall back to the as-of.
    snapshot_date = current_pipeline_snapshot_date or pipeline_as_of_date
    # Observation window: prefer the dedup inventory window, else the model evidence.
    window_start = evidence.get("observationWindowStart")
    window_end = evidence.get("observationWindowEnd")
    common = {
        "completionProbabilityBasis": completion_probability_basis,
        # Current pipeline snapshot (latest weekly extract) — distinct from window.
        "currentPipelineSnapshotDate": snapshot_date,
        "currentPipelineSourceFile": current_pipeline_source_file
            or (source_file.split("/")[-1] if source_file else None),
        "pipelineSourceFolderDate": pipeline_source_folder_date,
        # Historical probability observation window — distinct from the as-of date.
        "historicalObservationWindowStart": window_start,
        "historicalObservationWindowEnd": window_end,
        "observationWindowStart": window_start,
        "observationWindowEnd": window_end,
        "uniqueWeeklyExtractsUsed": evidence.get("uniqueWeeklyExtractsUsed"),
        "sourceFilesScanned": evidence.get("sourceFilesScanned"),
        "historicalModelEvidence": evidence,
    }
    if view == "pipeline":
        return {
            "view": "pipeline",
            "source": source_file or "governed weekly pipeline files",
            "metric": "expected_funded_amount",
            "weightedMetric": "expected_funded_amount × completion_probability",
            "pipelineAsOfDate": snapshot_date,
            **common,
            "explanation": "Origination pipeline (pre-funded), governed weekly extract; "
                           "completion probabilities from the historical weekly-snapshot model.",
        }
    return {
        "view": "forecast",
        "metric": "forecast_funded_balance",
        "formula": "forecast funded balance = funded balance + Σ(expected_funded_amount × completion_probability)",
        "fundedReportingDate": funded_reporting_date,
        "pipelineAsOfDate": snapshot_date,
        **common,
        "explanation": "Deterministic bridge: funded actuals + probability-weighted pipeline.",
    }
