"""Reconciliation helpers between pipeline-completed rows and funded stock rows.

First-pass principles:
- Pipeline COMPLETED is not funded truth; funded tape remains truth source.
- Reconciliation is deterministic and transparent, not fuzzy.
- Matching priority:
  1) pipeline.account_number == funded.loan_policy_number
  2) pipeline.account_number == funded.loan_identifier + '01'
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import pandas as pd


@dataclass(frozen=True)
class ReconciliationSummary:
    completed_pipeline_rows: int
    matched_pipeline_rows: int
    unmatched_pipeline_rows: int
    funded_rows: int
    funded_rows_not_seen_in_pipeline_completed: int


def _clean_text(series: pd.Series) -> pd.Series:
    return (
        series.astype("string")
        .fillna("")
        .str.strip()
        .str.replace(r"\s+", " ", regex=True)
    )


def _derive_funded_link_columns(funded_df: pd.DataFrame) -> pd.DataFrame:
    out = funded_df.copy()

    # Prefer explicit loan policy number when present.
    if "loan_policy_number" in out.columns:
        out["funded_loan_policy_number"] = _clean_text(out["loan_policy_number"])
    else:
        out["funded_loan_policy_number"] = pd.Series("", index=out.index, dtype="string")

    # loan_identifier is core canonical in existing funded flow.
    if "loan_identifier" in out.columns:
        out["funded_loan_identifier"] = _clean_text(out["loan_identifier"])
    else:
        out["funded_loan_identifier"] = pd.Series("", index=out.index, dtype="string")

    out["funded_account_from_loan_id"] = (
        out["funded_loan_identifier"].where(out["funded_loan_identifier"] != "", other="") + "01"
    )

    return out


def reconcile_completed_pipeline_to_funded(
    pipeline_df: pd.DataFrame,
    funded_df: pd.DataFrame,
) -> pd.DataFrame:
    """Return row-level reconciliation results for pipeline COMPLETED rows."""

    if pipeline_df is None or pipeline_df.empty:
        return pd.DataFrame()

    completed = pipeline_df[pipeline_df.get("pipeline_stage", "").astype(str).str.upper() == "COMPLETED"].copy()
    if completed.empty:
        return pd.DataFrame()

    if "account_number" not in completed.columns:
        completed["account_number"] = ""
    completed["account_number"] = _clean_text(completed["account_number"])

    funded = _derive_funded_link_columns(funded_df if funded_df is not None else pd.DataFrame())

    # Rule 1: account number = funded loan policy number
    by_policy = completed.merge(
        funded[["funded_loan_policy_number", "funded_loan_identifier"]],
        left_on="account_number",
        right_on="funded_loan_policy_number",
        how="left",
        indicator=False,
    )
    by_policy["match_rule"] = by_policy["funded_loan_policy_number"].apply(
        lambda v: "ACCOUNT=LOAN_POLICY_NUMBER" if isinstance(v, str) and v.strip() else ""
    )

    # Rule 2 fallback: account number = loan_identifier + '01'
    needs_fallback = by_policy["match_rule"].eq("")
    if needs_fallback.any():
        fallback = by_policy.loc[needs_fallback].drop(columns=["funded_loan_policy_number", "funded_loan_identifier", "match_rule"])
        fallback = fallback.merge(
            funded[["funded_account_from_loan_id", "funded_loan_identifier"]],
            left_on="account_number",
            right_on="funded_account_from_loan_id",
            how="left",
            indicator=False,
        )
        fallback["match_rule"] = fallback["funded_account_from_loan_id"].apply(
            lambda v: "ACCOUNT=LOAN_ID_PLUS_01" if isinstance(v, str) and v.strip() else ""
        )

        by_policy.loc[needs_fallback, "funded_loan_identifier"] = fallback["funded_loan_identifier"].values
        by_policy.loc[needs_fallback, "match_rule"] = fallback["match_rule"].values

    by_policy["match_status"] = by_policy["match_rule"].apply(lambda v: "MATCHED" if v else "UNMATCHED")

    keep_cols = [
        c for c in [
            "pipeline_opportunity_id",
            "snapshot_date",
            "pipeline_stage",
            "status_raw",
            "dpr_status_raw",
            "kfi_number",
            "account_number",
            "loan_amount",
            "broker",
            "property_region",
            "match_status",
            "match_rule",
            "funded_loan_identifier",
        ] if c in by_policy.columns
    ]

    return by_policy[keep_cols].copy()


def summarize_reconciliation(recon_df: pd.DataFrame, funded_df: Optional[pd.DataFrame] = None) -> ReconciliationSummary:
    completed_pipeline_rows = int(len(recon_df)) if recon_df is not None else 0
    matched = int((recon_df["match_status"] == "MATCHED").sum()) if completed_pipeline_rows else 0
    unmatched = completed_pipeline_rows - matched

    funded_rows = int(len(funded_df)) if funded_df is not None else 0
    funded_rows_not_seen = 0

    if funded_df is not None and not funded_df.empty and recon_df is not None and not recon_df.empty:
        funded_ids = set(
            _clean_text(funded_df.get("loan_identifier", pd.Series([], dtype="string"))).dropna().tolist()
        )
        matched_ids = set(
            _clean_text(recon_df.get("funded_loan_identifier", pd.Series([], dtype="string"))).dropna().tolist()
        )
        funded_rows_not_seen = int(len(funded_ids - matched_ids))

    return ReconciliationSummary(
        completed_pipeline_rows=completed_pipeline_rows,
        matched_pipeline_rows=matched,
        unmatched_pipeline_rows=unmatched,
        funded_rows=funded_rows,
        funded_rows_not_seen_in_pipeline_completed=funded_rows_not_seen,
    )
