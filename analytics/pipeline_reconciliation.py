"""Reconciliation helpers between pipeline rows and funded stock rows.

Principles:
- Pipeline stage is not funded truth; funded tape remains truth source.
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
    pipeline_rows: int
    matched_pipeline_rows: int
    unmatched_pipeline_rows: int
    funded_rows: int
    funded_rows_not_seen_in_pipeline: int


def _clean_text(series: pd.Series) -> pd.Series:
    return (
        series.astype("string")
        .fillna("")
        .str.strip()
        .str.replace(r"\s+", " ", regex=True)
    )


def _derive_funded_link_columns(funded_df: pd.DataFrame) -> pd.DataFrame:
    out = funded_df.copy()

    if "loan_policy_number" in out.columns:
        out["funded_loan_policy_number"] = _clean_text(out["loan_policy_number"])
    else:
        out["funded_loan_policy_number"] = pd.Series("", index=out.index, dtype="string")

    if "loan_identifier" in out.columns:
        out["funded_loan_identifier"] = _clean_text(out["loan_identifier"])
    else:
        out["funded_loan_identifier"] = pd.Series("", index=out.index, dtype="string")

    out["funded_account_from_loan_id"] = out["funded_loan_identifier"] + "01"

    return out


def reconcile_pipeline_to_funded(pipeline_df: pd.DataFrame, funded_df: pd.DataFrame) -> pd.DataFrame:
    """Return row-level reconciliation results for normalized pipeline rows."""
    if pipeline_df is None or pipeline_df.empty:
        return pd.DataFrame()

    out = pipeline_df.copy()
    if "account_number" not in out.columns:
        out["account_number"] = ""
    out["account_number"] = _clean_text(out["account_number"])

    funded = _derive_funded_link_columns(funded_df if funded_df is not None else pd.DataFrame())

    by_policy = out.merge(
        funded[["funded_loan_policy_number", "funded_loan_identifier"]],
        left_on="account_number",
        right_on="funded_loan_policy_number",
        how="left",
        indicator=False,
    )

    has_policy_match = by_policy["funded_loan_policy_number"].fillna("").astype("string").ne("")
    by_policy["reconciliation_match_rule"] = ""
    by_policy.loc[has_policy_match, "reconciliation_match_rule"] = "ACCOUNT=LOAN_POLICY_NUMBER"

    needs_fallback = by_policy["reconciliation_match_rule"].eq("")
    if needs_fallback.any():
        fallback = by_policy.loc[needs_fallback].drop(
            columns=["funded_loan_policy_number", "funded_loan_identifier", "reconciliation_match_rule"]
        )
        fallback = fallback.merge(
            funded[["funded_account_from_loan_id", "funded_loan_identifier"]],
            left_on="account_number",
            right_on="funded_account_from_loan_id",
            how="left",
            indicator=False,
        )
        has_fallback_match = fallback["funded_account_from_loan_id"].fillna("").astype("string").ne("")
        fallback["reconciliation_match_rule"] = ""
        fallback.loc[has_fallback_match, "reconciliation_match_rule"] = "ACCOUNT=LOAN_ID_PLUS_01"

        by_policy.loc[needs_fallback, "funded_loan_identifier"] = fallback["funded_loan_identifier"].values
        by_policy.loc[needs_fallback, "reconciliation_match_rule"] = fallback["reconciliation_match_rule"].values

    by_policy = by_policy.merge(
        funded[["funded_loan_identifier", "funded_loan_policy_number"]].drop_duplicates(),
        on="funded_loan_identifier",
        how="left",
        suffixes=("", "_linked"),
    )

    by_policy["matched_funded_identifier"] = _clean_text(by_policy["funded_loan_identifier"])
    by_policy["matched_funded_policy_number"] = _clean_text(
        by_policy["funded_loan_policy_number_linked"].fillna(by_policy["funded_loan_policy_number"])
    )

    by_policy["is_reconciled_to_funded"] = by_policy["reconciliation_match_rule"].ne("")
    by_policy["reconciliation_match_status"] = by_policy["is_reconciled_to_funded"].map(
        {True: "MATCHED", False: "UNMATCHED"}
    )

    # Backward-compatible aliases consumed by first-pass tab renderer.
    by_policy["match_status"] = by_policy["reconciliation_match_status"]
    by_policy["match_rule"] = by_policy["reconciliation_match_rule"]
    by_policy["funded_loan_identifier"] = by_policy["matched_funded_identifier"]

    keep_cols = [
        c
        for c in [
            "pipeline_opportunity_id",
            "snapshot_date",
            "stage",
            "pipeline_stage",
            "status_raw",
            "account_number",
            "loan_amount",
            "broker",
            "product",
            "property_region",
            "property_region_code",
            "is_reconciled_to_funded",
            "reconciliation_match_status",
            "reconciliation_match_rule",
            "matched_funded_identifier",
            "matched_funded_policy_number",
            "match_status",
            "match_rule",
            "funded_loan_identifier",
        ]
        if c in by_policy.columns
    ]
    return by_policy[keep_cols].copy()


def reconcile_completed_pipeline_to_funded(pipeline_df: pd.DataFrame, funded_df: pd.DataFrame) -> pd.DataFrame:
    """Backward-compatible wrapper for completed-only reconciliation."""
    if pipeline_df is None or pipeline_df.empty:
        return pd.DataFrame()

    stage_col = "stage" if "stage" in pipeline_df.columns else "pipeline_stage"
    completed = pipeline_df[pipeline_df.get(stage_col, "").astype(str).str.upper() == "COMPLETED"].copy()
    return reconcile_pipeline_to_funded(completed, funded_df)


def summarize_reconciliation(recon_df: pd.DataFrame, funded_df: Optional[pd.DataFrame] = None) -> ReconciliationSummary:
    pipeline_rows = int(len(recon_df)) if recon_df is not None else 0
    matched = int(recon_df.get("is_reconciled_to_funded", pd.Series(dtype=bool)).sum()) if pipeline_rows else 0
    unmatched = pipeline_rows - matched

    funded_rows = int(len(funded_df)) if funded_df is not None else 0
    funded_rows_not_seen = 0

    if funded_df is not None and not funded_df.empty:
        funded_ids = set(_clean_text(funded_df.get("loan_identifier", pd.Series([], dtype="string"))).tolist())
        matched_ids = set(_clean_text(recon_df.get("matched_funded_identifier", pd.Series([], dtype="string"))).tolist())
        funded_rows_not_seen = int(len({x for x in funded_ids if x} - {x for x in matched_ids if x}))

    return ReconciliationSummary(
        pipeline_rows=pipeline_rows,
        matched_pipeline_rows=matched,
        unmatched_pipeline_rows=unmatched,
        funded_rows=funded_rows,
        funded_rows_not_seen_in_pipeline=funded_rows_not_seen,
    )
