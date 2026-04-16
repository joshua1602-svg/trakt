"""Utilities for normalizing weekly pipeline snapshot files.

Design notes for first-pass implementation:
- Treat source files as snapshot truth (not event logs).
- Preserve raw stage values and map into a canonical stage taxonomy.
- Build a synthetic immutable opportunity key for cross-week tracking.
- Keep this module decoupled from Streamlit UI so it can be reused by
  scheduled jobs and future non-ERM adapters.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
import hashlib
from typing import Optional

import pandas as pd


CANONICAL_STAGE_ORDER = ["KFI", "APPLICATION", "OFFER", "COMPLETED", "WITHDRAWN", "OTHER"]


@dataclass(frozen=True)
class PipelinePrepConfig:
    """Configuration options for pipeline normalization."""

    snapshot_date: Optional[str] = None


def _clean_text(series: pd.Series) -> pd.Series:
    return (
        series.astype("string")
        .fillna("")
        .str.strip()
        .str.replace(r"\s+", " ", regex=True)
    )


def _parse_date(series: pd.Series) -> pd.Series:
    s = series.astype("string").str.strip()
    iso_mask = s.str.match(r"^\d{4}-\d{2}-\d{2}")
    out = pd.Series(pd.NaT, index=series.index, dtype="datetime64[ns]")
    if iso_mask.any():
        out.loc[iso_mask] = pd.to_datetime(s.loc[iso_mask], errors="coerce", format="%Y-%m-%d")
    non_iso = ~iso_mask
    if non_iso.any():
        out.loc[non_iso] = pd.to_datetime(s.loc[non_iso], errors="coerce", dayfirst=True)
    return out


def _map_stage(raw_status: pd.Series) -> pd.Series:
    s = _clean_text(raw_status).str.upper()

    mapped = pd.Series("OTHER", index=s.index, dtype="string")
    mapped.loc[s.str.contains("KFI", na=False)] = "KFI"
    mapped.loc[s.str.contains("APPLICATION", na=False)] = "APPLICATION"
    mapped.loc[s.str.contains("OFFER", na=False)] = "OFFER"
    mapped.loc[s.str.contains("COMPLET", na=False)] = "COMPLETED"
    mapped.loc[s.str.contains("WITHDRAW", na=False)] = "WITHDRAWN"

    return mapped


def _choose_snapshot_date(df: pd.DataFrame, configured_snapshot_date: Optional[str]) -> pd.Timestamp:
    if configured_snapshot_date:
        ts = pd.to_datetime(configured_snapshot_date, errors="coerce")
        if pd.notna(ts):
            return pd.Timestamp(ts).normalize()

    # Snapshot inference is explicit and conservative: choose latest known
    # milestone date in the file when available; otherwise today's date.
    date_candidates = [
        "kfi_submitted_date",
        "application_submitted_date",
        "offer_date",
        "date_funds_released",
    ]

    available = [c for c in date_candidates if c in df.columns]
    if available:
        max_date = pd.to_datetime(df[available].stack(), errors="coerce").max()
        if pd.notna(max_date):
            return pd.Timestamp(max_date).normalize()

    return pd.Timestamp(date.today())


def _synthetic_opportunity_key(df: pd.DataFrame) -> pd.Series:
    """Build deterministic key for pipeline opportunity rows.

    Preference order in composite seed:
      - KFI Number
      - Account Number
      - Broker + Product + Loan Amount + Application Submitted Date

    This is intentionally deterministic and transparent for first-pass use.
    """

    kfi = _clean_text(df.get("kfi_number", pd.Series("", index=df.index)))
    acc = _clean_text(df.get("account_number", pd.Series("", index=df.index)))
    broker = _clean_text(df.get("broker", pd.Series("", index=df.index)))
    product = _clean_text(df.get("product", pd.Series("", index=df.index)))
    amt = pd.to_numeric(df.get("loan_amount", pd.Series(0, index=df.index)), errors="coerce").fillna(0)
    app_dt = (
        pd.to_datetime(df.get("application_submitted_date", pd.Series(pd.NaT, index=df.index)), errors="coerce")
        .dt.strftime("%Y-%m-%d")
        .fillna("")
    )

    seeds = (
        "KFI=" + kfi +
        "|ACC=" + acc +
        "|BROKER=" + broker +
        "|PROD=" + product +
        "|AMT=" + amt.round(2).astype(str) +
        "|APP_DT=" + app_dt
    )

    return seeds.apply(lambda x: hashlib.sha1(x.encode("utf-8")).hexdigest())


def normalize_pipeline_snapshot(raw_df: pd.DataFrame, config: Optional[PipelinePrepConfig] = None) -> pd.DataFrame:
    """Normalize weekly pipeline snapshot into a lightweight canonical representation."""
    config = config or PipelinePrepConfig()

    if raw_df is None or raw_df.empty:
        return pd.DataFrame()

    out = raw_df.copy()

    # Normalize source headers into snake_case used by downstream helpers.
    rename_map = {
        "Company": "company",
        "Pool": "pool",
        "Account Number": "account_number",
        "KFI Number": "kfi_number",
        "Broker": "broker",
        "KFI Submitted Date": "kfi_submitted_date",
        "Loan Amount": "loan_amount",
        "Estimated Value": "estimated_value",
        "Product": "product",
        "Product Rate": "product_rate",
        "Loan Plan": "loan_plan",
        "Facility": "facility",
        "Max Facility": "max_facility",
        "Max Entitlement": "max_entitlement",
        "Property Region": "property_region",
        "PEG Percentage": "peg_percentage",
        "Fees Added": "fees_added",
        "Property Value": "property_value",
        "Loan Purpose": "loan_purpose",
        "Loan Purpose Detail": "loan_purpose_detail",
        "Status": "status_raw",
        "DPR Status": "dpr_status_raw",
        "Application Submitted Date": "application_submitted_date",
        "Offer Date": "offer_date",
        "Date Funds Released": "date_funds_released",
        "Rejection Reason A": "rejection_reason_a",
        "Rejection Reason B": "rejection_reason_b",
        "KFI Used For App": "kfi_used_for_app",
        "Contracted Payment Period": "contracted_payment_period",
        "Interest Payment Percentage": "interest_payment_percentage",
    }
    out = out.rename(columns={k: v for k, v in rename_map.items() if k in out.columns})

    for c in [
        "kfi_submitted_date",
        "application_submitted_date",
        "offer_date",
        "date_funds_released",
    ]:
        if c in out.columns:
            out[c] = _parse_date(out[c])

    for c in [
        "loan_amount",
        "estimated_value",
        "product_rate",
        "max_facility",
        "max_entitlement",
        "peg_percentage",
        "fees_added",
        "property_value",
        "interest_payment_percentage",
    ]:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")

    if "status_raw" not in out.columns:
        out["status_raw"] = ""
    if "dpr_status_raw" not in out.columns:
        out["dpr_status_raw"] = ""

    out["pipeline_stage"] = _map_stage(out["status_raw"])
    out["is_funded_stage"] = out["pipeline_stage"].eq("COMPLETED")
    out["is_terminal_stage"] = out["pipeline_stage"].isin(["COMPLETED", "WITHDRAWN"])
    out["is_live_stage"] = ~out["is_terminal_stage"]

    # Preserve withdrawn distinctly; classify rejection only when reasons exist.
    rej_a = _clean_text(out.get("rejection_reason_a", pd.Series("", index=out.index)))
    rej_b = _clean_text(out.get("rejection_reason_b", pd.Series("", index=out.index)))
    has_rejection_detail = (rej_a != "") | (rej_b != "")
    out["termination_reason_class"] = pd.Series("", index=out.index, dtype="string")
    out.loc[out["pipeline_stage"].eq("WITHDRAWN"), "termination_reason_class"] = "WITHDRAWN"
    out.loc[has_rejection_detail & out["is_terminal_stage"], "termination_reason_class"] = "REJECTED"

    out["snapshot_date"] = _choose_snapshot_date(out, config.snapshot_date)
    out["pipeline_opportunity_id"] = _synthetic_opportunity_key(out)

    # Keep external references and linkage fields explicit.
    for c in ["kfi_number", "account_number"]:
        if c not in out.columns:
            out[c] = ""
        out[c] = _clean_text(out[c])

    return out
