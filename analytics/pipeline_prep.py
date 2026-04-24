"""Utilities for normalizing weekly pipeline snapshot files.

Design notes:
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

from analytics.portfolio_semantics import normalize_region_labels, region_codes_from_labels, safe_ltv_percent


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


def _choose_snapshot_date(df: pd.DataFrame, configured_snapshot_date: Optional[str]) -> pd.Series:
    if configured_snapshot_date:
        ts = pd.to_datetime(configured_snapshot_date, errors="coerce")
        if pd.notna(ts):
            return pd.Series(pd.Timestamp(ts).normalize(), index=df.index, dtype="datetime64[ns]")

    if "snapshot_date" in df.columns:
        parsed = pd.to_datetime(df["snapshot_date"], errors="coerce")
        if parsed.notna().any():
            fallback = parsed.dropna().max().normalize()
            parsed = parsed.fillna(fallback)
            return parsed.dt.normalize()

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
            return pd.Series(pd.Timestamp(max_date).normalize(), index=df.index, dtype="datetime64[ns]")

    return pd.Series(pd.Timestamp(date.today()), index=df.index, dtype="datetime64[ns]")


def _synthetic_opportunity_key(df: pd.DataFrame) -> pd.Series:
    """Build deterministic key for pipeline opportunity rows."""

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
        "KFI=" + kfi
        + "|ACC=" + acc
        + "|BROKER=" + broker
        + "|PROD=" + product
        + "|AMT=" + amt.round(2).astype(str)
        + "|APP_DT=" + app_dt
    )

    return seeds.apply(lambda x: hashlib.sha1(x.encode("utf-8")).hexdigest())


def _derive_stage_date(out: pd.DataFrame) -> pd.Series:
    stage = out["stage"].astype("string")
    stage_date = pd.Series(pd.NaT, index=out.index, dtype="datetime64[ns]")

    stage_date.loc[stage.eq("KFI")] = out.get("kfi_submitted_date")
    stage_date.loc[stage.eq("APPLICATION")] = out.get("application_submitted_date")
    stage_date.loc[stage.eq("OFFER")] = out.get("offer_date")
    stage_date.loc[stage.eq("COMPLETED")] = out.get("date_funds_released")

    with_fallback = stage_date.isna()
    if with_fallback.any():
        fallback_cols = [
            c
            for c in ["date_funds_released", "offer_date", "application_submitted_date", "kfi_submitted_date"]
            if c in out.columns
        ]
        if fallback_cols:
            stage_date.loc[with_fallback] = out.loc[with_fallback, fallback_cols].bfill(axis=1).iloc[:, 0]

    return pd.to_datetime(stage_date, errors="coerce")



def normalize_pipeline_snapshot(raw_df: pd.DataFrame, config: Optional[PipelinePrepConfig] = None) -> pd.DataFrame:
    """Normalize weekly pipeline snapshot into canonical representation."""
    config = config or PipelinePrepConfig()

    if raw_df is None or raw_df.empty:
        return pd.DataFrame()

    out = raw_df.copy()

    rename_map = {
        "Snapshot Date": "snapshot_date",
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
        "Borrower Age": "borrower_age",
        "Youngest Borrower Age": "youngest_borrower_age",
    }
    out = out.rename(columns={k: v for k, v in rename_map.items() if k in out.columns})

    for c in [
        "snapshot_date",
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
        "borrower_age",
        "youngest_borrower_age",
    ]:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")

    if "youngest_borrower_age" not in out.columns and "borrower_age" in out.columns:
        out["youngest_borrower_age"] = out["borrower_age"]

    for c in ["status_raw", "dpr_status_raw", "broker", "product", "account_number", "property_region"]:
        if c not in out.columns:
            out[c] = ""
        out[c] = _clean_text(out[c])

    out["stage"] = _map_stage(out["status_raw"])
    out["pipeline_stage"] = out["stage"]  # backward-compatible alias for existing tab code
    out["is_funded_stage"] = out["stage"].eq("COMPLETED")
    out["is_terminal_stage"] = out["stage"].isin(["COMPLETED", "WITHDRAWN"])
    out["is_live_stage"] = ~out["is_terminal_stage"]

    rej_a = _clean_text(out.get("rejection_reason_a", pd.Series("", index=out.index)))
    rej_b = _clean_text(out.get("rejection_reason_b", pd.Series("", index=out.index)))
    has_rejection_detail = (rej_a != "") | (rej_b != "")
    out["termination_reason_class"] = pd.Series("", index=out.index, dtype="string")
    out.loc[out["stage"].eq("WITHDRAWN"), "termination_reason_class"] = "WITHDRAWN"
    out.loc[has_rejection_detail & out["is_terminal_stage"], "termination_reason_class"] = "REJECTED"

    out["snapshot_date"] = _choose_snapshot_date(out, config.snapshot_date)
    out["pipeline_opportunity_id"] = _synthetic_opportunity_key(out)

    out["stage_date"] = _derive_stage_date(out)
    out["days_in_stage"] = (
        (pd.to_datetime(out["snapshot_date"], errors="coerce") - pd.to_datetime(out["stage_date"], errors="coerce"))
        .dt.days
        .clip(lower=0)
    )

    if "loan_amount" not in out.columns:
        out["loan_amount"] = pd.NA
    if "estimated_value" not in out.columns:
        out["estimated_value"] = pd.NA

    out["current_ltv"] = safe_ltv_percent(out["loan_amount"], out["estimated_value"])

    out["property_region"] = normalize_region_labels(out["property_region"])
    out["property_region_code"] = region_codes_from_labels(out["property_region"])

    if "product_rate" not in out.columns:
        out["product_rate"] = pd.NA

    return out
