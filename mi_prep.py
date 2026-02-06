# mi_prep.py
"""MI preparation layer for the ERM dashboard.

Contract:
- INPUT must be the *trusted canonical output* from the pipeline (post transform + typing).
- This module must NOT change the economic meaning of canonical fields.
- Allowed operations: light type conversion for charting, creation of presentation aliases,
  bucketing, grouping helpers.

Notes:
- If you need to correct / standardise / derive *economic truth* (e.g., ISO dates,
  balances, currency codes, LTV), do it in canonical_transform, not here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# ----------------------------
# Formatting helpers
# ----------------------------
def format_currency(value: float, symbol: str = "£") -> str:
    """Format numeric value as a compact currency string for UI."""
    if pd.isna(value):
        return "N/A"
    try:
        v = float(value)
    except Exception:
        return "N/A"

    if abs(v) >= 1_000_000:
        return f"{symbol}{v / 1_000_000:.1f}M"
    if abs(v) >= 1_000:
        return f"{symbol}{v / 1_000:.0f}K"
    return f"{symbol}{v:.0f}"


def weighted_average(series: pd.Series, weights: pd.Series) -> float:
    """Weighted average with NaN-safe masking."""
    mask = series.notna() & weights.notna()
    if not mask.any():
        return np.nan
    return float(np.average(series[mask], weights=weights[mask]))


# ----------------------------
# Canonical-only guards
# ----------------------------
@dataclass(frozen=True)
class CanonicalCheckResult:
    ok: bool
    missing_required: List[str]
    notes: List[str]


def assert_trusted_canonical(
    df: pd.DataFrame,
    required_core_fields: Optional[List[str]] = None,
) -> CanonicalCheckResult:
    """Check whether df looks like pipeline output (best-effort).

    This is intentionally lightweight. It should prevent accidental upload of raw lender files
    into the dashboard, which causes 'two truths' drift.
    """
    required_core_fields = required_core_fields or [
        "loan_identifier",
        "data_cut_off_date",
    ]
    missing = [c for c in required_core_fields if c not in df.columns]
    notes: List[str] = []
    if missing:
        notes.append("Missing core fields; input may not be canonical pipeline output.")
    # crude raw-file heuristics (non-blocking)
    raw_smells = [c for c in df.columns if c.strip().lower() in {"loan id", "loan policy number", "date of completion"}]
    if raw_smells:
        notes.append("Input has lender-style headers; ensure you are using the pipeline canonical output.")
    return CanonicalCheckResult(ok=(len(missing) == 0), missing_required=missing, notes=notes)


# ----------------------------
# Presentation aliases
# ----------------------------
def add_presentation_aliases(df: pd.DataFrame) -> pd.DataFrame:
    """Add presentation-friendly aliases used by charts."""
    out = df.copy()

    # ========== ADD THIS AT THE START ==========
    # Ensure loan_id exists (many charts expect this)
    if "loan_id" not in out.columns:
        if "loan_identifier" in out.columns:
            out["loan_id"] = out["loan_identifier"]
        elif "unique_identifier" in out.columns:
            out["loan_id"] = out["unique_identifier"]
        else:
            out["loan_id"] = range(len(out))
    # ========== END LOAN_ID ALIAS ==========

    # Exposure weighting column for charts
    if "total_balance" not in out.columns:
        if "current_outstanding_balance" in out.columns:
            out["total_balance"] = pd.to_numeric(out["current_outstanding_balance"], errors="coerce")
        elif "current_principal_balance" in out.columns:
            out["total_balance"] = pd.to_numeric(out["current_principal_balance"], errors="coerce")
        else:
            out["total_balance"] = np.nan

    # Primary geography key for stratifications
    if "geographic_region" not in out.columns:
        if "geographic_region_classification" in out.columns:
            out["geographic_region"] = out["geographic_region_classification"]
        elif "geographic_region_obligor" in out.columns:
            out["geographic_region"] = out["geographic_region_obligor"]
        elif "geographic_region_collateral" in out.columns:
            out["geographic_region"] = out["geographic_region_collateral"]
        else:
            out["geographic_region"] = "Unknown"

    if "origination_date" in out.columns:
        od = pd.to_datetime(out["origination_date"], errors="coerce", dayfirst=True)
        out["origination_year"] = od.dt.year.astype("Int64")
        out["origination_month"] = od.dt.to_period("M")  # keep as Period, not string

    return out


# ----------------------------
# Bucketing for charts
# ----------------------------
def add_buckets(df: pd.DataFrame) -> pd.DataFrame:
    """Create buckets for standard stratifications."""
    out = df.copy()

    # LTV bucket (existing code - keep as is)
    if "current_loan_to_value" in out.columns and "ltv_bucket" not in out.columns:
        ltv = pd.to_numeric(out["current_loan_to_value"], errors="coerce")
        ltv_scaled = ltv.copy()
        if ltv.median() <= 1.0: 
             ltv_scaled = ltv * 100.0
        bins = [0, 20, 30, 40, 50, 60, 70, 80, 200]
        labels = ["0-20%", "20-30%", "30-40%", "40-50%", "50-60%", "60-70%", "70-80%", "80%+"]
        out["ltv_bucket"] = pd.cut(ltv_scaled, bins=bins, labels=labels, include_lowest=True)

    # Ticket size bucket (uses total_balance as exposure proxy)
    if "total_balance" in out.columns and "ticket_bucket" not in out.columns:
        bal = pd.to_numeric(out["total_balance"], errors="coerce")
        # Fill NaN with 0 before bucketing
        bal = bal.fillna(0)
        bins = [0, 50_000, 100_000, 150_000, 200_000, 300_000, 500_000, np.inf]
        labels = ["<50k", "50-100k", "100-150k", "150-200k", "200-300k", "300-500k", "500k+"]
        out["ticket_bucket"] = pd.cut(bal, bins=bins, labels=labels, include_lowest=True)
        # Replace any remaining NaN with "Unknown"
        out["ticket_bucket"] = out["ticket_bucket"].cat.add_categories(["Unknown"]).fillna("Unknown")
    
# Original LTV bucket
    if "original_loan_to_value" in out.columns and "original_ltv_bucket" not in out.columns:
        ol = pd.to_numeric(out["original_loan_to_value"], errors="coerce")       
        bins = [0, 20, 30, 40, 50, 60, 70, 80, 200]
        labels = ["0-20%", "20-30%", "30-40%", "40-50%", "50-60%", "60-70%", "70-80%", "80%+"]
        out["original_ltv_bucket"] = pd.cut(ol, bins=bins, labels=labels, include_lowest=True)
        out["original_ltv_bucket"] = out["original_ltv_bucket"].cat.add_categories(["Unknown"]).fillna("Unknown")


    # Interest rate bucket (existing code - keep as is)
    if "current_interest_rate" in out.columns and "rate_bucket" not in out.columns:
        r = pd.to_numeric(out["current_interest_rate"], errors="coerce")
        r_dec = r.where(r <= 1, r / 100.0)
        bins = [0, 0.02, 0.03, 0.04, 0.05, 0.06, 0.08, np.inf]
        labels = ["<2%", "2-3%", "3-4%", "4-5%", "5-6%", "6-8%", "8%+"]
        out["rate_bucket"] = pd.cut(r_dec, bins=bins, labels=labels, include_lowest=True)

    # ========== ADD THIS: AGE BUCKET ==========
    if "youngest_borrower_age" in out.columns and "age_bucket" not in out.columns:
        age = pd.to_numeric(out["youngest_borrower_age"], errors="coerce")
        bins = [0, 55, 60, 65, 70, 75, 80, 85, 120]
        labels = ["<55", "55-60", "60-65", "65-70", "70-75", "75-80", "80-85", "85+"]
        out["age_bucket"] = pd.cut(age, bins=bins, labels=labels, include_lowest=True)
    # ========== END AGE BUCKET ==========

    return out
