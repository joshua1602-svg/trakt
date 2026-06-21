"""Funded MI data preparation — promoted central lender tape -> analytics-ready.

The promoted ``18_central_lender_tape.csv`` is a thin canonical funded tape: it
carries per-loan canonical fields (balance, valuation, rate, dates, currency) but
not the derived MI **dimensions** the semantic layer strat-charts on (LTV / age /
ticket / rate / vintage / region buckets).

This step derives the bucket *source* fields that the tape supports and then runs
the EXISTING, canonical bucketing engine (``analytics_lib.buckets`` over
``config/mi/buckets.yaml``) — the same engine the Streamlit dashboard and the API
demo path use — so Streamlit / React / API stay consistent. No bucket edges or
React-side bucketing are introduced here.

Derivable from the funded tape:
  * ``current_loan_to_value``      = current_outstanding_balance / current_valuation_amount  -> ltv_bucket
  * ``months_on_book``             = months(reporting_date - origination_date)                -> time_on_book_bucket
  * ``vintage_year``               = year(origination_date)
  * ``original_loan_to_value``     = original_principal_balance / original_valuation_amount   -> original_ltv_bucket (when present)
Directly bucketable (already present):
  * ``current_interest_rate``      -> interest_rate_bucket
  * ``current_outstanding_balance``-> ticket_bucket

Unavailable without more source data (reported as missing): age (no borrower age),
region (no geography), pd/lgd/ead/term/arrears, origination channel / broker.
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import pandas as pd

# Core funded stratification dimensions the funded MI dashboard aims to provide.
CORE_FUNDED_DIMENSIONS = [
    "ltv_bucket", "interest_rate_bucket", "ticket_bucket", "time_on_book_bucket",
    "vintage_year", "age_bucket", "original_ltv_bucket",
    "geographic_region_obligor", "origination_channel",
]


def _to_num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def _derive_source_fields(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    """Derive bucket source fields the funded tape supports (numeric-safe)."""
    out = df.copy()
    derived: List[str] = []
    cols = set(out.columns)

    if "current_loan_to_value" not in cols and {
        "current_outstanding_balance", "current_valuation_amount"
    } <= cols:
        bal = _to_num(out["current_outstanding_balance"])
        val = _to_num(out["current_valuation_amount"])
        ltv = bal / val.where(val > 0)
        if ltv.notna().any():
            out["current_loan_to_value"] = ltv
            derived.append("current_loan_to_value")

    if "origination_date" in cols:
        od = pd.to_datetime(out["origination_date"], errors="coerce", dayfirst=True)
        if od.notna().any():
            if "vintage_year" not in cols:
                out["vintage_year"] = od.dt.year.astype("Int64")
                derived.append("vintage_year")
            rep_col = next((c for c in ("reporting_date", "data_cut_off_date", "cut_off_date")
                            if c in cols), None)
            if rep_col and "months_on_book" not in cols:
                rd = pd.to_datetime(out[rep_col], errors="coerce")
                mob = (rd.dt.year - od.dt.year) * 12 + (rd.dt.month - od.dt.month)
                if mob.notna().any():
                    out["months_on_book"] = mob.astype("Int64")
                    derived.append("months_on_book")

    if "original_loan_to_value" not in cols and {
        "original_principal_balance", "original_valuation_amount"
    } <= cols:
        ob = _to_num(out["original_principal_balance"])
        ov = _to_num(out["original_valuation_amount"])
        oltv = ob / ov.where(ov > 0)
        if oltv.notna().any():
            out["original_loan_to_value"] = oltv
            derived.append("original_loan_to_value")

    return out, derived


def prepare_funded_mi_dataset(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Return ``(analytics_ready_df, report)`` for a funded central lender tape.

    Reuses the canonical ``analytics_lib.buckets`` engine + ``config/mi/buckets.yaml``;
    never duplicates bucket edges. ``report`` describes what preparation produced.
    """
    out, derived = _derive_source_fields(df)

    applied: Dict[str, Any] = {}
    issues: List[Dict[str, Any]] = []
    try:
        from analytics_lib.buckets import load_bucket_config, materialise_buckets
        out, issues, applied = materialise_buckets(
            out, load_bucket_config(), target="semantic_field")
    except Exception as exc:  # bucketing is additive; never block the dataset
        issues = [{"bucket": "*", "code": "engine_error", "severity": "error",
                   "detail": str(exc)}]

    produced = sorted({c for c in applied.values() if c and c in out.columns})
    extra_dims = [d for d in ("vintage_year",) if d in out.columns]
    available = sorted(set(produced) | set(extra_dims))
    missing = [d for d in CORE_FUNDED_DIMENSIONS if d not in out.columns]

    report = {
        "preparation_applied": True,
        "derived_fields": derived,
        "buckets_applied": {k: v for k, v in applied.items() if v},
        "dimensions_available": available,
        "missing_dimensions": missing,
        "bucket_errors": [i for i in (issues or []) if i.get("severity") == "error"][:20],
    }
    return out, report
