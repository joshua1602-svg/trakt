"""mi_agent_api/cohorts.py — funded origination-vintage (static-pool) analysis.

Surfaces the per-vintage MI already derivable from the governed funded central
tape — balance, loan count, book share, and balance-weighted LTV / interest rate
/ months-on-book by origination year — using the shared cohort primitives
(:mod:`analytics_lib.cohort`) and the ``vintage_year`` / ``months_on_book`` fields
``funded_prep`` derives. Nothing is fabricated: redemption / completion /
performance curves are NOT computed in the MI path today, so this module does not
emit them. Each returned metric is present only when its source column exists, and
``metricsAvailable`` lists exactly what was computed.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from analytics_lib.numeric import coerce_numeric
from mi_agent.mi_dataset_profile import PERCENT_POINTS, percent_storage_scale

_BALANCE = "current_outstanding_balance"
_LTV = "current_loan_to_value"
_RATE = "current_interest_rate"
_MOB = "months_on_book"
_VINTAGE = "vintage_year"
_ORIG_DATE = "origination_date"

# Cohort dimensions (asset-class-agnostic). Each groups the static pool by a
# generic origination/risk attribute; metrics are identical across dimensions.
_AGE_BUCKET = "age_bucket"
_YOUNGEST_AGE = "youngest_borrower_age"
_ORIG_LTV_BUCKET = "original_ltv_bucket"
_LTV_BUCKET = "ltv_bucket"
_ORIG_LTV = "original_loan_to_value"
_ORIG_CHANNEL = "origination_channel"
_BROKER = "broker_channel"

_DIMENSION_LABELS = {
    "vintage": "Vintage", "age": "Borrower age",
    "ltv": "LTV band", "channel": "Origination channel",
}
# Fallback bands when the tape has no pre-bucketed column.
_AGE_BINS = [0, 60, 65, 70, 75, 80, 85, 200]
_AGE_LABELS = ["<60", "60–64", "65–69", "70–74", "75–79", "80–84", "85+"]
_LTV_BINS = [0, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 10]
_LTV_LABELS = ["<20%", "20–30%", "30–40%", "40–50%", "50–60%", "60–70%", "70–80%", "80%+"]


def _ltv_as_fraction(series: pd.Series) -> pd.Series:
    """LTV as a fraction (0.55) regardless of tape convention (0.55 or 55)."""
    v = coerce_numeric(series)
    if percent_storage_scale(series) == PERCENT_POINTS:
        v = v / 100.0
    return v


def _has_values(df: pd.DataFrame, col: str) -> bool:
    return col in df.columns and coerce_numeric(df[col]).notna().any()


def _has_labels(df: pd.DataFrame, col: str) -> bool:
    if col not in df.columns:
        return False
    s = df[col].astype("string").str.strip()
    return s.replace("", pd.NA).notna().any()


def _weighted_avg(values: pd.Series, weights: pd.Series) -> Optional[float]:
    v = coerce_numeric(values)
    w = coerce_numeric(weights)
    mask = v.notna() & w.notna()
    denom = float(w[mask].sum())
    if denom == 0:
        return None
    return round(float((v[mask] * w[mask]).sum() / denom), 4)


def _weighted_avg_pct(values: pd.Series, weights: pd.Series) -> Optional[float]:
    """Balance-weighted average of a PERCENT column, normalised to a FRACTION
    (0.0955 == 9.55%). The funded tape stores LTV as a fraction but the interest
    rate in points (9.55), so a single ×100 formatter turned 9.55% into 955%.
    Detect the column's storage scale and emit a fraction so the UI's percent
    formatter renders every rate/LTV correctly regardless of tape convention."""
    wavg = _weighted_avg(values, weights)
    if wavg is None:
        return None
    if percent_storage_scale(values) == PERCENT_POINTS:
        return round(wavg / 100.0, 6)
    return wavg


def _vintage_series(df: pd.DataFrame, grain: str = "Y") -> Optional[pd.Series]:
    """The origination-cohort label per row at ``grain`` (Y|Q|M). Parsed from
    ``origination_date`` (finer grains need the date); falls back to the derived
    ``vintage_year`` for year grain. None when neither exists.

    A finer grain (quarter / month) is useful for a YOUNG book where every loan
    shares one origination year — a single 'Y' bucket hides the seasoning that
    'Q'/'M' reveals."""
    g = (grain or "Y").upper()
    if _ORIG_DATE in df.columns:
        od = pd.to_datetime(df[_ORIG_DATE], errors="coerce", dayfirst=True)
        if od.notna().any():
            if g == "Q":
                return (od.dt.year.astype("Int64").astype("string") + "-Q"
                        + od.dt.quarter.astype("Int64").astype("string"))
            if g == "M":
                return od.dt.strftime("%Y-%m").astype("string").where(od.notna())
            return od.dt.year.astype("Int64")
    if g == "Y" and _VINTAGE in df.columns and df[_VINTAGE].notna().any():
        return df[_VINTAGE]
    return None


def _dimension_series(df: pd.DataFrame, dimension: str,
                      grain: str) -> Tuple[Optional[pd.Series], str]:
    """The per-row cohort label for ``dimension``, and its column header.

    Prefers a pre-bucketed column derived by ``funded_prep`` (age_bucket,
    original_ltv_bucket …); falls back to banding the raw value. Returns
    ``(None, header)`` when the tape carries no source for the dimension."""
    header = _DIMENSION_LABELS.get(dimension, "Cohort")
    if dimension == "vintage":
        return _vintage_series(df, grain), header
    if dimension == "age":
        if _has_labels(df, _AGE_BUCKET):
            return df[_AGE_BUCKET].astype("string"), header
        if _has_values(df, _YOUNGEST_AGE):
            banded = pd.cut(coerce_numeric(df[_YOUNGEST_AGE]), _AGE_BINS,
                            labels=_AGE_LABELS, right=False)
            return banded.astype("string"), header
        return None, header
    if dimension == "ltv":
        for col in (_ORIG_LTV_BUCKET, _LTV_BUCKET):
            if _has_labels(df, col):
                return df[col].astype("string"), header
        for col in (_ORIG_LTV, _LTV):
            if _has_values(df, col):
                banded = pd.cut(_ltv_as_fraction(df[col]), _LTV_BINS,
                                labels=_LTV_LABELS, right=False)
                return banded.astype("string"), header
        return None, header
    if dimension == "channel":
        for col in (_ORIG_CHANNEL, _BROKER):
            if _has_labels(df, col):
                return df[col].astype("string"), header
        return None, header
    return None, header


def _available_dimensions(df: pd.DataFrame) -> List[str]:
    """Which cohort dimensions the tape can actually support (drives the UI
    selector so it never offers a lens with no data)."""
    out: List[str] = []
    for dim in ("vintage", "age", "ltv", "channel"):
        series, _ = _dimension_series(df, dim, "Y")
        if series is not None and series.notna().any():
            out.append(dim)
    return out


def cohort_analysis(df: pd.DataFrame, *, client_id: str = "",
                    portfolio_id: str = "",
                    reporting_date: Optional[str] = None,
                    grain: str = "Y", dimension: str = "vintage") -> Dict[str, Any]:
    """Static-pool cohort table for a funded run, grouped by ``dimension``
    (vintage | age | ltv | channel), at ``grain`` (Y|Q|M — vintage only).

    Asset-class-agnostic: the same generic metrics (balance / loan count / book
    share and balance-weighted LTV, interest rate and months-on-book) are shown
    for every dimension. ``available`` is False (with a ``reason``) when the tape
    carries no source for the chosen dimension — the UI then shows an honest
    'no computed cohort data' state rather than a fabricated one.
    """
    dimension = dimension if dimension in _DIMENSION_LABELS else "vintage"
    available_dims = _available_dimensions(df) if df is not None and len(df) else []
    base = {
        "dataset": "cohorts",
        "portfolioId": portfolio_id or client_id,
        "cohortBasis": _ORIG_DATE,
        "period": (grain or "Y").upper(),
        "reportingDate": reporting_date,
        "dimension": dimension,
        "dimensionLabel": _DIMENSION_LABELS[dimension],
        "availableDimensions": available_dims,
    }
    if df is None or len(df) == 0:
        return {**base, "available": False, "reason": "no funded rows for this run",
                "cohorts": [], "metricsAvailable": []}

    series, header = _dimension_series(df, dimension, grain)
    if series is None:
        return {**base, "available": False,
                "reason": f"no {header.lower()} field on the funded tape",
                "cohorts": [], "metricsAvailable": []}

    work = df.copy()
    work["_vintage"] = series
    has_balance = _BALANCE in work.columns
    balance = coerce_numeric(work[_BALANCE]) if has_balance else None
    total_balance = float(balance.sum()) if balance is not None else None

    metrics_available: List[str] = ["loanCount"]
    if has_balance:
        metrics_available.append("balance")
    if _LTV in work.columns:
        metrics_available.append("waLtv")
    if _RATE in work.columns:
        metrics_available.append("waRate")
    if _MOB in work.columns:
        metrics_available.append("waMonthsOnBook")

    cohorts: List[Dict[str, Any]] = []
    # Rows with a missing label go into an explicit "Unknown" bucket so the table
    # reconciles to the book total (never silently dropped).
    for value, sub in work.groupby(work["_vintage"].astype("object"), dropna=False):
        if pd.isna(value) or str(value).strip() in ("", "nan", "None"):
            label = "Unknown"
        elif dimension == "vintage":
            try:
                label = str(int(value))
            except (TypeError, ValueError):
                label = str(value)
        else:
            label = str(value)
        sub_balance = coerce_numeric(sub[_BALANCE]) if has_balance else None
        bal = float(sub_balance.sum()) if sub_balance is not None else None
        # ``cohort`` is the generic label; ``vintage`` kept as an alias for
        # backward compatibility with the vintage-only contract.
        row: Dict[str, Any] = {
            "cohort": label,
            "vintage": label,
            "loanCount": int(len(sub)),
        }
        if bal is not None:
            row["balance"] = round(bal, 2)
            row["sharePct"] = (round(bal / total_balance * 100, 2)
                               if total_balance else None)
        if _LTV in sub.columns and sub_balance is not None:
            row["waLtv"] = _weighted_avg_pct(sub[_LTV], sub[_BALANCE])
        if _RATE in sub.columns and sub_balance is not None:
            row["waRate"] = _weighted_avg_pct(sub[_RATE], sub[_BALANCE])
        if _MOB in sub.columns and sub_balance is not None:
            row["waMonthsOnBook"] = _weighted_avg(sub[_MOB], sub[_BALANCE])
        cohorts.append(row)

    # Ordering: vintage chronological (lexicographic is chronological across
    # grains); age/LTV by the band's leading number; channel by balance desc.
    # The "Unknown" bucket always sinks to the end.
    def _key(r: Dict[str, Any]):
        label = str(r["cohort"])
        if label == "Unknown":
            return (2, 0.0, "")
        if dimension in ("age", "ltv"):
            m = re.search(r"\d+", label)
            return (0, float(m.group()) if m else 0.0, label)
        if dimension == "channel":
            return (0, -float(r.get("balance") or 0.0), label)
        return (0, 0.0, label)  # vintage — lexicographic

    cohorts.sort(key=_key)

    return {
        **base,
        "available": True,
        "totalBalance": (round(total_balance, 2) if total_balance is not None else None),
        "totalLoanCount": int(len(work)),
        "metricsAvailable": metrics_available,
        "cohorts": cohorts,
        "lineage": {
            "source": f"governed funded central lender tape (by {header.lower()})",
            "metric": "balance / loan count / book share and balance-weighted "
                      "LTV, interest rate and months-on-book",
            "note": "Point-in-time static-pool aggregates only. Redemption / "
                    "completion / performance curves are not computed in the MI "
                    "path and are not shown.",
        },
    }
