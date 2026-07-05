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

from typing import Any, Dict, List, Optional

import pandas as pd

from analytics_lib.numeric import coerce_numeric
from mi_agent.mi_dataset_profile import PERCENT_POINTS, percent_storage_scale

_BALANCE = "current_outstanding_balance"
_LTV = "current_loan_to_value"
_RATE = "current_interest_rate"
_MOB = "months_on_book"
_VINTAGE = "vintage_year"
_ORIG_DATE = "origination_date"


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


def cohort_analysis(df: pd.DataFrame, *, client_id: str = "",
                    portfolio_id: str = "",
                    reporting_date: Optional[str] = None,
                    grain: str = "Y") -> Dict[str, Any]:
    """Per-origination-vintage cohort table for a funded run, at ``grain``
    (Y|Q|M).

    Returns a UI-ready view-model. ``available`` is False (with a ``reason``)
    when the tape carries no origination vintage — the UI then shows an honest
    'no computed cohort data' state rather than a fabricated one.
    """
    base = {
        "dataset": "cohorts",
        "portfolioId": portfolio_id or client_id,
        "cohortBasis": _ORIG_DATE,
        "period": (grain or "Y").upper(),
        "reportingDate": reporting_date,
    }
    if df is None or len(df) == 0:
        return {**base, "available": False, "reason": "no funded rows for this run",
                "cohorts": [], "metricsAvailable": []}

    vintages = _vintage_series(df, grain)
    if vintages is None:
        return {**base, "available": False,
                "reason": "no origination date / vintage on the funded tape",
                "cohorts": [], "metricsAvailable": []}

    work = df.copy()
    work["_vintage"] = vintages
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
    # Drop rows with a missing vintage into an explicit bucket so the table
    # reconciles to the book total (never silently dropped).
    for vintage, sub in work.groupby(work["_vintage"].astype("object"), dropna=False):
        if pd.isna(vintage):
            label = "Unknown"
        else:
            try:
                label = str(int(vintage))
            except (TypeError, ValueError):
                label = str(vintage)
        sub_balance = coerce_numeric(sub[_BALANCE]) if has_balance else None
        bal = float(sub_balance.sum()) if sub_balance is not None else None
        row: Dict[str, Any] = {
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

    # Sort by vintage ascending; the Unknown bucket sinks to the end. Lexicographic
    # order is chronological for every grain ("2023" < "2023-Q2" < "2024",
    # "2025-03" < "2025-06").
    def _key(r: Dict[str, Any]):
        v = str(r["vintage"])
        return (1, "") if v == "Unknown" else (0, v)

    cohorts.sort(key=_key)

    return {
        **base,
        "available": True,
        "totalBalance": (round(total_balance, 2) if total_balance is not None else None),
        "totalLoanCount": int(len(work)),
        "metricsAvailable": metrics_available,
        "cohorts": cohorts,
        "lineage": {
            "source": "governed funded central lender tape (origination vintage)",
            "metric": "balance / loan count / book share and balance-weighted "
                      "LTV, interest rate and months-on-book by origination year",
            "note": "Point-in-time vintage aggregates only. Redemption / completion "
                    "/ performance curves are not computed in the MI path and are "
                    "not shown.",
        },
    }
