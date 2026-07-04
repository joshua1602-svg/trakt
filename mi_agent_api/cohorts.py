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


def _vintage_series(df: pd.DataFrame) -> Optional[pd.Series]:
    """The origination-year label per row: the derived ``vintage_year`` column
    when present, else parsed from ``origination_date``. None when neither
    exists (no vintage analysis possible)."""
    if _VINTAGE in df.columns and df[_VINTAGE].notna().any():
        return df[_VINTAGE]
    if _ORIG_DATE in df.columns:
        od = pd.to_datetime(df[_ORIG_DATE], errors="coerce", dayfirst=True)
        if od.notna().any():
            return od.dt.year.astype("Int64")
    return None


def cohort_analysis(df: pd.DataFrame, *, client_id: str = "",
                    portfolio_id: str = "",
                    reporting_date: Optional[str] = None) -> Dict[str, Any]:
    """Per-origination-year cohort table for a funded run.

    Returns a UI-ready view-model. ``available`` is False (with a ``reason``)
    when the tape carries no origination vintage — the UI then shows an honest
    'no computed cohort data' state rather than a fabricated one.
    """
    base = {
        "dataset": "cohorts",
        "portfolioId": portfolio_id or client_id,
        "cohortBasis": _ORIG_DATE,
        "period": "Y",
        "reportingDate": reporting_date,
    }
    if df is None or len(df) == 0:
        return {**base, "available": False, "reason": "no funded rows for this run",
                "cohorts": [], "metricsAvailable": []}

    vintages = _vintage_series(df)
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
            row["waLtv"] = _weighted_avg(sub[_LTV], sub[_BALANCE])
        if _RATE in sub.columns and sub_balance is not None:
            row["waRate"] = _weighted_avg(sub[_RATE], sub[_BALANCE])
        if _MOB in sub.columns and sub_balance is not None:
            row["waMonthsOnBook"] = _weighted_avg(sub[_MOB], sub[_BALANCE])
        cohorts.append(row)

    # Sort by vintage year ascending; the Unknown bucket sinks to the end.
    def _key(r: Dict[str, Any]):
        v = r["vintage"]
        return (1, 0) if v == "Unknown" else (0, int(v)) if v.isdigit() else (0, 0)

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
