"""analytics_lib.cohort — point-in-time cohort / vintage foundations.

Phase 1 shared analytics library. Pure functions to derive cohort periods from
an event date (origination / acquisition / funding) and to compute months-on-
book against a reporting/as-of date. This is point-in-time only: recurring
snapshot-to-snapshot migration is explicitly deferred (see
:mod:`analytics_lib.migration`).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from .stratify import stratify

# Issue codes (shared shape with analytics_lib.buckets).
UNAVAILABLE_FIELD = "unavailable_field"
INVALID_DATE = "invalid_date"
WARNING = "warning"
ERROR = "error"

_PERIOD_FREQS = {"Y": "Y", "year": "Y", "Q": "Q", "quarter": "Q",
                 "M": "M", "month": "M"}


def _issue(field: Optional[str], code: str, severity: str, count: int,
           message: str) -> Dict[str, Any]:
    return {"field": field, "code": code, "severity": severity,
            "count": int(count), "message": message}


def cohort_period(values: pd.Series, period: str = "Y") -> pd.Series:
    """Map a date Series to cohort-period labels (e.g. ``2021`` or ``2021Q2``)."""
    freq = _PERIOD_FREQS.get(period)
    if freq is None:
        raise ValueError(f"unsupported period {period!r} (use Y/Q/M)")
    dt = pd.to_datetime(values, errors="coerce")
    if freq == "Y":
        out = dt.dt.year.astype("Int64").astype("object")
        return out.where(dt.notna(), None)
    labels = dt.dt.to_period(freq).astype("object")
    return labels.where(dt.notna(), None)


def add_cohort_period(
    df: pd.DataFrame,
    date_col: str,
    *,
    period: str = "Y",
    out_col: Optional[str] = None,
) -> Tuple[pd.DataFrame, List[Dict[str, Any]]]:
    """Add a cohort-period column derived from *date_col*.

    Returns ``(df_out, issues)``. Reports an unavailable-field issue when
    *date_col* is missing, and an invalid-date issue for unparseable values.
    """
    out_col = out_col or f"{date_col}_cohort"
    if date_col not in df.columns:
        return df.copy(), [_issue(date_col, UNAVAILABLE_FIELD, ERROR, len(df),
                                  f"date field {date_col!r} not present")]
    out = df.copy()
    raw = out[date_col]
    parsed = pd.to_datetime(raw, errors="coerce")
    issues: List[Dict[str, Any]] = []
    n_bad = int((parsed.isna() & raw.notna()).sum())
    if n_bad:
        issues.append(_issue(date_col, INVALID_DATE, WARNING, n_bad,
                             f"{n_bad} unparseable date(s) in {date_col!r}"))
    out[out_col] = cohort_period(raw, period=period)
    return out, issues


def cohort_table(
    df: pd.DataFrame,
    date_col: str,
    balance_col: str,
    *,
    period: str = "Y",
    loan_id_col: Optional[str] = None,
    filters: Optional[Dict[str, Any]] = None,
) -> Tuple[pd.DataFrame, List[Dict[str, Any]]]:
    """Balance / count by cohort period for an event date.

    Works for origination, acquisition or funding cohorts simply by passing the
    relevant *date_col*. Returns ``(table, issues)`` where *table* is the
    stratification summary keyed by the cohort period.
    """
    out, issues = add_cohort_period(df, date_col, period=period,
                                    out_col=f"{date_col}_cohort")
    cohort_col = f"{date_col}_cohort"
    if cohort_col not in out.columns:
        return pd.DataFrame(), issues
    table = stratify(out, cohort_col, balance_col, loan_id_col=loan_id_col,
                     filters=filters, sort_by=cohort_col)
    return table, issues


def months_on_book(
    df: pd.DataFrame,
    start_date_col: str,
    as_of: Any,
    *,
    out_col: str = "months_on_book",
) -> Tuple[pd.DataFrame, List[Dict[str, Any]]]:
    """Compute whole months between *start_date_col* and *as_of*.

    *as_of* may be a scalar reporting/as-of date or the name of a column in
    *df*. Returns ``(df_out, issues)``; negative results (start after as-of) are
    clamped to 0 and reported as a warning.
    """
    if start_date_col not in df.columns:
        return df.copy(), [_issue(start_date_col, UNAVAILABLE_FIELD, ERROR,
                                  len(df),
                                  f"start date {start_date_col!r} not present")]
    out = df.copy()
    issues: List[Dict[str, Any]] = []

    start = pd.to_datetime(out[start_date_col], errors="coerce")
    if isinstance(as_of, str) and as_of in out.columns:
        end = pd.to_datetime(out[as_of], errors="coerce")
    else:
        end = pd.Series(pd.to_datetime(as_of, errors="coerce"), index=out.index)

    n_bad_start = int((start.isna() & out[start_date_col].notna()).sum())
    if n_bad_start:
        issues.append(_issue(start_date_col, INVALID_DATE, WARNING,
                             n_bad_start,
                             f"{n_bad_start} unparseable start date(s)"))

    months = (end.dt.year - start.dt.year) * 12 + (end.dt.month - start.dt.month)
    n_neg = int((months < 0).sum())
    if n_neg:
        issues.append(_issue(start_date_col, "negative_months", WARNING, n_neg,
                             f"{n_neg} loan(s) have start date after as-of date "
                             f"(clamped to 0)"))
    months = months.clip(lower=0)
    out[out_col] = months.astype("Int64")
    return out, issues
