"""Independent numerical oracle for the MI Agent assurance programme.

This module computes expected values for the question bank **independently of the
production calculation code**. It imports only ``pandas`` / ``numpy`` and the
standard library. It MUST NOT import:

* ``mi_agent`` / ``mi_agent_api`` / ``mi_workflows`` calculation code
* production aggregation, directionality, currency-guard or summary builders

The transparent reference implementations below are deliberately simple so a
reviewer can confirm each by eye. They express the *governed contract* the
production code is expected to honour — in particular:

* monetary totals are only produced for a single-currency population;
* weighted averages disclose their valid (strictly-positive-weight) denominator
  and never fall back to a simple mean on zero total weight;
* shares keep the unknown/missing bucket in the denominator;
* a point-in-time measure is taken as of a single reporting date.

The oracle operates on a raw canonical dataframe (the same CSV the agent loads)
using only the canonical column names, so it shares no code with the system
under test beyond ``pandas`` itself — which cannot alter the values.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

# Canonical column names (from the platform canonical schema).
BALANCE = "current_outstanding_balance"
LTV = "current_loan_to_value"
RATE = "current_interest_rate"
AGE = "youngest_borrower_age"
LOAN_ID = "loan_identifier"
SOURCE_PID = "source_portfolio_id"
SOURCE_TYPE = "source_portfolio_type"
CURRENCY_FIELDS = ("exposure_currency_denomination", "currency_denomination",
                   "collateral_currency")
DATE_FIELDS = ("reporting_date", "data_cut_off_date", "cut_off_date")

UNKNOWN = "(unknown)"


@dataclass
class MeasureResult:
    metric: str
    aggregation: str
    value: Optional[float]
    population: int
    denominator: Optional[float] = None
    unit: str = "currency"
    currency: Optional[str] = None
    monetary_suppressed: bool = False
    notes: List[str] = field(default_factory=list)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "metric": self.metric, "aggregation": self.aggregation,
            "value": self.value, "population": self.population,
            "denominator": self.denominator, "unit": self.unit,
            "currency": self.currency, "monetary_suppressed": self.monetary_suppressed,
            "notes": self.notes,
        }


# --------------------------------------------------------------------------- #
# Currency
# --------------------------------------------------------------------------- #
def currency_profile(df: pd.DataFrame) -> Dict[str, Any]:
    """The distinct non-null currencies present, from the first populated field."""
    for col in CURRENCY_FIELDS:
        if col in df.columns:
            vals = df[col].dropna().astype(str).str.strip()
            vals = vals[vals != ""]
            distinct = sorted(vals.unique())
            if distinct:
                return {"field": col, "currencies": distinct,
                        "mixed": len(distinct) > 1}
    return {"field": None, "currencies": [], "mixed": False}


def is_single_currency(df: pd.DataFrame) -> bool:
    return len(currency_profile(df)["currencies"]) <= 1


# --------------------------------------------------------------------------- #
# Reporting date
# --------------------------------------------------------------------------- #
def reporting_dates(df: pd.DataFrame) -> List[str]:
    for col in DATE_FIELDS:
        if col in df.columns:
            parsed = pd.to_datetime(df[col], errors="coerce").dropna()
            return sorted({d.date().isoformat() for d in parsed})
    return []


def as_of_latest(df: pd.DataFrame) -> pd.DataFrame:
    """Narrow to the latest reporting date (the point-in-time contract)."""
    for col in DATE_FIELDS:
        if col in df.columns:
            parsed = pd.to_datetime(df[col], errors="coerce")
            if parsed.notna().any():
                latest = parsed.max()
                return df[parsed == latest]
    return df


# --------------------------------------------------------------------------- #
# Point-in-time measures
# --------------------------------------------------------------------------- #
def _numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def total(df: pd.DataFrame, column: str = BALANCE, *, monetary: bool = True) -> MeasureResult:
    """Sum of a column as of the latest date. Monetary totals are suppressed on a
    mixed-currency population (the governed contract)."""
    frame = as_of_latest(df)
    prof = currency_profile(frame)
    if monetary and prof["mixed"]:
        return MeasureResult(column, "sum", None, int(len(frame)),
                             unit="currency", currency=None, monetary_suppressed=True,
                             notes=["mixed currency: monetary total suppressed"])
    vals = _numeric(frame[column])
    valid = vals.dropna()
    cur = prof["currencies"][0] if prof["currencies"] else None
    return MeasureResult(column, "sum", float(valid.sum()), int(valid.shape[0]),
                         unit="currency" if monetary else "count",
                         currency=cur if monetary else None)


def count(df: pd.DataFrame) -> MeasureResult:
    frame = as_of_latest(df)
    return MeasureResult(LOAN_ID, "count", float(len(frame)), int(len(frame)),
                         unit="count")


def simple_average(df: pd.DataFrame, column: str) -> MeasureResult:
    frame = as_of_latest(df)
    vals = _numeric(frame[column]).dropna()
    val = float(vals.mean()) if not vals.empty else None
    return MeasureResult(column, "average", val, int(vals.shape[0]),
                         denominator=float(vals.shape[0]), unit="ratio")


def weighted_average(df: pd.DataFrame, value_col: str,
                     weight_col: str = BALANCE) -> MeasureResult:
    """Weighted average with STRICTLY POSITIVE weights. On zero total weight the
    value is ``None`` — an unweighted mean is never substituted."""
    frame = as_of_latest(df)
    v = _numeric(frame[value_col])
    w = _numeric(frame[weight_col])
    mask = v.notna() & w.notna() & (w > 0)
    vv, ww = v[mask], w[mask]
    denom = float(ww.sum())
    if denom <= 0:
        return MeasureResult(value_col, "weighted_average", None, 0,
                             denominator=0.0, unit="ratio",
                             notes=["zero valid weight: no unweighted fallback"])
    val = float((vv * ww).sum() / denom)
    return MeasureResult(value_col, "weighted_average", val, int(mask.sum()),
                         denominator=denom, unit="ratio")


# --------------------------------------------------------------------------- #
# Filters and scope
# --------------------------------------------------------------------------- #
def apply_scope(df: pd.DataFrame, *, source_portfolio_id: Optional[str] = None,
                source_portfolio_type: Optional[str] = None) -> pd.DataFrame:
    frame = df
    if source_portfolio_id is not None and SOURCE_PID in frame.columns:
        frame = frame[frame[SOURCE_PID] == source_portfolio_id]
    if source_portfolio_type is not None and SOURCE_TYPE in frame.columns:
        frame = frame[frame[SOURCE_TYPE] == source_portfolio_type]
    return frame


# --------------------------------------------------------------------------- #
# Distributions and shares (unknown bucket kept in the denominator)
# --------------------------------------------------------------------------- #
def distribution(df: pd.DataFrame, dimension: str,
                 measure: str = BALANCE, *, monetary: bool = True) -> Dict[str, Any]:
    frame = as_of_latest(df).copy()
    dim = frame[dimension].astype("object").where(frame[dimension].notna(), UNKNOWN)
    frame = frame.assign(_dim=dim)
    prof = currency_profile(frame)
    suppress = monetary and prof["mixed"]
    rows: List[Dict[str, Any]] = []
    if suppress:
        grp = frame.groupby("_dim")[LOAN_ID].count()
        total_count = int(grp.sum())
        for cat, c in grp.items():
            rows.append({"category": str(cat), "count": int(c),
                         "count_share": (c / total_count) if total_count else None,
                         "exposure": None, "exposure_share": None})
    else:
        vals = _numeric(frame[measure]).fillna(0.0)
        frame = frame.assign(_val=vals)
        grp = frame.groupby("_dim").agg(count=(LOAN_ID, "count"),
                                        exposure=("_val", "sum"))
        total_count = int(grp["count"].sum())
        total_exp = float(grp["exposure"].sum())
        for cat, r in grp.iterrows():
            rows.append({
                "category": str(cat), "count": int(r["count"]),
                "count_share": (r["count"] / total_count) if total_count else None,
                "exposure": float(r["exposure"]),
                "exposure_share": (r["exposure"] / total_exp) if total_exp else None,
            })
    rows.sort(key=lambda x: (-(x["exposure_share"] or 0), x["category"]))
    return {"dimension": dimension, "rows": rows, "monetary_suppressed": suppress,
            "count_share_sum": sum(r["count_share"] or 0 for r in rows),
            "exposure_share_sum": (None if suppress
                                   else sum(r["exposure_share"] or 0 for r in rows))}


def concentration_top_n(df: pd.DataFrame, dimension: str, n: int,
                        measure: str = BALANCE) -> Dict[str, Any]:
    dist = distribution(df, dimension, measure)
    ranked = [r for r in dist["rows"]]
    top = ranked[:n]
    top_share = sum(r["exposure_share"] or 0 for r in top)
    return {"dimension": dimension, "n": n, "top": top,
            "top_share": top_share,
            "monetary_suppressed": dist["monetary_suppressed"]}


def single_name_top(df: pd.DataFrame, n: int = 1,
                    id_col: str = LOAN_ID, measure: str = BALANCE) -> Dict[str, Any]:
    frame = as_of_latest(df).copy()
    frame = frame.assign(_val=_numeric(frame[measure]).fillna(0.0))
    total_exp = float(frame["_val"].sum())
    ranked = frame.sort_values("_val", ascending=False)
    top = ranked.head(n)
    top_share = float(top["_val"].sum() / total_exp) if total_exp else None
    return {"n": n, "top_ids": list(top[id_col].astype(str)),
            "top_exposure": float(top["_val"].sum()), "top_share": top_share,
            "total_exposure": total_exp}
