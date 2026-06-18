"""analytics_lib.stratify — generic balance/count stratification.

Phase 1 shared analytics library. A single pure function that stratifies a
loan-level frame by one dimension (categorical *or* pre-bucketed) and returns a
tidy summary table. No chart output, no UI, no legacy imports.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Sequence

import pandas as pd

UNKNOWN_LABEL = "Unknown"


def _apply_filters(df: pd.DataFrame,
                   filters: Optional[Dict[str, Any]]) -> pd.DataFrame:
    """Filter rows. Each filter is ``col -> value`` or ``col -> [values]``."""
    if not filters:
        return df
    mask = pd.Series(True, index=df.index)
    for col, want in filters.items():
        if col not in df.columns:
            raise ValueError(f"filter column {col!r} not in DataFrame")
        if isinstance(want, (list, tuple, set)):
            mask &= df[col].isin(list(want))
        else:
            mask &= df[col] == want
    return df[mask]


def stratify(
    df: pd.DataFrame,
    dimension: str,
    balance_col: Optional[str] = None,
    *,
    count_col: Optional[str] = None,
    loan_id_col: Optional[str] = None,
    filters: Optional[Dict[str, Any]] = None,
    weighted_metrics: Optional[Sequence[str]] = None,
    weight_col: Optional[str] = None,
    unknown_label: str = UNKNOWN_LABEL,
    dropna: bool = False,
    sort_by: str = "balance_sum",
) -> pd.DataFrame:
    """Stratify *df* by *dimension*.

    Returns one row per category with: ``loan_count``, ``balance_sum``,
    ``balance_share`` (fraction 0..1), ``avg_balance``, and any requested
    weighted-average metrics (``{metric}_weighted_avg``).

    Missing dimension values are bucketed into *unknown_label* (explicit) unless
    *dropna* is set. Ordering is deterministic: by *sort_by* descending, then by
    the dimension value ascending (stable).
    """
    if dimension not in df.columns:
        raise ValueError(f"dimension column {dimension!r} not in DataFrame")

    work = _apply_filters(df, filters).copy()

    # Explicit missing/unknown handling.
    dim = work[dimension]
    if dropna:
        work = work[dim.notna()]
    else:
        work[dimension] = dim.astype("object").where(dim.notna(), unknown_label)

    has_balance = balance_col is not None
    if has_balance and balance_col not in work.columns:
        raise ValueError(f"balance column {balance_col!r} not in DataFrame")

    weight_col = weight_col or balance_col
    rows: List[Dict[str, Any]] = []

    for value, grp in work.groupby(dimension, dropna=False, sort=False):
        # Count: distinct loan ids > explicit count column > row count.
        if loan_id_col and loan_id_col in grp.columns:
            loan_count = int(grp[loan_id_col].nunique())
        elif count_col and count_col in grp.columns:
            loan_count = int(pd.to_numeric(grp[count_col],
                                           errors="coerce").fillna(0).sum())
        else:
            loan_count = int(len(grp))

        row: Dict[str, Any] = {dimension: value, "loan_count": loan_count}

        if has_balance:
            bal = pd.to_numeric(grp[balance_col], errors="coerce")
            balance_sum = float(bal.sum())
            row["balance_sum"] = balance_sum
            row["avg_balance"] = (balance_sum / loan_count
                                  if loan_count else 0.0)

        for metric in (weighted_metrics or []):
            if metric not in grp.columns:
                row[f"{metric}_weighted_avg"] = float("nan")
                continue
            val = pd.to_numeric(grp[metric], errors="coerce")
            wt = pd.to_numeric(grp[weight_col], errors="coerce") \
                if weight_col and weight_col in grp.columns else None
            if wt is not None and float(wt.fillna(0).sum()) != 0:
                paired = pd.DataFrame({"v": val, "w": wt}).dropna()
                wsum = float(paired["w"].sum())
                row[f"{metric}_weighted_avg"] = (
                    float((paired["v"] * paired["w"]).sum()) / wsum
                    if wsum else float("nan"))
            else:
                row[f"{metric}_weighted_avg"] = float(val.mean())

        rows.append(row)

    result = pd.DataFrame(rows)
    if result.empty:
        return result

    if has_balance:
        total = float(result["balance_sum"].sum())
        result["balance_share"] = (result["balance_sum"] / total
                                   if total else 0.0)
        # Tidy column order.
        ordered = [dimension, "loan_count", "balance_sum", "balance_share",
                   "avg_balance"]
        ordered += [c for c in result.columns if c not in ordered]
        result = result[ordered]

    # Deterministic ordering.
    if sort_by in result.columns:
        result = result.sort_values(
            by=[sort_by, dimension], ascending=[False, True],
            kind="mergesort").reset_index(drop=True)
    else:
        result = result.sort_values(
            by=[dimension], kind="mergesort").reset_index(drop=True)
    return result
