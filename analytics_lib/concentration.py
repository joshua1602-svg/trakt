"""analytics_lib.concentration — reusable concentration calculations.

Phase 1 shared analytics library. Pure functions for group shares, top-N
concentration, and simple limit-usage RAG status. This is NOT the MI risk
monitor (that is a later phase) — just the underlying, route-agnostic maths.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import pandas as pd

from .stratify import stratify

GREEN = "green"
AMBER = "amber"
RED = "red"

DEFAULT_THRESHOLDS = {"amber": 0.80, "red": 1.00}


def rag_status(usage: float, thresholds: Optional[Dict[str, float]] = None
               ) -> str:
    """Map a limit-usage ratio to a green/amber/red status.

    ``usage >= red`` -> red; ``usage >= amber`` -> amber; else green.
    """
    t = thresholds or DEFAULT_THRESHOLDS
    if usage is None or pd.isna(usage):
        return GREEN
    if usage >= t.get("red", 1.0):
        return RED
    if usage >= t.get("amber", 0.8):
        return AMBER
    return GREEN


def group_shares(
    df: pd.DataFrame,
    dimension: str,
    balance_col: str,
    *,
    loan_id_col: Optional[str] = None,
    count_col: Optional[str] = None,
    filters: Optional[Dict[str, Any]] = None,
) -> pd.DataFrame:
    """Per-group balance share and count share (one row per category).

    Adds ``count_share`` to the standard stratification table.
    """
    table = stratify(df, dimension, balance_col, loan_id_col=loan_id_col,
                     count_col=count_col, filters=filters)
    if table.empty:
        return table
    total_count = float(table["loan_count"].sum())
    table["count_share"] = (table["loan_count"] / total_count
                            if total_count else 0.0)
    return table


def top_n_concentration(
    df: pd.DataFrame,
    dimension: str,
    balance_col: str,
    n: int = 10,
    *,
    loan_id_col: Optional[str] = None,
    filters: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Combined balance/count share of the top-*n* groups by balance.

    Returns a dict with ``top_n``, ``n_groups``, ``balance_concentration``,
    ``count_concentration`` and the ``groups`` table (top-n rows).
    """
    table = group_shares(df, dimension, balance_col,
                         loan_id_col=loan_id_col, filters=filters)
    if table.empty:
        return {"top_n": n, "n_groups": 0, "balance_concentration": 0.0,
                "count_concentration": 0.0, "groups": table}

    top = table.head(n)
    return {
        "top_n": n,
        "n_groups": int(len(table)),
        "balance_concentration": float(top["balance_share"].sum()),
        "count_concentration": float(top["count_share"].sum()),
        "groups": top.reset_index(drop=True),
    }


def limit_usage(
    df: pd.DataFrame,
    dimension: str,
    balance_col: str,
    limits: Dict[str, float],
    *,
    thresholds: Optional[Dict[str, float]] = None,
    loan_id_col: Optional[str] = None,
    filters: Optional[Dict[str, Any]] = None,
) -> pd.DataFrame:
    """Evaluate per-group balance share against a simple limit config.

    *limits* maps a group value to its maximum allowed balance share (0..1).
    Returns the group-shares table with ``limit``, ``limit_usage`` (share /
    limit) and ``status`` (green/amber/red) for groups that carry a limit.
    """
    table = group_shares(df, dimension, balance_col,
                         loan_id_col=loan_id_col, filters=filters)
    if table.empty:
        return table

    table["limit"] = table[dimension].map(limits).astype("float")
    table["limit_usage"] = table.apply(
        lambda r: (r["balance_share"] / r["limit"])
        if pd.notna(r["limit"]) and r["limit"] else float("nan"),
        axis=1,
    )
    table["status"] = table["limit_usage"].apply(
        lambda u: rag_status(u, thresholds))
    return table
