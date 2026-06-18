"""mi_agent.risk_monitor.concentration — concentration early-warning.

Phase 5. Single-snapshot concentration with RAG status + approaching-limit
warning, and baseline/current (or funded/forecast) concentration movement.
Reuses ``analytics_lib.concentration``. Frame-in/frame-out, no charts.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import pandas as pd

from analytics_lib.concentration import group_shares, rag_status
from analytics_lib.concentration import top_n_concentration as _top_n

from . import models as RM
from .models import (
    BALANCE_COL,
    DEFAULT_KEY,
    RiskMonitorResult,
    get_approaching_at,
    get_concentration_thresholds,
    get_minimums,
    make_issue,
)


def funded_concentration(frame: pd.DataFrame, dimension: str, *,
                         balance_col: str = BALANCE_COL,
                         loan_id_col: str = DEFAULT_KEY,
                         thresholds: Optional[Dict[str, float]] = None,
                         approaching_at: Optional[float] = None,
                         minimum_balance: float = 0.0,
                         minimum_count: int = 0,
                         config: Optional[Dict[str, Any]] = None,
                         kind: str = "funded_concentration") -> RiskMonitorResult:
    """Group-share concentration with green/amber/red + approaching-limit flag."""
    issues: List[dict] = []
    thresholds = thresholds or get_concentration_thresholds(config)
    approaching_at = (approaching_at if approaching_at is not None
                      else get_approaching_at(config))
    if config is not None:
        mins = get_minimums(config)
        minimum_balance = minimum_balance or mins["balance"]
        minimum_count = minimum_count or mins["count"]

    if dimension not in frame.columns:
        issues.append(make_issue(
            RM.MISSING_CONCENTRATION_DIMENSION, RM.WARNING,
            f"concentration dimension {dimension!r} not present", field=dimension))
        return RiskMonitorResult(kind, pd.DataFrame(), issues,
                                 {"dimension": dimension})

    table = group_shares(frame, dimension, balance_col, loan_id_col=loan_id_col)
    if table.empty:
        issues.append(make_issue(RM.EMPTY_RISK_MONITOR_RESULT, RM.WARNING,
                                 "no rows to concentrate"))
        return RiskMonitorResult(kind, table, issues, {"dimension": dimension})

    red = thresholds.get("red", 0.30)
    n_below = 0
    statuses, approaching = [], []
    for _, r in table.iterrows():
        share = float(r["balance_share"])
        if (r["balance_sum"] < minimum_balance
                or r["loan_count"] < minimum_count):
            n_below += 1
            statuses.append("below_minimum")
            approaching.append(False)
            continue
        status = rag_status(share, thresholds)
        statuses.append(status)
        # Approaching = not yet red, but within the configured fraction of red.
        approaching.append(status != "red"
                           and (red * approaching_at) <= share < red)
    table = table.copy()
    table["status"] = statuses
    table["approaching_limit"] = approaching
    table["dimension"] = dimension

    if n_below:
        issues.append(make_issue(
            RM.CONCENTRATION_BELOW_MINIMUM_THRESHOLD, RM.INFO,
            f"{n_below} group(s) below the minimum balance/count threshold; "
            f"reported but not RAG-flagged", field=dimension, count=n_below))

    meta = {"dimension": dimension, "thresholds": thresholds,
            "approaching_at": approaching_at,
            "amber_or_worse": int(sum(s in ("amber", "red") for s in statuses))}
    return RiskMonitorResult(kind, table.reset_index(drop=True), issues, meta)


def concentration_movement(baseline: pd.DataFrame, current: pd.DataFrame,
                           dimension: str, *, balance_col: str = BALANCE_COL,
                           loan_id_col: str = DEFAULT_KEY,
                           thresholds: Optional[Dict[str, float]] = None,
                           config: Optional[Dict[str, Any]] = None,
                           baseline_balance_col: Optional[str] = None,
                           kind: str = "concentration_movement"
                           ) -> RiskMonitorResult:
    """Per-group share movement between two frames (baseline->current or
    funded->forecast)."""
    issues: List[dict] = []
    thresholds = thresholds or get_concentration_thresholds(config)
    base_bal = baseline_balance_col or balance_col

    for label, fr, col in (("baseline", baseline, base_bal),
                           ("current", current, balance_col)):
        if dimension not in fr.columns:
            issues.append(make_issue(
                RM.MISSING_CONCENTRATION_DIMENSION, RM.WARNING,
                f"dimension {dimension!r} not present on {label} frame",
                field=dimension))
            return RiskMonitorResult(kind, pd.DataFrame(), issues,
                                     {"dimension": dimension})

    bt = group_shares(baseline, dimension, base_bal, loan_id_col=loan_id_col)
    ct = group_shares(current, dimension, balance_col, loan_id_col=loan_id_col)
    bt = bt[[dimension, "balance_share"]].rename(
        columns={"balance_share": "baseline_share"})
    ct = ct[[dimension, "balance_share"]].rename(
        columns={"balance_share": "current_share"})
    merged = bt.merge(ct, on=dimension, how="outer").fillna(
        {"baseline_share": 0.0, "current_share": 0.0})
    merged["share_change"] = merged["current_share"] - merged["baseline_share"]
    merged["increasing"] = merged["share_change"] > 0
    merged["status_current"] = [rag_status(float(s), thresholds)
                                for s in merged["current_share"]]
    merged["dimension"] = dimension
    merged = merged.sort_values("current_share", ascending=False,
                                kind="mergesort").reset_index(drop=True)

    if merged.empty:
        issues.append(make_issue(RM.EMPTY_RISK_MONITOR_RESULT, RM.WARNING,
                                 "no groups to compare"))
    return RiskMonitorResult(kind, merged, issues, {"dimension": dimension,
                                                    "thresholds": thresholds})


def top_n_concentration(frame: pd.DataFrame, dimension: str, *,
                        balance_col: str = BALANCE_COL, n: int = 10,
                        loan_id_col: str = DEFAULT_KEY) -> RiskMonitorResult:
    """Thin wrapper over ``analytics_lib.concentration.top_n_concentration``."""
    issues: List[dict] = []
    if dimension not in frame.columns:
        issues.append(make_issue(
            RM.MISSING_CONCENTRATION_DIMENSION, RM.WARNING,
            f"dimension {dimension!r} not present", field=dimension))
        return RiskMonitorResult("top_n_concentration", pd.DataFrame(), issues,
                                 {"dimension": dimension})
    res = _top_n(frame, dimension, balance_col, n=n, loan_id_col=loan_id_col)
    meta = {"dimension": dimension, "top_n": n,
            "balance_concentration": res["balance_concentration"],
            "count_concentration": res["count_concentration"],
            "n_groups": res["n_groups"]}
    return RiskMonitorResult("top_n_concentration", res["groups"], issues, meta)
