"""mi_agent.risk_monitor.migration — two-snapshot migration matrices + flags.

Phase 5. Deterministic risk migration between a baseline and a current frame,
joined on a stable ``loan_id``. Ordering (improve/deteriorate) is taken from
config only; unordered dimensions are classified ``changed`` / ``unchanged`` and
never invent a direction. Frame-in/frame-out, no UI/charts/LLM/Azure.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from . import models as RM
from .models import (
    BALANCE_COL,
    DEFAULT_KEY,
    RiskMonitorResult,
    get_ordering,
    make_issue,
)


def _norm(value: Any) -> Optional[str]:
    if value is None or (isinstance(value, float) and value != value):
        return None
    text = str(value).strip()
    return text if text != "" else None


def classify_change(from_v: Any, to_v: Any,
                    ordering: Optional[List[str]]) -> Tuple[str, bool]:
    """Classify an in-both transition. Returns ``(movement_type, ordered)``.

    Never invents a direction: with no ordering (or a value outside it) a real
    change is ``changed``, not improved/deteriorated.
    """
    nf, nt = _norm(from_v), _norm(to_v)
    if nf is None and nt is None:
        return RM.UNKNOWN, False
    if nf is None or nt is None:
        return RM.UNKNOWN, False
    if nf == nt:
        return RM.UNCHANGED, bool(ordering)
    if ordering and nf in ordering and nt in ordering:
        i_from, i_to = ordering.index(nf), ordering.index(nt)
        if i_to > i_from:
            return RM.DETERIORATED, True
        if i_to < i_from:
            return RM.IMPROVED, True
        return RM.UNCHANGED, True
    return RM.CHANGED, False


def _prepared(frame: pd.DataFrame, key: str, dimension: str,
              balance_col: str) -> Tuple[Dict, Dict, Dict]:
    """Return ``(dim_by_key, balance_by_key, present_keys)`` (last row wins)."""
    f = frame.drop_duplicates(subset=key, keep="last")
    dim = (dict(zip(f[key], f[dimension])) if dimension in f.columns else {})
    if balance_col in f.columns:
        bal = dict(zip(f[key],
                       pd.to_numeric(f[balance_col], errors="coerce")))
    else:
        bal = {}
    return dim, bal, set(f[key])


def _classify_row(in_b: bool, in_c: bool, fv: Any, tv: Any,
                  ordering: Optional[List[str]]) -> Tuple[str, bool]:
    if not in_b:
        return RM.NEW, bool(ordering)
    if not in_c:
        return RM.EXITED, bool(ordering)
    return classify_change(fv, tv, ordering)


def migration_matrix(baseline: pd.DataFrame, current: pd.DataFrame,
                     dimension: str, *, key: str = DEFAULT_KEY,
                     balance_col: str = BALANCE_COL,
                     ordering: Optional[List[str]] = None,
                     config: Optional[Dict[str, Any]] = None) -> RiskMonitorResult:
    """Aggregate transition matrix for *dimension* across two snapshots."""
    issues: List[dict] = []
    if ordering is None:
        ordering = get_ordering(config, dimension)

    if key not in baseline.columns or key not in current.columns:
        issues.append(make_issue(
            RM.MISSING_STABLE_KEY_FOR_MIGRATION, RM.ERROR,
            f"stable key {key!r} absent on baseline and/or current",
            field=key))
        return RiskMonitorResult("migration_matrix", pd.DataFrame(), issues,
                                 {"dimension": dimension})
    if dimension not in baseline.columns and dimension not in current.columns:
        issues.append(make_issue(
            RM.MISSING_MIGRATION_DIMENSION, RM.WARNING,
            f"migration dimension {dimension!r} absent on both frames",
            field=dimension))
        return RiskMonitorResult("migration_matrix", pd.DataFrame(), issues,
                                 {"dimension": dimension})

    b_dim, b_bal, b_keys = _prepared(baseline, key, dimension, balance_col)
    c_dim, c_bal, c_keys = _prepared(current, key, dimension, balance_col)

    rows: List[dict] = []
    any_unordered_change = False
    for k in b_keys | c_keys:
        in_b, in_c = k in b_keys, k in c_keys
        fv = b_dim.get(k) if in_b else None
        tv = c_dim.get(k) if in_c else None
        mt, _ord = _classify_row(in_b, in_c, fv, tv, ordering)
        if mt == RM.CHANGED:
            any_unordered_change = True
        bal = c_bal.get(k) if in_c else b_bal.get(k)
        rows.append({"from_value": _norm(fv), "to_value": _norm(tv),
                     "movement_type": mt,
                     "balance": float(bal) if bal == bal and bal is not None
                     else 0.0})

    if any_unordered_change and not ordering:
        issues.append(make_issue(
            RM.UNORDERED_MIGRATION_DIMENSION, RM.INFO,
            f"{dimension!r} has no config ordering; real changes are classified "
            f"as 'changed' (no improve/deteriorate direction invented)",
            field=dimension))

    detail = pd.DataFrame(rows)
    if detail.empty:
        issues.append(make_issue(RM.EMPTY_RISK_MONITOR_RESULT, RM.WARNING,
                                 "no loans to migrate"))
        return RiskMonitorResult("migration_matrix", detail, issues,
                                 {"dimension": dimension})

    grouped = (detail.groupby(["from_value", "to_value", "movement_type"],
                              dropna=False)
               .agg(loan_count=("movement_type", "size"),
                    balance_sum=("balance", "sum"))
               .reset_index())
    total = float(grouped["balance_sum"].sum())
    grouped["balance_share"] = (grouped["balance_sum"] / total
                                if total else 0.0)
    grouped["dimension"] = dimension
    grouped = grouped[["dimension", "from_value", "to_value", "loan_count",
                       "balance_sum", "balance_share", "movement_type"]]
    grouped = grouped.sort_values(
        ["movement_type", "from_value", "to_value"], kind="mergesort",
        na_position="last").reset_index(drop=True)

    meta = {"dimension": dimension, "ordered": bool(ordering),
            "n_loans": int(len(detail)),
            "n_baseline": len(b_keys), "n_current": len(c_keys)}
    return RiskMonitorResult("migration_matrix", grouped, issues, meta)


def per_loan_movement(baseline: pd.DataFrame, current: pd.DataFrame,
                      dimension: str, *, key: str = DEFAULT_KEY,
                      balance_col: str = BALANCE_COL,
                      ordering: Optional[List[str]] = None,
                      config: Optional[Dict[str, Any]] = None) -> RiskMonitorResult:
    """Per-loan movement flags for the union of baseline/current loans."""
    issues: List[dict] = []
    if ordering is None:
        ordering = get_ordering(config, dimension)

    if key not in baseline.columns or key not in current.columns:
        issues.append(make_issue(
            RM.MISSING_STABLE_KEY_FOR_MIGRATION, RM.ERROR,
            f"stable key {key!r} absent on baseline and/or current",
            field=key))
        return RiskMonitorResult("per_loan_movement", pd.DataFrame(), issues,
                                 {"dimension": dimension})
    if dimension not in baseline.columns and dimension not in current.columns:
        issues.append(make_issue(
            RM.MISSING_MIGRATION_DIMENSION, RM.WARNING,
            f"migration dimension {dimension!r} absent on both frames",
            field=dimension))
        return RiskMonitorResult("per_loan_movement", pd.DataFrame(), issues,
                                 {"dimension": dimension})

    b_dim, b_bal, b_keys = _prepared(baseline, key, dimension, balance_col)
    c_dim, c_bal, c_keys = _prepared(current, key, dimension, balance_col)

    rows: List[dict] = []
    any_unordered_change = False
    for k in b_keys | c_keys:
        in_b, in_c = k in b_keys, k in c_keys
        fv = b_dim.get(k) if in_b else None
        tv = c_dim.get(k) if in_c else None
        mt, _ord = _classify_row(in_b, in_c, fv, tv, ordering)
        if mt == RM.CHANGED:
            any_unordered_change = True
        bb = b_bal.get(k) if in_b else None
        cb = c_bal.get(k) if in_c else None
        bb = float(bb) if bb == bb and bb is not None else 0.0
        cb = float(cb) if cb == cb and cb is not None else 0.0
        rows.append({
            "loan_id": k, "baseline_value": _norm(fv), "current_value": _norm(tv),
            "movement_type": mt, "balance_baseline": bb, "balance_current": cb,
            "balance_change": cb - bb,
            "deterioration_flag": mt == RM.DETERIORATED,
            "improvement_flag": mt == RM.IMPROVED,
        })

    if any_unordered_change and not ordering:
        issues.append(make_issue(
            RM.UNORDERED_MIGRATION_DIMENSION, RM.INFO,
            f"{dimension!r} has no config ordering; deterioration/improvement "
            f"flags are not set for unordered changes", field=dimension))

    frame = pd.DataFrame(rows)
    if not frame.empty:
        frame = frame.sort_values("loan_id", kind="mergesort").reset_index(
            drop=True)
    meta = {"dimension": dimension, "ordered": bool(ordering),
            "deteriorated": int(sum(r["deterioration_flag"] for r in rows)),
            "improved": int(sum(r["improvement_flag"] for r in rows))}
    return RiskMonitorResult("per_loan_movement", frame, issues, meta)
