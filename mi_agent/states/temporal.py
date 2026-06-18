"""mi_agent.states.temporal — deterministic temporal MI (compare / trend).

Phase 4. Adds compare-two-snapshots and trend-over-a-range capability on top of
the Phase 3 state assembler, driven by the Phase 2 ``SnapshotStore`` resolvers
and the Phase 0B route contracts. Pure, frame-in/frame-out, deterministic. No
orchestration, no MI Agent runtime wiring, no LLM, no Azure, no charts, no
risk-grade/PD migration matrices (Phase 5).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import pandas as pd

from analytics_lib import materialise_buckets as _materialise_buckets
from analytics_lib import stratify as _stratify
from snapshot.model import SnapshotNotFoundError
from snapshot.store import SnapshotStore

from . import models as M
from .assembler import BALANCE_COL, assemble_state
from .forecast import load_stage_probabilities
from .models import make_issue
from .route_contracts import canonical_state, validate_temporal_request

# Canonical state whose analytical measure is the forecast contribution.
_FORECAST_STATE = "total_forecast_funded"


@dataclass
class TemporalResult:
    """Outcome of a temporal (compare / trend) assembly."""

    mode: str
    state: str
    frame: pd.DataFrame
    issues: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return not any(i.get("severity") == M.ERROR for i in self.issues)

    @property
    def row_count(self) -> int:
        return int(len(self.frame))

    def issue_codes(self) -> List[str]:
        return [i["code"] for i in self.issues]


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _metric_col(state: str, balance_col: str) -> str:
    """Analytical measure column for a state."""
    return ("forecast_contribution"
            if canonical_state(state) == _FORECAST_STATE else balance_col)


def _apply_filters(frame: pd.DataFrame,
                   filters: Optional[Dict[str, Any]]) -> pd.DataFrame:
    if not filters:
        return frame
    out = frame
    for col, want in filters.items():
        if col not in out.columns:
            continue
        if isinstance(want, (list, tuple, set)):
            out = out[out[col].isin(list(want))]
        else:
            out = out[out[col] == want]
    return out


def _resolve_stage_probs(stage_probabilities: Optional[Dict[str, float]],
                         forecast_config_path: Optional[Path]
                         ) -> Optional[Dict[str, float]]:
    if stage_probabilities is not None:
        return stage_probabilities
    if forecast_config_path is not None:
        return load_stage_probabilities(forecast_config_path)
    return None


def _assemble(frame: pd.DataFrame, state: str,
              stage_probs: Optional[Dict[str, float]],
              state_kwargs: Dict[str, Any],
              filters: Optional[Dict[str, Any]]):
    work = _apply_filters(frame, filters)
    kwargs = dict(state_kwargs)
    if canonical_state(state) == _FORECAST_STATE and stage_probs is not None:
        kwargs["stage_probabilities"] = stage_probs
    # route already validated at the temporal layer ⇒ route=None here.
    return assemble_state(state, work, route=None, **kwargs)


def _agg_total(frame: pd.DataFrame, metric: str) -> Tuple[int, float]:
    if frame.empty:
        return 0, 0.0
    balance = (float(pd.to_numeric(frame[metric], errors="coerce").sum())
               if metric in frame.columns else 0.0)
    return int(len(frame)), balance


def _movement(base: pd.DataFrame, current: pd.DataFrame,
              issues: List[dict], key: str = "loan_id"
              ) -> Tuple[Optional[int], Optional[int], Optional[int]]:
    """Return ``(new, exited, retained)`` counts using a stable key."""
    if key not in base.columns or key not in current.columns:
        issues.append(make_issue(
            M.MISSING_STABLE_KEY_FOR_MOVEMENT, M.WARNING,
            f"stable key {key!r} not present on both frames; movement counts "
            f"unavailable", field=key))
        return None, None, None
    b = set(base[key].dropna())
    c = set(current[key].dropna())
    return len(c - b), len(b - c), len(b & c)


def _pct_change(baseline: float, current: float,
                issues: List[dict]) -> Optional[float]:
    if baseline == 0:
        issues.append(make_issue(
            M.PERCENTAGE_CHANGE_DIVIDE_BY_ZERO, M.INFO,
            "baseline value is zero; percentage change is undefined"))
        return None
    return (current - baseline) / baseline * 100.0


def _ensure_dimension(frame: pd.DataFrame, dim: str
                      ) -> Tuple[pd.DataFrame, Optional[dict]]:
    if dim in frame.columns:
        return frame, None
    try:
        out, _bissues, applied = _materialise_buckets(frame, buckets=[dim])
        if applied.get(dim):
            return out, None
    except Exception:  # pragma: no cover - defensive
        pass
    return frame, make_issue(M.UNAVAILABLE_DIMENSION, M.WARNING,
                             f"dimension {dim!r} not available for grouping",
                             field=dim)


def _carry_forecast_issues(res, issues: List[dict], seen: set) -> None:
    for i in res.issues:
        if i["code"] in (M.MISSING_FORECAST_PROBABILITY,
                         M.FORECAST_PROBABILITY_FROM_CONFIG) and \
                i["code"] not in seen:
            issues.append(i)
            seen.add(i["code"])


# --------------------------------------------------------------------------- #
# Compare
# --------------------------------------------------------------------------- #


def compare(store: SnapshotStore, state: str, client_id: str, *,
            route: Optional[str] = None, baseline_date: Any, current_date: Any,
            balance_col: str = BALANCE_COL, stratify_by: Optional[str] = None,
            segment: Optional[str] = None,
            filters: Optional[Dict[str, Any]] = None,
            stage_probabilities: Optional[Dict[str, float]] = None,
            forecast_config_path: Optional[Path] = None,
            routes_dir: Optional[Path] = None, **state_kwargs) -> TemporalResult:
    """Compare a state between a baseline and a current snapshot."""
    meta: Dict[str, Any] = {"mode": "compare", "state": state,
                            "client_id": client_id, "route": route}
    if route:
        block = validate_temporal_request(state, route, "compare",
                                          routes_dir=routes_dir)
        if block:
            meta["assembled"] = False
            return TemporalResult("compare", state, pd.DataFrame(), [block], meta)

    issues: List[dict] = []
    base_h = cur_h = None
    try:
        base_h = store.resolve_as_of(client_id, baseline_date, route=route)
    except SnapshotNotFoundError as exc:
        issues.append(make_issue(M.MISSING_BASELINE_SNAPSHOT, M.ERROR, str(exc)))
    try:
        cur_h = store.resolve_as_of(client_id, current_date, route=route)
    except SnapshotNotFoundError as exc:
        issues.append(make_issue(M.MISSING_CURRENT_SNAPSHOT, M.ERROR, str(exc)))
    if base_h is None or cur_h is None:
        meta["assembled"] = False
        return TemporalResult("compare", state, pd.DataFrame(), issues, meta)

    stage_probs = _resolve_stage_probs(stage_probabilities, forecast_config_path)
    base_res = _assemble(store.load_loans(base_h.snapshot_id), state,
                         stage_probs, state_kwargs, filters)
    cur_res = _assemble(store.load_loans(cur_h.snapshot_id), state,
                        stage_probs, state_kwargs, filters)
    seen: set = set()
    _carry_forecast_issues(base_res, issues, seen)
    _carry_forecast_issues(cur_res, issues, seen)

    metric = _metric_col(state, balance_col)
    group_by = stratify_by or segment
    meta.update({
        "assembled": True,
        "baseline_reporting_date": base_h.reporting_date,
        "current_reporting_date": cur_h.reporting_date,
        "baseline_snapshot_id": base_h.snapshot_id,
        "current_snapshot_id": cur_h.snapshot_id,
        "metric_column": metric,
    })

    if group_by:
        frame = _compare_grouped(base_res.frame, cur_res.frame, group_by,
                                 metric, issues)
        meta["group_by"] = group_by
    else:
        frame = _compare_total(base_res.frame, cur_res.frame, metric, state,
                               base_h, cur_h, issues, meta)

    if frame.empty:
        issues.append(make_issue(M.EMPTY_TEMPORAL_RESULT, M.WARNING,
                                 "compare produced no rows"))
    return TemporalResult("compare", state, frame, issues, meta)


def _compare_total(base: pd.DataFrame, current: pd.DataFrame, metric: str,
                   state: str, base_h, cur_h, issues: List[dict],
                   meta: Dict[str, Any]) -> pd.DataFrame:
    b_count, b_bal = _agg_total(base, metric)
    c_count, c_bal = _agg_total(current, metric)
    new, exited, retained = _movement(base, current, issues)
    bal_pct = _pct_change(b_bal, c_bal, issues)
    count_pct = _pct_change(float(b_count), float(c_count), issues)
    meta.update({"new_count": new, "exited_count": exited,
                 "retained_count": retained})
    row = {
        "state": state,
        "baseline_reporting_date": base_h.reporting_date,
        "current_reporting_date": cur_h.reporting_date,
        "baseline_count": b_count, "current_count": c_count,
        "count_change": c_count - b_count, "count_pct_change": count_pct,
        "baseline_balance": b_bal, "current_balance": c_bal,
        "balance_change": c_bal - b_bal, "balance_pct_change": bal_pct,
        "new_count": new, "exited_count": exited, "retained_count": retained,
    }
    return pd.DataFrame([row])


def _compare_grouped(base: pd.DataFrame, current: pd.DataFrame, group_by: str,
                     metric: str, issues: List[dict]) -> pd.DataFrame:
    base, b_issue = _ensure_dimension(base, group_by)
    current, c_issue = _ensure_dimension(current, group_by)
    for iss in (b_issue, c_issue):
        if iss and iss not in issues:
            issues.append(iss)
    if (group_by not in base.columns) or (group_by not in current.columns):
        return pd.DataFrame()

    def _table(frame: pd.DataFrame) -> pd.DataFrame:
        if frame.empty:
            return pd.DataFrame(columns=[group_by, "loan_count", "balance_sum"])
        t = _stratify(frame, group_by, metric)
        return t[[group_by, "loan_count", "balance_sum"]]

    bt = _table(base).rename(columns={"loan_count": "baseline_count",
                                      "balance_sum": "baseline_balance"})
    ct = _table(current).rename(columns={"loan_count": "current_count",
                                         "balance_sum": "current_balance"})
    merged = bt.merge(ct, on=group_by, how="outer").fillna(
        {"baseline_count": 0, "current_count": 0,
         "baseline_balance": 0.0, "current_balance": 0.0})
    merged["count_change"] = merged["current_count"] - merged["baseline_count"]
    merged["balance_change"] = (merged["current_balance"]
                                - merged["baseline_balance"])
    merged["balance_pct_change"] = [
        _pct_change(float(b), float(c), issues)
        for b, c in zip(merged["baseline_balance"], merged["current_balance"])]
    return merged.sort_values(group_by, kind="mergesort").reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Trend
# --------------------------------------------------------------------------- #


def trend(store: SnapshotStore, state: str, client_id: str, *,
          route: Optional[str] = None, start_date: Any, end_date: Any,
          balance_col: str = BALANCE_COL, stratify_by: Optional[str] = None,
          segment: Optional[str] = None,
          filters: Optional[Dict[str, Any]] = None,
          stage_probabilities: Optional[Dict[str, float]] = None,
          forecast_config_path: Optional[Path] = None,
          routes_dir: Optional[Path] = None, **state_kwargs) -> TemporalResult:
    """Assemble a state across an ordered range of snapshots."""
    meta: Dict[str, Any] = {"mode": "trend", "state": state,
                            "client_id": client_id, "route": route}
    if route:
        block = validate_temporal_request(state, route, "trend",
                                          routes_dir=routes_dir)
        if block:
            meta["assembled"] = False
            return TemporalResult("trend", state, pd.DataFrame(), [block], meta)

    issues: List[dict] = []
    headers = store.resolve_range(client_id, start_date, end_date, route=route)
    meta["n_snapshots"] = len(headers)
    if not headers:
        issues.append(make_issue(M.EMPTY_TEMPORAL_RESULT, M.WARNING,
                                 "no snapshots in the requested range"))
        meta["assembled"] = False
        return TemporalResult("trend", state, pd.DataFrame(), issues, meta)
    if len(headers) < 2:
        issues.append(make_issue(
            M.INSUFFICIENT_SNAPSHOTS_FOR_TREND, M.WARNING,
            f"a trend needs >=2 snapshots; found {len(headers)}"))

    stage_probs = _resolve_stage_probs(stage_probabilities, forecast_config_path)
    metric = _metric_col(state, balance_col)
    group_by = stratify_by or segment
    rows: List[dict] = []
    seen: set = set()

    for h in headers:  # resolve_range returns ascending reporting_date order
        res = _assemble(store.load_loans(h.snapshot_id), state, stage_probs,
                        state_kwargs, filters)
        _carry_forecast_issues(res, issues, seen)
        if group_by:
            framed, dissue = _ensure_dimension(res.frame, group_by)
            if dissue:
                if dissue not in issues:
                    issues.append(dissue)
                continue
            if framed.empty:
                continue
            table = _stratify(framed, group_by, metric)
            for _, r in table.iterrows():
                rows.append({
                    "reporting_date": h.reporting_date,
                    "snapshot_id": h.snapshot_id, "state": state,
                    group_by: r[group_by],
                    "count": int(r["loan_count"]),
                    "balance": float(r["balance_sum"]),
                })
        else:
            count, balance = _agg_total(res.frame, metric)
            rows.append({"reporting_date": h.reporting_date,
                         "snapshot_id": h.snapshot_id, "state": state,
                         "count": count, "balance": balance})

    frame = pd.DataFrame(rows)
    if not frame.empty:
        sort_cols = (["reporting_date", group_by] if group_by
                     else ["reporting_date"])
        frame = frame.sort_values(sort_cols, kind="mergesort").reset_index(
            drop=True)
    else:
        issues.append(make_issue(M.EMPTY_TEMPORAL_RESULT, M.WARNING,
                                 "trend produced no rows"))
    meta.update({"assembled": True, "group_by": group_by, "metric_column": metric,
                 "reporting_dates": [h.reporting_date for h in headers]})
    return TemporalResult("trend", state, frame, issues, meta)


# --------------------------------------------------------------------------- #
# Dispatcher
# --------------------------------------------------------------------------- #


def assemble_temporal(store: SnapshotStore, state: str, client_id: str, *,
                      mode: str = "compare", route: Optional[str] = None,
                      **kwargs) -> TemporalResult:
    """Dispatch by temporal *mode* (``compare`` / ``trend``)."""
    if mode == "compare":
        return compare(store, state, client_id, route=route, **kwargs)
    if mode in ("trend", "range"):
        return trend(store, state, client_id, route=route, **kwargs)
    return TemporalResult(
        mode, state, pd.DataFrame(),
        [make_issue(M.UNAVAILABLE_TEMPORAL_MODE, M.WARNING,
                    f"unsupported temporal mode {mode!r}")],
        {"assembled": False, "mode": mode, "state": state})
