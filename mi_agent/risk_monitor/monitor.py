"""mi_agent.risk_monitor.monitor — store-backed risk-monitor entry points.

Phase 5. Thin, deterministic wrappers that resolve snapshots from a
``SnapshotStore``, assemble Phase 3/4 state frames, and run the migration /
concentration / trajectory primitives. Route eligibility is enforced from the
route contract. No orchestration runtime, no MI Agent wiring, no charts.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from analytics_lib.concentration import rag_status
from mi_agent.states import assemble_state
from mi_agent.states.route_contracts import canonical_state, load_route_contract
from mi_agent.states.temporal import trend as _trend
from snapshot.model import SnapshotNotFoundError
from snapshot.store import SnapshotStore

from . import models as RM
from .concentration import concentration_movement, funded_concentration
from .migration import migration_matrix, per_loan_movement
from .models import (
    BALANCE_COL,
    DEFAULT_KEY,
    RiskMonitorResult,
    get_concentration_thresholds,
    get_trajectory_window,
    make_issue,
)

_FORECAST_STATE = "total_forecast_funded"


def _metric_col(state: Optional[str], balance_col: str) -> str:
    if state and canonical_state(state) == _FORECAST_STATE:
        return "forecast_contribution"
    return balance_col


# --------------------------------------------------------------------------- #
# Route eligibility
# --------------------------------------------------------------------------- #


def validate_risk_monitor_route(route: str, *, allow_mna_override: bool = False,
                                routes_dir: Optional[Path] = None
                                ) -> Optional[Dict[str, Any]]:
    """Return an ``unsupported_risk_monitor_route`` issue unless *route* enables
    the risk monitor (M&A only when explicitly overridden)."""
    contract = load_route_contract(route, routes_dir=routes_dir)
    if contract.get("risk_monitor") == "enabled":
        return None
    if route == "mna" and allow_mna_override:
        return None
    return make_issue(
        RM.UNSUPPORTED_RISK_MONITOR_ROUTE, RM.WARNING,
        f"route {route!r} does not enable the risk monitor "
        f"(risk_monitor={contract.get('risk_monitor')!r})", field="route",
        route=route)


# --------------------------------------------------------------------------- #
# Snapshot/state frame resolution
# --------------------------------------------------------------------------- #


def _frame_for(store: SnapshotStore, client_id: str, date: Any,
               state: Optional[str], route: Optional[str],
               stage_probabilities: Optional[Dict[str, float]],
               missing_code: str) -> Tuple[Optional[pd.DataFrame], Any, List[dict]]:
    try:
        header = store.resolve_as_of(client_id, date, route=route)
    except SnapshotNotFoundError as exc:
        return None, None, [make_issue(missing_code, RM.ERROR, str(exc))]
    frame = store.load_loans(header.snapshot_id)
    if state is None:
        return frame, header, []
    kwargs: Dict[str, Any] = {}
    if canonical_state(state) == _FORECAST_STATE and stage_probabilities is not None:
        kwargs["stage_probabilities"] = stage_probabilities
    res = assemble_state(state, frame, route=None, **kwargs)
    return res.frame, header, list(res.issues)


# --------------------------------------------------------------------------- #
# Migration
# --------------------------------------------------------------------------- #


def run_migration(store: SnapshotStore, client_id: str, dimension: str, *,
                  route: str, baseline_date: Any, current_date: Any,
                  state: Optional[str] = None, key: str = DEFAULT_KEY,
                  balance_col: str = BALANCE_COL, config: Optional[dict] = None,
                  ordering: Optional[List[str]] = None,
                  allow_mna_override: bool = False,
                  routes_dir: Optional[Path] = None,
                  per_loan: bool = False) -> RiskMonitorResult:
    block = validate_risk_monitor_route(route, allow_mna_override=allow_mna_override,
                                        routes_dir=routes_dir)
    if block:
        return RiskMonitorResult("migration_matrix", pd.DataFrame(), [block],
                                 {"assembled": False, "dimension": dimension})
    issues: List[dict] = []
    base, _bh, bi = _frame_for(store, client_id, baseline_date, state, route,
                               None, RM.MISSING_BASELINE_SNAPSHOT)
    cur, _ch, ci = _frame_for(store, client_id, current_date, state, route,
                              None, RM.MISSING_CURRENT_SNAPSHOT)
    issues.extend(i for i in bi + ci if i["severity"] == RM.ERROR)
    if base is None or cur is None:
        return RiskMonitorResult("migration_matrix", pd.DataFrame(), issues,
                                 {"assembled": False, "dimension": dimension})
    fn = per_loan_movement if per_loan else migration_matrix
    res = fn(base, cur, dimension, key=key, balance_col=balance_col,
             ordering=ordering, config=config)
    res.issues = issues + res.issues
    res.metadata["assembled"] = True
    return res


# --------------------------------------------------------------------------- #
# Concentration
# --------------------------------------------------------------------------- #


def run_concentration(store: SnapshotStore, client_id: str, dimension: str, *,
                      route: str, state: str = "total_funded",
                      reporting_date: Any = None,
                      balance_col: str = BALANCE_COL,
                      config: Optional[dict] = None,
                      stage_probabilities: Optional[Dict[str, float]] = None,
                      allow_mna_override: bool = False,
                      routes_dir: Optional[Path] = None) -> RiskMonitorResult:
    block = validate_risk_monitor_route(route, allow_mna_override=allow_mna_override,
                                        routes_dir=routes_dir)
    if block:
        return RiskMonitorResult("funded_concentration", pd.DataFrame(), [block],
                                 {"assembled": False, "dimension": dimension})
    if reporting_date is None:
        try:
            header = store.resolve_latest(client_id, route=route)
        except SnapshotNotFoundError as exc:
            return RiskMonitorResult(
                "funded_concentration", pd.DataFrame(),
                [make_issue(RM.MISSING_CURRENT_SNAPSHOT, RM.ERROR, str(exc))],
                {"assembled": False, "dimension": dimension})
        reporting_date = header.reporting_date

    frame, _h, _i = _frame_for(store, client_id, reporting_date, state, route,
                               stage_probabilities, RM.MISSING_CURRENT_SNAPSHOT)
    if frame is None:
        return RiskMonitorResult(
            "funded_concentration", pd.DataFrame(),
            [make_issue(RM.MISSING_CURRENT_SNAPSHOT, RM.ERROR,
                        "no snapshot resolved")],
            {"assembled": False, "dimension": dimension})
    res = funded_concentration(frame, dimension,
                               balance_col=_metric_col(state, balance_col),
                               config=config)
    res.metadata["assembled"] = True
    res.metadata["state"] = state
    res.metadata["reporting_date"] = reporting_date
    return res


def run_concentration_movement(store: SnapshotStore, client_id: str,
                               dimension: str, *, route: str,
                               baseline_date: Any, current_date: Any,
                               state: str = "total_funded",
                               balance_col: str = BALANCE_COL,
                               config: Optional[dict] = None,
                               allow_mna_override: bool = False,
                               routes_dir: Optional[Path] = None
                               ) -> RiskMonitorResult:
    block = validate_risk_monitor_route(route, allow_mna_override=allow_mna_override,
                                        routes_dir=routes_dir)
    if block:
        return RiskMonitorResult("concentration_movement", pd.DataFrame(),
                                 [block], {"assembled": False,
                                           "dimension": dimension})
    issues: List[dict] = []
    base, _bh, bi = _frame_for(store, client_id, baseline_date, state, route,
                               None, RM.MISSING_BASELINE_SNAPSHOT)
    cur, _ch, ci = _frame_for(store, client_id, current_date, state, route,
                              None, RM.MISSING_CURRENT_SNAPSHOT)
    issues.extend(i for i in bi + ci if i["severity"] == RM.ERROR)
    if base is None or cur is None:
        return RiskMonitorResult("concentration_movement", pd.DataFrame(),
                                 issues, {"assembled": False,
                                          "dimension": dimension})
    metric = _metric_col(state, balance_col)
    res = concentration_movement(base, cur, dimension, balance_col=metric,
                                 config=config)
    res.issues = issues + res.issues
    res.metadata["assembled"] = True
    return res


def run_funded_vs_forecast(store: SnapshotStore, client_id: str, dimension: str,
                           *, route: str, reporting_date: Any = None,
                           balance_col: str = BALANCE_COL,
                           config: Optional[dict] = None,
                           stage_probabilities: Optional[Dict[str, float]] = None,
                           allow_mna_override: bool = False,
                           routes_dir: Optional[Path] = None) -> RiskMonitorResult:
    block = validate_risk_monitor_route(route, allow_mna_override=allow_mna_override,
                                        routes_dir=routes_dir)
    if block:
        return RiskMonitorResult("funded_vs_forecast", pd.DataFrame(), [block],
                                 {"assembled": False, "dimension": dimension})
    if reporting_date is None:
        try:
            reporting_date = store.resolve_latest(client_id,
                                                  route=route).reporting_date
        except SnapshotNotFoundError as exc:
            return RiskMonitorResult(
                "funded_vs_forecast", pd.DataFrame(),
                [make_issue(RM.MISSING_CURRENT_SNAPSHOT, RM.ERROR, str(exc))],
                {"assembled": False, "dimension": dimension})

    funded, _fh, _fi = _frame_for(store, client_id, reporting_date,
                                  "total_funded", route, None,
                                  RM.MISSING_CURRENT_SNAPSHOT)
    forecast, _xh, fi = _frame_for(store, client_id, reporting_date,
                                   "total_forecast_funded", route,
                                   stage_probabilities, RM.MISSING_CURRENT_SNAPSHOT)
    if funded is None or forecast is None:
        return RiskMonitorResult("funded_vs_forecast", pd.DataFrame(), [],
                                 {"assembled": False, "dimension": dimension})
    res = concentration_movement(
        funded, forecast, dimension, balance_col="forecast_contribution",
        baseline_balance_col=balance_col, config=config,
        kind="funded_vs_forecast")
    # Carry forecast-probability issues from the forecast assembly.
    res.issues = [i for i in fi if i["code"] in (
        "missing_forecast_probability", "forecast_probability_from_config")
        ] + res.issues
    res.metadata["assembled"] = True
    res.metadata["reporting_date"] = reporting_date
    return res


# --------------------------------------------------------------------------- #
# Trajectory (trend-based early warning)
# --------------------------------------------------------------------------- #


def run_trajectory(store: SnapshotStore, client_id: str, dimension: str, *,
                   route: str, start_date: Any, end_date: Any,
                   state: str = "total_funded", balance_col: str = BALANCE_COL,
                   config: Optional[dict] = None,
                   thresholds: Optional[Dict[str, float]] = None,
                   min_snapshots: Optional[int] = None,
                   stage_probabilities: Optional[Dict[str, float]] = None,
                   allow_mna_override: bool = False,
                   routes_dir: Optional[Path] = None) -> RiskMonitorResult:
    """Conservative trajectory warning: a group whose share is non-decreasing
    across the window and rises overall is flagged when its latest share is
    amber-or-worse."""
    block = validate_risk_monitor_route(route, allow_mna_override=allow_mna_override,
                                        routes_dir=routes_dir)
    if block:
        return RiskMonitorResult("trajectory", pd.DataFrame(), [block],
                                 {"assembled": False, "dimension": dimension})

    thresholds = thresholds or get_concentration_thresholds(config)
    min_snapshots = min_snapshots or get_trajectory_window(config)
    issues: List[dict] = []

    # Reuse the Phase 4 trend (route already validated ⇒ route=None here).
    t_kwargs: Dict[str, Any] = {"stratify_by": dimension}
    if canonical_state(state) == _FORECAST_STATE and stage_probabilities is not None:
        t_kwargs["stage_probabilities"] = stage_probabilities
    tr = _trend(store, state, client_id, route=None, start_date=start_date,
                end_date=end_date, balance_col=balance_col, **t_kwargs)
    df = tr.frame
    if df.empty or dimension not in df.columns:
        issues.append(make_issue(RM.EMPTY_RISK_MONITOR_RESULT, RM.WARNING,
                                 "no trend rows for trajectory"))
        return RiskMonitorResult("trajectory", pd.DataFrame(), issues,
                                 {"assembled": False, "dimension": dimension})

    dates = sorted(df["reporting_date"].unique())
    if len(dates) < min_snapshots:
        issues.append(make_issue(
            RM.INSUFFICIENT_SNAPSHOTS_FOR_TRAJECTORY, RM.WARNING,
            f"trajectory needs >={min_snapshots} snapshots; found {len(dates)}",
            count=len(dates)))

    totals = df.groupby("reporting_date")["balance"].sum().to_dict()
    rows: List[dict] = []
    for group, sub in df.groupby(dimension):
        by_date = dict(zip(sub["reporting_date"], sub["balance"]))
        shares = [(by_date.get(d, 0.0) / totals[d]) if totals.get(d) else 0.0
                  for d in dates]
        non_decreasing = all(shares[i] <= shares[i + 1] + 1e-12
                             for i in range(len(shares) - 1))
        increasing = non_decreasing and shares[-1] > shares[0]
        warning = bool(increasing and len(dates) >= min_snapshots
                       and shares[-1] >= thresholds["amber"])
        rows.append({
            "dimension": dimension, dimension: group,
            "n_snapshots": len(dates), "first_share": shares[0],
            "last_share": shares[-1], "share_change": shares[-1] - shares[0],
            "increasing": increasing,
            "status_last": rag_status(shares[-1], thresholds),
            "warning": warning,
        })
    frame = pd.DataFrame(rows).sort_values(
        "last_share", ascending=False, kind="mergesort").reset_index(drop=True)
    meta = {"assembled": True, "dimension": dimension, "n_snapshots": len(dates),
            "reporting_dates": dates, "thresholds": thresholds,
            "min_snapshots": min_snapshots}
    return RiskMonitorResult("trajectory", frame, issues, meta)
