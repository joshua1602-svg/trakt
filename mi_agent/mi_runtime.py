"""mi_agent.mi_runtime — governed, read-only MI runtime dispatch (Phase 6).

Wires the Phase 2/3/4/5 foundations into the MI Agent behind ONE explicit
runtime boundary, preserving the existing flat single-CSV path. Inspects a
validated :class:`MIQuerySpec`, decides flat vs state vs temporal vs risk, runs
the right governed engine, and returns a consistent :class:`RuntimeResult`
(DataFrame + metadata + structured issues + an optional, governed chart
instruction). No UI, no LLM, no Azure, no new chart types.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from mi_agent.mi_query_executor import (
    MIQueryExecutionError,
    MIQueryResult,
    execute_mi_query,
)
from mi_agent.mi_query_spec import CHART_TYPES, MIQuerySpec
from mi_agent.mi_query_validator import load_mi_semantics
from mi_agent.states import assemble_state
from mi_agent.states.models import ERROR, INFO, WARNING, make_issue
from mi_agent.states.route_contracts import (
    canonical_state,
    load_route_contract,
    validate_state_for_route,
    validate_temporal_request,
)
from mi_agent.states.selectors import SnapshotSelector
from mi_agent.states.temporal import compare as temporal_compare
from mi_agent.states.temporal import trend as temporal_trend

# Phase 6 runtime issue codes.
SNAPSHOT_STORE_REQUIRED = "snapshot_store_required"
SNAPSHOT_STORE_MISSING = "snapshot_store_missing"
INVALID_ROUTE_FOR_STATE = "invalid_route_for_state"
INVALID_TEMPORAL_MODE_FOR_ROUTE = "invalid_temporal_mode_for_route"
UNSUPPORTED_CHART_FOR_RESULT = "unsupported_chart_for_result"
STATE_RESULT_EMPTY = "state_result_empty"
RISK_MONITOR_NOT_ENABLED = "risk_monitor_not_enabled"
TEMPORAL_SELECTOR_INCOMPLETE = "temporal_selector_incomplete"
MISSING_SNAPSHOT_CLIENT_ID = "missing_snapshot_client_id"
FALLBACK_TO_FLAT_EXECUTOR = "fallback_to_flat_executor"
VIRTUAL_FIELD_NOT_AVAILABLE_IN_FLAT_MODE = "virtual_field_not_available_in_flat_mode"
DATA_REQUIRED_FOR_FLAT = "data_required_for_flat_mode"

_FORECAST_STATE = "total_forecast_funded"
_REGULATORY_PREFIX = "regulatory"


@dataclass
class RuntimeResult:
    """Unified result for any MI runtime path."""

    mode: str                       # flat|state|temporal|risk
    result_type: str
    data: pd.DataFrame
    issues: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)
    chart_instruction: Optional[Dict[str, Any]] = None

    @property
    def ok(self) -> bool:
        return not any(i.get("severity") == ERROR for i in self.issues)

    @property
    def row_count(self) -> int:
        return int(len(self.data)) if self.data is not None else 0

    def issue_codes(self) -> List[str]:
        return [i["code"] for i in self.issues]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "mode": self.mode, "result_type": self.result_type,
            "row_count": self.row_count, "issues": self.issues,
            "metadata": self.metadata, "warnings": self.warnings,
            "chart_instruction": self.chart_instruction,
        }


# --------------------------------------------------------------------------- #
# Mode inference
# --------------------------------------------------------------------------- #


def infer_execution_mode(spec: MIQuerySpec) -> str:
    if spec.execution_mode:
        return spec.execution_mode
    if spec.risk_monitor:
        return "risk"
    if spec.temporal_mode in ("compare", "trend"):
        return "temporal"
    if spec.state:
        return "state"
    return "flat"


def _governed_chart(chart_type: Optional[str]) -> Optional[Dict[str, Any]]:
    if not chart_type or chart_type == "none":
        return None
    if chart_type not in CHART_TYPES:
        return None
    return {"chart_type": chart_type}


# --------------------------------------------------------------------------- #
# Snapshot store / client resolution
# --------------------------------------------------------------------------- #


def _resolve_store(store, store_root: Optional[str], spec: MIQuerySpec,
                   issues: List[dict]):
    if store is not None:
        return store
    root = store_root or spec.snapshot_store_root
    if not root:
        issues.append(make_issue(SNAPSHOT_STORE_REQUIRED, ERROR,
                                 "a snapshot store is required for "
                                 "state/temporal/risk queries"))
        return None
    if not Path(root).exists():
        issues.append(make_issue(SNAPSHOT_STORE_MISSING, ERROR,
                                 f"snapshot store root {root!r} does not exist"))
        return None
    from snapshot.adapters import LocalFsSnapshotStore
    return LocalFsSnapshotStore(root=root)


def _require_client(spec: MIQuerySpec, issues: List[dict]) -> Optional[str]:
    if not spec.snapshot_client_id:
        issues.append(make_issue(MISSING_SNAPSHOT_CLIENT_ID, ERROR,
                                 "snapshot_client_id is required"))
        return None
    return spec.snapshot_client_id


# --------------------------------------------------------------------------- #
# Flat path
# --------------------------------------------------------------------------- #


def _run_flat(spec: MIQuerySpec, data, semantics: dict, *,
              build_chart: bool) -> RuntimeResult:
    issues: List[dict] = []
    if data is None:
        issues.append(make_issue(DATA_REQUIRED_FOR_FLAT, ERROR,
                                 "flat mode requires a DataFrame/CSV"))
        return RuntimeResult("flat", "table", pd.DataFrame(), issues)

    df = data if isinstance(data, pd.DataFrame) else pd.read_csv(data)
    cols = set(df.columns)
    fields = semantics.get("fields", {})

    # Virtual snapshot-only fields cannot be served by the flat single-CSV path.
    for key in spec.referenced_fields():
        entry = fields.get(key)
        if entry and entry.get("virtual"):
            canon = entry.get("canonical_field", key)
            if canon not in cols and key not in cols:
                issues.append(make_issue(
                    VIRTUAL_FIELD_NOT_AVAILABLE_IN_FLAT_MODE, ERROR,
                    f"field {key!r} is a snapshot/state-only (virtual) field and "
                    f"is not present in the flat dataset; use a state/snapshot "
                    f"query", field=key))
    if any(i["severity"] == ERROR for i in issues):
        return RuntimeResult("flat", "table", pd.DataFrame(), issues)

    try:
        res: MIQueryResult = execute_mi_query(spec, df, semantics)
    except MIQueryExecutionError as exc:
        issues.append(make_issue("flat_execution_error", ERROR, str(exc)))
        return RuntimeResult("flat", "table", pd.DataFrame(), issues)

    chart_instruction = _governed_chart(spec.chart_type)
    meta = dict(res.metadata)
    meta["resolved_fields"] = res.resolved_fields
    if build_chart and chart_instruction:
        meta["chart"] = _safe_build_chart(res, semantics, issues)
    return RuntimeResult("flat", res.result_type, res.data,
                         issues + [], meta, list(res.warnings), chart_instruction)


def _safe_build_chart(result: MIQueryResult, semantics: dict,
                      issues: List[dict]):
    """Render via the existing governed chart factory; never raise."""
    try:
        from mi_agent.mi_chart_factory import create_mi_chart
        chart = create_mi_chart(result, semantics)
        return {"chart_type": getattr(chart, "chart_type", None), "rendered": True}
    except Exception as exc:  # pragma: no cover - defensive
        issues.append(make_issue(UNSUPPORTED_CHART_FOR_RESULT, WARNING,
                                 f"chart factory could not render: {exc}"))
        return None


# --------------------------------------------------------------------------- #
# Route gating
# --------------------------------------------------------------------------- #


def _route_is_regulatory(route: str) -> bool:
    return route.startswith(_REGULATORY_PREFIX)


def _gate_state_route(spec: MIQuerySpec, route: str, routes_dir) -> List[dict]:
    if _route_is_regulatory(route):
        return [make_issue(INVALID_ROUTE_FOR_STATE, ERROR,
                           f"regulatory route {route!r} may not run MI state "
                           f"queries", field=spec.state)]
    block = validate_state_for_route(spec.state, route, routes_dir=routes_dir)
    if block:
        return [make_issue(INVALID_ROUTE_FOR_STATE, ERROR, block["message"],
                           field=spec.state)]
    return []


# --------------------------------------------------------------------------- #
# State path
# --------------------------------------------------------------------------- #


def _run_state(spec, semantics, store, store_root, *, routes_dir,
               stage_probabilities) -> RuntimeResult:
    issues: List[dict] = []
    route = spec.route_id or "mi"
    issues += _gate_state_route(spec, route, routes_dir)
    client = _require_client(spec, issues)
    resolved_store = _resolve_store(store, store_root, spec, issues)
    if any(i["severity"] == ERROR for i in issues):
        return RuntimeResult("state", "state", pd.DataFrame(), issues,
                             {"state": spec.state})

    selector = (SnapshotSelector.as_of(client, spec.as_of_date, route=route)
                if spec.temporal_mode == "as_of" and spec.as_of_date
                else SnapshotSelector.latest(client, route=route))
    kwargs: Dict[str, Any] = {}
    if canonical_state(spec.state) == _FORECAST_STATE and stage_probabilities is not None:
        kwargs["stage_probabilities"] = stage_probabilities
    sr = assemble_state(spec.state, resolved_store, selector=selector,
                        route=None, **kwargs)
    issues += list(sr.issues)
    if sr.frame is None or sr.frame.empty:
        issues.append(make_issue(STATE_RESULT_EMPTY, WARNING,
                                 f"state {spec.state!r} produced no rows"))
    meta = dict(sr.metadata); meta["state"] = spec.state
    return RuntimeResult("state", "state", sr.frame, issues, meta)


# --------------------------------------------------------------------------- #
# Temporal path
# --------------------------------------------------------------------------- #


def _run_temporal(spec, semantics, store, store_root, *, routes_dir,
                  stage_probabilities) -> RuntimeResult:
    issues: List[dict] = []
    route = spec.route_id or "mi"
    mode = spec.temporal_mode
    if _route_is_regulatory(route):
        issues.append(make_issue(INVALID_ROUTE_FOR_STATE, ERROR,
                                 f"regulatory route {route!r} may not run "
                                 f"temporal MI", field=spec.state))
    else:
        block = validate_temporal_request(spec.state, route, mode,
                                          routes_dir=routes_dir)
        if block:
            issues.append(make_issue(INVALID_TEMPORAL_MODE_FOR_ROUTE, ERROR,
                                     block["message"], field=spec.state))
    client = _require_client(spec, issues)
    resolved_store = _resolve_store(store, store_root, spec, issues)
    if any(i["severity"] == ERROR for i in issues):
        return RuntimeResult("temporal", mode or "temporal", pd.DataFrame(),
                             issues, {"state": spec.state})

    common = dict(route=route, balance_col="current_outstanding_balance",
                  stratify_by=spec.dimension, segment=spec.segment,
                  stage_probabilities=stage_probabilities, routes_dir=routes_dir)
    if mode == "compare":
        if not (spec.baseline_date and spec.current_date):
            issues.append(make_issue(TEMPORAL_SELECTOR_INCOMPLETE, ERROR,
                                     "compare requires baseline_date and "
                                     "current_date"))
            return RuntimeResult("temporal", "compare", pd.DataFrame(), issues)
        tr = temporal_compare(resolved_store, spec.state, client,
                              baseline_date=spec.baseline_date,
                              current_date=spec.current_date, **common)
        chart = _governed_chart("bar")
    else:  # trend
        if not (spec.start_date and spec.end_date):
            issues.append(make_issue(TEMPORAL_SELECTOR_INCOMPLETE, ERROR,
                                     "trend requires start_date and end_date"))
            return RuntimeResult("temporal", "trend", pd.DataFrame(), issues)
        tr = temporal_trend(resolved_store, spec.state, client,
                            start_date=spec.start_date, end_date=spec.end_date,
                            **common)
        chart = _governed_chart("line")   # governed line path for trends
    issues += list(tr.issues)
    meta = dict(tr.metadata)
    return RuntimeResult("temporal", tr.mode, tr.frame, issues, meta, [], chart)


# --------------------------------------------------------------------------- #
# Risk path
# --------------------------------------------------------------------------- #


def _infer_risk_kind(spec: MIQuerySpec) -> str:
    rm = spec.risk_monitor
    if isinstance(rm, str) and rm in ("migration", "concentration", "trajectory"):
        return rm
    if spec.baseline_date and spec.current_date:
        return "migration"
    if spec.start_date and spec.end_date:
        return "trajectory"
    return "concentration"


def _run_risk(spec, semantics, store, store_root, *, routes_dir, config,
              allow_mna_risk, stage_probabilities) -> RuntimeResult:
    from mi_agent.risk_monitor import (
        run_concentration, run_migration, run_trajectory,
        validate_risk_monitor_route,
    )
    issues: List[dict] = []
    route = spec.route_id or "mi"
    kind = _infer_risk_kind(spec)
    block = validate_risk_monitor_route(route, allow_mna_override=allow_mna_risk,
                                        routes_dir=routes_dir)
    if block:
        issues.append(make_issue(RISK_MONITOR_NOT_ENABLED, ERROR,
                                 block["message"], field="route", route=route))
    client = _require_client(spec, issues)
    resolved_store = _resolve_store(store, store_root, spec, issues)
    if any(i["severity"] == ERROR for i in issues):
        return RuntimeResult("risk", kind, pd.DataFrame(), issues,
                             {"risk_kind": kind})

    dim = spec.dimension
    if kind == "migration":
        if not (spec.baseline_date and spec.current_date):
            issues.append(make_issue(TEMPORAL_SELECTOR_INCOMPLETE, ERROR,
                                     "migration requires baseline_date and "
                                     "current_date"))
            return RuntimeResult("risk", "migration", pd.DataFrame(), issues)
        rr = run_migration(resolved_store, client, dim, route=route,
                           baseline_date=spec.baseline_date,
                           current_date=spec.current_date, config=config,
                           allow_mna_override=allow_mna_risk, routes_dir=routes_dir)
        chart = _governed_chart("heatmap")
    elif kind == "trajectory":
        if not (spec.start_date and spec.end_date):
            issues.append(make_issue(TEMPORAL_SELECTOR_INCOMPLETE, ERROR,
                                     "trajectory requires start_date and "
                                     "end_date"))
            return RuntimeResult("risk", "trajectory", pd.DataFrame(), issues)
        rr = run_trajectory(resolved_store, client, dim, route=route,
                            start_date=spec.start_date, end_date=spec.end_date,
                            config=config, allow_mna_override=allow_mna_risk,
                            routes_dir=routes_dir)
        chart = _governed_chart("line")
    else:  # concentration
        rr = run_concentration(resolved_store, client, dim, route=route,
                               state=spec.state or "total_funded",
                               reporting_date=spec.as_of_date, config=config,
                               stage_probabilities=stage_probabilities,
                               allow_mna_override=allow_mna_risk,
                               routes_dir=routes_dir)
        chart = _governed_chart("bar")
    issues += list(rr.issues)
    meta = dict(rr.metadata); meta["risk_kind"] = kind
    return RuntimeResult("risk", rr.kind, rr.frame, issues, meta, [], chart)


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #


def run_mi_query(spec: MIQuerySpec, *, semantics, data=None, store=None,
                 store_root: Optional[str] = None, routes_dir=None,
                 risk_config: Optional[dict] = None, allow_mna_risk: bool = False,
                 stage_probabilities: Optional[Dict[str, float]] = None,
                 build_chart: bool = False) -> RuntimeResult:
    """Dispatch a validated :class:`MIQuerySpec` to the right governed engine.

    Flat mode preserves the existing single-CSV MI Agent exactly; state /
    temporal / risk modes run behind this boundary using a ``SnapshotStore``.
    """
    if isinstance(semantics, (str, Path)):
        semantics = load_mi_semantics(semantics)
    mode = infer_execution_mode(spec)

    if mode == "flat":
        result = _run_flat(spec, data, semantics, build_chart=build_chart)
    elif mode == "state":
        result = _run_state(spec, semantics, store, store_root,
                            routes_dir=routes_dir,
                            stage_probabilities=stage_probabilities)
    elif mode == "temporal":
        result = _run_temporal(spec, semantics, store, store_root,
                               routes_dir=routes_dir,
                               stage_probabilities=stage_probabilities)
    elif mode == "risk":
        result = _run_risk(spec, semantics, store, store_root, routes_dir=routes_dir,
                           config=risk_config, allow_mna_risk=allow_mna_risk,
                           stage_probabilities=stage_probabilities)
    else:
        return RuntimeResult(mode, "table", pd.DataFrame(),
                             [make_issue("unknown_execution_mode", ERROR,
                                         f"unknown execution_mode {mode!r}")])

    # Surface the active source-portfolio lens in the result metadata + title so
    # every table/chart/card states which book it covers (Total/Direct/Acquired/
    # specific cohort).
    lens = getattr(spec, "portfolio_lens", None)
    if lens:
        result.metadata["portfolio_lens"] = lens
        label = lens.get("label")
        if label and lens.get("name") != "total":
            existing = result.metadata.get("title")
            if existing and label not in str(existing):
                result.metadata["title"] = f"{existing} — {label}"
            elif not existing:
                result.metadata["portfolio_lens_label"] = label
    return result
