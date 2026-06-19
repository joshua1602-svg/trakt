"""mi_agent.mi_spec_validation — MIQuerySpec v2 spec-level validation (Phase 7).

Mode-aware, route-aware validation of a :class:`MIQuerySpec` that complements
(does not replace) the v1 chart-structure validator in
``mi_agent.mi_query_validator``. It validates the *single governed contract* for
both the flat and the snapshot/state/temporal/risk paths and returns structured
issues — it never loosens existing validation and never executes anything.

No external LLM calls, no Azure, no chart rendering.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from mi_agent.mi_query_spec import (
    AMBIGUOUS_DIMENSION_TERMS,
    BUCKET_STRATEGIES,
    EXECUTION_MODES,
    FORECAST_PROBABILITY_SOURCES,
    MIQuerySpec,
    OUTPUT_TYPES,
    RISK_MONITOR_MODES,
    STATES,
    TEMPORAL_MODES,
    TREND_GRAINS,
)
from mi_agent.states.models import ERROR, WARNING, make_issue

# Issue codes (string values match the Phase 6 runtime where they overlap).
INVALID_ENUM_VALUE = "invalid_enum_value"
MISSING_REQUIRED_STATE = "missing_required_state"
MISSING_SNAPSHOT_CLIENT_ID = "missing_snapshot_client_id"
SNAPSHOT_STORE_REQUIRED = "snapshot_store_required"
TEMPORAL_SELECTOR_INCOMPLETE = "temporal_selector_incomplete"
INVALID_RISK_MONITOR_SPEC = "invalid_risk_monitor_spec"
INVALID_ROUTE_FOR_STATE = "invalid_route_for_state"
INVALID_TEMPORAL_MODE_FOR_ROUTE = "invalid_temporal_mode_for_route"
RISK_MONITOR_NOT_ENABLED = "risk_monitor_not_enabled"
VIRTUAL_FIELD_NOT_AVAILABLE_IN_FLAT_MODE = "virtual_field_not_available_in_flat_mode"
AMBIGUOUS_DIMENSION = "ambiguous_dimension"

_REGULATORY_PREFIX = "regulatory"


@dataclass
class SpecValidationResult:
    ok: bool = True
    issues: List[Dict[str, Any]] = field(default_factory=list)
    normalized: Optional[MIQuerySpec] = None

    def add(self, issue: Dict[str, Any]) -> None:
        self.issues.append(issue)
        if issue.get("severity") == ERROR:
            self.ok = False

    def codes(self) -> List[str]:
        return [i["code"] for i in self.issues]


def _check_enum(result: SpecValidationResult, value: Optional[str],
                allowed: Set[str], field_name: str) -> None:
    if value is not None and value not in allowed:
        result.add(make_issue(INVALID_ENUM_VALUE, ERROR,
                              f"{field_name}={value!r} not in {sorted(allowed)}",
                              field=field_name))


def validate_query_spec(spec: MIQuerySpec, *, semantics: Optional[dict] = None,
                        available_columns: Optional[Set[str]] = None,
                        routes_dir: Optional[Path] = None
                        ) -> SpecValidationResult:
    """Validate an MIQuerySpec v2. Returns a :class:`SpecValidationResult`."""
    from mi_agent.states.route_contracts import validate_state_for_route, \
        validate_temporal_request

    norm = spec.normalized()
    result = SpecValidationResult(normalized=norm)
    route = norm.route_id or "mi"
    mode = norm.effective_execution_mode()

    # 1. Enum validation (only when set).
    _check_enum(result, norm.execution_mode, EXECUTION_MODES, "execution_mode")
    _check_enum(result, norm.state, STATES, "state")
    _check_enum(result, norm.temporal_mode, TEMPORAL_MODES, "temporal_mode")
    _check_enum(result, norm.risk_monitor_mode, RISK_MONITOR_MODES,
                "risk_monitor_mode")
    _check_enum(result, norm.bucket_strategy, BUCKET_STRATEGIES, "bucket_strategy")
    _check_enum(result, norm.trend_grain, TREND_GRAINS, "trend_grain")
    _check_enum(result, norm.forecast_probability_source,
                FORECAST_PROBABILITY_SOURCES, "forecast_probability_source")
    _check_enum(result, norm.output_type, OUTPUT_TYPES, "output_type")

    # 2. Bare ambiguous dimension must be resolved before reaching a spec.
    if norm.dimension in AMBIGUOUS_DIMENSION_TERMS:
        result.add(make_issue(
            AMBIGUOUS_DIMENSION, ERROR,
            f"dimension {norm.dimension!r} is ambiguous; resolve it to a concrete "
            f"field before building a spec", field="dimension"))

    # 3. Mode-specific requirements.
    if mode == "flat":
        _validate_flat(result, norm, semantics, available_columns)
    elif mode in ("state", "snapshot"):
        _validate_state(result, norm)
    elif mode == "temporal":
        _validate_temporal(result, norm)
    elif mode == "risk":
        _validate_risk(result, norm)

    # 4. Route gating (skip for flat; flat is route-agnostic by design).
    if mode != "flat":
        _validate_route(result, norm, route, mode, routes_dir,
                        validate_state_for_route, validate_temporal_request)

    return result


def _validate_flat(result, norm, semantics, available_columns) -> None:
    if not semantics or available_columns is None:
        return
    fields = semantics.get("fields", {})
    cols = set(available_columns)
    for key in norm.referenced_fields():
        entry = fields.get(key)
        if entry and entry.get("virtual"):
            canon = entry.get("canonical_field", key)
            if canon not in cols and key not in cols:
                result.add(make_issue(
                    VIRTUAL_FIELD_NOT_AVAILABLE_IN_FLAT_MODE, ERROR,
                    f"{key!r} is a snapshot/state-only (virtual) field and is not "
                    f"present in the flat dataset", field=key))


def _validate_state(result, norm) -> None:
    if not norm.state:
        result.add(make_issue(MISSING_REQUIRED_STATE, ERROR,
                              "state mode requires a 'state'", field="state"))
    if not norm.snapshot_client_id:
        result.add(make_issue(MISSING_SNAPSHOT_CLIENT_ID, ERROR,
                              "state mode requires snapshot_client_id",
                              field="snapshot_client_id"))


def _validate_temporal(result, norm) -> None:
    if not norm.snapshot_client_id:
        result.add(make_issue(MISSING_SNAPSHOT_CLIENT_ID, ERROR,
                              "temporal mode requires snapshot_client_id",
                              field="snapshot_client_id"))
    if norm.temporal_mode == "compare" and not (norm.baseline_date
                                                and norm.current_date):
        result.add(make_issue(TEMPORAL_SELECTOR_INCOMPLETE, ERROR,
                              "compare requires baseline_date and current_date"))
    if norm.temporal_mode == "trend" and not (norm.start_date and norm.end_date):
        result.add(make_issue(TEMPORAL_SELECTOR_INCOMPLETE, ERROR,
                              "trend requires start_date and end_date"))


def _validate_risk(result, norm) -> None:
    if not norm.snapshot_client_id:
        result.add(make_issue(MISSING_SNAPSHOT_CLIENT_ID, ERROR,
                              "risk mode requires snapshot_client_id",
                              field="snapshot_client_id"))
    rmode = norm.risk_monitor_mode
    if rmode is None and isinstance(norm.risk_monitor, str) \
            and norm.risk_monitor in RISK_MONITOR_MODES:
        rmode = norm.risk_monitor
    if rmode not in RISK_MONITOR_MODES:
        result.add(make_issue(INVALID_RISK_MONITOR_SPEC, ERROR,
                              "risk mode requires a valid risk_monitor_mode "
                              f"{sorted(RISK_MONITOR_MODES)}", field="risk_monitor_mode"))
        return
    dim = norm.effective_risk_dimension()
    if not dim:
        result.add(make_issue(INVALID_RISK_MONITOR_SPEC, ERROR,
                              f"risk {rmode} requires a dimension", field="dimension"))
    if rmode in ("migration", "flags") and not (norm.baseline_date
                                                and norm.current_date):
        result.add(make_issue(TEMPORAL_SELECTOR_INCOMPLETE, ERROR,
                              f"risk {rmode} requires baseline_date and current_date"))
    if rmode == "trajectory" and not (norm.start_date and norm.end_date):
        result.add(make_issue(TEMPORAL_SELECTOR_INCOMPLETE, ERROR,
                              "risk trajectory requires start_date and end_date"))


def _validate_route(result, norm, route, mode, routes_dir,
                    validate_state_for_route, validate_temporal_request) -> None:
    # Regulatory routes reject all MI state/temporal/risk execution.
    if route.startswith(_REGULATORY_PREFIX):
        result.add(make_issue(INVALID_ROUTE_FOR_STATE, ERROR,
                              f"regulatory route {route!r} may not run MI "
                              f"{mode} queries", field="route_id", route=route))
        return

    if mode in ("state", "snapshot") and norm.state:
        block = validate_state_for_route(norm.state, route, routes_dir=routes_dir)
        if block:
            result.add(make_issue(INVALID_ROUTE_FOR_STATE, ERROR,
                                  block["message"], field="state", route=route))
    elif mode == "temporal" and norm.state:
        block = validate_temporal_request(norm.state, route, norm.temporal_mode,
                                          routes_dir=routes_dir)
        if block:
            code = (INVALID_TEMPORAL_MODE_FOR_ROUTE
                    if block["code"] == "unavailable_temporal_mode"
                    else INVALID_ROUTE_FOR_STATE)
            result.add(make_issue(code, ERROR, block["message"],
                                  field="state", route=route))
    elif mode == "risk":
        from mi_agent.risk_monitor import validate_risk_monitor_route
        block = validate_risk_monitor_route(route, routes_dir=routes_dir)
        if block:
            result.add(make_issue(RISK_MONITOR_NOT_ENABLED, ERROR,
                                  block["message"], field="route_id", route=route))
