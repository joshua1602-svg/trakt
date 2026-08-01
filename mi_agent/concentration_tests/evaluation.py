"""mi_agent.concentration_tests.evaluation — the single evaluation service.

Evaluates an activated, operator-approved concentration-test configuration
against the funded canonical frame (and the governed prior frame, when one
exists). The React dashboard, MI Query and Copilot all consume THIS output —
none of them recalculates.

Fail-closed rules:

* an unapproved, unknown or unsupported test never evaluates here — only
  ``ActiveConfiguration`` versions minted by the approval workflow arrive;
* a missing field, empty frame or zero denominator yields ``unavailable`` /
  ``insufficient_data`` — never PASS, never zero;
* an external-index test with no configured provider is ``unavailable`` with
  the reason named;
* a test whose effective date is after the reporting date is
  ``pending_effective_date``; past its expiry date it is ``expired``;
* the prior period is compared only when the governed loader supplied a prior
  frame — never an arbitrary file — and its absence is disclosed explicitly.
"""

from __future__ import annotations

import datetime as _dt
from typing import Any, Dict, List, Optional

import pandas as pd

from .library import ConcentrationLibrary
from .metrics import (
    ExternalIndexProvider,
    MetricComputation,
    evaluate_metric,
    resolve_role_column,
)
from .models import (
    ActiveConfiguration,
    ActiveTest,
    DATA_EXTERNAL_UNCONFIGURED,
    DATA_MISSING,
    DATA_OK,
    OPERATOR_MIN,
    STATUS_BREACH,
    STATUS_EXPIRED,
    STATUS_INSUFFICIENT_DATA,
    STATUS_PASS,
    STATUS_PENDING_EFFECTIVE_DATE,
    STATUS_UNAVAILABLE,
    STATUS_WARNING,
)

#: Ordering used to detect deteriorations (higher = worse). Non-measured
#: statuses are not ordered against measured ones.
_SEVERITY_ORDER = {STATUS_PASS: 0, STATUS_WARNING: 1, STATUS_BREACH: 2}


def _now_iso() -> str:
    return _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds")


def _parse_date(value: str) -> Optional[_dt.date]:
    try:
        return _dt.date.fromisoformat(str(value)[:10])
    except (ValueError, TypeError):
        return None


def _status_for(value: Optional[float], threshold: Optional[float],
                operator: str, warning_fraction: float) -> str:
    if value is None:
        return STATUS_UNAVAILABLE
    if threshold is None:
        return STATUS_INSUFFICIENT_DATA
    if threshold == 0:
        if operator == OPERATOR_MIN:
            return STATUS_PASS if value >= 0 else STATUS_BREACH
        return STATUS_BREACH if value > 0 else STATUS_PASS
    if operator == OPERATOR_MIN:
        if value < threshold:
            return STATUS_BREACH
        if warning_fraction and value <= threshold / warning_fraction:
            return STATUS_WARNING
        return STATUS_PASS
    if value > threshold:
        return STATUS_BREACH
    if warning_fraction and value >= threshold * warning_fraction:
        return STATUS_WARNING
    return STATUS_PASS


def _utilization(value: Optional[float], threshold: Optional[float],
                 operator: str) -> Optional[float]:
    if value is None or threshold in (None, 0):
        return None
    if operator == OPERATOR_MIN:
        return round(threshold / value * 100.0, 2) if value else None
    return round(value / threshold * 100.0, 2)


def _headroom(value: Optional[float], threshold: Optional[float],
              operator: str) -> Optional[float]:
    if value is None or threshold is None:
        return None
    return round((value - threshold) if operator == OPERATOR_MIN
                 else (threshold - value), 4)


def _breach_amount(value: Optional[float], threshold: Optional[float],
                   operator: str, status: str) -> Optional[float]:
    if status != STATUS_BREACH or value is None or threshold is None:
        return None
    return round((threshold - value) if operator == OPERATOR_MIN
                 else (value - threshold), 4)


def _evaluate_one(df: Optional[pd.DataFrame], lib: ConcentrationLibrary,
                  test: ActiveTest, *, reporting_date: str,
                  external: Optional[ExternalIndexProvider]
                  ) -> "tuple[str, MetricComputation]":
    """(status, computation) for one active test on one frame."""
    metric = lib.get(test.metric_id)
    if metric is None:
        comp = MetricComputation(value=None, unit=test.unit,
                                 data_status=DATA_MISSING,
                                 notes=f"Metric '{test.metric_id}' is not in "
                                       "the governed library.")
        return STATUS_UNAVAILABLE, comp

    eff = _parse_date(test.effective_date)
    exp = _parse_date(test.expiry_date)
    as_of = _parse_date(reporting_date)
    if eff and as_of and eff > as_of:
        comp = MetricComputation(value=None, unit=metric.unit,
                                 data_status=DATA_OK,
                                 notes=f"Effective from {test.effective_date}.")
        return STATUS_PENDING_EFFECTIVE_DATE, comp
    if exp and as_of and exp < as_of:
        comp = MetricComputation(value=None, unit=metric.unit,
                                 data_status=DATA_OK,
                                 notes=f"Expired on {test.expiry_date}.")
        return STATUS_EXPIRED, comp

    if df is None or df.empty:
        comp = MetricComputation(value=None, unit=metric.unit,
                                 data_status=DATA_MISSING,
                                 notes="No funded data for this period.")
        return STATUS_INSUFFICIENT_DATA, comp

    params = dict(test.parameters or {})
    if metric.evaluator == "external_index_ratio":
        params["_reporting_date"] = reporting_date
    comp = evaluate_metric(df, lib, metric, params, external=external)
    if comp.data_status == DATA_EXTERNAL_UNCONFIGURED:
        return STATUS_UNAVAILABLE, comp
    if comp.data_status == DATA_MISSING:
        return STATUS_UNAVAILABLE, comp
    status = _status_for(comp.value, test.threshold, test.operator,
                         test.warning_fraction)
    return status, comp


def evaluate_active_tests(
    df: Optional[pd.DataFrame],
    prior_df: Optional[pd.DataFrame],
    config: ActiveConfiguration,
    lib: ConcentrationLibrary,
    *,
    reporting_date: str = "",
    prior_reporting_date: str = "",
    external: Optional[ExternalIndexProvider] = None,
) -> Dict[str, Any]:
    """Evaluate every test in an activated configuration version.

    Returns the full governed envelope: per-test results (current, prior,
    movement, status transition), a summary, and the configuration provenance.
    Keys are camelCase to match the existing ``/mi/risk-limits`` wire shape.
    """
    tests_out: List[Dict[str, Any]] = []
    evaluated_at = _now_iso()
    prior_available = prior_df is not None and not getattr(prior_df, "empty", True)

    for test in config.tests:
        status, comp = _evaluate_one(df, lib, test,
                                     reporting_date=reporting_date,
                                     external=external)
        prior_value = None
        prior_status = None
        if prior_available:
            prior_status, prior_comp = _evaluate_one(
                prior_df, lib, test, reporting_date=prior_reporting_date,
                external=external)
            prior_value = prior_comp.value

        absolute_change = (round(comp.value - prior_value, 4)
                           if comp.value is not None and prior_value is not None
                           else None)
        pp_change = absolute_change if comp.unit == "percent" else None
        relative_change = (round((comp.value - prior_value) / prior_value * 100.0, 2)
                           if comp.value is not None and prior_value not in (None, 0)
                           else None)
        deteriorated = (
            _SEVERITY_ORDER.get(prior_status) is not None
            and _SEVERITY_ORDER.get(status) is not None
            and _SEVERITY_ORDER[status] > _SEVERITY_ORDER[prior_status])
        transition = (f"{prior_status} -> {status}"
                      if prior_status and prior_status != status else None)

        metric = lib.get(test.metric_id)
        tests_out.append({
            "testId": test.test_id,
            "metricId": test.metric_id,
            "displayName": test.display_name or (metric.display_name if metric else test.metric_id),
            "category": test.category or (metric.category if metric else ""),
            "reportingDate": reporting_date or None,
            "currentValue": comp.value,
            "priorValue": prior_value,
            "priorReportingDate": prior_reporting_date or None,
            "priorAvailable": prior_available,
            "absoluteChange": absolute_change,
            "percentagePointChange": pp_change,
            "relativeChange": relative_change,
            "threshold": test.threshold,
            "operator": test.operator,
            "warningFraction": test.warning_fraction,
            "unit": comp.unit or test.unit,
            "utilization": _utilization(comp.value, test.threshold, test.operator),
            "headroom": _headroom(comp.value, test.threshold, test.operator),
            "status": status,
            "priorStatus": prior_status,
            "statusTransition": transition,
            "deteriorated": deteriorated,
            "breachAmount": _breach_amount(comp.value, test.threshold,
                                           test.operator, status),
            "dataStatus": comp.data_status,
            "missingFields": comp.missing_fields,
            "numeratorValue": comp.numerator_value,
            "denominatorValue": comp.denominator_value,
            "denominatorBasis": comp.denominator_basis,
            "loansInNumerator": comp.loans_in_numerator,
            "totalLoans": comp.total_loans,
            "severity": test.severity,
            "effectiveDate": test.effective_date or None,
            "expiryDate": test.expiry_date or None,
            "notes": comp.notes,
            "resolvedColumns": comp.resolved_columns,
            "parameters": test.parameters,
            "definition": {
                "numerator": metric.numerator if metric else "",
                "aggregation": metric.aggregation if metric else "",
                "description": metric.description if metric else "",
                "definitionNotes": test.definition_notes,
            },
            "provenance": {
                "sourceReference": test.evidence.source_reference,
                "sourceText": test.evidence.source_text,
                "locator": test.evidence.locator,
                "approvedBy": test.approval.operator,
                "approvedAt": test.approval.decided_at,
                "approvalComments": test.approval.comments,
                "proposalId": test.proposal_id,
            },
            "configurationVersion": config.version,
            "evaluatedAt": evaluated_at,
        })

    summary = summarise(tests_out, reporting_date=reporting_date,
                        prior_reporting_date=prior_reporting_date,
                        prior_available=prior_available)
    return {
        "configurationVersion": config.version,
        "configurationHash": config.content_hash,
        "libraryVersion": config.library_version,
        "activatedBy": config.activated_by,
        "activatedAt": config.activated_at,
        "reportingDate": reporting_date or None,
        "priorReportingDate": (prior_reporting_date or None) if prior_available else None,
        "priorAvailable": prior_available,
        "evaluatedAt": evaluated_at,
        "tests": tests_out,
        "summary": summary,
    }


def summarise(tests: List[Dict[str, Any]], *, reporting_date: str = "",
              prior_reporting_date: str = "",
              prior_available: bool = False) -> Dict[str, Any]:
    def count(status: str) -> int:
        return sum(1 for t in tests if t["status"] == status)

    breaches = count(STATUS_BREACH)
    warnings = count(STATUS_WARNING)
    passes = count(STATUS_PASS)
    unavailable = sum(1 for t in tests
                      if t["status"] in (STATUS_UNAVAILABLE,
                                         STATUS_INSUFFICIENT_DATA))
    measured = [t for t in tests
                if t["currentValue"] is not None and t["headroom"] is not None]
    closest = min(measured, key=lambda t: t["headroom"]) if measured else None
    if breaches:
        overall = STATUS_BREACH
    elif warnings:
        overall = STATUS_WARNING
    elif passes:
        overall = STATUS_PASS
    else:
        overall = STATUS_UNAVAILABLE
    return {
        "overallStatus": overall,
        "activeTests": len(tests),
        "breaches": breaches,
        "warnings": warnings,
        "passes": passes,
        "unavailable": unavailable,
        "pendingEffectiveDate": count(STATUS_PENDING_EFFECTIVE_DATE),
        "expired": count(STATUS_EXPIRED),
        "reportingDate": reporting_date or None,
        "priorReportingDate": (prior_reporting_date or None) if prior_available else None,
        "priorAvailable": prior_available,
        "deteriorations": sum(1 for t in tests if t.get("deteriorated")),
        "closestToLimit": ({"testId": closest["testId"],
                            "displayName": closest["displayName"],
                            "headroom": closest["headroom"],
                            "unit": closest["unit"]} if closest else None),
    }


# --------------------------------------------------------------------------- #
# Drill-through
# --------------------------------------------------------------------------- #
_DRILL_CONTEXT_ROLES = ("loan_id", "region", "balance_current",
                        "balance_original", "valuation_original",
                        "borrower_id", "borrower_age_youngest", "rate_type")


def drillthrough(df: Optional[pd.DataFrame], lib: ConcentrationLibrary,
                 test: ActiveTest, *, max_rows: int = 500,
                 external: Optional[ExternalIndexProvider] = None
                 ) -> Dict[str, Any]:
    """Contributing-loan population for one active test.

    Reuses the SAME evaluator and mask that produced the numerator, so the
    returned rows reconcile exactly: their count equals ``loansInNumerator``
    and their basis-balance sum equals ``numeratorValue``.
    """
    metric = lib.get(test.metric_id)
    if metric is None or df is None or df.empty:
        return {"available": False,
                "reason": "No data or unknown metric for drill-through.",
                "rows": [], "columns": []}
    comp = evaluate_metric(df, lib, metric, dict(test.parameters or {}),
                           external=external)
    if comp.mask is None:
        return {"available": False,
                "reason": comp.notes or "This test has no loan-level "
                                        "contributing population.",
                "rows": [], "columns": []}
    subset = df[comp.mask.reindex(df.index, fill_value=False)]
    columns: List[str] = []
    for role in _DRILL_CONTEXT_ROLES:
        col = resolve_role_column(df, lib, role)
        if col and col not in columns:
            columns.append(col)
    for col in comp.resolved_columns.values():
        if col not in columns:
            columns.append(col)
    present = [c for c in columns if c in subset.columns]
    rows = subset[present].head(max_rows)
    return {
        "available": True,
        "testId": test.test_id,
        "metricId": test.metric_id,
        "displayName": test.display_name,
        "columns": present,
        "rows": rows.where(rows.notna(), None).to_dict(orient="records"),
        "rowCount": int(len(subset)),
        "truncated": bool(len(subset) > max_rows),
        "loansInNumerator": comp.loans_in_numerator,
        "numeratorValue": comp.numerator_value,
        "denominatorValue": comp.denominator_value,
        "denominatorBasis": comp.denominator_basis,
        "reconciles": int(len(subset)) == comp.loans_in_numerator,
    }
