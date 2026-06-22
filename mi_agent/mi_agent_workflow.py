#!/usr/bin/env python3
"""
mi_agent_workflow.py

Non-UI orchestration for the Streamlit MI Agent: question -> spec -> validate ->
execute -> chart. Kept separate from ``streamlit_mi_agent.py`` so it can be unit
tested without a Streamlit runtime.

Control flow (the deterministic stack is the control layer; the LLM only
proposes an MIQuerySpec — it never executes anything or sees raw data):

    parse_with_repair  ->  validate_mi_query  ->  execute_mi_query  ->  create_mi_chart
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from .llm_query_parser import parse_with_repair
from .mi_dataset_profile import profile_dataset, validate_query_data
from .mi_chart_factory import MIChartError, MIChartResult, create_mi_chart
from .mi_query_executor import (
    MIQueryExecutionError,
    MIQueryResult,
    execute_mi_query,
)
from .mi_query_spec import MIQuerySpec
from .mi_query_validator import load_mi_semantics, validate_mi_query

# Chart types the chart factory can render (others are table/summary only).
_RENDERABLE = {"bar", "line", "scatter", "bubble", "heatmap", "treemap"}


def _dedupe(items: List[str]) -> List[str]:
    """De-duplicate a list of strings, preserving first-seen order."""
    seen: set = set()
    out: List[str] = []
    for item in items:
        if item not in seen:
            seen.add(item)
            out.append(item)
    return out


EXAMPLE_QUESTIONS = [
    "Show balance by region",
    "Show weighted average LTV by product type",
    "Show LTV by age bucket and region as a heatmap",
    "Show balance by region and broker as a treemap",
    "Show LTV by borrower age sized by balance",
    "Show top 10 brokers by balance",
    "Show redemptions over time",
]


# --------------------------------------------------------------------------- #
# Spec description (for the UI "Interpreted as:" panel)
# --------------------------------------------------------------------------- #

_AGG_LABEL = {
    "sum": "Sum", "avg": "Average", "weighted_avg": "Weighted average",
    "median": "Median", "count": "Count", "count_distinct": "Distinct count",
    "distribution": "Distribution", "loan_level": "Loan-level",
    "balance_sum": "Balance sum",
}


def _bn(semantics: dict, key: Optional[str]) -> Optional[str]:
    if not key:
        return None
    entry = semantics.get("fields", {}).get(key)
    if entry:
        return entry.get("business_name") or entry.get("display_name") or key
    return key


def describe_spec(spec: MIQuerySpec, semantics: dict,
                  parser_mode: str = "deterministic") -> Dict[str, Any]:
    """Human-readable summary of a spec for display."""
    out: Dict[str, Any] = {}
    out["Chart"] = (spec.chart_type.title() if spec.chart_type not in (None, "none")
                    else f"{spec.intent.title()} (no chart)")
    for slot, label in (("metric", "Metric"), ("dimension", "Dimension"),
                        ("x", "X"), ("y", "Y"), ("size", "Size"), ("color", "Colour")):
        val = getattr(spec, slot)
        if val:
            out[label] = _bn(semantics, val)
    if spec.dimensions:
        out["Dimensions"] = ", ".join(_bn(semantics, d) for d in spec.dimensions)
    if spec.hierarchy:
        out["Hierarchy"] = ", ".join(_bn(semantics, d) for d in spec.hierarchy)
    out["Aggregation"] = _AGG_LABEL.get(spec.aggregation, spec.aggregation)
    if spec.top_n is not None:
        out["Top N"] = spec.top_n
    if spec.filters:
        out["Filters"] = {_bn(semantics, k): v for k, v in spec.filters.items()}
    out["Parser"] = parser_mode
    return out


# --------------------------------------------------------------------------- #
# Main workflow
# --------------------------------------------------------------------------- #


def run_mi_agent_query(
    question: str,
    data,
    semantics,
    *,
    llm_enabled: bool = False,
    model: Optional[str] = None,
    parser_mode: str = "deterministic",
    max_repair_attempts: int = 1,
    catalog_mode: str = "core",
    zero_cost_first: bool = True,
    provider: str = "anthropic",
    llm_callable=None,
) -> Dict[str, Any]:
    """Run one MI question end to end.

    Parameters
    ----------
    question : natural-language MI question
    data     : pandas DataFrame or path to a canonical CSV
    semantics: dict or path to mi_semantics_field_registry.yaml
    llm_enabled : enable the LLM parser path (else deterministic)
    parser_mode : "deterministic" | "llm" (LLM only used when also llm_enabled)
    llm_callable: optional injected callable(prompt)->str for testing

    Returns a dict with: ok, error, parser_mode, spec, spec_obj, interpreted,
    validation, parse_metadata, query_result, chart_result, warnings, metadata.
    """
    if isinstance(semantics, (str, Path)):
        semantics = load_mi_semantics(semantics)

    result: Dict[str, Any] = {
        "ok": False,
        "error": None,
        "question": question,
        "parser_mode": "deterministic",
        "spec": None,
        "spec_obj": None,
        "interpreted": None,
        "validation": None,
        "parse_metadata": None,
        "query_result": None,
        "chart_result": None,
        "warnings": [],
        "metadata": {},
    }
    warnings: List[str] = []

    # ---- read data --------------------------------------------------------
    if isinstance(data, pd.DataFrame):
        df = data
    elif isinstance(data, (str, Path)):
        try:
            df = pd.read_csv(data)
        except Exception as exc:
            result["error"] = f"Could not read data CSV: {exc}"
            return result
    else:
        result["error"] = "data must be a pandas DataFrame or a path to a CSV"
        return result

    available_columns = set(df.columns)

    # ---- parse (deterministic or LLM, with repair) ------------------------
    effective_llm = bool(llm_enabled) and parser_mode == "llm"
    try:
        spec, parse_meta = parse_with_repair(
            question, semantics, available_columns=available_columns,
            llm_enabled=effective_llm, model=model,
            max_attempts=max_repair_attempts, llm_callable=llm_callable,
            provider=provider, catalog_mode=catalog_mode,
            zero_cost_first=zero_cost_first,
        )
    except Exception as exc:
        result["error"] = f"Parser error: {exc}"
        result["parse_metadata"] = {"parser_mode": parser_mode, "ok": False,
                                    "status": f"parser raised: {exc}"}
        return result

    result["parser_mode"] = parse_meta.get("parser_mode", "deterministic")
    # granular detail (deterministic_zero_cost / llm / llm_repaired / validation_failed)
    result["parser_mode_detail"] = parse_meta.get("parser_mode_detail",
                                                  result["parser_mode"])
    result["spec_obj"] = spec
    result["spec"] = spec.to_dict()
    result["parse_metadata"] = parse_meta
    result["interpreted"] = describe_spec(spec, semantics, result["parser_mode"])

    # ---- validate ---------------------------------------------------------
    vr = validate_mi_query(spec, semantics, available_columns=available_columns)
    result["validation"] = vr.to_dict()
    warnings.extend(vr.warnings)
    if not vr.ok:
        result["interpreted"]["Validation"] = "Failed"
        result["error"] = "The proposed query failed validation."
        result["warnings"] = _dedupe(warnings)
        result["metadata"] = {
            "parse_metadata": parse_meta,
            "parser_mode_detail": parse_meta.get("parser_mode_detail"),
            "repair_attempts": parse_meta.get("repair_attempts"),
            "repair_skipped_reason": parse_meta.get("repair_skipped_reason"),
            "llm": parse_meta.get("llm"),
        }
        return result

    # ---- data-aware validation (single dataset profile is the source of truth)
    # The name/role validator above never inspects values. Before claiming
    # "Validation: Passed", confirm the requested metric has numeric values, the
    # dimension(s) have non-blank values, filter fields carry values, and a
    # loan-level x/y/size has usable rows. A missing/empty LTV therefore fails
    # here with an exact reason instead of rendering an empty chart.
    profile = profile_dataset(df, semantics)
    result["display_hints"] = profile.get("display_hints", {})
    data_errors = validate_query_data(spec, df, semantics, profile)
    if data_errors:
        val = result.get("validation") or {"ok": True, "errors": [], "warnings": [],
                                            "resolved_fields": {}}
        val["ok"] = False
        val.setdefault("errors", []).extend(e["detail"] for e in data_errors)
        val["data_validation_errors"] = data_errors
        result["validation"] = val
        result["interpreted"]["Validation"] = "Failed"
        result["error"] = "The query cannot be answered from the prepared data: " + \
            "; ".join(f"{e['field']}: {e['reason']}" for e in data_errors)
        result["warnings"] = _dedupe(warnings)
        result["metadata"] = {
            "parse_metadata": parse_meta,
            "parser_mode_detail": parse_meta.get("parser_mode_detail"),
            "data_validation_errors": data_errors,
        }
        return result
    result["interpreted"]["Validation"] = "Passed"

    # ---- execute ----------------------------------------------------------
    try:
        qres: MIQueryResult = execute_mi_query(
            spec, df, semantics, validate=False,
        )
    except MIQueryExecutionError as exc:
        result["error"] = f"Execution failed: {exc}"
        # Surface as controlled validation output (never a raw 500). For a
        # duplicate-column defect, carry the structured fields the UI/API expose.
        val = result.get("validation") or {"ok": True, "errors": [], "warnings": [],
                                            "resolved_fields": {}}
        val["ok"] = False
        val.setdefault("errors", []).append(str(exc))
        dup = getattr(exc, "duplicate_columns", None)
        if dup is not None:
            val["duplicate_column_names"] = dup
            val["duplicate_query_fields_affected"] = getattr(exc, "affected_fields", [])
        result["validation"] = val
        if isinstance(result.get("interpreted"), dict):
            result["interpreted"]["Validation"] = "Failed"
        result["warnings"] = _dedupe(warnings)
        return result
    result["query_result"] = qres
    warnings.extend(qres.warnings)

    # A grouped / loan-level result with no rows after preparation is not a
    # "passed" query — surface it as a controlled validation failure with an
    # exact reason rather than rendering an empty chart.
    if qres.result_type in ("table", "loan_level") and qres.row_count == 0:
        reason = ("loan_level_no_usable_rows" if qres.result_type == "loan_level"
                  else "no_values_after_preparation")
        val = result.get("validation") or {"ok": True, "errors": [], "warnings": [],
                                            "resolved_fields": {}}
        val["ok"] = False
        val.setdefault("errors", []).append(
            f"{reason}: the query produced no usable rows after preparation")
        val["data_validation_errors"] = [{"field": spec.dimension or spec.x or "",
                                           "reason": reason,
                                           "detail": "no usable rows after preparation"}]
        result["validation"] = val
        result["interpreted"]["Validation"] = "Failed"
        result["error"] = f"The query produced no usable rows ({reason})."
        result["warnings"] = _dedupe(warnings)
        return result

    # ---- chart (only where a chart type is renderable) --------------------
    chart_result: Optional[MIChartResult] = None
    if spec.chart_type in _RENDERABLE:
        try:
            chart_result = create_mi_chart(qres, semantics)
            warnings.extend(chart_result.warnings)
        except MIChartError as exc:
            warnings.append(f"Chart not rendered: {exc}")
    else:
        warnings.append(
            f"No chart rendered (chart_type={spec.chart_type!r}); "
            f"showing the result table only."
        )
    result["chart_result"] = chart_result

    result["ok"] = True
    # The chart factory copies the executor's warnings onto its result, so the
    # same warning can arrive via both qres and chart_result — de-duplicate
    # while preserving order.
    result["warnings"] = _dedupe(warnings)
    result["metadata"] = {
        "parse_metadata": parse_meta,
        "parser_mode_detail": parse_meta.get("parser_mode_detail"),
        "repair_attempts": parse_meta.get("repair_attempts"),
        "repair_skipped_reason": parse_meta.get("repair_skipped_reason"),
        "llm": parse_meta.get("llm"),
        "executor_metadata": qres.metadata,
        "result_type": qres.result_type,
        "row_count": qres.row_count,
        "chart_type": spec.chart_type if chart_result else None,
    }
    return result


# --------------------------------------------------------------------------- #
# Export helpers (return bytes/strings; the UI wires them to download buttons)
# --------------------------------------------------------------------------- #


def result_csv_bytes(query_result: MIQueryResult) -> bytes:
    if query_result is None or query_result.data is None:
        return b""
    return query_result.data.to_csv(index=False).encode("utf-8")


def chart_html_str(chart_result: Optional[MIChartResult]) -> Optional[str]:
    if chart_result is None:
        return None
    return chart_result.to_html(include_plotlyjs="cdn")


def spec_json_str(spec) -> str:
    if isinstance(spec, MIQuerySpec):
        return spec.to_json()
    return json.dumps(spec, indent=2, default=str)


def metadata_json_str(workflow_result: Dict[str, Any]) -> str:
    payload = {
        "ok": workflow_result.get("ok"),
        "parser_mode": workflow_result.get("parser_mode"),
        "interpreted": workflow_result.get("interpreted"),
        "validation": workflow_result.get("validation"),
        "parse_metadata": workflow_result.get("parse_metadata"),
        "warnings": workflow_result.get("warnings"),
        "metadata": workflow_result.get("metadata"),
    }
    return json.dumps(payload, indent=2, default=str)
