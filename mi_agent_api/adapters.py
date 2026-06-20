"""Adapt MI Agent (`run_mi_agent_query`) output into the React artifact schema.

The React UI consumes a typed artifact union (kpi | chart | table | validation).
This module performs a *lossless-on-data* transform:

  - chart_result (a Plotly figure) for bar/line/scatter/bubble is rebuilt as a
    Recharts-friendly chart artifact from the underlying result table, and the
    raw Plotly figure JSON is carried in metadata for fidelity.
  - heatmap/treemap (not yet renderable in React) fall back to a table artifact.
  - summary results become KPI artifacts; tabular results become table artifacts.
  - validation errors/warnings become a validation artifact.

No analytics are performed here — only shape translation.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

# Brand palette (mirrors mi_chart_factory.DEFAULT_THEME / charts_plotly.py).
_PALETTE = ["#919dd1", "#232d55", "#3d4a82", "#36c2a8", "#e0a93b", "#c46b8f"]

# React-renderable chart types (the renderer does not yet draw heatmap/treemap).
_RENDERABLE_CHARTS = {"bar", "line", "scatter", "bubble"}

_FORMAT_MAP = {
    "currency": "gbp",
    "percent": "pct",
    "decimal": "decimal",
    "integer": "number",
    "date": "date",
    "string": "text",
}


def _uid(prefix: str = "art") -> str:
    return f"{prefix}_{uuid.uuid4().hex[:8]}"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _value_format(fmt: Optional[str]) -> str:
    return _FORMAT_MAP.get(fmt or "", "number")


def _source(spec: Dict[str, Any], ctx: "AdapterContext", label: str) -> Dict[str, Any]:
    return {
        "engine": "mi_agent.workflow",
        "label": label,
        "state": spec.get("state"),
        "spec": spec,
        "asOf": ctx.as_of,
        "portfolio": ctx.portfolio_id,
    }


class AdapterContext:
    def __init__(self, portfolio_id: Optional[str], as_of: Optional[str]):
        self.portfolio_id = portfolio_id
        self.as_of = as_of


def _resolved_format(resolved: Dict[str, Any], canonical: str) -> Optional[str]:
    for meta in resolved.values():
        if meta.get("canonical_field") == canonical:
            return meta.get("format")
    return None


def _dimension_column(resolved: Dict[str, Any], columns: List[str]) -> Optional[str]:
    for meta in resolved.values():
        if meta.get("role") == "dimension" and meta.get("canonical_field") in columns:
            return meta["canonical_field"]
    # Fallback: first non-numeric-looking column.
    return columns[0] if columns else None


def _infer_col_format(col: str, resolved: Dict[str, Any]) -> str:
    if col.endswith("_pct") or "concentration" in col:
        return "pct"
    # strip common aggregation suffixes to match a resolved canonical field
    base = col
    for suffix in ("_sum", "_avg", "_weighted_avg", "_median", "_count", "_count_distinct"):
        if base.endswith(suffix):
            base = base[: -len(suffix)]
            break
    fmt = _resolved_format(resolved, base)
    return _value_format(fmt)


def _kpi_label(key: str, resolved: Dict[str, Any]) -> str:
    base = key
    for suffix in ("_sum", "_avg", "_weighted_avg", "_median", "_count", "_count_distinct"):
        if base.endswith(suffix):
            base = base[: -len(suffix)]
            break
    return base.replace("_", " ").title()


def _format_kpi_value(value: Any, fmt: str) -> str:
    if not isinstance(value, (int, float)):
        return str(value)
    if fmt == "gbp":
        v = float(value)
        if abs(v) >= 1e9:
            return f"£{v / 1e9:.2f}BN"
        if abs(v) >= 1e6:
            return f"£{v / 1e6:.1f}MM"
        if abs(v) >= 1e3:
            return f"£{v / 1e3:.0f}K"
        return f"£{v:,.0f}"
    if fmt == "pct":
        return f"{float(value):.1f}%"
    if fmt == "decimal":
        return f"{float(value):.2f}"
    return f"{value:,.0f}" if isinstance(value, (int, float)) else str(value)


def _kpi_artifact(qr: Dict[str, Any], spec: Dict[str, Any], ctx: AdapterContext) -> Dict[str, Any]:
    rows = qr.get("data") or []
    resolved = qr.get("resolved_fields", {})
    row = rows[0] if rows else {}
    kpis = []
    for key, value in row.items():
        fmt = _infer_col_format(key, resolved)
        kpis.append(
            {
                "id": _uid("kpi"),
                "label": _kpi_label(key, resolved),
                "value": _format_kpi_value(value, fmt),
            }
        )
    return {
        "id": _uid(),
        "type": "kpi",
        "title": "Summary",
        "description": "Aggregated MI summary.",
        "source": _source(spec, ctx, "MI Agent · summary"),
        "createdAt": _now(),
        "mock": False,
        "kpis": kpis,
    }


def _table_artifact(
    qr: Dict[str, Any],
    spec: Dict[str, Any],
    ctx: AdapterContext,
    title: str,
    cr: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    rows = qr.get("data") or []
    resolved = qr.get("resolved_fields", {})
    columns = []
    if rows:
        for col in rows[0].keys():
            fmt = _infer_col_format(col, resolved)
            columns.append(
                {
                    "key": col,
                    "label": col.replace("_", " ").title(),
                    "align": "left" if fmt == "text" else "right",
                    "format": fmt,
                }
            )
    source = _source(spec, ctx, "MI Agent · table")
    # When this table stands in for an un-rendered chart (heatmap/treemap),
    # preserve the native chart type and raw Plotly figure so nothing is lost.
    if cr and cr.get("chart_type") not in _RENDERABLE_CHARTS:
        source["nativeChartType"] = cr.get("chart_type")
        source["figure"] = cr.get("figure")
    return {
        "id": _uid(),
        "type": "table",
        "title": title,
        "description": f"{qr.get('row_count', len(rows))} rows.",
        "source": source,
        "createdAt": _now(),
        "mock": False,
        "columns": columns,
        "rows": rows,
    }


def _chart_artifact(
    qr: Dict[str, Any],
    cr: Dict[str, Any],
    spec: Dict[str, Any],
    ctx: AdapterContext,
) -> Optional[Dict[str, Any]]:
    chart_type = cr.get("chart_type")
    if chart_type not in _RENDERABLE_CHARTS:
        return None
    rows = qr.get("data") or []
    if not rows:
        return None
    resolved = qr.get("resolved_fields", {})
    columns = list(rows[0].keys())

    if chart_type in ("scatter", "bubble"):
        # Loan-level x / y / (size) — the renderer reads series[0]=x, [1]=y, [2]=size.
        x_key = spec.get("x") if spec.get("x") in columns else (columns[0] if columns else None)
        y_key = spec.get("y") if spec.get("y") in columns else None
        size_key = spec.get("size") if spec.get("size") in columns else None
        if not x_key or not y_key:
            return None
        series = [
            {"key": x_key, "label": x_key.replace("_", " ").title(), "color": _PALETTE[0]},
            {"key": y_key, "label": y_key.replace("_", " ").title(), "color": _PALETTE[1]},
        ]
        if chart_type == "bubble" and size_key:
            series.append({"key": size_key, "label": size_key.replace("_", " ").title(), "color": _PALETTE[2]})
        value_format = _infer_col_format(y_key, resolved)
    else:
        # Grouped categorical (bar) / ordered (line): one dimension + one value.
        x_key = _dimension_column(resolved, columns)
        value_cols = [c for c in columns if c != x_key]
        value_cols.sort(key=lambda c: (c.endswith("_pct") or "concentration" in c))
        primary = value_cols[0] if value_cols else None
        if primary is None:
            return None
        series = [{"key": primary, "label": primary.replace("_", " ").title(), "color": _PALETTE[0]}]
        value_format = _infer_col_format(primary, resolved)

    return {
        "id": _uid(),
        "type": "chart",
        "title": cr.get("title") or "Chart",
        "description": cr.get("subtitle"),
        "source": {
            **_source(spec, ctx, f"MI Agent · {chart_type}"),
            # Backend-native chart type, kept distinct from the render type.
            "nativeChartType": chart_type,
            # Carry the raw Plotly figure for fidelity / future Plotly rendering.
            "figure": cr.get("figure"),
        },
        "createdAt": _now(),
        "mock": False,
        "chartType": chart_type,
        "xKey": x_key,
        "series": series,
        "rows": rows,
        "valueFormat": value_format,
        "warnings": cr.get("warnings", []),
    }


def _validation_artifact(validation: Dict[str, Any], spec: Dict[str, Any], ctx: AdapterContext) -> Optional[Dict[str, Any]]:
    errors = validation.get("errors", [])
    warnings = validation.get("warnings", [])
    if not errors and not warnings:
        return None
    issues = []
    for i, msg in enumerate(errors):
        issues.append(
            {"id": f"e{i}", "code": "MI.SPEC.ERROR", "title": msg, "severity": "blocker", "scope": "MI Agent · validation", "detail": msg}
        )
    for i, msg in enumerate(warnings):
        issues.append(
            {"id": f"w{i}", "code": "MI.SPEC.WARNING", "title": msg, "severity": "warning", "scope": "MI Agent · validation", "detail": msg}
        )
    return {
        "id": _uid(),
        "type": "validation",
        "title": "Query Validation",
        "description": "Validation of the interpreted MIQuerySpec.",
        "source": _source(spec, ctx, "MI Agent · validation"),
        "createdAt": _now(),
        "mock": False,
        "summary": {
            "blockers": len(errors),
            "warnings": len(warnings),
            "passed": 1 if validation.get("ok") else 0,
            "coverage": 100.0 if validation.get("ok") else 0.0,
        },
        "issues": issues,
    }


def _interpreted_string(interpreted: Any) -> str:
    if isinstance(interpreted, str):
        return interpreted
    if isinstance(interpreted, dict):
        parts = [f"{k}: {v}" for k, v in interpreted.items() if v not in (None, "", "none")]
        return " · ".join(parts)
    return ""


def _answer(interpreted: Any, qr: Optional[Dict[str, Any]], chart_type: Optional[str]) -> str:
    desc = _interpreted_string(interpreted)
    n = qr.get("row_count") if qr else None
    if desc and n is not None:
        return f"{desc} — {n} group(s)."
    return desc or "Query executed."


def adapt_workflow_result(
    workflow: Dict[str, Any],
    *,
    portfolio_id: Optional[str] = None,
    as_of: Optional[str] = None,
) -> Dict[str, Any]:
    """Map a `run_mi_agent_query` dict to the API/React response envelope."""
    ctx = AdapterContext(portfolio_id, as_of)
    spec = workflow.get("spec") or {}
    validation = workflow.get("validation") or {}

    # MIQueryResult / MIChartResult may be objects (in-process) or dicts.
    qr_obj = workflow.get("query_result")
    qr = qr_obj.to_dict() if hasattr(qr_obj, "to_dict") else qr_obj
    cr_obj = workflow.get("chart_result")
    if cr_obj is None:
        cr = None
    elif hasattr(cr_obj, "to_json"):
        import json

        cr = json.loads(cr_obj.to_json())
    else:
        cr = cr_obj

    artifacts: List[Dict[str, Any]] = []
    chart_type = cr.get("chart_type") if cr else None

    if qr:
        result_type = qr.get("result_type")
        if result_type == "summary":
            artifacts.append(_kpi_artifact(qr, spec, ctx))
        else:
            if cr:
                chart = _chart_artifact(qr, cr, spec, ctx)
                if chart:
                    artifacts.append(chart)
            artifacts.append(_table_artifact(qr, spec, ctx, cr.get("title") if cr else "Result", cr=cr))

    val_artifact = _validation_artifact(validation, spec, ctx)
    if val_artifact:
        artifacts.append(val_artifact)

    warnings = list(workflow.get("warnings", []))
    if chart_type in {"heatmap", "treemap"}:
        warnings.append(f"{chart_type} is not yet rendered in the React UI; showing the result table.")

    return {
        "ok": bool(workflow.get("ok")),
        "error": workflow.get("error"),
        "question": workflow.get("question"),
        "answer": _answer(workflow.get("interpreted"), qr, chart_type),
        "interpreted": _interpreted_string(workflow.get("interpreted")),
        "spec": spec,
        "validation": validation,
        "artifacts": artifacts,
        "warnings": warnings,
        # The MI Agent does not emit narrative assumptions; kept for schema parity.
        "assumptions": [],
        "metadata": {
            "portfolioId": portfolio_id,
            "asOfDate": as_of,
            "engine": "mi_agent",
            "source": "python",
            "mock": False,
            "parserMode": workflow.get("parser_mode"),
            "resultType": qr.get("result_type") if qr else None,
            "rowCount": qr.get("row_count") if qr else None,
            "chartType": chart_type,
        },
    }
