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

import re
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

# Technical diagnostics that are useful for engineers but must NOT appear in the
# normal user-facing MI card. They are retained in API metadata/diagnostics and
# logged backend-side instead. Business-facing warnings (missing data, unavailable
# dimension, validation failure, partial result) are NEVER matched here.
_TECHNICAL_WARNING_PATTERNS = (
    re.compile(r"percent-scale heuristically detected", re.I),
    re.compile(r"does NOT rescale percentages", re.I),
    re.compile(r"^No chart rendered \(chart_type=", re.I),
    re.compile(r"^filter .+ kept \d+/\d+ rows", re.I),
    re.compile(r"^top_n=.* applied", re.I),
    re.compile(r"^top_n ignored", re.I),
    re.compile(r"concentration_pct not (added|computed)", re.I),
    re.compile(r"^excluded \d+ row\(s\) with missing", re.I),
    re.compile(r"^dropped \d+ loan-level row\(s\)", re.I),
    re.compile(r"^loan-level output capped", re.I),
    re.compile(r"bucket field .+ not present in data", re.I),
)


def _is_technical_warning(message: str) -> bool:
    return any(p.search(message) for p in _TECHNICAL_WARNING_PATTERNS)


def split_warnings(warnings: List[str]) -> Tuple[List[str], List[str]]:
    """Partition warnings into ``(business_facing, technical_diagnostics)``.

    Technical diagnostics (e.g. the percent-scale heuristic note) are hidden from
    the main user-visible card and surfaced only in API metadata/diagnostics;
    business-facing warnings remain prominent.
    """
    business: List[str] = []
    diagnostics: List[str] = []
    for w in warnings:
        (diagnostics if _is_technical_warning(str(w)) else business).append(w)
    return business, diagnostics

# Brand palette (mirrors mi_chart_factory.DEFAULT_THEME / charts_plotly.py).
_PALETTE = ["#919dd1", "#232d55", "#3d4a82", "#36c2a8", "#e0a93b", "#c46b8f"]

_FORMAT_MAP = {
    "currency": "gbp",
    "percent": "pct",
    "decimal": "decimal",
    "integer": "number",
    "date": "date",
    "string": "text",
}


def _figure_has_content(figure: Any) -> bool:
    """True when a Plotly figure carries at least one trace worth rendering."""
    return isinstance(figure, dict) and bool(figure.get("data"))


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


def _dimension_columns(resolved: Dict[str, Any], columns: List[str]) -> List[str]:
    """All result columns whose resolved role is a dimension, in column order."""
    dim_fields = {
        meta.get("canonical_field")
        for meta in resolved.values()
        if meta.get("role") == "dimension"
    }
    return [c for c in columns if c in dim_fields]


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


def _format_kpi_value(value: Any, fmt: str, scale: Optional[str] = None) -> str:
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
        # Apply the storage scale from the dataset contract: a fraction (0.51)
        # displays as 51.0%, points (51) display as 51.0%. Never guessed.
        v = float(value)
        if scale == "percent_fraction":
            v *= 100.0
        return f"{v:.1f}%"
    if fmt == "decimal":
        return f"{float(value):.2f}"
    return f"{value:,.0f}" if isinstance(value, (int, float)) else str(value)


def _hint(hints: Optional[Dict[str, Any]], col: str) -> Dict[str, Any]:
    """The {format, scale} display hint for an emitted column, from the single
    dataset contract (suffix-aware). Falls back to format inference with no scale."""
    if hints:
        from mi_agent.mi_dataset_profile import display_hint_for
        return display_hint_for({"display_hints": hints}, col)
    return {"format": None, "scale": None}


def _kpi_artifact(qr: Dict[str, Any], spec: Dict[str, Any], ctx: AdapterContext,
                  hints: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    rows = qr.get("data") or []
    resolved = qr.get("resolved_fields", {})
    row = rows[0] if rows else {}
    kpis = []
    for key, value in row.items():
        h = _hint(hints, key)
        fmt = h.get("format") or _infer_col_format(key, resolved)
        kpis.append(
            {
                "id": _uid("kpi"),
                "label": _kpi_label(key, resolved),
                "value": _format_kpi_value(value, fmt, h.get("scale")),
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
    hints: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    rows = qr.get("data") or []
    resolved = qr.get("resolved_fields", {})
    columns = []
    if rows:
        for col in rows[0].keys():
            h = _hint(hints, col)
            fmt = h.get("format") or _infer_col_format(col, resolved)
            columns.append(
                {
                    "key": col,
                    "label": col.replace("_", " ").title(),
                    "align": "left" if fmt == "text" else "right",
                    "format": fmt,
                    # Storage scale so React renders fraction percents as points.
                    "scale": h.get("scale"),
                }
            )
    source = _source(spec, ctx, "MI Agent · table")
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


def _label_for(col: str, resolved: Dict[str, Any]) -> str:
    """A human label for a column, preferring the resolved business name."""
    for meta in resolved.values():
        if meta.get("canonical_field") == col and meta.get("business_name"):
            return meta["business_name"]
    return col.replace("_", " ").title()


def _chart_artifact(
    qr: Dict[str, Any],
    cr: Dict[str, Any],
    spec: Dict[str, Any],
    ctx: AdapterContext,
    hints: Optional[Dict[str, Any]] = None,
) -> Optional[Dict[str, Any]]:
    chart_type = cr.get("chart_type")
    has_figure = _figure_has_content(cr.get("figure"))
    rows = qr.get("data") or []
    resolved = qr.get("resolved_fields", {})
    columns = list(rows[0].keys()) if rows else []

    x_key: Optional[str] = None
    y_key: Optional[str] = None
    size_key: Optional[str] = None
    value_key: Optional[str] = None
    x_label: Optional[str] = None
    y_label: Optional[str] = None
    size_label: Optional[str] = None
    series: List[Dict[str, Any]] = []
    value_format = "number"

    def _value_column(exclude: List[str]) -> Optional[str]:
        cand = [c for c in columns if c not in exclude]
        cand.sort(key=lambda c: (c.endswith("_pct") or "concentration" in c))
        return cand[0] if cand else None

    if chart_type in ("scatter", "bubble"):
        # Loan-level: emit EXPLICIT role keys (xKey/yKey/sizeKey) + labels so the
        # renderer never infers axes from series order, and yKey is never null.
        x_key = spec.get("x") if spec.get("x") in columns else (columns[0] if columns else None)
        y_key = spec.get("y") if spec.get("y") in columns else None
        size_key = (spec.get("size") if spec.get("size") in columns else None) \
            if chart_type == "bubble" else None
        if x_key and y_key:
            x_label = _label_for(x_key, resolved)
            y_label = _label_for(y_key, resolved)
            series = [
                {"key": x_key, "label": x_label, "color": _PALETTE[0]},
                {"key": y_key, "label": y_label, "color": _PALETTE[1]},
            ]
            if chart_type == "bubble" and size_key:
                size_label = _label_for(size_key, resolved)
                series.append({"key": size_key, "label": size_label, "color": _PALETTE[2]})
            value_format = _hint(hints, y_key).get("format") or _infer_col_format(y_key, resolved)
    elif chart_type in ("bar", "line"):
        # Grouped categorical (bar) / ordered (line): one dimension + one value.
        x_key = _dimension_column(resolved, columns)
        primary = _value_column([x_key] if x_key else [])
        if primary is not None:
            series = [{"key": primary, "label": _label_for(primary, resolved), "color": _PALETTE[0]}]
            value_format = _hint(hints, primary).get("format") or _infer_col_format(primary, resolved)
    elif chart_type == "heatmap":
        # Native grid renderer needs two dimensions + an intensity measure.
        dims = _dimension_columns(resolved, columns)
        x_key = dims[0] if len(dims) > 0 else None
        y_key = dims[1] if len(dims) > 1 else None
        value_key = _value_column([d for d in (x_key, y_key) if d])
        value_format = _hint(hints, value_key or "").get("format") or _infer_col_format(value_key or "", resolved)
    elif chart_type == "treemap":
        # Native Recharts treemap needs a label dimension + a size measure.
        dims = _dimension_columns(resolved, columns)
        x_key = dims[0] if dims else (columns[0] if columns else None)
        value_key = _value_column([x_key] if x_key else [])
        value_format = _hint(hints, value_key or "").get("format") or _infer_col_format(value_key or "", resolved)
    else:
        x_key = columns[0] if columns else None

    # The Plotly figure is a *fallback* only for fidelity-sensitive types that
    # have no Recharts equivalent. Standard charts render natively, so we drop
    # their (redundant, heavy) figure payload.
    keep_figure = chart_type in ("heatmap", "treemap") or chart_type not in (
        "bar", "line", "scatter", "bubble", "heatmap", "treemap"
    )
    figure_out = cr.get("figure") if (keep_figure and has_figure) else None

    native_ok = bool(series) or (
        chart_type == "heatmap" and x_key and y_key and value_key and rows
    ) or (chart_type == "treemap" and x_key and value_key and rows)

    # Nothing to draw natively and no fallback figure → no chart artifact.
    if not native_ok and not figure_out:
        return None

    source = {
        **_source(spec, ctx, f"MI Agent · {chart_type}"),
        # Backend-native chart type, kept distinct from the render type.
        "nativeChartType": chart_type,
    }
    if figure_out is not None:
        source["figure"] = figure_out

    # Per-column display hints (format + storage scale) so React formats values —
    # especially fraction percents (0.51 -> 51.0%) — without guessing.
    display_hints = {c: _hint(hints, c) for c in columns} if hints else {}

    chart_warnings, chart_diagnostics = split_warnings(list(cr.get("warnings", [])))
    if chart_diagnostics:
        source["diagnostics"] = chart_diagnostics

    return {
        "id": _uid(),
        "type": "chart",
        "title": cr.get("title") or "Chart",
        "description": cr.get("subtitle"),
        "source": source,
        "createdAt": _now(),
        "mock": False,
        "chartType": chart_type,
        "xKey": x_key,
        "yKey": y_key,
        "sizeKey": size_key,
        "valueKey": value_key,
        "xLabel": x_label,
        "yLabel": y_label,
        "sizeLabel": size_label,
        "series": series,
        "rows": rows,
        "valueFormat": value_format,
        "displayHints": display_hints,
        # User-visible card carries business-facing warnings only; technical
        # diagnostics live in ``source.diagnostics``.
        "warnings": chart_warnings,
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
    # Single dataset contract: per-field {format, scale} from the workflow profile.
    hints = workflow.get("display_hints") or {}

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
    chart_emitted = False

    if qr:
        result_type = qr.get("result_type")
        if result_type == "summary":
            artifacts.append(_kpi_artifact(qr, spec, ctx, hints))
        else:
            if cr:
                chart = _chart_artifact(qr, cr, spec, ctx, hints)
                if chart:
                    artifacts.append(chart)
                    chart_emitted = True
            artifacts.append(_table_artifact(qr, spec, ctx,
                                             cr.get("title") if cr else "Result", hints))

    val_artifact = _validation_artifact(validation, spec, ctx)
    if val_artifact:
        artifacts.append(val_artifact)

    raw_warnings = list(workflow.get("warnings", []))
    # Only warn about degraded fidelity when we could not emit a chart at all
    # (no Plotly figure and no normalisable series). With a figure present,
    # heatmap/treemap render faithfully via the Plotly renderer.
    if chart_type in {"heatmap", "treemap"} and not chart_emitted:
        raw_warnings.append(f"{chart_type} could not be rendered; showing the result table.")

    # Hide technical diagnostics (e.g. the percent-scale heuristic note) from the
    # main user-facing output, but retain them in metadata.diagnostics.
    warnings, diagnostics = split_warnings(raw_warnings)

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
        "diagnostics": diagnostics,
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
            # Technical diagnostics retained for engineers / an expandable UI panel.
            "diagnostics": diagnostics,
        },
    }
