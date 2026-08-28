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
from typing import Any, Dict, List, Mapping, Optional, Tuple

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


def _is_bucket_column(col: Optional[str]) -> bool:
    """A bucketed dimension column (used as the heatmap COLUMN axis)."""
    return bool(col) and (col.endswith("_bucket") or col.endswith("_band")
                          or "bucket" in col)


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
    """The headline label for one executed measure — INCLUDING what was computed.

    The aggregation suffix was stripped and thrown away, so the KPI for
    `youngest_borrower_age_avg` read "Youngest Borrower Age: 74" beside a
    provenance line that correctly said "Average Borrower Age" and a rawValue
    of 74.33 — the portfolio AVERAGE, headlined as though it were the youngest
    borrower in the book. The number and the calculation are untouched; what
    changes is that the label now describes the calculation that produced it.

    The aggregation words come from `execution_receipt._MEASURE_AGG_WORDS`, the
    vocabulary the receipt already renders that provenance line from, so the
    headline and the receipt cannot say two different things about one figure.
    """
    # LONGEST SUFFIX FIRST. `_avg` matched before `_weighted_avg`, so
    # `current_loan_to_value_weighted_avg` lost only "_avg" and read
    # "Average Current Loan To Value Weighted".
    base, agg = key, ""
    for suffix in sorted(("_sum", "_avg", "_weighted_avg", "_median",
                          "_count", "_count_distinct"), key=len, reverse=True):
        if base.endswith(suffix):
            base, agg = base[: -len(suffix)], suffix[1:]
            break
    try:
        from mi_agent.execution_receipt import _MEASURE_AGG_WORDS
        prefix = _MEASURE_AGG_WORDS.get(agg, "")
    except Exception:  # noqa: BLE001 - no owner, the label is as it was
        prefix = ""
    # The FIELD's governed business name where the resolver knows one, so the
    # headline reads as the registry spells it rather than as the column is
    # keyed. Same helper the chart series labels already use.
    return f"{prefix}{_label_for(base, resolved)}".strip()


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
        from mi_agent.mi_dataset_profile import to_display_points
        return f"{to_display_points(float(value), scale):.1f}%"
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
        kpi = {
            "id": _uid("kpi"),
            "label": _kpi_label(key, resolved),
            "value": _format_kpi_value(value, fmt, h.get("scale")),
        }
        # ADDITIVE machine-readable evidence alongside the display string.
        #
        # ``value`` is deliberately formatted for a human ("£18.9MM"), which
        # means the headline figure of a governed answer could not be verified
        # by a machine without re-parsing display prose — and a rounded display
        # string cannot be reconciled to the underlying book at all. The raw
        # number, its canonical field and its unit are therefore carried beside
        # it. Nothing existing was renamed or removed, so every current consumer
        # is unaffected; a machine caller (Copilot, an assurance harness, a
        # future agent) now has the figure itself.
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            kpi["rawValue"] = float(value)
            kpi["valueFormat"] = fmt
            kpi["field"] = key
            canonical = (resolved.get(key) or {}).get("canonical_field")
            if canonical:
                kpi["canonicalField"] = canonical
        kpis.append(kpi)
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
    other_categories: Dict[str, List[str]] = {}
    promoted_to_heatmap = False

    def _value_column(exclude: List[str]) -> Optional[str]:
        cand = [c for c in columns if c not in exclude]
        # Supporting columns (share %, the loan-count denominator, the avg total)
        # are never the charted measure — push them to the end.
        cand.sort(key=lambda c: (c.endswith("_pct") or "concentration" in c
                                 or c == "loan_count" or c.endswith("_total")))
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
    elif chart_type in ("bar", "line") and len(_dimension_columns(resolved, columns)) >= 2:
        # Safe chart selection (never drop a dimension to fit a chart): a bar/line
        # carries a single categorical axis + one series, so a result grouped by
        # TWO categorical dimensions would silently lose the second on the chart.
        # Two categorical dimensions + one numeric metric → promote to a heatmap/
        # matrix so both dimensions survive. The full detail also stays in the
        # accompanying table artifact.
        dims = _dimension_columns(resolved, columns)
        a, b = dims[0], dims[1]
        if _is_bucket_column(b) and not _is_bucket_column(a):
            x_key, y_key = b, a
        else:
            x_key, y_key = a, b
        value_key = _value_column([d for d in (x_key, y_key) if d])
        value_format = _hint(hints, value_key or "").get("format") or _infer_col_format(value_key or "", resolved)
        chart_type = "heatmap"  # render as a matrix; both dimensions preserved
        promoted_to_heatmap = True
    elif chart_type in ("bar", "line"):
        # Grouped categorical (bar) / ordered (line): one dimension + one value.
        x_key = _dimension_column(resolved, columns)
        primary = _value_column([x_key] if x_key else [])
        if primary is not None:
            series = [{"key": primary, "label": _label_for(primary, resolved), "color": _PALETTE[0]}]
            value_format = _hint(hints, primary).get("format") or _infer_col_format(primary, resolved)
            value_key = primary
            # High-cardinality bar: cap to Top 10 (+ Other) for the visual; the
            # FULL detail stays in the table/export artifact alongside.
            if chart_type == "bar" and x_key:
                rows, capped = _cap_bar_rows(rows, x_key, primary, n=10)
                if capped:
                    # The "Other" bucket carries the SHOWN top-N category values so a
                    # drill on "Other" can be executed as <dim> NOT IN [those values]
                    # (recovering the underlying rows instead of matching a label).
                    other_categories[x_key] = [
                        str(r[x_key]) for r in rows
                        if str(r.get(x_key)) != "Other"]
    elif chart_type == "heatmap":
        # Native grid renderer needs two dimensions + an intensity measure.
        # Row/column convention: the bucket dimension is the COLUMN axis (xKey)
        # and the other (e.g. geography) is the ROW axis (yKey), so a matrix reads
        # geography down the side and the bucket across the top.
        dims = _dimension_columns(resolved, columns)
        a = dims[0] if len(dims) > 0 else None
        b = dims[1] if len(dims) > 1 else None
        if a and b and _is_bucket_column(b) and not _is_bucket_column(a):
            x_key, y_key = b, a
        else:
            x_key, y_key = a, b
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
    # A bar/line promoted to a heatmap has only the ORIGINAL (single-dimension)
    # bar figure attached — keeping it would re-introduce the dropped dimension.
    # Render the promoted matrix natively and discard the misleading figure.
    if promoted_to_heatmap:
        keep_figure = False
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

    # Heatmap/matrix row/column aliases (explicit, alongside x/y/value keys) so a
    # consumer can read the matrix as rows × columns without re-deriving them.
    matrix_keys: Dict[str, Any] = {}
    if chart_type == "heatmap" and x_key and y_key and value_key:
        matrix_keys = {"rowKey": y_key, "columnKey": x_key, "metricKey": value_key}

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
        **matrix_keys,
        "xLabel": x_label,
        "yLabel": y_label,
        "sizeLabel": size_label,
        "series": series,
        "rows": rows,
        "valueFormat": value_format,
        "displayHints": display_hints,
        # Drill metadata for the capped "Other" bucket (NOT IN the shown categories).
        "otherCategories": other_categories or None,
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


def _answer(interpreted: Any, qr: Optional[Dict[str, Any]], chart_type: Optional[str],
            hints: Optional[Dict[str, Any]] = None,
            spec: Optional[Mapping[str, Any]] = None) -> str:
    """A short, plain-English lead for the chat.

    Deliberately does NOT echo the parser interpretation (metric / dimension /
    aggregation / parser / validation) — that technical detail stays in the
    ``interpreted`` field, surfaced only behind the UI's collapsed Query Logic
    disclosure. The React response-presenter produces the grounded, data-aware
    sentence; this is a safe neutral fallback for any other API consumer.

    THE SINGLE-FIGURE CASE SAYS THE FIGURE. "What is the total balance for Wales
    loans?" answered *"Here is the result for your query, covering 1 group(s)"*
    and then a receipt that names the population and the loan count but not the
    money — so the headline number of the answer appeared nowhere in the prose,
    only inside the KPI artifact a text-first channel does not render. "1
    group(s)" is also not a sentence a lender writes.

    The figures come from the SAME row and the SAME formatter the KPI artifact
    uses, so the prose and the artifact cannot disagree; this adds a rendering,
    not a second calculation.
    """
    rows = (qr.get("data") or []) if qr else []
    n = qr.get("row_count") if qr else None
    if len(rows) == 1 and chart_type in (None, "none"):
        line = _scalar_line(rows[0], (qr or {}).get("resolved_fields") or {}, hints,
                            spec)
        if line:
            return line
    ranked = _ranked_lead(rows, (qr or {}).get("resolved_fields") or {}, hints, spec)
    if ranked:
        return ranked
    # A SINGLE ROW IS A SINGLE FIGURE, whatever chart was chosen for it.
    #
    # "What is the balance of offer stage cases?" narrows to one governed stage
    # and is rendered as a bar, so the scalar branch above — gated on
    # chart_type — was skipped and the reader got "Here is the bar for your
    # query, covering 1 group." The money was in the KPI artifact and nowhere
    # in the prose, which is the same defect that branch was added to fix, one
    # chart type further along.
    #
    # Purely additive and rendering-only: it is reached only where the next
    # line would have said "covering 1 group", it comes AFTER the ranked lead
    # so no ranked answer changes, and it uses the same row, labels and
    # formatters as the KPI artifact — a rendering, not a second calculation.
    if len(rows) == 1:
        line = _scalar_line(rows[0], (qr or {}).get("resolved_fields") or {},
                            hints, spec)
        if line:
            return line
    noun = "result" if chart_type in (None, "none") else chart_type
    if n is not None:
        groups = "1 group" if n == 1 else f"{n:,} groups"
        return f"Here is the {noun} for your query, covering {groups}."
    return "Here is the result for your query."


def _ranked_lead(rows, resolved: Mapping[str, Any],
                 hints: Optional[Dict[str, Any]],
                 spec: Optional[Mapping[str, Any]]) -> str:
    """"Scotland has the highest Total Balance: £28.9MM (7 groups)."

    A RANKING QUESTION IS ANSWERED BY NAMING THE GROUP. "Which region has the
    largest balance?" led with *"Here is the bar for your query, covering 7
    groups"* — true, and not the answer to the question asked. The reader has to
    read the chart to find out which region, and the same sentence appeared
    whether they asked for the largest or the smallest.

    Reads the FIRST row of the executed result and nothing else: the executor
    has already ordered it in the direction the spec asked for, so this states
    the ordering rather than deciding it. No new calculation, and the value goes
    through the same formatter the KPI and table artifacts use.
    """
    if not rows or not isinstance(spec, Mapping):
        return ""
    if not (spec.get("sort_by") or spec.get("ranking_mode") == "grouped"):
        return ""                       # a plain breakdown is not a ranking
    dimension = spec.get("dimension")
    if not dimension:
        return ""
    row = rows[0]
    if dimension not in row:
        return ""
    label = str(row.get(dimension) or "").strip()
    if not label:
        return ""
    # The ranked measure as the row spells it: "<metric>_<agg>", else the first
    # numeric column that is not the loan count.
    metric = spec.get("sort_by") or spec.get("metric") or ""
    keys = [k for k in row
            if k != dimension and isinstance(row.get(k), (int, float))
            and not isinstance(row.get(k), bool)]
    ranked_key = next((k for k in keys if metric and str(k).startswith(str(metric))),
                      next((k for k in keys if not str(k).endswith(("_count", "_pct"))),
                           None))
    if ranked_key is None:
        return ""
    h = _hint(hints, ranked_key)
    shown = _format_kpi_value(row.get(ranked_key),
                              h.get("format") or _infer_col_format(ranked_key, resolved),
                              h.get("scale"))
    superlative = ("lowest" if str(spec.get("sort_direction") or "desc").lower() == "asc"
                   else "highest")
    measure = _kpi_label(ranked_key, resolved)
    groups = "1 group" if len(rows) == 1 else f"{len(rows):,} groups"
    return f"{label} has the {superlative} {measure}: {shown} ({groups})."


def _scalar_line(row: Mapping[str, Any], resolved: Mapping[str, Any],
                 hints: Optional[Dict[str, Any]],
                 spec: Optional[Mapping[str, Any]] = None) -> str:
    """"Total Balance: £24.3MM · 93 loans." — the single-figure answer, in words.

    Built from the row the KPI artifact is built from, through the same label
    and value formatters, so there is one rendering of a figure and not two.
    """
    # THE MEASURE THE QUESTION ASKED FOR COMES FIRST, and the SPEC says which
    # one that is. The row's own order is the executor's, so a balance question
    # led with a loan count and a count question led with a balance — each
    # answering with the other's figure. Everything else follows in row order,
    # so nothing is dropped and the reader still gets the coverage.
    wanted = ""
    _grouped = False
    if isinstance(spec, Mapping):
        _grouped = bool(spec.get("dimension") or spec.get("dimensions"))
        if str(spec.get("aggregation") or "").lower() == "count":
            wanted = "_count"
        elif spec.get("metric"):
            wanted = str(spec.get("metric"))

    def _rank(key: str) -> int:
        key = str(key)
        # A SHARE answers a share question with the share. The share row also
        # carries its numerator, its denominator and the population total, and
        # a lead sentence that opens with "Current Outstanding Balance
        # Numerator: 5,835,756" buries the 3.4% the reader asked for. All four
        # stay in the KPI artifact and on the receipt; the sentence leads with
        # the figure and keeps the coverage.
        if key.endswith("_share_pct") or key.endswith("_pct"):
            # ...but a share OF ONE GROUP is not the share anyone asked for.
            # A grouped result carries a per-group concentration column, so
            # once this branch also served grouped answers "what is the
            # balance of offer stage cases?" led with "Concentration Pct:
            # 100.0%" — the share of the only group, which is 100% by
            # construction and reads like a finding. The share leads where the
            # share IS the answer: an ungrouped question about the book.
            return 0 if not _grouped else 3
        if key.endswith(("_numerator", "_denominator")) or key == "population_total":
            return 4
        if wanted == "_count":
            return 1 if key.endswith("_count") else 2
        if wanted and key.startswith(wanted):
            return 1
        return 2 if not key.endswith("_count") else 3

    # THREE FIGURES IS A SENTENCE; five is a dump. Ranked first, so what is kept
    # is what the question asked for.
    items = sorted((row or {}).items(), key=lambda kv: _rank(kv[0]))[:3]
    parts = []
    for key, value in items:
        h = _hint(hints, key)
        fmt = h.get("format") or _infer_col_format(key, resolved)
        shown = _format_kpi_value(value, fmt, h.get("scale"))
        if shown in (None, ""):
            continue
        label = _kpi_label(key, resolved)
        # A count reads as a count: "93 loans", not "Loan: 93".
        if str(key).endswith("_count") or fmt == "count":
            noun = (label or "record").strip().lower()
            parts.append(f"{shown} {noun}" if noun.endswith("s")
                         else f"{shown} {noun}s")
        else:
            parts.append(f"{label}: {shown}")
    return " · ".join(parts) + "." if parts else ""


def _with_receipt(answer: str, workflow: Dict[str, Any]) -> str:
    """Append the execution receipt to a successful substantive answer.

    A refusal already states its own reason and carries no receipt, so it is
    returned untouched. The receipt is derived from execution (see
    ``mi_agent.execution_receipt``), never from the question, which is what makes
    it capable of exposing a scope error rather than repeating one.
    """
    receipt = workflow.get("execution_receipt") or {}
    if not workflow.get("ok") or not receipt:
        return answer
    line = receipt.get("receipt")
    if not line:
        return answer
    parts = [answer.rstrip(), line]
    not_applied = receipt.get("notApplied") or []
    if not_applied:
        parts.append("Not applied: " + "; ".join(not_applied) + ".")
    return "\n\n".join(p for p in parts if p)


def _cap_bar_rows(rows: List[Dict[str, Any]], x_key: str, value_key: str,
                  n: int = 10) -> Tuple[List[Dict[str, Any]], bool]:
    """Cap a bar chart to the top ``n`` categories by ``value_key``.

    For an additive measure (``*_sum`` / ``count`` / ``*_total``) the remainder is
    folded into an aggregated ``Other`` row so the chart still totals correctly;
    for a non-additive measure (avg / weighted avg / share) the tail is dropped
    from the VISUAL only (the full detail remains in the table/export). No-op when
    there are ``<= n`` categories.
    """
    if not value_key or len(rows) <= n:
        return rows, False

    def _val(r: Dict[str, Any]) -> float:
        v = r.get(value_key)
        return float(v) if isinstance(v, (int, float)) else float("-inf")

    ordered = sorted(rows, key=_val, reverse=True)
    additive = (value_key.endswith("_sum") or value_key == "count"
                or value_key.endswith("_total"))
    if additive:
        head, tail = ordered[: n - 1], ordered[n - 1:]
        other: Dict[str, Any] = {x_key: "Other"}
        for k in rows[0].keys():
            if k == x_key:
                continue
            vals = [r.get(k) for r in tail if isinstance(r.get(k), (int, float))]
            other[k] = sum(vals) if vals else None
        return head + [other], True
    # Non-additive measure: show the top n; full detail stays in the table.
    return ordered[:n], True


def _source_notes(qr: Optional[Dict[str, Any]], spec: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Provenance / authoritative-source notes for the fields a query used.

    Surfaces any ``source_note`` declared on a referenced field's semantic-registry
    entry (the governed hook for "this field is sourced from the pipeline/KFI file —
    confirm it is authoritative for funded-book MI"). Empty when none apply.
    """
    if not qr:
        return []
    resolved = qr.get("resolved_fields", {}) or {}
    notes: List[Dict[str, Any]] = []
    for key, meta in resolved.items():
        note = meta.get("source_note") if isinstance(meta, dict) else None
        if note:
            notes.append({"field": key, "note": str(note)})
    return notes


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

    # Reconciliation / coverage footer (every artifact) + provenance source notes.
    reconciliation = (workflow.get("reconciliation")
                      or (workflow.get("metadata") or {}).get("reconciliation"))
    source_notes = _source_notes(qr, spec)

    artifacts: List[Dict[str, Any]] = []
    chart_type = cr.get("chart_type") if cr else None
    chart_emitted = False

    # P0: a declined answer must not ship the result it declined. The workflow
    # keeps ``query_result`` for the audit trail, but rendering it as a table
    # titled "Result" — or as a KPI reading "£1.96BN" — alongside "I have not
    # substituted a broader figure" hands the reader the very number the guard
    # just refused to present. This covers EVERY refusal, not only the semantic
    # guard's: the dimension invariant refuses the same way and shipped the
    # whole-book KPI next to its refusal.
    # ``qr`` itself is left intact so the metadata below still reports what ran.
    refused = (not workflow.get("ok")
               or bool(workflow.get("semantic_refusal"))
               or bool(workflow.get("controlled_refusal")))

    if qr and not refused:
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

    # Attach the reconciliation footer + source notes to every artifact so any
    # surfaced artifact (chat or workspace) can show coverage / provenance.
    for art in artifacts:
        if reconciliation and art.get("type") in ("chart", "table", "kpi"):
            art["reconciliation"] = reconciliation
        if source_notes and art.get("type") in ("chart", "table", "kpi"):
            art["sourceNotes"] = source_notes

    val_artifact = _validation_artifact(validation, spec, ctx)
    if val_artifact:
        artifacts.append(val_artifact)

    raw_warnings = list(workflow.get("warnings", []))
    # Only warn about degraded fidelity when we could not emit a chart at all
    # (no Plotly figure and no normalisable series). With a figure present,
    # heatmap/treemap render faithfully via the Plotly renderer.
    if chart_type in {"heatmap", "treemap"} and not chart_emitted and not refused:
        raw_warnings.append(f"{chart_type} could not be rendered; showing the result table.")

    # Hide technical diagnostics (e.g. the percent-scale heuristic note) from the
    # main user-facing output, but retain them in metadata.diagnostics.
    warnings, diagnostics = split_warnings(raw_warnings)

    # End-to-end query trace (parser → executor → renderer). The workflow builds
    # it without the artifact axes (those are produced here), so back-fill the
    # chart axes from the emitted chart artifact for the parser/executor/renderer
    # attribution to be complete.
    query_trace = workflow.get("query_trace")
    if isinstance(query_trace, dict):
        chart_art = next((a for a in artifacts if a.get("type") == "chart"), None)
        table_art = next((a for a in artifacts if a.get("type") == "table"), None)
        kpi_art = next((a for a in artifacts if a.get("type") == "kpi"), None)
        # The emitted artifact type (chart type wins, then table, then kpi, else none).
        artifact_type = (chart_art.get("chartType") if chart_art
                         else "table" if table_art else "kpi" if kpi_art else "none")
        query_trace = {**query_trace, "artifact_type": artifact_type}
        if chart_art is not None:
            query_trace["chartAxes"] = {
                "chartType": chart_art.get("chartType"),
                "xKey": chart_art.get("xKey"),
                "yKey": chart_art.get("yKey"),
                "valueKey": chart_art.get("valueKey"),
                "seriesKeys": [s.get("key") for s in (chart_art.get("series") or [])],
            }

    # Parser observability (mode detail / repair attempts / LLM status) from the
    # workflow metadata. Defined defensively — a missing block must never raise a
    # NameError that turns a good answer into a 500.
    _wf_meta = workflow.get("metadata") or {}
    parser_observability = {
        "parserModeDetail": _wf_meta.get("parser_mode_detail"),
        "repairAttempts": _wf_meta.get("repair_attempts"),
        "repairSkippedReason": _wf_meta.get("repair_skipped_reason"),
        "llm": _wf_meta.get("llm"),
    }

    return {
        "ok": bool(workflow.get("ok")),
        "error": workflow.get("error"),
        "question": workflow.get("question"),
        # A controlled response (unmapped question / unsupported concept)
        # carries its own user-facing answer — never replace it with the
        # generic lead sentence.
        # A declined answer states WHY. Falling through to the generic lead
        # sentence gave "Here is the result for your query" on a refusal — a
        # success-shaped answer to a question that was not answered.
        "answer": _with_receipt(
            workflow.get("answer")
            or (workflow.get("error") if refused else None)
            or _answer(workflow.get("interpreted"), qr, chart_type, hints, spec),
            workflow),
        # P0: the machine-derived statement of what was ACTUALLY executed —
        # measure, aggregation, filters that really narrowed the frame,
        # groupings that really grouped, population, period. Structured for
        # callers that want to render it themselves; also folded into `answer`
        # above so a text-first channel (Copilot) cannot miss it.
        "executionSummary": workflow.get("execution_receipt"),
        "semanticGuard": workflow.get("semantic_guard"),
        "interpreted": _interpreted_string(workflow.get("interpreted")),
        "spec": spec,
        "validation": validation,
        "artifacts": artifacts,
        "reconciliation": reconciliation,
        "sourceNotes": source_notes,
        "warnings": warnings,
        "diagnostics": diagnostics,
        # The MI Agent does not emit narrative assumptions; kept for schema parity.
        "assumptions": [],
        # Parser → executor → renderer diagnostics + the fail-closed dimension
        # invariant, so it is immediately obvious at which layer a dimension was
        # applied, rejected, or (never, by construction) silently dropped.
        "queryTrace": query_trace,
        "dimensionInvariant": workflow.get("dimension_invariant"),
        "filterInvariant": workflow.get("filter_invariant"),
        "metadata": {
            "portfolioId": portfolio_id,
            "asOfDate": as_of,
            "engine": "mi_agent",
            "source": "python",
            "mock": False,
            "parserMode": workflow.get("parser_mode"),
            "parserModeDetail": parser_observability["parserModeDetail"],
            "repairAttempts": parser_observability["repairAttempts"],
            "repairSkippedReason": parser_observability["repairSkippedReason"],
            "llm": parser_observability["llm"],
            "resultType": qr.get("result_type") if qr else None,
            "rowCount": qr.get("row_count") if qr else None,
            "chartType": chart_type,
            "unmappedQuestion": bool(workflow.get("unmapped_question")),
            "controlledUnsupported": bool(workflow.get("controlled_unsupported")),
            "controlledRefusal": refused,
            # Technical diagnostics retained for engineers / an expandable UI panel.
            "diagnostics": diagnostics,
        },
    }
