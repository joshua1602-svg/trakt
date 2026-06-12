#!/usr/bin/env python3
"""
mi_chart_factory.py

MI Chart Factory v1 — render enterprise-ready Plotly figures from the
deterministic :class:`MIQueryResult` produced by ``mi_query_executor``.

This module is DETERMINISTIC and ISOLATED:
  * no LLM calls
  * no Streamlit / Azure / pipeline integration
  * no arbitrary Plotly generation (only the chart types MIQuerySpec supports)
  * never re-runs the executor and never mutates ``result.data``

------------------------------------------------------------------------------
VISUAL DEFAULTS — repo inspection
------------------------------------------------------------------------------
The existing dashboard styling lives in ``analytics/charts_plotly.py`` and
``analytics/streamlit_app_erm.py`` (function ``apply_chart_theme``):

    PRIMARY_COLOR   = "#232D55"  (navy / slate)
    SECONDARY_COLOR = "#919DD1"  (muted blue)
    ACCENT_COLOR    = "#BFBFBF"  (grey)
    TEXT_DARK       = "#2D2D2D"
    BORDER_COLOR    = "#E0E0E0"
    font            = "Calibri"
    plot/paper bg   = white
    gridcolor       = "#F0F0F0"
    margins         = l20 r20 t60 b30, left-aligned title size 18 weight 600,
                      horizontal legend, hovermode "closest"

We intentionally DO NOT import that module (it imports ``mi_prep`` at module
load, coupling the MI agent to the analytics pipeline). Instead this file keeps
an isolated MI-Agent copy of the same look-and-feel (navy primary, muted-blue
secondary, grey accent, Calibri→Arial font, white background, subtle grid),
extended with restrained categorical/sequential palettes and a positive/
negative/neutral set for Financial-Services-grade, executive-ready output.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from .mi_query_executor import MIQueryResult
from .mi_query_validator import load_mi_semantics


class MIChartError(Exception):
    """Raised when a chart cannot be rendered from an MIQueryResult."""


# --------------------------------------------------------------------------- #
# Theme (isolated MI-Agent copy of the repo look-and-feel)
# --------------------------------------------------------------------------- #

DEFAULT_THEME: Dict[str, Any] = {
    # Aligned with analytics/charts_plotly.py (recreated here, not imported).
    "font_family": "Calibri, Arial, sans-serif",
    "paper_bgcolor": "white",
    "plot_bgcolor": "white",
    "primary": "#232D55",      # navy / slate
    "secondary": "#919DD1",    # muted blue
    "accent": "#BFBFBF",       # grey
    "text": "#2D2D2D",         # near-black slate
    "muted_text": "#5A6275",   # subtitle / secondary text
    "gridcolor": "#F0F0F0",    # subtle gridlines
    "axis_line": "#BFBFBF",
    # Restrained accents for FS dashboards (no toy-like colours).
    "positive": "#2E7D5B",
    "negative": "#B23A48",
    "neutral": "#8893A8",
    # Limited categorical palette derived from the brand navy/blue/slate.
    "categorical": ["#232D55", "#919DD1", "#5A6B9C", "#3E7C8C",
                    "#B0894A", "#8893A8", "#6E5B8C", "#4F6E7B"],
    # Muted sequential scale (light → navy) for heatmaps.
    "sequential": [[0.0, "#F2F4F8"], [0.5, "#919DD1"], [1.0, "#232D55"]],
    "title_size": 18,
    "subtitle_size": 12.5,
    "axis_title_size": 13,
    "tick_size": 11,
    "height": 460,
    "width": None,
    "margin": {"l": 80, "r": 36, "t": 92, "b": 64},
}


def _merge_theme(theme: Optional[dict]) -> dict:
    merged = dict(DEFAULT_THEME)
    if theme:
        merged.update(theme)
    return merged


# --------------------------------------------------------------------------- #
# Result schema
# --------------------------------------------------------------------------- #


@dataclass
class MIChartResult:
    fig: go.Figure
    chart_type: str
    title: str
    subtitle: Optional[str] = None
    warnings: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_html(self, path: Optional[str] = None,
                include_plotlyjs: str = "cdn") -> str:
        html = self.fig.to_html(include_plotlyjs=include_plotlyjs, full_html=True)
        if path:
            Path(path).write_text(html, encoding="utf-8")
        return html

    def write_html(self, path) -> str:
        return self.to_html(path=str(path))

    def to_json(self) -> str:
        return json.dumps({
            "chart_type": self.chart_type,
            "title": self.title,
            "subtitle": self.subtitle,
            "warnings": list(self.warnings),
            "metadata": self.metadata,
            "figure": json.loads(self.fig.to_json()),
        }, default=str)

    def write_image(self, path) -> str:
        try:
            import kaleido  # noqa: F401
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise MIChartError(
                "write_image requires the optional 'kaleido' package "
                "(pip install kaleido)."
            ) from exc
        self.fig.write_image(str(path))
        return str(path)


# --------------------------------------------------------------------------- #
# Value formatters
# --------------------------------------------------------------------------- #


def _is_missing(value) -> bool:
    try:
        return value is None or (isinstance(value, float) and math.isnan(value)) \
            or pd.isna(value)
    except (TypeError, ValueError):
        return False


def compact_currency(value, currency_symbol: str = "£") -> str:
    """£1.2m / £450k / £25,000 (k used from 100k, m from 1m)."""
    if _is_missing(value):
        return ""
    v = float(value)
    sign = "-" if v < 0 else ""
    a = abs(v)
    if a >= 1_000_000:
        s = f"{a / 1_000_000:.1f}".rstrip("0").rstrip(".")
        return f"{sign}{currency_symbol}{s}m"
    if a >= 100_000:
        return f"{sign}{currency_symbol}{a / 1_000:.0f}k"
    return f"{sign}{currency_symbol}{a:,.0f}"


def compact_number(value) -> str:
    """1.2m / 450k / 25,000 — compact form for large non-currency numbers."""
    if _is_missing(value):
        return ""
    v = float(value)
    sign = "-" if v < 0 else ""
    a = abs(v)
    if a >= 1_000_000:
        s = f"{a / 1_000_000:.1f}".rstrip("0").rstrip(".")
        return f"{sign}{s}m"
    if a >= 100_000:
        return f"{sign}{a / 1_000:.0f}k"
    return f"{sign}{a:,.0f}"


def format_percent(value, percent_scale: Optional[str] = None) -> str:
    """Respect executor percent-scale metadata; never mutate underlying data.

    fraction            -> 0.36 displayed as "36.0%"
    whole_number_percent-> 37.9 displayed as "37.9%"
    unknown / None       -> value shown as-is with "%"
    """
    if _is_missing(value):
        return ""
    v = float(value)
    if percent_scale == "fraction":
        v *= 100.0
    return f"{v:.1f}%"


def format_date_label(value) -> str:
    """'2026-01' -> 'Jan-26'; '2026' -> '2026'; else best-effort string."""
    if _is_missing(value):
        return ""
    s = str(value).strip()
    if len(s) == 7 and s[4] == "-":
        try:
            return datetime.strptime(s, "%Y-%m").strftime("%b-%y")
        except ValueError:
            return s
    if len(s) == 4 and s.isdigit():
        return s
    # ISO datetime / date
    try:
        return pd.to_datetime(s).strftime("%b-%y")
    except (ValueError, TypeError):
        return s


def format_axis_tick(value, format_type: str,
                     percent_scale: Optional[str] = None) -> str:
    if _is_missing(value):
        return ""
    if format_type == "currency":
        return compact_currency(value)
    if format_type == "percent":
        return format_percent(value, percent_scale)
    if format_type == "ratio":
        return f"{float(value):.2f}x"
    if format_type == "integer":
        return compact_number(value)
    if format_type == "decimal":
        return f"{float(value):.2f}"
    if format_type == "date":
        return format_date_label(value)
    return str(value)


def format_hover_value(value, format_type: str,
                       percent_scale: Optional[str] = None) -> str:
    if _is_missing(value):
        return "—"
    if format_type == "integer":
        return f"{int(round(float(value))):,}"
    if format_type == "currency":
        return compact_currency(value)
    if format_type == "percent":
        return format_percent(value, percent_scale)
    if format_type == "ratio":
        return f"{float(value):.2f}x"
    if format_type == "decimal":
        return f"{float(value):.2f}"
    if format_type == "date":
        return format_date_label(value)
    return str(value)


# --------------------------------------------------------------------------- #
# Field / format helpers
# --------------------------------------------------------------------------- #


_RATIO_FIELDS = {"current_debt_service_coverage_ratio", "debt_to_income_ratio"}


def _business_name(semantics: dict, key: Optional[str]) -> str:
    if not key:
        return ""
    entry = semantics.get("fields", {}).get(key)
    if entry:
        return entry.get("business_name") or entry.get("display_name") or key
    return str(key).replace("_", " ").title()


def _value_format_type(result: MIQueryResult, semantics: dict) -> str:
    """Format type for the aggregated metric column."""
    spec = result.spec
    agg = spec.aggregation
    if agg in ("count", "count_distinct"):
        return "integer"
    if agg == "balance_sum":
        return "currency"
    if spec.metric:
        if spec.metric in _RATIO_FIELDS:
            return "ratio"
        return semantics.get("fields", {}).get(spec.metric, {}).get("format", "decimal")
    return "integer"


def _field_format_type(semantics: dict, key: Optional[str]) -> str:
    if not key:
        return "decimal"
    if key in _RATIO_FIELDS:
        return "ratio"
    return semantics.get("fields", {}).get(key, {}).get("format", "decimal")


def _percent_scale(result: MIQueryResult) -> Optional[str]:
    return result.metadata.get("percent_scale_detected")


def _split_columns(data: pd.DataFrame) -> Tuple[List[str], Optional[str]]:
    """Return (group_columns, value_column) for a grouped/table result."""
    reserved = {"concentration_pct"}
    value_candidates = [c for c in data.columns
                        if c not in reserved and pd.api.types.is_numeric_dtype(data[c])]
    group_cols = [c for c in data.columns
                  if c not in reserved and c not in value_candidates]
    value_col = value_candidates[-1] if value_candidates else None
    return group_cols, value_col


# --------------------------------------------------------------------------- #
# Title / subtitle generation
# --------------------------------------------------------------------------- #

_AGG_PREFIX = {
    "weighted_avg": "Weighted Average ",
    "avg": "Average ",
    "median": "Median ",
    "distribution": "Distribution of ",
}


def generate_chart_title(result: MIQueryResult,
                         semantics: dict) -> Tuple[str, Optional[str]]:
    spec = result.spec
    ct = spec.chart_type
    bn = lambda k: _business_name(semantics, k)  # noqa: E731

    if spec.title:
        title = spec.title
    else:
        metric_name = bn(spec.metric) if spec.metric else "Loan Count"
        prefix = _AGG_PREFIX.get(spec.aggregation, "")
        if ct == "scatter":
            title = f"{bn(spec.y)} vs {bn(spec.x)}"
        elif ct == "bubble":
            title = f"{bn(spec.y)} by {bn(spec.x)}"
        elif ct == "heatmap":
            dims = (spec.dimensions[:2] if spec.dimensions
                    else [k for k in (spec.x, spec.y, spec.dimension, spec.color) if k][:2])
            title = f"{prefix}{metric_name} by " + " and ".join(bn(d) for d in dims)
        elif ct == "treemap":
            dims = list(spec.hierarchy) or list(spec.dimensions)
            if spec.dimension:
                dims = dims + [spec.dimension]
            title = f"{prefix}{metric_name} by " + " and ".join(bn(d) for d in dims if d)
        else:  # bar / line
            dim = spec.dimension or spec.x
            title = f"{prefix}{metric_name} by {bn(dim)}"

    # Subtitle: aggregation context + executor notes.
    parts: List[str] = []
    if spec.chart_type == "bubble" and spec.size:
        parts.append(f"Bubble size: {bn(spec.size)}")
    if spec.top_n is not None:
        rank = result.metadata.get("top_n_rank_priority", ["balance"])
        parts.append(f"Top {spec.top_n} by {rank[0] if rank else 'balance'}")
    if result.metadata.get("loan_level_sampled"):
        parts.append(
            f"Sampled {result.metadata.get('loan_level_returned_rows'):,} of "
            f"{result.metadata.get('loan_level_original_rows'):,} loans"
        )
    if (_value_format_type(result, semantics) == "percent"
            and _percent_scale(result) == "unknown"):
        parts.append("percent scale ambiguous — values shown unscaled")
    subtitle = " · ".join(parts) if parts else None
    return title, subtitle


# --------------------------------------------------------------------------- #
# Layout / theme application
# --------------------------------------------------------------------------- #


def _apply_theme(fig: go.Figure, theme: dict, title: str,
                 subtitle: Optional[str], height: Optional[int],
                 width: Optional[int]) -> None:
    fig.update_layout(
        template="none",  # never rely on a raw Plotly default template
        title=dict(
            text=title,
            x=0, xanchor="left",
            font=dict(family=theme["font_family"], size=theme["title_size"],
                      color=theme["primary"]),
            subtitle=dict(
                text=subtitle or "",
                font=dict(family=theme["font_family"],
                          size=theme["subtitle_size"], color=theme["muted_text"]),
            ),
        ),
        font=dict(family=theme["font_family"], size=theme["tick_size"],
                  color=theme["text"]),
        paper_bgcolor=theme["paper_bgcolor"],
        plot_bgcolor=theme["plot_bgcolor"],
        colorway=theme["categorical"],
        margin=theme["margin"],
        height=height or theme["height"],
        width=width or theme["width"],
        hovermode="closest",
        hoverlabel=dict(bgcolor="white", bordercolor=theme["accent"],
                        font=dict(family=theme["font_family"], size=11,
                                  color=theme["text"])),
        legend=dict(orientation="h", y=-0.18, x=0.5, xanchor="center",
                    font=dict(family=theme["font_family"], size=11,
                              color=theme["text"])),
    )


def _style_axes(fig: go.Figure, theme: dict, *, x_grid: bool = False,
                y_grid: bool = True) -> None:
    common = dict(
        gridcolor=theme["gridcolor"], gridwidth=1,
        title=dict(font=dict(family=theme["font_family"],
                             size=theme["axis_title_size"], color=theme["primary"])),
        tickfont=dict(family=theme["font_family"], size=theme["tick_size"],
                      color=theme["primary"]),
        showline=True, linewidth=1, linecolor=theme["axis_line"],
        zeroline=False, mirror=False,
    )
    fig.update_xaxes(showgrid=x_grid, **common)
    fig.update_yaxes(showgrid=y_grid, **common)


def _apply_value_axis(fig: go.Figure, axis: str, fmt: str,
                      percent_scale: Optional[str]) -> None:
    """Set tick formatting on the numeric (value) axis."""
    kw: Dict[str, Any] = {}
    if fmt == "currency":
        kw = dict(tickprefix="£", tickformat="~s")
    elif fmt == "percent":
        if percent_scale == "fraction":
            kw = dict(tickformat=".0%")
        else:  # whole_number_percent / unknown -> already 0-100
            kw = dict(ticksuffix="%", tickformat=".0f")
    elif fmt == "integer":
        kw = dict(tickformat="~s")
    elif fmt == "ratio":
        kw = dict(ticksuffix="x", tickformat=".1f")
    elif fmt == "decimal":
        kw = dict(tickformat=".2f")
    if not kw:
        return
    if axis == "x":
        fig.update_xaxes(**kw)
    else:
        fig.update_yaxes(**kw)


# --------------------------------------------------------------------------- #
# Bucket ordering
# --------------------------------------------------------------------------- #


def _bucket_sort_key(label: str):
    """Order bucket labels like '<55', '55-60', '80-85', '85+', '<50k', '50-100k'."""
    s = str(label).strip().lower().replace("£", "").replace("%", "")
    mult = 1
    if s.endswith("k"):
        mult = 1_000
    elif s.endswith("m"):
        mult = 1_000_000
    body = s.rstrip("km+")
    if body.startswith("<"):
        return (-1.0,)
    head = body.split("-")[0].split()[0] if body else ""
    try:
        return (float(head) * mult,)
    except ValueError:
        return (math.inf,)


# --------------------------------------------------------------------------- #
# Chart builders
# --------------------------------------------------------------------------- #


def _build_bar(result, semantics, theme):
    data = result.data.copy()
    group_cols, value_col = _split_columns(data)
    if not group_cols or value_col is None:
        raise MIChartError("bar chart requires a dimension column and a value column")
    cat_col = group_cols[0]
    spec = result.spec
    fmt = _value_format_type(result, semantics)
    ps = _percent_scale(result)
    has_conc = "concentration_pct" in data.columns

    dim_key = spec.dimension or spec.x
    is_date = (semantics.get("fields", {}).get(dim_key, {}).get("role") == "date")
    horizontal = len(data) > 6

    if horizontal and not is_date:
        # largest at top -> ascending for plotly's bottom-up ordering
        data = data.sort_values(value_col, ascending=True, kind="mergesort")

    cats = data[cat_col].astype(str).tolist()
    vals = data[value_col].tolist()
    text = [format_axis_tick(v, fmt, ps) for v in vals]

    cd_cols = [[format_hover_value(v, fmt, ps)] for v in vals]
    metric_name = (_business_name(semantics, spec.metric) if spec.metric else "Loan Count")
    hover = f"<b>%{{label}}</b><br>{metric_name}: %{{customdata[0]}}"
    if has_conc:
        for i, c in enumerate(data["concentration_pct"].tolist()):
            cd_cols[i].append(f"{float(c):.1f}%")
        hover += "<br>Concentration: %{customdata[1]}"
    hover += "<extra></extra>"
    customdata = np.array(cd_cols, dtype=object)

    fig = go.Figure()
    if horizontal:
        fig.add_trace(go.Bar(
            x=vals, y=cats, orientation="h", marker_color=theme["primary"],
            text=text, textposition="auto", customdata=customdata,
            hovertemplate=hover,
        ))
        fig.data[0].update(ids=cats)
        fig.update_layout(yaxis=dict(type="category"))
        _style_axes(fig, theme, x_grid=True, y_grid=False)
        _apply_value_axis(fig, "x", fmt, ps)
        fig.update_xaxes(title_text=metric_name)
        fig.update_yaxes(title_text=_business_name(semantics, dim_key))
    else:
        fig.add_trace(go.Bar(
            x=cats, y=vals, marker_color=theme["primary"],
            text=text, textposition="outside", customdata=customdata,
            hovertemplate=hover,
        ))
        fig.update_layout(xaxis=dict(type="category"))
        _style_axes(fig, theme, x_grid=False, y_grid=True)
        _apply_value_axis(fig, "y", fmt, ps)
        fig.update_yaxes(title_text=metric_name)
        fig.update_xaxes(title_text=_business_name(semantics, dim_key))
    fig.update_layout(showlegend=False)
    return fig


def _build_line(result, semantics, theme):
    data = result.data.copy()
    group_cols, value_col = _split_columns(data)
    if not group_cols or value_col is None:
        raise MIChartError("line chart requires a period column and a value column")
    period_col = group_cols[0]
    spec = result.spec
    fmt = _value_format_type(result, semantics)
    ps = _percent_scale(result)

    data = data.sort_values(period_col, kind="mergesort")
    periods = data[period_col].astype(str).tolist()
    labels = [format_date_label(p) for p in periods]
    vals = data[value_col].tolist()
    metric_name = (_business_name(semantics, spec.metric) if spec.metric else "Loan Count")
    customdata = np.array([[format_hover_value(v, fmt, ps)] for v in vals], dtype=object)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=labels, y=vals, mode="lines+markers" if len(labels) <= 24 else "lines",
        line=dict(color=theme["primary"], width=2),
        marker=dict(color=theme["primary"], size=6),
        customdata=customdata,
        hovertemplate=f"<b>%{{x}}</b><br>{metric_name}: %{{customdata[0]}}<extra></extra>",
    ))
    fig.update_layout(xaxis=dict(type="category"), showlegend=False)
    _style_axes(fig, theme, x_grid=False, y_grid=True)
    _apply_value_axis(fig, "y", fmt, ps)
    x_key = spec.x or spec.dimension
    fig.update_xaxes(title_text=_business_name(semantics, x_key) or "Period")
    fig.update_yaxes(title_text=metric_name)
    return fig


def _scatter_like(result, semantics, theme, *, bubble: bool):
    data = result.data.copy()
    spec = result.spec
    x_col = result.resolved_fields.get(spec.x, {}).get("canonical_field", spec.x)
    y_col = result.resolved_fields.get(spec.y, {}).get("canonical_field", spec.y)
    size_col = (result.resolved_fields.get(spec.size, {}).get("canonical_field", spec.size)
                if bubble and spec.size else None)
    color_col = (result.resolved_fields.get(spec.color, {}).get("canonical_field", spec.color)
                 if spec.color else None)
    x_fmt = _field_format_type(semantics, spec.x)
    y_fmt = _field_format_type(semantics, spec.y)
    ps = _percent_scale(result)
    x_name = _business_name(semantics, spec.x)
    y_name = _business_name(semantics, spec.y)

    def _marker_size(series):
        s = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
        finite = s[np.isfinite(s)]
        if finite.size == 0 or finite.max() <= 0:
            return None
        # area-mode scaling, capped so bubbles never look cartoonish
        return dict(size=s, sizemode="area",
                    sizeref=2.0 * finite.max() / (38.0 ** 2), sizemin=4)

    fig = go.Figure()
    groups = ([(None, data)] if not color_col
              else list(data.groupby(color_col, sort=False)))
    palette = theme["categorical"]
    for i, (gval, gdf) in enumerate(groups):
        cd = np.column_stack([
            [format_hover_value(v, x_fmt, ps) for v in gdf[x_col]],
            [format_hover_value(v, y_fmt, ps) for v in gdf[y_col]],
        ]).astype(object)
        hover = f"{x_name}: %{{customdata[0]}}<br>{y_name}: %{{customdata[1]}}"
        marker = dict(color=palette[i % len(palette)] if color_col else theme["primary"],
                      opacity=0.6, line=dict(width=0))
        if bubble and size_col:
            ms = _marker_size(gdf[size_col])
            if ms:
                ms["color"] = marker["color"]
                ms["opacity"] = 0.6
                ms["line"] = dict(width=0.5, color="white")
                marker = ms
            size_name = _business_name(semantics, spec.size)
            sz_fmt = _field_format_type(semantics, spec.size)
            cd = np.column_stack([
                cd,
                [format_hover_value(v, sz_fmt, ps) for v in gdf[size_col]],
            ]).astype(object)
            hover += f"<br>{size_name}: %{{customdata[2]}}"
        if color_col:
            hover += f"<br>{_business_name(semantics, spec.color)}: {gval}"
        hover += "<extra></extra>"
        fig.add_trace(go.Scatter(
            x=gdf[x_col], y=gdf[y_col], mode="markers", name=str(gval),
            marker=marker, customdata=cd, hovertemplate=hover,
        ))
    fig.update_layout(showlegend=bool(color_col))
    _style_axes(fig, theme, x_grid=True, y_grid=True)
    _apply_value_axis(fig, "x", x_fmt, ps)
    _apply_value_axis(fig, "y", y_fmt, ps)
    fig.update_xaxes(title_text=x_name)
    fig.update_yaxes(title_text=y_name)
    return fig


def _build_heatmap(result, semantics, theme):
    data = result.data.copy()
    group_cols, value_col = _split_columns(data)
    if len(group_cols) < 2 or value_col is None:
        raise MIChartError("heatmap requires two dimension columns and a value column")
    row_col, col_col = group_cols[0], group_cols[1]
    fmt = _value_format_type(result, semantics)
    ps = _percent_scale(result)

    pivot = data.pivot_table(index=row_col, columns=col_col,
                             values=value_col, aggfunc="first")
    pivot = pivot.reindex(sorted(pivot.index, key=_bucket_sort_key))
    pivot = pivot[sorted(pivot.columns, key=_bucket_sort_key)]

    z = pivot.to_numpy(dtype=float)
    text = [[format_hover_value(v, fmt, ps) for v in row] for row in z]
    row_name = _business_name(semantics, row_col)
    col_name = _business_name(semantics, col_col)
    metric_name = _business_name(semantics, result.spec.metric) or value_col

    fig = go.Figure(go.Heatmap(
        z=z, x=[str(c) for c in pivot.columns], y=[str(r) for r in pivot.index],
        text=text, colorscale=theme["sequential"],
        hoverongaps=False,
        colorbar=dict(title=dict(text=metric_name,
                                 font=dict(family=theme["font_family"], size=11)),
                      outlinewidth=0),
        hovertemplate=(f"{row_name}: %{{y}}<br>{col_name}: %{{x}}<br>"
                       f"{metric_name}: %{{text}}<extra></extra>"),
    ))
    _style_axes(fig, theme, x_grid=False, y_grid=False)
    fig.update_xaxes(title_text=col_name)
    fig.update_yaxes(title_text=row_name)
    return fig


def _build_treemap(result, semantics, theme):
    data = result.data.copy()
    group_cols, value_col = _split_columns(data)
    if not group_cols or value_col is None:
        raise MIChartError("treemap requires hierarchy columns and a value column")
    fmt = _value_format_type(result, semantics)
    ps = _percent_scale(result)
    metric_name = _business_name(semantics, result.spec.metric) or value_col

    fig = px.treemap(data, path=group_cols, values=value_col)
    tr = fig.data[0]
    # Format node values (parents carry plotly-aggregated totals).
    formatted = [format_hover_value(v, fmt, ps) for v in tr.values]
    has_conc = "concentration_pct" in data.columns
    if has_conc:
        # px builds leaf ids as the value path joined by '/', e.g. "North/Broker A";
        # parent nodes (e.g. "North") have no concentration.
        conc_by_path = {}
        for _, r in data.iterrows():
            path = "/".join(str(r[c]) for c in group_cols)
            conc_by_path[path] = f"{float(r['concentration_pct']):.1f}%"
        conc = [conc_by_path.get(str(i), "") for i in tr.ids]
        customdata = np.column_stack([formatted, conc]).astype(object)
        tr.hovertemplate = (f"<b>%{{label}}</b><br>{metric_name}: %{{customdata[0]}}"
                            f"<br>Concentration: %{{customdata[1]}}<extra></extra>")
    else:
        customdata = np.array([[f] for f in formatted], dtype=object)
        tr.hovertemplate = (f"<b>%{{label}}</b><br>{metric_name}: "
                            f"%{{customdata[0]}}<extra></extra>")
    tr.customdata = customdata
    tr.marker.colors = None
    tr.marker.colorscale = theme["sequential"]
    tr.marker.colors = tr.values
    tr.tiling.pad = 2
    tr.textfont = dict(family=theme["font_family"], color="white", size=12)
    return fig


_BUILDERS = {
    "bar": _build_bar,
    "line": _build_line,
    "scatter": lambda r, s, t: _scatter_like(r, s, t, bubble=False),
    "bubble": lambda r, s, t: _scatter_like(r, s, t, bubble=True),
    "heatmap": _build_heatmap,
    "treemap": _build_treemap,
}


# --------------------------------------------------------------------------- #
# Public entry point
# --------------------------------------------------------------------------- #


def create_mi_chart(
    result: MIQueryResult,
    semantics,
    *,
    theme: Optional[dict] = None,
    title: Optional[str] = None,
    subtitle: Optional[str] = None,
    height: Optional[int] = None,
    width: Optional[int] = None,
) -> MIChartResult:
    """Render an enterprise-ready Plotly figure from an MIQueryResult."""
    if not isinstance(result, MIQueryResult):
        raise MIChartError("create_mi_chart expects an MIQueryResult instance")
    if isinstance(semantics, (str, Path)):
        semantics = load_mi_semantics(semantics)

    spec = result.spec
    ct = spec.chart_type

    if spec.intent == "summary":
        raise MIChartError(
            "summary cards are not implemented in chart factory v1 "
            "(use the executor's summary table directly)"
        )
    if ct in (None, "none"):
        raise MIChartError(
            "table-only result (chart_type 'none') has no chart type to render"
        )
    if ct not in _BUILDERS:
        raise MIChartError(f"Unsupported chart_type for rendering: {ct!r}")

    theme = _merge_theme(theme)
    gen_title, gen_subtitle = generate_chart_title(result, semantics)
    final_title = title if title is not None else gen_title
    final_subtitle = subtitle if subtitle is not None else gen_subtitle

    warnings = list(result.warnings)

    fig = _BUILDERS[ct](result, semantics, theme)
    _apply_theme(fig, theme, final_title, final_subtitle, height, width)

    metadata = {
        "chart_type": ct,
        "result_type": result.result_type,
        "aggregation": spec.aggregation,
        "value_format": _value_format_type(result, semantics),
        "percent_scale_detected": _percent_scale(result),
        "theme": {k: theme[k] for k in ("primary", "secondary", "accent",
                                        "font_family", "paper_bgcolor")},
        "rows_rendered": int(len(result.data)),
        "top_n": spec.top_n,
        "loan_level_sampled": result.metadata.get("loan_level_sampled", False),
    }

    return MIChartResult(
        fig=fig, chart_type=ct, title=final_title, subtitle=final_subtitle,
        warnings=warnings, metadata=metadata,
    )
