# charts_plotly.py
"""Plotly chart layer for ERM Analytics Dashboard (extracted from streamlit_app_erm.py).

This module contains:
- apply_chart_theme (unchanged)
- strat_bar_chart_pure (pure chart factory; no Streamlit calls)

The UI layer wraps strat_bar_chart_pure to preserve st.warning / st.info behaviour.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go

# Theme defaults (overridden at runtime by streamlit_app_erm via YAML config)
PRIMARY_COLOR = "#232D55"
SECONDARY_COLOR = "#919DD1"
ACCENT_COLOR = "#BFBFBF"
TEXT_DARK = "#2D2D2D"

import mi_prep


def apply_chart_theme(fig: go.Figure, title: str = "") -> go.Figure:
    """Apply consistent Equity Release Europe chart theme."""
    fig.update_layout(
        title=dict(
            text=title,
            x=0,
            xanchor="left",
            font=dict(size=18, family="Calibri", color=PRIMARY_COLOR, weight=600),
        ),
        font=dict(family="Calibri", size=12, color=TEXT_DARK),
        plot_bgcolor="white",
        paper_bgcolor="white",
        margin=dict(l=20, r=20, t=60, b=30),
        hovermode="closest",
        legend=dict(
            orientation="h",
            y=-0.15,
            x=0.5,
            xanchor="center",
            font=dict(family="Calibri", size=11),
        ),
    )

    # Common axis style dict to ensure identical rendering
    axis_style = dict(
        showgrid=True,
        gridcolor="#F0F0F0",
        gridwidth=1,
        title=dict(font=dict(family="Calibri", size=13, color=PRIMARY_COLOR)),
        tickfont=dict(family="Calibri", size=11),
        showline=True,
        linewidth=2,
        linecolor=ACCENT_COLOR,
        zeroline=False,  # <--- CRITICAL: Prevents thick "zero" lines
        mirror=False,    # <--- CRITICAL: Prevents box borders affecting thickness
    )

    fig.update_xaxes(**axis_style)
    fig.update_yaxes(**axis_style)

    return fig


def strat_bar_chart_pure(
    df: pd.DataFrame,
    group_col: str,
    value_col: str = "total_balance",
    agg: str = "sum",
    title: str = ""
):
    """Pure stratification bar chart builder.

    Returns:
        (fig, msg, level)
        level in {"warning", "info"}
    """
    if group_col not in df.columns:
        return None, f"Column {group_col} not available", "warning"

    # Aggregate
    if agg == "sum":
        grouped = df.groupby(group_col, dropna=False).agg(
            value_sum=(value_col, "sum")
        ).reset_index()
    elif agg == "count":
        grouped = df.groupby(group_col, dropna=False).size().reset_index(name="value_sum")
    else:
        grouped = df.groupby(group_col, dropna=False)[value_col].agg(agg).reset_index()
        grouped = grouped.rename(columns={value_col: "value_sum"})

    grouped = grouped.replace([np.inf, -np.inf], np.nan)
    grouped = grouped.dropna(subset=["value_sum"])
    grouped = grouped[grouped["value_sum"] != 0]

    if grouped.empty:
        return None, "No data available for this stratification", "info"

    # Sort buckets in order if categorical, else by value
    col_series = grouped[group_col]
    if pd.api.types.is_categorical_dtype(col_series):
        grouped = grouped.sort_values(group_col)
    else:
        grouped = grouped.sort_values("value_sum", ascending=False)

    # Format labels
    if agg == "sum":
        text_labels = [mi_prep.format_currency(val) for val in grouped["value_sum"]]
    elif agg == "count":
        text_labels = [str(int(val)) for val in grouped["value_sum"]]
    else:
        text_labels = [f"{val:.1f}" for val in grouped["value_sum"]]

    grouped["text_label"] = text_labels

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=grouped[group_col],
        y=grouped["value_sum"],
        marker_color=PRIMARY_COLOR,
        text=grouped["text_label"],
        textposition="outside",
        textfont=dict(
            family="Lucida Sans, Lucida Sans Unicode, sans-serif",
            size=11,
            color=TEXT_DARK
        ),
        hovertemplate="<b>%{x}</b><br>%{text}<extra></extra>"
    ))

    fig = apply_chart_theme(fig, title)
    fig.update_xaxes(title_text=group_col.replace("_", " ").title())

    if agg == "sum" and value_col == "total_balance":
        fig.update_yaxes(title_text="Outstanding Balance (£)")
    elif agg == "count":
        fig.update_yaxes(title_text="Loan Count")

    fig.update_layout(showlegend=False)
    return fig, None, None
