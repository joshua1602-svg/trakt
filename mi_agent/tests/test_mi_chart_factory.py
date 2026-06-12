"""
Tests for the MI Chart Factory v1 (mi_agent/mi_chart_factory.py).

Charts are built from real executor output (executor called in-test on synthetic
data), so these exercise the executor -> chart-factory contract end to end.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from mi_agent.mi_chart_factory import (
    MIChartError,
    MIChartResult,
    compact_currency,
    compact_number,
    create_mi_chart,
    format_date_label,
    format_percent,
    generate_chart_title,
)
from mi_agent.mi_query_executor import execute_mi_query
from mi_agent.mi_query_spec import MIQuerySpec
from mi_agent.mi_query_validator import load_mi_semantics

REPO_ROOT = Path(__file__).resolve().parents[2]
SEMANTICS_PATH = REPO_ROOT / "mi_agent" / "mi_semantics_field_registry.yaml"


@pytest.fixture(scope="module")
def semantics():
    return load_mi_semantics(SEMANTICS_PATH)


@pytest.fixture
def df():
    regions7 = ["North", "South", "East", "West", "Wales", "Scotland", "NI"]
    rows = []
    for i in range(48):
        rows.append({
            "loan_identifier": f"L{i:04d}",
            "current_outstanding_balance": 100_000 + i * 5_000,
            "current_principal_balance": 95_000 + i * 5_000,
            "current_loan_to_value": 0.30 + (i % 5) * 0.05,   # fraction scale
            "current_interest_rate": 3.0 + (i % 4) * 0.5,
            "youngest_borrower_age": 55 + (i % 20),
            "geographic_region_obligor": regions7[i % 7],
            "broker_channel": ["Broker A", "Broker B"][i % 2],
            "account_status": ["Performing", "Arrears", "Default"][i % 3],
            "origination_date": pd.Timestamp("2021-01-01") + pd.Timedelta(days=40 * i),
            "age_bucket": ["55-60", "60-65", "65-70", "70-75"][i % 4],
        })
    return pd.DataFrame(rows)


def _chart(spec, df, semantics, **kw):
    result = execute_mi_query(spec, df, semantics)
    return create_mi_chart(result, semantics, **kw)


# --------------------------------------------------------------------------- #
# 1-2. bar (and horizontal when >6 categories)
# --------------------------------------------------------------------------- #


def test_bar_balance_by_region(df, semantics):
    # only 3 regions -> vertical bar
    small = df[df["geographic_region_obligor"].isin(["North", "South", "East"])]
    c = _chart(MIQuerySpec(intent="chart", chart_type="bar",
                           metric="current_outstanding_balance",
                           dimension="geographic_region_obligor",
                           aggregation="sum"), small, semantics)
    assert isinstance(c, MIChartResult)
    assert c.chart_type == "bar"
    assert c.fig.data[0].type == "bar"
    assert c.fig.data[0].orientation in (None, "v")


def test_bar_horizontal_when_many_categories(df, semantics):
    # 7 regions -> horizontal
    c = _chart(MIQuerySpec(intent="chart", chart_type="bar",
                           metric="current_outstanding_balance",
                           dimension="geographic_region_obligor",
                           aggregation="sum"), df, semantics)
    assert c.fig.data[0].orientation == "h"


# --------------------------------------------------------------------------- #
# 3. concentration in hover
# --------------------------------------------------------------------------- #


def test_bar_hover_includes_concentration(df, semantics):
    c = _chart(MIQuerySpec(intent="chart", chart_type="bar",
                           metric="current_outstanding_balance",
                           dimension="account_status", aggregation="sum"),
               df, semantics)
    assert "Concentration" in c.fig.data[0].hovertemplate
    # customdata carries the concentration string in the 2nd column
    assert c.fig.data[0].customdata.shape[1] == 2


# --------------------------------------------------------------------------- #
# 4. line
# --------------------------------------------------------------------------- #


def test_line_monthly_balance(df, semantics):
    c = _chart(MIQuerySpec(intent="chart", chart_type="line", x="origination_date",
                           metric="current_outstanding_balance", aggregation="sum"),
               df, semantics)
    assert c.chart_type == "line"
    assert c.fig.data[0].type == "scatter"
    xs = list(c.fig.data[0].x)
    # month labels like 'Jan-21'
    assert all("-" in x for x in xs)


# --------------------------------------------------------------------------- #
# 5-6. scatter / bubble
# --------------------------------------------------------------------------- #


def test_scatter_rate_vs_ltv(df, semantics):
    c = _chart(MIQuerySpec(intent="chart", chart_type="scatter",
                           x="current_interest_rate", y="current_loan_to_value",
                           aggregation="loan_level"), df, semantics)
    assert c.chart_type == "scatter"
    assert c.fig.data[0].mode == "markers"
    assert c.fig.data[0].marker.opacity < 1.0


def test_bubble_ltv_by_age_sized_by_balance(df, semantics):
    c = _chart(MIQuerySpec(intent="chart", chart_type="bubble",
                           x="youngest_borrower_age", y="current_loan_to_value",
                           size="current_outstanding_balance",
                           aggregation="loan_level"), df, semantics)
    assert c.chart_type == "bubble"
    assert c.fig.data[0].marker.sizemode == "area"
    assert c.fig.data[0].marker.opacity < 1.0


# --------------------------------------------------------------------------- #
# 7. heatmap
# --------------------------------------------------------------------------- #


def test_heatmap_wavg_ltv_by_age_bucket_and_region(df, semantics):
    c = _chart(MIQuerySpec(intent="chart", chart_type="heatmap",
                           metric="current_loan_to_value",
                           dimensions=["age_bucket", "geographic_region_obligor"],
                           aggregation="weighted_avg"), df, semantics)
    assert c.chart_type == "heatmap"
    assert c.fig.data[0].type == "heatmap"
    # buckets should be ordered numerically on the y axis
    assert list(c.fig.data[0].y) == ["55-60", "60-65", "65-70", "70-75"]


# --------------------------------------------------------------------------- #
# 8. treemap
# --------------------------------------------------------------------------- #


def test_treemap_balance_by_region_and_broker(df, semantics):
    c = _chart(MIQuerySpec(intent="chart", chart_type="treemap",
                           metric="current_outstanding_balance",
                           hierarchy=["geographic_region_obligor", "broker_channel"],
                           aggregation="sum"), df, semantics)
    assert c.chart_type == "treemap"
    assert c.fig.data[0].type == "treemap"
    assert "Concentration" in c.fig.data[0].hovertemplate


# --------------------------------------------------------------------------- #
# 9. compact currency formatting
# --------------------------------------------------------------------------- #


def test_currency_formatting_compact():
    assert compact_currency(1_200_000) == "£1.2m"
    assert compact_currency(2_000_000) == "£2m"
    assert compact_currency(450_000) == "£450k"
    assert compact_currency(25_000) == "£25,000"
    assert compact_number(1_500_000) == "1.5m"


def test_bar_value_labels_use_compact_currency(df, semantics):
    c = _chart(MIQuerySpec(intent="chart", chart_type="bar",
                           metric="current_outstanding_balance",
                           dimension="geographic_region_obligor",
                           aggregation="sum"), df, semantics)
    labels = list(c.fig.data[0].text)
    assert any(lbl.startswith("£") and (lbl.endswith("m") or lbl.endswith("k")
               or "," in lbl) for lbl in labels)
    # currency axis carries the £ prefix
    axis = c.fig.layout.xaxis if c.fig.data[0].orientation == "h" else c.fig.layout.yaxis
    assert axis.tickprefix == "£"


# --------------------------------------------------------------------------- #
# 10. percent scale respected
# --------------------------------------------------------------------------- #


def test_percent_scale_fraction():
    # 0.36 fraction -> 36.0%
    assert format_percent(0.36, "fraction") == "36.0%"
    # 37.9 whole-number -> 37.9%
    assert format_percent(37.9, "whole_number_percent") == "37.9%"


def test_percent_hover_respects_executor_scale(df, semantics):
    # Isolate LTV (stored as fractions 0.30..0.50) as the only percent field so
    # the executor unambiguously detects scale == 'fraction'; the factory must
    # then scale up to a "36.5%"-style label rather than print "0.365".
    frac_df = df.drop(columns=["current_interest_rate"])
    result = execute_mi_query(
        MIQuerySpec(intent="chart", chart_type="heatmap",
                    metric="current_loan_to_value",
                    dimensions=["age_bucket", "geographic_region_obligor"],
                    aggregation="weighted_avg"), frac_df, semantics)
    assert result.metadata["percent_scale_detected"] == "fraction"
    c = create_mi_chart(result, semantics)
    flat = [v for row in c.fig.data[0].text for v in row if v]
    assert flat and all(v.endswith("%") for v in flat)
    # scaled up (> 1) because the source was a fraction
    assert float(flat[0].rstrip("%")) > 1.0


# --------------------------------------------------------------------------- #
# 11. title generation uses business names
# --------------------------------------------------------------------------- #


def test_title_uses_business_names(df, semantics):
    result = execute_mi_query(
        MIQuerySpec(intent="chart", chart_type="bar",
                    metric="current_loan_to_value",
                    dimension="geographic_region_obligor",
                    aggregation="weighted_avg"), df, semantics)
    title, subtitle = generate_chart_title(result, semantics)
    assert title == "Weighted Average Current LTV by Region"


def test_title_respects_explicit_spec_title(df, semantics):
    result = execute_mi_query(
        MIQuerySpec(intent="chart", chart_type="bar",
                    metric="current_outstanding_balance",
                    dimension="geographic_region_obligor", aggregation="sum",
                    title="Portfolio Exposure by Region"), df, semantics)
    title, _ = generate_chart_title(result, semantics)
    assert title == "Portfolio Exposure by Region"


# --------------------------------------------------------------------------- #
# 12-13. error paths
# --------------------------------------------------------------------------- #


def test_table_only_result_raises(df, semantics):
    result = execute_mi_query(MIQuerySpec(intent="table", dimension="account_status",
                                          aggregation="count", chart_type="none"),
                              df, semantics)
    with pytest.raises(MIChartError, match="no chart type"):
        create_mi_chart(result, semantics)


def test_summary_result_raises(df, semantics):
    result = execute_mi_query(MIQuerySpec(intent="summary",
                                          metric="current_outstanding_balance",
                                          aggregation="sum"), df, semantics)
    with pytest.raises(MIChartError, match="summary"):
        create_mi_chart(result, semantics)


# --------------------------------------------------------------------------- #
# 14-15. export
# --------------------------------------------------------------------------- #


def test_to_html(df, semantics):
    c = _chart(MIQuerySpec(intent="chart", chart_type="bar",
                           metric="current_outstanding_balance",
                           dimension="account_status", aggregation="sum"),
               df, semantics)
    html = c.to_html()
    assert isinstance(html, str) and "<html" in html.lower()


def test_to_json(df, semantics):
    c = _chart(MIQuerySpec(intent="chart", chart_type="bar",
                           metric="current_outstanding_balance",
                           dimension="account_status", aggregation="sum"),
               df, semantics)
    payload = json.loads(c.to_json())
    assert payload["chart_type"] == "bar"
    assert "figure" in payload and payload["figure"]["data"]


def test_write_html(df, semantics, tmp_path):
    c = _chart(MIQuerySpec(intent="chart", chart_type="bar",
                           metric="current_outstanding_balance",
                           dimension="account_status", aggregation="sum"),
               df, semantics)
    out = tmp_path / "chart.html"
    c.write_html(out)
    assert out.exists() and out.stat().st_size > 0


# --------------------------------------------------------------------------- #
# 16. explicit (non-default) layout styling
# --------------------------------------------------------------------------- #


def test_layout_has_explicit_styling(df, semantics):
    c = _chart(MIQuerySpec(intent="chart", chart_type="bar",
                           metric="current_outstanding_balance",
                           dimension="account_status", aggregation="sum"),
               df, semantics)
    lay = c.fig.layout
    assert lay.font.family and "Calibri" in lay.font.family
    assert lay.paper_bgcolor == "white"
    assert lay.plot_bgcolor == "white"
    assert lay.margin.l is not None and lay.margin.t is not None
    # subtle gridlines configured explicitly
    assert lay.yaxis.gridcolor == "#F0F0F0"
    # restrained categorical palette set (not raw Plotly default colourway)
    assert lay.colorway[0] == "#232D55"


def test_date_label_helper():
    assert format_date_label("2026-01") == "Jan-26"
    assert format_date_label("2026") == "2026"
