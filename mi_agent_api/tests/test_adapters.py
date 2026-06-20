"""Adapter coverage across MIQuerySpec chart types.

Builds synthetic `run_mi_agent_query`-shaped dicts (no Plotly dependency) and
checks the adapter maps each chart type to the right React artifact(s).
"""

from __future__ import annotations

from mi_agent_api.adapters import adapt_workflow_result


def _workflow(chart_type, result_type, data, spec, resolved):
    return {
        "ok": True,
        "error": None,
        "question": "q",
        "parser_mode": "deterministic",
        "spec": spec,
        "interpreted": {"Chart": chart_type},
        "validation": {"ok": True, "errors": [], "warnings": [], "resolved_fields": resolved},
        "query_result": {
            "spec": spec,
            "result_type": result_type,
            "data": data,
            "resolved_fields": resolved,
            "row_count": len(data),
            "warnings": [],
            "metadata": {},
        },
        "chart_result": None
        if chart_type == "none"
        else {"chart_type": chart_type, "title": "T", "subtitle": None, "warnings": [], "metadata": {}, "figure": {"data": [], "layout": {}}},
        "warnings": [],
        "metadata": {},
    }


def _types(resp):
    return [a["type"] for a in resp["artifacts"]]


def test_bar_maps_to_chart_and_table():
    spec = {"chart_type": "bar", "dimension": "geographic_region_obligor", "metric": "current_outstanding_balance"}
    resolved = {"geographic_region_obligor": {"canonical_field": "geographic_region_obligor", "role": "dimension", "format": "string"}}
    data = [{"geographic_region_obligor": "London", "current_outstanding_balance_sum": 184.0}]
    resp = adapt_workflow_result(_workflow("bar", "table", data, spec, resolved))
    assert _types(resp) == ["chart", "table"]
    chart = resp["artifacts"][0]
    assert chart["chartType"] == "bar"
    assert chart["xKey"] == "geographic_region_obligor"
    assert len(chart["series"]) == 1


def test_line_maps_to_chart_and_table():
    spec = {"chart_type": "line", "x": "origination_date", "metric": "redemptions_received_in_period"}
    resolved = {"origination_date": {"canonical_field": "origination_date", "role": "date", "format": "date"}}
    data = [{"origination_date": "2026-01-01", "redemptions_received_in_period_sum": 12.0}]
    resp = adapt_workflow_result(_workflow("line", "table", data, spec, resolved))
    assert "chart" in _types(resp)
    assert resp["artifacts"][0]["chartType"] == "line"


def test_bubble_maps_x_y_size_series():
    spec = {"chart_type": "bubble", "x": "youngest_borrower_age", "y": "current_loan_to_value", "size": "current_outstanding_balance"}
    resolved = {
        "youngest_borrower_age": {"canonical_field": "youngest_borrower_age", "role": "metric", "format": "integer"},
        "current_loan_to_value": {"canonical_field": "current_loan_to_value", "role": "metric", "format": "percent"},
        "current_outstanding_balance": {"canonical_field": "current_outstanding_balance", "role": "metric", "format": "currency"},
    }
    data = [{"youngest_borrower_age": 71, "current_loan_to_value": 31.4, "current_outstanding_balance": 184000.0}]
    resp = adapt_workflow_result(_workflow("bubble", "loan_level", data, spec, resolved))
    chart = next(a for a in resp["artifacts"] if a["type"] == "chart")
    keys = [s["key"] for s in chart["series"]]
    assert keys == ["youngest_borrower_age", "current_loan_to_value", "current_outstanding_balance"]
    assert chart["xKey"] == "youngest_borrower_age"
    assert chart["source"]["nativeChartType"] == "bubble"


def test_scatter_maps_x_y_series_no_size():
    spec = {"chart_type": "scatter", "x": "current_valuation_amount", "y": "current_outstanding_balance"}
    resolved = {
        "current_valuation_amount": {"canonical_field": "current_valuation_amount", "role": "metric", "format": "currency"},
        "current_outstanding_balance": {"canonical_field": "current_outstanding_balance", "role": "metric", "format": "currency"},
    }
    data = [{"current_valuation_amount": 500000.0, "current_outstanding_balance": 184000.0}]
    resp = adapt_workflow_result(_workflow("scatter", "loan_level", data, spec, resolved))
    chart = next(a for a in resp["artifacts"] if a["type"] == "chart")
    assert [s["key"] for s in chart["series"]] == ["current_valuation_amount", "current_outstanding_balance"]


def test_heatmap_falls_back_to_table_with_figure_preserved():
    spec = {"chart_type": "heatmap", "dimensions": ["geographic_region_obligor", "ltv_bucket"]}
    resolved = {
        "geographic_region_obligor": {"canonical_field": "geographic_region_obligor", "role": "dimension", "format": "string"},
        "ltv_bucket": {"canonical_field": "ltv_bucket", "role": "dimension", "format": "string"},
    }
    data = [{"geographic_region_obligor": "London", "ltv_bucket": "30-40%", "current_outstanding_balance_sum": 50.0}]
    resp = adapt_workflow_result(_workflow("heatmap", "table", data, spec, resolved))
    # No chart artifact (not renderable); a table carries the figure + native type.
    assert "chart" not in _types(resp)
    table = next(a for a in resp["artifacts"] if a["type"] == "table")
    assert table["source"]["nativeChartType"] == "heatmap"
    assert table["source"]["figure"] is not None
    assert any("heatmap" in w for w in resp["warnings"])


def test_treemap_falls_back_to_table():
    spec = {"chart_type": "treemap", "hierarchy": ["geographic_region_obligor"]}
    resolved = {"geographic_region_obligor": {"canonical_field": "geographic_region_obligor", "role": "dimension", "format": "string"}}
    data = [{"geographic_region_obligor": "London", "current_outstanding_balance_sum": 184.0}]
    resp = adapt_workflow_result(_workflow("treemap", "table", data, spec, resolved))
    assert "chart" not in _types(resp)
    assert "table" in _types(resp)


def test_none_summary_maps_to_kpi():
    spec = {"chart_type": "none", "intent": "summary", "metric": "current_outstanding_balance"}
    resolved = {"current_outstanding_balance": {"canonical_field": "current_outstanding_balance", "role": "metric", "format": "currency"}}
    data = [{"current_outstanding_balance_sum": 842600000.0, "loan_count": 4318}]
    resp = adapt_workflow_result(_workflow("none", "summary", data, spec, resolved))
    assert _types(resp) == ["kpi"]
    kpi = resp["artifacts"][0]
    assert any("£" in k["value"] for k in kpi["kpis"])


def test_empty_result_produces_table_with_no_rows():
    spec = {"chart_type": "bar", "dimension": "geographic_region_obligor", "metric": "current_outstanding_balance"}
    resolved = {"geographic_region_obligor": {"canonical_field": "geographic_region_obligor", "role": "dimension", "format": "string"}}
    resp = adapt_workflow_result(_workflow("bar", "table", [], spec, resolved))
    # No data -> no chart artifact, but a (empty) table artifact is still present.
    assert "chart" not in _types(resp)
    assert "table" in _types(resp)
