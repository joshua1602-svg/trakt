"""Lightweight API tests for the MI Agent API.

These exercise the real MI Agent flow against the synthetic demo portfolio.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from mi_agent_api.app import app

client = TestClient(app)


def test_dataframe_has_materialised_bucket_dimensions():
    # The API must apply the existing bucketing engine so registry-named bucket
    # dimensions resolve in run_mi_agent_query (not only in Streamlit).
    from mi_agent_api.data_source import get_dataframe

    cols = set(get_dataframe().columns)
    assert {"age_bucket", "ltv_bucket", "ticket_bucket"} <= cols


def test_query_bucket_heatmap_resolves_natively():
    r = client.post(
        "/mi/query",
        json={"question": "Show LTV by age bucket and region as a heatmap", "asOfDate": "2026-05-31"},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] is True, body.get("validation")
    chart = next((a for a in body["artifacts"] if a["type"] == "chart"), None)
    assert chart is not None
    assert chart["chartType"] == "heatmap"
    # Native grid keys populated -> renders via the React HeatmapArtifactView.
    assert chart["xKey"] == "age_bucket"
    assert chart["yKey"]
    assert chart["valueKey"]


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] is True
    assert body["service"] == "mi_agent_api"


def test_catalogue_exposes_real_semantic_layer():
    r = client.get("/mi/catalogue")
    assert r.status_code == 200
    cat = r.json()
    # Real registry has 43 dimensions / 37 measures.
    assert len(cat["dimensions"]) > 30
    assert len(cat["measures"]) > 30
    # Chart types are the MIQuerySpec set — no legacy area/waterfall.
    assert set(cat["chart_types"]) == {"bar", "line", "scatter", "bubble", "heatmap", "treemap"}
    assert "area" not in cat["chart_types"]
    # Spot-check known registry keys.
    dim_keys = {d["key"] for d in cat["dimensions"]}
    assert "geographic_region_obligor" in dim_keys
    measure_keys = {m["key"] for m in cat["measures"]}
    assert "current_outstanding_balance" in measure_keys
    assert sorted(cat["aggregations"]) == sorted(
        ["sum", "avg", "weighted_avg", "count", "count_distinct", "median", "distribution", "loan_level", "balance_sum"]
    )


def test_query_known_question_returns_artifacts():
    r = client.post(
        "/mi/query",
        json={"question": "Show balance by region", "portfolioId": "erm-uk-master", "asOfDate": "2026-05-31"},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] is True
    assert body["spec"]["chart_type"] == "bar"
    assert body["metadata"]["engine"] == "mi_agent"
    assert body["metadata"]["mock"] is False
    types = {a["type"] for a in body["artifacts"]}
    # A bar query yields a chart + a table artifact.
    assert "chart" in types
    assert "table" in types
    chart = next(a for a in body["artifacts"] if a["type"] == "chart")
    assert chart["chartType"] == "bar"
    assert chart["xKey"]
    assert chart["rows"]
    # Lineage + mock disclosure present.
    assert chart["mock"] is False
    assert chart["source"]["engine"] == "mi_agent.workflow"


def test_invalid_spec_returns_structured_validation_failure():
    r = client.post("/mi/query", json={"question": "balance by nonexistent_dimension"})
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] is False
    assert body["validation"]["ok"] is False
    assert body["validation"]["errors"]
    # The failure is surfaced as a validation artifact for the UI.
    assert any(a["type"] == "validation" for a in body["artifacts"])


def test_empty_question_rejected():
    r = client.post("/mi/query", json={"question": ""})
    assert r.status_code == 422  # pydantic min_length


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
