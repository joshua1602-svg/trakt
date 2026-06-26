"""
Backend drill-through filter support: a caller (e.g. the UI drilling into one
region/broker/year/SPV/stage) passes ``extra_filters`` into run_mi_agent_query.
The filters must be applied BEFORE aggregation against the FULL dataset, and an
unknown filter field must be rejected as a controlled failure (never a 500).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from mi_agent.mi_agent_workflow import run_mi_agent_query
from mi_agent.mi_query_validator import load_mi_semantics

REPO_ROOT = Path(__file__).resolve().parents[2]
SEMANTICS_PATH = REPO_ROOT / "mi_agent" / "mi_semantics_field_registry.yaml"


@pytest.fixture(scope="module")
def semantics():
    return load_mi_semantics(SEMANTICS_PATH)


@pytest.fixture
def df():
    regions = ["North", "South", "East"]
    return pd.DataFrame([{
        "current_outstanding_balance": 100_000 + i * 1000,
        "geographic_region_obligor": regions[i % 3],
    } for i in range(30)])


def _dim_values(result, column):
    return set(result["query_result"].data[column].astype(str))


def test_baseline_has_all_regions(df, semantics):
    res = run_mi_agent_query("Show balance by region", df, semantics)
    assert res["ok"], res.get("error")
    dim = res["spec"]["dimension"]
    assert _dim_values(res, dim) == {"North", "South", "East"}


def test_drill_filter_narrows_to_one_region(df, semantics):
    res = run_mi_agent_query(
        "Show balance by region", df, semantics,
        extra_filters={"geographic_region_obligor": "South"})
    assert res["ok"], res.get("error")
    dim = res["spec"]["dimension"]
    # Filtered BEFORE aggregation: only the selected region survives.
    assert _dim_values(res, dim) == {"South"}
    # The drill filter is reflected on the echoed spec + a warning is surfaced.
    assert res["spec"]["filters"].get("geographic_region_obligor") == "South"
    assert any("drill-through filters applied" in w for w in res["warnings"])


def test_drill_filter_uses_full_dataset_not_displayed_rows(df, semantics):
    # South rows are i % 3 == 1 → balances 101k, 104k, ... (10 rows).
    expected = sum(100_000 + i * 1000 for i in range(30) if i % 3 == 1)
    res = run_mi_agent_query(
        "Show total balance by region", df, semantics,
        extra_filters={"geographic_region_obligor": "South"})
    assert res["ok"], res.get("error")
    data = res["query_result"].data
    metric_col = [c for c in data.columns if c != res["spec"]["dimension"]][0]
    assert float(data[metric_col].iloc[0]) == pytest.approx(expected)


def test_invalid_filter_field_rejected_safely(df, semantics):
    res = run_mi_agent_query(
        "Show balance by region", df, semantics,
        extra_filters={"not_a_real_field": "x"})
    # Controlled failure (ok False), never an exception / 500.
    assert res["ok"] is False
    blob = (res.get("error") or "") + str(res.get("validation") or "")
    assert "not_a_real_field" in blob


def test_no_extra_filters_is_unchanged(df, semantics):
    base = run_mi_agent_query("Show balance by region", df, semantics)
    none = run_mi_agent_query("Show balance by region", df, semantics, extra_filters=None)
    assert base["spec"]["filters"] == none["spec"]["filters"]
