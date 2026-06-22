#!/usr/bin/env python3
"""mi_agent/tests/test_mi_ranking_matrix.py

Parser + executor hardening for ranked queries, two-dimensional grouped (matrix /
heatmap) queries, and multi-filter counts. These EXTEND the existing deterministic
parser / MIQuerySpec / executor — no parallel parser or executor is introduced.

Covered behaviours:
  * `largest loan balance` -> loan-level ranking TABLE sorted desc (not a bar).
  * `largest balance by LTV` / `... by region` -> grouped ranking bar.
  * `balance by ltv by region` -> heatmap/matrix (two dims), NOT a loan-level
    bubble, with no duplicate-`current_outstanding_balance` failure.
  * `balance by region by ltv` -> same matrix semantics.
  * multi-filter count applies BOTH an age threshold and a region value.
  * heatmap validation fails gracefully when a dimension has no values.
  * existing simple queries (bar / bubble / single-filter count) still work.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from mi_agent.llm_query_parser import _deterministic_parse
from mi_agent.mi_agent_workflow import run_mi_agent_query
from mi_agent.mi_query_validator import load_mi_semantics

_SEMANTICS = _REPO_ROOT / "mi_agent" / "mi_semantics_field_registry.yaml"


@pytest.fixture(scope="module")
def semantics():
    return load_mi_semantics(_SEMANTICS)


@pytest.fixture(scope="module")
def df():
    """A prepared-funded-shaped frame with LTV / age / region / balance buckets."""
    from mi_agent_api.funded_prep import prepare_funded_mi_dataset
    n = 60
    raw = pd.DataFrame({
        "loan_identifier": [760000 + i for i in range(n)],
        "current_outstanding_balance": [100000.0 + i * 2500 for i in range(n)],
        "current_valuation_amount": [250000.0 + i * 2000 for i in range(n)],
        "current_interest_rate": [3.0 + (i % 6) * 0.4 for i in range(n)],
        "current_principal_balance": [100000.0 + i * 2500 for i in range(n)],
        "origination_date": ["2020-06-15"] * n,
        "reporting_date": ["2025-11-30"] * n,
        "geographic_region_obligor": (["South West", "London", "Wales", "North East"] * (n // 4 + 1))[:n],
        "youngest_borrower_age": [55 + (i % 30) for i in range(n)],
        "exposure_currency_denomination": ["GBP"] * n,
    })
    prepared, _report = prepare_funded_mi_dataset(raw)
    return prepared


def _cols(df):
    return set(df.columns)


def _run(df, q):
    return run_mi_agent_query(q, df, str(_SEMANTICS), parser_mode="deterministic")


# --------------------------------------------------------------------------- #
# B. Ranking grammar
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("q", [
    "largest loan balance",
    "largest loans by balance",
    "top 10 loans by balance",
    "largest current outstanding balance",
])
def test_loan_level_ranking_table(df, semantics, q):
    spec, _ = _deterministic_parse(q, semantics, available_columns=_cols(df))
    assert spec.intent == "table"
    assert spec.ranking_mode == "loan_level"
    assert spec.metric == "current_outstanding_balance"
    assert spec.sort_direction == "desc"

    res = _run(df, q)
    assert res["ok"], res.get("error")
    qr = res["query_result"].to_dict()
    rows = qr["data"]
    assert qr["result_type"] == "table"
    assert 0 < len(rows) <= 10
    # sorted descending by outstanding balance, and identifier exposed.
    bals = [r["current_outstanding_balance"] for r in rows]
    assert bals == sorted(bals, reverse=True)
    assert "loan_identifier" in rows[0]


def test_largest_balance_by_ltv_is_grouped_bar(df, semantics):
    spec, _ = _deterministic_parse("largest balance by LTV", semantics,
                                   available_columns=_cols(df))
    assert spec.chart_type == "bar"
    assert spec.metric == "current_outstanding_balance"
    assert spec.dimension == "ltv_bucket"
    assert spec.aggregation == "sum"
    res = _run(df, "largest balance by LTV")
    assert res["ok"], res.get("error")
    rows = res["query_result"].to_dict()["data"]
    vals = [r["current_outstanding_balance_sum"] for r in rows]
    assert vals == sorted(vals, reverse=True)  # ranked descending


def test_largest_balance_by_region_is_grouped_bar(df, semantics):
    spec, _ = _deterministic_parse("largest balance by region", semantics,
                                   available_columns=_cols(df))
    assert spec.chart_type == "bar"
    assert spec.dimension == "geographic_region_obligor"
    assert spec.metric == "current_outstanding_balance"


def test_top_n_brokers_still_bar_with_top_n(df, semantics):
    # Regression: the existing "top 10 <dim> by balance" grouped ranking is kept.
    spec, _ = _deterministic_parse("show top 10 brokers by balance", semantics)
    assert spec.chart_type == "bar"
    assert spec.dimension == "broker_channel"
    assert spec.top_n == 10
    assert spec.aggregation == "sum"


# --------------------------------------------------------------------------- #
# C/D. Two-dimensional grouped -> heatmap/matrix (NOT bubble)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("q", [
    "balance by ltv by region",
    "balance by region by ltv",
    "balance by ltv and region",
    "current outstanding balance by ltv bucket and geographic region",
])
def test_two_dim_grouped_is_heatmap(df, semantics, q):
    spec, _ = _deterministic_parse(q, semantics, available_columns=_cols(df))
    assert spec.chart_type == "heatmap", q
    assert spec.metric == "current_outstanding_balance"
    assert set(spec.dimensions) == {"ltv_bucket", "geographic_region_obligor"}
    # explicitly NOT a loan-level bubble
    assert spec.x is None and spec.y is None and spec.size is None


def test_heatmap_executes_without_duplicate_column_error(df):
    # The exact regression: 'balance by ltv by region' must not enter the bubble
    # path and must not fail with duplicate current_outstanding_balance.
    res = _run(df, "balance by ltv by region")
    assert res["ok"], res.get("error")
    assert res["error"] is None
    val = res.get("validation") or {}
    assert not any("duplicate" in str(e).lower() for e in val.get("errors", []))
    qr = res["query_result"].to_dict()
    cols = set(qr["data"][0].keys())
    assert {"ltv_bucket", "geographic_region_obligor"} <= cols
    assert "current_outstanding_balance_sum" in cols


def test_loan_count_by_two_dims_is_heatmap_count(df, semantics):
    spec, _ = _deterministic_parse("loan count by ltv by region", semantics,
                                   available_columns=_cols(df))
    assert spec.chart_type == "heatmap"
    assert spec.aggregation == "count"


def test_balance_by_ltv_by_age_stays_bubble(df, semantics):
    # Two NUMERIC axes remain a loan-level bubble (size = balance), distinct cols.
    spec, _ = _deterministic_parse("balance by ltv by age", semantics,
                                   available_columns=_cols(df))
    assert spec.chart_type == "bubble"
    assert spec.x and spec.y and spec.size
    assert len({spec.x, spec.y, spec.size}) == 3  # no duplicate role column
    res = _run(df, "balance by ltv by age")
    assert res["ok"], res.get("error")


def test_average_ltv_two_dims_is_weighted_avg_heatmap(df, semantics):
    spec, _ = _deterministic_parse("average ltv by region by age bucket", semantics,
                                   available_columns=_cols(df))
    assert spec.chart_type == "heatmap"
    assert spec.metric == "current_loan_to_value"
    assert spec.aggregation == "weighted_avg"
    assert spec.weight_field  # weighted_avg must carry a weight to validate
    res = _run(df, "average ltv by region by age bucket")
    assert res["ok"], res.get("error")


# --------------------------------------------------------------------------- #
# F. Multi-filter count
# --------------------------------------------------------------------------- #
def test_multi_filter_count_applies_both(df, semantics):
    q = ("how many loans with youngest age more than 70 and "
         "geographic region south west")
    spec, _ = _deterministic_parse(q, semantics, available_columns=_cols(df))
    assert spec.intent == "summary"
    assert spec.aggregation == "count"
    assert spec.filters["youngest_borrower_age"] == {"op": "gt", "value": 70.0}
    assert spec.filters["geographic_region_obligor"] == "South West"

    res = _run(df, q)
    assert res["ok"], res.get("error")
    row = res["query_result"].to_dict()["data"][0]
    # cross-check against a manual filter (case-insensitive region match).
    expect = df[(pd.to_numeric(df["youngest_borrower_age"]) > 70)
                & (df["geographic_region_obligor"].str.casefold() == "south west")]
    assert row["loan_count"] == len(expect)
    # balance for the filtered rows is also surfaced.
    assert any("balance" in k.lower() for k in row)


def test_single_filter_count_still_works(df, semantics):
    spec, _ = _deterministic_parse(
        "how many loans with youngest age more than 70", semantics,
        available_columns=_cols(df))
    assert spec.aggregation == "count"
    assert spec.filters == {"youngest_borrower_age": {"op": "gt", "value": 70.0}}


# --------------------------------------------------------------------------- #
# G. Validation stays data-aware (no raw 500)
# --------------------------------------------------------------------------- #
def test_heatmap_missing_dimension_fails_gracefully(df):
    # Region present, but request a dimension with no values in this data.
    no_region = df.drop(columns=["geographic_region_obligor"])
    res = _run(no_region, "balance by ltv by region")
    assert res["ok"] is False
    assert res["error"]  # controlled failure, not an exception
    val = res.get("validation") or {}
    assert val.get("ok") is False


def test_simple_bar_still_works(df, semantics):
    spec, _ = _deterministic_parse("current outstanding balance by ltv bucket",
                                   semantics, available_columns=_cols(df))
    assert spec.chart_type == "bar"
    assert spec.dimension == "ltv_bucket"
    assert spec.metric == "current_outstanding_balance"
    res = _run(df, "current outstanding balance by ltv bucket")
    assert res["ok"], res.get("error")


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
