"""
Tests for the derived bucket semantic fields added in v0.2.2.

These fields (age_bucket, ltv_bucket, ticket_bucket, vintage_year,
arrears_bucket, term_bucket, original_ltv_bucket, maturity_year) are derived
at the analytics layer (analytics/mi_prep.py::add_buckets) and are registered
here as first-class semantic dimensions so MI specs can group / heatmap /
treemap by them directly.

The executor logic was NOT changed for this work — these tests confirm that
the existing executor recognises the bucket fields if their columns are
present in the dataframe.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from mi_agent.mi_query_executor import MIQueryExecutionError, execute_mi_query
from mi_agent.mi_query_spec import MIQuerySpec
from mi_agent.mi_query_validator import load_mi_semantics, validate_mi_query

REPO_ROOT = Path(__file__).resolve().parents[2]
SEMANTICS_PATH = REPO_ROOT / "mi_agent" / "mi_semantics_field_registry.yaml"

DERIVED_BUCKETS = [
    "age_bucket",
    "ltv_bucket",
    "original_ltv_bucket",
    "ticket_bucket",
    "arrears_bucket",
    "term_bucket",
    "vintage_year",
    "maturity_year",
]


@pytest.fixture(scope="module")
def semantics():
    return load_mi_semantics(SEMANTICS_PATH)


@pytest.fixture
def df():
    """Synthetic canonical-shaped portfolio with pre-derived bucket columns
    (as analytics/mi_prep.py would have produced)."""
    regions = ["North", "South", "East"]
    age_bands = ["55-60", "60-65", "65-70", "70-75"]
    ltv_bands = ["20-30%", "30-40%", "40-50%", "50-60%"]
    ticket = ["<50k", "50-100k", "100-150k", "150-200k"]
    rows = []
    for i in range(40):
        rows.append({
            "loan_identifier": f"L{i:04d}",
            "current_outstanding_balance": 100_000 + i * 5_000,
            "current_principal_balance": 95_000 + i * 5_000,
            "current_loan_to_value": 0.30 + (i % 5) * 0.05,
            "youngest_borrower_age": 55 + (i % 20),
            "geographic_region_obligor": regions[i % 3],
            "origination_date": pd.Timestamp("2021-01-01") + pd.Timedelta(days=40 * i),
            # Pre-derived bucket columns (as analytics/mi_prep.py would emit):
            "age_bucket": age_bands[i % 4],
            "ltv_bucket": ltv_bands[i % 4],
            "ticket_bucket": ticket[i % 4],
            "vintage_year": str(2020 + (i % 5)),
        })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# 1. Registry contents
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("key", DERIVED_BUCKETS)
def test_bucket_field_present_in_registry(semantics, key):
    assert key in semantics["fields"], f"{key} missing from semantic registry"


@pytest.mark.parametrize("key", DERIVED_BUCKETS)
def test_bucket_field_metadata(semantics, key):
    entry = semantics["fields"][key]
    assert entry["mi_tier"] == "core"
    assert entry["role"] == "dimension"
    assert entry["format"] == "string"
    assert entry["chartable"] is True
    assert entry["source_criteria"] == ["derived_bucket"]
    assert entry.get("derived") is True
    assert entry.get("derived_from"), f"{key} should record what it is derived from"
    assert entry["business_name"], f"{key} needs a business_name"
    assert entry["business_description"], f"{key} needs a business_description"
    assert isinstance(entry["synonyms"], list) and entry["synonyms"], \
        f"{key} should have at least one synonym"
    roles = set(entry["allowed_chart_roles"])
    assert {"x", "group", "filter", "color"}.issubset(roles)
    assert entry["default_chart_role"] == "x"
    assert entry["allowed_aggregations"] == ["count", "balance_sum"]
    assert entry["default_aggregation"] == "count"
    # As a dimension, a bucket itself should not nominate another bucket.
    assert entry["bucket_field"] in (None,)


def test_registry_field_count_and_tiers(semantics):
    m = semantics["metadata"]
    assert m["derived_field_count"] == len(DERIVED_BUCKETS)
    assert m["field_count"] == m["core_field_count"] + m["extended_field_count"]
    # Sanity: 61 (v0.2.1 curated set) + 8 derived buckets + collateral_geography
    # (readable region display field added for MI Region) == 70.
    assert m["field_count"] == 70
    assert m["core_field_count"] == 46
    assert "derived bucket semantic fields added" in (m.get("cleanup_notes") or [])


def test_bucket_business_name_resolves_via_find_field(semantics):
    """The parser's existing keyword matcher (key / display_name / business_name)
    should resolve a bucket's *business_name* to the bucket key.  Full
    synonym-driven resolution is a v2 item and is intentionally not covered here.
    """
    from mi_agent.llm_query_parser import find_field
    # business_name "Age Bucket" matches "age bucket"
    assert find_field(semantics, role="dimension",
                      keywords=("age bucket",)) == "age_bucket"
    # business_name "LTV Bucket" matches "ltv bucket"
    assert find_field(semantics, role="dimension",
                      keywords=("ltv bucket",)) == "ltv_bucket"
    # key "vintage_year" matches "vintage_year"
    assert find_field(semantics, role="dimension",
                      keywords=("vintage_year",)) == "vintage_year"
    # business_name "Ticket Size" matches "ticket size"
    assert find_field(semantics, role="dimension",
                      keywords=("ticket size",)) == "ticket_bucket"


# --------------------------------------------------------------------------- #
# 2. Validator accepts bucket fields in heatmap / treemap / bar
# --------------------------------------------------------------------------- #


def test_validator_accepts_heatmap_with_age_bucket_and_region(semantics):
    spec = MIQuerySpec(intent="chart", chart_type="heatmap",
                       metric="current_loan_to_value",
                       dimensions=["age_bucket", "geographic_region_obligor"],
                       aggregation="weighted_avg")
    vr = validate_mi_query(spec, semantics)
    assert vr.ok, vr.errors


def test_validator_accepts_bar_balance_by_ticket_bucket(semantics):
    spec = MIQuerySpec(intent="chart", chart_type="bar",
                       metric="current_outstanding_balance",
                       dimension="ticket_bucket", aggregation="sum")
    vr = validate_mi_query(spec, semantics)
    assert vr.ok, vr.errors


# --------------------------------------------------------------------------- #
# 3. Executor recognises bucket columns when present in the dataframe
# --------------------------------------------------------------------------- #


def test_executor_heatmap_ltv_by_age_bucket_and_region(df, semantics):
    spec = MIQuerySpec(intent="chart", chart_type="heatmap",
                       metric="current_loan_to_value",
                       dimensions=["age_bucket", "geographic_region_obligor"],
                       aggregation="weighted_avg")
    res = execute_mi_query(spec, df, semantics)
    assert res.result_type == "table"
    for c in ("age_bucket", "geographic_region_obligor",
              "current_loan_to_value_weighted_avg"):
        assert c in res.data.columns
    # at least one row per realised (age_band, region) combination
    assert res.row_count > 0
    assert set(res.data["age_bucket"]).issubset(set(df["age_bucket"].astype(str)))


def test_executor_treemap_balance_by_ticket_and_region(df, semantics):
    spec = MIQuerySpec(intent="chart", chart_type="treemap",
                       metric="current_outstanding_balance",
                       hierarchy=["ticket_bucket", "geographic_region_obligor"],
                       aggregation="sum")
    res = execute_mi_query(spec, df, semantics)
    assert res.result_type == "table"
    assert "concentration_pct" in res.data.columns
    assert "ticket_bucket" in res.data.columns
    # total share sums to ~100
    assert res.data["concentration_pct"].sum() == pytest.approx(100.0)


def test_executor_bar_count_by_vintage_year_sorted(df, semantics):
    spec = MIQuerySpec(intent="chart", chart_type="bar",
                       dimension="vintage_year", aggregation="count")
    res = execute_mi_query(spec, df, semantics)
    assert "vintage_year" in res.data.columns
    assert "count" in res.data.columns
    assert int(res.data["count"].sum()) == len(df)


# --------------------------------------------------------------------------- #
# 4. Executor fails cleanly when a bucket column is missing from the dataframe
# --------------------------------------------------------------------------- #


def test_executor_missing_bucket_column_raises(df, semantics):
    broken = df.drop(columns=["age_bucket"])
    spec = MIQuerySpec(intent="chart", chart_type="heatmap",
                       metric="current_loan_to_value",
                       dimensions=["age_bucket", "geographic_region_obligor"],
                       aggregation="weighted_avg")
    with pytest.raises(MIQueryExecutionError):
        execute_mi_query(spec, broken, semantics)


# --------------------------------------------------------------------------- #
# 5. The metric-side bucket_field hint still routes to the bucket column when
#    available (executor's existing use_bucket=True path).
# --------------------------------------------------------------------------- #


def test_use_bucket_hint_still_works_via_metric_entry(df, semantics):
    # Even though the spec references the raw metric and a dimension, the
    # heatmap path passes use_bucket=True to dimensions, not to the metric.
    # This sanity-check just confirms the existing behaviour still holds for
    # legacy specs that name only the raw dimension.
    spec = MIQuerySpec(intent="chart", chart_type="treemap",
                       metric="current_outstanding_balance",
                       hierarchy=["geographic_region_obligor"],
                       aggregation="sum")
    res = execute_mi_query(spec, df, semantics)
    assert res.result_type == "table"
    assert "geographic_region_obligor" in res.data.columns
