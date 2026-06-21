"""
Tests for the MI Query Executor v1 (mi_agent/mi_query_executor.py).

Uses synthetic pandas data only — no real client data. Field keys are taken
from the locked v1 semantic registry; where helpful we discover keys from the
semantics by role rather than hard-coding.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from mi_agent.mi_query_executor import (
    MIQueryExecutionError,
    MIQueryResult,
    aggregate_series,
    execute_mi_query,
    resolve_weight_field,
    _resolve_group_column,
)
from mi_agent.mi_query_spec import MIQuerySpec
from mi_agent.mi_query_validator import load_mi_semantics

REPO_ROOT = Path(__file__).resolve().parents[2]
SEMANTICS_PATH = REPO_ROOT / "mi_agent" / "mi_semantics_field_registry.yaml"


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #


@pytest.fixture(scope="module")
def semantics():
    return load_mi_semantics(SEMANTICS_PATH)


@pytest.fixture
def df():
    """Synthetic canonical-shaped portfolio (30 loans)."""
    regions = ["North", "South", "East"]
    statuses = ["Performing", "Arrears", "Default"]
    brokers = ["Broker A", "Broker B"]
    rows = []
    for i in range(30):
        rows.append({
            "loan_identifier": f"L{i:04d}",
            "current_outstanding_balance": 100_000 + i * 5_000,
            "current_principal_balance": 95_000 + i * 5_000,
            "current_loan_to_value": 0.30 + (i % 5) * 0.05,   # fraction scale
            "indexed_loan_to_value": 0.28 + (i % 5) * 0.05,
            "current_interest_rate": 3.0 + (i % 4) * 0.5,
            "youngest_borrower_age": 55 + (i % 20),
            "number_of_days_in_arrears": (i % 6) * 15,
            "geographic_region_obligor": regions[i % 3],
            "broker_channel": brokers[i % 2],
            "account_status": statuses[i % 3],
            "origination_date": pd.Timestamp("2021-01-01") + pd.Timedelta(days=40 * i),
        })
    frame = pd.DataFrame(rows)
    # Introduce a few missing region values (for the exclusion test).
    frame.loc[[2, 9], "geographic_region_obligor"] = None
    return frame


def _exec(spec, df, semantics, **kw):
    return execute_mi_query(spec, df, semantics, **kw)


# --------------------------------------------------------------------------- #
# 1. bar: balance by region
# --------------------------------------------------------------------------- #


def test_bar_balance_by_region(df, semantics):
    spec = MIQuerySpec(intent="chart", chart_type="bar",
                       metric="current_outstanding_balance",
                       dimension="geographic_region_obligor", aggregation="sum")
    res = _exec(spec, df, semantics)
    assert res.result_type == "table"
    assert "geographic_region_obligor" in res.data.columns
    assert "current_outstanding_balance_sum" in res.data.columns
    # missing region rows excluded -> only the 3 real regions
    assert set(res.data["geographic_region_obligor"]) == {"North", "South", "East"}


# --------------------------------------------------------------------------- #
# 2. table: count by account_status
# --------------------------------------------------------------------------- #


def test_table_count_by_status(df, semantics):
    spec = MIQuerySpec(intent="table", dimension="account_status",
                       aggregation="count", chart_type="none")
    res = _exec(spec, df, semantics)
    assert res.result_type == "table"
    assert "count" in res.data.columns
    assert int(res.data["count"].sum()) == len(df)


# --------------------------------------------------------------------------- #
# 3. weighted average LTV by region
# --------------------------------------------------------------------------- #


def test_weighted_avg_ltv_by_region(df, semantics):
    spec = MIQuerySpec(intent="chart", chart_type="bar",
                       metric="current_loan_to_value",
                       dimension="geographic_region_obligor",
                       aggregation="weighted_avg")
    res = _exec(spec, df, semantics)
    col = "current_loan_to_value_weighted_avg"
    assert col in res.data.columns
    # hand-compute the weighted avg for North and compare
    north = df[df["geographic_region_obligor"] == "North"]
    w = north["current_outstanding_balance"]
    expected = (north["current_loan_to_value"] * w).sum() / w.sum()
    got = res.data.set_index("geographic_region_obligor").loc["North", col]
    assert got == pytest.approx(expected)
    assert res.metadata["balance_field_used"] == "current_outstanding_balance"


# --------------------------------------------------------------------------- #
# 4. line: balance by origination_date (monthly)
# --------------------------------------------------------------------------- #


def test_line_balance_by_origination_month(df, semantics):
    spec = MIQuerySpec(intent="chart", chart_type="line", x="origination_date",
                       metric="current_outstanding_balance", aggregation="sum")
    res = _exec(spec, df, semantics)
    assert res.result_type == "table"
    assert "origination_month" in res.data.columns
    periods = list(res.data["origination_month"])
    assert periods == sorted(periods)  # ascending
    assert all(len(p) == 7 and p[4] == "-" for p in periods)  # YYYY-MM


# --------------------------------------------------------------------------- #
# 5. bubble: LTV by age sized by balance
# --------------------------------------------------------------------------- #


def test_bubble_ltv_by_age_sized_by_balance(df, semantics):
    spec = MIQuerySpec(intent="chart", chart_type="bubble",
                       x="youngest_borrower_age", y="current_loan_to_value",
                       size="current_outstanding_balance", aggregation="loan_level")
    res = _exec(spec, df, semantics)
    assert res.result_type == "loan_level"
    assert set(res.data.columns) == {
        "youngest_borrower_age", "current_loan_to_value", "current_outstanding_balance"
    }


def test_duplicate_columns_raise_controlled_error_not_attributeerror(df, semantics):
    # A duplicated x/y/size column must yield a controlled MIDuplicateColumnError
    # (-> API validation failure), never a raw AttributeError 500 from
    # coerce_numeric receiving a DataFrame.
    from mi_agent.mi_query_executor import MIDuplicateColumnError
    dup = pd.concat([df, df[["current_loan_to_value"]]], axis=1)  # duplicate y
    assert dup.columns.duplicated().any()
    spec = MIQuerySpec(intent="chart", chart_type="bubble",
                       x="youngest_borrower_age", y="current_loan_to_value",
                       size="current_outstanding_balance", aggregation="loan_level")
    with pytest.raises(MIDuplicateColumnError) as exc:
        _exec(spec, dup, semantics)
    assert "current_loan_to_value" in exc.value.duplicate_columns
    assert any("current_loan_to_value" in a for a in exc.value.affected_fields)


# --------------------------------------------------------------------------- #
# 6. scatter: interest rate vs ltv
# --------------------------------------------------------------------------- #


def test_scatter_rate_vs_ltv(df, semantics):
    spec = MIQuerySpec(intent="chart", chart_type="scatter",
                       x="current_interest_rate", y="current_loan_to_value",
                       aggregation="loan_level")
    res = _exec(spec, df, semantics)
    assert res.result_type == "loan_level"
    assert list(res.data.columns) == ["current_interest_rate", "current_loan_to_value"]


# --------------------------------------------------------------------------- #
# 7. heatmap: weighted avg LTV by two dimensions
#    (age is a metric in v1, so we use two dimension-role fields, per registry)
# --------------------------------------------------------------------------- #


def test_heatmap_wavg_ltv_by_two_dimensions(df, semantics):
    spec = MIQuerySpec(intent="chart", chart_type="heatmap",
                       metric="current_loan_to_value",
                       dimensions=["geographic_region_obligor", "account_status"],
                       aggregation="weighted_avg")
    res = _exec(spec, df, semantics)
    assert res.result_type == "table"
    for c in ("geographic_region_obligor", "account_status",
              "current_loan_to_value_weighted_avg"):
        assert c in res.data.columns


# --------------------------------------------------------------------------- #
# 8. treemap: balance by region and broker
# --------------------------------------------------------------------------- #


def test_treemap_balance_by_region_and_broker(df, semantics):
    spec = MIQuerySpec(intent="chart", chart_type="treemap",
                       metric="current_outstanding_balance",
                       hierarchy=["geographic_region_obligor", "broker_channel"],
                       aggregation="sum")
    res = _exec(spec, df, semantics)
    assert res.result_type == "table"
    assert "concentration_pct" in res.data.columns
    for c in ("geographic_region_obligor", "broker_channel",
              "current_outstanding_balance_sum"):
        assert c in res.data.columns


# --------------------------------------------------------------------------- #
# 9. scalar filter
# --------------------------------------------------------------------------- #


def test_scalar_filter(df, semantics):
    spec = MIQuerySpec(intent="table", dimension="broker_channel",
                       aggregation="count", chart_type="none",
                       filters={"geographic_region_obligor": "North"})
    res = _exec(spec, df, semantics)
    expected = len(df[df["geographic_region_obligor"] == "North"])
    assert int(res.data["count"].sum()) == expected


# --------------------------------------------------------------------------- #
# 10. list filter
# --------------------------------------------------------------------------- #


def test_list_filter(df, semantics):
    spec = MIQuerySpec(intent="table", dimension="broker_channel",
                       aggregation="count", chart_type="none",
                       filters={"account_status": ["Performing", "Arrears"]})
    res = _exec(spec, df, semantics)
    expected = len(df[df["account_status"].isin(["Performing", "Arrears"])])
    assert int(res.data["count"].sum()) == expected


# --------------------------------------------------------------------------- #
# 11. top_n ranked by balance
# --------------------------------------------------------------------------- #


def test_top_n_ranked_by_balance(df, semantics):
    spec = MIQuerySpec(intent="chart", chart_type="bar",
                       metric="current_outstanding_balance",
                       dimension="account_status", aggregation="sum", top_n=2)
    res = _exec(spec, df, semantics)
    assert len(res.data) == 2
    # results are the two highest-balance statuses
    full = MIQuerySpec(intent="chart", chart_type="bar",
                       metric="current_outstanding_balance",
                       dimension="account_status", aggregation="sum")
    full_res = _exec(full, df, semantics)
    top2 = full_res.data.nlargest(2, "current_outstanding_balance_sum")
    assert set(res.data["account_status"]) == set(top2["account_status"])


# --------------------------------------------------------------------------- #
# 12. concentration_pct on grouped balance output
# --------------------------------------------------------------------------- #


def test_concentration_added(df, semantics):
    spec = MIQuerySpec(intent="chart", chart_type="bar",
                       metric="current_outstanding_balance",
                       dimension="geographic_region_obligor", aggregation="sum")
    res = _exec(spec, df, semantics)
    assert "concentration_pct" in res.data.columns
    assert res.data["concentration_pct"].sum() == pytest.approx(100.0)


# --------------------------------------------------------------------------- #
# 13. missing dimension values excluded + warning
# --------------------------------------------------------------------------- #


def test_missing_dimension_excluded_with_warning(df, semantics):
    spec = MIQuerySpec(intent="chart", chart_type="bar",
                       metric="current_outstanding_balance",
                       dimension="geographic_region_obligor", aggregation="sum")
    res = _exec(spec, df, semantics)
    assert any("missing/blank grouping" in w for w in res.warnings)
    # the two None regions are excluded
    assert "None" not in set(res.data["geographic_region_obligor"].astype(str))


# --------------------------------------------------------------------------- #
# 14. fails cleanly when a required canonical column is missing
# --------------------------------------------------------------------------- #


def test_missing_canonical_column_raises(df, semantics):
    broken = df.drop(columns=["current_outstanding_balance"])
    spec = MIQuerySpec(intent="chart", chart_type="bar",
                       metric="current_outstanding_balance",
                       dimension="geographic_region_obligor", aggregation="sum")
    with pytest.raises(MIQueryExecutionError):
        _exec(spec, broken, semantics)


# --------------------------------------------------------------------------- #
# 15. fails cleanly when validation fails
# --------------------------------------------------------------------------- #


def test_validation_failure_raises(df, semantics):
    spec = MIQuerySpec(intent="chart", chart_type="bar",
                       metric="not_a_real_semantic_field",
                       dimension="geographic_region_obligor", aggregation="sum")
    with pytest.raises(MIQueryExecutionError):
        _exec(spec, df, semantics)


# --------------------------------------------------------------------------- #
# 16. loan-level capping / sampling + warning
# --------------------------------------------------------------------------- #


def test_loan_level_capping(df, semantics):
    spec = MIQuerySpec(intent="chart", chart_type="bubble",
                       x="youngest_borrower_age", y="current_loan_to_value",
                       size="current_outstanding_balance", aggregation="loan_level")
    res = _exec(spec, df, semantics, max_loan_level_rows=10)
    assert res.row_count == 10
    assert res.metadata["loan_level_sampled"] is True
    assert res.metadata["loan_level_original_rows"] == 30
    assert res.metadata["sample_seed"] == 42
    assert any("capped" in w for w in res.warnings)
    # deterministic: same seed -> same rows
    res2 = _exec(spec, df, semantics, max_loan_level_rows=10)
    assert res.data.equals(res2.data)


# --------------------------------------------------------------------------- #
# 17. to_json works
# --------------------------------------------------------------------------- #


def test_result_to_json(df, semantics):
    spec = MIQuerySpec(intent="chart", chart_type="bar",
                       metric="current_outstanding_balance",
                       dimension="geographic_region_obligor", aggregation="sum")
    res = _exec(spec, df, semantics)
    payload = json.loads(res.to_json())
    assert payload["result_type"] == "table"
    assert isinstance(payload["data"], list) and payload["data"]
    assert payload["spec"]["metric"] == "current_outstanding_balance"
    assert "metadata" in payload and "warnings" in payload


# --------------------------------------------------------------------------- #
# 18. CLI smoke test
# --------------------------------------------------------------------------- #


def test_cli_smoke(df, semantics, tmp_path):
    from mi_agent.mi_query_executor import main
    data_path = tmp_path / "data.csv"
    df.to_csv(data_path, index=False)
    spec_path = tmp_path / "spec.json"
    spec = MIQuerySpec(intent="chart", chart_type="bar",
                       metric="current_outstanding_balance",
                       dimension="geographic_region_obligor", aggregation="sum")
    spec_path.write_text(spec.to_json(), encoding="utf-8")
    out_path = tmp_path / "result.csv"
    rc = main(["--semantics", str(SEMANTICS_PATH), "--spec", str(spec_path),
               "--data", str(data_path), "--out", str(out_path)])
    assert rc == 0
    assert out_path.exists()
    out_df = pd.read_csv(out_path)
    assert "current_outstanding_balance_sum" in out_df.columns


# --------------------------------------------------------------------------- #
# 19. scatter/bubble exclude loan identifiers by default
# --------------------------------------------------------------------------- #


def test_loan_level_excludes_identifiers(df, semantics):
    spec = MIQuerySpec(intent="chart", chart_type="bubble",
                       x="youngest_borrower_age", y="current_loan_to_value",
                       size="current_outstanding_balance", aggregation="loan_level")
    res = _exec(spec, df, semantics)
    assert "loan_identifier" not in res.data.columns
    assert res.metadata["identifiers_included"] is False


# --------------------------------------------------------------------------- #
# 20. weighted avg default weight + fallback
# --------------------------------------------------------------------------- #


def test_weight_field_default_and_fallback(df, semantics):
    ltv_entry = semantics["fields"]["current_loan_to_value"]
    spec = MIQuerySpec(metric="current_loan_to_value", aggregation="weighted_avg")

    # default: current_outstanding_balance present
    wf = resolve_weight_field(spec, ltv_entry, semantics, df.columns)
    assert wf == "current_outstanding_balance"

    # fallback: drop outstanding -> principal balance used
    cols_no_outstanding = [c for c in df.columns if c != "current_outstanding_balance"]
    wf2 = resolve_weight_field(spec, ltv_entry, semantics, cols_no_outstanding)
    assert wf2 == "current_principal_balance"

    # and the full execution still runs against the fallback frame
    spec_full = MIQuerySpec(intent="chart", chart_type="bar",
                            metric="current_loan_to_value",
                            dimension="geographic_region_obligor",
                            aggregation="weighted_avg")
    res = _exec(spec_full, df.drop(columns=["current_outstanding_balance"]), semantics)
    assert "current_loan_to_value_weighted_avg" in res.data.columns
    assert res.metadata["balance_field_used"] == "current_principal_balance"


# --------------------------------------------------------------------------- #
# Bucket-field reuse (Part 9) + aggregate_series unit checks
# --------------------------------------------------------------------------- #


def test_bucket_reuse_and_fallback(df, semantics):
    warnings = []
    # synthetic semantics entry with a bucket_field that exists in df
    fake = {"fields": {"x": {"canonical_field": "youngest_borrower_age",
                             "bucket_field": "age_bucket", "role": "metric"}}}
    work = df.copy()
    work["age_bucket"] = "65-70"
    col = _resolve_group_column("x", fake, work, warnings, use_bucket=True)
    assert col == "age_bucket"
    assert warnings == []

    # bucket named but absent -> falls back to canonical + warning
    warnings2 = []
    col2 = _resolve_group_column("x", fake, df, warnings2, use_bucket=True)
    assert col2 == "youngest_borrower_age"
    assert any("bucket field" in w for w in warnings2)


def test_aggregate_series_basic(df):
    assert aggregate_series(df, "current_outstanding_balance", "sum") == pytest.approx(
        df["current_outstanding_balance"].sum())
    assert aggregate_series(df, None, "count") == len(df)
    assert aggregate_series(df, "account_status", "count_distinct") == 3
    assert aggregate_series(df, "current_loan_to_value", "weighted_avg",
                            weight_col="current_outstanding_balance") == pytest.approx(
        (df["current_loan_to_value"] * df["current_outstanding_balance"]).sum()
        / df["current_outstanding_balance"].sum())
    with pytest.raises(MIQueryExecutionError):
        aggregate_series(df, "current_loan_to_value", "distribution")
