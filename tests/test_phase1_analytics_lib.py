"""Phase 1 — shared analytics library tests.

Pure-function tests over small synthetic DataFrames. They prove the bucket
engine, stratification, concentration and cohort foundations behave correctly
and degrade into *structured issues* (not crashes) on missing/invalid data.
No imports from the legacy ``analytics/`` Streamlit app.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from analytics_lib import (
    add_cohort_period,
    cohort_table,
    group_shares,
    limit_usage,
    load_bucket_config,
    materialise_buckets,
    months_on_book,
    normalise_scale,
    rag_status,
    stratify,
    top_n_concentration,
)
from analytics_lib import buckets as B

REPO_ROOT = Path(__file__).resolve().parents[1]

REQUIRED_BUCKETS = {
    "ltv_bucket", "borrower_age_bucket", "youngest_borrower_age_bucket",
    "interest_rate_bucket", "pd_bucket", "lgd_bucket", "ead_bucket",
    "balance_band", "time_on_book_bucket",
}


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #


@pytest.fixture(scope="module")
def bucket_config():
    return load_bucket_config()


@pytest.fixture
def full_df():
    """A loan-level frame carrying every bucket source field (decimal scales)."""
    return pd.DataFrame({
        "loan_id": [f"L{i}" for i in range(6)],
        "current_loan_to_value": [0.15, 0.25, 0.55, 0.95, 1.05, 0.45],
        "youngest_borrower_age": [50, 57, 62, 72, 90, 68],
        "current_interest_rate": [0.02, 0.045, 0.08, 0.035, 0.061, 0.049],
        "probability_of_default": [0.001, 0.003, 0.05, 0.30, 0.012, 0.07],
        "loss_given_default": [0.05, 0.15, 0.35, 0.80, 0.45, 0.25],
        "exposure_at_default": [25_000, 75_000, 250_000, 2_000_000, 120_000, 480_000],
        "current_outstanding_balance": [25_000, 75_000, 250_000, 2_000_000, 120_000, 480_000],
        "months_on_book": [3, 18, 200, 30, 70, 9],
        "region": ["North", "North", "South", "East", "South", "North"],
    })


# --------------------------------------------------------------------------- #
# 1. Bucket config loads
# --------------------------------------------------------------------------- #


def test_bucket_config_loads(bucket_config):
    assert REQUIRED_BUCKETS.issubset(set(bucket_config))
    for spec in bucket_config.values():
        assert "source_field" in spec and "edges" in spec and "labels" in spec


# --------------------------------------------------------------------------- #
# 2. Every declared bucket applies, or reports a clear unavailable-field issue
# --------------------------------------------------------------------------- #


def test_every_bucket_applies_on_full_frame(full_df, bucket_config):
    out, issues, applied = materialise_buckets(full_df, bucket_config)
    # Every bucket produced a column.
    for key in bucket_config:
        assert applied[key] == key, f"{key} did not materialise"
        assert key in out.columns
    # No ERROR-level issues on a complete frame.
    errors = [i for i in issues if i["severity"] == B.ERROR]
    assert not errors, errors


def test_missing_source_fields_report_unavailable(bucket_config):
    empty = pd.DataFrame({"unrelated": [1, 2, 3]})
    out, issues, applied = materialise_buckets(empty, bucket_config)
    for key in bucket_config:
        assert applied[key] is None
    codes = {i["code"] for i in issues}
    assert codes == {B.UNAVAILABLE_FIELD}
    # One unavailable issue per declared bucket.
    assert sum(i["code"] == B.UNAVAILABLE_FIELD for i in issues) == len(bucket_config)


def test_bucket_values_are_correct(full_df, bucket_config):
    out, _, _ = materialise_buckets(full_df, bucket_config)
    assert list(out["ltv_bucket"]) == [
        "<20%", "20-30%", "50-60%", "90-100%", ">=100%", "40-50%"]
    assert list(out["borrower_age_bucket"]) == [
        "<55", "55-60", "60-65", "70-75", "85+", "65-70"]
    assert list(out["balance_band"]) == [
        "<50k", "50-100k", "200-300k", ">=1m", "100-150k", "300-500k"]
    assert list(out["time_on_book_bucket"]) == [
        "0-6m", "1-2y", "10y+", "2-3y", "5-10y", "6-12m"]


# --------------------------------------------------------------------------- #
# 3. Decimal/percent normalisation
# --------------------------------------------------------------------------- #


def test_normalise_decimal_fraction_from_percent():
    s = pd.Series([15.0, 25.0, 55.0, 95.0])
    out, note = normalise_scale(s, "decimal_fraction")
    assert note is not None
    assert list(out) == [0.15, 0.25, 0.55, 0.95]


def test_normalise_percent_from_fraction():
    s = pd.Series([0.02, 0.045, 0.08])
    out, note = normalise_scale(s, "percent")
    assert note is not None
    assert out.round(3).tolist() == [2.0, 4.5, 8.0]


def test_normalise_noop_when_already_canonical():
    s = pd.Series([0.15, 0.25, 0.95])
    out, note = normalise_scale(s, "decimal_fraction")
    assert note is None
    assert list(out) == [0.15, 0.25, 0.95]


def test_ltv_percent_input_buckets_same_as_decimal(bucket_config):
    pct = pd.DataFrame({"current_loan_to_value": [15.0, 55.0, 105.0]})
    out, issues, applied = materialise_buckets(pct, bucket_config, ["ltv_bucket"])
    assert list(out["ltv_bucket"]) == ["<20%", "50-60%", ">=100%"]
    assert any(i["code"] == B.SCALE_NORMALISED for i in issues)


# --------------------------------------------------------------------------- #
# 4. Invalid numeric vs out-of-range are distinguished
# --------------------------------------------------------------------------- #


def test_invalid_numeric_recorded_and_not_silently_coerced(bucket_config):
    df = pd.DataFrame({"current_loan_to_value": [0.5, "n/a", 0.9]})
    out, issues, _ = materialise_buckets(df, bucket_config, ["ltv_bucket"])
    invalid = [i for i in issues if i["code"] == B.INVALID_NUMERIC]
    assert len(invalid) == 1 and invalid[0]["count"] == 1
    assert pd.isna(out["ltv_bucket"].iloc[1])


def test_out_of_range_recorded(bucket_config):
    df = pd.DataFrame({"youngest_borrower_age": [50, 300]})  # 300 >= cap 200
    out, issues, _ = materialise_buckets(df, bucket_config, ["borrower_age_bucket"])
    oor = [i for i in issues if i["code"] == B.OUT_OF_RANGE]
    assert len(oor) == 1 and oor[0]["count"] == 1
    assert pd.isna(out["borrower_age_bucket"].iloc[1])


# --------------------------------------------------------------------------- #
# 5. Stratification: counts, balances, shares
# --------------------------------------------------------------------------- #


def test_stratify_counts_balances_shares(full_df):
    table = stratify(full_df, "region", "current_outstanding_balance",
                     loan_id_col="loan_id")
    by_region = table.set_index("region")
    assert int(by_region.loc["North", "loan_count"]) == 3
    assert float(by_region.loc["North", "balance_sum"]) == 25_000 + 75_000 + 480_000
    assert pytest.approx(table["balance_share"].sum(), abs=1e-9) == 1.0
    # Deterministic: descending balance_sum.
    assert list(table["balance_sum"]) == sorted(table["balance_sum"], reverse=True)


def test_stratify_unknown_bucket_handling():
    df = pd.DataFrame({
        "grade": ["A", None, "B", None],
        "bal": [10.0, 20.0, 30.0, 40.0],
    })
    table = stratify(df, "grade", "bal")
    assert "Unknown" in set(table["grade"])
    assert int(table.set_index("grade").loc["Unknown", "loan_count"]) == 2


def test_stratify_weighted_metric(full_df):
    table = stratify(full_df, "region", "current_outstanding_balance",
                     weighted_metrics=["current_loan_to_value"])
    assert "current_loan_to_value_weighted_avg" in table.columns
    # North: LTVs 0.15,0.25,0.45 weighted by 25k,75k,480k.
    w = (0.15 * 25_000 + 0.25 * 75_000 + 0.45 * 480_000) / (25_000 + 75_000 + 480_000)
    got = table.set_index("region").loc["North", "current_loan_to_value_weighted_avg"]
    assert got == pytest.approx(w)


def test_stratify_missing_dimension_raises():
    df = pd.DataFrame({"bal": [1.0]})
    with pytest.raises(ValueError):
        stratify(df, "nope", "bal")


# --------------------------------------------------------------------------- #
# 6. Concentration: top-N, shares, limit usage / RAG
# --------------------------------------------------------------------------- #


def test_top_n_concentration(full_df):
    res = top_n_concentration(full_df, "region",
                              "current_outstanding_balance", n=1)
    assert res["n_groups"] == 3
    # East is a single 2m loan = the largest single region.
    top_region = res["groups"].iloc[0]["region"]
    assert top_region == "East"
    assert 0.0 < res["balance_concentration"] <= 1.0


def test_group_shares_sum_to_one(full_df):
    table = group_shares(full_df, "region", "current_outstanding_balance",
                         loan_id_col="loan_id")
    assert pytest.approx(table["balance_share"].sum(), abs=1e-9) == 1.0
    assert pytest.approx(table["count_share"].sum(), abs=1e-9) == 1.0


def test_rag_status_thresholds():
    assert rag_status(0.5) == "green"
    assert rag_status(0.85) == "amber"
    assert rag_status(1.2) == "red"
    assert rag_status(float("nan")) == "green"


def test_limit_usage_status(full_df):
    # East holds ~67% of balance; a 50% limit => over limit (red).
    limits = {"East": 0.50, "North": 0.50, "South": 0.50}
    table = limit_usage(full_df, "region", "current_outstanding_balance", limits)
    east = table.set_index("region").loc["East"]
    assert east["status"] == "red"
    assert east["limit_usage"] > 1.0


# --------------------------------------------------------------------------- #
# 7. Cohort / vintage foundations
# --------------------------------------------------------------------------- #


@pytest.fixture
def dated_df():
    return pd.DataFrame({
        "loan_id": ["L1", "L2", "L3", "L4"],
        "origination_date": ["2020-01-15", "2020-06-30", "2021-03-01", "2021-09-09"],
        "acquisition_date": ["2022-01-01", "2022-01-01", "2023-05-01", "2023-05-01"],
        "funding_date": ["2020-02-01", "2020-07-15", "2021-04-01", "2021-10-10"],
        "bal": [100.0, 200.0, 300.0, 400.0],
    })


@pytest.mark.parametrize("date_col", ["origination_date", "acquisition_date",
                                       "funding_date"])
def test_cohort_table_by_each_date(dated_df, date_col):
    table, issues = cohort_table(dated_df, date_col, "bal", period="Y")
    assert not issues
    assert pytest.approx(table["balance_share"].sum(), abs=1e-9) == 1.0
    assert int(table["loan_count"].sum()) == 4


def test_cohort_period_quarter(dated_df):
    out, issues = add_cohort_period(dated_df, "origination_date", period="Q")
    assert not issues
    assert str(out["origination_date_cohort"].iloc[0]) == "2020Q1"


def test_cohort_missing_date_field_reports_issue(dated_df):
    table, issues = cohort_table(dated_df, "nonexistent_date", "bal")
    assert table.empty
    assert issues and issues[0]["code"] == "unavailable_field"


def test_months_on_book_scalar_as_of():
    df = pd.DataFrame({"origination_date": ["2020-01-15", "2020-07-01"]})
    out, issues = months_on_book(df, "origination_date", "2021-07-15")
    assert not issues
    # 2020-01-15 -> 2021-07-15 = 18 months; 2020-07-01 -> 2021-07-15 = 12.
    assert int(out["months_on_book"].iloc[0]) == 18
    assert int(out["months_on_book"].iloc[1]) == 12


def test_months_on_book_negative_clamped():
    df = pd.DataFrame({"origination_date": ["2022-01-01"]})
    out, issues = months_on_book(df, "origination_date", "2021-01-01")
    assert int(out["months_on_book"].iloc[0]) == 0
    assert any(i["code"] == "negative_months" for i in issues)


# --------------------------------------------------------------------------- #
# 8. No legacy / UI / cloud imports in the package source
# --------------------------------------------------------------------------- #


def test_no_forbidden_imports_in_analytics_lib():
    pkg = REPO_ROOT / "analytics_lib"
    banned = ("from analytics", "import analytics_legacy", "import streamlit",
              "import plotly", "import azure", "from azure")
    # "import analytics " (legacy app) is forbidden, but "import analytics_lib"
    # (self) is fine — check word boundaries via a trailing dot/space.
    for py in pkg.glob("*.py"):
        text = py.read_text(encoding="utf-8")
        for token in banned:
            assert token not in text, f"{py.name} contains forbidden {token!r}"
        # Guard against importing the legacy top-level 'analytics' package.
        for line in text.splitlines():
            stripped = line.strip()
            assert not stripped.startswith("import analytics."), stripped
            assert stripped != "import analytics", stripped
