"""Tests for data resolution + registry-authorised metric resolution."""

from __future__ import annotations

import pandas as pd

from mi_agent_pptx.data_resolver import resolve_data
from mi_agent_pptx.metric_resolver import MetricResolver, format_value


# --------------------------------------------------------------- data_resolver
def test_balance_alias_and_ltv_normalisation(sample_tape, registries):
    rd = resolve_data(sample_tape, registries)
    # current_principal_balance -> canonical current_outstanding_balance.
    assert rd.balance_col == "current_outstanding_balance"
    assert "current_outstanding_balance" in rd.df.columns
    # LTV points (38..61) normalised to a 0..1 fraction.
    assert rd.df["current_loan_to_value"].max() <= 1.0
    assert rd.as_of_date == "2026-01-31"


def test_registry_buckets_materialised(sample_tape, registries):
    rd = resolve_data(sample_tape, registries)
    # LTV + age buckets should materialise from config/mi/buckets.yaml.
    assert rd.bucket_column("ltv_bucket") == "ltv_bucket"
    assert rd.bucket_column("borrower_age_bucket") == "borrower_age_bucket"
    assert rd.df["ltv_bucket"].notna().any()


def test_missing_balance_records_issue(registries):
    df = pd.DataFrame({"current_loan_to_value": [0.4, 0.5]})
    rd = resolve_data(df, registries)
    assert rd.balance_col is None
    assert any(i["code"] == "missing_balance_field" for i in rd.issues)


# ------------------------------------------------------------- metric_resolver
def _resolver(sample_tape, registries, analytics=None):
    rd = resolve_data(sample_tape, registries)
    return MetricResolver(rd, registries, analytics=analytics)


def test_sum_and_count_metrics(sample_tape, registries):
    mr = _resolver(sample_tape, registries)
    funded = mr.resolve({"key": "funded_balance",
                         "field": "current_outstanding_balance",
                         "aggregation": "sum", "format": "currency"})
    assert funded.ok
    assert abs(funded.value - sample_tape["current_principal_balance"].sum()) < 1e-6
    count = mr.resolve({"key": "loan_count", "kind": "count"})
    assert count.value == 6


def test_weighted_avg_uses_registry_weight(sample_tape, registries):
    mr = _resolver(sample_tape, registries)
    ltv = mr.resolve({"key": "wa_current_ltv", "field": "current_loan_to_value",
                     "aggregation": "weighted_avg", "format": "percent"})
    assert ltv.ok
    assert 0.0 < ltv.value < 1.0
    assert ltv.basis == "registry_computed"


def test_weighted_avg_falls_back_when_weight_unusable(registries):
    # Balance all-NaN -> weighted_avg must fall back to a simple mean.
    df = pd.DataFrame({
        "current_principal_balance": [None, None, None],
        "current_loan_to_value": [0.30, 0.40, 0.50],
        "youngest_borrower_age": [70, 72, 74],
    })
    rd = resolve_data(df, registries)
    mr = MetricResolver(rd, registries)
    ltv = mr.resolve({"key": "wa_current_ltv", "field": "current_loan_to_value",
                     "aggregation": "weighted_avg", "format": "percent"})
    assert ltv.ok
    assert abs(ltv.value - 0.40) < 1e-6
    assert "simple mean" in ltv.note


def test_missing_field_returns_unavailable(sample_tape, registries):
    mr = _resolver(sample_tape, registries)
    res = mr.resolve({"key": "forecast_funded_balance",
                     "field": "forecast_funded_balance",
                     "aggregation": "sum", "format": "currency"})
    assert not res.ok
    assert res.display == "—"
    assert res.basis == "unavailable"


def test_analytics_artifact_overrides_computation(sample_tape, registries):
    analytics = {"metrics": {"funded_balance": {"value": 9_999_999}}}
    mr = _resolver(sample_tape, registries, analytics=analytics)
    res = mr.resolve({"key": "funded_balance",
                     "field": "current_outstanding_balance",
                     "aggregation": "sum", "format": "currency"})
    assert res.value == 9_999_999
    assert res.basis == "analytics_artifact"


def test_largest_exposure_metric(sample_tape, registries):
    mr = _resolver(sample_tape, registries)
    res = mr.resolve({"key": "largest_region", "kind": "largest",
                     "dimension": "geographic_region_obligor", "format": "percent"})
    assert res.ok
    assert "·" in res.display  # "<region> · <share>%"


def test_format_value_variants():
    assert format_value(1_500_000, "currency") == "£1.5m"
    assert format_value(0.435, "percent") == "43.5%"
    assert format_value(7.63, "rate") == "7.63%"
    assert format_value(75.3, "years") == "75.3 yrs"
    assert format_value(None, "currency") == "—"
