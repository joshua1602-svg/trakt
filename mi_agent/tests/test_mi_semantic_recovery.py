#!/usr/bin/env python3
"""tests/test_mi_semantic_recovery.py

Validation-as-recovery, the governed average-loan-balance metric, expanded
reconciliation schema and extended alias resolution — all on the EXISTING
registry / parser / validator / executor (no parallel framework).
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from mi_agent.mi_query_validator import (
    load_mi_semantics, recover_chart_spec, validate_mi_query)
from mi_agent.mi_query_spec import MIQuerySpec
from mi_agent.llm_query_parser import _deterministic_parse, find_field
from mi_agent.mi_agent_workflow import run_mi_agent_query

_SEMANTICS = _REPO_ROOT / "mi_agent" / "mi_semantics_field_registry.yaml"


@pytest.fixture(scope="module")
def semantics():
    return load_mi_semantics(_SEMANTICS)


@pytest.fixture(scope="module")
def df():
    rng = np.random.default_rng(3)
    n = 90
    return pd.DataFrame({
        "current_outstanding_balance": rng.uniform(50_000, 300_000, n).round(2),
        "current_loan_to_value": rng.uniform(20, 80, n).round(1),
        "youngest_borrower_age": rng.integers(60, 90, n),
        "current_interest_rate": rng.uniform(3, 7, n).round(2),
        "broker_channel": rng.choice(["Alpha", "Beta", "Gamma"], n),
        "geographic_region_obligor": rng.choice(["North", "South"], n),
        "ltv_bucket": rng.choice(["<40", "40-60", "60-80"], n),
        "age_bucket": rng.choice(["60-69", "70-79"], n),
    })


# --------------------------------------------------------------------------- #
# Validation as a recovery / control layer
# --------------------------------------------------------------------------- #
class TestValidationRecovery:
    def test_metric_only_bar_recovers_to_kpi(self, semantics):
        cols = {"current_interest_rate", "current_outstanding_balance"}
        bad = MIQuerySpec(intent="chart", chart_type="bar",
                          metric="current_interest_rate", aggregation="weighted_avg",
                          weight_field="current_outstanding_balance")
        assert not validate_mi_query(bad, semantics, available_columns=cols).ok
        rec = recover_chart_spec(bad, semantics, cols)
        assert rec is not None
        assert rec.intent == "summary" and rec.chart_type == "none"
        assert validate_mi_query(rec, semantics, available_columns=cols).ok

    def test_recovery_does_not_mask_missing_dimension(self, semantics, df):
        # "balance by region" with no region column must FAIL (not silently KPI).
        d = pd.DataFrame({"current_outstanding_balance": [1.0, 2.0, 3.0]})
        res = run_mi_agent_query("Show balance by region", d, semantics)
        assert res["ok"] is False

    def test_workflow_recovers_metric_only_query(self, df, semantics):
        # The full workflow returns a KPI (no validation error) for a bare metric.
        res = run_mi_agent_query("interest rate", df, semantics)
        assert res["ok"] is True
        assert res["query_result"].result_type == "summary"
        assert res["chart_result"] is None


# --------------------------------------------------------------------------- #
# Governed average-loan-balance metric
# --------------------------------------------------------------------------- #
class TestGovernedAverageBalance:
    def test_registry_defines_average_loan_balance(self, semantics):
        defs = semantics.get("metadata", {}).get("metric_definitions", {})
        assert "average_loan_balance" in defs
        d = defs["average_loan_balance"]
        assert d["metric"] == "current_outstanding_balance"
        assert d["aggregation"] == "avg"
        assert "count(loans)" in d["formula"]

    def test_average_balance_by_broker_uses_avg_and_surfaces_definition(self, df, semantics):
        res = run_mi_agent_query("which broker has the highest average loan balance",
                                 df, semantics)
        assert res["ok"] is True
        assert res["spec"]["aggregation"] == "avg"
        assert res["spec"]["dimension"] == "broker_channel"
        md = res["query_result"].metadata.get("metric_definition")
        assert md and md["name"] == "average_loan_balance"
        # The governed breakdown (count + total) is in the table so avg is auditable.
        cols = list(res["query_result"].data.columns)
        assert "loan_count" in cols
        assert any(c.endswith("_total") for c in cols)


# --------------------------------------------------------------------------- #
# Expanded reconciliation schema
# --------------------------------------------------------------------------- #
class TestReconciliationSchema:
    def test_reconciliation_has_required_fields(self, df, semantics):
        warnings.simplefilter("ignore")
        res = run_mi_agent_query("balance by ltv bucket", df, semantics)
        recon = res["reconciliation"]
        for key in ("total_records", "total_balance", "records_included",
                    "balance_included", "records_excluded_missing",
                    "balance_excluded_missing", "coverage_by_balance_pct",
                    "missing_dimension_fields", "missing_measure_fields",
                    "filters"):
            assert key in recon, key

    def test_missing_measure_disclosed(self, semantics):
        # Some rows have a null balance (the measure) -> disclosed, not hidden.
        d = pd.DataFrame({
            "current_outstanding_balance": [100.0, 200.0, np.nan, np.nan],
            "broker_channel": ["A", "B", "A", "B"],
        })
        res = run_mi_agent_query("total balance by broker", d, semantics)
        recon = res["reconciliation"]
        assert "current_outstanding_balance" in recon["missing_measure_fields"]
        assert recon["records_missing_measure"] == 2


# --------------------------------------------------------------------------- #
# Extended alias resolution (existing registry synonyms)
# --------------------------------------------------------------------------- #
class TestAliasResolution:
    @pytest.mark.parametrize("phrase", ["age", "borrower age", "customer age", "youngest age"])
    def test_age_phrases_resolve_to_borrower_age(self, semantics, phrase):
        key = find_field(semantics, role="metric", fmt="integer",
                         keywords=(phrase,), strict=True)
        assert key == "youngest_borrower_age"

    def test_age_keyword_not_hijacked_by_months_on_book(self, semantics):
        # "age" must still resolve to the borrower-age field, not a synonym-only
        # look-alike (months_on_book lists "loan age").
        assert find_field(semantics, role="metric", fmt="integer",
                          keywords=("age",)) == "youngest_borrower_age"

    def test_bubble_customer_age_resolves_first_attempt(self, df, semantics):
        res = run_mi_agent_query("bubble chart of customer age vs ltv vs balance",
                                 df, semantics)
        assert res["ok"] is True
        spec = res["spec"]
        assert spec["x"] == "youngest_borrower_age"
        assert spec["y"] == "current_loan_to_value"
        assert spec["size"] == "current_outstanding_balance"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
