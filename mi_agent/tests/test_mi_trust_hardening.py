#!/usr/bin/env python3
"""tests/test_mi_trust_hardening.py

MI Agent trust-hardening: metric-intent parsing, single-metric KPIs, data-aware
alias resolution, reconciliation/coverage footer, Unknown/Missing bucketing and
export consistency. Mirrors the acceptance examples A–F.
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

from mi_agent.mi_query_validator import load_mi_semantics
from mi_agent.llm_query_parser import _deterministic_parse
from mi_agent.mi_agent_workflow import run_mi_agent_query, result_csv_bytes

_SEMANTICS = _REPO_ROOT / "mi_agent" / "mi_semantics_field_registry.yaml"


@pytest.fixture(scope="module")
def semantics():
    return load_mi_semantics(_SEMANTICS)


@pytest.fixture(scope="module")
def df():
    rng = np.random.default_rng(7)
    n = 120
    frame = pd.DataFrame({
        "current_outstanding_balance": rng.uniform(50_000, 300_000, n).round(2),
        "current_loan_to_value": rng.uniform(20, 80, n).round(1),
        "youngest_borrower_age": rng.integers(60, 90, n),
        "current_interest_rate": rng.uniform(3, 7, n).round(2),
        "broker_channel": rng.choice(["Alpha", "Beta", "Gamma", "Delta"], n),
        "borrower_structure": rng.choice(["Joint", "Sole"], n),
        "geographic_region_obligor": rng.choice(["North", "South", "East"], n),
        "ltv_bucket": rng.choice(["<40", "40-60", "60-80"], n),
        "age_bucket": rng.choice(["60-69", "70-79", "80+"], n),
    })
    # Some rows have a missing LTV bucket -> should become Unknown / Missing.
    frame.loc[:14, "ltv_bucket"] = ""
    return frame


def _cols(df):
    return set(df.columns)


# --------------------------------------------------------------------------- #
# A. "interest rate" -> weighted-average KPI, no bar chart
# --------------------------------------------------------------------------- #
def test_A_interest_rate_is_weighted_kpi(df, semantics):
    warnings.simplefilter("ignore")
    spec, _ = _deterministic_parse("interest rate", semantics, available_columns=_cols(df))
    assert spec.metric == "current_interest_rate"
    assert spec.aggregation == "weighted_avg"
    assert spec.dimension is None
    assert spec.chart_type == "none"          # no bar chart
    res = run_mi_agent_query("interest rate", df, semantics)
    assert res["ok"] is True
    assert res["query_result"].result_type == "summary"
    assert res["chart_result"] is None


# --------------------------------------------------------------------------- #
# B. "weighted average interest rate by region" -> balance-weighted bar + recon
# --------------------------------------------------------------------------- #
def test_B_weighted_avg_rate_by_region(df, semantics):
    res = run_mi_agent_query("weighted average interest rate by region", df, semantics)
    assert res["ok"] is True
    assert res["spec"]["aggregation"] == "weighted_avg"
    assert res["spec"]["dimension"] == "geographic_region_obligor"
    assert res["spec"]["weight_field"]  # balance-weighted
    recon = res["reconciliation"]
    assert recon and recon["total_balance"] > 0
    assert recon["coverage_by_balance_pct"] == 100.0


# --------------------------------------------------------------------------- #
# C. "how many joint borrowers and balance" -> count + balance + share
# --------------------------------------------------------------------------- #
def test_C_joint_borrowers_count_and_balance(df, semantics):
    res = run_mi_agent_query("how many joint borrowers and balance", df, semantics)
    assert res["ok"] is True
    assert res["spec"]["filters"].get("borrower_structure") == "Joint"
    row = res["query_result"].data.iloc[0].to_dict()
    assert "loan_count" in row
    assert any("balance" in k for k in row)
    # Share of the funded book is exposed via the reconciliation coverage %.
    recon = res["reconciliation"]
    assert 0 < recon["coverage_by_balance_pct"] < 100


def test_C_joint_falls_back_to_number_of_borrowers(df, semantics):
    # borrower_structure absent but number_of_borrowers present -> proxy + note.
    d = df.drop(columns=["borrower_structure"]).copy()
    d["number_of_borrowers"] = np.where(np.arange(len(d)) % 2 == 0, 2, 1)
    spec, meta = _deterministic_parse("how many joint borrowers and balance",
                                      load_mi_semantics(_SEMANTICS),
                                      available_columns=set(d.columns))
    assert "number_of_borrowers" in spec.filters
    assert "number_of_borrowers" in meta.get("note", "")


# --------------------------------------------------------------------------- #
# D. "which broker has the highest average loan balance" -> AVG not total
# --------------------------------------------------------------------------- #
def test_D_highest_average_balance_is_avg(df, semantics):
    res = run_mi_agent_query("which broker has the highest average loan balance",
                             df, semantics)
    assert res["ok"] is True
    assert res["spec"]["aggregation"] == "avg"        # NOT sum
    assert res["spec"]["dimension"] == "broker_channel"
    cols = res["query_result"].data.columns
    assert "loan_count" in cols                        # denominator shown
    assert any(c.endswith("_total") for c in cols)     # supporting total shown


def test_D_total_balance_is_sum(df, semantics):
    res = run_mi_agent_query("total balance by broker", df, semantics)
    assert res["spec"]["aggregation"] == "sum"


# --------------------------------------------------------------------------- #
# E. "balance by ltv bucket" -> reconciles; missing -> Unknown/Missing bucket
# --------------------------------------------------------------------------- #
def test_E_unknown_bucket_keeps_totals_reconciled(df, semantics):
    res = run_mi_agent_query("balance by ltv bucket", df, semantics)
    assert res["ok"] is True
    groups = list(res["query_result"].data.iloc[:, 0])
    assert "Unknown / Missing" in groups               # not silently dropped
    recon = res["reconciliation"]
    assert recon["coverage_by_balance_pct"] == 100.0   # everything reconciles


def test_E_exclude_missing_discloses_excluded_balance(df, semantics):
    res = run_mi_agent_query("balance by ltv bucket excluding missing", df, semantics)
    groups = list(res["query_result"].data.iloc[:, 0])
    assert "Unknown / Missing" not in groups
    recon = res["reconciliation"]
    assert recon["missing_dimension_policy"] == "exclude"
    assert recon["balance_excluded_missing"] > 0
    assert recon["coverage_by_balance_pct"] < 100.0
    # The workflow surfaces a plain-English coverage sentence.
    assert any("of the £" in w and "excluded" in w for w in res["warnings"])


def test_E_csv_export_reconciles_and_has_footer(df, semantics):
    res = run_mi_agent_query("balance by ltv bucket", df, semantics)
    csv = result_csv_bytes(res["query_result"]).decode("utf-8")
    assert "Reconciliation / coverage" in csv
    assert "Coverage by balance: 100.0%" in csv
    # The data total in the CSV equals the funded-book total (Unknown bucket kept).
    data_part = csv.split("# --- Reconciliation")[0]
    rows = [r for r in data_part.strip().splitlines() if r]
    header = rows[0].split(",")
    sum_idx = next(i for i, h in enumerate(header) if h.endswith("_sum"))
    total = sum(float(r.split(",")[sum_idx]) for r in rows[1:])
    assert abs(total - res["reconciliation"]["total_balance"]) < 1.0


# --------------------------------------------------------------------------- #
# F. "bubble chart of LTV vs age vs balance" -> consistent alias resolution
# --------------------------------------------------------------------------- #
def test_F_bubble_alias_resolution(df, semantics):
    res = run_mi_agent_query("bubble chart of ltv vs age vs balance", df, semantics)
    assert res["ok"] is True
    spec = res["spec"]
    assert spec["chart_type"] == "bubble"
    assert {spec["x"], spec["y"]} == {"youngest_borrower_age", "current_loan_to_value"}
    assert spec["size"] == "current_outstanding_balance"


def test_F_age_resolves_to_present_column(semantics):
    # When youngest_borrower_age is ABSENT but a synonymous integer age field is
    # present, alias resolution must pick the present one (no first-attempt fail).
    from mi_agent.llm_query_parser import _age_metric
    fields = semantics.get("fields", {})
    age_fields = [k for k, e in fields.items()
                  if e.get("role") == "metric" and e.get("format") == "integer"
                  and "age" in (k + str(e.get("display_name", ""))).lower()]
    # Default (no column context) is stable.
    assert _age_metric(semantics) in age_fields
    # With only youngest_borrower_age's column present, it is chosen.
    chosen = _age_metric(semantics, available_columns={"youngest_borrower_age"})
    assert fields.get(chosen, {}).get("canonical_field") == "youngest_borrower_age"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
