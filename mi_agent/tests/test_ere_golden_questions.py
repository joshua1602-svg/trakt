#!/usr/bin/env python3
"""tests/test_ere_golden_questions.py

Golden MI-question regression + governed query-template coverage. Runs the bank
through the EXISTING parser / registry / validator / executor — it does not build
a parallel framework. Proves coverage without constraining the agent to exact
library entries.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from mi_agent.mi_query_validator import load_mi_semantics
from mi_agent.llm_query_parser import _deterministic_parse
from mi_agent.mi_agent_workflow import run_mi_agent_query

_BANK = _REPO_ROOT / "config" / "mi" / "golden_questions" / "ere_mi_questions.yaml"
_SEMANTICS = _REPO_ROOT / "mi_agent" / "mi_semantics_field_registry.yaml"


@pytest.fixture(scope="module")
def bank():
    return yaml.safe_load(_BANK.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def semantics():
    return load_mi_semantics(_SEMANTICS)


@pytest.fixture(scope="module")
def funded_df():
    rng = np.random.default_rng(11)
    n = 200
    return pd.DataFrame({
        "current_outstanding_balance": rng.uniform(40_000, 400_000, n).round(2),
        "original_principal_balance": rng.uniform(40_000, 400_000, n).round(2),
        "current_loan_to_value": rng.uniform(15, 75, n).round(1),
        "youngest_borrower_age": rng.integers(60, 92, n),
        "current_interest_rate": rng.uniform(3, 9, n).round(2),
        "current_valuation_amount": rng.uniform(150_000, 900_000, n).round(2),
        "broker_channel": rng.choice(["Alpha", "Beta", "Gamma", "Delta", "Eps"], n),
        "origination_channel": rng.choice(["Direct", "Intermediary"], n),
        "geographic_region_obligor": rng.choice(["North", "South East", "East", "Wales"], n),
        "account_status": rng.choice(["Performing", "Watch"], n),
        "borrower_structure": rng.choice(["Joint", "Sole"], n),
        "erm_product_type": rng.choice(["Lump Sum", "Drawdown"], n),
        "ltv_bucket": rng.choice(["<40", "40-60", "60-80"], n),
        "age_bucket": rng.choice(["60-69", "70-79", "80+"], n),
        "origination_date": rng.choice(
            pd.date_range("2023-01-01", "2025-06-01", freq="MS").astype(str), n),
    })


# --------------------------------------------------------------------------- #
# Part D/E — bank size + variation capacity
# --------------------------------------------------------------------------- #
def test_bank_has_at_least_250_base_questions(bank):
    assert len(bank["questions"]) >= 250


def test_variation_metadata_supports_over_1000_phrasings(bank):
    capacity = 0
    templated = 0
    for q in bank["questions"]:
        ax = q.get("variation_axes")
        if not ax:
            continue
        templated += 1
        prod = 1
        for key in ("thresholds", "operators", "field_synonyms", "metric_synonyms"):
            prod *= max(1, len(ax.get(key, [])))
        capacity += prod
    assert templated >= 1
    assert capacity + len(bank["questions"]) > 1000, capacity


def test_every_question_has_required_schema(bank):
    for q in bank["questions"]:
        for key in ("id", "category", "question", "dataset", "expected_intent",
                    "must_reconcile"):
            assert key in q, (q.get("id"), key)


# --------------------------------------------------------------------------- #
# Part H — parse every base question: no crash, no hallucinated fields
# --------------------------------------------------------------------------- #
def test_no_base_question_references_a_hallucinated_field(bank, semantics):
    registry = set(semantics.get("fields", {}))
    bad = []
    for q in bank["questions"]:
        spec, _ = _deterministic_parse(q["question"], semantics)
        for fld in spec.referenced_fields():
            if fld not in registry:
                bad.append((q["id"], fld))
    assert not bad, bad


def test_supported_funded_questions_produce_a_governed_plan(bank, semantics):
    cols = None
    weak = []
    for q in bank["questions"]:
        if not q.get("supported") or q["dataset"] != "funded":
            continue
        if q["expected_intent"] in ("controlled_unsupported", "meta", "time_series"):
            continue
        spec, _ = _deterministic_parse(q["question"], semantics, available_columns=cols)
        # A governed plan references at least one field OR is an explicit count.
        has_plan = bool(spec.referenced_fields()) or spec.aggregation in (
            "count", "count_distinct")
        if not has_plan:
            weak.append(q["id"])
    assert not weak, weak


# --------------------------------------------------------------------------- #
# Part I.7 — curated acceptance assertions
# --------------------------------------------------------------------------- #
def test_acceptance_queries(semantics, funded_df):
    def parse(qtext):
        return _deterministic_parse(qtext, semantics, available_columns=set(funded_df.columns))[0]

    # interest rate -> weighted-avg KPI
    s = parse("interest rate")
    assert s.metric == "current_interest_rate" and s.aggregation == "weighted_avg"
    assert s.dimension is None and s.chart_type == "none"

    # weighted average interest rate by region -> grouped weighted avg
    s = parse("weighted average interest rate by region")
    assert s.aggregation == "weighted_avg" and s.dimension == "geographic_region_obligor"

    # highest average loan balance -> AVG not total
    s = parse("which broker has the highest average loan balance")
    assert s.aggregation == "avg" and s.dimension == "broker_channel"

    # balance by broker -> sum bar
    s = parse("balance by broker")
    assert s.aggregation == "sum" and s.dimension == "broker_channel"

    # bubble of LTV vs age vs balance
    s = parse("bubble chart of LTV vs age vs balance")
    assert s.chart_type == "bubble"
    assert {s.x, s.y} == {"youngest_borrower_age", "current_loan_to_value"}
    assert s.size == "current_outstanding_balance"

    # filtered count: age 70+ AND LTV above 50%
    s = parse("how many borrowers are 70 years or above with LTV above 50%")
    assert "youngest_borrower_age" in s.filters and "current_loan_to_value" in s.filters

    # show loans with LTV above 50% -> filter on LTV
    s = parse("show loans with LTV above 50%")
    assert "current_loan_to_value" in s.filters

    # origination year breakdown (a vintage cohort — grouped bar or cohort line)
    s = parse("show balance by origination year")
    assert "vintage_year" in ([s.dimension, s.x] + list(s.dimensions or []))


def test_forecast_and_pipeline_kpis_parse(semantics):
    # These compile to governed plans (executed against the right runtime context).
    for qtext in ("forecast funded balance", "pipeline amount"):
        spec, _ = _deterministic_parse(qtext, semantics)
        assert spec is not None  # no crash; a plan is produced


# --------------------------------------------------------------------------- #
# Part D — controlled unsupported (no hallucination on absent fields)
# --------------------------------------------------------------------------- #
_MUST_BE_CONTROLLED = [
    "How many loans are in arrears?",
    "Show defaulted balance by vintage.",
    "Show NNEG exposure by LTV bucket.",
    "Show credit score by broker.",
    "Show recoveries in period.",
    "Show indexed value by region.",
]


@pytest.mark.parametrize("qtext", _MUST_BE_CONTROLLED)
def test_unsupported_fields_return_controlled_response(qtext, semantics, funded_df):
    res = run_mi_agent_query(qtext, funded_df, semantics)
    assert res.get("controlled_unsupported") is True, res.get("error")
    assert res.get("missing_fields")
    # The funded balance is NOT silently returned in its place.
    assert res.get("query_result") is None


def test_controlled_questions_never_hallucinate(bank, semantics):
    registry = set(semantics.get("fields", {}))
    for q in bank["questions"]:
        if q.get("supported"):
            continue
        spec, _ = _deterministic_parse(q["question"], semantics)
        for fld in spec.referenced_fields():
            assert fld in registry, (q["id"], fld)


# --------------------------------------------------------------------------- #
# Part H — must_reconcile: executed funded questions carry a reconciliation block
# --------------------------------------------------------------------------- #
def test_must_reconcile_funded_sample(bank, semantics, funded_df):
    sample = [q for q in bank["questions"]
              if q["dataset"] == "funded" and q.get("supported")
              and q["must_reconcile"]
              and q["expected_intent"] in ("kpi", "grouped", "filtered_kpi", "ranking")][:25]
    assert sample
    for q in sample:
        res = run_mi_agent_query(q["question"], funded_df, semantics)
        if res.get("ok"):
            assert res.get("reconciliation") is not None, q["id"]


# --------------------------------------------------------------------------- #
# Part E — parameterised variation extraction (thresholds / operators / values)
# --------------------------------------------------------------------------- #
def test_ltv_threshold_extraction(semantics, funded_df):
    cols = set(funded_df.columns)
    for thr in (30, 40, 50, 60):
        s = _deterministic_parse(f"how many loans have LTV above {thr}%", semantics,
                                 available_columns=cols)[0]
        cond = s.filters.get("current_loan_to_value")
        assert cond and cond["op"] == "gt" and cond["value"] == float(thr)


def test_age_threshold_extraction(semantics, funded_df):
    cols = set(funded_df.columns)
    for thr in (70, 75, 80):
        s = _deterministic_parse(f"how many borrowers with age above {thr}", semantics,
                                 available_columns=cols)[0]
        cond = s.filters.get("youngest_borrower_age")
        assert cond and cond["op"] == "gt" and cond["value"] == float(thr)


def test_balance_threshold_and_between(semantics, funded_df):
    cols = set(funded_df.columns)
    s = _deterministic_parse("how many loans with balance above 200000", semantics,
                             available_columns=cols)[0]
    cond = s.filters.get("current_outstanding_balance")
    assert cond and cond["op"] == "gt" and cond["value"] == 200000.0
    s2 = _deterministic_parse("show loans where ltv is between 20 and 40", semantics,
                              available_columns=cols)[0]
    cond2 = s2.filters.get("current_loan_to_value")
    assert cond2 and cond2["op"] == "between" and cond2["value"] == [20.0, 40.0]


def test_generated_variation_capacity_is_samplable(bank, semantics, funded_df):
    # Generate a handful of phrasings per template and confirm they parse.
    cols = set(funded_df.columns)
    generated = 0
    for q in bank["questions"]:
        ax = q.get("variation_axes")
        if not ax or q["category"] != "funded_filtered_qa":
            continue
        field = q["parameterised_template"]["field"]
        if field not in ("current_loan_to_value", "youngest_borrower_age",
                         "current_outstanding_balance", "current_interest_rate"):
            continue
        for thr in ax["thresholds"][:3]:
            if not isinstance(thr, (int, float)):
                continue
            noun = {"current_loan_to_value": "LTV",
                    "youngest_borrower_age": "borrower age",
                    "current_outstanding_balance": "balance",
                    "current_interest_rate": "interest rate"}[field]
            spec = _deterministic_parse(f"how many loans have {noun} above {thr}",
                                        semantics, available_columns=cols)[0]
            assert field in spec.filters
            generated += 1
    assert generated >= 6


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
