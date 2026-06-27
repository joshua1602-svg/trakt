#!/usr/bin/env python3
"""tests/test_mi_analytical_intents.py

Parser coverage for the ERE securitisation sprint analytical intents:

  * loan / case COUNT evolution resolves to a COUNT metric (not balance/sum);
  * cross-period comparison compiles to a governed temporal_mode='compare' plan;
  * securitisation scale-up / run-rate questions compile to a forecast
    extrapolation plan (not a point-in-time KPI);
  * risk-limit / concentration questions compile to a risk-monitor plan.

These run through the EXISTING deterministic parser + registry — no parallel
framework — and assert no hallucinated fields are referenced.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from mi_agent.mi_query_validator import load_mi_semantics
from mi_agent.llm_query_parser import _deterministic_parse

_SEMANTICS = _REPO_ROOT / "mi_agent" / "mi_semantics_field_registry.yaml"


@pytest.fixture(scope="module")
def semantics():
    return load_mi_semantics(_SEMANTICS)


def _parse(q, semantics):
    return _deterministic_parse(q, semantics)[0]


# --------------------------------------------------------------------------- #
# Bug #1 — loan count evolution resolves to COUNT, not balance/sum
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("q", [
    "Show loan count evolution by month.",
    "Show funded loan count trend.",
    "Show number of loans by reporting month.",
    "Show pipeline case count evolution by week.",
    "Show number of cases by week.",
    "Show monthly loan count evolution by broker.",
])
def test_loan_count_evolution_is_count(q, semantics):
    s = _parse(q, semantics)
    assert s.chart_type == "line", q
    assert s.aggregation == "count", (q, s.aggregation)
    # A count time-series carries NO balance metric.
    assert s.metric is None, (q, s.metric)


def test_balance_evolution_remains_balance(semantics):
    s = _parse("Show funded balance evolution by month.", semantics)
    assert s.chart_type == "line"
    assert s.aggregation == "sum"
    assert s.metric == "current_outstanding_balance"


# --------------------------------------------------------------------------- #
# Bug #2 — cross-period comparison compiles to a governed temporal_compare plan
# --------------------------------------------------------------------------- #
def test_compare_funded_balance(semantics):
    s = _parse("Compare October and November funded balance.", semantics)
    assert s.temporal_mode == "compare"
    assert s.execution_mode == "temporal"
    assert s.compare_periods == ["October", "November"]
    assert s.metric == "current_outstanding_balance"
    assert s.aggregation == "sum"


def test_compare_loan_count_is_count(semantics):
    s = _parse("Compare October and November loan count.", semantics)
    assert s.temporal_mode == "compare"
    assert s.compare_periods == ["October", "November"]
    assert s.metric is None and s.aggregation == "count"


def test_compare_wa_ltv(semantics):
    s = _parse("Compare October and November WA LTV.", semantics)
    assert s.temporal_mode == "compare"
    assert s.metric == "current_loan_to_value"
    assert s.aggregation == "weighted_avg"


def test_compare_latest_prior_pipeline(semantics):
    s = _parse("Compare latest pipeline with prior pipeline.", semantics)
    assert s.temporal_mode == "compare"
    assert s.compare_periods == ["latest", "prior pipeline"]


@pytest.mark.parametrize("q,periods", [
    ("How did funded balance change from October to November?", ["October", "November"]),
    ("How did pipeline amount change from last week?", ["latest", "last week"]),
])
def test_compare_change_phrasing(q, periods, semantics):
    s = _parse(q, semantics)
    assert s.temporal_mode == "compare", q
    assert s.compare_periods == periods, q


# --------------------------------------------------------------------------- #
# Forecast scale-up / run-rate questions → forecast extrapolation plan
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("q,kind,target", [
    ("When do we reach £50m funded balance?", "reach_threshold", 50_000_000.0),
    ("When do we reach £100m funded balance?", "reach_threshold", 100_000_000.0),
    ("How much pipeline is needed to reach £100m?", "pipeline_needed", 100_000_000.0),
    ("What is the current completion run rate?", "run_rate", None),
    ("What is the annualised completion run rate?", "run_rate_annualised", None),
    ("Show the funded balance extrapolation curve.", "extrapolation_curve", None),
    ("What is the downside forecast?", "scenario_downside", None),
    ("What is the upside forecast?", "scenario_upside", None),
    ("What happens if completion run rate falls by 25%?", "scenario", None),
    ("What completion rate is assumed from KFI to completion?", "conversion", None),
    ("Compare current weighted pipeline forecast with run-rate extrapolation.",
     "compare_models", None),
])
def test_forecast_scale_questions(q, kind, target, semantics):
    s = _parse(q, semantics)
    assert s.forecast_mode == "extrapolation", q
    assert s.forecast_question == kind, (q, s.forecast_question)
    assert s.forecast_target_value == target, (q, s.forecast_target_value)


def test_forecast_scale_does_not_hijack_point_in_time(semantics):
    # The existing point-in-time forecast bridge question must NOT become a
    # scale-up extrapolation plan.
    for q in ("What is the forecast funded balance?", "Show forecast balance by region.",
              "What is the forecast bridge?"):
        s = _parse(q, semantics)
        assert s.forecast_mode != "extrapolation", q


# --------------------------------------------------------------------------- #
# Risk-limit / concentration questions → risk-monitor plan
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("q", [
    "Are we within our concentration limits?",
    "Which limits are breached?",
    "Show risk limit headroom.",
    "Show the Schedule 8 concentration tests.",
    "What is the headroom on the London concentration limit?",
    "Which risk limits need review?",
])
def test_risk_limit_questions(q, semantics):
    s = _parse(q, semantics)
    assert s.risk_limit_query is True, q
    assert s.risk_monitor_mode == "concentration", q
    assert s.execution_mode == "risk", q


def test_risk_limit_does_not_hijack_concentration_ranking(semantics):
    # Bare "concentration" ranking questions must remain ranking, not risk-limit.
    for q in ("What is the largest regional concentration?",
              "What is the top 5 broker concentration?"):
        s = _parse(q, semantics)
        assert not s.risk_limit_query, q


def test_company_name_with_limited_is_not_a_risk_limit(semantics):
    # "Equity Release Supermarket Limited" must not trigger the risk-limit intent.
    s = _parse("Show loans for Equity Release Supermarket Limited.", semantics)
    assert not s.risk_limit_query


# --------------------------------------------------------------------------- #
# No hallucinated fields across all new intents
# --------------------------------------------------------------------------- #
def test_new_intents_reference_no_hallucinated_fields(semantics):
    registry = set(semantics.get("fields", {}))
    questions = [
        "Compare October and November funded balance.",
        "When do we reach £50m funded balance?",
        "Are we within our concentration limits?",
        "Show loan count evolution by month.",
    ]
    for q in questions:
        s = _parse(q, semantics)
        for fld in s.referenced_fields():
            assert fld in registry, (q, fld)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
