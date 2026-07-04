#!/usr/bin/env python3
"""tests/test_chat_golden_regressions.py

Golden-question regressions for the live MI Agent chat. Each case here is a
question that produced a WRONG or MISLEADING answer in a live demo (see
due_diligence/MI_AGENT_CHATBOT_CRITICAL_REVIEW.md) — these pin the corrected
behaviour:

1. "Generate pipeline bridge to £100MM securitisation size" must compile to a
   governed scale-up forecast (gap to target), never a whole-book KPI summary.
2. "Ticket size by borrower type (i.e., single vs joint)" must cross-tab the
   two named dimensions — the " vs " token must NOT hijack the question into a
   hard-coded LTV-vs-age scatter.
3. "increase in completion conversion rates" must route to the governed
   KFI→completion conversion assumption, never a point-in-time summary.
4. A question that maps to NOTHING must be refused with a controlled
   "couldn't interpret" response — never silently answered with the whole book.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from mi_agent.llm_query_parser import (
    _deterministic_parse,
    _forecast_target_value,
    parse_with_repair,
)
from mi_agent.mi_agent_workflow import run_mi_agent_query
from mi_agent.mi_query_validator import load_mi_semantics

_SEMANTICS = _REPO_ROOT / "mi_agent" / "mi_semantics_field_registry.yaml"


@pytest.fixture(scope="module")
def semantics():
    return load_mi_semantics(_SEMANTICS)


@pytest.fixture(scope="module")
def funded_df():
    rng = np.random.default_rng(23)
    n = 150
    return pd.DataFrame({
        "current_outstanding_balance": rng.uniform(40_000, 400_000, n).round(2),
        "current_loan_to_value": rng.uniform(15, 75, n).round(1),
        "current_interest_rate": rng.uniform(3, 9, n).round(2),
        "youngest_borrower_age": rng.integers(60, 92, n),
        "borrower_type": rng.choice(["single", "joint"], n),
        "ticket_bucket": rng.choice(["<100k", "100-200k", "200k+"], n),
        "geographic_region_obligor": rng.choice(["North", "South East", "Wales"], n),
        "broker_channel": rng.choice(["Alpha", "Beta", "Gamma"], n),
    })


# --------------------------------------------------------------------------- #
# 1. Pipeline bridge to a securitisation target
# --------------------------------------------------------------------------- #
def test_pipeline_bridge_to_100mm_is_a_scaleup_forecast(semantics):
    spec, meta = _deterministic_parse(
        "generate pipeline bridge to £100mm securitisation size", semantics)
    assert spec.forecast_mode == "extrapolation", (
        "the bridge question fell through to a point-in-time plan")
    assert spec.forecast_question == "pipeline_needed"
    assert spec.forecast_target_value == 100_000_000.0
    assert meta["parser_confidence"] == "high"


@pytest.mark.parametrize("text,expected", [
    ("£100mm", 100_000_000.0),   # securitisation MM notation
    ("£100m", 100_000_000.0),
    ("£100 million", 100_000_000.0),
    ("100mm", 100_000_000.0),
    ("£0.1bn", 100_000_000.0),
    ("£250k", 250_000.0),
    ("£1.5 billion", 1_500_000_000.0),
])
def test_forecast_target_magnitudes(text, expected):
    assert _forecast_target_value(text.lower()) == expected


# --------------------------------------------------------------------------- #
# 2. Ticket size by borrower type — the " vs " scatter trap
# --------------------------------------------------------------------------- #
def test_ticket_size_by_borrower_type_is_a_cross_tab(semantics, funded_df):
    q = "ticket size by borrower type (i.e., single vs joint)"
    spec, _meta = _deterministic_parse(q, semantics)
    assert spec.chart_type == "heatmap", (
        f"expected a two-dimension cross-tab, got {spec.chart_type}")
    assert spec.dimensions == ["ticket_bucket", "borrower_type"]
    # And it executes over a funded frame that carries borrower_type.
    res = run_mi_agent_query(q, funded_df, semantics)
    assert res["ok"], res.get("error")


@pytest.mark.parametrize("q", [
    "ticket size by borrower type (i.e., single vs joint)",
    "outstanding balance by single vs joint borrowers",
    "balance split single vs joint",
])
def test_categorical_vs_phrasing_never_becomes_a_scatter(q, semantics):
    spec, _ = _deterministic_parse(q, semantics)
    assert spec.chart_type != "scatter", (
        "a categorical 'X vs Y' phrase was hijacked into a loan-level scatter")


def test_balance_by_borrower_type_never_substitutes_amortisation(semantics):
    spec, _ = _deterministic_parse("balance by borrower type", semantics)
    assert spec.dimension == "borrower_type"
    assert spec.metric == "current_outstanding_balance"


@pytest.mark.parametrize("q,expected_dim", [
    ("balance by originator", "originator_name"),
    ("balance by lender", "originator_name"),
    ("balance by intermediary", "broker_channel"),
    ("balance by employment status", "employment_status"),
    ("balance by portfolio cohort", "portfolio_cohort"),
])
def test_registry_synonyms_resolve_dimensions(q, expected_dim, semantics):
    spec, _ = _deterministic_parse(q, semantics)
    assert spec.dimension == expected_dim


def test_new_registry_synonym_is_understood_without_code_change(semantics):
    import copy
    sem = copy.deepcopy(semantics)
    sem["fields"]["broker_channel"]["synonyms"].append("distribution partner")
    spec, _ = _deterministic_parse("balance by distribution partner", sem)
    assert spec.dimension == "broker_channel"


@pytest.mark.parametrize("q", [
    "portfolio summary",                       # 'portfolio' must not become a dim
    "show me the ranking of brokers by balance",  # 'ranking' must not hijack to lien
])
def test_generic_words_are_not_hijacked_by_registry_terms(q, semantics):
    spec, _ = _deterministic_parse(q, semantics)
    # portfolio_id / lien must never be selected from these generic words.
    assert spec.dimension not in ("portfolio_id", "lien")


def test_numeric_vs_phrasing_still_produces_a_scatter(semantics):
    spec, _ = _deterministic_parse("ltv vs interest rate", semantics)
    assert spec.chart_type == "scatter"
    assert spec.x == "current_loan_to_value"
    assert spec.y == "current_interest_rate"

    spec2, _ = _deterministic_parse("scatter of balance vs age", semantics)
    assert spec2.chart_type == "scatter"
    assert spec2.x == "current_outstanding_balance"
    assert spec2.y == "youngest_borrower_age"


def test_scatter_axes_are_never_invented(semantics):
    # An explicit "scatter" with no resolvable axes must not default to
    # LTV-vs-age — it falls through to the summary/refusal grammar.
    spec, _ = _deterministic_parse("scatter of the portfolio please", semantics)
    assert spec.chart_type != "scatter"


# --------------------------------------------------------------------------- #
# 3. Completion conversion rates
# --------------------------------------------------------------------------- #
def test_completion_conversion_rates_route_to_conversion_forecast(semantics):
    spec, meta = _deterministic_parse(
        "increase in completion conversion rates", semantics)
    assert spec.forecast_mode == "extrapolation"
    assert spec.forecast_question == "conversion"
    assert meta["parser_confidence"] == "high"


# --------------------------------------------------------------------------- #
# 4. Unmapped questions are refused; genuine summaries still answer
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("q", [
    "tell me a joke about mortgages",
    "what is the meaning of life",
    "flibber jabber wocky",
])
def test_unmapped_question_is_refused_not_answered(q, semantics, funded_df):
    res = run_mi_agent_query(q, funded_df, semantics)
    assert res.get("unmapped_question") is True, (
        "an unintelligible question was silently answered")
    assert res.get("query_result") is None
    assert res.get("answer")  # a controlled, user-facing explanation
    assert not res["ok"]


@pytest.mark.parametrize("q", [
    "portfolio summary",
    "give me an overview of the book",
    "how many loans",
])
def test_summary_intent_still_answers_whole_book(q, semantics, funded_df):
    res = run_mi_agent_query(q, funded_df, semantics)
    assert res.get("unmapped_question") is None
    assert res["ok"], res.get("error")


# --------------------------------------------------------------------------- #
# 5. LLM gating: a heuristic (medium-confidence) deterministic parse must be
#    checked by the LLM when one is available; only HIGH confidence bypasses.
# --------------------------------------------------------------------------- #
def test_medium_confidence_parse_consults_llm_when_enabled(semantics):
    calls = []

    def mock_llm(prompt):
        calls.append(prompt)
        return ('{"intent":"chart","chart_type":"bar",'
                '"metric":"current_outstanding_balance",'
                '"dimension":"borrower_type","aggregation":"sum",'
                '"explanation":"mock"}')

    # "ltv vs age" resolves deterministically at MEDIUM confidence — with an
    # LLM available it must be verified, not short-circuited.
    spec, meta = parse_with_repair("ltv vs age", semantics,
                                   llm_enabled=True, llm_callable=mock_llm)
    assert len(calls) == 1
    assert meta["parser_mode"] == "llm"

    # A HIGH-confidence parse still short-circuits (zero cost).
    calls.clear()
    _spec, meta2 = parse_with_repair("balance by broker", semantics,
                                     llm_enabled=True, llm_callable=mock_llm)
    assert len(calls) == 0
    assert meta2["parser_mode_detail"] == "deterministic_zero_cost"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
