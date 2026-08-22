#!/usr/bin/env python3
"""tests/test_p1b_route_precedence.py — governed route precedence.

The rule P1B establishes:

    A specialist governed capability the deterministic parser positively
    recognises may not be shadowed by a generic LLM spec.

Why this can happen at all: the chat recognisers dispatch on SPEC FIELDS —
``risk_limit_query``, ``forecast_mode``, ``bridge_query``, ``cohort_progression``,
``temporal_mode`` — and only the deterministic parser sets them. Nothing in the
field catalogue the LLM is prompted with expresses "this is a covenant-headroom
question". When the LLM spec replaced the deterministic one wholesale, those
markers vanished and a purpose-built governed capability silently became a
generic chart:

  * "Am I close to breaching any concentration limits?" -> field-unavailable refusal
  * "What is the run rate of new lending?"              -> a 150-point line chart

``carry_specialist_intent`` preserves the markers at the one seam where both
specs exist. These tests pin the contract WITHOUT calling a live model: the LLM
is injected as a fixed spec, which is what makes them deterministic and CI-safe.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from mi_agent.llm_query_parser import (
    _SPECIALIST_INTENT_FIELDS,
    _deterministic_parse,
    carry_specialist_intent,
    parse_with_repair,
)
from mi_agent.mi_query_spec import MIQuerySpec
from mi_agent.mi_query_validator import load_mi_semantics

_SEMANTICS_PATH = _REPO_ROOT / "mi_agent" / "mi_semantics_field_registry.yaml"
_COLUMNS = {
    "current_outstanding_balance", "current_loan_to_value", "current_interest_rate",
    "youngest_borrower_age", "collateral_geography", "geographic_region_obligor",
    "account_status", "ltv_bucket", "origination_date", "data_cut_off_date",
}


@pytest.fixture(scope="module")
def semantics():
    return load_mi_semantics(_SEMANTICS_PATH)


#: A plausible generic spec of the kind the model returns for ANY question: a
#: valid balance-by-region bar with no specialist markers at all.
_GENERIC_LLM_SPEC = json.dumps({
    "intent": "chart", "chart_type": "bar",
    "metric": "current_outstanding_balance", "aggregation": "sum",
    "dimension": "collateral_geography",
    "explanation": "Balance by region.",
})


def _llm(_prompt):
    return _GENERIC_LLM_SPEC


# --------------------------------------------------------------------------- #
# The specialist markers each recogniser dispatches on
# --------------------------------------------------------------------------- #
#: (question, spec field the chat recogniser reads, expected value)
SPECIALIST = [
    ("Am I close to breaching any concentration limits?", "risk_limit_query", True),
    ("Which concentration limits are closest to breach?", "risk_limit_query", True),
    ("What covenant headroom do I have?", "risk_limit_query", True),
    ("Are any portfolio limits breached?", "risk_limit_query", True),
    ("What is the run rate of new lending?", "forecast_mode", "extrapolation"),
    ("What is the current origination run-rate?", "forecast_mode", "extrapolation"),
]


@pytest.mark.parametrize("question,field,expected", SPECIALIST,
                         ids=[q[:46] for q, _, _ in SPECIALIST])
def test_deterministic_parser_recognises_the_specialist_intent(
        question, field, expected, semantics):
    """Baseline: the marker the recogniser needs is genuinely set."""
    spec, _ = _deterministic_parse(question, semantics, available_columns=_COLUMNS)
    assert getattr(spec, field, None) == expected


@pytest.mark.parametrize("question,field,expected", SPECIALIST,
                         ids=[q[:46] for q, _, _ in SPECIALIST])
def test_a_generic_llm_spec_cannot_shadow_the_specialist_intent(
        question, field, expected, semantics):
    """THE P1B CONTRACT. The model returns a generic balance-by-region bar;
    the specialist marker must survive so the governed route still wins."""
    spec, meta = parse_with_repair(
        question, semantics, available_columns=_COLUMNS,
        llm_enabled=True, llm_callable=_llm, zero_cost_first=False)
    assert meta["parser_mode"] == "llm", "the LLM spec must be the one in play"
    assert getattr(spec, field, None) == expected, (
        f"{field} was lost — the generic spec shadowed the governed capability")
    assert field in (meta.get("specialist_intent_carried") or [])


def test_the_llm_keeps_its_own_value_when_it_expresses_one(semantics):
    """Carry-forward fills gaps; it never overrides a value the LLM stated."""
    det = MIQuerySpec(intent="summary", temporal_mode="compare")
    llm = MIQuerySpec(intent="summary", temporal_mode="trend")
    carried = carry_specialist_intent(llm, det)
    assert llm.temporal_mode == "trend"
    assert "temporal_mode" not in carried


def test_carry_forward_moves_intent_only_never_data(semantics):
    """Nothing that selects rows or measures may be carried: the LLM's own
    metric, dimensions and filters must be untouched."""
    for forbidden in ("metric", "dimension", "dimensions", "filters", "x", "y",
                      "aggregation", "weight_field", "top_n"):
        assert forbidden not in _SPECIALIST_INTENT_FIELDS

    det, _ = _deterministic_parse("Am I close to breaching any concentration limits?",
                                  semantics, available_columns=_COLUMNS)
    llm = MIQuerySpec(intent="chart", chart_type="bar",
                      metric="current_outstanding_balance", aggregation="sum",
                      dimensions=["collateral_geography"],
                      filters={"collateral_geography": "London"})
    carry_specialist_intent(llm, det)
    assert llm.metric == "current_outstanding_balance"
    assert llm.dimensions == ["collateral_geography"]
    assert llm.filters == {"collateral_geography": "London"}


def test_a_question_with_no_specialist_intent_carries_nothing(semantics):
    """No false positives: an ordinary analytical question is untouched, so the
    generic parser keeps its proper job."""
    spec, meta = parse_with_repair(
        "show me balance by region", semantics, available_columns=_COLUMNS,
        llm_enabled=True, llm_callable=_llm, zero_cost_first=False)
    assert not (meta.get("specialist_intent_carried") or [])
    for field in ("risk_limit_query", "forecast_mode", "bridge_query"):
        assert not getattr(spec, field, None)


def test_intent_survives_an_llm_spec_that_fails_validation(semantics):
    """A6's real shape: the model returned a spec referencing fields this book
    does not carry. Routing runs on the parsed spec BEFORE workflow validation,
    so the marker must survive that path too — otherwise the governed risk route
    is lost to a field-unavailable refusal."""
    def _invalid(_prompt):
        return json.dumps({"intent": "chart", "chart_type": "bar",
                           "metric": "not_a_real_field", "aggregation": "sum",
                           "dimensions": ["also_not_real"],
                           "explanation": "invalid"})

    spec, meta = parse_with_repair(
        "Am I close to breaching any concentration limits?", semantics,
        available_columns=_COLUMNS, llm_enabled=True, llm_callable=_invalid,
        zero_cost_first=False)
    # Whichever exit is taken — the LLM's invalid spec carried forward, or the
    # deterministic safety net — the ROUTING MARKER must survive, because that
    # is what decides whether the governed risk capability is reached.
    assert spec.risk_limit_query is True
    assert meta["parser_mode_detail"] in ("validation_failed", "deterministic_fallback")


# --------------------------------------------------------------------------- #
# Parser provenance — a fallback must never be reported as an LLM result
# --------------------------------------------------------------------------- #
def test_a_failed_llm_call_is_reported_as_a_fallback_not_as_an_llm_run(semantics):
    """The defect this exists to prevent: a revoked key produced a silent
    downgrade to deterministic, and an evaluation reported it as an LLM run."""
    from mi_agent_api.mi_service import _parser_provenance

    def _boom(_prompt):
        raise RuntimeError("Error code: 401 - authentication_error: API key is invalid.")

    _, meta = parse_with_repair(
        "show me balance by region", semantics, available_columns=_COLUMNS,
        llm_enabled=True, llm_callable=_boom, zero_cost_first=False)
    assert meta["parser_mode_detail"] == "deterministic_fallback"

    provenance = _parser_provenance({"metadata": {"parse_metadata": meta}})
    assert provenance["parser_used"] == "deterministic_fallback_after_llm_failure"
    assert provenance["llm_failure"] == "authentication"


def test_a_genuine_llm_parse_is_reported_as_llm(semantics):
    from mi_agent_api.mi_service import _parser_provenance

    _, meta = parse_with_repair(
        "show me balance by region", semantics, available_columns=_COLUMNS,
        llm_enabled=True, llm_callable=_llm, zero_cost_first=False)
    assert _parser_provenance(
        {"metadata": {"parse_metadata": meta}})["parser_used"] == "llm"


def test_a_deterministic_run_is_reported_as_deterministic(semantics):
    from mi_agent_api.mi_service import _parser_provenance

    _, meta = parse_with_repair("show me balance by region", semantics,
                                available_columns=_COLUMNS)
    assert _parser_provenance(
        {"metadata": {"parse_metadata": meta}})["parser_used"] == "deterministic"


# --------------------------------------------------------------------------- #
# Age threshold — the house convention (P1B Phase 7)
# --------------------------------------------------------------------------- #
# Strict where the language is strict, inclusive where it is inclusive:
#
#     over 85 / older than 85        -> age >  85
#     85 or older / at least 85 / 85+ -> age >= 85
#
# Pinned in the PARSER, so deterministic and LLM paths cannot disagree: the LLM
# spec states its own operator and is left alone, while a deterministic parse
# resolves the phrasing here. The receipt continues to show whichever predicate
# actually executed.
AGE_CONVENTION = [
    ("exposure to borrowers over 85", "gt", 85.0),
    ("borrowers older than 85", "gt", 85.0),
    ("borrowers aged 85 or older", "ge", 85.0),
    ("borrowers aged at least 85", "ge", 85.0),
    ("borrowers 85+", "ge", 85.0),
    ("borrowers 85 and over", "ge", 85.0),
    ("borrowers 40 and above", "ge", 40.0),
    ("borrowers under 70", "lt", 70.0),
    ("borrowers younger than 70", "lt", 70.0),
]


@pytest.mark.parametrize("question,op,value", AGE_CONVENTION,
                         ids=[q[:40] for q, _, _ in AGE_CONVENTION])
def test_age_threshold_house_convention(question, op, value, semantics):
    from mi_agent.llm_query_parser import _parse_filters

    condition = _parse_filters(question, semantics,
                               {"youngest_borrower_age",
                                "current_outstanding_balance"}
                               ).get("youngest_borrower_age")
    assert condition == {"op": op, "value": value}


# --------------------------------------------------------------------------- #
# Route coverage across phrasings (P1B Phase 8)
# --------------------------------------------------------------------------- #
# A capability recognised for one phrasing and lost for the next is a capability
# nobody can rely on. Each governed route below is pinned across the phrasings a
# reader actually uses, on BOTH parser paths.
BRIDGE_PHRASINGS = [
    "What drove the change in funded balance?",
    "What drove the movement in funded balance?",
    "Show me the balance bridge",
    "Bridge the funded balance movement",
    "Show me the funded balance bridge since last month",
]


@pytest.mark.parametrize("question", BRIDGE_PHRASINGS, ids=[q[:44] for q in BRIDGE_PHRASINGS])
def test_the_funded_bridge_marker_survives_every_phrasing(question, semantics):
    """``funded_bridge`` dispatches on ``bridge_query``, which only the
    deterministic parser sets — so every phrasing must keep it when the LLM spec
    is the one in play, or the bridge silently becomes a generic chart."""
    det, _ = _deterministic_parse(question, semantics, available_columns=_COLUMNS)
    assert det.bridge_query is True, "baseline: the deterministic parser sets it"

    spec, meta = parse_with_repair(
        question, semantics, available_columns=_COLUMNS,
        llm_enabled=True, llm_callable=_llm, zero_cost_first=False)
    assert meta["parser_mode"] == "llm"
    assert spec.bridge_query is True
    assert "bridge_query" in (meta.get("specialist_intent_carried") or [])


#: Period-change phrasings, and whether the governed period-change capability
#: owns them. The negatives matter as much as the positives: a route that claims
#: point-in-time questions is as broken as one that loses temporal ones.
PERIOD_CHANGE_PHRASINGS = [
    ("What changed since last month?", True),
    ("What moved month on month?", True),
    ("Which region grew the most last month?", True),
    ("Which region declined the most month-on-month?", True),
    ("How has the portfolio composition shifted since the prior month?", True),
    ("Rank regions by growth since last month.", True),
    # Not period change: no comparison at all.
    ("What is the weighted average LTV?", False),
    ("Where is the book concentrated?", False),
    ("Which region has the largest exposure?", False),
    # Not period change: owned by other governed capabilities by design.
    ("What is the run rate of new lending?", False),
    ("Show me the trend in funded balance", False),
]


@pytest.mark.parametrize("question,owned", PERIOD_CHANGE_PHRASINGS,
                         ids=[q[:44] for q, _ in PERIOD_CHANGE_PHRASINGS])
def test_period_change_recognition_across_phrasings(question, owned):
    from mi_agent.period_change import recognise

    assert bool(recognise(question, spec=None, view="funded").matched) is owned


@pytest.mark.parametrize("question,owned", PERIOD_CHANGE_PHRASINGS,
                         ids=[q[:44] for q, _ in PERIOD_CHANGE_PHRASINGS])
def test_geo_exposure_defers_to_period_change_and_only_to_it(question, owned):
    """The precedence rule, from the other side: the point-in-time geographic
    route stands down for exactly the questions the period-change capability
    claims, and for no others."""
    from mi_agent_api.chat_routing import _defers_to_period_change

    assert _defers_to_period_change(question) is owned


def test_a_point_in_time_geography_question_is_still_geo_exposure():
    from mi_agent_api.chat_routing import _is_geo_exposure

    assert _is_geo_exposure("Which region has the largest exposure?") is True
    assert _is_geo_exposure("Where is the largest geographic concentration?") is True
    # …and the same question with a period comparison is not.
    assert _is_geo_exposure("Which region grew the most last month?") is False


# --------------------------------------------------------------------------- #
# The age convention on the LLM path
# --------------------------------------------------------------------------- #
# Found by the P1C genuine-LLM run: on "what is my exposure to borrowers over
# 85?" the deterministic parser applied ``> 85`` (86 loans, £19.4m) and the
# model applied ``>= 85`` (136 loans, £31.1m). Two materially different answers
# to one question, decided by which parser happened to run.
CONVENTION_ON_LLM_PATH = [
    ("What is my exposure to borrowers over 85?", "ge", "gt"),
    ("What is my exposure to borrowers older than 85?", "ge", "gt"),
    ("What is my exposure to borrowers aged 85 or older?", "gt", "ge"),
    ("What is my exposure to borrowers aged at least 85?", "gt", "ge"),
]


@pytest.mark.parametrize("question,llm_op,expected", CONVENTION_ON_LLM_PATH,
                         ids=[q[:46] for q, _, _ in CONVENTION_ON_LLM_PATH])
def test_the_house_threshold_convention_survives_the_llm_path(
        question, llm_op, expected, semantics):
    def _llm_with(_prompt):
        return json.dumps({
            "intent": "summary", "chart_type": "none",
            "metric": "current_outstanding_balance", "aggregation": "sum",
            "filters": {"youngest_borrower_age": {"op": llm_op, "value": 85}},
            "explanation": "exposure by borrower age"})

    spec, meta = parse_with_repair(
        question, semantics, available_columns=_COLUMNS,
        llm_enabled=True, llm_callable=_llm_with, zero_cost_first=False)
    assert meta["parser_mode"] == "llm"
    assert spec.filters["youngest_borrower_age"]["op"] == expected
    assert spec.filters["youngest_borrower_age"]["value"] == 85
    assert "threshold_operator:youngest_borrower_age" in (
        meta.get("specialist_intent_carried") or [])


def test_only_the_operator_moves_never_the_field_or_the_value(semantics):
    """The reconciliation must not be able to change WHAT is filtered."""
    from mi_agent.llm_query_parser import reconcile_threshold_operators
    from mi_agent.mi_query_spec import MIQuerySpec

    det = MIQuerySpec(intent="summary",
                      filters={"youngest_borrower_age": {"op": "gt", "value": 85.0}})
    llm = MIQuerySpec(intent="summary",
                      filters={"youngest_borrower_age": {"op": "ge", "value": 85},
                               "current_loan_to_value": {"op": "lt", "value": 75}})
    reconcile_threshold_operators(llm, det)
    assert llm.filters == {"youngest_borrower_age": {"op": "gt", "value": 85},
                           "current_loan_to_value": {"op": "lt", "value": 75}}


@pytest.mark.parametrize("det_condition,llm_condition", [
    # A different value is a different question — leave it to the P0 guard.
    ({"op": "gt", "value": 85.0}, {"op": "ge", "value": 80}),
    # Opposite directions are a genuine disagreement, not a strictness nuance.
    ({"op": "gt", "value": 85.0}, {"op": "lt", "value": 85}),
    # An equality predicate is not a threshold.
    ({"op": "eq", "value": 85.0}, {"op": "ge", "value": 85}),
])
def test_the_reconciliation_stays_out_of_a_real_disagreement(det_condition,
                                                             llm_condition):
    from mi_agent.llm_query_parser import reconcile_threshold_operators
    from mi_agent.mi_query_spec import MIQuerySpec

    det = MIQuerySpec(intent="summary",
                      filters={"youngest_borrower_age": dict(det_condition)})
    llm = MIQuerySpec(intent="summary",
                      filters={"youngest_borrower_age": dict(llm_condition)})
    assert reconcile_threshold_operators(llm, det) == []
    assert llm.filters["youngest_borrower_age"] == llm_condition


def test_multi_clause_filters_still_split_after_the_and_fix(semantics):
    """The splitter change must not stop " and " joining two real predicates.

    The fix is a negative lookahead, so "and" still separates clauses everywhere
    except immediately before a bound word — which is the only place it was
    tearing one predicate in half.
    """
    from mi_agent.llm_query_parser import _parse_filters

    columns = {"youngest_borrower_age", "current_loan_to_value",
               "current_outstanding_balance"}
    for question in ("age over 70 and ltv above 50",
                     "balance over 100000 and ltv above 50",
                     "loans with ltv above 50 and age over 70"):
        applied = _parse_filters(question, semantics, columns)
        assert len(applied) == 2, f"{question!r} -> {applied}"
