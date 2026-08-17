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
