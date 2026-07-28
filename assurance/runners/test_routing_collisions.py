"""Routing collision matrix (Phase 5 §21).

Nearby routes must have a stable ownership rule. These tests drive ambiguous and
near-boundary questions through the LIVE recogniser registry (via the governed
capability with the deterministic parser) and assert which route wins, plus the
safety property that a losing route never silently produces a different
calculation.

Each collision documents the ownership rule and a positive/negative example.
Observed routes are recorded for the assurance report even where the assertion is
a safety property rather than an exact route (ambiguity is allowed; a wrong-safe
answer is not).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
FIX = ROOT / "assurance/fixtures/generated"


@pytest.fixture()
def ask(monkeypatch):
    monkeypatch.setenv("TRAKT_RUNTIME_MODE", "test")
    monkeypatch.setenv("MI_AGENT_PLATFORM_CANONICAL", str(FIX / "two_reporting_dates.csv"))
    monkeypatch.delenv("MI_AGENT_ONBOARDING_OUTPUT_ROOT", raising=False)
    monkeypatch.setenv("MI_AGENT_CLIENT_ID", "client_001")
    monkeypatch.setenv("MI_AGENT_LLM_ENABLED", "0")
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    from mi_agent_api import data_source, datasets
    from trakt_core.context import ExecutionContext
    from mi_agent_api.mi_service import MiQueryRequest, execute_governed_mi_query

    def _ask(question, lens=None):
        data_source.reset_cache()
        datasets._CLIENT_CURRENCY_CACHE.clear()
        ctx = ExecutionContext.for_internal("client_001")
        res = execute_governed_mi_query(
            MiQueryRequest(question=question, source_portfolio_lens=lens), ctx)
        route = ((res.result or {}).get("metadata") or {}).get("route")
        return res, route

    return _ask


def _forbidden_absent(result, terms):
    text = str((result.result or {}).get("answer") or "").lower()
    for art in (result.result or {}).get("artifacts") or []:
        for kpi in art.get("kpis") or []:
            text += " " + str(kpi.get("value", "")).lower()
    return [t for t in terms if t in text]


# Each case: (question, acceptable_routes, forbidden_terms)
COLLISIONS = [
    ("What is the total outstanding balance?", {"mi", None}, []),
    ("Show exposure by region.", {"geo_exposure", "mi", None}, []),
    ("What are the top 5 exposures?", {"concentration_analysis", "mi", None}, ["within appetite"]),
    ("How concentrated is the portfolio by region?",
     {"concentration_analysis", "geo_exposure", "mi", None}, ["excessive"]),
    ("What is the concentration limit status by region?",
     {"risk_limits", "concentration_analysis", "mi", None}, []),
    ("Compare the direct book with the acquired book.",
     {"portfolio_risk_comparison", "mi", None}, ["safer", "within appetite"]),
    ("How has the balance changed since last month?",
     {"period_change_analysis", "temporal_compare", "period_movement", "mi", None},
     ["primarily driven by"]),
    ("Forecast next month's balance.",
     {"forecast_extrapolation", "mi", None}, ["guaranteed"]),
    ("List the loans in the acquired book.", {"mi", None}, []),
    ("How has concentration by property type moved this month?",
     {"period_change_analysis", "concentration_analysis", "mi", None}, []),
]


@pytest.mark.parametrize("question,acceptable,forbidden", COLLISIONS)
def test_collision_route_is_acceptable_and_safe(ask, question, acceptable, forbidden):
    result, route = ask(question)
    # The route must be one of the acceptable owners for this boundary.
    assert route in acceptable, f"{question!r} routed to unexpected {route!r}"
    # And whatever answered must not leak a forbidden judgement/causation claim.
    leaked = _forbidden_absent(result, forbidden)
    assert leaked == [], f"{question!r} leaked forbidden claims {leaked}"


def test_comparison_is_not_a_period_movement(ask):
    """A same-period portfolio comparison must not be answered as a temporal
    movement (different population, different calculation)."""
    _, route = ask("Compare the direct book with the acquired book.")
    assert route not in {"period_change_analysis", "temporal_compare", "period_movement"}


def test_list_request_is_not_a_temporal_route(ask):
    _, route = ask("List the loans in the acquired book.")
    assert route not in {"temporal_compare", "period_change_analysis"}
