#!/usr/bin/env python3
"""tests/test_p1d_aggregate_contribution.py — contribution to a weighted aggregate.

The defect this exists to prevent, found by the P1C review run:

    "Which region contributes most to the weighted average LTV?" returned a
    chart TITLED WITH THE QUESTION and ordered by each region's own LTV, so
    West Midlands (43.90% LTV, 6.2% of the book) appeared first. The region
    that actually contributes most to the book's 43.15% is South East, at
    11.34pp against West Midlands' 2.72pp. The two rankings are near-inverted.

A portfolio weighted average is ``sum(w*v) / sum(w)``, so group g contributes
``sum over g of (w*v) / sum over the book of w``, which equals
``weight_share_g * value_g``, and the contributions sum back to the portfolio
figure. Every expectation below is computed from that definition with pandas
rather than taken from a previous run.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from mi_agent import execution_receipt as rc
from mi_agent.llm_query_parser import (
    _contribution_request, _deterministic_parse)
from mi_agent.mi_query_executor import execute_mi_query
from mi_agent.mi_query_spec import MIQuerySpec
from mi_agent.mi_query_validator import load_mi_semantics

_SEMANTICS_PATH = _REPO_ROOT / "mi_agent" / "mi_semantics_field_registry.yaml"
_REGION = "collateral_geography"
_LTV = "current_loan_to_value"
_BALANCE = "current_outstanding_balance"
_COLUMNS = {_REGION, _LTV, _BALANCE, "current_interest_rate",
            "youngest_borrower_age"}


@pytest.fixture(scope="module")
def semantics():
    return load_mi_semantics(_SEMANTICS_PATH)


@pytest.fixture(scope="module")
def book() -> pd.DataFrame:
    """A small book where the highest-LTV region is NOT the largest contributor.

    Constructed so the two rankings disagree, because a fixture where they agree
    cannot tell the two calculations apart.
    """
    return pd.DataFrame({
        _REGION: ["South East"] * 4 + ["West Midlands"] + ["London"] * 2,
        _BALANCE: [200.0, 200.0, 200.0, 200.0,   # SE    800 of 1,150
                   50.0,                          # WM     50
                   150.0, 150.0],                 # LDN   300
        _LTV: [40.0, 40.0, 40.0, 40.0,            # SE  LTV 40
               90.0,                              # WM  LTV 90  (highest)
               50.0, 50.0],                       # LDN LTV 50
    })


def _truth(book: pd.DataFrame) -> pd.DataFrame:
    """Contribution per region, from the definition, with pandas only."""
    frame = book.dropna(subset=[_LTV, _BALANCE])
    total_weight = frame[_BALANCE].sum()
    grouped = frame.assign(_wv=frame[_LTV] * frame[_BALANCE]).groupby(_REGION)
    out = pd.DataFrame({
        "contribution": grouped["_wv"].sum() / total_weight,
        "weight_share_pct": grouped[_BALANCE].sum() / total_weight * 100.0,
        "group_value": grouped["_wv"].sum() / grouped[_BALANCE].sum(),
    })
    return out.sort_values("contribution", ascending=False)


# =========================================================================== #
# The calculation
# =========================================================================== #
def test_contribution_matches_the_definition(book, semantics):
    spec = MIQuerySpec(intent="chart", chart_type="bar", metric=_LTV,
                       dimension=_REGION, x=_REGION, aggregation="contribution",
                       weight_field=_BALANCE)
    result = execute_mi_query(spec, book, semantics)
    truth = _truth(book)

    assert list(result.data[_REGION]) == list(truth.index)
    for _, row in result.data.iterrows():
        expected = truth.loc[row[_REGION]]
        assert row[f"{_LTV}_contribution"] == pytest.approx(expected["contribution"])
        assert row["weight_share_pct"] == pytest.approx(expected["weight_share_pct"])
        assert row[f"{_LTV}_weighted_avg"] == pytest.approx(expected["group_value"])


def test_contributions_sum_to_the_portfolio_figure(book, semantics):
    """The property that makes a contribution a contribution."""
    spec = MIQuerySpec(intent="chart", chart_type="bar", metric=_LTV,
                       dimension=_REGION, x=_REGION, aggregation="contribution",
                       weight_field=_BALANCE)
    result = execute_mi_query(spec, book, semantics)
    portfolio = (book[_LTV] * book[_BALANCE]).sum() / book[_BALANCE].sum()
    assert result.data[f"{_LTV}_contribution"].sum() == pytest.approx(portfolio)
    assert any("contributions sum to the portfolio" in w for w in result.warnings)


def test_the_largest_contributor_is_not_the_highest_value_group(book, semantics):
    """The whole point. West Midlands has the highest LTV; South East
    contributes the most. Ranking one as the other is the defect."""
    spec = MIQuerySpec(intent="chart", chart_type="bar", metric=_LTV,
                       dimension=_REGION, x=_REGION, aggregation="contribution",
                       weight_field=_BALANCE)
    data = execute_mi_query(spec, book, semantics).data
    assert data.iloc[0][_REGION] == "South East"
    assert data.loc[data[f"{_LTV}_weighted_avg"].idxmax()][_REGION] == "West Midlands"
    # …and the executed rows carry both figures, so a reader can see it.
    assert {f"{_LTV}_contribution", f"{_LTV}_weighted_avg",
            "weight_share_pct"} <= set(data.columns)


def test_a_zero_weight_book_is_refused_not_reported_as_zero(semantics):
    from mi_agent.mi_query_executor import MIQueryExecutionError

    empty = pd.DataFrame({_REGION: ["London"], _BALANCE: [0.0], _LTV: [50.0]})
    spec = MIQuerySpec(intent="chart", chart_type="bar", metric=_LTV,
                       dimension=_REGION, x=_REGION, aggregation="contribution",
                       weight_field=_BALANCE)
    with pytest.raises(MIQueryExecutionError):
        execute_mi_query(spec, empty, semantics)


def test_a_metric_with_no_governed_weighted_average_is_rejected(book, semantics):
    """A contribution to a weighted average is undefined where the registry
    does not allow a weighted average for the metric.

    The example was ``youngest_borrower_age`` until P1N, which permitted an
    exposure-weighted borrower age by product ruling. The invariant is unchanged
    — only the metric that illustrates it — so the case moves to a measure that
    still has no governed weighted average. Balance is the natural one: weighting
    a balance by balance is degenerate, which is why the registry omits it.
    """
    from mi_agent.mi_query_validator import validate_mi_query

    spec = MIQuerySpec(intent="chart", chart_type="bar",
                       metric="current_outstanding_balance", dimension=_REGION,
                       x=_REGION, aggregation="contribution")
    verdict = validate_mi_query(spec, semantics, available_columns=_COLUMNS)
    assert not verdict.ok
    assert any("contribution" in e.lower() for e in verdict.errors)


# =========================================================================== #
# Intent detection — and the distinctness that matters
# =========================================================================== #
CONTRIBUTION_PHRASINGS = [
    "Which region contributes most to the weighted average LTV?",
    "Which region contributes the most to the weighted average LTV?",
    "Which region is the largest contributor to the weighted average LTV?",
    "Which region is the biggest contributor to portfolio WA LTV?",
    "What drives most of the weighted average LTV?",
    "Which region accounts for most of the weighted average LTV?",
    "What is the regional contribution to the weighted average LTV?",
    "Which region contributes most to the weighted average interest rate?",
]


@pytest.mark.parametrize("question", CONTRIBUTION_PHRASINGS,
                         ids=[q[:48] for q in CONTRIBUTION_PHRASINGS])
def test_contribution_phrasings_are_recognised(question, semantics):
    assert _contribution_request(question.lower(), semantics, _COLUMNS) is not None


#: Questions that must NOT become a contribution. The first two are the
#: distinctness requirement: a per-group value ranking is a different intent.
NOT_CONTRIBUTION = [
    "Which region has the highest LTV?",
    "Which region has the highest weighted average LTV?",
    "Show me weighted average LTV by region",
    "Which region contributes most to the balance?",      # a sum, not weighted
    "Which region contributes most to the loan count?",   # a count, not weighted
    "What is the weighted average LTV?",
]


@pytest.mark.parametrize("question", NOT_CONTRIBUTION,
                         ids=[q[:48] for q in NOT_CONTRIBUTION])
def test_an_ordinary_ranking_is_never_converted(question, semantics):
    assert _contribution_request(question.lower(), semantics, _COLUMNS) is None


def test_the_two_intents_produce_different_specs(semantics):
    contribution, _ = _deterministic_parse(
        "Which region contributes most to the weighted average LTV?",
        semantics, available_columns=_COLUMNS)
    ranking, _ = _deterministic_parse(
        "Which region has the highest LTV?", semantics, available_columns=_COLUMNS)
    assert contribution.aggregation == "contribution"
    assert ranking.aggregation == "weighted_avg"
    # Same metric and same dimension — only the CALCULATION differs.
    assert contribution.metric == ranking.metric == _LTV
    assert contribution.dimension == ranking.dimension


def test_a_contribution_question_with_no_dimension_is_not_parsed_as_one(semantics):
    """Nothing to attribute across, so the parser stands down and the P0 facet
    refuses rather than this recogniser inventing a grouping."""
    spec, _ = _deterministic_parse("What drives most of the weighted average LTV?",
                                   semantics, available_columns={_LTV, _BALANCE})
    assert spec.aggregation != "contribution"


# =========================================================================== #
# P0 — a contribution question answered by a plain ranking must refuse
# =========================================================================== #
class _GroupedResult:
    result_type = "grouped"
    metadata = {"reconciliation": {}, "group_field_keys": [_REGION]}
    columns = [_REGION, f"{_LTV}_weighted_avg"]


_GUARD_SEMANTICS = {"fields": {
    _REGION: {"canonical_field": _REGION},
    _LTV: {"canonical_field": _LTV},
}}
_QUESTION = "Which region contributes most to the weighted average LTV?"


def _facets_for(aggregation: str):
    spec = MIQuerySpec(intent="chart", chart_type="bar", metric=_LTV,
                       dimension=_REGION, aggregation=aggregation)
    facets = rc.detect_requested_facets(
        _QUESTION, _GUARD_SEMANTICS, frame=None,
        requested_dimensions=[(_REGION, "region", ())])
    return spec, rc.reconcile_facets(
        facets, spec=spec, query_result=_GroupedResult(),
        semantics=_GUARD_SEMANTICS, available_columns={_REGION, _LTV})


def test_the_contribution_facet_is_detected_from_the_question():
    facets = rc.detect_requested_facets(
        _QUESTION, _GUARD_SEMANTICS, frame=None,
        requested_dimensions=[(_REGION, "region", ())])
    assert any(f.kind == rc.KIND_CONTRIBUTION for f in facets)


def test_a_plain_per_group_ranking_refuses_a_contribution_question():
    spec, facets = _facets_for("weighted_avg")
    contribution = [f for f in facets if f.kind == rc.KIND_CONTRIBUTION]
    assert contribution and contribution[0].status == rc.LOST
    verdict, message = rc.assess(rc.build_receipt(
        spec=spec, query_result=_GroupedResult(), semantics=_GUARD_SEMANTICS,
        facets=facets))
    assert verdict == rc.VERDICT_REFUSE
    assert "contribution to the portfolio weighted average" in message


def test_the_governed_contribution_calculation_passes_the_guard():
    spec, facets = _facets_for("contribution")
    contribution = [f for f in facets if f.kind == rc.KIND_CONTRIBUTION]
    assert contribution and contribution[0].status == rc.APPLIED
    verdict, _ = rc.assess(rc.build_receipt(
        spec=spec, query_result=_GroupedResult(), semantics=_GUARD_SEMANTICS,
        facets=facets))
    assert verdict == "ok"


@pytest.mark.parametrize("route", ["geo_exposure", "concentration_analysis",
                                   "risk_limits", "period_change_analysis"])
def test_no_routed_capability_may_claim_a_contribution_question(route):
    """None of them decomposes a weighted average across groups, so none may
    stand in for one."""
    facets = rc.reconcile_routed_facets(
        [rc.RequestedFacet(kind=rc.KIND_CONTRIBUTION,
                           label="contribution to the portfolio weighted average")],
        route=route, semantics=_GUARD_SEMANTICS)
    assert facets[0].status == rc.LOST


def test_an_ordinary_question_never_carries_a_contribution_facet():
    for question in NOT_CONTRIBUTION:
        facets = rc.detect_requested_facets(
            question, _GUARD_SEMANTICS, frame=None,
            requested_dimensions=[(_REGION, "region", ())])
        assert not [f for f in facets if f.kind == rc.KIND_CONTRIBUTION], question


# =========================================================================== #
# The chart title states the calculation, not the question
# =========================================================================== #
def test_the_chart_title_describes_the_executed_calculation(book, semantics):
    from mi_agent.mi_chart_factory import generate_chart_title

    spec = MIQuerySpec(intent="chart", chart_type="bar", metric=_LTV,
                       dimension=_REGION, x=_REGION, aggregation="contribution",
                       weight_field=_BALANCE, title=_QUESTION)
    result = execute_mi_query(spec, book, semantics)
    title, _subtitle = generate_chart_title(result, semantics)
    assert title == "Contribution to Portfolio Weighted Average Current LTV by Region"
    assert "contributes most" not in title.lower(), (
        "the chart must not be titled with the question it was asked")


# =========================================================================== #
# The LLM path — a generic spec must not answer a contribution question
# =========================================================================== #
def test_a_generic_llm_spec_for_b11_is_refused_not_answered(semantics):
    """The spec the model actually returned for B11: a plain weighted-average
    bar. It is a valid spec for a DIFFERENT question, so the guard refuses
    rather than presenting it — which is the P0 protection, not a fallback.

    Note the asymmetry this pins, deliberately: the deterministic parser
    ANSWERS this question and an LLM spec that misses the intent REFUSES it.
    Both outcomes are inside the boundary; neither is a silent wrong answer.
    """
    import json

    from mi_agent.llm_query_parser import parse_with_repair

    def _generic(_prompt):
        return json.dumps({
            "intent": "chart", "chart_type": "bar", "metric": _LTV,
            "x": _REGION, "aggregation": "weighted_avg",
            "sort_direction": "desc", "explanation": "WA LTV by region"})

    spec, meta = parse_with_repair(
        _QUESTION, semantics, available_columns=_COLUMNS,
        llm_enabled=True, llm_callable=_generic, zero_cost_first=False)
    assert meta["parser_mode"] == "llm"
    assert spec.aggregation == "weighted_avg"

    facets = rc.reconcile_facets(
        rc.detect_requested_facets(
            _QUESTION, _GUARD_SEMANTICS, frame=None,
            requested_dimensions=[(_REGION, "region", ())]),
        spec=spec, query_result=_GroupedResult(), semantics=_GUARD_SEMANTICS,
        available_columns={_REGION, _LTV})
    verdict, message = rc.assess(rc.build_receipt(
        spec=spec, query_result=_GroupedResult(), semantics=_GUARD_SEMANTICS,
        facets=facets))
    assert verdict == rc.VERDICT_REFUSE
    assert "contribution to the portfolio weighted average" in message
