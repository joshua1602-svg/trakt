"""Recognition: the questions this workflow claims, and the ones it must not.

The negative controls matter more than the positives. A recogniser that steals a
forecast, a trend series, a raw-tape request or an incumbent route's question is
worse than one that misses a few phrasings, because the user gets a confidently
wrong KIND of answer.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from mi_agent.business_semantics import load_business_semantics
from mi_agent.llm_query_parser import _deterministic_parse
from mi_agent.mi_query_validator import load_mi_semantics
from mi_agent.period_change import models as m
from mi_agent.period_change import recognition as rec

SEMANTICS_PATH = (Path(__file__).resolve().parents[1]
                  / "mi_semantics_field_registry.yaml")


@pytest.fixture(scope="module")
def semantics():
    return load_mi_semantics(SEMANTICS_PATH)


@pytest.fixture(scope="module")
def registry():
    return load_business_semantics()


@pytest.fixture
def ask(semantics, registry):
    def _ask(question: str, *, view: str = "funded"):
        spec, _meta = _deterministic_parse(question, semantics)
        return rec.recognise(question, spec=spec, view=view, registry=registry)
    return _ask


# --------------------------------------------------------------------------- #
# The questions the workflow exists to answer — §1
# --------------------------------------------------------------------------- #
HEADLINE_QUESTIONS = [
    "What changed in the portfolio this month?",
    "How has the portfolio changed since the previous reporting date?",
    "What moved materially between March and June?",
    "Compare the current portfolio with the prior period.",
    "What improved and deteriorated quarter-on-quarter?",
    "What are the main drivers of the balance movement?",
    "Which risk indicators changed the most?",
    "How has portfolio composition changed?",
    "What changed year-on-year?",
    "Explain the movement in arrears, LTV, balances and product mix.",
]


@pytest.mark.parametrize("question", HEADLINE_QUESTIONS)
def test_every_headline_question_is_recognised(ask, question):
    assert ask(question).matched, question


@pytest.mark.parametrize("question", [
    "What changed since the last reporting date?",
    "Show me the movement in arrears.",
    "Which balances increased between January and April?",
    "Has credit quality deteriorated month-on-month?",
    "What are the drivers of change in the book?",
    "How did the portfolio evolve quarter on quarter?",
    "What has decreased year to date?",
    "compare 2025-12-31 with 2026-03-31",
    "Show the portfolio evolution since the last reporting date.",
])
def test_the_wider_change_vocabulary_is_recognised(ask, question):
    assert ask(question).matched, question


def test_recognition_does_not_depend_on_the_words_period_change(ask):
    assert ask("What moved in the book since the prior reporting date?").matched


# --------------------------------------------------------------------------- #
# Negative controls — §4
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("question,reason", [
    ("What is the current funded balance?", rec.DECLINE_NO_CHANGE_LANGUAGE),
    ("Summarise the portfolio.", rec.DECLINE_NO_CHANGE_LANGUAGE),
    ("show me the prior month balance", rec.DECLINE_NO_CHANGE_LANGUAGE),
    ("balance by ltv bucket", rec.DECLINE_NO_CHANGE_LANGUAGE),
    ("Forecast the funded balance for next quarter.", rec.DECLINE_FORECAST),
    ("What if the run rate doubled — how does the balance change?",
     rec.DECLINE_FORECAST),
    ("How has the balance changed over time?", rec.DECLINE_TREND_SERIES),
    ("Show funded balance evolution by month.", rec.DECLINE_TREND_SERIES),
    ("Show me the loan level tape for loans that changed status.",
     rec.DECLINE_RAW_RECORDS),
    ("Reconcile the transactions that changed the balance.",
     rec.DECLINE_TRANSACTION_RECONCILIATION),
    ("How has the balance changed across portfolios?",
     rec.DECLINE_CROSS_PORTFOLIO),
])
def test_questions_belonging_to_another_capability_are_declined(ask, question, reason):
    intent = ask(question)
    assert not intent.matched, question
    assert intent.reason == reason


def test_a_pipeline_or_forecast_view_is_declined(ask):
    """The workflow reads the funded book; it does not silently answer a
    pipeline question from funded data."""
    for view in ("pipeline", "forecast"):
        assert not ask("What changed this month?", view=view).matched


def test_a_static_two_portfolio_comparison_is_declined(ask):
    """Reuses the governed lens layer rather than re-detecting cohorts."""
    intent = ask("How does direct compare with acquired, and what changed?")
    assert not intent.matched
    assert intent.reason == rec.DECLINE_CROSS_PORTFOLIO


# --------------------------------------------------------------------------- #
# Deference to the incumbent temporal-compare route
# --------------------------------------------------------------------------- #
class TestIncumbentDeference:
    def test_a_single_metric_named_month_comparison_is_deferred(self, ask):
        intent = ask("Compare October and November funded balance.")
        assert not intent.matched

    def test_a_named_month_loan_count_comparison_is_deferred(self, ask):
        assert not ask("Compare October and November loan count.").matched

    def test_a_portfolio_wide_named_period_question_is_not_deferred(self, ask):
        """"What moved between March and June" names no metric, so it is a
        portfolio-wide change question, not the incumbent's single-metric one."""
        assert ask("What moved materially between March and June?").matched


# --------------------------------------------------------------------------- #
# Interpretation — mode and periods
# --------------------------------------------------------------------------- #
class TestModeSelection:
    def test_a_broad_question_is_a_portfolio_overview(self, ask):
        assert ask("What changed in the portfolio this month?").mode == \
            m.MODE_PORTFOLIO_OVERVIEW

    def test_a_named_concept_is_concept_mode(self, ask):
        intent = ask("What changed in credit quality?")
        assert intent.mode == m.MODE_CONCEPT
        assert intent.requested_concepts == ("credit_quality",)

    def test_a_named_metric_is_requested_metric_mode(self, ask):
        intent = ask("How did current LTV change month-on-month?")
        assert intent.mode == m.MODE_REQUESTED_METRIC
        assert intent.requested_fields == ("current_loan_to_value",)

    def test_overview_language_beats_an_incidentally_resolved_metric(self, ask):
        """"What changed in the portfolio" is an overview even when the parser
        happens to resolve a metric from a stray noun."""
        assert ask("What has changed in the portfolio overall?").mode == \
            m.MODE_PORTFOLIO_OVERVIEW

    def test_a_composition_question_asks_for_distributions(self, ask):
        intent = ask("How has portfolio composition changed?")
        assert intent.mode == m.MODE_PORTFOLIO_OVERVIEW
        assert intent.composition_focus is True

    def test_a_drivers_question_asks_for_the_balance_bridge(self, ask):
        intent = ask("What are the main drivers of the balance movement?")
        assert intent.include_bridge is True
        assert intent.requested_fields == ("current_outstanding_balance",)

    def test_only_concepts_the_registry_defines_are_used(self, ask):
        intent = ask("What changed in credit quality?")
        registry = load_business_semantics()
        assert set(intent.requested_concepts) <= set(registry.concepts())


class TestPeriodInterpretation:
    @pytest.mark.parametrize("question,mode", [
        ("What changed month-on-month?", m.METHOD_MONTH_ON_MONTH),
        ("What changed month on month?", m.METHOD_MONTH_ON_MONTH),
        ("What changed quarter-on-quarter?", m.METHOD_QUARTER_ON_QUARTER),
        ("What changed year-on-year?", m.METHOD_YEAR_ON_YEAR),
        ("What changed year to date?", m.METHOD_YEAR_TO_DATE),
        ("What changed versus the previous period?", m.METHOD_CURRENT_VS_PREVIOUS),
        ("What changed since the previous reporting date?",
         m.METHOD_CURRENT_VS_PREVIOUS),
    ])
    def test_relative_period_language(self, ask, question, mode):
        assert ask(question).period_request.relative_mode == mode

    def test_explicit_month_names_are_carried_through(self, ask):
        request = ask("What moved between March and June?").period_request
        assert (request.requested_start, request.requested_end) == ("March", "June")
        assert request.relative_mode is None

    def test_explicit_iso_dates_are_carried_through(self, ask):
        request = ask("compare 2025-12-31 with 2026-03-31").period_request
        assert (request.requested_start, request.requested_end) == (
            "2025-12-31", "2026-03-31")

    def test_a_from_to_range_is_carried_through(self, ask):
        request = ask("What changed from January to April?").period_request
        assert (request.requested_start, request.requested_end) == (
            "January", "April")

    def test_a_bare_change_question_defers_the_period_decision(self, ask):
        """No period named: the resolver's latest-available-pair rule decides,
        and it records that it did."""
        request = ask("What moved in the book?").period_request
        assert request == rec.PeriodRequest()


# --------------------------------------------------------------------------- #
# Extension point
# --------------------------------------------------------------------------- #
def test_the_semantics_context_slot_overrides_the_controlled_vocabulary(ask,
                                                                        semantics):
    """``ParsedQuestion.semantics_context`` is the documented Business Semantics
    seam; a future resolver populates it and recognition honours it, with no
    signature change here."""
    spec, _ = _deterministic_parse("What changed?", semantics)
    intent = rec.recognise(
        "What changed?", spec=spec,
        semantics_context={"period_change_concepts": ["liquidity"]})
    assert intent.matched
    assert intent.requested_concepts == ("liquidity",)
    assert intent.mode == m.MODE_CONCEPT


def test_recognition_never_raises_on_a_malformed_spec():
    assert rec.recognise("What changed?", spec=object()).matched
    assert not rec.recognise("", spec=None).matched
    assert not rec.recognise(None, spec=None).matched
