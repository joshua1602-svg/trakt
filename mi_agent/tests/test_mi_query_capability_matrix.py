#!/usr/bin/env python3
"""mi_agent/tests/test_mi_query_capability_matrix.py

The MI Query Agent's semantic contract, expressed as a question bank.

Two kinds of test live here, deliberately together:

* **Guards** — behaviour that is correct and must stay correct. Several of these
  encode defects found during the production-readiness review and fixed with it;
  each names the defect so a future change cannot re-introduce it by accident.
* **Tracked gaps** — capabilities the agent does not have yet, recorded as
  ``xfail(strict=True)``. They are not silent TODOs: the suite fails if one is
  implemented without being promoted here, so the capability matrix in
  ``docs/mi_query_agent_production_readiness_review.md`` cannot drift away from
  what the agent actually does.

Everything runs through the real deterministic parser and the real registry.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from mi_agent import portfolio_lens as pl
from mi_agent.llm_query_parser import _deterministic_parse
from mi_agent.mi_query_validator import load_mi_semantics

_SEMANTICS = _REPO_ROOT / "mi_agent" / "mi_semantics_field_registry.yaml"


@pytest.fixture(scope="module")
def semantics():
    return load_mi_semantics(_SEMANTICS)


def parse(question, semantics):
    return _deterministic_parse(question, semantics)


def spec_of(question, semantics):
    return parse(question, semantics)[0]


def note_of(question, semantics):
    return parse(question, semantics)[1].get("note")


# --------------------------------------------------------------------------- #
# 1. Portfolio vocabulary must not capture measure vocabulary
# --------------------------------------------------------------------------- #
# Defect S1: "origination", "originated", "new lending" and "current book" were
# Direct-lens keywords. "show origination volumes by month" therefore silently
# filtered a Total workspace down to the direct portfolios — a scope mutation the
# user never asked for and was never told about.
@pytest.mark.parametrize("question", [
    "show origination volumes by month",
    "what were originations last month?",
    "show new lending by region",
    "what is the current book balance?",
    "show the origination trend",
    "how much did we originate?",
])
def test_measure_vocabulary_is_not_read_as_a_portfolio_scope(question):
    assert not pl.mentions_portfolio(question), (
        f"{question!r} is a measure question; reading it as a portfolio scope "
        "silently narrows the answer")
    assert pl.resolve_lens(question).name == pl.LENS_TOTAL


@pytest.mark.parametrize("question,expected", [
    ("show the direct book balance", pl.LENS_DIRECT),
    ("balance for directly originated loans", pl.LENS_DIRECT),
    ("originated loans, organic book", pl.LENS_DIRECT),
    ("show our own book", pl.LENS_DIRECT),
    ("balance for the acquired book", pl.LENS_ACQUIRED),
    ("balance across all portfolios", pl.LENS_TOTAL),
])
def test_portfolio_qualified_phrases_still_resolve(question, expected):
    """Narrowing the keyword list must not cost genuine scope recognition."""
    assert pl.resolve_lens(question).name == expected
    assert pl.mentions_portfolio(question)


@pytest.mark.parametrize("question", [
    "show the back book",
    "show the legacy book",
    "balance for the seasoned book",
    "show newly originated loans",
])
def test_a_seasoning_phrase_is_not_a_portfolio_scope(question):
    """P1J-1 (ruling R2a): front/back is the VINTAGE axis and direct/acquired is
    PROVENANCE. They are independent — a directly-originated loan written five
    years ago is DIRECT and BACK BOOK — so "the back book" must not resolve to
    the acquired lens. This case previously asserted the opposite, which is the
    conflation itself: it silently excluded every seasoned direct loan and
    included every recent acquired one. Seasoning resolves through
    mi_agent.seasoning; see tests/test_p1j1_vintage_seasoning.py."""
    assert pl.resolve_lens(question).name == pl.LENS_TOTAL


def test_an_exact_cohort_id_always_wins(semantics):
    assert pl.resolve_lens("balance for acquired_002").cohort_id == "acquired_002"
    assert pl.resolve_lens("direct_003 only").cohort_id == "direct_003"


def test_a_named_scope_overrides_the_workspace_default():
    default = pl.lens_from_selection("direct")
    assert pl.resolve_lens_with_default("the acquired book", default).name == pl.LENS_ACQUIRED
    # …and an unqualified question leaves the workspace selection alone.
    assert pl.resolve_lens_with_default("show balance by region", default).name == pl.LENS_DIRECT


# --------------------------------------------------------------------------- #
# 2. A predicate is resolved against its own clause, or refused
# --------------------------------------------------------------------------- #
# Defect S3: _filter_field_of scanned the WHOLE question, and "where" was not a
# clause boundary. "balance by region where flurb is above 3" bound the threshold
# to current_outstanding_balance — filtering the answer by a predicate the user
# never expressed.
def test_an_unknown_condition_never_binds_to_another_column(semantics):
    spec = spec_of("show balance by region where flurb is above 3", semantics)
    assert spec.filters == {}, (
        f"an unrecognised condition was bound to {list(spec.filters)}")
    assert spec.unavailable_filters, "the unapplied filter must be surfaced"
    assert "flurb" in " ".join(spec.unavailable_filters)


def test_a_recognised_condition_in_a_where_clause_still_applies(semantics):
    spec = spec_of("show balance by region where LTV is above 50%", semantics)
    assert "current_loan_to_value" in spec.filters
    assert spec.filters["current_loan_to_value"] == {"op": "gt", "value": 50.0}
    assert not spec.unavailable_filters


@pytest.mark.parametrize("question,field", [
    ("how many loans with youngest age more than 70", "youngest_borrower_age"),
    ("show loans with LTV above 50%", "current_loan_to_value"),
    ("how much balance is over 100000", "current_outstanding_balance"),
    ("how many loans where the interest rate is above 5", "current_interest_rate"),
])
def test_threshold_clauses_resolve_to_their_own_field(semantics, question, field):
    assert field in spec_of(question, semantics).filters


def test_a_question_that_is_only_an_unknown_predicate_is_refused(semantics):
    """When the filter IS the question, answering it unfiltered answers a
    different question. Refuse instead."""
    assert note_of("how many loans have Risk Score above 700", semantics) == "unmapped"
    assert note_of("how many loans have a widget index over 3", semantics) == "unmapped"


def test_a_valid_question_narrowed_by_an_unknown_predicate_still_answers(semantics):
    """The complement: the breakdown stands, with the unapplied filter disclosed."""
    spec = spec_of("show balance by region where flurb is above 3", semantics)
    assert spec.dimension, "the grouping is still a valid question"
    assert spec.unavailable_filters


# --------------------------------------------------------------------------- #
# 3. Rankings stay rankings
# --------------------------------------------------------------------------- #
# Defect S2: the ITL3 geographic-exposure route captured any question containing
# a geo term and a superlative, discarding top_n, the metric, the filters and the
# portfolio lens.
@pytest.mark.parametrize("question,limit", [
    ("show top 5 regions by balance", 5),
    ("show the top 10 areas by exposure", 10),
    ("show the bottom 5 regions by balance", 5),
])
def test_an_explicit_grouped_ranking_is_not_captured_by_the_geo_route(question, limit):
    from mi_agent_api.chat_routing import _is_geo_exposure

    assert _is_geo_exposure(question) is False, (
        "a ranking question was routed to the ITL3 engine, which ignores "
        "top_n, the metric and the portfolio lens")


@pytest.mark.parametrize("question", [
    "where is the largest geographic concentration?",
    "which region are we most exposed to?",
    "where is the largest area exposure?",
])
def test_genuine_concentration_questions_still_reach_the_geo_route(question):
    from mi_agent_api.chat_routing import _is_geo_exposure

    assert _is_geo_exposure(question) is True


@pytest.mark.parametrize("question,limit", [
    ("show top 5 regions by balance", 5),
    ("show the bottom 5 regions by balance", 5),
    ("show top 10 brokers by balance", 10),
])
def test_a_ranking_keeps_its_limit(semantics, question, limit):
    spec = spec_of(question, semantics)
    assert (spec.top_n or spec.limit) == limit


def test_bottom_n_keeps_both_its_direction_and_its_limit(semantics):
    """``_detect_top_n`` matched only "top N", so a Bottom-N question kept its
    ascending sort and silently returned every group."""
    spec = spec_of("show the bottom 5 regions by balance", semantics)
    assert spec.sort_direction == "asc"
    assert (spec.top_n or spec.limit) == 5


def test_ranking_direction_is_honoured(semantics):
    assert spec_of("show the smallest 5 regions by balance", semantics).sort_direction == "asc"
    assert spec_of("show the largest 5 regions by balance", semantics).sort_direction == "desc"


# --------------------------------------------------------------------------- #
# 4. The implemented capability set — these must not regress
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("question,check", [
    ("what is the total current balance?",
     lambda s: s.metric == "current_outstanding_balance" and s.aggregation == "sum"),
    ("what is the weighted average LTV?",
     lambda s: s.aggregation == "weighted_avg" and s.weight_field),
    ("what is the average borrower age?",
     lambda s: s.metric == "youngest_borrower_age" and s.aggregation == "avg"),
    ("what is the weighted average interest rate?",
     lambda s: s.metric == "current_interest_rate" and s.aggregation == "weighted_avg"),
    ("show balance by region", lambda s: s.dimension and s.chart_type == "bar"),
    ("show balance by age bucket", lambda s: s.dimension == "age_bucket"),
    ("show balance over time", lambda s: s.chart_type == "line"),
    ("show balance by region and age bucket",
     lambda s: len(s.dimensions or []) == 2 or s.chart_type == "heatmap"),
    ("show balance by region and broker as a treemap",
     lambda s: s.chart_type == "treemap" and len(s.hierarchy or []) >= 2),
    ("show LTV by borrower age sized by balance",
     lambda s: s.chart_type in ("bubble", "scatter") and s.x and s.y),
    ("how many loans are there?", lambda s: s.aggregation == "count"),
    ("show loans with LTV above 50%",
     lambda s: s.aggregation == "loan_level" and s.filters),
    ("portfolio summary", lambda s: s.intent == "summary"),
])
def test_implemented_capabilities_hold(semantics, question, check):
    assert check(spec_of(question, semantics)), question


@pytest.mark.parametrize("question", [
    "what is the weather like?",
    "tell me a joke",
    "what should I have for lunch?",
])
def test_an_uninterpretable_question_is_refused_not_answered(semantics, question):
    assert note_of(question, semantics) == "unmapped"


# --------------------------------------------------------------------------- #
# 5. Tracked capability gaps
# --------------------------------------------------------------------------- #
# Recorded as strict xfails so the documented capability matrix cannot drift:
# implementing one of these fails this suite until the expectation is promoted.
_GAPS = [
    pytest.param("show balance by quarter", "quarterly time grain",
                 marks=pytest.mark.xfail(strict=True, reason="quarterly grain")),
    pytest.param("show balance year on year", "year-on-year comparison",
                 marks=pytest.mark.xfail(strict=True, reason="YoY comparison")),
    pytest.param("how many distinct brokers are there?", "count_distinct",
                 marks=pytest.mark.xfail(strict=True, reason="count_distinct unreachable")),
    pytest.param("what percentage of the book is in the South East?", "share of book",
                 marks=pytest.mark.xfail(strict=True, reason="share-of-book intent")),
    pytest.param("why did the balance fall?", "explanatory intent",
                 marks=pytest.mark.xfail(strict=True, reason="explanatory intent")),
    pytest.param("what are the main drivers of the balance movement?", "driver intent",
                 marks=pytest.mark.xfail(strict=True, reason="driver attribution intent")),
    pytest.param("what explains the change in balance?", "explanatory intent",
                 marks=pytest.mark.xfail(strict=True, reason="explanatory intent")),
]


@pytest.mark.parametrize("question,capability", _GAPS)
def test_tracked_capability_gaps(semantics, question, capability):
    """A gap must REFUSE rather than answer a different question confidently.

    Every case here currently returns a plausible, unrelated figure with
    ``ok:true`` — the most damaging failure mode an MI agent has. The assertion
    is the behaviour we want (an honest refusal); the strict xfail records that
    we do not have it yet.
    """
    assert note_of(question, semantics) == "unmapped", (
        f"{capability!r} is not implemented, so {question!r} must be refused "
        "rather than answered with an unrelated number")


def test_the_gap_list_matches_the_documented_capability_matrix():
    """The review document and this suite must describe the same agent."""
    review = (_REPO_ROOT / "docs"
              / "mi_query_agent_production_readiness_review.md").read_text(encoding="utf-8")
    for marker in ("Percentages / share-of-book", "Count distinct",
                   "quarterly", "year-on-year", "Explanatory intents"):
        assert marker.lower() in review.lower(), (
            f"{marker!r} is tracked as a gap here but is not in the review's "
            "capability matrix")
