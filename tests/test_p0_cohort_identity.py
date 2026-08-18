#!/usr/bin/env python3
"""tests/test_p0_cohort_identity.py — cohort SEMANTIC IDENTITY.

The invariant P0 already held for measures, filters and dimensions, now held
for cohorts too:

    requested cohort concept
      -> resolved governed field
        -> executed grouping / comparison
          -> receipt

Before this, the guard proved only that a cohort comparison had *happened*. Any
grouping by a sourcing key satisfied any cohort question, so

    "Is the credit quality of new origination better or worse than the back
     book?"

— a question about SEASONING — was answered by splitting the book into direct
and acquired, which is how the loans were SOURCED. The arithmetic was correct
and the answer was to a different question.

Two cohorts is not the same as the right two cohorts.

Run: python -m pytest tests/test_p0_cohort_identity.py
"""
from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from demo_platform import config as cfg  # noqa: E402
from mi_agent import execution_receipt as er  # noqa: E402


@pytest.fixture(scope="module")
def governed_env():
    root = cfg.local_blob_root() / "processed" / "platform" / cfg.CLIENT_ID
    if not root.exists():
        pytest.skip("the demonstration platform has not been built — run "
                    "`python -m demo_platform.run_demo --generate --orchestrate`")
    before = dict(os.environ)
    os.environ.update(cfg.mi_env(period_role="current"))
    os.environ["MI_AGENT_DATA_CACHE_TTL"] = "0"
    os.environ["MI_AGENT_LLM_PARSER"] = "off"
    os.environ.pop("ANTHROPIC_API_KEY", None)
    logging.disable(logging.WARNING)
    from mi_agent_api import data_source

    data_source.reset_cache()
    yield
    logging.disable(logging.NOTSET)
    os.environ.clear()
    os.environ.update(before)
    data_source.reset_cache()


@pytest.fixture(scope="module")
def ask(governed_env):
    from trakt_core.context import ExecutionContext
    from mi_agent_api.mi_service import MiQueryRequest, execute_governed_mi_query

    ctx = ExecutionContext.for_internal(cfg.CLIENT_ID)
    cache: Dict[str, Dict[str, Any]] = {}

    def _ask(question: str) -> Dict[str, Any]:
        if question not in cache:
            cache[question] = (execute_governed_mi_query(
                MiQueryRequest(question=question), ctx).result or {})
        return cache[question]

    return _ask


@pytest.fixture(scope="module")
def semantics(governed_env):
    from mi_agent_api.data_source import semantics_path
    from mi_agent.mi_query_validator import load_mi_semantics

    return load_mi_semantics(semantics_path())


@pytest.fixture(scope="module")
def columns(governed_env):
    from mi_agent_api.data_source import get_dataframe

    return list(get_dataframe().columns)


def answer(envelope) -> str:
    return envelope.get("answer") or envelope.get("error") or ""


def cohort_facets(envelope):
    guard = envelope.get("semanticGuard") or {}
    return [f for f in (guard.get("facets") or [])
            if f.get("kind") == er.KIND_COHORT_COMPARISON]


# =========================================================================== #
# The concept vocabulary
# =========================================================================== #
@pytest.mark.parametrize("question,expected", [
    ("Is the credit quality of new origination better or worse than the back "
     "book?", "vintage"),
    ("How does new lending compare with the seasoned book on LTV?", "vintage"),
    ("Compare new origination against the back book on LTV.", "vintage"),
    ("How does the front book compare with the back book?", "vintage"),
    ("What is the average LTV of the direct book vs the acquired book?",
     "sourcing"),
    ("How does the direct book compare with the acquired book on borrower age?",
     "sourcing"),
    ("Compare the direct and acquired books on balance.", "sourcing"),
    ("How does the purchased book compare with organic lending?", "sourcing"),
])
def test_the_cohort_concept_is_identified(question, expected):
    named = [c for c, _desc, _fields in er.cohort_concepts_named(question)]
    assert expected in named, f"{question!r} -> {named}"


@pytest.mark.parametrize("question", [
    "Show me balance by region",
    "What is the total balance?",
    "Show me balance, loan count and weighted-average LTV by region.",
    "What is the largest single-loan exposure and what share of the book is it?",
    "Which region has grown the most in the last month?",
])
def test_a_question_with_no_cohort_names_no_concept(question):
    """Scope control: the vocabulary must not read cohorts into questions that
    do not have them."""
    assert er.cohort_concepts_named(question) == []


def test_a_cohort_concept_alone_is_not_a_comparison(semantics):
    """"The acquired book's LTV" asks about ONE cohort. The facet is raised
    only when the question also sets two things against each other, otherwise
    every scoped question would demand a comparison it never asked for."""
    facets = er.detect_requested_facets(
        "What is the weighted average LTV of the acquired book?", semantics)
    assert not [f for f in facets if f.kind == er.KIND_COHORT_COMPARISON]


# =========================================================================== #
# The invariant, end to end
# =========================================================================== #
VINTAGE_QUESTIONS = [
    "Is the credit quality of new origination better or worse than the back book?",
    "How does new lending compare with the seasoned book on LTV?",
    "Compare new origination against the back book on LTV.",
]


@pytest.mark.parametrize("question", VINTAGE_QUESTIONS)
def test_a_vintage_cohort_is_never_answered_by_sourcing(ask, question):
    """The defect this file exists for, restated after P1J-1.

    Vintage and provenance are INDEPENDENT axes. Before P1J-1 this book carried
    no governed vintage dimension, so the only safe outcome was a refusal naming
    the concept. P1J-1 governs seasoning, so these questions now ANSWER — but the
    invariant is unchanged and is what this asserts: a vintage question is never
    answered by sourcing. Whether it refuses or answers, the one thing it may
    never do is split the book by direct/acquired and call that a vintage
    comparison.
    """
    envelope = ask(question)
    text = answer(envelope).lower()
    spec = envelope.get("spec") or {}
    grouping = [spec.get("dimension")] + list(spec.get("dimensions") or [])
    filters = spec.get("filters") or {}

    # Never answered by provenance — the enduring invariant.
    assert "direct vs acquired" not in text
    assert "source_portfolio_type" not in grouping
    assert "source_portfolio_type" not in filters

    if envelope["ok"]:
        # Answered: it must have been answered on the VINTAGE axis.
        assert any(g in ("seasoning_segment", "seasoning_bucket", "vintage_year")
                   for g in grouping if g), (
            f"vintage question answered without a vintage grouping: {grouping!r}")
    else:
        # Refused: it must name the concept, not a field it reached for.
        assert "how long the loans have been on the book" in text


@pytest.mark.parametrize("question", [
    "What is the average LTV of the direct book vs the acquired book?",
    "How does the direct book compare with the acquired book on borrower age?",
    "Compare the direct and acquired books on balance, loan count, "
    "weighted-average LTV and average borrower age.",
])
def test_a_sourcing_cohort_still_answers(ask, question):
    """The other half of the invariant. Tightening identity must not refuse the
    comparisons the capability genuinely expresses."""
    envelope = ask(question)
    assert envelope["ok"] is True, envelope.get("error")
    facets = cohort_facets(envelope)
    assert facets and all(f.get("status") == "applied" for f in facets)


def test_the_applied_cohort_facet_names_the_concept(ask):
    envelope = ask("What is the average LTV of the direct book vs the "
                   "acquired book?")
    labels = [f.get("label") for f in cohort_facets(envelope)]
    assert any("sourced" in (l or "") for l in labels), labels


# =========================================================================== #
# Deliberately wrong substitutions
# =========================================================================== #
def _facet(concept, fields):
    return er.RequestedFacet(
        kind=er.KIND_COHORT_COMPARISON, label="a comparison by seasoning",
        concepts=(concept,), alt_keys=tuple(fields))


class _Result:
    def __init__(self, group_keys):
        self.metadata = {"group_field_keys": list(group_keys)}
        self.data = None
        self.result_type = "grouped"


def test_sourcing_grouping_cannot_satisfy_a_vintage_facet(semantics, columns):
    """The exact substitution B04 was making."""
    reconciled = er.reconcile_facets(
        [_facet("vintage", ("vintage_year", "origination_date"))],
        spec=object(), query_result=_Result(["source_portfolio_type"]),
        semantics=semantics, available_columns=columns)
    assert reconciled[0].status != er.APPLIED


def test_a_geography_grouping_cannot_satisfy_a_cohort_facet(semantics, columns):
    reconciled = er.reconcile_facets(
        [_facet("sourcing", ("source_portfolio_type",))],
        spec=object(), query_result=_Result(["collateral_geography"]),
        semantics=semantics, available_columns=columns)
    assert reconciled[0].status != er.APPLIED


def test_the_right_grouping_does_satisfy_it(semantics, columns):
    reconciled = er.reconcile_facets(
        [_facet("sourcing", ("source_portfolio_type",))],
        spec=object(), query_result=_Result(["source_portfolio_type"]),
        semantics=semantics, available_columns=columns)
    assert reconciled[0].status == er.APPLIED


def test_an_unavailable_cohort_concept_is_named_not_substituted(semantics,
                                                                columns):
    """A concept whose fields the book does not carry is named, not substituted.

    Since P1J-1 this tape DOES carry vintage, so the unavailable case is proven
    with a concept it genuinely lacks: the facet must report the concept and the
    fields it would need, rather than being marked applied by a split that
    happens to exist.
    """
    reconciled = er.reconcile_facets(
        [_facet("borrower structure", ("borrower_type", "number_of_borrowers"))],
        spec=object(), query_result=_Result([]),
        semantics=semantics, available_columns=columns)
    assert reconciled[0].status == er.UNAVAILABLE
    assert "no governed dimension" in reconciled[0].reason


def test_a_governed_vintage_dimension_is_now_available(semantics, columns):
    """The other side of the same guard: P1J-1 materialises vintage on this
    tape, so the vintage concept must NOT report itself unavailable."""
    assert "vintage_year" in columns and "seasoning_segment" in columns
    assert er._cohort_fields_available(
        ("vintage_year", "seasoning_segment"), columns, semantics)


def test_a_raw_origination_date_is_not_a_vintage_dimension(columns):
    """This tape HAS ``origination_date``. A date is not a cohort: splitting a
    book by it needs a derived year or seasoning band, and none is governed
    here. Listing the date as expressing the concept would make the guard say
    "we could have and didn't" when the truth is that it cannot."""
    from mi_agent.execution_receipt import _COHORT_CONCEPTS

    vintage = next(fields for concept, _d, _p, fields in _COHORT_CONCEPTS
                   if concept == "vintage")
    assert "origination_date" in columns
    assert "origination_date" not in vintage


def test_a_cohort_facet_that_is_not_applied_refuses():
    """A cohort is a SUBJECT facet: the answer is about those two populations,
    so losing it changes what the answer is about, not merely its shape."""
    assert er.KIND_COHORT_COMPARISON in er.NUMBER_OR_SUBJECT_FACETS
    assert er.KIND_COHORT_COMPARISON not in er.SHAPE_FACETS


# =========================================================================== #
# The routed path
# =========================================================================== #
def test_the_comparison_route_declares_which_cohort_it_compares(ask):
    """Evidence, not assumption. The route splits by portfolio scope, so it
    expresses sourcing; it says so, and the guard reads that rather than
    inferring it from the route's name."""
    envelope = ask("What is the average LTV of the direct book vs the "
                   "acquired book?")
    declared = (envelope.get("metadata") or {}).get("portfolioComparison") or {}
    assert declared.get("cohortConcept") == "sourcing"


def test_a_routed_sourcing_comparison_cannot_satisfy_a_vintage_facet(semantics,
                                                                     columns):
    """Even a perfectly executed portfolio comparison does not answer a
    seasoning question."""
    envelope = {"metadata": {"portfolioComparison": {
        "cohortConcept": "sourcing", "measuresCompared": ["current_loan_to_value"],
        "requestedMetric": "current_loan_to_value",
        "requestedMetricCompared": True}}}
    reconciled = er.reconcile_routed_facets(
        [_facet("vintage", ("vintage_year",))],
        route="portfolio_risk_comparison", semantics=semantics,
        available_columns=columns, envelope=envelope)
    assert reconciled[0].status == er.LOST
    assert "sourcing" in reconciled[0].reason


def test_a_routed_comparison_that_names_no_cohort_proves_nothing(semantics,
                                                                 columns):
    """A cohort route that does not state which cohorts it compared is not
    evidence that it compared the requested ones."""
    reconciled = er.reconcile_routed_facets(
        [_facet("vintage", ("vintage_year",))],
        route="portfolio_risk_comparison", semantics=semantics,
        available_columns=columns, envelope={"metadata": {}})
    assert reconciled[0].status == er.LOST
