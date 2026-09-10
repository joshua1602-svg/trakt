"""C, D, E · Unregistered, ambiguous and unsupported all fail closed.

The common rule under all three: the compiler never reaches for the nearest
available semantics to avoid refusing. Each test therefore asserts BOTH that the
outcome is not a plan AND that no plan object came back — a refusal that still
carried something runnable would be the failure mode these codes exist to
prevent.
"""

from __future__ import annotations

import pytest

from mi_agent.interpretation_v2 import (
    OUTCOME_CLARIFY,
    OUTCOME_PLAN,
    OUTCOME_REFUSE,
    CompilerContext,
    DeterministicCompiler,
    load_governed_vocabulary,
)

from .conftest import build_intent


# --------------------------------------------------------------------------- #
# C · unregistered concepts
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("term", [
    "ebitda", "churn_rate", "customer_satisfaction", "net_promoter_score",
    "balance_sheet_equity_ratio",
])
def test_an_unregistered_measure_refuses(term, compiler):
    result = compiler.compile(build_intent(measures=[{"concept": term}]))
    assert result.outcome == OUTCOME_REFUSE
    assert "UNREGISTERED_CONCEPT" in result.codes()
    assert result.plan is None


def test_an_unregistered_dimension_refuses(compiler):
    intent = build_intent(operation="breakdown", dimensions=["sales_territory"])
    result = compiler.compile(intent)
    assert result.outcome == OUTCOME_REFUSE
    assert "UNREGISTERED_CONCEPT" in result.codes()


def test_an_unregistered_filter_concept_refuses(compiler):
    intent = build_intent(filters=[{"concept": "credit_score_band",
                                    "comparator": "eq", "value": "A"}])
    result = compiler.compile(intent)
    assert result.outcome == OUTCOME_REFUSE
    assert "UNREGISTERED_CONCEPT" in result.codes()


def test_a_concept_absent_from_THIS_book_is_a_different_refusal(vocabulary):
    """"governed but not here" and "not governed anywhere" are not one answer.

    Collapsing them would tell a client that Trakt does not know what an
    indexed LTV is, when the truth is that their tape does not carry one.
    """
    scoped = CompilerContext(
        vocabulary, available_fields=["current_outstanding_balance"])
    compiler = DeterministicCompiler(scoped)
    result = compiler.compile(build_intent(measures=[{"concept": "indexed_ltv"}]))
    assert result.outcome == OUTCOME_REFUSE
    assert result.codes() == ["CONCEPT_UNAVAILABLE"]
    assert "UNREGISTERED_CONCEPT" not in result.codes()


# --------------------------------------------------------------------------- #
# D · ambiguous binding
# --------------------------------------------------------------------------- #

def test_a_term_two_governed_fields_both_claim_is_not_bound(vocabulary):
    """"Region" is claimed by two governed fields, so it is not a term at all.

    The vocabulary drops a colliding business name rather than picking one, so
    the compiler answers UNREGISTERED_CONCEPT instead of binding whichever came
    first in the file.
    """
    assert vocabulary.resolve("region") is None


@pytest.mark.parametrize("level", ["nuts3", "itl3"])
def test_geography_without_a_basis_clarifies_at_a_level_that_has_two(level, compiler):
    """At NUTS3 the borrower's region and the property's region both exist."""
    intent = build_intent(operation="breakdown",
                          geography={"requested": True, "level": level,
                                     "group_by": True})
    result = compiler.compile(intent)
    assert result.outcome == OUTCOME_CLARIFY
    assert "AMBIGUOUS_GEOGRAPHY" in result.codes()
    assert result.plan is None
    reason = next(r for r in result.reasons if r.code == "AMBIGUOUS_GEOGRAPHY")
    assert set(reason.spans) == {"obligor", "collateral"}


def test_a_blocking_model_ambiguity_clarifies(compiler):
    intent = build_intent(ambiguity=[{"slot": "measures", "blocking": True,
                                      "note": "could not tell which balance",
                                      "options": ["balance", "principal_balance"]}])
    result = compiler.compile(intent)
    assert result.outcome == OUTCOME_CLARIFY
    assert "MODEL_FLAGGED_AMBIGUITY" in result.codes()


def test_a_disclosed_reading_is_not_a_clarification(compiler):
    """A non-blocking note is a disclosure, and disclosure is not refusal."""
    intent = build_intent(ambiguity=[{"slot": "population.base", "blocking": False,
                                      "note": "funded book assumed"}])
    result = compiler.compile(intent)
    assert result.outcome == OUTCOME_PLAN
    assert any("funded book assumed" in note
               for note in result.plan.provenance.notes)


def test_an_unresolved_period_refuses(compiler):
    intent = build_intent(time={"form": "explicit_period"})
    result = compiler.compile(intent)
    assert result.outcome == OUTCOME_REFUSE
    assert "PERIOD_UNRESOLVED" in result.codes()


def test_a_span_with_no_span_clarifies(compiler):
    intent = build_intent(operation="series", time={"form": "series"})
    result = compiler.compile(intent)
    assert "AMBIGUOUS_PERIOD" in result.codes()
    assert result.plan is None


# --------------------------------------------------------------------------- #
# E · unsupported composition
# --------------------------------------------------------------------------- #

def test_an_operation_outside_its_capability_refuses(compiler):
    intent = build_intent(capability="portfolio_summary", operation="bridge")
    result = compiler.compile(intent)
    assert result.outcome == OUTCOME_REFUSE
    assert "UNSUPPORTED_OPERATION" in result.codes()


def test_a_statistic_the_registry_forbids_refuses(compiler):
    """A region cannot be summed, and "sum of product type" is not a question
    with a near-enough answer."""
    intent = build_intent(measures=[{"concept": "product_type",
                                     "statistic": "weighted_average"}])
    result = compiler.compile(intent)
    assert result.outcome == OUTCOME_REFUSE
    assert result.plan is None


def test_a_weight_on_a_statistic_that_takes_none_refuses(compiler):
    intent = build_intent(measures=[{"concept": "balance", "statistic": "sum",
                                     "weight": "balance"}])
    result = compiler.compile(intent)
    assert result.outcome == OUTCOME_REFUSE
    assert "WEIGHT_NOT_PERMITTED" in result.codes()


def test_a_breakdown_with_nothing_to_group_by_refuses(compiler):
    result = compiler.compile(build_intent(operation="breakdown"))
    assert result.outcome == OUTCOME_REFUSE
    assert "UNSUPPORTED_COMPOSITION" in result.codes()


def test_a_point_in_time_with_a_grouping_refuses_rather_than_reinterpreting(compiler):
    """A grouped point-in-time is a breakdown. Silently promoting it would be
    the compiler deciding what the question meant."""
    intent = build_intent(operation="point_in_time", dimensions=["product_type"])
    result = compiler.compile(intent)
    assert result.outcome == OUTCOME_REFUSE
    assert "UNSUPPORTED_COMPOSITION" in result.codes()


def test_a_movement_over_a_single_period_refuses(compiler):
    intent = build_intent(capability="period_movement", operation="movement",
                          time={"form": "current"})
    result = compiler.compile(intent)
    assert result.outcome == OUTCOME_REFUSE
    assert "UNSUPPORTED_COMPOSITION" in result.codes()


def test_an_ordered_comparator_on_an_unordered_dimension_refuses(compiler):
    intent = build_intent(filters=[{"concept": "product_type",
                                    "comparator": "gt", "value": "drawdown"}])
    result = compiler.compile(intent)
    assert result.outcome == OUTCOME_REFUSE
    assert "UNSUPPORTED_FILTER" in result.codes()


def test_between_needs_exactly_two_bounds(compiler):
    intent = build_intent(filters=[{"concept": "current_ltv",
                                    "comparator": "between", "value": [50]}])
    result = compiler.compile(intent)
    assert result.outcome == OUTCOME_REFUSE
    assert "UNSUPPORTED_FILTER" in result.codes()


def test_an_unavailable_capability_refuses(vocabulary):
    scoped = CompilerContext(vocabulary, capabilities=["generic_analysis"])
    compiler = DeterministicCompiler(scoped)
    intent = build_intent(capability="borrowing_base", operation="headroom")
    result = compiler.compile(intent)
    assert result.outcome == OUTCOME_REFUSE
    assert "CAPABILITY_UNAVAILABLE" in result.codes()


def test_one_non_clarifiable_reason_refuses_the_whole_compilation(compiler):
    """A question that is partly unanswerable is not answered in part."""
    intent = build_intent(
        measures=[{"concept": "balance"}, {"concept": "ebitda"}])
    result = compiler.compile(intent)
    assert result.outcome == OUTCOME_REFUSE
    assert result.plan is None


def test_a_refusal_never_carries_a_plan(compiler):
    for intent in (build_intent(measures=[{"concept": "ebitda"}]),
                   build_intent(operation="breakdown"),
                   build_intent(capability="portfolio_summary", operation="bridge")):
        result = compiler.compile(intent)
        assert result.outcome != OUTCOME_PLAN
        assert result.plan is None
        assert result.reasons
