"""tests/test_p0_segment_pair_routing.py — naming both sides IS the comparison.

The defect this pins
--------------------
"How have direct and acquired balances moved over the periods?" was REFUSED:

    "I understood that you asked for Direct and Acquired tracked separately,
     but that could not be applied to the calculation."

True of the route it took, FALSE of the product — "Compare balance over time
for direct and acquired" is the same request with a comparison verb, and it
composes and answers completely.

Three defects were stacked behind that one refusal:

1. the metric did not resolve, because the registry carries "balance" and a
   lender typed "balances";
2. `planner._population_pair` required a comparison VERB before it would accept
   a direct/acquired pair it could already see;
3. `intent.classify` raised no `SIGNAL_COMPARISON` for the same reason, so
   `_plan_population_movement_comparison` declined even once the pair resolved.

**The P0 guard is not weakened by any of this.** It still refuses a whole-book
series returned for a segmented request. It stops firing on this question
because the route now produces the segmented series it was protecting — which
these tests assert directly.
"""
from __future__ import annotations

import pytest

from mi_workflows.analytical import intent as I


# --------------------------------------------------------------------------- #
# 1. THE COMPARISON DEFINITION
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("question", [
    "How have direct and acquired balances moved over the periods?",
    "how have direct and acquired balances moved",
    "direct and acquired over time",
    "How has the front book moved over time compared with the back book?",
    "front book and back book over the periods",
])
def test_naming_both_sides_of_a_governed_pair_is_comparative(question):
    assert I.is_comparative(question) is True


@pytest.mark.parametrize("question", [
    # TWO DIMENSIONS coordinated by "and" is NOT two values of one dimension.
    # These are the P0 time-axis refusals and they must stay refusals.
    "Show me balance by month by region and LTV band",
    "balance by month broken down by LTV band and region",
    "balance over time by region and ticket size",
    # One side only is not a comparison.
    "balance over time for direct",
    "How has balance moved over time for the front book?",
    "Show me balance by month",
])
def test_these_are_not_comparative(question):
    assert I.is_comparative(question) is False


def test_the_p0_time_axis_refusals_name_no_side_of_any_pair():
    """Structural, not incidental: the two P0 refusals that must hold cannot
    reach the pair branch at all, because they name no value of either governed
    binary dimension."""
    for q in ("Show me balance by month by region and LTV band",
              "balance by month broken down by LTV band and region"):
        assert I.names_both_sides_of_a_pair(I.normalise(q)) is False


def test_classify_and_is_comparative_cannot_drift():
    """They read the same sentence through the same definition. A second copy of
    the vocabulary is how they came to disagree in the first place."""
    for q in ("How have direct and acquired balances moved over the periods?",
              "Show me balance by month by region and LTV band",
              "Compare balance over time for direct and acquired"):
        reading = I.classify(q)
        assert (I.SIGNAL_COMPARISON in reading.signals) == I.is_comparative(q)


# --------------------------------------------------------------------------- #
# 2. THE PLURAL THAT COST A METRIC
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def semantics():
    from pathlib import Path
    from mi_agent.mi_query_validator import load_mi_semantics
    path = (Path(__file__).resolve().parent.parent / "mi_agent"
            / "mi_semantics_field_registry.yaml")
    return load_mi_semantics(str(path))


def _metric(question, semantics):
    from mi_agent import llm_query_parser as P
    spec, _meta = P.parse_with_repair(question, semantics, llm_enabled=False)
    return getattr(spec, "metric", None)


@pytest.mark.parametrize("plural,singular", [
    ("How have direct and acquired balances moved over the periods?",
     "How have direct and acquired balance moved over the periods?"),
    ("Compare balances over time for direct and acquired",
     "Compare balance over time for direct and acquired"),
    ("show me balances by region", "show me balance by region"),
])
def test_a_plural_resolves_the_same_metric_as_its_singular(plural, singular,
                                                           semantics):
    """A lender types plurals. Before this, "balances" resolved NO metric, and a
    question with no metric cannot compose an analytical plan."""
    singular_metric = _metric(singular, semantics)
    assert singular_metric is not None, \
        "the singular must resolve, or this test proves nothing"
    assert _metric(plural, semantics) == singular_metric


def test_a_word_that_is_not_a_metric_still_resolves_none(semantics):
    """The plural rule must not turn every trailing -s into a measure."""
    assert _metric("show me widgets by region", semantics) is None


def test_a_generic_plural_is_still_dropped_as_generic(semantics):
    """"totals" is the plural of a token the registry deliberately refuses to
    treat as a metric name; pluralising must not smuggle it back in."""
    from mi_agent.llm_query_parser import _registry_metric_terms
    terms = _registry_metric_terms(semantics)
    for generic in ("total", "value", "amount", "loan"):
        assert generic not in terms
        assert generic + "s" not in terms, \
            "the plural of a generic token is just as generic"
