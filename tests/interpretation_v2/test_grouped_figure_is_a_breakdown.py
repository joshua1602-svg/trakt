"""A grouped single figure is a breakdown (normalisation rule 7, P0 design §24).

The contract let one request be written two ways — `point_in_time` with a
grouping, and `breakdown` with the same grouping — and refused the first with a
reason that named the second ("a grouping makes output 'primary' a breakdown,
not a point_in_time"). Must-answer [135] was refused on exactly that on the
2026-09-30 10:26 check. The operation label now has one canonical value when
every output groups, and nothing else about the reading moves.
"""
from __future__ import annotations

import pytest

from mi_agent.interpretation_v2.compiler import (
    CAPABILITY_OPERATIONS, CompilerContext, DeterministicCompiler)
from mi_agent.interpretation_v2.intent import parse_candidate_intent
from mi_agent.interpretation_v2.normalise import canonical_intent
from mi_agent.interpretation_v2.outcomes import UNSUPPORTED_COMPOSITION
from mi_agent.interpretation_v2.vocabulary import load_governed_vocabulary

_VOCAB = load_governed_vocabulary()


def _intent(**over):
    payload = {"schema_version": "candidate_intent/1.0",
               "capability": "generic_analysis", "operation": "point_in_time",
               "population": {"base": "funded"},
               "measures": [{"concept": "current_outstanding_balance",
                             "statistic": "sum"}],
               "time": {"form": "current"}}
    payload.update(over)
    return parse_candidate_intent(payload)


def _normalise(intent):
    return canonical_intent(intent, _VOCAB,
                            capability_operations=CAPABILITY_OPERATIONS)


def _compile(intent):
    return DeterministicCompiler(CompilerContext()).compile(intent)


def test_a_grouped_single_figure_compiles_as_the_breakdown_it_names():
    grouped = _intent(dimensions=["property_region"])
    stated = _intent(operation="breakdown", dimensions=["property_region"])
    a, b = _compile(grouped), _compile(stated)
    assert a.plan is not None, [(r.code, r.detail) for r in a.reasons]
    assert a.plan.plan_id == b.plan.plan_id
    assert a.intent.operation == "point_in_time"      # the claim survives


def test_a_geography_grouping_counts_as_a_grouping():
    result = _normalise(_intent(geography={"requested": True,
                                           "group_by": True}))
    assert result.intent.operation == "breakdown"


def test_an_ungrouped_single_figure_is_untouched():
    result = _normalise(_intent())
    assert result.intent.operation == "point_in_time" and not result.applied


def test_a_filter_is_not_a_grouping():
    """"The balance in London" is one figure, and stays one."""
    result = _normalise(_intent(filters=[{"concept": "property_region",
                                          "comparator": "eq",
                                          "value": "London"}]))
    assert result.intent.operation == "point_in_time"


def test_a_member_question_keeps_its_filter_on_the_breakdown():
    intent = _intent(dimensions=["property_region"],
                     filters=[{"concept": "property_region", "comparator": "eq",
                               "value": "London"}])
    result = _normalise(intent)
    assert result.intent.operation == "breakdown"
    assert result.intent.filters == intent.filters
    assert result.intent.dimensions == intent.dimensions
    assert result.intent.measures == intent.measures


def test_a_mixed_intent_keeps_its_refusal():
    """One grouped output and one single figure: rewriting would make the
    single figure the unsupported one, so the compiler's refusal stands."""
    measure = [{"concept": "current_outstanding_balance", "statistic": "sum"}]
    intent = _intent(outputs=[
        {"id": "by_region", "measures": measure,
         "dimensions": ["property_region"]},
        {"id": "total", "measures": measure}])
    assert _normalise(intent).intent.operation == "point_in_time"
    assert UNSUPPORTED_COMPOSITION in _compile(intent).codes()


def test_a_capability_that_makes_no_breakdown_keeps_its_refusal():
    assert "breakdown" not in CAPABILITY_OPERATIONS["limit_assessment"]
    intent = _intent(capability="limit_assessment",
                     dimensions=["property_region"])
    assert _normalise(intent).intent.operation == "point_in_time"


@pytest.mark.parametrize("operation", ["breakdown", "rank", "distribution",
                                       "compare"])
def test_no_other_operation_is_rewritten(operation):
    result = _normalise(_intent(operation=operation,
                                dimensions=["property_region"]))
    assert result.intent.operation == operation
    assert not any("grouped_figure" in a for a in result.applied)


def test_the_rule_is_idempotent():
    once = _normalise(_intent(dimensions=["property_region"])).intent
    twice = _normalise(once)
    assert twice.intent == once and not any(
        "grouped_figure" in a for a in twice.applied)
