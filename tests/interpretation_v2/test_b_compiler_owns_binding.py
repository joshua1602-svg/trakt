"""B · The compiler binds concepts to fields, not the model.

Two claims, and the second is the one that matters:

  * every canonical field on a plan came from the governed vocabulary's map;
  * no canonical field on a plan appears anywhere in what the model said.

The second is the sharp one. If a plan could reach a field by copying a string
the model supplied, the architecture would be a naming convention rather than a
boundary.
"""

from __future__ import annotations

import json

from mi_agent.interpretation_v2 import OUTCOME_PLAN, load_governed_vocabulary

from .conftest import build_intent


def _model_supplied_strings(intent) -> set:
    """Every string the model put in the intent, at any depth."""
    found = set()

    def walk(node):
        if isinstance(node, str):
            found.add(node)
        elif isinstance(node, dict):
            for value in node.values():
                walk(value)
        elif isinstance(node, (list, tuple)):
            for value in node:
                walk(value)

    body = intent.to_dict()
    body.pop("provenance", None)
    walk(body)
    return found


def test_every_bound_field_comes_from_the_governed_vocabulary(compiler):
    intent = build_intent(
        operation="breakdown",
        measures=[{"concept": "current_ltv", "statistic": "weighted_average"}],
        dimensions=["product_type"],
        filters=[{"concept": "borrower_age", "comparator": "gt", "value": 75}],
    )
    result = compiler.compile(intent)
    assert result.outcome == OUTCOME_PLAN

    vocabulary = load_governed_vocabulary()
    by_field = {c.canonical_field: c.term for c in vocabulary.concepts.values()
                if c.canonical_field}
    for field in result.plan.bound_fields():
        assert field in by_field, f"{field!r} is not a governed binding"


def test_no_bound_field_was_a_string_the_model_supplied(compiler):
    """The plan's fields are the compiler's choices, not the model's words."""
    intent = build_intent(
        operation="breakdown",
        measures=[{"concept": "current_ltv", "statistic": "weighted_average",
                   "weight": "balance"}],
        dimensions=["product_type"],
        filters=[{"concept": "borrower_age", "comparator": "gt", "value": 75}],
        geography={"requested": True, "level": "itl3", "basis": "collateral",
                   "group_by": True},
    )
    result = compiler.compile(intent)
    assert result.outcome == OUTCOME_PLAN

    said = _model_supplied_strings(intent)
    fields = set(result.plan.bound_fields())
    assert fields, "the plan bound nothing, so this proves nothing"
    assert fields.isdisjoint(said), (
        "a canonical field on the plan is a string the model supplied: "
        f"{sorted(fields & said)}")


def test_the_weight_field_is_supplied_by_the_registry_not_the_question(compiler):
    """"average LTV" carries no weight; the governed registry does."""
    intent = build_intent(measures=[{"concept": "current_ltv",
                                     "statistic": "weighted_average"}])
    result = compiler.compile(intent)
    assert result.outcome == OUTCOME_PLAN
    measure = result.plan.outputs[0].measures[0]
    assert measure.weight_field == "current_outstanding_balance"
    assert measure.weight_concept == "balance"
    # And it is disclosed, not silent.
    assert any("weight" in note for note in result.plan.provenance.notes)


def test_a_defaulted_statistic_is_recorded_as_defaulted(compiler):
    intent = build_intent(measures=[{"concept": "balance"}])
    result = compiler.compile(intent)
    assert result.outcome == OUTCOME_PLAN
    measure = result.plan.outputs[0].measures[0]
    assert measure.statistic == "sum"
    assert measure.statistic_defaulted is True


def test_provenance_separates_what_the_model_claimed_from_what_was_bound(compiler):
    intent = build_intent(measures=[{"concept": "balance"}])
    result = compiler.compile(intent)
    provenance = result.plan.provenance

    claims = json.dumps(provenance.intent_claims)
    bindings = json.dumps(provenance.compiler_bindings)
    assert "current_outstanding_balance" not in claims, (
        "a canonical field leaked into the model's recorded claims")
    assert "current_outstanding_balance" in bindings
    assert provenance.compiler_version
    assert provenance.vocabulary_version
