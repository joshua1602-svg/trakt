"""B · The compiler binds concepts to fields, not the model.

Opus may now READ the governed registry and name a concept identifier out of
it, so "the model never said this string" is no longer the test — it would fail
on a correct reading. The claim that replaces it is stronger and is the one
that was always meant:

  * every field on a plan is RE-DERIVED from the authoritative index, and is
    the registry's answer rather than the model's string;
  * a string the registry does not carry never becomes a field, however
    column-shaped it looks;
  * the statistic, the weight and the physical binding come from the registry
    even when the question named none of them.
"""

from __future__ import annotations

import json

from mi_agent.interpretation_v2 import OUTCOME_PLAN, load_governed_vocabulary

from .conftest import build_intent


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
    governed = {c.canonical_field for c in vocabulary.concepts.values()
                if c.canonical_field}
    for field in result.plan.bound_fields():
        assert field in governed, f"{field!r} is not a governed binding"


def test_a_column_shaped_string_the_registry_lacks_never_becomes_a_field(compiler):
    """The boundary is the registry, not the spelling.

    Every one of these looks exactly like a canonical column. None is in the
    index, so none reaches a plan — which is what stops "the model saw the
    schema" from becoming "the model can address the schema".
    """
    for invented in ("current_outstanding_balance_gross", "loan__balance",
                     "v_loans.current_balance", "balance_2024"):
        result = compiler.compile(build_intent(measures=[{"concept": invented}]))
        assert result.plan is None, invented
        assert "UNREGISTERED_CONCEPT" in result.codes()


def test_the_binding_is_the_registrys_answer_not_the_models_word(compiler):
    """An alias and its identifier compile to the SAME field.

    If the compiler were copying the model's string, these two would produce
    different plans; they produce one, because both are re-resolved through the
    index before anything is bound.
    """
    by_alias = compiler.compile(build_intent(
        measures=[{"concept": "balance", "statistic": "sum"}]))
    by_id = compiler.compile(build_intent(
        measures=[{"concept": "current_outstanding_balance", "statistic": "sum"}]))
    assert by_alias.plan.plan_id == by_id.plan.plan_id


def test_the_weight_field_is_supplied_by_the_registry_not_the_question(compiler):
    """"average LTV" carries no weight; the governed registry does."""
    intent = build_intent(measures=[{"concept": "current_ltv",
                                     "statistic": "weighted_average"}])
    result = compiler.compile(intent)
    assert result.outcome == OUTCOME_PLAN
    measure = result.plan.outputs[0].measures[0]
    assert measure.weight_field == "current_outstanding_balance"
    # The weight is recorded by its governed IDENTIFIER, not by whatever word
    # would have named it — the question named no weight at all.
    assert measure.weight_concept == "current_outstanding_balance"
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
    # The model said "balance". The compiler said the field. Both are on the
    # record, apart, so an auditor can see which half decided what.
    assert '"balance"' in claims
    assert "current_outstanding_balance" not in claims
    assert "current_outstanding_balance" in bindings
    assert provenance.compiler_version
    assert provenance.vocabulary_version
