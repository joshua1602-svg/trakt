"""The four analytical forms, as structured intents rather than as sentences.

WHY THE BANK IS STRUCTURED AND NOT WRITTEN IN ENGLISH. Sprint A changes the
CONTRACT, not the model. Whether Opus reads a given sentence as one form or
another is a live measurement and is made separately; what can be proved offline
is that once a form is stated, the contract resolves it to one owner, the same
way, for every lender.

SO NO SENTENCE APPEARS BELOW. Each case is the structured shape a question of
that kind arrives as. The three lender contexts differ only in the governed
concepts their books carry — which is the point: the decision never reads them.
"""
from __future__ import annotations

import pytest

from mi_agent.interpretation_v2.compiler import CompilerContext, DeterministicCompiler
from mi_agent.interpretation_v2.intent import parse_candidate_intent
from mi_agent.interpretation_v2.outcomes import (CHANGE_FORM_NOT_CONNECTED,
                                                 MISSING_REQUIRED_SLOT,
                                                 OUTCOME_CLARIFY, OUTCOME_PLAN,
                                                 OUTCOME_REFUSE)

#: One governed measure per lender context. They are different concepts on
#: different books; the form-of-change decision must not depend on which.
_MEASURES = {
    "generic_currency": "current_outstanding_balance",
    "generic_count": "loan",
    "generic_ratio": "current_loan_to_value",
}

_PAIR = {"form": "relative_pair", "periods_back": 1}


def _compile(**over):
    body = {
        "schema_version": "candidate_intent/1.0",
        "capability": "generic_analysis",
        "operation": "movement",
        "measures": [{"concept": _MEASURES["generic_currency"]}],
        "time": dict(_PAIR),
    }
    body.update(over)
    return DeterministicCompiler(CompilerContext()).compile(
        parse_candidate_intent(body))


# --------------------------------------------------------------------------- #
# 1-4. a stated metric delta reaches one owner, whatever the metric is
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("concept", sorted(_MEASURES.values()))
def test_a_metric_delta_reaches_the_movement_owner_for_any_governed_metric(concept):
    result = _compile(change_form="metric_delta",
                      measures=[{"concept": concept}])
    assert result.outcome == OUTCOME_PLAN
    assert result.plan.capability == "period_movement"


@pytest.mark.parametrize("claimed", ["generic_analysis", "period_movement",
                                     "portfolio_summary"])
def test_the_models_capability_guess_does_not_change_the_owner(claimed):
    # The model answers what was asked; the contract decides who runs it. Three
    # different guesses, one owner — which is the whole repair.
    result = _compile(change_form="metric_delta", capability=claimed)
    assert result.plan is not None and result.plan.capability == "period_movement"


# --------------------------------------------------------------------------- #
# 5-6. an explicit decomposition stays a decomposition
# --------------------------------------------------------------------------- #

def test_attribution_reaches_the_bridge_owner():
    result = _compile(change_form="attribution", operation="bridge",
                      measures=[{"concept": "funded_balance_movement"}])
    assert result.outcome == OUTCOME_PLAN
    assert result.plan.capability == "funded_bridge"


def test_an_explicit_bridge_measure_is_never_collapsed_into_a_delta():
    # The repair must not turn a genuine attribution request into a net figure.
    result = _compile(change_form="attribution", operation="bridge",
                      measures=[{"concept": "bridge_component"}])
    assert result.plan is not None and result.plan.capability == "funded_bridge"


# --------------------------------------------------------------------------- #
# 7. levels compared stay levels
# --------------------------------------------------------------------------- #

def test_a_level_comparison_is_not_turned_into_movement():
    result = _compile(change_form="level_comparison", operation="compare")
    assert result.outcome == OUTCOME_PLAN
    assert result.plan.capability == "generic_analysis"
    assert result.plan.operation == "compare"


# --------------------------------------------------------------------------- #
# 8. the broad request reaches its owner, and a narrowed one does not
# --------------------------------------------------------------------------- #
# This section used to assert that the broad request was DECLINED, because the
# funded material-change composition did not exist. It does now and the form is
# connected — so what these controls pin is the thing that had to stay true
# either way: a broad request is never answered by a NARROWER analysis, and a
# narrowed one is never answered by the broad composition. Before, both were
# enforced by refusing everything. Now the first is enforced by the compiler
# binding the form to one owner, and the second by that owner's own perimeter.


@pytest.mark.parametrize("operation", ["compare", "movement", "summary"])
def test_a_material_summary_reaches_one_owner_whatever_operation_accompanies_it(
        operation):
    # ONE owner and ONE mode, whichever operation the reader's sentence carried.
    # The operation states the result SHAPE; it never re-decides the owner.
    result = _compile(change_form="material_summary", operation=operation)
    assert result.outcome == OUTCOME_PLAN
    binding = result.plan.provenance.compiler_bindings["change_form"]
    assert binding["capability"] == "period_movement"
    assert binding["mode"] == "portfolio_overview"


# --------------------------------------------------------------------------- #
# 8b. who owns the candidate set — the four structured invariants
# --------------------------------------------------------------------------- #
# WHERE `change_form` AND `operation` OVERLAP, THE FORM WINS ON OWNERSHIP.
# `material_summary` MEANS "determine which governed changes are materially
# relevant", so the composition owns the candidate set BY DEFINITION and no
# measure need be named. `operation` stays authoritative over the result SHAPE.
#
# Everything below is built from structured intent. No sentence is parsed, no
# wording is matched and no recogniser is involved — which is the point: these
# are properties of the contract, not of any phrasing that reaches it.


def test_A_a_bare_material_summary_executes_without_a_named_measure():
    result = _compile(change_form="material_summary", operation="summary",
                      measures=[])
    assert result.outcome == OUTCOME_PLAN
    assert result.plan.capability == "period_movement"
    assert all(not output.measures for output in result.plan.outputs)


@pytest.mark.parametrize("operation", ["movement", "compare", "summary"])
def test_B_every_spelling_of_the_action_reaches_one_execution_contract(operation):
    # One analytical form, three equally correct spellings of its action, ONE
    # executable shape. The model is not made to emit an implementation's
    # spelling to make its plan runnable.
    result = _compile(change_form="material_summary", operation=operation,
                      measures=[])
    assert result.outcome == OUTCOME_PLAN
    assert result.plan.operation == "summary", operation
    binding = result.plan.provenance.compiler_bindings["change_form"]
    assert binding["capability"] == "period_movement"
    assert binding["mode"] == "portfolio_overview"
    # What the model actually said is still recoverable; only the EXECUTION was
    # canonicalised, and the rewrite says so in the provenance notes.
    assert result.plan.provenance.intent_claims["change_form"] == "material_summary"
    if operation != "summary":
        assert any("operation" in note and "change_form" in note
                   for note in result.plan.provenance.notes), operation


@pytest.mark.parametrize("operation", ["movement", "compare"])
def test_C_metric_delta_still_requires_a_named_measure(operation):
    # The correction is scoped to ONE form. A metric delta names its metric, and
    # a metric delta that does not is still a clarification.
    result = _compile(change_form="metric_delta", operation=operation,
                      measures=[])
    assert result.outcome == OUTCOME_CLARIFY, operation
    assert MISSING_REQUIRED_SLOT in [r.code for r in result.reasons], operation


@pytest.mark.parametrize("operation", ["movement", "compare"])
def test_D_no_change_form_is_never_read_as_a_material_summary(operation):
    # THE INFERENCE THAT WAS NOT ADDED. Whether a bare "what changed" IS a
    # material summary is an interpretation question. With the slot empty the
    # compiler does not guess it, and the pre-existing contract answers.
    result = _compile(operation=operation, measures=[])
    assert result.outcome == OUTCOME_CLARIFY, operation
    assert MISSING_REQUIRED_SLOT in [r.code for r in result.reasons], operation
    assert result.plan is None


@pytest.mark.parametrize("form,operation", [("attribution", "movement"),
                                            ("level_comparison", "compare"),
                                            ("level_comparison", "movement")])
def test_the_other_forms_measure_requirements_are_untouched(form, operation):
    result = _compile(change_form=form, operation=operation, measures=[])
    assert result.outcome == OUTCOME_CLARIFY, (form, operation)
    assert MISSING_REQUIRED_SLOT in [r.code for r in result.reasons]


@pytest.mark.parametrize("operation", ["rank", "breakdown", "series",
                                       "distribution"])
def test_an_operation_stating_a_shape_the_form_cannot_produce_is_not_flattened(
        operation):
    # A compatible linguistic variant is canonicalised; an incompatible SHAPE is
    # not. `rank` and `breakdown` ask for an ordering and a grouping this
    # composition does not produce, and turning them into a summary would answer
    # a different question while reporting the one that was asked.
    result = _compile(change_form="material_summary", operation=operation,
                      measures=[])
    assert result.outcome != OUTCOME_PLAN, operation
    assert result.plan is None


def test_the_broad_request_does_not_borrow_the_overview_measure():
    # `portfolio_summary` is a LEVEL summary of one period. Answering "what
    # changed" with it would substitute a neighbouring owner, which is the defect
    # this slot exists to end. The disagreement between the form's owner and the
    # nominated specialist measure is RECORDED and left unresolved rather than
    # silently settled in the measure's favour, and the plan that results is
    # refused by the material-summary perimeter rather than served by it.
    from mi_agent import plan_material_summary as runtime

    result = _compile(change_form="material_summary", operation="compare",
                      measures=[{"concept": "portfolio_overview"}],
                      capability="portfolio_summary")
    if result.plan is None:
        return
    assert any("change_form" in note for note in result.plan.provenance.notes)
    eligible, _why, _detail = runtime.check_eligibility(result.plan)
    assert not eligible


def test_a_named_measure_is_refused_by_the_owners_perimeter_not_flattened():
    # The compiler binds the form to its owner; the OWNER decides whether this
    # particular plan is one it can answer. A plan naming a measure is a metric
    # delta — a different question, run in a different governed mode — and the
    # perimeter says so rather than running a portfolio overview over everything
    # and reporting it as the reader's.
    from mi_agent import plan_material_summary as runtime

    result = _compile(change_form="material_summary", operation="summary")
    assert result.outcome == OUTCOME_PLAN
    assert runtime.claims(result.plan)
    eligible, why, _detail = runtime.check_eligibility(result.plan)
    assert not eligible
    assert why == runtime.MEASURE_NARROWS_THE_SUMMARY


# --------------------------------------------------------------------------- #
# 9-10. population is untouched by the form
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("base", ["funded", "pipeline"])
def test_the_form_does_not_move_the_population(base):
    result = _compile(change_form="metric_delta",
                      population={"base": base})
    if result.plan is not None:
        assert result.plan.population.base == base


@pytest.mark.parametrize("lens", ["direct", "acquired"])
def test_the_form_does_not_move_the_lens(lens):
    result = _compile(change_form="metric_delta",
                      population={"base": "funded", "lens": lens})
    if result.plan is not None:
        assert result.plan.population.lens == lens


def test_the_form_does_not_move_a_filter():
    result = _compile(change_form="metric_delta",
                      filters=[{"concept": "current_loan_to_value",
                                "comparator": "gt", "value": 50}])
    assert result.plan is not None
    assert result.plan.filters, "the requested filter did not survive"


# --------------------------------------------------------------------------- #
# paraphrase invariance, at the level the contract can actually guarantee
# --------------------------------------------------------------------------- #

def test_one_form_and_one_metric_give_one_owner_however_it_was_labelled():
    """The guarantee this sprint actually makes: the OWNER agrees.

    Three arrivals of the same request differing only in the capability the
    model happened to name — which is the slot it is no longer required to get
    right.
    """
    owners = {
        _compile(change_form="metric_delta", capability=cap).plan.capability
        for cap in ("generic_analysis", "period_movement", "portfolio_summary")
    }
    assert owners == {"period_movement"}


def test_identical_arrivals_are_byte_identical_contracts():
    from mi_agent.interpretation_v2.equivalence import plan_fingerprint

    prints = {
        plan_fingerprint(_compile(change_form="metric_delta",
                                  capability=cap).plan)
        for cap in ("generic_analysis", "period_movement")
    }
    assert len(prints) == 1


def test_the_operation_is_not_normalised_by_the_form():
    """A stated limit of this sprint, pinned so it is not mistaken for a bug.

    `change_form` decides WHO executes. It does not rewrite `operation`, because
    an operation is part of what was asked — `compare` and `movement` are two
    different requests of the same owner, and collapsing them would be deciding
    for the reader rather than routing what they said. A form arriving with an
    operation its owner supports keeps that operation.
    """
    movement = _compile(change_form="metric_delta", operation="movement")
    compared = _compile(change_form="metric_delta", operation="compare")
    assert movement.plan.capability == compared.plan.capability
    assert movement.plan.operation != compared.plan.operation
