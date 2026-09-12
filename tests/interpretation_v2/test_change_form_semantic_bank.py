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
                                                 OUTCOME_PLAN, OUTCOME_REFUSE)

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
# 8. the broad request is understood and declined, not substituted
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("operation", ["compare", "movement", "summary"])
def test_a_material_summary_refuses_whatever_operation_accompanies_it(operation):
    result = _compile(change_form="material_summary", operation=operation)
    assert result.outcome == OUTCOME_REFUSE
    assert CHANGE_FORM_NOT_CONNECTED in [r.code for r in result.reasons]


def test_the_broad_request_does_not_borrow_the_overview_measure():
    result = _compile(change_form="material_summary", operation="compare",
                      measures=[{"concept": "portfolio_overview"}],
                      capability="portfolio_summary")
    assert result.plan is None


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
