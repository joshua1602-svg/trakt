"""The form of a change is its own semantic, and it names its own owner.

WHAT THIS PINS. One business question — "how did the book change last month?" —
reached the compiler as three different contracts, because the reading was
spread across three slots that can disagree: `capability`, `operation` and the
MEASURE. A specialist measure determines its owner, so the owner followed the
measure rather than the question.

`change_form` states the analytical question separately from what is measured
and from who executes it. These controls are written against that contract, not
against the questions that exposed it: no benchmark id, no benchmark wording,
and no lender's vocabulary appears below, because a rule that needed them would
not survive a second lender.
"""
from __future__ import annotations

import re

import pytest

from mi_agent.interpretation_v2.compiler import CompilerContext, DeterministicCompiler
from mi_agent.interpretation_v2.intent import (candidate_intent_json_schema,
                                               parse_candidate_intent)
from mi_agent.interpretation_v2.normalise import canonical_intent
from mi_agent.interpretation_v2.outcomes import (OUTCOME_PLAN, OUTCOME_REFUSE,
                                                 CHANGE_FORM_NOT_CONNECTED)
from mi_agent.interpretation_v2.vocabulary import (CAPABILITY_OPERATIONS,
                                                   CHANGE_FORM_CAPABILITY,
                                                   CHANGE_FORMS,
                                                   load_governed_vocabulary)

_BASE = {
    "schema_version": "candidate_intent/1.0",
    "capability": "generic_analysis",
    "operation": "movement",
    "measures": [{"concept": "current_outstanding_balance"}],
    "time": {"form": "relative_pair", "periods_back": 1},
}


def _intent(**over):
    body = dict(_BASE)
    body.update(over)
    return parse_candidate_intent(body)


def _compile(**over):
    return DeterministicCompiler(CompilerContext()).compile(_intent(**over))


def _normalise(**over):
    return canonical_intent(_intent(**over), load_governed_vocabulary(),
                            capability_operations=CAPABILITY_OPERATIONS)


# --------------------------------------------------------------------------- #
# the slot
# --------------------------------------------------------------------------- #

def test_the_slot_is_optional_so_nothing_recorded_before_it_breaks():
    # Every intent in the committed corpus predates this slot. If the contract
    # required it, replaying that evidence would fail for a reason that has
    # nothing to do with what the evidence says.
    assert _intent().change_form is None
    assert "change_form" not in candidate_intent_json_schema()["required"]


def test_an_ungoverned_form_is_refused_rather_than_ignored():
    with pytest.raises(Exception) as excinfo:
        _intent(change_form="whatever_the_model_felt_like")
    assert "change_form" in str(excinfo.value)


def test_the_model_is_offered_exactly_the_governed_forms():
    schema = candidate_intent_json_schema()["properties"]["change_form"]
    assert set(schema["enum"]) == set(CHANGE_FORMS)


def test_the_model_facing_description_names_no_lender_and_no_benchmark():
    # The whole point of the slot is that it means the same thing for a
    # residential mortgage book, an equity-release book and an asset-finance
    # book. A description that reached for any of their words would not.
    text = candidate_intent_json_schema()["properties"]["change_form"]["description"]
    for forbidden in ("Q18", "Q19", "Q20", "ERE", "mortgage", "equity release",
                      "asset finance", "drawdown", "balance", "LTV", "broker"):
        assert not re.search(rf"\b{re.escape(forbidden)}\b", text, re.I), forbidden


def test_the_description_asks_what_was_asked_not_who_should_run_it():
    text = candidate_intent_json_schema()["properties"]["change_form"]["description"]
    for owner in ("period_movement", "funded_bridge", "portfolio_summary",
                  "generic_analysis", "capability"):
        assert owner not in text, owner


# --------------------------------------------------------------------------- #
# the owner is derived, not claimed
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("form,owner", sorted(
    (f, o) for f, o in CHANGE_FORM_CAPABILITY.items() if o))
def test_every_implemented_form_names_its_own_owner(form, owner):
    supported = CAPABILITY_OPERATIONS.get(owner, frozenset())
    operation = "movement" if "movement" in supported else sorted(supported)[0]
    result = _normalise(change_form=form, operation=operation)
    assert result.intent.capability == owner


def test_the_derivation_is_recorded_and_never_silent():
    result = _normalise(change_form="metric_delta")
    assert any("change_form" in note for note in result.applied)


def test_an_intent_without_the_slot_is_left_exactly_as_it_was():
    # The regression guarantee for every question that is not about a change.
    before = _intent()
    after = _normalise()
    assert after.intent.capability == before.capability
    assert not [n for n in after.applied if "change_form" in n]


def test_normalisation_stays_idempotent():
    once = _normalise(change_form="metric_delta").intent
    twice = canonical_intent(once, load_governed_vocabulary(),
                             capability_operations=CAPABILITY_OPERATIONS)
    assert twice.intent.capability == once.capability


# --------------------------------------------------------------------------- #
# precedence, and what happens when two signals disagree
# --------------------------------------------------------------------------- #

def test_an_explicitly_named_specialist_measure_still_wins():
    # An explicit semantic requirement outranks a structural implication. The
    # reader named a measure only one capability owns; that is not overridden.
    result = _normalise(change_form="metric_delta", capability="funded_bridge",
                        measures=[{"concept": "funded_balance_movement"}])
    assert result.intent.capability == "funded_bridge"


def test_a_disagreement_is_recorded_and_left_for_the_compiler():
    # Preferring either signal would decide, for the reader, whether they asked
    # for a decomposition or a delta.
    result = _normalise(change_form="metric_delta", capability="funded_bridge",
                        measures=[{"concept": "funded_balance_movement"}])
    assert any("left unresolved" in note for note in result.applied)


def test_an_owner_is_never_bound_to_an_operation_it_does_not_support():
    # Outcome-neutral: a rewrite may not turn a plan into a composition the
    # owner cannot express.
    result = _normalise(change_form="attribution", operation="distribution")
    supported = CAPABILITY_OPERATIONS["funded_bridge"]
    assert "distribution" not in supported
    assert result.intent.capability != "funded_bridge"


# --------------------------------------------------------------------------- #
# a form with no owner refuses; it does not borrow one
# --------------------------------------------------------------------------- #

def test_an_unimplemented_form_refuses_rather_than_substituting():
    result = _compile(change_form="material_summary", operation="compare")
    assert result.outcome == OUTCOME_REFUSE
    assert [r.code for r in result.reasons] == [CHANGE_FORM_NOT_CONNECTED]
    assert result.plan is None


def test_the_refusal_says_what_was_understood():
    result = _compile(change_form="material_summary", operation="compare")
    assert "material_summary" in result.reasons[0].subject


def test_the_unimplemented_form_is_not_clarifiable():
    # Asking the reader to rephrase cannot build an owner.
    result = _compile(change_form="material_summary", operation="compare")
    assert not result.reasons[0].clarifiable


@pytest.mark.parametrize("substitute", ["period_movement", "funded_bridge",
                                        "portfolio_summary", "generic_analysis"])
def test_it_is_never_answered_by_a_neighbouring_owner(substitute):
    result = _compile(change_form="material_summary", operation="compare")
    assert result.plan is None, substitute


# --------------------------------------------------------------------------- #
# the compiler's mapping is the only place the owner is decided
# --------------------------------------------------------------------------- #

def test_every_governed_form_has_a_stated_owner_or_a_stated_absence():
    assert set(CHANGE_FORM_CAPABILITY) == set(CHANGE_FORMS)


def test_the_module_reads_no_question():
    import inspect

    from mi_agent.interpretation_v2 import normalise
    source = inspect.getsource(normalise)
    for seam in ("re.search", "re.match", "re.compile", "question.lower",
                 "in question", "source_span", "quote"):
        assert seam not in source, seam
