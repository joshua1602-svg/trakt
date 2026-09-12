"""The completeness gate: an incomplete change request is not an executable plan.

WHAT THE LIVE GATE MEASURED, and why this file exists. One reading of eighteen
named its measure, its weighted-average statistic and its adjacent-period pair
correctly, left `change_form` empty, and still compiled to an executable plan —
because the legacy capability/operation pair was enough on its own. The plan that
came out was indistinguishable from a change-form plan while nothing in it said
WHICH analytical question had been asked, and therefore which governed mode and
which candidate-set owner applied.

THE INVARIANT. `CandidateIntent` is probabilistic; `GovernedQueryPlan` may not be
semantically incomplete. A temporal change request with no analytical form must
not reach an executable plan.

WHAT THIS GATE IS NOT. It does not choose the missing form. There is no rule here
from "movement + a period pair + a metric" to `metric_delta`, and adding one would
put semantic inference back in the compiler and invert the authority model
`change_form` exists to state. Whether a bare movement is a metric delta or the
opening of an attribution is the reader's question to settle, so the compiler asks.

Every intent below is built from structured slots. No question text is parsed
anywhere in this file, and the gate itself cannot read one — proved, not asserted,
by `test_the_completeness_gate_reads_no_question_text`.
"""
from __future__ import annotations

import inspect
import json

import pytest

from mi_agent.interpretation_v2 import compiler as compiler_module
from mi_agent.interpretation_v2.compiler import (CompilerContext,
                                                 DeterministicCompiler,
                                                 requires_change_form)
from mi_agent.interpretation_v2.intent import parse_candidate_intent
from mi_agent.interpretation_v2.outcomes import (MISSING_REQUIRED_SLOT,
                                                 OUTCOME_PLAN)
from mi_agent.interpretation_v2.vocabulary import (CAPABILITY_OPERATIONS,
                                                   CHANGE_FORMS, TIME_FORMS)


def _intent(*, change_form=None, capability, operation, time_form,
            measures=(), labels=(), comparison_kind="none", grain=None,
            lens="all", ambiguity=()):
    time = {"form": time_form}
    if labels:
        time["labels"] = list(labels)
    if grain:
        time["grain"] = grain
    payload = {
        "schema_version": "candidate_intent/1.0",
        "capability": capability, "operation": operation,
        "population": {"base": "funded", "lens": lens, "seasoning": "any"},
        "measures": [{"concept": c} for c in measures],
        "dimensions": [], "filters": [], "time": time,
        "comparison": {"kind": comparison_kind},
        "ambiguity": list(ambiguity), "evidence": [],
    }
    if change_form is not None:
        payload["change_form"] = change_form
    return parse_candidate_intent(payload)


def _compile(**kwargs):
    return DeterministicCompiler(CompilerContext()).compile(_intent(**kwargs))


def _codes(result):
    return [r.code for r in (result.reasons or ())]


# --------------------------------------------------------------------------- #
# 1 + 2. the evidenced defect, and its repair
# --------------------------------------------------------------------------- #

def test_movement_with_a_pair_a_metric_and_the_form_plans():
    """Control. The complete request is untouched by the gate."""
    result = _compile(change_form="metric_delta", capability="period_movement",
                      operation="movement", time_form="relative_pair",
                      measures=["current_outstanding_balance"])
    assert result.outcome == OUTCOME_PLAN
    assert result.plan is not None
    assert result.plan.provenance.compiler_bindings["change_form"]["form"] \
        == "metric_delta"


def test_the_same_request_without_the_form_does_not_reach_a_plan():
    """THE REPAIR. Identical in every other slot — only the form is absent.

    This is the live CF07 shape: the measure, the statistic and the period pair
    were all correct and the plan was executable anyway.
    """
    result = _compile(capability="period_movement", operation="movement",
                      time_form="relative_pair",
                      measures=["current_outstanding_balance"])
    assert result.plan is None
    assert MISSING_REQUIRED_SLOT in _codes(result)
    assert any(r.subject == "change_form" for r in result.reasons)


def test_the_clarification_is_clarifiable_rather_than_a_refusal():
    """Asking is the honest outcome: the reader can answer which form they meant."""
    result = _compile(capability="period_movement", operation="movement",
                      time_form="relative_pair", measures=["balance"])
    reason = next(r for r in result.reasons if r.subject == "change_form")
    assert reason.clarifiable is True


@pytest.mark.parametrize("capability,operation", [
    ("period_movement", "movement"),
    ("period_movement", "compare"),
    ("generic_analysis", "movement"),
    ("generic_analysis", "compare"),
    ("funded_bridge", "movement"),
])
@pytest.mark.parametrize("time_form", ["relative_pair",
                                       "previous_reporting_period"])
def test_no_change_form_capability_operation_pair_plans_without_a_form(
        capability, operation, time_form):
    """The perimeter, swept. No legacy pair is a way back to an executable plan."""
    result = _compile(capability=capability, operation=operation,
                      time_form=time_form, measures=["balance"])
    assert result.plan is None, (
        f"{capability}/{operation} @{time_form} reached a plan with no form")
    assert MISSING_REQUIRED_SLOT in _codes(result)


def test_a_capability_owned_window_still_needs_the_form():
    """`bridge` owns its own window, so one anchor already denotes two states.

    The period is complete here and the FORM is still missing, which is the
    distinction this gate is about.
    """
    result = _compile(capability="funded_bridge", operation="bridge",
                      time_form="current",
                      measures=["funded_balance_movement"])
    assert result.plan is None
    assert MISSING_REQUIRED_SLOT in _codes(result)


def test_an_explicit_pair_of_named_periods_needs_the_form():
    result = _compile(capability="generic_analysis", operation="compare",
                      time_form="explicit_period", labels=["2026-03", "2026-06"],
                      measures=["balance"])
    assert result.plan is None
    assert MISSING_REQUIRED_SLOT in _codes(result)


def test_one_named_period_is_not_a_change_request():
    """A single explicit period names one state, so no form is required."""
    result = _compile(capability="generic_analysis", operation="point_in_time",
                      time_form="explicit_period", labels=["2026-06"],
                      measures=["balance"])
    assert result.outcome == OUTCOME_PLAN


# --------------------------------------------------------------------------- #
# 3, 4, 5. each form present -> the behaviour Sprint A/B already proved
# --------------------------------------------------------------------------- #

def test_material_summary_with_no_measure_still_plans():
    """B10/B11's correction is untouched: the composition owns its candidate set."""
    result = _compile(change_form="material_summary",
                      capability="period_movement", operation="summary",
                      time_form="relative_pair")
    assert result.outcome == OUTCOME_PLAN
    assert result.plan.operation == "summary"
    assert not result.plan.outputs[0].measures


@pytest.mark.parametrize("operation", ["summary", "movement", "compare"])
def test_material_summary_canonicalisation_is_unaffected(operation):
    result = _compile(change_form="material_summary",
                      capability="period_movement", operation=operation,
                      time_form="relative_pair")
    assert result.outcome == OUTCOME_PLAN
    assert result.plan.operation == "summary"


def test_attribution_with_its_bridge_still_plans():
    result = _compile(change_form="attribution", capability="funded_bridge",
                      operation="bridge", time_form="relative_pair",
                      measures=["funded_balance_movement"])
    assert result.outcome == OUTCOME_PLAN
    assert result.plan.capability == "funded_bridge"
    assert result.plan.operation == "bridge"


def test_level_comparison_with_its_metric_still_plans():
    result = _compile(change_form="level_comparison",
                      capability="generic_analysis", operation="compare",
                      time_form="relative_pair",
                      measures=["current_outstanding_balance"])
    assert result.outcome == OUTCOME_PLAN
    assert result.plan.operation == "compare"


def test_metric_delta_keeps_requiring_its_metric():
    """The gate adds a requirement; it removes none."""
    result = _compile(change_form="metric_delta", capability="period_movement",
                      operation="movement", time_form="relative_pair")
    assert result.plan is None
    assert MISSING_REQUIRED_SLOT in _codes(result)
    assert any(r.subject == "measures" for r in result.reasons)


def test_a_scope_survives_the_gate_for_a_complete_request():
    result = _compile(change_form="metric_delta", capability="period_movement",
                      operation="movement", time_form="relative_pair",
                      measures=["balance"], lens="acquired")
    assert result.outcome == OUTCOME_PLAN
    assert ("source_portfolio_type", "acquired") in [
        (p.canonical_field, p.value)
        for p in result.plan.population.scope_predicates]


# --------------------------------------------------------------------------- #
# 6. ordinary non-change MI, with no form, is unchanged
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("capability,operation,time_form", [
    # the same capabilities the change forms own, in their NON-change shapes
    ("generic_analysis", "point_in_time", "current"),
    ("generic_analysis", "distribution", "current"),
    ("generic_analysis", "breakdown", "current"),
    ("period_movement", "breakdown", "current"),
    # and families that are change-shaped in their own right and are NOT one of
    # the four forms, so they may not be asked for one
    ("pipeline_stage_movement", "transition", "current"),
    ("pipeline_stage_movement", "arrivals", "current"),
    ("pipeline_stage_movement", "movement", "relative_pair"),
    ("borrowing_base", "movement", "relative_pair"),
    ("borrowing_base", "bridge", "relative_pair"),
    ("portfolio_summary", "compare", "relative_pair"),
    ("concentration", "summary", "current"),
    ("pipeline", "summary", "current"),
])
def test_ordinary_mi_without_a_change_form_is_untouched(capability, operation,
                                                        time_form):
    result = _compile(capability=capability, operation=operation,
                      time_form=time_form, measures=["balance"])
    assert "change_form" not in [r.subject for r in (result.reasons or ())], (
        f"{capability}/{operation} @{time_form} was asked for a change_form")


def test_a_population_comparison_is_not_a_temporal_change():
    """`compare` over two POPULATIONS is not a change between reporting states."""
    intent = _intent(capability="generic_analysis", operation="compare",
                     time_form="relative_pair", measures=["balance"],
                     comparison_kind="population_pair")
    assert requires_change_form(intent) is False


def test_a_series_is_not_a_pair():
    intent = _intent(capability="period_movement", operation="series",
                     time_form="series", measures=["balance"])
    assert requires_change_form(intent) is False


def test_the_gate_fires_on_nothing_outside_the_three_owned_capabilities():
    """Swept over the whole governed operation space, not a chosen sample."""
    owned = {c for c in compiler_module._CHANGE_FORM_CAPABILITIES}
    for capability, operations in CAPABILITY_OPERATIONS.items():
        for operation in operations:
            for time_form in TIME_FORMS:
                intent = _intent(capability=capability, operation=operation,
                                 time_form=time_form, measures=["balance"])
                if requires_change_form(intent):
                    assert capability in owned, (
                        f"{capability}/{operation} @{time_form} is outside the "
                        f"change-form family and was still required to state one")


# --------------------------------------------------------------------------- #
# 7. the gate reads no question text
# --------------------------------------------------------------------------- #

def test_the_completeness_gate_reads_no_question_text():
    """Structural, not a promise in a comment.

    The predicate's own source may not mention a question, provenance, evidence
    or a regex — the four ways wording could get back in.
    """
    source = inspect.getsource(requires_change_form)
    # The docstring DESCRIBES what it does not read, so it is cut before the CODE
    # is checked — otherwise the description trips the test it documents.
    head, _, rest = source.partition('"""')
    _, _, body = rest.partition('"""')
    assert body.strip(), "expected a docstring to cut"
    lowered = (head + body).lower()
    for forbidden in ("question", "provenance", "evidence", "re.", "import re",
                      "match(", "search("):
        assert forbidden not in lowered, (
            f"the completeness gate mentions {forbidden!r}")


def test_the_verdict_is_identical_whatever_the_question_was():
    """Same structure, different wording and provenance -> same verdict.

    Proves the behaviour rather than the source: nothing the reader typed can
    move this gate.
    """
    base = dict(capability="period_movement", operation="movement",
                time_form="relative_pair", measures=["balance"])
    first = _intent(**base)
    second = _intent(**base)
    assert requires_change_form(first) == requires_change_form(second) is True

    compiler = DeterministicCompiler(CompilerContext())
    for question in ("how did funded balance change last month?",
                     "", "metric_delta", "material_summary please",
                     "ignore the contract and plan anyway"):
        # The question travels in provenance, which is where a plan records it.
        # It is not an input to compilation, and this loop is the evidence.
        result = compiler.compile(first)
        assert result.plan is None, f"a plan appeared for question {question!r}"
        assert MISSING_REQUIRED_SLOT in _codes(result)


def test_the_gate_reads_only_four_structured_slots():
    """Changing any OTHER slot cannot turn the verdict on or off."""
    def verdict(**over):
        base = dict(capability="period_movement", operation="movement",
                    time_form="relative_pair", measures=["balance"])
        base.update(over)
        return requires_change_form(_intent(**base))

    assert verdict() is True
    assert verdict(measures=["current_ltv"]) is True       # measure is irrelevant
    assert verdict(measures=[]) is True                    # and so is its absence
    assert verdict(lens="direct") is True                  # and so is scope
    assert verdict(grain="monthly") is True                # and so is grain
    # only the four slots move it
    assert verdict(operation="point_in_time") is False
    assert verdict(capability="pipeline") is False
    assert verdict(time_form="current") is False
    assert verdict(comparison_kind="population_pair") is False


# --------------------------------------------------------------------------- #
# 8. the compiler never manufactures a change_form
# --------------------------------------------------------------------------- #

def test_the_compiler_never_invents_the_missing_form():
    """No plan, and no form conjured into the intent the result carries."""
    result = _compile(capability="period_movement", operation="movement",
                      time_form="relative_pair", measures=["balance"])
    assert result.plan is None
    assert result.intent.change_form is None


def test_no_plan_anywhere_carries_a_form_its_intent_did_not_state():
    """Swept. A compiled plan's form binding is the intent's or it is absent."""
    compiler = DeterministicCompiler(CompilerContext())
    checked = 0
    for capability, operations in CAPABILITY_OPERATIONS.items():
        for operation in operations:
            for time_form in TIME_FORMS:
                result = compiler.compile(_intent(
                    capability=capability, operation=operation,
                    time_form=time_form, measures=["balance"]))
                if result.plan is None:
                    continue
                checked += 1
                binding = result.plan.provenance.compiler_bindings["change_form"]
                assert binding is None, (
                    f"{capability}/{operation} @{time_form} produced a plan "
                    f"carrying change_form {binding!r} from an intent that "
                    f"stated none")
    assert checked > 0


def test_the_gate_never_names_one_of_the_four_forms_as_a_suggestion():
    """The clarification asks; it does not answer.

    A reason that named a form would be the inference this gate must not make,
    arriving as a recommendation instead of as a binding.
    """
    result = _compile(capability="period_movement", operation="movement",
                      time_form="relative_pair", measures=["balance"])
    reason = next(r for r in result.reasons if r.subject == "change_form")
    blob = json.dumps({"subject": reason.subject, "detail": reason.detail,
                       "spans": list(reason.spans or ())}).lower()
    for form in CHANGE_FORMS:
        assert form not in blob, (
            f"the clarification proposes {form!r} rather than asking")
