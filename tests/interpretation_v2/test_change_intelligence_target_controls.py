#!/usr/bin/env python3
"""Target-state controls for the change-intelligence family.

WHY THESE ARE NOT SCORED AGAINST THE 135 BANK. That bank records, for all nine
questions of this family, `capability=period_movement, operation=movement`. All
nine expect the same thing, so the fixture cannot express the distinction this
sprint exists to make: a broad "what changed?" and a named-metric delta are
different questions with different owners and different governed modes, and the
coarse fixture calls them the same. It is not wrong about the capability; it
simply predates the semantic. These controls state the target instead.

WHAT IS AND IS NOT UNDER TEST. Every control below is built from STRUCTURED
INTENT. No question is parsed, no wording is matched, and no model is called —
`LIVE_MODEL_CALLS = 0`. The question text appears only in a comment, to say
which reader's request each control stands for. Nothing in the product
references any of them, and the controls would read identically for a different
lender asking the same four things.

WHAT THEY THEREFORE PROVE, AND WHAT THEY DO NOT. They prove that ONCE
INTERPRETATION SUPPLIES A `change_form`, the structured request reaches exactly
one deterministic owner in exactly one governed mode. They do NOT prove that the
interpreter assigns the right form to any particular sentence — that is the live
interpretation gate, which remains pending and is a separate measurement.
"""
from __future__ import annotations

import pytest

from mi_agent import plan_material_summary as material_summary
from mi_agent.interpretation_v2.compiler import CompilerContext, DeterministicCompiler
from mi_agent.interpretation_v2.intent import parse_candidate_intent
from mi_agent.interpretation_v2.outcomes import OUTCOME_PLAN

#: The month-on-month pair every question in this family asks for.
_LAST_MONTH = {"form": "relative_pair", "periods_back": 1}


def _plan(**over):
    body = {
        "schema_version": "candidate_intent/1.0",
        "capability": "generic_analysis",
        "operation": "movement",
        "measures": [],
        "population": {"base": "funded"},
        "time": dict(_LAST_MONTH),
    }
    body.update(over)
    result = DeterministicCompiler(CompilerContext()).compile(
        parse_candidate_intent(body))
    assert result.outcome == OUTCOME_PLAN, [r.code for r in result.reasons]
    return result.plan


def _form_binding(plan):
    return plan.provenance.compiler_bindings["change_form"]


# --------------------------------------------------------------------------- #
# Q18 — the broad request, whole book
# --------------------------------------------------------------------------- #
# "How did the book change in the last month?"
# "What changed in the portfolio since last month?"
# "Give me a summary of how the funded book moved over the last month."
#
# Three sentences, one analytical form, and no metric named in any of them. The
# three natural operations a reader's phrasing produces are all variants of the
# same action, so all three must reach one execution contract.
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("operation", ["movement", "compare", "summary"])
def test_Q18_broad_reaches_the_material_summary_execution_contract(operation):
    plan = _plan(change_form="material_summary", operation=operation)

    assert plan.capability == "period_movement"
    assert plan.operation == "summary"
    binding = _form_binding(plan)
    assert binding["form"] == "material_summary"
    assert binding["mode"] == "portfolio_overview"

    # And the runtime that owns that contract accepts it, with no measure named.
    assert material_summary.claims(plan)
    eligible, why, detail = material_summary.check_eligibility(plan)
    assert eligible, f"{why}: {detail}"


def test_Q18_names_no_measure_and_is_not_asked_to():
    plan = _plan(change_form="material_summary", operation="movement")
    assert all(not output.measures for output in plan.outputs)


def test_Q18_resolves_to_a_snapshot_pair_the_resolver_owns():
    plan = _plan(change_form="material_summary", operation="movement")
    request = material_summary.period_request(plan)
    # A relative pair states a METHOD, not two dates. Nothing here computed a
    # calendar month, and nothing assumed one: the compiler's normal form leaves
    # this request's grain absent, which means the ADJACENT GOVERNED SNAPSHOTS —
    # whatever cadence the book reports on — and the resolver owns which two
    # those are.
    assert request.relative_mode == "current_vs_previous"
    assert request.requested_start is None and request.requested_end is None


@pytest.mark.parametrize("grain,method", [("monthly", "month_on_month"),
                                          ("quarterly", "quarter_on_quarter"),
                                          ("annual", "year_on_year")])
def test_a_stated_grain_translates_to_its_own_governed_method(grain, method):
    plan = _plan(change_form="material_summary", operation="movement",
                 time=dict(_LAST_MONTH, grain=grain))
    assert material_summary.period_request(plan).relative_mode == method


def test_a_distance_with_no_governed_method_is_refused_not_approximated():
    # "Three periods back" is a reasonable question with no relative method on
    # the resolver. Synthesising the two dates here would be the period
    # arithmetic the resolver owns, so it is refused and says so.
    plan = _plan(change_form="material_summary", operation="movement",
                 time={"form": "relative_pair", "periods_back": 3})
    eligible, why, _detail = material_summary.check_eligibility(plan)
    assert not eligible
    assert why == material_summary.PERIOD_GRAIN_UNTRANSLATABLE


# --------------------------------------------------------------------------- #
# Q18 explicit attribution variant
# --------------------------------------------------------------------------- #
# "What DROVE the change in the book last month?" — a decomposition is asked
# for. Same subject, same period, different analytical form and a different
# owner. This is the substitution the change_form slot exists to prevent, in
# both directions.
# --------------------------------------------------------------------------- #

def test_Q18_attribution_variant_reaches_the_bridge_owner_not_the_summary():
    plan = _plan(change_form="attribution", operation="bridge")
    assert plan.capability == "funded_bridge"
    assert plan.operation == "bridge"
    assert _form_binding(plan)["capability"] == "funded_bridge"
    # The material-summary runtime does not claim it, so the two can never both
    # answer the same plan.
    assert not material_summary.claims(plan)


# --------------------------------------------------------------------------- #
# Q19 — the same broad request, scoped to one origination channel
# --------------------------------------------------------------------------- #
# "How did the Direct book change last month?"
# "What changed in the Direct portfolio since last month?"
# "Summarise the month-on-month movement in the Direct book."
#
# The scope must survive the form. A material summary of the Direct book that
# quietly reported the whole book would be the same defect as substituting a
# neighbouring owner, expressed through the population instead.
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("operation", ["movement", "compare", "summary"])
def test_Q19_broad_and_scoped_reaches_the_same_contract_with_its_scope(operation):
    plan = _plan(change_form="material_summary", operation=operation,
                 population={"base": "funded", "lens": "direct"})

    assert plan.capability == "period_movement"
    assert plan.operation == "summary"
    assert _form_binding(plan)["mode"] == "portfolio_overview"

    # THE SCOPE SURVIVED, as a governed predicate on a governed field.
    assert plan.population.lens == "direct"
    predicates = {(p.canonical_field, p.value)
                  for p in plan.population.scope_predicates}
    assert ("source_portfolio_type", "direct") in predicates

    eligible, why, detail = material_summary.check_eligibility(plan)
    assert eligible, f"{why}: {detail}"


def test_Q19_and_Q18_are_the_same_form_and_differ_only_in_population():
    whole = _plan(change_form="material_summary", operation="movement")
    direct = _plan(change_form="material_summary", operation="movement",
                   population={"base": "funded", "lens": "direct"})
    assert whole.capability == direct.capability
    assert whole.operation == direct.operation
    assert whole.population.lens != direct.population.lens


def test_Q19_attribution_variant_reaches_the_bridge_owner_with_its_scope():
    plan = _plan(change_form="attribution", operation="bridge",
                 population={"base": "funded", "lens": "direct"})
    assert plan.capability == "funded_bridge"
    assert plan.population.lens == "direct"
    assert not material_summary.claims(plan)


# --------------------------------------------------------------------------- #
# Q20 — a named metric, and nothing about this sprint may move it
# --------------------------------------------------------------------------- #
# "How did drawdown loans change last month?" read as a request about a NAMED
# governed metric is a metric delta: the reader said which number they meant, so
# the composition does not get to choose the candidate set. Its routing is
# unchanged by this sprint, and that is the control.
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("operation", ["movement", "compare"])
def test_Q20_metric_delta_routing_is_unchanged(operation):
    plan = _plan(change_form="metric_delta", operation=operation,
                 measures=[{"concept": "current_outstanding_balance"}])
    assert plan.capability == "period_movement"
    # NOT canonicalised to `summary`. The operation the reader's request implied
    # is the operation that executes, because this form has no canonical
    # operation of its own to collapse into.
    assert plan.operation == operation
    assert _form_binding(plan)["mode"] == "requested_metric"
    assert [m.concept for o in plan.outputs for m in o.measures] == \
        ["current_outstanding_balance"]


def test_Q20_is_not_claimed_by_the_material_summary_runtime():
    plan = _plan(change_form="metric_delta", operation="movement",
                 measures=[{"concept": "current_outstanding_balance"}])
    assert not material_summary.claims(plan)


def test_a_metric_delta_that_names_nothing_is_still_a_clarification():
    # The correction is scoped to one form. Q20 without its metric does not
    # become Q18.
    result = DeterministicCompiler(CompilerContext()).compile(
        parse_candidate_intent({
            "schema_version": "candidate_intent/1.0",
            "capability": "generic_analysis", "operation": "movement",
            "change_form": "metric_delta", "measures": [],
            "population": {"base": "funded"}, "time": dict(_LAST_MONTH)}))
    assert result.outcome != OUTCOME_PLAN
    assert result.plan is None


# --------------------------------------------------------------------------- #
# The family, side by side
# --------------------------------------------------------------------------- #

def test_the_three_forms_reach_three_distinct_execution_contracts():
    """The distinction the coarse fixture could not express, in one assertion."""
    broad = _plan(change_form="material_summary", operation="movement")
    delta = _plan(change_form="metric_delta", operation="movement",
                  measures=[{"concept": "current_outstanding_balance"}])
    drivers = _plan(change_form="attribution", operation="bridge")

    contracts = {
        (p.capability, p.operation, _form_binding(p)["mode"])
        for p in (broad, delta, drivers)}
    assert len(contracts) == 3, contracts
    assert ("period_movement", "summary", "portfolio_overview") in contracts
    assert ("period_movement", "movement", "requested_metric") in contracts
    assert ("funded_bridge", "bridge", None) in contracts

    # Exactly one of them is the material-summary runtime's.
    claimed = [p for p in (broad, delta, drivers) if material_summary.claims(p)]
    assert claimed == [broad]
