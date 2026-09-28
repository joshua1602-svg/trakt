"""A current anchor is an endpoint; the deterministic owner supplies the other end.

THE FINDING. The v1 live gate read "what materially changed in the direct funded
portfolio over the latest reporting period?" as `material_summary` anchored at
`current`, with the Direct scope intact — and the runtime refused it
`PERIOD_NOT_A_PAIR`. The model's reading was defensible: "the latest reporting
period" names the endpoint, and for a form that intrinsically compares against the
preceding governed state the other end is not the reader's to author.

THE TEMPORAL AUTHORITY RULE, which these controls pin:

    the reader / interpreter   supplies the anchor, or an explicit pair
    the analytical owner       supplies any comparison state the FORM requires

    current anchor + a form that intrinsically compares
        -> the resolver's existing `current_vs_previous`
        -> two GOVERNED snapshots, FROM and TO, read from the catalogue

WHAT IS NOT DONE HERE. No new temporal slot, no `period_pair_owner`, no new enum,
and no rewriting of the intent to pretend the model emitted `relative_pair`. The
interpreted form stays `current` in provenance; the resolution method is the
owner's, recorded beside it. The comparison state is never selected by the model.

NO QUESTION TEXT IS READ. Every intent below is built from structured slots, and
`test_no_question_text_reaches_the_period_decision` proves the decision cannot see
one.
"""
from __future__ import annotations

import inspect
import re

import pandas as pd
import pytest

from mi_agent import plan_material_summary as material_summary
from mi_agent import plan_temporal_runtime as temporal
from mi_agent.interpretation_v2.compiler import (CompilerContext,
                                                 DeterministicCompiler)
from mi_agent.interpretation_v2.intent import parse_candidate_intent
from mi_agent.interpretation_v2.outcomes import OUTCOME_PLAN, OUTCOME_REFUSE
from mi_agent.period_change.models import (METHOD_CURRENT_VS_PREVIOUS,
                                           METHOD_MONTH_ON_MONTH,
                                           PeriodChangeFailure, SnapshotFrame)
from mi_agent.period_change.periods import PeriodRequest, resolve_periods


def _intent(*, change_form, operation, time_form, capability="period_movement",
            measures=(), grain=None, labels=(), lens="all"):
    time = {"form": time_form}
    if grain:
        time["grain"] = grain
    if labels:
        time["labels"] = list(labels)
    return parse_candidate_intent({
        "schema_version": "candidate_intent/1.0",
        "capability": capability, "operation": operation,
        "change_form": change_form,
        "population": {"base": "funded", "lens": lens, "seasoning": "any"},
        "measures": [{"concept": c} for c in measures],
        "dimensions": [], "filters": [], "time": time,
        "comparison": {"kind": "none"}, "ambiguity": [], "evidence": [],
    })


def _plan(**kwargs):
    result = DeterministicCompiler(CompilerContext()).compile(_intent(**kwargs))
    assert result.plan is not None, [r.code for r in (result.reasons or ())]
    return result.plan


def _snapshot(snapshot_id, reporting_date):
    return SnapshotFrame(snapshot_id=snapshot_id, reporting_date=reporting_date,
                         frame=pd.DataFrame({"loan_id": [1]}),
                         dataset_label=snapshot_id,
                         dataset_reference=snapshot_id, row_count=1,
                         portfolio_ids=())


GOVERNED_BOOK = [_snapshot("s1", "2026-04-30"),
                 _snapshot("s2", "2026-05-31"),
                 _snapshot("s3", "2026-06-30")]


# --------------------------------------------------------------------------- #
# 1. material_summary + current -> current_vs_previous
# --------------------------------------------------------------------------- #

def test_material_summary_anchored_at_current_is_admitted():
    plan = _plan(change_form="material_summary", operation="summary",
                 time_form="current")
    eligible, reason, detail = material_summary.check_eligibility(plan)
    assert eligible, f"{reason}: {detail}"


def test_material_summary_at_current_asks_the_owner_for_the_adjacent_pair():
    plan = _plan(change_form="material_summary", operation="summary",
                 time_form="current")
    request = material_summary.period_request(plan)
    assert request.relative_mode == METHOD_CURRENT_VS_PREVIOUS
    # The anchor is not turned into a date here. Nothing is requested by label.
    assert request.requested_start is None
    assert request.requested_end is None


def test_the_owner_resolves_that_request_to_two_governed_snapshots():
    """The whole point: FROM and TO are real catalogue snapshots, not arithmetic."""
    plan = _plan(change_form="material_summary", operation="summary",
                 time_form="current")
    resolution = resolve_periods(GOVERNED_BOOK,
                                 material_summary.period_request(plan))
    assert resolution.resolution_method == METHOD_CURRENT_VS_PREVIOUS
    assert resolution.start_snapshot.snapshot_id == "s2"
    assert resolution.end_snapshot.snapshot_id == "s3"
    assert resolution.start_snapshot.reporting_date == "2026-05-31"
    assert resolution.end_snapshot.reporting_date == "2026-06-30"
    # Neither end was nudged to fit: both are snapshots the book actually has.
    assert resolution.start_adjusted is False
    assert resolution.end_adjusted is False


def test_the_scope_survives_a_current_anchor():
    """The v1 case exactly: the Direct lens must not be lost on the way."""
    plan = _plan(change_form="material_summary", operation="summary",
                 time_form="current", lens="direct")
    assert material_summary.check_eligibility(plan)[0] is True
    assert ("source_portfolio_type", "direct") in [
        (p.canonical_field, p.value) for p in plan.population.scope_predicates]


def test_current_is_an_anchor_and_is_not_reclassified_as_a_pair():
    """The provenance distinction. `current` still does not MEAN two snapshots."""
    assert "current" in material_summary.ANCHOR_PERIOD_FORMS
    assert "current" not in material_summary.PAIR_PERIOD_FORMS
    assert "current" in material_summary.ADMITTED_PERIOD_FORMS

    plan = _plan(change_form="material_summary", operation="summary",
                 time_form="current")
    # the intent was not rewritten to claim a pair
    assert plan.period.form == "current"


# --------------------------------------------------------------------------- #
# 2. an explicit pair is preserved, not overwritten by the anchor rule
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("grain,method", [
    (None, METHOD_CURRENT_VS_PREVIOUS),
    ("monthly", METHOD_MONTH_ON_MONTH),
])
def test_an_explicit_relative_pair_keeps_its_own_method(grain, method):
    plan = _plan(change_form="material_summary", operation="summary",
                 time_form="relative_pair", grain=grain)
    assert material_summary.period_request(plan).relative_mode == method


def test_an_explicitly_named_span_is_still_passed_through_as_labels():
    plan = _plan(change_form="material_summary", operation="summary",
                 time_form="explicit_period", labels=["2026-03", "2026-06"])
    request = material_summary.period_request(plan)
    assert (request.requested_start, request.requested_end) == ("2026-03",
                                                                "2026-06")
    assert request.relative_mode is None


# --------------------------------------------------------------------------- #
# 3. an unavailable or untranslatable comparison state refuses
# --------------------------------------------------------------------------- #

def test_a_book_with_one_snapshot_refuses_rather_than_inventing_a_date():
    plan = _plan(change_form="material_summary", operation="summary",
                 time_form="current")
    with pytest.raises(PeriodChangeFailure) as excinfo:
        resolve_periods([_snapshot("only", "2026-06-30")],
                        material_summary.period_request(plan))
    assert "two governed portfolio snapshots" in str(excinfo.value)


def test_a_series_is_still_not_an_anchor():
    """Many states is not one endpoint, and is not completed into a pair."""
    assert "series" not in material_summary.ADMITTED_PERIOD_FORMS


def test_an_untranslatable_grain_is_still_refused():
    plan = _plan(change_form="material_summary", operation="summary",
                 time_form="relative_pair", grain="weekly")
    eligible, reason, _ = material_summary.check_eligibility(plan)
    assert eligible is False
    assert reason == material_summary.PERIOD_GRAIN_UNTRANSLATABLE


# --------------------------------------------------------------------------- #
# 4 + 5. the other forms are NOT broadened because material_summary was
# --------------------------------------------------------------------------- #

def test_metric_delta_with_an_explicit_pair_is_unchanged():
    plan = _plan(change_form="metric_delta", operation="movement",
                 time_form="relative_pair",
                 measures=["current_outstanding_balance"])
    assert plan.capability == "period_movement"
    assert plan.operation == "movement"
    assert plan.period.form == "relative_pair"


def test_metric_delta_at_a_bare_current_keeps_its_existing_refusal():
    """NOT broadened. A named metric's movement still needs more than one state.

    `material_summary` gained the anchor because a comparison state is part of
    what THAT form is. `metric_delta` names one quantity and asks how much it
    moved; the existing contract requires a period richer than a single point,
    and this sprint does not relax it just because its sibling changed.
    """
    result = DeterministicCompiler(CompilerContext()).compile(
        _intent(change_form="metric_delta", operation="movement",
                time_form="current", measures=["current_outstanding_balance"]))
    assert result.plan is None
    assert result.outcome == OUTCOME_REFUSE


def test_the_anchor_rule_lives_only_inside_the_material_summary_perimeter():
    """No other form's plan is claimed by this runtime, at `current` or at all."""
    for change_form, operation, measures in [
            ("metric_delta", "movement", ["current_outstanding_balance"]),
            ("attribution", "bridge", ["funded_balance_movement"]),
            ("level_comparison", "compare", ["current_outstanding_balance"]),
    ]:
        capability = {"metric_delta": "period_movement",
                      "attribution": "funded_bridge",
                      "level_comparison": "generic_analysis"}[change_form]
        result = DeterministicCompiler(CompilerContext()).compile(
            _intent(change_form=change_form, operation=operation,
                    capability=capability, time_form="current",
                    measures=measures))
        if result.plan is None:
            continue
        assert material_summary.claims(result.plan) is False, change_form


# --------------------------------------------------------------------------- #
# 6. level_comparison keeps its own comparison semantics
# --------------------------------------------------------------------------- #

def test_level_comparison_still_requires_a_pair_at_its_own_runtime():
    """Its owner is the generic temporal runtime, and that owner wants a pair.

    A `compare` at a bare `current` states one state to put side by side with
    nothing, and is refused by the owner rather than completed for it. Note the
    asymmetry is deliberate: `material_summary` gained the anchor because a
    comparison state is part of what that form IS, whereas a level comparison is
    a request to show two states the reader has in mind — so the reader names
    them, and a single anchor is genuinely incomplete.
    """
    plan = _plan(change_form="level_comparison", operation="compare",
                 capability="generic_analysis", time_form="current",
                 measures=["current_outstanding_balance"])
    eligible, reason, _ = temporal.check_temporal_eligibility(plan)
    assert eligible is False
    assert reason == temporal.PERIOD_NOT_TEMPORAL


def test_level_comparison_with_a_pair_is_accepted_by_its_owner():
    plan = _plan(change_form="level_comparison", operation="compare",
                 capability="generic_analysis", time_form="relative_pair",
                 measures=["current_outstanding_balance"])
    eligible, reason, _ = temporal.check_temporal_eligibility(plan)
    assert eligible is True, reason


# --------------------------------------------------------------------------- #
# 7. attribution: only as far as B3 proved
# --------------------------------------------------------------------------- #

def test_attribution_at_current_compiles_with_a_capability_owned_window():
    """What IS true today, and no more.

    `bridge` owns its own window, so a `current` anchor compiles and the plan
    records that the capability owns the period. What does NOT exist yet is a
    governed-plan runtime for `funded_bridge`: the calculation owner accepts a
    `current_vs_previous` request (it shares the resolver with every other
    period-change figure), but nothing routes a PLAN to it. That is a bounded
    connectivity item for the serving sprint, and is deliberately not built here.
    """
    plan = _plan(change_form="attribution", operation="bridge",
                 capability="funded_bridge", time_form="current",
                 measures=["funded_balance_movement"])
    assert plan.period.form == "current"
    assert plan.period.owned_by_capability is True
    # and no runtime in this repository claims it
    assert material_summary.claims(plan) is False
    assert temporal.check_temporal_eligibility(plan)[0] is False


def test_the_bridge_calculation_owner_accepts_the_governed_adjacent_pair():
    """The calculation half of B3, proved without a plan runtime.

    `balance_bridge` is computed from the pair `resolve_periods` returns, so the
    temporal behaviour attribution needs already exists and is the same resolver
    `material_summary` uses. Only the plan route is missing.
    """
    resolution = resolve_periods(
        GOVERNED_BOOK, PeriodRequest(relative_mode=METHOD_CURRENT_VS_PREVIOUS))
    assert resolution.resolution_method == METHOD_CURRENT_VS_PREVIOUS
    assert (resolution.start_snapshot.snapshot_id,
            resolution.end_snapshot.snapshot_id) == ("s2", "s3")


# --------------------------------------------------------------------------- #
# 8. no question text reaches the period decision
# --------------------------------------------------------------------------- #

def test_no_question_text_reaches_the_period_decision():
    """The period decision cannot READ a question, proved structurally.

    Not "the word never appears": these functions explain themselves in comments
    and name the reader in their refusal messages, which is prose and is the point
    of a governed refusal. What must be absent is an ACCESS — a `question`
    parameter, an attribute or key read of one, or a regex over anything.
    """
    for function in (material_summary.period_request,
                     material_summary.check_eligibility):
        name = function.__name__
        assert "question" not in inspect.signature(function).parameters, name

        source = inspect.getsource(function)
        for forbidden in (r"\.question\b",
                          r"\[\s*['\"]question['\"]\s*\]",
                          r"get\(\s*['\"]question['\"]",
                          r"\bimport re\b", r"\bre\.match\(",
                          r"\bre\.search\(", r"\bre\.compile\("):
            assert not re.search(forbidden, source), (name, forbidden)


def test_the_period_request_is_decided_by_the_form_alone():
    """Same period slot, different everything else -> the same request."""
    a = material_summary.period_request(
        _plan(change_form="material_summary", operation="summary",
              time_form="current"))
    b = material_summary.period_request(
        _plan(change_form="material_summary", operation="summary",
              time_form="current", lens="acquired"))
    assert a.relative_mode == b.relative_mode == METHOD_CURRENT_VS_PREVIOUS


# --------------------------------------------------------------------------- #
# 9. the receipt records the snapshots that were actually read
# --------------------------------------------------------------------------- #

class _Result:
    """Stands in for the workflow result, carrying a REAL period resolution."""

    workflow_id = "period_change.workflow"
    request_interpretation = {"mode": "portfolio_overview"}

    def __init__(self, resolution):
        self.period_resolution = resolution


def test_the_receipt_records_the_interpreted_anchor_and_the_resolved_pair():
    """Both halves, kept apart: what was asked, and what the owner decided."""
    plan = _plan(change_form="material_summary", operation="summary",
                 time_form="current")
    resolution = resolve_periods(GOVERNED_BOOK,
                                 material_summary.period_request(plan))
    receipt = material_summary.receipt(
        plan, _Result(resolution),
        {"config_source": "config/mi/insights.yaml", "insight_count": 0,
         "omitted": (), "status": "ok"})

    # the INTERPRETED form — the model's reading, unrewritten
    assert receipt["interpreted_period_form"] == "current"
    assert receipt["period_completed_by_form"] is True

    # the OWNER's resolution — method, FROM and TO, from the catalogue
    resolved = receipt["period_resolution"]
    assert resolved["resolution_method"] == METHOD_CURRENT_VS_PREVIOUS
    assert resolved["resolved_start_snapshot"]["reporting_date"] == "2026-05-31"
    assert resolved["resolved_end_snapshot"]["reporting_date"] == "2026-06-30"
    assert resolved["resolved_start_snapshot"]["snapshot_id"] == "s2"
    assert resolved["resolved_end_snapshot"]["snapshot_id"] == "s3"


def test_an_explicit_pair_is_receipted_as_not_completed_by_the_form():
    plan = _plan(change_form="material_summary", operation="summary",
                 time_form="relative_pair")
    resolution = resolve_periods(GOVERNED_BOOK,
                                 material_summary.period_request(plan))
    receipt = material_summary.receipt(
        plan, _Result(resolution),
        {"config_source": "x", "insight_count": 0, "omitted": (),
         "status": "ok"})
    assert receipt["interpreted_period_form"] == "relative_pair"
    assert receipt["period_completed_by_form"] is False


# --------------------------------------------------------------------------- #
# the completeness repair is untouched by any of this
# --------------------------------------------------------------------------- #

def test_a_current_anchor_does_not_excuse_a_missing_change_form():
    """Temporal normalisation must not infer the form. Both rules still hold."""
    result = DeterministicCompiler(CompilerContext()).compile(
        parse_candidate_intent({
            "schema_version": "candidate_intent/1.0",
            "capability": "funded_bridge", "operation": "bridge",
            "population": {"base": "funded", "lens": "all",
                           "seasoning": "any"},
            "measures": [{"concept": "funded_balance_movement"}],
            "dimensions": [], "filters": [], "time": {"form": "current"},
            "comparison": {"kind": "none"}, "ambiguity": [], "evidence": [],
        }))
    assert result.plan is None
    assert result.intent.change_form is None


def test_the_anchor_rule_does_not_manufacture_a_form():
    plan = _plan(change_form="material_summary", operation="summary",
                 time_form="current")
    binding = plan.provenance.compiler_bindings["change_form"]
    assert binding["form"] == "material_summary"
    assert plan.provenance.intent_claims["change_form"] == "material_summary"
