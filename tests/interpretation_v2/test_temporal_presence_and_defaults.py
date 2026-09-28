"""Absence must not masquerade as explicit semantics.

THE FINDING. The live v3 gate read three questions correctly — the analytical form
was right in all eighteen — and in three of them deliberately stated NO temporal
slot, saying so outright: *"no period is stated on the intent because the
funded_bridge capability defines the movement period itself."* The interpreter was
right; restating deterministic execution policy is not its job. But
`SemanticTime.form` carries a construction default of `current`, so those readings
arrived at the governed boundary indistinguishable from a reader who had asked
about the current state.

    A CONSTRUCTION DEFAULT IS NOT A BUSINESS DEFAULT.

THREE STATES, AND THE BOUNDARY MUST TELL THEM APART:

    EXPLICIT    the reading stated a temporal form        stated=True
    DEFAULTED   it stated none, and the FORM owns one     stated=False, defaulted=True
    ABSENT      it stated none, and no form owns one      stated=False, defaulted=False

WHERE THE SIGNAL COMES FROM. Key presence in the model's payload — `"form" in raw`
— which is a structural fact. No wording, no source span, no phrase rule; proved by
`test_nothing_in_the_presence_path_reads_a_question`.

WHAT WAS NOT ADDED. No new semantic slot: `stated` is provenance, excluded from
`SemanticTime.key()` and from the plan's identity via `_DERIVATION_KEYS`, so two
readings of one question still hash alike. No new period engine: the default is the
resolver's existing `current_vs_previous`. No model change of any kind.
"""
from __future__ import annotations

import inspect
import re

import pandas as pd
import pytest

from mi_agent import plan_material_summary as material_summary
from mi_agent import plan_temporal_runtime as temporal
from mi_agent.interpretation_v2 import compiler as compiler_module
from mi_agent.interpretation_v2.compiler import (CompilerContext,
                                                 DeterministicCompiler)
from mi_agent.interpretation_v2.intent import (SemanticTime,
                                               parse_candidate_intent)
from mi_agent.interpretation_v2.outcomes import OUTCOME_PLAN, OUTCOME_REFUSE
from mi_agent.interpretation_v2.plan import GovernedQueryPlan
from mi_agent.interpretation_v2.vocabulary import (
    CHANGE_FORM_ABSENT_PERIOD_DEFAULT, CHANGE_FORMS)
from mi_agent.period_change.models import (METHOD_CURRENT_VS_PREVIOUS,
                                           METHOD_MONTH_ON_MONTH,
                                           PeriodChangeFailure, SnapshotFrame)
from mi_agent.period_change.periods import resolve_periods

_CAPABILITY = {"material_summary": "period_movement",
               "metric_delta": "period_movement",
               "attribution": "funded_bridge",
               "level_comparison": "generic_analysis"}
_OPERATION = {"material_summary": "summary", "metric_delta": "movement",
              "attribution": "bridge", "level_comparison": "compare"}
_MEASURES = {"material_summary": (), "metric_delta": ("current_outstanding_balance",),
             "attribution": ("funded_balance_movement",),
             "level_comparison": ("current_outstanding_balance",)}


def _payload(change_form, *, time="OMIT", lens="all", grain=None, labels=()):
    body = {
        "schema_version": "candidate_intent/1.0",
        "capability": _CAPABILITY[change_form],
        "operation": _OPERATION[change_form],
        "change_form": change_form,
        "population": {"base": "funded", "lens": lens, "seasoning": "any"},
        "measures": [{"concept": c} for c in _MEASURES[change_form]],
        "dimensions": [], "filters": [],
        "comparison": {"kind": "none"}, "ambiguity": [], "evidence": [],
    }
    if time != "OMIT":
        block = {"form": time} if time is not None else {}
        if grain:
            block["grain"] = grain
        if labels:
            block["labels"] = list(labels)
        body["time"] = block
    return body


def _compile(change_form, **kwargs):
    return DeterministicCompiler(CompilerContext()).compile(
        parse_candidate_intent(_payload(change_form, **kwargs)))


def _plan(change_form, **kwargs):
    result = _compile(change_form, **kwargs)
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

RECEIPT_BRIEF = {"config_source": "config/mi/insights.yaml", "insight_count": 0,
                 "omitted": (), "status": "ok"}


class _Result:
    workflow_id = "period_change.workflow"
    request_interpretation = {"mode": "portfolio_overview"}

    def __init__(self, resolution):
        self.period_resolution = resolution


def _receipt(plan):
    resolution = resolve_periods(GOVERNED_BOOK,
                                 material_summary.period_request(plan))
    return material_summary.receipt(plan, _Result(resolution), RECEIPT_BRIEF)


# --------------------------------------------------------------------------- #
# the defect itself: the three states are distinguishable
# --------------------------------------------------------------------------- #

def test_an_absent_time_slot_is_not_recorded_as_an_explicit_current():
    """THE DEFECT, directly. Same `form`, different provenance."""
    absent = parse_candidate_intent(_payload("attribution"))
    explicit = parse_candidate_intent(_payload("attribution", time="current"))

    # the construction value is the same, which is why this was invisible
    assert absent.time.form == explicit.time.form == "current"
    # and the provenance now separates them
    assert absent.time.stated is False
    assert explicit.time.stated is True


def test_a_time_block_with_no_form_has_not_stated_one_either():
    intent = parse_candidate_intent(_payload("attribution", time=None,
                                             grain="monthly"))
    assert intent.time.grain == "monthly"
    assert intent.time.stated is False


def test_presence_is_provenance_and_never_changes_meaning_or_identity():
    """Excluded from the semantic key AND from the plan hash.

    Two readings of one question must not look divergent because one of them
    reached its period through a default.
    """
    assert SemanticTime(form="current", stated=True).key() == \
        SemanticTime(form="current", stated=False).key()
    assert "stated" in GovernedQueryPlan._DERIVATION_KEYS

    absent = _plan("attribution")
    explicit = _plan("attribution", time="current")
    assert absent.plan_id == explicit.plan_id


# --------------------------------------------------------------------------- #
# 1, 2, 3. material_summary: explicit pair, explicit anchor, absent
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("grain,method", [(None, METHOD_CURRENT_VS_PREVIOUS),
                                          ("monthly", METHOD_MONTH_ON_MONTH)])
def test_1_material_summary_with_an_explicit_pair_keeps_it(grain, method):
    plan = _plan("material_summary", time="relative_pair", grain=grain)
    assert plan.period.stated is True
    assert plan.period.defaulted is False
    assert material_summary.period_request(plan).relative_mode == method


def test_2_material_summary_with_an_explicit_current_anchor_resolves_the_pair():
    plan = _plan("material_summary", time="current")
    assert plan.period.stated is True
    assert plan.period.defaulted is False          # the reader stated it
    assert material_summary.check_eligibility(plan)[0] is True
    assert material_summary.period_request(plan).relative_mode == \
        METHOD_CURRENT_VS_PREVIOUS

    receipt = _receipt(plan)
    assert receipt["interpreted_time_present"] is True
    assert receipt["interpreted_period_form"] == "current"
    assert receipt["temporal_default_applied"] is False


def test_3_material_summary_with_no_time_gets_the_authorised_default():
    plan = _plan("material_summary")
    assert plan.period.stated is False
    assert plan.period.defaulted is True
    assert plan.period.default_method == METHOD_CURRENT_VS_PREVIOUS
    assert plan.period.default_owner == "material_summary"
    assert material_summary.check_eligibility(plan)[0] is True
    assert material_summary.period_request(plan).relative_mode == \
        METHOD_CURRENT_VS_PREVIOUS


def test_3b_the_receipt_says_absent_and_defaulted_and_names_the_snapshots():
    plan = _plan("material_summary")
    receipt = _receipt(plan)

    assert receipt["interpreted_time_present"] is False
    assert receipt["interpreted_period_form"] is None
    assert receipt["temporal_default_applied"] is True
    assert receipt["temporal_default_method"] == METHOD_CURRENT_VS_PREVIOUS
    assert receipt["temporal_default_owner"] == "material_summary"

    resolved = receipt["period_resolution"]
    assert resolved["resolution_method"] == METHOD_CURRENT_VS_PREVIOUS
    assert resolved["resolved_start_snapshot"]["reporting_date"] == "2026-05-31"
    assert resolved["resolved_end_snapshot"]["reporting_date"] == "2026-06-30"


def test_3c_the_two_routes_execute_the_same_pair_and_stay_distinguishable():
    """The requirement stated exactly: same snapshots, different evidence."""
    absent = _receipt(_plan("material_summary"))
    explicit = _receipt(_plan("material_summary", time="current"))

    for receipt in (absent, explicit):
        resolved = receipt["period_resolution"]
        assert resolved["resolution_method"] == METHOD_CURRENT_VS_PREVIOUS
        assert resolved["resolved_start_snapshot"]["snapshot_id"] == "s2"
        assert resolved["resolved_end_snapshot"]["snapshot_id"] == "s3"

    assert (absent["interpreted_time_present"],
            absent["temporal_default_applied"]) == (False, True)
    assert (explicit["interpreted_time_present"],
            explicit["temporal_default_applied"]) == (True, False)


# --------------------------------------------------------------------------- #
# 4, 5. attribution
# --------------------------------------------------------------------------- #

def test_4_attribution_with_an_explicit_current_is_recorded_as_explicit():
    plan = _plan("attribution", time="current")
    assert plan.period.stated is True
    assert plan.period.defaulted is False
    assert plan.period.owned_by_capability is True
    # and its calculation owner resolves that method over governed snapshots
    resolution = resolve_periods(
        GOVERNED_BOOK,
        material_summary.period_request(_plan("material_summary",
                                              time="current")))
    assert resolution.resolution_method == METHOD_CURRENT_VS_PREVIOUS


def test_5_attribution_with_no_time_gets_the_authorised_default():
    plan = _plan("attribution")
    assert plan.period.stated is False
    assert plan.period.defaulted is True
    assert plan.period.default_method == METHOD_CURRENT_VS_PREVIOUS
    assert plan.period.default_owner == "attribution"


def test_5b_attribution_is_still_not_claimed_by_the_material_summary_runtime():
    """The default authorises a WINDOW, not a change of owner."""
    for kwargs in ({}, {"time": "current"}):
        assert material_summary.claims(_plan("attribution", **kwargs)) is False


# --------------------------------------------------------------------------- #
# 6, 7. metric_delta is NOT broadened
# --------------------------------------------------------------------------- #

def test_6_metric_delta_with_an_explicit_pair_is_unchanged():
    plan = _plan("metric_delta", time="relative_pair")
    assert plan.period.stated is True
    assert plan.period.defaulted is False
    assert plan.capability == "period_movement"
    assert plan.operation == "movement"


def test_7_metric_delta_with_no_time_remains_incomplete():
    """No sibling's default is inherited. This is the same refusal as before."""
    result = _compile("metric_delta")
    assert result.plan is None
    assert result.outcome == OUTCOME_REFUSE


def test_7b_metric_delta_owns_no_authorised_default_in_the_policy():
    assert CHANGE_FORM_ABSENT_PERIOD_DEFAULT["metric_delta"] is None


# --------------------------------------------------------------------------- #
# 8, 9. level_comparison is NOT broadened
# --------------------------------------------------------------------------- #

def test_8_level_comparison_with_an_explicit_pair_is_unchanged():
    plan = _plan("level_comparison", time="relative_pair")
    assert plan.period.stated is True
    assert plan.period.defaulted is False
    assert temporal.check_temporal_eligibility(plan)[0] is True


def test_9_level_comparison_with_no_time_remains_incomplete():
    plan = _plan("level_comparison")
    assert plan.period.stated is False
    assert plan.period.defaulted is False, "inherited a default it does not own"
    eligible, reason, _ = temporal.check_temporal_eligibility(plan)
    assert eligible is False
    assert reason == temporal.PERIOD_NOT_TEMPORAL


def test_9b_level_comparison_owns_no_authorised_default_in_the_policy():
    assert CHANGE_FORM_ABSENT_PERIOD_DEFAULT["level_comparison"] is None


def test_9c_only_the_two_proved_forms_own_a_default():
    owned = {form for form, method
             in CHANGE_FORM_ABSENT_PERIOD_DEFAULT.items() if method}
    assert owned == {"material_summary", "attribution"}
    # every governed form has an explicit entry, so a new one cannot inherit by
    # omission
    assert set(CHANGE_FORM_ABSENT_PERIOD_DEFAULT) == set(CHANGE_FORMS)


def test_9d_the_policy_names_the_resolvers_own_method_and_cannot_drift():
    """The vocabulary writes the method as a literal; this is the drift guard."""
    for method in CHANGE_FORM_ABSENT_PERIOD_DEFAULT.values():
        if method:
            assert method == METHOD_CURRENT_VS_PREVIOUS


def test_9e_an_intent_with_no_change_form_gets_no_default_at_all():
    """The completeness repair still owns that case; it never reaches a plan."""
    body = _payload("metric_delta")
    body.pop("change_form")
    body["time"] = {"form": "relative_pair"}
    result = DeterministicCompiler(CompilerContext()).compile(
        parse_candidate_intent(body))
    assert result.plan is None


# --------------------------------------------------------------------------- #
# 10, 11. refusal without invention; scope survives
# --------------------------------------------------------------------------- #

def test_10_one_snapshot_refuses_and_no_prior_date_is_synthesised():
    plan = _plan("material_summary")
    assert plan.period.defaulted is True
    with pytest.raises(PeriodChangeFailure) as excinfo:
        resolve_periods([_snapshot("only", "2026-06-30")],
                        material_summary.period_request(plan))
    assert "two governed portfolio snapshots" in str(excinfo.value)


@pytest.mark.parametrize("lens", ["direct", "acquired"])
@pytest.mark.parametrize("time", ["OMIT", "current", "relative_pair"])
def test_11_the_scope_survives_every_temporal_route(lens, time):
    plan = _plan("material_summary", lens=lens, time=time)
    assert ("source_portfolio_type", lens) in [
        (p.canonical_field, p.value) for p in plan.population.scope_predicates]
    assert material_summary.check_eligibility(plan)[0] is True


# --------------------------------------------------------------------------- #
# 12. no raw-question reads anywhere in the presence path
# --------------------------------------------------------------------------- #

def test_nothing_in_the_presence_path_reads_a_question():
    """Presence comes from a KEY, and these are the functions that decide it."""
    from mi_agent.interpretation_v2 import intent as intent_module

    for function in (intent_module._parse_time,
                     compiler_module.DeterministicCompiler._bind_period,
                     material_summary._period_defaulted_to_owner,
                     material_summary.period_request,
                     material_summary.check_eligibility):
        name = getattr(function, "__name__", str(function))
        assert "question" not in inspect.signature(function).parameters, name
        source = inspect.getsource(function)
        for forbidden in (r"\.question\b", r"\[\s*['\"]question['\"]\s*\]",
                          r"get\(\s*['\"]question['\"]", r"\bimport re\b",
                          r"\bre\.match\(", r"\bre\.search\(",
                          r"\bre\.compile\("):
            assert not re.search(forbidden, source), (name, forbidden)


def test_the_presence_flag_is_read_from_key_presence_only():
    """Behavioural: the same structure with different wording decides the same.

    `provenance.question` is the only place a sentence lives, and changing it
    cannot move the presence verdict.
    """
    from mi_agent.interpretation_v2.intent import IntentProvenance

    for question in ("what changed?", "", "state no period please",
                     "use relative_pair"):
        absent = parse_candidate_intent(
            _payload("material_summary"),
            provenance=IntentProvenance(question=question))
        explicit = parse_candidate_intent(
            _payload("material_summary", time="current"),
            provenance=IntentProvenance(question=question))
        assert absent.time.stated is False
        assert explicit.time.stated is True
