"""What moved in the whole pipeline — stage movement's `material_summary` (§28).

The 15:53 check read "pipeline movement since the prior extract" as a "what
moved" summary over the stage-movement figures (cases and amount moved, since
the previous extract). The form was bound to the FUNDED book's owner of
`material_summary`, the plan stayed with stage movement, and the runtime
refused an operation it had no owner for.

The movement owner already publishes the answer: every case between its latest
pair of extracts classified once as arrived, moved stage, left or stayed, with
counts and amounts and a reconciliation to both extracts. The pair is the
latest extract and the one before it — D15's "previous". These tests pin that
the owner of the figures owns the form, that the answer is the owner's own
totals, and what the summary refuses.
"""
from __future__ import annotations

import pytest

from mi_agent import plan_stage_movement_runtime as stage_rt
from mi_agent.interpretation_v2.compiler import CompilerContext, DeterministicCompiler
from mi_agent.interpretation_v2.intent import parse_candidate_intent
from mi_agent_api.mi_service import _governed_plan_coverage
from tests.interpretation_v2 import test_specialist_runtime_stage_movement as sm
from tests.interpretation_v2.test_specialist_runtime_stage_movement import (  # noqa: F401
    payload)

#: The reading the model gave (15:53, vocabulary 2.16.0): an open "what moved"
#: over the stage-movement figures, since the previous extract.
_READING = {
    "schema_version": "candidate_intent/1.0",
    "capability": "pipeline_stage_movement", "operation": "movement",
    "change_form": "material_summary",
    "population": {"base": "pipeline", "lens": "all", "seasoning": "any",
                   "source_reference": None},
    "measures": [{"concept": "cases_moved"}, {"concept": "amount_moved"}],
    "filters": [], "dimensions": [],
    "time": {"form": "previous_reporting_period",
             "labels": ["since the prior extract"]},
}


def _plan(**over):
    result = DeterministicCompiler(CompilerContext()).compile(
        parse_candidate_intent(dict(_READING, **over)))
    assert result.plan is not None, [(r.code, r.detail) for r in result.reasons]
    return result.plan.to_dict()


def test_the_owner_of_the_figures_owns_the_form():
    plan = _plan()
    assert plan["capability"] == "pipeline_stage_movement"
    binding = plan["provenance"]["compiler_bindings"]["change_form"]
    assert binding["form"] == "material_summary"
    # was `period_movement` — the funded book's owner of the form
    assert binding["capability"] == "pipeline_stage_movement"
    assert stage_rt.is_summary(plan)
    assert stage_rt.check_eligibility(plan) == (True, "", "")


def test_the_answer_is_the_owners_own_totals(monkeypatch, payload):
    served, record = sm._served(dict(_READING), monkeypatch)
    assert served is not None, record.get("execution")
    answer = served["answer"]
    totals = payload["event_totals"]
    by_stage = payload["reconciliation"]["by_stage"]
    assert f"{payload['comparison_date']}" in answer
    assert f"{payload['as_of_date']}" in answer
    for cls, words in (("new_arrival", "arrived"),
                       ("stage_transition", "moved to another stage"),
                       ("departure", "left the extracts"),
                       ("stayer", "stayed at their stage")):
        n = totals[cls]["case_count"]
        assert f"{n} case{'' if n == 1 else 's'} {words}" in answer, (cls, answer)
    opening = sum(r["opening_case_count"] for r in by_stage)
    closing = sum(r["closing_case_count"] for r in by_stage)
    assert f"went from {opening:,} cases to {closing:,} cases" in answer
    opening_amount = sum(r["opening_amount"] for r in by_stage)
    closing_amount = sum(r["closing_amount"] for r in by_stage)
    if round(closing_amount - opening_amount):
        assert (" (up " in answer) or (" (down " in answer)
    assert "reconciles to both extracts" in answer
    # the per-stage reconciliation is the table
    tables = [a for a in served["artifacts"] if a["type"] == "table"]
    assert tables and len(tables[0]["rows"]) == len(by_stage)
    assert _governed_plan_coverage(served)["unaccounted"] == []


def test_a_summary_naming_no_figure_is_answered_too(monkeypatch):
    served, record = sm._served(dict(_READING, measures=[]), monkeypatch)
    assert served is not None, record.get("execution")


def test_one_stages_movement_is_its_reconciliation_not_the_summary():
    plan = _plan(filters=[{"concept": "pipeline_stage", "comparator": "eq",
                           "value": "KFI"}])
    ok, reason, _ = stage_rt.check_eligibility(plan)
    assert (ok, reason) == (False, stage_rt.FILTERS_NOT_SUPPORTED)


@pytest.mark.parametrize("time", [
    {"form": "relative_pair", "periods_back": 2},
    {"form": "relative_pair", "periods_back": 1, "grain": "monthly"},
    {"form": "explicit_period", "labels": ["August", "September"]},
])
def test_any_pair_but_the_latest_two_extracts_is_refused(time):
    """The owner holds only its latest pair; "since the last snapshot" in any
    shape names it (`plan_reading.names_latest_pair`), another pair does not."""
    plan = _plan(time=time)
    ok, reason, _ = stage_rt.check_eligibility(plan)
    assert (ok, reason) == (False, stage_rt.PERIOD_NOT_SUPPORTED)


def test_a_figure_the_summary_does_not_state_is_refused():
    plan = _plan(measures=[{"concept": "cases_moved"},
                           {"concept": "stage_opening"}])
    ok, reason, _ = stage_rt.check_eligibility(plan)
    assert (ok, reason) == (False, stage_rt.MEASURE_NOT_SUPPORTED)


def test_the_summary_is_one_owner_answer_not_a_composition():
    from mi_agent import plan_composition as composition
    assert stage_rt.serves_figures_together(_plan())
    assert not composition.needs_composition(_plan())


@pytest.mark.parametrize("time", [None, {"form": "current"},
                                  {"form": "previous_reporting_period"},
                                  {"form": "relative_pair", "periods_back": 1}])
def test_every_way_of_naming_the_latest_pair_is_served(time):
    reading = {k: v for k, v in _READING.items() if k != "time"}
    if time is not None:
        reading["time"] = time
    result = DeterministicCompiler(CompilerContext()).compile(
        parse_candidate_intent(reading))
    assert result.plan is not None, [(r.code, r.detail) for r in result.reasons]
    assert stage_rt.check_eligibility(result.plan.to_dict()) == (True, "", "")
