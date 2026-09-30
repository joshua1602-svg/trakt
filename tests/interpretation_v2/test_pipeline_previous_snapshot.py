"""The pipeline's "previous" is the snapshot before the latest (owner decision
D15, 2026-09-30, P0 design §26).

"Week on week is difficult because the reporting of the pipeline is adhoc — so
it should strictly be between the two most recent pipeline snapshots."

The 13:49 check (2026-09-30) asked "How has the pipeline changed since the
previous extract?" and "Latest pipeline against the one before it: what
moved?". The model read both as a `material_summary` — what moved, no measure
named — and normalisation sent them to the FUNDED book's owner of that form,
which refused the pipeline population. The form's owner is now chosen with the
population, and the pipeline answers what moved with each of its headline
figures between its two most recent snapshots, however far apart they are.
"""
from __future__ import annotations

import pytest

from mi_agent import plan_pipeline_runtime as pipeline_rt
from mi_agent import semantic_engine as engine
from mi_agent.interpretation_v2.compiler import CompilerContext, DeterministicCompiler
from mi_agent.interpretation_v2.intent import parse_candidate_intent
from tests.interpretation_v2.test_specialist_runtime_pipeline_dated import (  # noqa: F401
    _CLIENT, _owner, _served, history)

#: The fixture's two most recent snapshots are eleven months apart — the ad
#: hoc reporting D15 is about.
_PREVIOUS, _LATEST = "2025-11-27", "2026-10-29"


def _recorded(question_labels, operation="movement"):
    """The model's own reading at 13:49, as the readback recorded it."""
    return {"schema_version": "candidate_intent/1.0",
            "capability": "pipeline_stage_movement",
            "change_form": "material_summary", "operation": operation,
            "population": {"base": "pipeline"},
            "time": {"form": "relative_pair", "labels": question_labels}}


_SINCE_THE_PREVIOUS = _recorded(["previous extract"])
_THE_ONE_BEFORE = _recorded(["latest pipeline", "the one before it"])


def _compile(payload):
    return DeterministicCompiler(CompilerContext()).compile(
        parse_candidate_intent(payload))


@pytest.mark.parametrize("payload", [_SINCE_THE_PREVIOUS, _THE_ONE_BEFORE])
def test_what_moved_in_the_pipeline_is_the_pipelines_to_answer(payload):
    result = _compile(payload)
    assert result.plan is not None, [(r.code, r.detail) for r in result.reasons]
    plan = result.plan.to_dict()
    assert (plan["capability"], plan["operation"]) == ("pipeline", "summary")
    binding = plan["provenance"]["compiler_bindings"]["change_form"]
    assert (binding["form"], binding["capability"]) == ("material_summary",
                                                        "pipeline")
    assert pipeline_rt.check_eligibility(plan) == (True, "", "")


def test_a_funded_what_moved_still_goes_to_the_funded_owner():
    payload = dict(_SINCE_THE_PREVIOUS, capability="generic_analysis",
                   population={"base": "funded"})
    plan = _compile(payload).plan.to_dict()
    assert plan["capability"] == "period_movement"


def test_what_moved_is_every_headline_figure_between_the_two_latest(history):
    plan = _compile(_SINCE_THE_PREVIOUS).plan.to_dict()
    outcome = pipeline_rt.execute_dated(plan, root=history, client_id=_CLIENT)
    assert outcome.ok, outcome.detail
    assert outcome.receipt["selected_periods"] == [_PREVIOUS, _LATEST]
    by_date, _ = _owner(history)
    for row in outcome.cells:
        metric = row["measure"]
        assert {k: row[k] for k in ("from", "to", "change", "change_pct")} == \
            engine.period_change(by_date[_PREVIOUS]["metrics"][metric],
                                 by_date[_LATEST]["metrics"][metric])
    assert [r["measure"] for r in outcome.cells] == list(pipeline_rt.SUMMARY_MEASURES)


@pytest.mark.parametrize("time", [
    {"form": "relative_pair", "labels": ["latest", "prior"]},
    {"form": "relative_pair", "grain": "weekly", "periods_back": 1,
     "labels": ["week on week"]},
    {"form": "previous_reporting_period", "labels": ["last week"]},
])
def test_previous_prior_and_week_on_week_are_the_same_two_snapshots(history, time):
    payload = {"schema_version": "candidate_intent/1.0", "capability": "pipeline",
               "change_form": "metric_delta", "operation": "movement",
               "population": {"base": "pipeline"},
               "measures": [{"concept": "pipeline_amount"}], "time": time}
    plan = _compile(payload).plan.to_dict()
    outcome = pipeline_rt.execute_dated(plan, root=history, client_id=_CLIENT)
    assert outcome.ok, outcome.detail
    assert outcome.receipt["selected_periods"] == [_PREVIOUS, _LATEST]
    assert all(r["rule"].startswith(pipeline_rt.SNAPSHOT_RULE)
               for r in outcome.receipt["period_resolution"])


def test_the_answer_names_both_snapshots_and_the_gap(monkeypatch, history):
    from mi_agent_api.mi_service import _governed_plan_coverage

    payload, record = _served(_SINCE_THE_PREVIOUS, monkeypatch, history)
    assert payload is not None, record.get("execution")
    answer = payload["answer"]
    assert answer.startswith(f"Between the previous snapshot ({_PREVIOUS}) and "
                             f"the latest snapshot ({_LATEST}), 336 days apart: ")
    assert "live pipeline amount" in answer and "live pipeline case count" in answer
    assert "week before" not in answer
    assert _governed_plan_coverage(payload)["unaccounted"] == []


def test_a_named_measure_change_is_worded_the_same_way(monkeypatch, history):
    payload, _ = _served({"schema_version": "candidate_intent/1.0",
                          "capability": "pipeline", "operation": "compare",
                          "change_form": "level_comparison",
                          "population": {"base": "pipeline"},
                          "measures": [{"concept": "pipeline_amount"}],
                          "time": {"form": "relative_pair",
                                   "labels": ["latest", "prior"]}},
                         monkeypatch, history)
    answer = payload["answer"]
    assert f"the previous snapshot ({_PREVIOUS})" in answer
    assert f"the latest snapshot ({_LATEST}), 336 days apart." in answer
    assert "week before" not in answer
