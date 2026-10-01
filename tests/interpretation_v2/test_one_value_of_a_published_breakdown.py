"""One value of a breakdown a figure is published by is that figure's own
value (twins run 2026-10-01).

"What is the pipeline amount at Offer stage?" and "When does the upside
forecast reach £100m?" were read correctly and refused: the pipeline runtime
served no filter but expected-completion timing, and the forecast's milestone
path served none at all. Each figure is already PUBLISHED by that dimension —
the Pipeline tab's amount by stage, the milestone's downside / base / upside
dates — so the one value asked for is read off the published breakdown, never
recomputed narrowed: the same rule the shared semantic engine already applies
("Filter on ONE origin_stage").
"""
from __future__ import annotations

import pytest

from mi_agent import plan_decline as decline
from mi_agent import plan_pipeline_runtime as pipeline_rt
from tests.interpretation_v2.test_specialist_runtime_pipeline import (
    _PIPELINE_INTENT, _SOURCE, _plan, _semantics, _served)


def _stage(value):
    return [{"concept": "pipeline_stage", "comparator": "eq", "value": value}]


def _by_stage(measure):
    plan = _plan(operation="breakdown", measures=[{"concept": measure}],
                 dimensions=["pipeline_stage"])
    out = pipeline_rt.execute_current(plan, source=_SOURCE,
                                      semantics=_semantics())
    assert out.ok, out.detail
    return {str(c["pipeline_stage"]).upper(): c["value"] for c in out.cells}


@pytest.mark.parametrize("measure", ["pipeline_amount", "loan"])
def test_one_stage_is_the_by_stage_breakdowns_value(measure):
    plan = _plan(operation="point_in_time", measures=[{"concept": measure}],
                 filters=_stage("OFFER"))
    assert pipeline_rt.check_eligibility(plan) == (True, "", "")
    out = pipeline_rt.execute_current(plan, source=_SOURCE,
                                      semantics=_semantics())
    assert out.ok, out.detail
    assert out.value == pytest.approx(_by_stage(measure)["OFFER"])
    assert out.receipt["result_shape"] == "scalar"
    assert out.receipt["applied_predicates"] == [
        {"field": "pipeline_stage", "op": "eq", "values": ["OFFER"]}]
    assert out.receipt["member"]["dimension"] == "pipeline_stage"


def test_a_closed_stage_is_not_published_for_the_live_pipeline():
    plan = _plan(operation="point_in_time", filters=_stage("COMPLETED"))
    out = pipeline_rt.execute_current(plan, source=_SOURCE,
                                      semantics=_semantics())
    assert not out.ok
    assert out.reason == pipeline_rt.MEMBER_NOT_PUBLISHED
    assert "is not one of them" in out.detail
    assert decline.plain_reason(out.reason) == (
        "the value it narrows to is not one the figure is published for")


def test_one_value_is_not_served_together_with_another_breakdown():
    plan = _plan(operation="breakdown", filters=_stage("OFFER"),
                 dimensions=["broker_channel"])
    ok, why, _ = pipeline_rt.check_eligibility(plan)
    assert (ok, why) == (False, pipeline_rt.FILTERS_NOT_SUPPORTED)


def test_the_answer_names_the_value(monkeypatch):
    payload, record = _served(dict(_PIPELINE_INTENT, operation="point_in_time",
                                   filters=_stage("OFFER")), monkeypatch,
                              semantics=_semantics())
    assert payload is not None, record.get("execution")
    assert record["serving"]["decision"] == "NEW"
    assert payload["answer"].startswith("The live pipeline amount for stage Offer is £")
