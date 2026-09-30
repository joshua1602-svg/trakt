"""D20 (owner decision 2026-09-30): "over time" is every reporting date held.

The 19:59 full bank declined "Show pipeline amount by stage over time" and
"Show pipeline cases by stage over time" [74, 75]: the model read a series and
stated no span, and the compiler refused it as AMBIGUOUS_PERIOD, so the agent
asked back. The owner's rule: a series stating no span, grain or count is every
reporting date the owner holds, at the owner's own cadence — the pipeline's
extracts as reported, the funded book's monthly runs. Not the latest two
dates: that is "previous" (D15), a comparison; "over time" asks for the trend.

The default is recorded on the plan and disclosed on the answer. A RANGE with
no bounds is still an incomplete request.
"""
from __future__ import annotations

import pytest

from mi_agent import plan_pipeline_runtime as pipeline_rt
from mi_agent import plan_temporal_runtime as temporal
from mi_agent.interpretation_v2.compiler import CompilerContext, DeterministicCompiler
from mi_agent.interpretation_v2.intent import parse_candidate_intent
from mi_agent.interpretation_v2.vocabulary import (
    SERIES_ABSENT_SPAN_DEFAULT, SERIES_ABSENT_SPAN_RULE)
from mi_agent.tests import temporal_snapshot_fixture as fixture
from mi_agent.tests.test_plan_temporal_runtime import (  # noqa: F401
    BASE_INTENT, history, store)
from tests.interpretation_v2.test_specialist_runtime_pipeline import (
    _HISTORY_ROOT, _CLIENT, _PIPELINE_INTENT, _served)

_OVER_TIME = {"form": "series"}


def _compile(payload):
    return DeterministicCompiler(CompilerContext()).compile(
        parse_candidate_intent(payload))


def _pipeline(**over):
    body = dict(_PIPELINE_INTENT, operation="series", time=dict(_OVER_TIME))
    body.update(over)
    return body


# --------------------------------------------------------------------------- #
# the compiler records the default
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("payload", [
    _pipeline(dimensions=["pipeline_stage"]),
    _pipeline(measures=[{"concept": "pipeline_case_count"}],
              dimensions=["pipeline_stage"]),
    _pipeline(),
    dict(BASE_INTENT, operation="series", time=dict(_OVER_TIME)),
])
def test_an_open_series_is_every_reporting_date_and_says_so(payload):
    result = _compile(payload)
    assert result.plan is not None, result.codes()
    period = result.plan.to_dict()["period"]
    assert period["form"] == "series"
    assert period["defaulted"] is True
    assert period["default_method"] == SERIES_ABSENT_SPAN_DEFAULT
    assert period["default_reason"] == SERIES_ABSENT_SPAN_RULE
    assert period["default_owner"] == payload["capability"]


def test_a_range_with_no_bounds_still_asks():
    result = _compile(_pipeline(time={"form": "range"}))
    assert result.plan is None
    assert "AMBIGUOUS_PERIOD" in result.codes()


def test_a_stated_span_is_not_a_default():
    for time in ({"form": "series", "grain": "weekly"},
                 {"form": "series", "periods_back": 6, "grain": "weekly"}):
        period = _compile(_pipeline(time=time)).plan.to_dict()["period"]
        assert not period["defaulted"], time


# --------------------------------------------------------------------------- #
# the pipeline: every extract, as reported
# --------------------------------------------------------------------------- #

def _weekly_owner():
    from mi_agent_api import evolution as evolution_mod
    return evolution_mod.pipeline_evolution(_HISTORY_ROOT, _CLIENT, None)


def test_the_pipeline_by_stage_over_time_is_every_extract(monkeypatch):
    """[74]: the same series "by week" answers, stated from first to last."""
    served, record = _served(_pipeline(dimensions=["pipeline_stage"]), monkeypatch)
    assert served is not None, record.get("execution")
    weeks = sorted({str(p.get("week") or p.get("extract_date"))
                    for p in _weekly_owner()["periods"]})
    executed = served["metadata"]["governedPlan"]["executed"]
    assert sorted(executed["selected_periods"]) == weeks
    answer = served["answer"]
    assert f"across {len(weeks)} weekly extracts ({weeks[0]} to {weeks[-1]})" in answer
    assert " from " in answer and " to £" in answer
    notes = {n["field"]: n["note"] for n in served["sourceNotes"]}
    assert notes.get("period: over time") == SERIES_ABSENT_SPAN_RULE


def test_the_open_series_equals_the_weekly_series(monkeypatch):
    open_series, _ = _served(_pipeline(), monkeypatch)
    weekly, _ = _served(_pipeline(time={"form": "series", "grain": "weekly"}),
                        monkeypatch)
    chart = lambda p: next(a for a in p["artifacts"] if a["type"] == "chart")["rows"]
    assert chart(open_series) == chart(weekly)


def test_the_pipeline_runtime_takes_it():
    plan = _compile(_pipeline(dimensions=["pipeline_stage"])).plan.to_dict()
    assert pipeline_rt.check_eligibility(plan) == (True, "", "")


# --------------------------------------------------------------------------- #
# the funded book: every monthly run
# --------------------------------------------------------------------------- #

def test_the_funded_book_over_time_is_every_run(store):
    plan = _compile(dict(BASE_INTENT, operation="series",
                         time=dict(_OVER_TIME))).plan.to_dict()
    resolution = temporal.resolve_temporal(plan, store, client_id=fixture.CLIENT_ID,
                                           route=fixture.ROUTE)
    assert resolution.ok, (resolution.reason, resolution.detail)
    assert resolution.basis == "every_reporting_date"
    whole = temporal.resolve_temporal(
        _compile(dict(BASE_INTENT, operation="series",
                      time={"form": "series", "labels": ["over time"]})
                 ).plan.to_dict(),
        store, client_id=fixture.CLIENT_ID, route=fixture.ROUTE)
    assert [h.reporting_date for h in resolution.headers] == \
        [h.reporting_date for h in whole.headers]
