"""D22 (owner decision 2026-09-30): the completion run-rate is measured on the
calendar, over any whole number of weeks the history covers.

Found reading how "What is the 8-week completion run rate?" [125] could be
answered: the forecast's "5-week average" was the mean of the last five
extract-to-extract changes in the completed stock, each treated as one week.
The pipeline is reported ad hoc (D15) — the production history holds 90
extracts in about 54 weeks, the latest two three days apart — so each "week"
was a few days of completions, and the run-rate (and every scale date built on
it) was understated by however close together the extracts were.

The owner's rule: a run-rate over N weeks is the amount of the cases that
completed in the N x 7 days to the latest extract, by each case's own
completion date, per week over those weeks and per month at 52/12 weeks a
month. The history owner publishes it for every window from three weeks to the
span of the extracts; the forecast's own window is five weeks; a window the
history does not cover is declined, never shortened.
"""
from __future__ import annotations

import pytest

from mi_agent import plan_forecast_runtime as forecast_rt
from mi_agent import semantic_model
from mi_agent.interpretation_v2.compiler import CompilerContext, DeterministicCompiler
from mi_agent.interpretation_v2.intent import parse_candidate_intent
from mi_agent.interpretation_v2.vocabulary import SPECIALIST_MEASURE_DEFINITIONS
from mi_agent_api import forecast_extrapolation as fx
from mi_agent_api import pipeline_contract as pc
from mi_agent_api import pipeline_history as history
from tests.interpretation_v2.test_specialist_runtime_forecast import (  # noqa: F401
    _CLIENT, _NEAR, _owner, _served, funded_root)


def _case(on, amount, stated=True):
    return {"completed_on": on, "completed_amount": amount,
            "completed_on_stated": stated}


# --------------------------------------------------------------------------- #
# the owner: completions by date, over whole weeks
# --------------------------------------------------------------------------- #

_TIMELINES = {
    "a": _case("2026-09-24", 100_000.0),        # on the last day
    "b": _case("2026-09-18", 200_000.0),
    "c": _case("2026-08-28", 300_000.0),        # day 28 before: in 4 weeks
    "d": _case("2026-08-27", 400_000.0),        # day 29 before: in 5 weeks
    "e": _case("2026-06-01", 999_000.0),        # outside every window below
    "f": {"completed_on": None},                # never completed
}


def test_a_window_is_the_completions_dated_inside_it():
    rr = history.completion_run_rate(_TIMELINES, ["2026-07-30", "2026-09-24"])
    assert (rr["minWeeks"], rr["maxWeeks"]) == (3, 8)
    by = {w["weeks"]: w for w in rr["byWindow"]}
    assert sorted(by) == list(range(3, 9))
    four, five = by[4], by[5]
    assert (four["from"], four["to"]) == ("2026-08-28", "2026-09-24")
    assert (four["cases"], four["amount"]) == (3, 600_000.0)
    assert (five["cases"], five["amount"]) == (4, 1_000_000.0)
    assert five["weeklyAmount"] == 200_000.0
    assert five["monthlyAmount"] == pytest.approx(200_000.0 * 52 / 12, abs=0.01)


def test_the_extracts_spacing_does_not_move_the_rate():
    """The finding: the same completions, reported every 3 days or every 7,
    are the same run-rate."""
    weekly = [f"2026-{d}" for d in ("07-30", "08-06", "08-13", "08-20", "08-27",
                                    "09-03", "09-10", "09-17", "09-24")]
    ad_hoc = ["2026-07-30", "2026-08-02", "2026-08-05", "2026-09-21", "2026-09-24"]
    assert history.completion_run_rate(_TIMELINES, weekly)["byWindow"] == \
        history.completion_run_rate(_TIMELINES, ad_hoc)["byWindow"]


def test_a_window_the_history_does_not_cover_is_not_published():
    rr = history.completion_run_rate(_TIMELINES, ["2026-09-03", "2026-09-24"])
    assert rr["maxWeeks"] == 3 and [w["weeks"] for w in rr["byWindow"]] == [3]
    short = history.completion_run_rate(_TIMELINES, ["2026-09-14", "2026-09-24"])
    assert not short["available"] and short["byWindow"] == []
    assert "at least 3" in short["reason"]


def test_a_completion_with_no_amount_leaves_the_window_without_one():
    timelines = dict(_TIMELINES, g=_case("2026-09-20", None))
    by = {w["weeks"]: w for w in history.completion_run_rate(
        timelines, ["2026-07-30", "2026-09-24"])["byWindow"]}
    assert by[3]["casesWithoutAmount"] == 1
    assert by[3]["amount"] is None and by[3]["monthlyAmount"] is None


def test_the_history_model_publishes_it_from_the_extracts():
    model = pc.build_pipeline_history(_NEAR, _CLIENT)
    rr = model["completionRunRate"]
    assert rr["method"] == "calendar" and rr["asOf"] == "2025-12-01"
    assert rr["maxWeeks"] == 8
    cut = pc.build_pipeline_history(_NEAR, _CLIENT, as_of="2025-11-01")
    assert cut["completionRunRate"]["asOf"] == "2025-11-01"


# --------------------------------------------------------------------------- #
# the forecast reads it
# --------------------------------------------------------------------------- #

def test_the_forecast_run_rate_is_the_five_week_calendar_window(funded_root):
    rr = _owner(funded_root, _NEAR)["completionRunRateForecast"]
    five = history.run_rate_window(pc.build_pipeline_history(_NEAR, _CLIENT), 5)
    assert rr["baseMonthlyRunRate"] == pytest.approx(five["monthlyAmount"], abs=0.01)
    assert rr["runRateWindow"] == five
    assert rr["runRateMethod"] == "calendar"
    assert f"the 5 weeks {five['from']} to {five['to']}" in \
        rr["assumptions"]["completionSignal"]
    assert rr["runRateByWindow"][0]["weeks"] == 3


# --------------------------------------------------------------------------- #
# the agent: any window the history covers, stated as weeks
# --------------------------------------------------------------------------- #

def _window_plan(**time):
    result = DeterministicCompiler(CompilerContext()).compile(parse_candidate_intent({
        "schema_version": "candidate_intent/1.0", "capability": "forecast",
        "operation": "point_in_time", "population": {"base": "forecast"},
        "measures": [{"concept": "forecast_completion_rate"}], "time": time}))
    assert result.plan is not None, result.codes()
    return result.plan.to_dict()


def _run(plan, funded_root):
    return forecast_rt.execute(plan, output_root=funded_root, pipeline_root=_NEAR,
                               client_id=_CLIENT)


@pytest.mark.parametrize("weeks", [3, 8])
def test_a_named_window_is_the_owners_row_for_it(weeks, funded_root):
    plan = _window_plan(form="range", grain="weekly", periods_back=weeks,
                        labels=[f"{weeks}-week"])
    assert forecast_rt.check_eligibility(plan)[:2] == (True, "")
    out = _run(plan, funded_root)
    assert out.ok, out.detail
    row = history.run_rate_window(pc.build_pipeline_history(_NEAR, _CLIENT), weeks)
    assert out.value == row["monthlyAmount"]
    assert out.receipt["window"]["length"] == weeks
    assert out.receipt["explain"] == ""     # the owner's 5-week companions are not this


def test_a_window_past_the_history_is_declined_naming_what_it_covers(funded_root):
    out = _run(_window_plan(form="range", grain="weekly", periods_back=20),
               funded_root)
    assert (out.ok, out.reason) == (False, forecast_rt.PERIOD_NOT_SUPPORTED)
    assert "3 to 8 weeks" in out.detail


@pytest.mark.parametrize("time", [
    {"form": "range", "labels": ["8-week"]},
    {"form": "range", "grain": "monthly", "periods_back": 3},
])
def test_a_window_not_stated_in_weeks_is_declined(time):
    ok, why, _ = forecast_rt.check_eligibility(_window_plan(**time))
    assert (ok, why) == (False, forecast_rt.PERIOD_NOT_SUPPORTED)


def test_the_answer_states_the_window(monkeypatch, funded_root):
    from mi_agent_api.mi_service import _governed_plan_coverage
    intent = {"schema_version": "candidate_intent/1.0", "capability": "forecast",
              "operation": "point_in_time", "population": {"base": "forecast"},
              "measures": [{"concept": "forecast_completion_rate"}],
              "time": {"form": "range", "grain": "weekly", "periods_back": 8,
                       "labels": ["8-week"]}}
    payload, record = _served(intent, monkeypatch, funded_root, pipeline_root=_NEAR)
    assert payload is not None, record.get("execution")
    row = history.run_rate_window(pc.build_pipeline_history(_NEAR, _CLIENT), 8)
    answer = payload["answer"]
    assert answer.startswith(f"Completion run-rate over the 8 weeks {row['from']} "
                             f"to {row['to']}: ")
    assert f"{row['cases']} cases completed in the window" in answer
    assert _governed_plan_coverage(payload)["unaccounted"] == []


def test_the_owners_own_window_is_stated_too(monkeypatch, funded_root):
    intent = {"schema_version": "candidate_intent/1.0", "capability": "forecast",
              "operation": "point_in_time", "population": {"base": "forecast"},
              "measures": [{"concept": "forecast_completion_rate"}],
              "time": {"form": "current"}}
    payload, record = _served(intent, monkeypatch, funded_root, pipeline_root=_NEAR)
    assert payload is not None, record.get("execution")
    five = history.run_rate_window(pc.build_pipeline_history(_NEAR, _CLIENT), 5)
    assert f"over the 5 weeks {five['from']} to {five['to']}" in payload["answer"]


def test_the_model_is_told_how_to_state_a_window():
    definition = SPECIALIST_MEASURE_DEFINITIONS["forecast_completion_rate"]
    assert "`grain` weekly" in definition and "`periods_back`" in definition
    assert semantic_model.load("forecast").measures[
        "forecast_completion_rate"].window["grain"] == "weekly"
