"""The owner's production bank, replayed through today's perimeter. Offline.

The model's own output for all 135 questions of the 2026-09-28 production run
was read back from the canary's evidence sink (`qb-plan-readback.yml`) and is
committed without answers or figures. This replays each one through today's
compiler and the specialist runtimes' perimeters and pins the result — so
admitting a question to the governed path is always a decision a test records,
never a side effect of widening a runtime.

WHY PIN BOTH SIDES. A widened perimeter that admits a question the model
mis-encoded answers it with the wrong figure, and on the governed path that is
worse than the legacy answer it replaces. The replay found eight such readings
among the first forecast shapes built; they are pinned as held here so that
enabling their shape requires the vocabulary fix that separates them.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from mi_agent import plan_forecast_runtime as forecast_rt
from mi_agent import plan_pipeline_runtime as pipeline_rt
from mi_agent import plan_runtime_adapter as adapter
from mi_agent import plan_stage_movement_runtime as stage_rt
from mi_agent.interpretation_v2.compiler import CompilerContext, DeterministicCompiler
from mi_agent.interpretation_v2.intent import parse_candidate_intent

_REPO = Path(__file__).resolve().parents[2]
_FIXTURE = (_REPO / "due_diligence" / "evidence" / "qb_plan_readback"
            / "qb_recorded_intents.json")
_CASES = json.loads(_FIXTURE.read_text())["cases"]
_RUNTIMES = (("pipeline", pipeline_rt), ("stage_movement", stage_rt),
             ("forecast", forecast_rt))

#: Newly admitted to the governed path — the whole list, by bank number.
ADMITTED = {
    80: "pipeline",   # pipeline by stage for October and November (D7)
    81: "pipeline",   # October and November pipeline amount (D7)
    85: "forecast",   # the forecast funded balance (D6, the composer)
    107: "forecast", 108: "forecast", 109: "forecast", 110: "forecast",
    111: "forecast",  # when do we reach £25m ... £150m
    # Catalogue batch 1: the Pipeline tab's own breakdowns, each read correctly
    # by the recorded model output (checked one by one, 2026-09-29).
    52: "pipeline", 55: "pipeline",     # pipeline amount by broker; the largest
    65: "pipeline", 129: "pipeline",    # pipeline amount by product
    131: "pipeline",                    # pipeline case count by product
    66: "pipeline", 130: "pipeline",    # pipeline amount by LTV bucket / band
    # The Pipeline tab's region chart is the reporting region (owner, same day).
    53: "pipeline",                     # pipeline amount by region
    # D13 (§20.3): the pipeline's change between two dated extracts — read
    # correctly as a metric delta on the pipeline amount, October to November.
    83: "pipeline",                     # pipeline growth October to November
}

#: Plans the runtime would answer with a figure the question did not ask for.
MISREAD = {
    76: forecast_rt.PERIOD_NOT_SUPPORTED,   # expected completions BY MONTH
    77: forecast_rt.PERIOD_NOT_SUPPORTED,   # weighted expected amount BY MONTH
    127: forecast_rt.PERIOD_NOT_SUPPORTED,  # projection over twelve months
    94: forecast_rt.AMBIGUOUS_READING,      # the FUNDED share of the forecast
    114: forecast_rt.AMBIGUOUS_READING,     # the extrapolation CURVE
    117: forecast_rt.AMBIGUOUS_READING,     # the BASE scenario
    98: forecast_rt.AMBIGUOUS_READING,      # the forecast's METHOD
    121: forecast_rt.AMBIGUOUS_READING,     # a KFI->completion CONVERSION %
}

#: Correct readings held with the misreads because their plan is the same
#: shape. Legacy answers them correctly meanwhile.
HELD_WITH_THEM = {87, 112, 113}


def _compiled(case):
    return DeterministicCompiler(CompilerContext()).compile(
        parse_candidate_intent(json.loads(json.dumps(case["raw_model_payload"]))))


def _route(plan):
    """`(runtime name, eligible, reason)` for a specialist plan, else None."""
    for name, runtime in _RUNTIMES:
        if runtime.claims(plan):
            ok, why, _ = runtime.check_eligibility(plan)
            if ok:
                ok, why, _ = adapter.check_population_base(
                    plan, runtime.EXECUTION_POPULATION,
                    executable=runtime.EXECUTABLE_POPULATIONS)
            return name, ok, why
    return None


@pytest.fixture(scope="module")
def replay():
    out = {}
    for case in _CASES:
        result = _compiled(case)
        plan = result.plan.to_dict() if result.plan is not None else None
        out[case["n"]] = {"case": case, "result": result, "plan": plan,
                          "route": _route(plan) if plan else None}
    return out


def test_the_fixture_is_the_whole_run():
    assert len(_CASES) == 135
    assert all(case["raw_model_payload"] for case in _CASES)


def test_todays_compiler_reproduces_the_recorded_plans(replay):
    """Only normalisation rule 6 moves a plan, and only its base."""
    moved = {}
    for n, row in replay.items():
        recorded = row["case"]["recorded"]
        assert str(row["result"].outcome) == recorded["compile_outcome"], n
        if row["plan"] and row["plan"]["plan_id"] != recorded["plan_id"]:
            moved[n] = row["plan"]["population"]["base"]
    # 43 ("balance by borrower structure") now binds the governed
    # borrower_type: the legacy concept is folded into it (`superseded_by`).
    # 83 ("pipeline growth October to November"): normalisation keeps the
    # measure's owner, the pipeline, which implements the metric delta (§20.3).
    assert moved == {112: "forecast", 113: "forecast", 125: "forecast",
                     43: "funded", 83: "pipeline"}


def test_exactly_the_pinned_questions_are_newly_admitted(replay):
    admitted = {n: row["route"][0] for n, row in replay.items()
                if row["route"] and row["route"][1]
                and row["case"]["recorded"]["serving_decision"] != "NEW"}
    assert admitted == ADMITTED


def test_every_measured_misread_is_held(replay):
    held = {n: replay[n]["route"][2] for n in MISREAD}
    assert held == MISREAD


def test_correct_readings_of_a_held_shape_are_held_too(replay):
    for n in HELD_WITH_THEM:
        name, ok, why = replay[n]["route"]
        assert (name, ok, why) == ("forecast", False, forecast_rt.AMBIGUOUS_READING)


def test_nothing_served_on_the_governed_path_before_is_now_refused(replay):
    for n, row in replay.items():
        if row["case"]["recorded"]["serving_decision"] == "NEW" and row["route"]:
            assert row["route"][1], (n, row["route"])
