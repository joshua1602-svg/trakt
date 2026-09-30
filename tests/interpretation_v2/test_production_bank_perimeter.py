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
    # THE RUN-RATE HOLD RELEASED (P0 design §31) on the 19:59 full bank's
    # readings (2026-09-30, vocabulary 2.18.0): every question in the shape
    # now reaches the forecast runtime. 112 and 113 are this run's correct
    # run-rate readings; 98 and 121 are this run's MISREADS in the same shape
    # (the method, a KFI conversion), released because today's model no
    # longer makes them — 98 asks back and 121 reads the stage completion rate
    # (`test_the_run_rate_hold_is_released_on_the_1959_readings`).
    98: "forecast", 112: "forecast", 113: "forecast", 121: "forecast",
    # D20 (§32.1): "over time" with no span is every reporting date held — the
    # pipeline by stage across every extract, amount and cases.
    74: "pipeline", 75: "pipeline",
    # D22 (§32.3): the run-rate over a named window of weeks, read as a range
    # of 8 weekly periods, is the history owner's calendar 8-week run-rate.
    125: "forecast",
}

#: Plans the runtime would answer with a figure the question did not ask for.
MISREAD = {
    76: forecast_rt.PERIOD_NOT_SUPPORTED,   # expected completions BY MONTH
    77: forecast_rt.PERIOD_NOT_SUPPORTED,   # weighted expected amount BY MONTH
    127: forecast_rt.PERIOD_NOT_SUPPORTED,  # projection over twelve months
    94: forecast_rt.AMBIGUOUS_READING,      # the FUNDED share of the forecast
    114: forecast_rt.AMBIGUOUS_READING,     # the extrapolation CURVE
    117: forecast_rt.AMBIGUOUS_READING,     # the BASE scenario
}

#: Misreads of this run whose shape's hold was RELEASED on later evidence
#: (§31): the run-rate shape. Today's model reads both questions otherwise.
MISREAD_RELEASED = {98: "the forecast's METHOD",
                    121: "a KFI->completion CONVERSION %"}

#: Correct readings held with the misreads because their plan is the same
#: shape. Legacy answers them correctly meanwhile.
HELD_WITH_THEM = {87}


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


#: RECORDED OUTCOMES NORMALISATION RULE 7 MOVES (normal form 1.1, P0 design
#: §24.1): a single figure whose every output groups IS the breakdown. Four
#: readings of this run labelled a grouping `point_in_time`:
#:   100, 101  "which stages use historical / config fallback rates?" — a
#:             stage grouping with NO measure and a blocking ambiguity: no
#:             longer refused for the label, so the ambiguity asks back
#:             (fine to decline / ask back).
#:   105, 106  "forecast balance by stage (and broker)" — now a plan, and
#:             refused by the forecast runtime: its owner publishes no split
#:             by pipeline stage (`test_rule_7_admits_nothing_an_owner_does_
#:             not_publish`). The refusal moved from a label to the owner.
RULE_7_MOVED = {100: ("REFUSE", "CLARIFY"), 101: ("REFUSE", "CLARIFY"),
                105: ("REFUSE", "PLAN"), 106: ("REFUSE", "PLAN")}

#: D20 (owner decision 2026-09-30, §32.1): a series stating no span is every
#: reporting date the owner holds, not a request to ask back. 79 (broker over
#: time) and 99 (a conversion basis over time) become plans their runtimes
#: still refuse — neither is published over time.
D20_MOVED = {74: ("CLARIFY", "PLAN"), 75: ("CLARIFY", "PLAN"),
             79: ("CLARIFY", "PLAN"), 99: ("CLARIFY", "PLAN")}


def test_todays_compiler_reproduces_the_recorded_plans(replay):
    """Only normalisation rule 6 moves a plan, and only its base; rule 7 moves
    exactly the outcomes it names."""
    moved = {}
    outcome_moved = {}
    for n, row in replay.items():
        recorded = row["case"]["recorded"]
        now = str(row["result"].outcome)
        if now != recorded["compile_outcome"]:
            outcome_moved[n] = (recorded["compile_outcome"], now)
            continue
        if row["plan"] and row["plan"]["plan_id"] != recorded["plan_id"]:
            moved[n] = row["plan"]["population"]["base"]
    # 43 ("balance by borrower structure") now binds the governed
    # borrower_type: the legacy concept is folded into it (`superseded_by`).
    # 83 ("pipeline growth October to November"): normalisation keeps the
    # measure's owner, the pipeline, which implements the metric delta (§20.3).
    # 16 ("what is the smallest loan?"): the minimum balance counts only
    # balances above zero, a predicate on the plan (owner decision D14, §21.3).
    assert moved == {112: "forecast", 113: "forecast", 125: "forecast",
                     43: "funded", 83: "pipeline", 16: "funded"}
    assert outcome_moved == {**RULE_7_MOVED, **D20_MOVED}
    for n in D20_MOVED:
        assert replay[n]["plan"]["period"]["default_method"] == \
            "every_reporting_date", n
    for n in RULE_7_MOVED:
        assert any("grouped_figure" in note
                   for note in replay[n]["result"].plan.provenance.notes) \
            if replay[n]["result"].plan else replay[n]["case"]["raw_model_payload"][
                "operation"] == "point_in_time"


def test_rule_7_admits_nothing_an_owner_does_not_publish(replay):
    for n in (105, 106):
        plan = replay[n]["plan"]
        assert plan["operation"] == "breakdown"
        ok, why, _ = forecast_rt.check_eligibility(plan)
        assert (ok, why) == (False, "DIMENSION_NOT_SUPPORTED")


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


_LATEST = json.loads((_FIXTURE.parent / "qb_recorded_intents_20260930.json")
                     .read_text())["cases"]


def _in_the_run_rate_shape(case):
    payload = case["raw_model_payload"] or {}
    return (payload.get("operation") == "point_in_time"
            and [m.get("concept") for m in payload.get("measures") or ()]
            == ["forecast_completion_rate"])


def test_why_the_run_rate_hold_stayed_on_the_0710_readings():
    """The 2026-09-30 07:10 full bank moved every reading the hold was built
    for to its own concept — the annualised run-rate [113], the KFI-to-
    completion rate [121], the Offer pull-through [134] — and the forecast's
    method [98] asks back. But the 8- and 12-week run-rates [125, 126] arrived
    in the shape with their window dropped: served, they would be answered
    with the forecast's own window. The hold stayed until they carry their
    window (vocabulary 2.13.0) and a live run shows it (P0 design §22) — which
    the 19:59 run did (below)."""
    latest = {case["n"]: case for case in _LATEST}
    assert len(latest) == 135
    shape = sorted(n for n, case in latest.items() if _in_the_run_rate_shape(case))
    assert shape == [98, 112, 125, 126]
    assert latest[98]["recorded"]["compile_outcome"] == "CLARIFY"
    for n, concept in ((113, "annualised_completion_run_rate"),
                       (121, "stage_completion_rate"),
                       (134, "stage_pull_through")):
        payload = latest[n]["raw_model_payload"]
        assert [m["concept"] for m in payload["measures"]] == [concept], n


_RELEASE = json.loads((_FIXTURE.parent / "qb_recorded_intents_20260930_1959.json")
                      .read_text())["cases"]


def test_the_run_rate_hold_is_released_on_the_1959_readings():
    """The hold's own condition, met by a live run (19:59, vocabulary 2.18.0;
    P0 design §31): in the run-rate shape, the method [98] asks back, the
    8- and 12-week run-rates [125, 126] carry their window — which the
    measure, stated only for its current window, refuses — and what remains
    is the current run-rate [112], which it answers. The KFI-to-completion
    rate [121] and the annualised run-rate [113] read as their own concepts.
    [134] met the provider's credit limit and is unmeasured, not misread."""
    latest = {case["n"]: case for case in _RELEASE}
    assert len(latest) == 135
    shape = sorted(n for n, case in latest.items() if _in_the_run_rate_shape(case))
    assert shape == [98, 112, 125, 126]
    assert latest[98]["recorded"]["compile_outcome"] == "CLARIFY"
    assert latest[112]["raw_model_payload"]["time"] == {"form": "current"}
    for n, window in ((125, "8-week"), (126, "12-week")):
        assert latest[n]["raw_model_payload"]["time"]["labels"] == [window]
    for n, concept in ((113, "annualised_completion_run_rate"),
                       (121, "stage_completion_rate")):
        payload = latest[n]["raw_model_payload"]
        assert [m["concept"] for m in payload["measures"]] == [concept], n
    assert latest[134]["recorded"]["reason_codes"] == ["MODEL_UNAVAILABLE"]
    assert ("point_in_time", "forecast_completion_rate") \
        not in forecast_rt.HELD_READINGS
    routes = {}
    for n in (112, 125, 126):
        plan = _compiled(latest[n]).plan.to_dict()
        routes[n] = forecast_rt.check_eligibility(plan)[:2]
    assert routes[112] == (True, "")
    assert routes[125] == (False, forecast_rt.PERIOD_NOT_SUPPORTED)
    assert routes[126][0] is False


def test_a_run_rate_that_keeps_its_window_is_declined_not_answered():
    """What 2.13.0 asks the model to do with "the 8-week run-rate": keep the
    window. The runtime owns one window, so a stated other one is refused."""
    result = DeterministicCompiler(CompilerContext()).compile(parse_candidate_intent({
        "schema_version": "candidate_intent/1.0", "capability": "forecast",
        "operation": "point_in_time",
        "measures": [{"concept": "forecast_completion_rate"}],
        "time": {"form": "range", "labels": ["the last 8 weeks"]}}))
    if result.plan is None:          # refused at compile: declined either way
        return
    plan = result.plan.to_dict()
    assert not forecast_rt.check_eligibility(plan)[0]
    # And with the hold off — the guard the release will rely on.
    from mi_agent import semantic_engine as engine
    ok, why, _ = engine.check(forecast_rt.MODEL, plan, "forecast_completion_rate",
                              "point_in_time", held=None)
    assert not ok and why == forecast_rt.PERIOD_NOT_SUPPORTED, why


def test_nothing_served_on_the_governed_path_before_is_now_refused(replay):
    for n, row in replay.items():
        if row["case"]["recorded"]["serving_decision"] == "NEW" and row["route"]:
            assert row["route"][1], (n, row["route"])
