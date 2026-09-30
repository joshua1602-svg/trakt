"""When the live pipeline is expected to complete — a date, from the book's own
history (owner decision 2026-09-30, P0 design §23).

"When are pipeline cases expected to complete?" is must-answer [135]. On the
2026-09-30 full bank it was read as the amount and the case count by the month
each case's own record gives, and neither path answered it. The owner's
answer: "a date answer based on historical time to complete using the
client's time series".

The history owner (`pipeline_history.build_historical_completion_model`)
already measures, per stage, the median days this book's cases took from being
first seen at the stage to completing. It now applies that to each LIVE case —
first seen at its current stage, plus the stage's median — and publishes the
median date per stage and over the whole live pipeline. The stage-movement
semantic model declares the figure; the one engine reads it. Every date here
is compared with the owner's own output.
"""
from __future__ import annotations

import datetime as dt

from mi_agent import semantic_model
from tests.interpretation_v2.test_stage_conversion import (  # noqa: F401
    _plan, _run, _served, _stage, history)

_MODEL = semantic_model.load("pipeline_stage_movement")


def _iso(value):
    return dt.date.fromisoformat(value)


def test_each_stage_date_rests_on_the_stages_own_timing(history):
    """The median days behind each date are the timing the owner already
    publishes, and the date is no earlier than first-seen plus those days
    could make it for a case seen at the latest extract."""
    timing = history["historicalCompletionTimingByStage"]
    latest = _iso(history["observationWindowEnd"])
    for stage, row in history["expectedCompletionByStage"].items():
        assert row["liveCases"] >= 1
        assert row["liveCases"] == row["lapsedCases"] + row["datedCases"] or \
            "medianDate" not in row
        if stage not in timing:
            assert "medianDate" not in row    # no completion from it, no date
            continue
        assert row["medianDays"] == timing[stage]["medianDays"]
        assert row["completionsObserved"] == timing[stage]["observed"]
        if "medianDate" not in row:
            continue                          # every live case lapsed
        assert _iso(row["medianDate"]) <= latest + dt.timedelta(
            days=row["medianDays"])
        assert 0 <= row["pastTypical"] <= row["datedCases"]


def test_the_date_by_stage_is_the_owners(history):
    out = _run(_plan("expected_completion_date", operation="breakdown",
                     dimensions=["origin_stage"]), history)
    owner = history["expectedCompletionByStage"]
    assert {c["origin_stage"]: c["value"] for c in out.cells} == {
        s: row.get("medianDate") for s, row in owner.items()}
    assert out.receipt["provisional_members"] == [
        s for s, row in owner.items() if not row["sufficient"]]


def test_one_stage_is_that_stages_date_with_its_evidence(history):
    owner = history["expectedCompletionByStage"]["OFFER"]
    out = _run(_plan("expected_completion_date",
                     filters=_stage("origin_stage", "OFFER")), history)
    assert out.value == owner["medianDate"]
    assert out.receipt["member_evidence"] == {
        k: owner.get(k) for k in ("liveCases", "lapsedCases", "datedCases",
                                  "windowDays", "medianDays",
                                  "completionsObserved", "pastTypical")}


def test_the_whole_live_pipeline_is_the_owners_median(history):
    out = _run(_plan("expected_completion_date"), history)
    assert out.value == history["expectedCompletion"]["medianDate"]
    assert out.receipt["inputs"]["case_history"]["as_of"] == \
        history["observationWindowEnd"]


def test_the_answer_is_a_date_that_says_it_is_conditional(monkeypatch, history):
    payload = _served(_intent_for_the_whole_pipeline(), monkeypatch, history)
    answer = payload["answer"]
    assert answer.startswith("Expected completion date: "
                             + history["expectedCompletion"]["medianDate"])
    assert "that complete" in answer and "not a promise" in answer
    assert f"{history['expectedCompletion']['datedCases']:,} live cases" in answer
    assert f"{history['expectedCompletion']['lapsedCases']:,} more have sat" in answer


def _intent_for_the_whole_pipeline():
    return {"schema_version": "candidate_intent/1.0",
            "capability": "pipeline_stage_movement",
            "operation": "point_in_time", "population": {"base": "pipeline"},
            "measures": [{"concept": "expected_completion_date"}],
            "time": {"form": "current"}}


def test_the_model_is_told_it_is_a_date_from_history_and_not_the_records():
    from mi_agent.interpretation_v2.vocabulary import (
        SPECIALIST_DIMENSION_DEFINITIONS, load_governed_vocabulary)
    concept = load_governed_vocabulary().concepts["expected_completion_date"]
    assert concept.owning_capability == "pipeline_stage_movement"
    assert "WHEN live cases will complete" in concept.description
    assert "NOT the completion date each case's own" in concept.description
    month = SPECIALIST_DIMENSION_DEFINITIONS["expected_completion_month"]
    assert "when are cases expected to complete" not in month
    assert "`expected_completion_date`" in month


# --------------------------------------------------------------------------- #
# the 10:26 check's reading: a single figure, grouped (normalisation rule 7)
# --------------------------------------------------------------------------- #

def _read_at_1026():
    """The model's own reading of [135] on the 2026-09-30 10:26 check, as the
    readback printed it: the right measure and axis, labelled a single figure."""
    return {"schema_version": "candidate_intent/1.0",
            "capability": "pipeline_stage_movement",
            "operation": "point_in_time", "population": {"base": "pipeline"},
            "measures": [{"concept": "expected_completion_date"}],
            "dimensions": ["origin_stage"], "time": {"form": "current"}}


def test_the_1026_reading_is_the_breakdown_by_stage(history):
    from mi_agent.interpretation_v2.compiler import (CompilerContext,
                                                     DeterministicCompiler)
    from mi_agent.interpretation_v2.intent import parse_candidate_intent
    result = DeterministicCompiler(CompilerContext()).compile(
        parse_candidate_intent(_read_at_1026()))
    assert result.plan is not None, [(r.code, r.detail) for r in result.reasons]
    plan = result.plan.to_dict()
    assert plan["operation"] == "breakdown"
    assert any("grouped_figure" in note for note in plan["provenance"]["notes"])
    # The model's own label still travels.
    assert result.intent.operation == "point_in_time"
    out = _run(plan, history)
    assert {c["origin_stage"]: c["value"] for c in out.cells} == {
        s: row.get("medianDate")
        for s, row in history["expectedCompletionByStage"].items()}


def test_the_1026_reading_is_answered_with_a_date_per_stage(monkeypatch,
                                                            history):
    payload = _served(_read_at_1026(), monkeypatch, history)
    assert payload is not None
    answer = payload["answer"]
    # The stage a live case is at now — not the "from stage" of a rate.
    assert answer.startswith("Expected completion date by current stage: ")
    for stage, row in history["expectedCompletionByStage"].items():
        if row.get("medianDate"):
            assert row["medianDate"] in answer, (stage, answer)
    # The caveat goes with every shape, not only the single figure.
    assert "not a promise that they will" in answer


def test_one_stage_says_it_is_the_current_stage_and_carries_the_caveat(
        monkeypatch, history):
    intent = dict(_intent_for_the_whole_pipeline(),
                  filters=[{"concept": "origin_stage", "comparator": "eq",
                            "value": "OFFER"}])
    answer = _served(intent, monkeypatch, history)["answer"]
    assert answer.startswith("Expected completion date (current stage: Offer): ")
    assert "not a promise that they will" in answer



# --------------------------------------------------------------------------- #
# D17: lapsed cases are not dated (owner decision 2026-09-30)
# --------------------------------------------------------------------------- #

def test_a_lapsed_case_is_counted_and_left_out_of_the_date():
    """The 13:49 check answered 2026-04-01 for a pipeline as at 2026-09-24:
    most live cases were KFIs sat far past the stage's validity window. The
    forecast gives such a case no weight; the date now leaves it out too."""
    from mi_agent_api.pipeline_history import _expected_completion
    latest = "2026-09-24"
    timelines = {
        # Entered KFI in January: past a 14-day window — lapsed.
        "stale": {"last_seen": latest, "final_stage": "KFI",
                  "stages": {"KFI": "2026-01-05"}, "kfi_date": "2026-01-05"},
        # Entered KFI last week: inside the window — dated.
        "fresh": {"last_seen": latest, "final_stage": "KFI",
                  "stages": {"KFI": "2026-09-17"}, "kfi_date": "2026-09-17"},
        # No entry date: the forecast cannot measure its time in stage, so it
        # does not lapse it — nor does this.
        "undated": {"last_seen": latest, "final_stage": "KFI",
                    "stages": {"KFI": "2026-09-10"}},
    }
    timing = {"KFI": {"medianDays": 60, "observed": 40}}
    by_stage, overall = _expected_completion(
        timelines, timing, latest, 20, windows={"KFI": 14},
        window_basis={"KFI": "measured"})
    row = by_stage["KFI"]
    assert (row["liveCases"], row["lapsedCases"], row["datedCases"]) == (3, 1, 2)
    assert row["windowDays"] == 14 and row["windowBasis"] == "measured"
    assert row["medianDate"] == "2026-11-09"          # 2026-09-10 + 60 days
    assert overall["medianDate"] >= latest
    assert (overall["liveCases"], overall["lapsedCases"],
            overall["datedCases"]) == (3, 1, 2)


def test_the_window_is_the_forecasts_own():
    """One definition of lapsed: the forecast's tier-4 rule and the date read
    the same function over the same run-off model."""
    import inspect
    from mi_agent_api import pipeline_history, pipeline_prep
    assert "stage_validity_windows(runoff)" in inspect.getsource(
        pipeline_history.build_historical_completion_model)
    assert "windows = stage_validity_windows(runoff)" in inspect.getsource(
        pipeline_prep._derive_probabilities_and_amounts)


def test_the_history_publishes_the_window_it_applied(history):
    windows = pipeline_prep_windows(history)
    for stage, row in history["expectedCompletionByStage"].items():
        assert row["windowDays"] == windows.get(stage)


def pipeline_prep_windows(history):
    from mi_agent_api.pipeline_prep import stage_validity_windows
    return stage_validity_windows(history["runoff"])
