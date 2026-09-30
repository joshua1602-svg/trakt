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
        if stage not in timing:
            assert "medianDate" not in row    # no completion from it, no date
            continue
        assert row["medianDays"] == timing[stage]["medianDays"]
        assert row["completionsObserved"] == timing[stage]["observed"]
        assert _iso(row["medianDate"]) <= latest + dt.timedelta(
            days=row["medianDays"])
        assert 0 <= row["pastTypical"] <= row["liveCases"]


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
        k: owner[k] for k in ("liveCases", "medianDays", "completionsObserved",
                              "pastTypical")}


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
    assert f"{history['expectedCompletion']['liveCases']:,} live cases" in answer


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
    assert "when are pipeline cases expected to complete" in concept.description
    assert "NOT the completion date each case's own" in concept.description
    month = SPECIALIST_DIMENSION_DEFINITIONS["expected_completion_month"]
    assert "when are cases expected to complete" not in month
    assert "`expected_completion_date`" in month
