"""What is expected to complete in a month is stated at face value, with the
weighted figure alongside (owner decision D16, 2026-09-30, P0 design §26).

On the 13:49 check "How much pipeline is expected to complete next month?"
answered £4.8m (the weighted figure) and "What's due to complete out of the
pipeline next month?" answered £7.8m (face value): two readings of one
question, both of them figures the Pipeline tab publishes for the same cases.
The owner's rule: face value is the headline, the weighted figure goes beside
it. The model is told so, and every answer carries all three of the tab's
figures for the month, so a reading that names the other measure can no longer
hide the one the owner leads with.
"""
from __future__ import annotations

import pytest

from mi_agent import plan_pipeline_runtime as pipeline_rt
from mi_agent.interpretation_v2.vocabulary import (
    SPECIALIST_DIMENSION_DEFINITIONS, SPECIALIST_MEASURE_DEFINITIONS)
from tests.interpretation_v2.test_pipeline_catalogue_batch1 import (  # noqa: F401
    _TIMING, _plan, _run, overdue_case, semantics)


def test_the_model_is_told_the_month_is_face_value():
    timing = SPECIALIST_DIMENSION_DEFINITIONS["expected_completion_timing"]
    weighted = SPECIALIST_MEASURE_DEFINITIONS["weighted_expected_funded_amount"]
    assert "`pipeline_amount` of those cases, at face value" in timing
    assert "only when the question asks for weighting" in timing
    assert "at face value" in weighted and "D16" in weighted


@pytest.mark.parametrize("measure", ["pipeline_amount",
                                     "weighted_expected_funded_amount",
                                     "pipeline_case_count"])
def test_every_reading_carries_all_three_of_the_tabs_figures(measure, semantics,
                                                             overdue_case):
    out = _run(_plan(measures=[{"concept": measure}],
                     filters=_TIMING("current_month")), semantics)
    summary = overdue_case["expectedCompletionSummary"]
    assert out.receipt["timing_figures"] == pytest.approx({
        "amount": summary["currentMonthExpectedCompletionAmount"],
        "count": summary["currentMonthExpectedCompletionCount"],
        "weighted": summary["currentMonthExpectedCompletionWeightedAmount"]})


def test_the_answer_states_the_other_figures_beside_the_headline():
    from mi_agent.plan_serving_canary import _timing_companions
    figures = {"amount": 7_800_000.0, "count": 23.0, "weighted": 4_800_000.0}
    face = _timing_companions("amount", figures)
    assert face.startswith(" Weighted by each case's chance of completing £4.8m")
    assert "23 cases" in face and "£7.8m" not in face
    weighted = _timing_companions("weighted", figures)
    assert weighted.startswith(" At face value £7.8m") and "23 cases" in weighted
