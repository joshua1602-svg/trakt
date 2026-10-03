"""D21 (owner decision 2026-09-30): measured, or declined.

"Measured or decline if not enough history" — for each stage's chance of
completing and, by the owner's follow-up, for each stage's validity window.
Nothing configured stands in: a weighted figure (and a forecast built on one)
that depends on a stage the client's history cannot yet measure is not stated,
and the agent declines it with the reason — never a zero, never a partial sum.

The fixture book is far below the production thresholds, so read as production
reads it, it measures nothing; read at test-book scale
(`tests.measured_history`), the same book measures every stage.
"""
from __future__ import annotations

import pytest

from mi_agent import plan_decline
from mi_agent import plan_pipeline_runtime as pipeline_rt
from mi_agent import semantic_engine
from mi_agent_api import pipeline_contract as pc
from tests.interpretation_v2.test_pipeline_catalogue_batch1 import (  # noqa: F401
    _HISTORY, _TIMING, _WEIGHTED, _plan, _source, semantics)


def _execute(plan, semantics, history=None):
    ok, why, detail = pipeline_rt.check_eligibility(plan)
    assert ok, (why, detail)
    return pipeline_rt.execute_current(plan, source=_source(), semantics=semantics,
                                       history_model=history)


def test_the_weighted_pipeline_is_declined_without_measured_rates(semantics):
    out = _execute(_plan(measures=_WEIGHTED), semantics)
    assert (out.ok, out.reason) == (False, pipeline_rt.RATE_NOT_MEASURED)
    assert "no configured value is used" in out.detail
    assert "D21" in out.detail


def test_with_measured_rates_it_is_the_tabs(semantics):
    out = _execute(_plan(measures=_WEIGHTED), semantics, _HISTORY)
    assert out.ok, out.detail
    frame, report = pc.load_prepared_pipeline(_source(), historical_model=_HISTORY)
    assert out.value == pytest.approx(pc.open_totals(frame)["weighted"])


def test_the_face_value_is_still_answered_where_only_rates_are_unmeasured(semantics):
    """The pipeline amount needs no rate: it is answered whatever the history."""
    out = _execute(_plan(measures=[{"concept": "pipeline_amount"}]), semantics)
    assert out.ok, out.detail


def test_a_weighted_breakdown_declines_when_no_group_can_be_stated(semantics):
    out = _execute(_plan(operation="breakdown", measures=_WEIGHTED,
                         dimensions=["expected_completion_month"]), semantics)
    assert (out.ok, out.reason) == (False, pipeline_rt.RATE_NOT_MEASURED)


def test_the_decline_says_why_in_plain_words():
    """Not "the data could not be read; try again" — the history is short."""
    assert plan_decline.kind("INELIGIBLE:RATE_NOT_MEASURED") != plan_decline.UNAVAILABLE
    for code in ("RATE_NOT_MEASURED", "FIGURE_WITHHELD"):
        reason = plan_decline.plain_reason(code)
        assert "history is not yet enough" in reason
        assert "no configured rate" in reason


def test_the_engine_declines_a_figure_its_owner_withholds():
    from mi_agent import semantic_model
    m = semantic_model.load("forecast").measures["forecast_funded_balance"]
    payload = {"forecastBridge": {"forecastFundedBalance": None,
                                  "forecastWithheldReason": "the history is short"}}
    with pytest.raises(semantic_engine.Refusal) as refused:
        semantic_engine.serve_figure(m, payload, axis=None, member=None,
                                     binding={}, period={})
    assert refused.value.reason == semantic_engine.FIGURE_WITHHELD
    assert refused.value.detail == "the history is short"


def test_no_configured_rate_or_window_reaches_a_case():
    frame, report = pc.load_prepared_pipeline(_source())
    sources = set(frame["completion_probability_source"].astype(str))
    assert "configured_stage_rate" not in sources
    assert "stage_conversion_probability" not in frame.columns
    assert report["completion_probability_basis"] == "insufficient_history"
    assert report["weighting_complete"] is False
