"""P0 — the temporal honouring property, proved from the artifact.

Every test here is written against ROWS. None reads `executionSummary`,
`dimensionsApplied` or a guard verdict, because the case that made this rule
necessary had a TRUTHFUL receipt: an 88-group heatmap whose `dimensionsApplied`
correctly read ['Region','LTV Bucket'] and was simply silent about time.
"""
from __future__ import annotations

import pytest

from mi_agent import execution_receipt as R


def _art(rows, kind="table"):
    return [{"type": kind, "rows": rows}]


SERIES = _art([{"period": "2026-04", "value": 1},
               {"period": "2026-05", "value": 2},
               {"period": "2026-06", "value": 3}], "chart")
#: The T5 shape: a real breakdown, correctly labelled, over ONE period.
HEATMAP = _art([{"collateral_geography": "London", "ltv_bucket": "50-60%",
                 "current_outstanding_balance_sum": 10},
                {"collateral_geography": "South East", "ltv_bucket": "60-70%",
                 "current_outstanding_balance_sum": 20}], "chart")
#: The movement shape: two points in time, one per COLUMN.
MOVEMENT = _art([{"measure": "Balance", "population": "Direct",
                  "period": "2026-04-30 → 2026-06-30",
                  "prior": "£1.36bn", "current": "£1.39bn", "change": "+£21.5m"},
                 {"measure": "Balance", "population": "Acquired",
                  "period": "2026-04-30 → 2026-06-30",
                  "prior": "£568.3m", "current": "£579.4m", "change": "+£11.1m"}])


# --------------------------------------------------------------------------- #
# The artifact reader
# --------------------------------------------------------------------------- #
def test_a_series_proves_the_axis_from_its_rows():
    assert "3 distinct values in period" == R.artifact_time_axis(SERIES)


def test_a_movement_table_proves_the_axis_from_its_columns():
    """Form B. A movement expresses its two points as a COLUMN PAIR.

    Rejecting this form would refuse `period_change_analysis` and
    `analytical_composition` answers that are correct today — three of them on
    the time-series surface alone.
    """
    assert R.artifact_time_axis(MOVEMENT) == "columns naming prior and current"


def test_a_time_column_with_one_value_proves_nothing():
    """`period` present and constant is a stamp, not an axis.

    MOVEMENT's own `period` column holds one value on both rows; it passes on
    prior/current, not on that column.
    """
    rows = [{"period": "30 June 2026", "region": "London", "value": 1},
            {"period": "30 June 2026", "region": "South East", "value": 2}]
    assert R.artifact_time_axis(_art(rows)) is None


def test_a_correct_breakdown_over_one_period_proves_nothing():
    assert R.artifact_time_axis(HEATMAP) is None


def test_an_end_name_must_be_a_whole_word_prefix_or_suffix():
    """`current_outstanding_balance` is an end name; `recurrent_fees` is not."""
    assert R._names_end("current_outstanding_balance", "current")
    assert R._names_end("balance_current", "current")
    assert R._names_end("current", "current")
    assert not R._names_end("recurrent_fees", "current")


@pytest.mark.parametrize("first,second", R._ARTIFACT_END_PAIRS)
def test_every_end_pair_is_reachable(first, second):
    """No dead vocabulary: each declared pair must be able to prove an axis."""
    rows = [{first + "_value": 1, second + "_value": 2, "category": "London"}]
    assert R.artifact_time_axis(_art(rows)) == (
        "columns naming %s and %s" % (first, second))


def test_one_end_alone_proves_nothing():
    rows = [{"current_balance": 1, "category": "London"}]
    assert R.artifact_time_axis(_art(rows)) is None


def test_it_looks_at_every_artifact_the_answer_shipped():
    """A route may ship a chart AND a table; either may carry the proof."""
    both = HEATMAP + SERIES
    assert R.artifact_time_axis(both)


# --------------------------------------------------------------------------- #
# The facet raiser
# --------------------------------------------------------------------------- #
def test_a_time_axis_asked_for_and_absent_is_raised():
    facets = R.temporal_honouring_facets(
        "Show me balance by month by region and LTV band", HEATMAP)
    assert [f.kind for f in facets] == [R.KIND_SERIES_AXIS]
    assert facets[0].label == "a series by month"
    assert facets[0].status == R.LOST


def test_a_time_axis_asked_for_and_present_raises_nothing():
    assert R.temporal_honouring_facets("Show me balance by month", SERIES) == []


def test_a_question_that_asked_for_no_axis_raises_nothing():
    assert R.temporal_honouring_facets("balance by region", HEATMAP) == []


def test_an_answer_that_ships_no_object_is_out_of_scope():
    """"completions by month" replies that no extracts exist, with no artifact
    and no figure. Refusing that would replace an honest statement of
    incapacity with a refusal that says less."""
    assert R.temporal_honouring_facets("completions by month", []) == []
    assert R.temporal_honouring_facets("completions by month", None) == []


def test_the_kind_blocks_rather_than_discloses():
    """Honour-or-clarify: a single position presented for a question about
    movement is a substitution, not a partial answer."""
    assert R.KIND_SERIES_AXIS in R.NUMBER_OR_SUBJECT_FACETS
    assert R.KIND_SERIES_AXIS not in R.SHAPE_FACETS


def test_the_refusal_sentence_comes_from_assess():
    """NOT re-authored here. The wording must be the one the eighteen existing
    refusals on this surface already use."""
    facets = R.temporal_honouring_facets("balance over time", HEATMAP)
    verdict, message = R.assess(R.ExecutionReceipt(facets=facets))
    assert verdict == R.VERDICT_REFUSE
    assert message.startswith("I understood that you asked for a series over time,"
                              " but that could not be applied to the calculation")
    assert message.endswith("I have not substituted a broader figure.")
    assert message.count("—") == 1


# --------------------------------------------------------------------------- #
# THE CAN-FAIL PROOF
# --------------------------------------------------------------------------- #
def test_the_reader_would_notice_a_receipt_being_trusted(monkeypatch):
    """The whole point, stated as a test that fails if the point is lost.

    T5's receipt was TRUE. Give the heatmap the receipt it really carried and
    the reader must still refuse it — a reader consulting `dimensionsApplied`
    would pass it.
    """
    withreceipt = [{"type": "chart", "rows": HEATMAP[0]["rows"],
                    "dimensionsApplied": ["Region", "LTV Bucket"],
                    "executionSummary": {"dimensionsApplied": ["Region",
                                                               "LTV Bucket"]}}]
    assert R.artifact_time_axis(withreceipt) is None
    assert R.temporal_honouring_facets("balance by month", withreceipt)


def test_the_suite_would_notice_a_reader_that_always_proves(monkeypatch):
    monkeypatch.setattr(R, "artifact_time_axis", lambda _a: "anything at all")
    assert R.temporal_honouring_facets("balance by month", HEATMAP) == []


def test_the_suite_would_notice_a_reader_that_never_proves(monkeypatch):
    monkeypatch.setattr(R, "artifact_time_axis", lambda _a: None)
    assert R.temporal_honouring_facets("Show me balance by month", SERIES)


# --------------------------------------------------------------------------- #
# The call sites, enumerated
# --------------------------------------------------------------------------- #
def test_both_paths_reach_the_temporal_guard():
    """Two sites, and exactly two: the routed envelope and the point-in-time
    result. Both must be inside `_run_analysis`, after the artifacts exist.

    Enumerated as a test because the recurring failure in this programme is a
    decision that arrives at one site and not another — six of the seven
    consolidations closed exactly that.
    """
    import inspect

    from mi_agent_api import mi_service

    source = inspect.getsource(mi_service._run_analysis)
    assert source.count("_guard_temporal_honouring(") == 2
    assert "_guard_temporal_honouring(routed" in source
    assert "_guard_temporal_honouring(result" in source
    whole = inspect.getsource(mi_service)
    assert whole.count("_guard_temporal_honouring(") == 3  # two calls + the def
