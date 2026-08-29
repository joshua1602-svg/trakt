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


# --------------------------------------------------------------------------- #
# Limb 2 — the segments the sentence named must survive
# --------------------------------------------------------------------------- #
#: A whole-book series: one row per point in time, nothing else.
WHOLE_BOOK = _art([{"period": "2026-04", "funded_balance": 1},
                   {"period": "2026-05", "funded_balance": 2},
                   {"period": "2026-06", "funded_balance": 3}], "chart")
#: The table `cohort_progression` ships BESIDE that series. Its `wa_ltv` and
#: `wa_interest_rate` vary across the periods, which is what defeated the first
#: draft's measure-word blacklist.
WHOLE_BOOK_WITH_MEASURES = WHOLE_BOOK + _art(
    [{"period": "2026-04", "loan_count": 11035, "funded_balance": 1,
      "wa_ltv": 43.1, "wa_interest_rate": 6.4, "nneg_headroom_pct": 11.0},
     {"period": "2026-05", "loan_count": 11040, "funded_balance": 2,
      "wa_ltv": 43.5, "wa_interest_rate": 6.5, "nneg_headroom_pct": 11.2},
     {"period": "2026-06", "loan_count": 11035, "funded_balance": 3,
      "wa_ltv": 45.4, "wa_interest_rate": 6.6, "nneg_headroom_pct": 11.4}])
#: A segmented series: one row per (segment, point).
SEGMENTED = _art([{"period": p, "population": g, "value": 1}
                  for p in ("2026-04", "2026-05", "2026-06")
                  for g in ("Direct", "Acquired")], "chart")

VALUES = {"direct": "origination_channel", "acquired": "source_portfolio_type",
          "front book": "seasoning_segment", "back book": "seasoning_segment",
          "london": "collateral_geography", "south east": "collateral_geography",
          "east": "collateral_geography"}
ASKS_FOR_BOTH = "How have direct and acquired balances moved over the periods?"


def test_a_whole_book_series_proves_no_cut():
    assert R.artifact_segment_cut(WHOLE_BOOK) is None


def test_a_measure_that_varies_is_not_a_cut():
    """The defect that killed the first draft, pinned as a test.

    A blacklist of measure words can never be complete. The shape of the object
    does not need one: three rows over three points is uncut whatever the
    columns are called.
    """
    assert R.artifact_segment_cut(WHOLE_BOOK_WITH_MEASURES) is None


def test_a_segmented_series_proves_a_cut():
    assert R.artifact_segment_cut(SEGMENTED) == (
        "6 rows where an uncut answer of this shape carries 3")


def test_a_movement_table_is_uncut_at_one_row_and_cut_above_it():
    """Form B counts differently, and the difference is load-bearing.

    A movement table holds both points in a column pair, so ONE row is the
    uncut shape. Counting its rows against its points instead refused the two
    `analytical_composition` answers that correctly track front book against
    back book.
    """
    one = _art([{"measure": "Balance", "population": "Total",
                 "prior": 1, "current": 2, "change": 1}])
    assert R.artifact_segment_cut(one) is None
    assert R.artifact_segment_cut(MOVEMENT) == (
        "2 rows where an uncut answer of this shape carries 1")


def test_the_segments_are_read_from_the_book_not_from_a_word_list():
    assert R.segments_named_in(ASKS_FOR_BOTH, VALUES) == ["direct", "acquired"]
    assert R.segments_named_in(ASKS_FOR_BOTH, None) == []
    assert R.segments_named_in("What is the total balance?", VALUES) == []


def test_a_value_inside_a_longer_value_is_not_a_second_segment():
    """"south east" contains "east"; naming one region is not a comparison."""
    assert R.segments_named_in("balance over time for the South East",
                               VALUES) == ["south east"]


def test_the_segments_are_returned_in_the_order_the_sentence_says_them():
    assert R.segments_named_in("acquired and direct over time", VALUES) == [
        "acquired", "direct"]


def test_one_segment_named_is_a_scope_and_not_a_comparison():
    """"over time for the front book" is a POPULATION, already owned elsewhere.
    It must not be read as a request to track two things."""
    assert R.temporal_honouring_facets(
        "How has balance moved over time for the front book?",
        WHOLE_BOOK, VALUES) == []


def test_two_segments_named_and_a_whole_book_series_refuses():
    facets = R.temporal_honouring_facets(ASKS_FOR_BOTH, WHOLE_BOOK_WITH_MEASURES,
                                         VALUES)
    assert [f.kind for f in facets] == [R.KIND_SERIES_AXIS]
    assert facets[0].label == "Direct and Acquired tracked separately"


def test_two_segments_named_and_a_segmented_series_stands():
    assert R.temporal_honouring_facets(ASKS_FOR_BOTH, SEGMENTED, VALUES) == []


def test_two_segments_named_and_a_two_row_movement_stands():
    assert R.temporal_honouring_facets(
        "Compare balance over time for direct and acquired", MOVEMENT,
        VALUES) == []


def test_limb_two_never_runs_before_limb_one():
    """A missing time axis is the FIRST loss, and the reader is owed that one.

    Told "your segments were dropped" about an answer that has no time axis at
    all, a reader would fix the wrong half of the question.
    """
    facets = R.temporal_honouring_facets(
        "balance by month for direct and acquired", HEATMAP, VALUES)
    assert len(facets) == 1
    assert facets[0].label == "a series by month"


def test_limb_two_would_notice_a_cut_reader_that_always_proves(monkeypatch):
    monkeypatch.setattr(R, "artifact_segment_cut", lambda _a: "anything at all")
    assert R.temporal_honouring_facets(ASKS_FOR_BOTH, WHOLE_BOOK, VALUES) == []


def test_limb_two_would_notice_a_cut_reader_that_never_proves(monkeypatch):
    monkeypatch.setattr(R, "artifact_segment_cut", lambda _a: None)
    assert R.temporal_honouring_facets(
        "Compare balance over time for direct and acquired", MOVEMENT, VALUES)


def test_both_limbs_ship_the_same_kind_because_it_is_one_property():
    """One property, one kind, one refusal shape. The label says which axis."""
    lost_time = R.temporal_honouring_facets("balance by month", HEATMAP, VALUES)
    lost_cut = R.temporal_honouring_facets(ASKS_FOR_BOTH, WHOLE_BOOK, VALUES)
    assert {f.kind for f in lost_time + lost_cut} == {R.KIND_SERIES_AXIS}
    for facets in (lost_time, lost_cut):
        verdict, message = R.assess(R.ExecutionReceipt(facets=facets))
        assert verdict == R.VERDICT_REFUSE
        assert message.endswith("I have not substituted a broader figure.")


# --------------------------------------------------------------------------- #
# The designed hole, proved rather than asserted
# --------------------------------------------------------------------------- #
# `stamp_coverage` reported KIND_SERIES_AXIS as a LIVE HOLE in sixteen
# (route, kind) cells — a kind that can be raised where nothing can confirm it.
# It is declared in DESIGNED_HOLES, and the reason given there is a CLAIM about
# behaviour: the facet never enters a reconciler. These tests are that claim's
# proof. A declaration without one is how an instrument gets talked out of a
# finding.
def test_no_reconciler_ever_receives_a_series_axis_facet():
    """The real raiser's output must not be reconcilable, because it is already
    adjudicated. Exercised through the raiser, not a hand-built facet."""
    facets = R.temporal_honouring_facets("balance by month", HEATMAP, VALUES)
    assert facets and all(f.status == R.LOST and f.reason for f in facets)


def test_only_one_place_constructs_this_kind():
    """One raiser, and the declaration in DESIGNED_HOLES depends on it.

    A second construction site — especially one before execution — would put a
    series_axis facet into a reconciler's list, and the hole would stop being
    designed.
    """
    import inspect
    from pathlib import Path

    root = Path(R.__file__).resolve().parents[1]
    sites = []
    for path in root.rglob("*.py"):
        if "test" in path.name or "/tests/" in str(path):
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        for line_no, line in enumerate(text.splitlines(), start=1):
            if "KIND_SERIES_AXIS" in line and "kind=" in line:
                sites.append("%s:%d" % (path.relative_to(root), line_no))
    assert len(sites) == 2, sites          # limb 1 and limb 2, both in the raiser
    assert all("execution_receipt.py" in s for s in sites), sites
    source = inspect.getsource(R.temporal_honouring_facets)
    assert source.count("KIND_SERIES_AXIS") == 2


def test_the_designed_hole_is_declared():
    """If the kind is ever removed or renamed, the declaration must not linger."""
    from question_interpretation import stamp_coverage

    assert "KIND_SERIES_AXIS" in stamp_coverage.DESIGNED_HOLES
    assert hasattr(R, "KIND_SERIES_AXIS")
