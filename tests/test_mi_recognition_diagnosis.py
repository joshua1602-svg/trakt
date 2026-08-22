"""tests/test_mi_recognition_diagnosis.py — the attribution must be able to fail.

This instrument decides whether a failure is cheap (wording) or expensive
(capability). Getting that wrong does not produce an obviously wrong number — it
produces a plausible one that misdirects a build.

The specific error it must not make: reading an unresolved element straight off
the parse. `Show me balance by month by region` resolves NO dimension, but the
vocabulary knows "region" perfectly well — the spec cannot hold a time axis and a
dimension at once, and the guard names the dimension back in its refusal. Scoring
that as recognition would report ~70% recognition instead of the 33% that
survives the control, and would move a build's worth of work into the cheap
column.
"""
from __future__ import annotations

import pytest

from question_interpretation import mi_recognition_diagnosis as D


def _row(shape, question, target, delivered=False, cause=D.CAUSE_NONE,
         named_back=False, route="evolution"):
    return {"shape": shape, "question": question, "target": target,
            "delivered": delivered, "cause": cause, "named_back": named_back,
            "route": route, "claimed_by_a_route": route is not None}


# --------------------------------------------------------------------------- #
# 1. UNDERSTOOD-BUT-UNCARRIED IS CAPABILITY, NOT RECOGNITION
# --------------------------------------------------------------------------- #
def test_an_element_named_back_by_the_guard_is_capability_not_recognition():
    """The exact T3 case: no dimension in the spec, but the refusal quotes it."""
    rows = D.assign_verdicts([
        _row("T3", "Show me balance by month by region", "region",
             cause=D.CAUSE_NO_DIMENSION, named_back=True)])
    assert rows[0]["verdict"] == D.CAPABILITY, \
        "the vocabulary knew the word; the spec could not carry both limbs"


def test_an_element_neither_resolved_nor_named_back_is_unparsed():
    rows = D.assign_verdicts([
        _row("T1", "how is the loan book tracking month to month", "none",
             cause=D.CAUSE_NO_METRIC, named_back=False, route=None)])
    assert rows[0]["verdict"] == D.UNPARSED


def test_a_failure_with_a_delivering_sibling_is_wording_whatever_the_parse_says():
    """A reachable request is a wording problem even if the parse dropped
    something — the capability demonstrably answers under other words."""
    rows = D.assign_verdicts([
        _row("T1", "balance over time", "none", delivered=True),
        _row("T1", "outstanding balances by period", "none",
             cause=D.CAUSE_NO_TIME, named_back=False, route=None)])
    assert rows[1]["verdict"] == D.WORDING


def test_a_sibling_of_a_DIFFERENT_target_does_not_make_it_wording():
    """The standing sibling rule. T7's region ranking working must not make T7's
    LTV-band ranking a wording problem — that would move build work into the
    cheap column."""
    rows = D.assign_verdicts([
        _row("T7", "Which region has grown fastest?", "region", delivered=True),
        _row("T7", "Which LTV band moved most between periods?", "ltv_band",
             cause=D.CAUSE_NO_TIME, named_back=False)])
    assert rows[1]["verdict"] != D.WORDING
    assert rows[1]["verdict"] == D.UNPARSED


def test_everything_resolved_and_still_undelivered_is_capability():
    rows = D.assign_verdicts([
        _row("T6", "What moved between periods by region?", "region",
             cause=D.CAUSE_NONE, named_back=False)])
    assert rows[0]["verdict"] == D.CAPABILITY


def test_delivered_always_wins():
    rows = D.assign_verdicts([
        _row("T1", "balance over time", "none", delivered=True,
             cause=D.CAUSE_NO_METRIC)])
    assert rows[0]["verdict"] == D.DELIVERED


# --------------------------------------------------------------------------- #
# 2. THE CAUSE ORDER
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("want,got,expected", [
    ([], {"metric": False, "time": False, "dimension": False}, D.CAUSE_NO_METRIC),
    ([], {"metric": True, "time": False, "dimension": False}, D.CAUSE_NO_TIME),
    (["region"], {"metric": True, "time": True, "dimension": False},
     D.CAUSE_NO_DIMENSION),
    (["region"], {"metric": True, "time": True, "dimension": True}, D.CAUSE_NONE),
    # A shape that asks for no dimension is not faulted for lacking one.
    ([], {"metric": True, "time": True, "dimension": False}, D.CAUSE_NONE),
])
def test_cause_names_the_first_unresolved_element(want, got, expected):
    assert D.cause(want, got) == expected


def test_named_back_reads_the_honour_or_clarify_message():
    assert D._named_back(
        "Show me balance by month by region", ["region"],
        "I understood that you asked for region, but that could not be applied")
    assert not D._named_back("Show me balance by month by region", ["region"],
                             "Here is the result for your query.")


def test_named_back_is_false_on_an_empty_answer():
    assert not D._named_back("anything", ["region"], "")


# --------------------------------------------------------------------------- #
# 3. THE NO-ROUTE COLUMN IS INDEPENDENT OF THE VERDICT
# --------------------------------------------------------------------------- #
def test_no_route_is_recorded_separately_from_the_verdict():
    """A question can reach a route and still be a wording failure, or reach none
    and be a capability failure. Collapsing the two would hide either."""
    rows = D.assign_verdicts([
        _row("T7", "Which region has grown fastest?", "region", delivered=True),
        _row("T7", "Rank regions by balance growth over time", "region",
             cause=D.CAUSE_NO_DIMENSION, named_back=True,
             route="period_change_analysis"),
    ])
    assert rows[1]["verdict"] == D.WORDING
    assert rows[1]["claimed_by_a_route"] is True
