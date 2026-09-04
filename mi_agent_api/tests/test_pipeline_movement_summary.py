#!/usr/bin/env python3
"""The whole pipeline's movement for one governed interval, as one structure.

WHY THIS EXISTS. `pipeline_stage_movement` answers about ONE stage or ONE
transition — it needs a named stage or a source/destination pair, and returns
None without them. Two questions in the live bank name neither:

    Give me the stage movement summary.
    Compare stage movement with the prior period.

Routing them at the existing capability would have forced it to INVENT a stage,
which is the substitution the estate forbids, so `names_a_stage_movement` is
deliberately untouched. What was missing is not a route: it is an ALL-STAGES
movement summary.

IT COMPOSES, IT DOES NOT CALCULATE. Every figure below is read from the governed
payload `movement_detail.resolve_stage_transition_detail` already publishes —
the same payload the stage route, the React movement-detail endpoint and the
PPTX deck consume. No new economics: this module reshapes and reconciles, and
where the governed data does not evidence an element it OMITS it and says so.

WITHDRAWALS ARE THE TEST OF THAT. `governed_outcome` is a canonical terminal
stage only where the prior extract recorded one; everything else stays
`unclassified_departure`. A summary that reported those as withdrawals would be
inventing an economic fact, so they are reported as departures, the withdrawal
split is omitted, and the omission is named.

ONE RESULT, TWO CONSUMERS. The MI query route renders it; the Teams proactive
briefing will consume the same structure. The result is therefore data, not
prose — nothing here formats a sentence.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import pytest

from mi_agent_api import pipeline_movement_summary as PMS


@pytest.fixture(scope="module")
def payload():
    """The real governed payload from the module fixture's pipeline root."""
    sys.argv = ["pytest"]
    from mi_agent_api.tests import test_stage_movement_query as T
    from mi_agent_api import movement_detail as detail

    T._ensure_env()
    return detail.resolve_stage_transition_detail(
        os.environ["MI_AGENT_PIPELINE_ROOT"], "client_001")


@pytest.fixture(scope="module")
def summary(payload):
    return PMS.build(payload)


# ------------------------------------------------------------- the contract #
def test_it_is_available_and_names_its_window(summary):
    assert summary["available"] is True
    assert summary["window"]["opening_date"] == "2026-06-05"
    assert summary["window"]["closing_date"] == "2026-06-12"
    assert summary["measure"] == "current_outstanding_balance"


def test_it_carries_opening_closing_and_net(summary):
    for block in ("opening", "closing", "net"):
        assert set(summary[block]) == {"cases", "balance"}, block


def test_the_net_is_the_difference_and_not_a_separate_calculation(summary):
    assert summary["net"]["cases"] == (summary["closing"]["cases"]
                                       - summary["opening"]["cases"])
    assert summary["net"]["balance"] == pytest.approx(
        summary["closing"]["balance"] - summary["opening"]["balance"])


def test_the_case_counts_are_the_governed_ones(summary, payload):
    assert summary["closing"]["cases"] == payload["counts"]["current"]
    assert summary["opening"]["cases"] == payload["counts"]["comparison"]
    assert summary["net"]["cases"] == payload["counts"]["change"]


def test_entrants_progressions_and_departures_are_all_present(summary):
    assert summary["entrants"]["cases"] >= 0
    assert summary["progressions"], "no stage-to-stage progressions"
    for row in summary["progressions"]:
        assert set(row) >= {"source", "destination", "cases", "balance",
                            "balance_change"}
    assert summary["departures"]["cases"] >= 0


def test_per_stage_rows_carry_opening_arrivals_departures_closing(summary):
    assert summary["by_stage"], "no per-stage rows"
    for row in summary["by_stage"]:
        assert set(row) >= {"stage", "opening_cases", "arrivals", "departures",
                            "closing_cases", "opening_balance",
                            "closing_balance"}


def test_persistent_cases_report_their_value_change(summary):
    persistent = summary["persistent"]
    assert persistent["cases"] >= 0
    assert "balance_change" in persistent


def test_the_largest_movements_are_ranked_not_invented(summary):
    largest = summary["largest"]
    progressions = largest["progressions"]
    assert progressions == sorted(progressions,
                                  key=lambda r: -abs(r["balance"]))
    for row in progressions:
        assert row in summary["progressions"], "ranked a row that is not a fact"


# ------------------------------------------- what the data does not evidence #
def test_an_unclassified_departure_is_never_counted_as_a_withdrawal(summary):
    """`governed_outcome` is a terminal stage only where the prior extract
    recorded one. The rest are departures, and saying otherwise would invent an
    economic fact.

    An earlier version of this test assumed the fixture evidenced NOTHING and
    asserted the whole element was omitted. It evidences two outcomes, so the
    real invariant is the one below: what is unattributed stays unattributed,
    and the omission is named."""
    outcomes = {row["outcome"] for row in summary["departures"]["by_outcome"]}
    assert PMS.UNCLASSIFIED_DEPARTURE in outcomes
    withdrawn = summary["withdrawals"]["by_outcome"]
    assert PMS.UNCLASSIFIED_DEPARTURE not in {r["outcome"] for r in withdrawn}
    assert "departure_outcome_split" in {o["element"] for o in summary["omitted"]}


def test_a_completion_is_not_a_withdrawal(summary):
    """Both are departures with an evidenced terminal stage. Merging them would
    report cases that FINISHED as cases that fell out — which the first draft of
    this module did, and this test caught."""
    withdrawn = {r["outcome"] for r in summary["withdrawals"]["by_outcome"]}
    assert PMS.COMPLETED_STAGE not in {o.upper() for o in withdrawn}
    assert PMS.COMPLETED_STAGE not in {
        str(r["outcome"]).upper() for r in summary["largest"]["attrition"]}


def test_when_nothing_is_evidenced_the_element_is_omitted_with_its_reason():
    payload = {"available": True, "measure": "current_outstanding_balance",
               "as_of_date": "2026-06-12", "comparison_date": "2026-06-05",
               "counts": {"current": 1, "comparison": 1, "change": 0},
               "event_totals": {}, "transitions": [], "new_arrivals": [],
               "stayers": [],
               "departures": [{"source_stage": "OFFER",
                               "governed_outcome": "unclassified_departure",
                               "case_count": 3, "prior_amount": 10.0}],
               "reconciliation": {"by_stage": [
                   {"stage": "OFFER", "opening_case_count": 1,
                    "closing_case_count": 1,
                    "count_reconciliation_residual": 0,
                    "opening_amount": 0.0, "closing_amount": 0.0,
                    "amount_reconciliation_residual": 0.0}]}}
    out = PMS.build(payload)
    assert out["withdrawals"] is None
    assert "withdrawals" in {o["element"] for o in out["omitted"]}


def test_every_omission_says_why(summary):
    for entry in summary["omitted"]:
        assert entry["element"] and entry["reason"]


def test_completions_come_from_a_governed_transition_only(summary):
    """A completion is a transition INTO the governed completed stage — a fact
    the payload states. It is never derived from a departure."""
    if summary.get("completions") is not None:
        assert summary["completions"]["cases"] >= 0
    else:
        assert "completions" in {o["element"] for o in summary["omitted"]}


# ------------------------------------------------------------ reconciliation #
def test_the_summary_reconciles_and_says_so(summary):
    rec = summary["reconciliation"]
    assert rec["ok"] is True
    assert rec["count_residual"] == 0


def test_every_stage_row_reconciles_individually(summary):
    for row in summary["by_stage"]:
        assert row["count_residual"] == 0, row["stage"]


def test_the_stage_openings_sum_to_the_pipeline_opening(summary):
    assert sum(r["opening_cases"] for r in summary["by_stage"]) == \
        summary["opening"]["cases"]
    assert sum(r["closing_cases"] for r in summary["by_stage"]) == \
        summary["closing"]["cases"]


def test_a_payload_that_does_not_reconcile_is_reported_not_hidden():
    broken = {"available": True, "measure": "current_outstanding_balance",
              "as_of_date": "2026-06-12", "comparison_date": "2026-06-05",
              "counts": {"current": 10, "comparison": 12, "change": -2},
              "event_totals": {}, "transitions": [], "new_arrivals": [],
              "departures": [], "stayers": [],
              "reconciliation": {"by_stage": [
                  {"stage": "KFI", "opening_case_count": 4, "new_arrivals": 1,
                   "transitions_in": 0, "transitions_out": 2, "departures": 0,
                   "stayers": 2, "closing_case_count": 9,
                   "count_reconciliation_residual": 6,
                   "opening_amount": 0.0, "closing_amount": 0.0,
                   "amount_reconciliation_residual": 0.0}]}}
    out = PMS.build(broken)
    assert out["reconciliation"]["ok"] is False
    assert out["reconciliation"]["count_residual"] != 0


# ---------------------------------------------------------- unavailable data #
def test_an_unavailable_payload_refuses_rather_than_returning_zeroes():
    out = PMS.build({"available": False, "reason": "no prior snapshot",
                     "reason_code": "no_comparison"})
    assert out["available"] is False
    assert out["reason"]
    assert "opening" not in out


def test_it_never_raises_on_a_shape_it_does_not_recognise():
    for bad in (None, {}, {"available": True}, []):
        out = PMS.build(bad)
        assert out["available"] is False
