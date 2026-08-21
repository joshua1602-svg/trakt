#!/usr/bin/env python3
"""Standing condition 4 for the per-conversion lexical-decision instrument.

Stage 3 converts one consumer per commit, and this is what proves a conversion
changed nothing. It is finer than the answer-text diff on purpose: a conversion
that moved a consumer's decision but happened not to reach the answer would pass
the answer diff and must fail here.

So it has to be shown to catch a moved decision, a dropped question and a new
one — not merely to report a number.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from question_interpretation.lexical_decisions import compare  # noqa: E402


def _rec(ident, question, **decisions):
    base = {"answer_type.subject_side": "balance",
            "answer_type.asked": "currency",
            "llm_query_parser._metric_slot": "balance",
            "period_request.requested_unit": None,
            "period_request.requested_span": None,
            "execution_receipt._is_filter_subject": []}
    base.update(decisions)
    return {"surface": "ere_mi_calibration_250", "id": ident,
            "question": question, "decisions": base}


def test_identical_decisions_do_not_move():
    before = [_rec("a", "balance where LTV above 50%")]
    after = [_rec("a", "balance where LTV above 50%")]
    result = compare(before, after)
    assert result["moved"] == [] and result["identical"] == 1


def test_a_moved_subject_side_is_caught_and_attributed():
    """The exact failure conversion 1 must not cause: the split changes, so the
    measure slot changes, so the metric could change."""
    before = [_rec("a", "balance where LTV above 50%",
                   **{"answer_type.subject_side": "balance"})]
    after = [_rec("a", "balance where LTV above 50%",
                  **{"answer_type.subject_side": "balance where LTV"})]
    result = compare(before, after)
    assert len(result["moved"]) == 1
    assert result["per_function"] == {"answer_type.subject_side": 1}
    changed = result["moved"][0]["changed"]["answer_type.subject_side"]
    assert changed["before"] == "balance"
    assert changed["after"] == "balance where LTV"


def test_each_function_is_attributed_separately():
    """One conversion per commit means a move must name which consumer moved."""
    before = [_rec("a", "q1"), _rec("b", "q2")]
    after = [_rec("a", "q1", **{"llm_query_parser._metric_slot": "changed"}),
             _rec("b", "q2", **{"period_request.requested_unit": "month"})]
    result = compare(before, after)
    assert result["per_function"] == {
        "llm_query_parser._metric_slot": 1,
        "period_request.requested_unit": 1}


def test_a_none_to_value_move_is_caught():
    """Stage 5 turns requested_unit into something carried. A slot going from
    None to a value is a move, not an absence."""
    before = [_rec("a", "funded balance by month")]
    after = [_rec("a", "funded balance by month",
                  **{"period_request.requested_unit": "month"})]
    assert len(compare(before, after)["moved"]) == 1


def test_a_dropped_or_added_question_is_reported():
    before = [_rec("a", "q1"), _rec("b", "q2")]
    after = [_rec("a", "q1"), _rec("c", "q3")]
    result = compare(before, after)
    assert result["only_before"] == ["ere_mi_calibration_250/b"]
    assert result["only_after"] == ["ere_mi_calibration_250/c"]


def test_the_is_filter_subject_probe_moves_when_its_verdict_flips():
    before = [_rec("a", "balance where LTV above 50%",
                   **{"execution_receipt._is_filter_subject":
                      [{"at": "ltv", "is_filter_subject": True}]})]
    after = [_rec("a", "balance where LTV above 50%",
                  **{"execution_receipt._is_filter_subject":
                     [{"at": "ltv", "is_filter_subject": False}]})]
    assert len(compare(before, after)["moved"]) == 1


def test_the_real_corpus_is_covered_and_stable():
    """The instrument must actually see the corpus it claims to, and give the
    same answer twice."""
    from question_interpretation.lexical_decisions import collect, corpus
    cases = corpus()
    # 697, not 693: b16_001..004 were added to the calibration bank as B16a's
    # surface, declared failing BEFORE the fix. Before them, pop_253/254/255 were
    # added the same way for the point-in-time population hole. The corpus grows
    # when a defect earns a case; no existing decision moved, which is what the
    # stability half asserts.
    assert len(cases) == 697, len(cases)
    surfaces = {c["surface"] for c in cases}
    assert surfaces == {"ere_mi_calibration_250", "ere_mi_questions",
                        "nl_robustness_alderbridge", "nl_robustness_kestrelmoor"}
    once, twice = collect(), collect()
    assert compare(once, twice)["moved"] == []
