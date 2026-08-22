#!/usr/bin/env python3
"""Standing condition 4 for the answer-text differ.

Standing condition 7 makes this instrument the acceptance gate for every stage,
and the argument for leaving the grading path alone rests entirely on it being
STRICTER than the bank. So it has to be shown to catch what the bank cannot —
not merely to report a number.

The central case is Finding 1: a portfolio summary that silently drops its
balance measure PASSES the calibration bank, because `expected_answer_type:
mixed` is satisfied by an observed type of `count`. It must fail here.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from question_interpretation.answer_diff import (  # noqa: E402
    _digest, _keyed, _user_visible, compare,
)

#: A portfolio summary as it answers today: a count AND a balance.
_SUMMARY_TODAY = {
    "ok": True,
    "answer": "The book holds 11,035 loans with a funded balance of £1,964.89m.",
    "artifacts": [{"type": "kpi", "title": "Portfolio summary", "rows": [{}],
                   "columns": ["loan_count", "current_outstanding_balance_sum"]}],
    "warnings": [],
}

#: The same summary having silently dropped its balance measure. This is the
#: regression the bank cannot see.
_SUMMARY_LOST_A_MEASURE = {
    "ok": True,
    "answer": "The book holds 11,035 loans.",
    "artifacts": [{"type": "kpi", "title": "Portfolio summary", "rows": [{}],
                   "columns": ["loan_count"]}],
    "warnings": [],
}


def _record(surface, ident, question, payload):
    r = _user_visible(payload)
    return {"surface": surface, "id": ident, "question": question,
            "digest": _digest(r), "record": r}


def test_identical_answers_do_not_move():
    before = [_record("calibration_bank", "kpi_028", "portfolio summary",
                      _SUMMARY_TODAY)]
    after = [_record("calibration_bank", "kpi_028", "portfolio summary",
                     _SUMMARY_TODAY)]
    result = compare(before, after)
    assert result["moved"] == []
    assert result["identical"] == 1


def test_a_summary_that_loses_a_measure_is_caught():
    """The Finding 1 regression. The bank passes it; this must not."""
    before = [_record("calibration_bank", "kpi_028", "portfolio summary",
                      _SUMMARY_TODAY)]
    after = [_record("calibration_bank", "kpi_028", "portfolio summary",
                     _SUMMARY_LOST_A_MEASURE)]
    result = compare(before, after)
    assert len(result["moved"]) == 1
    moved = result["moved"][0]
    assert moved["id"] == "kpi_028"
    assert moved["before"]["answer"] != moved["after"]["answer"]
    assert (moved["before"]["artifacts"][0]["columns"]
            != moved["after"]["artifacts"][0]["columns"])


def test_the_bank_would_have_passed_that_same_regression():
    """Why the differ is needed at all, asserted rather than claimed.

    `of_measure(None, 'count')` types both the healthy and the broken summary as
    `count`, and `_SATISFIES[MIXED]` accepts `count`. So the declared
    expectation cannot separate them.
    """
    from mi_agent import answer_type as AT
    healthy_type = AT.of_measure(None, "count", None)
    broken_type = AT.of_measure(None, "count", None)
    assert healthy_type == broken_type == AT.COUNT
    assert AT.satisfies(AT.MIXED, broken_type) is True


def test_a_changed_refusal_reason_is_caught():
    """A refusal that starts citing a different reason is a user-visible move,
    even though ok=False either way and the bank grades both as refusals."""
    before = [_record("calibration_bank", "x", "q",
                      {"ok": False, "error": "No loans match that filter (region)."})]
    after = [_record("calibration_bank", "x", "q",
                     {"ok": False, "error": "Arrears Bucket is not available."})]
    assert len(compare(before, after)["moved"]) == 1


def test_added_metadata_does_not_count_as_a_move():
    """Stage 2 ADDS the interpretation object to the result. The diff must not
    flag that: it measures what reaches the reader, not what the payload
    carries."""
    before = [_record("calibration_bank", "x", "q", dict(_SUMMARY_TODAY))]
    after_payload = dict(_SUMMARY_TODAY)
    after_payload["metadata"] = {"questionInterpretation": {"operation": "amount"}}
    after_payload["execution_receipt"] = {"facets": []}
    after = [_record("calibration_bank", "x", "q", after_payload)]
    assert compare(before, after)["moved"] == []


def test_a_dropped_case_is_reported_not_silently_ignored():
    before = [_record("calibration_bank", "a", "q1", _SUMMARY_TODAY),
              _record("calibration_bank", "b", "q2", _SUMMARY_TODAY)]
    after = [_record("calibration_bank", "a", "q1", _SUMMARY_TODAY)]
    result = compare(before, after)
    assert result["only_before"] == ["calibration_bank/b"]


def test_two_records_cannot_collapse_onto_one_key():
    """The defect this instrument shipped with for one run.

    Q8 runs the same four sentence shapes against three governed X/Y pairs, so
    (intent, variation) repeats three times per book. Keying on it alone
    dropped 16 of 88 robustness answers — precisely the seasoning family Stage 4
    must leave unmoved. Collapsing must raise, never silently lose a case.
    """
    records = [
        _record("robustness_alderbridge", "Q8.1", "…front book…", _SUMMARY_TODAY),
        _record("robustness_alderbridge", "Q8.1", "…South East…", _SUMMARY_TODAY),
    ]
    with pytest.raises(ValueError, match="duplicate answer-diff key"):
        _keyed(records)


def test_the_real_q8_identities_are_distinct():
    """And the fix holds on the real bank: 44 distinct ids per book."""
    sys.path.insert(0, str(_REPO_ROOT / "due_diligence" / "evidence" / "analytical_intent_v1"))
    import nl_bank
    for book in ("alderbridge", "kestrelmoor"):
        idents = []
        for entry in nl_bank.bank(book):
            pair = entry.get("pair")
            ident = "%s.%s" % (entry["intent"], entry["variation"])
            if pair:
                ident = "%s[%s]" % (ident, pair if isinstance(pair, str)
                                    else "|".join(str(x) for x in pair))
            idents.append(ident)
        assert len(idents) == 44, (book, len(idents))
        assert len(set(idents)) == 44, (book, len(idents) - len(set(idents)))
