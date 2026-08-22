#!/usr/bin/env python3
"""Stage 3, conversion 2: execution_receipt._is_filter_subject reads the owner.

The third copy, and the one whose vocabulary genuinely DIFFERS — which is the
interesting part. It is not merged with the condition openers, because it
answers a different question: `subject_side` asks where the subject clause ends,
this asks whether the measure word at a given position is the subject of a
predicate. Comparators, not clause openers.

This one is on the parser path: `llm_query_parser._measure_hits` imports it.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from question_interpretation import lexical  # noqa: E402


def test_the_receipt_no_longer_declares_the_regexes():
    from mi_agent import execution_receipt as R
    assert not hasattr(R, "_FILTER_AFTER_RE")
    assert not hasattr(R, "_FILTER_BEFORE_RE")
    assert not hasattr(R, "_FILTER_FILLER")


def test_the_two_vocabularies_are_distinct_and_that_is_deliberate():
    """Merging them would be the wrong fix. This test records why."""
    comparators = set(lexical.COMPARATORS) | set(lexical.COMPARATORS_AFTER_ONLY)
    openers = set(lexical.CONDITION_OPENERS)
    # Only the predicate test carries these:
    assert {"between", "exceeding", "in excess of"} <= comparators
    assert not ({"between", "exceeding", "in excess of"} & openers)
    # Only the clause-opener test carries these:
    assert {"where", "with", "for", "fewer than"} <= openers
    assert not ({"where", "with", "for", "fewer than"} & comparators)


def test_it_delegates_rather_than_reimplementing(monkeypatch):
    from mi_agent import execution_receipt as R
    monkeypatch.setattr(lexical, "is_filter_subject", lambda t, s, e: "SENTINEL")
    assert R._is_filter_subject("balance where ltv above 50%", 15, 18) == "SENTINEL"


def test_the_delegation_check_can_fail():
    from mi_agent import execution_receipt as R
    q = "balance where ltv above 50%"
    assert R._is_filter_subject(q, q.index("ltv"), q.index("ltv") + 3) is True


@pytest.mark.parametrize("question,word,expected", [
    ("balance where ltv above 50%", "ltv", True),          # comparator after
    ("above 50% ltv by region", "ltv", True),              # comparator before
    ("balance by region", "balance", False),               # neither side
    ("ltv of more than 40%", "ltv", True),                 # filler then comparator
    ("ltv between 40 and 60", "ltv", True),                # after-only comparator
    ("what is the average ltv", "ltv", False),
])
def test_the_rule_is_unchanged(question, word, expected):
    idx = question.index(word)
    assert lexical.is_filter_subject(question, idx, idx + len(word)) is expected


def test_the_window_is_bounded_and_named():
    """A predicate binds close. Widening the window would start matching a
    comparator from a different clause, so the bound is a named constant."""
    assert lexical.PREDICATE_WINDOW == 32
    far = "ltv" + (" " * 40) + "above 50%"
    assert lexical.is_filter_subject(far, 0, 3) is False
    near = "ltv above 50%"
    assert lexical.is_filter_subject(near, 0, 3) is True


def test_equivalence_at_every_offset_on_a_sample():
    """The per-conversion evidence, recomputed against the ORIGINAL regexes so
    it keeps holding after the source they came from is gone.

    The full run covered 400,097 positions across the corpus; this pins a
    representative sample so a regression is caught by the suite rather than
    only by a corpus run.
    """
    import re
    filler = r"(?:of|is|are|that is|which is|at|with|having)?"
    after = re.compile(
        r"^\s*" + filler + r"\s*(?:[<>]=?|=)|"
        r"^\s*" + filler + r"\s*\b(?:above|below|over|under|more than|"
        r"less than|greater than|at least|at most|between|exceeding|"
        r"in excess of)\b|"
        r"^\s*\d[\d,.]*\s*(?:%|\+)", re.I)
    before = re.compile(
        r"\b(?:above|below|over|under|more than|less than|greater than|at least|"
        r"at most|exceeding|in excess of|[<>]=?)\s*£?\s*\d[\d,.]*\s*%?\s*$", re.I)

    def original(q, start, end):
        if after.search(q[end:end + 32]):
            return True
        return bool(before.search(q[max(0, start - 32):start]))

    samples = [
        "balance where ltv above 50%",
        "how many loans have a balance above £250k",
        "above 50% ltv by region",
        "ltv of more than 40% and balance is above £100k",
        "what is the balance of loans with borrower age 70+ and ltv above 50%?",
        "ltv between 40 and 60",
        "balance by region",
    ]
    checked = 0
    for q in samples:
        for i in range(len(q) + 1):
            for j in range(i, min(i + 20, len(q)) + 1):
                assert original(q, i, j) == lexical.is_filter_subject(q, i, j), (q, i, j)
                checked += 1
    assert checked > 3000, checked
