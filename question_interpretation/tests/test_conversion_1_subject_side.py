#!/usr/bin/env python3
"""Stage 3, conversion 1: answer_type.subject_side reads the single owner.

What it used to decide: the subject-side span, using its own copy of thirteen
condition openers, declared in answer_type and again in
llm_query_parser._metric_slot.

What the object now supplies: the same span, from
question_interpretation.lexical, which declares those openers once.

The diff in decisions across the corpus: none — 690 of 690 identical.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
_EVIDENCE = _REPO_ROOT / "due_diligence" / "evidence" / "analytical_intent_v1"
for p in (str(_REPO_ROOT), str(_EVIDENCE)):
    if p not in sys.path:
        sys.path.insert(0, p)

from question_interpretation import lexical  # noqa: E402


def test_answer_type_no_longer_declares_the_vocabulary():
    """The point of the conversion. Two declarations agreeing by maintenance is
    the defect; one declaration is the fix."""
    from mi_agent import answer_type as AT
    assert not hasattr(AT, "_CONDITION_OPENERS")
    assert not hasattr(AT, "_CONDITION_RE")


def test_the_vocabulary_is_declared_once_and_unchanged():
    """Same thirteen terms. A conversion that quietly widened them would change
    which questions get cut."""
    assert set(lexical.CONDITION_OPENERS) == {
        "where", "with", "for", "above", "below", "over", "under",
        "greater than", "less than", "at least", "at most",
        "more than", "fewer than"}
    assert len(lexical.CONDITION_OPENERS) == 13


def test_answer_type_delegates_rather_than_reimplementing(monkeypatch):
    """Proof the delegation is live: redirect the owner and answer_type follows.

    A test that only compared outputs would pass against a copy-paste.
    """
    from mi_agent import answer_type as AT
    monkeypatch.setattr(lexical, "subject_side", lambda q: "SENTINEL")
    assert AT.subject_side("balance where LTV above 50%") == "SENTINEL"


def test_the_delegation_check_can_fail():
    """Without the redirect it returns the real answer, so the test above is
    detecting delegation rather than a constant."""
    from mi_agent import answer_type as AT
    assert AT.subject_side("balance where LTV above 50%") == "balance"


@pytest.mark.parametrize("question,expected", [
    ("balance where LTV above 50%", "balance"),
    ("balance by region where borrower age is over 70", "balance"),
    ("regions with the highest LTV", "regions with the highest LTV"),
    ("loans with LTV above 50%", "loans"),
    ("for 3 loans by broker", "for 3 loans"),
    ("above 50", "above 50"),
    ("by region", ""),
    ("", ""),
])
def test_the_rule_is_unchanged(question, expected):
    assert lexical.subject_side(question) == expected


def test_the_whole_corpus_is_unchanged():
    """The per-conversion evidence, asserted rather than only reported.

    Recomputed against the rule as it was written in answer_type before the
    conversion, so this keeps holding even after that source is gone.
    """
    import re
    from question_interpretation.lexical_decisions import corpus

    openers = ("where", "with", "for", "above", "below", "over", "under",
               "greater than", "less than", "at least", "at most",
               "more than", "fewer than")
    cond = re.compile(r"\b(" + "|".join(re.escape(t) for t in
                                        sorted(openers, key=len, reverse=True)) + r")\b")

    def original(question):
        head = re.split(r"\bby\b", question or "")[0]
        m = cond.search(head)
        while m:
            if re.search(r"\d", head[m.end():]):
                return head[:m.start()].strip() or head.strip()
            m = cond.search(head, m.end())
        return head.strip()

    cases = corpus()
    assert len(cases) == 690
    differing = [c["question"] for c in cases
                 if original(c["question"]) != lexical.subject_side(c["question"])]
    assert differing == [], differing[:5]


def test_the_object_carries_the_subject_side_span():
    """'Reads the object' means the object holds it. The span is over the
    ORIGINAL question, so a consumer can show where its decision came from."""
    from question_interpretation.schema import QuestionInterpretation
    from question_interpretation.projection import _subject

    class _Spec:
        metric = "current_outstanding_balance"
        aggregation = "sum"

    qi = QuestionInterpretation(question="balance where LTV above 50%")
    _subject(qi, _Spec())
    assert qi.subject.raw_text == "balance"
    assert qi.subject.has_span
    start, end = qi.subject.span.start, qi.subject.span.end
    assert qi.question[start:end].strip() == "balance"
