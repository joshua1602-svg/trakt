#!/usr/bin/env python3
"""question_interpretation/lexical.py

The single owner of the subject-side clause split.

The inventory found this one decision implemented three times, from three
separately-declared vocabularies that agree today by maintenance rather than by
construction:

    answer_type.subject_side            13 terms, declared in answer_type
    llm_query_parser._metric_slot       the SAME 13 terms, declared again
    execution_receipt._is_filter_subject
                                        a THIRD vocabulary, which adds
                                        "between", "exceeding", "in excess of"
                                        and the comparison operators, and drops
                                        "where", "with", "for", "fewer than"

Stage 3 converts them onto this module one at a time, each conversion proved by
`question_interpretation.lexical_decisions` to change no decision on any of the
690 real-surface questions.

Nothing here imports `mi_agent`. It reads the question text and nothing else —
no registry, no frame, no spec — so it can be the owner without becoming a
dependency cycle, and so a consumer converting onto it gains no new knowledge it
did not have.
"""
from __future__ import annotations

import re
from typing import Optional, Tuple

#: Clause openers that introduce a CONDITION.
#:
#: A measure word inside a condition names the field being filtered ON, not the
#: thing being reported. The first version of the answer-type classifier scanned
#: the whole question and typed "balance by region where borrower age is over
#: 70" as an AGE question — the very defect the sweep was built to find,
#: reproduced in the instrument looking for it.
#:
#: This is the ONE declaration. A consumer that needs these terms imports them.
CONDITION_OPENERS: Tuple[str, ...] = (
    "where", "with", "for", "above", "below", "over", "under",
    "greater than", "less than", "at least", "at most",
    "more than", "fewer than",
)

_CONDITION_RE = re.compile(
    r"\b(" + "|".join(re.escape(t) for t in
                      sorted(CONDITION_OPENERS, key=len, reverse=True)) + r")\b")

_BY_RE = re.compile(r"\bby\b")
_DIGIT_RE = re.compile(r"\d")


def condition_cut(text: str) -> Optional[int]:
    """Offset where the first CONDITION clause begins, or None.

    An opener counts only where a numeric bound follows it, which is what makes
    "loans with LTV above 50%" cut while "regions with the highest LTV" does
    not. Returning the offset rather than the slice is deliberate: the caller
    decides what to do with it, and a span can be recorded on the object.
    """
    match = _CONDITION_RE.search(text or "")
    while match:
        if _DIGIT_RE.search(text[match.end():]):
            return match.start()
        match = _CONDITION_RE.search(text, match.end())
    return None


def grouping_cut(text: str) -> Optional[int]:
    """Offset where the first GROUPING clause begins, or None.

    "balance by LTV bucket" is a currency question grouped by a rate band, and
    reading past "by" typed it as a rate — the condition defect, one clause
    along.
    """
    match = _BY_RE.search(text or "")
    return match.start() if match else None


def subject_side_span(question: str) -> Tuple[int, int]:
    """(start, end) of the span that may name the measure being reported.

    Grouping first, then condition — the order `answer_type.subject_side` has
    always applied. The span is over the ORIGINAL question, so a consumer can
    record where its decision came from.
    """
    text = question or ""
    end = len(text)
    cut = grouping_cut(text)
    if cut is not None:
        end = cut
    head = text[:end]
    cut = condition_cut(head)
    if cut is not None:
        # `or head.strip()` in the original: a cut at position 0 leaves nothing,
        # and the whole head is the better answer than an empty string.
        if head[:cut].strip():
            end = cut
    return (0, end)


def subject_side(question: str) -> str:
    """The subject-side text. Byte-identical to the three implementations it
    replaces, proved across the corpus at each conversion."""
    start, end = subject_side_span(question)
    return (question or "")[start:end].strip()
