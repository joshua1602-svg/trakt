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


# --------------------------------------------------------------------------- #
# The predicate test — a DIFFERENT lexical decision on the same sentence.
#
# `subject_side` asks "where does the subject clause end". This asks "is the
# measure word at THIS position the subject of a predicate rather than the thing
# being measured" — "balance by region where LTV above 50%" measures balance and
# filters on LTV, and counting the filter subject as a second requested measure
# would refuse a perfectly good filtered breakdown.
#
# Its vocabulary genuinely differs from CONDITION_OPENERS, and the difference is
# CORRECT rather than drift. It carries `between`, `exceeding`, `in excess of`
# and the comparison operators, and it does NOT carry `where`, `with` or `for`,
# because those are clause openers and this test wants comparators. The
# inventory's finding was that the three splits were separately DECLARED, not
# that they should be identical — so ownership is consolidated here while the
# two vocabularies stay distinct and named.
# --------------------------------------------------------------------------- #

#: Comparators, for the predicate test. Distinct from CONDITION_OPENERS.
COMPARATORS: Tuple[str, ...] = (
    "above", "below", "over", "under", "more than", "less than",
    "greater than", "at least", "at most", "exceeding", "in excess of",
)
#: `between` reads as a comparator only AFTER the measure word: "LTV between 40
#: and 60" is a predicate, while "between" before it is not a bound on it.
COMPARATORS_AFTER_ONLY: Tuple[str, ...] = ("between",)

#: Filler that can sit between a measure word and its comparator
#: ("LTV of more than 40%", "balance is above £100k").
_FILTER_FILLER = r"(?:of|is|are|that is|which is|at|with|having)?"

_FILTER_AFTER_RE = re.compile(
    r"^\s*" + _FILTER_FILLER + r"\s*(?:[<>]=?|=)|"
    r"^\s*" + _FILTER_FILLER + r"\s*\b(?:"
    + "|".join(COMPARATORS + COMPARATORS_AFTER_ONLY) + r")\b|"
    r"^\s*\d[\d,.]*\s*(?:%|\+)", re.I)

_FILTER_BEFORE_RE = re.compile(
    r"\b(?:" + "|".join(COMPARATORS) + r"|[<>]=?)"
    r"\s*£?\s*\d[\d,.]*\s*%?\s*$", re.I)

#: How far either side of the measure word to look. A predicate binds close;
#: widening this would start matching a comparator from a different clause.
PREDICATE_WINDOW = 32


def is_filter_subject(text: str, start: int, end: int) -> bool:
    """True when the word at ``[start:end]`` is the subject of a predicate.

    Both sides are checked: a comparator immediately after ("LTV above 50%") or
    a comparator and number immediately before ("above 50% LTV").
    """
    if _FILTER_AFTER_RE.search(text[end:end + PREDICATE_WINDOW]):
        return True
    return bool(_FILTER_BEFORE_RE.search(
        text[max(0, start - PREDICATE_WINDOW):start]))


def metric_slot(text: str) -> str:
    """The span that may legitimately NAME the metric.

    The same condition truncation as `subject_side`, WITHOUT the grouping cut:
    the parser splits grouping clauses upstream in `_grouping_segments`, so
    `_metric_slot` composes with that split rather than repeating it.

    That difference is why the owner exposes `condition_cut` and `grouping_cut`
    separately — two consumers need the same vocabulary applied in different
    compositions, and a single fused function could serve only one of them.
    """
    head = text or ""
    cut = condition_cut(head)
    if cut is not None and head[:cut].strip():
        return head[:cut].strip()
    return head.strip()
