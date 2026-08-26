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
from dataclasses import dataclass
from typing import Optional, Tuple

# The contract's controlled ordering vocabulary. Imported rather than restated:
# a second copy of `increase`/`decrease`/`absolute` here would be exactly the
# duplication this module exists to end.
from question_interpretation.schema import (  # noqa: E402
    ORDER_BASIS_ABSOLUTE, ORDER_BASIS_COUNT, ORDER_BASIS_PERCENT,
    ORDER_BASIS_SHARE, ORDER_DECREASE, ORDER_EITHER, ORDER_INCREASE)

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


def condition_span(text: str) -> Optional[Tuple[int, int]]:
    """(start, end) of the first CONDITION clause, or None.

    `condition_cut` gives only the START, which is all a caller needs when the
    condition is stated LAST — everything before it is the subject. Stated
    FIRST it is not enough: truncating at the start throws away the subject,
    because the subject is what follows the condition.

    The end is the next clause boundary, or the end of the text. Punctuation is
    a boundary (see `_CLAUSE_PUNCTUATION`), which is what gives a leading
    condition an end at all.
    """
    body = text or ""
    match = _CONDITION_RE.search(body)
    while match:
        if _DIGIT_RE.search(body[match.end():]):
            break
        match = _CONDITION_RE.search(body, match.end())
    if match is None:
        return None
    # THE CLAUSE ENDS AFTER ITS BOUND, not at the next connective.
    #
    # An opener is followed by the connective that introduces its own body —
    # "for loans" then "with" — so searching for a boundary from the end of the
    # opener finds one INSIDE the condition and reports a two-word clause.
    # Measured: (0, 10), "for loans ", for a question whose condition runs to
    # the comma. A condition only counts when a numeric bound follows it
    # (`condition_cut`'s own rule), so the bound is where the clause's content
    # ends and the search for its boundary begins.
    bound = _DIGIT_RE.search(body, match.end())
    if bound is None:
        return None
    boundary = _CLAUSE_SPLIT_RE.search(body, bound.end())
    return (match.start(), boundary.start() if boundary else len(body))


#: WORDS THAT NAME A FORWARD HORIZON. One governed reader, because a forward
#: question that states how far ahead it looks and is answered over a different
#: horizon has had a declared element replaced — the same defect class as a
#: replaced period, one tense along. Measured before this existed: "what will
#: the book be worth in five years?", "…in 12 months?" and "…in 5 years?" all
#: returned the identical open-pipeline composition, with the horizon named
#: nowhere in the answer.
_HORIZON_UNITS: Tuple[Tuple[str, int], ...] = (
    ("year", 12), ("yr", 12), ("quarter", 3), ("month", 1), ("week", 0))

_HORIZON_WORDS = {
    "a": 1, "an": 1, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10, "twelve": 12,
    "eighteen": 18, "twenty": 20, "next": 1,
}

_HORIZON_RE = re.compile(
    r"\b(?:in|over|within|across|after|for)\s+(?:the\s+)?(?:next\s+)?"
    r"([0-9]+|" + "|".join(_HORIZON_WORDS) + r")\s*"
    r"(year|yr|quarter|month|week)s?\b", re.I)

_HORIZON_NEXT_RE = re.compile(r"\bnext\s+(year|quarter|month)\b", re.I)


def forecast_horizon_months(question: Optional[str]) -> Optional[int]:
    """How many months ahead the question looks, or ``None``.

    Returns MONTHS so a consumer compares one number against its own projection
    horizon. A week-scale horizon returns 0 — stated, and shorter than a month —
    rather than None, because "none stated" and "very short" are different facts.
    """
    text = (question or "").lower()
    match = _HORIZON_RE.search(text)
    if match:
        token, unit = match.group(1), match.group(2)
        count = (int(token) if token.isdigit()
                 else _HORIZON_WORDS.get(token))
        if count is None:
            return None
        factor = dict(_HORIZON_UNITS).get(unit, 1)
        return int(count * factor)
    bare = _HORIZON_NEXT_RE.search(text)
    if bare:
        return dict(_HORIZON_UNITS).get(bare.group(1), 1)
    return None


def grouping_cut(text: str) -> Optional[int]:
    """Offset where the first GROUPING clause begins, or None.

    "balance by LTV bucket" is a currency question grouped by a rate band, and
    reading past "by" typed it as a rate — the condition defect, one clause
    along.
    """
    match = AXIS_MARKER_RE.search(text or "")
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


# --------------------------------------------------------------------------- #
# THE SELECTOR MARK — a decision with no owner until now
# --------------------------------------------------------------------------- #
# `is_filter_subject` above owns "this mention is the subject of a predicate",
# and it owns it for NUMERIC predicates only: both its patterns require a
# comparator, a `<>=` symbol or a leading digit, and `condition_cut` needs a
# digit after the opener before it will cut. Measured:
#
#   "how many loans have LTV above 50%"            is_filter_subject -> True
#   "balance where account status is active"       is_filter_subject -> False
#   "balance for interest roll-up loans"           is_filter_subject -> False
#   "show loans in the South East"                 is_filter_subject -> False
#
# The numeric cases are exactly the ones the parser already resolves — four of
# the five fields that ever reach `spec.filters` are numeric bounds on measures.
# So the CATEGORICAL selector had no owner at all, and a question that says
# "where account status is active" was answered over the whole book, split by
# status, with the breakdown certified: 11,035 loans for a narrowed question.
#
# This is that owner, and it is deliberately narrow. The cost of a false positive
# is a refused answer to a question that was fine, which is what 32c263a cost.
#: Words that can introduce a row selector. A superset of CONDITION_OPENERS
#: because a categorical selector uses prepositions a numeric bound does not
#: ("for interest roll-up loans", "of the front book"), and a subset in the other
#: direction because the comparators are numeric-only and live there.
SELECTOR_OPENERS: Tuple[str, ...] = (
    "where", "with", "for", "whose", "having", "of", "in",
    "is", "are", "equals", "equal to",
)

#: Words that mark an AXIS rather than a selector — "balance by broker" is a
#: breakdown, and `Broker` is also a value of `origination_channel`, so the
#: collision is real and the separation matters.
#:
#: There is NO runtime check against these, and that is deliberate. A check was
#: written and then removed because it could never fire: the two vocabularies are
#: DISJOINT, so a mention preceded by an axis marker cannot also be preceded by a
#: selector opener within the window. The invariant that makes the check
#: unnecessary is asserted instead —
#: `test_the_selector_and_axis_vocabularies_are_disjoint` — which fails loudly
#: the moment someone adds "by" to `SELECTOR_OPENERS` and the collision becomes
#: reachable.
#:
#: Recorded this way because of the standing rule earned in D7: a branch that
#: fires zero times is unmeasured, not unused. Here the paths WERE searched, and
#: there is none.
AXIS_MARKERS: Tuple[str, ...] = (
    "by", "per", "across", "split by", "broken down by", "grouped by",
)


def axis_marker_alternation() -> str:
    """A regex alternation of every grouping marker, longest first.

    Item 2 — THE READ OWNER for "where does the grouping clause start?".
    Four consumers were asking it and two hard-coded `\bby\b`:
    `llm_query_parser._grouping_segments`, which SPLITS the sentence, and
    `grouping_cut` below, which returns an OFFSET. `split by`, `broken down by`
    and `grouped by` passed them INCIDENTALLY — those phrases contain the word
    "by" — so only `per` and `across` exposed the gap, and a question grouped
    "across LTV and ticket size" never had its grouping clause cut. The axis
    field then stayed visible to `_detect_metric`, which masks nothing, and was
    read as the MEASURE: a two-dimension breakdown of balance became a
    one-dimension breakdown of LTV, and the substitution guard refused.

    The declared list above was already correct. Nobody read it.

    Longest first so `broken down by` is not shadowed by `by`. Each consumer
    keeps its own implementation — split, region, suffix, offset — because
    those are genuinely four different jobs over one fact.
    """
    return "|".join(re.escape(m) for m in
                    sorted(AXIS_MARKERS, key=len, reverse=True))


#: The marker as a standalone word/phrase. Built from the owner, never retyped.
AXIS_MARKER_RE = re.compile(r"\b(?:" + "|".join(
    re.escape(m) for m in sorted(AXIS_MARKERS, key=len, reverse=True)) + r")\b",
    re.IGNORECASE)

_SELECTOR_BEFORE_RE = re.compile(
    r"\b(?:" + "|".join(sorted((re.escape(t) for t in SELECTOR_OPENERS),
                               key=len, reverse=True)) + r"|=)"
    r"\s+(?:the\s+|a\s+|an\s+|all\s+|our\s+|its\s+)?$", re.I)

#: How far back to look for the opener. Short, so an opener from a different
#: clause cannot claim a mention two phrases away.
SELECTOR_WINDOW = 24


def selector_mark(text: str, start: int, end: int) -> bool:
    """True when the sentence uses the mention at ``[start:end]`` to SELECT rows.

    An opener immediately before it ("for **interest roll-up** loans", "where
    account status is **active**"). "balance by **broker**" is a breakdown and is
    not marked, because `AXIS_MARKERS` and `SELECTOR_OPENERS` are disjoint — the
    invariant, not a second test here.

    This answers only whether the SENTENCE selected. It says nothing about which
    field, which value, or whether anything resolved it — those belong to the
    readers that consume this, not here.
    """
    before = (text or "")[max(0, start - SELECTOR_WINDOW):start]
    return bool(_SELECTOR_BEFORE_RE.search(before))


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
    span = condition_span(head)
    if span is None:
        return head.strip()
    start, end = span
    if head[:start].strip():
        # CONDITION STATED LAST — everything before it is the subject, which is
        # what this has always returned.
        return head[:start].strip()
    # CONDITION STATED FIRST. Truncating at its start leaves nothing, and the
    # old code then handed the WHOLE question to the detector — so "for loans
    # with LTV above 50%, balance by region" resolved its measure to LTV, the
    # field named inside the condition, while the same question with the
    # condition last resolved correctly to balance. The clause is REMOVED
    # instead, leaving the subject on the other side of it.
    remainder = (head[:start] + " " + head[end:]).strip()
    return remainder or head.strip()


# --------------------------------------------------------------------------- #
# Time granularity — the grain a question names.
#
# A question can name a time UNIT without naming a count. "Based on the last few
# weeks" pins no number, so there is no span to honour or fail — but it does pin
# a GRANULARITY, and a series that cannot express weeks cannot answer it. The
# count being vague does not make the unit vague.
#
# This is Stage 5's input: the reading already exists and is correct, and the
# carriage is what is missing.
# --------------------------------------------------------------------------- #

#: Ordered FINEST FIRST, so the finest unit a question names is the one returned.
UNIT_PATTERNS: Tuple[Tuple[str, str], ...] = (
    # `day` was missing until Stage 5's baseline went looking for it. "show me
    # daily funded balance" read as naming NO time unit at all and was answered
    # as a single whole-book KPI — the most complete of the three silent
    # substitutions on the time axis, because no time axis survived to be
    # disclosed.
    #
    # Matched only in GRAIN constructions, never on the bare noun. Drafted as
    # `\bdays?\b` first, it moved five corpus questions and every one was
    # arrears — "more than 90 days in arrears", "number of days in arrears by
    # broker" — where "days" is part of a MEASURE, not a reporting level. Zero
    # were daily-series requests. In this domain the bare noun is overwhelmingly
    # a duration, which is not true of the other units' bare nouns, so `day`
    # needs the construction and they do not.
    #
    # No corpus question exercises this reading in either direction. It is
    # generated coverage under Note 2, and its tests say so.
    ("day", r"\bdaily\b|\bday[- ]by[- ]day\b|\bper day\b|\beach day\b"
            r"|\bby day\b|\bdaily basis\b"),
    ("week", r"\bweeks?\b|\bweekly\b|\bfortnight\b"),
    ("month", r"\bmonths?\b|\bmonthly\b"),
    ("quarter", r"\bquarters?\b|\bquarterly\b"),
    # B11. `year` carried the same defect `day` did — a unit word that names a
    # reporting level in some constructions and an ATTRIBUTE in others. "How
    # many loans are over 80 years old" read as a yearly grain, which put two
    # corpus questions into the B9 series-substitution class that were never
    # asking for a series. Unlike `day`, this one was already live: `day` was
    # caught before it shipped and this was not.
    #
    # Narrower than `day`'s treatment, deliberately. "By year", "year to date"
    # and "annual" are ordinary grain wordings and stay; only the age compounds
    # are excluded, because those are the whole of the observed defect and a
    # wider restriction would drop readings that work.
    ("year", r"\byears?\b(?!\s+(?:old|of age))|\bannual\b|\bannually\b|\bytd\b"),
)

#: Coarsest last, so a request's unit can be compared to a series'.
UNIT_ORDER = {"day": 0, "week": 1, "month": 2, "quarter": 3, "year": 4}

_UNIT_RES = tuple((unit, re.compile(pattern)) for unit, pattern in UNIT_PATTERNS)


def requested_unit(question: str) -> Optional[str]:
    """The finest time UNIT the question names, or None.

    Padded with spaces before matching, as the original did: several patterns
    rely on a word boundary at the string edge.
    """
    text = " %s " % (question or "").lower().strip()
    for unit, pattern in _UNIT_RES:
        if pattern.search(text):
            return unit
    return None


def finer_than(requested: Optional[str], available: str) -> bool:
    """Is the requested unit finer than the one a series can express?"""
    if not requested:
        return False
    return UNIT_ORDER.get(requested, 99) < UNIT_ORDER.get(available, 99)


# --------------------------------------------------------------------------- #
# P0 — "does this sentence ask the answer to VARY OVER TIME?"
# --------------------------------------------------------------------------- #
# THE ONE OWNER of that question, and deliberately a COMPOSITION rather than a
# new vocabulary. Two readings already exist and both are already owned here:
#
#   AXIS_MARKER_RE   where a grouping clause starts ("by", "per", "across",
#                    "split by", "broken down by", "grouped by")
#   requested_unit   what counts as a time unit (day / week / month / quarter /
#                    year), including B11's exclusion of the age compounds
#
# "By month" is the first followed immediately by the second. Writing a fresh
# list of unit words here would have been the twelfth interpreter this contract
# exists to prevent, and it would have gone wrong in a specific, checkable way:
# a hand-written list drafted for this rule included `vintage` and `snapshot`,
# which would have fired on rt_017 ("forecast run rate by vintage") and rt_028
# ("balance by vintage, ignoring the forecast") — two routed-surface answers
# that are correct today. A vintage is a loan ATTRIBUTE, the grouping owner
# already handles it, and `requested_unit` had always said so. Asking the owner
# instead of retyping its vocabulary is what excluded them.
#
# What is genuinely new is the grain-agnostic phrasing: "over time" names no
# unit at all, so no unit vocabulary could ever hold it, and nothing else owned
# it either. That list — and only that list — is declared below.

#: Series wordings that name a time axis WITHOUT naming a grain. No existing
#: vocabulary can hold these, because there is no unit in them to hold.
SERIES_PHRASES: Tuple[str, ...] = (
    "over time", "through time", "time series",
    "over the period", "over the periods",
    "across periods", "across the periods",
    "over successive periods",
    "month by month", "quarter by quarter", "period by period",
    # LENDER VOICE. Each added with a phrasing from the measured banks that
    # needed it — no term here without evidence. See
    # docs/mi_time_axis_vocabulary_prediction.md for the evidence table.
    #
    # "x to x" and "x on x" are the same series request as "x by x", which was
    # already held; only the preposition differs.
    "month to month", "month on month", "month-on-month",
    "quarter to quarter", "quarter on quarter", "quarter-on-quarter",
    "year on year", "year-on-year",
    "period to period", "period on period", "period-on-period",
    # A DISTRIBUTIVE determiner over a time noun names the axis: "balances per
    # region each month". `_AXIS_FILLER_RE` already treats each/every as
    # transparent AFTER an axis marker ("by each month"); these are the same
    # words with no marker in front, which is how a lender writes it.
    "each month", "every month", "each quarter", "every quarter",
    "each period", "every period",
    # "between periods" names two ends of one axis. Kept to the PLURAL: "between
    # January and March" is a span, not a series request, and must not match.
    "between periods", "between the periods", "between reporting periods",
)

_SERIES_PHRASE_RE = re.compile(
    r"\b(?:" + "|".join(re.escape(p) for p in
                        sorted(SERIES_PHRASES, key=len, reverse=True)) + r")\b",
    re.I)

#: Determiners and qualifiers that may sit between an axis marker and its noun
#: without changing what the noun is: "by each month", "by the reporting period".
_AXIS_FILLER_RE = re.compile(r"^(?:the|each|every|a|calendar|governed)\s+", re.I)

#: "period" is a time axis word that is NOT a unit — it names the axis while
#: leaving its grain to the book. `UNIT_PATTERNS` correctly does not carry it
#: (there is no `UNIT_ORDER` position for "whatever the book reports at"), so it
#: is matched separately rather than by widening a vocabulary that means
#: something else.
_PERIOD_NOUN_RE = re.compile(r"^(?:reporting\s+|governed\s+)?periods?\b", re.I)


def time_axis_request(question: Optional[str]) -> Optional[str]:
    """The wording by which this sentence asked the answer to vary over time.

    Returns the matched wording — "by month", "over the periods" — so a refusal
    can quote the reader's own words back, or ``None``.

    Two forms, and the asymmetry between them is the point:

    * a SERIES PHRASE names the axis and no grain ("balance over time");
    * an AXIS MARKER followed by a time unit names both ("balance by month").

    The unit must be the very next word after the marker, allowing only
    determiners. That adjacency is what separates "balance by month" from
    "balance by region for loans under 12 months old" — the second names a
    month, but not as an axis, and reading a unit from anywhere in the sentence
    would have turned a filter into a series request.
    """
    q = str(question or "")
    match = _SERIES_PHRASE_RE.search(q)
    if match:
        return match.group(0)
    for marker in AXIS_MARKER_RE.finditer(q):
        rest = q[marker.end():]
        offset = len(rest) - len(rest.lstrip())
        tail = rest.lstrip()
        while True:
            filler = _AXIS_FILLER_RE.match(tail)
            if not filler:
                break
            offset += filler.end()
            tail = tail[filler.end():]
        period = _PERIOD_NOUN_RE.match(tail)
        if period:
            return q[marker.start():marker.end() + offset + period.end()]
        word = re.match(r"[A-Za-z]+", tail)
        if not word:
            continue
        # THE OWNER IS ASKED ABOUT THE MARKER AND THE NOUN TOGETHER, not about
        # the noun alone. `UNIT_PATTERNS` holds `day` as PHRASES — "by day",
        # "per day", "daily" — and never as the bare word, because "the day the
        # loan completed" is not a reporting grain. Handing it "day" therefore
        # returned None and "balance by day" read as no time axis at all.
        # Handing it the whole candidate span is both correct and the only form
        # that keeps this a composition rather than a second unit vocabulary.
        if requested_unit("%s %s" % (marker.group(0), word.group(0))):
            return q[marker.start():marker.end() + offset + word.end()]
    return None


# --------------------------------------------------------------------------- #
# Clause splitting.
#
# "<age> 70+ with LTV above 50" is two independent thresholds, and a threshold
# must only ever be resolved against its own clause.
#
# The connective list carries one exception that is easy to lose: "and" does NOT
# split when a comparator follows it, because "borrowers 85 and over" is one
# phrase. Splitting it tore the bound off and the filter vanished entirely. The
# splitter decides what a clause IS, so the threshold matcher never has to know
# about the exception.
#
# `consumed` is how a span already claimed by another matcher is excluded
# without destroying the string. A `between A and B` match contains an "and"
# that is not a clause boundary; marking its span consumed is what stops the
# split, and — unlike excising the text — it leaves every offset recoverable.
# --------------------------------------------------------------------------- #

CLAUSE_CONNECTIVES: Tuple[str, ...] = ("and", "with", "where", "whose", "having")

#: Comparators after which an "and" is part of the phrase, not a boundary.
CLAUSE_AND_EXCEPTIONS: Tuple[str, ...] = (
    "over", "above", "older", "under", "below", "younger", "more", "less")

#: PUNCTUATION IS A CLAUSE BOUNDARY, and its absence was a defect.
#:
#: The connectives alone could not end a clause, so a filter stated FIRST ran to
#: the end of the sentence: "for loans with LTV above 50%, balance by region"
#: split at "with" and produced one clause — " ltv above 50%, balance by region"
#: — which swallowed the measure and the dimension. The field was then resolved
#: from the swallowed words, so the predicate landed on `balance` instead of
#: `ltv`, and "borrower age above 70, balance by region" produced a threshold of
#: seventy BILLION. Stated last, the same filter parsed correctly, because there
#: the clause genuinely does run to the end.
#:
#: A comma with digits on BOTH sides is a thousands separator, not a boundary,
#: so "over 1,500,000" stays one number. Guarding on either side alone was
#: wrong and measured wrong: "above 70, balance by region" has a digit before
#: the comma, so a lookbehind-only guard refused to split exactly the clause
#: this fix exists to end.
_CLAUSE_PUNCTUATION = r"(?:(?<![0-9])[,;]|[,;](?![0-9]))"

_CLAUSE_SPLIT_RE = re.compile(
    r"\band\b(?!\s+(?:" + "|".join(CLAUSE_AND_EXCEPTIONS) + r")\b)"
    r"|" + "|".join(r"\b%s\b" % c for c in CLAUSE_CONNECTIVES if c != "and")
    + r"|" + _CLAUSE_PUNCTUATION)


def _overlaps_any(start: int, end: int,
                  ranges: "Tuple[Tuple[int, int], ...]") -> bool:
    return any(not (end <= a or start >= b) for a, b in ranges)


def blank_consumed(text: str, start: int, end: int,
                   consumed: "Tuple[Tuple[int, int], ...]" = ()) -> str:
    """``text[start:end]`` with any consumed sub-range replaced by one space.

    One space, not nothing: it is what keeps the words either side of a claimed
    span from running together into a token neither of them is.
    """
    out = []
    cursor = start
    for a, b in sorted(consumed):
        a, b = max(a, start), min(b, end)
        if a >= b:
            continue
        out.append(text[cursor:a])
        out.append(" ")
        cursor = b
    out.append(text[cursor:end])
    return "".join(out)


def clause_spans(text: str,
                 consumed: "Tuple[Tuple[int, int], ...]" = ()
                 ) -> "list":
    """``[(start, end)]`` of each clause, over the ORIGINAL text.

    Splits on the connectives, ignoring any that falls inside a consumed span.
    Returns spans rather than strings so a caller can say WHERE a filter came
    from — which is the parser half of the filter join.
    """
    text = text or ""
    consumed = tuple(consumed or ())
    spans = []
    cursor = 0
    for match in _CLAUSE_SPLIT_RE.finditer(text):
        if _overlaps_any(match.start(), match.end(), consumed):
            continue
        spans.append((cursor, match.start()))
        cursor = match.end()
    spans.append((cursor, len(text)))
    return spans


# --------------------------------------------------------------------------- #
# THE COMPARATOR VOCABULARY — one fact about English, two consumers
# --------------------------------------------------------------------------- #
#: Item 1. "Is this phrase a comparator, and in which direction?" had TWO
#: owners with different word lists: `llm_query_parser._FILTER_COMPARATORS`,
#: which builds the predicate that narrows rows, and
#: `execution_receipt._THRESHOLD_PATTERNS`, which records that the SENTENCE
#: asked for a narrowing. They agreed on 16 of 30 phrases. Where both were
#: blind — `bigger than`, `larger than`, `higher than`, `smaller than`,
#: `lower than` — the narrowing vanished, no facet was raised, the
#: honour-or-clarify guard had nothing to honour, and the whole book came back
#: as fact: 43.15% weighted LTV over 11,035 loans for a question about the
#: 5,857 loans over £150k.
#:
#: WHAT IS SHARED IS THE VOCABULARY, NOT THE OWNER, and that is deliberate.
#: The two consumers must keep detecting INDEPENDENTLY: if the receipt derived
#: its facet from the parser's output, a threshold the parser missed would
#: never be raised and the guard could never catch it. That independence is
#: what makes `exceeding`, `in excess of`, `minimum of`, `beneath`, `up to`,
#: `maximum of` and `capped at` refuse today rather than answer wrongly — the
#: receipt sees a threshold the parser does not. Collapsing them into one owner
#: would convert those seven from safe to silent.
#:
#: This is the INVERSE of `portfolio_lens._qualified_span_re`, which shares an
#: implementation across genuinely different vocabularies. The precedent is not
#: "always parameterise the implementation" — it is SHARE WHAT IS ONE FACT AND
#: SEPARATE WHAT IS TWO. There, hard-coding one noun list dropped five governed
#: phrases; here, keeping two comparator lists dropped five comparators.
#:
#: Ordering is load-bearing: longest phrase first, so `greater than or equal
#: to` is not shadowed by `greater than`, and `no more than` is not read as
#: `more than` with the negation discarded — which would invert the filter.

#: op -> the word a receipt uses for it. The receipt renders the OPERATOR; it
#: does not keep a second list of phrases mapping to words.
COMPARATOR_WORD = {
    "gt": "over", "ge": "at least", "lt": "under", "le": "at most",
    "eq": "exactly", "between": "between",
}

#: (phrase, op). THE list. Sorted longest-first at import so a caller building
#: an alternation cannot reintroduce the shadowing bug by reordering.
COMPARATOR_PHRASES: "Tuple[Tuple[str, str], ...]" = tuple(sorted(
    (
        # --- between ---------------------------------------------------- #
        ("between", "between"),
        # --- >= : the negated and explicit forms MUST precede `gt` ------- #
        ("greater than or equal to", "ge"), ("no less than", "ge"),
        ("not less than", "ge"), ("at least", "ge"), ("minimum of", "ge"),
        ("a minimum of", "ge"), ("or above", "ge"), ("or over", "ge"),
        ("or more", "ge"), ("or older", "ge"), ("or greater", "ge"),
        # --- <= : likewise before `lt` ----------------------------------- #
        ("less than or equal to", "le"), ("no more than", "le"),
        ("not more than", "le"), ("at most", "le"), ("maximum of", "le"),
        ("a maximum of", "le"), ("capped at", "le"), ("up to", "le"),
        ("or below", "le"), ("or under", "le"), ("or less", "le"),
        ("or younger", "le"), ("or fewer", "le"),
        # --- > ------------------------------------------------------------ #
        ("greater than", "gt"), ("more than", "gt"), ("bigger than", "gt"),
        ("larger than", "gt"), ("higher than", "gt"), ("older than", "gt"),
        ("longer than", "gt"), ("in excess of", "gt"), ("exceeding", "gt"),
        ("exceeds", "gt"), ("over", "gt"), ("above", "gt"),
        # --- < ------------------------------------------------------------ #
        ("less than", "lt"), ("fewer than", "lt"), ("smaller than", "lt"),
        ("lower than", "lt"), ("younger than", "lt"), ("shorter than", "lt"),
        ("under", "lt"), ("below", "lt"), ("beneath", "lt"),
        # --- = ------------------------------------------------------------ #
        ("equal to", "eq"), ("equals", "eq"), ("exactly", "eq"),
    ),
    key=lambda pair: (-len(pair[0]), pair[0])))


def comparator_alternation(ops: "Tuple[str, ...]" = ()) -> str:
    """A regex alternation of every phrase meaning one of ``ops``.

    Longest-first, always. A caller that wants only the `gt` phrases passes
    ``("gt",)``; a caller that wants all of them passes nothing. Both consumers
    build their own pattern around this — the parser needs a captured value and
    a suffix, the receipt needs a number and a span — but neither keeps its own
    idea of which words are comparators.
    """
    wanted = tuple(ops or ())
    phrases = [p for p, op in COMPARATOR_PHRASES if not wanted or op in wanted]
    return "|".join(re.escape(p) for p in phrases)


def comparator_ops() -> "Tuple[str, ...]":
    """Every operator the vocabulary can express, in a stable order."""
    return tuple(dict.fromkeys(op for _, op in COMPARATOR_PHRASES))


# --------------------------------------------------------------------------- #
# THE THRESHOLD SUBJECT — which noun a threshold is on
# --------------------------------------------------------------------------- #
#: Item 3. "Which field is this threshold on?" had TWO owners with near-identical
#: vocabularies and DIFFERENT RULES:
#:
#:   llm_query_parser._filter_field_of    -> a field key, by the subject NEAREST
#:                                           BEFORE the comparator
#:   execution_receipt._threshold_subject -> a display name, by the FIRST entry
#:                                           in a fixed priority list
#:
#: They disagree whenever a measure is named earlier in the sentence than the
#: threshold's own noun. "What is the LTV for loans with a balance above
#: £150,000" bound `current_outstanding_balance` and disclosed "LTV over
#: 150000" — the receipt naming a field execution did not filter. Three of eight
#: probed sentences did that.
#:
#: And the same decision, never asked, is why "the LTV for loan TICKETS above
#: £150k" refused: `ticket` names a registry dimension, `dimension_role` had no
#: source for "this word is the subject of a threshold", so the role fell to
#: UNRESOLVED and the guard clarified — over a question whose measure, filter
#: and 5,857-loan population were all already resolved.
#:
#: ONE FACT: which noun the threshold is on. Proximity to the comparator, which
#: is the rule `_filter_field_of` documents and item 1 hardened. TWO: the
#: renderings — a field key is not a display name, exactly as an operator was not
#: a receipt word in item 1.
#:
#: This owner returns the KIND. Each consumer renders it.
THRESHOLD_SUBJECT_PATTERNS: "Tuple[Tuple[str, str], ...]" = (
    (r"\bltv\b|\bloan[- ]to[- ]value\b", "ltv"),
    (r"\b(?:age|aged|youngest|borrowers?|years?|yrs?|yo|year[- ]?old|older|"
     r"younger)\b", "age"),
    (r"\brate\b|\binterest\b|\bcoupon\b", "rate"),
    (r"\bbalance\b|\boutstanding\b|\bexposure\b|\bloan size\b|\bticket\b|"
     r"\btickets\b", "balance"),
    (r"\bvaluation\b|\bproperty value\b|\bcollateral\b", "valuation"),
)

#: kind -> the word a receipt uses for it. The receipt renders the KIND; it does
#: not keep a second list of patterns mapping to names.
THRESHOLD_SUBJECT_WORD = {
    "ltv": "LTV", "age": "borrower age", "rate": "interest rate",
    "balance": "balance", "valuation": "valuation",
}


def threshold_subject_kind(text: "Optional[str]",
                           anchor: "Optional[int]" = None,
                           value_end: "Optional[int]" = None) -> "Optional[str]":
    """The kind of field a threshold is on, or ``None``.

    ``anchor`` is the comparator's offset. When given, THE SUBJECT NEAREST
    BEFORE IT WINS — which is what a reader does and what the predicate means.
    Without it the whole text is searched and the nearest match to the end wins,
    which is the same rule with the end of the text as the anchor.

    Priority order is NOT used to break ties, and that is the correction: the
    receipt used to return the first entry of an ordered list, so "the LTV for
    loans with a balance above £150,000" disclosed a threshold on LTV while
    execution filtered the balance.
    """
    if not text:
        return None
    text = str(text)

    # POSTFIX BINDS TIGHTEST. "above 50% LTV" names its subject AFTER the value,
    # and that subject is the threshold's, not whatever was mentioned earlier.
    # `_filter_field_of` has always had this rule; the first version of this
    # owner shared the vocabulary and implemented only the nearest-BEFORE half,
    # which silently dropped the subject from "What percentage of the book is
    # above 50% LTV?" — the facet label went from "LTV over 50" to "over 50".
    #
    # One corpus answer moved and that is how it was found. Item 1's rule again:
    # a consolidation is complete when the consumers have been exercised across
    # the full range, and prefix-only was not the full range.
    if value_end is not None:
        tail = text[value_end:value_end + 28]
        for pattern, kind in THRESHOLD_SUBJECT_PATTERNS:
            if re.search(pattern, tail, re.IGNORECASE):
                return kind

    head = text[:anchor] if anchor is not None else text
    best = None
    for pattern, kind in THRESHOLD_SUBJECT_PATTERNS:
        for match in re.finditer(pattern, head, re.IGNORECASE):
            if best is None or match.start() > best[0]:
                best = (match.start(), kind)
    return best[1] if best else None


# --------------------------------------------------------------------------- #
# THE PIPELINE STAGE AXIS — a governed concept with no reader until now
# --------------------------------------------------------------------------- #
#: The governed field key. `config/mi/pipeline_field_contract.yaml` already
#: declares `pipeline_stage` with `role: dimension` and
#: `semantic_registry_field: pipeline_stage`, and
#: `config/mi/stratification_catalogue.yaml` already declares it categorical over
#: `total_pipeline`. The dimension was governed on the DATA side the whole time;
#: what did not exist was any way for a QUESTION to name it, which is why
#: `_route_evolution` re-read the raw sentence to choose its sub-route.
PIPELINE_STAGE_FIELD = "pipeline_stage"

#: Matches the stage AXIS itself ("by stage", "stage migration", "stage
#: balances"). Whole-word, so "stagecoach" and "staged" do not qualify.
_STAGE_AXIS_RE = re.compile(r"\bstages?\b", re.IGNORECASE)

_STAGE_VOCAB_CACHE: "Optional[Dict[str, str]]" = None


def pipeline_stage_vocabulary() -> "Dict[str, str]":
    """Question-side spellings of the governed stages -> canonical stage.

    Derived from the ONE authoritative normalisation map,
    `pipeline_prep._STAGE_CANON`, never redeclared here. Two adjustments, both
    rules rather than hand-lists:

    A data-value normalisation map is not a question vocabulary. `_STAGE_CANON`
    maps ``"funded" -> COMPLETED``, which is right for a tape cell and wrong for
    a sentence, where *funded* names the governed DATASET. Every spelling that
    collides with a governed view name is therefore dropped, read from the view
    registry so a newly registered view cannot silently reintroduce the clash.
    Without this, *"Show funded balance evolution by month"* — the most ordinary
    question in the corpus — would acquire a COMPLETED stage.

    The import is deferred because `question_interpretation` is imported by the
    parser that `mi_agent_api` itself imports; at module scope this is a cycle.
    """
    global _STAGE_VOCAB_CACHE
    if _STAGE_VOCAB_CACHE is not None:
        return _STAGE_VOCAB_CACHE
    from mi_agent_api.pipeline_prep import _STAGE_CANON
    from mi_agent_api.workspace import VIEWS
    collides = {str(v).lower() for v in (VIEWS or ())}
    kept = {k: v for k, v in _STAGE_CANON.items() if k.lower() not in collides}

    # A truncation of a longer spelling for the SAME stage is a word fragment,
    # not a stage noun, and fragments are where the false positives live:
    # `complete` (a prefix of `completed`/`completion`) matched *"How complete is
    # interest rate?"* — five corpus questions about DATA COMPLETENESS acquiring a
    # COMPLETED stage — and `app` (a prefix of `application`) is the same shape.
    # Dropped by that rule rather than by a hand-list, so the exclusion cannot
    # drift from the map it is derived from. The canonical token itself is always
    # kept: `offer` is a prefix of `offer issued` and IS the stage.
    _STAGE_VOCAB_CACHE = {
        k: v for k, v in kept.items()
        if k.lower() == v.lower()
        or not any(o != k and o.startswith(k) and kept[o] == v for o in kept)
    }
    return _STAGE_VOCAB_CACHE


def canonical_pipeline_stages() -> "Tuple[str, ...]":
    """The governed stage set, in funnel order, from the governed bucket map."""
    from mi_agent_api.pipeline_prep import _STAGE_BUCKET
    order = {"early": 0, "mid": 1, "late": 2, "completed": 3, "withdrawn": 4}
    return tuple(sorted(_STAGE_BUCKET, key=lambda s: order.get(_STAGE_BUCKET[s], 9)))


def pipeline_stage_request(question: "Optional[str]"
                           ) -> "Tuple[Optional[str], bool]":
    """What this question says about the pipeline stage axis.

    Returns ``(canonical_stage, names_the_axis)``:

      ``canonical_stage`` a specific governed stage the question names, or None
      ``names_the_axis``  whether it names the stage DIMENSION ("by stage")

    THE ONE PLACE A STAGE IS READ FROM A QUESTION. `_route_evolution` currently
    holds two more — a membership test against `_FUNNEL_KEYWORDS` and a
    substring test against three hard-coded phrases — and those are the duplicate
    owners this exists to retire. It is deliberately a reader and not a policy:
    it says what the sentence names, and leaves to the projection what role that
    plays.

    Both halves are needed and they are not the same question. *"Show pipeline
    amount by stage over time"* names the axis and no stage; *"Show the KFI
    trend"* names a stage and not the axis; *"How have offer-stage cases
    changed?"* names both, and a specific stage beats the axis because the
    question asks about one stage rather than for a split across all of them.

    Longest spelling first, so ``offer issued`` is not read as ``offer``, and
    ``undisclaimed_mention`` so a stage every occurrence of which the sentence
    rules out does not select — the same bar the dataset owner already uses.
    """
    text = str(question or "")
    if not text:
        return None, False
    from mi_agent.portfolio_lens import undisclaimed_mention

    # Funnel order, so a sentence naming two stages resolves to the EARLIER one
    # — *"How much is sitting at offer today and how much will complete?"* is a
    # question about the offer stage with a completion verb in it, and the
    # shipped handler already resolves it that way. The order is read from the
    # governed bucket map, not asserted here.
    order = {st: i for i, st in enumerate(canonical_pipeline_stages())}
    stage: "Optional[str]" = None
    vocab = pipeline_stage_vocabulary()
    for spelling in sorted(vocab, key=lambda k: (order.get(vocab[k], 9), -len(k))):
        # Plural allowed on the trailing word: "How many KFIs have we issued?"
        # names a stage, and a reader that missed it would UNDER-reach where the
        # handler it replaces reaches, which is the one direction a replacement
        # may not fail in.
        if not re.search(r"\b%s(?:e?s)?\b" % re.escape(spelling), text,
                         re.IGNORECASE):
            continue
        if not undisclaimed_mention(text, spelling):
            continue
        stage = vocab[spelling]
        break

    # The bare word "stage" is ordinary English — *"What stage is the
    # securitisation at?"* is not a pipeline question — so unlike a canonical
    # stage NAME it is not self-evidencing. It names the axis only where the
    # sentence puts it in an axis position ("by stage", "per stage"), or where
    # the governed dataset owner has already decided this is a pipeline
    # question. That is the asymmetry: a stage name is its own evidence, the
    # word "stage" needs some.
    axis = False
    match = _STAGE_AXIS_RE.search(text)
    if match:
        before = text[max(0, match.start() - SELECTOR_WINDOW):match.start()]
        if AXIS_MARKER_RE.search(before):
            axis = True
        else:
            from mi_agent_api.workspace import resolve_dataset
            axis = resolve_dataset(text) == "pipeline"
    return stage, axis


# --------------------------------------------------------------------------- #
# LEVEL versus MOVEMENT — the single owner
# --------------------------------------------------------------------------- #
# THE QUESTION THIS ANSWERS, and nothing else:
#
#     Is the quantity asked for a LEVEL at one point in time, or a CHANGE
#     between two points in time?
#
# It lives here because it is a reading of the QUESTION's vocabulary, which is
# what this module owns, and because it must not live in any route: the estate
# had FIVE components inferring it independently and they disagreed on 30 of 882
# corpus questions, with no reader a superset of any other —
#
#     A  period_change.recognition.has_change_language      17
#     B  llm_query_parser._COMPARE_TRIGGER_RE               21
#     C  spec.temporal_mode == "compare"                     5
#     D  interpreter.deterministic's compare branch         20
#     E  concentration_query's compare_concentration gate    0
#     union                                                 30
#
# — and each missed part of the union (A 13, B 9, C 25, D 10, E 30). Reader A
# missed "How did the balance change since last month?", the most canonical
# movement question in the estate, because CHANGE_MARKERS carried "changed",
# "change in" and "has changed" but not the bare verb.
#
# TWO THINGS THAT ARE NOT A MOVEMENT, and the reason each is excluded:
#
#   comparing two POPULATIONS  "how does the front book compare with our older
#                              lending" contrasts two slices of one snapshot.
#                              Reader B called it a movement because its trigger
#                              is a bare "compare" with no period requirement.
#   comparing against a PLAN   "compare current funded balance to expected",
#                              "the largest concentration versus limit" — the
#                              second operand is a forecast or a threshold, not
#                              an earlier date.
#
# So every construction below requires a PERIOD on both sides, or an explicit
# change verb. A bare "compare" is not evidence of anything temporal.
#
# A SERIES IS NOT A TWO-POINT MOVEMENT either. "balance by month" is a sequence
# of levels; the estate already treats it as a separate decline reason
# (DECLINE_TREND_SERIES) and folding it in here would claim a movement wherever
# a time axis appears.

LEVEL = "level"
MOVEMENT = "movement"
TEMPORAL_ASPECTS = (LEVEL, MOVEMENT)

#: Verbs and nouns that name a change. Superset of the estate's readers: the
#: bare conjugations `change`, `changing`, `declined`, `dropped` and `growing`
#: were missing from the widest of them.
CHANGE_WORDS: Tuple[str, ...] = (
    # "added"/"gained"/"lost" were MISSING, and the omission was not cosmetic:
    # "which region added the most balance?" read as a LEVEL, so a question
    # about a movement was answered with a position. That is defect D2
    # reoccurring through the owner that exists to prevent it.
    "added", "add", "adds", "adding", "gained", "gain", "gains", "lost",
    "change", "changed", "changes", "changing",
    "movement", "movements", "moved", "moves", "move",
    "increase", "increased", "increasing", "decrease", "decreased", "decreasing",
    "grew", "grow", "grown", "growing", "growth",
    "declined", "decline", "declining", "dropped", "drop", "dropping",
    "shrank", "shrunk", "shrinking", "rose", "risen", "rising",
    "fell", "fallen", "falling",
    "improved", "improvement", "deteriorated", "deterioration",
    "worsened", "shifted", "shift",
)
_CHANGE_WORD_RE = re.compile(
    r"\b(?:" + "|".join(sorted(CHANGE_WORDS, key=len, reverse=True)) + r")\b",
    re.IGNORECASE)

#: Phrases that name a period-over-period comparison outright.
COMPARISON_PERIOD_WORDS: Tuple[str, ...] = (
    "month on month", "month-on-month", " mom ", "quarter on quarter",
    "quarter-on-quarter", " qoq ", "year on year", "year-on-year", " yoy ",
    "year to date", "year-to-date", " ytd ",
    "current versus previous", "current vs previous",
    "versus the previous", "versus the prior", "vs the previous", "vs the prior",
    "with the previous", "with the prior",
    "since the previous", "since the prior", "since the last",
)

#: What counts as naming a period. A construction only reads as temporal when a
#: period stands on BOTH sides of it — that is the whole difference between
#: comparing two dates and comparing two books.
_PERIOD = (r"(?:\d{4}-\d{2}(?:-\d{2})?|\d{4}|q[1-4]|h[12]|"
           r"jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|"
           r"jul(?:y)?|aug(?:ust)?|sep(?:tember)?|oct(?:ober)?|nov(?:ember)?|"
           r"dec(?:ember)?|"
           r"(?:last|previous|prior|next|this|current|latest)\s+"
           r"(?:day|week|month|quarter|year|run|pipeline|period)|"
           # Relative recency. A reader says "now compared with a few months
           # ago" and means two dates as plainly as "October versus November";
           # without these the owner read that as a level.
           r"(?:a\s+few\s+|\d+\s+|a\s+|several\s+)?"
           r"(?:days?|weeks?|months?|quarters?|years?)\s+ago|"
           r"earlier\s+(?:in\s+the\s+)?(?:year|quarter|month|period)|"
           r"latest|today|yesterday|now|recent(?:ly)?)")

#: Seasoning SEGMENTS are not periods. "the front book", "our seasoned loans"
#: and "the back book" name cohorts of the CURRENT book, so a question
#: contrasting two of them is cross-sectional however comparative its wording.
#: The estate has a `seasoning` owner for exactly that axis, and the difference
#: between it and this one is the difference between "what we hold" and "what
#: moved".

_BETWEEN_PERIODS_RE = re.compile(
    rf"\bbetween\s+{_PERIOD}\s+and\s+{_PERIOD}", re.IGNORECASE)
_FROM_TO_PERIODS_RE = re.compile(
    rf"\bfrom\s+{_PERIOD}\s+(?:to|until|through)\s+{_PERIOD}", re.IGNORECASE)
_PERIOD_VERSUS_RE = re.compile(
    rf"{_PERIOD}\s+(?:versus|vs\.?|against|compared\s+(?:to|with))\s+"
    rf"{_PERIOD}", re.IGNORECASE)
#: "compare October AND November". A bare `and` between two periods is NOT
#: enough on its own — "show pipeline by stage for October and November" asks
#: for two LEVELS side by side, not their difference — so the connective only
#: counts when an explicit comparison verb governs it.
_COMPARE_PERIODS_RE = re.compile(
    rf"\bcompar\w*\b[^.?!]{{0,40}}?{_PERIOD}\s+(?:and|with|to|versus|vs\.?)\s+"
    rf"{_PERIOD}", re.IGNORECASE)
#: Two DISTINCT periods anywhere, joined by a comparison word. The shape
#: "how does recent lending compare with what we were originating earlier in the
#: year" puts the periods too far from the connective for the patterns above.
_COMPARE_WORD_RE = re.compile(
    r"\b(?:compare[ds]?|comparing|versus|vs\.?|against|different|difference)\b",
    re.IGNORECASE)
_PERIOD_TOKEN_RE = re.compile(_PERIOD, re.IGNORECASE)
_SINCE_PERIOD_RE = re.compile(rf"\bsince\s+{_PERIOD}", re.IGNORECASE)
#: "how did/has X change" — the shape reader A missed entirely.
_HOW_CHANGED_RE = re.compile(
    r"\bhow\s+(?:did|has|have|is|are)\b.{0,60}?\b(?:chang|mov|grow|grew|"
    r"declin|f[ae]ll|drop|increas|decreas|shift)", re.IGNORECASE)


@dataclass(frozen=True)
class TemporalAspect:
    """Whether the question asks for a level or a change, and what said so."""

    verdict: str
    evidence: Tuple[str, ...] = ()

    @property
    def is_movement(self) -> bool:
        return self.verdict == MOVEMENT


def temporal_aspect(question: str) -> TemporalAspect:
    """LEVEL or MOVEMENT, with the signals that decided it.

    MOVEMENT requires POSITIVE evidence. Absence of evidence is LEVEL, because
    a question that names no change is asking what the position is — which is
    what every route in the estate already assumes, and saying so explicitly is
    what lets a consumer stop guessing.

    The evidence is returned so a receipt can show WHY a question was read as a
    change. A verdict with no evidence is a level, and that is checkable.
    """
    text = f" {str(question or '').strip().lower()} "
    evidence: list = []
    if _CHANGE_WORD_RE.search(text):
        evidence.append("change_word")
    if any(w in text for w in COMPARISON_PERIOD_WORDS):
        evidence.append("comparison_period_phrase")
    if _BETWEEN_PERIODS_RE.search(text):
        evidence.append("between_periods")
    if _FROM_TO_PERIODS_RE.search(text):
        evidence.append("from_period_to_period")
    if _PERIOD_VERSUS_RE.search(text):
        evidence.append("period_versus_period")
    if _COMPARE_PERIODS_RE.search(text):
        evidence.append("compare_period_and_period")
    if (_COMPARE_WORD_RE.search(text)
            and len({m.group(0).strip().lower()
                     for m in _PERIOD_TOKEN_RE.finditer(text)}) >= 2):
        evidence.append("two_periods_and_a_comparison")
    if _SINCE_PERIOD_RE.search(text):
        evidence.append("since_period")
    if _HOW_CHANGED_RE.search(text):
        evidence.append("how_did_it_change")
    return TemporalAspect(MOVEMENT if evidence else LEVEL, tuple(evidence))


def is_movement_question(question: str) -> bool:
    """The boolean form, for a caller that does not need the evidence."""
    return temporal_aspect(question).is_movement


# --------------------------------------------------------------------------- #
# ORDERING — the single owner of direction, basis and limit
# --------------------------------------------------------------------------- #
# The same argument as LEVEL versus MOVEMENT above, and the same remedy. The
# vocabulary lived in `mi_agent.period_change.rank_request`, inside a ROUTE
# package, and the contract had to import a route to learn what the reader
# asked to order by. It also required a resolved dimension term before it would
# answer at all, so a question that named an ordering but no dimension carried
# no ordering on the contract — 15 of 97 ranking questions could not be planned
# for that reason alone.
#
# Direction, basis and limit are properties of the QUESTION, not of the
# dimension, so they are read here and the dimension is bound separately.

ORDER_LIMIT_WORDS = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
    "seven": 7, "eight": 8, "nine": 9, "ten": 10, "twenty": 20,
}
#: The rank words that can govern a count. `bottom`/`worst`/`smallest` carry a
#: DIRECTION as well, which is why they appear in both maps.
_TOP_WORDS = r"top|first|best|largest|biggest|greatest|highest|leading"
#: `last` is NOT here. It matched "since last month" and made every
#: month-relative question read as a bottom-N ranking with a decreasing
#: direction — the ordering owner inventing an ordering out of a period.
_BOTTOM_WORDS = r"bottom|worst|smallest|lowest|least"
_ORDER_WORDS = f"{_TOP_WORDS}|{_BOTTOM_WORDS}"
_WORDNUM = "|".join(ORDER_LIMIT_WORDS)

#: How many results the reader asked for. Deliberately bounded shapes, so a
#: number meaning something else — "the 3 months to June", "over 85", "LTV above
#: 50%" — is never read as a limit.
#:
#:     "top 3", "bottom five"       a rank word immediately before a number
#:     "three largest"              a number immediately before a rank word
#:     "which two regions"          "which" + a number opening the question
_ORDER_LIMIT_RE = re.compile(
    rf"\b(?:{_ORDER_WORDS})\s+(?P<a_digits>\d+)\b|"
    rf"\b(?:{_ORDER_WORDS})\s+(?P<a_word>{_WORDNUM})\b|"
    rf"\b(?P<b_word>{_WORDNUM})\s+(?:{_ORDER_WORDS})\b|"
    rf"\bwhich\s+(?P<c_word>{_WORDNUM})\b|"
    rf"\bwhich\s+(?P<c_digits>\d+)\b", re.I)

#: DIRECTION COMES FROM THE VERB, NOT FROM THE SUPERLATIVE.
#:
#: `most`, `largest`, `top` are RANK words: they say an ordering was asked for,
#: and say nothing about which way. Putting them in the increase set made
#: "which region saw the LARGEST FALL" read as increase-and-decrease-at-once,
#: which resolves to `either` — an ordering by magnitude that would have ranked
#: a riser top of a question about falls.
_ORDER_DECREASE_RE = re.compile(
    rf"\b(?:{_BOTTOM_WORDS})\b|\b(?:declin\w*|f[ae]ll\w*|drop\w*|decreas\w*|"
    rf"shr(?:ank|unk|ink\w*)|reduc\w*|lost|los(?:e|ing)|down)\b", re.I)
_ORDER_INCREASE_RE = re.compile(
    r"\b(?:grew|grow|grown|growth|increas\w*|ris(?:e|en|ing)|gain\w*|"
    r"expand\w*|added|add|up)\b", re.I)
_ORDER_ANY_RE = re.compile(r"\bmovement\b|\bmoved\b|\bchang(?:e|ed|es)\b", re.I)

_ORDER_SHARE_RE = re.compile(
    r"\bshare\b|\bproportion\b|\bcomposition\b|\bmix\b", re.I)
_ORDER_PERCENT_RE = re.compile(
    r"\bfastest\b|\bin percentage terms\b|\bpercentage\b|\bpercent\b|\b%\b|"
    r"\brelative growth\b", re.I)
_ORDER_COUNT_RE = re.compile(
    r"\b(?:loan|loans|case|cases|account|accounts|number of)\b", re.I)
_ORDER_AMOUNT_RE = re.compile(
    r"\bbalance\b|\bexposure\b|\bbook\b|\bvalue\b|£", re.I)

#: A superlative or explicit rank instruction. Without one, no ordering.
_ORDER_REQUESTED_RE = re.compile(
    rf"\b(?:most|least|{_ORDER_WORDS}|fastest|rank|order)\b", re.I)


@dataclass(frozen=True)
class OrderingRequest:
    """What the question asked to order by, if anything."""

    requested: bool
    direction: Optional[str] = None
    basis: Optional[str] = None
    limit: Optional[int] = None


def ordering_limit(question: str) -> Optional[int]:
    """How many results the question asked for, or None.

    "which TWO regions", "top 3", "bottom five". The word map used to stop at
    three|four|five|ten, so "which two regions grew the most" carried no limit
    and the planner ranked every riser — silently wrong rather than refused.
    """
    match = _ORDER_LIMIT_RE.search(f" {str(question or '').strip().lower()} ")
    if not match:
        return None
    digits = match.group("a_digits") or match.group("c_digits")
    if digits:
        try:
            value = int(digits)
        except ValueError:
            return None
        return value if value > 0 else None
    word = (match.group("a_word") or match.group("b_word")
            or match.group("c_word") or "")
    return ORDER_LIMIT_WORDS.get(word.lower())


def ordering_request(question: str) -> OrderingRequest:
    """Direction, basis and limit — WITHOUT needing a resolved dimension.

    Requiring one was the old reader's mistake: a question orders by a
    direction and a basis whether or not the dimension it names resolves, and
    withholding the ordering because the dimension did not resolve loses two
    facts to explain one.
    """
    text = f" {str(question or '').strip().lower()} "
    if not _ORDER_REQUESTED_RE.search(text):
        return OrderingRequest(requested=False)

    down = bool(_ORDER_DECREASE_RE.search(text))
    up = bool(_ORDER_INCREASE_RE.search(text))
    if down and not up:
        direction = ORDER_DECREASE
    elif up and not down:
        direction = ORDER_INCREASE
    elif _ORDER_ANY_RE.search(text) or (up and down):
        # Both named, or the question says "movement" — order by magnitude
        # rather than guess which the reader meant.
        direction = ORDER_EITHER
    else:
        # A bare rank instruction with no verb is a descending ranking.
        direction = ORDER_INCREASE

    if _ORDER_SHARE_RE.search(text):
        basis = ORDER_BASIS_SHARE
    elif _ORDER_PERCENT_RE.search(text):
        basis = ORDER_BASIS_PERCENT
    elif _ORDER_COUNT_RE.search(text) and not _ORDER_AMOUNT_RE.search(text):
        basis = ORDER_BASIS_COUNT
    else:
        basis = ORDER_BASIS_ABSOLUTE

    return OrderingRequest(requested=True, direction=direction, basis=basis,
                           limit=ordering_limit(question))
