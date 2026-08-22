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
    cut = condition_cut(head)
    if cut is not None and head[:cut].strip():
        return head[:cut].strip()
    return head.strip()


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

_CLAUSE_SPLIT_RE = re.compile(
    r"\band\b(?!\s+(?:" + "|".join(CLAUSE_AND_EXCEPTIONS) + r")\b)"
    r"|" + "|".join(r"\b%s\b" % c for c in CLAUSE_CONNECTIVES if c != "and"))


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
