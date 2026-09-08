#!/usr/bin/env python3
"""Why "above 6%" filtered and "7% or more" did not.

THE OBSERVATION, from the 100-question atomic perimeter run. Two questions that
differ only in how the reader wrote the same bound:

    "How many funded loans have an interest rate above 6%?"   → count, filtered
    "How many funded loans have an interest rate of 7% or more?"
        → *"Weighted-average Interest Rate: 6.9% · 10 loans · entire pipeline"*

The second is the worst class of failure in that bank: it answered confidently,
about the whole book, with a measure nobody asked for, and said nothing. F032 on
the funded frame and P030 on the pipeline frame are the same sentence.

THE CAUSE IS ONE CONCEPT WITH TWO OWNERS, which is the shape of nearly every
defect this estate has found. A finance value has a governed grammar — `_VALUE`
plus `_amount()` — that accepts a currency prefix, thousands commas, a k/m/bn
multiplier and a trailing percent, and it is what the PREFIX comparators
("above 6%", "under £200k") are parsed with. The POSTFIX comparators ("70+",
"7% or more") carried a second, private number grammar:

    (-?\\d+(?:\\.\\d+)?)\\s*(?:years?|yrs?)?\\s*(?:\\+|\\bor (?:above|over|…)\\b)

— bare digits, no `%`, no currency, no multiplier, and `years?` hard-coded as
the only unit a number may wear. So the age questions it was written for all
worked (20/20 in that bank), and every postfix bound on money or a rate was
invisible to it.

INVISIBLE IS THE OPERATIVE WORD, and it is why this is silent rather than
refused. The facet guard reports a requested facet that could not be applied —
it is what makes "total pipeline amount for cases with a rate above 6%" decline
honestly. Here nothing recognised a threshold at all, so there was no request to
report, and the parser, left with a rate word and no filter, fell through to the
weighted-average rate. A grammar gap upstream of the guard cannot be caught by
the guard.

THE FIX IS TO DELETE THE SECOND GRAMMAR, not to extend it: the postfix patterns
now embed `_VALUE` and coerce through `_amount()`, so there is exactly one
definition of what a number may look like in a threshold, and "£200k or more"
works for the same reason "7% or more" does.

WHAT THIS FILE PINS. The contrast pairs the brief asked for — the same bound
under three different measure requests, and the same phrasing across both
frames — plus the age behaviour that already worked, so the fix cannot be
mistaken for a widening that traded one theme for another.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from mi_agent.llm_query_parser import _deterministic_parse      # noqa: E402
from mi_agent.mi_query_validator import load_mi_semantics       # noqa: E402

_SEMANTICS = load_mi_semantics(
    str(_REPO_ROOT / "mi_agent" / "mi_semantics_field_registry.yaml"))

RATE = "current_interest_rate"
AGE = "youngest_borrower_age"
BALANCE = "current_outstanding_balance"
LTV = "current_loan_to_value"


def parse(question: str):
    spec, _meta = _deterministic_parse(question, _SEMANTICS)
    return spec


class TestTheBoundSurvivesHoweverItIsWritten(unittest.TestCase):
    """One bound, six spellings. Every one of them is `>= 7` on the rate."""

    #: Postfix spellings of the SAME bound. The first three parsed before this
    #: work; the last three did not, and differ only in the format of the number.
    SPELLINGS = (
        "7 or more", "7 or above", "7 or greater",
        "7% or more", "7 % or above", "7.0% or more",
    )

    def test_every_spelling_binds_the_same_rate_threshold(self):
        for spelling in self.SPELLINGS:
            question = f"How many funded loans have an interest rate of {spelling}?"
            with self.subTest(spelling=spelling):
                self.assertEqual(parse(question).filters,
                                 {RATE: {"op": "ge", "value": 7.0}})

    def test_a_currency_bound_is_a_number_too(self):
        """Never exercised by the bank, and broken for the same reason: the
        private grammar had no currency prefix and no k/m/bn."""
        for spelling, expected in (("£200,000 or more", 200000.0),
                                   ("£200k or more", 200000.0),
                                   ("200000 or more", 200000.0)):
            with self.subTest(spelling=spelling):
                self.assertEqual(
                    parse(f"How many funded loans have a balance of {spelling}?"
                          ).filters,
                    {BALANCE: {"op": "ge", "value": expected}})

    def test_the_lower_bound_reads_the_same_grammar(self):
        self.assertEqual(
            parse("How many funded loans have an interest rate of 5% or below?"
                  ).filters,
            {RATE: {"op": "le", "value": 5.0}})


class TestTheThreeMeasureRequests(unittest.TestCase):
    """The brief's contrast set: one filter, three things asked ABOUT it.

    The filter must be identical in all three; only the measure and the
    aggregation may differ. This is the assertion that separates "the count
    request was lost" from "the rate filter was lost" — F032 lost both, and
    fixing only one of them would still answer a different question.
    """

    QUESTIONS = {
        "count": ("How many funded loans have an interest rate of 7% or more?",
                  None, "count"),
        "weighted_avg": ("What is the weighted average LTV for funded loans "
                         "with an interest rate of 7% or more?",
                         LTV, "weighted_avg"),
        "sum": ("What is the balance of funded loans with an interest rate "
                "of 7% or more?", BALANCE, "sum"),
    }

    def test_each_request_keeps_the_filter_and_owns_its_aggregation(self):
        for name, (question, metric, agg) in self.QUESTIONS.items():
            with self.subTest(request=name):
                spec = parse(question)
                self.assertEqual(spec.filters, {RATE: {"op": "ge", "value": 7.0}},
                                 f"{name}: the rate filter did not survive")
                self.assertEqual(spec.aggregation, agg)
                self.assertEqual(spec.metric, metric)

    def test_a_field_cannot_be_the_measure_and_its_own_predicate(self):
        """A BOUNDARY THIS FIX DOES NOT MOVE, recorded so it is not mistaken for
        one it did.

        "the weighted average RATE on loans with an interest RATE of 7% or more"
        names one field twice — once as the thing measured, once as the subject
        of the predicate. `is_filter_subject` excludes BOTH mentions, so no
        measure survives and the question is `unmapped`.

        It is out of Phase 1's scope for a reason that is checkable rather than
        convenient: it behaves IDENTICALLY under the prefix form, which this
        work did not touch, so it is not this seam; and `unmapped` is a
        controlled refusal, not a silent wrong answer, so it is not one of the
        six. Asserted in both forms — if a later fix closes it, both change
        together, and if one form starts answering while the other refuses, that
        is the drift this file exists to catch.
        """
        for phrasing in ("of 7% or more", "above 7%"):
            with self.subTest(phrasing=phrasing):
                spec, meta = _deterministic_parse(
                    "What is the weighted average rate on funded loans with an "
                    f"interest rate {phrasing}?", _SEMANTICS)
                self.assertIsNone(spec.metric)
                self.assertEqual((meta or {}).get("note"), "unmapped")

    def test_a_count_question_never_returns_the_filtered_field_as_a_measure(self):
        """The precise silent substitution: the field named in the PREDICATE
        became the field reported as the ANSWER."""
        spec = parse("How many funded loans have an interest rate of 7% or more?")
        self.assertNotEqual(spec.metric, RATE)
        self.assertEqual(spec.aggregation, "count")


class TestBothFramesReadOneGrammar(unittest.TestCase):
    """F032 and P030 are the same sentence about two populations. A fix that
    needed a pipeline clause would be a fifth private vocabulary."""

    def test_the_pipeline_wording_parses_identically(self):
        funded = parse("How many funded loans have an interest rate of 7% or more?")
        pipeline = parse("How many pipeline cases have an interest rate of 7% or more?")
        self.assertEqual(funded.filters, pipeline.filters)
        self.assertEqual(funded.aggregation, pipeline.aggregation)
        self.assertEqual(funded.metric, pipeline.metric)


class TestWhatAlreadyWorkedStillWorks(unittest.TestCase):
    """The postfix grammar was written for age, and age scored 20/20. None of
    it may change — including the bare `70+` and the `years` unit the private
    grammar named explicitly."""

    def test_age_postfix_bounds_are_unchanged(self):
        for spelling, op, value in (
                ("85 or older", "ge", 85.0),
                ("70+", "ge", 70.0),
                ("70 years or above", "ge", 70.0),
                ("65 or younger", "le", 65.0),
                ("80 and over", "ge", 80.0)):
            with self.subTest(spelling=spelling):
                self.assertEqual(
                    parse(f"How many funded loans have a borrower aged {spelling}?"
                          ).filters,
                    {AGE: {"op": op, "value": value}})

    def test_the_prefix_grammar_is_untouched(self):
        self.assertEqual(
            parse("How many funded loans have an interest rate above 6%?").filters,
            {RATE: {"op": "gt", "value": 6.0}})
        self.assertEqual(
            parse("How many funded loans have a balance above £200k?").filters,
            {BALANCE: {"op": "gt", "value": 200000.0}})


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
