#!/usr/bin/env python3
"""What the reader asked for, recorded once, however they said it.

THE DEFECT. Five owners in this module decide whether a count was requested,
and three of them are adjacency-bound — they spell it
``how\\s+many\\s+(?:loans|cases|accounts)``, so a single adjective between the
words breaks them:

    how many loans are there            _wants_count True   _COUNT_MEASURE True
    how many FUNDED loans are there     _wants_count False  _COUNT_MEASURE False
    how many PIPELINE cases are there   _wants_count False  _COUNT_MEASURE False
    how many JOINT loans are there      _wants_count False  _COUNT_MEASURE False

`is_count_q`, a fifth reading written inline in the middle of the parse, is
`\\bhow many\\b` and sees all four. So the estate simultaneously knows and does
not know that these are counts, and which answer you get depends on which owner
the sentence happens to reach.

WHY IT COSTS A WHOLE COMPOSITION FAMILY. The two adjacency-bound owners are
exactly the two that feed the MEASURE SET. `detect_measure_set` needs two
measures before it will report any, so losing the count leaves one, and the
request falls past `_measure_set_recognizer` into a single-output branch where
the second output has nowhere to go:

    "How many funded loans are to joint borrowers, and what is their funded
     balance?"          → count only; the balance is never represented

    "What is the funded balance for joint borrowers, and how many loans are
     there?"            → both, executed together

Two spellings of one question, two different contracts. The reader cannot see
why, and neither could the estate: nothing records what was asked for, so
nothing can notice that an output went missing.

WHAT THIS FILE PINS — the invariant, not the sentences.

    Equivalent requests resolve to equivalent REQUESTED-OUTPUT SETS, and a
    modifier between the interrogative and the row noun is not a semantic
    difference.

The cases below are built by combining a phrasing template with a modifier and
a row noun, so passing them by teaching the parser any particular sentence is
not possible — there is no sentence here that a fix could target without
covering the whole grammar. That is deliberate: §18 of the sprint brief makes
the banks oracles, and a test that could be satisfied phrase-by-phrase would
measure nothing the banks do not already measure.
"""

from __future__ import annotations

import itertools
import sys
import unittest
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from mi_agent.llm_query_parser import (                          # noqa: E402
    _deterministic_parse, detect_measure_set)
from mi_agent.mi_query_validator import load_mi_semantics        # noqa: E402

_SEMANTICS = load_mi_semantics(
    str(_REPO_ROOT / "mi_agent" / "mi_semantics_field_registry.yaml"))

COUNT = "loan_count"
BALANCE = "current_outstanding_balance"
LTV = "current_loan_to_value"


def outputs(question: str):
    """The requested-output set, as (field, aggregation) pairs.

    Read from `detect_measure_set` because that is what the multi-output path
    consumes. When the estate grows a single requested-output owner this is the
    one line that moves.
    """
    return {(m["field"], m["aggregation"])
            for m in detect_measure_set(question.lower(), _SEMANTICS)}


def spec_of(question: str):
    spec, _meta = _deterministic_parse(question, _SEMANTICS)
    return spec


# --------------------------------------------------------------------------- #
# The grammar under test, built rather than listed.
# --------------------------------------------------------------------------- #
#: A modifier between the interrogative and the row noun. The empty string is
#: the case that already worked, and it is included so the parametrisation
#: proves the OTHERS are the defect rather than the whole family being broken.
MODIFIERS = ("", "funded ", "joint ", "acquired ")

ROW_NOUNS = ("loans", "cases")


class TestAModifierIsNotASemanticDifference(unittest.TestCase):

    def test_count_is_recognised_whatever_sits_before_the_row_noun(self):
        """The narrowest statement of the defect, over the whole grammar."""
        for modifier, noun in itertools.product(MODIFIERS, ROW_NOUNS):
            question = (f"How many {modifier}{noun} are there "
                        f"and what is their balance?")
            with self.subTest(modifier=modifier or "(none)", noun=noun):
                self.assertEqual(outputs(question),
                                 {(COUNT, "count"), (BALANCE, "sum")})

    def test_the_owners_agree_with_each_other(self):
        """The five readings, asked the same question.

        Asserted on the OWNERS rather than only on outcomes, because the outcome
        can be right while the owners still disagree — and the next branch to
        consult the wrong one puts the defect straight back.
        """
        import re

        from mi_agent.llm_query_parser import (
            _counts_a_row_noun, _COUNT_MEASURE_RE, _wants_count)

        for modifier, noun in itertools.product(MODIFIERS, ROW_NOUNS):
            question = f"how many {modifier}{noun} are there"
            with self.subTest(question=question):
                readings = {
                    "_wants_count": _wants_count(question),
                    "_COUNT_MEASURE_RE": bool(_COUNT_MEASURE_RE.search(question)),
                    "is_count_q": bool(re.search(
                        r"\bhow many\b|\bnumber of\b|\bcount of\b", question)),
                    "_counts_a_row_noun": _counts_a_row_noun(question),
                }
                self.assertEqual(set(readings.values()), {True}, readings)


class TestEquivalentPhrasingsAreOneContract(unittest.TestCase):
    """§17's requested-output contract: syntax varies, the output set does not."""

    #: Every one of these asks for a count and a balance over the same book.
    SAME_REQUEST = (
        "Give me the loan count and balance.",
        "How many loans are there and what is their balance?",
        "Show the number of loans plus balance.",
        "What is the balance, and how many loans are there?",
        "How many funded loans are there and what is the funded balance?",
    )

    def test_all_of_them_request_the_same_two_outputs(self):
        expected = {(COUNT, "count"), (BALANCE, "sum")}
        for question in self.SAME_REQUEST:
            with self.subTest(question=question):
                self.assertEqual(outputs(question), expected)

    def test_the_order_of_the_clauses_does_not_change_the_contract(self):
        forward = outputs("How many funded loans are to joint borrowers, "
                          "and what is their funded balance?")
        reverse = outputs("What is the funded balance for joint borrowers, "
                          "and how many loans are there?")
        self.assertEqual(forward, reverse)


class TestTheSharedScopeSurvivesComposition(unittest.TestCase):
    """A composed request must not lose the population while gaining outputs.

    This is the failure mode the fix could introduce: routing family A into the
    measure-set path answers both outputs, and would be worse than the defect if
    it answered them over the whole book.
    """

    def test_the_population_is_carried_with_the_outputs(self):
        spec = spec_of("How many funded loans are to joint borrowers, "
                       "and what is their funded balance?")
        self.assertEqual({(m["field"], m["aggregation"]) for m in spec.measures},
                         {(COUNT, "count"), (BALANCE, "sum")})
        self.assertEqual(spec.filters.get("borrower_type"), "Joint")

    def test_a_threshold_population_is_carried_too(self):
        spec = spec_of("How many funded loans have an interest rate above 6%, "
                       "and what is their balance?")
        self.assertEqual({(m["field"], m["aggregation"]) for m in spec.measures},
                         {(COUNT, "count"), (BALANCE, "sum")})
        self.assertEqual(spec.filters.get("current_interest_rate"),
                         {"op": "gt", "value": 6.0})


class TestSingleOutputRequestsAreUntouched(unittest.TestCase):
    """`detect_measure_set` reports nothing below two measures, so a one-output
    question must keep exactly the parse it has. This is the blast boundary."""

    def test_a_bare_count_is_still_a_bare_count(self):
        for question in ("How many funded loans are there?",
                         "How many pipeline cases are there?",
                         "How many joint loans are there?"):
            with self.subTest(question=question):
                spec = spec_of(question)
                self.assertEqual(spec.aggregation, "count")
                self.assertIsNone(spec.metric)
                self.assertFalse(spec.measures)

    def test_a_bare_balance_is_still_a_bare_balance(self):
        spec = spec_of("What is the total funded balance?")
        self.assertEqual(spec.metric, BALANCE)
        self.assertEqual(spec.aggregation, "sum")
        self.assertFalse(spec.measures)

    def test_a_filtered_count_keeps_its_filter_and_gains_no_measure(self):
        spec = spec_of("How many funded loans have an interest rate of 7% or more?")
        self.assertEqual(spec.aggregation, "count")
        self.assertIsNone(spec.metric)
        self.assertEqual(spec.filters,
                         {"current_interest_rate": {"op": "ge", "value": 7.0}})


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
