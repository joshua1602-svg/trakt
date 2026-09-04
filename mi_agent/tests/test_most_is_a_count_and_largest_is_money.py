#!/usr/bin/env python3
"""Two different questions that produced one identical plan.

THE OBSERVATION, from the 100-question atomic perimeter run:

    "Which product type has the most funded loans?"
    "Which product type has the largest funded balance?"

parsed to the SAME spec — metric `current_outstanding_balance`, aggregation
`sum`, dimension `erm_product_type` — so the first was answered with the product
carrying the most MONEY, which on a book with a few large lump-sum cases is not
the product with the most LOANS. F039 on the funded frame and P040 on broker
("Which broker has the most pipeline cases?" → *"Balance: £4.7MM · Broker: …"*)
are the same defect on two dimensions, which is why they must not have two fixes.

WHY IT HAPPENED. `_RANK_DESC` is ("largest", "biggest", "highest", "greatest",
"top ") — "most" is deliberately absent, so `_detect_ranking` does not even see
these as rankings. They become an ordinary grouped bar, and the grouped branch
had exactly two arms: an explicit count (`_wants_count`) or the balance default.
A superlative over a bare row noun matched neither, so it took the default.

THE OWNER ALREADY EXISTS AND IS ALREADY TRUSTED FOR THIS. `_counts_a_row_noun`
reads the bare governed row noun standing as the subject and excludes anything
carrying a money word, and it was wired into the TREND branch for precisely this
reason — "a trend of things is a count of them", 2026-09-04. A superlative is the
same request said with a ranking word instead of a period, and the grouped branch
simply did not ask it. It is asked now; no ranking vocabulary was added.

    "most funded loans"      → True      "largest funded balance"  → False
    "most pipeline cases"    → True      "largest pipeline amount"  → False

THE SECOND DEFECT IN THE SAME LINE. `metric_defaulted` was False on all four of
those specs, including the three where nobody named a measure. The trend branch
records its default; the grouped branch did not, so a substituted balance was
indistinguishable downstream from a balance the reader asked for — the exact
condition the disclosure field exists to prevent. Pinned below.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from mi_agent.llm_query_parser import (                          # noqa: E402
    _counts_a_row_noun, _deterministic_parse)
from mi_agent.mi_query_validator import load_mi_semantics        # noqa: E402

_SEMANTICS = load_mi_semantics(
    str(_REPO_ROOT / "mi_agent" / "mi_semantics_field_registry.yaml"))

BALANCE = "current_outstanding_balance"


def parse(question: str):
    spec, _meta = _deterministic_parse(question, _SEMANTICS)
    return spec


class TestThePairsDisagree(unittest.TestCase):
    """The brief's requirement: prove it on three unrelated dimensions, so the
    rule is about the SUPERLATIVE and not about product, or broker, or any
    particular axis."""

    #: (dimension phrase, expected bound dimension)
    DIMENSIONS = (
        ("product type", "erm_product_type"),
        ("broker", "broker_channel"),
        ("region", None),          # whichever region field this book prefers
    )

    def test_most_ranks_by_count(self):
        for phrase, expected_dim in self.DIMENSIONS:
            with self.subTest(dimension=phrase):
                spec = parse(f"Which {phrase} has the most funded loans?")
                self.assertEqual(spec.aggregation, "count")
                self.assertIsNone(spec.metric)
                if expected_dim:
                    self.assertEqual(spec.dimension, expected_dim)
                else:
                    self.assertIsNotNone(spec.dimension)

    def test_largest_ranks_by_money(self):
        for phrase, expected_dim in self.DIMENSIONS:
            with self.subTest(dimension=phrase):
                spec = parse(f"Which {phrase} has the largest funded balance?")
                self.assertEqual(spec.aggregation, "sum")
                self.assertEqual(spec.metric, BALANCE)
                if expected_dim:
                    self.assertEqual(spec.dimension, expected_dim)

    def test_the_two_no_longer_produce_one_plan(self):
        """The finding, stated as the thing that must never be true again."""
        for phrase, _dim in self.DIMENSIONS:
            with self.subTest(dimension=phrase):
                most = parse(f"Which {phrase} has the most funded loans?")
                largest = parse(f"Which {phrase} has the largest funded balance?")
                self.assertNotEqual(
                    (most.metric, most.aggregation),
                    (largest.metric, largest.aggregation))


class TestThePipelineFramesSaysItTheSameWay(unittest.TestCase):

    def test_most_cases_is_a_count_and_largest_amount_is_money(self):
        cases = parse("Which broker has the most pipeline cases?")
        self.assertEqual(cases.aggregation, "count")
        self.assertIsNone(cases.metric)
        amount = parse("Which broker has the largest pipeline amount?")
        self.assertEqual(amount.aggregation, "sum")
        self.assertEqual(amount.metric, BALANCE)


class TestTheOwnerIsTheExistingPredicate(unittest.TestCase):
    """If a later change makes the grouped branch and `_counts_a_row_noun`
    disagree, this is where it shows — rather than in a bank score."""

    def test_the_predicate_discriminates_the_pairs(self):
        for question, expected in (
                ("which product type has the most funded loans?", True),
                ("which product type has the largest funded balance?", False),
                ("which broker has the most pipeline cases?", True),
                ("which broker has the largest pipeline amount?", False)):
            with self.subTest(question=question):
                self.assertIs(_counts_a_row_noun(question), expected)

    def test_the_branch_agrees_with_the_predicate(self):
        for question in ("Which product type has the most funded loans?",
                         "Which broker has the most pipeline cases?",
                         "Which product type has the largest funded balance?",
                         "Which broker has the largest pipeline amount?"):
            with self.subTest(question=question):
                spec = parse(question)
                self.assertIs(spec.aggregation == "count",
                              _counts_a_row_noun(question.lower()))


class TestTheSameDefectWithoutASuperlative(unittest.TestCase):
    """A THIRD ROW THE BANK NEVER REACHED, found by writing this file.

        "How many funded loans by broker?"
            → metric current_outstanding_balance, aggregation sum

    on the code before this change. No superlative, no ranking word — just a
    grouped question whose measure is a bare row noun. `_wants_count` returns
    False for it (its vocabulary is "loan count" / "case count", not "how many
    … by"), so the branch fell to the balance default exactly as F039 did.

    It is recorded here rather than in the "untouched" class below because it
    was NOT untouched: it was already wrong, and it is the same defect. Its
    presence is why the fix is stated as "the grouped branch asks
    `_counts_a_row_noun`" and not as "the ranking branch handles most" — a
    ranking-specific rule would have left this one broken and no bank row would
    have said so.
    """

    def test_how_many_by_dimension_is_a_count(self):
        for question in ("How many funded loans by broker?",
                         "How many funded loans are there by broker?",
                         "How many pipeline cases by stage?"):
            with self.subTest(question=question):
                spec = parse(question)
                self.assertEqual(spec.aggregation, "count")
                self.assertIsNone(spec.metric)
                self.assertIsNotNone(spec.dimension)


class TestASubstitutedMeasureSaysSo(unittest.TestCase):

    def test_a_defaulted_balance_is_disclosed(self):
        """Nobody named a measure here, so the balance is the parser's choice
        and the spec must say so."""
        self.assertTrue(parse("Which product type has the largest amount?"
                              ).metric_defaulted)

    def test_a_named_measure_is_not_marked_defaulted(self):
        self.assertFalse(parse("Which product type has the largest funded balance?"
                               ).metric_defaulted)

    def test_a_count_is_not_a_defaulted_measure(self):
        """A count is a measure the reader asked for, not one substituted for
        them — marking it defaulted would tell the receipt the opposite."""
        self.assertFalse(parse("Which product type has the most funded loans?"
                               ).metric_defaulted)


class TestExplicitCountLanguageIsUntouched(unittest.TestCase):
    """`_wants_count` was already the first arm of this branch and stays first.
    These are the phrasings it owns, and none of them change."""

    def test_loan_count_by_dimension_still_counts(self):
        for question in ("Show loan count by product type",
                         "Case count by pipeline stage"):
            with self.subTest(question=question):
                spec = parse(question)
                self.assertEqual(spec.aggregation, "count")
                self.assertIsNone(spec.metric)

    def test_a_named_measure_by_dimension_is_untouched(self):
        for question, metric, agg in (
                ("Total balance by product type", BALANCE, "sum"),
                ("Weighted average LTV by broker", "current_loan_to_value",
                 "weighted_avg"),
                ("Average borrower age by region", "youngest_borrower_age",
                 "avg")):
            with self.subTest(question=question):
                spec = parse(question)
                self.assertEqual(spec.metric, metric)
                self.assertEqual(spec.aggregation, agg)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
