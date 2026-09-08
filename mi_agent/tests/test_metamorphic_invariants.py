#!/usr/bin/env python3
"""Relations between answers that must hold whatever the answers are.

WHY THIS IS DIFFERENT FROM A TRUTH BANK. A truth bank asserts a FIGURE, so it
can only cover the questions somebody wrote down and computed an expectation
for. A metamorphic invariant asserts a RELATION between two answers — the whole
and its parts, the same request said two ways, a narrowing and the cell it
selects — and it holds for questions nobody has thought of yet. It needs no
oracle: the product's own answers are the inputs, and the relation is what
cannot be wrong.

These are the relations a reader would notice were broken, stated the way a
reader states them:

  A  a narrowing never grows a total
  B  the parts add up to the whole
  C  narrowing to one value is the cell that value has in the breakdown
  D  the ORDER two narrowings are written in changes nothing
  E  a two-axis breakdown's margins are the one-axis breakdowns
  F  a count is the number of rows the same population has
  G  a threshold that excludes nothing changes nothing

Each is asserted from ENGLISH, through the whole product, over a real book. A
relation that fails names a defect without anyone having had to predict it.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from mi_agent.mi_agent_workflow import run_mi_agent_query               # noqa: E402
from mi_agent.mi_query_validator import load_mi_semantics               # noqa: E402
from mi_agent.tests import portfolio_truth_oracle as truth              # noqa: E402

_SEMANTICS = load_mi_semantics(
    str(_REPO_ROOT / "mi_agent" / "mi_semantics_field_registry.yaml"))
_BOOK = truth.canonical_book()

BALANCE = truth.BALANCE
_SUM = f"{BALANCE}_sum"


def answer(question: str):
    """The executed frame for one question, or an AssertionError naming why not."""
    result = run_mi_agent_query(question, _BOOK, _SEMANTICS)
    assert result.get("ok"), f"{question!r}: {result.get('error')!r}"
    return result["query_result"].data


def total(question: str) -> float:
    frame = answer(question)
    assert _SUM in frame.columns, f"{question!r} produced no balance: {list(frame.columns)}"
    return float(frame[_SUM].sum())


def count(question: str) -> int:
    frame = answer(question)
    assert "loan_count" in frame.columns, (
        f"{question!r} produced no count: {list(frame.columns)}")
    return int(frame["loan_count"].sum())


def cells(question: str, dimension: str, column: str = _SUM):
    frame = answer(question)
    assert dimension in frame.columns, (
        f"{question!r} did not group by {dimension}: {list(frame.columns)}")
    return {str(row[dimension]): float(row[column]) for _, row in frame.iterrows()}


#: The narrowings these relations are exercised over. Written as a reader writes
#: them, one of each governed kind.
NARROWINGS = ("for joint borrowers", "in Scotland", "for lump sum loans",
              "for Alpha", "for loans with LTV over 50%",
              "for borrowers over 75")


class TestANarrowingNeverGrowsATotal(unittest.TestCase):

    def test_every_narrowing_is_a_subset(self):
        whole = total("What is the total balance?")
        for narrowing in NARROWINGS:
            with self.subTest(narrowing=narrowing):
                self.assertLessEqual(
                    total(f"What is the total balance {narrowing}?"), whole + 0.01,
                    "a narrowed population produced MORE balance than the book")

    def test_two_narrowings_are_a_subset_of_one(self):
        one = total("What is the total balance for joint borrowers?")
        both = total("What is the total balance for joint borrowers in Scotland?")
        self.assertLessEqual(both, one + 0.01)
        self.assertGreater(one, 0.0)


class TestThePartsAddUpToTheWhole(unittest.TestCase):
    """The relation a wrong grouping is most likely to break, and the one a
    total on its own cannot see."""

    #: Every governed axis this book carries, categorical and bucketed alike.
    #: The bucketed ones are here because a bucket engine is a second way for
    #: the parts to stop adding up.
    AXES = ("region", "product", "broker", "borrower type", "ltv", "age")

    def test_a_breakdown_sums_to_the_unbroken_total(self):
        whole = total("What is the total balance?")
        for axis in self.AXES:
            with self.subTest(axis=axis):
                self.assertAlmostEqual(
                    sum(cells(f"Total balance by {axis}",
                              _AXIS_COLUMN[axis]).values()),
                    whole, places=2)

    def test_a_narrowed_breakdown_sums_to_the_narrowed_total(self):
        for narrowing in NARROWINGS:
            for axis in self.AXES:
                with self.subTest(narrowing=narrowing, axis=axis):
                    self.assertAlmostEqual(
                        sum(cells(f"Total balance by {axis} {narrowing}",
                                  _AXIS_COLUMN[axis]).values()),
                        total(f"What is the total balance {narrowing}?"),
                        places=2)


#: The governed column each axis word resolves to on this book.
_AXIS_COLUMN = {"region": "collateral_geography",
                "product": "erm_product_type",
                "broker": "broker_channel",
                "borrower type": "borrower_type",
                "ltv": "ltv_bucket",
                "age": "age_bucket"}


class TestNarrowingToAValueIsThatValuesCell(unittest.TestCase):
    """The relation that ties the two shapes together: a filtered total and the
    corresponding bar of the breakdown are the same number, or one of them is
    wrong."""

    CASES = (("region", "collateral_geography", "Scotland", "in Scotland"),
             ("product", "erm_product_type", "Lump Sum", "for lump sum loans"),
             ("broker", "broker_channel", "Alpha", "for Alpha"))

    def test_the_cell_equals_the_filtered_total(self):
        for axis, column, value, phrase in self.CASES:
            with self.subTest(value=value):
                breakdown = cells(f"Total balance by {axis}", column)
                self.assertIn(value, breakdown)
                self.assertAlmostEqual(
                    breakdown[value],
                    total(f"What is the total balance {phrase}?"), places=2)


class TestTheOrderOfNarrowingsDoesNotMatter(unittest.TestCase):

    PAIRS = (("What is the total balance for joint borrowers in Scotland?",
              "What is the total balance in Scotland for joint borrowers?"),
             ("What is the total balance for lump sum loans in Scotland?",
              "What is the total balance in Scotland for lump sum loans?"))

    def test_both_orders_agree(self):
        for first, second in self.PAIRS:
            with self.subTest(first=first):
                self.assertAlmostEqual(total(first), total(second), places=2)
                self.assertEqual(count(first), count(second))


class TestTheMarginsOfATwoAxisBreakdownAreTheOneAxisBreakdowns(unittest.TestCase):

    def test_summing_out_one_axis_gives_the_other(self):
        frame = answer("Total balance by region and product")
        margin = {}
        for _, row in frame.iterrows():
            key = str(row["collateral_geography"])
            margin[key] = margin.get(key, 0.0) + float(row[_SUM])
        self.assertEqual(set(margin), set(cells("Total balance by region",
                                                "collateral_geography")))
        for key, value in cells("Total balance by region",
                                "collateral_geography").items():
            self.assertAlmostEqual(margin[key], value, places=2)


class TestACountIsTheNumberOfRows(unittest.TestCase):

    def test_the_count_matches_the_populations_size(self):
        for phrase, predicates in (
                ("", ()),
                ("for joint borrowers", (("borrower_type", "eq", "Joint"),)),
                ("in Scotland", (("collateral_geography", "eq", "Scotland"),)),
                ("for loans with LTV over 50%", ((truth.LTV, "gt", 50.0),))):
            question = f"How many loans are there {phrase}?".replace("  ", " ")
            with self.subTest(question=question):
                self.assertEqual(count(question),
                                 truth.row_count(_BOOK, predicates))


class TestAThresholdThatExcludesNothingChangesNothing(unittest.TestCase):
    """A predicate every row satisfies must leave the answer alone — and the
    estate has a defect in exactly this shape on record, where a filter was
    declared LOST because the row count did not move."""

    def test_an_always_true_threshold_is_a_no_op(self):
        whole = total("What is the total balance?")
        rows = count("How many loans are there?")
        for phrase in ("with LTV over 0%", "with a balance above 1"):
            with self.subTest(phrase=phrase):
                self.assertAlmostEqual(
                    total(f"What is the total balance for loans {phrase}?"),
                    whole, places=2)
                self.assertEqual(
                    count(f"How many loans are there {phrase}?"), rows)


if __name__ == "__main__":
    unittest.main()
