#!/usr/bin/env python3
"""The narrowed breakdown, from English to a number, checked against pandas.

WHAT THIS ADDS TO `test_portfolio_truth_bank`. That bank constructs plans
directly and never speaks English — deliberately, so a failure there is
arithmetic or population and never vocabulary. It therefore cannot see a
question that is parsed into the WRONG SHAPE, and the shape is exactly what was
wrong: "total balance by region for joint borrowers" resolved to one number over
the joint book instead of a breakdown of it, and every plan-level check agreed
with itself about the summary it was given.

So this bank starts at the sentence and finishes at the cells, and the expected
value comes from `portfolio_truth_oracle`, which imports nothing from the
product. A row here is only green when the question was READ as a breakdown, the
narrowing was applied, and every cell equals an independently computed one.

CELL BY CELL, not by total. A cross-tab whose total agrees can still have the
mass in the wrong groups, and the total is the number a wrong grouping is most
likely to get right — which is precisely how a whole-book figure passed for a
joint-book one.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from mi_agent.mi_agent_workflow import run_mi_agent_query                # noqa: E402
from mi_agent.mi_query_validator import load_mi_semantics                # noqa: E402
from mi_agent.tests import portfolio_truth_oracle as truth               # noqa: E402

_SEMANTICS = load_mi_semantics(
    str(_REPO_ROOT / "mi_agent" / "mi_semantics_field_registry.yaml"))
_BOOK = truth.canonical_book()

BALANCE = truth.BALANCE
LTV = truth.LTV

JOINT = ("borrower_type", "eq", "Joint")
SCOTLAND = ("collateral_geography", "eq", "Scotland")
LUMP_SUM = ("erm_product_type", "eq", "Lump Sum")
LTV_OVER_50 = (LTV, "gt", 50.0)

#: ``(question, grouping column, measure column or None for a count,
#:   predicates the oracle applies)``
CASES = (
    ("Total balance by region for joint borrowers",
     "collateral_geography", BALANCE, (JOINT,)),
    ("Total balance by product in Scotland",
     "erm_product_type", BALANCE, (SCOTLAND,)),
    ("How much balance by broker for joint borrowers",
     "broker_channel", BALANCE, (JOINT,)),
    ("Total balance by broker for loans with LTV over 50%",
     "broker_channel", BALANCE, (LTV_OVER_50,)),
    ("Total balance by region for lump sum loans",
     "collateral_geography", BALANCE, (LUMP_SUM,)),
    ("How many loans in Scotland by product?",
     "erm_product_type", None, (SCOTLAND,)),
    ("Loan count by region for joint borrowers",
     "collateral_geography", None, (JOINT,)),
    ("How many loans by broker for loans with LTV over 50%",
     "broker_channel", None, (LTV_OVER_50,)),
)


def _executed_cells(question: str, dimension: str, measure):
    """``{group: figure}`` as the product computed it, or an AssertionError."""
    result = run_mi_agent_query(question, _BOOK, _SEMANTICS)
    assert result.get("ok"), (
        f"{question!r} was not answered: {result.get('error')!r}")
    frame = result["query_result"].data
    assert dimension in frame.columns, (
        f"{question!r} did not group by {dimension}: columns {list(frame.columns)}")
    column = f"{measure}_sum" if measure else "count"
    assert column in frame.columns, (
        f"{question!r} produced no {column}: columns {list(frame.columns)}")
    return {str(row[dimension]): float(row[column])
            for _, row in frame.iterrows()}


def _oracle_cells(dimension: str, measure, predicates):
    grouped = _BOOK[truth.mask_for(_BOOK, predicates)].groupby(dimension)
    series = grouped.size() if measure is None else grouped[measure].sum()
    return {str(key): float(value) for key, value in series.items()}


class TestANarrowedBreakdownIsArithmeticallyRight(unittest.TestCase):

    def test_every_cell_matches_an_independently_computed_one(self):
        for question, dimension, measure, predicates in CASES:
            with self.subTest(question=question):
                executed = _executed_cells(question, dimension, measure)
                expected = _oracle_cells(dimension, measure, predicates)
                self.assertEqual(set(executed), set(expected),
                                 "the breakdown has the wrong groups")
                for group, value in expected.items():
                    self.assertAlmostEqual(
                        executed[group], value, places=2,
                        msg=f"{question!r}: {group} is wrong")

    def test_the_narrowing_actually_narrowed(self):
        """The cells must NOT equal the whole-book cells.

        Without this the bank would pass on a product that ignored the
        population entirely, whenever the population happens to be most of the
        book — which is the shape of the defect that made this file necessary.
        """
        for question, dimension, measure, predicates in CASES:
            with self.subTest(question=question):
                narrowed = _oracle_cells(dimension, measure, predicates)
                whole_book = _oracle_cells(dimension, measure, ())
                self.assertNotEqual(
                    narrowed, whole_book,
                    "this row cannot detect a dropped narrowing on this book")
                self.assertNotEqual(
                    _executed_cells(question, dimension, measure), whole_book,
                    "the answer is the whole book's")


if __name__ == "__main__":
    unittest.main()
