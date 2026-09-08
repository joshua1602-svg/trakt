#!/usr/bin/env python3
"""Eight books that break the assumptions, and what the product does with them.

WHY. Every figure this estate publishes is computed over a real portfolio, and
real portfolios are not the tidy 400-row frame the banks use. They arrive empty
while a feed is late; they arrive with one loan on day one; they arrive with a
column entirely null because the source system does not populate it yet; they
arrive from a regional subsidiary where every loan is in one place. None of
those is an error condition — they are Tuesday — and what the product must never
do on any of them is raise, or answer confidently from a calculation that did
not mean what it says.

The eight shapes: empty; one row; a measure entirely null; half a grouping
column missing; a grouping column with ONE distinct value; every balance zero;
some balances negative; a filter column entirely null.

WHAT IS ASSERTED, and it is deliberately not a table of expected figures. The
right answer for most of these is a refusal, and pinning which ones refuse would
make an improvement look like a regression. Three properties hold whatever the
product decides:

  * NOTHING RAISES. An uncaught exception is not a governed refusal; it is the
    reader seeing a stack trace where a sentence belongs.
  * AN ANSWER IS ARITHMETICALLY TRUE OF THE BOOK IT WAS COMPUTED OVER. Where a
    figure comes back it equals the one the oracle computes from the same
    frame — so a degenerate shape may not quietly change what a total means.
  * A REFUSAL EXPLAINS ITSELF. It names something; it is never empty.

THE ONE PINNED CASE is the single-region book, because a defect lived there: a
narrowing that every row satisfies was reported LOST, since the receipt decided
whether narrowing had happened by asking whether the row count went down.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from mi_agent.mi_agent_workflow import run_mi_agent_query               # noqa: E402
from mi_agent.mi_query_validator import load_mi_semantics               # noqa: E402
from mi_agent.tests import portfolio_truth_oracle as truth              # noqa: E402

_SEMANTICS = load_mi_semantics(
    str(_REPO_ROOT / "mi_agent" / "mi_semantics_field_registry.yaml"))

BALANCE = truth.BALANCE
_SUM = f"{BALANCE}_sum"

QUESTIONS = (
    "What is the total balance?",
    "Total balance by region",
    "How many loans are there?",
    "What is the total balance in Scotland?",
    "Total balance by region for joint borrowers",
    "What is the average loan size?",
)


def _books():
    """The eight shapes, built fresh so no test can contaminate another."""
    book = truth.canonical_book()
    yield "empty", book.iloc[0:0]

    book = truth.canonical_book()
    yield "one row", book.iloc[[0]]

    book = truth.canonical_book()
    book[BALANCE] = np.nan
    yield "the measure is entirely null", book

    book = truth.canonical_book()
    book.loc[book.index[:200], "collateral_geography"] = np.nan
    yield "half the grouping column is missing", book

    book = truth.canonical_book()
    book["collateral_geography"] = "Scotland"
    yield "one distinct region", book

    book = truth.canonical_book()
    book[BALANCE] = 0.0
    yield "every balance is zero", book

    book = truth.canonical_book()
    book.loc[book.index[:5], BALANCE] = -1000.0
    yield "some balances are negative", book

    book = truth.canonical_book()
    book["borrower_type"] = np.nan
    yield "a filter column is entirely null", book


class TestNothingRaisesAndNothingLies(unittest.TestCase):

    def test_no_question_raises_on_any_shape(self):
        for name, book in _books():
            for question in QUESTIONS:
                with self.subTest(book=name, question=question):
                    try:
                        run_mi_agent_query(question, book, _SEMANTICS)
                    except Exception as exc:  # noqa: BLE001 - that is the assertion
                        self.fail(f"{type(exc).__name__}: {exc}")

    def test_a_refusal_says_why(self):
        for name, book in _books():
            for question in QUESTIONS:
                result = run_mi_agent_query(question, book, _SEMANTICS)
                if result.get("ok"):
                    continue
                with self.subTest(book=name, question=question):
                    self.assertTrue(
                        str(result.get("error") or "").strip(),
                        "refused with no explanation")

    def test_an_unfiltered_total_is_true_of_the_book_it_ran_over(self):
        """Where a whole-book figure comes back it equals the oracle's over the
        SAME frame — so a degenerate shape cannot quietly change what a total
        means."""
        for name, book in _books():
            result = run_mi_agent_query("What is the total balance?", book,
                                        _SEMANTICS)
            if not result.get("ok"):
                continue
            frame = result["query_result"].data
            if _SUM not in frame.columns:
                continue
            with self.subTest(book=name):
                self.assertAlmostEqual(float(frame[_SUM].sum()),
                                       float(book[BALANCE].sum(min_count=1) or 0.0),
                                       places=2)

    def test_a_breakdown_still_adds_up_when_values_are_missing(self):
        """The missing-value policy may bucket or exclude, but the parts must
        still account for the whole of what was measured."""
        book = truth.canonical_book()
        book.loc[book.index[:200], "collateral_geography"] = np.nan
        grouped = run_mi_agent_query("Total balance by region", book, _SEMANTICS)
        whole = run_mi_agent_query("What is the total balance?", book, _SEMANTICS)
        if not (grouped.get("ok") and whole.get("ok")):
            self.skipTest("this book refuses one of the two shapes")
        self.assertAlmostEqual(
            float(grouped["query_result"].data[_SUM].sum()),
            float(whole["query_result"].data[_SUM].sum()), places=2)


class TestTheSingleRegionBook(unittest.TestCase):
    """Pinned, because a defect lived here: a narrowing that every row satisfies
    was reported LOST, because "was it narrowed?" was answered by asking whether
    the row count went down."""

    def test_a_narrowing_every_row_satisfies_is_answered(self):
        book = truth.canonical_book()
        book["collateral_geography"] = "Scotland"
        result = run_mi_agent_query("What is the total balance in Scotland?",
                                    book, _SEMANTICS)
        self.assertTrue(result.get("ok"), result.get("error"))
        self.assertAlmostEqual(
            float(result["query_result"].data[_SUM].sum()),
            float(book[BALANCE].sum()), places=2)


if __name__ == "__main__":
    unittest.main()
