#!/usr/bin/env python3
"""A filter that excludes nothing was applied. It is not lost.

THE DEFECT, and this estate has already fixed it once.

On a book whose loans are all in Scotland, "what is the total balance in
Scotland?" was REFUSED:

    "I understood that you asked for Scotland, but that could not be applied to
     the calculation (Scotland (Region) — the geographic scope was not applied
     to the calculation). I have not substituted a broader figure."

The filter WAS applied. The executor recorded applying it. It simply excluded
no rows, because every row satisfied it — and the receipt decided whether a
narrowing had happened by asking whether the row count went DOWN.

    narrowed = rows_after < rows_before

That inference is wrong whenever a population happens to be the whole book, and
a single-region book is an ordinary thing: a regional subsidiary, a drilled
view, a small portfolio. The same question about a product ("for lump sum
loans") answers correctly on a book that is entirely lump sum, because the
THRESHOLD and narrowing branches were converted to read the executor's own
`applied_filter_fields` — evidence of what ran — when this class was found
before. The geographic branch was not, so one owner still infers where the
others observe.

THE RULE. Whether a narrowing was applied is a fact the executor reports. It is
never deduced from how many rows survived it.

A false refusal is not a wrong answer, and the fail-closed direction is the safe
one to be wrong in. It is still a capability the reader does not have, and it
fails precisely on the books where the narrowing matters least and the question
is most obviously answerable.
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

BALANCE = truth.BALANCE
_SUM = f"{BALANCE}_sum"


def _book_where_every_row_matches(column: str, value):
    book = truth.canonical_book()
    book[column] = value
    return book


class TestANarrowingThatKeepsEveryRowIsStillApplied(unittest.TestCase):

    #: ``(column, value, the question, the phrase the reader used)``. One of
    #: each governed narrowing kind, because the defect is one owner inferring
    #: where the others observe — a test on geography alone would not say that.
    CASES = (
        ("collateral_geography", "Scotland",
         "What is the total balance in Scotland?"),
        ("collateral_geography", "Scotland",
         "How many loans are there in Scotland?"),
        ("erm_product_type", "Lump Sum",
         "What is the total balance for lump sum loans?"),
        ("borrower_type", "Joint",
         "What is the total balance for joint borrowers?"),
        ("broker_channel", "Alpha",
         "What is the total balance for Alpha?"),
    )

    def test_the_question_is_answered_over_the_whole_of_that_book(self):
        for column, value, question in self.CASES:
            book = _book_where_every_row_matches(column, value)
            with self.subTest(question=question, column=column):
                result = run_mi_agent_query(question, book, _SEMANTICS)
                self.assertTrue(
                    result.get("ok"),
                    "a narrowing that every row satisfies was reported lost: "
                    f"{result.get('error')!r}")
                frame = result["query_result"].data
                if _SUM in frame.columns:
                    self.assertAlmostEqual(float(frame[_SUM].sum()),
                                           float(book[BALANCE].sum()), places=2)
                else:
                    self.assertEqual(int(frame["loan_count"].sum()), len(book))

    def test_a_narrowing_that_excludes_everything_is_still_refused(self):
        """The other side of the rule, so the repair cannot be a widening. A
        population the book does not contain must still refuse — it is nothing
        to calculate, not a whole-book figure."""
        book = _book_where_every_row_matches("collateral_geography", "Wales")
        result = run_mi_agent_query("What is the total balance in Scotland?",
                                    book, _SEMANTICS)
        self.assertFalse(result.get("ok"),
                         "an empty population returned a figure")

    def test_a_narrowing_that_is_genuinely_dropped_is_still_refused(self):
        """And a narrowing the estate cannot apply must still be disclosed, on a
        book where it excludes nothing. `number_of_borrowers` is a governed
        field this book does not carry, so the population cannot be applied and
        the reader must be told."""
        book = truth.canonical_book()
        result = run_mi_agent_query(
            "What is the total balance in Atlantis?", book, _SEMANTICS)
        self.assertFalse(result.get("ok"),
                         "a place the book does not carry was answered over the "
                         "whole book")


if __name__ == "__main__":
    unittest.main()
