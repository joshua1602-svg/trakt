#!/usr/bin/env python3
"""A population is not an axis, even when it is written next to one.

THE DEFECT, and it is the only silent wrong answer this sprint found.

    "Total balance by LTV for joint borrowers"

was answered as a breakdown by NUMBER OF BORROWERS. Not refused, not disclosed —
answered, `ok`, with the semantic guard reporting `ok`, as a single bar labelled
"2" carrying 100% of the joint book. The reader asked how their balance is
distributed across LTV and received a chart about how many people are on each
mortgage.

    "Total balance by LTV for joint borrowers"    -> by number_of_borrowers
    "Total balance by age for joint borrowers"    -> by number_of_borrowers
    "Total balance by LTV bucket for joint borrowers"  -> by ltv_bucket    (fine)
    "Total balance by region for joint borrowers"      -> by region        (fine)

It hid for two reasons. The estate's own books do not carry
`number_of_borrowers`, so the executor raised "not available in this dataset"
and the question refused — the right outcome for the wrong reason, and one that
disappears the moment a book carries the column, which is an ordinary thing for
a book to do since the field is in the registry. And it needs BOTH a bare
numeric axis (which the explicit-dimension reader leaves unresolved, so the
fallback below runs) AND a borrower-shaped population.

THE CAUSE. Where no dimension resolved, the fallback offered the whole text
after the last "by" as axis KEYWORDS:

    "by LTV for joint borrowers"  ->  keywords ('ltv', 'for', 'joint', 'borrowers')

and `find_field` matched "borrowers". The qualifier's words were competing to be
the axis. This estate already owns that boundary — `_axis_phrase` is the segment
without its qualifier, and the population resolver, the grouping splitter and
the value resolver all read it — so the fix is to consult the owner rather than
the raw text.

WHAT THIS FILE ASSERTS. Not that "borrowers" is excluded: that a QUALIFIER never
supplies the axis, whatever the qualifier says. The axis a question is answered
along must be the axis it names, and adding a population to a question must not
change it.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import mi_agent.execution_receipt as receipt                            # noqa: E402
from mi_agent.llm_query_parser import _deterministic_parse              # noqa: E402
from mi_agent.mi_agent_workflow import run_mi_agent_query               # noqa: E402
from mi_agent.mi_query_validator import load_mi_semantics               # noqa: E402
from mi_agent.tests import portfolio_truth_oracle as truth              # noqa: E402

_SEMANTICS = load_mi_semantics(
    str(_REPO_ROOT / "mi_agent" / "mi_semantics_field_registry.yaml"))
_BOOK = truth.canonical_book()
_COLUMNS = receipt.book_columns(_BOOK)
_VALUES = receipt.book_values(_BOOK, _SEMANTICS)


def parse(question: str):
    spec, _meta = _deterministic_parse(question, _SEMANTICS,
                                       available_columns=_COLUMNS,
                                       available_values=_VALUES)
    return spec


def axes(spec):
    return list(spec.dimensions or []) or ([spec.dimension] if spec.dimension else [])


#: Every axis word this bank exercises, with the governed axis it must resolve
#: to. The bare numeric ones are the shape that failed; the others are here so a
#: repair cannot fix one reading by breaking another.
AXES = (("ltv", "ltv_bucket"),
        ("ltv bucket", "ltv_bucket"),
        ("age", "age_bucket"),
        ("age bucket", "age_bucket"),
        ("region", "collateral_geography"),
        ("product", "erm_product_type"),
        ("broker", "broker_channel"))

#: Populations written as qualifiers. Each names words that ANOTHER governed
#: field would answer to, which is the whole point.
QUALIFIERS = ("for joint borrowers",
              "for borrowers over 75",
              "for single borrower loans",
              "in Scotland",
              "for lump sum loans")


class TestAQualifierNeverSuppliesTheAxis(unittest.TestCase):

    def test_the_axis_is_the_one_the_question_names(self):
        for word, expected in AXES:
            for qualifier in QUALIFIERS:
                question = f"Total balance by {word} {qualifier}"
                with self.subTest(question=question):
                    self.assertEqual(axes(parse(question)), [expected])

    def test_adding_a_population_does_not_change_the_axis(self):
        """Stated as a difference, so it holds for any axis the registry
        resolves rather than for the list above."""
        for word, _expected in AXES:
            plain = axes(parse(f"Total balance by {word}"))
            for qualifier in QUALIFIERS:
                with self.subTest(axis=word, qualifier=qualifier):
                    self.assertEqual(
                        axes(parse(f"Total balance by {word} {qualifier}")),
                        plain, "the population changed the axis")


class TestTheWrongAxisIsNotHiddenByAMissingColumn(unittest.TestCase):
    """The reason this reached production undetected: the substituted field is
    absent from the estate's books, so the executor refused and the defect
    looked like a data gap. This runs the same question over a book that DOES
    carry it — an ordinary thing for a book to do, since the field is in the
    registry — where a wrong axis is answered instead of refused.
    """

    @classmethod
    def setUpClass(cls):
        book = truth.canonical_book()
        book["number_of_borrowers"] = np.where(book["borrower_type"] == "Joint",
                                               2, 1)
        cls.book = book

    def test_the_answer_is_broken_down_by_the_axis_that_was_asked_for(self):
        for word, expected in (("ltv", "ltv_bucket"), ("age", "age_bucket")):
            question = f"Total balance by {word} for joint borrowers"
            with self.subTest(question=question):
                result = run_mi_agent_query(question, self.book, _SEMANTICS)
                self.assertTrue(result.get("ok"),
                                f"not answered: {result.get('error')!r}")
                frame = result["query_result"].data
                self.assertIn(expected, frame.columns,
                              f"answered along {list(frame.columns)}")
                self.assertNotIn("number_of_borrowers", frame.columns)
                self.assertGreater(len(frame), 1,
                                   "a breakdown with one bar is not a breakdown")


if __name__ == "__main__":
    unittest.main()
