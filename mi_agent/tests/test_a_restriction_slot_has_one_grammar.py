#!/usr/bin/env python3
"""Every restriction a question states, wherever it stands and whatever it modifies.

TWO GRAMMAR DEFECTS, one sentence apart, and together they are why an
unrecognised population could vanish in silence.

D — THE SLOT ONLY EXISTED IN FRONT OF A ROW NOUN.

    "Scottish loans"    the attributive scan anchors on "loans", offers
                        "scottish" to the resolvers, and the estate refuses
                        honestly with `unknown category: 'scottish'`
    "Scottish balance"  no row noun, no anchor, nothing offered to anything —
                        so the whole book came back with no disclosure at all

A reader restricts a MEASURE as readily as a row: "Scottish balance", "joint
borrower exposure", "lump sum lending". The slot is the run of modifiers before
the head noun, and whether that head names a row or a measure is not a
difference the reader can be expected to know about.

E — THE SCAN STOPPED AT ITS FIRST SUCCESS.

    "Scottish lump sum loans"

resolves "lump sum", returns, and never looks at the words in front of it. The
answer covered every Lump Sum loan in the book — 195 where 45 was asked for —
and "Scottish" left no trace: not applied, not disclosed, not recorded. A
question may state several independent narrowings, and the owner whose docstring
says "EVERY categorical narrowing the text states, not the last one" is the one
that must collect them.

AND THE REGION OWNER IS ASKED, at last. `region_resolution.resolve` already maps
an alias through the governed ITL ladder onto whatever the book actually stores —
`resolve("scottish", ["London", "North West", "Scotland", "Wales"])` returns
`["Scotland"]` — and the executor has always used it. The population resolver
never did, so a term the estate's own region owner could resolve was treated as
a category the book does not carry. Asking it is one owner consultation, not a
second geography system.

ONE GRAMMAR, TWO READERS. Where the restriction slots are is a fact about the
question, so it is owned by the lexical layer and read by both the parser (which
resolves what stands in them) and the semantic-accounting layer (which checks
that something did). Two copies of that boundary is the defect this estate has
paid for more than once.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

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

BALANCE = truth.BALANCE
_SUM = f"{BALANCE}_sum"

SCOTLAND = ("collateral_geography", "eq", "Scotland")
LUMP_SUM = ("erm_product_type", "eq", "Lump Sum")
JOINT = ("borrower_type", "eq", "Joint")


def parse(question: str):
    spec, _meta = _deterministic_parse(question, _SEMANTICS,
                                       available_columns=_COLUMNS,
                                       available_values=_VALUES)
    return spec


class TestARestrictionInFrontOfAMeasure(unittest.TestCase):
    """D. The head noun may name a measure."""

    def test_an_adjectival_region_before_a_measure_narrows(self):
        for question in ("Give me the Scottish balance.",
                         "What is the Scottish balance?",
                         "Scottish balance"):
            with self.subTest(question=question):
                spec = parse(question)
                self.assertEqual((spec.filters or {}).get("collateral_geography"),
                                 "Scotland",
                                 "the restriction in front of the measure was lost")

    def test_it_means_what_the_prepositional_form_means(self):
        """The two ways of saying it are one request."""
        for adjectival, prepositional in (
                ("Scottish balance", "balance in Scotland"),
                ("Welsh balance", "balance in Wales")):
            with self.subTest(adjectival=adjectival):
                self.assertEqual((parse(adjectival).filters or {}),
                                 (parse(prepositional).filters or {}))


class TestSeveralRestrictionsInOneSlot(unittest.TestCase):
    """E. The scan collects every narrowing, not the first one it resolves."""

    def test_both_narrowings_survive(self):
        spec = parse("How many Scottish lump sum loans are there?")
        self.assertEqual((spec.filters or {}).get("collateral_geography"),
                         "Scotland")
        self.assertEqual((spec.filters or {}).get("erm_product_type"), "Lump Sum")

    def test_the_order_they_are_written_in_does_not_matter(self):
        first = parse("How many Scottish lump sum loans are there?")
        second = parse("How many lump sum Scottish loans are there?")
        self.assertEqual(first.filters or {}, second.filters or {})

    def test_three_narrowings_in_one_slot(self):
        spec = parse("How many joint Scottish lump sum loans are there?")
        self.assertEqual((spec.filters or {}).get("collateral_geography"),
                         "Scotland")
        self.assertEqual((spec.filters or {}).get("erm_product_type"), "Lump Sum")
        self.assertEqual((spec.filters or {}).get("borrower_type"), "Joint")


class TestTheNumbersAreRight(unittest.TestCase):
    """Independently computed, because reading the grammar correctly and
    computing the population correctly are two different claims."""

    CASES = (("Give me the Scottish balance.", "total", (SCOTLAND,)),
             ("How many Scottish lump sum loans are there?", "count",
              (SCOTLAND, LUMP_SUM)),
             ("What is the Scottish balance for joint borrowers?", "total",
              (SCOTLAND, JOINT)))

    def test_each_matches_the_oracle(self):
        for question, kind, predicates in self.CASES:
            with self.subTest(question=question):
                result = run_mi_agent_query(question, _BOOK, _SEMANTICS)
                self.assertTrue(result.get("ok"),
                                f"not answered: {result.get('error')!r}")
                frame = result["query_result"].data
                if kind == "total":
                    self.assertAlmostEqual(
                        float(frame[_SUM].sum()),
                        truth.total(_BOOK, BALANCE, predicates), places=2)
                else:
                    self.assertEqual(int(frame["loan_count"].sum()),
                                     truth.row_count(_BOOK, predicates))
                self.assertNotEqual(int(frame["loan_count"].sum()), len(_BOOK),
                                    "the answer is still the whole book's")


class TestNothingElseBecomesANarrowing(unittest.TestCase):
    """The guard. Widening where a restriction may stand must not make ordinary
    words into populations."""

    def test_ordinary_requests_narrow_nothing(self):
        for question in ("show me the total balance", "give me the balance",
                         "what is the average loan size?",
                         "How many loans are there?", "Total balance by region",
                         "please show the total balance"):
            with self.subTest(question=question):
                self.assertEqual(parse(question).filters or {}, {})

    def test_a_two_letter_postcode_area_is_not_a_place_here(self):
        """The region ladder answers TRUE for "me" (Medway) and "so"
        (Southampton). Consulting that owner for a restriction slot must never
        turn "give ME the balance" into a Medway question."""
        for question in ("give me the balance", "so what is the balance",
                         "show me the total balance"):
            with self.subTest(question=question):
                filters = parse(question).filters or {}
                self.assertNotIn("collateral_geography", filters)

    def test_an_unrecognised_restriction_still_refuses(self):
        """"Platinum" names no governed value, and the estate's existing refusal
        for it must survive the widening."""
        result = run_mi_agent_query("How many platinum loans do we have?",
                                    _BOOK, _SEMANTICS)
        self.assertFalse(result.get("ok"))


if __name__ == "__main__":
    unittest.main()
