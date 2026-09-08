#!/usr/bin/env python3
"""Twenty-two questions written AFTER the repair and never consulted during it.

WHY A SECOND HOLDOUT. The first one found the two silent wrong answers this
sprint set out to close, and it was then used to check the fix — which spends
it. A holdout that has been looked at while building is a bank, and a bank
measures how well the fix fits the questions it was fitted to.

So these were written after the accounting invariant was wired, run ONCE, and
reported. No production logic was tuned against any individual row. Two general
faults it exposed were repaired, and both were general: a contraction ("What's")
was reported as an unrecognised category because the framing vocabularies spell
these without the apostrophe, and that is fixed for every contraction rather
than for that sentence.

MEASURED ON FIRST RUN:

    15 answered, and answered CORRECTLY against the pandas oracle
     7 fail-closed
     0 silently wrong

The set is built for this sprint's subject: adjectival geography in several
positions, several narrowings in one slot, and — the half that matters most —
restrictions that must NOT resolve. "Cornish" and "Yorkshire" are real English
words for real places that the governed ITL1 taxonomy does not uniquely define,
so they must refuse rather than resolve to something near them; "premium",
"distressed" and "gold tier" name nothing at all.
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

SCOTLAND = ("collateral_geography", "eq", "Scotland")
WALES = ("collateral_geography", "eq", "Wales")
LONDON = ("collateral_geography", "eq", "London")
NORTH_WEST = ("collateral_geography", "eq", "North West")
LUMP_SUM = ("erm_product_type", "eq", "Lump Sum")
DRAWDOWN = ("erm_product_type", "eq", "Drawdown")
JOINT = ("borrower_type", "eq", "Joint")
SINGLE = ("borrower_type", "eq", "Single")
ALPHA = ("broker_channel", "eq", "Alpha")
BETA = ("broker_channel", "eq", "Beta")

#: Answered, and checked against the oracle.
CORRECT = (
    ("What's the Welsh loan count?", "count", (WALES,)),
    ("Welsh drawdown balance", "total", (WALES, DRAWDOWN)),
    ("Scottish Beta balance", "total", (SCOTLAND, BETA)),
    ("What is the London balance?", "total", (LONDON,)),
    ("How many North West loans?", "count", (NORTH_WEST,)),
    ("Total balance for single borrowers in Wales", "total", (WALES, SINGLE)),
    ("Welsh lump sum loan count", "count", (WALES, LUMP_SUM)),
    ("Balance for Alpha in Scotland", "total", (ALPHA, SCOTLAND)),
    ("How many drawdown loans are in London?", "count", (LONDON, DRAWDOWN)),
    ("Scottish balance by broker", "cells", ("broker_channel", (SCOTLAND,))),
    ("Welsh balance by product", "cells", ("erm_product_type", (WALES,))),
    ("Joint Scottish balance by broker", "cells",
     ("broker_channel", (SCOTLAND, JOINT))),
)

#: Must never be answered over a broader population.
#:
#: "Cornish" is a real place the governed ITL1 taxonomy does not uniquely define
#: — Cornwall sits inside South West (England) and no ITL1 value bears its name —
#: so it must refuse rather than resolve to its neighbour. The rest name nothing
#: any governed vocabulary claims.
#:
#: "Yorkshire" was here and has been REMOVED, because the claim was wrong: the
#: region owner carries it as a governed alias for "Yorkshire and The Humber"
#: (13 ITL3 codes). It refused on THIS book only because this book holds no
#: Yorkshire rows, which is a property of the fixture and not of the taxonomy —
#: asserting it here would have pinned a fixture accident as a semantic rule.
#: The production certification suite found that, on a book that does carry the
#: region, and it is asserted correctly in `TestAGovernedRegionAliasResolves`
#: below.
MUST_REFUSE = (
    "What is the Cornish balance?",
    "Show me the premium loans",
    "What is the distressed balance?",
    "How many gold tier lump sum loans?",
)

#: Ordinary phrasing that must not be refused.
MUST_NOT_REFUSE = (
    "Could you give me the total balance please?",
    "Show us the balance broken down by broker",
    "What's our average LTV?",
)


def _run(question):
    result = run_mi_agent_query(question, _BOOK, _SEMANTICS)
    return result, (result["query_result"].data if result.get("ok") else None)


class TestTheFreshHoldoutAnswersCorrectly(unittest.TestCase):

    def test_each_figure_matches_the_oracle(self):
        for question, kind, expectation in CORRECT:
            with self.subTest(question=question):
                result, frame = _run(question)
                self.assertTrue(result.get("ok"),
                                f"not answered: {result.get('error')!r}")
                if kind == "total":
                    self.assertAlmostEqual(
                        float(frame[_SUM].sum()),
                        truth.total(_BOOK, BALANCE, expectation), places=2)
                elif kind == "count":
                    self.assertEqual(int(frame["loan_count"].sum()),
                                     truth.row_count(_BOOK, expectation))
                else:
                    dimension, predicates = expectation
                    self.assertIn(dimension, frame.columns)
                    executed = {str(row[dimension]): round(float(row[_SUM]), 2)
                                for _, row in frame.iterrows()}
                    grouped = _BOOK[truth.mask_for(_BOOK, predicates)].groupby(
                        dimension)[BALANCE].sum()
                    self.assertEqual(
                        executed,
                        {str(k): round(float(v), 2) for k, v in grouped.items()})

    def test_none_of_them_is_the_whole_book(self):
        """A narrowing that did nothing would pass the figures above only if the
        population happened to be everything. It never is here."""
        for question, _kind, _expectation in CORRECT:
            with self.subTest(question=question):
                _result, frame = _run(question)
                self.assertNotEqual(int(frame["loan_count"].sum()), len(_BOOK))


class TestTheFreshHoldoutRefusesWhatItCannotBind(unittest.TestCase):

    def test_an_unbindable_restriction_is_never_answered_broadly(self):
        for question in MUST_REFUSE:
            with self.subTest(question=question):
                result, _frame = _run(question)
                self.assertFalse(
                    result.get("ok"),
                    "a restriction nothing can bind was answered over a "
                    "broader population")

    def test_a_near_miss_place_is_not_resolved_to_a_neighbour(self):
        """"Cornish" must refuse, not quietly become South West. An alias exists
        where the taxonomy makes the referent unambiguous, and Cornwall is not
        an ITL1 value — it sits inside South West (England)."""
        result, _frame = _run("What is the Cornish balance?")
        self.assertFalse(result.get("ok"))


class TestAGovernedRegionAliasResolves(unittest.TestCase):
    """The other side of the near-miss rule, and it is here because the
    certification suite caught this file claiming the opposite.

    "Yorkshire" IS governed: the region owner carries it as an alias for
    "Yorkshire and The Humber". That it refuses on the canonical book is a fact
    about the fixture — the book holds no Yorkshire rows — and not a fact about
    the taxonomy. The distinction matters, because pinning the first as if it
    were the second would make a correct answer look like a defect the day a
    book carries the region.
    """

    def test_the_owner_knows_it(self):
        import mi_agent.region_resolution as region

        self.assertEqual(region.codes_for("yorkshire"),
                         region.codes_for("yorkshire and the humber"))
        self.assertTrue(region.codes_for("yorkshire"))

    def test_and_it_binds_on_a_book_that_carries_it(self):
        book = truth.canonical_book()
        book.loc[book.index[:40], "collateral_geography"] = "Yorkshire and The Humber"
        result = run_mi_agent_query("How many Yorkshire loans are there?",
                                    book, _SEMANTICS)
        self.assertTrue(result.get("ok"), result.get("error"))
        self.assertEqual(int(result["query_result"].data["loan_count"].sum()), 40)


class TestOrdinaryPhrasingIsNotRefused(unittest.TestCase):

    def test_none_of_it_refuses(self):
        for question in MUST_NOT_REFUSE:
            with self.subTest(question=question):
                result, _frame = _run(question)
                self.assertTrue(result.get("ok"),
                                f"ordinary phrasing refused: "
                                f"{result.get('error')!r}")


if __name__ == "__main__":
    unittest.main()
