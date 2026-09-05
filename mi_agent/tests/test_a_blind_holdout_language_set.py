#!/usr/bin/env python3
"""Questions written WITHOUT looking at the corpus, and what happened to them.

WHY A HOLDOUT. Every bank in this estate was written after a defect, so every
bank is shaped by the defects already found. The corpus has the same problem: it
is the language the estate has already been taught. A capability measured only
against it reports how well the product answers questions it has seen the shape
of, which is not the question anyone is actually asking.

So these twenty-eight are ordinary MI requests phrased the way a person phrases
them at a desk — contractions, possessives, demonyms, "split", "spread",
"what's on the book" — and each carries an expectation computed by the pandas
oracle rather than by the product. Measured when written:

    17 answered, and answered CORRECTLY
     9 refused, every one of them fail-closed and honestly explained
     2 answered WRONGLY, in silence

THE TWO WRONG ONES ARE THE FINDING, and they are recorded here as expected
failures rather than deleted or quietly fixed, so the gap is executable:

    "Give me the Scottish balance."
        returns the WHOLE BOOK where Scotland was asked for. `ok`, no
        disclosure, no substitution warning.

    "How many Scottish lump sum loans are there?"
        applies Lump Sum, drops Scottish, answers 195 where 45 was asked for.

WHAT THEY MEAN, and it is the most important sentence in this file. The
fail-closed guarantee is not universal. It holds for a population the estate
can SEE — a value it recognises, or a word standing where an unrecognised
category would be noticed. A population it cannot see at all is not disclosed,
because there is nothing to disclose. "How many Scottish loans are there?"
refuses correctly with `unknown category: 'scottish'`; add one resolvable
category beside it, or take away the row noun that anchors the scan, and the
same word disappears silently.

Not repaired here deliberately. The repair is either new vocabulary (a demonym
table the registry does not carry) or a widening of the residue rule that has
already been measured refusing three working questions. Both need their own
evidence chain, and a rushed one is how a silent wrong answer becomes two.
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

JOINT = ("borrower_type", "eq", "Joint")
SCOTLAND = ("collateral_geography", "eq", "Scotland")
LUMP_SUM = ("erm_product_type", "eq", "Lump Sum")
DRAWDOWN = ("erm_product_type", "eq", "Drawdown")
ALPHA = ("broker_channel", "eq", "Alpha")
LTV_OVER_50 = (truth.LTV, "gt", 50.0)
OVER_75 = (truth.AGE, "gt", 75)

#: ``(question, kind, expectation)``. "total" and "count" carry the oracle's
#: predicates; "cells" carries ``(dimension, predicates)``; "any" only requires
#: an answer, for shapes whose exact figure another bank already owns.
ANSWERED_CORRECTLY = (
    ("How big is the book?", "any", None),
    ("How much have we lent to borrowers over 75?", "total", (OVER_75,)),
    ("What is our exposure to Scotland?", "total", (SCOTLAND,)),
    ("How many mortgages are on the book?", "count", ()),
    ("Number of joint cases.", "count", (JOINT,)),
    ("Loan count for Drawdown.", "count", (DRAWDOWN,)),
    ("Break the balance down by region.", "cells", ("collateral_geography", ())),
    ("Balance per product type.", "cells", ("erm_product_type", ())),
    ("Show me the regional split of the joint book.", "cells",
     ("collateral_geography", (JOINT,))),
    ("For lump sum, balance by region.", "cells",
     ("collateral_geography", (LUMP_SUM,))),
    ("Loan counts by product for joint borrowers.", "any", None),
    ("Balance grouped by LTV band.", "cells", ("ltv_bucket", ())),
    ("Balance per age band for joint borrowers.", "cells",
     ("age_bucket", (JOINT,))),
    ("Balance by region and product.", "any", None),
    ("Balance across region and broker for joint borrowers.", "any", None),
    ("What's the average loan size?", "any", None),
    ("Average balance in Scotland.", "any", None),
    # The two that used to be answered wrongly, in silence.
    ("Give me the Scottish balance.", "total", (SCOTLAND,)),
    ("How many Scottish lump sum loans are there?", "count",
     (SCOTLAND, LUMP_SUM)),
)

#: Refused, and every one of them fail-closed with an explanation naming what
#: could not be applied. A refusal is a legitimate reading; what it must never
#: be is a number.
REFUSED = (
    "What are we lending in Scotland?",
    "What's on the book for joint applicants?",
    "Total lending through Alpha.",
    "Balance where LTV exceeds 50%.",
    "Split lending across brokers.",
    "Balance by broker, Scotland only.",
    "How is the Scottish book spread across brokers?",
    "Balance for joint borrowers in Scotland on lump sum.",
    "Alpha's joint lending.",
)

#: RETIRED. Both were answered, `ok`, and wrong, in silence — the whole book for
#: "Scottish balance", and 195 loans for "Scottish lump sum" where 45 was asked
#: for. They are here as history rather than as expectations: three independent
#: causes had to be repaired before either could answer, and each has its own
#: bank —
#:
#:   the governed region vocabulary carried no adjectival form of Scotland
#:       (test_adjectival_region_names)
#:   a restriction slot existed only in front of a ROW noun, so "Scottish
#:   balance" offered its adjective to nothing at all, and the scan stopped at
#:   its first success, so "lump sum" masked "Scottish"
#:       (test_a_restriction_slot_has_one_grammar)
#:
#: They now sit in ANSWERED_CORRECTLY above, checked cell for cell against the
#: oracle like every other row.
FORMERLY_SILENTLY_WRONG = (
    "Give me the Scottish balance.",
    "How many Scottish lump sum loans are there?",
)


def _run(question):
    result = run_mi_agent_query(question, _BOOK, _SEMANTICS)
    return result, (result["query_result"].data if result.get("ok") else None)


def _matches(frame, kind, expectation) -> bool:
    if kind == "any":
        return True
    if kind == "total":
        return (_SUM in frame.columns
                and abs(float(frame[_SUM].sum())
                        - truth.total(_BOOK, BALANCE, expectation)) < 0.01)
    if kind == "count":
        return ("loan_count" in frame.columns
                and int(frame["loan_count"].sum())
                == truth.row_count(_BOOK, expectation))
    dimension, predicates = expectation
    if dimension not in frame.columns:
        return False
    executed = {str(row[dimension]): round(float(row[_SUM]), 2)
                for _, row in frame.iterrows()}
    grouped = _BOOK[truth.mask_for(_BOOK, predicates)].groupby(dimension)[BALANCE]
    return executed == {str(k): round(float(v), 2) for k, v in grouped.sum().items()}


class TestTheHoldoutStillAnswers(unittest.TestCase):
    """The seventeen that work. A regression here is a capability lost to
    language the estate was never tuned on, which is the only kind that
    generalises."""

    def test_each_is_answered_and_the_figure_is_right(self):
        for question, kind, expectation in ANSWERED_CORRECTLY:
            with self.subTest(question=question):
                result, frame = _run(question)
                self.assertTrue(result.get("ok"),
                                f"lost: {result.get('error')!r}")
                self.assertTrue(_matches(frame, kind, expectation),
                                "answered, with the wrong figure")


class TestTheRefusalsAreRefusals(unittest.TestCase):
    """The nine that refuse. They may become answers — that is an improvement,
    and this test says so rather than pinning them shut — but a refusal must
    never quietly become a WRONG answer, so each is checked against the oracle
    if it starts answering."""

    #: Where an expectation exists for a refused question, so an improvement can
    #: be told from a silent regression.
    EXPECTATIONS = {
        "What are we lending in Scotland?": ("total", (SCOTLAND,)),
        "What's on the book for joint applicants?": ("total", (JOINT,)),
        "Total lending through Alpha.": ("total", (ALPHA,)),
        "Balance where LTV exceeds 50%.": ("total", (LTV_OVER_50,)),
        "Balance by broker, Scotland only.": ("cells",
                                              ("broker_channel", (SCOTLAND,))),
        "How is the Scottish book spread across brokers?":
            ("cells", ("broker_channel", (SCOTLAND,))),
        "Balance for joint borrowers in Scotland on lump sum.":
            ("total", (JOINT, SCOTLAND, LUMP_SUM)),
        "Alpha's joint lending.": ("total", (ALPHA, JOINT)),
    }

    def test_a_refusal_never_becomes_a_wrong_answer(self):
        for question in REFUSED:
            with self.subTest(question=question):
                result, frame = _run(question)
                if not result.get("ok"):
                    continue                      # still refused: fail-closed
                kind, expectation = self.EXPECTATIONS.get(question,
                                                          ("any", None))
                self.assertTrue(
                    _matches(frame, kind, expectation),
                    "a question that used to refuse now answers, wrongly")


class TestTheTwoThatWereSilentlyWrong(unittest.TestCase):
    """The primary acceptance target of the repair, stated as figures.

    Both used to answer `ok` over a broader population with no disclosure of any
    kind. What makes them a fix rather than two patched sentences is that
    neither question, nor any word in it, appears in the production logic: the
    region owner gained three adjectives derived from its own ITL1 taxonomy, and
    the restriction-slot grammar gained a measure head and lost its first-hit
    behaviour.
    """

    def test_neither_is_answered_over_the_whole_book(self):
        for question in FORMERLY_SILENTLY_WRONG:
            with self.subTest(question=question):
                result, frame = _run(question)
                if not result.get("ok"):
                    continue                       # refusing would also be safe
                self.assertNotEqual(
                    int(frame["loan_count"].sum()), len(_BOOK),
                    "still answering over the whole book")


class TestTheDisclosureCapabilitySurvives(unittest.TestCase):
    """The repair resolves more language; it must not resolve language it cannot
    justify. "Platinum" names no governed value under any owner — not the
    catalogue, not the registry, not the region ladder — so it must still refuse
    rather than quietly widen to the book."""

    def test_an_unresolvable_restriction_in_front_of_a_row_noun_refuses(self):
        """The position the estate's residue scan has always covered."""
        result, _frame = _run("How many platinum loans do we have?")
        self.assertFalse(result.get("ok"),
                         "an unrecognised category was answered over the book")

    #: THE CLASS IS NOT CLOSED, only its two known instances.
    #:
    #: Resolving "Scottish" fixed the two questions. It did not fix the two
    #: BLIND SPOTS that let them fail silently, and an unresolvable term still
    #: finds both:
    #:
    #:   "What is the platinum balance?"          measure head, no row noun,
    #:                                            so nothing is offered to any
    #:                                            resolver and nothing is
    #:                                            reported
    #:   "How many platinum lump sum loans?"      the RESIDUE scan still stops
    #:                                            at its first success, so a
    #:                                            resolvable category masks an
    #:                                            unresolvable one
    #:
    #: Resolution and disclosure are separate machinery, and only resolution was
    #: repaired here. Closing the class is the semantic-accounting invariant's
    #: job; these two rows are its acceptance test, and they become unexpected
    #: successes when it lands.
    UNCLOSED = ("What is the platinum balance?",
                "How many platinum lump sum loans are there?")

    @unittest.expectedFailure
    def test_an_unresolvable_restriction_in_other_positions_refuses(self):
        for question in self.UNCLOSED:
            result, _frame = _run(question)
            self.assertFalse(
                result.get("ok"),
                f"{question!r} was answered over a broader population")


if __name__ == "__main__":
    unittest.main()
