#!/usr/bin/env python3
"""A narrowing stated about ONE output may not narrow the others.

THE DEFECT, measured:

    "How many joint loans are there, what is their balance,
     and how much of that balance has LTV above 40%?"

        measures  [loan_count/count, balance/sum]
        filters   {borrower_type: Joint, current_loan_to_value: >40}

Two things are wrong and the second is worse than the first. The third output
does not exist — the measure set dedupes by field, so "that balance ... above
40%" collapses into the same balance already requested. And the LTV bound,
which qualifies only the third clause, was applied to the WHOLE request: the
count and the balance are computed over joint loans above 40% LTV, which is not
what either clause asked. The reader is given three figures, two of them
silently narrowed, with nothing in the envelope to say so.

WHAT SEPARATES A LOCAL PREDICATE FROM A SHARED ONE. Not position — both of
these state their bound in the final clause:

    "What is the balance and weighted average LTV of loans above 6%?"
        → above 6% qualifies the POPULATION both outputs describe.  SHARED

    "How many joint loans are there, what is their balance, and how much
     of that balance has LTV above 40%?"
        → above 40% qualifies a NEW output carved out of a prior one.  LOCAL

The difference is the BACK-REFERENCE. "of that balance", "of those loans", "of
the £38m" name a figure or population the request has already established, and
a clause that opens by naming one is asking a further question ABOUT it rather
than adding a condition to it. Without the back-reference the bound has nothing
to attach to but the whole request, which is why the first sentence is shared.

That reading is not local to same-turn composition. "Of that balance, how much
is above 80% LTV?" is the same sentence split across two turns, and §13 of the
sprint brief requires one population model for both — so the back-reference
vocabulary lives in `question_interpretation.lexical` with the comparators and
the count, and the conversational work reads the same owner.

WHAT THIS FILE PINS, AND WHAT IT DELIBERATELY DOES NOT. It pins the invariant:
a clause-local predicate never reaches another output's population. Executing
the outputs under DIFFERENT populations is the QueryPlan work and is not built
here; until it is, a request that needs it is REFUSED rather than answered over
a silently narrowed book. §19 is explicit that an honest governed refusal beats
a widened answer, and this is the case it was written for.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from mi_agent.llm_query_parser import _deterministic_parse       # noqa: E402
from mi_agent.mi_query_validator import load_mi_semantics        # noqa: E402

_SEMANTICS = load_mi_semantics(
    str(_REPO_ROOT / "mi_agent" / "mi_semantics_field_registry.yaml"))

LTV = "current_loan_to_value"
RATE = "current_interest_rate"


def parse(question: str):
    return _deterministic_parse(question, _SEMANTICS)


class TestABackReferenceMarksAClauseLocalNarrowing(unittest.TestCase):
    """The discriminator, on its own, across several fields and both frames."""

    LOCAL = (
        "How many joint loans are there, what is their balance, and how much "
        "of that balance has LTV above 40%?",
        "How many funded loans are there, what is the balance, and how much "
        "of that balance is on loans with a rate above 6%?",
        "How many pipeline cases are there, what is the amount, and how much "
        "of those cases have an LTV above 50%?",
    )

    SHARED = (
        "What is the balance and weighted average LTV of loans with an "
        "interest rate above 6%?",
        "Give me the loan count and balance for loans with an LTV above 40%.",
        "For joint borrowers, give me the funded loan count and balance.",
    )

    def test_a_local_narrowing_never_reaches_the_shared_population(self):
        for question in self.LOCAL:
            with self.subTest(question=question[:60]):
                spec, _meta = parse(question)
                for field in (LTV, RATE):
                    self.assertNotIn(
                        field, spec.filters or {},
                        "a clause-local bound narrowed every output")

    def test_a_shared_narrowing_still_reaches_the_population(self):
        """The other half. A rule that simply dropped trailing predicates would
        satisfy the class above and destroy the composition that works."""
        for question in self.SHARED:
            with self.subTest(question=question[:60]):
                spec, _meta = parse(question)
                if "above 6%" in question or "above 40%" in question:
                    self.assertTrue(
                        set(spec.filters or {}) & {LTV, RATE},
                        "a shared bound was dropped")


class TestTheRequestIsRefusedRatherThanWidened(unittest.TestCase):
    """Until the outputs can be executed under different populations, a request
    that needs it must not be answered over one."""

    def test_a_clause_local_request_declines(self):
        spec, meta = parse(
            "How many joint loans are there, what is their balance, and how "
            "much of that balance has LTV above 40%?")
        self.assertEqual((meta or {}).get("note"), "clause_local_unsupported")
        self.assertFalse(spec.measures)

    def test_the_refusal_says_what_it_could_not_do(self):
        spec, _meta = parse(
            "How many joint loans are there, what is their balance, and how "
            "much of that balance has LTV above 40%?")
        explanation = (spec.explanation or "").lower()
        self.assertIn("population", explanation)


class TestTheOwnerIsShared(unittest.TestCase):
    """§13: same-turn and multi-turn read one back-reference vocabulary."""

    def test_the_vocabulary_recognises_a_reference_to_a_prior_result(self):
        from question_interpretation.lexical import refers_to_prior_result

        for text in ("of that balance", "of those loans", "of that amount",
                     "of the £38m", "of those cases", "that population"):
            with self.subTest(text=text):
                self.assertTrue(refers_to_prior_result(text))

    def test_it_does_not_fire_on_an_ordinary_narrowing(self):
        from question_interpretation.lexical import refers_to_prior_result

        for text in ("of loans above 6%", "for joint borrowers",
                     "with an LTV above 40%", "in Scotland"):
            with self.subTest(text=text):
                self.assertFalse(refers_to_prior_result(text))


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
