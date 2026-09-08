#!/usr/bin/env python3
"""Material analytical language cannot silently disappear from a successful answer.

THE HOLE. The estate fails closed on a requested population it has first
NOTICED. Every requested-versus-executed guard compares something the question
STATED with something the execution DID — and a qualifier no owner recognised
states nothing, so there is nothing to reconcile:

    user states a material restriction
        -> no owner recognises it
        -> no canonical facet exists
        -> the guards have nothing to compare
        -> a whole-book figure is returned, confidently, in silence

"Give me the Scottish balance." returned £115,450,800.70 where £25,405,654.23
was asked for, `ok`, with no disclosure of any kind. Its sibling "How many
Scottish loans are there?" refused correctly, and the only difference between
them was that the second had a row noun for the residue scan to anchor on.

THE INVARIANT. A normal successful analytical response is allowed only where
every semantically material part of the request is either bound to a governed
role, claimed as structural language by an owner, or surfaced as unresolved —
in which case ordinary success is prevented.

NOT A SECOND PARSER, and that is the design. The layer decides nothing about
what a question means. It asks the owners that already ship — the value
catalogue, the region ladder, the measure, statistic, dimension, period, scope,
seasoning, threshold and framing vocabularies — which SPANS they claimed, and
looks at what is left standing in a restriction position. It never resolves a
population of its own; residue is recorded as an unresolved category in the
estate's existing vocabulary, so the refusal has one owner rather than two.

    ONE semantic interpretation  +  ONE completeness check

POSITION, NOT PLAUSIBILITY. Materiality is decided by grammar. A word standing
attributively before a row noun or a measure noun is where a governed
restriction goes; a word elsewhere is not. That is what lets this be strict
without being fuzzy — and it matters, because the region ladder answers TRUE for
"me" and "so" (ME is Medway, SO is Southampton). A rule that asked "could this
be a place?" would refuse "give me the balance". This one never asks.

MEASURED BEFORE IT WAS WIRED. Over the estate's own 882-question corpus the
layer reports residue on 10, and every one of those already refuses for another
reason: wiring it changed no answer that was previously given. What it changed
is that the two blind spots — a restriction before a MEASURE noun, and a
resolvable category masking an unresolvable one — no longer let a population
vanish.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import mi_agent.execution_receipt as receipt                            # noqa: E402
from mi_agent.mi_agent_workflow import run_mi_agent_query               # noqa: E402
from mi_agent.mi_query_validator import load_mi_semantics               # noqa: E402
from mi_agent.tests import portfolio_truth_oracle as truth              # noqa: E402
from question_interpretation import semantic_accounting as accounting   # noqa: E402

_SEMANTICS = load_mi_semantics(
    str(_REPO_ROOT / "mi_agent" / "mi_semantics_field_registry.yaml"))
_BOOK = truth.canonical_book()
_COLUMNS = receipt.book_columns(_BOOK)
_VALUES = receipt.book_values(_BOOK, _SEMANTICS)


def residue(question: str):
    return [r.text for r in accounting.material_residue(
        question, _SEMANTICS, available_columns=_COLUMNS,
        available_values=_VALUES)]


def answered(question: str) -> bool:
    return bool(run_mi_agent_query(question, _BOOK, _SEMANTICS).get("ok"))


class TestUnresolvableLanguagePreventsOrdinarySuccess(unittest.TestCase):
    """"Platinum" names no governed value under any owner — not the catalogue,
    not the registry, not the region ladder. It must be refused in every
    position a restriction can stand."""

    POSITIONS = (
        ("before a row noun", "How many platinum loans do we have?"),
        ("before a measure noun", "What is the platinum balance?"),
        ("beside a category that resolves",
         "How many platinum lump sum loans are there?"),
        ("before a measure, with a grouping",
         "Show platinum balance by region."),
    )

    def test_it_is_reported_as_residue(self):
        for position, question in self.POSITIONS:
            with self.subTest(position=position):
                self.assertIn("platinum", residue(question))

    def test_it_is_never_answered_over_a_broader_population(self):
        for position, question in self.POSITIONS:
            with self.subTest(position=position):
                self.assertFalse(answered(question),
                                 "a whole-book figure answered a narrower "
                                 "question")


class TestOrdinaryEnglishIsNotResidue(unittest.TestCase):
    """The negative controls, and they are the reason this can be strict.

    An accounting layer that refused ordinary prose would be worse than the
    defect it closes. Every phrase here is claimed by an OWNER — articles,
    auxiliaries, presentation verbs, aggregation and chart words, the estate's
    own analytical idiom — not by an ignore list.
    """

    ORDINARY = (
        "show me the total balance",
        "give me the balance",
        "please show the total balance",
        "what is the average loan size?",
        "How many loans are there?",
        "Can you show me the balance by region?",
        "I need the total balance for joint borrowers",
        "Total balance by region for joint borrowers",
        "balance in Scotland for lump sum loans",
        "For joint borrowers, chart balance by LTV by age",
        "WA LTV",
        "weighted average LTV by region",
        "original balance by region",
        "borrower age by region",
        "What is the total collateral valuation",
        "total property valuation",
        "What is the current funded balance?",
        "how much balance is above 60% LTV",
        "How many loans are in the 60-70% LTV bucket?",
        "Show loans with current balance above current valuation.",
        "What is the forecast loan count?",
        "When do we reach £25m funded balance?",
        "How complete is LTV?",
        "compare current funded balance to expected funded",
        "Drill into South East loans.",
        "Show monthly balance evolution by broker.",
        "If the current pipeline converts as expected, what will our funded "
        "balance be?",
    )

    def test_none_of_it_is_material_residue(self):
        for question in self.ORDINARY:
            with self.subTest(question=question):
                self.assertEqual(residue(question), [],
                                 "ordinary language was reported as an "
                                 "unaccounted analytical restriction")


class TestATwoLetterPostcodeIsNotAPlace(unittest.TestCase):
    """The trap this design exists to avoid. The governed region ladder resolves
    POSTCODE AREAS, so `codes_for("me")` and `codes_for("so")` are non-empty —
    Medway and Southampton. Any layer that asked "might this word be a place?"
    would refuse "give me the balance"."""

    def test_the_ladder_really_does_resolve_them(self):
        import mi_agent.region_resolution as region

        self.assertTrue(region.codes_for("me"))
        self.assertTrue(region.codes_for("so"))

    def test_and_none_of_them_becomes_residue_or_a_filter(self):
        for question in ("give me the balance", "show me the total balance",
                         "so how many loans do we have",
                         "can you give me the balance by region"):
            with self.subTest(question=question):
                self.assertEqual(residue(question), [])
                result = run_mi_agent_query(question, _BOOK, _SEMANTICS)
                if result.get("ok"):
                    spec = result.get("spec") or {}
                    self.assertNotIn("collateral_geography",
                                     spec.get("filters") or {})


class TestItValidatesRatherThanInterprets(unittest.TestCase):
    """§10 — the layer must not become a second parser.

    It reports SPANS nobody claimed. It never says what they mean, and it never
    supplies a population: "Scottish balance" is resolved by the parser through
    the governed region owner, and by the time this layer looks there is nothing
    left to report.
    """

    def test_it_reports_nothing_once_an_owner_has_claimed_the_span(self):
        for question in ("Give me the Scottish balance.",
                         "How many Scottish lump sum loans are there?",
                         "What is the Welsh balance?"):
            with self.subTest(question=question):
                self.assertEqual(residue(question), [])

    def test_it_returns_text_not_meaning(self):
        """A residue names the span and where it stood. It carries no field, no
        value and no resolution — there is nothing in it for a caller to bind."""
        found = accounting.material_residue(
            "What is the platinum balance?", _SEMANTICS,
            available_columns=_COLUMNS, available_values=_VALUES)
        self.assertEqual(len(found), 1)
        self.assertEqual(found[0].text, "platinum")
        self.assertEqual(found[0].position, "measure")
        self.assertEqual(len(found[0]), 3, "a residue is (text, head, position)")


class TestTheTwoLayersAreComplementary(unittest.TestCase):
    """§15 — the existing requested-versus-executed guards keep their job.

    Semantic accounting catches what was lost AT INTERPRETATION; reconciliation
    catches what was lost AFTER it. Neither replaces the other, and this pins
    both still firing.
    """

    def test_a_recognised_filter_that_cannot_be_applied_is_still_caught(self):
        """Lost after interpretation: the value is understood, the book does not
        carry the field."""
        book = _BOOK.drop(columns=["borrower_type"])
        result = run_mi_agent_query(
            "What is the total balance for joint borrowers?", book, _SEMANTICS)
        self.assertFalse(result.get("ok"))

    def test_an_unrecognised_qualifier_is_caught_at_interpretation(self):
        """Lost at interpretation: nothing understood the word at all."""
        self.assertFalse(answered("What is the platinum balance?"))


if __name__ == "__main__":
    unittest.main()
