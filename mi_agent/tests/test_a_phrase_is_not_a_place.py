#!/usr/bin/env python3
"""Nothing becomes a place merely because nothing else claimed it.

WHERE THIS LIVES. `_parse_categorical_filter` has two endings. When the book's
own value catalogue is available it decides the field from the values the book
carries, and a phrase no governed field claims is RECORDED as unresolved rather
than bound — that ending is correct and is not touched here. When no catalogue
was supplied it fell through to:

    field = _preferred_region(...) or "geographic_region_obligor"
    return field, value.title()

— any captured phrase, bound to geography, in title case. Measured on the
882-question corpus: "What is the largest geographic concentration versus
limit?" binds `collateral_geography = 'Concentration Versus Limit'`. It was
invisible because a later rule dropped every filter whose field was the grouping
dimension; the moment that rule was narrowed to the case it was written for, an
invented geography appeared in a corpus question.

THE GOVERNED MAPPING ALREADY KNOWS WHAT A PLACE IS.
`region_resolution.looks_like_region_term` answers "is this a term the governed
ITL mapping knows, in any representation" — name, alias, ITL code or postcode —
and it was written for exactly this question and used nowhere. A term it knows
still binds with no catalogue, so every real place keeps working; a term it does
not know is not bound, and the narrowing is DISCLOSED rather than dropped, using
the same owner test the catalogued ending uses so the two agree about what
counts as an unrecognised category.

WHAT THIS IS NOT. It is not a vocabulary: this module adds no place names, and
asks the one owner that holds the mapping the canonical transformation itself
uses. And it is not a substitute for the catalogue — where the book's values are
available they still decide, first, exactly as before.
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from mi_agent import llm_query_parser as P
from mi_agent import region_resolution as RR
from mi_agent.mi_query_validator import load_mi_semantics

_SEMANTICS_PATH = _REPO_ROOT / "mi_agent" / "mi_semantics_field_registry.yaml"
_GEO = "collateral_geography"
_COLUMNS = {_GEO, "current_outstanding_balance", "current_loan_to_value"}

#: The corpus question that exposed it.
CONCENTRATION = "what is the largest geographic concentration versus limit?"


class _Fixture(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.semantics = load_mi_semantics(_SEMANTICS_PATH)

    def _parse(self, clause, values=None, notes=None):
        return P._parse_categorical_filter(
            clause, self.semantics, _COLUMNS, values, unresolved=notes)


class TestWithNoCatalogue(_Fixture):
    """The ending this changes: no book values were supplied."""

    def test_an_analytic_phrase_is_not_bound_to_geography(self):
        notes = []
        self.assertIsNone(self._parse(CONCENTRATION, notes=notes))

    def test_a_place_the_governed_mapping_knows_still_binds(self):
        """Every real place keeps working with no catalogue — that is what
        makes this a narrowing of the fallback rather than its removal."""
        for clause, place in (
                ("what is the total balance for loans in wales?", "Wales"),
                ("what is the average ltv in the south east?", "South East"),
                ("what is the total balance for loans in london?", "London")):
            with self.subTest(clause=clause):
                self.assertTrue(RR.looks_like_region_term(place))
                self.assertEqual(self._parse(clause), (_GEO, place))

    def test_a_place_the_mapping_does_not_know_is_disclosed_not_bound(self):
        """"Atlantis" used to bind, match nothing and refuse — the right answer
        for the wrong reason. It is now declined, and SAID: a narrowing dropped
        in silence is a whole-book figure answering a narrower question."""
        notes = []
        self.assertIsNone(self._parse("what is the average ltv in atlantis?",
                                      notes=notes))
        self.assertTrue(notes)
        self.assertIn("atlantis", " ".join(notes).lower())
        # And it REFUSES rather than warning — see the test below.
        self.assertEqual(P.unknown_category_names(notes), ["'atlantis'"])

    def test_the_disclosure_is_the_one_the_catalogued_ending_writes(self):
        """The same obstacle, the same sentence.

        A warning would not refuse, and the invented binding this replaces DID —
        it matched no rows. Removing it while writing a note that only warns
        would answer over the whole book for the first time. `unknown_category_
        refusal` exists so "a reader who asks the same question two ways cannot
        be told two different things about the same obstacle"; whether the
        caller passed the book's values is not a difference a reader should
        hear.
        """
        with_catalogue, without = [], []
        self._parse("what is the average ltv in atlantis?",
                    TestWithACatalogue.BOOK, notes=with_catalogue)
        self._parse("what is the average ltv in atlantis?", notes=without)
        self.assertEqual(P.unknown_category_names(without), ["'atlantis'"])
        self.assertEqual(without, with_catalogue)

class TestWithACatalogue(_Fixture):
    """The ending that already worked. Nothing here may move."""

    BOOK = {_GEO: ["Wales", "Scotland"], "erm_product_type": ["Lump Sum"]}

    def test_the_book_s_own_values_still_decide(self):
        self.assertEqual(
            self._parse("what is the total balance for loans in wales?",
                        self.BOOK), (_GEO, "Wales"))

    def test_a_value_the_book_does_not_carry_is_still_recorded(self):
        notes = []
        self.assertIsNone(self._parse("what is the average ltv in atlantis?",
                                      self.BOOK, notes=notes))
        self.assertEqual(P.unknown_category_names(notes), ["'atlantis'"])

    def test_a_product_type_is_still_never_geography(self):
        self.assertEqual(
            self._parse("what is the total balance for lump sum loans?",
                        self.BOOK), ("erm_product_type", "Lump Sum"))


class TestTheSpecItReaches(_Fixture):
    """At the spec, which is where a filter becomes an answer.

    A GROUPED question no longer shows this — the grouped-filter rule now keeps
    a restriction on the axis only when its values are the book's own, so an
    invented one is dropped there. An UNGROUPED question has no such rule in
    front of it, which is why the spec is asserted on one.
    """

    def test_an_unknown_place_reaches_no_spec_filter(self):
        spec, _meta = P._deterministic_parse(
            "what is the average ltv in atlantis?", self.semantics,
            available_columns=_COLUMNS)
        self.assertEqual((spec.to_dict().get("filters") or {}) if spec else {}, {})

    def test_a_known_place_still_reaches_the_spec(self):
        spec, _meta = P._deterministic_parse(
            "what is the average ltv in wales?", self.semantics,
            available_columns=_COLUMNS)
        self.assertEqual((spec.to_dict().get("filters") or {}) if spec else {},
                         {_GEO: "Wales"})


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
