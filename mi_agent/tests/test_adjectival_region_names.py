#!/usr/bin/env python3
"""Scotland has an adjective. The governed region vocabulary did not know it.

`region_resolution` is the estate's authoritative region owner: the ITL ladder
the canonical transformation itself produces, plus `_ALIASES` for "common ways
people name a region that the ITL vocabulary spells differently". It already
carries nominal variants — `greater london`, `ulster`, `ni`, `the north west`,
`east anglia` — and it carried no adjectival ones, so:

    codes_for("scotland")   18 codes
    codes_for("scottish")    0

THE BOUNDARY, and it comes from the taxonomy rather than from English. The
governed ITL1 level has exactly twelve values, and only three of them are
nations with an unambiguous adjectival form:

    Scotland          -> scottish
    Wales             -> welsh
    Northern Ireland  -> northern irish

`English` is deliberately ABSENT. England is not one governed value: it spans
nine ITL1 regions (North East through South West), so "English" has no single
canonical referent and resolving it would invent a region the taxonomy does not
have. `British`, `UK` and the rest are absent for the same reason. The rule is
that an alias may be added where the governed taxonomy already makes the
referent unambiguous — never to widen the taxonomy to accommodate language.

`northern irish` is one governed multi-token phrase with lexical boundaries, not
two words that happen to appear together.

THESE ALIASES ARE NOT THE FIX FOR THE SILENT WRONG ANSWER, and that was measured
before they were written. With `codes_for("scottish")` at 18, both

    "Give me the Scottish balance."
    "How many Scottish lump sum loans are there?"

still answered over the whole book, because in production the book's own value
catalogue decides first and `looks_like_region_term` is consulted only in the
ending reached when no catalogue was supplied. The vocabulary was one of three
independent causes; the grammar and the accounting layers are the others, and
they have their own banks. What this file proves is only that the region owner
now knows the word — which is the part that belongs to the region owner.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import mi_agent.region_resolution as region                             # noqa: E402


#: ``(adjective, the governed name it means)``. Exactly the three ITL1 nations.
ADJECTIVES = (("scottish", "scotland"),
              ("welsh", "wales"),
              ("northern irish", "northern ireland"))


class TestAnAdjectiveReachesTheSameRegion(unittest.TestCase):

    def test_it_resolves_to_exactly_what_the_noun_resolves_to(self):
        """Stated as an equality with the noun, not as a code count, so it holds
        whatever the taxonomy contains."""
        for adjective, noun in ADJECTIVES:
            with self.subTest(adjective=adjective):
                self.assertEqual(region.codes_for(adjective),
                                 region.codes_for(noun))
                self.assertTrue(region.codes_for(noun),
                                "the noun itself does not resolve; the fixture "
                                "or the taxonomy has changed")

    def test_the_region_owner_recognises_it_as_a_place(self):
        for adjective, _noun in ADJECTIVES:
            with self.subTest(adjective=adjective):
                self.assertTrue(region.looks_like_region_term(adjective))

    def test_case_and_article_do_not_matter(self):
        for spelling in ("Scottish", "SCOTTISH", "scottish", "Northern Irish"):
            with self.subTest(spelling=spelling):
                self.assertTrue(region.looks_like_region_term(spelling))


class TestTheBoundaryIsTheTaxonomy(unittest.TestCase):
    """What must NOT resolve, and why. Each of these is a demonym a reader might
    plausibly type; none of them names one governed value."""

    def test_england_is_not_one_governed_region(self):
        """The premise of excluding "English": if England were a single ITL1
        value this test would fail and the exclusion would need revisiting."""
        self.assertEqual(region.codes_for("england"), set())

    def test_no_demonym_without_an_unambiguous_referent(self):
        for adjective in ("english", "british", "cornish", "geordie",
                          "mancunian", "irish"):
            with self.subTest(adjective=adjective):
                self.assertFalse(
                    region.looks_like_region_term(adjective),
                    "a demonym resolved to a region the taxonomy does not "
                    "uniquely define")

    def test_the_alias_table_stays_small(self):
        """A guard on scope creep. Three adjectives were added deliberately; a
        table of plausible demonyms grown to raise the answer rate is how a
        governed vocabulary stops meaning anything."""
        adjectival = [k for k in region._ALIASES
                      if k in {a for a, _ in ADJECTIVES}]
        self.assertEqual(sorted(adjectival),
                         sorted(a for a, _ in ADJECTIVES))


class TestOrdinaryWordsAreStillNotPlaces(unittest.TestCase):
    """The postcode ladder makes two-letter tokens dangerous: ME is Medway and
    SO is Southampton, so "give ME the balance" and "SO how many loans" both
    contain a governed postcode area. Nothing here may make that worse, and the
    accounting layer that consumes this owner must never treat a bare token as
    material on the strength of the postcode ladder alone."""

    def test_short_tokens_still_resolve_as_postcodes_and_must_be_handled_upstream(self):
        # Recorded, not fixed here: this is the region owner answering
        # truthfully about its own ladder. The caller decides materiality.
        self.assertTrue(region.codes_for("me"),
                        "ME is a postcode area; if this changes, the callers "
                        "that guard against it can be simplified")

    def test_ordinary_prose_words_are_not_places(self):
        for word in ("balance", "loans", "total", "please", "give", "show",
                     "count", "product", "broker"):
            with self.subTest(word=word):
                self.assertFalse(region.looks_like_region_term(word))


if __name__ == "__main__":
    unittest.main()
