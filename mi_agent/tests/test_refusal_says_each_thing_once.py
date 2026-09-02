#!/usr/bin/env python3
"""A refusal echoes the request once, and keeps every reason it has.

WHY THIS FILE EXISTS. Measured against the live book, ten of the accepted
questions refused with an opening clause that read like a defect:

    "I understood that you asked for ranking by region and region, but that
     could not be applied to the calculation (ranking by region — this answer
     does not rank that dimension; region — this answer covers the whole
     population; it is neither narrowed to nor broken down by region)."

"region and region" is not a mistake in the analysis. *"Which region added the
most balance?"* registers two facets — the RANKING and the dimension it ranks —
and both genuinely block, for different reasons. The redundancy is in how the
request is echoed back, and only there.

So the de-duplication is confined to the opening clause. The detail keeps every
facet, because the two reasons are not the same reason and a reader who is being
refused is owed both.
"""
from __future__ import annotations

import unittest

from mi_agent.execution_receipt import _speech_list


class _Facet:
    def __init__(self, speech):
        self.speech = speech


class TestTheRequestIsEchoedOnce(unittest.TestCase):

    def test_the_phrasing_measured_on_the_live_book(self):
        said = _speech_list([_Facet("ranking by region"), _Facet("region")])
        self.assertEqual(said, "ranking by region")
        self.assertNotIn("region and region", said)

    def test_order_does_not_decide_which_survives(self):
        """The reader's phrase wins whichever order the facets arrive in."""
        for facets in (["ranking by region", "region"],
                       ["region", "ranking by region"]):
            with self.subTest(order=facets):
                self.assertEqual(
                    _speech_list([_Facet(f) for f in facets]),
                    "ranking by region")

    def test_a_four_facet_question_collapses_to_its_distinct_concepts(self):
        """Q21A on the live book: one threshold and one ranking, said twice each."""
        said = _speech_list([_Facet("LTV over 50"), _Facet("ranking by region"),
                             _Facet("region"),
                             _Facet("loans where Current LTV over 50")])
        self.assertEqual(said,
                         "ranking by region and loans where Current LTV over 50")


class TestNothingDistinctIsLost(unittest.TestCase):
    """The guard must never quietly drop a concept the reader actually named."""

    def test_two_unrelated_concepts_are_both_named(self):
        said = _speech_list([_Facet("region"), _Facet("broker channel")])
        self.assertEqual(said, "region and broker channel")

    def test_a_single_concept_is_unchanged(self):
        self.assertEqual(_speech_list([_Facet("product type")]), "product type")

    def test_empty_speech_is_skipped_not_rendered(self):
        self.assertEqual(_speech_list([_Facet(""), _Facet("region")]), "region")

    def test_similar_but_distinct_phrases_both_survive(self):
        """Containment is the test, not similarity: neither contains the other."""
        said = _speech_list([_Facet("LTV over 50"), _Facet("LTV over 40")])
        self.assertIn("LTV over 50", said)
        self.assertIn("LTV over 40", said)


if __name__ == "__main__":
    unittest.main()
