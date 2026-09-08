#!/usr/bin/env python3
""""Moved into Offer" is an arrival at Offer, and was read as nothing.

WHY THIS FILE EXISTS. Asked live against the deployed agent:

    "how many loans moved into Offer in the last reporting period"

    "I understood this as a pipeline, movement trend question, but I have not
     answered it: this asks how something changed, which needs two governed
     reporting snapshots to compare."

Twenty governed weekly extracts existed, and the stage-movement capability
answers "how many cases moved from KFI to Application" from exactly those
snapshots. The sentence named one governed stage and a direction, `read`
returned None, and the question fell to the generic engine — which correctly
declines a movement question and then explains a limitation that is not the
reason. THE MESSAGE WAS WRONG BECAUSE THE ROUTING WAS WRONG, and a reader has
no way to tell those two apart.

The same hole swallowed "Show arrivals into Completion by prior stage" (SM31):
`_ARRIVAL_WORDS` carried "new arrivals" and not the bare "arrivals".

A movement verb, an INTO-shaped connector and exactly ONE governed stage can
only mean arrivals at that stage. Where the sentence names BOTH ends the
directional pair claims it as a transition first, so this can never reach one.
"""
from __future__ import annotations

import unittest

from mi_agent_api.stage_movement_query import NEW_ARRIVAL, TRANSITION, read


class TestTheQuestionAskedLive(unittest.TestCase):

    def test_moved_into_a_stage_is_an_arrival_at_it(self):
        r = read("how many loans moved into Offer in the last reporting period")
        self.assertIsNotNone(r, "still unread; it will fall to the generic "
                                "engine and be refused for the wrong reason")
        self.assertEqual(r.subtype, NEW_ARRIVAL)
        self.assertEqual(r.destination, "OFFER")

    def test_the_measure_still_comes_from_the_sentence(self):
        self.assertEqual(read("What balance moved into Offer?").measure, "amount")
        self.assertEqual(read("How many cases moved into Offer?").measure, "count")

    def test_the_bare_noun_arrivals(self):
        r = read("Show arrivals into Completion by prior stage.")
        self.assertIsNotNone(r)
        self.assertEqual(r.destination, "COMPLETED")


class TestTransitionsStillOutrankArrivals(unittest.TestCase):
    """Both ends named is a transition, and must not become an arrival."""

    def test_two_stages_with_a_direction_stay_transitions(self):
        for q, src, dst in (
                ("How many cases moved from KFI to Application?", "KFI", "APPLICATION"),
                ("How much balance moved from Application to Offer?", "APPLICATION", "OFFER"),
                ("How many cases went from KFI into Application?", "KFI", "APPLICATION")):
            with self.subTest(q=q):
                r = read(q)
                self.assertEqual(r.subtype, TRANSITION)
                self.assertEqual((r.source, r.destination), (src, dst))


class TestNothingElseIsClaimed(unittest.TestCase):
    """The verb list is gated on a GOVERNED STAGE, so it cannot spread."""

    def test_a_movement_into_something_that_is_not_a_stage(self):
        for q in ("How much balance moved into arrears?",
                  "How many loans moved into default?",
                  "What moved into the funded book?"):
            with self.subTest(q=q):
                self.assertIsNone(read(q), "claimed a movement into a "
                                           "non-governed stage")

    def test_the_other_subtypes_are_untouched(self):
        self.assertEqual(read("How many cases stayed in Application?").subtype,
                         "stayer")
        self.assertEqual(read("Where did cases leaving Offer go?").subtype,
                         "departure")

    def test_summary_and_measure_questions_are_still_not_movements(self):
        for q in ("What is the pipeline balance?",
                  "Summarise the current pipeline.",
                  "How many cases are in the pipeline?",
                  "Show the pipeline by stage."):
            with self.subTest(q=q):
                self.assertIsNone(read(q))

    def test_a_verb_with_no_connector_is_not_an_arrival(self):
        """"moved" alone names no direction, so it binds nothing."""
        self.assertIsNone(read("How many Offer cases moved?"))


if __name__ == "__main__":
    unittest.main()
