#!/usr/bin/env python3
"""A relative period is not a licence to drop the stage the reader named.

WHAT WAS MEASURED, on the deployed build and reproduced here against
``tests/fixtures/pipeline_transition_2w``:

    "...moved into Offer in the last reporting period"  -> pipeline_stage_movement -> ANSWERED
    "...moved into Offer stage in the last month"       -> temporal_compare        -> refused
    "...moved into Offer in the last week?"             -> temporal_compare        -> refused

One analytic, three phrasings, two outcomes. `pipeline_stage_movement` narrows
by stage and answers this question 45 times over in the live corpus.
`temporal_compare` compares a whole-population metric across two reporting
periods and CANNOT narrow to a stage, so its receipt correctly refuses —
"stage — this answer covers the whole population". The refusal is right; the
route is wrong.

IT IS A ROUTING-PRECEDENCE DEFECT, NOT A CAPABILITY GAP. A relative time
expression sets ``spec.temporal_mode = "compare"``, `temporal_compare`
recognises on that alone at priority 90, and the stage route is registered last
at 120 — so the period word outranks the stage recogniser and hands the
question to the one route that cannot honour it.

THE FIX IS WHICH ROUTE CLAIMS IT. `temporal_compare` yields a sentence that
carries a governed stage-movement construction. It was NOT taught to narrow,
and `test_temporal_compare_was_not_taught_to_narrow` below is what holds that
line: a question that NAMES a stage without a movement construction still
reaches `temporal_compare`, and is still refused rather than answered narrowly.
"""

from __future__ import annotations

import unittest

from mi_agent_api import stage_movement_query as SM
from mi_agent_api.tests.test_stage_movement_query import FIXTURE, ask

ROUTE = SM.ROUTE_NAME
COMPARE = "temporal_compare"

#: The governed window the fixture's two extracts define, quoted in every answer.
WINDOW = "between 2026-06-05 and 2026-06-12"

#: One analytic, three phrasings. The first is the one that already answered;
#: the other two are the defect.
ARRIVALS_INTO_OFFER = (
    "How many loans moved into Offer in the last reporting period?",
    "How many loans moved into Offer stage in the last month?",
    "How many loans moved into Offer in the last week?",
)


def _route_of(envelope) -> str:
    return str((envelope.get("metadata") or {}).get("route") or "")


class TestThePeriodWordDoesNotDecideTheRoute(unittest.TestCase):

    def test_the_fixture_is_the_one_the_oracle_describes(self):
        self.assertTrue((FIXTURE).exists(), FIXTURE)

    def test_every_phrasing_of_one_analytic_reaches_the_stage_route(self):
        for question in ARRIVALS_INTO_OFFER:
            with self.subTest(question=question):
                envelope = ask(question)
                self.assertEqual(_route_of(envelope), ROUTE,
                                 envelope.get("answer") or envelope.get("error"))
                self.assertTrue(envelope.get("ok"),
                                envelope.get("error") or envelope.get("answer"))

    def test_every_phrasing_of_one_analytic_gets_one_answer(self):
        """Not merely 'answered' — the SAME answer. The route ignores the
        request's as-of and states its own governed window, so a reader who says
        "last week" and a reader who says "the last reporting period" are told
        the same thing about the same pair of extracts."""
        answers = {ask(q).get("answer") for q in ARRIVALS_INTO_OFFER}
        self.assertEqual(len(answers), 1, answers)
        self.assertIn(WINDOW, answers.pop())

    def test_a_transition_with_a_relative_period_is_the_transition(self):
        """Two stages and a direction. The fixture's oracle: KFI -> APPLICATION
        is 2 cases."""
        envelope = ask("How many cases moved from KFI into Application "
                       "in the last month?")
        self.assertEqual(_route_of(envelope), ROUTE,
                         envelope.get("answer") or envelope.get("error"))
        self.assertTrue(envelope.get("ok"), envelope.get("error"))
        self.assertIn("2 cases", envelope.get("answer") or "")

    def test_the_question_that_already_answered_still_answers(self):
        """The one phrasing that worked before any of this. It must not move."""
        envelope = ask("How many cases left KFI in the last week?")
        self.assertEqual(_route_of(envelope), ROUTE)
        self.assertTrue(envelope.get("ok"), envelope.get("error"))


class TestTheBoundaryOfTheYield(unittest.TestCase):
    """What `temporal_compare` keeps, so the fix cannot be read as wider."""

    def test_temporal_compare_was_not_taught_to_narrow(self):
        """A question NAMING a stage, with no movement construction, is not a
        stage-movement question. It still reaches `temporal_compare`, and is
        still REFUSED for a narrowing that route cannot honour — which is the
        capability boundary this change deliberately leaves alone."""
        question = "Compare KFI balance this month vs last month"
        self.assertIsNone(SM.read(question),
                          "no movement construction — nothing for the stage "
                          "route to claim")
        envelope = ask(question)
        self.assertEqual(_route_of(envelope), COMPARE)
        self.assertFalse(envelope.get("ok"))
        # WHICH reason leads changed on 2026-09-04, when a requested time grain
        # the series cannot express became a blocking facet. This question
        # carries TWO true reasons — the route cannot narrow to KFI, and the
        # pipeline has no monthly series — and the facet guard now refuses
        # first, so `_enforce_semantic_coverage` stands down by its own rule
        # ("a refusal is already a refusal"). The BOUNDARY this test exists for
        # is untouched: the question still reaches `temporal_compare`, and that
        # route is still not taught to narrow.
        said = envelope.get("answer") or envelope.get("error") or ""
        self.assertRegex(said, r"KFI|month")
        self.assertIn("not substituted a broader figure", said)

    def test_a_comparison_with_no_stage_is_untouched(self):
        """The stage route reads nothing here, so nothing yields."""
        for question in ("Compare total balance this month vs last month",
                         "How is the total balance this month compared to "
                         "last month?"):
            with self.subTest(question=question):
                self.assertIsNone(SM.read(question))
                self.assertNotEqual(_route_of(ask(question)), ROUTE)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
