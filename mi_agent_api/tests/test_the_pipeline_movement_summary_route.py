#!/usr/bin/env python3
"""The two questions that named the capability and had nowhere to go.

    Give me the stage movement summary.
    Give me the pipeline movement summary.
    Compare stage movement with the prior period.
    How has pipeline movement changed since last period?

The first two ask for the whole pipeline's movement over one governed interval;
the last two ask for that summary against the one before it. None names a stage,
so `stage_movement_query` correctly declines all four — and it still does: this
route does not weaken `names_a_stage_movement`, it answers what that reading
returns None for.

THE COMPARATIVE FORM COMPARES TWO GOVERNED SUMMARIES. It never substitutes a
point-in-time snapshot for a movement, which is what `temporal_compare` was
doing when it claimed "Compare stage movement with the prior period" and then
refused for a stage it could not narrow to.

WHAT MUST NOT MOVE: every question `pipeline_stage_movement` answers today.
That capability is 48 of 49 in the live bank and this route registers after it,
so a sentence it claims is a sentence this one never sees. Pinned below.
"""
from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from mi_agent_api import pipeline_movement_summary as PMS
from mi_agent_api import stage_movement_query as SM
from mi_agent_api.tests.test_stage_movement_query import ask

ROUTE = "pipeline_movement_summary"

SUMMARY_QUESTIONS = (
    "Give me the stage movement summary.",
    "Give me the pipeline movement summary.",
    "Summarise pipeline stage movement this period.",
)
COMPARATIVE_QUESTIONS = (
    "Compare stage movement with the prior period.",
    "How has pipeline movement changed since last period?",
)
#: Answered by `pipeline_stage_movement` today. None may move.
STAGE_QUESTIONS = (
    "How many cases moved from KFI to Application?",
    "Reconcile the Application stage from opening to closing.",
    "Show the destinations of Offer-stage departures.",
    "How many cases stayed in Application?",
    "Compare Offer departures with the previous reporting period.",
)


def _route(envelope):
    return (envelope.get("metadata") or {}).get("route")


class TestItReadsWhatTheStageRouteDeclines(unittest.TestCase):

    def test_the_stage_route_still_declines_all_of_them(self):
        """`names_a_stage_movement` is NOT weakened — that is the point."""
        for question in SUMMARY_QUESTIONS + COMPARATIVE_QUESTIONS:
            with self.subTest(question=question):
                self.assertFalse(SM.names_a_stage_movement(question))

    def test_this_route_reads_them(self):
        for question in SUMMARY_QUESTIONS:
            with self.subTest(question=question):
                reading = PMS.read(question)
                self.assertIsNotNone(reading)
                self.assertFalse(reading.comparative)

    def test_the_comparative_form_is_read_as_comparative(self):
        for question in COMPARATIVE_QUESTIONS:
            with self.subTest(question=question):
                reading = PMS.read(question)
                self.assertIsNotNone(reading)
                self.assertTrue(reading.comparative)

    def test_it_reads_nothing_that_names_a_single_stage_movement(self):
        for question in STAGE_QUESTIONS:
            with self.subTest(question=question):
                self.assertIsNone(PMS.read(question))

    def test_it_reads_no_unrelated_question(self):
        for question in ("Show balance by region.",
                         "What is the funded balance?",
                         "Summarise the funded portfolio.",
                         "Show pipeline evolution."):
            with self.subTest(question=question):
                self.assertIsNone(PMS.read(question))


class TestItAnswers(unittest.TestCase):

    def test_the_summary_questions_answer_on_this_route(self):
        for question in SUMMARY_QUESTIONS:
            with self.subTest(question=question):
                envelope = ask(question)
                self.assertTrue(envelope.get("ok"), envelope.get("error"))
                self.assertEqual(_route(envelope), ROUTE)

    def test_the_answer_carries_the_structured_result(self):
        envelope = ask(SUMMARY_QUESTIONS[0])
        summary = (envelope.get("metadata") or {}).get("pipelineMovementSummary")
        self.assertTrue(summary)
        self.assertTrue(summary["available"])
        self.assertEqual(summary["version"], PMS.SUMMARY_VERSION)
        self.assertTrue(summary["reconciliation"]["ok"])

    def test_the_answer_states_the_window_it_covers(self):
        answer = ask(SUMMARY_QUESTIONS[0]).get("answer") or ""
        self.assertIn("2026-06-12", answer)
        self.assertIn("2026-06-05", answer)

    def test_the_comparative_form_compares_two_movement_summaries(self):
        envelope = ask(COMPARATIVE_QUESTIONS[0])
        meta = envelope.get("metadata") or {}
        block = meta.get("pipelineMovementComparison")
        if envelope.get("ok"):
            self.assertEqual(_route(envelope), ROUTE)
            self.assertTrue(block)
            # BOTH SIDES ARE MOVEMENT SUMMARIES, never a point-in-time stock.
            for side in ("current", "prior"):
                self.assertEqual(block[side]["version"], PMS.SUMMARY_VERSION)
                self.assertIn("opening", block[side])
        else:
            # Only one governed interval exists in this fixture: refuse, and say
            # so, rather than compare a movement with a snapshot.
            self.assertIn("prior", str(envelope.get("error") or "").lower())


class TestNothingTheStageRouteAnswersMoves(unittest.TestCase):

    def test_every_stage_question_keeps_its_route_and_its_answer(self):
        for question in STAGE_QUESTIONS:
            with self.subTest(question=question):
                envelope = ask(question)
                self.assertTrue(envelope.get("ok"), envelope.get("error"))
                self.assertEqual(_route(envelope), "pipeline_stage_movement")


#: THE ENVIRONMENT THIS MODULE PERTURBS, restored so it cannot cost a neighbour.
#:
#: `test_stage_movement_query.ask` calls `_ensure_env()`, which sets the pipeline
#: and onboarding roots GLOBALLY and never puts them back — deliberately, because
#: re-asserting them per ask is what recovered fifteen order-dependent nodes in
#: that module. Importing `ask` inherits the leak, and
#: `test_pipeline_runtime_materialisation` (which sorts after these files) then
#: discovered a pipeline root it never set. Measured: three failures that appear
#: only in a whole-directory run.
_LEAKED = ("MI_AGENT_PIPELINE_ROOT", "MI_AGENT_ONBOARDING_OUTPUT_ROOT",
           "MI_AGENT_AUTH_ENABLED", "MI_AGENT_LLM_PARSER")
_SAVED_ENV = {}


def setUpModule():                                          # noqa: N802
    _SAVED_ENV.update({k: os.environ.get(k) for k in _LEAKED})


def tearDownModule():                                       # noqa: N802
    for key, value in _SAVED_ENV.items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
