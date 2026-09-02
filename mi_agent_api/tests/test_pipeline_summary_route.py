#!/usr/bin/env python3
"""The pipeline gets the summary the funded book already has.

WHY THIS FILE EXISTS. `portfolio_summary` reads `output_root`, has no pipeline
frame, and therefore declines every pipeline question — correctly, and for a
measured reason: "Summarise the current pipeline." was once answered *"the
portfolio holds 640 loans with a funded balance of [figure]"*, from the FUNDED
book. `_names_another_dataset` closed that.

What the guard never had was a sibling to hand the question to. So a pipeline
summary fell through to the generic executor, and "What does the current
pipeline look like?" came back as AMBIGUOUS_QUESTION — measured live on the
funded bank (Q10C) and the stage bank (SM87).

THE CLAIM IS GATED ON THE DATASET OWNER, not on words. `workspace.resolve_dataset`
must say PIPELINE before any vocabulary below is consulted, so nothing on the
funded side is reachable however the pipeline list grows. The tests below pin
that boundary in both directions, because a summary route that drifts onto the
funded book is the exact defect this route's sibling was built to stop.
"""
from __future__ import annotations

import unittest

from mi_agent_api.chat_routing import _is_pipeline_summary, _is_portfolio_summary


class TestItClaimsThePipelineSummaries(unittest.TestCase):

    def test_the_questions_that_fell_through_live(self):
        for q in ("What does the current pipeline look like?",     # Q10C
                  "Show pipeline progression."):                    # SM87
            with self.subTest(q=q):
                self.assertTrue(_is_pipeline_summary(q))

    def test_the_plain_pipeline_summary(self):
        self.assertTrue(_is_pipeline_summary("Summarise the current pipeline."))


class TestItCannotReachTheFundedBook(unittest.TestCase):
    """The gate is the dataset owner. No funded question may be claimed."""

    def test_no_funded_summary_is_taken(self):
        for q in ("Summarise the portfolio.",
                  "Give me a management summary of the current book.",
                  "Give me a concise overview of the funded portfolio.",
                  "What are the headline numbers?",
                  "How is the book doing?"):
            with self.subTest(q=q):
                self.assertFalse(_is_pipeline_summary(q),
                                 "claimed a FUNDED question: the dataset gate "
                                 "is not holding")

    def test_the_funded_summary_keeps_everything_it_had(self):
        for q in ("Summarise the portfolio.",
                  "Give me a management summary of the current book."):
            with self.subTest(q=q):
                self.assertTrue(_is_portfolio_summary(q))

    def test_the_two_never_both_claim_a_question(self):
        """Disjoint by construction — one reads funded, the other pipeline."""
        for q in ("Summarise the portfolio.",
                  "What does the current pipeline look like?",
                  "Show pipeline progression.",
                  "How is the book doing?"):
            with self.subTest(q=q):
                funded = _is_portfolio_summary(q) and not _is_pipeline_summary(q)
                pipe = _is_pipeline_summary(q)
                self.assertFalse(funded and pipe)


class TestItTakesNothingFromAnotherRoute(unittest.TestCase):
    """Every exclusion the funded summary applies, applied here too."""

    def test_a_stratification_is_not_a_summary(self):
        self.assertFalse(_is_pipeline_summary(
            "Give me an overview of the pipeline by size and stage."))

    def test_a_movement_question_belongs_to_stage_movement(self):
        for q in ("How did cases move through the funnel?",
                  "How many cases moved from KFI to Application?",
                  "What changed in the pipeline since last month?"):
            with self.subTest(q=q):
                self.assertFalse(_is_pipeline_summary(q))

    def test_a_measure_question_is_not_a_summary(self):
        for q in ("What is the pipeline balance?",
                  "How many cases are in the pipeline?",
                  "Show the pipeline by stage."):
            with self.subTest(q=q):
                self.assertFalse(_is_pipeline_summary(q))


class TestTheRouteComputesNothing(unittest.TestCase):
    """Every figure is a key on the governed snapshot, or the route defers."""

    def test_the_handler_reads_the_governed_payload_only(self):
        import pathlib
        from mi_agent_api import chat_routing
        src = pathlib.Path(chat_routing.__file__).read_text()
        body = src.split("def _route_pipeline_summary", 1)[1].split("\ndef ", 1)[0]
        self.assertIn("compute_pipeline_snapshot", body)
        for arithmetic in ("groupby", ".sum()", ".mean()", ".count()"):
            self.assertNotIn(arithmetic, body,
                             "the route is computing %s; every figure must be a "
                             "key on the governed snapshot" % arithmetic)

    def test_scope_is_disclosed_not_claimed(self):
        import pathlib
        from mi_agent_api import chat_routing
        src = pathlib.Path(chat_routing.__file__).read_text()
        body = src.split("def _route_pipeline_summary", 1)[1].split("\ndef ", 1)[0]
        self.assertIn("lens_applied=False", body)
        self.assertIn("Scope not narrowed", body)


class TestWhereItRegisters(unittest.TestCase):
    """The placement test `test_existing_route_order_is_preserved` requires.

    That test pins the twelve MIGRATED routes in order and excludes anything
    added since, on the condition that each addition proves its own placement
    here. This is that proof.
    """

    def _names(self):
        import mi_agent_api.chat_routing  # noqa: F401 - registers the routes
        from mi_agent_api.recogniser_registry import REGISTRY

        return list(REGISTRY.names())

    def test_it_registers_immediately_after_the_funded_summary(self):
        names = self._names()
        self.assertIn("pipeline_summary", names)
        self.assertEqual(names[names.index("portfolio_summary") + 1],
                         "pipeline_summary",
                         "the two summaries must sit together: they are one "
                         "capability split by dataset, and reading the chain "
                         "should say so")

    def test_it_sits_before_period_change_analysis(self):
        names = self._names()
        self.assertLess(names.index("pipeline_summary"),
                        names.index("period_change_analysis"))

    def test_the_migrated_chain_is_untouched(self):
        """Order among the pre-existing routes is exactly as it was."""
        names = self._names()
        chain = [n for n in names
                 if n in ("scenario", "cohort_conversion",
                          "forecast_extrapolation", "funded_bridge",
                          "cohort_progression", "geo_exposure",
                          "period_movement", "portfolio_summary",
                          "period_change_analysis", "temporal_compare",
                          "risk_limits", "evolution")]
        self.assertEqual(chain, [
            "scenario", "cohort_conversion", "forecast_extrapolation",
            "funded_bridge", "cohort_progression", "geo_exposure",
            "period_movement", "portfolio_summary", "period_change_analysis",
            "temporal_compare", "risk_limits", "evolution"])


if __name__ == "__main__":
    unittest.main()
