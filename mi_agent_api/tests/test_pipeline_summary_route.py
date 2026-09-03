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


class TestItSaysWhichDatasetItRead(unittest.TestCase):
    """The route must DECLARE the pipeline, not merely read it.

    MEASURED LIVE, 2026-09-03. "Summarise the current pipeline" routed here,
    computed the right snapshot, disclosed its scope — and was then replaced by
    "I understood that you asked about pipeline, but I could not confirm it was
    applied to this calculation." The envelope carried no `reconciliation` at
    all, so `completeness._carried` compared a stated `dataset: pipeline`
    concept against an empty record of what was read, reported it UNACCOUNTED,
    and `_enforce_semantic_coverage` refused a correct answer.

    The guard was right. A route that cannot say what it read cannot be checked
    against what it was asked for, and `workspace.datasets_read` exists because
    three routes once wrote the dataset as a literal and were wrong about
    themselves undetectably. So the fix is the route supplying the evidence,
    never the ledger being told to trust it — which is why the assertions below
    are about the ENVELOPE's own record and about `_carried` reaching RESOLVED
    from it, not about the refusal being suppressed.
    """

    def _envelope_from_the_route(self):
        from mi_agent_api import chat_routing as CR
        from mi_agent_api import workspace as W
        return CR._envelope(
            ok=True, question="Summarise the current pipeline.",
            answer="...", spec={}, artifacts=[], route="pipeline_summary",
            reconciliation=W.reconciliation_for(
                W.datasets_read(pipeline_root={"client_id": "c"}),
                reporting_date="2026-01-12"))

    def test_the_envelope_names_the_pipeline_as_what_it_read(self):
        env = self._envelope_from_the_route()
        self.assertEqual((env.get("reconciliation") or {}).get("dataset"),
                         "pipeline")

    def test_the_stated_pipeline_concept_is_carried(self):
        """The ledger's own reader, on the ledger's own terms."""
        from question_interpretation import completeness as C
        concept = C.StatedConcept("dataset", "view", "pipeline", "",
                                  "workspace.resolve_dataset")
        contract = C.from_envelope(self._envelope_from_the_route())
        self.assertTrue(C._carried(concept, contract))

    def test_an_envelope_without_the_declaration_is_still_refused(self):
        """The falsification. Remove the declaration and the concept goes back
        to uncarried — so this suite fails against the code that shipped, and
        the guard is proven still able to catch a route that stays silent."""
        from mi_agent_api import chat_routing as CR
        from question_interpretation import completeness as C
        silent = CR._envelope(ok=True, question="Summarise the current pipeline.",
                              answer="...", spec={}, artifacts=[],
                              route="pipeline_summary")
        concept = C.StatedConcept("dataset", "view", "pipeline", "",
                                  "workspace.resolve_dataset")
        self.assertFalse(C._carried(concept, C.from_envelope(silent)))

    def test_it_does_not_claim_the_funded_book(self):
        """Declaring MORE than it read would be the same defect pointed the
        other way: `funded+pipeline` would let a funded concept pass unchecked."""
        env = self._envelope_from_the_route()
        self.assertNotIn("funded", (env.get("reconciliation") or {}).get("dataset", ""))


class TestTheRouteItselfDeclaresIt(unittest.TestCase):
    """The assertions above exercise `_envelope` and `workspace`. This one
    exercises THE CALL SITE, which is where the declaration was missing — a
    suite that builds the envelope by hand would pass against the code that
    shipped the bug."""

    def _run_the_route(self):
        from unittest import mock
        from mi_agent_api import chat_routing as CR
        # Import them so they are attributes of the package: the route resolves
        # `from . import datasets` at call time, and patch() cannot replace an
        # attribute that import has not yet created.
        import mi_agent_api.datasets  # noqa: F401
        import mi_agent_api.pipeline_contract  # noqa: F401

        snapshot = {"pipelineRowCount": 12, "pipelineAmount": 1_000_000.0,
                    "pipelineAsOfDate": "2026-01-12",
                    "stageBreakdown": [{"stage": "Offer", "count": 4}],
                    "weightedExpectedFundedAmount": 500_000.0}
        ds = mock.MagicMock()
        ds._resolve_pipeline_source.return_value = {"client_id": "c", "run_id": "r"}
        pc = mock.MagicMock()
        pc.load_prepared_pipeline.return_value = (mock.MagicMock(), {})
        pc.compute_pipeline_snapshot.return_value = snapshot
        with mock.patch("mi_agent_api.datasets", ds), \
             mock.patch("mi_agent_api.pipeline_contract", pc):
            return CR._route_pipeline_summary(
                "Summarise the current pipeline.", {},
                client_id="c", run_id="r")

    def test_the_route_publishes_the_pipeline_reconciliation(self):
        env = self._run_the_route()
        self.assertIsNotNone(env, "the route declined its own question")
        self.assertEqual((env.get("reconciliation") or {}).get("dataset"),
                         "pipeline")

    def test_the_route_reports_the_extract_date_it_read(self):
        env = self._run_the_route()
        self.assertEqual((env.get("reconciliation") or {}).get("reporting_date"),
                         "2026-01-12")


class TestTheAnswerSurvivesTheCoverageGate(unittest.TestCase):
    """END TO END, THROUGH THE GATE THAT ACTUALLY REFUSED IT.

    The suites above prove the route DECLARES the pipeline and that
    `completeness._carried` reaches RESOLVED from that declaration. Neither
    drives `_enforce_semantic_coverage`, which is what replaced the live answer
    with "I could not confirm it was applied to this calculation" — so neither
    can honestly claim the question is answered again. This one runs the real
    ledger over the real envelope and asserts the gate lets it through, and
    asserts the same gate still refuses the undeclared envelope so the test
    cannot pass by the gate having gone soft.
    """

    QUESTION = "Summarise the current pipeline."

    def _semantics(self):
        from mi_agent_api.datasets import load_mi_semantics, semantics_path
        return load_mi_semantics(semantics_path())

    def _gated(self, envelope):
        """The envelope after the service's own stamp-then-enforce sequence."""
        from mi_agent_api import mi_service as MS
        envelope.setdefault("metadata", {})
        MS._stamp_semantic_coverage(envelope, question=self.QUESTION,
                                    semantics=self._semantics(), frame=None)
        return MS._enforce_semantic_coverage(envelope)

    def _envelope(self, *, declared: bool):
        from mi_agent_api import chat_routing as CR
        from mi_agent_api import workspace as W
        recon = (W.reconciliation_for(W.datasets_read(pipeline_root={"c": 1}),
                                      reporting_date="2026-01-12")
                 if declared else None)
        return CR._envelope(
            ok=True, question=self.QUESTION,
            answer="At the weekly extract of 12 Jan 2026 the pipeline holds ...",
            spec={}, artifacts=[], route="pipeline_summary",
            lens_applied=False, reconciliation=recon)

    def test_the_declared_answer_is_not_refused(self):
        gated = self._gated(self._envelope(declared=True))
        self.assertTrue(gated.get("ok"), (
            "the coverage gate still refuses the pipeline summary: "
            + str(gated.get("error"))))
        self.assertNotIn("could not confirm it was applied",
                         str(gated.get("answer")))

    def test_the_pipeline_concept_is_not_left_unaccounted(self):
        gated = self._gated(self._envelope(declared=True))
        ledger = (gated.get("metadata") or {}).get("semanticCoverage") or {}
        unaccounted = {str(e.get("value")) for e in (ledger.get("unaccounted") or ())}
        self.assertNotIn("pipeline", unaccounted)

    def test_the_undeclared_answer_is_still_refused(self):
        """THE FALSIFICATION, and the guard's own protection. If this ever
        passes, the gate has stopped catching a route that reads one dataset
        and is asked about another — which is the defect it exists for."""
        gated = self._gated(self._envelope(declared=False))
        self.assertFalse(gated.get("ok"),
                         "the coverage gate no longer catches an undeclared read")
        self.assertIn("could not confirm it was applied", str(gated.get("answer")))
