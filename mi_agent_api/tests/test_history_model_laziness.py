#!/usr/bin/env python3
"""Phase 1A item 1 — the historical completion model must be built ON DEMAND.

Building it replays every retained weekly pipeline extract (see
``pipeline_history.build_historical_completion_model``), so it must never be
constructed for a question that does not use it. The serving path used to pass
it as an eagerly-evaluated argument to ``chat_routing.try_route``, which charged
every MI and Copilot question — including "balance by region" — for analysis
almost none of them consume.

These tests pin the contract in both directions:

  * an intent that does not need the model never calls the builder;
  * an intent that does need it still gets it, exactly once;
  * a builder that fails degrades to "no model", never to a failed request.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from mi_agent_api.recogniser_registry import RouteRequest


def _request(**kw) -> RouteRequest:
    base = dict(question="q", spec=object(), spec_dict={}, semantics={},
                view="funded", client_id="client_001", run_id=None,
                portfolio_id=None)
    base.update(kw)
    return RouteRequest(**base)


class TestResolveHistoryModel(unittest.TestCase):
    """The lazy accessor itself."""

    def test_provider_is_not_called_until_asked(self):
        calls = []
        _request(history_model_provider=lambda: calls.append(1) or {"m": 1})
        self.assertEqual(calls, [],
                         "constructing a RouteRequest must not build the model")

    def test_provider_is_called_once_and_memoised(self):
        calls = []

        def build():
            calls.append(1)
            return {"uniqueWeeklyExtractsUsed": 4}

        req = _request(history_model_provider=build)
        first = req.resolve_history_model()
        second = req.resolve_history_model()
        self.assertEqual(len(calls), 1, "the model must be built at most once")
        self.assertEqual(first, {"uniqueWeeklyExtractsUsed": 4})
        self.assertIs(first, second)

    def test_eager_model_wins_and_provider_is_never_called(self):
        calls = []
        req = _request(history_model={"eager": True},
                       history_model_provider=lambda: calls.append(1) or {"lazy": True})
        self.assertEqual(req.resolve_history_model(), {"eager": True})
        self.assertEqual(calls, [], "an explicit model must not trigger a build")

    def test_no_provider_and_no_model_is_none(self):
        self.assertIsNone(_request().resolve_history_model())

    def test_failing_provider_degrades_to_none(self):
        def boom():
            raise RuntimeError("weekly extracts unreadable")

        req = _request(history_model_provider=boom)
        self.assertIsNone(req.resolve_history_model(),
                          "a history fault must degrade, not raise")

    def test_failing_provider_is_not_retried(self):
        calls = []

        def boom():
            calls.append(1)
            raise RuntimeError("nope")

        req = _request(history_model_provider=boom)
        req.resolve_history_model()
        req.resolve_history_model()
        self.assertEqual(len(calls), 1, "a failed build must be memoised too")

    def test_memo_does_not_break_request_equality(self):
        # Share every other field so the only candidate difference is the memo.
        spec = object()
        a = _request(spec=spec)
        b = _request(spec=spec)
        self.assertEqual(a, b, "two identical requests must compare equal")
        a.resolve_history_model()  # populates a's memo, not b's
        self.assertEqual(a, b,
                         "the memo field must be excluded from equality")


class TestRoutingDoesNotBuildForIrrelevantIntents(unittest.TestCase):
    """End-to-end through ``try_route``: recognition must not touch the model."""

    def _run(self, question: str):
        from mi_agent.mi_query_validator import load_mi_semantics
        from mi_agent_api import chat_routing
        from mi_agent_api.data_source import semantics_path

        calls = []

        def provider():
            calls.append(1)
            return {"uniqueWeeklyExtractsUsed": 3,
                    "cumulativeCohortConversion": {}, "cohortProgression": {}}

        try:
            chat_routing.try_route(
                question, portfolio_id="client_001", view="funded",
                output_root=None, pipeline_root=None,
                semantics=load_mi_semantics(semantics_path()),
                history_model_provider=provider)
        except Exception:  # noqa: BLE001 - routing failures are not this test's subject
            pass
        return calls

    def test_plain_portfolio_question_never_builds_the_model(self):
        self.assertEqual(
            self._run("What is the total funded balance by region?"), [],
            "a point-in-time question must not build the historical model")

    def test_stratification_question_never_builds_the_model(self):
        self.assertEqual(
            self._run("Show me the LTV distribution"), [],
            "a stratification question must not build the historical model")

    def test_top_n_question_never_builds_the_model(self):
        self.assertEqual(
            self._run("What are the top 10 loans by balance?"), [],
            "a top-N question must not build the historical model")

    def test_conversion_question_still_builds_the_model(self):
        self.assertEqual(
            len(self._run("What is the cohort conversion rate?")), 1,
            "cohort conversion genuinely needs the historical model")


if __name__ == "__main__":
    unittest.main()
