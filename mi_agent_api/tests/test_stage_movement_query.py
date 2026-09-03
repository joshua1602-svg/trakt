#!/usr/bin/env python3
"""MI Query as a consumer of the governed pipeline stage-transition capability.

The capability itself is pinned by ``test_pipeline_stage_transition.py`` and is
untouched here. What these tests pin is the ADAPTER: that Query recognises a
stage-movement question, binds the stages and the measure the reader named, gets
its figures from ``resolve_stage_transition_detail`` and from nothing else, and
declines rather than substituting when it cannot bind.

THE FIXTURE IS THE ORACLE. ``tests/fixtures/pipeline_transition_2w`` — fourteen
cases across 2026-06-05 and 2026-06-12 — so every expected figure below is
arithmetic on that table, not an opinion:

    KFI -> APPLICATION      2 cases   prior 900,000    latest 920,000
    APPLICATION -> OFFER    2 cases   prior 1,300,000  latest 1,290,000
    OFFER -> COMPLETED      1 case    800,000
    new arrivals into KFI   1 case    900,000
    stayers in APPLICATION  1 case    300,000 -> 280,000
    departures from OFFER   1 case    1,200,000, outcome unclassified
    APPLICATION             opening 4, +1 new, +2 in, -2 out, -1 gone, closing 4

THE DEFECT AT THE STARTING SHA, pinned as a negative below: "How many cases went
from KFI into Application?" was answered *"3 loans · £1.2MM · Pipeline Stage =
KFI"* — the current KFI STOCK, for a question about a transition.
"""

from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path
from typing import Any, Dict

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

FIXTURE = _REPO_ROOT / "tests" / "fixtures" / "pipeline_transition_2w"
PORTFOLIO = "client_001/mi_2026_06"
AS_OF = "2026-06-30"
ROUTE = "pipeline_stage_movement"

#: The governed window the fixture's two extracts define.
LATEST, PRIOR = "2026-06-12", "2026-06-05"


def _write_funded_tape(root: Path) -> None:
    """A minimal governed funded tape, so the MI service has a book to resolve.

    Nothing here is read by the stage-movement route — its figures come entirely
    from the pipeline extracts — but ``/mi/query`` resolves a funded frame for
    every request, so one has to exist.
    """
    import numpy as np
    import pandas as pd

    rng = np.random.default_rng(20260630)
    n = 120
    out = root / "client_001" / "mi_2026_06" / "output" / "central"
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({
        "loan_identifier": [f"L{i:05d}" for i in range(n)],
        "current_outstanding_balance": rng.uniform(60_000, 480_000, n).round(2),
        "current_loan_to_value": rng.uniform(15, 75, n).round(1),
        "current_interest_rate": rng.uniform(3, 9, n).round(2),
        "youngest_borrower_age": rng.integers(60, 92, n),
        "broker_channel": rng.choice(["Alpha", "Beta"], n),
        "geographic_region_obligor": rng.choice(["North", "Scotland"], n),
        "reporting_date": ["2026-06-30"] * n,
    }).to_csv(out / "18_central_lender_tape.csv", index=False)


_CLIENT = None
_ROOT = None


def _ensure_env() -> None:
    """The roots this module's client needs, RE-ASSERTED before every ask.

    The app reads these per request, and several sibling modules in this
    directory set and then POP the same variables in their own setUp/tearDown.
    Whichever of them runs first leaves `MI_AGENT_PIPELINE_ROOT` unset, and
    every question here then came back "No governed pipeline data is available
    for the pipeline view" — twenty-odd failures that depend on nothing but
    collection order, and that do not appear when this file is run alone.

    Setting them once at client construction was the assumption; the process
    environment is shared, so it does not hold.
    """
    global _ROOT
    if _ROOT is None:
        import tempfile

        _ROOT = Path(tempfile.mkdtemp()) / "onboarding_output"
        _write_funded_tape(_ROOT)
    os.environ["MI_AGENT_ONBOARDING_OUTPUT_ROOT"] = str(_ROOT)
    os.environ["MI_AGENT_PIPELINE_ROOT"] = str(FIXTURE)
    os.environ["MI_AGENT_AUTH_ENABLED"] = "false"
    os.environ.setdefault("MI_AGENT_LLM_PARSER", "off")


def _client():
    global _CLIENT
    if _CLIENT is None:
        _ensure_env()
        from fastapi.testclient import TestClient
        from mi_agent_api.app import app

        _CLIENT = TestClient(app)
    return _CLIENT


def ask(question: str) -> Dict[str, Any]:
    client = _client()
    _ensure_env()
    return client.post("/mi/query", json={
        "question": question, "portfolioId": PORTFOLIO, "asOfDate": AS_OF}).json()


def answer(question: str) -> str:
    env = ask(question)
    assert env.get("ok"), "%r was declined: %s" % (question,
                                                   env.get("error") or env.get("answer"))
    assert (env.get("metadata") or {}).get("route") == ROUTE, (
        "%r was answered by %r, not the governed stage-movement route"
        % (question, (env.get("metadata") or {}).get("route")))
    return env.get("answer") or ""


# --------------------------------------------------------------------------- #
# 1-6. Source -> destination transitions, counts and amounts
# --------------------------------------------------------------------------- #
class TestTransitions(unittest.TestCase):

    def test_kfi_to_application_case_count(self):
        """The governed answer is two. The KFI STOCK is three."""
        text = answer("How many cases moved from KFI to Application?")
        self.assertIn("2 cases", text)
        self.assertIn("KFI", text)
        self.assertIn("Application", text)

    def test_kfi_to_application_natural_language_variants(self):
        """Four independent formulations, one governed figure."""
        for question in (
                "How many KFI cases progressed to Application?",
                "How many cases went from KFI into Application?",
                "What number of cases transitioned KFI to Application?",
                "How many cases advanced from KFI to Application?"):
            with self.subTest(question=question):
                self.assertIn("2 cases", answer(question))

    def test_application_to_offer_amount(self):
        """£1,290,000 arrived at OFFER; the same cases left APPLICATION at
        £1,300,000. Both are governed fields; neither is computed here."""
        text = answer("How much balance moved from Application to Offer?")
        self.assertIn("£1.3m", text)
        self.assertIn("2 cases", text)

    def test_amount_language_variants_all_bind_the_amount(self):
        """balance / value / amount / how much are one measure."""
        for question in (
                "What value progressed from Application to Offer?",
                "How much pipeline moved from Application into Offer?",
                "What amount transitioned from Application to Offer?"):
            with self.subTest(question=question):
                self.assertIn("£1.3m", answer(question))

    def test_offer_to_completion_case_count(self):
        text = answer("How many cases moved from Offer to Completion?")
        self.assertIn("1 case", text)
        self.assertIn("Completion", text)

    def test_offer_to_completion_amount(self):
        self.assertIn("£800k",
                      answer("How much balance moved from Offer to Completion?"))


# --------------------------------------------------------------------------- #
# 7-11. The other governed event classes
# --------------------------------------------------------------------------- #
class TestEventClasses(unittest.TestCase):

    def test_new_arrivals_into_a_stage(self):
        text = answer("How many new cases entered KFI?")
        self.assertIn("1 case", text)
        self.assertIn("£900k", text)

    def test_stayers_in_a_stage(self):
        """One case stayed at APPLICATION. Four are AT application — the stock."""
        text = answer("How many cases stayed in Application?")
        self.assertIn("1 case", text)
        self.assertNotIn("4 cases", text)

    def test_stayer_amount_change_is_the_governed_amendment(self):
        """£300k -> £280k. The route reports the governed change, not a stock."""
        text = answer("What was the amount change on cases that stayed in Application?")
        self.assertIn("£20k", text)
        self.assertIn("down", text)
        self.assertIn("£300k", text)
        self.assertIn("£280k", text)

    def test_departures_are_broken_down_by_destination(self):
        """One OFFER case completed; one left with no evidenced outcome. The
        second is reported as unevidenced, never resolved into a withdrawal."""
        text = answer("Where did cases leaving Offer go?")
        self.assertIn("Completion", text)
        self.assertIn("no outcome the data evidences", text)

    def test_stage_reconciliation(self):
        """opening 4 + 1 new + 2 in - 2 out - 1 departed = closing 4."""
        text = answer("Reconcile Application stage this period.")
        for phrase in ("opening 4 cases", "1 case newly arrived",
                       "2 cases transferred in", "2 cases transferred out",
                       "1 case departed", "closing 4 cases"):
            self.assertIn(phrase, text)


# --------------------------------------------------------------------------- #
# 12. The reporting window is stated on every answer
# --------------------------------------------------------------------------- #
class TestReportingWindow(unittest.TestCase):

    def test_every_answer_states_the_governed_window(self):
        for question in ("How many cases moved from KFI to Application?",
                         "How many new cases entered KFI?",
                         "How many cases stayed in Application?",
                         "Where did cases leaving Offer go?",
                         "Reconcile Application stage this period."):
            with self.subTest(question=question):
                text = answer(question)
                self.assertIn(PRIOR, text)
                self.assertIn(LATEST, text)


# --------------------------------------------------------------------------- #
# 13-16. Refusal, and the substitution that must never happen
# --------------------------------------------------------------------------- #
class TestRefusals(unittest.TestCase):

    #: A window whose governed reconciliation carries KFI and APPLICATION only.
    #: The fixture itself carries all five canonical stages, so a stage genuinely
    #: absent from a window has to be posed to `compose` directly — which is
    #: where the rule lives anyway.
    _TWO_STAGE_WINDOW = {
        "available": True, "as_of_date": LATEST, "comparison_date": PRIOR,
        "transitions": [], "new_arrivals": [], "stayers": [], "departures": [],
        "reconciliation": {"by_stage": [{"stage": "KFI"},
                                        {"stage": "APPLICATION"}]},
    }

    def test_absent_source_stage_is_refused_not_answered(self):
        """"No cases moved" and "that stage is not in this pipeline" are
        different statements, and the second must never be told as the first."""
        from mi_agent_api import stage_movement_query as sm

        reading = sm.read("How many cases moved from Offer to Application?")
        text, rows, refusal = sm.compose(reading, self._TWO_STAGE_WINDOW, money=str)
        self.assertIsNone(text)
        self.assertEqual(rows, [])
        self.assertIn("Offer is not a stage", refusal)
        self.assertIn("from", refusal)

    def test_absent_destination_stage_is_refused_not_answered(self):
        from mi_agent_api import stage_movement_query as sm

        reading = sm.read("How many cases moved from KFI to Offer?")
        text, rows, refusal = sm.compose(reading, self._TWO_STAGE_WINDOW, money=str)
        self.assertIsNone(text)
        self.assertEqual(rows, [])
        self.assertIn("Offer is not a stage", refusal)
        self.assertIn("into", refusal)

    def test_a_stage_pair_with_no_transitions_says_so_rather_than_refusing(self):
        """The other half of the same rule. Both stages ARE governed here and no
        case made that move, which is an answer — nought — not a refusal."""
        text = answer("How many cases moved from KFI to Withdrawn?")
        self.assertIn("No cases moved from KFI to Withdrawn", text)

    def test_unavailable_capability_refuses_with_the_governed_reason(self):
        """A payload the capability marks unavailable is never rendered as zero."""
        from mi_agent_api import movement_detail as md
        from mi_agent_api import stage_movement_query as sm

        unavailable = md.stage_transition_unavailable(
            "client_001", reason_code=md.REASON_NO_COMPARISON,
            reason="There is no prior governed pipeline snapshot.")
        reading = sm.read("How many cases moved from KFI to Application?")
        text, rows, refusal = sm.compose(reading, unavailable, money=str)
        self.assertIsNone(text)
        self.assertEqual(rows, [])
        self.assertIn("no prior governed pipeline snapshot", refusal)

    def test_a_transition_is_never_answered_with_the_current_stage_stock(self):
        """THE STARTING-SHA DEFECT. Three cases are AT KFI in the latest extract;
        two MOVED from KFI to Application. Every formulation gets the transition."""
        for question in ("How many cases moved from KFI to Application?",
                         "How many KFI cases progressed to Application?",
                         "How many cases went from KFI into Application?",
                         "What number of cases transitioned KFI to Application?"):
            with self.subTest(question=question):
                text = answer(question)
                self.assertIn("2 cases", text)
                self.assertNotIn("3 loans", text)
                self.assertNotIn("3 cases", text)


# --------------------------------------------------------------------------- #
# 17-18. The delegation itself
# --------------------------------------------------------------------------- #
class TestDelegation(unittest.TestCase):

    def test_query_calls_the_governed_resolver(self):
        """Proven by observation, not by reading the source: the route's answer
        disappears when the governed resolver is unavailable."""
        from mi_agent_api import movement_detail as md

        calls = []
        original = md.resolve_stage_transition_detail

        def spy(*args, **kwargs):
            calls.append((args, kwargs))
            return original(*args, **kwargs)

        md.resolve_stage_transition_detail = spy
        try:
            answer("How many cases moved from KFI to Application?")
        finally:
            md.resolve_stage_transition_detail = original
        self.assertEqual(len(calls), 1,
                         "the route must call the governed resolver exactly once")

    def test_query_performs_no_stage_transition_arithmetic(self):
        """Every figure the adapter publishes is a key lookup on the governed
        payload. Swap the payload's numbers and the ANSWER follows them — which
        it could not do if the adapter recomputed anything."""
        from mi_agent_api import stage_movement_query as sm

        payload = {
            "available": True, "as_of_date": LATEST, "comparison_date": PRIOR,
            "transitions": [{"source_stage": "KFI", "destination_stage": "APPLICATION",
                             "case_count": 99, "prior_amount": 1.0,
                             "latest_amount": 1.0, "amount_change": 0.0}],
            "reconciliation": {"by_stage": [{"stage": "KFI"}, {"stage": "APPLICATION"}]},
        }
        reading = sm.read("How many cases moved from KFI to Application?")
        text, _, refusal = sm.compose(reading, payload, money=str)
        self.assertIsNone(refusal)
        self.assertIn("99 cases", text)

    def test_the_module_reads_no_snapshot_and_owns_no_stage_table(self):
        """A structural check on the adapter's own source: it must not import
        pandas, read a pipeline file, or restate the governed stage map."""
        import ast

        path = _REPO_ROOT / "mi_agent_api" / "stage_movement_query.py"
        tree = ast.parse(path.read_text())
        # Docstrings NAME what the module defers to, which is the point of them.
        # The check is on the CODE, so every string literal is stripped first.
        for node in ast.walk(tree):
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                node.value = ""
        code = ast.unparse(tree)
        for forbidden in ("pandas", "read_csv", "load_prepared_pipeline",
                          "weekly_extract_inventory", "_STAGE_CANON",
                          "groupby", "merge("):
            self.assertNotIn(forbidden, code,
                             "the adapter must not use %s" % forbidden)


# --------------------------------------------------------------------------- #
# 19-23. Route ownership — the near neighbours keep their owners
# --------------------------------------------------------------------------- #
class TestRouteOwnership(unittest.TestCase):
    """Recognition must not claim a question another capability owns.

    Asserted on the READER rather than end to end, so the assertion holds
    whatever fixture a deployment has: a question this reader declines cannot
    reach the stage-movement route at all.
    """

    def _declines(self, question: str) -> None:
        from mi_agent_api import stage_movement_query as sm

        self.assertIsNone(sm.read(question),
                          "%r must not be claimed as stage movement" % question)

    def test_funded_movement_is_not_stage_movement(self):
        self._declines("What is funded balance movement?")
        self._declines("Why did funded balance increase?")
        self._declines("Show movement by region.")

    def test_pipeline_evolution_is_not_stage_movement(self):
        self._declines("Show pipeline evolution.")
        self._declines("Show weekly pipeline cases.")
        self._declines("How has the pipeline changed over time?")

    def test_pipeline_stage_stock_is_not_stage_movement(self):
        """One stage named is a POSITION. It takes two, directionally."""
        self._declines("How much pipeline is currently in Offer?")
        self._declines("Show pipeline amount by stage.")
        self._declines("What is pipeline by stage?")

    def test_conversion_is_not_stage_movement(self):
        """Cohort conversion owns "conversion", including between two stages."""
        self._declines("What is the conversion rate?")
        self._declines("How has conversion changed?")
        self._declines("What is the KFI to Offer conversion rate?")

    def test_forecast_and_expectation_are_not_stage_movement(self):
        self._declines("What is forecast funded balance?")
        self._declines("How much of the offer pipeline is expected to complete?")

    def test_the_route_registers_after_every_existing_recogniser(self):
        """Deference is structural: last in the registry, default confidence."""
        from mi_agent_api import chat_routing  # noqa: F401 - populates REGISTRY
        from mi_agent_api.recogniser_registry import DEFAULT_CONFIDENCE, REGISTRY

        names = REGISTRY.names()
        self.assertEqual(names[-1], ROUTE,
                         "the stage-movement route must be evaluated last")
        entry = REGISTRY.get(ROUTE)
        self.assertIsNotNone(entry)
        self.assertGreater(entry.priority,
                           max(r.priority for r in REGISTRY.ordered()
                               if r.name != ROUTE))
        from mi_agent_api import stage_movement_query as sm

        verdict = sm.recognise(_FakeRequest(
            "How many cases moved from KFI to Application?"))
        self.assertTrue(verdict.matched)
        self.assertEqual(verdict.confidence, DEFAULT_CONFIDENCE)


class _FakeRequest:
    """The two things ``recognise`` touches on a request, and nothing else."""

    def __init__(self, question: str) -> None:
        self.question = question
        self._memo: Dict[str, Any] = {}

    def remember_recognition(self, key: str, value: Any) -> Any:
        self._memo[key] = value
        return value

    def recalled_recognition(self, key: str) -> Any:
        return self._memo.get(key)


if __name__ == "__main__":
    unittest.main()
