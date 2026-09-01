#!/usr/bin/env python3
"""The two defects the multivariate pipeline audit found, and their boundaries.

DEFECT A — a share divided by the wrong denominator.
    "What share of Offer pipeline is joint borrowers?" answered 19.2%: the
    numerator honoured both filters, the denominator was all forty pipeline
    cases rather than the ten at Offer. The governed answer is 59.68%.

DEFECT B — a region that never bound as a filter on the pipeline dataset.
    "How much Application-stage pipeline is in London?" refused with "No loans
    in this book match that filter ('london')" on a book holding four such
    cases worth 1,910,000 — a false statement about the client's data.

Half of these tests are about what must NOT change. A share of the whole book
must still divide by the whole book; an unknown region must still refuse; a
known region with no rows in the narrowed stage must say so without inventing a
figure; and the stage + temporal questions, whose semantics are deliberately
deferred, must behave exactly as they did.

The oracle is arithmetic on the governed prepared frame — see
``scripts/prove_multivariate_pipeline_fixture.py``:

    Offer total          4,960,000 · 10 cases     Offer joint   2,960,000 · 6
    Application total    4,345,000 · 12 cases     London only   1,910,000 · 4
    Offer / Scotland       450,000 · 1 case
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

PIPELINE_FIXTURE = _REPO_ROOT / "tests" / "fixtures" / "pipeline_multivariate"
PORTFOLIO = "client_001/mi_2026_06"
AS_OF = "2026-06-30"

_CLIENT = None


#: The environment these tests need, and what it replaced.
#:
#: THIS IS SHARED STATE AND IT LEAKS BOTH WAYS. The client is a module-level
#: singleton, so the env was set ONCE on first use and then left standing for
#: the rest of the pytest session: a later module inherited this book and this
#: pipeline fixture, and `test_channel_parity` failed against a tape it never
#: asked for. The mirror image bit too — once another module had overwritten
#: these keys, the singleton was already built, so nothing re-applied them and
#: THESE tests failed instead. Both were invisible while this was the only
#: module using the harness and nothing ran between its first and last test.
#:
#: So the env is re-applied on EVERY call (idempotent and cheap) and restored
#: at module teardown. `test_calibration_only` shares this harness and does the
#: same, which is what keeps the two modules from stranding each other.
_ENV: Dict[str, str] = {}
_SAVED_ENV: Dict[str, Any] = {}


def _apply_env() -> None:
    for key, value in _ENV.items():
        _SAVED_ENV.setdefault(key, os.environ.get(key))
        os.environ[key] = value


def restore_env() -> None:
    """Put back what was there before this module's first request."""
    for key, value in _SAVED_ENV.items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value


def _client():
    global _CLIENT
    if _CLIENT is None:
        import importlib.util
        import tempfile

        spec = importlib.util.spec_from_file_location(
            "_mv_bank_runner",
            _REPO_ROOT / "scripts" / "run_mi_query_stage_movement_banks.py")
        runner = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(runner)

        root = Path(tempfile.mkdtemp(prefix="mv_fix_")) / "onboarding_output"
        for run_id, date, rows in runner.MONTHS:
            runner.write_funded_tape(root, run_id, date, rows)
        _ENV.update({
            "MI_AGENT_ONBOARDING_OUTPUT_ROOT": str(root),
            "MI_AGENT_PIPELINE_ROOT": str(PIPELINE_FIXTURE),
            "MI_AGENT_AUTH_ENABLED": "false",
            "MI_AGENT_LLM_PARSER": os.environ.get("MI_AGENT_LLM_PARSER", "off"),
            "MI_AGENT_CONCEPT_MERGE": "off",
        })
        _apply_env()
        from fastapi.testclient import TestClient
        from mi_agent_api.app import app

        _CLIENT = TestClient(app)
    _apply_env()
    return _CLIENT


def tearDownModule():
    restore_env()


def ask(question: str) -> Dict[str, Any]:
    return _client().post("/mi/query", json={
        "question": question, "portfolioId": PORTFOLIO, "asOfDate": AS_OF}).json()


def answer(question: str) -> str:
    env = ask(question)
    assert env.get("ok"), "%r was declined: %s" % (
        question, env.get("error") or env.get("answer"))
    return env.get("answer") or ""


# --------------------------------------------------------------------------- #
# DEFECT A — the denominator is the population the question named
# --------------------------------------------------------------------------- #
class TestFilteredShareDenominator(unittest.TestCase):

    def test_share_of_a_named_stage_uses_that_stage_as_the_denominator(self):
        """2,960,000 of the 4,960,000 at Offer = 59.68%, not of all 15,400,000."""
        text = answer("What share of Offer pipeline is joint borrowers?")
        self.assertIn("59.7", text)
        self.assertNotIn("19.2", text)

    def test_the_denominator_population_is_stated_and_is_the_stage(self):
        """Ten cases are at Offer. The receipt must show the reader that."""
        text = answer("What share of Offer pipeline is joint borrowers?")
        self.assertIn("Population Total: 10", text)
        self.assertNotIn("Population Total: 40", text)

    def test_the_numerator_still_honours_both_filters(self):
        """Six joint cases at Offer — the numerator was never the defect."""
        text = answer("What share of Offer pipeline is joint borrowers?")
        self.assertIn("6 loans", text)

    def test_a_whole_book_share_still_divides_by_the_whole_book(self):
        """THE REGRESSION THAT MATTERS. "of the book" names no governed
        population, so no contextual narrowing survives and the denominator is
        the whole book — exactly as before this change."""
        text = answer("What share of the book is drawdown?")
        self.assertIn("Population Total: 640", text)

    def test_a_whole_book_threshold_share_is_unchanged(self):
        text = answer("What proportion of the book is below 75% LTV?")
        self.assertIn("Population Total: 640", text)

    def test_a_whole_book_categorical_share_is_unchanged(self):
        text = answer("What proportion of the book is in Scotland?")
        self.assertIn("Population Total: 640", text)

    def test_the_selection_reader_claims_only_what_it_can_resolve(self):
        """Unit-level, on the two shapes that decide the denominator.

        A claimed selection leaves the REST of the sentence as the denominator
        population. An unclaimed one leaves nothing, and the denominator stays
        the whole book — which is today's behaviour, and is the direction this
        reader is built to fail in.
        """
        from mi_agent import llm_query_parser as parser
        from mi_agent.mi_query_validator import load_mi_semantics
        from mi_agent_api.data_source import semantics_path

        semantics = load_mi_semantics(semantics_path())
        columns = {"borrower_type", "pipeline_stage",
                   "current_outstanding_balance"}
        values = {"borrower_type": {"joint": "Joint", "single": "Single"}}

        self.assertEqual(
            parser._share_selection_fields(
                "What share of Offer pipeline is joint borrowers?",
                semantics, columns, values),
            ["borrower_type"])

        # No selection clause at all, and a clause naming nothing governed.
        for question in ("Summarise the portfolio.",
                         "What share of the book is unicorns?"):
            with self.subTest(question=question):
                self.assertEqual(
                    parser._share_selection_fields(question, semantics,
                                                   columns, values), [])

    def test_the_executor_divides_by_the_whole_frame_when_nothing_is_claimed(self):
        """The executor's own guard, exercised rather than described.

        Same spec, same frame, two selection settings. With nothing claimed the
        denominator is all six rows (today's behaviour); with the stage claimed
        as context the denominator is the two Offer rows.
        """
        import pandas as pd

        from mi_agent.mi_query_executor import _execute_share
        from mi_agent.mi_query_spec import MIQuerySpec

        frame = pd.DataFrame({
            "pipeline_stage": ["OFFER", "OFFER", "KFI", "KFI", "KFI", "KFI"],
            "borrower_type": ["Joint", "Single", "Joint", "Joint",
                              "Single", "Single"],
        })
        work = frame[(frame.pipeline_stage == "OFFER")
                     & (frame.borrower_type == "Joint")]
        semantics = {"fields": {
            "pipeline_stage": {"role": "dimension",
                               "canonical_field": "pipeline_stage"},
            "borrower_type": {"role": "dimension",
                              "canonical_field": "borrower_type"}}}
        filters = {"pipeline_stage": "OFFER", "borrower_type": "Joint"}

        whole = MIQuerySpec(intent="summary", aggregation="share",
                            filters=filters, share_selection_fields=[])
        data, _ = _execute_share(whole, frame, work, semantics, [], None)
        self.assertEqual(int(data.iloc[0]["population_total"]), 6)

        scoped = MIQuerySpec(intent="summary", aggregation="share",
                             filters=filters,
                             share_selection_fields=["borrower_type"])
        data, _ = _execute_share(scoped, frame, work, semantics, [], None)
        self.assertEqual(int(data.iloc[0]["population_total"]), 2)
        self.assertAlmostEqual(
            float(data.iloc[0]["loan_count_share_pct"]), 50.0, places=6)


# --------------------------------------------------------------------------- #
# DEFECT B — a governed region binds as a filter on the pipeline dataset
# --------------------------------------------------------------------------- #
class TestPipelineRegionFilter(unittest.TestCase):

    def test_stage_plus_region_binds_and_answers(self):
        for question in (
                "How much Application-stage pipeline is in London?",
                "What is the Application pipeline balance for London?",
                "How much pipeline at Application is in London?",
                "Show Application-stage exposure in London."):
            with self.subTest(question=question):
                text = answer(question)
                self.assertIn("1.9MM", text)
                self.assertIn("4 loans", text)
                self.assertNotIn("4.3MM", text)

    def test_another_governed_region_works_too(self):
        """The mechanism is governed, not a London special case."""
        text = answer("How much Offer-stage pipeline is in Scotland?")
        self.assertIn("450K", text)

    def test_an_unknown_region_still_refuses_safely(self):
        """No fabricated value, and no whole-book figure in its place."""
        env = ask("How much Application-stage pipeline is in Atlantis?")
        self.assertFalse(env.get("ok"))
        body = (env.get("answer") or "") + (env.get("error") or "")
        self.assertIn("atlantis", body.lower())
        self.assertNotIn("4.3MM", body)

    def test_a_known_region_absent_from_the_narrowed_stage_is_not_invented(self):
        """Wales holds pipeline cases but none at COMPLETED. The answer must be
        empty or a refusal — never a figure borrowed from a wider population."""
        env = ask("How much Completed-stage pipeline is in Wales?")
        body = (env.get("answer") or "") + (env.get("error") or "")
        for borrowed in ("2.1MM", "4.3MM", "5.0MM"):
            self.assertNotIn(borrowed, body)

    def test_the_funded_book_region_filter_is_untouched(self):
        """Region already bound on funded and must keep binding, on its own
        governed field."""
        env = ask("What is the funded balance in Scotland?")
        self.assertTrue(env.get("ok"))
        self.assertIn("geographic_region_obligor",
                      str((env.get("spec") or {}).get("filters")))

    def test_the_catalogue_collapses_a_duplicated_dimension_not_a_named_one(self):
        """Unit-level proof that the rule is about DATA, not about a name list.

        Two dimensions whose columns are element-wise identical are one
        narrowing; two that share a vocabulary but differ per row stay two, and
        the ambiguity rule still protects them.
        """
        import pandas as pd

        from mi_agent import execution_receipt as receipt

        semantics = {"fields": {
            "a": {"role": "dimension", "canonical_field": "a"},
            "b": {"role": "dimension", "canonical_field": "b"},
            "c": {"role": "dimension", "canonical_field": "c"},
        }}
        twin = pd.DataFrame({"a": ["London", "Wales"],
                             "b": ["London", "Wales"],      # identical -> one
                             "c": ["Wales", "London"]})     # same words, differs
        catalogue = receipt.book_values(twin, semantics)
        self.assertIn("a", catalogue)
        self.assertNotIn("b", catalogue, "an identical column is catalogued once")
        self.assertIn("c", catalogue, "a genuinely different column is kept")

        from mi_agent.categorical_spans import value_field
        self.assertIsNone(value_field("london", catalogue),
                          "a value two DIFFERENT fields claim is still ambiguous")


# --------------------------------------------------------------------------- #
# OUT OF SCOPE — deliberately unchanged
# --------------------------------------------------------------------------- #
class TestDeferredBehaviourIsUnchanged(unittest.TestCase):
    """The audit found more than two gaps. Only two were repaired."""

    def test_stage_plus_previous_month_still_refuses_without_substituting(self):
        """PIPELINE TEMPORAL SEMANTICS ARE DEFERRED. Weekly governed snapshots
        may not carry a monthly comparison, and choosing one is a decision this
        sprint did not take. The refusal must stand, and no prior-week figure
        may stand in for a prior month."""
        for question in (
                "How much pipeline is in Application stage and how does that "
                "compare with the previous month?",
                "What is Application pipeline now versus a month ago?",
                "Compare current Application-stage balance with the previous month.",
                "How has Application-stage pipeline changed over the last month?"):
            with self.subTest(question=question):
                env = ask(question)
                self.assertFalse(env.get("ok"),
                                 "stage + temporal must still refuse")

    def test_the_deferred_recognition_gaps_are_still_gaps(self):
        """Not fixed here, and not accidentally fixed either."""
        for question in ("How much Offer-stage pipeline is on cases over 500k?",
                         "Show Offer pipeline where the loan amount exceeds 500k."):
            with self.subTest(question=question):
                self.assertFalse(ask(question).get("ok"))


# --------------------------------------------------------------------------- #
# The successful constructions the audit measured must survive both fixes
# --------------------------------------------------------------------------- #
class TestWorkingConstructionsPreserved(unittest.TestCase):

    def test_stage_plus_borrower_type_plus_balance(self):
        text = answer("How much Offer pipeline is joint borrowers?")
        self.assertIn("3.0MM", text)
        self.assertIn("6 loans", text)

    def test_stage_plus_borrower_type_plus_weighted_ltv(self):
        """The hardest construction the audit found working — 53.7%, weighted,
        with BOTH narrowings bound. Neither may disappear."""
        env = ask("What is WA LTV for joint borrowers in Application?")
        self.assertTrue(env.get("ok"))
        text = env.get("answer") or ""
        self.assertIn("53.7", text)
        self.assertNotIn("55.9", text)   # the borrower filter dropped
        self.assertNotIn("49.7", text)   # an unweighted mean
        filters = str((env.get("spec") or {}).get("filters"))
        self.assertIn("APPLICATION", filters)
        self.assertIn("Joint", filters)

    def test_stage_plus_ltv_threshold(self):
        text = answer("How much Offer-stage pipeline has LTV above 60%?")
        self.assertIn("2.8MM", text)
        self.assertIn("4 loans", text)

    def test_stage_plus_weighted_ltv(self):
        self.assertIn("58.4", answer("What is WA LTV for Offer-stage pipeline?"))

    def test_stage_plus_region_grouping(self):
        env = ask("Break down Offer pipeline by region.")
        self.assertTrue(env.get("ok"))
        rows = max([len(a.get("rows") or [])
                    for a in (env.get("artifacts") or [])] or [0])
        self.assertGreaterEqual(rows, 5)

    def test_two_dimensional_stage_by_borrower_type(self):
        env = ask("Break down pipeline balance by stage and single versus "
                  "joint borrower.")
        self.assertTrue(env.get("ok"))
        rows = max([len(a.get("rows") or [])
                    for a in (env.get("artifacts") or [])] or [0])
        self.assertGreaterEqual(rows, 6)


if __name__ == "__main__":
    unittest.main()
