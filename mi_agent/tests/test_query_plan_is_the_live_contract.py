#!/usr/bin/env python3
"""The live route runs through the plan, and nothing moves because of it.

D4. For a spec whose complete semantics a `QueryPlan` can carry, the request is
lifted and the spec that executes is the one the plan compiles to. The plan owns
what the request MEANS; the executor still owns every calculation.

WHY A NO-OP IS THE RIGHT OUTCOME. Routing production through a new contract is
only safe if the contract is provably lossless first, and the whole D1 argument
was that round trip. So the acceptance evidence for D4 is not that answers
changed — it is that they demonstrably did not, while the semantic ownership
moved.

WHAT MEASURING FIRST CAUGHT, recorded because each would have shipped:

  * the compiler RE-DERIVED presentation from the dimension count, which would
    have changed what a reader sees on 265 of 703 liftable questions — 37 line
    charts becoming text summaries — with identical analysis underneath;
  * `metric_defaulted` was dropped, turning a measure the parser SUBSTITUTED
    into one indistinguishable from a measure the reader named;
  * chart axes were normalised on 450 charted specs.

None of the three is a calculation, which is exactly why a test comparing
figures would have passed while the product got worse.
"""

from __future__ import annotations

import dataclasses
import sys
import unittest
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from mi_agent.llm_query_parser import _deterministic_parse           # noqa: E402
from mi_agent.mi_query_executor import _all_group_dims               # noqa: E402
from mi_agent.mi_query_spec import MIQuerySpec                       # noqa: E402
from mi_agent.mi_query_validator import load_mi_semantics            # noqa: E402
from mi_agent.query_plan_adapter import compiled_spec_for            # noqa: E402

_SEMANTICS = load_mi_semantics(
    str(_REPO_ROOT / "mi_agent" / "mi_semantics_field_registry.yaml"))

#: Deliberately spread across the shapes the plan models, and three it does not.
QUESTIONS = (
    "What is the total funded balance?",
    "How many funded loans are there?",
    "What is the weighted average LTV?",
    "What is the funded balance for joint borrowers?",
    "How many funded loans have an interest rate above 6%?",
    "Show balance by region",
    "Balance by LTV bucket by age bucket for joint borrowers",
    "Show balance by LTV band and age band",
    "Loan count by broker channel",
    "Average borrower age by product type",
)

#: These must NOT lift — §25 requires their behaviour to be untouched.
UNLIFTABLE = (
    "Which broker has the largest funded balance?",
    "Show the balance trend over the last 6 months",
    "Are we within our concentration limits?",
)


def parse(question):
    return _deterministic_parse(question, _SEMANTICS)[0]


class TestTheLiftPreservesWhatDecidesAnAnswer(unittest.TestCase):

    def test_the_executed_grouping_is_unchanged(self):
        """The executor's own authority on which axes exist."""
        for question in QUESTIONS:
            with self.subTest(question=question):
                spec = parse(question)
                planned = compiled_spec_for(spec)
                if planned is None:
                    continue
                self.assertEqual(_all_group_dims(planned),
                                 _all_group_dims(spec))

    def test_population_measure_and_aggregation_are_unchanged(self):
        for question in QUESTIONS:
            with self.subTest(question=question):
                spec = parse(question)
                planned = compiled_spec_for(spec)
                if planned is None:
                    continue
                self.assertEqual(planned.filters, spec.filters)
                self.assertEqual(planned.metric, spec.metric)
                self.assertEqual(planned.aggregation, spec.aggregation)
                self.assertEqual(
                    [(m["field"], m["aggregation"]) for m in planned.measures],
                    [(m["field"], m["aggregation"]) for m in spec.measures])

    def test_presentation_and_disclosure_are_unchanged(self):
        """The three divergences measuring caught before they shipped."""
        for question in QUESTIONS:
            with self.subTest(question=question):
                spec = parse(question)
                planned = compiled_spec_for(spec)
                if planned is None:
                    continue
                for field in ("intent", "chart_type", "output_format",
                              "metric_defaulted", "x", "y", "title"):
                    self.assertEqual(getattr(planned, field),
                                     getattr(spec, field), field)

    def test_only_the_grouping_normalisation_differs(self):
        """Stated as a closed set, so a NEW divergence fails here."""
        allowed = {"dimensions", "dimension"}
        names = [f.name for f in dataclasses.fields(MIQuerySpec)]
        for question in QUESTIONS:
            with self.subTest(question=question):
                spec = parse(question)
                planned = compiled_spec_for(spec)
                if planned is None:
                    continue
                differing = {n for n in names
                             if getattr(planned, n) != getattr(spec, n)}
                self.assertTrue(differing <= allowed, differing - allowed)


class TestUnliftableQuestionsKeepTheShippedPath(unittest.TestCase):
    """§25 — the 179 deliberately unmodelled shapes must not change."""

    def test_they_are_declined(self):
        for question in UNLIFTABLE:
            with self.subTest(question=question):
                self.assertIsNone(compiled_spec_for(parse(question)))


class TestTheLiveRouteRecordsWhichContractOwnedTheRequest(unittest.TestCase):

    def test_a_lifted_request_is_marked(self):
        import numpy as np
        import pandas as pd

        from mi_agent.mi_agent_workflow import run_mi_agent_query

        rng = np.random.default_rng(3)
        n = 120
        frame = pd.DataFrame({
            "loan_identifier": [f"L{i}" for i in range(n)],
            "current_outstanding_balance": rng.uniform(50_000, 400_000, n).round(2),
            "current_loan_to_value": rng.uniform(15, 70, n).round(1),
            "collateral_geography": rng.choice(["Scotland", "Wales"], n),
        })
        result = run_mi_agent_query("Show balance by region", frame, _SEMANTICS)
        self.assertTrue(result.get("ok"), result.get("error"))
        self.assertEqual(
            (result.get("parse_metadata") or {}).get("semantic_contract"),
            "query_plan")


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
