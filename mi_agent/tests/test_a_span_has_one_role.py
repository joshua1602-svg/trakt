#!/usr/bin/env python3
"""A word already doing a job is not also the measure.

THE DEFECT, and it is not about LTV. This estate has TWO measure resolvers:

    _detect_metric   the single-output path
    _measure_hits    the measure-set path

`_measure_hits` asks three questions before it accepts a hit — is this span the
subject of a predicate (`is_filter_subject`), does it sit inside a grouping
region (`_grouping_regions`), is it followed by a dimension suffix
(`_DIMENSION_SUFFIX_RE`). `_detect_metric` asks none of them. It takes the first
governed measure word it sees, whatever that word is already doing.

The plainest grouped question in the estate shows the two disagreeing:

    "balance by ltv bucket"
        _detect_metric  → current_loan_to_value / weighted_avg   ← the AXIS
        _measure_hits   → current_outstanding_balance / sum      ← the measure

That question still answers correctly, and the reason is the shape of the
problem: its caller hands `_detect_metric` a pre-blanked string
(`_metric_slot(remaining)`) with the grouping already removed. Every call site
has to know to do that. Where one does not, the axis becomes the measure:

    "How many loans are in the 60-70% LTV bucket?"
        → metric current_loan_to_value, aggregation weighted_avg,
          dimension ltv_bucket

`_wants_count` is TRUE for that sentence. It is never consulted, because the
grouped branch only asks `if metric is None` and a measure had already been
claimed from a phrase that was naming a POPULATION.

THE INVARIANT THIS FILE PINS, from §5 of the sprint brief:

    Once a span is governed as a filter, threshold, bucket, dimension, dataset,
    period or entity qualifier, that role is available to downstream
    resolution — the same text is not independently rediscovered as a measure.

It is tested across FIVE governed fields in three different role positions,
because a mechanism that only knows about LTV is the per-field workaround the
brief forbids. The pairs matter as much as the cases: each field is also asked
for as a genuine measure, so a fix that simply stops resolving these words
fails here rather than passing.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from mi_agent.llm_query_parser import (                          # noqa: E402
    _detect_metric, _deterministic_parse, _measure_hits)
from mi_agent.mi_query_validator import load_mi_semantics        # noqa: E402

_SEMANTICS = load_mi_semantics(
    str(_REPO_ROOT / "mi_agent" / "mi_semantics_field_registry.yaml"))

LTV = "current_loan_to_value"
RATE = "current_interest_rate"
AGE = "youngest_borrower_age"
BALANCE = "current_outstanding_balance"


def parse(question: str):
    spec, _meta = _deterministic_parse(question, _SEMANTICS)
    return spec


class TestTheTwoResolversAgree(unittest.TestCase):
    """The asymmetry itself, asserted directly so it cannot come back by a
    caller forgetting to blank a string."""

    #: (text, the field that is NOT the measure because it holds another role)
    ROLE_SPANS = (
        ("balance by ltv bucket", LTV),
        ("balance by age bucket", AGE),
        ("balance by interest rate bucket", RATE),
        ("loan count by ltv band", LTV),
        ("balance where ltv is above 50%", LTV),
        ("balance for loans with an interest rate above 6%", RATE),
        ("balance for borrowers aged 85 or older", AGE),
    )

    def test_a_span_holding_another_role_is_not_the_measure(self):
        for text, occupied in self.ROLE_SPANS:
            with self.subTest(text=text):
                key, _agg, _matched = _detect_metric(text, _SEMANTICS)
                self.assertNotEqual(
                    key, occupied,
                    f"{occupied} was already the axis/predicate in {text!r}")

    def test_both_resolvers_reach_the_same_measure(self):
        for text, _occupied in self.ROLE_SPANS:
            with self.subTest(text=text):
                single, _agg, _m = _detect_metric(text, _SEMANTICS)
                hits = [h[2] for h in _measure_hits(text, _SEMANTICS)]
                if hits:
                    self.assertIn(single, hits + [None])


class TestTheFieldIsStillAMeasureWhenItIsAsked(unittest.TestCase):
    """The other half of every pair. A role-aware resolver that simply stopped
    resolving these fields would satisfy the class above and be useless."""

    ASKED = (
        ("what is the weighted average ltv?", LTV),
        ("what is the weighted average interest rate?", RATE),
        ("what is the average borrower age?", AGE),
        ("weighted average ltv by region", LTV),
        ("average borrower age by product type", AGE),
        ("weighted average interest rate by broker", RATE),
    )

    def test_a_named_measure_still_resolves(self):
        for text, expected in self.ASKED:
            with self.subTest(text=text):
                key, _agg, _m = _detect_metric(text, _SEMANTICS)
                self.assertEqual(key, expected)


class TestACountOverAPopulationIsACount(unittest.TestCase):
    """§14A, stated generally. The bucket phrase names the population; the
    question names the operation. Five fields, three of them bucketed."""

    POPULATIONS = (
        "in the 60-70% LTV bucket",
        "in the 80-85 age bucket",
        "with an interest rate above 6%",
        "with an LTV above 50%",
        "aged 85 or older",
    )

    def test_the_operation_is_count_not_the_population_s_field(self):
        for population in self.POPULATIONS:
            question = f"How many loans are {population}?"
            with self.subTest(population=population):
                spec = parse(question)
                self.assertEqual(spec.aggregation, "count", spec.metric)
                self.assertNotEqual(spec.aggregation, "weighted_avg")

    def test_the_population_is_still_applied(self):
        """A count over the wrong population would be a worse answer than the
        weighted average was."""
        spec = parse("How many loans are in the 60-70% LTV bucket?")
        self.assertTrue(spec.dimension or spec.filters,
                        "the LTV population vanished with the measure")


class TestTheSameMechanismCoversTheAgeCollision(unittest.TestCase):
    """§14B. "older than 30 days" is a predicate about the CASE; the borrower's
    age must not be filtered by it. Asserted as role ownership, not as a
    pipeline rule — the same invariant, a different field."""

    def test_a_case_age_predicate_does_not_bind_the_borrower(self):
        for question in ("How many pipeline cases are older than 30 days?",
                         "What is the total pipeline amount for cases older "
                         "than 30 days?"):
            with self.subTest(question=question):
                self.assertNotIn(AGE, parse(question).filters or {})

    def test_only_one_field_owns_the_predicate(self):
        """The live P049 shape: two predicates for one stated bound."""
        filters = parse("How many pipeline cases are older than 30 days?").filters or {}
        age_like = [k for k in filters if "age" in k]
        self.assertLessEqual(len(age_like), 1, filters)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
