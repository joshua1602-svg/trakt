#!/usr/bin/env python3
"""A filter that matched everything is not a filter that never ran.

THE DEFECT. The receipt guard decides whether a stated threshold reached the
calculation by asking whether the POPULATION SHRANK:

    narrowed = bool(total is not None and after is not None and after < total)

    elif facet.kind == KIND_THRESHOLD:
        thresholds_seen += 1
        if narrowed and comparison_ops >= thresholds_seen:
            APPLIED
        else:
            LOST → "the threshold was not applied to the calculation"

Row count is a SIDE EFFECT of applying a predicate, not evidence of it. A
predicate every row satisfies is arithmetically indistinguishable from one that
was dropped, and the guard calls both of them lost:

    "How many pipeline cases are older than 30 days?"
        executor  → filter pipeline_case_age_days gt 30.0 kept 10/10 rows
        guard     → "I understood that you asked for over 30, but that could
                     not be applied to the calculation"

The figure was right, the filter ran, and the reader was told the question could
not be answered. That is the fail-closed machinery firing on a correct answer,
which costs exactly as much trust as a wrong one.

WHY IT SURVIVED THIS LONG. Almost every threshold in the standing banks removes
rows, so the heuristic agreed with the truth on nearly all of them. It is wrong
whenever a bound sits at or outside the range the book actually holds — a
concentrated portfolio, a young pipeline, a narrow product — and those are
exactly the books where an operator most needs to trust the answer.

THE EVIDENCE ALREADY EXISTS. `mi_query_executor` publishes
`metadata["applied_filter_fields"]` — the fields it actually filtered on — and
the POPULATION facet one branch above already reads it via `population_applied`.
The threshold branch is the one that guessed. This file pins that it stops.

The row-count heuristic is KEPT as a fallback, because a route that publishes no
executed-filter record still needs an answer, and the fallback can only ever add
APPLIED verdicts to what the record already proves.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


class _Result:
    def __init__(self, metadata, rows):
        self.metadata = metadata
        self.row_count = rows


class TestTheGuardReadsTheExecutedRecord(unittest.TestCase):

    def _assess(self, *, applied_fields, filters, total, after):
        from mi_agent.execution_receipt import (
            KIND_THRESHOLD, RequestedFacet, reconcile_facets)

        class _Spec:
            pass

        spec = _Spec()
        spec.filters = filters
        facets = [RequestedFacet(kind=KIND_THRESHOLD, label="over 30",
                                 field_key="pipeline_case_age_days")]
        result = _Result({"applied_filter_fields": applied_fields,
                          "reconciliation": {"total_records": total,
                                             "records_after_filters": after}},
                         after)
        reconcile_facets(facets, spec=spec, query_result=result,
                         semantics={"fields": {}}, available_columns=())
        return facets[0]

    def test_a_predicate_that_kept_every_row_is_applied(self):
        """THE DEFECT. Ten of ten rows satisfied the bound; it still ran."""
        facet = self._assess(
            applied_fields=["pipeline_case_age_days"],
            filters={"pipeline_case_age_days": {"op": "gt", "value": 30.0}},
            total=10, after=10)
        self.assertEqual(facet.status, "applied", facet.reason)

    def test_a_predicate_the_executor_never_recorded_is_still_lost(self):
        """The other half. Removing the heuristic must not make everything pass:
        an executor that filtered on nothing has not applied the threshold, and
        an unchanged row count is then genuine evidence of loss."""
        facet = self._assess(
            applied_fields=[],
            filters={"pipeline_case_age_days": {"op": "gt", "value": 30.0}},
            total=10, after=10)
        self.assertEqual(facet.status, "lost")

    def test_a_narrowing_predicate_is_applied_as_before(self):
        facet = self._assess(
            applied_fields=["current_interest_rate"],
            filters={"current_interest_rate": {"op": "gt", "value": 6.0}},
            total=10, after=5)
        self.assertEqual(facet.status, "applied", facet.reason)

    def test_the_row_count_fallback_still_answers_without_a_record(self):
        """A route that publishes no `applied_filter_fields` keeps the shipped
        behaviour rather than regressing to LOST."""
        facet = self._assess(
            applied_fields=None,
            filters={"current_interest_rate": {"op": "gt", "value": 6.0}},
            total=10, after=5)
        self.assertEqual(facet.status, "applied", facet.reason)


class TestTheLiveShapeAnswers(unittest.TestCase):
    """End to end on the governed pipeline frame, because the unit test above
    constructs its own metadata and could agree with a guard that no real
    execution reaches."""

    def test_a_case_age_threshold_answers_rather_than_refusing(self):
        from mi_agent_api.tests.test_stage_movement_query import ask

        envelope = ask("How many pipeline cases are older than 30 days?")
        self.assertTrue(envelope.get("ok"), envelope.get("error"))
        self.assertIn("10", envelope.get("answer") or "")

    def test_the_borrower_is_not_filtered_by_a_bound_stated_in_days(self):
        """§14B, from the live path rather than the parse."""
        from mi_agent_api.tests.test_stage_movement_query import ask

        diagnostics = " ".join(
            (ask("How many pipeline cases are older than 30 days?")
             .get("metadata") or {}).get("diagnostics") or ())
        self.assertIn("pipeline_case_age_days", diagnostics)
        self.assertNotIn("youngest_borrower_age", diagnostics)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
