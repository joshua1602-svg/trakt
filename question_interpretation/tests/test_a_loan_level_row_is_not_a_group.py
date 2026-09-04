#!/usr/bin/env python3
"""A loan-level table reports ROWS, so it has no axis to give a concept.

MEASURED ON THE DEPLOYED BUILD, by the 115-question replay of what real users
asked. "Where was the greatest pipeline attrition?" had ANSWERED before and came
back:

    parsed dimension(s) neither applied nor rejected: pipeline_stage.
    Refusing to answer with a silently dropped dimension.

The deterministic parser builds that question as a loan-level ranking —
`aggregation="loan_level"`, no dimension — and reproducing the parse at both the
before and after commits over 48 column/value combinations gives byte-identical
specs, so the parse did not move. What moved is that the CONCEPT-MERGE ARM
proposed `pipeline stage` as a dimension and `_apply_to_spec` filled the empty
slot. A loan-level result has no group columns, so the dimension was then
neither applied nor rejected, and the invariant refused the answer — correctly.
The arm may change whether Trakt answers; it may not change what it answers.

THE SEAM ALREADY EXISTED AND WAS ONE ENTRY SHORT. `OperationProfile
.accepts_grouping_axis` is exactly this rule, and `_AGGREGATIONS_WITHOUT_AN_AXIS`
listed only `share`. `loan_level` belongs beside it for the same reason, held to
the same measured bar: across the 882-question corpus the deterministic parser
builds `loan_level` 29 times and carries a dimension in NONE of them — as with
`share`, this stops the merge building a shape the parser never builds.

Reproduced with the arm stubbed, so the test needs no model, no credit and no
network: `mi_agent_api/tests/test_the_arm_may_not_add_an_axis.py`.
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from mi_agent.mi_query_spec import MIQuerySpec
from question_interpretation import claim_merge as CM


def _spec(aggregation):
    return MIQuerySpec(intent="table", chart_type="none", output_format="table",
                       metric="current_outstanding_balance",
                       aggregation=aggregation, title="t")


class TestWhichOperationsCarryAnAxis(unittest.TestCase):

    def test_a_loan_level_table_accepts_no_grouping_axis(self):
        self.assertFalse(
            CM.operation_profile(_spec("loan_level")).accepts_grouping_axis)

    def test_a_share_still_accepts_none(self):
        """The entry this joins. It must not move."""
        self.assertFalse(CM.operation_profile(_spec("share")).accepts_grouping_axis)

    def test_every_grouping_aggregation_still_accepts_one(self):
        """The 660 corpus questions that DO group. A profile that refused these
        would silence the arm on the analytics it exists to complete."""
        for aggregation in ("sum", "count", "weighted_avg", "avg", "median"):
            with self.subTest(aggregation=aggregation):
                self.assertTrue(
                    CM.operation_profile(_spec(aggregation)).accepts_grouping_axis)

    def test_the_declined_axis_is_recorded_not_silently_dropped(self):
        """A refusal the estate cannot see is the defect in the other
        direction. The merge reports the role it would not give."""
        from question_interpretation.concept_proposal import (BoundConcept,
                                                              ProposedConcept)

        proposal = ProposedConcept("dimension", "pipeline stage", "pipeline attrition")
        bound = BoundConcept(proposal, "pipeline_stage", None, "test")
        result = CM.merge(CM.deterministic_slots(None), [bound], [],
                          profile=CM.operation_profile(_spec("loan_level")))
        self.assertEqual(list(result.filled_by_model), [])
        self.assertTrue(result.findings, "the decline left no finding")


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
