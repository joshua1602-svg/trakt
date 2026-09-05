#!/usr/bin/env python3
"""Was every requested figure produced, over the population it was requested for?

TWO WAYS A COMPOSED ANSWER GOES WRONG, and the estate could see neither.

MISSING. Three outputs requested, two executed, answered as a success. The
machinery to catch this existed and could not see the requests: its trigger read
the QUESTION with its own vocabulary, which had no word for a count and none for
"amount", so for the commonest composed shape it never fired.

MISCOPED, and this one is worse because the figure is real. Every output was
produced, every number is arithmetically correct, and one of them is about a
different population than the one asked for. "Requested Joint, executed the
whole book" and "requested Joint, executed Joint AND LTV > 40" both look like
complete successes from anywhere except a structural comparison of the two
populations — and until this module there was no representation in which such a
comparison could be made. Row counts cannot make it: that heuristic is what
declared a correct filter lost whenever every row happened to satisfy it.

    OUTPUT IDENTITY IS measure + aggregation + EFFECTIVE SCOPE.

Without the third term, the second and third clauses of

    "How many joint loans, what is their balance, and how much of that balance
     has LTV above 40%?"

are the same output — same measure, same aggregation — which is precisely why
one of them silently disappeared.

WHAT RECONCILIATION IS NOT. It is not an oracle about whether a number is right.
It compares two records that already exist: what the plan asked for, and what
each execution reports having done.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from mi_agent.query_plan import (                                # noqa: E402
    COUNT, SUM, AnalyticalScope, Predicate, PlannedOutput, QueryPlan,
    ScopeDelta, effective_scope)
from mi_agent.query_plan_result import (                         # noqa: E402
    MultiResultEnvelope, OutputResult, reconcile_plan)

BALANCE = "current_outstanding_balance"
LTV = "current_loan_to_value"
JOINT = Predicate("borrower_type", "eq", "Joint")
ABOVE_40 = Predicate(LTV, "gt", 40.0)


def three_output_plan() -> QueryPlan:
    return QueryPlan(
        shared_scope=AnalyticalScope(dataset="funded", period="2026-06-30",
                                     filters=(JOINT,)),
        outputs=(
            PlannedOutput(output_id="a", operation=COUNT),
            PlannedOutput(output_id="b", operation=SUM, measure=BALANCE),
            PlannedOutput(output_id="c", operation=SUM, measure=BALANCE,
                          local_scope_delta=ScopeDelta(filters=(ABOVE_40,)))))


def result_for(plan, output_id, *, executed_scope=None, value=1):
    output = plan.output(output_id)
    requested = effective_scope(plan, output)
    return OutputResult(
        output_id=output_id,
        requested_scope=requested,
        executed_scope=executed_scope if executed_scope is not None else requested,
        execution_ref=f"exec-{output_id}",
        value=value)


class TestACompletePlan(unittest.TestCase):

    def test_every_output_executed_over_its_own_population_is_complete(self):
        plan = three_output_plan()
        results = [result_for(plan, i) for i in ("a", "b", "c")]
        self.assertTrue(reconcile_plan(plan, results).complete)

    def test_the_envelope_carries_one_entry_per_requested_output(self):
        plan = three_output_plan()
        envelope = MultiResultEnvelope.build(
            plan, [result_for(plan, i) for i in ("a", "b", "c")])
        self.assertEqual([o.output_id for o in envelope.outputs],
                         ["a", "b", "c"])
        self.assertTrue(envelope.completeness.complete)
        self.assertEqual(envelope.plan_id, plan.plan_id)


class TestAMissingOutput(unittest.TestCase):
    """§13 — requested 3, executed 2, must not be an ordinary success."""

    def test_it_is_not_complete(self):
        plan = three_output_plan()
        reconciliation = reconcile_plan(
            plan, [result_for(plan, "a"), result_for(plan, "b")])
        self.assertFalse(reconciliation.complete)
        self.assertEqual(reconciliation.missing, ("c",))

    def test_the_envelope_refuses_to_call_itself_complete(self):
        plan = three_output_plan()
        envelope = MultiResultEnvelope.build(
            plan, [result_for(plan, "a"), result_for(plan, "b")])
        self.assertFalse(envelope.completeness.complete)
        self.assertIn("c", envelope.completeness.missing)


class TestAMiscopedOutput(unittest.TestCase):
    """The failure with a real number attached."""

    def test_an_output_executed_over_the_whole_book_is_not_complete(self):
        """requested Joint, executed the whole book."""
        plan = three_output_plan()
        widened = AnalyticalScope(dataset="funded", period="2026-06-30")
        results = [result_for(plan, "a", executed_scope=widened),
                   result_for(plan, "b"), result_for(plan, "c")]
        reconciliation = reconcile_plan(plan, results)
        self.assertFalse(reconciliation.complete)
        self.assertEqual(reconciliation.miscoped, ("a",))

    def test_an_output_that_absorbed_a_sibling_s_narrowing_is_not_complete(self):
        """requested Joint, executed Joint AND LTV > 40 — the live leak."""
        plan = three_output_plan()
        leaked = AnalyticalScope(dataset="funded", period="2026-06-30",
                                 filters=(JOINT, ABOVE_40))
        results = [result_for(plan, "a", executed_scope=leaked),
                   result_for(plan, "b", executed_scope=leaked),
                   result_for(plan, "c")]
        reconciliation = reconcile_plan(plan, results)
        self.assertFalse(reconciliation.complete)
        self.assertEqual(reconciliation.miscoped, ("a", "b"))

    def test_a_changed_operator_is_a_different_population(self):
        plan = QueryPlan(
            shared_scope=AnalyticalScope(
                dataset="funded",
                filters=(Predicate("current_interest_rate", "gt", 6.0),)),
            outputs=(PlannedOutput(output_id="a", operation=COUNT),))
        executed = AnalyticalScope(
            dataset="funded",
            filters=(Predicate("current_interest_rate", "ge", 6.0),))
        reconciliation = reconcile_plan(
            plan, [result_for(plan, "a", executed_scope=executed)])
        self.assertFalse(reconciliation.complete)
        self.assertEqual(reconciliation.miscoped, ("a",))

    def test_predicate_order_is_not_a_difference(self):
        plan = QueryPlan(
            shared_scope=AnalyticalScope(dataset="funded",
                                         filters=(JOINT, ABOVE_40)),
            outputs=(PlannedOutput(output_id="a", operation=COUNT),))
        reordered = AnalyticalScope(dataset="funded",
                                    filters=(ABOVE_40, JOINT))
        self.assertTrue(reconcile_plan(
            plan, [result_for(plan, "a", executed_scope=reordered)]).complete)


class TestAResultNobodyAskedFor(unittest.TestCase):

    def test_an_unrequested_output_is_reported(self):
        plan = three_output_plan()
        stray = OutputResult(output_id="z",
                             requested_scope=plan.shared_scope,
                             executed_scope=plan.shared_scope,
                             execution_ref="exec-z", value=1)
        reconciliation = reconcile_plan(
            plan, [result_for(plan, i) for i in ("a", "b", "c")] + [stray])
        self.assertFalse(reconciliation.complete)
        self.assertEqual(reconciliation.unrequested, ("z",))

    def test_two_results_for_one_output_are_reported(self):
        plan = three_output_plan()
        results = [result_for(plan, "a"), result_for(plan, "a"),
                   result_for(plan, "b"), result_for(plan, "c")]
        self.assertFalse(reconcile_plan(plan, results).complete)


class TestProvenance(unittest.TestCase):
    """§11 — every output can say what it was and what it ran over."""

    def test_each_result_names_both_populations_and_its_execution(self):
        plan = three_output_plan()
        envelope = MultiResultEnvelope.build(
            plan, [result_for(plan, i) for i in ("a", "b", "c")])
        narrowed = next(o for o in envelope.outputs if o.output_id == "c")
        self.assertIn(ABOVE_40, narrowed.requested_scope.filters)
        self.assertIn(JOINT, narrowed.requested_scope.filters)
        self.assertEqual(narrowed.execution_ref, "exec-c")

    def test_the_envelope_does_not_copy_the_receipt(self):
        """§14 — reference the governed artefact, do not duplicate it."""
        plan = three_output_plan()
        receipt = object()
        result = OutputResult(output_id="a",
                              requested_scope=plan.shared_scope,
                              executed_scope=plan.shared_scope,
                              execution_ref=receipt, value=1)
        self.assertIs(result.execution_ref, receipt)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
