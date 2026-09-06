#!/usr/bin/env python3
"""The analytical contracts, exercised as objects rather than as sentences.

WHY THIS FILE HAS NO QUESTIONS IN IT. Every composition defect this programme
has found was a disagreement between two owners about one concept, and every
one of them was first observed through a sentence — which is why they were
mistaken for parser bugs for so long. The semantics below are properties of the
CONTRACTS: they hold or fail without any natural language reaching them, and a
regression here cannot be explained away as a phrasing the parser has not
learnt.

WHAT THE CONTRACTS MEAN

    AnalyticalScope   which rows a calculation is about
    ScopeDelta        a narrowing of a scope, belonging to ONE output
    PlannedOutput     one requested figure: operation, measure, aggregation,
                      and optionally its own narrowing
    QueryPlan         one shared scope and the outputs asked of it

The invariant the whole structure exists to hold, and the defect it replaces:

    "How many joint loans are there, what is their balance, and how much of
     that balance has LTV above 40%?"

applied the LTV bound to every output, so the count and the balance were
computed over a population neither clause asked for. A delta belongs to its
output. It may not reach the shared scope, and it may not reach a sibling.

SCOPE IDENTITY IS STRUCTURAL. `interest_rate > 6` and `interest_rate >= 6`
select different rows, so they are different scopes; two scopes with the same
predicates in a different order are the same scope. Neither fact can be read
off a rendered string, and neither can be inferred from a row count — which is
how the receipt guard used to decide whether a filter had run, and why it
refused correct answers whenever a bound matched every row.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from mi_agent.query_plan import (                                # noqa: E402
    COUNT, SUM, WEIGHTED_AVERAGE, AnalyticalScope, Predicate, PlannedOutput,
    QueryPlan, ScopeDelta, effective_scope, scope_equivalent)

BALANCE = "current_outstanding_balance"
LTV = "current_loan_to_value"
RATE = "current_interest_rate"


def joint_book() -> AnalyticalScope:
    return AnalyticalScope(
        dataset="funded",
        portfolio_lens="total",
        period="2026-06-30",
        filters=(Predicate("borrower_type", "eq", "Joint"),))


def plan_with_local_delta() -> QueryPlan:
    """The sentence that motivated the structure, as a plan."""
    return QueryPlan(
        shared_scope=joint_book(),
        outputs=(
            PlannedOutput(output_id="a", operation=COUNT),
            PlannedOutput(output_id="b", operation=SUM, measure=BALANCE),
            PlannedOutput(
                output_id="c", operation=SUM, measure=BALANCE,
                local_scope_delta=ScopeDelta(
                    filters=(Predicate(LTV, "gt", 40.0),))),
        ))


# --------------------------------------------------------------------------- #
# §17 — one output, shared scope
# --------------------------------------------------------------------------- #
class TestOneOutput(unittest.TestCase):

    def test_its_effective_scope_is_the_shared_scope(self):
        scope = joint_book()
        plan = QueryPlan(shared_scope=scope,
                         outputs=(PlannedOutput(output_id="a", operation=COUNT),))
        self.assertTrue(scope_equivalent(effective_scope(plan, plan.outputs[0]),
                                         scope))


# --------------------------------------------------------------------------- #
# §17 — several outputs, one scope
# --------------------------------------------------------------------------- #
class TestSeveralOutputsOneScope(unittest.TestCase):

    def test_every_output_resolves_to_the_shared_scope(self):
        scope = joint_book()
        plan = QueryPlan(shared_scope=scope, outputs=(
            PlannedOutput(output_id="a", operation=COUNT),
            PlannedOutput(output_id="b", operation=SUM, measure=BALANCE),
            PlannedOutput(output_id="c", operation=WEIGHTED_AVERAGE,
                          measure=LTV)))
        for output in plan.outputs:
            with self.subTest(output=output.output_id):
                self.assertTrue(
                    scope_equivalent(effective_scope(plan, output), scope))

    def test_a_count_is_an_output_in_its_own_right(self):
        """Not a property of a top-level metric. The estate's five count owners
        all existed because a count had nowhere to be."""
        output = PlannedOutput(output_id="a", operation=COUNT)
        self.assertEqual(output.operation, COUNT)
        self.assertIsNone(output.measure)


# --------------------------------------------------------------------------- #
# §17 — one narrowed output, and no sibling leakage
# --------------------------------------------------------------------------- #
class TestOutputLocalNarrowing(unittest.TestCase):

    def test_the_narrowed_output_sees_both_predicates(self):
        plan = plan_with_local_delta()
        scope = effective_scope(plan, plan.outputs[2])
        self.assertIn(Predicate(LTV, "gt", 40.0), scope.filters)
        self.assertIn(Predicate("borrower_type", "eq", "Joint"), scope.filters)

    def test_the_siblings_do_not_see_it(self):
        """THE INVARIANT. This is the live defect, stated structurally."""
        plan = plan_with_local_delta()
        for output in plan.outputs[:2]:
            with self.subTest(output=output.output_id):
                scope = effective_scope(plan, output)
                self.assertTrue(scope_equivalent(scope, plan.shared_scope))
                self.assertNotIn(Predicate(LTV, "gt", 40.0), scope.filters)

    def test_the_shared_scope_is_not_mutated(self):
        """§18. Resolving an output may not change the plan it came from."""
        plan = plan_with_local_delta()
        before = plan.shared_scope
        for output in plan.outputs:
            effective_scope(plan, output)
        self.assertTrue(scope_equivalent(plan.shared_scope, before))
        self.assertEqual(len(plan.shared_scope.filters), 1)

    def test_a_scope_cannot_be_mutated_in_place(self):
        scope = joint_book()
        with self.assertRaises(Exception):
            scope.filters = ()                                   # type: ignore


# --------------------------------------------------------------------------- #
# §12 — population equivalence is structural
# --------------------------------------------------------------------------- #
class TestScopeEquivalence(unittest.TestCase):

    def test_predicate_order_does_not_change_a_population(self):
        one = AnalyticalScope(dataset="funded", filters=(
            Predicate(LTV, "gt", 40.0), Predicate(RATE, "gt", 6.0)))
        other = AnalyticalScope(dataset="funded", filters=(
            Predicate(RATE, "gt", 6.0), Predicate(LTV, "gt", 40.0)))
        self.assertTrue(scope_equivalent(one, other))

    def test_a_missing_narrowing_is_a_different_population(self):
        """requested Joint, executed whole book."""
        self.assertFalse(scope_equivalent(
            joint_book(), AnalyticalScope(dataset="funded")))

    def test_an_extra_narrowing_is_a_different_population(self):
        """requested Joint, executed Joint + LTV > 40 — the leak, detected."""
        narrowed = AnalyticalScope(dataset="funded", filters=(
            Predicate("borrower_type", "eq", "Joint"),
            Predicate(LTV, "gt", 40.0)))
        self.assertFalse(scope_equivalent(joint_book(), narrowed))

    def test_a_different_operator_is_a_different_population(self):
        """requested rate > 6, executed rate >= 6. Same field, same number,
        different rows."""
        self.assertFalse(scope_equivalent(
            AnalyticalScope(dataset="funded",
                            filters=(Predicate(RATE, "gt", 6.0),)),
            AnalyticalScope(dataset="funded",
                            filters=(Predicate(RATE, "ge", 6.0),))))

    def test_a_different_dataset_is_a_different_population(self):
        self.assertFalse(scope_equivalent(
            AnalyticalScope(dataset="funded"),
            AnalyticalScope(dataset="pipeline")))

    def test_a_different_period_is_a_different_population(self):
        self.assertFalse(scope_equivalent(
            AnalyticalScope(dataset="funded", period="2026-06-30"),
            AnalyticalScope(dataset="funded", period="2026-03-31")))


# --------------------------------------------------------------------------- #
# §7 — the delta narrows; it does not replace, contradict or widen
# --------------------------------------------------------------------------- #
class TestScopeAlgebra(unittest.TestCase):

    def test_a_delta_that_contradicts_the_shared_scope_fails_closed(self):
        """Two bounds on one field are a range or a mistake, and this sprint
        does not guess which. Refusing is the governed answer."""
        from mi_agent.query_plan import ScopeConflict

        plan = QueryPlan(
            shared_scope=AnalyticalScope(
                dataset="funded", filters=(Predicate(LTV, "gt", 40.0),)),
            outputs=(PlannedOutput(
                output_id="a", operation=COUNT,
                local_scope_delta=ScopeDelta(
                    filters=(Predicate(LTV, "lt", 20.0),))),))
        with self.assertRaises(ScopeConflict):
            effective_scope(plan, plan.outputs[0])

    def test_a_delta_may_not_widen(self):
        """A delta that removes a shared narrowing would make an output cover
        MORE than the request. There is no representation for it, by design."""
        self.assertFalse(hasattr(ScopeDelta((), ), "remove_filters"))

    def test_an_empty_delta_is_the_shared_scope(self):
        plan = QueryPlan(shared_scope=joint_book(), outputs=(
            PlannedOutput(output_id="a", operation=COUNT,
                          local_scope_delta=ScopeDelta(filters=())),))
        self.assertTrue(scope_equivalent(
            effective_scope(plan, plan.outputs[0]), plan.shared_scope))


# --------------------------------------------------------------------------- #
# §18 — stable identity
# --------------------------------------------------------------------------- #
class TestIdentity(unittest.TestCase):

    def test_every_output_has_a_stable_id(self):
        plan = plan_with_local_delta()
        ids = [o.output_id for o in plan.outputs]
        self.assertEqual(len(set(ids)), len(ids))
        self.assertEqual(ids, [o.output_id for o in plan.outputs])

    def test_a_plan_rejects_duplicate_output_ids(self):
        with self.assertRaises(ValueError):
            QueryPlan(shared_scope=joint_book(), outputs=(
                PlannedOutput(output_id="a", operation=COUNT),
                PlannedOutput(output_id="a", operation=SUM, measure=BALANCE)))

    def test_output_identity_includes_its_effective_scope(self):
        """§13. Two outputs with the same measure and aggregation are NOT the
        same output when their populations differ — which is exactly the third
        clause of the motivating sentence, and why it collapsed into the second
        when identity was measure+aggregation alone."""
        plan = plan_with_local_delta()
        b, c = plan.outputs[1], plan.outputs[2]
        self.assertEqual((b.operation, b.measure), (c.operation, c.measure))
        self.assertFalse(scope_equivalent(effective_scope(plan, b),
                                          effective_scope(plan, c)))


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
