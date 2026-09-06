#!/usr/bin/env python3
"""A plan, run through the real executor, judged on the real receipts.

Everything until now was structural: contracts agreeing with contracts. This
file runs plans against the governed executor on a real frame and reads the
population back out of what the executor published, because the two failures
composition has to catch are only visible there.

    MISSING     three requested, two executed
    MISCOPED    all three executed, one over the wrong population

The second is the one that reached production: "For joint borrowers, give me
the count, the balance and the WA LTV" returned three correct figures over the
whole book, and every check the estate had said complete.

NO ROW COUNTS ARE CONSULTED, here or in the module under test. A predicate the
executor did not record applying is absent from the executed scope, and the
structural comparison then fails — regardless of how many rows survived.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from mi_agent.mi_query_validator import load_mi_semantics            # noqa: E402
from mi_agent.query_plan import (                                    # noqa: E402
    COUNT, SUM, WEIGHTED_AVERAGE, AnalyticalScope, Predicate, PlannedOutput,
    QueryPlan, ScopeDelta)
from mi_agent.query_plan_execution import execute_query_plan         # noqa: E402

BALANCE = "current_outstanding_balance"
LTV = "current_loan_to_value"
JOINT = Predicate("borrower_type", "eq", "Joint")
ABOVE_40 = Predicate(LTV, "gt", 40.0)

_SEMANTICS = load_mi_semantics(
    str(_REPO_ROOT / "mi_agent" / "mi_semantics_field_registry.yaml"))


def book(n: int = 200) -> pd.DataFrame:
    rng = np.random.default_rng(11)
    return pd.DataFrame({
        "loan_identifier": [f"L{i:04d}" for i in range(n)],
        BALANCE: rng.uniform(60_000, 420_000, n).round(2),
        LTV: rng.uniform(15, 75, n).round(1),
        "current_interest_rate": rng.uniform(3, 9, n).round(2),
        "youngest_borrower_age": rng.integers(60, 92, n),
        "borrower_type": rng.choice(["Joint", "Single"], n),
        "collateral_geography": rng.choice(["Scotland", "Wales"], n),
        "erm_product_type": rng.choice(["Lump Sum", "Drawdown"], n),
    })


def joint_scope() -> AnalyticalScope:
    return AnalyticalScope(dataset="funded", period="2026-06-30",
                           filters=(JOINT,))


class TestSameScopeMultiOutput(unittest.TestCase):
    """§6 / §25E — adding an output cannot change the shared population."""

    def _plan(self):
        return QueryPlan(shared_scope=joint_scope(), outputs=(
            PlannedOutput(output_id="a", operation=COUNT),
            PlannedOutput(output_id="b", operation=SUM, measure=BALANCE),
            PlannedOutput(output_id="c", operation=WEIGHTED_AVERAGE,
                          measure=LTV, weight_field=BALANCE)))

    def test_it_runs_as_one_execution_and_reconciles(self):
        envelope = execute_query_plan(self._plan(), book(), _SEMANTICS)
        self.assertTrue(envelope.completeness.complete,
                        envelope.completeness.reason())
        self.assertEqual(len(envelope.outputs), 3)

    def test_every_output_ran_over_the_requested_population(self):
        envelope = execute_query_plan(self._plan(), book(), _SEMANTICS)
        for output in envelope.outputs:
            with self.subTest(output=output.output_id):
                self.assertIn(JOINT, output.executed_scope.filters)

    def test_the_figures_come_from_one_execution(self):
        envelope = execute_query_plan(self._plan(), book(), _SEMANTICS)
        refs = {id(o.execution_ref) for o in envelope.outputs}
        self.assertEqual(len(refs), 1)

    def test_the_count_matches_the_frame_the_executor_filtered(self):
        frame = book()
        envelope = execute_query_plan(self._plan(), frame, _SEMANTICS)
        count = next(o for o in envelope.outputs if o.output_id == "a")
        self.assertEqual(int(count.value),
                         int((frame["borrower_type"] == "Joint").sum()))


class TestOutputLocalNarrowing(unittest.TestCase):
    """§7 / §25F — the capability MIQuerySpec could not express, executed."""

    def _plan(self):
        return QueryPlan(shared_scope=joint_scope(), outputs=(
            PlannedOutput(output_id="a", operation=COUNT),
            PlannedOutput(output_id="b", operation=SUM, measure=BALANCE),
            PlannedOutput(output_id="c", operation=SUM, measure=BALANCE,
                          local_scope_delta=ScopeDelta(filters=(ABOVE_40,)))))

    def test_it_reconciles_complete(self):
        envelope = execute_query_plan(self._plan(), book(), _SEMANTICS)
        self.assertTrue(envelope.completeness.complete,
                        envelope.completeness.reason())

    def test_the_narrowed_output_ran_over_both_predicates(self):
        envelope = execute_query_plan(self._plan(), book(), _SEMANTICS)
        narrowed = next(o for o in envelope.outputs if o.output_id == "c")
        self.assertIn(JOINT, narrowed.executed_scope.filters)
        self.assertIn(ABOVE_40, narrowed.executed_scope.filters)

    def test_no_sibling_ran_over_the_local_narrowing(self):
        """§16D. The live defect, now checkable from the execution record."""
        envelope = execute_query_plan(self._plan(), book(), _SEMANTICS)
        for output_id in ("a", "b"):
            with self.subTest(output=output_id):
                sibling = next(o for o in envelope.outputs
                               if o.output_id == output_id)
                self.assertNotIn(ABOVE_40, sibling.executed_scope.filters)

    def test_the_figures_differ_because_the_populations_do(self):
        frame = book()
        envelope = execute_query_plan(self._plan(), frame, _SEMANTICS)
        whole = next(o for o in envelope.outputs if o.output_id == "b")
        narrowed = next(o for o in envelope.outputs if o.output_id == "c")
        joint = frame["borrower_type"] == "Joint"
        self.assertAlmostEqual(float(whole.value),
                               float(frame.loc[joint, BALANCE].sum()), places=2)
        self.assertAlmostEqual(
            float(narrowed.value),
            float(frame.loc[joint & (frame[LTV] > 40), BALANCE].sum()), places=2)

    def test_two_populations_became_two_governed_executions(self):
        envelope = execute_query_plan(self._plan(), book(), _SEMANTICS)
        refs = {id(o.execution_ref) for o in envelope.outputs}
        self.assertEqual(len(refs), 2)


class TestAWrongPopulationIsCaught(unittest.TestCase):
    """§11 / §16F — the failure with a real number attached.

    A finding worth recording: the executor ALREADY fails closed when a
    requested filter names a column the book does not carry. It raises rather
    than quietly executing over the whole book, so that particular route cannot
    produce the live false positive on its own. The production defect came from
    a layer above — the parser never produced the `borrower_type` predicate at
    all, so nothing downstream had anything to drop.

    What reconciliation adds is the case the executor cannot see: an execution
    that RAN and recorded applying fewer predicates than the output asked for.
    Both halves are asserted, because "the executor raises" is only a guarantee
    while it stays true.
    """

    def test_the_executor_refuses_a_filter_it_cannot_apply(self):
        from mi_agent.mi_query_executor import MIQueryExecutionError

        frame = book().drop(columns=["borrower_type"])
        plan = QueryPlan(shared_scope=joint_scope(), outputs=(
            PlannedOutput(output_id="a", operation=COUNT),))
        with self.assertRaises(MIQueryExecutionError):
            execute_query_plan(plan, frame, _SEMANTICS, validate=False)

    def test_an_execution_that_dropped_a_predicate_is_miscoped(self):
        """The executed scope is read from `applied_filter_fields`, so an
        execution reporting fewer applied predicates than the output requested
        cannot reconcile — whatever its figures say."""
        from mi_agent.query_plan_compiler import compile_query_plan
        from mi_agent.query_plan_execution import _executed_scope

        plan = QueryPlan(shared_scope=joint_scope(), outputs=(
            PlannedOutput(output_id="a", operation=COUNT),))
        execution = compile_query_plan(plan)[0]

        class _RanButNarrowedNothing:
            metadata = {"applied_filter_fields": []}
            data = None
            result_type = "summary"

        executed = _executed_scope(execution, _RanButNarrowedNothing())
        self.assertEqual(executed.filters, ())
        from mi_agent.query_plan_result import OutputResult, reconcile_plan
        reconciliation = reconcile_plan(plan, [OutputResult(
            output_id="a", requested_scope=execution.effective_scope,
            executed_scope=executed, execution_ref=None, value=1)])
        self.assertFalse(reconciliation.complete)
        self.assertEqual(reconciliation.miscoped, ("a",))


class TestGenerality(unittest.TestCase):
    """§17 — several categories, one path. No production branch per category."""

    CASES = (
        ("categorical", Predicate("collateral_geography", "eq", "Scotland")),
        ("numeric threshold", Predicate("current_interest_rate", "gt", 6.0)),
        ("borrower structure", JOINT),
        ("age bound", Predicate("youngest_borrower_age", "ge", 80.0)),
    )

    def test_each_population_executes_and_reconciles(self):
        for name, predicate in self.CASES:
            with self.subTest(population=name):
                plan = QueryPlan(
                    shared_scope=AnalyticalScope(dataset="funded",
                                                 filters=(predicate,)),
                    outputs=(PlannedOutput(output_id="a", operation=COUNT),
                             PlannedOutput(output_id="b", operation=SUM,
                                           measure=BALANCE)))
                envelope = execute_query_plan(plan, book(), _SEMANTICS)
                self.assertTrue(envelope.completeness.complete,
                                envelope.completeness.reason())
                for output in envelope.outputs:
                    self.assertIn(predicate, output.executed_scope.filters)

    def test_a_grouped_plan_executes_over_its_axis(self):
        plan = QueryPlan(
            shared_scope=AnalyticalScope(dataset="funded", filters=(JOINT,),
                                         dimensions=("erm_product_type",)),
            outputs=(PlannedOutput(output_id="a", operation=COUNT),
                     PlannedOutput(output_id="b", operation=SUM,
                                   measure=BALANCE)))
        envelope = execute_query_plan(plan, book(), _SEMANTICS)
        self.assertTrue(envelope.completeness.complete,
                        envelope.completeness.reason())
        for output in envelope.outputs:
            self.assertIn("erm_product_type", output.executed_scope.dimensions)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
