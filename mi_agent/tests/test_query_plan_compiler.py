#!/usr/bin/env python3
"""One plan in, governed execution contracts out — and no second engine.

WHAT THE COMPILER IS FOR. A `QueryPlan` says what a request MEANS. The executor
already knows how to calculate: filtering, grouping, count, sum, averages,
weighted averages. The compiler is the seam between them, and its whole job is
to produce the contracts the executor already accepts.

    outputs sharing one effective scope
        → ONE existing MIQuerySpec, using the multi-measure path that ships

    outputs whose effective scopes differ
        → SEVERAL ordinary governed executions, one per population

The second case is what `MIQuerySpec` could never express: it carries a single
`filters` dict, so a request about two populations had to be flattened into one
and answered over whichever survived. Compiling to several specs — each of them
an ordinary spec the executor has always accepted — is what makes the third
clause of a composed question a real output rather than a collision with the
second.

THE OPTIMISATION MAY NEVER CHANGE THE SEMANTICS. Grouping compatible outputs
into one execution is an efficiency, and correctness of population outranks it:
two outputs go into one spec only when their effective scopes are structurally
equivalent, never when they merely look similar.

BACKWARD COMPATIBILITY IS THE POINT OF THE SEAM. A single-output plan must
compile to the spec the atomic path already produces — same metric, same
aggregation, same filters — so the new model sits ABOVE the deterministic layer
without disturbing it. That is asserted here rather than assumed, because a
compiler that quietly produced a different spec for simple questions would move
every atomic answer in the estate.
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
from mi_agent.query_plan_compiler import compile_query_plan      # noqa: E402

BALANCE = "current_outstanding_balance"
LTV = "current_loan_to_value"
JOINT = Predicate("borrower_type", "eq", "Joint")


def scope(*filters, dataset="funded", dimensions=()):
    return AnalyticalScope(dataset=dataset, period="2026-06-30",
                           filters=tuple(filters), dimensions=tuple(dimensions))


class TestOneOutputCompilesToTheAtomicContract(unittest.TestCase):
    """§9 / §25D — existing atomic analysis keeps the existing executor path."""

    def test_a_single_measure_plan_produces_one_ordinary_spec(self):
        plan = QueryPlan(shared_scope=scope(JOINT), outputs=(
            PlannedOutput(output_id="a", operation=SUM, measure=BALANCE),))
        executions = compile_query_plan(plan)
        self.assertEqual(len(executions), 1)
        spec = executions[0].spec
        self.assertEqual(spec.metric, BALANCE)
        self.assertEqual(spec.aggregation, SUM)
        self.assertEqual(spec.filters, {"borrower_type": "Joint"})

    def test_a_single_count_plan_produces_a_count_spec(self):
        plan = QueryPlan(shared_scope=scope(JOINT), outputs=(
            PlannedOutput(output_id="a", operation=COUNT),))
        spec = compile_query_plan(plan)[0].spec
        self.assertEqual(spec.aggregation, COUNT)
        self.assertIsNone(spec.metric)

    def test_a_threshold_survives_into_the_spec_in_the_executor_s_shape(self):
        plan = QueryPlan(shared_scope=scope(Predicate(LTV, "gt", 40.0)),
                         outputs=(PlannedOutput(output_id="a", operation=COUNT),))
        self.assertEqual(compile_query_plan(plan)[0].spec.filters,
                         {LTV: {"op": "gt", "value": 40.0}})


class TestSameScopeOutputsShareOneExecution(unittest.TestCase):
    """§8 / §10 — reuse the multi-measure path that already ships."""

    def test_three_same_scope_outputs_compile_to_one_execution(self):
        plan = QueryPlan(shared_scope=scope(JOINT), outputs=(
            PlannedOutput(output_id="a", operation=COUNT),
            PlannedOutput(output_id="b", operation=SUM, measure=BALANCE),
            PlannedOutput(output_id="c", operation=WEIGHTED_AVERAGE,
                          measure=LTV)))
        executions = compile_query_plan(plan)
        self.assertEqual(len(executions), 1)
        self.assertEqual(set(executions[0].output_ids), {"a", "b", "c"})
        self.assertEqual(
            [(m["field"], m["aggregation"]) for m in executions[0].spec.measures],
            [("loan_count", COUNT), (BALANCE, SUM), (LTV, WEIGHTED_AVERAGE)])

    def test_the_shared_population_reaches_that_execution(self):
        plan = QueryPlan(shared_scope=scope(JOINT), outputs=(
            PlannedOutput(output_id="a", operation=COUNT),
            PlannedOutput(output_id="b", operation=SUM, measure=BALANCE)))
        self.assertEqual(compile_query_plan(plan)[0].spec.filters,
                         {"borrower_type": "Joint"})


class TestDifferentScopesCompileToSeparateExecutions(unittest.TestCase):
    """§8 — the case MIQuerySpec could not express."""

    def _three_output_plan(self):
        return QueryPlan(shared_scope=scope(JOINT), outputs=(
            PlannedOutput(output_id="a", operation=COUNT),
            PlannedOutput(output_id="b", operation=SUM, measure=BALANCE),
            PlannedOutput(output_id="c", operation=SUM, measure=BALANCE,
                          local_scope_delta=ScopeDelta(
                              filters=(Predicate(LTV, "gt", 40.0),)))))

    def test_two_populations_produce_two_executions(self):
        executions = compile_query_plan(self._three_output_plan())
        self.assertEqual(len(executions), 2)

    def test_each_execution_carries_only_its_own_population(self):
        by_output = {}
        for execution in compile_query_plan(self._three_output_plan()):
            for output_id in execution.output_ids:
                by_output[output_id] = execution.spec.filters
        self.assertEqual(by_output["a"], {"borrower_type": "Joint"})
        self.assertEqual(by_output["b"], {"borrower_type": "Joint"})
        self.assertEqual(by_output["c"], {"borrower_type": "Joint",
                                          LTV: {"op": "gt", "value": 40.0}})

    def test_the_narrowed_output_is_not_folded_into_its_twin(self):
        """b and c share measure AND aggregation. Only the population differs,
        and that is exactly what made the third clause disappear before."""
        executions = compile_query_plan(self._three_output_plan())
        homes = {output_id: index
                 for index, execution in enumerate(executions)
                 for output_id in execution.output_ids}
        self.assertNotEqual(homes["b"], homes["c"])

    def test_every_requested_output_is_compiled_exactly_once(self):
        plan = self._three_output_plan()
        compiled = [output_id for execution in compile_query_plan(plan)
                    for output_id in execution.output_ids]
        self.assertEqual(sorted(compiled), ["a", "b", "c"])
        self.assertEqual(len(compiled), len(set(compiled)))


class TestTheCompilerCarriesTheScopeItCompiled(unittest.TestCase):
    """§11 — every execution can say which population it is for, without
    re-deriving it from the spec it produced."""

    def test_each_execution_reports_its_effective_scope(self):
        plan = QueryPlan(shared_scope=scope(JOINT), outputs=(
            PlannedOutput(output_id="a", operation=COUNT),
            PlannedOutput(output_id="c", operation=SUM, measure=BALANCE,
                          local_scope_delta=ScopeDelta(
                              filters=(Predicate(LTV, "gt", 40.0),)))))
        for execution in compile_query_plan(plan):
            for output_id in execution.output_ids:
                with self.subTest(output=output_id):
                    self.assertTrue(scope_equivalent(
                        execution.effective_scope,
                        effective_scope(plan, plan.output(output_id))))

    def test_the_dataset_and_period_travel_with_the_execution(self):
        """They select the FRAME and are not spec fields, so the compiler
        surfaces them rather than dropping them."""
        plan = QueryPlan(shared_scope=scope(JOINT, dataset="pipeline"),
                         outputs=(PlannedOutput(output_id="a", operation=COUNT),))
        execution = compile_query_plan(plan)[0]
        self.assertEqual(execution.effective_scope.dataset, "pipeline")
        self.assertEqual(execution.effective_scope.period, "2026-06-30")


class TestGrouping(unittest.TestCase):
    """A dimension is part of the population's shape, so it compiles with it."""

    def test_a_grouped_scope_produces_a_grouped_spec(self):
        plan = QueryPlan(
            shared_scope=scope(JOINT, dimensions=("erm_product_type",)),
            outputs=(PlannedOutput(output_id="a", operation=COUNT),
                     PlannedOutput(output_id="b", operation=SUM,
                                   measure=BALANCE)))
        spec = compile_query_plan(plan)[0].spec
        self.assertEqual(spec.dimension, "erm_product_type")


class TestGenerality(unittest.TestCase):
    """§25E — several canonical measures, filters and dimensions, with no
    production special case for any of them."""

    CASES = (
        (SUM, BALANCE, Predicate(LTV, "gt", 40.0)),
        (WEIGHTED_AVERAGE, LTV, Predicate("current_interest_rate", "ge", 7.0)),
        (WEIGHTED_AVERAGE, "current_interest_rate",
         Predicate("youngest_borrower_age", "ge", 85.0)),
        (SUM, BALANCE, Predicate("collateral_geography", "eq", "Scotland")),
        (COUNT, None, Predicate("pipeline_case_age_days", "gt", 30.0)),
    )

    def test_each_measure_and_filter_compiles_the_same_way(self):
        for operation, measure, predicate in self.CASES:
            with self.subTest(measure=measure, predicate=predicate.field):
                plan = QueryPlan(
                    shared_scope=scope(predicate),
                    outputs=(PlannedOutput(output_id="a", operation=operation,
                                           measure=measure),))
                spec = compile_query_plan(plan)[0].spec
                self.assertEqual(spec.metric, measure)
                self.assertEqual(spec.aggregation, operation)
                self.assertIn(predicate.field, spec.filters)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
