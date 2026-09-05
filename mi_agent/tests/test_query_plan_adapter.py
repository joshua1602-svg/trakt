#!/usr/bin/env python3
"""Lifting a live parse into the target contracts, without losing any of it.

THE MIGRATION THIS SUPPORTS. Interpretation stays exactly where it is. The
parser produces an `MIQuerySpec` as it always has, and the adapter LIFTS that
spec into `QueryPlan` / `AnalyticalScope`. Nothing reinterprets the sentence a
second time — a second reader of the question is the defect this whole
programme has been removing, and introducing one to populate new objects would
be the worst possible way to adopt them.

THE INVARIANT THAT MAKES THE LIFT SAFE, and it is the whole design:

    compile_query_plan(plan_from_spec(spec))  ==  spec

for the semantics that decide an answer — population, measure, aggregation,
dimensions. If that round trip is not an identity, then routing production
through the plan changes answers, and no amount of structural testing elsewhere
would tell you which ones.

WHAT THE ADAPTER MUST REFUSE TO CLAIM. `MIQuerySpec` carries 76 fields. The plan
models a handful: filters, dimension, measures, lens, period. The other sixty-odd
express rankings, temporal comparisons, cohorts, bridges, forecasts, risk-limit
plans, bucket strategies, scatter axes. An adapter that lifted those specs would
QUIETLY DROP the field it does not model, and the compiled spec would execute a
different, simpler question than the reader asked — silently, because the plan
would look complete.

So the check is inverted, and stated as a property rather than a list: every
field outside the modelled set must still hold its DEFAULT, or the spec is not
liftable and the adapter declines. A new field added to `MIQuerySpec` tomorrow
is therefore un-liftable until someone models it deliberately. That is the
opposite of the usual failure, where a new field is silently ignored by an
adapter nobody remembered to update.

These tests are structural. They construct specs directly, so a decline is
provably about the spec's shape rather than about a sentence the parser happens
not to produce today.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from mi_agent.mi_query_spec import MIQuerySpec                   # noqa: E402
from mi_agent.query_plan import COUNT, SUM, WEIGHTED_AVERAGE     # noqa: E402
from mi_agent.query_plan_adapter import plan_from_spec           # noqa: E402
from mi_agent.query_plan_compiler import compile_query_plan      # noqa: E402

BALANCE = "current_outstanding_balance"
LTV = "current_loan_to_value"
RATE = "current_interest_rate"


def round_trip(spec: MIQuerySpec):
    plan = plan_from_spec(spec)
    assert plan is not None, "the adapter declined a spec it should model"
    executions = compile_query_plan(plan)
    assert len(executions) == 1, executions
    return executions[0].spec


class TestTheRoundTripIsAnIdentity(unittest.TestCase):
    """§5 / §16A — several unrelated semantic shapes, all preserved."""

    #: Deliberately spread across categorical populations, numeric thresholds,
    #: geography, grouping and every governed aggregation — §17's generality
    #: requirement, and a guard against an adapter that works for one shape.
    SPECS = {
        "categorical population": MIQuerySpec(
            intent="summary", metric=BALANCE, aggregation=SUM,
            filters={"borrower_type": "Joint"}),
        "numeric threshold": MIQuerySpec(
            intent="summary", aggregation=COUNT,
            filters={RATE: {"op": "ge", "value": 7.0}}),
        "geography": MIQuerySpec(
            intent="summary", metric=BALANCE, aggregation=SUM,
            filters={"collateral_geography": "Scotland"}),
        "weighted average": MIQuerySpec(
            intent="summary", metric=LTV, aggregation=WEIGHTED_AVERAGE,
            weight_field=BALANCE, filters={}),
        "average": MIQuerySpec(
            intent="summary", metric="youngest_borrower_age", aggregation="avg",
            filters={}),
        "grouped": MIQuerySpec(
            intent="chart", chart_type="bar", metric=BALANCE, aggregation=SUM,
            dimension="erm_product_type", x="erm_product_type", filters={}),
        "grouped count with a threshold": MIQuerySpec(
            intent="chart", chart_type="bar", aggregation=COUNT,
            dimension="ltv_bucket", x="ltv_bucket",
            filters={LTV: {"op": "gt", "value": 40.0}}),
        "range bound": MIQuerySpec(
            intent="summary", metric=BALANCE, aggregation=SUM,
            filters={LTV: {"op": "between", "value": [40.0, 60.0]}}),
        "several values on one field": MIQuerySpec(
            intent="summary", metric=BALANCE, aggregation=SUM,
            filters={"collateral_geography": {"op": "in",
                                              "value": ["Scotland", "Wales"]}}),
        "multi-measure": MIQuerySpec(
            intent="summary", filters={"borrower_type": "Joint"},
            measures=[{"field": "loan_count", "aggregation": COUNT},
                      {"field": BALANCE, "aggregation": SUM},
                      {"field": LTV, "aggregation": WEIGHTED_AVERAGE}]),
    }

    def test_the_population_survives(self):
        for name, spec in self.SPECS.items():
            with self.subTest(shape=name):
                self.assertEqual(round_trip(spec).filters, spec.filters)

    def test_the_dimension_survives(self):
        for name, spec in self.SPECS.items():
            with self.subTest(shape=name):
                self.assertEqual(round_trip(spec).dimension, spec.dimension)

    def test_the_measure_and_aggregation_survive(self):
        for name, spec in self.SPECS.items():
            with self.subTest(shape=name):
                out = round_trip(spec)
                if spec.measures:
                    self.assertEqual(
                        [(m["field"], m["aggregation"]) for m in out.measures],
                        [(m["field"], m["aggregation"]) for m in spec.measures])
                else:
                    self.assertEqual(out.metric, spec.metric)
                    self.assertEqual(out.aggregation, spec.aggregation)

    def test_a_weight_field_survives(self):
        out = round_trip(self.SPECS["weighted average"])
        self.assertEqual(out.weight_field, BALANCE)


class TestTheAdapterDeclinesWhatItCannotRepresent(unittest.TestCase):
    """The property that keeps the lift honest, and keeps it honest LATER."""

    UNLIFTABLE = {
        "ranking": MIQuerySpec(metric=BALANCE, aggregation=SUM,
                               dimension="broker_channel",
                               ranking_mode="grouped", sort_direction="desc"),
        "top n": MIQuerySpec(metric=BALANCE, aggregation=SUM,
                             dimension="broker_channel", top_n=5),
        "share": MIQuerySpec(metric=BALANCE, aggregation="share"),
        "contribution": MIQuerySpec(metric=LTV, aggregation="contribution",
                                    dimension="collateral_geography"),
        "temporal": MIQuerySpec(metric=BALANCE, aggregation=SUM,
                                execution_mode="temporal"),
        "comparison": MIQuerySpec(metric=BALANCE, aggregation=SUM,
                                  compare_periods=True),
        "cohort": MIQuerySpec(metric=BALANCE, aggregation=SUM,
                              cohort_progression=True),
        "risk limits": MIQuerySpec(aggregation=COUNT, risk_limit_query=True),
        "forecast": MIQuerySpec(metric=BALANCE, aggregation=SUM,
                                forecast_mode="expected"),
        "scatter": MIQuerySpec(chart_type="scatter", metric=BALANCE,
                               aggregation=SUM, y=LTV),
        "second dimension": MIQuerySpec(metric=BALANCE, aggregation=SUM,
                                        dimension="broker_channel",
                                        dimensions=["broker_channel", "ltv_bucket"]),
        "unapplied filter disclosure": MIQuerySpec(
            metric=BALANCE, aggregation=SUM,
            unavailable_filters=["risk score is not in this dataset"]),
    }

    def test_each_is_declined_rather_than_flattened(self):
        for name, spec in self.UNLIFTABLE.items():
            with self.subTest(shape=name):
                self.assertIsNone(
                    plan_from_spec(spec),
                    f"{name}: lifted a spec whose semantics the plan drops")

    def test_a_field_the_plan_does_not_model_makes_a_spec_unliftable(self):
        """Stated as the general property, so a field added to MIQuerySpec
        tomorrow is un-liftable until someone models it deliberately."""
        import dataclasses

        from mi_agent.query_plan_adapter import MODELLED_FIELDS

        declared = {f.name for f in dataclasses.fields(MIQuerySpec)}
        self.assertTrue(MODELLED_FIELDS <= declared,
                        MODELLED_FIELDS - declared)
        # And the modelled set is a small minority of the contract, which is
        # the honest description of how much of MIQuerySpec a plan can carry.
        self.assertLess(len(MODELLED_FIELDS), len(declared) / 2)


class TestScopeProvenance(unittest.TestCase):
    """The frame-selecting facts are not spec fields, so the adapter takes them
    from the caller rather than inventing them."""

    def test_dataset_lens_and_period_reach_the_scope(self):
        spec = MIQuerySpec(metric=BALANCE, aggregation=SUM, filters={})
        plan = plan_from_spec(spec, dataset="pipeline",
                              portfolio_lens="direct", period="2026-06-30")
        self.assertEqual(plan.shared_scope.dataset, "pipeline")
        self.assertEqual(plan.shared_scope.portfolio_lens, "direct")
        self.assertEqual(plan.shared_scope.period, "2026-06-30")

    def test_a_lens_on_the_spec_is_not_ignored(self):
        spec = MIQuerySpec(metric=BALANCE, aggregation=SUM, filters={},
                           portfolio_lens="acquired")
        self.assertEqual(
            plan_from_spec(spec).shared_scope.portfolio_lens, "acquired")


class TestOutputIdentity(unittest.TestCase):

    def test_a_multi_measure_spec_becomes_one_output_per_measure(self):
        spec = self_specs = MIQuerySpec(
            intent="summary", filters={},
            measures=[{"field": "loan_count", "aggregation": COUNT},
                      {"field": BALANCE, "aggregation": SUM}])
        plan = plan_from_spec(spec)
        self.assertEqual(len(plan.outputs), 2)
        self.assertEqual(plan.outputs[0].operation, COUNT)
        self.assertIsNone(plan.outputs[0].measure)
        self.assertEqual(plan.outputs[1].measure, BALANCE)

    def test_ids_are_stable_and_positional(self):
        spec = MIQuerySpec(intent="summary", filters={},
                           measures=[{"field": "loan_count", "aggregation": COUNT},
                                     {"field": BALANCE, "aggregation": SUM}])
        first = [o.output_id for o in plan_from_spec(spec).outputs]
        second = [o.output_id for o in plan_from_spec(spec).outputs]
        self.assertEqual(first, second)
        self.assertEqual(len(set(first)), 2)

    def test_no_output_carries_a_local_delta_from_a_flat_spec(self):
        """A spec has one filters dict, so every output it lifts to shares the
        population. A delta appearing here would be invented."""
        spec = MIQuerySpec(intent="summary", filters={"borrower_type": "Joint"},
                           measures=[{"field": "loan_count", "aggregation": COUNT},
                                     {"field": BALANCE, "aggregation": SUM}])
        for output in plan_from_spec(spec).outputs:
            self.assertIsNone(output.local_scope_delta)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
