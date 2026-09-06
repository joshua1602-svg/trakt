#!/usr/bin/env python3
"""Does the MI Query Agent's number equal an independently calculated one?

WHAT THIS LAYER ADDS. Every other acceptance layer checks that the product
agrees with ITSELF — the spec named the right field, the receipt recorded the
right filter, the plan reconciled against the execution. All necessary; none of
it evidence that a NUMBER is right. The live composition audit made the gap
concrete: three figures with correct field names and correct aggregations,
computed over the whole book instead of the joint one, and every check the
estate had said complete.

`portfolio_truth_oracle` imports nothing from the product. It is pandas and
explicit column names, and it is deliberately simpler than the thing it judges.
Where the product is cleverer — registry resolution, bucket engines,
percent-scale detection, missing-value policy — is exactly where the comparison
earns its keep.

GROUPED RESULTS ARE COMPARED CELL BY CELL, not by total. A cross-tab whose
totals agree can still have the mass in the wrong cells, and a total is the one
number a wrong grouping is most likely to get right.

NO NATURAL LANGUAGE. The bank constructs plans directly. That is deliberate: it
separates "can the analytical engine compute this correctly" from "can the
parser recognise this phrasing", and the second question already has three banks
of its own. A failure here is arithmetic or population, never vocabulary.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from mi_agent.mi_query_validator import load_mi_semantics            # noqa: E402
from mi_agent.query_plan import (                                    # noqa: E402
    AVERAGE, COUNT, SUM, WEIGHTED_AVERAGE, AnalyticalScope, PlannedOutput,
    Predicate, QueryPlan, ScopeDelta)
from mi_agent.query_plan_execution import execute_query_plan         # noqa: E402
from mi_agent.tests import portfolio_truth_oracle as truth           # noqa: E402

BALANCE = truth.BALANCE
LTV = truth.LTV
RATE = truth.RATE
AGE = truth.AGE

_SEMANTICS = load_mi_semantics(
    str(_REPO_ROOT / "mi_agent" / "mi_semantics_field_registry.yaml"))
_BOOK = truth.canonical_book()

JOINT = ("borrower_type", "eq", "Joint")
SCOTLAND = ("collateral_geography", "eq", "Scotland")
LUMP_SUM = ("erm_product_type", "eq", "Lump Sum")
ALPHA = ("broker_channel", "eq", "Alpha")
RATE_OVER_6 = (RATE, "gt", 6.0)
LTV_OVER_40 = (LTV, "gt", 40.0)


def plan_for(outputs, *, predicates=(), dimensions=()):
    return QueryPlan(
        shared_scope=AnalyticalScope(
            dataset="funded", period="2026-06-30",
            filters=tuple(Predicate(*p) for p in predicates),
            dimensions=tuple(dimensions)),
        outputs=tuple(outputs))


def run(plan):
    envelope = execute_query_plan(plan, _BOOK, _SEMANTICS, validate=False)
    assert envelope.completeness.complete, envelope.completeness.reason()
    return envelope


def cells(envelope, output_id, column):
    """``{group key tuple: figure}`` from the executed frame."""
    result = next(o for o in envelope.outputs
                  if o.output_id == output_id).execution_ref
    frame = result.data
    keys = list((result.metadata or {}).get("group_field_keys") or ())
    return {tuple(str(row[k]) for k in keys): float(row[column])
            for _, row in frame.iterrows()}


# --------------------------------------------------------------------------- #
# 1. Simple measures
# --------------------------------------------------------------------------- #
class TestSimpleMeasures(unittest.TestCase):

    def test_loan_count(self):
        envelope = run(plan_for([PlannedOutput(output_id="a", operation=COUNT)]))
        self.assertEqual(int(envelope.outputs[0].value), truth.row_count(_BOOK))

    def test_total_balance(self):
        envelope = run(plan_for([PlannedOutput(output_id="a", operation=SUM,
                                               measure=BALANCE)]))
        self.assertAlmostEqual(float(envelope.outputs[0].value),
                               truth.total(_BOOK, BALANCE), places=2)

    def test_weighted_average_ltv(self):
        envelope = run(plan_for([PlannedOutput(
            output_id="a", operation=WEIGHTED_AVERAGE, measure=LTV,
            weight_field=BALANCE)]))
        self.assertAlmostEqual(
            float(envelope.outputs[0].value),
            truth.weighted_average(_BOOK, LTV, BALANCE), places=4)

    def test_weighted_average_rate(self):
        envelope = run(plan_for([PlannedOutput(
            output_id="a", operation=WEIGHTED_AVERAGE, measure=RATE,
            weight_field=BALANCE)]))
        self.assertAlmostEqual(
            float(envelope.outputs[0].value),
            truth.weighted_average(_BOOK, RATE, BALANCE), places=4)

    def test_average_borrower_age(self):
        envelope = run(plan_for([PlannedOutput(output_id="a", operation=AVERAGE,
                                               measure=AGE)]))
        self.assertAlmostEqual(float(envelope.outputs[0].value),
                               float(_BOOK[AGE].mean()), places=4)


# --------------------------------------------------------------------------- #
# 2. Single filters — five populations, one path
# --------------------------------------------------------------------------- #
class TestFilteredMeasures(unittest.TestCase):

    POPULATIONS = {
        "borrower structure": JOINT,
        "geography": SCOTLAND,
        "product": LUMP_SUM,
        "broker": ALPHA,
        "numeric threshold": RATE_OVER_6,
    }

    def test_the_filtered_count_matches_truth(self):
        for name, predicate in self.POPULATIONS.items():
            with self.subTest(population=name):
                envelope = run(plan_for(
                    [PlannedOutput(output_id="a", operation=COUNT)],
                    predicates=[predicate]))
                self.assertEqual(int(envelope.outputs[0].value),
                                 truth.row_count(_BOOK, [predicate]))

    def test_the_filtered_balance_matches_truth(self):
        for name, predicate in self.POPULATIONS.items():
            with self.subTest(population=name):
                envelope = run(plan_for(
                    [PlannedOutput(output_id="a", operation=SUM,
                                   measure=BALANCE)],
                    predicates=[predicate]))
                self.assertAlmostEqual(
                    float(envelope.outputs[0].value),
                    truth.total(_BOOK, BALANCE, [predicate]), places=2)

    def test_two_filters_narrow_together(self):
        envelope = run(plan_for(
            [PlannedOutput(output_id="a", operation=SUM, measure=BALANCE)],
            predicates=[JOINT, LTV_OVER_40]))
        self.assertAlmostEqual(
            float(envelope.outputs[0].value),
            truth.total(_BOOK, BALANCE, [JOINT, LTV_OVER_40]), places=2)


# --------------------------------------------------------------------------- #
# 3. One-dimensional grouping — every cell
# --------------------------------------------------------------------------- #
class TestOneDimensionalGrouping(unittest.TestCase):

    AXES = ("ltv_bucket", "age_bucket", "broker_channel",
            "collateral_geography", "erm_product_type")

    def test_balance_by_each_axis_matches_cell_for_cell(self):
        for axis in self.AXES:
            with self.subTest(axis=axis):
                envelope = run(plan_for(
                    [PlannedOutput(output_id="a", operation=SUM,
                                   measure=BALANCE)],
                    dimensions=[axis]))
                got = cells(envelope, "a", f"{BALANCE}_sum")
                want = truth.grouped(_BOOK, [axis], column=BALANCE)
                self.assertEqual(set(got), set(want))
                for key in want:
                    self.assertAlmostEqual(got[key], want[key], places=2)

    def test_count_by_broker_matches_cell_for_cell(self):
        envelope = run(plan_for(
            [PlannedOutput(output_id="a", operation=COUNT)],
            dimensions=["broker_channel"]))
        got = cells(envelope, "a", "loan_count")
        want = truth.grouped(_BOOK, ["broker_channel"], how="count")
        self.assertEqual(got, want)


# --------------------------------------------------------------------------- #
# 4. Two-dimensional grouping — the sprint's headline capability
# --------------------------------------------------------------------------- #
class TestTwoDimensionalGrouping(unittest.TestCase):

    GRIDS = (("ltv_bucket", "age_bucket"),
             ("collateral_geography", "erm_product_type"),
             ("broker_channel", "borrower_type"))

    def test_every_cell_of_every_grid_matches_truth(self):
        for grid in self.GRIDS:
            with self.subTest(grid=" x ".join(grid)):
                envelope = run(plan_for(
                    [PlannedOutput(output_id="a", operation=SUM,
                                   measure=BALANCE)],
                    dimensions=list(grid)))
                got = cells(envelope, "a", f"{BALANCE}_sum")
                want = truth.grouped(_BOOK, list(grid), column=BALANCE)
                self.assertEqual(set(got), set(want), "the grid differs")
                for key in want:
                    self.assertAlmostEqual(got[key], want[key], places=2,
                                           msg=f"cell {key}")

    def test_the_cells_reconcile_to_the_population_total(self):
        """§18 — the displayed mass equals the filtered population's mass."""
        envelope = run(plan_for(
            [PlannedOutput(output_id="a", operation=SUM, measure=BALANCE)],
            dimensions=["ltv_bucket", "age_bucket"]))
        got = cells(envelope, "a", f"{BALANCE}_sum")
        self.assertAlmostEqual(sum(got.values()),
                               truth.total(_BOOK, BALANCE), places=2)


# --------------------------------------------------------------------------- #
# 5. Two dimensions AND a filter — §1's worked example
# --------------------------------------------------------------------------- #
class TestTwoDimensionsAndAFilter(unittest.TestCase):

    def test_balance_by_ltv_by_age_for_joint_borrowers(self):
        envelope = run(plan_for(
            [PlannedOutput(output_id="a", operation=SUM, measure=BALANCE)],
            predicates=[JOINT], dimensions=["ltv_bucket", "age_bucket"]))
        got = cells(envelope, "a", f"{BALANCE}_sum")
        want = truth.grouped(_BOOK, ["ltv_bucket", "age_bucket"],
                             column=BALANCE, predicates=[JOINT])
        self.assertEqual(set(got), set(want))
        for key in want:
            self.assertAlmostEqual(got[key], want[key], places=2,
                                   msg=f"cell {key}")

    def test_the_filtered_grid_reconciles_to_the_filtered_total(self):
        envelope = run(plan_for(
            [PlannedOutput(output_id="a", operation=SUM, measure=BALANCE)],
            predicates=[JOINT], dimensions=["ltv_bucket", "age_bucket"]))
        got = cells(envelope, "a", f"{BALANCE}_sum")
        self.assertAlmostEqual(sum(got.values()),
                               truth.total(_BOOK, BALANCE, [JOINT]), places=2)

    def test_a_filtered_grid_is_strictly_smaller_than_the_unfiltered_one(self):
        """A filter that changed nothing would pass every cell test above by
        accident. This is the guard against a population that never narrowed."""
        joint = truth.total(_BOOK, BALANCE, [JOINT])
        whole = truth.total(_BOOK, BALANCE)
        self.assertLess(joint, whole)
        envelope = run(plan_for(
            [PlannedOutput(output_id="a", operation=SUM, measure=BALANCE)],
            predicates=[JOINT], dimensions=["ltv_bucket", "age_bucket"]))
        self.assertAlmostEqual(
            sum(cells(envelope, "a", f"{BALANCE}_sum").values()), joint,
            places=2)

    def test_region_by_product_for_one_broker(self):
        envelope = run(plan_for(
            [PlannedOutput(output_id="a", operation=SUM, measure=BALANCE)],
            predicates=[ALPHA],
            dimensions=["collateral_geography", "erm_product_type"]))
        got = cells(envelope, "a", f"{BALANCE}_sum")
        want = truth.grouped(_BOOK, ["collateral_geography", "erm_product_type"],
                             column=BALANCE, predicates=[ALPHA])
        self.assertEqual(set(got), set(want))
        for key in want:
            self.assertAlmostEqual(got[key], want[key], places=2)


# --------------------------------------------------------------------------- #
# 6. Multi-output, and multi-output over a grid
# --------------------------------------------------------------------------- #
class TestMultiOutput(unittest.TestCase):

    def test_count_balance_and_wa_ltv_over_one_population(self):
        envelope = run(plan_for([
            PlannedOutput(output_id="a", operation=COUNT),
            PlannedOutput(output_id="b", operation=SUM, measure=BALANCE),
            PlannedOutput(output_id="c", operation=WEIGHTED_AVERAGE,
                          measure=LTV, weight_field=BALANCE)],
            predicates=[JOINT]))
        values = {o.output_id: o.value for o in envelope.outputs}
        self.assertEqual(int(values["a"]), truth.row_count(_BOOK, [JOINT]))
        self.assertAlmostEqual(float(values["b"]),
                               truth.total(_BOOK, BALANCE, [JOINT]), places=2)
        self.assertAlmostEqual(
            float(values["c"]),
            truth.weighted_average(_BOOK, LTV, BALANCE, [JOINT]), places=4)

    def test_count_and_balance_by_age_bucket_cell_for_cell(self):
        envelope = run(plan_for([
            PlannedOutput(output_id="a", operation=COUNT),
            PlannedOutput(output_id="b", operation=SUM, measure=BALANCE)],
            predicates=[JOINT], dimensions=["age_bucket"]))
        counts = cells(envelope, "a", "loan_count")
        balances = cells(envelope, "b", f"{BALANCE}_sum")
        self.assertEqual(counts, truth.grouped(_BOOK, ["age_bucket"],
                                               how="count", predicates=[JOINT]))
        want = truth.grouped(_BOOK, ["age_bucket"], column=BALANCE,
                             predicates=[JOINT])
        for key in want:
            self.assertAlmostEqual(balances[key], want[key], places=2)

    def test_three_measures_over_a_two_dimensional_grid(self):
        """§9 — multi-output and multi-dimension composing, not two paths."""
        envelope = run(plan_for([
            PlannedOutput(output_id="a", operation=COUNT),
            PlannedOutput(output_id="b", operation=SUM, measure=BALANCE),
            PlannedOutput(output_id="c", operation=WEIGHTED_AVERAGE,
                          measure=RATE, weight_field=BALANCE)],
            predicates=[JOINT], dimensions=["age_bucket", "ltv_bucket"]))
        balances = cells(envelope, "b", f"{BALANCE}_sum")
        want = truth.grouped(_BOOK, ["age_bucket", "ltv_bucket"],
                             column=BALANCE, predicates=[JOINT])
        self.assertEqual(set(balances), set(want))
        for key in want:
            self.assertAlmostEqual(balances[key], want[key], places=2)


# --------------------------------------------------------------------------- #
# 7. Output-local narrowing, numerically
# --------------------------------------------------------------------------- #
class TestOutputLocalNarrowing(unittest.TestCase):

    def test_the_narrowed_output_and_its_siblings_each_match_their_own_truth(self):
        envelope = run(plan_for([
            PlannedOutput(output_id="a", operation=COUNT),
            PlannedOutput(output_id="b", operation=SUM, measure=BALANCE),
            PlannedOutput(output_id="c", operation=SUM, measure=BALANCE,
                          local_scope_delta=ScopeDelta(
                              filters=(Predicate(*LTV_OVER_40),)))],
            predicates=[JOINT]))
        values = {o.output_id: o.value for o in envelope.outputs}
        self.assertEqual(int(values["a"]), truth.row_count(_BOOK, [JOINT]))
        self.assertAlmostEqual(float(values["b"]),
                               truth.total(_BOOK, BALANCE, [JOINT]), places=2)
        self.assertAlmostEqual(
            float(values["c"]),
            truth.total(_BOOK, BALANCE, [JOINT, LTV_OVER_40]), places=2)

    def test_the_narrowed_output_is_genuinely_smaller(self):
        """If the delta had leaked onto the siblings, or been dropped from the
        narrowed output, these two would be equal."""
        self.assertLess(truth.total(_BOOK, BALANCE, [JOINT, LTV_OVER_40]),
                        truth.total(_BOOK, BALANCE, [JOINT]))

    def test_a_narrowed_output_over_a_grid(self):
        envelope = run(plan_for([
            PlannedOutput(output_id="a", operation=SUM, measure=BALANCE),
            PlannedOutput(output_id="b", operation=SUM, measure=BALANCE,
                          local_scope_delta=ScopeDelta(
                              filters=(Predicate(*LTV_OVER_40),)))],
            predicates=[JOINT], dimensions=["age_bucket"]))
        base = cells(envelope, "a", f"{BALANCE}_sum")
        narrowed = cells(envelope, "b", f"{BALANCE}_sum")
        self.assertEqual(base, truth.grouped(_BOOK, ["age_bucket"],
                                             column=BALANCE, predicates=[JOINT]))
        want = truth.grouped(_BOOK, ["age_bucket"], column=BALANCE,
                             predicates=[JOINT, LTV_OVER_40])
        for key in want:
            self.assertAlmostEqual(narrowed[key], want[key], places=2)
        self.assertLess(sum(narrowed.values()), sum(base.values()))


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
