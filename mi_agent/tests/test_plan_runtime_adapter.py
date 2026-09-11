#!/usr/bin/env python3
"""Slice 1: does the adapter dispatch the right plans, and refuse the rest?

The eligibility half matters more than the execution half. An adapter that
quietly handed a specialist plan, a stated historical period or an explicit
Direct lens to the generic executor would compute a simpler question than the
reader asked and return a plausible number — which is the exact failure class
this programme has spent its length removing.

Numbers are checked against `portfolio_truth_oracle`, which imports nothing from
the product. The adapter is never asked to agree with the legacy path: the legacy
path is a control, not the truth.
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from mi_agent import plan_runtime_adapter as adapter              # noqa: E402
from mi_agent.mi_query_validator import load_mi_semantics         # noqa: E402
from mi_agent.tests import portfolio_truth_oracle as truth        # noqa: E402

_SEMANTICS = load_mi_semantics(
    str(_REPO_ROOT / "mi_agent" / "mi_semantics_field_registry.yaml"))
_BOOK = truth.canonical_book()

BALANCE, LTV, AGE = truth.BALANCE, truth.LTV, truth.AGE
REGION = "collateral_geography"
PRODUCT = "erm_product_type"


def plan(*, capability="generic_analysis", operation="point_in_time",
         measures=None, dimensions=(), filters=(), output_filters=(),
         period_form="current", lens="all", outputs=None, geography=None,
         output_geography=None,
         comparison_kind="none", target=None, plan_id="plan_test"):
    """A GovernedQueryPlan-shaped dict. The adapter reads plans, not objects."""
    if measures is None:
        measures = [{"concept": "current_outstanding_balance",
                     "canonical_field": BALANCE, "statistic": "sum",
                     "weight_field": None, "capability_owner": None}]
    body_outputs = outputs if outputs is not None else [{
        "id": "primary", "measures": list(measures),
        "dimensions": [{"concept": d, "canonical_field": d} for d in dimensions],
        "filters": list(output_filters), "geography": output_geography}]
    return {
        "schema_version": "governed_query_plan/1.0",
        "capability": capability, "operation": operation,
        "population": {"base": "funded", "lens": lens, "seasoning": "any",
                       "scope_predicates": []},
        "outputs": body_outputs,
        "period": {"form": period_form, "labels": [], "grain": None,
                   "periods_back": None, "contract": "latest_governed_reporting_period",
                   "resolved": True, "owned_by_capability": False},
        "comparison_kind": comparison_kind, "comparison_left": None,
        "comparison_right": None,
        "filters": list(filters), "geography": geography, "target": target,
        "plan_id": plan_id,
    }


#: A geography binding of the shape the compiler actually emits — taken from the
#: recorded run-8 plans for Q14 and Q16, not invented.
GEOGRAPHY_BINDING = {
    "requested_basis": None, "requested_level": "reporting",
    "resolved_level": "reporting", "canonical_field": "canonical_region_reporting",
    "group_by": True, "values": [], "defaulted": True,
    "default_reason": "no basis was stated; level 'reporting' is governed by the "
                      "reporting taxonomy",
}


def flt(field, value, comparator="eq"):
    return {"concept": field, "canonical_field": field,
            "comparator": comparator, "value": value}


def run(p, frame=None):
    return adapter.execute_shadow_governed_plan(
        p, _BOOK if frame is None else frame, _SEMANTICS)


# --------------------------------------------------------------------------- #
# eligible, and numerically right against the independent oracle
# --------------------------------------------------------------------------- #
class TestEligibleExecution(unittest.TestCase):

    def test_ordinary_sum(self):
        out = run(plan())
        self.assertTrue(out.eligible, out.reason)
        self.assertEqual(out.error, "")
        self.assertAlmostEqual(out.value, truth.total(_BOOK, BALANCE), places=2)

    def test_count(self):
        out = run(plan(measures=[{"concept": "loan", "canonical_field": None,
                                  "statistic": "count", "weight_field": None}]))
        self.assertTrue(out.eligible, out.reason)
        self.assertAlmostEqual(out.value, float(truth.row_count(_BOOK)), places=2)

    def test_simple_average(self):
        out = run(plan(measures=[{"concept": "youngest_borrower_age",
                                  "canonical_field": AGE, "statistic": "average",
                                  "weight_field": None}]))
        self.assertTrue(out.eligible, out.reason)
        self.assertAlmostEqual(out.value, float(_BOOK[AGE].mean()), places=4)

    def test_weighted_average(self):
        out = run(plan(measures=[{"concept": "current_loan_to_value",
                                  "canonical_field": LTV,
                                  "statistic": "weighted_average",
                                  "weight_field": BALANCE}]))
        self.assertTrue(out.eligible, out.reason)
        self.assertAlmostEqual(out.value,
                               truth.weighted_average(_BOOK, LTV, BALANCE),
                               places=4)

    def test_one_filter(self):
        pred = ("borrower_type", "eq", "Joint")
        out = run(plan(filters=[flt("borrower_type", "Joint")]))
        self.assertTrue(out.eligible, out.reason)
        self.assertAlmostEqual(out.value, truth.total(_BOOK, BALANCE, [pred]),
                               places=2)

    def test_multiple_filters(self):
        preds = [("borrower_type", "eq", "Joint"), (REGION, "eq", "Scotland")]
        out = run(plan(filters=[flt("borrower_type", "Joint"),
                                flt(REGION, "Scotland")]))
        self.assertTrue(out.eligible, out.reason)
        self.assertAlmostEqual(out.value, truth.total(_BOOK, BALANCE, preds),
                               places=2)

    def test_directional_filter_keeps_its_operator(self):
        pred = (LTV, "gt", 40.0)
        out = run(plan(filters=[flt(LTV, 40.0, comparator="gt")]))
        self.assertTrue(out.eligible, out.reason)
        self.assertAlmostEqual(out.value, truth.total(_BOOK, BALANCE, [pred]),
                               places=2)

    def test_one_dimension_is_eligible_and_grouped(self):
        out = run(plan(operation="breakdown", dimensions=[PRODUCT]))
        self.assertTrue(out.eligible, out.reason)
        self.assertEqual(out.error, "")
        self.assertEqual(out.receipt["group_field_keys"], [PRODUCT])
        # A grouped execution has no scalar; inventing a total would be the
        # adapter answering a question the plan did not ask.
        self.assertIsNone(out.value)

    def test_two_dimensions_are_eligible(self):
        out = run(plan(operation="breakdown", dimensions=[PRODUCT, REGION]))
        self.assertTrue(out.eligible, out.reason)
        self.assertEqual(out.receipt["group_field_keys"], [PRODUCT, REGION])

    def test_two_dimension_cells_match_the_oracle_cell_for_cell(self):
        """The load-bearing check behind `test_two_dimensions_are_eligible`.

        Presentation falls back to a table for two axes because the second
        field has no `y` chart role — so the risk worth testing is that the
        second axis was dropped along with its chart role. A cross-tab whose
        total agrees can still have the mass in the wrong cells, so this compares
        every cell against the independent oracle.
        """
        dims = [PRODUCT, REGION]
        out = run(plan(operation="breakdown", dimensions=dims))
        self.assertTrue(out.eligible, out.reason)
        self.assertEqual(out.error, "")
        spec = adapter.spec_for_plan(plan(operation="breakdown", dimensions=dims))
        from mi_agent.mi_query_executor import execute_mi_query
        frame = execute_mi_query(spec, _BOOK, _SEMANTICS).data
        got = {(str(r[PRODUCT]), str(r[REGION])): float(r[f"{BALANCE}_sum"])
               for _, r in frame.iterrows()}
        want = dict(truth.grouped(_BOOK, dims, column=BALANCE, how="sum"))
        self.assertEqual(set(got), set(want))
        for key, value in want.items():
            self.assertAlmostEqual(got[key], value, places=3, msg=f"cell {key}")

    def test_the_configured_region_dimension(self):
        """The asset-config basis for equity_release is collateral, whose
        reporting field is `collateral_geography`. Grouping on it must bind."""
        from mi_agent import mi_geography as geo
        self.assertEqual(geo.default_primary_basis("equity_release"), "collateral")
        self.assertEqual(geo.field_for_basis("collateral", frame=_BOOK), REGION)
        out = run(plan(operation="breakdown", dimensions=[REGION]))
        self.assertTrue(out.eligible, out.reason)
        self.assertEqual(out.receipt["group_field_keys"], [REGION])

    def test_zero_row_filtered_population(self):
        pred = (LTV, "gt", 1000.0)
        self.assertEqual(truth.row_count(_BOOK, [pred]), 0)
        out = run(plan(filters=[flt(LTV, 1000.0, comparator="gt")]))
        self.assertTrue(out.eligible, out.reason)
        self.assertEqual(out.receipt["filtered_row_count"], 0)

    def test_null_measure_is_not_zero_filled(self):
        import numpy as np
        book = _BOOK.copy()
        book.loc[book.index[:40], LTV] = np.nan
        out = run(plan(measures=[{"concept": "current_loan_to_value",
                                  "canonical_field": LTV, "statistic": "average",
                                  "weight_field": None}]), frame=book)
        self.assertTrue(out.eligible, out.reason)
        self.assertAlmostEqual(out.value, float(book[LTV].mean()), places=4)
        self.assertNotAlmostEqual(out.value, float(book[LTV].fillna(0).mean()),
                                  places=4)

    def test_the_receipt_records_what_was_executed(self):
        out = run(plan(filters=[flt("borrower_type", "Joint")]))
        self.assertEqual(out.receipt["aggregation"], "sum")
        self.assertEqual(out.receipt["input_row_count"], truth.row_count(_BOOK))
        self.assertTrue(out.receipt["applied_predicates"])

    def test_the_requested_half_is_transcribed_not_reinterpreted(self):
        out = run(plan(filters=[flt("borrower_type", "Joint")],
                       operation="breakdown", dimensions=[PRODUCT]))
        self.assertEqual(out.requested["measure_field"], BALANCE)
        self.assertEqual(out.requested["statistic"], "sum")
        self.assertEqual(out.requested["dimensions"], [PRODUCT])
        self.assertEqual(out.requested["filters"],
                         [{"field": "borrower_type", "comparator": "eq",
                           "value": "Joint"}])
        self.assertEqual(out.requested["period_form"], "current")

    def test_the_requested_half_keeps_the_direction_of_every_predicate(self):
        """A direction survives into the ledger row.

        The resolved half records `op` per predicate; if the requested half
        collapses to `{field: value}` then `>` and `=` look identical on the
        requested side and a divergence cannot be adjudicated from the row
        afterwards.
        """
        out = run(plan(filters=[flt(LTV, 50, "gt"), flt(AGE, 70, "lt"),
                                flt("borrower_type", "Joint")]))
        self.assertEqual(out.requested["filters"], [
            {"field": LTV, "comparator": "gt", "value": 50},
            {"field": AGE, "comparator": "lt", "value": 70},
            {"field": "borrower_type", "comparator": "eq", "value": "Joint"},
        ])


# --------------------------------------------------------------------------- #
# ineligible — and never quietly executed anyway
# --------------------------------------------------------------------------- #
class TestIneligibility(unittest.TestCase):

    def _refused(self, p, expected_reason):
        out = run(p)
        self.assertFalse(out.eligible,
                         f"expected {expected_reason}, but the plan was executed")
        self.assertEqual(out.reason, expected_reason, out.detail)
        self.assertIsNone(out.value)
        self.assertIsNone(out.spec)
        return out

    def test_explicit_historical_period(self):
        self._refused(plan(period_form="explicit_period"),
                      adapter.PERIOD_NOT_CURRENT)

    def test_previous_reporting_period(self):
        self._refused(plan(period_form="previous_reporting_period"),
                      adapter.PERIOD_NOT_CURRENT)

    def test_relative_pair(self):
        self._refused(plan(period_form="relative_pair"),
                      adapter.PERIOD_NOT_CURRENT)

    def test_time_series(self):
        self._refused(plan(period_form="series"), adapter.PERIOD_NOT_CURRENT)

    def test_forward_looking(self):
        self._refused(plan(period_form="forward_looking"),
                      adapter.PERIOD_NOT_CURRENT)

    def test_a_geography_axis_the_adapter_cannot_bind(self):
        """The corpus-replay finding: a stated region axis is never dropped.

        This module binds no geography axis, so a plan that states one has to be
        refused. Executing it would group by the OTHER axis alone and present the
        answer as though it were the breakdown the reader authorised.
        """
        self._refused(plan(operation="breakdown", dimensions=["ltv_bucket"],
                           geography=GEOGRAPHY_BINDING),
                      adapter.GEOGRAPHY_REQUESTED)

    def test_a_geography_binding_on_the_output_is_equally_refused(self):
        self._refused(plan(operation="breakdown",
                           output_geography=GEOGRAPHY_BINDING),
                      adapter.GEOGRAPHY_REQUESTED)

    def test_a_geography_restriction_without_grouping_is_also_refused(self):
        """`values` restricts the population, and this slice binds no predicate."""
        binding = dict(GEOGRAPHY_BINDING, group_by=False, values=["Scotland"])
        self._refused(plan(geography=binding), adapter.GEOGRAPHY_REQUESTED)

    # THE TWO GOVERNED ROLES ARE NO LONGER REFUSED. They were, and these two
    # cases asserted it: the compiler bound the lens to a `source_portfolio_type`
    # predicate, but nothing carried that predicate to the executor, so refusing
    # was the only honest answer. `plan_predicates` carries it now, and refusing
    # a plan this adapter can express would be the wrong kind of caution. What is
    # still refused is a lens the governed vocabulary does not define, which
    # never reaches a plan at all — see TestScopeFailsClosed below.
    #
    # EXPLICIT_LENS itself is kept rather than deleted: it is the reason a lens
    # is refused when a deployment narrows ELIGIBLE_POPULATION_LENS, and a reason
    # code that no longer exists cannot be read in an old evidence record.
    def _scoped(self, role):
        scoped = plan(lens=role)
        scoped["population"]["scope_predicates"] = [
            {"concept": "portfolio_lens", "comparator": "eq",
             "canonical_field": "source_portfolio_type", "value": role}]
        return scoped

    def test_direct_lens_is_eligible(self):
        ok, why, _ = adapter.check_eligibility(self._scoped("direct"))
        self.assertTrue(ok, f"the direct role was refused: {why}")

    def test_acquired_lens_is_eligible(self):
        ok, why, _ = adapter.check_eligibility(self._scoped("acquired"))
        self.assertTrue(ok, f"the acquired role was refused: {why}")

    def test_a_role_with_no_scope_predicate_is_refused(self):
        """The worst available shape: a plan that says "acquired" and would be
        computed over the whole book. The compiler cannot emit it; the adapter
        refuses it anyway rather than trusting that."""
        self._refused(plan(lens="acquired"), adapter.SCOPE_NOT_BOUND)

    def test_a_lens_outside_the_eligible_set_is_still_EXPLICIT_LENS(self):
        """The guard still guards; only its membership changed."""
        previous = adapter.ELIGIBLE_POPULATION_LENS
        adapter.ELIGIBLE_POPULATION_LENS = frozenset({"", "all", "total", "none"})
        try:
            self._refused(self._scoped("acquired"), adapter.EXPLICIT_LENS)
        finally:
            adapter.ELIGIBLE_POPULATION_LENS = previous

    def test_specialist_bridge(self):
        self._refused(plan(capability="funded_bridge", operation="bridge"),
                      adapter.CAPABILITY_NOT_GENERIC)

    def test_specialist_period_movement(self):
        self._refused(plan(capability="period_movement", operation="movement"),
                      adapter.CAPABILITY_NOT_GENERIC)

    def test_borrowing_base(self):
        self._refused(plan(capability="borrowing_base", operation="headroom"),
                      adapter.CAPABILITY_NOT_GENERIC)

    def test_concentration(self):
        self._refused(plan(capability="concentration", operation="rank"),
                      adapter.CAPABILITY_NOT_GENERIC)

    def test_limit_assessment(self):
        self._refused(plan(capability="limit_assessment", operation="headroom"),
                      adapter.CAPABILITY_NOT_GENERIC)

    def test_distribution_is_not_a_generic_groupby(self):
        self._refused(plan(operation="distribution", dimensions=[PRODUCT]),
                      adapter.OPERATION_NOT_GENERIC)

    def test_rank_is_not_generic(self):
        self._refused(plan(operation="rank", dimensions=[PRODUCT]),
                      adapter.OPERATION_NOT_GENERIC)

    def test_summary_is_not_generic(self):
        self._refused(plan(operation="summary"), adapter.OPERATION_NOT_GENERIC)

    def test_three_dimensions(self):
        self._refused(plan(operation="breakdown",
                           dimensions=[PRODUCT, REGION, "borrower_type"]),
                      adapter.TOO_MANY_DIMENSIONS)

    def test_multi_output(self):
        one = plan()["outputs"][0]
        self._refused(plan(outputs=[one, dict(one, id="second")]),
                      adapter.NOT_SINGLE_OUTPUT)

    def test_several_measures_in_one_output(self):
        self._refused(plan(measures=[
            {"concept": "current_outstanding_balance", "canonical_field": BALANCE,
             "statistic": "sum", "weight_field": None},
            {"concept": "loan", "canonical_field": None, "statistic": "count",
             "weight_field": None}]), adapter.NOT_SINGLE_OUTPUT)

    def test_no_measure(self):
        self._refused(plan(measures=[]), adapter.NO_MEASURE)

    def test_capability_owned_statistic(self):
        self._refused(plan(measures=[{"concept": "funded_balance_movement",
                                      "canonical_field": None,
                                      "statistic": "capability",
                                      "weight_field": None}]),
                      adapter.MEASURE_NOT_GENERIC)

    def test_unbound_measure_field(self):
        self._refused(plan(measures=[{"concept": "mystery",
                                      "canonical_field": None,
                                      "statistic": "sum", "weight_field": None}]),
                      adapter.MEASURE_UNBOUND)

    def test_weighted_average_with_no_weight(self):
        self._refused(plan(measures=[{"concept": "current_loan_to_value",
                                      "canonical_field": LTV,
                                      "statistic": "weighted_average",
                                      "weight_field": None}]),
                      adapter.MEASURE_UNBOUND)

    def test_unbound_dimension(self):
        p = plan(operation="breakdown")
        p["outputs"][0]["dimensions"] = [{"concept": "vibes",
                                          "canonical_field": None}]
        self._refused(p, adapter.DIMENSION_UNBOUND)

    def test_two_predicates_on_one_field_are_refused_not_collapsed(self):
        """`MIQuerySpec.filters` is keyed by field, so the second would win alone.

        An LTV band stated as two bounds would arrive at the executor as `< 80`
        with the `> 50` gone, and the answer would be a population the reader
        never asked for. The band belongs in a single `between`.
        """
        out = self._refused(plan(filters=[flt(LTV, 50, "gt"), flt(LTV, 80, "lt")]),
                            adapter.FILTER_NOT_EXPRESSIBLE)
        self.assertIn(LTV, out.detail)

    def test_a_band_stated_once_is_carried_whole(self):
        """The expressible form of the same restriction still works, as a list.

        A multi-valued bound reaches the executor as the LIST its filters have
        always held — the estate's own `_executor_value` shape — not as a tuple.
        """
        p = plan(filters=[flt(LTV, (40, 60), "between")])
        spec = adapter.spec_for_plan(p)
        self.assertEqual(spec.filters, {LTV: {"op": "between", "value": [40, 60]}})

    def test_unbound_filter(self):
        p = plan()
        p["filters"] = [{"concept": "vibes", "canonical_field": None,
                         "comparator": "eq", "value": "x"}]
        self._refused(p, adapter.FILTER_UNBOUND)

    def test_comparison(self):
        self._refused(plan(comparison_kind="population_pair"),
                      adapter.COMPARISON_REQUESTED)

    def test_target(self):
        self._refused(plan(target={"concept": "balance", "value": 1e8,
                                   "comparator": "gte"}),
                      adapter.TARGET_REQUESTED)

    def test_clarify_or_refuse_has_no_plan(self):
        """CLARIFY and REFUSE produce no plan at all, so there is nothing to
        dispatch — the adapter must say so rather than improvise."""
        for empty in (None, {}, ""):
            out = run(empty)
            self.assertFalse(out.eligible)
            self.assertEqual(out.reason, adapter.NOT_A_PLAN)


# --------------------------------------------------------------------------- #
# the module's own discipline
# --------------------------------------------------------------------------- #
class TestAdapterDiscipline(unittest.TestCase):

    def setUp(self):
        self.source = (_REPO_ROOT / "mi_agent" / "plan_runtime_adapter.py").read_text()

    def test_it_cannot_read_the_question(self):
        import ast
        tree = ast.parse(self.source)
        names = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
        names |= {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)}
        names |= {a.arg for n in ast.walk(tree)
                  if isinstance(n, ast.arguments) for a in n.args}
        for forbidden in ("question", "text", "sentence", "parsed"):
            self.assertNotIn(forbidden, names,
                             f"the adapter reaches {forbidden!r}")

    def test_it_cannot_pattern_match(self):
        import ast
        tree = ast.parse(self.source)
        imported = {n.module for n in ast.walk(tree)
                    if isinstance(n, ast.ImportFrom) and n.module}
        imported |= {a.name for n in ast.walk(tree) if isinstance(n, ast.Import)
                     for a in n.names}
        self.assertNotIn("re", imported)

    def test_it_imports_no_interpreter_and_no_parser(self):
        for forbidden in ("llm_query_parser", "parsed_question",
                          "question_interpretation", "opus_interpreter",
                          "chat_routing"):
            self.assertNotIn(forbidden, self.source,
                             f"the adapter reaches {forbidden}")

    def test_the_flag_defaults_to_off(self):
        import os
        previous = os.environ.pop(adapter.SHADOW_ENV_VAR, None)
        try:
            self.assertEqual(adapter.shadow_mode(), adapter.SHADOW_OFF)
            os.environ[adapter.SHADOW_ENV_VAR] = "shadow"
            self.assertEqual(adapter.shadow_mode(), adapter.SHADOW_ON)
            os.environ[adapter.SHADOW_ENV_VAR] = "serve"
            self.assertEqual(adapter.shadow_mode(), adapter.SHADOW_OFF,
                             "there is no SERVE state in slice 1")
        finally:
            os.environ.pop(adapter.SHADOW_ENV_VAR, None)
            if previous is not None:
                os.environ[adapter.SHADOW_ENV_VAR] = previous

    def test_a_failing_execution_is_captured_not_raised(self):
        out = adapter.execute_shadow_governed_plan(plan(), object(), _SEMANTICS)
        self.assertTrue(out.eligible)
        self.assertTrue(out.error, "an executor failure must be recorded")
        self.assertIsNone(out.value)

    def test_no_plan_facet_is_dropped_to_obtain_a_result(self):
        """Every bound facet the plan states must reach the spec."""
        p = plan(operation="breakdown", dimensions=[PRODUCT, REGION],
                 filters=[flt("borrower_type", "Joint")],
                 output_filters=[flt(PRODUCT, "Lump Sum")])
        spec = adapter.spec_for_plan(p)
        self.assertEqual(spec.metric, BALANCE)
        self.assertEqual(spec.aggregation, "sum")
        self.assertEqual(list(spec.dimensions), [PRODUCT, REGION])
        self.assertEqual(set(spec.filters), {"borrower_type", PRODUCT})

    def test_every_facet_of_the_plan_contract_is_classified(self):
        """The guard that the geography miss got past, written structurally.

        The earlier facet test checked the facets the local `plan()` helper could
        express, and that helper hard-coded `geography: None` — so a whole facet
        of the real contract was never put to the adapter at all. Reading the
        facet list off `GovernedQueryPlan` itself closes that: each field has to
        be declared CARRIED into the spec, REFUSED as an ineligibility, or
        IDENTITY/provenance, and a facet added to the contract later fails this
        test until somebody decides which it is.
        """
        import dataclasses
        from mi_agent.interpretation_v2.plan import GovernedQueryPlan, OutputPlan

        # `population` MOVED from refused to carried in slice 3. Its lens is a
        # governed Direct/Acquired role, the compiler binds it to a
        # `source_portfolio_type` predicate, and `plan_predicates` now carries
        # that predicate into the spec — so refusing it would be the adapter
        # declining a plan it can express. The carried assertion below is what
        # keeps it honest: a facet listed here has to actually arrive.
        carried = {"capability", "operation", "outputs", "filters",
                   "measures", "dimensions", "population"}
        refused = {"period", "comparison_kind", "comparison_left",
                   "comparison_right", "geography", "target"}
        identity = {"schema_version", "provenance", "id"}

        facets = {f.name for f in dataclasses.fields(GovernedQueryPlan)}
        facets |= {f.name for f in dataclasses.fields(OutputPlan)}
        unclassified = facets - carried - refused - identity
        self.assertFalse(
            unclassified,
            f"the plan contract grew facets this slice has not classified: "
            f"{sorted(unclassified)}. Each must be carried into the spec or "
            f"refused by check_eligibility — never silently ignored.")

        # A CARRIED FACET MUST ACTUALLY ARRIVE. Listing `population` above is a
        # claim, and this is the evidence for it: the governed role reaches the
        # executor's filters rather than being quietly dropped, which is the
        # exact failure mode "carried" is supposed to rule out.
        for role in ("direct", "acquired"):
            scoped = plan(lens=role)
            # The predicate the compiler binds for this lens. The local helper
            # states the lens without one, which is a shape no compiler emits —
            # and which `check_structure` now refuses outright.
            scoped["population"]["scope_predicates"] = [
                {"concept": "portfolio_lens", "comparator": "eq",
                 "canonical_field": "source_portfolio_type", "value": role}]
            spec = adapter.spec_for_plan(scoped)
            self.assertEqual(spec.filters.get("source_portfolio_type"), role,
                             f"the {role} lens never reached the spec")
        self.assertNotIn("source_portfolio_type",
                         adapter.spec_for_plan(plan(lens="all")).filters,
                         "the default population became a predicate")

        # And the refusable ones must actually refuse, not merely be listed here.
        for p, reason in (
            (plan(period_form="explicit_period"), adapter.PERIOD_NOT_CURRENT),
            (plan(comparison_kind="period_over_period"),
             adapter.COMPARISON_REQUESTED),
            (plan(target={"concept": "funding_target", "value": 1.0}),
             adapter.TARGET_REQUESTED),
            (plan(geography=GEOGRAPHY_BINDING), adapter.GEOGRAPHY_REQUESTED),
            (plan(output_geography=GEOGRAPHY_BINDING),
             adapter.GEOGRAPHY_REQUESTED),
        ):
            self.assertEqual(adapter.check_eligibility(p)[1], reason)



# --------------------------------------------------------------------------- #
# SLICE 3 — the governed Direct/Acquired role scope
# --------------------------------------------------------------------------- #
#
# The representation was never missing. `CandidateIntent.population.lens` has
# always admitted {direct, acquired, all}, and `compiler._bind_population` has
# always bound a non-default lens to a governed predicate on
# `source_portfolio_type` — held in `population.scope_predicates` rather than in
# `filters`, because the population axis resolves before the output does.
#
# What was missing was two lines of plumbing. `ELIGIBLE_POPULATION_LENS` refused
# every explicit lens as EXPLICIT_LENS, and `_filters_for` read only the two
# `filters` slots — so an admitted lens would have been SILENTLY DROPPED between
# the plan and the executor and the answer would have read as a correct total.
#
# `plan_predicates` is the one accessor all three readers now share: the
# duplicate-field check, the executor bind, and the requested half of the
# coverage ledger. Three readers of the same thing is how a requested side and
# an executed side drift apart.

import pandas as _pd

ROLE_FIELD = "source_portfolio_type"


def provenanced_book(n: int = 400):
    """The oracle's book plus the governed provenance contract.

    Roles are assigned by ROW INDEX, never from a name, so the oracle can count
    them without borrowing any product judgement about what "acquired" means.
    """
    from mi_agent.tests import portfolio_truth_oracle as _truth
    book = _truth.canonical_book(n=n).reset_index(drop=True)
    book[ROLE_FIELD] = ["direct" if i % 3 else "acquired" for i in range(len(book))]
    return book


BOOK = provenanced_book()


def scoped_intent(lens="all", **overrides):
    payload = {
        "schema_version": "candidate_intent/1.0",
        "capability": "generic_analysis", "operation": "point_in_time",
        "population": {"base": "funded", "lens": lens, "seasoning": "any"},
        "measures": [{"concept": "current_outstanding_balance",
                      "statistic": "sum"}],
        "dimensions": [], "filters": [], "geography": {"requested": False},
        "comparison": {"kind": "none"}, "time": {"form": "current"},
    }
    payload.update(overrides)
    return payload


def compiled(payload):
    from mi_agent.interpretation_v2.compiler import DeterministicCompiler
    from mi_agent.interpretation_v2.intent import parse_candidate_intent
    return DeterministicCompiler().compile(parse_candidate_intent(payload))


def executed(payload, frame=None):
    """Plan -> eligibility -> spec -> the one calculation owner."""
    from mi_agent.mi_query_executor import execute_mi_query
    result = compiled(payload)
    assert result.is_plan, f"did not compile: {result.outcome} {result.codes()}"
    plan = result.plan.to_dict()
    ok, why, detail = adapter.check_eligibility(plan)
    assert ok, f"ineligible: {why} {detail}"
    spec = adapter.spec_for_plan(plan)
    return plan, spec, execute_mi_query(spec, BOOK if frame is None else frame,
                                        _SEMANTICS)


def receipt_fields(result):
    return sorted(e.get("canonical_field")
                  for e in (result.metadata.get("applied_predicates") or ()))


def oracle_sum(role=None, frame=None):
    from mi_agent.tests import portfolio_truth_oracle as _truth
    book = BOOK if frame is None else frame
    rows = book if role is None else book[book[ROLE_FIELD] == role]
    return float(rows[_truth.BALANCE].sum())


class TestTheDefaultIsTotalFunded(unittest.TestCase):
    """The contract this slice must not disturb."""

    def test_no_lens_produces_no_scope_predicate(self):
        plan, spec, _ = executed(scoped_intent())
        assert not (plan["population"].get("scope_predicates") or ())
        assert ROLE_FIELD not in (spec.filters or {})

    def test_no_lens_is_the_whole_book(self):
        from mi_agent.tests import portfolio_truth_oracle as _truth
        _, _, result = executed(scoped_intent())
        served = float(result.data[f"{_truth.BALANCE}_sum"].iloc[0])
        assert abs(served - oracle_sum()) < 0.01

    def test_lens_all_is_identical_to_no_lens(self):
        from mi_agent.tests import portfolio_truth_oracle as _truth
        _, _, default = executed(scoped_intent())
        _, _, explicit = executed(scoped_intent(lens="all"))
        column = f"{_truth.BALANCE}_sum"
        assert float(default.data[column].iloc[0]) == float(
            explicit.data[column].iloc[0])


class TestAGovernedRoleScopesThePopulation(unittest.TestCase):

    def test_the_role_selects_exactly_its_own_rows(self):
        from mi_agent.tests import portfolio_truth_oracle as _truth
        for role in ("direct", "acquired"):
            with self.subTest(role=role):
                _, _, result = executed(scoped_intent(lens=role))
                served = float(result.data[f"{_truth.BALANCE}_sum"].iloc[0])
                assert abs(served - oracle_sum(role)) < 0.01

    def test_the_scope_reaches_the_executor_and_the_receipt(self):
        for role in ("direct", "acquired"):
            with self.subTest(role=role):
                _, spec, result = executed(scoped_intent(lens=role))
                assert spec.filters.get(ROLE_FIELD) == role
                assert ROLE_FIELD in receipt_fields(result)

    def test_the_role_binds_to_the_canonical_role_column_not_a_portfolio_id(self):
        """A role is a semantic, never a client's physical portfolio id."""
        result = compiled(scoped_intent(lens="acquired"))
        predicates = result.plan.population.scope_predicates
        assert [p.canonical_field for p in predicates] == [ROLE_FIELD]
        assert [p.value for p in predicates] == ["acquired"]

    def test_scope_composes_with_ordinary_predicates(self):
        _, spec, result = executed(scoped_intent(
            lens="acquired",
            measures=[{"concept": "loan", "statistic": "count"}],
            filters=[{"concept": "erm_product_type", "comparator": "eq",
                      "value": "drawdown"},
                     {"concept": "current_loan_to_value", "comparator": "gt",
                      "value": 50}]))
        expected = int(((BOOK[ROLE_FIELD] == "acquired")
                        & (BOOK.erm_product_type.str.lower() == "drawdown")
                        & (BOOK.current_loan_to_value > 50)).sum())
        assert int(result.data["loan_count"].iloc[0]) == expected
        assert receipt_fields(result) == sorted(
            ["current_loan_to_value", "erm_product_type", ROLE_FIELD])

    def test_scope_composes_with_a_governed_dimension(self):
        _, _, result = executed(scoped_intent(
            lens="acquired", operation="breakdown",
            measures=[{"concept": "loan", "statistic": "count"}],
            dimensions=["ltv_bucket"]))
        served = result.data.set_index("ltv_bucket")["loan_count"].to_dict()
        acquired = BOOK[BOOK[ROLE_FIELD] == "acquired"]
        assert served == acquired.groupby("ltv_bucket").size().to_dict()


class TestScopeIsNeverSilentlyDropped(unittest.TestCase):
    """The failure this wiring exists to prevent: an answer that reads as a
    correct total because the scope never reached the executor."""

    def test_every_reader_sees_the_scope_predicate(self):
        plan = compiled(scoped_intent(lens="acquired")).plan.to_dict()
        output = (plan.get("outputs") or ({},))[0]
        fields = [p.get("canonical_field")
                  for p in adapter.plan_predicates(plan, output)]
        assert ROLE_FIELD in fields
        assert ROLE_FIELD in adapter.spec_for_plan(plan).filters
        assert ROLE_FIELD in [f["field"] for f in
                              adapter.requested_semantics(plan)["filters"]]

    def test_the_requested_ledger_states_the_role_that_was_asked(self):
        plan = compiled(scoped_intent(lens="acquired")).plan.to_dict()
        stated = [f for f in adapter.requested_semantics(plan)["filters"]
                  if f["field"] == ROLE_FIELD]
        assert stated == [{"field": ROLE_FIELD, "comparator": "eq",
                           "value": "acquired"}]

    def test_a_receipt_without_the_scope_fails_reconciliation(self):
        import copy
        _, spec, result = executed(scoped_intent(lens="acquired"))
        stripped = copy.deepcopy(result)
        stripped.metadata["applied_predicates"] = [
            e for e in result.metadata["applied_predicates"]
            if e.get("canonical_field") != ROLE_FIELD]
        ok, why = adapter.reconcile_receipt(spec, stripped)
        assert not ok and ROLE_FIELD in why


class TestScopeFailsClosed(unittest.TestCase):

    def test_an_unknown_role_never_becomes_a_plan(self):
        from mi_agent.interpretation_v2.intent import (IntentParseError,
                                                       parse_candidate_intent)
        with self.assertRaises(IntentParseError):
            parse_candidate_intent(scoped_intent(lens="wholesale"))

    def test_a_book_without_the_role_column_refuses_rather_than_widening(self):
        """The control that matters most: an explicit Acquired against a book
        that cannot prove the role must REFUSE, never answer over Total."""
        from mi_agent.interpretation_v2.compiler import (CompilerContext,
                                                         DeterministicCompiler)
        from mi_agent.interpretation_v2.intent import parse_candidate_intent
        from mi_agent.tests import portfolio_truth_oracle as _truth
        bare = DeterministicCompiler(CompilerContext(
            available_fields=frozenset({_truth.BALANCE, "loan_identifier"})))
        result = bare.compile(parse_candidate_intent(scoped_intent(lens="acquired")))
        assert not result.is_plan
        assert "CONCEPT_UNAVAILABLE" in result.codes()

    def test_the_adapter_admits_exactly_the_governed_vocabulary(self):
        """It does not widen what a lens may say; it stops refusing it."""
        from mi_agent.interpretation_v2.vocabulary import POPULATION_LENSES
        assert adapter.GOVERNED_LENS_ROLES == POPULATION_LENSES - {"all"}
        assert adapter.GOVERNED_LENS_ROLES <= adapter.ELIGIBLE_POPULATION_LENS


if __name__ == "__main__":
    unittest.main()
