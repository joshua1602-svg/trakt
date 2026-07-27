#!/usr/bin/env python3
"""tests/test_governed_portfolio_architecture.py

The governed multi-portfolio MI architecture, end to end:

    canonical provenance
        → governed portfolio registry + dynamic scope resolution
        → governed capability resolution
        → shared query / calculation execution
        → governed result envelope (scope + coverage)
        → React · Copilot · exports

The fixture deliberately carries FOUR source portfolios — ``direct_001``,
``direct_002``, ``acquired_001``, ``acquired_002`` — because the defect this
architecture replaces was an implicit assumption that a client has exactly one
direct book and one acquired book. Every assertion below is written so that it
would fail if any layer hard-coded ``_001``, matched on an id prefix where
governed metadata exists, or treated ``Total`` as ``direct_001 + acquired_001``.

Run: python -m pytest tests/test_governed_portfolio_architecture.py
"""

from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from mi_agent import portfolio_scope as pscope
from mi_agent.mi_agent_workflow import run_mi_agent_query
from trakt_core import portfolio as P

_SEM = str(_REPO / "mi_agent" / "mi_semantics_field_registry.yaml")


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #
#: (portfolio_id, type, loan_count). Two direct + two acquired books: the shape
#: that breaks any "one of each" assumption.
_FOUR = [("direct_001", "direct", 3), ("direct_002", "direct", 2),
         ("acquired_001", "acquired", 4), ("acquired_002", "acquired", 2)]


def _frame(portfolios=_FOUR, *, ltv_missing_in=(), reporting_dates=None):
    """A canonical-shaped frame for the given portfolios.

    ``ltv_missing_in`` nulls ``current_loan_to_value`` for those portfolios, so
    partial field coverage is exercised with real data rather than a stub.
    """
    rows = []
    for pid, ptype, count in portfolios:
        dates = (reporting_dates or {}).get(pid, ["2025-12-31"])
        for date in dates:
            for i in range(count):
                rows.append({
                    "loan_identifier": f"{pid}-{date}-{i}",
                    "current_outstanding_balance": 100.0 * (i + 1),
                    "current_loan_to_value": (None if pid in ltv_missing_in
                                              else 0.40 + 0.01 * i),
                    "geographic_region_collateral": "South West",
                    "source_portfolio_id": pid,
                    "source_portfolio_type": ptype,
                    "source_portfolio_label": pid.replace("_", " ").title(),
                    "portfolio_cohort": pid,
                    "reporting_date": date,
                })
    return pd.DataFrame(rows)


def _registry(df=None, metadata=None):
    return pscope.registry_for_frame(_frame() if df is None else df,
                                     client_id="ERE", metadata=metadata or {})


# --------------------------------------------------------------------------- #
# 1. Dynamic hierarchy
# --------------------------------------------------------------------------- #
class TestDynamicHierarchy(unittest.TestCase):
    """The hierarchy is derived from governed metadata, never from fixed ids."""

    def test_total_resolves_every_governed_source_portfolio(self):
        scope = P.resolve_scope(_registry(), "total")
        self.assertEqual(scope.context_kind, P.CONTEXT_KIND_TOTAL)
        self.assertEqual(sorted(scope.portfolio_ids),
                         ["acquired_001", "acquired_002", "direct_001", "direct_002"])

    def test_direct_resolves_all_direct_portfolios(self):
        scope = P.resolve_scope(_registry(), "direct")
        self.assertEqual(scope.context_kind, P.CONTEXT_KIND_TYPE)
        self.assertEqual(sorted(scope.portfolio_ids), ["direct_001", "direct_002"])

    def test_acquired_resolves_all_acquired_portfolios(self):
        scope = P.resolve_scope(_registry(), "acquired")
        self.assertEqual(sorted(scope.portfolio_ids), ["acquired_001", "acquired_002"])

    def test_individual_context_resolves_one_portfolio(self):
        for pid in ("direct_002", "acquired_002"):
            scope = P.resolve_scope(_registry(), pid)
            self.assertEqual(scope.context_kind, P.CONTEXT_KIND_PORTFOLIO)
            self.assertEqual(scope.portfolio_ids, (pid,))

    def test_adding_a_direct_portfolio_needs_no_code_change(self):
        """direct_003 appears in Direct and Total purely by being in the data."""
        before = P.resolve_scope(_registry(), "direct").portfolio_ids
        widened = _registry(_frame(_FOUR + [("direct_003", "direct", 1)]))
        after = P.resolve_scope(widened, "direct").portfolio_ids
        self.assertNotIn("direct_003", before)
        self.assertEqual(sorted(after), ["direct_001", "direct_002", "direct_003"])
        self.assertIn("direct_003", P.resolve_scope(widened, "total").portfolio_ids)

    def test_adding_an_acquired_portfolio_needs_no_code_change(self):
        widened = _registry(_frame(_FOUR + [("acquired_003", "acquired", 1)]))
        self.assertEqual(sorted(P.resolve_scope(widened, "acquired").portfolio_ids),
                         ["acquired_001", "acquired_002", "acquired_003"])

    def test_removing_a_portfolio_updates_group_scope_automatically(self):
        narrowed = _registry(_frame([p for p in _FOUR if p[0] != "direct_002"]))
        self.assertEqual(P.resolve_scope(narrowed, "direct").portfolio_ids,
                         ("direct_001",))
        self.assertNotIn("direct_002", P.resolve_scope(narrowed, "total").portfolio_ids)

    def test_group_membership_follows_metadata_not_the_id(self):
        """A book NAMED acquired_002 but governed as direct joins Direct.

        This is the assertion that fails the moment anything infers a group from
        an id prefix rather than from ``source_portfolio_type``.
        """
        df = _frame()
        df.loc[df["source_portfolio_id"] == "acquired_002", "source_portfolio_type"] = "direct"
        registry = pscope.registry_for_frame(df, client_id="ERE", metadata={})
        self.assertIn("acquired_002", P.resolve_scope(registry, "direct").portfolio_ids)
        self.assertNotIn("acquired_002", P.resolve_scope(registry, "acquired").portfolio_ids)

    def test_hierarchy_is_total_then_types_then_portfolios(self):
        contexts = _registry().contexts()
        self.assertEqual(contexts[0]["context_id"], "total")
        self.assertEqual(contexts[0]["depth"], 0)
        depths = {c["context_id"]: c["depth"] for c in contexts}
        self.assertEqual(depths["direct"], 1)
        self.assertEqual(depths["direct_002"], 2)
        parents = {c["context_id"]: c["parent_id"] for c in contexts}
        self.assertEqual(parents["direct"], "total")
        self.assertEqual(parents["acquired_002"], "acquired")

    def test_unknown_context_widens_to_total_and_discloses_it(self):
        """A stale selection never silently narrows to one book."""
        scope = P.resolve_scope(_registry(), "direct_999")
        self.assertTrue(scope.is_total)
        self.assertTrue(scope.fell_back_to_total)
        self.assertEqual(scope.requested_context_id, "direct_999")


# --------------------------------------------------------------------------- #
# 2. Consolidation
# --------------------------------------------------------------------------- #
class TestConsolidation(unittest.TestCase):
    """Total is Σ Direct + Σ Acquired, computed from rows — never assumed."""

    def _balance(self, context_id, df=None):
        frame = _frame() if df is None else df
        registry = pscope.registry_for_frame(frame, client_id="ERE", metadata={})
        scope = P.resolve_scope(registry, context_id)
        scoped = pscope.apply_scope(frame, scope)
        return float(scoped["current_outstanding_balance"].sum())

    def test_total_equals_all_direct_plus_all_acquired(self):
        self.assertAlmostEqual(self._balance("total"),
                               self._balance("direct") + self._balance("acquired"))

    def test_total_is_not_just_the_001_portfolios(self):
        """The specific defect: Total must exceed direct_001 + acquired_001."""
        pair = self._balance("direct_001") + self._balance("acquired_001")
        self.assertGreater(self._balance("total"), pair)

    def test_direct_equals_the_sum_of_every_direct_portfolio(self):
        self.assertAlmostEqual(
            self._balance("direct"),
            self._balance("direct_001") + self._balance("direct_002"))

    def test_acquired_equals_the_sum_of_every_acquired_portfolio(self):
        self.assertAlmostEqual(
            self._balance("acquired"),
            self._balance("acquired_001") + self._balance("acquired_002"))

    def test_additive_metrics_aggregate_across_the_group(self):
        registry = _registry()
        scope = P.resolve_scope(registry, "total")
        scoped = pscope.apply_scope(_frame(), scope)
        self.assertEqual(len(scoped), sum(c for _, _, c in _FOUR))

    def test_weighted_metrics_recalculate_from_rows_not_from_portfolio_averages(self):
        """A weighted average is recomputed over the pooled rows.

        Averaging the two per-portfolio weighted averages gives a different
        number whenever the books differ in size, which is exactly the error
        this asserts against.
        """
        df = _frame()
        registry = pscope.registry_for_frame(df, client_id="ERE", metadata={})

        def wavg(context_id):
            scoped = pscope.apply_scope(df, P.resolve_scope(registry, context_id))
            w = scoped["current_outstanding_balance"]
            v = scoped["current_loan_to_value"]
            return float((v * w).sum() / w.sum())

        pooled = wavg("direct")
        naive = (wavg("direct_001") + wavg("direct_002")) / 2
        self.assertNotAlmostEqual(pooled, naive, places=6)
        # The pooled figure must sit between the two portfolio figures.
        lo, hi = sorted((wavg("direct_001"), wavg("direct_002")))
        self.assertGreaterEqual(pooled, lo)
        self.assertLessEqual(pooled, hi)

    def test_missing_values_are_not_treated_as_zero(self):
        """A portfolio with no LTV is EXCLUDED, not counted as LTV zero."""
        df = _frame(ltv_missing_in=("direct_002",))
        registry = pscope.registry_for_frame(df, client_id="ERE", metadata={})
        scope = P.resolve_scope(registry, "direct")
        coverage = pscope.coverage_for_frame(df, scope, fields=["current_loan_to_value"])
        self.assertEqual(coverage.portfolios_used, ("direct_001",))
        self.assertIn("direct_002", coverage.portfolios_excluded)

        scoped = pscope.apply_scope(df, scope)
        pooled = float((scoped["current_loan_to_value"]
                        * scoped["current_outstanding_balance"]).sum()
                       / scoped.loc[scoped["current_loan_to_value"].notna(),
                                    "current_outstanding_balance"].sum())
        self.assertGreater(pooled, 0.0)

    def test_partial_coverage_is_never_presented_as_a_total(self):
        df = _frame(ltv_missing_in=("direct_002",))
        registry = pscope.registry_for_frame(df, client_id="ERE", metadata={})
        coverage = pscope.coverage_for_frame(
            df, P.resolve_scope(registry, "total"), fields=["current_loan_to_value"])
        self.assertFalse(coverage.is_fully_consolidated)
        self.assertIn("NOT a fully consolidated", coverage.disclosure())


# --------------------------------------------------------------------------- #
# 3. Capability matrix
# --------------------------------------------------------------------------- #
class TestCapabilityMatrix(unittest.TestCase):
    """Business applicability is resolved once, in the backend, from metadata."""

    def _caps(self, context_id, *, registry=None, pipeline=None):
        reg = registry or _registry()
        return P.resolve_capabilities(reg, P.resolve_scope(reg, context_id),
                                      pipeline_portfolios=pipeline)

    # ---- individual portfolios ------------------------------------------- #
    def test_direct_portfolio_supports_pipeline_and_origination_forecast(self):
        caps = self._caps("direct_001", pipeline=["direct_001", "direct_002"])
        for cap in (P.CAP_FUNDED, P.CAP_RISK, P.CAP_EVOLUTION, P.CAP_COHORTS,
                    P.CAP_PIPELINE, P.CAP_ORIGINATION_FORECAST):
            self.assertTrue(caps[cap].enabled, cap)

    def test_acquired_portfolio_cannot_select_pipeline(self):
        caps = self._caps("acquired_001", pipeline=["direct_001", "direct_002"])
        self.assertFalse(caps[P.CAP_PIPELINE].enabled)
        self.assertEqual(caps[P.CAP_PIPELINE].reason_code, P.REASON_NON_ORIGINATING)

    def test_acquired_portfolio_cannot_select_origination_forecast(self):
        caps = self._caps("acquired_002", pipeline=["direct_001"])
        self.assertFalse(caps[P.CAP_ORIGINATION_FORECAST].enabled)
        self.assertEqual(caps[P.CAP_ORIGINATION_FORECAST].reason_code,
                         P.REASON_NON_ORIGINATING)

    def test_acquired_portfolio_keeps_full_funded_capability(self):
        """Acquired is not a second-class book: funded MI depth is identical."""
        caps = self._caps("acquired_001")
        for cap in (P.CAP_FUNDED, P.CAP_RISK, P.CAP_EVOLUTION, P.CAP_COHORTS,
                    P.CAP_DRILL_THROUGH, P.CAP_EXPORT):
            self.assertTrue(caps[cap].enabled, cap)

    def test_runoff_forecast_needs_a_supplied_governed_curve(self):
        caps = self._caps("acquired_001")
        self.assertFalse(caps[P.CAP_RUNOFF_FORECAST].enabled)
        self.assertEqual(caps[P.CAP_RUNOFF_FORECAST].reason_code,
                         P.REASON_NO_RUNOFF_PROFILE)

        with_curve = _registry(metadata={
            "acquired_001": {"runoff_profile_id": "seller_a_2026",
                             "runoff_curve": [0.99, 0.98, 0.97]}})
        caps = self._caps("acquired_001", registry=with_curve)
        self.assertTrue(caps[P.CAP_RUNOFF_FORECAST].enabled)
        self.assertEqual(caps[P.CAP_RUNOFF_FORECAST].contributing_portfolios,
                         ("acquired_001",))

    # ---- groups ----------------------------------------------------------- #
    def test_direct_group_capability_reflects_all_direct_portfolios(self):
        caps = self._caps("direct", pipeline=["direct_001", "direct_002"])
        self.assertTrue(caps[P.CAP_PIPELINE].enabled)
        self.assertEqual(sorted(caps[P.CAP_PIPELINE].contributing_portfolios),
                         ["direct_001", "direct_002"])

    def test_direct_group_pipeline_is_enabled_if_at_least_one_originates(self):
        """Only direct_002 supplies pipeline data — the group is still enabled,
        and the response discloses which book is behind it."""
        caps = self._caps("direct", pipeline=["direct_002"])
        self.assertTrue(caps[P.CAP_PIPELINE].enabled)
        self.assertEqual(caps[P.CAP_PIPELINE].contributing_portfolios, ("direct_002",))
        self.assertIn("direct_001", caps[P.CAP_PIPELINE].excluded_portfolios)
        self.assertTrue(caps[P.CAP_PIPELINE].partial)

    def test_acquired_group_pipeline_disabled_when_none_originate(self):
        caps = self._caps("acquired", pipeline=["direct_001", "direct_002"])
        self.assertFalse(caps[P.CAP_PIPELINE].enabled)
        self.assertEqual(caps[P.CAP_PIPELINE].reason_code, P.REASON_NON_ORIGINATING)

    def test_an_originating_acquired_vehicle_is_supported_by_metadata(self):
        """Acquired is not hard-coded as non-originating: governed metadata can
        say otherwise, and the capability follows without a code change."""
        registry = _registry(metadata={"acquired_002": {"originates": True}})
        caps = self._caps("acquired", registry=registry, pipeline=["acquired_002"])
        self.assertTrue(caps[P.CAP_PIPELINE].enabled)
        self.assertEqual(caps[P.CAP_PIPELINE].contributing_portfolios, ("acquired_002",))
        self.assertTrue(caps[P.CAP_ORIGINATION_FORECAST].enabled)

    def test_mixed_group_capability_is_disclosed_per_portfolio(self):
        registry = _registry(metadata={
            "acquired_001": {"runoff_profile_id": "seller_a", "runoff_curve": [0.99]}})
        caps = self._caps("acquired", registry=registry)
        runoff = caps[P.CAP_RUNOFF_FORECAST]
        self.assertTrue(runoff.enabled)
        self.assertTrue(runoff.partial)
        self.assertEqual(runoff.contributing_portfolios, ("acquired_001",))
        self.assertEqual(runoff.excluded_portfolios, ("acquired_002",))
        self.assertIn("acquired_002", runoff.detail)

    # ---- total ------------------------------------------------------------ #
    def test_total_pipeline_uses_every_eligible_originating_portfolio(self):
        caps = self._caps("total", pipeline=["direct_001", "direct_002"])
        pipeline = caps[P.CAP_PIPELINE]
        self.assertTrue(pipeline.enabled)
        self.assertEqual(sorted(pipeline.contributing_portfolios),
                         ["direct_001", "direct_002"])
        self.assertEqual(sorted(pipeline.excluded_portfolios),
                         ["acquired_001", "acquired_002"])
        self.assertIn("non-originating", pipeline.detail)

    def test_total_funded_and_risk_use_every_portfolio_in_scope(self):
        caps = self._caps("total", pipeline=["direct_001"])
        for cap in (P.CAP_FUNDED, P.CAP_RISK):
            self.assertEqual(sorted(caps[cap].contributing_portfolios),
                             ["acquired_001", "acquired_002", "direct_001", "direct_002"])

    def test_total_consolidated_forecast_includes_every_portfolio(self):
        caps = self._caps("total", pipeline=["direct_001"])
        consolidated = caps[P.CAP_CONSOLIDATED_FORECAST]
        self.assertTrue(consolidated.enabled)
        self.assertEqual(len(consolidated.contributing_portfolios), 4)

    def test_no_pipeline_data_anywhere_disables_pipeline_with_a_reason(self):
        caps = self._caps("total", pipeline=[])
        self.assertFalse(caps[P.CAP_PIPELINE].enabled)
        self.assertEqual(caps[P.CAP_PIPELINE].reason_code, P.REASON_NO_PIPELINE_DATA)


# --------------------------------------------------------------------------- #
# 4. Field coverage
# --------------------------------------------------------------------------- #
class TestFieldCoverage(unittest.TestCase):

    def _coverage(self, context_id, fields, *, missing=()):
        df = _frame(ltv_missing_in=missing)
        registry = pscope.registry_for_frame(df, client_id="ERE", metadata={})
        return pscope.coverage_for_frame(
            df, P.resolve_scope(registry, context_id), fields=fields)

    def test_field_present_in_every_portfolio_is_fully_consolidated(self):
        cov = self._coverage("total", ["current_loan_to_value"])
        self.assertTrue(cov.is_fully_consolidated)
        self.assertEqual(len(cov.portfolios_used), 4)
        self.assertIn("Fully consolidated", cov.disclosure())

    def test_field_present_in_only_some_portfolios_is_disclosed_by_portfolio(self):
        cov = self._coverage("total", ["current_loan_to_value"],
                             missing=("direct_002",))
        self.assertFalse(cov.is_fully_consolidated)
        self.assertEqual(sorted(cov.portfolios_used),
                         ["acquired_001", "acquired_002", "direct_001"])
        self.assertEqual(list(cov.portfolios_excluded), ["direct_002"])
        self.assertEqual(cov.portfolios_excluded["direct_002"]["reason_code"],
                         P.REASON_FIELD_NOT_AVAILABLE)
        field = cov.field_coverage["current_loan_to_value"]
        self.assertEqual(field.unavailable_in, ("direct_002",))

    def test_field_present_only_in_direct(self):
        cov = self._coverage("total", ["current_loan_to_value"],
                             missing=("acquired_001", "acquired_002"))
        self.assertEqual(sorted(cov.portfolios_used), ["direct_001", "direct_002"])
        self.assertEqual(sorted(cov.portfolios_excluded),
                         ["acquired_001", "acquired_002"])

    def test_field_present_only_in_acquired(self):
        cov = self._coverage("total", ["current_loan_to_value"],
                             missing=("direct_001", "direct_002"))
        self.assertEqual(sorted(cov.portfolios_used),
                         ["acquired_001", "acquired_002"])

    def test_field_absent_from_every_portfolio(self):
        cov = self._coverage("total", ["nonexistent_field"])
        self.assertEqual(cov.portfolios_used, ())
        self.assertIn("not available in the canonical platform data",
                      cov.disclosure())

    def test_a_column_that_is_entirely_null_counts_as_unavailable(self):
        """Present-but-null is unavailable, so it can never be summed as zero."""
        df = _frame()
        df["current_loan_to_value"] = None
        registry = pscope.registry_for_frame(df, client_id="ERE", metadata={})
        cov = pscope.coverage_for_frame(df, P.resolve_scope(registry, "total"),
                                        fields=["current_loan_to_value"])
        self.assertEqual(cov.portfolios_used, ())

    def test_envelope_shape_matches_the_governed_contract(self):
        cov = self._coverage("total", ["current_loan_to_value"],
                             missing=("direct_002",)).to_dict()
        for key in ("scope_requested", "portfolios_in_scope", "portfolios_used",
                    "portfolios_excluded", "is_fully_consolidated",
                    "field_coverage", "disclosure"):
            self.assertIn(key, cov)
        self.assertEqual(cov["scope_requested"]["context_id"], "total")
        self.assertEqual(cov["scope_requested"]["context_kind"], "total")


# --------------------------------------------------------------------------- #
# 5. MI Agent — identical depth across portfolio types
# --------------------------------------------------------------------------- #
class TestMiAgentAcrossPortfolios(unittest.TestCase):
    """One parser, one registry, one executor, one envelope — for every book."""

    QUESTIONS = ["What is the total current balance?",
                 "How many loans are in the book?",
                 "What is the weighted average LTV?",
                 "Show balance by collateral region."]

    def _run(self, question, context, df=None):
        return run_mi_agent_query(question, _frame() if df is None else df, _SEM,
                                  source_portfolio_lens=context)

    def test_funded_depth_is_identical_for_direct_and_acquired(self):
        for question in self.QUESTIONS:
            direct = self._run(question, "direct_001")
            acquired = self._run(question, "acquired_001")
            self.assertEqual(direct["ok"], acquired["ok"], question)
            self.assertTrue(acquired["ok"], f"acquired lost capability on: {question}")
            # Same parse, same governed spec shape — no separate acquired path.
            self.assertEqual(direct["spec"].get("metric"),
                             acquired["spec"].get("metric"), question)
            self.assertEqual(direct["spec"].get("aggregation"),
                             acquired["spec"].get("aggregation"), question)

    def test_every_context_class_answers(self):
        for context in ("total", "direct", "acquired",
                        "direct_001", "direct_002", "acquired_001", "acquired_002"):
            result = self._run("What is the total current balance?", context)
            self.assertTrue(result["ok"], context)
            self.assertEqual(result["portfolio_scope"]["context_id"], context)

    def test_group_queries_span_every_member_portfolio(self):
        result = self._run("What is the total current balance?", "acquired")
        self.assertEqual(sorted(result["portfolio_scope"]["portfolio_ids"]),
                         ["acquired_001", "acquired_002"])
        self.assertEqual(sorted(result["portfolio_coverage"]["portfolios_used"]),
                         ["acquired_001", "acquired_002"])

    def test_total_equals_direct_plus_acquired_through_the_agent(self):
        def balance(context):
            r = self._run("What is the total current balance?", context)
            return float(r["query_result"].data.iloc[0, 0])

        self.assertAlmostEqual(balance("total"),
                               balance("direct") + balance("acquired"), places=6)

    def test_partial_coverage_is_disclosed_by_individual_portfolio(self):
        df = _frame(ltv_missing_in=("direct_002",))
        result = self._run("What is the weighted average LTV?", "total", df)
        coverage = result["portfolio_coverage"]
        self.assertFalse(coverage["is_fully_consolidated"])
        self.assertIn("direct_002", coverage["portfolios_excluded"])
        self.assertTrue(any("direct_002" in w for w in result["warnings"]))

    def test_scope_filter_is_the_resolved_id_list_not_a_type_string(self):
        result = self._run("What is the total current balance?", "direct")
        filters = result["spec_obj"].filters
        self.assertEqual(sorted(filters["source_portfolio_id"]),
                         ["direct_001", "direct_002"])
        self.assertNotIn("source_portfolio_type", filters)

    def test_a_dataset_without_provenance_is_unaffected(self):
        """Single-portfolio deployments keep their existing behaviour exactly."""
        df = pd.DataFrame({"loan_identifier": ["L1"],
                           "current_outstanding_balance": [10.0]})
        result = run_mi_agent_query("What is the total current balance?", df, _SEM,
                                    source_portfolio_lens="acquired")
        self.assertTrue(result["ok"])
        self.assertNotIn("portfolio_scope", result)


# --------------------------------------------------------------------------- #
# 6. Portfolio-aware forecast
# --------------------------------------------------------------------------- #
class TestPortfolioForecast(unittest.TestCase):

    def _projections(self, context_id, *, metadata=None, weighted=0.0,
                     pipeline=None, horizon=12):
        from mi_agent_api import forecast_bridge as fb
        df = _frame()
        registry = pscope.registry_for_frame(df, client_id="ERE",
                                             metadata=metadata or {})
        scope = P.resolve_scope(registry, context_id)
        return fb.portfolio_projections(
            pscope.apply_scope(df, scope), registry, scope,
            weighted_pipeline=weighted, pipeline_portfolios=pipeline,
            horizon_months=horizon)

    def test_total_projection_is_the_sum_of_every_portfolio_projection(self):
        proj = self._projections("total", weighted=1000.0, pipeline=["direct_001"])
        summed = sum(p["projectedBalance"] for p in proj["portfolios"])
        self.assertAlmostEqual(proj["totalProjectedBalance"], summed, places=2)
        self.assertEqual(len(proj["portfolios"]), 4)

    def test_acquired_portfolios_contribute_no_new_originations(self):
        proj = self._projections("total", weighted=1000.0,
                                 pipeline=["direct_001", "direct_002"])
        for entry in proj["portfolios"]:
            if entry["portfolioType"] == "acquired":
                self.assertEqual(entry["expectedNewOriginations"], 0.0)

    def test_direct_portfolios_receive_the_expected_originations(self):
        proj = self._projections("direct", weighted=1000.0,
                                 pipeline=["direct_001", "direct_002"])
        got = {p["portfolioId"]: p["expectedNewOriginations"] for p in proj["portfolios"]}
        self.assertEqual(got, {"direct_001": 500.0, "direct_002": 500.0})

    def test_acquired_without_a_runoff_curve_is_held_constant_with_disclosure(self):
        proj = self._projections("acquired")
        for entry in proj["portfolios"]:
            self.assertEqual(entry["projectedBalance"], entry["currentBalance"])
            self.assertFalse(entry["runoffModelled"])
            self.assertIn("Runoff not modelled", entry["disclosure"])
        self.assertEqual(sorted(proj["runoffNotModelled"]),
                         ["acquired_001", "acquired_002"])

    def test_disclosure_names_each_portfolio_whose_runoff_is_not_modelled(self):
        proj = self._projections("acquired", metadata={
            "acquired_001": {"runoff_profile_id": "seller_a",
                             "runoff_curve": [0.99] * 12}})
        self.assertEqual(proj["runoffModelled"], ["acquired_001"])
        self.assertEqual(proj["runoffNotModelled"], ["acquired_002"])
        joined = " ".join(proj["disclosures"])
        self.assertIn("Runoff not modelled for acquired_002", joined)

    def test_a_supplied_runoff_curve_is_consumed(self):
        proj = self._projections("acquired_001", horizon=3, metadata={
            "acquired_001": {"runoff_profile_id": "seller_a",
                             "runoff_curve": [0.95, 0.90, 0.85]}})
        entry = proj["portfolios"][0]
        self.assertTrue(entry["runoffModelled"])
        self.assertAlmostEqual(entry["balanceRetentionFactor"], 0.85)
        self.assertAlmostEqual(entry["projectedBalance"],
                               round(entry["currentBalance"] * 0.85, 2), places=2)

    def test_no_mortality_or_decay_assumption_is_generated_internally(self):
        """With no supplied curve the balance is flat — Trakt invents nothing."""
        proj = self._projections("total", horizon=60)
        for entry in proj["portfolios"]:
            if not entry["originates"]:
                self.assertEqual(entry["projectedBalance"], entry["currentBalance"])
        self.assertIn("never generated" if "never generated" in proj["basis"]
                      else "only where the client has supplied", proj["basis"])

    def test_unattributable_pipeline_is_reported_not_split_by_guesswork(self):
        proj = self._projections("direct", weighted=900.0, pipeline=[])
        self.assertEqual(proj["unattributedExpectedOriginations"], 900.0)
        for entry in proj["portfolios"]:
            self.assertEqual(entry["expectedNewOriginations"], 0.0)
        self.assertTrue(any("not attributed" in d for d in proj["disclosures"]))


# --------------------------------------------------------------------------- #
# 7. Evolution / cohorts across multiple portfolios
# --------------------------------------------------------------------------- #
class TestEvolutionAndCohorts(unittest.TestCase):

    def test_one_reporting_date_does_not_imply_a_trend(self):
        from mi_agent_api import evolution as evo
        df = _frame()
        frames = [{"run_id": "r1", "reporting_date": "2025-12-31", "df": df}]
        result = evo.assemble_funded_evolution(frames, "ERE", "r1")
        self.assertTrue(result["singlePeriod"])
        self.assertEqual(len(result["periods"]), 1)

    def test_multiple_reporting_dates_produce_a_real_series(self):
        from mi_agent_api import evolution as evo
        frames = [
            {"run_id": "r1", "reporting_date": "2025-11-30",
             "df": _frame(reporting_dates={p[0]: ["2025-11-30"] for p in _FOUR})},
            {"run_id": "r2", "reporting_date": "2025-12-31", "df": _frame()},
        ]
        result = evo.assemble_funded_evolution(frames, "ERE", "r2")
        self.assertFalse(result["singlePeriod"])
        self.assertEqual(len(result["periods"]), 2)

    def test_the_scope_filter_is_applied_to_every_period_frame(self):
        from mi_agent_api import evolution as evo
        registry = _registry()
        scope = P.resolve_scope(registry, "acquired")
        frames = [{"run_id": f"r{i}", "reporting_date": d, "df": _frame()}
                  for i, d in enumerate(("2025-11-30", "2025-12-31"))]
        scoped = evo._apply_scope_to_frames(frames, scope)
        for frame in scoped:
            ids = set(frame["df"]["source_portfolio_id"].unique())
            self.assertEqual(ids, {"acquired_001", "acquired_002"})

    def test_a_group_series_covers_multiple_portfolios(self):
        from mi_agent_api import evolution as evo
        registry = _registry()
        for context, expected in (("direct", 2), ("acquired", 2), ("total", 4)):
            scope = P.resolve_scope(registry, context)
            frames = evo._apply_scope_to_frames(
                [{"run_id": "r1", "reporting_date": "2025-12-31", "df": _frame()}], scope)
            self.assertEqual(frames[0]["df"]["source_portfolio_id"].nunique(), expected)

    def test_an_individual_portfolio_is_isolated(self):
        from mi_agent_api import evolution as evo
        registry = _registry()
        scope = P.resolve_scope(registry, "acquired_002")
        frames = evo._apply_scope_to_frames(
            [{"run_id": "r1", "reporting_date": "2025-12-31", "df": _frame()}], scope)
        self.assertEqual(set(frames[0]["df"]["source_portfolio_id"].unique()),
                         {"acquired_002"})

    def test_portfolios_with_missing_reporting_periods_are_visible(self):
        """A book absent from a period is discoverable, not silently filled."""
        df = _frame(reporting_dates={
            "direct_001": ["2025-11-30", "2025-12-31"],
            "direct_002": ["2025-12-31"],            # no November cut
            "acquired_001": ["2025-11-30", "2025-12-31"],
            "acquired_002": ["2025-11-30", "2025-12-31"],
        })
        november = df[df["reporting_date"] == "2025-11-30"]
        registry = pscope.registry_for_frame(df, client_id="ERE", metadata={})
        scope = P.resolve_scope(registry, "total")
        coverage = pscope.coverage_for_frame(november, scope)
        self.assertNotIn("direct_002", coverage.portfolios_used)
        self.assertEqual(coverage.portfolios_excluded["direct_002"]["reason_code"],
                         P.REASON_NO_ROWS_IN_SCOPE)


# --------------------------------------------------------------------------- #
# 8. Funded charts
# --------------------------------------------------------------------------- #
class TestFundedChartCoverage(unittest.TestCase):

    def _strats(self, context_id, df=None, *, drop=()):
        from mi_agent_api import snapshots as snap
        frame = _frame() if df is None else df
        frame = frame.drop(columns=[c for c in drop if c in frame.columns])
        registry = pscope.registry_for_frame(frame, client_id="ERE", metadata={})
        scope = P.resolve_scope(registry, context_id)
        strats = snap._funded_stratifications(pscope.apply_scope(frame, scope), scope)
        return {s["key"]: s for s in strats}

    def test_populated_dimensions_render_for_direct_and_acquired_alike(self):
        for context in ("direct_001", "acquired_001", "direct", "acquired", "total"):
            strats = self._strats(context)
            self.assertTrue(strats["ltv"]["bars"], context)
            self.assertEqual(strats["ltv"]["availability"], snap_available(), context)

    def test_group_charts_combine_every_eligible_portfolio(self):
        total = self._strats("total")["ltv"]
        direct = self._strats("direct")["ltv"]
        acquired = self._strats("acquired")["ltv"]
        total_balance = sum(b["balance"] for b in total["bars"])
        self.assertAlmostEqual(
            total_balance,
            sum(b["balance"] for b in direct["bars"])
            + sum(b["balance"] for b in acquired["bars"]), places=2)

    def test_an_all_null_field_shows_an_explicit_unavailable_state(self):
        df = _frame()
        df["current_loan_to_value"] = None
        strats = self._strats("total", df)
        self.assertEqual(strats["ltv"]["availability"], "present_but_all_null")
        self.assertEqual(strats["ltv"]["bars"], [])
        self.assertTrue(strats["ltv"]["reason"])

    def test_a_missing_field_shows_not_supplied_rather_than_an_empty_chart(self):
        strats = self._strats("total", drop=("current_loan_to_value",))
        self.assertEqual(strats["ltv"]["availability"], "not_supplied")
        self.assertIn("not supplied", strats["ltv"]["reason"])

    def test_partial_availability_names_the_contributing_portfolios(self):
        df = _frame(ltv_missing_in=("direct_002",))
        strats = self._strats("total", df)
        ltv = strats["ltv"]
        self.assertEqual(ltv["availability"], "partially_available")
        self.assertIn("direct_002", ltv["missingPortfolios"])
        self.assertIn("direct_001", ltv["contributingPortfolios"])

    def test_no_chart_assumes_only_001_portfolios_exist(self):
        strats = self._strats("total", _frame(_FOUR + [("direct_003", "direct", 2)]))
        ltv = strats["ltv"]
        self.assertEqual(len(ltv.get("contributingPortfolios", [])), 5)


def snap_available():
    from mi_agent_api import snapshots as snap
    return snap.CHART_AVAILABLE


# --------------------------------------------------------------------------- #
# 9. The HTTP surface: one contract for every channel
# --------------------------------------------------------------------------- #
class TestGovernedApiSurface(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        try:
            from fastapi.testclient import TestClient  # noqa: F401
        except Exception:  # pragma: no cover - fastapi not installed
            raise unittest.SkipTest("fastapi not installed")

    def setUp(self):
        from engine import platform_assembler as pa
        from fastapi.testclient import TestClient

        # Exercise the ROUTES, not the auth guard (covered by test_auth.py).
        self._saved = {k: os.environ.get(k) for k in
                       ("MI_AGENT_AUTH_ENABLED", "MI_AGENT_PLATFORM_CANONICAL")}
        os.environ["MI_AGENT_AUTH_ENABLED"] = "false"
        self._tmp = tempfile.TemporaryDirectory()
        path = Path(self._tmp.name) / pa.PLATFORM_CANONICAL_NAME
        _frame(ltv_missing_in=("direct_002",)).to_csv(path, index=False)
        os.environ["MI_AGENT_PLATFORM_CANONICAL"] = str(path)

        from mi_agent_api import data_source as ds
        self.ds = ds
        ds.reset_cache()
        from mi_agent_api.app import app
        self.client = TestClient(app)

    def tearDown(self):
        for key, value in self._saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        self.ds.reset_cache()
        self._tmp.cleanup()

    # ---- the context contract -------------------------------------------- #
    def test_portfolio_context_exposes_the_dynamic_hierarchy(self):
        body = self.client.get("/mi/portfolio-context").json()
        self.assertTrue(body["available"])
        ids = [c["context_id"] for c in body["contexts"]]
        self.assertEqual(ids, ["total", "direct", "direct_001", "direct_002",
                               "acquired", "acquired_001", "acquired_002"])
        self.assertEqual(body["default_context_id"], "total")

    def test_every_context_carries_its_resolved_capabilities(self):
        body = self.client.get("/mi/portfolio-context").json()
        by_id = {c["context_id"]: c for c in body["contexts"]}
        acquired = by_id["acquired_001"]["capabilities"]
        self.assertFalse(acquired["pipeline"]["enabled"])
        self.assertFalse(acquired["origination_forecast"]["enabled"])
        self.assertTrue(acquired["funded"]["enabled"])
        self.assertTrue(acquired["cohorts"]["enabled"])
        self.assertTrue(acquired["evolution"]["enabled"])

    def test_portfolio_metadata_is_exposed_for_every_book(self):
        body = self.client.get("/mi/portfolio-context").json()
        by_id = {p["portfolio_id"]: p for p in body["portfolios"]}
        self.assertEqual(len(by_id), 4)
        self.assertTrue(by_id["direct_001"]["originates"])
        self.assertFalse(by_id["acquired_001"]["originates"])
        self.assertEqual(by_id["acquired_001"]["forecast_treatment"], "hold_constant")

    # ---- scoped funded surfaces ------------------------------------------ #
    def test_the_snapshot_honours_the_selected_context(self):
        def balance(context):
            body = self.client.get(
                "/mi/snapshot", params={"portfolioId": "ERE/latest",
                                        "portfolioContext": context}).json()
            return body["current_outstanding_balance"]

        self.assertAlmostEqual(balance("total"),
                               balance("direct") + balance("acquired"), places=2)
        self.assertGreater(balance("total"),
                           balance("direct_001") + balance("acquired_001"))

    def test_the_snapshot_carries_the_governed_scope_block(self):
        body = self.client.get(
            "/mi/snapshot", params={"portfolioId": "ERE/latest",
                                    "portfolioContext": "acquired"}).json()
        scope = body["portfolioScope"]
        self.assertEqual(scope["scope"]["context_id"], "acquired")
        self.assertEqual(sorted(scope["portfolios_in_scope"]),
                         ["acquired_001", "acquired_002"])
        self.assertIn("capabilities", scope)

    def test_geo_and_cohorts_honour_the_selected_context(self):
        for path in ("/mi/geo/exposure", "/mi/cohorts"):
            body = self.client.get(path, params={"portfolioId": "ERE/latest",
                                                 "portfolioContext": "direct"}).json()
            self.assertEqual(sorted(body["portfolioScope"]["portfolios_in_scope"]),
                             ["direct_001", "direct_002"], path)

    # ---- pipeline gating -------------------------------------------------- #
    def test_pipeline_refuses_an_acquired_scope_with_a_business_reason(self):
        body = self.client.get("/mi/pipeline/snapshot",
                               params={"portfolioId": "ERE/latest",
                                       "portfolioContext": "acquired"}).json()
        self.assertFalse(body["applicable"])
        self.assertEqual(body["reasonCode"], P.REASON_NON_ORIGINATING)
        # The selected context is PRESERVED — never silently switched to Direct.
        self.assertEqual(body["portfolioScope"]["context_id"], "acquired")

    def test_pipeline_evolution_and_funnel_refuse_the_same_way(self):
        for path in ("/mi/evolution/pipeline", "/mi/evolution/funnel"):
            body = self.client.get(path, params={"portfolioId": "ERE/latest",
                                                 "portfolioContext": "acquired_001"}).json()
            self.assertFalse(body["applicable"], path)
            self.assertEqual(body["portfolioScope"]["context_id"], "acquired_001", path)

    # ---- the MI Agent envelope ------------------------------------------- #
    def test_query_returns_the_governed_coverage_envelope(self):
        body = self.client.post("/mi/query", json={
            "question": "What is the weighted average LTV?",
            "sourcePortfolioLens": "total"}).json()
        self.assertTrue(body["ok"])
        coverage = body["portfolioCoverage"]
        self.assertFalse(coverage["is_fully_consolidated"])
        self.assertEqual(sorted(coverage["portfolios_in_scope"]),
                         ["acquired_001", "acquired_002", "direct_001", "direct_002"])
        self.assertIn("direct_002", coverage["portfolios_excluded"])
        self.assertEqual(coverage["portfolios_excluded"]["direct_002"]["reason_code"],
                         P.REASON_FIELD_NOT_AVAILABLE)

    def test_the_governance_block_carries_the_same_scope(self):
        body = self.client.post("/mi/query", json={
            "question": "What is the weighted average LTV?",
            "sourcePortfolioLens": "direct"}).json()
        scope = body["governance"]["scope"]
        self.assertEqual(scope["scope_requested"]["context_id"], "direct")
        self.assertEqual(sorted(scope["portfolios_in_scope"]),
                         ["direct_001", "direct_002"])

    def test_copilot_receives_the_identical_provenance(self):
        """One backend fact set: React and Copilot cannot disagree."""
        question = "What is the weighted average LTV?"
        react = self.client.post("/mi/query", json={"question": question}).json()
        saved = os.environ.get("TRAKT_COPILOT_AUTH_MODE")
        os.environ["TRAKT_COPILOT_AUTH_MODE"] = "disabled"
        try:
            copilot = self.client.post("/v1/copilot/mi/query",
                                       json={"question": question}).json()
        finally:
            if saved is None:
                os.environ.pop("TRAKT_COPILOT_AUTH_MODE", None)
            else:
                os.environ["TRAKT_COPILOT_AUTH_MODE"] = saved
        self.assertEqual(copilot["portfolioCoverage"]["portfolios_used"],
                         react["portfolioCoverage"]["portfolios_used"])
        self.assertEqual(copilot["portfolioCoverage"]["is_fully_consolidated"],
                         react["portfolioCoverage"]["is_fully_consolidated"])
        self.assertEqual(copilot["portfolioCoverage"]["disclosure"],
                         react["portfolioCoverage"]["disclosure"])

    def test_an_unscoped_request_behaves_exactly_as_before(self):
        """Backward compatibility: no portfolioContext → the consolidated book."""
        scoped = self.client.get("/mi/snapshot",
                                 params={"portfolioId": "ERE/latest",
                                         "portfolioContext": "total"}).json()
        unscoped = self.client.get("/mi/snapshot",
                                   params={"portfolioId": "ERE/latest"}).json()
        self.assertEqual(unscoped["current_outstanding_balance"],
                         scoped["current_outstanding_balance"])
        self.assertEqual(unscoped["loan_count"], scoped["loan_count"])


if __name__ == "__main__":
    unittest.main()
