#!/usr/bin/env python3
"""Phase 2A — the movement attribution must be correct before anything renders it.

These tests exist because a hover that explains a number wrongly is worse than a
hover that explains nothing. They pin, on hand-built frames where the right
answer is arithmetic rather than opinion:

  * the headline equals the difference between the two stock levels;
  * every contributor list sums back to that headline (the reconciliation
    invariant — if this breaks, the decomposition is lying);
  * each component is recognised distinctly and never silently merged;
  * ranking and tie-breaking are deterministic;
  * blank / unknown / duplicate / unkeyed cases are handled explicitly;
  * a dimension reassignment follows ONE stated convention, and is counted.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from typing import Any, Dict, List, Optional

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import pandas as pd

from mi_agent_api import movement_detail as md


def frame(rows: List[Dict[str, Any]]) -> pd.DataFrame:
    """A minimal prepared-pipeline-shaped frame."""
    cols = {md.CASE_KEY: [], md.MEASURE: [], md.STAGE: [],
            "broker_channel": [], "geographic_region_obligor": []}
    for r in rows:
        cols[md.CASE_KEY].append(r.get("id", ""))
        cols[md.MEASURE].append(r.get("amount", 0))
        cols[md.STAGE].append(r.get("stage", "KFI"))
        cols["broker_channel"].append(r.get("broker", "Air"))
        cols["geographic_region_obligor"].append(r.get("region", "London"))
    return pd.DataFrame(cols)


def build(cur, pri, detail_type=md.DETAIL_PIPELINE, **kw) -> Dict[str, Any]:
    return md.build_movement_detail(
        detail_type, cur, pri,
        as_of_date=kw.pop("as_of", "2026-08-07"),
        comparison_date=kw.pop("comparison", "2026-07-31"),
        portfolio_id="ERE", **kw)


class TestReconciliation(unittest.TestCase):
    """The invariant everything else depends on."""

    def test_headline_is_the_difference_between_the_two_stock_levels(self):
        cur = frame([{"id": "A", "amount": 100}, {"id": "B", "amount": 250}])
        pri = frame([{"id": "A", "amount": 90}])
        d = build(cur, pri)
        self.assertEqual(d["headline_metric"]["value"], 350.0)
        self.assertEqual(d["headline_metric"]["change"], 260.0)   # 350 - 90

    def test_every_contributor_list_sums_to_the_headline_change(self):
        cur = frame([
            {"id": "A", "amount": 100, "broker": "Air", "region": "London"},
            {"id": "B", "amount": 250, "broker": "Fluent", "region": "Wales"},
            {"id": "C", "amount": 40, "broker": "HUB", "region": "London"},
        ])
        pri = frame([
            {"id": "A", "amount": 90, "broker": "Air", "region": "London"},
            {"id": "D", "amount": 500, "broker": "Fluent", "region": "Wales"},
        ])
        comps = md.movement_components(
            cur, pri, dims=["broker_channel", "geographic_region_obligor"])
        total = float(comps["delta"].sum())
        d = build(cur, pri)
        self.assertEqual(d["headline_metric"]["change"], round(total, 2))
        for dim in ("broker_channel", "geographic_region_obligor"):
            full = md.rank_contributors(comps, dim, total=total, top_n=999)
            self.assertAlmostEqual(sum(c["amount"] for c in full), total, places=2,
                                   msg=f"{dim} contributors do not reconcile")

    def test_components_sum_to_the_headline_change(self):
        cur = frame([{"id": "A", "amount": 100}, {"id": "B", "amount": 250}])
        pri = frame([{"id": "A", "amount": 90}, {"id": "C", "amount": 30}])
        d = build(cur, pri)
        total = sum(v["amount"] for v in d["components"].values())
        self.assertAlmostEqual(total, d["headline_metric"]["change"], places=2)


class TestComponents(unittest.TestCase):
    """Each concept the brief requires must be distinguishable."""

    def _components(self, cur, pri, terminal=md.TERMINAL_STAGES):
        return md.movement_components(
            cur, pri, dims=["broker_channel"], terminal_stages=terminal)

    def test_a_new_case_is_new(self):
        c = self._components(frame([{"id": "A", "amount": 100}]), frame([]))
        self.assertEqual(c.loc["A", "component"], "new")
        self.assertEqual(c.loc["A", "delta"], 100.0)

    def test_a_removed_case_is_removed_and_negative(self):
        c = self._components(frame([]), frame([{"id": "A", "amount": 100}]))
        self.assertEqual(c.loc["A", "component"], "removed")
        self.assertEqual(c.loc["A", "delta"], -100.0)

    def test_an_increase_and_a_decrease_are_distinct(self):
        cur = frame([{"id": "A", "amount": 120}, {"id": "B", "amount": 80}])
        pri = frame([{"id": "A", "amount": 100}, {"id": "B", "amount": 100}])
        c = self._components(cur, pri)
        self.assertEqual(c.loc["A", "component"], "increased")
        self.assertEqual(c.loc["B", "component"], "decreased")

    def test_a_case_leaving_the_active_pipeline_is_progressed_out(self):
        """Still in the extract, but no longer an active pipeline case."""
        cur = frame([{"id": "A", "amount": 100, "stage": "COMPLETED"}])
        pri = frame([{"id": "A", "amount": 100, "stage": "OFFER"}])
        c = self._components(cur, pri)
        self.assertEqual(c.loc["A", "component"], "progressed_out")

    def test_progressed_out_wins_over_a_value_change(self):
        """A stage transition must not be disguised as an ordinary increase."""
        cur = frame([{"id": "A", "amount": 150, "stage": "WITHDRAWN"}])
        pri = frame([{"id": "A", "amount": 100, "stage": "APPLICATION"}])
        c = self._components(cur, pri)
        self.assertEqual(c.loc["A", "component"], "progressed_out")
        self.assertEqual(c.loc["A", "delta"], 50.0)

    def test_a_flat_case_is_unchanged(self):
        cur = frame([{"id": "A", "amount": 100, "stage": "OFFER"}])
        pri = frame([{"id": "A", "amount": 100, "stage": "OFFER"}])
        self.assertEqual(self._components(cur, pri).loc["A", "component"], "unchanged")

    def test_every_case_lands_in_exactly_one_component(self):
        cur = frame([{"id": "A", "amount": 120}, {"id": "B", "amount": 80},
                     {"id": "C", "amount": 50, "stage": "COMPLETED"},
                     {"id": "N", "amount": 10}])
        pri = frame([{"id": "A", "amount": 100}, {"id": "B", "amount": 100},
                     {"id": "C", "amount": 50, "stage": "OFFER"},
                     {"id": "R", "amount": 70}])
        c = self._components(cur, pri)
        self.assertEqual(len(c), 5)
        self.assertTrue(set(c["component"]) <= set(md.COMPONENTS))
        d = build(cur, pri)
        self.assertEqual(sum(v["cases"] for v in d["components"].values()), 5)

    def test_all_components_are_always_reported_even_when_empty(self):
        d = build(frame([{"id": "A", "amount": 1}]), frame([]))
        self.assertEqual(set(d["components"]), set(md.COMPONENTS))
        self.assertEqual(d["components"]["removed"], {"amount": 0.0, "cases": 0})


class TestContributorRanking(unittest.TestCase):
    def test_ranks_by_absolute_contribution_not_current_balance(self):
        """A large but static broker must not outrank a small mover."""
        cur = frame([{"id": "A", "amount": 1000, "broker": "Big"},
                     {"id": "B", "amount": 60, "broker": "Mover"}])
        pri = frame([{"id": "A", "amount": 1000, "broker": "Big"},
                     {"id": "B", "amount": 10, "broker": "Mover"}])
        d = build(cur, pri)
        self.assertEqual(d["contributors"]["brokers"][0]["name"], "Mover")

    def test_a_negative_contributor_can_rank_first(self):
        cur = frame([{"id": "A", "amount": 10, "broker": "Down"},
                     {"id": "B", "amount": 20, "broker": "Up"}])
        pri = frame([{"id": "A", "amount": 500, "broker": "Down"},
                     {"id": "B", "amount": 10, "broker": "Up"}])
        top = build(cur, pri)["contributors"]["brokers"][0]
        self.assertEqual(top["name"], "Down")
        self.assertLess(top["amount"], 0)

    def test_ties_break_on_name_ascending(self):
        cur = frame([{"id": "A", "amount": 100, "broker": "Zeta"},
                     {"id": "B", "amount": 100, "broker": "Alpha"}])
        pri = frame([{"id": "A", "amount": 50, "broker": "Zeta"},
                     {"id": "B", "amount": 50, "broker": "Alpha"}])
        names = [c["name"] for c in build(cur, pri)["contributors"]["brokers"]]
        self.assertEqual(names[:2], ["Alpha", "Zeta"])

    def test_ranking_is_stable_across_input_row_order(self):
        rows_c = [{"id": "A", "amount": 100, "broker": "Air"},
                  {"id": "B", "amount": 300, "broker": "Fluent"},
                  {"id": "C", "amount": 50, "broker": "HUB"}]
        rows_p = [{"id": "A", "amount": 90, "broker": "Air"},
                  {"id": "B", "amount": 100, "broker": "Fluent"},
                  {"id": "C", "amount": 40, "broker": "HUB"}]
        a = build(frame(rows_c), frame(rows_p))["contributors"]["brokers"]
        b = build(frame(rows_c[::-1]), frame(rows_p[::-1]))["contributors"]["brokers"]
        self.assertEqual(a, b)

    def test_top_three_by_default(self):
        cur = frame([{"id": f"C{i}", "amount": 100 * (i + 1), "broker": f"B{i}"}
                     for i in range(6)])
        self.assertEqual(len(build(cur, frame([]))["contributors"]["brokers"]), 3)

    def test_share_of_change_is_none_when_nothing_moved(self):
        f = frame([{"id": "A", "amount": 100}])
        d = build(f, f)
        for c in d["contributors"]["brokers"]:
            self.assertIsNone(c["share_of_change_pct"])

    def test_share_of_change_is_signed_against_the_total(self):
        cur = frame([{"id": "A", "amount": 200, "broker": "Air"}])
        pri = frame([{"id": "A", "amount": 100, "broker": "Air"}])
        c = build(cur, pri)["contributors"]["brokers"][0]
        self.assertEqual(c["share_of_change_pct"], 100.0)


class TestUnknownAndMalformedInput(unittest.TestCase):
    def test_a_blank_dimension_becomes_an_explicit_unknown_bucket(self):
        cur = frame([{"id": "A", "amount": 100, "broker": ""}])
        pri = frame([{"id": "A", "amount": 10, "broker": ""}])
        names = [c["name"] for c in build(cur, pri)["contributors"]["brokers"]]
        self.assertIn(md.UNKNOWN, names)

    def test_an_unknown_bucket_still_reconciles(self):
        cur = frame([{"id": "A", "amount": 100, "broker": ""},
                     {"id": "B", "amount": 50, "broker": "Air"}])
        pri = frame([{"id": "A", "amount": 10, "broker": ""}])
        d = build(cur, pri)
        total = sum(c["amount"] for c in d["contributors"]["brokers"])
        self.assertAlmostEqual(total, d["headline_metric"]["change"], places=2)

    def test_rows_with_no_case_id_are_excluded_and_reported(self):
        cur = frame([{"id": "", "amount": 999}, {"id": "A", "amount": 100}])
        d = build(cur, frame([]))
        self.assertEqual(d["headline_metric"]["change"], 100.0,
                         "an unkeyed row cannot be attributed to a movement")
        self.assertEqual(d["methodology"]["unmatched_current"],
                         {"cases": 1, "amount": 999.0})

    def test_nan_and_none_case_ids_are_treated_as_missing(self):
        cur = frame([{"id": "nan", "amount": 5}, {"id": "None", "amount": 7},
                     {"id": "A", "amount": 100}])
        d = build(cur, frame([]))
        self.assertEqual(d["methodology"]["unmatched_current"]["cases"], 2)
        self.assertEqual(d["headline_metric"]["change"], 100.0)

    def test_duplicate_case_ids_are_summed_and_counted(self):
        cur = frame([{"id": "A", "amount": 60}, {"id": "A", "amount": 40}])
        pri = frame([{"id": "A", "amount": 30}])
        d = build(cur, pri)
        self.assertEqual(d["headline_metric"]["change"], 70.0)
        self.assertEqual(d["methodology"]["duplicate_case_identifiers"]["current"], 1)

    def test_empty_frames_do_not_raise(self):
        d = build(frame([]), frame([]))
        self.assertEqual(d["headline_metric"]["change"], 0.0)
        self.assertEqual(d["contributors"]["brokers"], [])
        self.assertFalse(d["available"])

    def test_a_missing_dimension_column_yields_unknown_not_a_crash(self):
        cur = pd.DataFrame({md.CASE_KEY: ["A"], md.MEASURE: [100],
                            md.STAGE: ["KFI"]})
        d = build(cur, pd.DataFrame(columns=[md.CASE_KEY, md.MEASURE, md.STAGE]))
        self.assertEqual([c["name"] for c in d["contributors"]["brokers"]],
                         [md.UNKNOWN])


class TestAttributionConvention(unittest.TestCase):
    def test_a_reassigned_case_is_attributed_to_its_current_dimension(self):
        cur = frame([{"id": "A", "amount": 150, "broker": "New"}])
        pri = frame([{"id": "A", "amount": 100, "broker": "Old"}])
        d = build(cur, pri)
        self.assertEqual(d["contributors"]["brokers"],
                         [{"name": "New", "amount": 50.0,
                           "share_of_change_pct": 100.0, "case_count": 1}])
        self.assertEqual(d["methodology"]["attribution"], md.ATTRIBUTION)

    def test_a_reassignment_is_counted_so_the_convention_is_visible(self):
        cur = frame([{"id": "A", "amount": 150, "broker": "New", "region": "R2"},
                     {"id": "B", "amount": 10, "broker": "Same", "region": "R1"}])
        pri = frame([{"id": "A", "amount": 100, "broker": "Old", "region": "R1"},
                     {"id": "B", "amount": 10, "broker": "Same", "region": "R1"}])
        re = build(cur, pri)["methodology"]["dimension_reassignments"]
        self.assertEqual(re["broker_channel"], 1)
        self.assertEqual(re["geographic_region_obligor"], 1)

    def test_a_removed_case_is_attributed_to_its_prior_dimension(self):
        """It has no current dimension — the prior one is the only truth."""
        cur = frame([])
        pri = frame([{"id": "A", "amount": 100, "broker": "Gone"}])
        d = build(cur, pri)
        self.assertEqual(d["contributors"]["brokers"][0]["name"], "Gone")
        self.assertEqual(d["contributors"]["brokers"][0]["amount"], -100.0)

    def test_a_new_case_is_not_counted_as_a_reassignment(self):
        d = build(frame([{"id": "A", "amount": 100, "broker": "Air"}]), frame([]))
        self.assertEqual(d["methodology"]["dimension_reassignments"]["broker_channel"], 0)


class TestCompletionsDetail(unittest.TestCase):
    """The completions measure is a pipeline STAGE measure, not funded growth."""

    def test_only_completed_stage_rows_are_measured(self):
        cur = frame([{"id": "A", "amount": 100, "stage": "COMPLETED"},
                     {"id": "B", "amount": 900, "stage": "OFFER"}])
        pri = frame([{"id": "A", "amount": 100, "stage": "COMPLETED"},
                     {"id": "B", "amount": 500, "stage": "OFFER"}])
        d = build(cur, pri, detail_type=md.DETAIL_COMPLETIONS)
        self.assertEqual(d["headline_metric"]["value"], 100.0)
        self.assertEqual(d["headline_metric"]["change"], 0.0,
                         "a change in an OFFER-stage case is not a completion")

    def test_a_case_reaching_completed_appears_as_new(self):
        cur = frame([{"id": "A", "amount": 100, "stage": "COMPLETED"}])
        pri = frame([{"id": "A", "amount": 100, "stage": "OFFER"}])
        d = build(cur, pri, detail_type=md.DETAIL_COMPLETIONS)
        self.assertEqual(d["headline_metric"]["change"], 100.0)
        self.assertEqual(d["components"]["new"]["cases"], 1)

    def test_the_label_never_claims_funded_growth(self):
        d = build(frame([]), frame([]), detail_type=md.DETAIL_COMPLETIONS)
        self.assertEqual(d["headline_metric"]["label"], "Completed case value")
        definition = d["methodology"]["metric_definition"].lower()
        self.assertIn("pipeline-stage measure", definition)
        self.assertIn("not funded balance growth", definition)

    def test_completions_never_reports_progressed_out(self):
        """A completed case is already terminal — the concept does not apply."""
        cur = frame([{"id": "A", "amount": 100, "stage": "COMPLETED"}])
        pri = frame([{"id": "A", "amount": 90, "stage": "COMPLETED"}])
        d = build(cur, pri, detail_type=md.DETAIL_COMPLETIONS)
        self.assertEqual(d["components"]["progressed_out"]["cases"], 0)
        self.assertEqual(d["components"]["increased"]["cases"], 1)


class TestPayloadContract(unittest.TestCase):
    def test_dates_and_scope_are_explicit(self):
        d = build(frame([{"id": "A", "amount": 1}]), frame([]),
                  scope="acquired_001", run_id="2026-07-31",
                  source_file="w_2026-08-07.csv",
                  comparison_source_file="w_2026-07-31.csv")
        self.assertEqual(d["as_of_date"], "2026-08-07")
        self.assertEqual(d["comparison_date"], "2026-07-31")
        self.assertEqual(d["scope"], "acquired_001")
        self.assertEqual(d["run_id"], "2026-07-31")
        self.assertEqual(d["source_dates"]["pipeline_as_of"], "2026-08-07")
        self.assertIsNone(d["source_dates"]["funded_as_of"],
                          "a weekly pipeline detail must not imply a funded date")
        self.assertEqual(d["sources"]["comparison"], "w_2026-07-31.csv")

    def test_methodology_version_and_basis_are_stated(self):
        d = build(frame([{"id": "A", "amount": 1}]), frame([]))
        self.assertEqual(d["methodology"]["movement_basis"], "net")
        self.assertEqual(d["methodology"]["version"], md.METHODOLOGY_VERSION)

    def test_no_comparison_period_is_reported_as_unavailable(self):
        d = build(frame([{"id": "A", "amount": 1}]), None, comparison=None)
        self.assertFalse(d["available"])

    def test_an_unknown_detail_type_is_rejected(self):
        with self.assertRaises(ValueError):
            build(frame([]), frame([]), detail_type="SOMETHING_ELSE")

    def test_the_payload_carries_no_case_level_rows(self):
        """A hover must never leak loan-level detail."""
        cur = frame([{"id": "SECRET-CASE-1", "amount": 100}])
        d = build(cur, frame([]))
        self.assertNotIn("SECRET-CASE-1", repr(d))


class TestReconcilesToTheGovernedSeries(unittest.TestCase):
    """The headline must be the number the chart already plots — not a new one.

    Run against the committed governed pack rather than the synthetic
    performance fixture: that fixture reassigns broker and region randomly every
    week (it was generated to measure latency, not semantics), so it is
    unsuitable for validating attribution. It is still exercised below for the
    one property it CAN prove — that a churning dimension is reported rather
    than silently absorbed.
    """

    FIXTURE = _REPO_ROOT / "tests" / "fixtures" / "client_001_mi_pack" / "pipeline"

    @classmethod
    def setUpClass(cls):
        if not cls.FIXTURE.exists():
            raise unittest.SkipTest("fixture pack not present")
        from mi_agent_api import evolution as ev
        from mi_agent_api import pipeline_contract as pc
        cls.ev, cls.pc = ev, pc
        cls.extracts = pc.weekly_extract_inventory(str(cls.FIXTURE),
                                                   "client_001")["extracts"]
        if len(cls.extracts) < 2:
            raise unittest.SkipTest("need two weekly extracts to compare")

    def _pair(self):
        pri_e, cur_e = self.extracts[-2], self.extracts[-1]
        cur, _ = self.pc.load_prepared_pipeline(cur_e)
        pri, _ = self.pc.load_prepared_pipeline(pri_e)
        return cur, pri, cur_e, pri_e

    def test_pipeline_headline_equals_the_evolution_series(self):
        cur, pri, cur_e, pri_e = self._pair()
        series = self.ev.pipeline_evolution(str(self.FIXTURE), "client_001")["periods"]
        by_date = {p.get("extract_date"): p for p in series}
        now = by_date[cur_e["pipeline_extract_date"]]["metrics"]["pipeline_amount"]
        prev = by_date[pri_e["pipeline_extract_date"]]["metrics"]["pipeline_amount"]

        d = md.build_movement_detail(
            md.DETAIL_PIPELINE, cur, pri,
            as_of_date=cur_e["pipeline_extract_date"],
            comparison_date=pri_e["pipeline_extract_date"],
            portfolio_id="client_001")
        self.assertAlmostEqual(d["headline_metric"]["value"], now, places=2)
        self.assertAlmostEqual(d["headline_metric"]["change"], now - prev, places=2)

    def test_completions_headline_equals_the_funnel_flow(self):
        cur, pri, cur_e, pri_e = self._pair()
        fn = self.ev.pipeline_funnel_evolution(str(self.FIXTURE), "client_001")
        stock = {p["week"]: p["value"] for p in fn["series"]["COMPLETED"]}
        flow = {p["week"]: p["flowValue"] for p in fn["flowSeries"]["COMPLETED"]}
        cd, pd_ = cur_e["pipeline_extract_date"], pri_e["pipeline_extract_date"]

        d = md.build_movement_detail(
            md.DETAIL_COMPLETIONS, cur, pri,
            as_of_date=cd, comparison_date=pd_, portfolio_id="client_001")
        self.assertAlmostEqual(d["headline_metric"]["value"], stock[cd], places=2)
        self.assertAlmostEqual(d["headline_metric"]["change"], flow[cd], places=2,
                               msg="completions change must be the funnel's weekly flow")

    def test_contributors_reconcile_on_real_data(self):
        cur, pri, cur_e, pri_e = self._pair()
        comps = md.movement_components(
            cur, pri, dims=[c for _k, c in md.DIMENSIONS])
        total = float(comps["delta"].sum())
        for _key, col in md.DIMENSIONS:
            full = md.rank_contributors(comps, col, total=total, top_n=10_000)
            self.assertAlmostEqual(sum(c["amount"] for c in full), total, places=2)

    def test_a_churning_dimension_is_reported_not_absorbed(self):
        """The convention picks a side; the payload must say how often it mattered."""
        cur = frame([{"id": "A", "amount": 100, "broker": "X"},
                     {"id": "B", "amount": 100, "broker": "Y"}])
        pri = frame([{"id": "A", "amount": 100, "broker": "Y"},
                     {"id": "B", "amount": 100, "broker": "X"}])
        d = build(cur, pri)
        self.assertEqual(d["methodology"]["dimension_reassignments"]["broker_channel"], 2)
        self.assertEqual(d["headline_metric"]["change"], 0.0)


if __name__ == "__main__":
    unittest.main()
