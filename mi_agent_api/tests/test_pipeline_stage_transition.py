#!/usr/bin/env python3
"""The governed pipeline stage-transition capability.

A net stage movement cannot answer a gross question. An OFFER stock that falls
from three cases to one is a net -2, and that is equally consistent with two
leaving and none arriving, or four leaving and two arriving. These tests pin the
capability that closes that gap, on a fixture where the right answer is
arithmetic rather than opinion:

  * identity is the governed ``pipeline_case_identifier`` and nothing else — an
    amount amendment never splits one case into a departure plus an arrival;
  * every case is classified into exactly one of four event classes;
  * a departure's outcome is stated only where the governed data evidences it,
    and is left visibly unclassified where it does not;
  * every stage reconciles, in counts EXACTLY and in amounts to a floating-point
    tolerance, from opening stock to closing stock;
  * the whole identifier population reconciles — nothing disappears, nothing is
    counted twice;
  * the capability is reachable through the PRODUCTION preparation path, not a
    test-only computation; and
  * it takes ownership of no existing pipeline or forecast metric.
"""

from __future__ import annotations

import copy
import sys
import unittest
from pathlib import Path
from typing import Any, Dict, List, Optional

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import pandas as pd

from mi_agent_api import movement_detail as md

#: The two-snapshot pack built by ``build_fixture.py`` next to it.
FIXTURE = _REPO_ROOT / "tests" / "fixtures" / "pipeline_transition_2w"
#: The five-week pack the existing pipeline views are already asserted against.
HISTORY_FIXTURE = _REPO_ROOT / "tests" / "fixtures" / "pipeline_history_5w"


def frame(rows: List[Dict[str, Any]]) -> pd.DataFrame:
    """A minimal prepared-pipeline-shaped frame."""
    return pd.DataFrame({
        md.CASE_KEY: [r.get("id", "") for r in rows],
        md.MEASURE: [r.get("amount", 0) for r in rows],
        md.STAGE: [r.get("stage", "KFI") for r in rows],
    })


def build(cur, pri, **kw) -> Dict[str, Any]:
    return md.build_stage_transition_detail(
        cur, pri,
        as_of_date=kw.pop("as_of", "2026-06-12"),
        comparison_date=kw.pop("comparison", "2026-06-05"),
        portfolio_id="ERE", **kw)


def by_key(rows: List[Dict[str, Any]], *keys: str) -> Dict[Any, Dict[str, Any]]:
    return {tuple(r[k] for k in keys) if len(keys) > 1 else r[keys[0]]: r
            for r in rows}


# --------------------------------------------------------------------------- #
# 1-4. Identity
# --------------------------------------------------------------------------- #
class TestIdentity(unittest.TestCase):
    """One case ID. Two snapshots. One governed classification."""

    def test_matching_uses_the_governed_case_identifier(self):
        """Not a row position, not an amount, not a derived key."""
        self.assertEqual(md.CASE_KEY, "pipeline_case_identifier")
        cur = frame([{"id": "B", "amount": 50}, {"id": "A", "amount": 100}])
        pri = frame([{"id": "A", "amount": 100}, {"id": "B", "amount": 50}])
        events = md.stage_transition_events(cur, pri)
        # Row order differs between the snapshots; identity does not.
        self.assertEqual(set(events.index), {"A", "B"})
        self.assertEqual(set(events["event_class"]), {md.EVENT_STAYER})
        d = build(cur, pri)
        self.assertEqual(d["identifier"], "pipeline_case_identifier")
        self.assertEqual(d["methodology"]["identity_basis"],
                         "pipeline_case_identifier")

    def test_an_amount_change_does_not_alter_identity(self):
        """The defect this capability exists to avoid: an amendment read as a
        departure plus an arrival, doubling both sides of the movement."""
        cur = frame([{"id": "A", "amount": 220_000, "stage": "KFI"}])
        pri = frame([{"id": "A", "amount": 200_000, "stage": "KFI"}])
        events = md.stage_transition_events(cur, pri)
        self.assertEqual(len(events), 1)
        self.assertEqual(events.loc["A", "event_class"], md.EVENT_STAYER)
        self.assertEqual(events.loc["A", "amount_change"], 20_000.0)
        d = build(cur, pri)
        self.assertEqual(d["event_totals"][md.EVENT_DEPARTURE]["case_count"], 0)
        self.assertEqual(d["event_totals"][md.EVENT_NEW_ARRIVAL]["case_count"], 0)

    def test_duplicate_identifiers_make_the_capability_unavailable(self):
        """Deterministic matching is impossible, so the capability refuses
        rather than picking a row or silently summing two cases into one."""
        cur = frame([{"id": "A", "amount": 100}, {"id": "A", "amount": 100}])
        pri = frame([{"id": "A", "amount": 100}])
        d = build(cur, pri)
        self.assertFalse(d["available"])
        self.assertEqual(d["reason_code"], md.REASON_DUPLICATE_IDENTIFIERS)
        self.assertIsNone(d["reconciliation"])
        self.assertEqual(d["transitions"], [])
        self.assertEqual(
            d["methodology"]["duplicate_case_identifiers"]["current"], 1)

    def test_a_duplicate_in_the_prior_snapshot_also_refuses(self):
        d = build(frame([{"id": "A", "amount": 100}]),
                  frame([{"id": "A", "amount": 100}, {"id": "A", "amount": 100}]))
        self.assertFalse(d["available"])
        self.assertEqual(d["reason_code"], md.REASON_DUPLICATE_IDENTIFIERS)

    def test_a_missing_identifier_makes_the_capability_unavailable(self):
        cur = pd.DataFrame({md.MEASURE: [100.0], md.STAGE: ["KFI"]})
        pri = pd.DataFrame({md.MEASURE: [100.0], md.STAGE: ["KFI"]})
        d = build(cur, pri)
        self.assertFalse(d["available"])
        self.assertEqual(d["reason_code"], md.REASON_MISSING_IDENTIFIER)

    def test_a_blank_identifier_column_makes_the_capability_unavailable(self):
        d = build(frame([{"id": "", "amount": 100}]),
                  frame([{"id": "", "amount": 100}]))
        self.assertFalse(d["available"])
        self.assertEqual(d["reason_code"], md.REASON_MISSING_IDENTIFIER)

    def test_no_fallback_identity_is_invented_for_an_unkeyed_row(self):
        """A row with no key is excluded and REPORTED, never matched on amount,
        row number or any other mutable field."""
        cur = frame([{"id": "A", "amount": 100}, {"id": "", "amount": 999}])
        pri = frame([{"id": "A", "amount": 100}, {"id": "", "amount": 999}])
        d = build(cur, pri)
        self.assertTrue(d["available"])
        self.assertEqual(d["methodology"]["unmatched_current"],
                         {"cases": 1, "amount": 999.0})
        self.assertEqual(d["counts"]["current"], 1)
        self.assertEqual(d["reconciliation"]["global"]["union_population"], 1)

    def test_an_empty_snapshot_is_not_treated_as_a_missing_identifier(self):
        """Everything departing is a governed situation the data can answer;
        refusing it as a missing identifier would withhold a real answer."""
        d = build(frame([]), frame([{"id": "A", "amount": 100, "stage": "OFFER"}]))
        self.assertTrue(d["available"], d.get("reason"))
        self.assertEqual(d["event_totals"][md.EVENT_DEPARTURE]["case_count"], 1)
        self.assertEqual(d["reconciliation"]["count_reconciliation_residual"], 0)

    def test_two_empty_snapshots_are_a_governed_unavailability(self):
        d = build(frame([]), frame([]))
        self.assertFalse(d["available"])
        self.assertEqual(d["reason_code"], md.REASON_NO_CASES)

    def test_no_prior_snapshot_is_a_governed_unavailability_not_a_zero(self):
        d = md.build_stage_transition_detail(
            frame([{"id": "A", "amount": 100}]), None,
            as_of_date="2026-06-12", comparison_date=None, portfolio_id="ERE")
        self.assertFalse(d["available"])
        self.assertEqual(d["reason_code"], md.REASON_NO_COMPARISON)


# --------------------------------------------------------------------------- #
# 5-10. Classification
# --------------------------------------------------------------------------- #
class TestClassification(unittest.TestCase):
    """Four classes, mutually exclusive and collectively exhaustive."""

    def _events(self, cur, pri):
        return md.stage_transition_events(cur, pri)

    def test_a_latest_only_identifier_is_a_new_arrival(self):
        e = self._events(frame([{"id": "A", "amount": 100, "stage": "KFI"}]), frame([]))
        self.assertEqual(e.loc["A", "event_class"], md.EVENT_NEW_ARRIVAL)
        self.assertEqual(e.loc["A", "destination_stage"], "KFI")
        self.assertEqual(e.loc["A", "latest_amount"], 100.0)

    def test_a_new_arrival_is_not_given_a_prior_stage_it_never_had(self):
        """A synthetic source token here would be indistinguishable downstream
        from a real governed stage."""
        e = self._events(frame([{"id": "A", "amount": 100}]), frame([]))
        self.assertIsNone(e.loc["A", "source_stage"])
        self.assertTrue(pd.isna(e.loc["A", "prior_amount"]))

    def test_both_snapshots_same_stage_is_a_stayer(self):
        e = self._events(frame([{"id": "A", "amount": 120, "stage": "OFFER"}]),
                         frame([{"id": "A", "amount": 100, "stage": "OFFER"}]))
        self.assertEqual(e.loc["A", "event_class"], md.EVENT_STAYER)
        self.assertEqual(e.loc["A", "source_stage"], "OFFER")
        self.assertEqual(e.loc["A", "destination_stage"], "OFFER")

    def test_both_snapshots_different_stage_is_a_stage_transition(self):
        e = self._events(frame([{"id": "A", "amount": 100, "stage": "APPLICATION"}]),
                         frame([{"id": "A", "amount": 100, "stage": "KFI"}]))
        self.assertEqual(e.loc["A", "event_class"], md.EVENT_STAGE_TRANSITION)
        self.assertEqual(e.loc["A", "source_stage"], "KFI")
        self.assertEqual(e.loc["A", "destination_stage"], "APPLICATION")

    def test_a_prior_only_identifier_is_a_departure(self):
        e = self._events(frame([]), frame([{"id": "A", "amount": 100, "stage": "OFFER"}]))
        self.assertEqual(e.loc["A", "event_class"], md.EVENT_DEPARTURE)
        self.assertEqual(e.loc["A", "source_stage"], "OFFER")
        self.assertIsNone(e.loc["A", "destination_stage"])

    def test_a_terminal_outcome_is_classified_only_where_evidence_supports_it(self):
        e = self._events(frame([]), frame([
            {"id": "C", "amount": 100, "stage": "COMPLETED"},
            {"id": "W", "amount": 100, "stage": "WITHDRAWN"}]))
        self.assertEqual(e.loc["C", "governed_outcome"], "COMPLETED")
        self.assertEqual(e.loc["C", "outcome_evidence"], md.EVIDENCE_PRIOR_TERMINAL)
        self.assertEqual(e.loc["W", "governed_outcome"], "WITHDRAWN")
        self.assertEqual(e.loc["W", "outcome_evidence"], md.EVIDENCE_PRIOR_TERMINAL)

    def test_an_unclassifiable_departure_is_never_guessed(self):
        """Absence from the latest extract is NOT evidence of a withdrawal."""
        e = self._events(frame([]), frame([
            {"id": "K", "amount": 100, "stage": "KFI"},
            {"id": "O", "amount": 100, "stage": "OFFER"},
            {"id": "U", "amount": 100, "stage": "UNKNOWN"}]))
        for case in ("K", "O", "U"):
            self.assertEqual(e.loc[case, "governed_outcome"],
                             md.UNCLASSIFIED_DEPARTURE,
                             msg=f"{case} was given an outcome the data does not support")
            self.assertEqual(e.loc[case, "outcome_evidence"], md.EVIDENCE_NONE)

    def test_the_unclassified_distinction_survives_into_the_payload(self):
        d = build(frame([]), frame([{"id": "C", "amount": 100, "stage": "COMPLETED"},
                                    {"id": "O", "amount": 100, "stage": "OFFER"}]))
        outcomes = {r["source_stage"]: r["governed_outcome"] for r in d["departures"]}
        self.assertEqual(outcomes["COMPLETED"], "COMPLETED")
        self.assertEqual(outcomes["OFFER"], md.UNCLASSIFIED_DEPARTURE)

    def test_every_case_lands_in_exactly_one_class(self):
        cur = frame([{"id": "S", "amount": 100, "stage": "KFI"},
                     {"id": "T", "amount": 100, "stage": "OFFER"},
                     {"id": "N", "amount": 100, "stage": "KFI"}])
        pri = frame([{"id": "S", "amount": 100, "stage": "KFI"},
                     {"id": "T", "amount": 100, "stage": "APPLICATION"},
                     {"id": "D", "amount": 100, "stage": "OFFER"}])
        e = md.stage_transition_events(cur, pri)
        self.assertEqual(len(e), 4)
        self.assertEqual(e.index.nunique(), 4)
        self.assertEqual(set(e["event_class"]), set(md.EVENT_CLASSES))


# --------------------------------------------------------------------------- #
# 11-18. The fixture truth table
# --------------------------------------------------------------------------- #
class TestFixtureTruth(unittest.TestCase):
    """The expected answer is independently obvious from ``build_fixture.py``.

    Built through the PRODUCTION resolution path — the governed extract
    inventory, ``load_prepared_pipeline`` and the same immediate-neighbour rule
    the pipeline views use — so this is not a test-only computation.
    """

    @classmethod
    def setUpClass(cls):
        if not FIXTURE.exists():
            raise unittest.SkipTest("transition fixture pack not present")
        cls.detail = md.resolve_stage_transition_detail(str(FIXTURE), "client_001")

    def test_the_capability_is_available_on_the_governed_pair(self):
        self.assertTrue(self.detail["available"], self.detail.get("reason"))
        self.assertEqual(self.detail["as_of_date"], "2026-06-12")
        self.assertEqual(self.detail["comparison_date"], "2026-06-05")
        self.assertEqual(self.detail["counts"],
                         {"current": 10, "comparison": 12, "change": -2})

    # --- transitions ------------------------------------------------------- #
    def test_kfi_to_application_count_is_correct(self):
        """3004 and 3005 both move KFI -> APPLICATION."""
        t = by_key(self.detail["transitions"], "source_stage", "destination_stage")
        self.assertEqual(t[("KFI", "APPLICATION")]["case_count"], 2)

    def test_kfi_to_application_amount_is_correct(self):
        """400k + 500k out, 400k + 520k in: 3005 was amended by +20k in flight."""
        row = by_key(self.detail["transitions"],
                     "source_stage", "destination_stage")[("KFI", "APPLICATION")]
        self.assertEqual(row["prior_amount"], 900_000.0)
        self.assertEqual(row["latest_amount"], 920_000.0)
        self.assertEqual(row["amount_change"], 20_000.0)

    def test_application_to_offer_count_is_correct(self):
        t = by_key(self.detail["transitions"], "source_stage", "destination_stage")
        self.assertEqual(t[("APPLICATION", "OFFER")]["case_count"], 2)

    def test_application_to_offer_amount_is_correct(self):
        """600k + 700k out, 600k + 690k in: 3007 was reduced by 10k."""
        row = by_key(self.detail["transitions"],
                     "source_stage", "destination_stage")[("APPLICATION", "OFFER")]
        self.assertEqual(row["prior_amount"], 1_300_000.0)
        self.assertEqual(row["latest_amount"], 1_290_000.0)
        self.assertEqual(row["amount_change"], -10_000.0)

    def test_the_terminal_transition_is_published(self):
        """3008 reaches COMPLETED while still present in the latest extract —
        a governed transition, not a departure."""
        row = by_key(self.detail["transitions"],
                     "source_stage", "destination_stage")[("OFFER", "COMPLETED")]
        self.assertEqual(row["case_count"], 1)
        self.assertEqual(row["prior_amount"], 800_000.0)
        self.assertEqual(row["latest_amount"], 800_000.0)
        self.assertEqual(row["amount_change"], 0.0)

    def test_multiple_transitions_aggregate_and_nothing_else_appears(self):
        self.assertEqual(
            [(r["source_stage"], r["destination_stage"], r["case_count"])
             for r in self.detail["transitions"]],
            [("KFI", "APPLICATION", 2),
             ("APPLICATION", "OFFER", 2),
             ("OFFER", "COMPLETED", 1)])
        self.assertEqual(
            self.detail["event_totals"][md.EVENT_STAGE_TRANSITION]["case_count"], 5)

    def test_transitions_are_ordered_by_the_governed_funnel_order(self):
        from mi_agent_api.pipeline_prep import canonical_stage_order
        order = canonical_stage_order()
        positions = [order.index(r["source_stage"]) for r in self.detail["transitions"]]
        self.assertEqual(positions, sorted(positions))

    # --- new arrivals and departures ---------------------------------------- #
    def test_new_arrivals_are_reported_by_entry_stage(self):
        a = by_key(self.detail["new_arrivals"], "destination_stage")
        self.assertEqual(a["KFI"], {"destination_stage": "KFI", "case_count": 1,
                                    "latest_amount": 900_000.0})
        self.assertEqual(a["APPLICATION"],
                         {"destination_stage": "APPLICATION", "case_count": 1,
                          "latest_amount": 150_000.0})

    def test_departures_split_governed_outcomes_from_unclassified_ones(self):
        d = by_key(self.detail["departures"], "source_stage")
        self.assertEqual(d["COMPLETED"]["governed_outcome"], "COMPLETED")
        self.assertEqual(d["COMPLETED"]["prior_amount"], 1_000_000.0)
        self.assertEqual(d["WITHDRAWN"]["governed_outcome"], "WITHDRAWN")
        self.assertEqual(d["WITHDRAWN"]["prior_amount"], 1_100_000.0)
        # 3013 (OFFER) and 3014 (APPLICATION) simply vanish — no evidence.
        self.assertEqual(d["OFFER"]["governed_outcome"], md.UNCLASSIFIED_DEPARTURE)
        self.assertEqual(d["APPLICATION"]["governed_outcome"], md.UNCLASSIFIED_DEPARTURE)
        self.assertEqual(
            sum(r["case_count"] for r in self.detail["departures"]
                if r["governed_outcome"] == md.UNCLASSIFIED_DEPARTURE), 2)

    # --- amendments ---------------------------------------------------------- #
    def test_an_amount_increase_on_a_stayer_is_recorded(self):
        """3002: KFI 200k -> 220k, alongside 3001 unchanged at 100k."""
        kfi = by_key(self.detail["stayers"], "stage")["KFI"]
        self.assertEqual(kfi["case_count"], 2)
        self.assertEqual(kfi["prior_amount"], 300_000.0)
        self.assertEqual(kfi["latest_amount"], 320_000.0)
        self.assertEqual(kfi["amount_change"], 20_000.0)

    def test_an_amount_decrease_on_a_stayer_is_recorded(self):
        """3003: APPLICATION 300k -> 280k."""
        app = by_key(self.detail["stayers"], "stage")["APPLICATION"]
        self.assertEqual(app["case_count"], 1)
        self.assertEqual(app["amount_change"], -20_000.0)

    def test_an_amendment_on_a_transitioning_case_stays_one_case(self):
        """3005 is KFI GBP500k -> APPLICATION GBP520k: ONE case, ONE transition,
        +GBP20k — never a departure plus an arrival."""
        events = self._events()
        row = events.loc["ACC3005"]
        self.assertEqual(row["event_class"], md.EVENT_STAGE_TRANSITION)
        self.assertEqual(row["source_stage"], "KFI")
        self.assertEqual(row["destination_stage"], "APPLICATION")
        self.assertEqual(row["amount_change"], 20_000.0)
        self.assertEqual(int((events.index == "ACC3005").sum()), 1)

    def test_a_stayer_with_an_amendment_is_still_a_stayer(self):
        events = self._events()
        self.assertEqual(events.loc["ACC3002", "event_class"], md.EVENT_STAYER)
        self.assertEqual(events.loc["ACC3003", "event_class"], md.EVENT_STAYER)

    @classmethod
    def _prepared_pair(cls):
        from mi_agent_api import pipeline_contract as pc
        extracts = pc.weekly_extract_inventory(str(FIXTURE), "client_001")["extracts"]
        cur, _ = pc.load_prepared_pipeline(extracts[-1])
        pri, _ = pc.load_prepared_pipeline(extracts[-2])
        return cur, pri

    def _events(self):
        cur, pri = self._prepared_pair()
        return md.stage_transition_events(cur, pri)


# --------------------------------------------------------------------------- #
# 19-22. Reconciliation
# --------------------------------------------------------------------------- #
class TestReconciliation(unittest.TestCase):
    """The capability is not complete unless it reconciles."""

    @classmethod
    def setUpClass(cls):
        if not FIXTURE.exists():
            raise unittest.SkipTest("transition fixture pack not present")
        cls.detail = md.resolve_stage_transition_detail(str(FIXTURE), "client_001")
        cls.recon = cls.detail["reconciliation"]

    def test_every_stage_count_reconciliation_has_a_zero_residual(self):
        for row in self.recon["by_stage"]:
            self.assertEqual(
                row["opening_case_count"] + row["new_arrivals"]
                + row["transitions_in"] - row["transitions_out"]
                - row["departures"],
                row["closing_case_count"],
                msg=f"{row['stage']} count identity does not hold")
            self.assertEqual(row["count_reconciliation_residual"], 0,
                             msg=f"{row['stage']} carries a count residual")
        self.assertEqual(self.recon["count_reconciliation_residual"], 0)

    def test_every_stage_amount_reconciliation_is_within_tolerance(self):
        tol = self.recon["amount_tolerance"]
        for row in self.recon["by_stage"]:
            lhs = (row["opening_amount"] + row["new_arrival_amount"]
                   + row["transferred_in_latest_amount"]
                   - row["transferred_out_prior_amount"]
                   - row["departure_prior_amount"]
                   + row["stayer_amount_change"])
            self.assertAlmostEqual(lhs, row["closing_amount"], delta=tol,
                                   msg=f"{row['stage']} amount identity does not hold")
            self.assertLessEqual(abs(row["amount_reconciliation_residual"]), tol,
                                 msg=f"{row['stage']} carries an amount residual")
        self.assertLessEqual(self.recon["amount_reconciliation_residual"], tol)

    def test_the_per_stage_opening_and_closing_are_the_real_stage_stocks(self):
        """The reconciliation is anchored to the two snapshots' own stage stock,
        not to a total the capability invented for itself."""
        from mi_agent_api import pipeline_contract as pc
        extracts = pc.weekly_extract_inventory(str(FIXTURE), "client_001")["extracts"]
        cur, _ = pc.load_prepared_pipeline(extracts[-1])
        pri, _ = pc.load_prepared_pipeline(extracts[-2])
        opening = pri[md.STAGE].value_counts().to_dict()
        closing = cur[md.STAGE].value_counts().to_dict()
        for row in self.recon["by_stage"]:
            self.assertEqual(row["opening_case_count"], opening.get(row["stage"], 0))
            self.assertEqual(row["closing_case_count"], closing.get(row["stage"], 0))

    def test_the_global_identifier_reconciliation_is_exact(self):
        g = self.recon["global"]
        self.assertEqual(g["prior_only"] + g["in_both"] + g["latest_only"],
                         g["union_population"])
        self.assertEqual(g["classified_events"], g["union_population"])
        self.assertEqual(g["residual"], 0)
        self.assertEqual(g["duplicate_classifications"], 0)
        self.assertEqual((g["prior_population"], g["latest_population"]), (12, 10))
        self.assertEqual((g["prior_only"], g["in_both"], g["latest_only"]), (4, 8, 2))

    def test_the_event_classes_partition_the_identifier_union(self):
        totals = self.detail["event_totals"]
        self.assertEqual(sum(v["case_count"] for v in totals.values()),
                         self.recon["global"]["union_population"])
        self.assertEqual({k: v["case_count"] for k, v in totals.items()},
                         {md.EVENT_NEW_ARRIVAL: 2, md.EVENT_STAYER: 3,
                          md.EVENT_STAGE_TRANSITION: 5, md.EVENT_DEPARTURE: 4})

    def test_source_destination_totals_reconcile_to_the_event_population(self):
        moved = sum(r["case_count"] for r in self.detail["transitions"])
        self.assertEqual(
            moved, self.detail["event_totals"][md.EVENT_STAGE_TRANSITION]["case_count"])
        self.assertEqual(
            sum(r["case_count"] for r in self.detail["new_arrivals"])
            + sum(r["case_count"] for r in self.detail["stayers"])
            + moved
            + sum(r["case_count"] for r in self.detail["departures"]),
            self.recon["global"]["union_population"])

    def test_the_transition_matrix_amounts_reconcile_to_the_event_totals(self):
        totals = self.detail["event_totals"][md.EVENT_STAGE_TRANSITION]
        self.assertAlmostEqual(sum(r["prior_amount"] for r in self.detail["transitions"]),
                               totals["prior_amount"], places=2)
        self.assertAlmostEqual(sum(r["latest_amount"] for r in self.detail["transitions"]),
                               totals["latest_amount"], places=2)

    def test_a_residual_would_be_published_rather_than_hidden(self):
        """The residual fields exist and are read from the identity, so a real
        break shows up as a number instead of vanishing."""
        for row in self.recon["by_stage"]:
            self.assertIn("count_reconciliation_residual", row)
            self.assertIn("amount_reconciliation_residual", row)


# --------------------------------------------------------------------------- #
# 14. Production reachability
# --------------------------------------------------------------------------- #
class TestProductionReachability(unittest.TestCase):
    """The capability must read the SAME prepared data production MI reads."""

    @classmethod
    def setUpClass(cls):
        if not FIXTURE.exists():
            raise unittest.SkipTest("transition fixture pack not present")

    def test_the_capability_resolves_through_the_governed_extract_inventory(self):
        from mi_agent_api import pipeline_contract as pc
        extracts = pc.weekly_extract_inventory(str(FIXTURE), "client_001")["extracts"]
        self.assertEqual([e["pipeline_extract_date"] for e in extracts],
                         ["2026-06-05", "2026-06-12"])
        d = md.resolve_stage_transition_detail(str(FIXTURE), "client_001")
        self.assertEqual(d["as_of_date"], extracts[-1]["pipeline_extract_date"])
        self.assertEqual(d["comparison_date"], extracts[-2]["pipeline_extract_date"])
        self.assertEqual(d["sources"]["current"],
                         Path(extracts[-1]["source_file"]).name)

    def test_it_compares_the_immediately_prior_snapshot(self):
        """The same neighbour rule the movement detail and the charts use."""
        from mi_agent_api import pipeline_contract as pc
        extracts = pc.weekly_extract_inventory(str(FIXTURE), "client_001")["extracts"]
        cur_e, pri_e = md.select_pair(extracts)
        self.assertEqual(cur_e["pipeline_extract_date"], "2026-06-12")
        self.assertEqual(pri_e["pipeline_extract_date"], "2026-06-05")

    def test_the_earliest_snapshot_has_nothing_to_compare_against(self):
        d = md.resolve_stage_transition_detail(str(FIXTURE), "client_001",
                                               as_of="2026-06-05")
        self.assertFalse(d["available"])
        self.assertEqual(d["reason_code"], md.REASON_NO_COMPARISON)

    def test_the_stage_vocabulary_is_the_one_the_pipeline_views_use(self):
        """Not a second stage list: every stage the capability publishes is a
        canonical token produced by ``pipeline_prep.canonical_stage``."""
        from mi_agent_api.pipeline_prep import canonical_stage, canonical_stage_order
        d = md.resolve_stage_transition_detail(str(FIXTURE), "client_001")
        seen = set()
        for r in d["transitions"]:
            seen.update((r["source_stage"], r["destination_stage"]))
        seen.update(r["destination_stage"] for r in d["new_arrivals"])
        seen.update(r["stage"] for r in d["stayers"])
        seen.update(r["source_stage"] for r in d["departures"])
        self.assertTrue(seen)
        for stage in seen:
            self.assertEqual(canonical_stage(stage), stage,
                             msg=f"{stage} is not a canonical governed stage")
            self.assertIn(stage, canonical_stage_order())

    def test_it_also_runs_against_the_existing_five_week_governed_pack(self):
        """A second, independently built governed pack — the capability is not
        tuned to one fixture."""
        if not HISTORY_FIXTURE.exists():
            self.skipTest("five-week fixture not present")
        d = md.resolve_stage_transition_detail(str(HISTORY_FIXTURE), "client_001")
        self.assertTrue(d["available"], d.get("reason"))
        self.assertEqual(d["reconciliation"]["count_reconciliation_residual"], 0)
        self.assertLessEqual(d["reconciliation"]["amount_reconciliation_residual"],
                             md.AMOUNT_TOLERANCE)
        self.assertEqual(d["reconciliation"]["global"]["residual"], 0)
        # Week 4 -> week 5: 2002 APPLICATION->OFFER and 2008 KFI->OFFER.
        self.assertEqual(
            [(r["source_stage"], r["destination_stage"], r["case_count"])
             for r in d["transitions"]],
            [("KFI", "OFFER", 1), ("APPLICATION", "OFFER", 1)])

    def test_the_payload_carries_no_case_identifiers(self):
        """Same discipline as the movement payload it sits beside."""
        d = md.resolve_stage_transition_detail(str(FIXTURE), "client_001")
        self.assertNotIn("ACC3001", repr(d))
        self.assertNotIn("ACC3013", repr(d))


# --------------------------------------------------------------------------- #
# 23-28. Non-regression — the capability owns no existing metric
# --------------------------------------------------------------------------- #
class TestExistingOutputsUnchanged(unittest.TestCase):
    """This sprint is additive. Nothing above may move.

    The proof is a pure-reader one: every governed pipeline / forecast output is
    computed on the five-week pack, the new capability is exercised against the
    SAME prepared frames, and each output is recomputed and compared. A
    capability that changed a stock, a stage stock, an evolution point, a
    weighted expectation or a funnel conversion would fail here.
    """

    @classmethod
    def setUpClass(cls):
        if not HISTORY_FIXTURE.exists():
            raise unittest.SkipTest("five-week fixture not present")
        from mi_agent_api import evolution as ev
        from mi_agent_api import pipeline_contract as pc
        cls.ev, cls.pc = ev, pc
        cls.root = str(HISTORY_FIXTURE)

    def _prepared(self):
        extracts = self.pc.weekly_extract_inventory(self.root, "client_001")["extracts"]
        cur, cur_rep = self.pc.load_prepared_pipeline(extracts[-1])
        pri, pri_rep = self.pc.load_prepared_pipeline(extracts[-2])
        return cur, cur_rep, pri, pri_rep

    def _exercise_capability(self, cur, pri):
        d = md.build_stage_transition_detail(
            cur, pri, as_of_date="2026-05-29", comparison_date="2026-05-22",
            portfolio_id="client_001")
        self.assertTrue(d["available"], d.get("reason"))
        return d

    def test_live_pipeline_amount_and_case_count_are_unchanged(self):
        cur, rep, pri, _ = self._prepared()
        before = (rep.get("total_pipeline_amount"), int(len(cur)),
                  float(cur[md.MEASURE].sum()))
        self._exercise_capability(cur, pri)
        after = (rep.get("total_pipeline_amount"), int(len(cur)),
                 float(cur[md.MEASURE].sum()))
        self.assertEqual(before, after)

    def test_pipeline_stage_stock_is_unchanged(self):
        cur, rep, pri, _ = self._prepared()
        before = copy.deepcopy(rep.get("stage_counts"))
        before_frame = cur[md.STAGE].value_counts().to_dict()
        self._exercise_capability(cur, pri)
        self.assertEqual(rep.get("stage_counts"), before)
        self.assertEqual(cur[md.STAGE].value_counts().to_dict(), before_frame)

    def test_weighted_expected_pipeline_is_unchanged(self):
        cur, rep, pri, _ = self._prepared()
        before = (rep.get("weighted_expected_funded_amount"),
                  float(cur["weighted_expected_funded_amount"].fillna(0).sum()))
        self._exercise_capability(cur, pri)
        after = (rep.get("weighted_expected_funded_amount"),
                 float(cur["weighted_expected_funded_amount"].fillna(0).sum()))
        self.assertEqual(before, after)

    def test_pipeline_evolution_is_unchanged(self):
        before = self.ev.pipeline_evolution(self.root, "client_001")
        cur, _, pri, _ = self._prepared()
        self._exercise_capability(cur, pri)
        self.assertEqual(self.ev.pipeline_evolution(self.root, "client_001"), before)

    def test_funnel_and_conversion_are_unchanged(self):
        before = self.ev.pipeline_funnel_evolution(self.root, "client_001")
        cur, _, pri, _ = self._prepared()
        self._exercise_capability(cur, pri)
        self.assertEqual(
            self.ev.pipeline_funnel_evolution(self.root, "client_001"), before)

    def test_the_existing_net_movement_decomposition_is_unchanged(self):
        cur, _, pri, _ = self._prepared()
        before = md.build_movement_detail(
            md.DETAIL_PIPELINE, cur, pri, as_of_date="2026-05-29",
            comparison_date="2026-05-22", portfolio_id="client_001")
        self._exercise_capability(cur, pri)
        after = md.build_movement_detail(
            md.DETAIL_PIPELINE, cur, pri, as_of_date="2026-05-29",
            comparison_date="2026-05-22", portfolio_id="client_001")
        self.assertEqual(before, after)

    def test_the_capability_does_not_mutate_the_prepared_frames(self):
        """It reads the production frames; anything else would make every
        downstream number depend on whether a hover was opened."""
        cur, _, pri, _ = self._prepared()
        cur_before, pri_before = cur.copy(deep=True), pri.copy(deep=True)
        md.stage_transition_events(cur, pri)
        self._exercise_capability(cur, pri)
        pd.testing.assert_frame_equal(cur, cur_before)
        pd.testing.assert_frame_equal(pri, pri_before)

    def test_the_new_capability_is_a_distinct_detail_type(self):
        """It does not re-label, replace or reuse an existing detail type."""
        self.assertNotIn(md.DETAIL_STAGE_TRANSITION,
                         (md.DETAIL_PIPELINE, md.DETAIL_COMPLETIONS))
        self.assertEqual(md.build_stage_transition_detail(
            frame([{"id": "A", "amount": 1}]), frame([{"id": "A", "amount": 1}]),
            as_of_date="b", comparison_date="a",
            portfolio_id="X")["detail_type"], md.DETAIL_STAGE_TRANSITION)


# --------------------------------------------------------------------------- #
# The gross/net distinction this capability exists for
# --------------------------------------------------------------------------- #
class TestGrossIsNotNet(unittest.TestCase):

    def test_a_net_minus_two_is_resolved_into_its_gross_parts(self):
        """The brief's worked example: OFFER goes from 3 cases to 1. The net is
        -2 either way; only the gross classification says which happened."""
        pri = frame([{"id": "A", "amount": 100, "stage": "OFFER"},
                     {"id": "B", "amount": 100, "stage": "OFFER"},
                     {"id": "C", "amount": 100, "stage": "OFFER"}])
        cur = frame([{"id": "A", "amount": 100, "stage": "COMPLETED"},
                     {"id": "B", "amount": 100, "stage": "COMPLETED"},
                     {"id": "C", "amount": 100, "stage": "COMPLETED"},
                     {"id": "D", "amount": 100, "stage": "OFFER"}])
        offer = by_key(build(cur, pri)["reconciliation"]["by_stage"], "stage")["OFFER"]
        self.assertEqual(offer["opening_case_count"], 3)
        self.assertEqual(offer["closing_case_count"], 1)
        # THREE left and ONE arrived — not two and none, which the net -2
        # would equally have permitted.
        self.assertEqual(offer["transitions_out"], 3)
        self.assertEqual(offer["transitions_in"], 0)
        self.assertEqual(offer["new_arrivals"], 1)
        self.assertEqual(offer["departures"], 0)
        self.assertEqual(offer["count_reconciliation_residual"], 0)

    def test_the_payload_declares_itself_gross(self):
        d = build(frame([{"id": "A", "amount": 1}]), frame([{"id": "A", "amount": 1}]))
        self.assertEqual(d["methodology"]["movement_basis"], "gross")


if __name__ == "__main__":
    unittest.main()
