#!/usr/bin/env python3
"""tests/test_onboarding_capability_gate.py

Integration of the product-profile + capability-readiness + date-semantics layers
into the LIVE blocking gates:

  * Gate 4 / 28c human decision queue + 34 target-first decisions
    (target_coverage.run_target_first_coverage), and
  * 07 gap questions severity (gap_analyzer).

Proves the required behaviour for mode=mi_only + equity_release:
  1. Gate 4 produces no blocking decisions for maturity_date / amortisation_type /
     risk fields / SPV+acquisition fields / originator / pipeline_snapshot_date
     when they are not required for base MI (and these DO block without a profile).
  2. 07 gap questions do not leave product-profile not_applicable/defaulted/derived
     core fields blocking.
  3. pipeline_snapshot_date is inferred from a pipeline role/date folder and is not
     confused with the funded reporting_date.
  4. Missing risk fields mark risk_migration unavailable while base_mi stays
     promotable.
  5. regulatory_mi requirements remain strict (profile never relaxes them).
"""

from __future__ import annotations

import sys
import tempfile
import unittest
import warnings
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from engine.onboarding_agent import target_coverage as tcov
from engine.onboarding_agent import gap_analyzer as ga
from engine.onboarding_agent import product_profile as pp
from engine.onboarding_agent import capability_readiness as cr
from engine.onboarding_agent import date_semantics as ds

EQUITY_CTX = {
    "asset_class": "equity_release_mortgage", "product_type": "lifetime_mortgage",
    "jurisdiction": "UK", "confidence": 0.9, "asset_signal_strength": 6,
    "reporting_regime": "mi_only",
    "rationale": "equity release lifetime mortgage roll-up nneg drawdown",
}
GENERIC_CTX = {  # no asset evidence -> profile must NOT apply
    "asset_class": "equity_release_mortgage", "asset_signal_strength": 0,
    "reporting_regime": "mi_only",
}

# Fields that must NOT be base-MI blockers under the equity-release profile.
NON_BASE_FIELDS = {
    "maturity_date", "amortisation_type", "funded_status",
    "current_outstanding_balance", "ifrs9_stage", "probability_of_default",
    "loss_given_default", "exposure_at_default", "internal_risk_grade",
    "spv_id", "acquisition_date", "acquired_portfolio_id",
    "pipeline_snapshot_date", "originator_name",
}


def _ev(col, registry_field="", *, null_rate=0.0, distinct=10, dtype="string"):
    return {
        "source_file": "loan_tape.csv", "source_sheet": "", "source_column": col,
        "normalized_column": col.lower().replace(" ", "_"),
        "domain_guess": "funded_loan", "file_domain_guess": "funded_loan",
        "null_rate": null_rate, "distinct_count": distinct, "data_type_guess": dtype,
        "candidate_existing_registry_fields": registry_field,
        "candidate_existing_pipeline_contract_fields": "",
        "candidate_alias_matches": "", "candidate_semantic_alignment_matches": "",
        "known_client_memory_matches": "", "candidate_value_profile_matches": "",
    }


# Base-MI source evidence (so base_mi is satisfiable; non-base fields are the gaps).
BASE_EVIDENCE = [
    _ev("Current Balance", "current_principal_balance", dtype="decimal"),
    _ev("Interest Rate", "current_interest_rate", dtype="decimal"),
    _ev("Origination Date", "origination_date", dtype="date"),
    _ev("Current Valuation", "current_valuation_amount", dtype="decimal"),
    _ev("Reporting Date", "reporting_date", dtype="date"),
]


def _run_coverage(context):
    out = Path(tempfile.mkdtemp(prefix="capgate_"))
    warnings.simplefilter("ignore")
    res = tcov.run_target_first_coverage(
        mode="mi_only", context=context, evidence_rows=BASE_EVIDENCE,
        resolved_rows=[], output_dir=out, client_id="t", run_id="r1")
    return res, out


def _blocking_fields(decision_queue):
    return {d["target_field"] for d in decision_queue if d.get("blocking")}


# --------------------------------------------------------------------------- #
# 1 — Gate 4 / 28c: profile removes non-base blockers (before/after)
# --------------------------------------------------------------------------- #
class TestGate4CapabilityScope(unittest.TestCase):
    def test_profile_applied_removes_non_base_blockers(self):
        res, _ = _run_coverage(EQUITY_CTX)
        self.assertTrue(res["product_profile_scope"]["product_profile"]["applied"])
        blocking = _blocking_fields(res["decision_queue"])
        leaked = NON_BASE_FIELDS & blocking
        self.assertEqual(leaked, set(),
                         f"these should not block base MI: {sorted(leaked)}")

    def test_before_after_blocker_drop(self):
        generic, _ = _run_coverage(GENERIC_CTX)
        equity, _ = _run_coverage(EQUITY_CTX)
        self.assertFalse(generic["product_profile_scope"]["product_profile"]["applied"])
        before = len(_blocking_fields(generic["decision_queue"]))
        after = len(_blocking_fields(equity["decision_queue"]))
        # Materially fewer Gate-4 blockers once the profile is applied.
        self.assertLess(after, before)
        # And specifically: generic DOES block several of the non-base fields.
        self.assertTrue(NON_BASE_FIELDS & _blocking_fields(generic["decision_queue"]))

    def test_scope_changes_are_audited(self):
        res, out = _run_coverage(EQUITY_CTX)
        scope = res["product_profile_scope"]
        self.assertTrue(scope["capability_scope_applied"])
        self.assertTrue((out / "28d_product_profile_scope.json").exists())
        # Each change records the rationale / capability.
        for ch in scope["capability_scope_changes"]:
            self.assertIn("rationale", ch)
            self.assertEqual(ch["new_required_status"], "optional")

    def test_base_fields_still_resolve(self):
        # The genuine base-MI fields are mapped from source (not blocking).
        res, _ = _run_coverage(EQUITY_CTX)
        blocking = _blocking_fields(res["decision_queue"])
        for f in ("current_principal_balance", "current_interest_rate",
                  "origination_date", "current_valuation_amount", "reporting_date"):
            self.assertNotIn(f, blocking)


# --------------------------------------------------------------------------- #
# 2 — 07 gap questions: profile demotes not_applicable/defaulted/derived
# --------------------------------------------------------------------------- #
class _FakeFieldScope:
    def __init__(self, core):
        self.included_fields = set(core)
        self.core_canonical_fields = set(core)
        self.blocking_fields = set(core)        # all core block by default
        self.excluded_fields = set()
        self.regulatory_fields = set()


class TestGapSeverity(unittest.TestCase):
    def setUp(self):
        self.core = ["current_principal_balance", "origination_date",
                     "maturity_date", "amortisation_type", "funded_status",
                     "ifrs9_stage", "spv_id"]
        self.fs = _FakeFieldScope(self.core)

    def _severities(self, resolved):
        qs = ga._missing_core_field_questions(self.fs, [], 0, resolved)
        return {q.subject: q.severity for q in qs}

    def test_without_profile_all_core_block(self):
        sev = self._severities(None)
        self.assertEqual(sev["maturity_date"], "blocking")
        self.assertEqual(sev["amortisation_type"], "blocking")

    def test_with_profile_non_base_demoted(self):
        resolved = pp.resolve_product_profile(EQUITY_CTX, profiles_path=str(
            _REPO_ROOT / "config" / "asset" / "product_profiles.yaml"))
        self.assertTrue(resolved.applied)
        sev = self._severities(resolved)
        # Profile-non-blocking fields demoted to visible (high), not blocking.
        for f in ("maturity_date", "amortisation_type", "funded_status",
                  "ifrs9_stage", "spv_id"):
            self.assertEqual(sev[f], "high", f)
        # Genuine base fields still block.
        self.assertEqual(sev["current_principal_balance"], "blocking")
        self.assertEqual(sev["origination_date"], "blocking")


# --------------------------------------------------------------------------- #
# 3 — pipeline_snapshot_date inferred from pipeline folder, not funded date
# --------------------------------------------------------------------------- #
class TestPipelineSnapshotInference(unittest.TestCase):
    def test_pipeline_folder_not_confused_with_funded(self):
        arts = [
            {"file_name": "loan.csv", "role": "current_loan_report",
             "folder": "input/funded/2025-11-30/loan.csv"},
            {"file_name": "kfi.csv", "role": "pipeline_report",
             "folder": "input/pipeline/2025-12-01/kfi.csv"},
        ]
        dated = {a.role: a for a in ds.assign_artefact_dates(arts)}
        self.assertEqual(dated["current_loan_report"].canonical_field, ds.REPORTING_DATE)
        self.assertEqual(dated["current_loan_report"].date, "2025-11-30")
        self.assertEqual(dated["pipeline_report"].canonical_field,
                         ds.PIPELINE_SNAPSHOT_DATE)
        self.assertEqual(dated["pipeline_report"].date, "2025-12-01")
        res = ds.validate_date_consistency(ds.assign_artefact_dates(arts))
        self.assertFalse(res["blocking"])
        self.assertEqual(res["funded_reporting_date"], "2025-11-30")
        self.assertEqual(res["pipeline_snapshot_date"], "2025-12-01")


# --------------------------------------------------------------------------- #
# 4 — missing risk fields: risk_migration unavailable, base_mi promotable
# --------------------------------------------------------------------------- #
class TestRiskUnavailableBasePromotable(unittest.TestCase):
    def test_risk_unavailable_base_promotable(self):
        prof = pp.load_product_profiles(str(
            _REPO_ROOT / "config" / "asset" / "product_profiles.yaml"))[
            "equity_release_lifetime_mortgage"]
        satisfied = ["loan_identifier", "current_principal_balance",
                     "current_interest_rate", "origination_date",
                     "current_valuation_amount", "reporting_date"]
        readiness = cr.compute_capability_readiness(
            profile=prof, satisfied_fields=satisfied,
            artefact_roles=["current_loan_report"])
        self.assertEqual(readiness["risk_migration"].status, cr.UNAVAILABLE)
        decision = cr.promotion_decision(readiness, mode="mi_only")
        self.assertTrue(decision["promotable"])
        self.assertIn("risk_migration", decision["unavailable_capabilities"])


# --------------------------------------------------------------------------- #
# 5 — regulatory_mi is unchanged (profile never relaxes it)
# --------------------------------------------------------------------------- #
class TestRegulatoryUnchanged(unittest.TestCase):
    def test_capability_scope_noop_for_regulatory(self):
        _cid, _csrc, fields = tcov.load_target_contract("regulatory_mi", EQUITY_CTX)
        before = [dict(f) for f in fields]
        resolved = pp.resolve_product_profile(EQUITY_CTX, profiles_path=str(
            _REPO_ROOT / "config" / "asset" / "product_profiles.yaml"))
        changes = tcov.apply_mi_capability_scope(fields, resolved, "regulatory_mi")
        self.assertEqual(changes, [])
        # required_status of regulatory fields is untouched.
        self.assertEqual([f["required_status"] for f in fields],
                         [f["required_status"] for f in before])

    def test_overlay_not_applied_for_regulatory(self):
        overlay = tcov.load_mi_applicability_overlay(
            "regulatory_mi", EQUITY_CTX,
            resolved_profile=pp.resolve_product_profile(EQUITY_CTX, profiles_path=str(
                _REPO_ROOT / "config" / "asset" / "product_profiles.yaml")))
        self.assertEqual(overlay, {})  # regulatory contract -> no MI overlay


if __name__ == "__main__":
    unittest.main(verbosity=2)
