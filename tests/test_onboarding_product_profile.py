#!/usr/bin/env python3
"""tests/test_onboarding_product_profile.py

Asset-agnostic, config-driven product-profile hardening for onboarding.

Proves the design objectives:
  * generic (no-profile) mode stays stricter where appropriate;
  * an equity_release profile is base-MI ready without maturity_date /
    amortisation_type;
  * current_principal_balance satisfies current_outstanding_balance (proxy);
  * funded_status derives from the funded loan-extract artefact role;
  * pipeline_stage comes from the pipeline artefact, not the loan extract;
  * months_on_book derives from origination_date + reporting_date;
  * number_of_borrowers is NEVER derived from a unique loan_id count;
  * missing risk fields do not block base MI but disable risk migration;
  * profile resolution: explicit trusted, high-confidence applied, medium asks;
  * the coverage classifier treats profile fields as non-blocking (before/after
    blocker count);
  * regulatory readiness is never relaxed by a product profile.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from engine.onboarding_agent import product_profile as pp
from engine.onboarding_agent import capability_readiness as cr
from engine.onboarding_agent import target_coverage as tc

PROFILES = str(_REPO_ROOT / "config" / "asset" / "product_profiles.yaml")
ERE_PROFILE_ID = "equity_release_lifetime_mortgage"


def _profiles():
    return pp.load_product_profiles(PROFILES)


def _ere_profile():
    return _profiles()[ERE_PROFILE_ID]


# --------------------------------------------------------------------------- #
# Profile loading / resolution
# --------------------------------------------------------------------------- #
class TestProfileResolution(unittest.TestCase):
    def test_profiles_load(self):
        profs = _profiles()
        self.assertIn(ERE_PROFILE_ID, profs)
        self.assertTrue(_ere_profile().field_policies)

    def test_explicit_profile_trusted(self):
        res = pp.resolve_product_profile(
            {"asset_class": "anything"}, explicit_profile_id=ERE_PROFILE_ID,
            profiles_path=PROFILES)
        self.assertEqual(res.decision, pp.DECISION_EXPLICIT)
        self.assertTrue(res.applied)
        self.assertFalse(res.needs_confirmation)
        self.assertEqual(res.confidence, 1.0)

    def test_explicit_unknown_profile_surfaced_not_guessed(self):
        res = pp.resolve_product_profile(
            {"asset_class": "equity_release_mortgage"},
            explicit_profile_id="does_not_exist", profiles_path=PROFILES)
        self.assertFalse(res.applied)
        self.assertTrue(res.needs_confirmation)

    def test_high_confidence_detection_applies(self):
        ctx = {
            "asset_class": "equity_release_mortgage",
            "product_type": "lifetime_mortgage",
            "confidence": 0.95,
            "rationale": "equity release lifetime mortgage roll-up nneg",
            "supporting_evidence": ["drawdown plan"],
        }
        res = pp.resolve_product_profile(ctx, profiles_path=PROFILES)
        self.assertEqual(res.profile_id, ERE_PROFILE_ID)
        self.assertEqual(res.decision, pp.DECISION_DETECTED)
        self.assertTrue(res.applied)
        self.assertGreaterEqual(res.confidence, _ere_profile().apply_confidence)
        self.assertTrue(res.evidence)

    def test_medium_confidence_asks_for_confirmation(self):
        # asset_class alone (0.6) lands in the confirm band, below apply (0.8).
        ctx = {"asset_class": "equity_release_mortgage"}
        res = pp.resolve_product_profile(ctx, profiles_path=PROFILES)
        self.assertEqual(res.profile_id, ERE_PROFILE_ID)
        self.assertEqual(res.decision, pp.DECISION_NEEDS_CONFIRMATION)
        self.assertFalse(res.applied)
        self.assertTrue(res.needs_confirmation)

    def test_no_match_generic_behaviour(self):
        res = pp.resolve_product_profile(
            {"asset_class": "auto_loan", "product_type": "hire_purchase"},
            profiles_path=PROFILES)
        self.assertEqual(res.decision, pp.DECISION_NONE)
        self.assertFalse(res.applied)


# --------------------------------------------------------------------------- #
# Field policies: non-blocking, defaults, derivations
# --------------------------------------------------------------------------- #
class TestFieldPolicies(unittest.TestCase):
    def setUp(self):
        self.prof = _ere_profile()

    def test_maturity_date_not_applicable(self):
        self.assertEqual(self.prof.base_mi_policy("maturity_date"),
                         pp.POLICY_NOT_APPLICABLE)
        self.assertTrue(self.prof.is_non_blocking_for_base_mi("maturity_date"))
        rec = self.prof.not_applicable_record("maturity_date")
        self.assertIsNotNone(rec)
        self.assertTrue(rec.rationale)

    def test_amortisation_type_defaulted_to_canonical_enum(self):
        self.assertEqual(self.prof.base_mi_policy("amortisation_type"),
                         pp.POLICY_DEFAULTED)
        rec = self.prof.default_record("amortisation_type")
        self.assertIsNotNone(rec)
        # canonical enum value for capitalising / roll-up / no scheduled amortisation
        self.assertEqual(rec.value, "OTHR")
        self.assertTrue(rec.rationale)
        self.assertEqual(rec.profile_id, ERE_PROFILE_ID)

    def test_current_outstanding_balance_proxy_from_principal(self):
        row = {"current_principal_balance": "123456.78"}
        rec = self.prof.derive_current_outstanding_balance(row)
        self.assertIsNotNone(rec)
        self.assertEqual(rec.value, "123456.78")
        self.assertEqual(rec.source, "current_principal_balance")
        self.assertEqual(rec.method, "from_field")

    def test_current_outstanding_balance_not_fabricated(self):
        # No current-balance source -> no record (never invented).
        self.assertIsNone(self.prof.derive_current_outstanding_balance({"foo": "1"}))

    def test_funded_status_from_loan_extract_role(self):
        rec = self.prof.derive_funded_status("current_loan_report")
        self.assertIsNotNone(rec)
        self.assertEqual(rec.value, "funded")
        self.assertEqual(rec.method, "from_artefact_role")
        self.assertIn("artefact_role", rec.source)

    def test_funded_status_source_status_can_contradict(self):
        rec = self.prof.derive_funded_status("current_loan_report",
                                             source_status="Redeemed")
        self.assertIsNotNone(rec)
        self.assertEqual(rec.method, "from_source_status")

    def test_funded_status_not_derived_from_pipeline_role(self):
        # A pipeline artefact must not yield funded status from the role.
        self.assertIsNone(self.prof.derive_funded_status("pipeline_report"))

    def test_months_on_book_from_dates(self):
        rec = self.prof.derive_months_on_book("2020-01-15", "2025-10-31")
        self.assertIsNotNone(rec)
        self.assertEqual(rec.value, 69)   # 5y 9m + partial day handling
        self.assertEqual(rec.method, "from_dates")

    def test_months_on_book_requires_both_dates(self):
        self.assertIsNone(self.prof.derive_months_on_book("", "2025-10-31"))

    def test_number_of_borrowers_from_borrower_fields(self):
        row = {"borrower_1_name": "A", "borrower_2_name": "B", "loan_identifier": "L1"}
        rec = self.prof.derive_number_of_borrowers(row)
        self.assertIsNotNone(rec)
        self.assertEqual(rec.value, 2)
        self.assertEqual(rec.method, "from_borrower_fields")

    def test_number_of_borrowers_never_from_loan_id(self):
        # Only loan-id-like fields present -> must NOT count them as borrowers.
        row = {"loan_identifier": "L1", "account_number": "A1", "policy_number": "P1"}
        self.assertIsNone(self.prof.derive_number_of_borrowers(row))

    def test_risk_fields_non_blocking_but_capability_gated(self):
        for f in ("ifrs9_stage", "probability_of_default", "loss_given_default",
                  "exposure_at_default", "internal_risk_grade"):
            self.assertTrue(self.prof.is_non_blocking_for_base_mi(f), f)
        self.assertIn("risk_monitor",
                      self.prof.required_for_capabilities("probability_of_default"))
        self.assertIn("risk_migration",
                      self.prof.required_for_capabilities("ifrs9_stage"))

    def test_segmentation_keys_non_blocking(self):
        for f in ("spv_id", "acquired_portfolio_id", "acquisition_date"):
            self.assertTrue(self.prof.is_non_blocking_for_base_mi(f), f)
        self.assertIn("spv_segmentation", self.prof.required_for_capabilities("spv_id"))
        self.assertIn("mna_segmentation",
                      self.prof.required_for_capabilities("acquired_portfolio_id"))


# --------------------------------------------------------------------------- #
# Capability-based readiness + promotion
# --------------------------------------------------------------------------- #
class TestCapabilityReadiness(unittest.TestCase):
    def setUp(self):
        self.prof = _ere_profile()
        # Base-MI fields satisfied, NO maturity_date / amortisation_type, NO risk fields.
        self.base_satisfied = [
            "loan_identifier", "current_principal_balance", "current_interest_rate",
            "origination_date", "current_valuation_amount", "reporting_date",
        ]

    def test_base_mi_ready_without_maturity_or_amortisation(self):
        readiness = cr.compute_capability_readiness(
            profile=self.prof, satisfied_fields=self.base_satisfied,
            artefact_roles=["current_loan_report"])
        base = readiness["base_mi"]
        self.assertTrue(base.ready, base.rationale)
        self.assertEqual(base.status, cr.READY)

    def test_current_principal_satisfies_outstanding_equivalence(self):
        # Only current_principal_balance present; equivalence group satisfies the
        # "current balance" requirement even if the contract names outstanding.
        readiness = cr.compute_capability_readiness(
            profile=self.prof, satisfied_fields=self.base_satisfied)
        self.assertTrue(readiness["base_mi"].ready)

    def test_missing_risk_fields_disable_risk_capabilities(self):
        readiness = cr.compute_capability_readiness(
            profile=self.prof, satisfied_fields=self.base_satisfied)
        self.assertEqual(readiness["risk_migration"].status, cr.UNAVAILABLE)
        self.assertEqual(readiness["risk_monitor"].status, cr.UNAVAILABLE)
        self.assertFalse(readiness["risk_migration"].ready)

    def test_risk_migration_ready_when_risk_field_present(self):
        readiness = cr.compute_capability_readiness(
            profile=self.prof,
            satisfied_fields=self.base_satisfied + ["ifrs9_stage"])
        self.assertEqual(readiness["risk_migration"].status, cr.READY)

    def test_pipeline_mi_requires_pipeline_artefact(self):
        # No pipeline artefact -> pipeline_mi unavailable (not a blocker).
        readiness = cr.compute_capability_readiness(
            profile=self.prof, satisfied_fields=self.base_satisfied,
            artefact_roles=["current_loan_report"])
        self.assertEqual(readiness["pipeline_mi"].status, cr.UNAVAILABLE)
        # With a pipeline artefact AND pipeline_stage -> ready.
        readiness2 = cr.compute_capability_readiness(
            profile=self.prof,
            satisfied_fields=self.base_satisfied + ["pipeline_stage"],
            artefact_roles=["current_loan_report", "pipeline_report"])
        self.assertEqual(readiness2["pipeline_mi"].status, cr.READY)

    def test_promotion_mi_only_without_risk_migration(self):
        readiness = cr.compute_capability_readiness(
            profile=self.prof, satisfied_fields=self.base_satisfied,
            artefact_roles=["current_loan_report"])
        decision = cr.promotion_decision(readiness, mode="mi_only")
        self.assertTrue(decision["promotable"], decision["rationale"])
        self.assertIn("base_mi", decision["ready_capabilities"])
        self.assertIn("risk_migration", decision["unavailable_capabilities"])
        self.assertEqual(decision["blocking_capabilities"], [])

    def test_promotion_blocked_when_base_mi_incomplete(self):
        readiness = cr.compute_capability_readiness(
            profile=self.prof,
            satisfied_fields=["loan_identifier"],  # missing core base-MI fields
            artefact_roles=["current_loan_report"])
        decision = cr.promotion_decision(readiness, mode="mi_only")
        self.assertFalse(decision["promotable"])
        self.assertIn("base_mi", decision["blocking_capabilities"])

    def test_regulatory_not_relaxed_by_profile(self):
        # No regulatory_reporting contract is relaxed; promotion in regulatory_mi
        # requires it explicitly (and the profile never marks it ready).
        readiness = cr.compute_capability_readiness(
            profile=self.prof, satisfied_fields=self.base_satisfied,
            artefact_roles=["current_loan_report"])
        decision = cr.promotion_decision(readiness, mode="regulatory_mi")
        self.assertFalse(decision["promotable"])


# --------------------------------------------------------------------------- #
# Coverage-overlay integration: before/after blocker count
# --------------------------------------------------------------------------- #
class TestCoverageOverlayIntegration(unittest.TestCase):
    def _tf(self, field):
        return {
            "target_field": field, "match_field": field, "synonyms": [],
            "required_status": "mandatory", "applicability_status": "applicable",
            "default_value": "", "derived": "", "derivation_rule": "",
            "configured_value_source": "",
        }

    def test_required_field_blocks_without_profile(self):
        # Generic: a required field with no source/derivation/default blocks.
        out = tc._classify(self._tf("maturity_date"), [], overlay_rule=None)
        self.assertEqual(out["coverage_status"], tc.MISSING_REQUIRED)
        self.assertTrue(out["blocking"])

    def test_profile_overlay_makes_field_non_blocking(self):
        prof = _ere_profile()
        rules = pp.profile_overlay_rules(prof)
        self.assertIn("maturity_date", rules)
        out = tc._classify(self._tf("maturity_date"), [],
                           overlay_rule=rules["maturity_date"])
        self.assertFalse(out["blocking"])
        self.assertNotEqual(out["coverage_status"], tc.MISSING_REQUIRED)

    def test_overlay_loader_merges_applied_profile(self):
        ctx = {"asset_class": "equity_release_mortgage",
               "jurisdiction": "DE",  # outside static UK overlay scope
               "product_type": "lifetime_mortgage", "confidence": 0.95,
               "rationale": "equity release lifetime mortgage roll-up"}
        res = pp.resolve_product_profile(ctx, profiles_path=PROFILES)
        self.assertTrue(res.applied)
        overlay = tc.load_mi_applicability_overlay(
            "mi_only", ctx, resolved_profile=res)
        # Profile gap-fills amortisation_type even though the UK YAML overlay
        # (scoped to UK) does not apply here.
        self.assertIn("amortisation_type", overlay)
        self.assertFalse(overlay["amortisation_type"].get("blocking", False))

    def test_unapplied_profile_contributes_nothing(self):
        ctx = {"asset_class": "equity_release_mortgage", "jurisdiction": "DE"}
        res = pp.resolve_product_profile(ctx, profiles_path=PROFILES)
        self.assertFalse(res.applied)  # confirm band only
        overlay = tc.load_mi_applicability_overlay(
            "mi_only", ctx, resolved_profile=res)
        self.assertNotIn("amortisation_type", overlay)

    def test_before_after_blocker_count(self):
        """Quantify the blocker reduction the applied profile delivers."""
        fields = ["maturity_date", "amortisation_type", "funded_status",
                  "current_outstanding_balance", "ifrs9_stage", "spv_id"]

        def count_blockers(overlay):
            n = 0
            for f in fields:
                rule = overlay.get(f) if overlay else None
                if tc._classify(self._tf(f), [], overlay_rule=rule)["blocking"]:
                    n += 1
            return n

        before = count_blockers(None)
        prof = _ere_profile()
        overlay = pp.profile_overlay_rules(prof)
        after = count_blockers(overlay)
        self.assertEqual(before, len(fields))   # all block without a profile
        self.assertEqual(after, 0)              # none block with the profile


if __name__ == "__main__":
    unittest.main(verbosity=2)
