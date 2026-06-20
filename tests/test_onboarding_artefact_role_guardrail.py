#!/usr/bin/env python3
"""tests/test_onboarding_artefact_role_guardrail.py

Narrow artefact-role guardrail: a pipeline artefact must not auto-fill funded-book
base-MI target fields (it may stay a lower-priority alternative).

Proves:
  * funded Loan Interest Rate is selected over Pipeline product rate;
  * funded Current Outstanding Balance sources the balance; pipeline loan amount
    is excluded and the equity proxy never derives principal from pipeline;
  * funded Policy Completion Date sources origination_date over pipeline dates;
  * a pipeline-only candidate for a funded field is NOT auto-selected;
  * pipeline columns still map to pipeline-specific targets;
  * the role preference / exclusion is recorded for audit.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from engine.onboarding_agent import target_coverage as tcov
from engine.onboarding_agent import product_profile as pp

ROLES = {"LoanExtract One.csv": "current_loan_report", "Pipeline.csv": "pipeline_report"}


def _tf(field, *, required="mandatory", domain="loan", synonyms=None):
    return {
        "target_field": field, "match_field": field, "esma_code": "",
        "projected_source_field": "", "target_domain": domain, "target_label": field,
        "required_status": required, "applicability_status": "applicable",
        "synonyms": synonyms or [], "derived": False, "derivation_rule": "",
        "default_rule": "", "default_value": "", "default_rule_source": "",
        "default_reason": "", "nd_allowed": [], "configured_value_source": "",
    }


def _ev(source_file, col, canonical):
    return {
        "source_file": source_file, "source_sheet": "", "source_column": col,
        "normalized_column": col.lower().replace(" ", "_"),
        "domain_guess": "loan", "file_domain_guess": "unknown",
        "null_rate": 0.0, "distinct_count": 10, "data_type_guess": "string",
        "candidate_existing_registry_fields": canonical,
        "candidate_existing_pipeline_contract_fields": "",
        "candidate_alias_matches": "", "candidate_semantic_alignment_matches": "",
        "known_client_memory_matches": "", "candidate_value_profile_matches": "",
    }


def _cov(target_fields, evidence):
    rows, _ = tcov.build_target_coverage(
        "mi_only", {}, "mi_semantics_field_registry", "src", target_fields,
        evidence, resolved_rows=[], artefact_roles=ROLES)
    return {r["target_field"]: r for r in rows}


class TestRoleGuardrail(unittest.TestCase):
    def test_funded_interest_rate_wins_over_pipeline(self):
        ev = [
            _ev("LoanExtract One.csv", "Loan Interest Rate", "current_interest_rate"),
            _ev("Pipeline.csv", "product rate", "current_interest_rate"),
        ]
        row = _cov([_tf("current_interest_rate")], ev)["current_interest_rate"]
        self.assertEqual(row["selected_source_file"], "LoanExtract One.csv")
        self.assertEqual(row["artefact_role_selected"], "funded")
        # Pipeline survives only as a lower-priority alternative.
        self.assertIn("Pipeline.csv", row["alternative_source_candidates"])
        self.assertIn("(pipeline)", row["alternative_source_candidates"])
        self.assertTrue(row["role_preference_note"])

    def test_funded_origination_date_wins_over_pipeline_dates(self):
        ev = [
            _ev("LoanExtract One.csv", "Policy Completion Date", "origination_date"),
            _ev("Pipeline.csv", "application date", "origination_date"),
            _ev("Pipeline.csv", "date funds released", "origination_date"),
        ]
        row = _cov([_tf("origination_date")], ev)["origination_date"]
        self.assertEqual(row["selected_source_file"], "LoanExtract One.csv")
        self.assertEqual(row["artefact_role_selected"], "funded")

    def test_pipeline_only_funded_field_not_auto_selected(self):
        # Only a pipeline candidate exists for a funded-book field -> not selected.
        ev = [_ev("Pipeline.csv", "product rate", "current_interest_rate")]
        row = _cov([_tf("current_interest_rate")], ev)["current_interest_rate"]
        self.assertNotEqual(row["coverage_status"], tcov.SOURCE_MAPPED)
        self.assertNotEqual(row["coverage_status"], tcov.SOURCE_MAPPED_ALT)
        self.assertEqual(row["selected_source_file"], "")
        # Pipeline candidate is still visible + the exclusion is recorded.
        self.assertIn("(pipeline)", row["alternative_source_candidates"])
        self.assertIn("excluded", row["role_preference_note"])

    def test_pipeline_target_still_maps_from_pipeline(self):
        # A pipeline-specific field is NOT protected -> pipeline source is fine.
        ev = [_ev("Pipeline.csv", "pipeline stage", "pipeline_stage")]
        row = _cov([_tf("pipeline_stage", domain="pipeline")], ev)["pipeline_stage"]
        self.assertEqual(row["selected_source_file"], "Pipeline.csv")
        self.assertIn(row["coverage_status"], (tcov.SOURCE_MAPPED, tcov.SOURCE_MAPPED_ALT))

    def test_guardrail_noop_without_roles(self):
        ev = [_ev("Pipeline.csv", "product rate", "current_interest_rate")]
        rows, _ = tcov.build_target_coverage(
            "mi_only", {}, "mi", "src", [_tf("current_interest_rate")], ev,
            resolved_rows=[], artefact_roles=None)
        # No role map -> legacy behaviour (pipeline can map).
        self.assertEqual(rows[0]["selected_source_file"], "Pipeline.csv")


class TestProxyRespectsGuardrail(unittest.TestCase):
    def _resolved(self):
        return pp.resolve_product_profile(
            {"asset_class": "equity_release_mortgage", "product_type": "lifetime_mortgage",
             "confidence": 0.9, "asset_signal_strength": 6,
             "rationale": "equity release lifetime mortgage roll-up"},
            profiles_path=str(_REPO_ROOT / "config" / "asset" / "product_profiles.yaml"))

    def test_principal_not_derived_from_pipeline_balance(self):
        # Pipeline 'loan amount' is the only balance candidate -> excluded -> the
        # equity proxy must NOT derive current_principal_balance from it.
        ev = [_ev("Pipeline.csv", "loan amount", "current_outstanding_balance")]
        rows, _ = tcov.build_target_coverage(
            "mi_only", {}, "mi", "src",
            [_tf("current_outstanding_balance"), _tf("current_principal_balance")],
            ev, resolved_rows=[], artefact_roles=ROLES)
        changes = tcov.apply_profile_proxy_derivations(rows, self._resolved(), run_id="")
        by = {r["target_field"]: r for r in rows}
        self.assertNotEqual(by["current_outstanding_balance"]["coverage_status"],
                            tcov.SOURCE_MAPPED)
        self.assertFalse(any(c["target_field"] == "current_principal_balance"
                             for c in changes))

    def test_principal_derived_from_funded_balance(self):
        ev = [_ev("LoanExtract One.csv", "Total OSBalance", "current_outstanding_balance")]
        rows, _ = tcov.build_target_coverage(
            "mi_only", {}, "mi", "src",
            [_tf("current_outstanding_balance"), _tf("current_principal_balance")],
            ev, resolved_rows=[], artefact_roles=ROLES)
        by = {r["target_field"]: r for r in rows}
        self.assertEqual(by["current_outstanding_balance"]["coverage_status"],
                         tcov.SOURCE_MAPPED)
        self.assertEqual(by["current_outstanding_balance"]["artefact_role_selected"], "funded")
        changes = tcov.apply_profile_proxy_derivations(rows, self._resolved(), run_id="")
        self.assertTrue(any(c["target_field"] == "current_principal_balance"
                            for c in changes))


if __name__ == "__main__":
    unittest.main(verbosity=2)
