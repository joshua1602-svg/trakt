#!/usr/bin/env python3
"""
tests/test_onboarding_regulatory_preference.py

PART 1/2 — the regulatory-preference ambiguity rule and its artefacts.

The core rule is exercised as a pure function over constructed candidate lists
(so it is deterministic and independent of registry token scoring); the
artefact writer and review pack are exercised end to end.
"""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from engine.onboarding_agent.ambiguity_rule import (
    REASON_MI_ONLY_CORE,
    REASON_MI_ONLY_DIVERTED,
    REASON_MNA_DD,
    REASON_REGULATORY_MI,
    Candidate,
    load_ambiguity_delta,
    resolve_regulatory_preference,
)
from engine.onboarding_agent.onboarding_models import MappingAmbiguity, OnboardingProject
from engine.onboarding_agent.onboarding_orchestrator import _write_artifacts
from engine.onboarding_agent.review_pack_builder import build_review_pack


# Example 1 candidates (original principal): regulatory core vs analytics.
def _example1():
    return [
        Candidate("original_principal_balance", 0.86, "regulatory", True),
        Candidate("loan_amount_analytics", 0.84, "analytics", False),
    ]


# Example 2 candidates (employment type): regulatory non-core vs analytics.
def _example2():
    return [
        Candidate("employment_status", 0.76, "regulatory", False),
        Candidate("borrower_segment", 0.73, "analytics", False),
    ]


class TestRuleLogic(unittest.TestCase):
    def test_1_regulatory_mi_selects_regulatory_and_reviews(self):
        res = resolve_regulatory_preference(_example1(), mode="regulatory_mi", delta_threshold=0.10)
        self.assertIsNotNone(res)
        self.assertEqual(res.selected.field, "original_principal_balance")
        self.assertTrue(res.review_required)
        self.assertEqual(res.reason, REASON_REGULATORY_MI)
        self.assertEqual(res.alternative.field, "loan_amount_analytics")

    def test_2_mna_dd_selects_regulatory_nonblocking_review(self):
        res = resolve_regulatory_preference(_example2(), mode="mna_dd", delta_threshold=0.10)
        self.assertIsNotNone(res)
        self.assertEqual(res.selected.field, "employment_status")
        self.assertTrue(res.review_required)
        self.assertEqual(res.reason, REASON_MNA_DD)

    def test_3_mi_only_does_not_select_regulatory_noncore(self):
        res = resolve_regulatory_preference(_example2(), mode="mi_only", delta_threshold=0.10)
        self.assertIsNotNone(res)
        # The regulatory non-core field is NOT selected; analytics is.
        self.assertEqual(res.selected.field, "borrower_segment")
        self.assertEqual(res.selected.category, "analytics")
        self.assertNotEqual(res.selected.field, "employment_status")
        self.assertTrue(res.divert_regulatory_to_out_of_scope)
        self.assertEqual(res.diverted_field.field, "employment_status")
        self.assertEqual(res.reason, REASON_MI_ONLY_DIVERTED)

    def test_4_mi_only_can_select_regulatory_core(self):
        res = resolve_regulatory_preference(_example1(), mode="mi_only", delta_threshold=0.10)
        self.assertIsNotNone(res)
        # Regulatory CORE field may be selected even in MI-only.
        self.assertEqual(res.selected.field, "original_principal_balance")
        self.assertTrue(res.selected.core_canonical)
        self.assertFalse(res.divert_regulatory_to_out_of_scope)
        self.assertEqual(res.reason, REASON_MI_ONLY_CORE)

    def test_warehouse_prefers_operational_unless_enabled(self):
        # Non-core regulatory vs analytics, regulatory reporting OFF -> prefer non-reg.
        res = resolve_regulatory_preference(
            _example2(), mode="warehouse_securitisation", delta_threshold=0.10,
            regulatory_reporting_enabled=False)
        self.assertEqual(res.selected.field, "borrower_segment")
        # With regulatory reporting ON -> prefer regulatory.
        res_on = resolve_regulatory_preference(
            _example2(), mode="warehouse_securitisation", delta_threshold=0.10,
            regulatory_reporting_enabled=True)
        self.assertEqual(res_on.selected.field, "employment_status")
        # A regulatory CORE field is preferred even when reporting is off.
        res_core = resolve_regulatory_preference(
            _example1(), mode="warehouse_securitisation", delta_threshold=0.10,
            regulatory_reporting_enabled=False)
        self.assertEqual(res_core.selected.field, "original_principal_balance")

    def test_no_ambiguity_when_delta_too_large(self):
        cands = [
            Candidate("original_principal_balance", 0.90, "regulatory", True),
            Candidate("loan_amount_analytics", 0.50, "analytics", False),
        ]
        self.assertIsNone(
            resolve_regulatory_preference(cands, mode="regulatory_mi", delta_threshold=0.10))

    def test_no_ambiguity_below_confidence_floor(self):
        cands = [
            Candidate("employment_status", 0.33, "regulatory", False),
            Candidate("borrower_segment", 0.33, "analytics", False),
        ]
        self.assertIsNone(resolve_regulatory_preference(
            cands, mode="regulatory_mi", delta_threshold=0.10,
            min_candidate_confidence=0.60))

    def test_config_delta_threshold_loads(self):
        self.assertAlmostEqual(load_ambiguity_delta(), 0.10, places=4)


def _project_with_ambiguity(tmp: Path, mode: str) -> OnboardingProject:
    proj = OnboardingProject(
        project_id="amb", client_name="AMB", input_dir=str(tmp), output_dir=str(tmp),
        onboarding_mode=mode,
    )
    proj.mapping_ambiguities = [
        MappingAmbiguity(
            source_file="monthly_loan_report.csv",
            source_column="original principal",
            selected_canonical_field="original_principal_balance",
            selected_category="regulatory", selected_core_canonical=True,
            selected_confidence=0.86,
            alternative_canonical_field="loan_amount_analytics",
            alternative_category="analytics", alternative_core_canonical=False,
            alternative_confidence=0.84, confidence_delta=0.02,
            ambiguity_rule_applied=REASON_REGULATORY_MI, review_required=True,
            reason=REASON_REGULATORY_MI, mode=mode,
        )
    ]
    return proj


class TestArtefacts(unittest.TestCase):
    def test_5_ambiguity_written_to_05b(self):
        tmp = Path(tempfile.mkdtemp(prefix="amb05b_"))
        proj = _project_with_ambiguity(tmp, "regulatory_mi")
        _write_artifacts(proj)
        csv = tmp / "05b_mapping_ambiguities.csv"
        self.assertTrue(csv.exists())
        text = csv.read_text(encoding="utf-8")
        self.assertIn("original principal", text)
        self.assertIn("original_principal_balance", text)
        self.assertIn("loan_amount_analytics", text)
        self.assertIn(REASON_REGULATORY_MI, text)
        self.assertTrue((tmp / "05b_mapping_ambiguities.json").exists())

    def test_6_review_pack_shows_ambiguity_section(self):
        tmp = Path(tempfile.mkdtemp(prefix="amb_pack_"))
        proj = _project_with_ambiguity(tmp, "regulatory_mi")
        pack = tmp / "08_onboarding_review_pack.html"
        build_review_pack(proj, pack)
        html = pack.read_text(encoding="utf-8")
        self.assertIn("Mapping ambiguities resolved by policy", html)
        self.assertIn("regulatory interpretation as the safer default", html)
        self.assertIn("original_principal_balance", html)

    def test_6b_review_pack_mi_only_wording(self):
        tmp = Path(tempfile.mkdtemp(prefix="amb_pack_mi_"))
        proj = _project_with_ambiguity(tmp, "mi_only")
        pack = tmp / "08_onboarding_review_pack.html"
        build_review_pack(proj, pack)
        html = pack.read_text(encoding="utf-8")
        self.assertIn("Regulatory non-core candidates were not selected", html)


class TestIntegration(unittest.TestCase):
    """A real propose_mappings run that genuinely fires the rule (low floor)."""

    def test_propose_mappings_records_ambiguity(self):
        import pandas as pd

        from engine.gate_1_alignment.semantic_alignment import load_field_registry
        from engine.onboarding_agent.field_scope import resolve_field_scope
        from engine.onboarding_agent.mapping_proposer import propose_mappings
        from engine.onboarding_agent.mode_policy import load_mode_policy
        from engine.onboarding_agent.onboarding_models import FileInventoryItem

        registry_path = _REPO_ROOT / "config" / "system" / "fields_registry.yaml"
        aliases = _REPO_ROOT / "config" / "system"
        scope = resolve_field_scope(load_field_registry(registry_path),
                                    load_mode_policy("regulatory_mi"))
        inv = [FileInventoryItem(file_path="x.csv", file_name="x.csv",
                                 file_type="csv", classification="loan_report")]
        df = pd.DataFrame({"principal outstanding": [1, 2, 3]})
        frames = {"x.csv": df}
        # Low floor forces the token-overlap competitors to count as ambiguous.
        cands, oos, ambiguities = propose_mappings(
            inv, frames, registry_path, aliases, field_scope=scope,
            min_candidate_confidence=0.0,
        )
        self.assertTrue(ambiguities, "expected at least one ambiguity to be recorded")
        for a in ambiguities:
            self.assertTrue(a.review_required)
            self.assertEqual(a.mode, "regulatory_mi")


if __name__ == "__main__":
    unittest.main()
