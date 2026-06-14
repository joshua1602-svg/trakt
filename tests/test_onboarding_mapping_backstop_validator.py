#!/usr/bin/env python3
"""tests/test_onboarding_mapping_backstop_validator.py — PART 13 (10, 11, 12)."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
for _p in (str(_REPO_ROOT), str(_REPO_ROOT / "tests")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from engine.gate_1_alignment.semantic_alignment import load_field_registry
from engine.onboarding_agent import mapping_backstop_validator as bv
from engine.onboarding_agent.field_scope import resolve_field_scope
from engine.onboarding_agent.mode_policy import load_mode_policy

REGISTRY = str(_REPO_ROOT / "config" / "system" / "fields_registry.yaml")


def _reg():
    return load_field_registry(Path(REGISTRY)).get("fields", {})


class TestBackstop(unittest.TestCase):
    # 10. Backstop blocks out-of-scope fields.
    def test_blocks_out_of_scope(self):
        reg = _reg()
        fs = resolve_field_scope(REGISTRY, load_mode_policy("mi_only"))
        rows = bv.validate_mappings([{
            "source_column": "Estimated Value", "proposed_target_field": "current_valuation_amount",
            "candidate_source": "alias", "confidence": "high", "type_compatible": True,
        }], registry_fields=reg, field_scope=fs)
        self.assertEqual(rows[0]["validation_status"], bv.OUT_OF_SCOPE)
        self.assertTrue(rows[0]["requires_user_approval"])

    # 11. Backstop blocks type-incompatible mappings.
    def test_blocks_type_incompatible(self):
        reg = _reg()
        fs = resolve_field_scope(REGISTRY, load_mode_policy("regulatory_mi"))
        rows = bv.validate_mappings([{
            "source_column": "Some Date", "proposed_target_field": "current_principal_balance",
            "candidate_source": "semantic_alignment", "confidence": "high",
            "type_compatible": False,
        }], registry_fields=reg, field_scope=fs)
        self.assertEqual(rows[0]["validation_status"], bv.UNSAFE)

    # 12. Backstop routes ambiguous material economic fields to user review.
    def test_material_ambiguous_to_review(self):
        reg = _reg()
        fs = resolve_field_scope(REGISTRY, load_mode_policy("regulatory_mi"))
        rows = bv.validate_mappings([{
            "source_column": "Balance?", "proposed_target_field": "current_principal_balance",
            "candidate_source": "semantic_alignment", "confidence": "medium",
            "type_compatible": True, "ambiguity_flags": ["ambiguous"],
        }], registry_fields=reg, field_scope=fs)
        self.assertEqual(rows[0]["validation_status"], bv.REVIEW_REQUIRED)
        self.assertFalse(rows[0]["auto_approvable"])

    # 12b. Material regulatory field never auto-approves from a fuzzy/LLM source.
    def test_material_regulatory_not_auto_from_semantic(self):
        reg = _reg()
        fs = resolve_field_scope(REGISTRY, load_mode_policy("regulatory_mi"))
        rows = bv.validate_mappings([{
            "source_column": "Orig Bal", "proposed_target_field": "original_principal_balance",
            "candidate_source": "semantic_alignment", "confidence": "high",
            "type_compatible": True,
        }], registry_fields=reg, field_scope=fs)
        self.assertEqual(rows[0]["validation_status"], bv.REVIEW_REQUIRED)

    # Conservative auto-approval: exact pipeline-contract match auto-approves.
    def test_pipeline_contract_auto_approves(self):
        reg = _reg()
        fs = resolve_field_scope(REGISTRY, load_mode_policy("regulatory_mi"))
        rows = bv.validate_mappings([{
            "source_column": "Offer Date", "proposed_target_field": "offer_date",
            "candidate_source": "pipeline_contract", "confidence": "high",
            "type_compatible": True, "is_pipeline_field": True,
        }], registry_fields=reg, field_scope=fs)
        self.assertEqual(rows[0]["validation_status"], bv.AUTO_APPROVED)

    # Two columns mapped to the same target -> conflict.
    def test_conflicting_target_assignment(self):
        reg = _reg()
        fs = resolve_field_scope(REGISTRY, load_mode_policy("regulatory_mi"))
        rows = bv.validate_mappings([
            {"source_column": "A", "proposed_target_field": "offer_date",
             "candidate_source": "pipeline_contract", "confidence": "high",
             "type_compatible": True, "is_pipeline_field": True},
            {"source_column": "B", "proposed_target_field": "offer_date",
             "candidate_source": "pipeline_contract", "confidence": "high",
             "type_compatible": True, "is_pipeline_field": True},
        ], registry_fields=reg, field_scope=fs)
        self.assertTrue(any(r["validation_status"] == bv.CONFLICTS_MAPPING for r in rows))


if __name__ == "__main__":
    unittest.main()
