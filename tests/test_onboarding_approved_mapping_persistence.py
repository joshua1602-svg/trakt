#!/usr/bin/env python3
"""tests/test_onboarding_approved_mapping_persistence.py — PART 13 (16-20, 28)."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import yaml

_REPO_ROOT = Path(__file__).resolve().parent.parent
for _p in (str(_REPO_ROOT), str(_REPO_ROOT / "tests")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from engine.onboarding_agent import mapping_memory as mm
from engine.onboarding_agent import mapping_persistence as mp


class TestPersistence(unittest.TestCase):
    # 16. Approved mapping writes to client memory.
    def test_persist_to_client_memory(self):
        out = Path(tempfile.mkdtemp())
        res = mp.persist_to_client_memory(
            [{"source_column": "Offer Date", "canonical_field": "offer_date",
              "source_file_pattern": "*", "mode": "regulatory_mi", "domain": "pipeline"}],
            client_id="c1", output_dir=str(out), run_id="r1")
        self.assertEqual(res["saved"], 1)
        store = mm.MappingMemoryStore(res["memory_dir"], client_id="c1")
        self.assertEqual(store.counts()["total"], 1)

    # 17. Approved alias writes to the alias library ONLY when user confirms.
    def test_persist_aliases_requires_confirm(self):
        root = Path(tempfile.mkdtemp())
        item = {"alias": "funds released date", "canonical_field": "date_funds_released",
                "approved_by": "user", "source_client_id": "c1", "source_run_id": "r1",
                "llm_used": False}
        dry = mp.persist_aliases([item], confirm=False, repo_root=root)
        self.assertFalse(dry["written"])
        self.assertFalse((root / mp.ALIAS_PIPELINE_FILE).exists())
        wet = mp.persist_aliases([item], confirm=True, repo_root=root)
        self.assertTrue(wet["written"])
        data = yaml.safe_load((root / mp.ALIAS_PIPELINE_FILE).read_text())
        self.assertIn("funds released date", data["date_funds_released"]["aliases"])
        # Companion metadata captures provenance.
        meta = yaml.safe_load((root / mp.ALIAS_PIPELINE_META).read_text())
        self.assertTrue(any(m["alias"] == "funds released date" for m in meta))

    # 18. Proposed new field writes to a registry PATCH, not the core registry.
    def test_propose_registry_patch_not_core(self):
        out = Path(tempfile.mkdtemp())
        path = mp.propose_registry_patch(
            [{"field_name": "kfi_identifier", "format": "string", "domain": "pipeline"}],
            output_dir=out, client_id="c1", run_id="r1")
        self.assertEqual(path.name, "36_registry_patch_proposed.yaml")
        patch = yaml.safe_load(path.read_text())
        self.assertIn("kfi_identifier", patch["fields"])
        self.assertEqual(patch["fields"]["kfi_identifier"]["category"], "pipeline")
        # The core registry is untouched.
        core = _REPO_ROOT / "config" / "system" / "fields_registry.yaml"
        self.assertNotIn("kfi_identifier", yaml.safe_load(core.read_text())["fields"])

    # 28. A regulatory field is NEVER created from an LLM-assisted proposal.
    def test_regulatory_field_refused(self):
        out = Path(tempfile.mkdtemp())
        path = mp.propose_registry_patch(
            [{"field_name": "sneaky_reg_field", "format": "decimal", "category": "regulatory"}],
            output_dir=out)
        patch = yaml.safe_load(path.read_text())
        self.assertNotIn("sneaky_reg_field", patch["fields"])
        self.assertIn("sneaky_reg_field", patch["refused_regulatory_fields"])

    # 19. Approved registry patch can be applied to the pipeline registry extension.
    def test_apply_patch_to_pipeline_registry(self):
        out = Path(tempfile.mkdtemp())
        root = Path(tempfile.mkdtemp())
        path = mp.propose_registry_patch(
            [{"field_name": "kfi_identifier", "format": "string"}], output_dir=out)
        dry = mp.apply_registry_patch(path, confirm=False, repo_root=root)
        self.assertFalse(dry["applied"])
        wet = mp.apply_registry_patch(path, confirm=True, repo_root=root)
        self.assertTrue(wet["applied"])
        data = yaml.safe_load((root / mp.PIPELINE_REGISTRY_FILE).read_text())
        self.assertIn("kfi_identifier", data["fields"])

    # 20. Future runs use client memory deterministically (no LLM).
    def test_future_run_uses_memory_deterministically(self):
        from engine.onboarding_agent.onboarding_models import MappingCandidate
        out = Path(tempfile.mkdtemp())
        mp.persist_to_client_memory(
            [{"source_column": "loan amount", "canonical_field": "original_principal_balance",
              "source_file_pattern": "kfi*", "mode": "regulatory_mi"}],
            client_id="c1", output_dir=str(out), run_id="r1")
        store = mm.MappingMemoryStore(
            mm.resolve_memory_dir(output_dir=str(out), client_id="c1"), client_id="c1")
        cand = MappingCandidate(source_file="kfi.csv", source_column="loan_amount",
                                candidate_canonical_field="", confidence=0.0)
        res = mm.apply_mapping_memory([cand], store, mode="regulatory_mi")
        self.assertEqual(cand.candidate_canonical_field, "original_principal_balance")
        self.assertEqual(cand.method, "client_memory")
        self.assertEqual(res["applied"], 1)


if __name__ == "__main__":
    unittest.main()
