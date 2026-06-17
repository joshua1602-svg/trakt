#!/usr/bin/env python3
"""tests/test_annex2_handoff_validation.py

Validation of the Onboarding → Transformation & Validation handoff package for
the ESMA Annex 2 (regulatory) target contract.

This focuses on the acceptance criteria and the failure-mode guard:
  * the regulatory workflow creates a formal handoff package under output/handoff/;
  * the manifest states it is a canonical_onboarding_package for the
    transformation_validation agent, not raw source, and that Gate 1 must not be
    re-run on the central tape;
  * the package does not claim XML readiness;
  * the documented guard against running raw Gate 1 on the central tape exists.
"""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
import warnings
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from engine.onboarding_agent import workflow as wf

PACK = str(_REPO_ROOT / "synthetic_demo" / "input")
REGISTRY = str(_REPO_ROOT / "config" / "system" / "fields_registry.yaml")
ALIASES = str(_REPO_ROOT / "config" / "system")


class TestAnnex2HandoffValidation(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        warnings.simplefilter("ignore")
        cls.out = Path(tempfile.mkdtemp(prefix="annex2_handoff_"))
        cls.summary = wf.run_operator_workflow(
            input_dir=PACK, client_name="CLIENT_001_TEST", client_id="client_001",
            run_id="annex2_handoff", project_dir=str(cls.out), mode="regulatory_mi",
            registry=REGISTRY, aliases_dir=ALIASES)
        cls.hf = cls.out / "output" / "handoff"
        cls.manifest = json.loads(
            (cls.hf / "24_onboarding_handoff_manifest.json").read_text())
        cls.readiness = json.loads(
            (cls.hf / "25_onboarding_handoff_readiness.json").read_text())

    def test_handoff_package_created(self):
        self.assertTrue(self.hf.is_dir())
        self.assertTrue((self.hf / "24_onboarding_handoff_manifest.json").exists())
        self.assertTrue((self.hf / "24_onboarding_handoff_manifest.yaml").exists())

    def test_manifest_governance_statements(self):
        m = self.manifest
        self.assertEqual(m["handoff_type"], "canonical_onboarding_package")
        self.assertEqual(m["next_agent"], "transformation_validation")
        self.assertEqual(m["not_raw_source"], True)
        self.assertEqual(m["do_not_rerun_gate1_on_central_tape"], True)

    def test_targets_esma_annex_2(self):
        self.assertEqual(self.manifest["target_contract_id"], "esma_annex_2")

    def test_consumable_without_reprofiling(self):
        # The next agent gets the coverage matrix, decision queue, field contract
        # and lineage — so it never has to re-profile raw sources or re-map.
        m = self.manifest
        for key in ("target_coverage_matrix_path", "decision_queue_path",
                    "field_contract_path", "lineage_path", "central_tape_path"):
            self.assertTrue(m.get(key), key)
        self.assertTrue(m["ready_for_transformation_validation"])

    def test_not_xml_ready(self):
        self.assertFalse(self.manifest["ready_for_xml_delivery"])
        self.assertTrue(self.manifest["not_xml_ready"])
        self.assertFalse(self.readiness["ready_for_xml_delivery"])

    def test_unresolved_items_classified_not_failed(self):
        # Pending regime rules / defaults are classified for the next agent, not
        # treated as onboarding failures that block the handoff.
        m = self.manifest
        self.assertGreaterEqual(m["pending_regime_rule_count"], 0)
        self.assertGreaterEqual(m["downstream_default_required_count"], 0)
        self.assertEqual(m["blocking_decision_count"], 0)
        self.assertTrue(m["ready_for_transformation_validation"])

    def test_contract_doc_states_guard(self):
        doc = (_REPO_ROOT / "due_diligence" / "ONBOARDING_HANDOFF_CONTRACT.md")
        self.assertTrue(doc.exists())
        text = doc.read_text()
        self.assertIn("Do not run raw Gate 1 canonicalisation on", text)
        self.assertIn("output/central/18_central_lender_tape.csv", text)
        self.assertIn("Transformation & Validation Agent", text)


class TestTraktRunGuard(unittest.TestCase):
    """The raw Gate 1 orchestrator refuses the central lender tape as input."""

    def test_guard_refuses_central_tape(self):
        from engine.orchestrator.trakt_run import _guard_not_onboarding_central_tape
        d = Path(tempfile.mkdtemp()) / "central"
        d.mkdir(parents=True)
        f = d / "18_central_lender_tape.csv"
        f.write_text("a,b\n1,2\n")
        with self.assertRaises(SystemExit):
            _guard_not_onboarding_central_tape(f)

    def test_guard_can_be_forced(self):
        from engine.orchestrator.trakt_run import _guard_not_onboarding_central_tape
        d = Path(tempfile.mkdtemp()) / "central"
        d.mkdir(parents=True)
        f = d / "18_central_lender_tape.csv"
        f.write_text("a,b\n1,2\n")
        _guard_not_onboarding_central_tape(f, force=True)  # no raise

    def test_guard_allows_raw_tape(self):
        from engine.orchestrator.trakt_run import _guard_not_onboarding_central_tape
        f = Path(tempfile.mkdtemp()) / "raw_client_tape.csv"
        f.write_text("a\n1\n")
        _guard_not_onboarding_central_tape(f)  # no raise


if __name__ == "__main__":
    unittest.main()
