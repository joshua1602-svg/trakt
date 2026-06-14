#!/usr/bin/env python3
"""
tests/test_onboarding_document_policy.py

PART 9 tests 11-15 — document extraction minimisation + client-scoped
persistence guard.
"""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import yaml

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from engine.onboarding_agent import file_classifier
from engine.onboarding_agent.document_extractor import (
    PersistenceScopeError,
    assert_within_project,
    extract_documents,
    load_document_policy,
    write_document_extraction_summary,
)
from engine.onboarding_agent.onboarding_orchestrator import run_onboarding

PACK = _REPO_ROOT / "synthetic_onboarding_pack"
REGISTRY = _REPO_ROOT / "config" / "system" / "fields_registry.yaml"
ALIASES = _REPO_ROOT / "config" / "system"

# Long, document-only passages that would only appear if full text (not a short
# capped excerpt) were persisted. Short headings/values may legitimately appear
# inside a policy-capped evidence excerpt, so we assert on long unique passages.
_FULL_TEXT_FRAGMENTS = [
    "contains no real counterparty data",
    "Minimum borrower age at origination",
    "Underlying exposures funded under this facility must satisfy",
    "no later than five business",
]


class TestDocumentPolicy(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.policy = load_document_policy()
        cls.inventory = file_classifier.classify_directory(PACK)
        cls.extractions = extract_documents(cls.inventory, cls.policy)

    def test_policy_disables_full_text(self):
        self.assertFalse(self.policy.get("persist_full_text", True))
        self.assertFalse(self.policy.get("persist_raw_chunks", True))
        self.assertTrue(self.policy.get("client_scoped_only", False))

    def test_only_allowed_fields_extracted(self):
        allowed_map = self.policy["allowed_extraction_fields"]
        cls_by_doc = {i.file_name: i.classification for i in self.inventory}
        for ex in self.extractions:
            allowed = allowed_map.get(cls_by_doc.get(ex.source_document, ""), [])
            self.assertIn(ex.field, allowed, f"{ex.field} not allowed for {ex.source_document}")

    def test_retained_evidence_capped(self):
        cap = int(self.policy["allowed_retained_evidence_chars"])
        for ex in self.extractions:
            self.assertLessEqual(len(ex.retained_evidence), cap)

    def test_extractions_have_warehouse_terms(self):
        fields = {e.field for e in self.extractions}
        self.assertIn("advance_rate", fields)
        self.assertIn("warehouse_facility_present", fields)


class TestPersistenceGuard(unittest.TestCase):
    def test_within_project_ok(self):
        tmp = Path(tempfile.mkdtemp())
        target = assert_within_project(tmp / "17_document_extraction_summary.yaml", tmp)
        self.assertTrue(str(target).startswith(str(tmp.resolve())))

    def test_outside_project_rejected(self):
        tmp = Path(tempfile.mkdtemp())
        with self.assertRaises(PersistenceScopeError):
            assert_within_project(_REPO_ROOT / "config" / "system" / "x.yaml", tmp)

    def test_global_config_dir_rejected(self):
        # Even if somehow under the project root, global config trees are refused.
        tmp = Path(tempfile.mkdtemp())
        with self.assertRaises(PersistenceScopeError):
            assert_within_project(Path("/etc/passwd"), tmp)


class TestEndToEndPersistence(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp(prefix="docpol_"))
        cls.project = run_onboarding(
            input_dir=str(PACK), client_name="TEST", output_dir=str(cls.tmp),
            registry_path=str(REGISTRY), aliases_dir=str(ALIASES),
            mode="warehouse_securitisation",
        )

    def test_summary_written_under_project(self):
        self.assertTrue((self.tmp / "17_document_extraction_summary.yaml").exists())

    def test_no_full_document_text_in_any_artifact(self):
        for path in self.tmp.iterdir():
            if not path.is_file():
                continue
            text = path.read_text(encoding="utf-8", errors="ignore")
            for fragment in _FULL_TEXT_FRAGMENTS:
                self.assertNotIn(
                    fragment, text, f"full-text fragment leaked into {path.name}"
                )

    def test_summary_only_allowed_fields(self):
        data = yaml.safe_load((self.tmp / "17_document_extraction_summary.yaml").read_text())
        policy = load_document_policy()
        allowed_all = set()
        for fields in policy["allowed_extraction_fields"].values():
            allowed_all.update(fields)
        for ex in data["document_extractions"]:
            self.assertIn(ex["field"], allowed_all)

    def test_global_config_not_modified(self):
        # The system config files must be unchanged by an onboarding run.
        import subprocess

        result = subprocess.run(
            ["git", "status", "--porcelain", "config/system", "config/regime", "config/client"],
            cwd=str(_REPO_ROOT), capture_output=True, text=True,
        )
        # Only allow the policy files we intentionally added in this feature; no
        # run-time mutation of config trees should appear.
        dirty = [
            ln for ln in result.stdout.splitlines()
            if ln and not ln.endswith((
                "onboarding_modes.yaml", "onboarding_agent.yaml",
                "aliases_onboarding_lending.yaml", "aliases_onboarding_kfi.yaml",
            ))
        ]
        self.assertEqual(dirty, [], f"unexpected config changes: {dirty}")


if __name__ == "__main__":
    unittest.main()
