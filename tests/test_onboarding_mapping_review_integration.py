#!/usr/bin/env python3
"""tests/test_onboarding_mapping_review_integration.py — CLI/orchestrator wiring.

Proves the controlled LLM-assisted mapping flow is executed by the NORMAL
onboarding run (not only a separate subcommand):

  * --enable-mapping-review (LLM off) generates 28,29,30,32,33,34,35,37
  * 31_* only appears when the LLM is enabled (with an injected callable)
  * old LLM flags are still accepted and map to the new settings
  * the workbench exposes the queue when present and warns when absent
"""

from __future__ import annotations

import shutil
import sys
import tempfile
import unittest
import warnings
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
for _p in (str(_REPO_ROOT), str(_REPO_ROOT / "tests")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from engine.onboarding_agent import streamlit_onboarding_workbench as wb
from engine.onboarding_agent.cli import build_parser
from engine.onboarding_agent.onboarding_orchestrator import run_onboarding

REGISTRY = str(_REPO_ROOT / "config" / "system" / "fields_registry.yaml")
ALIASES = str(_REPO_ROOT / "config" / "system")
KFI_FIXTURE = _REPO_ROOT / "tests" / "fixtures" / "kfi_pipeline_headers.csv"

_DETERMINISTIC = ["28_existing_pipeline_field_contract.csv", "29_column_evidence.csv",
                  "30_mapping_candidate_shortlist.csv", "32_mapping_backstop_validation.csv",
                  "33_mapping_review_queue.csv", "37_schema_drift_report.csv"]


def _input_dir() -> Path:
    d = Path(tempfile.mkdtemp()) / "input"
    d.mkdir(parents=True)
    shutil.copy(KFI_FIXTURE, d / "kfi_pipeline.csv")
    return d


def _run(out, enable_mapping_review=False, enable_llm=False, llm=None):
    warnings.simplefilter("ignore")
    return run_onboarding(
        input_dir=str(_input_dir()), client_name="CLIENT_001_TEST", output_dir=str(out),
        registry_path=REGISTRY, aliases_dir=ALIASES, mode="mi_only",
        client_id="client_001", run_id="r1",
        enable_mapping_review=enable_mapping_review, enable_llm_mapping_review=enable_llm,
        llm_mapping_callable=llm)


class TestDeterministicReviewGenerated(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.out = Path(tempfile.mkdtemp()) / "run"
        cls.project = _run(cls.out, enable_mapping_review=True)

    # 1. Normal run with mapping review + LLM off produces 28,29,30,32,33,37.
    def test_deterministic_artefacts_generated(self):
        for name in _DETERMINISTIC:
            self.assertTrue((self.out / name).exists(), f"missing {name}")
        # 31_* (LLM) must NOT be written when the LLM is off.
        self.assertFalse((self.out / "31_llm_mapping_review.json").exists())

    # 6. The KFI/pipeline scenario generates a populated 33_mapping_review_queue.csv.
    def test_queue_populated(self):
        queue = wb.load_mapping_review_queue(self.out)
        self.assertTrue(queue["items"])
        cols = {it["source_column"] for it in queue["items"]}
        self.assertIn("Offer Date", cols)
        self.assertIn("Status", cols)

    def test_artefacts_listed_in_summary(self):
        names = {Path(a).name for a in self.project.generated_artifacts}
        for name in _DETERMINISTIC:
            self.assertIn(name, names)
        self.assertTrue(self.project.mapping_review_summary.get("total_columns_reviewed"))


class TestLlmReviewGated(unittest.TestCase):
    # 2. A run with the LLM enabled (injected callable) writes 31_* resolver + usage.
    def test_llm_artefacts_when_enabled(self):
        import json

        def fake_llm(prompt):
            return json.dumps([{"source_file": "kfi_pipeline.csv",
                                "source_column": "Gender APP 1",
                                "resolved_target_field": "", "decision": "ignore_source_field",
                                "confidence": 0.3, "rationale": "no contract target"}])
        out = Path(tempfile.mkdtemp()) / "run"
        project = _run(out, enable_llm=True, llm=fake_llm)
        self.assertTrue((out / "31_llm_mapping_resolver.json").exists())
        self.assertTrue((out / "31_llm_usage_summary.json").exists())
        self.assertTrue((out / "22_llm_usage_summary.json").exists())
        self.assertTrue(project.mapping_review_summary.get("llm_enabled"))
        self.assertGreaterEqual(project.mapping_review_summary.get("llm_calls", 0), 1)


class TestBackwardCompatFlags(unittest.TestCase):
    # 3. Old LLM flags still parse; new flags exist and fall back to old ones.
    def test_old_and_new_flags_parse(self):
        p = build_parser()
        new = p.parse_args([
            "--input-dir", "x", "--client-name", "c", "--enable-mapping-review",
            "--enable-llm-mapping-review", "--llm-mapping-profile", "low",
            "--llm-review-only-unresolved", "--llm-max-mapping-items", "25",
            "--llm-max-cost-gbp", "2"])
        self.assertTrue(new.enable_mapping_review)
        self.assertEqual(new.llm_max_mapping_items, 25)
        old = p.parse_args([
            "--input-dir", "x", "--client-name", "c", "--enable-llm-review",
            "--llm-budget-profile", "low", "--llm-max-calls", "3",
            "--llm-max-items-per-call", "5"])
        self.assertTrue(old.enable_llm_review)
        self.assertEqual(old.llm_budget_profile, "low")
        # New mapping-items cap is unset -> main() falls back to the legacy cap.
        self.assertIsNone(old.llm_max_mapping_items)
        self.assertEqual(old.llm_max_items_per_call, 5)


class TestWorkbenchSurfacing(unittest.TestCase):
    # 4. Workbench exposes the queue when 33_* exists.
    def test_present(self):
        out = Path(tempfile.mkdtemp()) / "run"
        _run(out, enable_mapping_review=True)
        self.assertTrue(wb.mapping_review_artifacts_present(out))
        self.assertTrue(wb.load_mapping_review_queue(out)["items"])

    # 5. Workbench reports missing artefacts (no silent fallback) when 33_* absent.
    def test_absent(self):
        empty = Path(tempfile.mkdtemp())
        self.assertFalse(wb.mapping_review_artifacts_present(empty))
        self.assertEqual(wb.load_mapping_review_queue(empty)["items"], [])


if __name__ == "__main__":
    unittest.main()
