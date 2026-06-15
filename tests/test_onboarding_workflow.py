#!/usr/bin/env python3
"""tests/test_onboarding_workflow.py

Thin operator workflow wrapper (engine.onboarding_agent.workflow): first/second
pass orchestration, 40 workflow summary, 41 legacy audit, status derivation and
path defaults. The wrapper must not change deterministic onboarding behaviour.
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
import warnings
from pathlib import Path

import yaml

_REPO_ROOT = Path(__file__).resolve().parent.parent
for _p in (str(_REPO_ROOT), str(_REPO_ROOT / "tests")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from engine.onboarding_agent import workflow as wf

PACK = str(_REPO_ROOT / "synthetic_demo" / "input")
REGISTRY = str(_REPO_ROOT / "config" / "system" / "fields_registry.yaml")
ALIASES = str(_REPO_ROOT / "config" / "system")


def _first_pass(out: Path, advisor=False):
    warnings.simplefilter("ignore")
    return wf.run_operator_workflow(
        input_dir=PACK, client_name="CLIENT_001_TEST", client_id="client_001",
        run_id="r1", project_dir=str(out), mode="mi_only", registry=REGISTRY,
        aliases_dir=ALIASES, enable_llm_target_advisor=advisor)


# --------------------------------------------------------------------------- #
# 4. Status derivation (pure)
# --------------------------------------------------------------------------- #
class TestStatusDerivation(unittest.TestCase):
    def test_empty_is_ready(self):
        self.assertEqual(wf.derive_status(0, 0, False), wf.READY)

    def test_non_blocking_is_needs_confirmation(self):
        self.assertEqual(wf.derive_status(8, 0, False), wf.NEEDS_CONFIRMATION)

    def test_blocking_is_blocked(self):
        self.assertEqual(wf.derive_status(3, 2, False), wf.BLOCKED)

    def test_missing_artifacts_is_failed(self):
        self.assertEqual(wf.derive_status(0, 0, True), wf.FAILED)


# --------------------------------------------------------------------------- #
# 1. First-pass workflow
# --------------------------------------------------------------------------- #
class TestFirstPass(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.out = Path(tempfile.mkdtemp(prefix="wf1_"))
        cls.summary = _first_pass(cls.out)

    def test_summary_artifacts_written(self):
        self.assertTrue((self.out / "40_operator_workflow_summary.json").exists())
        self.assertTrue((self.out / "40_operator_workflow_summary.md").exists())

    def test_decision_template_written(self):
        self.assertTrue((self.out / "34_target_first_decisions.yaml").exists())

    def test_required_artifacts_exist(self):
        for n in ("08_onboarding_review_pack.html", "28a_target_coverage_matrix.csv",
                  "28c_human_decision_queue.csv", "34_target_first_decisions.yaml"):
            self.assertTrue((self.out / n).exists(), n)

    def test_status_is_target_first_based(self):
        self.assertIn(self.summary["status"], (wf.READY, wf.NEEDS_CONFIRMATION, wf.BLOCKED))
        self.assertEqual(self.summary["workflow_stage"], wf.FIRST_PASS)
        # Status matches the 28c state, not legacy gaps.
        dec = json.loads((self.out / "28c_human_decision_queue.json").read_text())
        nblock = dec["summary"]["blocking_decisions"]
        ntotal = dec["summary"]["human_decision_rows_total"]
        expected = wf.derive_status(ntotal, nblock, False)
        self.assertEqual(self.summary["status"], expected)

    def test_next_action_present(self):
        self.assertTrue(self.summary["next_operator_action"])


# --------------------------------------------------------------------------- #
# 2. Second-pass workflow
# --------------------------------------------------------------------------- #
class TestSecondPass(unittest.TestCase):
    def test_apply_reduces_28c_and_writes_log(self):
        out1 = Path(tempfile.mkdtemp(prefix="wf2a_"))
        s1 = _first_pass(out1)
        before = s1["human_decision_queue_count_after"]
        self.assertGreater(before, 0)
        # Approve every decision as mark_not_applicable.
        tpl = yaml.safe_load((out1 / "34_target_first_decisions.yaml").read_text())
        for d in tpl["decisions"]:
            d["status"] = "approved"
            d["selected_action"] = "mark_not_applicable"
        approved = out1 / "34_target_first_decisions_APPROVED.yaml"
        approved.write_text(yaml.safe_dump(tpl, sort_keys=False), encoding="utf-8")

        out2 = Path(tempfile.mkdtemp(prefix="wf2b_"))
        warnings.simplefilter("ignore")
        s2 = wf.run_operator_workflow(
            input_dir=PACK, client_name="CLIENT_001_TEST", client_id="client_001",
            run_id="r2", project_dir=str(out2), mode="mi_only", registry=REGISTRY,
            aliases_dir=ALIASES, target_first_decisions=str(approved))
        self.assertEqual(s2["workflow_stage"], wf.SECOND_PASS)
        self.assertTrue((out2 / "35_target_first_decision_application_log.json").exists())
        self.assertTrue((out2 / "35_target_first_decision_application_log.csv").exists())
        self.assertEqual(s2["applied_decisions_count"], len(tpl["decisions"]))
        self.assertLess(s2["human_decision_queue_count_after"], before)
        self.assertEqual(s2["human_decision_queue_count_before"],
                         s2["human_decision_queue_count_after"] + s2["applied_decisions_count"])
        # Status reflects the resulting (empty) 28c.
        self.assertEqual(s2["status"], wf.READY)


# --------------------------------------------------------------------------- #
# 3. LLM advisor optional + advisory only
# --------------------------------------------------------------------------- #
class TestAdvisorOptional(unittest.TestCase):
    def test_without_advisor_no_36_required(self):
        out = Path(tempfile.mkdtemp(prefix="wf_noadv_"))
        s = _first_pass(out, advisor=False)
        self.assertFalse(s["llm_target_advisor_enabled"])
        self.assertFalse((out / "36_target_first_llm_recommendations.csv").exists())
        self.assertEqual(s["warnings"], [])

    def test_with_advisor_writes_36_and_summary_fields(self):
        out = Path(tempfile.mkdtemp(prefix="wf_adv_"))
        s = _first_pass(out, advisor=True)
        self.assertTrue(s["llm_target_advisor_enabled"])
        self.assertTrue((out / "36_target_first_llm_recommendations.csv").exists())
        self.assertTrue((out / "36_target_first_llm_usage_summary.json").exists())
        for k in ("llm_target_advisor_file", "llm_target_advisor_rows",
                  "llm_target_advisor_advised_count", "llm_target_advisor_parse_failed_count",
                  "llm_target_advisor_estimated_cost_gbp"):
            self.assertIn(k, s)
        # Advisory only: 28c is unchanged vs a no-advisor run (deterministic).
        out2 = Path(tempfile.mkdtemp(prefix="wf_noadv2_"))
        s2 = _first_pass(out2, advisor=False)
        self.assertEqual(s["human_decision_queue_count_after"],
                         s2["human_decision_queue_count_after"])


# --------------------------------------------------------------------------- #
# 5. Path defaults
# --------------------------------------------------------------------------- #
class TestPathDefaults(unittest.TestCase):
    def test_default_project_dir_and_output_root(self):
        warnings.simplefilter("ignore")
        rid = "wf_default_paths_test"
        expected = _REPO_ROOT / "onboarding_output" / "client_001" / rid
        import shutil
        if expected.exists():
            shutil.rmtree(expected)
        try:
            s = wf.run_operator_workflow(
                input_dir=PACK, client_name="CLIENT_001_TEST", client_id="client_001",
                run_id=rid, mode="mi_only", registry=REGISTRY, aliases_dir=ALIASES)
            self.assertEqual(Path(s["project_dir"]), expected)
            self.assertEqual(Path(s["output_root"]), expected / "output")
            self.assertTrue(expected.exists())
        finally:
            if expected.exists():
                shutil.rmtree(expected)


# --------------------------------------------------------------------------- #
# 6. Legacy audit (non-mutating)
# --------------------------------------------------------------------------- #
class TestLegacyAudit(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.out = Path(tempfile.mkdtemp(prefix="wf_audit_"))
        _first_pass(cls.out)
        cls.audit = json.loads(
            (cls.out / "41_onboarding_legacy_file_audit.json").read_text())

    def test_audit_written(self):
        self.assertTrue((self.out / "41_onboarding_legacy_file_audit.json").exists())
        self.assertTrue((self.out / "41_onboarding_legacy_file_audit.md").exists())
        self.assertTrue(self.audit["non_mutating"])

    def test_target_first_are_keep_core(self):
        by_name = {e["name"]: e for e in self.audit["entries"]}
        for n in ("28a_target_coverage_matrix.csv", "28c_human_decision_queue.csv",
                  "34_target_first_decisions.yaml"):
            self.assertEqual(by_name[n]["classification"], "keep_core")

    def test_legacy_source_column_not_marked_removal(self):
        by_name = {e["name"]: e for e in self.audit["entries"]}
        for n in ("33_mapping_review_queue.csv", "34_mapping_review_decisions.yaml",
                  "35_mapping_review_action_log.json"):
            self.assertIn(by_name[n]["classification"],
                          ("keep_legacy_audit", "candidate_for_deprecation"))
            self.assertNotEqual(by_name[n]["classification"],
                                "candidate_for_removal_after_migration")

    def test_audit_does_not_delete_repo_files(self):
        # Files referenced by the audit still exist on disk.
        for n in ("engine/onboarding_agent/target_coverage.py",
                  "engine/onboarding_agent/mapping_review_queue.py"):
            self.assertTrue((_REPO_ROOT / n).exists())


# --------------------------------------------------------------------------- #
# 7. Existing lower-level CLI still works
# --------------------------------------------------------------------------- #
class TestLowerLevelCliUnaffected(unittest.TestCase):
    def test_cli_help_runs(self):
        r = subprocess.run([sys.executable, "-m", "engine.onboarding_agent.cli", "--help"],
                           cwd=str(_REPO_ROOT), capture_output=True, text=True)
        self.assertEqual(r.returncode, 0)
        self.assertIn("--input-dir", r.stdout)

    def test_workflow_help_runs(self):
        r = subprocess.run([sys.executable, "-m", "engine.onboarding_agent.workflow", "--help"],
                           cwd=str(_REPO_ROOT), capture_output=True, text=True)
        self.assertEqual(r.returncode, 0)
        self.assertIn("--target-first-decisions", r.stdout)


if __name__ == "__main__":
    unittest.main()
