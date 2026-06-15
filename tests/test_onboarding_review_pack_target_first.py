#!/usr/bin/env python3
"""tests/test_onboarding_review_pack_target_first.py

The onboarding review HTML pack must present the target-contract-first artefacts
(28a/28b/28c) as the PRIMARY managed-service workflow, and demote the legacy
33 source-column queue to audit detail.

Checks:
  * HTML includes the target coverage section (Gate 3).
  * HTML includes the compact decision queue section (Gate 4).
  * HTML includes the residual source field section.
  * HTML labels the 33 queue as audit detail, not the primary gate.
  * HTML does not lead with the 33 source-column approval burden.
  * Headline summary counts are derived from 28a/28b/28c where available.
"""

from __future__ import annotations

import sys
import tempfile
import unittest
import warnings
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
for _p in (str(_REPO_ROOT), str(_REPO_ROOT / "tests")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from engine.onboarding_agent.onboarding_orchestrator import run_onboarding

PACK = _REPO_ROOT / "synthetic_demo" / "input"  # dir holding the synthetic ERE tape
REGISTRY = _REPO_ROOT / "config" / "system" / "fields_registry.yaml"
ALIASES = _REPO_ROOT / "config" / "system"


class TestReviewPackTargetFirst(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        warnings.simplefilter("ignore")
        tmp = tempfile.mkdtemp(prefix="review_pack_tf_")
        cls.project = run_onboarding(
            input_dir=str(PACK), client_name="CLIENT_001_TEST", output_dir=tmp,
            registry_path=str(REGISTRY), aliases_dir=str(ALIASES), mode="mi_only",
            client_id="client_001", run_id="r_tf", enable_mapping_review=True,
        )
        cls.out = Path(tmp)
        cls.html = (cls.out / "08_onboarding_review_pack.html").read_text(encoding="utf-8")

    def test_html_generated(self):
        self.assertTrue((self.out / "08_onboarding_review_pack.html").exists())
        self.assertIn("Onboarding Review Pack", self.html)

    def test_includes_target_coverage_section(self):
        self.assertIn("Gate 3 — Target coverage matrix", self.html)
        self.assertIn("Coverage status counts", self.html)

    def test_includes_compact_decision_queue_section(self):
        self.assertIn("Gate 4 — Compact human decision queue", self.html)
        self.assertIn("Decision type counts", self.html)

    def test_includes_residual_source_field_section(self):
        self.assertIn("Residual source fields", self.html)
        self.assertIn("Residual class counts", self.html)
        # Made clear these are not primary approvals.
        self.assertIn("NOT part of the primary", self.html)

    def test_labels_33_as_audit_detail(self):
        self.assertIn("Detailed source-column audit queue", self.html)
        self.assertIn("Source-column audit detail, not the primary gate", self.html)
        self.assertIn("audit only", self.html)

    def test_does_not_lead_with_33_approval_burden(self):
        # The legacy 33 audit section must come AFTER the target-first gates.
        # Anchor on the <h2> card headings (gate names also appear in callouts).
        pos_exec = self.html.find("<h2>1. Executive onboarding summary")
        pos_g3 = self.html.find("<h2>3. Gate 3 — Target coverage matrix")
        pos_g4 = self.html.find("<h2>4. Gate 4 — Compact human decision queue")
        pos_audit = self.html.find("<h2>7. Detailed source-column audit queue")
        self.assertGreater(pos_g3, pos_exec)
        self.assertGreater(pos_g4, pos_g3)
        self.assertGreater(pos_audit, pos_g4)
        # The old 33 approval count is explicitly labelled audit-only, not the headline.
        self.assertIn("Old 33 approvals (audit only)", self.html)

    def test_headline_status_from_28c_not_legacy(self):
        # The headline (before Gate 2) must be driven by 28c, not the legacy
        # "N blocking question(s)" gap-question banner.
        head = self.html[: self.html.find("<h2>2. Gate 2")]
        self.assertIn("Only ONE blocking target decision remains", head)
        self.assertNotIn("blocking question(s)", head)

    def test_headline_counts_from_target_first(self):
        # Headline KPIs come from 28a/28b/28c.
        import json
        cov = json.loads((self.out / "28a_target_coverage_matrix.json").read_text())
        dec = json.loads((self.out / "28c_human_decision_queue.json").read_text())
        self.assertIn("Target fields", self.html)
        self.assertIn(str(cov["summary"]["target_fields_total"]), self.html)
        self.assertIn("Compact decision queue", self.html)
        # The blocking decision count from 28c is surfaced in Gate 4.
        n_block = dec["summary"]["blocking_decisions"]
        if n_block == 1:
            self.assertIn("Only ONE blocking decision remains", self.html)

    def test_preserves_existing_supporting_sections(self):
        # Existing managed-service detail still present (re-homed under gates).
        for s in ("Data domain coverage", "Azure-ready run metadata",
                  "Field scope for this onboarding mode", "Central tapes",
                  "Mapping ambiguities resolved by policy"):
            self.assertIn(s, self.html)


class TestArtifactLoader(unittest.TestCase):
    """The loader must find 28a/28b/28c in all plausible run locations."""

    def _write(self, d: Path, name: str, payload: dict):
        import json
        d.mkdir(parents=True, exist_ok=True)
        (d / name).write_text(json.dumps(payload), encoding="utf-8")

    def test_finds_files_in_project_dir_root(self):
        from engine.onboarding_agent.review_pack_builder import _load_target_first_artifacts
        proj = Path(tempfile.mkdtemp(prefix="loader_root_"))
        self._write(proj, "28a_target_coverage_matrix.json",
                    {"summary": {"target_fields_total": 3}, "rows": [{"target_field": "x"}]})
        self._write(proj, "28c_human_decision_queue.json",
                    {"summary": {"blocking_decisions": 1}, "rows": [{"blocking": True}]})
        tf = _load_target_first_artifacts(proj)
        self.assertIsNotNone(tf["coverage"])
        self.assertIsNotNone(tf["decision"])
        self.assertEqual(tf["coverage"]["summary"]["target_fields_total"], 3)

    def test_finds_files_under_output_dir(self):
        from engine.onboarding_agent.review_pack_builder import _load_target_first_artifacts
        proj = Path(tempfile.mkdtemp(prefix="loader_out_"))
        self._write(proj / "output", "28b_source_residual_register.json",
                    {"summary": {"residual_source_columns_total": 5}, "rows": []})
        tf = _load_target_first_artifacts(proj)
        self.assertIsNotNone(tf["residual"])
        self.assertEqual(tf["residual"]["summary"]["residual_source_columns_total"], 5)

    def test_finds_files_under_output_root_and_parent(self):
        from engine.onboarding_agent.review_pack_builder import _load_target_first_artifacts
        proj = Path(tempfile.mkdtemp(prefix="loader_root2_"))
        output_root = proj / "output"
        # 28a in output_root, 28c in parent(output_root) == proj.
        self._write(output_root, "28a_target_coverage_matrix.json",
                    {"summary": {"target_fields_total": 9}, "rows": []})
        self._write(proj, "28c_human_decision_queue.json",
                    {"summary": {"blocking_decisions": 0}, "rows": []})
        tf = _load_target_first_artifacts(proj, output_root)
        self.assertIsNotNone(tf["coverage"])
        self.assertIsNotNone(tf["decision"])

    def test_csv_fallback_when_no_json(self):
        from engine.onboarding_agent.review_pack_builder import _load_target_first_artifacts
        proj = Path(tempfile.mkdtemp(prefix="loader_csv_"))
        proj.mkdir(parents=True, exist_ok=True)
        (proj / "28c_human_decision_queue.csv").write_text(
            "decision_id,decision_type,blocking\nDQ-1,missing_required_target,True\n",
            encoding="utf-8")
        tf = _load_target_first_artifacts(proj)
        self.assertIsNotNone(tf["decision"])
        # Summary derived from CSV rows; CSV 'True' string coerced to a real bool.
        self.assertEqual(tf["decision"]["summary"]["blocking_decisions"], 1)


if __name__ == "__main__":
    unittest.main()
