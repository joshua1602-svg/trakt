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
        self.assertIn("Gate 3 — Target coverage summary", self.html)
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

    def test_decision_queue_before_full_target_matrix(self):
        # Operator-first: the compact queue (Gate 4) must precede the full matrix.
        pos_g4 = self.html.find("<h2>2. Gate 4 — Compact human decision queue")
        pos_g3 = self.html.find("<h2>3. Gate 3 — Target coverage summary")
        pos_matrix = self.html.find("Full target coverage matrix (audit/detail)")
        pos_appendix = self.html.find("<h2>6. Appendices")
        self.assertGreater(pos_g4, 0)
        self.assertGreater(pos_g3, pos_g4)
        self.assertGreater(pos_matrix, pos_g4)
        self.assertGreater(pos_appendix, pos_g3)

    def test_full_matrix_is_collapsed_detail_only(self):
        # The 72-row matrix must live inside a collapsible <details> block.
        self.assertIn("Full target coverage matrix (audit/detail)", self.html)
        idx = self.html.find("Full target coverage matrix (audit/detail)")
        pre = self.html[:idx]
        # The nearest preceding tag opening the disclosure must be <details>.
        self.assertEqual(pre.rfind("<details>"), pre.rfind("<details"),
                         "matrix label should sit inside a <details> block")
        self.assertGreater(pre.rfind("<details>"), pre.rfind("</details>"))

    def test_legacy_sections_are_appendix_only(self):
        appendix_start = self.html.find("<h2>6. Appendices")
        main, appendix = self.html[:appendix_start], self.html[appendix_start:]
        # Legacy diagnostics live only in the appendix.
        for s in ("Legacy readiness assessment", "Legacy / supporting gap questions",
                  "Deterministic mapping trace", "Mapping candidates"):
            self.assertNotIn(s, main, f"{s} leaked into the main body")
            self.assertIn(s, appendix)
        self.assertIn("superseded by target-first", appendix)
        # No legacy red blocking styling survives in the appendix.
        self.assertNotIn("badge b-block", appendix)
        self.assertNotIn("callout block", appendix)

    def test_headline_status_from_28c_not_legacy(self):
        # The headline (before the appendix) must be driven by 28c, never the
        # legacy "N blocking question(s)" gap-question banner / "Readiness: BLOCKED".
        main = self.html[: self.html.find("<h2>6. Appendices")]
        self.assertNotIn("blocking question(s)", main)
        self.assertNotIn("Readiness: BLOCKED", main)
        # This synthetic pack has exactly one genuine blocker.
        self.assertIn("Only ONE blocking target decision remains", main)

    def test_headline_counts_from_target_first(self):
        # Headline KPIs come from 28a/28b/28c.
        import json
        cov = json.loads((self.out / "28a_target_coverage_matrix.json").read_text())
        self.assertIn("Target fields", self.html)
        self.assertIn(str(cov["summary"]["target_fields_total"]), self.html)
        self.assertIn("Compact decision queue", self.html)
        self.assertIn("Old source-column approvals — audit only", self.html)

    def test_gate5_handoff_status_not_legacy_blocked(self):
        # Gate 5 MI handoff status is derived from 28c, never the legacy readiness.
        self.assertIn("Gate 5 — MI handoff readiness", self.html)
        self.assertIn("MI handoff:", self.html)

    def test_preserves_existing_supporting_sections(self):
        # Existing managed-service detail still present (re-homed under gates/appendix).
        for s in ("Data domain coverage", "Azure-ready run metadata",
                  "Field scope for this onboarding mode", "Central tapes",
                  "Mapping ambiguities resolved by policy"):
            self.assertIn(s, self.html)


class TestZeroBlockingHeadline(unittest.TestCase):
    """With a hand-built 28c that has zero blocking rows, the headline must be
    'NEEDS CONFIRMATION' with no red/legacy blocking language in the main body."""

    @classmethod
    def setUpClass(cls):
        import json
        from engine.onboarding_agent.onboarding_models import OnboardingProject
        from engine.onboarding_agent.review_pack_builder import build_review_pack
        cls.dir = Path(tempfile.mkdtemp(prefix="zero_block_"))
        # 28a: a couple of covered fields, no missing_required.
        (cls.dir / "28a_target_coverage_matrix.json").write_text(json.dumps({
            "target_contract_id": "mi_semantics_field_registry",
            "summary": {"target_fields_total": 2,
                        "coverage_status_counts": {"source_mapped": 1, "needs_confirmation": 1},
                        "source_mapped_fields": 1, "needs_confirmation_fields": 1,
                        "missing_required_fields": 0},
            "rows": [
                {"target_field": "a", "target_domain": "core", "required_status": "required",
                 "coverage_status": "source_mapped", "blocking": False},
                {"target_field": "b", "target_domain": "core", "required_status": "required",
                 "coverage_status": "needs_confirmation", "blocking": False},
            ]}), encoding="utf-8")
        # 28b: residuals.
        (cls.dir / "28b_source_residual_register.json").write_text(json.dumps({
            "summary": {"residual_source_columns_total": 3, "suppressed_from_main_queue": 3,
                        "operator_visible": 3, "residual_class_counts": {"not_relevant_to_current_mode": 3}},
            "rows": []}), encoding="utf-8")
        # 28c: 8 decisions, NONE blocking.
        (cls.dir / "28c_human_decision_queue.json").write_text(json.dumps({
            "summary": {"human_decision_rows_total": 8, "blocking_decisions": 0,
                        "decision_type_counts": {"source_priority_confirmation": 8}},
            "rows": [{"decision_id": f"DQ-{i}", "decision_type": "source_priority_confirmation",
                      "priority": "medium", "target_field": f"f{i}", "blocking": False,
                      "operator_question": "confirm", "recommendation": "ok",
                      "options": "", "evidence_summary": ""} for i in range(8)]}),
            encoding="utf-8")
        # A blocked legacy project, to prove legacy BLOCKED never leaks to the headline.
        proj = OnboardingProject(project_id="p", client_name="C", input_dir="i",
                                 output_dir=str(cls.dir), onboarding_mode="mi_only")
        proj.review_status = "blocked"
        build_review_pack(proj, cls.dir / "08_onboarding_review_pack.html")
        cls.html = (cls.dir / "08_onboarding_review_pack.html").read_text(encoding="utf-8")
        cls.main = cls.html[: cls.html.find("<h2>6. Appendices")]

    def test_headline_needs_confirmation_no_red(self):
        self.assertIn("NEEDS CONFIRMATION", self.main)
        self.assertIn("8 non-blocking confirmations remain", self.main)
        # No blocked/red language in the headline/main body.
        self.assertNotIn("Status: BLOCKED", self.main)
        self.assertNotIn("Readiness: BLOCKED", self.main)
        self.assertNotIn("BLOCKED</span>", self.main)

    def test_hero_badge_is_target_first_not_legacy_blocked(self):
        hero = self.html[: self.html.find("<h2>1.")]
        self.assertIn("NEEDS CONFIRMATION", hero)
        self.assertNotIn("BLOCKED", hero)

    def test_gate5_status_needs_confirmation(self):
        self.assertIn("MI handoff: NEEDS CONFIRMATION", self.main)


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
