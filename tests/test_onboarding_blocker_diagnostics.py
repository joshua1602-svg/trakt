#!/usr/bin/env python3
"""tests/test_onboarding_blocker_diagnostics.py

Operator-first "Why is the MI pipeline blocked?" diagnostics.

Covers:
  * source-pack composition classification (pipeline-only vs funded present);
  * a funded-book blocker (origination_date) on a pipeline-only pack yields a
    plain-English reason, source-pack context and the A/B/C next actions;
  * the compact CLI text answers what/why/which-file/action and render outlook;
  * the structured report flags central artifacts as NOT rendered while blocked;
  * a pack WITH funded files (and no blockers) reads as renderable;
  * the HTML review pack shows a first-screen "Why blocked?" box (before the
    executive summary) and collapses the source-column audit detail.
"""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from engine.onboarding_agent import blocker_diagnostics as bd


def _pipeline_only_project(tmp: Path, *, n_pipeline=12, blocking_field="origination_date"):
    inv = [{"file_name": f"M2L_KFI_{i}.csv", "classification": "pipeline_report"}
           for i in range(n_pipeline)]
    (tmp / "01_file_inventory.json").write_text(json.dumps(inv))
    rows = []
    if blocking_field:
        rows = [{"decision_id": "TFD1", "target_field": blocking_field,
                 "decision_type": "missing_required_target", "blocking": True,
                 "operator_question": f"No source for {blocking_field}"}]
    (tmp / "28c_human_decision_queue.json").write_text(
        json.dumps({"summary": {"blocking_decisions": len(rows)}, "rows": rows}))


# --------------------------------------------------------------------------- #
# Source-pack composition
# --------------------------------------------------------------------------- #
class TestSourcePackComposition(unittest.TestCase):
    def test_pipeline_only_pack(self):
        comp = bd.classify_source_pack(["pipeline_report"] * 12)
        self.assertEqual(comp["buckets"]["pipeline_report"], 12)
        self.assertEqual(comp["funded_source_count"], 0)
        self.assertFalse(comp["funded_present"])
        self.assertTrue(comp["pipeline_only"])

    def test_funded_present_pack(self):
        comp = bd.classify_source_pack(
            ["pipeline_report", "current_loan_report", "collateral_report",
             "cashflow_report", "warehouse_agreement", "data_dictionary"])
        self.assertEqual(comp["buckets"]["funded_loan_tape"], 1)
        self.assertEqual(comp["buckets"]["property_tape"], 1)
        self.assertEqual(comp["buckets"]["cashflow_funder_tape"], 2)
        self.assertEqual(comp["buckets"]["docs"], 1)
        self.assertTrue(comp["funded_present"])
        self.assertFalse(comp["pipeline_only"])


# --------------------------------------------------------------------------- #
# Blocker analysis
# --------------------------------------------------------------------------- #
class TestBlockerAnalysis(unittest.TestCase):
    def test_origination_date_on_pipeline_only(self):
        rows = [{"target_field": "origination_date",
                 "decision_type": "missing_required_target", "blocking": True}]
        rep = bd.analyze_blockers(rows, ["pipeline_report"] * 12)
        self.assertEqual(rep["status"], "BLOCKED")
        self.assertTrue(rep["is_blocked"])
        self.assertEqual(rep["blocking_count"], 1)
        item = rep["blocking_items"][0]
        self.assertEqual(item["target_field"], "origination_date")
        self.assertIn("required funded MI field has no source", item["reason"])
        self.assertIn("no funded loan tape detected", item["source_pack_context"])
        self.assertIn("pipeline-only mode", item["suggested_action"])
        # "Because" bullets cover field, funded-source absence and pipeline-only.
        joined = " | ".join(rep["because"])
        self.assertIn("Missing required target field: origination_date", joined)
        self.assertIn("No funded-book source file", joined)
        self.assertIn("pipeline-only", joined)
        # Artifacts must not render while blocked.
        self.assertFalse(rep["central_artifacts_rendered"])
        self.assertEqual(rep["reason_not_rendered"], "blocking Gate 4 decision remains.")

    def test_not_blocked_pack_renders(self):
        rows = [{"target_field": "broker_channel",
                 "decision_type": "source_priority_confirmation", "blocking": False}]
        rep = bd.analyze_blockers(rows, ["current_loan_report", "pipeline_report"])
        self.assertFalse(rep["is_blocked"])
        self.assertEqual(rep["non_blocking_count"], 1)
        self.assertTrue(rep["central_artifacts_rendered"])
        self.assertEqual(rep["reason_not_rendered"], "")


# --------------------------------------------------------------------------- #
# CLI rendering + disk loading
# --------------------------------------------------------------------------- #
class TestCliRendering(unittest.TestCase):
    def test_compact_cli_output(self):
        with tempfile.TemporaryDirectory() as d:
            tmp = Path(d)
            _pipeline_only_project(tmp)
            rep = bd.load_blocker_report(tmp)
            text = bd.format_cli(rep)
            self.assertIn("Status: BLOCKED", text)
            self.assertIn("1. origination_date", text)
            self.assertIn("Reason: required funded MI field has no source", text)
            self.assertIn("Source-pack context: no funded loan tape detected", text)
            self.assertIn("Suggested action: add funded LoanExtract", text)
            self.assertIn("Non-blocking confirmations: 0", text)
            self.assertIn("pipeline_report: 12", text)
            self.assertIn("funded_loan_tape: 0", text)
            self.assertIn("no funded source files detected", text.lower())
            self.assertIn("Central artifacts rendered: no", text)
            self.assertIn("Reason artifacts not rendered: blocking Gate 4 decision remains.",
                          text)

    def test_load_report_marks_inventory_and_queue_present(self):
        with tempfile.TemporaryDirectory() as d:
            tmp = Path(d)
            _pipeline_only_project(tmp)
            rep = bd.load_blocker_report(tmp)
            self.assertTrue(rep["decision_queue_present"])
            self.assertTrue(rep["inventory_present"])


# --------------------------------------------------------------------------- #
# HTML review-pack first-screen box
# --------------------------------------------------------------------------- #
class TestReviewPackWhyBlockedBox(unittest.TestCase):
    def test_first_screen_box_and_collapsed_audit(self):
        from engine.onboarding_agent.onboarding_models import (
            OnboardingProject, FileInventoryItem)
        from engine.onboarding_agent import review_pack_builder as rpb

        with tempfile.TemporaryDirectory() as d:
            tmp = Path(d)
            dec = {"summary": {"blocking_decisions": 1}, "rows": [
                {"decision_id": "TFD1", "target_field": "origination_date",
                 "decision_type": "missing_required_target", "blocking": True,
                 "operator_question": "No source for origination_date"}]}
            (tmp / "28c_human_decision_queue.json").write_text(json.dumps(dec))

            p = OnboardingProject(client_name="client_001",
                                  project_id="client_001/mi_2025_11")
            p.onboarding_mode = "mi_only"
            p.input_dir = str(tmp / "input")
            p.output_dir = str(tmp)
            p.file_inventory = [
                FileInventoryItem(file_name=f"M2L_KFI_{i}.csv",
                                  classification="pipeline_report",
                                  row_count=10, column_count=20) for i in range(12)]

            out = rpb.build_review_pack(p, tmp / "review_pack.html", output_root=tmp)
            html = Path(out).read_text()

            # The "Why blocked?" box leads the document (before the exec summary).
            self.assertIn("Why blocked? — operator summary", html)
            self.assertLess(html.index("Why blocked?"),
                            html.index("1. Executive onboarding summary"))
            # Plain-English content + A/B/C actions + composition + render outlook.
            self.assertIn("BLOCKED because:", html)
            self.assertIn("Missing required target field: origination_date", html)
            self.assertIn("Option A — Full MI", html)
            self.assertIn("LoanExtract", html)
            self.assertIn("Option B — Pipeline-only", html)
            self.assertIn("Option C — Manual mapping", html)
            self.assertIn("Date Funds Released", html)
            self.assertIn("Source-pack composition", html)
            self.assertIn("No funded source files detected", html)
            self.assertIn("Will artifacts render?", html)
            # Audit detail (matrix / residual / source-column queue) is collapsed.
            self.assertIn("Show full audit detail", html)


if __name__ == "__main__":
    unittest.main(verbosity=2)
