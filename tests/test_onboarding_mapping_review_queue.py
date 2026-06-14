#!/usr/bin/env python3
"""tests/test_onboarding_mapping_review_queue.py — PART 13 (6, 13, 14, 15)."""

from __future__ import annotations

import sys
import tempfile
import unittest
import warnings
from pathlib import Path

import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parent.parent
for _p in (str(_REPO_ROOT), str(_REPO_ROOT / "tests")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from engine.onboarding_agent.llm_assisted_mapping import run_llm_assisted_mapping
from engine.onboarding_agent import streamlit_onboarding_workbench as wb

FIXTURE = str(_REPO_ROOT / "tests" / "fixtures" / "kfi_pipeline_headers.csv")


class TestReviewQueue(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        warnings.simplefilter("ignore")
        cls.out = Path(tempfile.mkdtemp())
        cls.res = run_llm_assisted_mapping(
            input_file=FIXTURE, output_dir=str(cls.out), mode="regulatory_mi",
            client_id="kfi", run_id="r1")
        cls.queue = cls.res["review_queue"]

    # 6. Candidate shortlist includes the expected deterministic sources.
    def test_shortlist_sources(self):
        sources = {r["candidate_source"] for r in self.res["shortlist"]}
        self.assertIn("alias", sources)
        self.assertIn("pipeline_contract", sources)
        # semantic_alignment and registry/value_profile appear across the file.
        self.assertTrue({"semantic_alignment", "registry_description", "value_profile"} & sources)

    # 13. The queue groups high-confidence, review, missing-target and OOS items.
    def test_groups_present(self):
        groups = {it["group"] for it in self.queue["items"]}
        self.assertIn("high_confidence_approvals", groups)
        self.assertIn("missing_trakt_fields", groups)

    # 14. The queue is concise + prioritised (not a flat wall): a top summary with
    #     an estimated review time, and items sorted by priority.
    def test_concise_and_prioritised(self):
        s = self.queue["summary"]
        for key in ("total_columns_reviewed", "auto_approved", "needs_review",
                    "high_priority_decisions", "estimated_review_minutes"):
            self.assertIn(key, s)
        priorities = [it["priority"] for it in self.queue["items"]]
        self.assertEqual(priorities, sorted(priorities))
        # Most KFI columns auto-approve against the known pipeline contract, so the
        # review burden is small.
        self.assertLessEqual(s["needs_review"], s["total_columns_reviewed"] // 2)

    # 15. Bulk approve only approves safe high-confidence (auto-approved) mappings.
    def test_bulk_approve_only_safe(self):
        wb.write_queue_decisions = getattr(wb, "write_queue_decisions", None)  # noop guard
        safe = wb.bulk_approve_safe_mappings({"items": self.queue["items"]})
        # Every bulk-approved item came from an auto_approved_candidate row.
        auto_cols = {it["source_column"] for it in self.queue["items"]
                     if it["validation_status"] == "auto_approved_candidate"}
        self.assertEqual({s["source_column"] for s in safe}, auto_cols)
        # No review-required or missing-target item is ever bulk-approved.
        review_cols = {it["source_column"] for it in self.queue["items"]
                       if it["validation_status"] != "auto_approved_candidate"}
        self.assertFalse({s["source_column"] for s in safe} & review_cols)

    def test_queue_artefacts_written(self):
        for name in ("33_mapping_review_queue.csv", "33_mapping_review_queue.json",
                     "34_mapping_review_decisions.yaml", "35_mapping_review_action_log.json"):
            self.assertTrue((self.out / name).exists())


if __name__ == "__main__":
    unittest.main()
