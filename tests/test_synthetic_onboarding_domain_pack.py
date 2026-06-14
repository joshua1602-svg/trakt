#!/usr/bin/env python3
"""tests/test_synthetic_onboarding_domain_pack.py — PART 15 (25) + pack integrity."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
_TESTS_DIR = Path(__file__).resolve().parent
for _p in (str(_REPO_ROOT), str(_TESTS_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import pandas as pd

from onboarding_domain_fixtures import PACK, REGISTRY, SCENARIO_A, build_run
from engine.onboarding_agent import central_tape_builder


class TestPackIntegrity(unittest.TestCase):
    def test_scenario_a_files_present(self):
        a = PACK / SCENARIO_A
        for name in ("master_loan_collateral_tape.csv", "cashflow_report.csv",
                     "pipeline_report.csv", "warehouse_funding_agreement.md"):
            self.assertTrue((a / name).exists(), f"missing {name}")

    def test_scenario_b_files_present(self):
        b = PACK / "scenario_b_split"
        for name in ("loan_report.csv", "collateral_report.csv", "cashflow_report.csv",
                     "pipeline_report.csv", "warehouse_funding_agreement.md"):
            self.assertTrue((b / name).exists(), f"missing {name}")

    def test_master_tape_carries_collateral_columns(self):
        df = pd.read_csv(PACK / SCENARIO_A / "master_loan_collateral_tape.csv")
        for col in ("property_post_code", "collateral_region", "valuation_amount"):
            self.assertIn(col, df.columns)

    def test_pipeline_has_linked_and_application_only(self):
        df = pd.read_csv(PACK / SCENARIO_A / "pipeline_report.csv")
        linked = df["linked_loan_id"].fillna("").astype(str).str.strip()
        self.assertTrue((linked != "").any(), "expected some linked applications")
        self.assertTrue((linked == "").any(), "expected some application-only rows")


class TestReviewPack(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.project, cls.pdir, cls.rp = build_run(SCENARIO_A, mode="regulatory_mi",
                                                  ingest=True)
        # Build tapes + refresh the review pack with promotion results.
        from engine.onboarding_agent import promotion_planner
        from engine.onboarding_agent.review_pack_builder import refresh_review_pack_promotion
        res = central_tape_builder.build_central_tapes(cls.pdir, cls.rp, str(REGISTRY),
                                                       mode="regulatory_mi")
        promotion_planner.build_promotion_plan(
            cls.pdir, cls.rp, res, cls.project.domain_coverage, "regulatory_mi", False,
            client_name="CLIENT_X", project_id="client_x",
        )
        refresh_review_pack_promotion(cls.pdir, Path(cls.rp.output_root))
        cls.html = (cls.pdir / "08_onboarding_review_pack.html").read_text(encoding="utf-8")

    # 25. Review pack shows domain coverage, Azure metadata and central tape status.
    def test_review_pack_sections(self):
        self.assertIn("Data domain coverage", self.html)
        self.assertIn("Azure-ready run metadata", self.html)
        self.assertIn("Central tapes", self.html)
        # Azure metadata values surfaced.
        self.assertIn("client_x", self.html)
        self.assertIn("Storage backend", self.html)
        self.assertIn("Run ID", self.html)
        # Central tape status after promotion.
        self.assertIn("Central lender tape created", self.html)
        self.assertIn("Ready for MI", self.html)


if __name__ == "__main__":
    unittest.main()
