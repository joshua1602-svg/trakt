#!/usr/bin/env python3
"""tests/test_onboarding_domain_coverage.py — PART 15 (4–7)."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
_TESTS_DIR = Path(__file__).resolve().parent
for _p in (str(_REPO_ROOT), str(_TESTS_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from onboarding_domain_fixtures import SCENARIO_A, SCENARIO_B, build_run
from engine.onboarding_agent import domain_coverage as dc


def _by_domain(project):
    return {d.domain: d for d in project.domain_coverage}


class TestCombinedFile(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.project, cls.pdir, cls.rp = build_run(SCENARIO_A, ingest=False)

    # 4. Combined master loan/collateral file covers both loan and collateral.
    def test_master_tape_detected_as_loan_and_collateral(self):
        inv = {i.file_name: i for i in cls_inv(self.project)}
        master = inv["master_loan_collateral_tape.csv"]
        self.assertIn("loan", master.domains_detected)
        self.assertIn("collateral", master.domains_detected)

    # 5. Coverage does not require a separate collateral file.
    def test_collateral_not_missing_in_combined(self):
        cov = _by_domain(self.project)["collateral"]
        self.assertIn(cov.status, ("covered", "partially_covered"))
        self.assertTrue(cov.source_files)
        # Collateral is sourced from the combined master tape, not a separate file.
        self.assertIn("master_loan_collateral_tape.csv", cov.source_files)
        self.assertNotIn("file missing", cov.notes.lower())


class TestSplitFiles(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.project, cls.pdir, cls.rp = build_run(SCENARIO_B, ingest=False)

    # 6. Split loan/collateral files also produce coverage.
    def test_split_collateral_coverage(self):
        cov = _by_domain(self.project)
        self.assertIn(cov["collateral"].status, ("covered", "partially_covered"))
        self.assertIn(cov["loan"].status, ("covered", "partially_covered"))
        self.assertIn("collateral_report.csv", cov["collateral"].source_files)


class TestMissingDomain(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # A pack with only a loan file: pipeline domain is absent.
        tmp = Path(tempfile.mkdtemp(prefix="loanonly_"))
        src = _REPO_ROOT / "synthetic_onboarding_pack_domain_based" / SCENARIO_B / "loan_report.csv"
        (tmp / "loan_report.csv").write_text(src.read_text(encoding="utf-8"), encoding="utf-8")
        cls.project, cls.pdir, cls.rp = build_run(
            mode="mna_dd", ingest=False, input_dir=str(tmp)
        )

    # 7. Missing domain produces a domain gap, not a missing-file error.
    def test_pipeline_missing_is_domain_based(self):
        cov = _by_domain(self.project)["pipeline"]
        self.assertEqual(cov.status, "missing")
        # Plain-English, domain-based wording (never "expected pipeline file missing").
        self.assertIn("were found in the provided onboarding pack", cov.notes)
        self.assertNotIn("file missing", cov.notes.lower())


def cls_inv(project):
    return project.file_inventory


if __name__ == "__main__":
    unittest.main()
