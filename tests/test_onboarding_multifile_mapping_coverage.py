#!/usr/bin/env python3
"""tests/test_onboarding_multifile_mapping_coverage.py — multi-file coverage.

Proves the controlled mapping review runs across EVERY parseable tabular file
(not only the pipeline/KFI workbook), reports per-file coverage (29a), threads
source_file into the queue, handles multi-sheet workbooks, and explicitly
reports document-only (.docx) and unsupported (.xlsb) files.
"""

from __future__ import annotations

import shutil
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

from engine.onboarding_agent import streamlit_onboarding_workbench as wb
from engine.onboarding_agent.onboarding_orchestrator import run_onboarding

REGISTRY = str(_REPO_ROOT / "config" / "system" / "fields_registry.yaml")
ALIASES = str(_REPO_ROOT / "config" / "system")
KFI = _REPO_ROOT / "tests" / "fixtures" / "kfi_pipeline_headers.csv"


def _build_input() -> Path:
    warnings.simplefilter("ignore")
    d = Path(tempfile.mkdtemp()) / "input"
    d.mkdir(parents=True)
    # Funded loan tape (xlsx).
    pd.DataFrame({"loan_id": ["L1", "L2"], "current_balance": [100000, 200000],
                  "original_principal": [110000, 220000], "interest_rate": [4.5, 4.6],
                  "maturity_date": ["2045-01-01", "2046-01-01"]}).to_excel(
        d / "LoanExtract One - OMNI.xlsx", index=False)
    # Cashflow (xlsx).
    pd.DataFrame({"loan_id": ["L1", "L2"], "payment_date": ["2025-12-01", "2025-12-01"],
                  "principal_outstanding": [100000, 200000],
                  "scheduled_interest_payment": [500, 600]}).to_excel(
        d / "Principal And Interest - OMNI.xlsx", index=False)
    # Multi-sheet finance MI (xlsx).
    with pd.ExcelWriter(d / "Finance Report MI - ERE.xlsx") as xw:
        pd.DataFrame({"metric": ["AUM", "NIM"], "value": [1000, 3.2]}).to_excel(
            xw, sheet_name="Summary", index=False)
        pd.DataFrame({"month": ["2025-11", "2025-12"], "redemptions": [5, 7]}).to_excel(
            xw, sheet_name="Redemptions", index=False)
    # KFI/pipeline (csv stand-in for the workbook).
    shutil.copy(KFI, d / "M2L KFI and Pipeline.csv")
    # Document-only + unsupported.
    (d / "Schedule 8 Concentration.docx").write_text("Concentration limit 10%.")
    (d / "Redemptions Dec 2025.xlsb").write_bytes(b"\x00fake")
    return d


class TestMultiFileCoverage(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.inp = _build_input()
        cls.out = Path(tempfile.mkdtemp()) / "run"
        cls.project = run_onboarding(
            input_dir=str(cls.inp), client_name="CLIENT_001_TEST", output_dir=str(cls.out),
            registry_path=REGISTRY, aliases_dir=ALIASES, mode="mi_only",
            client_id="client_001", run_id="r1", enable_mapping_review=True)
        cls.ev = pd.read_csv(cls.out / "29_column_evidence.csv")
        cls.queue = pd.read_csv(cls.out / "33_mapping_review_queue.csv")
        cls.cov = pd.read_csv(cls.out / "29a_column_evidence_file_coverage.csv")

    # Evidence rows are created for ALL parseable tabular files (not just KFI).
    def test_evidence_across_all_parseable_files(self):
        files = set(self.ev["source_file"].unique())
        for f in ("M2L KFI and Pipeline.csv", "LoanExtract One - OMNI.xlsx",
                  "Principal And Interest - OMNI.xlsx", "Finance Report MI - ERE.xlsx"):
            self.assertIn(f, files, f"no evidence for {f}")

    # Multi-sheet workbook: both sheets profiled, with source_sheet set.
    def test_multi_sheet_profiled(self):
        fin = self.ev[self.ev["source_file"] == "Finance Report MI - ERE.xlsx"]
        sheets = set(fin["source_sheet"].dropna().unique())
        self.assertEqual(sheets, {"Summary", "Redemptions"})

    # The review queue carries source_file/source_sheet/domain columns.
    def test_queue_has_source_file(self):
        for col in ("source_file", "source_sheet", "source_column", "domain_guess",
                    "file_domain_guess", "group", "priority", "suggested_mapping",
                    "validation_status", "is_pipeline_field", "evidence_summary", "risk"):
            self.assertIn(col, self.queue.columns)
        self.assertGreater(self.queue["source_file"].nunique(), 1)

    # 29a reports included AND excluded files with reasons.
    def test_coverage_reports_all_files(self):
        cov = {r["file_name"]: r for _, r in self.cov.iterrows()}
        self.assertEqual(len(cov), 6)
        # docx -> document_only
        self.assertEqual(cov["Schedule 8 Concentration.docx"]["parse_status"], "document_only")
        # xlsb -> unsupported, explicit reason + next action
        xlsb = cov["Redemptions Dec 2025.xlsb"]
        self.assertEqual(xlsb["parse_status"], "unsupported_file_type")
        self.assertIn("xlsb parser unavailable", xlsb["reason_excluded"])
        self.assertIn("pyxlsb", xlsb["recommended_next_action"])
        # parsed files have evidence rows.
        self.assertGreater(cov["LoanExtract One - OMNI.xlsx"]["column_evidence_rows"], 0)

    # Non-pipeline MI files are NOT silently excluded in mi_only.
    def test_non_pipeline_files_not_excluded(self):
        cov = {r["file_name"]: r for _, r in self.cov.iterrows()}
        self.assertGreater(cov["LoanExtract One - OMNI.xlsx"]["review_queue_rows"], 0)
        self.assertGreater(cov["Principal And Interest - OMNI.xlsx"]["review_queue_rows"], 0)

    # The run summary records partial file coverage.
    def test_run_summary_file_coverage(self):
        fc = self.project.mapping_review_summary.get("file_coverage", {})
        self.assertEqual(fc.get("files_inventoried"), 6)
        self.assertEqual(fc.get("files_with_column_evidence"), 4)
        self.assertEqual(fc.get("files_excluded"), 2)


class TestWorkbenchFileCoverage(unittest.TestCase):
    # Workbench exposes the coverage summary + partial-coverage condition.
    def test_workbench_coverage_helpers(self):
        inp = _build_input()
        out = Path(tempfile.mkdtemp()) / "run"
        run_onboarding(input_dir=str(inp), client_name="C", output_dir=str(out),
                       registry_path=REGISTRY, aliases_dir=ALIASES, mode="mi_only",
                       client_id="c", run_id="r1", enable_mapping_review=True)
        coverage = wb.load_file_coverage(out)
        fc = wb.file_coverage_summary(coverage)
        self.assertEqual(fc["files_inventoried"], 6)
        self.assertEqual(fc["files_with_evidence"], 4)
        # Partial coverage -> the UI must warn (with_evidence < inventoried).
        self.assertLess(fc["files_with_evidence"], fc["files_inventoried"])


if __name__ == "__main__":
    unittest.main()
