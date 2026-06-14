#!/usr/bin/env python3
"""tests/test_onboarding_file_signature_diagnostics.py — fake-xlsx / OLE handling.

A lender file named *.xlsx whose bytes are an OLE compound document must be
diagnosed explicitly (extension mismatch, parsers attempted, parse error,
conversion availability, next action) — never silently dropped or left with a
blank reason. A true OOXML .xlsx must still parse and reach the review queue.
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

from engine.onboarding_agent import source_table_loader as stl
from engine.onboarding_agent import streamlit_onboarding_workbench as wb
from engine.onboarding_agent.onboarding_orchestrator import run_onboarding

REGISTRY = str(_REPO_ROOT / "config" / "system" / "fields_registry.yaml")
ALIASES = str(_REPO_ROOT / "config" / "system")
KFI = _REPO_ROOT / "tests" / "fixtures" / "kfi_pipeline_headers.csv"
OLE = bytes([0xD0, 0xCF, 0x11, 0xE0, 0xA1, 0xB1, 0x1A, 0xE1])


def _inv(path: Path, ftype: str = ""):
    return [{"file_path": str(path), "file_name": path.name,
             "file_type": ftype or path.suffix.lstrip("."),
             "classification": "unknown", "domains_detected": []}]


class TestSignatureDetection(unittest.TestCase):
    def test_signatures(self):
        d = Path(tempfile.mkdtemp())
        (d / "ole.xlsx").write_bytes(OLE + b"\x00" * 64)
        (d / "zip.xlsx").write_bytes(b"PK\x03\x04rest")
        (d / "data.csv").write_text("a,b,c\n1,2,3\n")
        (d / "page.xlsx").write_text("<html><table><tr><td>1</td></tr></table></html>")
        (d / "x.xml").write_text("<?xml version='1.0'?><Workbook></Workbook>")
        self.assertEqual(stl.detect_container_type(d / "ole.xlsx"), stl.C_OLE)
        self.assertEqual(stl.detect_container_type(d / "zip.xlsx"), stl.C_ZIP)
        self.assertEqual(stl.detect_container_type(d / "data.csv"), stl.C_CSV)
        self.assertEqual(stl.detect_container_type(d / "page.xlsx"), stl.C_HTML)
        self.assertEqual(stl.detect_container_type(d / "x.xml"), stl.C_XML)


class TestOleFakeXlsx(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        d = Path(tempfile.mkdtemp())
        cls.ole = d / "LoanExtract One - OMNI.xlsx"
        cls.ole.write_bytes(OLE + b"\x00" * 2048)
        cls.cov = stl.load_source_tables(_inv(cls.ole, "xlsx"))[1][0]

    def test_extension_mismatch_flagged(self):
        self.assertEqual(self.cov.declared_extension, ".xlsx")
        self.assertEqual(self.cov.detected_container_type, stl.C_OLE)
        self.assertTrue(self.cov.extension_mismatch_detected)
        self.assertEqual(self.cov.detected_excel_format,
                         "ole_compound_unknown_or_unreadable_excel")

    def test_parsers_attempted_and_error(self):
        self.assertIn("openpyxl", self.cov.parser_attempted)
        self.assertIn("xlrd", self.cov.parser_attempted)
        self.assertEqual(self.cov.engine_used, "none")
        self.assertEqual(self.cov.parse_status, "parse_error")
        self.assertTrue(self.cov.parse_error.strip(), "parse_error must not be blank")

    def test_recommended_action_explicit(self):
        self.assertIn("OLE compound", self.cov.recommended_next_action)
        self.assertIn("resave", self.cov.recommended_next_action.lower())


class TestConversionAvailability(unittest.TestCase):
    def setUp(self):
        self.ole = Path(tempfile.mkdtemp()) / "x.xlsx"
        self.ole.write_bytes(OLE + b"\x00" * 64)

    def test_conversion_unavailable_reported(self):
        cov = stl.load_source_tables(_inv(self.ole, "xlsx"),
                                     converter=(False, "none"))[1][0]
        self.assertFalse(cov.conversion_available)
        self.assertEqual(cov.conversion_tool, "none")
        self.assertFalse(cov.conversion_attempted)

    def test_fallback_without_converter_does_not_crash(self):
        cov = stl.load_source_tables(_inv(self.ole, "xlsx"), enable_conversion=True,
                                     converter=(False, "none"))[1][0]
        self.assertEqual(cov.conversion_status, "unavailable")
        self.assertEqual(cov.conversion_error, "No libreoffice/soffice executable found")
        # The run still produced a coverage record (no exception).
        self.assertEqual(cov.parse_status, "parse_error")


class TestTrueXlsxUnchanged(unittest.TestCase):
    def test_true_xlsx_parses(self):
        warnings.simplefilter("ignore")
        d = Path(tempfile.mkdtemp())
        p = d / "M2L KFI and Pipeline.xlsx"
        pd.read_csv(KFI).to_excel(p, index=False)
        tables, cov, sheets = stl.load_source_tables(_inv(p, "xlsx"))
        c = cov[0]
        self.assertEqual(c.detected_container_type, stl.C_ZIP)
        self.assertFalse(c.extension_mismatch_detected)
        self.assertEqual(c.engine_used, "openpyxl")
        self.assertEqual(c.parse_status, "parsed")
        self.assertTrue(tables)


class TestEndToEndDiagnostics(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        warnings.simplefilter("ignore")
        d = Path(tempfile.mkdtemp()) / "input"
        d.mkdir(parents=True)
        for nm in ("Finance Report MI - ERE.xlsx", "Principal And Interest - OMNI.xlsx"):
            (d / nm).write_bytes(OLE + b"\x00" * 2048)
        pd.read_csv(KFI).to_excel(d / "M2L KFI and Pipeline.xlsx", index=False)
        (d / "Schedule 8.docx").write_text("Concentration limit.")
        cls.out = Path(tempfile.mkdtemp()) / "run"
        run_onboarding(input_dir=str(d), client_name="C", output_dir=str(cls.out),
                       registry_path=REGISTRY, aliases_dir=ALIASES, mode="mi_only",
                       client_id="c", run_id="r1", enable_mapping_review=True)
        cls.cov = pd.read_csv(cls.out / "29a_column_evidence_file_coverage.csv")
        cls.sheets = pd.read_csv(cls.out / "29b_excel_sheet_parse_coverage.csv")

    def test_29a_and_29b_written(self):
        self.assertTrue((self.out / "29a_column_evidence_file_coverage.csv").exists())
        self.assertTrue((self.out / "29b_excel_sheet_parse_coverage.csv").exists())

    def test_29a_no_blank_parse_error(self):
        pe = self.cov[self.cov["parse_status"] == "parse_error"]
        self.assertTrue(len(pe) >= 2)
        blanks = pe["parse_error"].isna() | (pe["parse_error"].astype(str).str.strip() == "")
        self.assertEqual(int(blanks.sum()), 0, "parse_error rows must never be blank")
        nexts = pe["recommended_next_action"].astype(str).str.strip()
        self.assertTrue((nexts != "").all(), "every parse_error needs a next action")

    def test_true_xlsx_in_review_queue(self):
        m2l = self.cov[self.cov["file_name"] == "M2L KFI and Pipeline.xlsx"].iloc[0]
        self.assertEqual(m2l["parse_status"], "parsed")
        self.assertGreater(m2l["column_evidence_rows"], 0)
        self.assertTrue(m2l["included_in_review_queue"])

    def test_workbench_surfaces_diagnostics(self):
        coverage = wb.load_file_coverage(self.out)
        self.assertTrue(coverage)
        ole = next(c for c in coverage if c["detected_container_type"] == "ole_compound")
        for key in ("declared_extension", "detected_container_type", "parser_attempted",
                    "parse_error", "conversion_available", "recommended_next_action"):
            self.assertIn(key, ole)
        fc = wb.file_coverage_summary(coverage)
        self.assertLess(fc["files_with_evidence"], fc["files_inventoried"])


if __name__ == "__main__":
    unittest.main()
