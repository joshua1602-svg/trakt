#!/usr/bin/env python3
"""tests/test_onboarding_column_evidence.py — PART 13 (2, 3, 4, 5)."""

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

from engine.onboarding_agent import column_evidence as ce

FIXTURE = _REPO_ROOT / "tests" / "fixtures" / "kfi_pipeline_headers.csv"


class TestColumnEvidence(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        warnings.simplefilter("ignore")
        df = pd.read_csv(FIXTURE)
        cls.ev = {e["source_column"]: e for e in ce.build_column_evidence(df, "kfi.csv")}

    # 2. Evidence detects dates, identifiers, amounts, enums, null rates.
    def test_type_detection(self):
        self.assertEqual(self.ev["Offer Date"]["data_type_guess"], "date")
        self.assertEqual(self.ev["Account Number"]["data_type_guess"], "identifier")
        self.assertEqual(self.ev["KFI Number"]["data_type_guess"], "identifier")
        self.assertEqual(self.ev["Loan Amount"]["data_type_guess"], "amount")
        self.assertIn(self.ev["Status"]["data_type_guess"], ("enum",))
        self.assertEqual(self.ev["Gender APP 1"]["data_type_guess"], "enum")
        # null rate is captured (App 2 fields are partly null).
        self.assertGreater(self.ev["Gender APP 2"]["null_rate"], 0.0)

    # 3. Product Rate and Interest Payment Percentage are distinguished.
    def test_rate_vs_percentage(self):
        self.assertEqual(self.ev["Product Rate"]["data_type_guess"], "rate")
        self.assertEqual(self.ev["Interest Payment Percentage"]["data_type_guess"], "percentage")
        self.assertGreaterEqual(self.ev["Product Rate"]["rate_like_score"], 0.5)
        self.assertGreaterEqual(self.ev["Interest Payment Percentage"]["percentage_like_score"], 0.6)

    # 4. KFI/Application/Offer/Funds Released dates produce chronology evidence.
    def test_date_chronology(self):
        chrono = self.ev["KFI Submitted Date"]["chronology_relationships"]
        self.assertIn("Application Submitted Date", chrono)
        self.assertIn("Offer Date", chrono)

    # 5. Loan Amount / Max Facility / Max Entitlement produce amount relationships.
    def test_amount_relationships(self):
        rel = self.ev["Loan Amount"]["amount_relationships"]
        self.assertIn("Max Facility", rel)
        self.assertIn("Max Entitlement", rel)
        self.assertIn("Estimated Value", rel)

    def test_artefacts_written(self):
        out = Path(tempfile.mkdtemp())
        df = pd.read_csv(FIXTURE)
        paths = ce.write_evidence_artifacts(ce.build_column_evidence(df, "kfi.csv"), out)
        for k in ("csv", "json", "summary_md"):
            self.assertTrue(Path(paths[k]).exists())

    def test_postcode_and_gender_scores(self):
        self.assertGreaterEqual(self.ev["Gender APP 1"]["gender_like_score"], 0.8)
        self.assertGreaterEqual(self.ev["Status"]["stage_like_score"], 0.6)


if __name__ == "__main__":
    unittest.main()
