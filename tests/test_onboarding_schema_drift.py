#!/usr/bin/env python3
"""tests/test_onboarding_schema_drift.py — PART 13 (21, 22)."""

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
from engine.onboarding_agent import schema_drift as sd

FIXTURE = _REPO_ROOT / "tests" / "fixtures" / "kfi_pipeline_headers.csv"


def _evidence(df):
    warnings.simplefilter("ignore")
    return ce.build_column_evidence(df, "kfi.csv")


class TestSchemaDrift(unittest.TestCase):
    def setUp(self):
        self.mem = Path(tempfile.mkdtemp())
        self.df = pd.read_csv(FIXTURE)

    # 22. Repeat run with no changes asks only about nothing new (all unchanged).
    def test_repeat_run_unchanged(self):
        ev = _evidence(self.df)
        sd.save_signature(sd.build_signature(ev), self.mem)
        prev = sd.load_signature(self.mem)
        rows = sd.detect_drift(ev, prev)
        self.assertTrue(all(r["drift_status"] == sd.UNCHANGED for r in rows))
        self.assertEqual(sd.columns_needing_review(rows), set())

    # 21. Schema drift detects a changed value profile for a known column.
    def test_value_profile_changed(self):
        ev1 = _evidence(self.df)
        sd.save_signature(sd.build_signature(ev1), self.mem)
        prev = sd.load_signature(self.mem)
        # Same numeric type, but the value profile shifts sharply (null rate jump).
        df2 = self.df.copy()
        df2["Loan Amount"] = [150000.0, None, None, None, None]
        ev2 = _evidence(df2)
        rows = sd.detect_drift(ev2, prev)
        by_col = {r["source_column"]: r for r in rows}
        self.assertEqual(by_col["Loan Amount"]["drift_status"], sd.VALUE_PROFILE_CHANGED)
        self.assertIn("Loan Amount", sd.columns_needing_review(rows))
        # Unchanged columns are not re-flagged.
        self.assertEqual(by_col["Offer Date"]["drift_status"], sd.UNCHANGED)

    # 22b. A new column is detected and routed to review.
    def test_new_column_detected(self):
        ev1 = _evidence(self.df)
        sd.save_signature(sd.build_signature(ev1), self.mem)
        prev = sd.load_signature(self.mem)
        df2 = self.df.copy()
        df2["Brand New Column"] = [1, 2, 3, 4, 5]
        rows = sd.detect_drift(_evidence(df2), prev)
        by_col = {r["source_column"]: r for r in rows}
        self.assertEqual(by_col["Brand New Column"]["drift_status"], sd.NEW_COLUMN)

    # Missing previously-mapped column is detected.
    def test_missing_column_detected(self):
        ev1 = _evidence(self.df)
        sd.save_signature(sd.build_signature(ev1), self.mem)
        prev = sd.load_signature(self.mem)
        df2 = self.df.drop(columns=["Offer Date"])
        rows = sd.detect_drift(_evidence(df2), prev)
        by_col = {r["source_column"]: r for r in rows}
        self.assertEqual(by_col["Offer Date"]["drift_status"], sd.MISSING_COLUMN)

    def test_first_run_all_new(self):
        rows = sd.detect_drift(_evidence(self.df), None)
        self.assertTrue(all(r["drift_status"] == sd.NEW_COLUMN for r in rows))


if __name__ == "__main__":
    unittest.main()
