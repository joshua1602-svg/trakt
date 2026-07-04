#!/usr/bin/env python3
"""tests/test_borrower_type_and_dob_age.py

Two intertwined concerns for lifetime-mortgage MI:

  1. UK day-first DOB parsing. Raw M2L extracts carry dd/mm/yyyy dates. Parsing
     them month-first (dayfirst=False) silently dropped every DOB whose day > 12
     to NaT and month/day-swapped the rest, which zeroed out youngest_borrower_age
     for ~60% of rows — the field that drives NNEG (no-negative-equity-guarantee)
     youngest-life exposure. Both funded and pipeline preps must parse day-first.

  2. borrower_type (single vs joint) is derived from whether ANY second-applicant
     field is populated, and is registered as a first-class categorical dimension
     on BOTH the pipeline and funded datasets so the MI Agent can run single-vs-joint
     cohort analysis and stratifications (e.g. LTV by borrower_type).

Run: python -m unittest tests.test_borrower_type_and_dob_age
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import pandas as pd

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from mi_agent_api.funded_prep import CORE_FUNDED_DIMENSIONS, prepare_funded_mi_dataset
from mi_agent_api.pipeline_prep import _DIMENSION_FIELDS, prepare_pipeline_mi_dataset


class TestPipelineDayFirstDob(unittest.TestCase):

    def test_day_gt_12_dobs_parse_and_age_is_populated_for_all_rows(self):
        # Rows 2,3,4 have day > 12 — month-first parsing would drop these to NaT.
        df = pd.DataFrame({
            "Deal ID": ["D1", "D2", "D3", "D4", "D5"],
            "Loan Amount": [250000, 180000, 320000, 150000, 410000],
            "Property Value": [500000, 400000, 640000, 300000, 700000],
            "DOB App 1": ["12/03/1955", "08/07/1960", "22/10/1947",
                          "14/05/1946", "18/03/1950"],
        })
        out, _rep = prepare_pipeline_mi_dataset(df, as_of_date="2025-09-08")
        self.assertIn("youngest_borrower_age", out.columns)
        # No NaT ages: every row parsed, including the day>12 DOBs.
        self.assertEqual(int(out["youngest_borrower_age"].isna().sum()), 0)
        # 22/10/1947 as of 2025-09-08 is 77 completed years (not month/day-swapped).
        self.assertEqual(int(out.loc[2, "youngest_borrower_age"]), 77)


class TestPipelineBorrowerType(unittest.TestCase):

    def _prep(self):
        df = pd.DataFrame({
            "Deal ID": ["D1", "D2", "D3", "D4"],
            "Loan Amount": [250000, 180000, 320000, 150000],
            "Property Value": [500000, 400000, 640000, 300000],
            "DOB App 1": ["12/03/1955", "22/10/1947", "14/05/1946", "18/03/1950"],
            "DOB App 2": ["15/06/1957", "", "20/08/1952", None],
        })
        return prepare_pipeline_mi_dataset(df, as_of_date="2025-09-08")

    def test_joint_when_second_applicant_present_single_otherwise(self):
        out, _rep = self._prep()
        self.assertIn("borrower_type", out.columns)
        self.assertEqual(list(out["borrower_type"]),
                         ["joint", "single", "joint", "single"])

    def test_registered_as_dimension_and_reported_available(self):
        out, rep = self._prep()
        self.assertIn("borrower_type", _DIMENSION_FIELDS)
        self.assertIn("borrower_type", rep.get("dimensions_available", []))

    def test_not_derivable_without_second_applicant_column(self):
        # No second-applicant column at all → borrower_type is not fabricated.
        df = pd.DataFrame({
            "Deal ID": ["D1", "D2"],
            "Loan Amount": [250000, 180000],
            "Property Value": [500000, 400000],
            "DOB App 1": ["12/03/1955", "22/10/1947"],
        })
        out, _rep = prepare_pipeline_mi_dataset(df, as_of_date="2025-09-08")
        self.assertNotIn("borrower_type", out.columns)


class TestFundedBorrowerType(unittest.TestCase):

    def _prep(self):
        df = pd.DataFrame({
            "loan_id": ["L1", "L2", "L3", "L4"],
            "current_outstanding_balance": [200000, 150000, 300000, 180000],
            "current_valuation_amount": [400000, 300000, 600000, 360000],
            "reporting_date": ["2025-11-30"] * 4,
            "borrower_1_DOB": ["12/03/1955", "22/10/1947", "14/05/1946", "18/03/1950"],
            "borrower_2_DOB": ["15/06/1957", "", "20/08/1952", None],
        })
        return prepare_funded_mi_dataset(df)

    def test_joint_single_derivation(self):
        out, _rep = self._prep()
        self.assertIn("borrower_type", out.columns)
        self.assertEqual(list(out["borrower_type"]),
                         ["joint", "single", "joint", "single"])

    def test_core_dimension_and_reported_available(self):
        out, rep = self._prep()
        self.assertIn("borrower_type", CORE_FUNDED_DIMENSIONS)
        self.assertIn("borrower_type", rep.get("dimensions_available", []))

    def test_funded_dob_day_first(self):
        # borrower_2_DOB 20/08/1952 (day>12) must parse; youngest age populated.
        out, _rep = self._prep()
        self.assertEqual(int(out["youngest_borrower_age"].isna().sum()), 0)


if __name__ == "__main__":
    unittest.main()
