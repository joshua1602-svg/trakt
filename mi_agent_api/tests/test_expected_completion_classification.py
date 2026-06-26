#!/usr/bin/env python3
"""mi_agent_api/tests/test_expected_completion_classification.py

The "Next expected completions" classification must distinguish overdue / current
/ next relative to the pipeline as-of month, so a PAST month is never labelled
"next". The chart breakdown itself stays unchanged.
"""

from __future__ import annotations

import sys
import unittest
import warnings
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import pandas as pd
import yaml

from mi_agent_api import pipeline_contract as pc
from mi_agent_api.pipeline_contract import _expected_completion_summary
from mi_agent_api.pipeline_prep import prepare_pipeline_mi_dataset

_BREAKDOWN = [
    {"month": "2025-10", "caseCount": 34, "expectedFundedAmount": 5_000_000.0, "weightedExpectedFundedAmount": 4_200_000.0},
    {"month": "2025-11", "caseCount": 12, "expectedFundedAmount": 1_200_000.0, "weightedExpectedFundedAmount": 900_000.0},
    {"month": "2025-12", "caseCount": 20, "expectedFundedAmount": 2_000_000.0, "weightedExpectedFundedAmount": 1_500_000.0},
    {"month": "2026-01", "caseCount": 8, "expectedFundedAmount": 800_000.0, "weightedExpectedFundedAmount": 500_000.0},
]


class TestExpectedCompletionClassification(unittest.TestCase):
    def test_past_month_not_selected_as_next(self):
        s = _expected_completion_summary(_BREAKDOWN, "2025-11-01")
        # 2025-10 is overdue, 2025-11 current, next is the first FUTURE month 2025-12.
        self.assertEqual(s["nextExpectedCompletionMonth"], "2025-12")
        self.assertNotEqual(s["nextExpectedCompletionMonth"], "2025-10")
        self.assertEqual(s["nextExpectedCompletionCount"], 20)

    def test_overdue_bucket_exposed_separately(self):
        s = _expected_completion_summary(_BREAKDOWN, "2025-11-01")
        self.assertEqual(s["overdueExpectedCompletionCount"], 34)
        self.assertEqual(s["overdueExpectedCompletionWeightedAmount"], 4_200_000.0)
        self.assertEqual(s["currentMonthExpectedCompletionCount"], 12)

    def test_only_past_buckets_next_is_null(self):
        past = [{"month": "2025-09", "caseCount": 5, "weightedExpectedFundedAmount": 100.0},
                {"month": "2025-10", "caseCount": 3, "weightedExpectedFundedAmount": 50.0}]
        s = _expected_completion_summary(past, "2025-11-01")
        self.assertIsNone(s["nextExpectedCompletionMonth"])
        self.assertEqual(s["nextExpectedCompletionCount"], 0)
        self.assertEqual(s["overdueExpectedCompletionCount"], 8)

    def test_snapshot_exposes_classification_and_chart_unchanged(self):
        warnings.simplefilter("ignore")
        # A pipeline frame with completion months Oct/Nov/Dec and an as-of of Nov.
        n = 9
        df = pd.DataFrame({
            "Account Number": [f"A{i}" for i in range(n)],
            "KFI Number": [f"K{i}" for i in range(n)],
            "Status": ["Offer"] * n, "Loan Amount": [100000.0] * n,
            "Date Funds Released": (["2025-10-15"] * 3 + ["2025-11-15"] * 3 + ["2025-12-15"] * 3),
        })
        prep, rep = prepare_pipeline_mi_dataset(df, as_of_date="2025-11-01")
        semantics = yaml.safe_load(
            (_REPO_ROOT / "mi_agent" / "mi_semantics_field_registry.yaml").read_text())
        snap = pc.compute_pipeline_snapshot(prep, rep, semantics, client_id="client_001",
                                            run_id="mi_2025_11",
                                            source={"pipeline_as_of_date": "2025-11-01"})
        # Chart breakdown is unchanged: still all months ascending.
        months = [r["month"] for r in snap["expectedCompletionBreakdown"]]
        self.assertEqual(months, ["2025-10", "2025-11", "2025-12"])
        # Next is the first future month (2025-12), not the past 2025-10.
        self.assertEqual(snap["nextExpectedCompletionMonth"], "2025-12")
        self.assertEqual(snap["overdueExpectedCompletionCount"], 3)
        self.assertEqual(snap["currentMonthExpectedCompletionCount"], 3)
        self.assertEqual(snap["expectedCompletionSummary"]["asOfMonth"], "2025-11")


if __name__ == "__main__":
    unittest.main(verbosity=2)
