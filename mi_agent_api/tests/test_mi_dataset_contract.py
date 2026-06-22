#!/usr/bin/env python3
"""mi_agent_api/tests/test_mi_dataset_contract.py

Stage 8-13 consolidation regression cover:

  * the single dataset contract reports per-field semantic type + storage scale;
  * bubble artifacts carry explicit xKey / yKey / sizeKey (+ labels + scale hint);
  * fractional percents format as points (0.51 -> 51.0%);
  * a missing/empty metric/dimension fails data validation (not "Passed");
  * the filtered-count intent ("how many ... more than N") answers a number;
  * duplicate columns surface as controlled validation, never a raw 500.
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

from mi_agent.mi_agent_workflow import run_mi_agent_query
from mi_agent.mi_dataset_profile import (
    PERCENT_FRACTION, PERCENT_POINTS, percent_storage_scale, profile_dataset,
)
from mi_agent_api.adapters import _format_kpi_value, adapt_workflow_result
from mi_agent_api.mi_dataset_contract import build_dataset_contract

_SEM = str(_REPO_ROOT / "mi_agent" / "mi_semantics_field_registry.yaml")


def _prepared(n: int = 33, ltv_nan: bool = False) -> pd.DataFrame:
    import numpy as np
    return pd.DataFrame({
        "loan_identifier": [f"L{i}" for i in range(n)],
        "current_outstanding_balance": [127515.15] * n,
        "youngest_borrower_age": [60 + (i % 30) for i in range(n)],
        # LTV stored as a FRACTION (0.29..0.56) — the contract must say so.
        "current_loan_to_value": ([np.nan] * n if ltv_nan
                                  else [0.29 + (i % 28) * 0.01 for i in range(n)]),
        "current_interest_rate": [3.10 + (i % 5) * 0.05 for i in range(n)],
        "ltv_bucket": ([np.nan] * n if ltv_nan else ["50-60%"] * n),
    })


def _run(question: str, df: pd.DataFrame):
    wf = run_mi_agent_query(question, df, _SEM, parser_mode="deterministic")
    return wf, adapt_workflow_result(wf, portfolio_id="client_001/mi_2025_10")


class TestStorageScaleContract(unittest.TestCase):
    def test_fraction_vs_points_detection(self):
        self.assertEqual(percent_storage_scale(pd.Series([0.29, 0.51, 0.56])), PERCENT_FRACTION)
        self.assertEqual(percent_storage_scale(pd.Series([29.0, 51.0, 56.0])), PERCENT_POINTS)

    def test_contract_reports_scale_and_availability(self):
        import yaml
        sem = yaml.safe_load(Path(_SEM).read_text())
        contract = build_dataset_contract(_prepared(), sem)
        by = {f["field"]: f for f in contract["fields"]}
        self.assertEqual(by["current_loan_to_value"]["semantic_type"], "percent")
        self.assertEqual(by["current_loan_to_value"]["storage_scale"], PERCENT_FRACTION)
        self.assertTrue(by["current_loan_to_value"]["metric_available"])
        self.assertIn("ltv_bucket", contract["dimensions_available"])

    def test_contract_marks_empty_dimension_missing(self):
        import yaml
        sem = yaml.safe_load(Path(_SEM).read_text())
        contract = build_dataset_contract(_prepared(ltv_nan=True), sem)
        self.assertNotIn("ltv_bucket", contract["dimensions_available"])
        missing = {m["dimension"] for m in contract["dimensions_missing"]}
        self.assertIn("ltv_bucket", missing)


class TestKpiPercentDisplay(unittest.TestCase):
    def test_fraction_formats_as_points(self):
        self.assertEqual(_format_kpi_value(0.51, "pct", PERCENT_FRACTION), "51.0%")

    def test_points_unchanged(self):
        self.assertEqual(_format_kpi_value(51.0, "pct", PERCENT_POINTS), "51.0%")


class TestBubbleArtifactKeys(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        warnings.simplefilter("ignore")
        cls.wf, cls.resp = _run("balance by ltv by age", _prepared())

    def test_ok_and_row_count(self):
        self.assertTrue(self.resp["ok"])
        self.assertEqual(self.resp["metadata"]["rowCount"], 33)

    def test_explicit_role_keys(self):
        chart = next(a for a in self.resp["artifacts"] if a["type"] == "chart")
        self.assertEqual(chart["chartType"], "bubble")
        self.assertEqual(chart["xKey"], "youngest_borrower_age")
        self.assertEqual(chart["yKey"], "current_loan_to_value")
        self.assertEqual(chart["sizeKey"], "current_outstanding_balance")
        self.assertIsNotNone(chart["yKey"])  # never null for bubble
        self.assertTrue(chart["xLabel"] and chart["yLabel"] and chart["sizeLabel"])

    def test_y_axis_scale_hint_is_fraction(self):
        chart = next(a for a in self.resp["artifacts"] if a["type"] == "chart")
        self.assertEqual(chart["displayHints"]["current_loan_to_value"]["scale"], PERCENT_FRACTION)
        self.assertEqual(chart["displayHints"]["current_loan_to_value"]["format"], "pct")


class TestFilteredCountIntent(unittest.TestCase):
    def test_how_many_more_than_routes_to_count(self):
        warnings.simplefilter("ignore")
        wf, resp = _run("how many loans with youngest age more than 70", _prepared())
        self.assertTrue(resp["ok"])
        self.assertEqual(wf["spec"]["intent"], "summary")
        self.assertEqual(wf["spec"]["aggregation"], "count")
        self.assertEqual(wf["spec"]["filters"],
                         {"youngest_borrower_age": {"op": "gt", "value": 70.0}})
        kpi = next(a for a in resp["artifacts"] if a["type"] == "kpi")
        loan = next(k for k in kpi["kpis"] if "loan" in k["label"].lower())
        self.assertEqual(int(loan["value"]), 19)  # ages 71..89


class TestGracefulFailureNoValues(unittest.TestCase):
    def test_missing_ltv_fails_with_reason_not_passed(self):
        warnings.simplefilter("ignore")
        wf, resp = _run("balance by ltv by age", _prepared(n=73, ltv_nan=True))
        self.assertFalse(resp["ok"])
        self.assertFalse(resp["validation"]["ok"])
        self.assertNotEqual(wf["interpreted"].get("Validation"), "Passed")
        reasons = {e["reason"] for e in resp["validation"]["data_validation_errors"]}
        self.assertIn("loan_level_no_usable_rows", reasons)
        # A controlled validation artifact is emitted (not a silent empty chart).
        self.assertTrue(any(a["type"] == "validation" for a in resp["artifacts"]))
        self.assertFalse(any(a["type"] == "chart" for a in resp["artifacts"]))

    def test_missing_ltv_bucket_dimension_fails(self):
        warnings.simplefilter("ignore")
        wf, resp = _run("current outstanding balance by ltv bucket",
                        _prepared(n=73, ltv_nan=True))
        self.assertFalse(resp["ok"])
        reasons = {e["reason"] for e in resp["validation"]["data_validation_errors"]}
        self.assertIn("dimension_no_values", reasons)


class TestDuplicateColumnsControlled(unittest.TestCase):
    def test_duplicate_columns_controlled_validation(self):
        warnings.simplefilter("ignore")
        df = _prepared(n=10)
        df = pd.concat([df, df[["current_loan_to_value"]]], axis=1)
        wf, resp = _run("balance by ltv by age", df)  # must not raise
        self.assertFalse(resp["ok"])
        self.assertFalse(resp["validation"]["ok"])
        self.assertIn("current_loan_to_value",
                      resp["validation"].get("duplicate_column_names", []))


if __name__ == "__main__":
    unittest.main(verbosity=2)
