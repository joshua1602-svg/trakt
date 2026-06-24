#!/usr/bin/env python3
"""mi_agent_api/tests/test_pipeline_phase3_refinement.py — Pipeline MI Phase 3.

Refinements: stage-classified watchlist (withdrawn INFO vs active WARNING),
top-10 broker/channel capping with Other, an empirical historical completion-rate
model, the governed probability hierarchy, and forecast-bridge disclosure.
"""

from __future__ import annotations

import sys
import tempfile
import unittest
import warnings
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import numpy as np
import pandas as pd
import yaml

from mi_agent_api import forecast_bridge as fb
from mi_agent_api import pipeline_contract as pc
from mi_agent_api.pipeline_history import build_historical_completion_model
from mi_agent_api.pipeline_prep import (
    classify_forecast_gaps,
    diagnostics_by_severity,
    prepare_pipeline_mi_dataset,
)

_FIXTURES = _REPO_ROOT / "tests" / "fixtures" / "client_001_mi_pack" / "pipeline"
_OCT = _FIXTURES / "2025-10-01" / "M2L_KFI_and_Pipeline_2025_10_01.csv"
_SEMANTICS = yaml.safe_load(
    (_REPO_ROOT / "mi_agent" / "mi_semantics_field_registry.yaml").read_text())


# --------------------------------------------------------------------------- #
# 1 + 2. Watchlist classification (probability + expected completion date)
# --------------------------------------------------------------------------- #
class TestForecastGapClassification(unittest.TestCase):
    def _frame(self):
        return pd.DataFrame({
            "pipeline_stage": ["OFFER", "WITHDRAWN", "UNKNOWN", "KFI"],
            "completion_probability": [0.75, np.nan, np.nan, np.nan],
            "expected_completion_date": [pd.Timestamp("2025-12-01"), pd.NaT, pd.NaT, pd.NaT],
            "current_outstanding_balance": [100000.0, 80000.0, 50000.0, 90000.0],
        })

    def test_withdrawn_probability_is_info_excluded(self):
        gaps = {g["check"]: g for g in classify_forecast_gaps(self._frame())}
        w = gaps["withdrawn_excluded_from_weighting"]
        self.assertEqual(w["severity"], "info")
        self.assertTrue(w["excluded"])
        self.assertFalse(w["weighted"])
        self.assertEqual(w["by_stage"], {"WITHDRAWN": 1})
        self.assertNotIn("without a completion probability", w["detail"])

    def test_active_missing_probability_is_warning(self):
        gaps = {g["check"]: g for g in classify_forecast_gaps(self._frame())}
        a = gaps["active_missing_completion_probability"]
        self.assertEqual(a["severity"], "warning")
        self.assertEqual(a["by_stage"], {"KFI": 1})

    def test_unknown_stage_diagnosed_separately(self):
        gaps = {g["check"]: g for g in classify_forecast_gaps(self._frame())}
        self.assertIn("unknown_stage_no_probability", gaps)
        self.assertEqual(gaps["unknown_stage_no_probability"]["severity"], "warning")
        self.assertEqual(gaps["unknown_stage_no_probability"]["by_stage"], {"UNKNOWN": 1})

    def test_expected_completion_date_classified_by_stage(self):
        gaps = {g["check"]: g for g in classify_forecast_gaps(self._frame())}
        # withdrawn + unknown have no date -> INFO (not required); KFI active -> WARNING.
        self.assertEqual(gaps["expected_completion_date_not_required"]["severity"], "info")
        self.assertEqual(gaps["active_missing_expected_completion_date"]["severity"], "warning")
        self.assertEqual(
            gaps["active_missing_expected_completion_date"]["by_stage"], {"KFI": 1})

    def test_fixture_withdrawn_is_info_not_generic_warning(self):
        warnings.simplefilter("ignore")
        df, rep = prepare_pipeline_mi_dataset(pd.read_csv(_OCT), source_file=_OCT.name)
        groups = diagnostics_by_severity(rep)
        checks = {d["check"] for d in rep["data_quality"]}
        self.assertIn("withdrawn_excluded_from_weighting", checks)
        # The old conflated WARNING must be gone for a withdrawn-only gap.
        warn_checks = {d["check"] for d in groups["warning"]}
        self.assertNotIn("missing_completion_probability", warn_checks)


# --------------------------------------------------------------------------- #
# 3. Broker/channel top-10 with Other
# --------------------------------------------------------------------------- #
class TestTopTenCap(unittest.TestCase):
    def _rows(self, n):
        return [{"key": f"B{i}", "caseCount": n - i, "pipelineAmount": float((n - i) * 1000),
                 "weightedExpectedFundedAmount": float((n - i) * 500)} for i in range(n)]

    def test_caps_to_ten_with_other(self):
        rows = self._rows(14)
        capped = pc.cap_breakdown(rows, 10)
        self.assertEqual(len(capped), 10)
        self.assertEqual(capped[-1]["key"], "Other")
        self.assertTrue(capped[-1]["isOther"])
        self.assertEqual(capped[-1]["categoriesIncluded"], 5)

    def test_totals_reconcile(self):
        rows = self._rows(20)
        capped = pc.cap_breakdown(rows, 10)
        self.assertAlmostEqual(sum(r["pipelineAmount"] for r in capped),
                               sum(r["pipelineAmount"] for r in rows), places=2)
        self.assertEqual(sum(r["caseCount"] for r in capped),
                         sum(r["caseCount"] for r in rows))

    def test_no_cap_when_ten_or_fewer(self):
        rows = self._rows(8)
        self.assertEqual(pc.cap_breakdown(rows, 10), rows)

    def test_snapshot_exposes_capped_and_full(self):
        warnings.simplefilter("ignore")
        # 14 distinct brokers across enough rows.
        n = 14
        df = pd.DataFrame({
            "Account Number": [f"A{i}" for i in range(n)],
            "KFI Number": [f"K{i}" for i in range(n)],
            "Status": ["Offer"] * n, "Loan Amount": [100000.0] * n,
            "Broker": [f"Broker {i}" for i in range(n)],
        })
        prep, rep = prepare_pipeline_mi_dataset(df, as_of_date="2025-11-01")
        snap = pc.compute_pipeline_snapshot(prep, rep, _SEMANTICS,
                                            client_id="client_001", run_id="mi_2025_11")
        self.assertLessEqual(len(snap["brokerBreakdown"]), 10)
        self.assertEqual(snap["brokerBreakdown"][-1]["key"], "Other")
        self.assertEqual(len(snap["brokerBreakdownFull"]), 14)  # full detail retained


# --------------------------------------------------------------------------- #
# 4. Historical completion-rate model
# --------------------------------------------------------------------------- #
class TestHistoricalModel(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        warnings.simplefilter("ignore")
        cls.tmp = Path(tempfile.mkdtemp(prefix="hist_"))
        n = 14
        # Week 1: 14 OFFER cases.
        pd.DataFrame({"Account Number": [f"A{i}" for i in range(n)],
                      "KFI Number": [f"K{i}" for i in range(n)], "Status": ["Offer"] * n,
                      "Loan Amount": [100000.0] * n, "Broker": ["B"] * n}).to_csv(
            cls.tmp / "M2L_KFI_2025_10_01.csv", index=False)
        # Week 2: 10 of them completed, 4 still OFFER.
        pd.DataFrame({"Account Number": [f"A{i}" for i in range(n)],
                      "KFI Number": [f"K{i}" for i in range(n)],
                      "Status": ["Completed" if i < 10 else "Offer" for i in range(n)],
                      "Loan Amount": [100000.0] * n, "Broker": ["B"] * n,
                      "Date Funds Released": ["2025-11-20" if i < 10 else "" for i in range(n)]}).to_csv(
            cls.tmp / "M2L_KFI_2025_11_01.csv", index=False)
        cls.entries = [
            {"source_file": str(cls.tmp / "M2L_KFI_2025_10_01.csv"), "pipeline_extract_date": "2025-10-01"},
            {"source_file": str(cls.tmp / "M2L_KFI_2025_11_01.csv"), "pipeline_extract_date": "2025-11-01"},
        ]

    def test_tracks_cases_and_derives_offer_rate(self):
        m = build_historical_completion_model(self.entries, min_observations=12)
        self.assertEqual(m["casesTracked"], 14)
        self.assertEqual(m["snapshotCount"], 2)
        offer = m["historicalCompletionRateByStage"]["OFFER"]
        self.assertEqual(offer["observed"], 14)
        self.assertEqual(offer["completed"], 10)
        self.assertAlmostEqual(offer["rate"], round(10 / 14, 4), places=4)
        self.assertTrue(offer["sufficient"])
        self.assertEqual(m["stage_rates"]["OFFER"], round(10 / 14, 4))

    def test_window_is_chronological(self):
        m = build_historical_completion_model(self.entries)
        self.assertEqual(m["historicalCompletionRateWindow"]["fromDate"], "2025-10-01")
        self.assertEqual(m["historicalCompletionRateWindow"]["toDate"], "2025-11-01")

    def test_insufficient_history_falls_back_to_config(self):
        m = build_historical_completion_model(self.entries, min_observations=50)
        self.assertFalse(m["available"])
        self.assertEqual(m["stage_rates"], {})  # nothing trusted -> prep uses config

    def test_prep_uses_historical_then_config(self):
        m = build_historical_completion_model(self.entries, min_observations=12)
        df = pd.read_csv(self.tmp / "M2L_KFI_2025_11_01.csv")
        prep, rep = prepare_pipeline_mi_dataset(df, as_of_date="2025-11-01",
                                                historical_model=m)
        srcs = set(prep["completion_probability_source"])
        self.assertIn("historical_stage_rate", srcs)   # OFFER cases
        self.assertIn("configured_stage_rate", srcs)   # COMPLETED via config 1.0
        self.assertEqual(rep["completion_probability_basis"], "mixed_historical_and_config")
        # OFFER rows carry the empirical rate, not the configured 0.75.
        offer = prep[prep["pipeline_stage"] == "OFFER"]
        self.assertTrue((offer["completion_probability"].round(4) == round(10 / 14, 4)).all())


# --------------------------------------------------------------------------- #
# 5. Forecast bridge disclosure + hierarchy
# --------------------------------------------------------------------------- #
class TestForecastDisclosure(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        warnings.simplefilter("ignore")
        cls.pdf, cls.prep = prepare_pipeline_mi_dataset(pd.read_csv(_OCT), source_file=_OCT.name)
        funded = pd.DataFrame({"loan_identifier": range(73),
                               "current_outstanding_balance": [100000.0] * 73})
        snap = pc.compute_pipeline_snapshot(cls.pdf, cls.prep, _SEMANTICS,
                                            client_id="client_001", run_id="mi_2025_11")
        cls.env = fb.compute_forecast_bridge(
            client_id="client_001", run_id="mi_2025_11", funded_reporting_date="2025-11-30",
            funded_df=funded, pipeline_df=cls.pdf, pipeline_report=cls.prep,
            pipeline_snapshot=snap)
        cls.b = cls.env["forecastBridge"]

    def test_discloses_gross_excluded_and_basis(self):
        self.assertEqual(self.b["grossPipelineAmount"], self.prep["total_pipeline_amount"])
        # One withdrawn case (£80k) is excluded from weighting.
        self.assertEqual(self.b["excludedFromWeightingAmount"], 80000.0)
        self.assertEqual(self.b["excludedCaseCount"], 1)
        self.assertEqual(self.b["completionProbabilityBasis"], "stage_config")

    def test_blended_conversion_present(self):
        self.assertIsNotNone(self.b["blendedWeightedConversion"])
        # weighted / active gross.
        active = self.b["activeGrossPipelineAmount"]
        expected = round(self.b["weightedExpectedFundedAmount"] / active, 4)
        self.assertAlmostEqual(self.b["blendedWeightedConversion"], expected, places=3)

    def test_forecast_identity_preserved(self):
        self.assertAlmostEqual(
            self.b["forecastFundedBalance"],
            round(self.b["fundedBalance"] + self.b["weightedExpectedFundedAmount"], 2),
            places=2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
