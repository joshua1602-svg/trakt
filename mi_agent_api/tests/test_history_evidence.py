#!/usr/bin/env python3
"""mi_agent_api/tests/test_history_evidence.py — historical model evidence lineage.

Exposes how many weekly pipeline files / rows / cases the completion estimate
relies on, keeps the current snapshot as-of date distinct from the historical
observation window, and excludes non-pipeline (funder principal) files.
"""

from __future__ import annotations

import os
import sys
import tempfile
import unittest
import warnings
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import pandas as pd

from mi_agent_api import data_source
from mi_agent_api import pipeline_contract as pc
from mi_agent_api.pipeline_history import build_historical_completion_model, historical_model_evidence

_WEEKS = [("2025_10_06", "2025-10-06"), ("2025_11_03", "2025-11-03"), ("2025_12_01", "2025-12-01")]
_N = 16


def _weekly_df(week_key: str):
    completed = [week_key != "2025_10_06" and i < 10 for i in range(_N)]
    return pd.DataFrame({
        "Account Number": [f"A{i}" for i in range(_N)],
        "KFI Number": [f"K{i}" for i in range(_N)],
        "Status": ["Completed" if c else "Offer" for c in completed],
        "Loan Amount": [100000.0] * _N, "Broker": ["B"] * _N,
        "Date Funds Released": ["2025-11-20" if c else "" for c in completed],
    })


def _write_pack(pdir: Path):
    pdir.mkdir(parents=True, exist_ok=True)
    for wk, _dt in _WEEKS:
        _weekly_df(wk).to_csv(pdir / f"M2L_KFI_and_Pipeline_{wk}_120000.csv", index=False)
    # A funder principal file that must NOT be counted as pipeline evidence.
    pd.DataFrame({"Account Number": [f"F{i}" for i in range(5)], "Principal": [1] * 5}).to_csv(
        pdir / "Funder Principal And Interest_test.csv", index=False)


# --------------------------------------------------------------------------- #
# Unit: model evidence + source filtering
# --------------------------------------------------------------------------- #
class TestHistoryEvidenceUnit(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        warnings.simplefilter("ignore")
        cls.tmp = Path(tempfile.mkdtemp(prefix="hist_ev_"))
        _write_pack(cls.tmp / "client_001" / "output" / "pipeline" / "2025-10-01")

    def test_model_exposes_evidence_fields(self):
        entries = pc.collect_weekly_history(self.tmp, "client_001")
        model = build_historical_completion_model(entries)
        ev = historical_model_evidence(model, "mixed_historical_and_config")
        self.assertEqual(ev["weeklyFilesUsed"], 3)
        self.assertEqual(len(ev["weeklyFileNames"]), 3)
        self.assertEqual(ev["trackedCaseCount"], 16)
        self.assertEqual(ev["observedCompletionCount"], 10)
        self.assertGreater(ev["historicalRowsUsed"], 0)
        self.assertEqual(ev["observationWindowStart"], "2025-10-06")
        self.assertEqual(ev["observationWindowEnd"], "2025-12-01")
        self.assertIn("Account Number", ev["stableIdentifierUsed"])
        self.assertIn("OFFER", ev["stagesUsingHistoricalRates"])
        self.assertEqual(ev["completionProbabilityBasis"], "mixed_historical_and_config")

    def test_cohort_progression_is_a_monotonic_funnel(self):
        model = build_historical_completion_model(
            pc.collect_weekly_history(self.tmp, "client_001"))
        prog = model["cohortProgression"]
        self.assertIsNotNone(prog)
        self.assertEqual(prog["cohortSize"], 16)
        self.assertEqual(prog["stages"], ["KFI", "APPLICATION", "OFFER", "COMPLETED"])
        s = prog["series"]
        # 10 of 16 cases funded -> cumulative cohort conversion 62.5% to date.
        self.assertEqual(s["COMPLETED"][-1], 62.5)
        self.assertEqual(model["cumulativeCohortConversion"], 62.5)
        # Every week: Funded <= Offer <= Application <= KFI (nested funnel).
        for i in range(len(prog["weeks"])):
            self.assertLessEqual(s["COMPLETED"][i], s["OFFER"][i])
            self.assertLessEqual(s["OFFER"][i], s["APPLICATION"][i])
            self.assertLessEqual(s["APPLICATION"][i], s["KFI"][i])
        # Completions are non-decreasing over time (cumulative).
        self.assertTrue(all(s["COMPLETED"][i] <= s["COMPLETED"][i + 1]
                            for i in range(len(s["COMPLETED"]) - 1)))

    def test_funder_file_not_counted(self):
        names = [Path(e["source_file"]).name for e in pc.collect_weekly_history(self.tmp, "client_001")]
        self.assertTrue(all("Funder" not in n and "Principal" not in n for n in names), names)
        model = build_historical_completion_model(pc.collect_weekly_history(self.tmp, "client_001"))
        self.assertTrue(all("Funder" not in n for n in model["weeklyFileNames"]))


# --------------------------------------------------------------------------- #
# Integration: forecast + workspace expose the evidence; dates stay distinct
# --------------------------------------------------------------------------- #
class TestHistoryEvidenceApi(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        warnings.simplefilter("ignore")
        cls.root = Path(tempfile.mkdtemp(prefix="hist_api_"))
        base = cls.root / "client_001" / "mi_2025_11" / "output"
        _write_pack(base / "pipeline" / "2025-10-01")
        central = base / "central"
        central.mkdir(parents=True)
        pd.DataFrame({"loan_identifier": range(73),
                      "current_outstanding_balance": [120000.0] * 73,
                      "reporting_date": ["2025-11-30"] * 73}).to_csv(
            central / "18_central_lender_tape.csv", index=False)
        cls.tape = str(central / "18_central_lender_tape.csv")

    def setUp(self):
        os.environ["MI_AGENT_ONBOARDING_OUTPUT_ROOT"] = str(self.root)
        os.environ["MI_AGENT_PIPELINE_ROOT"] = str(self.root)
        os.environ["MI_AGENT_CENTRAL_TAPE"] = self.tape
        os.environ["MI_AGENT_CLIENT_ID"] = "client_001"
        os.environ["MI_AGENT_RUN_ID"] = "mi_2025_11"
        data_source.reset_cache()

    def tearDown(self):
        for k in ("MI_AGENT_ONBOARDING_OUTPUT_ROOT", "MI_AGENT_PIPELINE_ROOT",
                  "MI_AGENT_CENTRAL_TAPE", "MI_AGENT_CLIENT_ID", "MI_AGENT_RUN_ID"):
            os.environ.pop(k, None)
        data_source.reset_cache()

    def _client(self):
        from fastapi.testclient import TestClient
        from mi_agent_api.app import app
        return TestClient(app)

    def test_forecast_exposes_evidence_and_basis_not_null(self):
        body = self._client().get("/mi/forecast/snapshot",
                                  params={"portfolioId": "client_001/mi_2025_11"}).json()
        ev = body["historicalModelEvidence"]
        self.assertEqual(ev["weeklyFilesUsed"], 3)
        self.assertEqual(ev["observationWindowStart"], "2025-10-06")
        self.assertEqual(ev["observationWindowEnd"], "2025-12-01")
        # Top-level basis is not null when the model is available.
        self.assertIsNotNone(body["completionProbabilityBasis"])
        self.assertEqual(body["lineage"]["completionProbabilityBasis"],
                         body["completionProbabilityBasis"])

    def test_pipeline_and_forecast_views_expose_same_evidence(self):
        c = self._client()
        pv = c.get("/mi/workspace/view",
                   params={"portfolioId": "client_001/mi_2025_11", "view": "pipeline"}).json()
        fv = c.get("/mi/workspace/view",
                   params={"portfolioId": "client_001/mi_2025_11", "view": "forecast"}).json()
        pe = pv["lineage"]["pipeline"]["historicalModelEvidence"]
        fe = fv["lineage"]["forecast"]["historicalModelEvidence"]
        self.assertEqual(pe["weeklyFilesUsed"], 3)
        self.assertEqual(fe["weeklyFilesUsed"], 3)
        self.assertEqual(pe["observationWindowEnd"], fe["observationWindowEnd"])

    def test_snapshot_as_of_distinct_from_observation_window(self):
        ln = self._client().get(
            "/mi/workspace/view",
            params={"portfolioId": "client_001/mi_2025_11", "view": "pipeline"}).json()["lineage"]["pipeline"]
        # Current snapshot as-of is the latest weekly extract (Dec 1); the window
        # spans the FIRST weekly file (Oct 6) to the latest — they are distinct.
        self.assertEqual(ln["pipelineAsOfDate"], "2025-12-01")
        self.assertEqual(ln["observationWindowStart"], "2025-10-06")
        self.assertNotEqual(ln["pipelineAsOfDate"], ln["observationWindowStart"])

    def test_pipeline_snapshot_carries_evidence(self):
        body = self._client().get("/mi/pipeline/snapshot",
                                  params={"portfolioId": "client_001/mi_2025_11"}).json()
        self.assertIn("historicalModelEvidence", body)
        self.assertEqual(body["historicalModelEvidence"]["weeklyFilesUsed"], 3)


if __name__ == "__main__":
    unittest.main(verbosity=2)
