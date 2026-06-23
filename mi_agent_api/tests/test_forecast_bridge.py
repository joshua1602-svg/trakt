#!/usr/bin/env python3
"""mi_agent_api/tests/test_forecast_bridge.py — Pipeline MI Phase 2.

The deterministic funded + pipeline forecast bridge:

    forecast_funded_balance = funded_balance + sum(expected * probability)

composed at the aggregate level from the funded snapshot + the Phase 1 pipeline
snapshot, with probabilities sourced from config (never the frontend, never
invented). Covers Part 6 backend requirements: bridge calc, pipeline snapshot
integration, missing-pipeline handling, readiness states, funded/pipeline
separation, probability governance, and funded endpoints unchanged.
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

from mi_agent_api import forecast_bridge as fb
from mi_agent_api import pipeline_contract as pc
from mi_agent_api.pipeline_prep import prepare_pipeline_mi_dataset

_FIXTURE_PACK = _REPO_ROOT / "tests" / "fixtures" / "client_001_mi_pack"
_NOV = _FIXTURE_PACK / "pipeline" / "2025-11-01" / "M2L_KFI_and_Pipeline_2025_11_01.csv"


def _funded_df(n: int = 73, per: float = 120000.0) -> pd.DataFrame:
    return pd.DataFrame({
        "loan_identifier": [760000 + i for i in range(n)],
        "current_outstanding_balance": [per] * n,
        "reporting_date": ["2025-11-30"] * n,
    })


def _pipeline(path: Path = _NOV):
    df = pd.read_csv(path)
    return prepare_pipeline_mi_dataset(df, source_file=path.name)


def _bridge_for(funded_df, pipeline_df, pipeline_report, *, run_id="mi_2025_11"):
    import yaml
    semantics = yaml.safe_load(
        (_REPO_ROOT / "mi_agent" / "mi_semantics_field_registry.yaml").read_text())
    # Pipeline source scope carries the weekly extract date (distinct from the
    # funded reporting date), modelling the Nov scope's latest weekly file.
    source = {"pipeline_source_folder_date": "2025-11-01",
              "pipeline_extract_date": "2025-12-01",
              "pipeline_as_of_date": "2025-12-01"}
    snap = None
    if pipeline_df is not None:
        snap = pc.compute_pipeline_snapshot(
            pipeline_df, pipeline_report, semantics,
            client_id="client_001", run_id=run_id, source=source)
    return fb.compute_forecast_bridge(
        client_id="client_001", run_id=run_id, funded_reporting_date="2025-11-30",
        funded_df=funded_df, pipeline_df=pipeline_df,
        pipeline_report=pipeline_report, pipeline_snapshot=snap,
        pipeline_source=source)


# --------------------------------------------------------------------------- #
# 1. Forecast bridge calculation
# --------------------------------------------------------------------------- #
class TestForecastBridgeCalc(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        warnings.simplefilter("ignore")
        cls.fdf = _funded_df()
        cls.pdf, cls.prep = _pipeline()
        cls.env = _bridge_for(cls.fdf, cls.pdf, cls.prep)
        cls.b = cls.env["forecastBridge"]

    def test_forecast_equals_funded_plus_weighted(self):
        self.assertAlmostEqual(
            self.b["forecastFundedBalance"],
            round(self.b["fundedBalance"] + self.b["weightedExpectedFundedAmount"], 2),
            places=2)

    def test_funded_headline_present(self):
        self.assertEqual(self.b["fundedLoanCount"], 73)
        self.assertAlmostEqual(self.b["fundedBalance"], 73 * 120000.0, places=2)

    def test_forecast_loan_count_is_funded_plus_pipeline(self):
        self.assertEqual(self.b["forecastLoanCount"],
                         self.b["fundedLoanCount"] + self.b["pipelineCaseCount"])


# --------------------------------------------------------------------------- #
# 2. Pipeline snapshot integration
# --------------------------------------------------------------------------- #
class TestPipelineIntegration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        warnings.simplefilter("ignore")
        cls.pdf, cls.prep = _pipeline()
        cls.env = _bridge_for(_funded_df(), cls.pdf, cls.prep)

    def test_bridge_carries_pipeline_metrics(self):
        b = self.env["forecastBridge"]
        self.assertEqual(b["pipelineCaseCount"], 10)
        self.assertGreater(b["pipelineAmount"], 0)
        self.assertGreater(b["weightedExpectedFundedAmount"], 0)
        self.assertTrue(b["stageBreakdown"])

    def test_full_pipeline_snapshot_embedded(self):
        snap = self.env["pipelineSnapshot"]
        self.assertEqual(snap["recordType"], "pipeline")
        self.assertIn("pipeline_stage", snap["availableDimensions"])
        self.assertTrue(snap["expectedCompletionBreakdown"])


# --------------------------------------------------------------------------- #
# 3. Missing pipeline handling
# --------------------------------------------------------------------------- #
class TestMissingPipeline(unittest.TestCase):
    def test_no_pipeline_does_not_500_and_is_controlled(self):
        env = _bridge_for(_funded_df(), None, None)
        b = env["forecastBridge"]
        self.assertTrue(env["ok"])
        self.assertFalse(b["pipelineAvailable"])
        self.assertEqual(b["forecastReadiness"]["status"], "blocked")
        self.assertIn("No pipeline data available for this run.",
                      b["forecastReadiness"]["warnings"])
        # Forecast falls back to the funded balance, never NaN/500.
        self.assertEqual(b["forecastFundedBalance"], b["fundedBalance"])
        self.assertEqual(b["weightedExpectedFundedAmount"], 0.0)


# --------------------------------------------------------------------------- #
# 4. Forecast readiness states
# --------------------------------------------------------------------------- #
class TestForecastReadiness(unittest.TestCase):
    def test_ready_when_amount_and_probability_available(self):
        pdf, prep = _pipeline()
        b = _bridge_for(_funded_df(), pdf, prep)["forecastBridge"]
        self.assertEqual(b["forecastReadiness"]["status"], "ready")
        self.assertEqual(b["forecastReadiness"]["missingRequiredFields"], [])

    def test_blocked_when_amount_missing(self):
        raw = pd.read_csv(_NOV).drop(columns=["Loan Amount", "Facility"])
        pdf, prep = prepare_pipeline_mi_dataset(raw, source_file=_NOV.name)
        b = _bridge_for(_funded_df(), pdf, prep)["forecastBridge"]
        self.assertEqual(b["forecastReadiness"]["status"], "blocked")
        self.assertIn("expected_amount", b["forecastReadiness"]["missingRequiredFields"])

    def test_partial_when_amount_coverage_incomplete(self):
        raw = pd.read_csv(_NOV)
        raw["Loan Amount"] = raw["Loan Amount"].astype(object)
        raw.loc[0, "Loan Amount"] = ""  # one row without an amount -> warning
        pdf, prep = prepare_pipeline_mi_dataset(raw, source_file=_NOV.name)
        b = _bridge_for(_funded_df(), pdf, prep)["forecastBridge"]
        self.assertEqual(b["forecastReadiness"]["status"], "partial")


# --------------------------------------------------------------------------- #
# 5. Separation of funded and pipeline
# --------------------------------------------------------------------------- #
class TestSeparation(unittest.TestCase):
    def test_funded_df_not_polluted_with_pipeline_rows(self):
        fdf = _funded_df()
        pdf, prep = _pipeline()
        _env = _bridge_for(fdf, pdf, prep)
        # The funded frame passed in is unchanged (no pipeline columns/rows added).
        self.assertEqual(len(fdf), 73)
        self.assertNotIn("pipeline_stage", fdf.columns)
        self.assertNotIn("record_type", fdf.columns)
        # The pipeline frame is tagged pipeline and never carries a funded loan id.
        self.assertTrue((pdf["record_type"] == "pipeline").all())
        self.assertNotIn("loan_identifier", pdf.columns)


# --------------------------------------------------------------------------- #
# 6. Probability governance
# --------------------------------------------------------------------------- #
class TestProbabilityGovernance(unittest.TestCase):
    def test_basis_is_stage_config_and_matches_config_values(self):
        import yaml
        cfg = yaml.safe_load((_REPO_ROOT / "config" / "client"
                              / "pipeline_expected_funding.yaml").read_text())
        probs = {k.upper(): float(v) for k, v in cfg["stage_probabilities"].items()}
        pdf, prep = _pipeline()
        b = _bridge_for(_funded_df(), pdf, prep)["forecastBridge"]
        self.assertEqual(b["completionProbabilityBasis"], "stage_config")
        # KFI cases must carry exactly the configured 0.20 — not a frontend/ad-hoc value.
        kfi = pdf[pdf["pipeline_stage"] == "KFI"]
        self.assertTrue((kfi["completion_probability"] == probs["KFI"]).all())
        # Weighted < unweighted because non-completed stages discount.
        self.assertLess(b["weightedExpectedFundedAmount"], b["pipelineAmount"])

    def test_endpoint_takes_no_probability_input(self):
        # The forecast endpoint signature accepts only portfolio/run identifiers;
        # probabilities cannot be injected by the caller.
        import inspect
        from mi_agent_api.app import forecast_snapshot
        params = set(inspect.signature(forecast_snapshot).parameters)
        self.assertEqual(params, {"portfolioId", "client_id", "runId", "run_id"})


# --------------------------------------------------------------------------- #
# 7. Existing funded endpoints unchanged + end-to-end forecast endpoint
# --------------------------------------------------------------------------- #
class TestEndToEndApi(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        warnings.simplefilter("ignore")
        cls.root = Path(tempfile.mkdtemp(prefix="fc_api_"))
        central = cls.root / "client_001" / "mi_2025_11" / "output" / "central"
        central.mkdir(parents=True)
        _funded_df().to_csv(central / "18_central_lender_tape.csv", index=False)

    def setUp(self):
        os.environ["MI_AGENT_ONBOARDING_OUTPUT_ROOT"] = str(self.root)
        os.environ["MI_AGENT_PIPELINE_ROOT"] = str(_FIXTURE_PACK)

    def tearDown(self):
        for k in ("MI_AGENT_ONBOARDING_OUTPUT_ROOT", "MI_AGENT_PIPELINE_ROOT"):
            os.environ.pop(k, None)

    def _client(self):
        from fastapi.testclient import TestClient
        from mi_agent_api.app import app
        return TestClient(app)

    def test_forecast_endpoint_composes_funded_and_pipeline(self):
        body = self._client().get(
            "/mi/forecast/snapshot",
            params={"portfolioId": "client_001/mi_2025_11"}).json()
        self.assertTrue(body["ok"])
        b = body["forecastBridge"]
        self.assertEqual(b["fundedLoanCount"], 73)
        self.assertGreater(b["pipelineCaseCount"], 0)
        self.assertAlmostEqual(
            b["forecastFundedBalance"],
            round(b["fundedBalance"] + b["weightedExpectedFundedAmount"], 2), places=2)
        self.assertIsNotNone(body["pipelineSnapshot"])

    def test_bridge_distinguishes_funded_and_pipeline_dates(self):
        body = self._client().get(
            "/mi/forecast/snapshot",
            params={"portfolioId": "client_001/mi_2025_11"}).json()
        b = body["forecastBridge"]
        # Funded reporting date is the month-end (Nov 30); pipeline as-of follows
        # the latest weekly extract (Dec 1) — they are NOT conflated.
        self.assertEqual(b["fundedReportingDate"], "2025-11-30")
        self.assertEqual(b["pipelineAsOfDate"], "2025-12-01")
        self.assertEqual(b["pipelineSourceFolderDate"], "2025-11-01")
        self.assertNotIn("reportingDate", b)  # no single ambiguous field

    def test_funded_snapshot_endpoint_still_works(self):
        body = self._client().get(
            "/mi/snapshot", params={"portfolioId": "client_001/mi_2025_11"}).json()
        self.assertTrue(body["ok"])
        self.assertEqual(body["loan_count"], 73)
        # Funded snapshot carries no pipeline/forecast keys (unchanged contract).
        self.assertNotIn("forecastBridge", body)
        self.assertNotIn("pipelineSnapshot", body)


if __name__ == "__main__":
    unittest.main(verbosity=2)
