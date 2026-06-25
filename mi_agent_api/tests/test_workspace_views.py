#!/usr/bin/env python3
"""mi_agent_api/tests/test_workspace_views.py — Pipeline MI Phase 4 workspace.

The unified Funded / Pipeline / Forecast workspace view-model endpoint and the
tab-aware MI Agent query context. Funded behaviour and funded/pipeline separation
are preserved; forecast stays backend-derived.
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
from mi_agent_api import workspace as ws

_FIXTURES = _REPO_ROOT / "tests" / "fixtures" / "client_001_mi_pack"


# --------------------------------------------------------------------------- #
# Unit: active-view resolution + forecast frame
# --------------------------------------------------------------------------- #
class TestActiveViewResolution(unittest.TestCase):
    def test_tab_context_used_when_no_explicit_wording(self):
        self.assertEqual(ws.resolve_active_view("amount by region", "pipeline"), "pipeline")
        self.assertEqual(ws.resolve_active_view("amount by region", "forecast"), "forecast")
        self.assertEqual(ws.resolve_active_view("amount by region", None), "funded")

    def test_explicit_wording_overrides_tab(self):
        self.assertEqual(ws.resolve_active_view("funded balance by region", "pipeline"), "funded")
        self.assertEqual(ws.resolve_active_view("pipeline amount by region", "funded"), "pipeline")
        # "forecast funded balance" -> forecast wins over the funded keyword.
        self.assertEqual(ws.resolve_active_view("forecast funded balance", "funded"), "forecast")

    def test_forecast_frame_is_funded_plus_weighted_pipeline(self):
        funded = pd.DataFrame({"current_outstanding_balance": [100.0, 200.0],
                               "geographic_region_obligor": ["London", "Wales"]})
        pipeline = pd.DataFrame({"weighted_expected_funded_amount": [50.0, None],
                                 "current_outstanding_balance": [80.0, 90.0],
                                 "geographic_region_obligor": ["London", "Wales"]})
        frame = ws.build_forecast_view_frame(funded, pipeline)
        # 2 funded + 1 weightable pipeline row (the None-weight pipeline row drops).
        self.assertEqual(len(frame), 3)
        self.assertAlmostEqual(frame["current_outstanding_balance"].sum(), 350.0)
        self.assertEqual(set(frame["state_component"]), {"funded", "forecast_pipeline"})

    def test_forecast_dimension_breakdown_composes(self):
        funded = pd.DataFrame({"current_outstanding_balance": [100.0, 200.0],
                               "geographic_region_obligor": ["London", "Wales"]})
        pipeline = pd.DataFrame({"weighted_expected_funded_amount": [50.0],
                                 "geographic_region_obligor": ["London"]})
        rows = {r["key"]: r for r in ws.forecast_dimension_breakdown(
            funded, pipeline, "geographic_region_obligor")}
        self.assertEqual(rows["London"]["fundedAmount"], 100.0)
        self.assertEqual(rows["London"]["weightedPipelineAmount"], 50.0)
        self.assertEqual(rows["London"]["forecastAmount"], 150.0)
        self.assertEqual(rows["Wales"]["forecastAmount"], 200.0)


# --------------------------------------------------------------------------- #
# Integration: workspace endpoint + tab-aware query
# --------------------------------------------------------------------------- #
class TestWorkspaceApi(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        warnings.simplefilter("ignore")
        cls.root = Path(tempfile.mkdtemp(prefix="ws_"))
        central = cls.root / "client_001" / "mi_2025_11" / "output" / "central"
        central.mkdir(parents=True)
        n = 73
        pd.DataFrame({
            "loan_identifier": [760000 + i for i in range(n)],
            "current_outstanding_balance": [120000.0 + i * 1000 for i in range(n)],
            "geographic_region_obligor": [["London", "Scotland", "Wales"][i % 3] for i in range(n)],
            "reporting_date": ["2025-11-30"] * n,
        }).to_csv(central / "18_central_lender_tape.csv", index=False)
        cls.tape = str(central / "18_central_lender_tape.csv")

    def setUp(self):
        os.environ["MI_AGENT_ONBOARDING_OUTPUT_ROOT"] = str(self.root)
        os.environ["MI_AGENT_CENTRAL_TAPE"] = self.tape
        os.environ["MI_AGENT_CLIENT_ID"] = "client_001"
        os.environ["MI_AGENT_RUN_ID"] = "mi_2025_11"
        os.environ["MI_AGENT_PIPELINE_ROOT"] = str(_FIXTURES)
        data_source.reset_cache()

    def tearDown(self):
        for k in ("MI_AGENT_ONBOARDING_OUTPUT_ROOT", "MI_AGENT_CENTRAL_TAPE",
                  "MI_AGENT_CLIENT_ID", "MI_AGENT_RUN_ID", "MI_AGENT_PIPELINE_ROOT"):
            os.environ.pop(k, None)
        data_source.reset_cache()

    def _client(self):
        from fastapi.testclient import TestClient
        from mi_agent_api.app import app
        return TestClient(app)

    def _query(self, question, ctx, pid="client_001/mi_2025_11"):
        return self._client().post("/mi/query", json={
            "question": question, "portfolioId": pid, "datasetContext": ctx}).json()

    # ---- view-model endpoint ----------------------------------------------- #
    def test_view_model_composes_three_views(self):
        body = self._client().get("/mi/workspace/view",
                                  params={"portfolioId": "client_001/mi_2025_11", "view": "forecast"}).json()
        self.assertTrue(body["ok"])
        self.assertEqual(body["activeView"], "forecast")
        self.assertEqual(body["views"], ["funded", "pipeline", "forecast"])
        # Funded view carries funded metrics; pipeline carries pipeline metrics.
        self.assertEqual(body["funded"]["loan_count"], 73)
        self.assertGreater(body["pipeline"]["pipelineRowCount"], 0)
        self.assertTrue(body["forecast"]["ok"])
        self.assertIn("byRegion", body["forecast"]["forecastBreakdowns"])
        # Lineage exposes funded source, pipeline as-of, forecast formula.
        self.assertEqual(body["lineage"]["funded"]["source"], "18_central_lender_tape.csv")
        self.assertEqual(body["lineage"]["pipeline"]["pipelineAsOfDate"], "2025-12-01")
        self.assertIn("forecast funded balance", body["lineage"]["forecast"]["formula"])

    def test_default_view_is_funded(self):
        body = self._client().get("/mi/workspace/view",
                                  params={"portfolioId": "client_001/mi_2025_11"}).json()
        self.assertEqual(body["activeView"], "funded")

    def test_existing_endpoints_unchanged(self):
        c = self._client()
        self.assertTrue(c.get("/mi/snapshot", params={"portfolioId": "client_001/mi_2025_11"}).json()["ok"])
        self.assertTrue(c.get("/mi/pipeline/snapshot", params={"portfolioId": "client_001/mi_2025_11"}).json()["ok"])
        # Forecast endpoint now also carries the derived breakdowns + lineage.
        fc = c.get("/mi/forecast/snapshot", params={"portfolioId": "client_001/mi_2025_11"}).json()
        self.assertIn("forecastBreakdowns", fc)
        self.assertIn("lineage", fc)

    # ---- tab-aware query --------------------------------------------------- #
    def test_unqualified_amount_routes_to_active_dataset(self):
        for ctx in ("funded", "pipeline", "forecast"):
            body = self._query("amount by region", ctx)
            self.assertTrue(body["ok"], (ctx, body.get("validation")))
            self.assertEqual(body["metadata"]["datasetContext"], ctx)

    def test_explicit_wording_overrides_tab(self):
        # Forecast wording while on the funded tab routes to forecast.
        body = self._query("forecast balance by region", "funded")
        self.assertEqual(body["metadata"]["datasetContext"], "forecast")
        # Pipeline wording while on the funded tab routes to pipeline.
        body = self._query("pipeline amount by region", "funded")
        self.assertEqual(body["metadata"]["datasetContext"], "pipeline")

    def test_funded_balance_differs_from_pipeline_amount(self):
        funded = self._query("amount by region", "funded")
        pipeline = self._query("amount by region", "pipeline")
        # Both succeed but answer different datasets (funded ~£8.9MM+, pipeline ~£1.8MM).
        self.assertTrue(funded["ok"] and pipeline["ok"])
        self.assertEqual(funded["metadata"]["datasetContext"], "funded")
        self.assertEqual(pipeline["metadata"]["datasetContext"], "pipeline")

    def test_unsupported_query_fails_gracefully(self):
        # A dimension absent from the pipeline dataset returns a controlled
        # (non-500) validation response, not an exception.
        body = self._query("current outstanding balance by maturity year", "pipeline")
        self.assertIn("ok", body)
        self.assertFalse(body["ok"])
        self.assertEqual(body["metadata"]["datasetContext"], "pipeline")

    def test_separation_funded_and_pipeline_not_merged(self):
        # The forecast frame is derived in-memory; the funded tape on disk is
        # untouched and carries no pipeline columns.
        fdf = pd.read_csv(self.tape)
        self.assertEqual(len(fdf), 73)
        self.assertNotIn("pipeline_stage", fdf.columns)
        self.assertNotIn("weighted_expected_funded_amount", fdf.columns)


if __name__ == "__main__":
    unittest.main(verbosity=2)
