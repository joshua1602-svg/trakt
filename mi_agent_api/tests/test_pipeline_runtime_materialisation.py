#!/usr/bin/env python3
"""mi_agent_api/tests/test_pipeline_runtime_materialisation.py — Pipeline MI runtime fix.

A clean E2E onboarding/promote of an M2L KFI weekly pipeline extract must, at the
canonical ``onboarding_output/<client>/<run>/output/`` layout:

  * build a NON-empty central pipeline tape at
    ``output/central/18a_central_pipeline_tape.csv`` (fixes "Central pipeline
    tape: False (0 applications)"), keyed on the KFI / account number;
  * materialise the governed pipeline SOURCE under ``output/pipeline/<folder>/``
    (the contributing weekly extract), WITHOUT copying funded loan files;
  * be discoverable by ``/mi/pipeline/snapshots`` from ``onboarding_output``
    (the raw ``input/`` copy is NOT discovered) and return non-zero rows;
  * drive ``/mi/forecast/snapshot`` from the generated outputs;
  * keep the funded book separate and unchanged.

Pipeline dates remain weekly-operational (the Dec-1 extract inside the Nov scope),
never conflated with the funded reporting date.
"""

from __future__ import annotations

import importlib.util
import os
import shutil
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

_REGISTRY = str(_REPO_ROOT / "config" / "system" / "fields_registry.yaml")
_FIXTURES = _REPO_ROOT / "tests" / "fixtures" / "client_001_mi_pack" / "pipeline"
_WEEKLY = _FIXTURES / "2025-11-01" / "M2L_KFI_and_Pipeline_2025_12_01_115711.csv"
_HAS_OPENPYXL = importlib.util.find_spec("openpyxl") is not None


class TestPipelineRuntimeMaterialisation(unittest.TestCase):
    """Clean promote at the canonical onboarding_output layout."""

    @classmethod
    def setUpClass(cls):
        warnings.simplefilter("ignore")
        cls.tmp = Path(tempfile.mkdtemp(prefix="pl_rt_"))
        cls.onb = cls.tmp / "onboarding_output"
        base = cls.onb / "client_001" / "mi_2025_11"
        inp = base / "input"
        (inp / "pipeline" / "2025-11-01").mkdir(parents=True)
        n = 12
        ids = [760000 + i for i in range(n)]
        pd.DataFrame({
            # Month Run aligns with the mi_2025_11 run period so the funded
            # universe is eligible (period-gated by source_period_eligibility).
            "Loan Policy Number": ids, "Month Run": ["November"] * n,
            "Loan Interest Rate": [3.0 + (i % 5) * 0.5 for i in range(n)],
            "Current Outstanding Balance": [100000.0 + i * 5000 for i in range(n)],
            "Policy Completion Date": ["2020-01-01"] * n,
        }).to_csv(inp / "LoanExtract One.csv", index=False)
        # The real M2L KFI weekly extract (Dec-1 file) under the Nov source folder.
        shutil.copy(_WEEKLY, inp / "pipeline" / "2025-11-01" / _WEEKLY.name)

        from engine.onboarding_agent import workflow as wf, storage_paths, central_tape_builder
        cls.out_root = base / "output"
        proj = base / "proj"
        wf.run_operator_workflow(
            input_dir=str(inp), client_name="Client 001", client_id="client_001",
            run_id="mi_2025_11", mode="mi_only", project_dir=str(proj),
            product_profile="equity_release_lifetime_mortgage")
        rp = storage_paths.resolve_run_paths(
            project_dir=str(proj), input_dir=str(inp), output_root=str(cls.out_root),
            client_id="client_001", run_id="mi_2025_11", storage_backend="local",
            input_uri="", output_uri="")
        cls.res = central_tape_builder.build_central_tapes(
            str(proj), rp, _REGISTRY, mode="mi_only")

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    # ---- promote materialisation -------------------------------------------- #
    def test_central_pipeline_tape_at_canonical_path(self):
        # Was "False (0 applications)" before the KFI/account key fix.
        self.assertTrue(self.res["central_pipeline_tape_created"])
        self.assertGreater(self.res["pipeline_count"], 0)
        canonical = self.out_root / "central" / "18a_central_pipeline_tape.csv"
        self.assertTrue(canonical.exists(), canonical)
        self.assertGreater(len(pd.read_csv(canonical)), 0)

    def test_funded_tape_unchanged_and_separate(self):
        funded = self.out_root / "central" / "18_central_lender_tape.csv"
        self.assertTrue(funded.exists())
        fdf = pd.read_csv(funded)
        self.assertEqual(len(fdf), 12)              # funded universe intact
        # Pipeline application ids never pollute the funded central tape.
        self.assertNotIn("application_id", fdf.columns)
        self.assertNotIn("pipeline_stage", fdf.columns)

    def test_pipeline_source_materialised_without_funded_pollution(self):
        self.assertEqual(self.res["pipeline_sources_materialised"], 1)
        psd = Path(self.res["pipeline_source_dir"])
        self.assertTrue((psd / "2025-11-01" / _WEEKLY.name).exists())
        names = [p.name for p in psd.rglob("*") if p.is_file()]
        self.assertNotIn("LoanExtract One.csv", names)  # no funded file copied

    # ---- API from generated outputs ----------------------------------------- #
    def setUp(self):
        os.environ["MI_AGENT_ONBOARDING_OUTPUT_ROOT"] = str(self.onb)
        os.environ["MI_AGENT_CENTRAL_TAPE"] = self.res["central_lender_tape_path"]
        os.environ["MI_AGENT_CLIENT_ID"] = "client_001"
        os.environ["MI_AGENT_RUN_ID"] = "mi_2025_11"
        data_source.reset_cache()

    def tearDown(self):
        for k in ("MI_AGENT_ONBOARDING_OUTPUT_ROOT", "MI_AGENT_CENTRAL_TAPE",
                  "MI_AGENT_CLIENT_ID", "MI_AGENT_RUN_ID"):
            os.environ.pop(k, None)
        data_source.reset_cache()

    def _client(self):
        from fastapi.testclient import TestClient
        from mi_agent_api.app import app
        return TestClient(app)

    def test_snapshots_discovers_output_not_input(self):
        body = self._client().get("/mi/pipeline/snapshots",
                                  params={"portfolioId": "client_001"}).json()
        # Exactly ONE governed scope — the raw input/ copy is excluded.
        self.assertEqual(len(body["sources"]), 1)
        scope = body["sources"][0]
        self.assertIn("/output/pipeline/", scope["source_file"].replace("\\", "/"))
        self.assertNotIn("/input/", scope["source_file"].replace("\\", "/"))
        self.assertEqual(scope["pipeline_source_folder_date"], "2025-11-01")
        self.assertEqual(scope["pipeline_as_of_date"], "2025-12-01")
        self.assertEqual(scope["run_id"], "mi_2025_11")

    def test_pipeline_snapshot_returns_non_zero_rows(self):
        body = self._client().get(
            "/mi/pipeline/snapshot",
            params={"portfolioId": "client_001/mi_2025_11"}).json()
        self.assertTrue(body["ok"])
        self.assertGreater(body["pipelineRowCount"], 0)
        self.assertGreater(body["pipelineAmount"], 0)
        self.assertTrue(body["stageBreakdown"])
        self.assertTrue(body["availableMetrics"])
        self.assertTrue(body["availableDimensions"])
        self.assertIn("dataQuality", body)
        # Weekly-operational dates, distinct from the funded reporting date.
        self.assertEqual(body["pipelineAsOfDate"], "2025-12-01")
        self.assertEqual(body["pipelineSourceFolderDate"], "2025-11-01")
        self.assertNotIn("reportingDate", body)

    def test_forecast_snapshot_works_from_generated_outputs(self):
        body = self._client().get(
            "/mi/forecast/snapshot",
            params={"portfolioId": "client_001/mi_2025_11"}).json()
        self.assertTrue(body["ok"])
        b = body["forecastBridge"]
        self.assertEqual(b["fundedLoanCount"], 12)           # funded book separate
        self.assertGreater(b["pipelineCaseCount"], 0)
        self.assertTrue(b["pipelineAvailable"])
        self.assertIn(b["forecastReadiness"]["status"], ("ready", "partial"))
        self.assertAlmostEqual(
            b["forecastFundedBalance"],
            round(b["fundedBalance"] + b["weightedExpectedFundedAmount"], 2), places=2)
        # Funded vs pipeline dates kept distinct.
        self.assertEqual(b["fundedReportingDate"], "2025-11-30")
        self.assertEqual(b["pipelineAsOfDate"], "2025-12-01")
        self.assertNotIn("reportingDate", b)


@unittest.skipUnless(_HAS_OPENPYXL, "openpyxl required to read .xlsx pipeline extracts")
class TestXlsxWeeklyExtract(unittest.TestCase):
    """M2L KFI weekly extracts may be .xlsx; discovery selects the latest readable
    weekly file and reads xlsx via the same path as csv."""

    @classmethod
    def setUpClass(cls):
        warnings.simplefilter("ignore")
        cls.tmp = Path(tempfile.mkdtemp(prefix="pl_xlsx_"))
        folder = cls.tmp / "client_001" / "output" / "pipeline" / "2025-11-01"
        folder.mkdir(parents=True)
        # Monthly CSV (older) + a later weekly .xlsx (newer) in the same scope.
        shutil.copy(_FIXTURES / "2025-11-01" / "M2L_KFI_and_Pipeline_2025_11_01.csv",
                    folder / "M2L_KFI_and_Pipeline_2025_11_01.csv")
        pd.read_csv(_WEEKLY).to_excel(
            folder / "M2L KFI and Pipeline 2025_12_08_093000.xlsx", index=False)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def test_latest_xlsx_weekly_is_selected_and_readable(self):
        scope = pc.resolve_pipeline_source(self.tmp, "client_001", "mi_2025_11")
        self.assertIsNotNone(scope)
        self.assertEqual(scope["pipeline_source_folder_date"], "2025-11-01")
        # The later .xlsx weekly is the operational as-of (not the monthly csv).
        self.assertEqual(scope["pipeline_extract_date"], "2025-12-08")
        self.assertEqual(scope["pipeline_as_of_date"], "2025-12-08")
        self.assertTrue(scope["source_file"].endswith(".xlsx"))
        # And it preps to non-zero rows (xlsx read through the same path as csv).
        df, report = pc.load_prepared_pipeline(scope)
        self.assertGreater(report["row_count"], 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
