#!/usr/bin/env python3
"""mi_agent_api/tests/test_pipeline_runtime_materialisation.py — Pipeline MI runtime fix.

A clean E2E onboarding/promote of an M2L KFI weekly pipeline extract must:
  * build a NON-empty central pipeline tape (fixes "Central pipeline tape: False
    (0 applications)"), keyed on the KFI / account number;
  * materialise the governed pipeline SOURCE under ``output/pipeline/<folder>/``
    (preserving the weekly-extract file), WITHOUT copying funded loan files;
  * be discoverable by ``/mi/pipeline/snapshots`` and return non-zero rows from
    ``/mi/pipeline/snapshot``;
  * drive ``/mi/forecast/snapshot`` from the generated outputs;
  * keep the funded book separate and unchanged.

Pipeline dates remain weekly-operational (the Dec-1 extract inside the Nov scope),
never conflated with the funded reporting date.
"""

from __future__ import annotations

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

_REGISTRY = str(_REPO_ROOT / "config" / "system" / "fields_registry.yaml")
_WEEKLY = (_REPO_ROOT / "tests" / "fixtures" / "client_001_mi_pack" / "pipeline"
           / "2025-11-01" / "M2L_KFI_and_Pipeline_2025_12_01_115711.csv")


class TestPipelineRuntimeMaterialisation(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        warnings.simplefilter("ignore")
        cls.root = Path(tempfile.mkdtemp(prefix="pl_rt_"))
        inp = cls.root / "input"
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
        proj = cls.root / "proj"
        wf.run_operator_workflow(
            input_dir=str(inp), client_name="Client 001", client_id="client_001",
            run_id="mi_2025_11", mode="mi_only", project_dir=str(proj),
            product_profile="equity_release_lifetime_mortgage")
        rp = storage_paths.resolve_run_paths(
            project_dir=str(proj), input_dir=str(inp), output_root=None,
            client_id="client_001", run_id="mi_2025_11", storage_backend="local",
            input_uri="", output_uri="")
        cls.res = central_tape_builder.build_central_tapes(
            str(proj), rp, _REGISTRY, mode="mi_only")
        cls.output_dir = Path(cls.res["central_lender_tape_path"]).parent.parent

    # ---- promote materialisation -------------------------------------------- #
    def test_central_pipeline_tape_now_created(self):
        # Was "False (0 applications)" before the KFI/account key fix.
        self.assertTrue(self.res["central_pipeline_tape_created"])
        self.assertGreater(self.res["pipeline_count"], 0)

    def test_pipeline_source_materialised_without_funded_pollution(self):
        self.assertEqual(self.res["pipeline_sources_materialised"], 1)
        psd = Path(self.res["pipeline_source_dir"])
        self.assertTrue(psd.is_dir())
        files = [p.name for p in psd.rglob("*") if p.is_file() and p.suffix == ".csv"]
        # Only the M2L KFI weekly extract — never the funded LoanExtract.
        self.assertIn(_WEEKLY.name, files)
        self.assertNotIn("LoanExtract One.csv", files)
        # Preserved under the November source-folder scope.
        self.assertTrue((psd / "2025-11-01" / _WEEKLY.name).exists())

    # ---- API from generated outputs ----------------------------------------- #
    def setUp(self):
        os.environ["MI_AGENT_CENTRAL_TAPE"] = self.res["central_lender_tape_path"]
        os.environ["MI_AGENT_CLIENT_ID"] = "client_001"
        os.environ["MI_AGENT_RUN_ID"] = "mi_2025_11"
        os.environ["MI_AGENT_PIPELINE_ROOT"] = str(self.output_dir)
        data_source.reset_cache()

    def tearDown(self):
        for k in ("MI_AGENT_CENTRAL_TAPE", "MI_AGENT_CLIENT_ID", "MI_AGENT_RUN_ID",
                  "MI_AGENT_PIPELINE_ROOT"):
            os.environ.pop(k, None)
        data_source.reset_cache()

    def _client(self):
        from fastapi.testclient import TestClient
        from mi_agent_api.app import app
        return TestClient(app)

    def test_pipeline_snapshots_discovers_generated_output(self):
        body = self._client().get("/mi/pipeline/snapshots",
                                  params={"portfolioId": "client_001"}).json()
        self.assertGreaterEqual(len(body["sources"]), 1)
        scope = body["sources"][-1]
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
        # Weekly-operational dates, distinct from the funded reporting date.
        self.assertEqual(body["pipelineAsOfDate"], "2025-12-01")
        self.assertEqual(body["pipelineSourceFolderDate"], "2025-11-01")

    def test_forecast_snapshot_works_from_generated_outputs(self):
        body = self._client().get(
            "/mi/forecast/snapshot",
            params={"portfolioId": "client_001/mi_2025_11"}).json()
        self.assertTrue(body["ok"])
        b = body["forecastBridge"]
        self.assertEqual(b["fundedLoanCount"], 12)           # funded book separate
        self.assertGreater(b["pipelineCaseCount"], 0)
        self.assertTrue(b["pipelineAvailable"])
        self.assertAlmostEqual(
            b["forecastFundedBalance"],
            round(b["fundedBalance"] + b["weightedExpectedFundedAmount"], 2), places=2)
        # Funded vs pipeline dates kept distinct.
        self.assertEqual(b["pipelineAsOfDate"], "2025-12-01")
        self.assertNotIn("reportingDate", b)


if __name__ == "__main__":
    unittest.main(verbosity=2)
