#!/usr/bin/env python3
"""mi_agent_api/tests/test_pipeline_source_selection.py

Pipeline MI source-selection correctness. The CURRENT pipeline snapshot must be
the latest valid governed weekly extract ordered by parsed extract date across
ALL source folders — never the latest file inside whichever folder sorts first.
Historical evidence must deduplicate the same weekly file (cross-folder copies,
or .xlsx/.csv of the same extract), and the current snapshot date must stay
distinct from the historical observation window start.
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

import pandas as pd

from mi_agent_api import pipeline_contract as pc


def _write(path: Path, rows: int = 5) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame({
        "Account Number": [f"A{i}" for i in range(rows)],
        "KFI Number": [f"K{i}" for i in range(rows)],
        "Status": ["Offer"] * rows,
        "Loan Amount": [100_000.0] * rows,
    })
    if path.suffix.lower() == ".xlsx":
        df.to_excel(path, index=False)
    else:
        df.to_csv(path, index=False)
    return path


class TestPipelineSourceSelection(unittest.TestCase):
    def setUp(self):
        warnings.simplefilter("ignore")
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.pipe = self.root / "client_001" / "pipeline"
        # The LATEST extract (Dec 1) deliberately lives in the EARLIEST folder
        # (2025-10-01) so a folder-first selection would wrongly miss it. It is
        # ALSO copied into the 2025-11-01 folder, and exists as both .xlsx and
        # .csv — those are the same weekly extract and must dedupe to one.
        _write(self.pipe / "2025-10-01" / "M2L KFI and Pipeline 2025_10_01.xlsx")
        _write(self.pipe / "2025-10-01" / "M2L KFI and Pipeline 2025_12_01_115711.xlsx")
        _write(self.pipe / "2025-10-01" / "M2L KFI and Pipeline 2025_12_01_115711.csv")
        _write(self.pipe / "2025-11-01" / "M2L KFI and Pipeline 2025_11_01.xlsx")
        _write(self.pipe / "2025-11-01" / "M2L KFI and Pipeline 2025_12_01_115711.xlsx")

    def tearDown(self):
        self._tmp.cleanup()

    def _resolve(self):
        return pc.resolve_pipeline_source(self.root, "client_001", "mi_2025_11")

    # 1. latest extract selected across folders
    def test_latest_extract_selected_across_folders(self):
        scope = self._resolve()
        self.assertEqual(scope["pipeline_as_of_date"], "2025-12-01")
        self.assertIn("2025_12_01", scope["current_pipeline_source_file"])

    # 2. earliest folder not selected merely for sorting first
    def test_earliest_folder_not_selected_for_sorting_first(self):
        scope = self._resolve()
        # The 2025-10-01 folder sorts first but the current extract is Dec, NOT the
        # Oct extract that folder also contains.
        self.assertNotEqual(scope["pipeline_as_of_date"], "2025-10-01")
        self.assertNotIn("2025_10_01", scope["current_pipeline_source_file"])

    # 3. 2025-12-01 selected as current
    def test_current_snapshot_is_2025_12_01(self):
        scope = self._resolve()
        self.assertEqual(scope["current_pipeline_snapshot_date"], "2025-12-01")
        self.assertEqual(scope["pipeline_extract_date"], "2025-12-01")

    # 4. historical evidence deduplicates
    def test_historical_evidence_deduplicates(self):
        inv = pc.weekly_extract_inventory(self.root, "client_001")
        # 5 files scanned -> 3 unique extracts (Oct, Nov, Dec).
        self.assertEqual(inv["sourceFilesScanned"], 5)
        self.assertEqual(inv["uniqueWeeklyExtractsUsed"], 3)
        self.assertEqual(inv["duplicatesExcluded"], 2)
        dates = sorted(e["pipeline_extract_date"] for e in inv["extracts"])
        self.assertEqual(dates, ["2025-10-01", "2025-11-01", "2025-12-01"])

    # 5. .xlsx/.csv not double-counted (governed primary preferred)
    def test_xlsx_and_csv_not_double_counted(self):
        inv = pc.weekly_extract_inventory(self.root, "client_001")
        dec = [e for e in inv["extracts"] if e["pipeline_extract_date"] == "2025-12-01"]
        self.assertEqual(len(dec), 1)
        # The governed primary representation (.xlsx) wins over the .csv re-export.
        self.assertTrue(dec[0]["source_file"].endswith(".xlsx"))
        self.assertEqual(inv["primarySourcePreference"], "xlsx_over_csv")

    # 6. evidence reports both scanned and unique
    def test_evidence_reports_scanned_and_unique(self):
        model = pc.build_pipeline_history(self.root, "client_001")
        from mi_agent_api.pipeline_history import historical_model_evidence
        ev = historical_model_evidence(model, "configured_stage_rate")
        self.assertEqual(ev["sourceFilesScanned"], 5)
        self.assertEqual(ev["uniqueWeeklyExtractsUsed"], 3)
        self.assertEqual(ev["duplicatesExcluded"], 2)
        # Unique extracts used must not exceed files scanned.
        self.assertLessEqual(ev["uniqueWeeklyExtractsUsed"], ev["sourceFilesScanned"])

    # 7. forecast uses latest snapshot
    def test_forecast_uses_latest_snapshot(self):
        import os
        os.environ["MI_AGENT_PIPELINE_ROOT"] = str(self.root)
        try:
            from fastapi.testclient import TestClient
            from mi_agent_api.app import app
            body = TestClient(app).get(
                "/mi/forecast/snapshot",
                params={"portfolioId": "client_001/mi_2025_11"}).json()
            self.assertEqual(body["pipelineAsOfDate"], "2025-12-01")
            self.assertEqual(body["pipelineSnapshot"]["currentPipelineSnapshotDate"],
                             "2025-12-01")
        finally:
            os.environ.pop("MI_AGENT_PIPELINE_ROOT", None)

    # 8. current date distinct from observation window start
    def test_current_date_distinct_from_window_start(self):
        scope = self._resolve()
        self.assertEqual(scope["historical_observation_window_start"], "2025-10-01")
        self.assertEqual(scope["historical_observation_window_end"], "2025-12-01")
        self.assertNotEqual(scope["current_pipeline_snapshot_date"],
                            scope["historical_observation_window_start"])

    def test_snapshot_exposes_separated_fields(self):
        import os
        os.environ["MI_AGENT_PIPELINE_ROOT"] = str(self.root)
        try:
            from fastapi.testclient import TestClient
            from mi_agent_api.app import app
            body = TestClient(app).get(
                "/mi/pipeline/snapshot",
                params={"portfolioId": "client_001/mi_2025_11"}).json()
            self.assertEqual(body["currentPipelineSnapshotDate"], "2025-12-01")
            self.assertIn("2025_12_01", body["currentPipelineSourceFile"])
            self.assertEqual(body["historicalObservationWindowStart"], "2025-10-01")
            self.assertEqual(body["historicalObservationWindowEnd"], "2025-12-01")
            self.assertEqual(body["uniqueWeeklyExtractsUsed"], 3)
            self.assertEqual(body["sourceFilesScanned"], 5)
            self.assertEqual(body["duplicatesExcluded"], 2)
            self.assertEqual(body["primarySourcePreference"], "xlsx_over_csv")
        finally:
            os.environ.pop("MI_AGENT_PIPELINE_ROOT", None)


if __name__ == "__main__":
    unittest.main(verbosity=2)
