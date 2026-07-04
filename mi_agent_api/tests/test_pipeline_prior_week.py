#!/usr/bin/env python3
"""mi_agent_api/tests/test_pipeline_prior_week.py

Week-on-week prior-week aggregates for the pipeline tiles. The backend must:
  * pick the prior weekly extract (prefer exactly 7 days before; else the latest
    earlier extract);
  * aggregate it with the SAME preparation as the current snapshot;
  * NEVER fabricate a prior week (None when no earlier extract / unreadable file);
  * handle zero cases / null amounts safely;
  * surface the block as ``priorWeek`` on the pipeline snapshot.
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


def _entry(path: str, extract_date: str, folder_date: str = "2025-11-01") -> dict:
    return {
        "source_file": path,
        "pipeline_extract_date": extract_date,
        "pipeline_source_folder_date": folder_date,
    }


class TestSelectPriorWeekExtract(unittest.TestCase):
    """Pure selection logic — no IO."""

    def test_prefers_exact_seven_days_prior(self):
        weekly = [
            _entry("a", "2025-11-17"),
            _entry("b", "2025-11-24"),  # exactly 7 days before 12-01
            _entry("c", "2025-11-28"),  # closer, but not exactly 7 days
            _entry("d", "2025-12-01"),  # current
        ]
        prior = pc.select_prior_week_extract(weekly, "2025-12-01")
        self.assertEqual(prior["pipeline_extract_date"], "2025-11-24")

    def test_falls_back_to_latest_earlier_when_no_exact_week(self):
        weekly = [
            _entry("a", "2025-11-10"),
            _entry("b", "2025-11-20"),  # latest earlier (no exact 7-day prior)
            _entry("c", "2025-12-01"),  # current
        ]
        prior = pc.select_prior_week_extract(weekly, "2025-12-01")
        self.assertEqual(prior["pipeline_extract_date"], "2025-11-20")

    def test_returns_none_when_no_earlier_extract(self):
        weekly = [_entry("c", "2025-12-01")]  # only the current snapshot
        self.assertIsNone(pc.select_prior_week_extract(weekly, "2025-12-01"))

    def test_multiple_priors_picks_seven_day_among_them(self):
        weekly = [
            _entry("a", "2025-10-06"),
            _entry("b", "2025-11-03"),
            _entry("c", "2025-11-24"),  # exact 7-day prior to 12-01
            _entry("d", "2025-12-01"),
        ]
        prior = pc.select_prior_week_extract(weekly, "2025-12-01")
        self.assertEqual(prior["pipeline_extract_date"], "2025-11-24")

    def test_unparseable_or_missing_as_of_yields_none(self):
        weekly = [_entry("a", "2025-11-24"), _entry("b", "2025-12-01")]
        self.assertIsNone(pc.select_prior_week_extract(weekly, None))
        self.assertIsNone(pc.select_prior_week_extract([], "2025-12-01"))


def _write_pipeline_csv(path: Path, rows: int, amount: float | None = 100_000.0) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "Account Number": [f"A{i}" for i in range(rows)],
        "KFI Number": [f"K{i}" for i in range(rows)],
        "Status": ["Offer"] * rows,
    }
    if amount is not None:
        data["Loan Amount"] = [amount] * rows
    pd.DataFrame(data).to_csv(path, index=False)


class TestComputePriorWeekAggregates(unittest.TestCase):
    def setUp(self):
        warnings.simplefilter("ignore")
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def _source(self, weekly, as_of):
        return {"pipeline_as_of_date": as_of, "weekly_files": weekly}

    def test_happy_path_aggregates_prior_extract(self):
        prior_file = self.root / "M2L KFI and Pipeline 2025_11_24.csv"
        cur_file = self.root / "M2L KFI and Pipeline 2025_12_01.csv"
        _write_pipeline_csv(prior_file, rows=4, amount=100_000.0)
        _write_pipeline_csv(cur_file, rows=6, amount=100_000.0)
        weekly = [_entry(str(prior_file), "2025-11-24"), _entry(str(cur_file), "2025-12-01")]
        agg = pc.compute_prior_week_aggregates(self._source(weekly, "2025-12-01"))
        self.assertIsNotNone(agg)
        self.assertEqual(agg["snapshotDate"], "2025-11-24")
        self.assertEqual(agg["pipelineRowCount"], 4)
        self.assertIsNotNone(agg["pipelineAmount"])
        self.assertGreater(agg["pipelineAmount"], 0)
        self.assertEqual(agg["sourceFile"], "M2L KFI and Pipeline 2025_11_24.csv")

    def test_no_prior_returns_none(self):
        cur_file = self.root / "M2L KFI and Pipeline 2025_12_01.csv"
        _write_pipeline_csv(cur_file, rows=6)
        weekly = [_entry(str(cur_file), "2025-12-01")]
        self.assertIsNone(pc.compute_prior_week_aggregates(self._source(weekly, "2025-12-01")))

    def test_zero_case_prior_handled_safely(self):
        prior_file = self.root / "M2L KFI and Pipeline 2025_11_24.csv"
        cur_file = self.root / "M2L KFI and Pipeline 2025_12_01.csv"
        _write_pipeline_csv(prior_file, rows=0)
        _write_pipeline_csv(cur_file, rows=6)
        weekly = [_entry(str(prior_file), "2025-11-24"), _entry(str(cur_file), "2025-12-01")]
        agg = pc.compute_prior_week_aggregates(self._source(weekly, "2025-12-01"))
        self.assertIsNotNone(agg)
        self.assertEqual(agg["pipelineRowCount"], 0)

    def test_missing_amount_column_handled_safely(self):
        prior_file = self.root / "M2L KFI and Pipeline 2025_11_24.csv"
        cur_file = self.root / "M2L KFI and Pipeline 2025_12_01.csv"
        _write_pipeline_csv(prior_file, rows=3, amount=None)  # no Loan Amount column
        _write_pipeline_csv(cur_file, rows=6)
        weekly = [_entry(str(prior_file), "2025-11-24"), _entry(str(cur_file), "2025-12-01")]
        agg = pc.compute_prior_week_aggregates(self._source(weekly, "2025-12-01"))
        self.assertIsNotNone(agg)
        self.assertEqual(agg["pipelineRowCount"], 3)
        # amount may be 0/None but must never raise.

    def test_unreadable_prior_file_yields_none(self):
        cur_file = self.root / "M2L KFI and Pipeline 2025_12_01.csv"
        _write_pipeline_csv(cur_file, rows=6)
        weekly = [
            _entry(str(self.root / "missing_2025_11_24.csv"), "2025-11-24"),
            _entry(str(cur_file), "2025-12-01"),
        ]
        self.assertIsNone(pc.compute_prior_week_aggregates(self._source(weekly, "2025-12-01")))


class TestSnapshotPriorWeekField(unittest.TestCase):
    """The pipeline snapshot must carry the additive ``priorWeek`` block."""

    def setUp(self):
        warnings.simplefilter("ignore")
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.pipe = self.root / "client_001" / "pipeline" / "2025-11-01"
        _write_pipeline_csv(self.pipe / "M2L KFI and Pipeline 2025_11_24.csv", rows=4)
        _write_pipeline_csv(self.pipe / "M2L KFI and Pipeline 2025_12_01.csv", rows=6)

    def tearDown(self):
        self._tmp.cleanup()

    def _semantics(self):
        from mi_agent.mi_query_validator import load_mi_semantics
        from mi_agent_api.data_source import semantics_path
        return load_mi_semantics(semantics_path())

    def test_priorWeek_present_when_history_exists(self):
        source = pc.resolve_pipeline_source(self.root, "client_001", "mi_2025_11")
        self.assertIsNotNone(source)
        df, report = pc.load_prepared_pipeline(source)
        prior = pc.compute_prior_week_aggregates(source)
        snap = pc.compute_pipeline_snapshot(
            df, report, self._semantics(), client_id="client_001",
            run_id="mi_2025_11", source=source, prior_week=prior)
        self.assertIn("priorWeek", snap)
        self.assertIsNotNone(snap["priorWeek"])
        self.assertEqual(snap["priorWeek"]["snapshotDate"], "2025-11-24")
        self.assertEqual(snap["priorWeek"]["pipelineRowCount"], 4)

    def test_priorWeek_null_when_omitted(self):
        source = pc.resolve_pipeline_source(self.root, "client_001", "mi_2025_11")
        df, report = pc.load_prepared_pipeline(source)
        snap = pc.compute_pipeline_snapshot(
            df, report, self._semantics(), client_id="client_001",
            run_id="mi_2025_11", source=source)  # no prior_week passed
        self.assertIn("priorWeek", snap)
        self.assertIsNone(snap["priorWeek"])


class TestLatestPointerPriorWeekEnrichment(unittest.TestCase):
    """Regression (Task 5): a source resolved via the ``latest/`` pointer (a
    single CSV with no embedded history) must be enriched with the governed
    dated-extract window so prior-week tile deltas still compute. Reproduces the
    acceptance scenario: latest extract 2026-01-12, prior extract 2026-01-05.
    """

    def setUp(self):
        warnings.simplefilter("ignore")
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.folder = self.root / "client_001" / "pipeline" / "2026-01-01"
        self.prior = self.folder / "M2L KFI and Pipeline 2026_01_05.csv"
        self.latest = self.folder / "M2L KFI and Pipeline 2026_01_12.csv"
        _write_pipeline_csv(self.prior, rows=8, amount=100_000.0)
        _write_pipeline_csv(self.latest, rows=11, amount=100_000.0)
        self._env = {}
        for k in ("MI_AGENT_PIPELINE_URI", "MI_AGENT_PIPELINE_SOURCE",
                  "MI_AGENT_PIPELINE_ROOT"):
            self._env[k] = __import__("os").environ.pop(k, None)
        import os
        # Simulate the blob latest/ pointer resolving to the latest local CSV,
        # with the discovery root holding both dated extracts.
        os.environ["MI_AGENT_PIPELINE_SOURCE"] = str(self.latest)
        os.environ["MI_AGENT_PIPELINE_ROOT"] = str(self.root)

    def tearDown(self):
        import os
        for k, v in self._env.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        self._tmp.cleanup()

    def test_latest_pointer_source_carries_weekly_window(self):
        from mi_agent_api import app
        source = app._resolve_pipeline_source("client_001", "mi_2026_01")
        self.assertIsNotNone(source)
        self.assertEqual(source["pipeline_as_of_date"], "2026-01-12")
        dates = {e.get("pipeline_extract_date") for e in source.get("weekly_files", [])}
        self.assertIn("2026-01-05", dates)
        self.assertIn("2026-01-12", dates)

    def test_prior_week_selected_from_enriched_window(self):
        from mi_agent_api import app
        source = app._resolve_pipeline_source("client_001", "mi_2026_01")
        agg = pc.compute_prior_week_aggregates(source)
        self.assertIsNotNone(agg)
        self.assertEqual(agg["snapshotDate"], "2026-01-05")
        self.assertEqual(agg["pipelineRowCount"], 8)


if __name__ == "__main__":
    unittest.main()
