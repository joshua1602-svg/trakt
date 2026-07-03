#!/usr/bin/env python3
"""tests/test_period_and_skips.py

Real weekly pipeline folders arrive as ``2025_09_08`` (underscores), while funded
folders arrive as ``2025-10-31`` (hyphens). The underscore period failed to parse,
so the backfill SILENTLY skipped the entire weekly pipeline. Two fixes:

  * the path parser normalises an underscore period to canonical hyphen form
    (``2025_09_08`` → ``2025-09-08``), so weekly pipeline folders enumerate; and
  * the backfill REPORTS any folder it could not enumerate (never a silent drop).

Run: python -m unittest tests.test_period_and_skips
"""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from apps.blob_trigger_app import backfill as BF
from apps.blob_trigger_app.layout import Layout
from apps.blob_trigger_app.path_parser import parse_blob_path, PathParseError
from apps.blob_trigger_app.persistence import ProductionPersistence
from apps.blob_trigger_app.source_registry import SourceRegistry
from apps.blob_trigger_app.storage import Storage


class TestPeriodNormalisation(unittest.TestCase):

    def test_underscore_date_period_parses_and_normalises(self):
        pp = parse_blob_path(
            "raw-v2/ERE/direct/pipeline/weekly/direct_001/2025_09_08/M2L KFI and Pipeline.xlsx",
            "raw-v2")
        self.assertEqual(pp.reporting_period, "2025-09-08")   # normalised to hyphens
        self.assertEqual(pp.dataset, "pipeline")
        self.assertEqual(pp.frequency, "weekly")

    def test_underscore_iso_week_normalises(self):
        pp = parse_blob_path(
            "raw-v2/ERE/direct/pipeline/weekly/direct_001/2025_W36/x.csv", "raw-v2")
        self.assertEqual(pp.reporting_period, "2025-W36")

    def test_hyphen_period_unchanged(self):
        pp = parse_blob_path(
            "raw-v2/ERE/direct/funded/monthly/direct_001/2025-10-31/x.csv", "raw-v2")
        self.assertEqual(pp.reporting_period, "2025-10-31")

    def test_genuinely_bad_period_still_fails(self):
        with self.assertRaises(PathParseError):
            parse_blob_path(
                "raw-v2/ERE/direct/pipeline/weekly/direct_001/badperiod/x.csv", "raw-v2")


class TestBackfillScanReportsSkips(unittest.TestCase):

    def _tree(self, root: Path):
        specs = [
            # Weekly pipeline with UNDERSCORE period — must now enumerate.
            ("ERE/direct/pipeline/weekly/direct_001/2025_09_08", "PipelineExtract.csv"),
            # Funded monthly (hyphen) — control.
            ("ERE/direct/funded/monthly/direct_001/2025-10-31", "LoanExtract.csv"),
            # Genuinely malformed period — must be REPORTED as skipped, not dropped.
            ("ERE/direct/pipeline/weekly/direct_001/not-a-real-period", "junk.csv"),
        ]
        for folder, fname in specs:
            d = root / "raw-v2" / folder
            d.mkdir(parents=True, exist_ok=True)
            (d / fname).write_text("a,b,c\n1,2,3\n")
            (d / "_READY.json").write_text("{}")

    def test_weekly_underscore_enumerates_and_bad_folder_reported(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            self._tree(root)
            storage = Storage(root)
            packs, skipped = BF.scan_folders(storage, "raw-v2")

            periods = {(p.dataset, p.frequency): p.reporting_period for p in packs}
            self.assertEqual(periods.get(("pipeline", "weekly")), "2025-09-08")
            self.assertEqual(periods.get(("funded", "monthly")), "2025-10-31")

            reasons = " ".join(s["reason"] for s in skipped)
            folders = " ".join(s["folder"] for s in skipped)
            self.assertIn("path_parse_error", reasons)
            self.assertIn("not-a-real-period", folders)

    def test_dry_run_lists_weekly_and_flags_unparseable(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            self._tree(root)
            storage = Storage(root)
            layout = Layout()
            persistence = ProductionPersistence(storage, layout)
            registry = SourceRegistry("blob://trakt-state/registry/source_registry.yaml",
                                      storage=storage)
            results = BF.run_backfill(storage, persistence, registry, container="raw-v2",
                                      dry_run=True, out_dir=str(root / "out"))
            planned = [r for r in results if r.get("planned_route")]
            unparseable = [r for r in results if r.get("status") == "skipped_unparseable"]
            self.assertTrue(any(r["frequency"] == "weekly" and r["period"] == "2025-09-08"
                                for r in planned))
            self.assertTrue(unparseable)
            self.assertIn("not-a-real-period", unparseable[0]["prefix"])


if __name__ == "__main__":
    unittest.main()
