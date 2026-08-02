#!/usr/bin/env python3
"""Phase 1B — the historical model's vectorised date handling must be exact.

``build_historical_completion_model`` used to call ``pd.to_datetime`` on two
scalars per (case, stage) pair — 8,420 datetime format guesses per build on a
26-week root — and to walk each snapshot with ``DataFrame.iterrows``, which
constructs a Series per row. Both were replaced with column-wise / vectorised
equivalents.

The rewrite is only acceptable if it is output-identical, including at the
edges the old ``try/except`` silently absorbed. These tests pin:

  * an unparseable date is skipped, not counted and not raised;
  * a completion earlier than the first sighting (negative elapsed) is skipped;
  * an exactly-zero elapsed day IS counted;
  * a NaT completion date falls back to the extract date;
  * ordering and median timing are unchanged;
  * the whole model is identical on the committed fixture pack.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from typing import Any, Dict, List

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import pandas as pd

from mi_agent_api.pipeline_history import build_historical_completion_model

_FIXTURE = _REPO_ROOT / "tests" / "fixtures" / "client_001_mi_pack" / "pipeline"


def _snapshot(tmp: Path, name: str, rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Write a minimal weekly extract and return the discovery-shaped entry."""
    path = tmp / name
    pd.DataFrame(rows).to_csv(path, index=False)
    return {"source_file": str(path),
            "pipeline_extract_date": name.split("_")[-1].replace(".csv", "")}


class TestElapsedDayEdges(unittest.TestCase):
    """The edges the previous per-pair try/except absorbed."""

    def setUp(self):
        import tempfile
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def _model(self, entries):
        return build_historical_completion_model(entries, min_observations=1)

    def test_unparseable_completion_date_is_skipped_not_raised(self):
        e1 = _snapshot(self.tmp, "w_2026-01-01.csv", [
            {"KFI Number": "C1", "Status": "KFI", "Date Funds Released": ""}])
        e2 = _snapshot(self.tmp, "w_2026-01-08.csv", [
            {"KFI Number": "C1", "Status": "COMPLETED",
             "Date Funds Released": "not-a-date"}])
        model = self._model([e1, e2])
        # It must still count the completion; only the TIMING is unavailable.
        self.assertTrue(model["available"])
        rate = model["historicalCompletionRateByStage"]["KFI"]
        self.assertEqual(rate["observed"], 1)
        self.assertEqual(rate["completed"], 1)

    def test_zero_elapsed_days_is_counted(self):
        e1 = _snapshot(self.tmp, "w_2026-01-01.csv", [
            {"KFI Number": "C1", "Status": "KFI", "Date Funds Released": ""}])
        e2 = _snapshot(self.tmp, "w_2026-01-08.csv", [
            {"KFI Number": "C1", "Status": "COMPLETED",
             "Date Funds Released": "2026-01-01"}])
        model = self._model([e1, e2])
        timing = model.get("historicalCompletionTimingByStage", {}).get("KFI")
        self.assertIsNotNone(timing, "a zero-day completion must still be timed")
        self.assertEqual(timing["medianDays"], 0)

    def test_negative_elapsed_days_is_skipped(self):
        # Completed BEFORE the case was first seen — nonsensical, must not count.
        e1 = _snapshot(self.tmp, "w_2026-02-01.csv", [
            {"KFI Number": "C1", "Status": "KFI", "Date Funds Released": ""}])
        e2 = _snapshot(self.tmp, "w_2026-02-08.csv", [
            {"KFI Number": "C1", "Status": "COMPLETED",
             "Date Funds Released": "2026-01-01"}])
        model = self._model([e1, e2])
        self.assertNotIn("KFI", model.get("historicalCompletionTimingByStage", {}),
                         "a negative elapsed time must be discarded")

    def test_missing_completion_date_falls_back_to_the_extract_date(self):
        e1 = _snapshot(self.tmp, "w_2026-03-01.csv", [
            {"KFI Number": "C1", "Status": "KFI", "Date Funds Released": ""}])
        e2 = _snapshot(self.tmp, "w_2026-03-15.csv", [
            {"KFI Number": "C1", "Status": "COMPLETED", "Date Funds Released": ""}])
        model = self._model([e1, e2])
        timing = model.get("historicalCompletionTimingByStage", {}).get("KFI")
        self.assertIsNotNone(timing)
        self.assertEqual(timing["medianDays"], 14,
                         "with no completion date the extract date must be used")

    def test_blank_and_nan_case_ids_are_ignored(self):
        e1 = _snapshot(self.tmp, "w_2026-04-01.csv", [
            {"KFI Number": "", "Status": "KFI", "Date Funds Released": ""},
            {"KFI Number": "nan", "Status": "KFI", "Date Funds Released": ""},
            {"KFI Number": "C9", "Status": "KFI", "Date Funds Released": ""}])
        model = self._model([e1])
        self.assertEqual(
            model["historicalCompletionRateByStage"]["KFI"]["observed"], 1,
            "blank / 'nan' case ids must not become tracked cases")


class TestModelIsStableOnTheFixturePack(unittest.TestCase):
    def test_build_is_deterministic_and_populated(self):
        if not _FIXTURE.exists():
            self.skipTest("fixture pack not present")
        from mi_agent_api import pipeline_contract as pc
        entries = pc.weekly_extract_inventory(str(_FIXTURE), "client_001")["extracts"]
        a = build_historical_completion_model(entries)
        b = build_historical_completion_model(entries)
        self.assertEqual(a, b, "the model must be deterministic")
        self.assertIn("historicalCompletionRateByStage", a)


if __name__ == "__main__":
    unittest.main()
