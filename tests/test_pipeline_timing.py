#!/usr/bin/env python3
"""tests/test_pipeline_timing.py

Pipeline/funded date-alignment product rules:

  * the selected FUNDED reporting date must NOT truncate pipeline history — the
    pipeline defaults to its LATEST weekly extract even when funded actuals lag;
  * the API exposes BOTH anchors (funded actuals as-of, pipeline extract as-of);
  * a non-blocking timing disclosure is emitted when the pipeline extract is later
    than the funded date, strengthened above the 45-day threshold.

Run: python -m unittest tests.test_pipeline_timing
"""

from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from mi_agent_api.pipeline_timing import timing_disclosure


class TestTimingDisclosureLogic(unittest.TestCase):

    def test_no_disclosure_when_pipeline_not_later(self):
        d = timing_disclosure("2025-11-30", "2025-11-10")
        self.assertEqual(d["level"], "none")
        self.assertIsNone(d["message"])
        self.assertEqual(d["lagDays"], -20)
        # anchors always echoed
        self.assertEqual(d["fundedActualsAsOf"], "2025-11-30")
        self.assertEqual(d["pipelineExtractAsOf"], "2025-11-10")

    def test_info_when_pipeline_later_within_threshold(self):
        d = timing_disclosure("2025-11-30", "2025-12-20")   # 20 days
        self.assertEqual(d["level"], "info")
        self.assertEqual(d["lagDays"], 20)
        self.assertIn("latest weekly extract dated 2025-12-20", d["message"])
        self.assertIn("Funded actuals are as of 2025-11-30", d["message"])

    def test_warning_above_threshold(self):
        d = timing_disclosure("2025-11-30", "2026-01-12")   # 43 days -> still info
        self.assertEqual(d["level"], "info")
        d2 = timing_disclosure("2025-11-30", "2026-01-20")  # 51 days -> warning
        self.assertEqual(d2["level"], "warning")
        self.assertEqual(d2["lagDays"], 51)
        self.assertIn("51 days after", d2["message"])
        self.assertIn("Confirm funded actuals are pending", d2["message"])

    def test_threshold_is_configurable(self):
        d = timing_disclosure("2025-11-30", "2025-12-20", warn_days=10)  # 20 > 10
        self.assertEqual(d["level"], "warning")
        self.assertEqual(d["warnThresholdDays"], 10)

    def test_missing_date_is_none_level(self):
        self.assertEqual(timing_disclosure(None, "2026-01-12")["level"], "none")
        self.assertEqual(timing_disclosure("2025-11-30", None)["level"], "none")


class _EnvGuard:
    def __init__(self, **kw):
        self.kw, self.saved = kw, {}

    def __enter__(self):
        for k, v in self.kw.items():
            self.saved[k] = os.environ.get(k)
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        return self

    def __exit__(self, *a):
        for k, v in self.saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


_SNAP = ("deal_id,current_outstanding_balance,current_valuation_amount,"
         "current_interest_rate,pipeline_stage\n"
         "D1,250000,500000,0.062,Offer\nD2,180000,400000,0.059,Application\n")


class _PipelineFixture:
    """A blob pipeline root with dated weekly extracts THROUGH 2026-01-12 and a
    latest/ pointer, backed by the filesystem storage backend."""

    DATES = ["2025-09-08", "2025-10-13", "2025-11-10", "2025-12-08", "2026-01-12"]
    ROOT = "blob://processed-v2/pipeline/ERE/"
    URI = "blob://processed-v2/pipeline/ERE/latest/pipeline_snapshot.csv"

    def __init__(self, td: str):
        self.blobroot = Path(td) / "blobstore"
        base = self.blobroot / "processed-v2" / "pipeline" / "ERE"
        for d in self.DATES:
            (base / d).mkdir(parents=True, exist_ok=True)
            (base / d / "pipeline_snapshot.csv").write_text(_SNAP)
        (base / "latest").mkdir(parents=True, exist_ok=True)
        (base / "latest" / "pipeline_snapshot.csv").write_text(_SNAP)
        self.scratch = Path(td) / "scratch"

    def env(self, **extra):
        e = {"TRAKT_STORAGE_BACKEND": "file",
             "TRAKT_LOCAL_BLOB_ROOT": str(self.blobroot),
             "MI_AGENT_PIPELINE_ROOT": self.ROOT,
             "MI_AGENT_PIPELINE_URI": self.URI,
             "MI_AGENT_SCRATCH": str(self.scratch),
             "MI_AGENT_PIPELINE_SOURCE": None}
        e.update(extra)
        return _EnvGuard(**e)


def _reset_caches():
    import mi_agent_api.app as app
    app._PIPELINE_MIRROR_CACHE.update(root=None, sig=None, local=None)
    app._PIPELINE_URI_CACHE.update(etag=None, path=None)


class TestPipelineNotTruncatedByFundedDate(unittest.TestCase):

    def test_pipeline_evolution_includes_dates_after_funded(self):
        import mi_agent_api.app as app
        with tempfile.TemporaryDirectory() as td:
            fx = _PipelineFixture(td)
            _reset_caches()
            with fx.env():
                # Selected FUNDED run is 2025-11-30; pipeline data runs to 2026-01-12.
                evo = app.pipeline_evolution(portfolioId="direct_001/2025-11-30")
        dates = [p.get("extract_date") for p in evo.get("periods", [])]
        self.assertIn("2026-01-12", dates)                       # NOT capped at Nov
        self.assertIn("2025-12-08", dates)
        self.assertEqual(max(dates), "2026-01-12")
        self.assertFalse(evo["singlePeriod"])
        # timing disclosure present on the evolution response
        self.assertEqual(evo["pipelineTiming"]["pipelineExtractAsOf"], "2026-01-12")
        self.assertEqual(evo["pipelineTiming"]["fundedActualsAsOf"], "2025-11-30")

    def test_pipeline_snapshot_defaults_to_latest_extract(self):
        import mi_agent_api.app as app
        with tempfile.TemporaryDirectory() as td:
            fx = _PipelineFixture(td)
            _reset_caches()
            with fx.env():
                snap = app.pipeline_snapshot(portfolioId="direct_001/2025-11-30")
        # Even resolved via the latest/ pointer, the as-of is the true latest extract.
        self.assertEqual(snap.get("pipelineAsOfDate"), "2026-01-12")
        # 2025-11-30 -> 2026-01-12 is 43 days (< 45) -> info, not warning.
        self.assertEqual(snap["pipelineTiming"]["level"], "info")
        self.assertEqual(snap["pipelineTiming"]["lagDays"], 43)
        self.assertEqual(snap["pipelineTiming"]["pipelineExtractAsOf"], "2026-01-12")
        self.assertEqual(snap["pipelineTiming"]["fundedActualsAsOf"], "2025-11-30")


class TestForecastAnchorsAndDisclosure(unittest.TestCase):

    def test_forecast_exposes_both_anchors_and_disclosure(self):
        import mi_agent_api.app as app
        with tempfile.TemporaryDirectory() as td:
            fx = _PipelineFixture(td)
            _reset_caches()
            # Funded resolves from a platform canonical for the same client so the
            # bridge has a funded side; pipeline stays the latest extract.
            canon = Path(td) / "platform_canonical_typed.csv"
            canon.write_text(
                "loan_id,source_portfolio_id,current_outstanding_balance,reporting_date\n"
                "L1,direct_001,250000,2025-11-30\n")
            with fx.env(MI_AGENT_PLATFORM_CANONICAL=str(canon),
                        MI_AGENT_ONBOARDING_OUTPUT_ROOT=None):
                env = app.forecast_snapshot(portfolioId="direct_001/2025-11-30")
        timing = env["pipelineTiming"]
        self.assertEqual(timing["fundedActualsAsOf"], "2025-11-30")
        self.assertEqual(timing["pipelineExtractAsOf"], "2026-01-12")
        self.assertIn(timing["level"], ("info", "warning"))
        self.assertGreater(timing["lagDays"], 0)


if __name__ == "__main__":
    unittest.main()
