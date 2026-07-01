#!/usr/bin/env python3
"""tests/test_platform_discovery — portfolio/lens discovery from the platform canonical.

When MI_AGENT_PLATFORM_URI is configured (no on-disk onboarding root), both
/mi/snapshots (portfolio + reporting-date dropdown) and /mi/source-portfolios
(lenses) must derive from the loaded platform canonical — not snapshot discovery.
Also: a blank source_portfolio_label must fall back to the source_portfolio_id,
never the string "nan".
"""

from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import pandas as pd
from fastapi.testclient import TestClient

_REPO = Path(__file__).resolve().parents[2]
_SYNTH = _REPO / "synthetic_demo" / "output" / "SYNTHETIC_ERE_Portfolio_012026_canonical_typed.csv"


def _platform_env(blobroot: Path, scratch: Path) -> dict:
    env = {k: v for k, v in os.environ.items()
           if not k.startswith("MI_AGENT_")
           and k not in ("WEBSITE_INSTANCE_ID", "WEBSITE_SITE_NAME",
                         "TRAKT_BLOB_CONNECTION", "MI_AGENT_ONBOARDING_OUTPUT_ROOT")}
    env.update({
        "TRAKT_STORAGE_BACKEND": "file",
        "TRAKT_LOCAL_BLOB_ROOT": str(blobroot),
        "MI_AGENT_PLATFORM_URI":
            "blob://processed-v2/platform/ERE/latest/platform_canonical_typed.csv",
        "MI_AGENT_SCRATCH": str(scratch),
    })
    return env


def _write_canonical(blobroot: Path, *, blank_label: bool = False,
                     date_col: str = "reporting_date",
                     date_value: str = "2026-01-31") -> None:
    dest = (blobroot / "processed-v2" / "platform" / "ERE" / "latest"
            / "platform_canonical_typed.csv")
    dest.parent.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(_SYNTH)
    df["source_portfolio_id"] = "direct_001"
    df["source_portfolio_type"] = "direct"
    df["source_portfolio_label"] = (float("nan") if blank_label else "Direct Book 001")
    for c in ("reporting_date", "data_cut_off_date", "cut_off_date"):
        if c in df.columns:
            df = df.drop(columns=[c])
    df[date_col] = date_value                       # only this date column present
    df.to_csv(dest, index=False)


@unittest.skipUnless(_SYNTH.exists(), "synthetic canonical fixture not present")
class TestPlatformDiscovery(unittest.TestCase):

    def _client(self, td: Path, *, blank_label: bool = False, **canon) -> TestClient:
        blobroot = td / "blobstore"
        _write_canonical(blobroot, blank_label=blank_label, **canon)
        env = _platform_env(blobroot, td / "scratch")
        self._ctx = mock.patch.dict(os.environ, env, clear=True)
        self._ctx.start()
        from mi_agent_api import data_source as DS
        DS.reset_cache()
        from mi_agent_api.app import app
        return TestClient(app)

    def tearDown(self):
        from mi_agent_api import data_source as DS
        DS.reset_cache()
        if getattr(self, "_ctx", None):
            self._ctx.stop()

    def test_snapshots_derived_from_source_portfolio_id(self):
        # Portfolio entries come from source_portfolio_id → direct_001 selectable.
        with tempfile.TemporaryDirectory() as td:
            c = self._client(Path(td))
            idx = c.get("/mi/snapshots").json()
            self.assertEqual(len(idx["portfolios"]), 1)
            pf = idx["portfolios"][0]
            self.assertEqual(pf["client_id"], "direct_001")   # not "ERE"
            self.assertEqual(pf.get("source_portfolio_id"), "direct_001")
            self.assertEqual(pf["label"], "Direct Book 001")
            self.assertEqual(len(pf["runs"]), 1)
            self.assertEqual(pf["runs"][0]["run_id"], "latest")
            self.assertEqual(pf["runs"][0]["reporting_date"], "2026-01-31")
            self.assertGreater(pf["runs"][0]["loan_count"], 0)
            self.assertNotEqual(idx["source"], "unavailable")

    def test_reporting_date_from_data_cut_off_date(self):
        # Live-shaped: no reporting_date column, only data_cut_off_date populated.
        with tempfile.TemporaryDirectory() as td:
            c = self._client(Path(td), date_col="data_cut_off_date", date_value="2026-01-31")
            idx = c.get("/mi/snapshots").json()
            run = idx["portfolios"][0]["runs"][0]
            self.assertEqual(run["reporting_date"], "2026-01-31")   # NOT null
            self.assertIsNotNone(run["reporting_date"])

    def test_snapshot_loads_direct_001_latest(self):
        with tempfile.TemporaryDirectory() as td:
            c = self._client(Path(td), date_col="data_cut_off_date")
            snap = c.get("/mi/snapshot?portfolioId=direct_001/latest").json()
            self.assertTrue(snap.get("ok"))
            self.assertTrue(snap.get("kpis"))
            self.assertEqual((snap.get("portfolio") or {}).get("reporting_date"), "2026-01-31")

    def test_source_portfolios_lenses_from_platform_canonical(self):
        with tempfile.TemporaryDirectory() as td:
            c = self._client(Path(td), blank_label=False)
            sp = c.get("/mi/source-portfolios").json()
            ids = [l["id"] for l in sp["lenses"]]
            self.assertIn("total", ids)
            self.assertIn("direct", ids)
            self.assertIn("direct_001", ids)
            self.assertTrue(sp["available"])

    def test_blank_label_falls_back_to_id_not_nan(self):
        with tempfile.TemporaryDirectory() as td:
            c = self._client(Path(td), blank_label=True)
            sp = c.get("/mi/source-portfolios").json()
            cohort = next(l for l in sp["lenses"] if l["id"] == "direct_001")
            self.assertEqual(cohort["label"], "direct_001")     # not "nan"
            self.assertNotEqual(cohort["label"].lower(), "nan")

    def test_label_preserved_when_present(self):
        with tempfile.TemporaryDirectory() as td:
            c = self._client(Path(td), blank_label=False)
            sp = c.get("/mi/source-portfolios").json()
            cohort = next(l for l in sp["lenses"] if l["id"] == "direct_001")
            self.assertEqual(cohort["label"], "Direct Book 001")


class TestPlatformReportingDate(unittest.TestCase):
    """Reporting-date derivation priority (never null when a date exists)."""

    def _rd(self, df, env):
        base = {k: v for k, v in os.environ.items() if not k.startswith("MI_AGENT_")}
        base.update(env)
        with mock.patch.dict(os.environ, base, clear=True):
            from mi_agent_api import app as A
            return A._platform_reporting_date(df, "latest")

    _URI_LATEST = "blob://processed-v2/platform/ERE/latest/platform_canonical_typed.csv"

    def test_prefers_reporting_date_column(self):
        df = pd.DataFrame({"reporting_date": ["2026-01-31"],
                           "data_cut_off_date": ["2025-12-31"]})
        self.assertEqual(self._rd(df, {"MI_AGENT_PLATFORM_URI": self._URI_LATEST}),
                         "2026-01-31")

    def test_falls_back_to_data_cut_off_date(self):
        df = pd.DataFrame({"data_cut_off_date": ["2026-01-31"]})   # no reporting_date
        self.assertEqual(self._rd(df, {"MI_AGENT_PLATFORM_URI": self._URI_LATEST}),
                         "2026-01-31")

    def test_parses_date_period_from_platform_uri(self):
        df = pd.DataFrame({"loan_id": [1, 2]})                     # no date columns
        env = {"MI_AGENT_PLATFORM_URI":
               "blob://processed-v2/platform/ERE/2026-01-31/platform_canonical_typed.csv"}
        self.assertEqual(self._rd(df, env), "2026-01-31")

    def test_parses_month_period_from_platform_uri_to_month_end(self):
        df = pd.DataFrame({"loan_id": [1]})
        env = {"MI_AGENT_PLATFORM_URI":
               "blob://processed-v2/platform/ERE/2026-02/platform_canonical_typed.csv"}
        self.assertEqual(self._rd(df, env), "2026-02-28")

    def test_env_override(self):
        df = pd.DataFrame({"loan_id": [1]})
        self.assertEqual(
            self._rd(df, {"MI_AGENT_PLATFORM_URI": self._URI_LATEST,
                          "MI_AGENT_REPORTING_DATE": "2026-03-31"}),
            "2026-03-31")

    def test_scans_any_date_like_column_last(self):
        df = pd.DataFrame({"loan_id": [1], "servicer_cutoff_date": ["2026-01-31"]})
        self.assertEqual(self._rd(df, {"MI_AGENT_PLATFORM_URI": self._URI_LATEST}),
                         "2026-01-31")

    def test_null_only_when_no_date_anywhere(self):
        df = pd.DataFrame({"loan_id": [1], "balance": [100.0]})
        self.assertIsNone(self._rd(df, {"MI_AGENT_PLATFORM_URI": self._URI_LATEST}))


if __name__ == "__main__":
    unittest.main()
