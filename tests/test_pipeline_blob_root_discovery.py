#!/usr/bin/env python3
"""tests/test_pipeline_blob_root_discovery.py

MI_AGENT_PIPELINE_ROOT must support ``blob://`` roots so /mi/pipeline/snapshots
(and the evolution + historical-model paths that must share the SAME discovered
dated sources) return the full history of published pipeline snapshots.

Layout under the root (blob://processed-v2/pipeline/ERE/):
  {YYYY-MM-DD}/pipeline_snapshot.csv   <- dated historical snapshots (discovered)
  latest/pipeline_snapshot.csv         <- current pointer (EXCLUDED from history)

Covered:
  * local (filesystem) MI_AGENT_PIPELINE_ROOT still discovers sources unchanged;
  * a blob-style root with multiple dated snapshots returns > 1 source;
  * the latest/ folder is excluded;
  * chronological ordering is stable;
  * evolution + history use the same dated set (>= 2 → multi-week model built).

Run: python -m unittest tests.test_pipeline_blob_root_discovery
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

_ROWS = ("deal_id,current_outstanding_balance,current_valuation_amount,"
         "current_interest_rate,pipeline_stage\n")


def _snapshot_csv(seed: int) -> str:
    # A couple of rich rows so discovery reads a non-empty frame per snapshot.
    return (_ROWS
            + f"D{seed}0,{200000 + seed},{400000 + seed},0.06,Offer\n"
            + f"D{seed}1,{150000 + seed},{300000 + seed},0.061,Application\n")


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


class _BlobRootFixture:
    """A blob:// pipeline root backed by the filesystem storage backend."""

    DATES = ["2025-09-08", "2025-10-13", "2025-11-10", "2026-01-12"]
    ROOT = "blob://processed-v2/pipeline/ERE/"

    def __init__(self, td: str):
        self.local_blob_root = Path(td) / "blobstore"
        base = self.local_blob_root / "processed-v2" / "pipeline" / "ERE"
        for i, d in enumerate(self.DATES):
            (base / d).mkdir(parents=True, exist_ok=True)
            (base / d / "pipeline_snapshot.csv").write_text(_snapshot_csv(i))
        # The current pointer under latest/ — must be EXCLUDED from dated history.
        (base / "latest").mkdir(parents=True, exist_ok=True)
        (base / "latest" / "pipeline_snapshot.csv").write_text(_snapshot_csv(99))

    def env(self, **extra):
        e = {"TRAKT_STORAGE_BACKEND": "file",
             "TRAKT_LOCAL_BLOB_ROOT": str(self.local_blob_root),
             "MI_AGENT_PIPELINE_ROOT": self.ROOT,
             "MI_AGENT_SCRATCH": str(self.local_blob_root.parent / "scratch"),
             # keep other resolution paths out of the way
             "MI_AGENT_PIPELINE_URI": None, "MI_AGENT_PIPELINE_SOURCE": None,
             "MI_AGENT_ONBOARDING_OUTPUT_ROOT": None}
        e.update(extra)
        return _EnvGuard(**e)


def _reset_mirror():
    import mi_agent_api.app as app
    app._PIPELINE_MIRROR_CACHE.update(root=None, sig=None, local=None)


class TestBlobListing(unittest.TestCase):

    def test_lists_dated_excludes_latest_sorted(self):
        import mi_agent_api.app as app
        from apps.blob_trigger_app.storage import open_storage
        with tempfile.TemporaryDirectory() as td:
            fx = _BlobRootFixture(td)
            with fx.env():
                storage = open_storage()
                dated = app._blob_dated_snapshots(fx.ROOT, storage)
        got_dates = [d["date"] for d in dated]
        self.assertEqual(got_dates, sorted(fx.DATES))          # chronological
        self.assertEqual(len(got_dates), 4)                    # all dated present
        self.assertTrue(all("/latest/" not in d["uri"] for d in dated))  # excluded


class TestBlobRootSnapshots(unittest.TestCase):

    def test_blob_root_returns_multiple_dated_sources(self):
        import mi_agent_api.app as app
        with tempfile.TemporaryDirectory() as td:
            fx = _BlobRootFixture(td)
            _reset_mirror()
            with fx.env():
                resp = app.pipeline_snapshots(portfolioId="ERE")
        sources = resp["sources"]
        self.assertGreater(len(sources), 1)                    # > 1 source
        self.assertEqual(len(sources), 4)
        dates = [s.get("pipeline_as_of_date") for s in sources]
        self.assertEqual(dates, sorted(fx.DATES))              # stable chronological
        # latest/ never becomes a dated source.
        self.assertNotIn("latest", " ".join(
            s.get("pipeline_source_folder", "") for s in sources).lower())
        # The endpoint reports the ORIGINAL blob root, not the local mirror.
        self.assertEqual(resp["source"], fx.ROOT)

    def test_client_and_run_id_resolved_on_mirror(self):
        import mi_agent_api.app as app
        with tempfile.TemporaryDirectory() as td:
            fx = _BlobRootFixture(td)
            _reset_mirror()
            with fx.env():
                resp = app.pipeline_snapshots(portfolioId="ERE")
        self.assertTrue(all(s["client_id"] == "ERE" for s in resp["sources"]))


class TestBlobRootEvolutionAndHistory(unittest.TestCase):

    def test_evolution_uses_same_dated_sources(self):
        import mi_agent_api.app as app
        with tempfile.TemporaryDirectory() as td:
            fx = _BlobRootFixture(td)
            _reset_mirror()
            with fx.env():
                evo = app.pipeline_evolution(portfolioId="ERE")
        # One period per dated snapshot (latest/ excluded), chronological.
        edates = [p.get("extract_date") for p in evo.get("periods", [])]
        self.assertEqual(edates, sorted(fx.DATES))

    def test_history_built_from_multiple_dated_snapshots(self):
        import mi_agent_api.app as app
        with tempfile.TemporaryDirectory() as td:
            fx = _BlobRootFixture(td)
            _reset_mirror()
            # The latest pointer being set must NOT suppress multi-week history.
            uri = ("blob://processed-v2/pipeline/ERE/latest/pipeline_snapshot.csv")
            with fx.env(MI_AGENT_PIPELINE_URI=uri):
                model = app._pipeline_history("ERE")
        self.assertIsNotNone(model)
        self.assertGreaterEqual(int(model.get("uniqueWeeklyExtractsUsed", 0)), 2)


class TestLocalRootUnchanged(unittest.TestCase):

    def test_local_filesystem_root_still_discovers(self):
        import mi_agent_api.app as app
        import mi_agent_api.pipeline_contract as pc
        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / "pipeline_root"
            # A classic M2L weekly extract fixture (dated filename).
            folder = root / "ERE" / "2025-11-01"
            folder.mkdir(parents=True)
            (folder / "M2L_KFI_and_Pipeline_2025_11_10_120000.csv").write_text(
                _snapshot_csv(1))
            _reset_mirror()
            with _EnvGuard(MI_AGENT_PIPELINE_ROOT=str(root),
                           TRAKT_STORAGE_BACKEND=None, MI_AGENT_PIPELINE_URI=None,
                           MI_AGENT_PIPELINE_SOURCE=None,
                           MI_AGENT_ONBOARDING_OUTPUT_ROOT=None):
                # _materialise_pipeline_root returns a filesystem root unchanged.
                self.assertEqual(app._materialise_pipeline_root(str(root)), str(root))
                resp = app.pipeline_snapshots(portfolioId="ERE")
            self.assertEqual(len(resp["sources"]), 1)
            self.assertEqual(resp["sources"][0]["pipeline_extract_date"], "2025-11-10")


if __name__ == "__main__":
    unittest.main()
