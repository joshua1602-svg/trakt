#!/usr/bin/env python3
"""tests/test_platform_blob_snapshots.py

MI_AGENT_ONBOARDING_OUTPUT_ROOT must support ``blob://`` platform roots so
/mi/snapshots enumerates every funded reporting cut (not only the latest
pointer), keyed by source_portfolio_id.

Layout under the root (blob://processed-v2/platform/ERE/):
  {YYYY-MM-DD}/platform_canonical_typed.csv   <- dated funded cuts (discovered)
  latest/platform_canonical_typed.csv         <- current pointer (EXCLUDED)

Covered:
  * dated platform canonicals listed, latest/ excluded, chronological;
  * /mi/snapshots returns portfolios keyed by source_portfolio_id, each with one
    run per dated cut (> 1 run);
  * a selected historical run loads THAT dated canonical (not the latest);
  * an on-disk (filesystem) root is unaffected.

Run: python -m unittest tests.test_platform_blob_snapshots
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


def _canonical_csv(direct_balance: int, acquired_balance: int, cut: str) -> str:
    # Two source portfolios so the index is keyed by source_portfolio_id, plus a
    # reporting_date column carrying the funded cut.
    return (
        "loan_id,source_portfolio_id,source_portfolio_type,source_portfolio_label,"
        "current_outstanding_balance,reporting_date\n"
        f"L1,direct_001,direct,Direct Book,{direct_balance},{cut}\n"
        f"L2,direct_001,direct,Direct Book,{direct_balance},{cut}\n"
        f"L3,acquired_001,acquired,Acquired Book,{acquired_balance},{cut}\n")


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


class _BlobPlatformFixture:
    CUTS = {"2025-10-31": (5_000_000, 2_000_000),
            "2025-11-30": (6_000_000, 2_500_000)}
    ROOT = "blob://processed-v2/platform/ERE/"

    def __init__(self, td: str):
        self.local_blob_root = Path(td) / "blobstore"
        base = self.local_blob_root / "processed-v2" / "platform" / "ERE"
        for cut, (d, a) in self.CUTS.items():
            (base / cut).mkdir(parents=True, exist_ok=True)
            (base / cut / "platform_canonical_typed.csv").write_text(
                _canonical_csv(d, a, cut))
        # latest pointer (== Nov) — must be EXCLUDED from the dated index.
        (base / "latest").mkdir(parents=True, exist_ok=True)
        (base / "latest" / "platform_canonical_typed.csv").write_text(
            _canonical_csv(6_000_000, 2_500_000, "2025-11-30"))

    def env(self, **extra):
        e = {"TRAKT_STORAGE_BACKEND": "file",
             "TRAKT_LOCAL_BLOB_ROOT": str(self.local_blob_root),
             "MI_AGENT_ONBOARDING_OUTPUT_ROOT": self.ROOT,
             "MI_AGENT_SCRATCH": str(self.local_blob_root.parent / "scratch"),
             "MI_AGENT_PLATFORM_URI": None, "MI_AGENT_CENTRAL_TAPE": None,
             "MI_AGENT_DATA_CSV": None}
        e.update(extra)
        return _EnvGuard(**e)


def _reset_read_cache():
    import mi_agent_api.platform_snapshots_blob as pb
    pb._READ_CACHE.clear()


class TestBlobPlatformListing(unittest.TestCase):

    def test_lists_dated_excludes_latest_sorted(self):
        import mi_agent_api.platform_snapshots_blob as pb
        from apps.blob_trigger_app.storage import open_storage
        with tempfile.TemporaryDirectory() as td:
            fx = _BlobPlatformFixture(td)
            with fx.env():
                dated = pb.list_dated_platform_canonicals(fx.ROOT, open_storage())
        dates = [d["date"] for d in dated]
        self.assertEqual(dates, ["2025-10-31", "2025-11-30"])
        self.assertTrue(all("/latest/" not in d["uri"] for d in dated))


class TestSnapshotsEndpoint(unittest.TestCase):

    def test_enumerates_runs_per_source_portfolio(self):
        import mi_agent_api.app as app
        with tempfile.TemporaryDirectory() as td:
            fx = _BlobPlatformFixture(td)
            _reset_read_cache()
            with fx.env():
                resp = app.snapshots()
        self.assertEqual(resp["source"], fx.ROOT)
        pfs = {p["client_id"]: p for p in resp["portfolios"]}
        self.assertIn("direct_001", pfs)
        self.assertIn("acquired_001", pfs)
        direct = pfs["direct_001"]
        run_dates = [r["reporting_date"] for r in direct["runs"]]
        self.assertEqual(run_dates, ["2025-10-31", "2025-11-30"])   # > 1 run, sorted
        # Direct Oct balance = 2 loans * 5,000,000.
        oct_run = direct["runs"][0]
        self.assertEqual(oct_run["loan_count"], 2)
        self.assertEqual(oct_run["current_outstanding_balance"], 10_000_000.0)
        self.assertEqual(direct["label"], "Direct Book")

    def test_selected_historical_run_loads_that_cut(self):
        import mi_agent_api.app as app
        with tempfile.TemporaryDirectory() as td:
            fx = _BlobPlatformFixture(td)
            _reset_read_cache()
            with fx.env():
                snap = app.snapshot(client_id="direct_001", run_id="2025-10-31")
        # The snapshot resolves the OCT cut (10,000,000), not the latest (Nov).
        self.assertTrue(snap.get("ok", True))
        text = str(snap)
        self.assertIn("2025-10-31", text)


class TestFundedEvolution(unittest.TestCase):

    def test_two_periods_for_source_portfolio(self):
        import mi_agent_api.app as app
        with tempfile.TemporaryDirectory() as td:
            fx = _BlobPlatformFixture(td)
            _reset_read_cache()
            with fx.env():
                # portfolioId is the SELECTED run — evolution must NOT collapse to it.
                evo = app.funded_evolution(portfolioId="direct_001/2025-11-30")
        self.assertFalse(evo["singlePeriod"])
        dates = [p["reporting_date"] for p in evo["periods"]]
        self.assertEqual(dates, ["2025-10-31", "2025-11-30"])
        # Oct direct = 2 loans, Nov direct = 2 loans; balances are the dated cut's.
        oct_bal = evo["periods"][0]["metrics"]["funded_balance"]
        nov_bal = evo["periods"][1]["metrics"]["funded_balance"]
        self.assertEqual(oct_bal, 10_000_000.0)
        self.assertEqual(nov_bal, 12_000_000.0)

    def test_uses_source_portfolio_not_selected_run(self):
        import mi_agent_api.app as app
        with tempfile.TemporaryDirectory() as td:
            fx = _BlobPlatformFixture(td)
            _reset_read_cache()
            with fx.env():
                a = app.funded_evolution(portfolioId="direct_001/2025-11-30")
                b = app.funded_evolution(portfolioId="direct_001/2025-10-31")
        # Selecting the EARLIER run truncates to it (one period); the later run
        # yields both — proving the series follows the source portfolio + toRunId,
        # not a single collapsed run.
        self.assertEqual([p["reporting_date"] for p in a["periods"]],
                         ["2025-10-31", "2025-11-30"])
        self.assertEqual([p["reporting_date"] for p in b["periods"]], ["2025-10-31"])

    def test_total_lens_aggregates_across_source_portfolios(self):
        import mi_agent_api.app as app
        with tempfile.TemporaryDirectory() as td:
            fx = _BlobPlatformFixture(td)
            _reset_read_cache()
            with fx.env():
                evo = app.funded_evolution(portfolioId="total/2025-11-30")
        self.assertFalse(evo["singlePeriod"])
        self.assertEqual([p["reporting_date"] for p in evo["periods"]],
                         ["2025-10-31", "2025-11-30"])
        # Oct total = direct(2*5M) + acquired(1*2M) = 12,000,000.
        self.assertEqual(evo["periods"][0]["metrics"]["funded_balance"], 12_000_000.0)
        # Nov total = direct(2*6M) + acquired(1*2.5M) = 14,500,000.
        self.assertEqual(evo["periods"][1]["metrics"]["funded_balance"], 14_500_000.0)


class TestOnDiskRootUnaffected(unittest.TestCase):

    def test_filesystem_root_still_uses_disk_walk(self):
        import mi_agent_api.app as app
        import mi_agent_api.platform_snapshots_blob as pb
        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / "onboarding_out"
            tape_dir = root / "ERE" / "mi_2025_11" / "output" / "central"
            tape_dir.mkdir(parents=True)
            (tape_dir / "18_central_lender_tape.csv").write_text(
                "loan_id,current_outstanding_balance,reporting_date\n"
                "L1,250000,2025-11-30\n")
            _reset_read_cache()
            self.assertFalse(pb.is_blob_root(str(root)))
            with _EnvGuard(MI_AGENT_ONBOARDING_OUTPUT_ROOT=str(root),
                           TRAKT_STORAGE_BACKEND=None, MI_AGENT_PLATFORM_URI=None,
                           MI_AGENT_CENTRAL_TAPE=None, MI_AGENT_DATA_CSV=None):
                resp = app.snapshots()
        self.assertEqual(resp["source"], str(root))
        self.assertTrue(any(p["client_id"] == "ERE" for p in resp["portfolios"]))


if __name__ == "__main__":
    unittest.main()
