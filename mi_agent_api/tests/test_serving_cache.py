#!/usr/bin/env python3
"""Phase 1A — the serving caches must be safe before they are fast.

Covers the guarantees the caches are only acceptable because of:
tenant and portfolio isolation, immutable source identity, bounded eviction,
protection of mutable cached objects, no caching of failures, and graceful
degradation when the cache is disabled or unusable.
"""

from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from mi_agent_api import serving_cache as sc


class TestCacheKey(unittest.TestCase):
    def test_same_inputs_produce_the_same_key(self):
        a = sc.key_for(tenant="ERE", scope="total", identity=["u", "etag-1"])
        b = sc.key_for(tenant="ERE", scope="total", identity=["u", "etag-1"])
        self.assertEqual(a, b)
        self.assertIsNotNone(a)

    def test_different_tenant_is_a_different_key(self):
        a = sc.key_for(tenant="ERE", scope="total", identity=["u", "etag-1"])
        b = sc.key_for(tenant="OTHER", scope="total", identity=["u", "etag-1"])
        self.assertNotEqual(a, b, "a cache entry must never cross tenants")

    def test_different_scope_is_a_different_key(self):
        a = sc.key_for(tenant="ERE", scope="total", identity=["u", "etag-1"])
        b = sc.key_for(tenant="ERE", scope="direct_001", identity=["u", "etag-1"])
        self.assertNotEqual(a, b, "a cache entry must never cross portfolios")

    def test_changed_source_identity_is_a_different_key(self):
        a = sc.key_for(tenant="ERE", scope="total", identity=["u", "etag-1"])
        b = sc.key_for(tenant="ERE", scope="total", identity=["u", "etag-2"])
        self.assertNotEqual(a, b, "a republished source must not be reused")

    def test_changed_methodology_version_is_a_different_key(self):
        a = sc.key_for(tenant="ERE", scope="total", identity=["u"], version="1")
        b = sc.key_for(tenant="ERE", scope="total", identity=["u"], version="2")
        self.assertNotEqual(a, b)

    def test_unidentifiable_source_is_not_cacheable(self):
        self.assertIsNone(sc.key_for(tenant="ERE", scope="total",
                                     identity=["u", None]))
        self.assertIsNone(sc.key_for(tenant="ERE", scope="total", identity=[]))

    def test_duplicate_filenames_with_different_content_do_not_collide(self):
        """Two extracts sharing a basename but not their bytes must differ."""
        a = sc.key_for(tenant="ERE", scope="total",
                       identity=[("/a/pipeline_snapshot.csv", "111:10")])
        b = sc.key_for(tenant="ERE", scope="total",
                       identity=[("/b/pipeline_snapshot.csv", "222:20")])
        self.assertNotEqual(a, b)

    def test_scope_token_distinguishes_portfolios(self):
        class Scope:
            def __init__(self, cid, ids):
                self.context_id, self.portfolio_ids, self.filters = cid, ids, {}

        self.assertEqual(sc.scope_token(None), "total")
        self.assertNotEqual(sc.scope_token(Scope("direct", ["direct_001"])),
                            sc.scope_token(Scope("acquired", ["acquired_001"])))

    def test_model_fingerprint_tracks_stage_rates(self):
        a = sc.model_fingerprint({"historicalCompletionRateByStage": {"KFI": 0.2}})
        b = sc.model_fingerprint({"historicalCompletionRateByStage": {"KFI": 0.3}})
        self.assertNotEqual(a, b, "a different model must re-prepare the frame")
        self.assertEqual(sc.model_fingerprint(None), "none")


class TestBoundedCache(unittest.TestCase):
    def setUp(self):
        self.cache = sc.BoundedCache("unit_test", max_entries=3)
        self.cache.clear()

    def test_hit_avoids_a_second_build(self):
        calls = []
        for _ in range(3):
            self.cache.get_or_build("k", lambda: calls.append(1) or "v")
        self.assertEqual(len(calls), 1)
        self.assertEqual(self.cache.hits, 2)

    def test_eviction_is_bounded_and_least_recently_used(self):
        for i in range(5):
            self.cache.get_or_build(f"k{i}", lambda i=i: i)
        self.assertEqual(len(self.cache), 3, "the cache must stay bounded")
        self.assertIs(self.cache.peek("k0"), sc._MISSING)
        self.assertEqual(self.cache.peek("k4"), 4)

    def test_none_key_bypasses_and_never_stores(self):
        calls = []
        for _ in range(2):
            self.cache.get_or_build(None, lambda: calls.append(1) or "v")
        self.assertEqual(len(calls), 2, "an unidentifiable source is recomputed")
        self.assertEqual(len(self.cache), 0)

    def test_a_failing_build_is_not_cached(self):
        calls = []

        def boom():
            calls.append(1)
            raise ValueError("source unreadable")

        for _ in range(2):
            with self.assertRaises(ValueError):
                self.cache.get_or_build("k", boom)
        self.assertEqual(len(calls), 2, "a failure must be retried, not served")
        self.assertEqual(len(self.cache), 0,
                         "a failed build must never be published")

    def test_cached_none_is_distinguished_from_absent(self):
        calls = []
        for _ in range(2):
            self.cache.get_or_build("k", lambda: calls.append(1) or None)
        self.assertEqual(len(calls), 1, "a legitimate None must be cached")

    def test_disabled_by_env_falls_back_to_the_calculation(self):
        os.environ["TRAKT_SERVING_CACHE_UNIT_TEST"] = "off"
        try:
            calls = []
            for _ in range(3):
                self.cache.get_or_build("k", lambda: calls.append(1) or "v")
            self.assertEqual(len(calls), 3,
                             "the kill switch must restore the original path")
        finally:
            os.environ.pop("TRAKT_SERVING_CACHE_UNIT_TEST", None)

    def test_global_kill_switch(self):
        os.environ["TRAKT_SERVING_CACHE"] = "off"
        try:
            calls = []
            for _ in range(2):
                self.cache.get_or_build("k", lambda: calls.append(1) or "v")
            self.assertEqual(len(calls), 2)
        finally:
            os.environ.pop("TRAKT_SERVING_CACHE", None)

    def test_concurrent_cold_requests_do_not_corrupt_the_cache(self):
        import threading
        barrier = threading.Barrier(8)
        results = []

        def worker():
            barrier.wait()
            results.append(self.cache.get_or_build("k", lambda: "v"))

        threads = [threading.Thread(target=worker) for _ in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        self.assertEqual(results, ["v"] * 8)
        self.assertEqual(len(self.cache), 1)


class TestFileIdentity(unittest.TestCase):
    def test_identity_changes_when_a_file_is_rewritten(self):
        import tempfile
        import time
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "extract.csv"
            path.write_text("a,b\n1,2\n")
            first = sc.file_identity(path)
            time.sleep(0.01)
            path.write_text("a,b\n1,2\n3,4\n")   # corrected upload
            self.assertNotEqual(first, sc.file_identity(path),
                                "a corrected upload must invalidate reuse")

    def test_missing_file_has_no_identity(self):
        self.assertIsNone(sc.file_identity("/nonexistent/path/x.csv"))


class TestPreparedPipelineIsolation(unittest.TestCase):
    """A cached frame must not be contaminated by a later request."""

    def test_prepared_frame_is_not_shared_mutably(self):
        from mi_agent_api import pipeline_contract as pc
        fixture = (_REPO_ROOT / "tests" / "fixtures" / "client_001_mi_pack"
                   / "pipeline" / "2025-11-01" / "M2L_KFI_and_Pipeline_2025_11_01.csv")
        if not fixture.exists():
            self.skipTest("fixture pack not present")
        pc._PIPELINE_PREP_CACHE.clear()
        first, _ = pc.load_prepared_pipeline(str(fixture))
        first["__injected_by_a_previous_request"] = 1
        second, _ = pc.load_prepared_pipeline(str(fixture))
        self.assertNotIn("__injected_by_a_previous_request", second.columns,
                         "one request's column must not leak into the next")

    def test_report_is_not_shared_mutably(self):
        from mi_agent_api import pipeline_contract as pc
        fixture = (_REPO_ROOT / "tests" / "fixtures" / "client_001_mi_pack"
                   / "pipeline" / "2025-11-01" / "M2L_KFI_and_Pipeline_2025_11_01.csv")
        if not fixture.exists():
            self.skipTest("fixture pack not present")
        pc._PIPELINE_PREP_CACHE.clear()
        _, first = pc.load_prepared_pipeline(str(fixture))
        first["__injected"] = True
        _, second = pc.load_prepared_pipeline(str(fixture))
        self.assertNotIn("__injected", second)

    def test_unreadable_source_still_raises_and_is_not_cached(self):
        from mi_agent_api import pipeline_contract as pc
        with self.assertRaises(FileNotFoundError):
            pc.load_prepared_pipeline("/nonexistent/pipeline_snapshot.csv")


if __name__ == "__main__":
    unittest.main()
