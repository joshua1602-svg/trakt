#!/usr/bin/env python3
"""Phase 1A — structural performance budgets.

These assert REPEATED-WORK COUNTS, not milliseconds. Wall-clock is machine- and
load-dependent and would make CI flaky; the counts are what actually regressed
and what the Phase 1A caches fix. If someone reintroduces an eager historical
model, drops a cache key component, or adds a per-period re-preparation, these
fail immediately and for an obvious reason.

Measured through the real app against a small representative fixture, using the
same :mod:`trakt_core.perf` instrumentation that runs in production.
"""

from __future__ import annotations

import json
import logging
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any, Dict

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


class _Sink(logging.Handler):
    def __init__(self) -> None:
        super().__init__(level=logging.INFO)
        self.records = []

    def emit(self, record: logging.LogRecord) -> None:
        try:
            msg = record.getMessage()
            if msg.startswith("perf "):
                self.records.append(json.loads(msg.split("perf ", 1)[1]))
        except Exception:  # noqa: BLE001
            pass


class TestWarmRequestBudgets(unittest.TestCase):
    """A WARM request must not repeat expensive preparation."""

    @classmethod
    def setUpClass(cls):
        from tests.perf.generate_fixture import build
        from tests.perf.measure import configure_env, snapshot_env

        cls.tmp = tempfile.TemporaryDirectory()
        # Several periods and several weeks, so a per-period or per-extract
        # regression shows up as a count > 0 rather than being masked.
        build(Path(cls.tmp.name), months=4, funded_rows=80,
              weeks=4, pipeline_rows=50)
        # Same contract as test_http_conditional: configure_env mutates the
        # process environment, so snapshot before and restore after.
        cls._env_snapshot = snapshot_env()
        configure_env(Path(cls.tmp.name))

        from fastapi.testclient import TestClient
        from mi_agent_api.app import app
        cls.client = TestClient(app)
        cls.client.__enter__()
        cls.sink = _Sink()
        cls.logger = logging.getLogger("mi_agent_api.perf")
        cls.logger.setLevel(logging.INFO)
        cls.logger.addHandler(cls.sink)
        cls.logger.propagate = False

    @classmethod
    def tearDownClass(cls):
        cls.logger.removeHandler(cls.sink)
        try:
            cls.client.__exit__(None, None, None)
        finally:
            try:
                cls.tmp.cleanup()
            finally:
                from tests.perf.measure import restore_env
                restore_env(cls._env_snapshot)

    def _warm(self, path: str, **params) -> Dict[str, Any]:
        """Issue the request twice and return the SECOND request's record."""
        self.client.get(path, params=params)
        self.sink.records.clear()
        self.client.get(path, params=params)
        self.assertTrue(self.sink.records, f"no perf record for {path}")
        return self.sink.records[-1]

    def _calls(self, record: Dict[str, Any], stage: str) -> int:
        return int((record.get("stage_calls") or {}).get(stage, 0))

    def _counter(self, record: Dict[str, Any], prefix: str) -> int:
        return sum(v for k, v in (record.get("counters") or {}).items()
                   if k.startswith(prefix))

    # -- the historical completion model ---------------------------------- #
    def test_warm_pipeline_routes_never_rebuild_the_historical_model(self):
        for path in ("/mi/evolution/pipeline", "/mi/evolution/funnel",
                     "/mi/forecast/snapshot", "/mi/forecast/extrapolation"):
            rec = self._warm(path, portfolioId="ERE/2026-07-31")
            self.assertEqual(
                self._calls(rec, "historical_completion_model_build"), 0,
                f"{path} rebuilt the historical model on a warm request")

    def test_a_query_that_does_not_need_history_never_builds_it(self):
        self.sink.records.clear()
        self.client.post("/mi/query", json={
            "question": "What is the total funded balance by region?",
            "portfolioId": "ERE/2026-07-31", "datasetContext": "funded"})
        rec = self.sink.records[-1]
        self.assertEqual(
            self._calls(rec, "historical_completion_model_build"), 0,
            "a point-in-time question must never build the historical model")

    # -- file re-reading --------------------------------------------------- #
    def test_warm_requests_do_not_re_read_source_files(self):
        for path in ("/mi/evolution/pipeline", "/mi/evolution/funnel",
                     "/mi/evolution/funded", "/mi/risk-limits",
                     "/mi/concentration-tests", "/mi/geo/exposure",
                     "/mi/cohorts", "/mi/portfolio-context"):
            rec = self._warm(path, portfolioId="ERE/2026-07-31")
            self.assertEqual(
                self._counter(rec, "file.read."), 0,
                f"{path} re-read source files on a warm request")

    def test_warm_evolution_does_not_re_prepare_funded_periods(self):
        rec = self._warm("/mi/evolution/funded", portfolioId="ERE/2026-07-31")
        self.assertEqual(self._calls(rec, "funded_prep"), 0,
                         "funded evolution re-prepared its periods when warm")

    # -- conditional responses -------------------------------------------- #
    def test_a_matching_validator_does_no_analytical_work(self):
        first = self.client.get("/mi/snapshot",
                                params={"portfolioId": "ERE/2026-07-31"})
        etag = first.headers.get("etag")
        self.assertTrue(etag)
        self.sink.records.clear()
        second = self.client.get("/mi/snapshot",
                                 params={"portfolioId": "ERE/2026-07-31"},
                                 headers={"If-None-Match": etag})
        self.assertEqual(second.status_code, 304)
        rec = self.sink.records[-1]
        self.assertEqual(self._calls(rec, "funded_snapshot_compute"), 0,
                         "a 304 must skip the computation, not just the body")
        self.assertEqual(rec.get("response_bytes"), 0)


class TestKillSwitchesRestoreThePreviousPath(unittest.TestCase):
    """Every Phase 1A optimisation must be revertible by configuration."""

    def test_serving_cache_switch_is_honoured(self):
        import os

        from mi_agent_api import serving_cache as sc
        cache = sc.BoundedCache("killswitch_probe", max_entries=4)
        calls = []
        cache.get_or_build("k", lambda: calls.append(1) or "v")
        cache.get_or_build("k", lambda: calls.append(1) or "v")
        self.assertEqual(len(calls), 1, "cache should be active by default")
        os.environ["TRAKT_SERVING_CACHE"] = "off"
        try:
            cache.get_or_build("k", lambda: calls.append(1) or "v")
            self.assertEqual(len(calls), 2,
                             "the kill switch must restore the calculation path")
        finally:
            os.environ.pop("TRAKT_SERVING_CACHE", None)

    def test_http_cache_switch_is_honoured(self):
        import os

        from mi_agent_api import http_cache
        self.assertTrue(http_cache.enabled())
        os.environ["TRAKT_HTTP_CACHE"] = "off"
        try:
            self.assertFalse(http_cache.enabled())
        finally:
            os.environ.pop("TRAKT_HTTP_CACHE", None)

    def test_instrumentation_switch_is_honoured(self):
        import os

        from trakt_core import perf
        self.assertTrue(perf.enabled())
        os.environ["TRAKT_PERF_INSTRUMENTATION"] = "off"
        try:
            self.assertFalse(perf.enabled())
            with perf.collect(route="/x") as c:
                self.assertIsNone(c, "disabled instrumentation must not collect")
        finally:
            os.environ.pop("TRAKT_PERF_INSTRUMENTATION", None)


if __name__ == "__main__":
    unittest.main()
