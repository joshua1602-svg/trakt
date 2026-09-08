#!/usr/bin/env python3
"""The app must come up whether or not the governed tape is ready.

WHY THIS FILE EXISTS. On 2026-09-02 trakt-mi-api could not start. The lifespan
called `_warm_caches()` inline, that warm is a governed-tape download plus a
prepare, and the platform gives a container a fixed window (230s) to answer its
startup probe. Three consecutive boots reached "Waiting for application startup"
and none reached "complete": Azure killed each container on the probe, and the
next boot restarted the same download from nothing, because the scratch copy does
not survive a restart. Fourteen minutes of 500s, and the loop could not converge
on its own.

The warm had a `try/except` — it was guarded against FAILING. Slowness was the
mode that took the site down, and nothing guarded that. These tests pin the
distinction: a warm that never finishes must cost the FIRST QUESTION, never the
app's ability to answer at all.
"""
from __future__ import annotations

import threading
import time
import unittest
from unittest import mock

from fastapi.testclient import TestClient

from mi_agent_api import app as app_mod
from mi_agent_api import data_source


class TestStartupDoesNotBlock(unittest.TestCase):
    """Startup completes even when the cache warm never does."""

    def test_a_warm_that_never_finishes_still_lets_the_app_start(self):
        # A warm that blocks forever — the unbounded download, made explicit.
        never = threading.Event()

        def _hangs() -> None:
            never.wait(30)

        with mock.patch.object(app_mod, "_warm_caches", _hangs):
            started = time.monotonic()
            # Entering the context manager RUNS THE LIFESPAN. If the warm were
            # still inline this would sit here until the event timed out.
            with TestClient(app_mod.app) as client:
                elapsed = time.monotonic() - started
                self.assertLess(
                    elapsed, 5.0,
                    "startup waited on the cache warm: %.1fs" % elapsed)
                # And the process genuinely serves while the warm is in flight.
                self.assertEqual(client.get("/").status_code, 200)
            never.set()

    def test_the_warm_actually_runs(self):
        """Non-blocking must not mean never-running: it is still an optimisation."""
        ran = threading.Event()
        with mock.patch.object(app_mod, "_warm_caches", ran.set):
            with TestClient(app_mod.app):
                self.assertTrue(ran.wait(5.0),
                                "the warm never ran; the first query stays cold")


class TestLivenessDoesNotTouchData(unittest.TestCase):
    """`/` is the liveness probe, so it must not become the cold cost it reports."""

    def test_root_never_resolves_the_dataset(self):
        client = TestClient(app_mod.app)
        with mock.patch.object(data_source, "_load_active",
                               side_effect=AssertionError(
                                   "the liveness probe loaded the dataset")):
            data_source.reset_cache()
            body = client.get("/").json()
        self.assertEqual(body["service"], "mi_agent_api")
        # It REPORTS warmth without causing it.
        self.assertIn("warm", body)
        self.assertFalse(body["warm"])


class TestOneLoadNotTwo(unittest.TestCase):
    """The warm thread and the first request must not both fetch the tape.

    Before the lock this was invisible because the warm finished before any
    request arrived. Moving it onto its own thread is exactly what makes the
    race reachable, so the lock is part of that change, not a separate tidy-up.
    """

    def test_concurrent_callers_load_once(self):
        data_source.reset_cache()
        calls = []
        barrier = threading.Barrier(2, timeout=10)

        def _slow_load():
            calls.append(1)
            time.sleep(0.3)          # long enough for the second caller to arrive
            return ("frame", {"kind": "test", "label": "test"})

        with mock.patch.object(data_source, "_load_active", _slow_load), \
                mock.patch.object(data_source, "_source_signature",
                                  return_value="sig-1"):
            def _go():
                barrier.wait()
                data_source._active()

            threads = [threading.Thread(target=_go) for _ in range(2)]
            for t in threads:
                t.start()
            for t in threads:
                t.join(timeout=10)

        self.assertEqual(len(calls), 1,
                         "the tape was fetched %d times concurrently" % len(calls))
        data_source.reset_cache()


if __name__ == "__main__":
    unittest.main()
