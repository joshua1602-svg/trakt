#!/usr/bin/env python3
"""Phase 1A item 7 — conditional responses on the stable MI read routes.

A conditional response is only acceptable if it cannot leak across a tenant, a
portfolio scope or a run; cannot survive a new published run; cannot be issued
for an error; and cannot bypass authentication. Each of those is pinned here.
"""

from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from mi_agent_api import http_cache


class TestEtagConstruction(unittest.TestCase):
    def _tag(self, **kw):
        base = dict(route="mi.snapshot", tenant="ERE", scope="total",
                    identity=["blob://x", "etag-1"])
        base.update(kw)
        return http_cache.build_etag(**base)

    def test_deterministic(self):
        self.assertEqual(self._tag(), self._tag())
        self.assertTrue(self._tag().startswith('W/"'))

    def test_tenant_scope_run_and_source_all_discriminate(self):
        base = self._tag()
        self.assertNotEqual(base, self._tag(tenant="OTHER"))
        self.assertNotEqual(base, self._tag(scope="direct_001"))
        self.assertNotEqual(base, self._tag(route="mi.evolution.funded"))
        self.assertNotEqual(base, self._tag(identity=["blob://x", "etag-2"]))

    def test_unknown_identity_yields_no_validator(self):
        self.assertIsNone(self._tag(identity=["blob://x", None]))
        self.assertIsNone(self._tag(identity=[]))

    def test_disabled_by_env(self):
        os.environ["TRAKT_HTTP_CACHE"] = "off"
        try:
            self.assertFalse(http_cache.enabled())
            self.assertIsNone(http_cache.begin(object(), route="r", identity=["a"]))
        finally:
            os.environ.pop("TRAKT_HTTP_CACHE", None)


class TestSuccessGating(unittest.TestCase):
    class _Resp:
        def __init__(self):
            self.headers = {}

    def test_successful_payload_gets_the_validator(self):
        resp = self._Resp()
        http_cache.finish(resp, 'W/"abc"', {"ok": True, "kpis": []})
        self.assertEqual(resp.headers["ETag"], 'W/"abc"')
        self.assertEqual(resp.headers["Cache-Control"], "private, no-cache")

    def test_error_payload_is_never_given_a_validator(self):
        for payload in ({"ok": False, "error": "boom"},
                        {"available": False, "reason": "no config"},
                        {"error": "storage unavailable"}):
            resp = self._Resp()
            http_cache.finish(resp, 'W/"abc"', payload)
            self.assertNotIn("ETag", resp.headers, payload)
            self.assertEqual(resp.headers["Cache-Control"], "no-store", payload)

    def test_reason_alone_is_not_a_failure(self):
        resp = self._Resp()
        http_cache.finish(resp, 'W/"abc"', {"available": True, "reason": "note"})
        self.assertIn("ETag", resp.headers)

    def test_caching_is_private_never_public(self):
        self.assertIn("private", http_cache.CACHE_CONTROL)
        self.assertNotIn("public", http_cache.CACHE_CONTROL)

    def test_no_long_lived_freshness_window(self):
        """A new approved run must never be masked by a stale window."""
        self.assertIn("no-cache", http_cache.CACHE_CONTROL)
        self.assertNotIn("max-age=", http_cache.CACHE_CONTROL)


class TestConditionalOverHttp(unittest.TestCase):
    """End-to-end against the real app and the representative fixture."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = None
        fixture = os.environ.get("MI_PERF_FIXTURE")
        if not fixture or not Path(fixture).exists():
            import tempfile
            from tests.perf.generate_fixture import build
            cls.tmp = tempfile.TemporaryDirectory()
            build(Path(cls.tmp.name), months=2, funded_rows=60,
                  weeks=2, pipeline_rows=40)
            fixture = cls.tmp.name
        # configure_env mutates a dozen process-wide variables, including the
        # auth switch. Snapshot before, restore in tearDownClass — otherwise
        # every test that runs after this class inherits a fixture's idea of
        # the environment.
        from tests.perf.measure import configure_env, snapshot_env
        cls._env_snapshot = snapshot_env()
        configure_env(Path(fixture))
        from fastapi.testclient import TestClient
        from mi_agent_api.app import app
        cls.client = TestClient(app)
        cls.client.__enter__()
        cls.params = {"portfolioId": "ERE/2026-07-31"}

    @classmethod
    def tearDownClass(cls):
        try:
            cls.client.__exit__(None, None, None)
        finally:
            try:
                if cls.tmp:
                    cls.tmp.cleanup()
            finally:
                from tests.perf.measure import restore_env
                restore_env(cls._env_snapshot)

    def test_cold_request_returns_a_validator(self):
        r = self.client.get("/mi/snapshot", params=self.params)
        self.assertEqual(r.status_code, 200)
        self.assertTrue(r.headers.get("etag"), "a stable read must be revalidatable")

    def test_matching_validator_returns_304_with_no_body(self):
        first = self.client.get("/mi/snapshot", params=self.params)
        etag = first.headers["etag"]
        second = self.client.get("/mi/snapshot", params=self.params,
                                 headers={"If-None-Match": etag})
        self.assertEqual(second.status_code, 304)
        self.assertEqual(second.content, b"")
        self.assertEqual(second.headers.get("etag"), etag)

    def test_unchanged_body_when_revalidation_misses(self):
        first = self.client.get("/mi/snapshot", params=self.params)
        second = self.client.get("/mi/snapshot", params=self.params,
                                 headers={"If-None-Match": 'W/"stale"'})
        self.assertEqual(second.status_code, 200)
        self.assertEqual(first.json(), second.json(),
                         "a conditional miss must return the same body as before")

    def test_a_different_scope_does_not_reuse_the_validator(self):
        base = self.client.get("/mi/snapshot", params=self.params)
        scoped = self.client.get(
            "/mi/snapshot",
            params={**self.params, "portfolioContext": "direct_001"},
            headers={"If-None-Match": base.headers["etag"]})
        self.assertEqual(scoped.status_code, 200,
                         "another portfolio scope must never 304 on this validator")

    def test_a_different_run_does_not_reuse_the_validator(self):
        base = self.client.get("/mi/snapshot", params=self.params)
        other = self.client.get("/mi/snapshot",
                                params={"portfolioId": "ERE/2026-06-30"},
                                headers={"If-None-Match": base.headers["etag"]})
        self.assertNotEqual(other.status_code, 304,
                            "another run must never 304 on this validator")

    def test_a_different_route_does_not_reuse_the_validator(self):
        base = self.client.get("/mi/snapshot", params=self.params)
        other = self.client.get("/mi/evolution/funded", params=self.params,
                                headers={"If-None-Match": base.headers["etag"]})
        self.assertNotEqual(other.status_code, 304)

    def test_server_timing_is_present(self):
        r = self.client.get("/mi/snapshot", params=self.params)
        self.assertIn("server-timing", {k.lower() for k in r.headers})


class TestConditionalRequiresAuthentication(unittest.TestCase):
    """A conditional request is authenticated exactly like an unconditional one."""

    def test_unauthenticated_conditional_request_is_refused_not_304(self):
        import importlib

        # Capture whatever was set before — including "not set at all" — so the
        # finally block can put it back exactly.
        #
        # The previous version hard-coded "false" on the way out, which is the
        # value the perf fixture happens to want, not the value this process
        # started with. In a full-suite run that leaked auth-disabled into every
        # test that ran afterwards, and
        # tests/test_agent_identity_and_api.py::
        # test_the_easy_auth_guard_exempts_the_agent_prefix then found
        # ``auth_guard`` permitting a path it asserts is guarded. A leaked auth
        # switch is the worst kind of test pollution: it makes a SECURITY
        # assertion pass or fail depending on collection order, and it fails in
        # the permissive direction.
        previous = os.environ.get("MI_AGENT_AUTH_ENABLED")
        os.environ["MI_AGENT_AUTH_ENABLED"] = "true"
        try:
            from mi_agent_api import auth as auth_mod
            importlib.reload(auth_mod)
            from fastapi.testclient import TestClient
            from mi_agent_api.app import app
            with TestClient(app) as client:
                r = client.get("/mi/snapshot",
                               params={"portfolioId": "ERE/2026-07-31"},
                               headers={"If-None-Match": 'W/"anything"'})
            self.assertIn(r.status_code, (401, 403),
                          "auth must run before any conditional short-circuit")
            self.assertNotEqual(r.status_code, 304)
        finally:
            if previous is None:
                os.environ.pop("MI_AGENT_AUTH_ENABLED", None)
            else:
                os.environ["MI_AGENT_AUTH_ENABLED"] = previous
            from mi_agent_api import auth as auth_mod
            importlib.reload(auth_mod)


if __name__ == "__main__":
    unittest.main()
