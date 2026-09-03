#!/usr/bin/env python3
"""The load probe's two promises, pinned.

A capacity test is pointed at production and its output is pasted into chat, so
it has exactly two ways to do harm and both are worth a test rather than a
comment:

  * it must never write the bearer token anywhere — the file is shared;
  * it must never retain a response body — those carry balances, counts and
    answer prose, and the standing rule is that a probe emits route, outcome
    and shape, never figures.

The third test is the one that decides whether the measurement means anything:
a session that fires fewer calls than the browser does measures a fraction of
the real load and reports a box as adequate when it is not.
"""
from __future__ import annotations

import io
import json
import os
import sys
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent))
import load_probe as LP  # noqa: E402


_TOKEN = "SECRET-TOKEN-VALUE-do-not-leak"
_BODY = b'{"answer":"the book holds 958 loans of GBP 196000000","ok":true}'


class _FakeResponse:
    status = 200

    def read(self):
        return _BODY

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


class TestItNeverLeaksTheToken(unittest.TestCase):

    def _run(self, tmp):
        out = tmp / "load.json"
        with mock.patch.dict(os.environ, {"MI_BEARER": _TOKEN}), \
             mock.patch.object(LP.urllib.request, "urlopen",
                               return_value=_FakeResponse()), \
             redirect_stdout(io.StringIO()) as printed:
            LP.main(["--users", "2", "--out", str(out), "--settle", "0"])
        return out.read_text(encoding="utf-8"), printed.getvalue()

    def test_the_token_is_not_in_the_output_file(self):
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            written, _ = self._run(Path(d))
            self.assertNotIn(_TOKEN, written)

    def test_the_token_is_not_printed(self):
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            _, printed = self._run(Path(d))
            self.assertNotIn(_TOKEN, printed)

    def test_it_refuses_to_run_without_a_token_rather_than_guessing(self):
        with mock.patch.dict(os.environ, {"MI_BEARER": ""}), \
             redirect_stdout(io.StringIO()):
            self.assertEqual(LP.main(["--users", "1"]), 2)


class TestItNeverRetainsAnAnswer(unittest.TestCase):
    """Timing and status are diagnostics. A response body is client data."""

    def test_no_response_content_reaches_the_record(self):
        with mock.patch.object(LP.urllib.request, "urlopen",
                               return_value=_FakeResponse()):
            rec = LP._call("https://x/api", _TOKEN, "GET", "/me", None,
                           "ERE/2026-06-30", 10.0)
        blob = json.dumps(rec)
        for leak in ("958", "196000000", "answer", "the book holds"):
            self.assertNotIn(leak, blob,
                             "response content reached the record: %r" % rec)
        # The SHAPE is kept deliberately: a zero-length 200 is worth seeing.
        self.assertEqual(rec["bytes"], len(_BODY))
        self.assertEqual(rec["status"], 200)

    def test_a_transport_failure_records_the_class_not_the_message(self):
        """An exception string can carry the URL, and the URL names a
        portfolio."""
        boom = RuntimeError("failed for https://x/api/mi/snapshot?portfolioId=ERE")
        with mock.patch.object(LP.urllib.request, "urlopen", side_effect=boom):
            rec = LP._call("https://x/api", _TOKEN, "GET", "/mi/snapshot", None,
                           "ERE/2026-06-30", 10.0)
        self.assertEqual(rec["status"], 0)
        self.assertEqual(rec["error"], "RuntimeError")
        self.assertNotIn("ERE", json.dumps(rec)[len('{"endpoint": "/mi/snapshot"'):])


class TestTheSessionIsTheRealPageLoad(unittest.TestCase):

    def test_it_fires_the_whole_browser_burst(self):
        """Measured 2026-09-03: a dashboard load issues nine calls. Fewer here
        and the box is being asked an easier question than users ask it."""
        self.assertGreaterEqual(len(LP.SESSION), 9)

    def test_the_expensive_question_is_included(self):
        posts = [(m, p) for m, p, _ in LP.SESSION if m == "POST"]
        self.assertIn(("POST", "/mi/query"), posts)

    def test_every_user_runs_every_call(self):
        seen = []
        with mock.patch.object(LP, "_call",
                               side_effect=lambda *a, **k: {
                                   "endpoint": a[3], "method": a[2], "status": 200,
                                   "ms": 1, "bytes": 0, "error": "",
                                   "over_gateway_timeout": False}):
            LP._run_round("https://x/api", _TOKEN, "p", 3, 10.0)
        # 3 users x the full session.
        with mock.patch.object(LP, "_call",
                               side_effect=lambda *a, **k: (
                                   seen.append(a[3]) or {
                                       "endpoint": a[3], "method": a[2],
                                       "status": 200, "ms": 1, "bytes": 0,
                                       "error": "", "over_gateway_timeout": False})):
            LP._run_round("https://x/api", _TOKEN, "p", 3, 10.0)
        self.assertEqual(len(seen), 3 * len(LP.SESSION))


class TestItCountsTheGatewayCeiling(unittest.TestCase):
    """46s is where the auth layer abandoned four requests on 2026-09-03. A
    request past it is a 500 to the user however healthy the app is, so the
    summary has to count it as a failure rather than a slow success."""

    def test_a_slow_success_still_counts_as_degraded(self):
        records = [{"endpoint": "/mi/query", "method": "POST", "status": 200,
                    "ms": 50_000, "bytes": 10, "error": "",
                    "over_gateway_timeout": True}]
        s = LP._summarise(records, users=1)
        self.assertEqual(s["over_gateway_timeout"], 1)

    def test_the_threshold_matches_what_was_measured(self):
        self.assertAlmostEqual(LP.GATEWAY_TIMEOUT_S, 46.0, places=1)


if __name__ == "__main__":
    unittest.main()


class TestItToleratesTheHeaderAsCopied(unittest.TestCase):
    """What devtools shows is `Bearer eyJ...`, and that is what gets copied.

    `_call` adds the scheme, so an un-stripped paste sends it twice and every
    request 401s. That would read as a total failure at every concurrency
    level -- a capacity verdict produced by a formatting slip.
    """

    def _token_used(self, env_value):
        seen = {}

        def _fake(base, token, method, path, body, pid, timeout,
                  inflight=None):
            seen["token"] = token
            return {"endpoint": path, "method": method, "status": 200, "ms": 1,
                    "bytes": 0, "error": "", "over_gateway_timeout": False}

        import tempfile
        with tempfile.TemporaryDirectory() as d, \
             mock.patch.dict(os.environ, {"MI_BEARER": env_value}), \
             mock.patch.object(LP, "_call", side_effect=_fake), \
             redirect_stdout(io.StringIO()):
            LP.main(["--users", "1", "--settle", "0",
                     "--out", str(Path(d) / "o.json")])
        return seen["token"]

    def test_a_pasted_scheme_prefix_is_stripped(self):
        self.assertEqual(self._token_used("Bearer " + _TOKEN), _TOKEN)

    def test_a_bare_token_is_untouched(self):
        self.assertEqual(self._token_used(_TOKEN), _TOKEN)

    def test_the_check_is_case_insensitive_and_survives_whitespace(self):
        self.assertEqual(self._token_used("  bearer   " + _TOKEN + "  "), _TOKEN)


class TestBurstIsActuallyABurst(unittest.TestCase):
    """A burst mode that quietly serialises would be worse than none.

    The scenario this exists for — everyone opening the dashboard in the two
    hours after a reporting cycle — is the one the sustained run cannot speak
    to, because it issues nine calls per user one after another and never has
    more than `users` requests open. If `--burst` reported the same shape, a
    "6 users, all 200" result would be read as proof of something it never
    tested.
    """

    def _round(self, users, burst):
        import time as _t

        def _slow(base, token, method, path, body, pid, timeout, inflight=None):
            ctx = inflight if inflight is not None else LP._InFlight()
            with ctx:
                _t.sleep(0.05)          # long enough for overlap to be real
            return {"endpoint": path, "method": method, "status": 200,
                    "ms": 50, "bytes": 0, "error": "",
                    "over_gateway_timeout": False}

        with mock.patch.object(LP, "_call", side_effect=_slow):
            return LP._run_round("https://x/api", _TOKEN, "p", users, 10.0,
                                 burst=burst)

    def test_burst_puts_the_whole_session_in_flight_per_user(self):
        s = self._round(2, burst=True)
        self.assertEqual(s["mode"], "burst")
        self.assertGreater(s["peak_in_flight"], 2, (
            "burst mode never exceeded one request per user — it is "
            "serialising, and the result would describe sustained load"))
        self.assertLessEqual(s["peak_in_flight"], 2 * len(LP.SESSION))

    def test_sustained_keeps_one_request_per_user(self):
        s = self._round(3, burst=False)
        self.assertEqual(s["mode"], "sustained")
        self.assertLessEqual(s["peak_in_flight"], 3)

    def test_both_modes_still_run_every_call(self):
        for burst in (True, False):
            with self.subTest(burst=burst):
                s = self._round(2, burst=burst)
                self.assertEqual(s["requests"], 2 * len(LP.SESSION))

    def test_the_peak_is_reported_so_a_throttled_run_is_visible(self):
        """`users x 9` is what a burst SHOULD reach. Reporting the peak is
        what makes a run that fell short of it detectable: if the client, the
        network or the OS throttled before the server did, the round measured
        the load generator and the number says so."""
        s = self._round(2, burst=True)
        self.assertIn("peak_in_flight", s)
        self.assertIsInstance(s["peak_in_flight"], int)
        self.assertGreaterEqual(s["peak_in_flight"], 1)


class TestAnExpiredTokenIsNotACapacityVerdict(unittest.TestCase):
    """MEASURED 2026-09-03. The cold-burst run returned 54/54 401s in 3.3
    seconds and printed "FIRST DEGRADED AT: 6 concurrent user(s)".

    Everything about that output reads like a capacity ceiling — a DEGRADED
    verdict, 54 failures, a breaking point named. It measured an expired Entra
    token. Believed, it buys a larger App Service plan to fix a credential
    that lapsed while the previous ramp was running, which is exactly how long
    an hour-long token lasts against a ramp plus a restart plus a warm-up.

    A probe may fail to measure. It may not fail in a way that looks like a
    finding.
    """

    def _round_of(self, status):
        def _fake(base, token, method, path, body, pid, timeout, inflight=None):
            return {"endpoint": path, "method": method, "status": status,
                    "ms": 5, "bytes": 0, "error": "",
                    "over_gateway_timeout": False}

        with mock.patch.object(LP, "_call", side_effect=_fake):
            return LP._run_round("https://x/api", _TOKEN, "p", 2, 10.0)

    def test_an_all_401_round_is_flagged_as_not_a_measurement(self):
        s = self._round_of(401)
        self.assertTrue(s["auth_failed"])
        out = io.StringIO()
        with redirect_stdout(out):
            LP._print_round(s)
        self.assertIn("NOT A MEASUREMENT", out.getvalue())
        self.assertNotIn("DEGRADED", out.getvalue())

    def test_403_counts_too(self):
        self.assertTrue(self._round_of(403)["auth_failed"])

    def test_a_real_failure_is_still_a_real_failure(self):
        """The 500s at six burst users were the finding. This must not swallow
        them."""
        s = self._round_of(500)
        self.assertFalse(s["auth_failed"])
        out = io.StringIO()
        with redirect_stdout(out):
            LP._print_round(s)
        self.assertIn("DEGRADED", out.getvalue())

    def test_a_healthy_round_is_untouched(self):
        s = self._round_of(200)
        self.assertFalse(s["auth_failed"])

    def test_a_mixed_round_is_not_written_off_as_auth(self):
        """Some 401s among real traffic is a different problem — and a real
        one. Only a round that is ENTIRELY auth failures measured nothing."""
        recs = [{"endpoint": "/a", "method": "GET", "status": 401, "ms": 5,
                 "bytes": 0, "error": "", "over_gateway_timeout": False},
                {"endpoint": "/b", "method": "GET", "status": 500, "ms": 45000,
                 "bytes": 0, "error": "", "over_gateway_timeout": False}]
        self.assertFalse(LP._auth_failed(recs))

    def test_the_run_stops_rather_than_ramping_on(self):
        """Continuing produces a full ramp of clean-looking failures and a
        confident verdict about a server never reached."""
        def _fake(base, token, method, path, body, pid, timeout, inflight=None):
            return {"endpoint": path, "method": method, "status": 401, "ms": 5,
                    "bytes": 0, "error": "", "over_gateway_timeout": False}

        import tempfile
        with tempfile.TemporaryDirectory() as d, \
             mock.patch.dict(os.environ, {"MI_BEARER": _TOKEN}), \
             mock.patch.object(LP, "_call", side_effect=_fake), \
             redirect_stdout(io.StringIO()) as out:
            rc = LP.main(["--ramp", "1,2,4,6", "--settle", "0",
                          "--out", str(Path(d) / "o.json")])
        self.assertEqual(rc, 3, "an all-auth-failure run must not exit 0")
        self.assertNotIn("FIRST DEGRADED AT", out.getvalue())


class TestTheSessionCanModelTheDeferredPageLoad(unittest.TestCase):
    """This probe calls the API directly, so a DASHBOARD change is invisible
    to it unless the session list is told about one.

    Deferring the speculative forecast prefetch took `forecast/snapshot` and
    `weekly-brief` out of the page-load burst. Re-running with the old
    nine-call list would report no improvement whatever and send the search
    somewhere else -- a measurement that is wrong in a way that looks like a
    result.
    """

    def _labels(self, page_load):
        seen = []

        def _fake(base, token, method, path, body, pid, timeout, inflight=None):
            seen.append(LP._label(path))
            return {"endpoint": LP._label(path), "method": method, "status": 200,
                    "ms": 1, "bytes": 0, "error": "",
                    "over_gateway_timeout": False}

        import tempfile
        with tempfile.TemporaryDirectory() as d, \
             mock.patch.dict(os.environ, {"MI_BEARER": _TOKEN}), \
             mock.patch.object(LP, "_call", side_effect=_fake), \
             redirect_stdout(io.StringIO()):
            LP.main(["--users", "1", "--settle", "0", "--page-load", page_load,
                     "--out", str(Path(d) / "o.json")])
        return seen

    def test_before_issues_the_whole_traced_burst(self):
        self.assertEqual(len(self._labels("before")), len(LP.SESSION))

    def test_after_drops_exactly_the_deferred_calls(self):
        after = self._labels("after")
        for gone in LP.DEFERRED_BY_LAZY_LOAD:
            self.assertNotIn(gone, after)
        self.assertEqual(len(after), len(LP.SESSION) - len(LP.DEFERRED_BY_LAZY_LOAD))

    def test_after_keeps_the_question(self):
        """The query is the point of the product. Dropping it would make
        'after' look better by measuring less."""
        self.assertIn("/mi/query", self._labels("after"))

    def test_before_is_the_default_so_a_rerun_compares_like_with_like(self):
        self.assertEqual(len(self._labels("before")), 9)
