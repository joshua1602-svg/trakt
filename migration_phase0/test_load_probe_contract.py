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

        def _fake(base, token, method, path, body, pid, timeout):
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
