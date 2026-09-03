#!/usr/bin/env python3
"""The replay probe's promises.

Its whole purpose is to answer "what moved", so the verdict logic is the thing
under test — a probe that cannot see a regression is worse than no probe,
because it reports a clean bill of health.

Three properties matter, and each is a lesson from 2026-09-03:

  * a question that ANSWERED before and does not now is a REGRESSION, and the
    run exits non-zero. Re-running only the failures would never see one;
  * a question whose prior runs DISAGREED has no "before". "Summarise the
    funded portfolio." answered at 14:52:31 and errored 47 seconds later, and
    counting that as a regression (or as a pass) would both be fiction;
  * no answer text reaches the output. The probe reads the same envelope a
    client does, and that envelope carries balances.
"""
from __future__ import annotations

import io
import json
import os
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent))
import replay_probe as RP  # noqa: E402


def _log(rows):
    fh = tempfile.NamedTemporaryFile("w", suffix=".json", delete=False)
    json.dump({"ok": True, "count": len(rows), "queries": rows}, fh)
    fh.close()
    return fh.name


class TestTheCorpusComesFromTheLog(unittest.TestCase):

    def test_it_reads_the_api_envelope(self):
        c = RP.load_corpus(_log([{"question": "a", "outcome": "ANSWERED"}]))
        self.assertEqual([x["question"] for x in c], ["a"])

    def test_it_also_reads_a_bare_list(self):
        fh = tempfile.NamedTemporaryFile("w", suffix=".json", delete=False)
        json.dump([{"question": "a", "outcome": "ERROR"}], fh)
        fh.close()
        self.assertEqual(RP.load_corpus(fh.name)[0]["prior"], "ERROR")

    def test_duplicates_collapse_to_one_question(self):
        """"Summarise the current pipeline." appears ~25 times live. Asking it
        25 times measures the same thing 25 times and costs 25 Opus calls."""
        c = RP.load_corpus(_log([{"question": "a", "outcome": "ANSWERED"}] * 25))
        self.assertEqual(len(c), 1)
        self.assertEqual(c[0]["prior_counts"], {"ANSWERED": 25})

    def test_disagreeing_priors_become_MIXED(self):
        c = RP.load_corpus(_log([{"question": "a", "outcome": "ANSWERED"},
                                 {"question": "a", "outcome": "ERROR"}]))
        self.assertEqual(c[0]["prior"], RP.MIXED)
        self.assertEqual(c[0]["prior_counts"], {"ANSWERED": 1, "ERROR": 1})

    def test_order_is_the_log_order(self):
        c = RP.load_corpus(_log([{"question": "b", "outcome": "ANSWERED"},
                                 {"question": "a", "outcome": "ANSWERED"}]))
        self.assertEqual([x["question"] for x in c], ["b", "a"])


class TestTheVerdictSeesRegressions(unittest.TestCase):

    def test_answered_then_not_is_a_regression(self):
        self.assertEqual(RP._verdict("ANSWERED", "ERROR"), RP.REGRESSED)
        self.assertEqual(RP._verdict("ANSWERED", "REFUSED"), RP.REGRESSED)

    def test_not_answered_then_answered_is_fixed(self):
        self.assertEqual(RP._verdict("ERROR", "ANSWERED"), RP.FIXED)
        self.assertEqual(RP._verdict("REFUSED", "ANSWERED"), RP.FIXED)

    def test_a_mixed_prior_is_never_called_a_regression(self):
        """There is no single "before" to have regressed from. Calling it one
        would manufacture a finding out of known flakiness."""
        for now in ("ANSWERED", "REFUSED", "ERROR"):
            with self.subTest(now=now):
                self.assertEqual(RP._verdict(RP.MIXED, now), RP.WAS_MIXED)

    def test_a_refusal_becoming_an_error_is_not_progress(self):
        self.assertEqual(RP._verdict("REFUSED", "ERROR"), RP.STILL_FAILING)


class TestItNeverEmitsAnAnswer(unittest.TestCase):

    def test_no_answer_text_reaches_the_record(self):
        body = json.dumps({
            "ok": True, "answer": "the book holds 958 loans of £196,000,000",
            "metadata": {"route": "portfolio_summary"}}).encode()

        class _Resp:
            status = 200
            def read(self): return body
            def __enter__(self): return self
            def __exit__(self, *a): return False

        with mock.patch.object(RP.urllib.request, "urlopen", return_value=_Resp()):
            rec = RP._ask("https://x/api", "tok", "q", None, "p", 10.0)
        blob = json.dumps(rec)
        for leak in ("958", "196,000,000", "the book holds", "answer"):
            self.assertNotIn(leak, blob, "answer content leaked: %r" % rec)
        self.assertEqual(rec["route"], "portfolio_summary")

    def test_a_refusal_reason_is_redacted_but_keeps_dates(self):
        r = RP._redact("No new cases entered Offer between 2026-01-05 and "
                       "2026-01-12, worth £562.9m.")
        self.assertIn("2026-01-05", r)
        self.assertIn("2026-01-12", r)
        self.assertNotIn("562", r)

    def test_a_transport_failure_records_the_class_not_the_message(self):
        boom = RuntimeError("failed for https://x/api?portfolioId=ERE")
        with mock.patch.object(RP.urllib.request, "urlopen", side_effect=boom):
            rec = RP._ask("https://x/api", "tok", "q", None, "ERE/2026-06-30", 5.0)
        self.assertEqual(rec["transport_error"], "RuntimeError")
        self.assertNotIn("ERE", json.dumps(rec))


class TestTheRunFailsOnARegression(unittest.TestCase):
    """A probe whose exit code ignores a regression is a probe nobody has to
    read."""

    def _run(self, prior, now):
        rows = [{"question": "q", "outcome": prior}]
        def _fake(*a, **k):
            return {"outcome": now, "http": 200, "route": "r",
                    "error_code": None, "reason": "", "ms": 1,
                    "transport_error": ""}
        with tempfile.TemporaryDirectory() as d, \
             mock.patch.dict(os.environ, {"MI_BEARER": "t"}), \
             mock.patch.object(RP, "_ask", side_effect=_fake), \
             redirect_stdout(io.StringIO()) as out:
            rc = RP.main(["--from-log", _log(rows),
                          "--out", str(Path(d) / "o.json")])
        return rc, out.getvalue()

    def test_a_regression_exits_non_zero(self):
        rc, printed = self._run("ANSWERED", "ERROR")
        self.assertEqual(rc, 1)
        self.assertIn("REGRESSED", printed)

    def test_a_clean_run_exits_zero(self):
        self.assertEqual(self._run("ANSWERED", "ANSWERED")[0], 0)

    def test_a_fix_alone_does_not_fail_the_run(self):
        self.assertEqual(self._run("ERROR", "ANSWERED")[0], 0)


if __name__ == "__main__":
    unittest.main()
