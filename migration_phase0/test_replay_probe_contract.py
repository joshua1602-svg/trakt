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


class TestAQuestionIsReplayedAgainstItsOwnPortfolio(unittest.TestCase):
    """The prior outcome was recorded against a named portfolio. Replaying the
    question against a different one compares two different questions, and the
    disagreement would be read as the model having changed."""

    def test_the_logged_portfolio_travels_with_the_question(self):
        corpus = RP.load_corpus(_log([
            {"question": "a", "outcome": "ANSWERED", "portfolio_id": "P/1"},
            {"question": "b", "outcome": "ERROR", "portfolio_id": "P/2"}]))
        self.assertEqual([c["portfolio"] for c in corpus], ["P/1", "P/2"])

    def test_a_log_without_a_portfolio_falls_back_to_the_flag(self):
        corpus = RP.load_corpus(_log([{"question": "a", "outcome": "ANSWERED"}]))
        self.assertIsNone(corpus[0]["portfolio"])

    def test_the_flag_does_not_override_the_log(self):
        sent = []

        def _fake(base, token, q, lens, portfolio, timeout):
            sent.append(portfolio)
            return {"outcome": RP.ANSWERED, "http": 200, "route": "r",
                    "error_code": None, "reason": "", "ms": 1,
                    "transport_error": "", "gateway_cut": False}

        rows = [{"question": "a", "outcome": "ANSWERED", "portfolio_id": "P/1"},
                {"question": "b", "outcome": "ANSWERED"}]
        with tempfile.TemporaryDirectory() as d, \
             mock.patch.dict(os.environ, {"MI_BEARER": "t"}), \
             mock.patch.object(RP, "_ask", side_effect=_fake), \
             redirect_stdout(io.StringIO()):
            RP.main(["--from-log", _log(rows), "--portfolio", "FALLBACK",
                     "--out", str(Path(d) / "o.json")])
        self.assertEqual(sent, ["P/1", "FALLBACK"])


class TestItRecordsWhatTheModelUnderstood(unittest.TestCase):
    """`route` is null for every question the general path answers, which is
    most of them. The spec is what distinguishes one defect from another."""

    def test_the_spec_names_measure_dimension_and_period(self):
        d = RP._spec_digest({"spec": {
            "metric": "balance", "dimension": "region",
            "temporal_mode": "compare", "as_of_date": "2026-06-30",
            "execution_mode": "temporal"}})
        self.assertEqual(d["metric"], "balance")
        self.assertEqual(d["dimension"], "region")
        self.assertEqual(d["temporal_mode"], "compare")
        self.assertEqual(d["as_of_date"], "2026-06-30")

    def test_filter_values_survive_because_they_are_the_finding(self):
        """Scotland applied and Scotland dropped are different defects, and
        the value came from the user's question, not the tape."""
        d = RP._spec_digest({"spec": {"filters": {"region": "Scotland"}}})
        self.assertIn("Scotland", d["filters"]["region"])

    def test_a_figure_inside_a_filter_is_still_redacted(self):
        d = RP._spec_digest({"spec": {"filters": {"balance": "£562,900,000"}}})
        self.assertNotIn("562", d["filters"]["balance"])

    def test_no_answer_text_or_artefact_can_ride_out_on_the_spec(self):
        d = RP._spec_digest({"spec": {
            "metric": "balance", "title": "Funded balance is 562.9m",
            "explanation": "The book stands at ...", "rows": [[1, 2]],
            "snapshot_store_root": "/governed/tape"}})
        for banned in ("title", "explanation", "rows", "snapshot_store_root"):
            self.assertNotIn(banned, d)

    def test_a_defaulted_measure_is_flagged(self):
        """'Show me the trend' answered as a funded-balance trend is a
        different defect from a trend the reader asked for."""
        def _open(req, timeout=0):
            class _R:
                status = 200

                def read(self):
                    return json.dumps({"ok": True, "spec": {
                        "metric": "funded_balance",
                        "metric_defaulted": True}}).encode()

                def __enter__(self):
                    return self

                def __exit__(self, *a):
                    return False
            return _R()

        with mock.patch.object(RP.urllib.request, "urlopen", _open):
            res = RP._ask("http://h", "t", "show me the trend", None, None, 5.0)
        self.assertTrue(res["metric_defaulted"])
        self.assertTrue(res["spec"]["metric_defaulted"])

    def test_an_unrouted_question_is_grouped_by_how_it_executed(self):
        """'(no route)' for four questions in five says nothing about where the
        model is weak."""
        def _fake(*a, **k):
            return {"outcome": RP.ANSWERED, "http": 200, "route": None,
                    "error_code": None, "reason": "", "ms": 1,
                    "transport_error": "", "gateway_cut": False,
                    "metric_defaulted": False,
                    "spec": {"execution_mode": "temporal"}}

        with tempfile.TemporaryDirectory() as d, \
             mock.patch.dict(os.environ, {"MI_BEARER": "t"}), \
             mock.patch.object(RP, "_ask", side_effect=_fake), \
             redirect_stdout(io.StringIO()):
            RP.main(["--from-log", _log([{"question": "q",
                                          "outcome": "ANSWERED"}]),
                     "--out", str(Path(d) / "o.json")])
            written = json.loads((Path(d) / "o.json").read_text())
        self.assertEqual(list(written["by_route"]), ["pit:temporal"])

    def test_failures_are_counted_by_error_code(self):
        def _fake(*a, **k):
            return {"outcome": RP.REFUSED, "http": 200, "route": None,
                    "error_code": "UNSUPPORTED_QUESTION", "reason": "",
                    "ms": 1, "transport_error": "", "gateway_cut": False,
                    "metric_defaulted": False, "spec": {}}

        with tempfile.TemporaryDirectory() as d, \
             mock.patch.dict(os.environ, {"MI_BEARER": "t"}), \
             mock.patch.object(RP, "_ask", side_effect=_fake), \
             redirect_stdout(io.StringIO()):
            RP.main(["--from-log", _log([{"question": "q",
                                          "outcome": "REFUSED"}]),
                     "--out", str(Path(d) / "o.json")])
            written = json.loads((Path(d) / "o.json").read_text())
        self.assertEqual(written["by_error_code"], {"UNSUPPORTED_QUESTION": 1})


class TestAGatewayCutIsNotAModelFailure(unittest.TestCase):
    """The failure this class exists to stop.

    The prior outcomes in the telemetry log were written by the APP, after it
    had decided. If the client scores a gateway cut-off as ERROR, a question
    the model still answers is reported REGRESSED, and the recalibration that
    follows chases a defect that is not there.
    """

    def _ask(self, status, body, ms=0.0, boom=None):
        calls = {"n": 0}

        class _Resp:
            status_code = status

            def __init__(self):
                self.status = status

            def read(self):
                return json.dumps(body).encode() if body is not None else b"x"

            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

        def _open(req, timeout=0):
            calls["n"] += 1
            if boom:
                raise boom
            if status >= 400:
                raise RP.urllib.error.HTTPError(
                    "u", status, "e", {}, io.BytesIO(_Resp().read()))
            return _Resp()

        times = iter([0.0, ms])
        with mock.patch.object(RP.urllib.request, "urlopen", _open), \
             mock.patch.object(RP.time, "time", lambda: next(times, ms)):
            return RP._ask("http://h", "t", "q", None, None, 5.0), calls

    def test_a_500_with_no_envelope_is_not_measured(self):
        res, _ = self._ask(500, None, ms=46.0)
        self.assertEqual(res["outcome"], RP.NOT_MEASURED)
        self.assertTrue(res["gateway_cut"])

    def test_a_500_carrying_an_envelope_is_a_real_error(self):
        """The app's own failures still count. Only what never reached it does
        not."""
        res, _ = self._ask(500, {"ok": False, "errorCode": "INTERNAL"})
        self.assertEqual(res["outcome"], RP.ERROR)
        self.assertFalse(res["gateway_cut"])

    def test_a_dead_connection_is_not_measured(self):
        res, _ = self._ask(200, {}, boom=TimeoutError("read timeout"))
        self.assertEqual(res["outcome"], RP.NOT_MEASURED)

    def test_an_unmeasured_question_is_never_a_regression(self):
        self.assertEqual(RP._verdict("ANSWERED", RP.NOT_MEASURED), RP.UNMEASURED)

    def test_an_unmeasured_question_is_never_a_fix_either(self):
        self.assertEqual(RP._verdict("ERROR", RP.NOT_MEASURED), RP.UNMEASURED)

    def test_a_cut_is_retried_before_it_is_believed(self):
        """One retry, because a warm second attempt usually answers -- and a
        run that gives up on the first cut measures the box, not the model."""
        seen = []

        def _fake(*a, **k):
            seen.append(1)
            out = RP.NOT_MEASURED if len(seen) == 1 else RP.ANSWERED
            return {"outcome": out, "http": 200, "route": "r",
                    "error_code": None, "reason": "", "ms": 1,
                    "transport_error": "", "gateway_cut": False}

        with tempfile.TemporaryDirectory() as d, \
             mock.patch.dict(os.environ, {"MI_BEARER": "t"}), \
             mock.patch.object(RP, "_ask", side_effect=_fake), \
             redirect_stdout(io.StringIO()):
            rc = RP.main(["--from-log", _log([{"question": "q",
                                               "outcome": "ANSWERED"}]),
                          "--out", str(Path(d) / "o.json")])
            written = json.loads((Path(d) / "o.json").read_text())
        self.assertEqual(len(seen), 2)
        self.assertEqual(rc, 0)
        self.assertEqual(written["results"][0]["verdict"], RP.UNCHANGED_OK)

    def test_a_refusal_is_not_retried(self):
        """Re-asking a refusal would average away the non-determinism this run
        exists to find."""
        seen = []

        def _fake(*a, **k):
            seen.append(1)
            return {"outcome": RP.REFUSED, "http": 200, "route": "r",
                    "error_code": "UNSUPPORTED_QUESTION", "reason": "",
                    "ms": 1, "transport_error": "", "gateway_cut": False}

        with tempfile.TemporaryDirectory() as d, \
             mock.patch.dict(os.environ, {"MI_BEARER": "t"}), \
             mock.patch.object(RP, "_ask", side_effect=_fake), \
             redirect_stdout(io.StringIO()):
            RP.main(["--from-log", _log([{"question": "q",
                                          "outcome": "REFUSED"}]),
                     "--out", str(Path(d) / "o.json")])
        self.assertEqual(len(seen), 1)

    def test_an_unmeasured_question_does_not_pollute_no_route(self):
        """'(no route)' means the MODEL could not attribute the question. A
        question the model never saw must not appear there."""
        def _fake(*a, **k):
            return {"outcome": RP.NOT_MEASURED, "http": 500, "route": None,
                    "error_code": None, "reason": "", "ms": 46000,
                    "transport_error": "", "gateway_cut": True}

        with tempfile.TemporaryDirectory() as d, \
             mock.patch.dict(os.environ, {"MI_BEARER": "t"}), \
             mock.patch.object(RP, "_ask", side_effect=_fake), \
             redirect_stdout(io.StringIO()) as out:
            rc = RP.main(["--from-log", _log([{"question": "q",
                                               "outcome": "ANSWERED"}]),
                          "--retries", "0",
                          "--out", str(Path(d) / "o.json")])
            written = json.loads((Path(d) / "o.json").read_text())
        self.assertEqual(rc, 0)
        self.assertEqual(written["by_route"], {})
        self.assertIn("never reached the model", out.getvalue())


if __name__ == "__main__":
    unittest.main()
