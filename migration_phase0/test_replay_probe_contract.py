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


#: A response shaped like the one the live API actually returns, taken from a
#: structure dump of two real answers on 2026-09-03. The previous version of
#: these tests was written against the shape the SOURCE suggested, and passed
#: while the probe read a `governance.errorCode` that does not exist.
def _live_refusal():
    return {
        "ok": False,
        "error": "'funded' is not a governed measure in this dataset.",
        "governance": {"status": "error", "capability": "mi_query",
                       "error": {"code": "UNSUPPORTED_QUESTION",
                                 "category": "capability",
                                 "message": "not a governed measure",
                                 "retryable": False}},
        "spec": {"metric": None, "dimension": None, "aggregation": "count",
                 "execution_mode": None, "filters": {}, "measures": []},
        "metadata": {"parserMode": "deterministic", "controlledRefusal": True,
                     "controlledUnsupported": True, "unmappedQuestion": False,
                     "semanticCoverage": {"unaccounted": ["funded"]}},
    }


def _live_answer():
    return {
        "ok": True,
        "governance": {"status": "success", "error": None},
        "spec": {"metric": "current_outstanding_balance",
                 "dimension": "pipeline_stage", "aggregation": "count",
                 "execution_mode": None, "filters": {}},
        "metadata": {"parserMode": "llm", "rowCount": 7,
                     "llm": {"model": "x", "calls": 2, "total_tokens": 900,
                             "estimated_total_cost": 0.0121,
                             "prompt_cache_used": True}},
        "queryTrace": {"requested_dimensions": ["pipeline_stage"],
                       "rejected_dimensions": [], "parserConfidence": 0.9,
                       "normalisedQuery": "which stage has the most cases"},
        "filterInvariant": {"ok": True, "dropped": [],
                            "parsed_filters": {}, "applied_filters": {}},
        "executionSummary": {"measure": "case count", "period": "2026-01-12",
                             "comparisonPeriod": None, "populationTotal": 11035,
                             "groupCount": 7, "populationLabel": "pipeline"},
    }


class TestTheProbeClassifiesLikeTheLog(unittest.TestCase):
    """A probe that split ANSWERED/REFUSED/ERROR differently from the telemetry
    would report a change in every question where the two merely disagree."""

    def _ask(self, body, status=200):
        class _R:
            def __init__(self):
                self.status = status

            def read(self):
                return json.dumps(body).encode()

            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

        def _open(req, timeout=0):
            if status >= 400:
                raise RP.urllib.error.HTTPError(
                    "u", status, "e", {}, io.BytesIO(json.dumps(body).encode()))
            return _R()

        with mock.patch.object(RP.urllib.request, "urlopen", _open):
            return RP._ask("http://h", "t", "q", None, None, 5.0)

    def test_the_error_code_is_read_from_governance_error_code(self):
        """NOT `governance.errorCode`, which does not exist -- the field the
        probe read until a live response was looked at."""
        res = self._ask(_live_refusal())
        self.assertEqual(res["error_code"], "UNSUPPORTED_QUESTION")
        self.assertEqual(res["category"], "capability")

    def test_a_capability_decline_is_a_refusal_not_an_error(self):
        self.assertEqual(self._ask(_live_refusal())["outcome"], RP.REFUSED)

    def test_an_infrastructure_failure_is_an_error(self):
        body = _live_refusal()
        body["governance"]["error"] = {"code": "STORAGE_UNAVAILABLE",
                                       "category": "infrastructure"}
        self.assertEqual(self._ask(body)["outcome"], RP.ERROR)

    def test_calculation_failed_is_an_error_despite_being_a_capability_code(self):
        body = _live_refusal()
        body["governance"]["error"] = {"code": "CALCULATION_FAILED",
                                       "category": "capability"}
        self.assertEqual(self._ask(body)["outcome"], RP.ERROR)

    def test_a_governed_status_beats_the_envelope_ok_flag(self):
        """`ok` is true for a partial success; the telemetry counts only
        `status == success` as answered."""
        body = _live_answer()
        body["governance"]["status"] = "partial_success"
        body["governance"]["error"] = {"code": "PARTIAL", "category": "capability"}
        self.assertEqual(self._ask(body)["outcome"], RP.REFUSED)

    def test_a_success_is_answered(self):
        self.assertEqual(self._ask(_live_answer())["outcome"], RP.ANSWERED)

    def test_the_rule_still_matches_the_telemetry_that_owns_it(self):
        """The probe carries a copy because it runs as a standalone file. This
        is the seam that keeps the copy honest."""
        try:
            sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
            from operations_control import mi_query_telemetry as T
        except Exception as exc:  # noqa: BLE001
            self.skipTest("telemetry not importable: %s" % type(exc).__name__)
        self.assertEqual(set(RP._ERROR_CODES), {str(c) for c in T._ERROR_CODES})
        self.assertEqual(set(RP._ERROR_CATEGORIES),
                         {str(c) for c in T._ERROR_CATEGORIES})


class TestItRecordsWhatTheModelUnderstood(unittest.TestCase):
    """`route` is null for every question the general path answers and
    `execution_mode` is null even on a correct answer. These blocks are what
    can actually locate a defect."""

    def test_it_captures_which_scopes_were_parsed_and_then_dropped(self):
        d = RP._digest({"filterInvariant": {
            "ok": False, "parsed_filters": {"region": "Scotland"},
            "applied_filters": {}, "dropped": ["region"]}})
        self.assertEqual(d["filterInvariant"]["dropped"], ["region"])
        self.assertIn("Scotland", str(d["filterInvariant"]["parsed_filters"]))

    def test_it_captures_a_comparison_period_the_reader_did_not_ask_for(self):
        d = RP._digest({"executionSummary": {
            "period": "2026-06-30", "comparisonPeriod": "2026-05-31"}})
        self.assertEqual(d["execution"]["comparisonPeriod"], "2026-05-31")

    def test_it_captures_which_concepts_went_unaccounted(self):
        d = RP._digest(_live_refusal())
        self.assertEqual(d["meta"]["semanticCoverage"]["unaccounted"],
                         ["funded"])

    def test_no_figure_reaches_the_output_on_any_key(self):
        """The standing rule, enforced rather than intended: a number is
        dropped unless its key is named, so an allow-list entry cannot leak
        one."""
        d = RP._digest(_live_answer())
        self.assertNotIn("populationTotal", d["execution"])
        self.assertNotIn("groupCount", d["execution"])
        self.assertNotIn("estimated_total_cost", d["llm"])
        blob = json.dumps(d)
        for figure in ("11035", "0.0121"):
            self.assertNotIn(figure, blob)

    def test_a_row_count_becomes_whether_there_were_rows(self):
        self.assertTrue(RP._digest(_live_answer())["hasRows"])
        body = _live_answer()
        body["metadata"]["rowCount"] = 0
        self.assertFalse(RP._digest(body)["hasRows"])

    def test_our_own_token_usage_survives_because_it_is_not_client_data(self):
        d = RP._digest(_live_answer())
        self.assertEqual(d["llm"]["calls"], 2)
        self.assertEqual(d["llm"]["total_tokens"], 900)

    def test_a_figure_in_a_filter_value_is_still_redacted(self):
        d = RP._digest({"spec": {"filters": {"balance": "£562,900,000"}}})
        self.assertNotIn("562", json.dumps(d))

    def test_no_answer_text_or_artefact_can_ride_out(self):
        d = RP._digest({"answer": "The book stands at [x]", "artifacts": [{}],
                        "spec": {"title": "t", "explanation": "e"},
                        "reconciliation": {"total_balance": 1}})
        for banned in ("answer", "artifacts", "reconciliation"):
            self.assertNotIn(banned, d)
        self.assertNotIn("title", d.get("spec", {}))

    def test_an_unrouted_question_is_grouped_by_parser_mode(self):
        def _fake(*a, **k):
            return {"outcome": RP.ANSWERED, "http": 200, "route": None,
                    "error_code": None, "category": None, "reason": "",
                    "ms": 1, "transport_error": "", "gateway_cut": False,
                    "metric_defaulted": False,
                    "spec": {"meta": {"parserMode": "llm"}}}

        with tempfile.TemporaryDirectory() as d, \
             mock.patch.dict(os.environ, {"MI_BEARER": "t"}), \
             mock.patch.object(RP, "_ask", side_effect=_fake), \
             redirect_stdout(io.StringIO()):
            RP.main(["--from-log", _log([{"question": "q",
                                          "outcome": "ANSWERED"}]),
                     "--out", str(Path(d) / "o.json")])
            written = json.loads((Path(d) / "o.json").read_text())
        self.assertEqual(list(written["by_route"]), ["pit:llm"])

    def test_failures_are_counted_by_error_code(self):
        def _fake(*a, **k):
            return {"outcome": RP.REFUSED, "http": 200, "route": None,
                    "error_code": "UNSUPPORTED_QUESTION", "category": None,
                    "reason": "", "ms": 1, "transport_error": "",
                    "gateway_cut": False, "metric_defaulted": False, "spec": {}}

        with tempfile.TemporaryDirectory() as d, \
             mock.patch.dict(os.environ, {"MI_BEARER": "t"}), \
             mock.patch.object(RP, "_ask", side_effect=_fake), \
             redirect_stdout(io.StringIO()):
            RP.main(["--from-log", _log([{"question": "q",
                                          "outcome": "REFUSED"}]),
                     "--out", str(Path(d) / "o.json")])
            written = json.loads((Path(d) / "o.json").read_text())
        self.assertEqual(written["by_error_code"], {"UNSUPPORTED_QUESTION": 1})


class TestAnExpiredTokenIsNotSixtyFourRegressions(unittest.TestCase):
    """WHAT HAPPENED ON 2026-09-03, in production, on a run that mattered.

    A token expired 29 questions into a 115-question replay. The remaining 86
    came back HTTP 401 in under 400ms each, were recorded as ERROR, and were
    scored against priors that said ANSWERED. The run printed "REGRESSIONS: 64
    question(s) answered before and do not now" and every one of them was an
    authentication failure.

    `load_probe` has had this guard since a cold-burst run returned 54/54 401s
    and printed a capacity verdict off it — which, believed, buys a bigger App
    Service plan to fix an expired token. The lesson did not travel to this
    probe until it cost a second false alarm.
    """

    def _ask(self, status):
        def _open(req, timeout=0):
            raise RP.urllib.error.HTTPError("u", status, "e", {},
                                            io.BytesIO(b""))
        with mock.patch.object(RP.urllib.request, "urlopen", _open):
            return RP._ask("http://h", "t", "q", None, None, 5.0)

    def test_a_401_is_not_a_model_outcome(self):
        res = self._ask(401)
        self.assertEqual(res["outcome"], RP.NOT_MEASURED)
        self.assertTrue(res["auth_failed"])

    def test_a_403_reads_the_same_way(self):
        self.assertEqual(self._ask(403)["outcome"], RP.NOT_MEASURED)

    def test_an_ordinary_server_error_is_still_not_an_auth_failure(self):
        res = self._ask(500)
        self.assertFalse(res["auth_failed"])

    def test_it_is_never_scored_as_a_regression(self):
        self.assertEqual(RP._verdict("ANSWERED", RP.NOT_MEASURED),
                         RP.UNMEASURED)

    def test_the_run_stops_rather_than_filling_the_corpus_with_401s(self):
        """A partial run is not a baseline. Continuing produces a file that
        looks like a measurement and is not one."""
        seen = []

        def _fake(*a, **k):
            seen.append(1)
            ok = len(seen) <= 2
            return {"outcome": RP.ANSWERED if ok else RP.NOT_MEASURED,
                    "http": 200 if ok else 401, "route": None,
                    "error_code": None, "category": None, "reason": "",
                    "ms": 1, "transport_error": "", "gateway_cut": False,
                    "auth_failed": not ok, "metric_defaulted": False,
                    "spec": {}}

        rows = [{"question": "q%d" % n, "outcome": "ANSWERED"}
                for n in range(10)]
        with tempfile.TemporaryDirectory() as d, \
             mock.patch.dict(os.environ, {"MI_BEARER": "t"}), \
             mock.patch.object(RP, "_ask", side_effect=_fake), \
             redirect_stdout(io.StringIO()):
            rc = RP.main(["--from-log", _log(rows),
                          "--out", str(Path(d) / "o.json")])
        self.assertEqual(rc, 3)
        self.assertEqual(len(seen), 3, "it kept asking after the token died")

    def test_a_dead_token_is_not_retried(self):
        seen = []

        def _fake(*a, **k):
            seen.append(1)
            return {"outcome": RP.NOT_MEASURED, "http": 401, "route": None,
                    "error_code": None, "category": None, "reason": "",
                    "ms": 1, "transport_error": "", "gateway_cut": False,
                    "auth_failed": True, "metric_defaulted": False, "spec": {}}

        with tempfile.TemporaryDirectory() as d, \
             mock.patch.dict(os.environ, {"MI_BEARER": "t"}), \
             mock.patch.object(RP, "_ask", side_effect=_fake), \
             redirect_stdout(io.StringIO()):
            RP.main(["--from-log", _log([{"question": "q",
                                          "outcome": "ANSWERED"}]),
                     "--retries", "3", "--out", str(Path(d) / "o.json")])
        self.assertEqual(len(seen), 1)


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
