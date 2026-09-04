#!/usr/bin/env python3
"""replay_probe — re-ask what real users already asked, and say what moved.

WHY THIS SHAPE. The temptation after a day of fixes is to re-run only the
questions that failed. That measures the fixes and nothing else: it cannot see
a question that used to answer and now does not, which is the failure that
actually costs trust. So the corpus is EVERY question in the log, and the
verdict for each is a comparison against what that question did before.

THE CORPUS IS NOT TYPED IN. It is read from the MI Query telemetry the OCC
already keeps -- pass the `/ops/mi-queries` response straight to `--from-log`.
A bank someone transcribes is a bank someone paraphrases, and the whole lesson
of 2026-09-03 is that a corpus whose phrasings drifted from what users type
reports 36/36 while the product refuses three questions in a row.

DUPLICATES ARE EVIDENCE, NOT NOISE. "Summarise the current pipeline." appears
~25 times in the live log, and it did NOT always do the same thing: it
answered at 14:52:31 and errored 47 seconds later. A question whose prior runs
disagree is recorded as MIXED and never counted as a regression, because there
is no single "before" for it to have regressed from. Those are the questions
worth fixing first, and averaging them away would hide them.

A CUT IS NOT A VERDICT. The prior outcome for each question was written by
the APP, after it had decided. The gateway in front of it gives up at ~46s and
returns a body that is not an MI envelope -- so a client that scores that as an
error reports a REGRESSION in a question the model still answers, and the
recalibration that follows chases a defect that is not there. Anything without
an envelope is NOT_MEASURED: retried once, then scored neither way and counted
separately as the capacity finding it is.

DIAGNOSTICS ONLY, the standing rule. Outcome, route, error code, duration and
a REDACTED reason. Never the answer text: money and long numbers are stripped
with the same reader `live_bank_probe` uses, ISO dates kept, because "between
2026-01-05 and 2026-01-12" is a finding and "£562.9m" is client data.

USAGE
    # save the /ops/mi-queries response as queries.json first
    export MI_BEARER='<paste your dashboard token>'
    python3 replay_probe.py --from-log queries.json --out replay.json
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
from typing import Any, Dict, List, Optional, Tuple

ANSWERED, REFUSED, ERROR, MIXED, UNKNOWN = (
    "ANSWERED", "REFUSED", "ERROR", "MIXED", "UNKNOWN")

#: The request never reached a verdict from the MI service -- the gateway cut
#: it, or the connection died. NOT a model outcome, and deliberately not ERROR:
#: the prior outcomes in the telemetry log were written by the app AFTER it had
#: decided, so a client-side cut-off scored as ERROR would report a regression
#: in a question the model still answers. See `_ask`.
NOT_MEASURED = "NOT_MEASURED"

#: Verdicts, and the only one that should stop a release.
FIXED = "FIXED"                 # did not answer before; answers now
REGRESSED = "REGRESSED"         # answered before; does not now
UNCHANGED_OK = "UNCHANGED_OK"   # answered before and after
STILL_FAILING = "STILL_FAILING"  # did not answer before or after
WAS_MIXED = "WAS_MIXED"         # prior runs disagreed; no single "before"
UNMEASURED = "UNMEASURED"       # the request never reached the model

#: A 401/403 is the TOKEN failing, never the model. It arrives in
#: milliseconds, touches no analytic, and — before this — was recorded as an
#: ERROR and scored against a prior that said ANSWERED. On 2026-09-03 a token
#: expired 29 questions into a 115-question replay and the run reported
#: "REGRESSIONS: 64 question(s) answered before and do not now": eighty-six
#: authentication failures presented as a model that had fallen apart.
#:
#: `load_probe` has carried this guard since a cold-burst run returned 54/54
#: 401s and printed a capacity verdict off it. The lesson did not travel here.
_AUTH_STATUSES = (401, 403)

#: The Azure auth sidecar gives up at ~46s and returns "Backend call failure"
#: -- a body that is not an MI envelope. Recorded so a slow answer is visible
#: as a capacity finding rather than silently absorbed into the model's score.
GATEWAY_CUT_MS = 44_000

#: Copied from `live_bank_probe`, deliberately: ISO dates first so they survive,
#: then money and long numbers out. One rule, two probes, same behaviour.
_ISO_DATE = re.compile(r"\b\d{4}-\d{2}-\d{2}\b")
_FIGURE = re.compile(
    r"[£$€]\s?[\d,.]+\s*(?:[kKmMbB]{1,2}|MM)?|\b\d[\d,]{3,}(?:\.\d+)?\b|\b\d+\.\d+\b")


def _redact(text: Any) -> str:
    if not text:
        return ""
    keep: Dict[str, str] = {}

    def _stash(m):
        token = "\x00%d\x00" % len(keep)
        keep[token] = m.group(0)
        return token

    out = _ISO_DATE.sub(_stash, str(text))
    out = _FIGURE.sub("[figure]", out)
    for token, original in keep.items():
        out = out.replace(token, original)
    return out.strip()[:300]


def load_corpus(path: str) -> List[Dict[str, Any]]:
    """``[{question, prior, prior_counts}]`` from a telemetry response.

    Accepts the `/ops/mi-queries` envelope or a bare list of rows, because a
    reader should not have to know which one they saved.
    """
    with open(path, "r", encoding="utf-8") as fh:
        raw = json.load(fh)
    rows = raw.get("queries") if isinstance(raw, dict) else raw
    if not isinstance(rows, list):
        raise SystemExit("no queries found in %s" % path)

    seen: Dict[str, Dict[str, int]] = {}
    portfolios: Dict[str, str] = {}
    order: List[str] = []
    for row in rows:
        q = str((row or {}).get("question") or "").strip()
        if not q:
            continue
        if q not in seen:
            seen[q] = {}
            order.append(q)
        outcome = str(row.get("outcome") or UNKNOWN).upper()
        seen[q][outcome] = seen[q].get(outcome, 0) + 1
        # The prior outcome was recorded against THIS portfolio. Replaying the
        # question against a different one compares two different questions,
        # and any disagreement would be read as the model having changed.
        pid = row.get("portfolio_id")
        if pid and q not in portfolios:
            portfolios[q] = str(pid)

    corpus = []
    for q in order:
        counts = seen[q]
        # A single prior outcome is the baseline. Several means the question
        # was not deterministic BEFORE this run, so it has no "before" to have
        # regressed from — recorded, never averaged.
        prior = next(iter(counts)) if len(counts) == 1 else MIXED
        corpus.append({"question": q, "prior": prior, "prior_counts": counts,
                       "portfolio": portfolios.get(q)})
    return corpus


def _ask(base: str, token: str, question: str, lens: Optional[str],
         portfolio: Optional[str], timeout: float) -> Dict[str, Any]:
    """One question. Returns diagnostics; never the answer."""
    body: Dict[str, Any] = {"question": question}
    if lens:
        body["sourcePortfolioLens"] = lens
    if portfolio:
        body["portfolioId"] = portfolio
    req = urllib.request.Request(
        base.rstrip("/") + "/mi/query",
        data=json.dumps(body).encode("utf-8"), method="POST")
    req.add_header("Authorization", "Bearer " + token)
    req.add_header("Content-Type", "application/json")

    t0 = time.time()
    status: Any = 0
    payload: Dict[str, Any] = {}
    transport = ""
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            status = resp.status
            payload = json.loads(resp.read().decode("utf-8") or "{}")
    except urllib.error.HTTPError as exc:
        status = exc.code
        try:
            payload = json.loads(exc.read().decode("utf-8") or "{}")
        except Exception:  # noqa: BLE001
            payload = {}
    except Exception as exc:  # noqa: BLE001 - the class only; a URL names a portfolio
        transport = type(exc).__name__
    elapsed = int((time.time() - t0) * 1000)

    meta = payload.get("metadata") or {}
    gov = payload.get("governance") if isinstance(
        payload.get("governance"), dict) else {}
    gov_error = gov.get("error") if isinstance(gov.get("error"), dict) else {}
    # `governance.error.code` -- NOT `governance.errorCode`, which does not
    # exist and which this probe read until a live response was looked at.
    code = payload.get("errorCode") or gov_error.get("code")
    category = gov_error.get("category")
    gov_status = gov.get("status")
    # The MI service ALWAYS answers with an envelope carrying `ok`. Its absence
    # means the response was written by something in front of the app -- the
    # auth sidecar, a proxy, or nothing at all -- so the model never gave a
    # verdict and this run has no measurement of it.
    envelope = isinstance(payload, dict) and "ok" in payload
    if status in _AUTH_STATUSES:
        # The token, not the model. Never a verdict about a question.
        outcome = NOT_MEASURED
    elif transport or status == 0:
        outcome = NOT_MEASURED
    elif status >= 500 and not envelope:
        outcome = NOT_MEASURED
    elif gov_status:
        # THE TELEMETRY'S OWN RULE, on the telemetry's own fields, so a verdict
        # compares like with like. Reading `ok` instead would disagree with the
        # log wherever the two diverge -- a partial success, most of all.
        if gov_status == _STATUS_SUCCESS:
            outcome = ANSWERED
        elif not code:
            outcome = ERROR if gov_status == _STATUS_ERROR else REFUSED
        elif code in _ERROR_CODES or category in _ERROR_CATEGORIES:
            outcome = ERROR
        else:
            outcome = REFUSED
    else:
        # No governance block at all -- an older or non-governed response.
        outcome = ANSWERED if payload.get("ok") else ERROR
    return {
        "outcome": outcome,
        "http": status,
        # Captured from the LIVE response, so route attribution does not wait
        # on the telemetry fix being deployed.
        "route": meta.get("route") or None,
        "error_code": code or None,
        "reason": _redact(payload.get("error") or gov_error.get("message")),
        "ms": elapsed,
        "transport_error": transport,
        "spec": _digest(payload),
        "category": category or None,
        # Did the model choose the measure, or did it choose one FOR the reader?
        # "Show me the trend" answered as a funded-balance trend is a different
        # defect from a trend the reader asked for.
        "metric_defaulted": bool((payload.get("spec") or {}).get(
            "metric_defaulted")) if isinstance(payload.get("spec"), dict) else False,
        # Corroborating, never the discriminator: the envelope is. A cut at 46s
        # and a cut at 3s are both unmeasured; only one is a capacity finding.
        "gateway_cut": elapsed >= GATEWAY_CUT_MS and outcome == NOT_MEASURED,
        "auth_failed": status in _AUTH_STATUSES,
    }


#: THE SAME RULE THE TELEMETRY USES, copied from
#: `operations_control.mi_query_telemetry.outcome_for`. A probe that
#: classified outcomes differently from the log it compares against would
#: report a change in every question where the two merely disagree. A test
#: imports the owner's own constants and asserts these still match.
_STATUS_SUCCESS = "success"
_STATUS_ERROR = "error"
_ERROR_CODES = frozenset({"CALCULATION_FAILED"})
_ERROR_CATEGORIES = frozenset({"infrastructure"})

#: The ONLY keys whose numeric value may leave this probe. Everything else
#: numeric is dropped. This is the diagnostics-only rule made enforceable
#: rather than aspirational: a field added to an allow-list below cannot leak
#: a balance, a count or a rate, because `_safe` never lets a number through
#: on a key that is not named here. None of these are portfolio figures --
#: they are parser confidence, our own token usage and repair attempts.
_NUMERIC_OK = frozenset({
    "parserConfidence", "parser_confidence", "calls", "total_tokens",
    "input_tokens", "output_tokens", "cache_read_tokens",
    "cache_write_tokens", "repairAttempts",
})


def _safe(value: Any, key: str = "") -> Any:
    """Whatever survives the diagnostics-only rule, and nothing else.

    Strings are redacted, numbers are dropped unless their key is named in
    `_NUMERIC_OK`, and containers are walked. Enforced here, once, so an
    allow-list entry below cannot become a leak.
    """
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value if key in _NUMERIC_OK else None
    if isinstance(value, str):
        return _redact(value)[:160] or None
    if isinstance(value, (list, tuple, set)):
        out = [_safe(v, key) for v in list(value)[:12]]
        out = [v for v in out if v not in (None, "", [], {})]
        return out or None
    if isinstance(value, dict):
        out = {k: _safe(v, k) for k, v in list(value.items())[:30]}
        out = {k: v for k, v in out.items() if v not in (None, "", [], {})}
        return out or None
    return None


def _pick(block: Any, keys: Tuple[str, ...]) -> Dict[str, Any]:
    """The named keys of a block, each through `_safe`."""
    if not isinstance(block, dict):
        return {}
    out: Dict[str, Any] = {}
    for k in keys:
        v = _safe(block.get(k), k)
        if v not in (None, "", [], {}):
            out[k] = v
    return out


#: WHAT THE MODEL UNDERSTOOD -- read from the response the API actually
#: returns, not from the one the source suggested it returns. `route` is null
#: for every question the general path answers, and `spec.execution_mode` is
#: null even when the answer is correct, so neither can locate a defect. These
#: blocks can: `filterInvariant` says which scopes were parsed and then
#: dropped, `queryTrace` says which dimensions and measures were requested and
#: which were rejected, and `executionSummary` says which period was used and
#: whether a comparison was introduced.
_SPEC_KEYS = ("intent", "chart_type", "output_format", "metric",
              "metric_defaulted", "aggregation", "dimension", "dimensions",
              "temporal_mode", "execution_mode", "state", "as_of_date",
              "start_date", "end_date", "baseline_date", "current_date",
              "reporting_date", "segment", "ranking_mode", "sort_by",
              "compare_periods", "trend_grain", "bucket_field",
              "bucket_strategy", "filters", "measures", "unavailable_filters")
_META_KEYS = ("parserMode", "parserModeDetail", "controlledRefusal",
              "controlledUnsupported", "unmappedQuestion", "resultType",
              "runRequired", "asOfDate", "repairAttempts",
              "repairSkippedReason", "semanticCoverage", "parserProvenance",
              # WHICH MODELS TOUCHED THE ANSWER, both arms totalled. Without it
              # the bank evidence showed `llm.calls = 0` for a request the
              # concept-merge arm had changed, and a replay could not tell a
              # deterministic answer from a model-assisted one.
              "modelUsage")
_TRACE_KEYS = ("intent", "metric", "aggregation", "parserMode",
               "parserConfidence", "portfolioLens", "resultType",
               "requested_dimensions", "applied_dimensions",
               "rejected_dimensions", "rejectedDimensions",
               "dimensionsParsed", "requested_filters", "applied_filters",
               "rejected_filters", "rejectedFilters", "filtersParsed",
               "requested_measures", "executed_measures", "normalisedQuery")
_FILTER_INV_KEYS = ("ok", "dropped", "parsed_filters", "applied_filters",
                    "filters_applied", "rejected_filters",
                    "unavailable_filters")
_DIM_INV_KEYS = ("ok", "applied", "dropped", "rejected")
_GUARD_KEYS = ("verdict", "message", "substitution", "facets")
#: NOT populationTotal, groupCount, population or receipt -- those are counts
#: and balances off the client's own tape.
_EXEC_KEYS = ("measure", "aggregation", "period", "comparisonPeriod",
              "populationLabel", "filtersApplied", "dimensionsApplied",
              "notApplied", "narrowed", "ranking", "scenario",
              "parserConfidence")
_LLM_KEYS = ("model", "calls", "total_tokens", "prompt_cache_used",
             "prompt_cache_supported")


def _digest(payload: Dict[str, Any]) -> Dict[str, Any]:
    """What the query model made of the question, and what it dropped."""
    if not isinstance(payload, dict):
        return {}
    meta = payload.get("metadata") if isinstance(
        payload.get("metadata"), dict) else {}
    out: Dict[str, Any] = {}
    for name, block, keys in (
            ("spec", payload.get("spec"), _SPEC_KEYS),
            ("meta", meta, _META_KEYS),
            ("trace", payload.get("queryTrace"), _TRACE_KEYS),
            ("filterInvariant", payload.get("filterInvariant"),
             _FILTER_INV_KEYS),
            ("dimensionInvariant", payload.get("dimensionInvariant"),
             _DIM_INV_KEYS),
            ("guard", payload.get("semanticGuard"), _GUARD_KEYS),
            ("execution", payload.get("executionSummary"), _EXEC_KEYS),
            ("llm", meta.get("llm"), _LLM_KEYS)):
        picked = _pick(block, keys)
        if picked:
            out[name] = picked
    # A count of the client's rows is a figure; whether ANY came back is a
    # finding, and the difference between them is the whole standing rule.
    if isinstance(meta.get("rowCount"), (int, float)):
        out["hasRows"] = bool(meta["rowCount"])
    return out


def _verdict(prior: str, now: str) -> str:
    # Checked BEFORE the prior, because a question we did not measure has no
    # verdict to give whatever it used to do.
    if now == NOT_MEASURED:
        return UNMEASURED
    if prior == MIXED:
        return WAS_MIXED
    was_ok, is_ok = prior == ANSWERED, now == ANSWERED
    if was_ok and is_ok:
        return UNCHANGED_OK
    if was_ok and not is_ok:
        return REGRESSED
    if not was_ok and is_ok:
        return FIXED
    return STILL_FAILING


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--from-log", required=True,
                    help="the /ops/mi-queries response, saved as JSON")
    ap.add_argument("--base", default="https://app.traktinfra.io/api")
    ap.add_argument("--portfolio", default="ERE/2026-06-30",
                    help="fallback only -- a question logged with its own "
                         "portfolio is replayed against that one")
    ap.add_argument("--lens", default=None,
                    help="sourcePortfolioLens, e.g. direct. Omit for none.")
    ap.add_argument("--timeout", type=float, default=180.0)
    ap.add_argument("--pause", type=float, default=0.0,
                    help="seconds between questions (default 0)")
    ap.add_argument("--limit", type=int, default=0, help="first N only (0 = all)")
    ap.add_argument("--retries", type=int, default=1,
                    help="re-ask a question the gateway cut, N times (default 1). "
                         "Only NOT_MEASURED is retried -- a refusal and an error "
                         "are answers, and re-asking them would hide "
                         "non-determinism this run exists to find.")
    ap.add_argument("--out", default="replay.json")
    args = ap.parse_args(argv)

    token = os.environ.get("MI_BEARER", "").strip()
    if token.lower().startswith("bearer "):
        token = token[7:].strip()
    if not token:
        print("MI_BEARER is not set. Copy the Authorization header value from "
              "devtools and:\n    export MI_BEARER='<token>'", file=sys.stderr)
        return 2

    corpus = load_corpus(args.from_log)
    if args.limit:
        corpus = corpus[:args.limit]
    total_rows = sum(sum(c["prior_counts"].values()) for c in corpus)
    print("%d distinct questions (from %d logged runs)" % (len(corpus), total_rows))
    logged = sorted({c["portfolio"] for c in corpus if c.get("portfolio")})
    print("target %s  portfolio %s  lens %s"
          % (args.base,
             ", ".join(logged) if logged else args.portfolio + " (fallback)",
             args.lens or "-"))

    results = []
    for i, item in enumerate(corpus, 1):
        attempts = 0
        while True:
            res = _ask(args.base, token, item["question"], args.lens,
                       item.get("portfolio") or args.portfolio, args.timeout)
            attempts += 1
            if res.get("auth_failed"):
                break          # a second attempt with the same token is the
                               # same answer, more slowly
            if res["outcome"] != NOT_MEASURED or attempts > args.retries:
                break
        res["attempts"] = attempts
        verdict = _verdict(item["prior"], res["outcome"])
        row = {**item, **res, "verdict": verdict}
        results.append(row)
        flag = "  <-- REGRESSED" if verdict == REGRESSED else ""
        if res.get("auth_failed"):
            print("\n*** NOT A MEASUREMENT — the token was rejected (HTTP %s) at "
                  "question %d of %d.\n    Everything from here would be "
                  "recorded as a model failure and scored against a prior that\n"
                  "    says ANSWERED. Get a fresh token and re-run; a partial "
                  "run is not a baseline.\n" % (res["http"], i, len(corpus)),
                  file=sys.stderr)
            return 3
        if res["outcome"] == NOT_MEASURED:
            flag = "  <-- not measured (%s)" % (
                "gateway cut" if res["gateway_cut"]
                else res["transport_error"] or ("http %s" % res["http"]))
        print("[%3d/%3d] %-13s %-13s %-22s %5dms  %s%s"
              % (i, len(corpus), item["prior"], res["outcome"],
                 (res["route"] or "-")[:22], res["ms"],
                 item["question"][:52], flag))
        # The reason, on the failures only. A run that prints why it failed as
        # it goes is diagnosable while it is still running.
        if res["outcome"] in (REFUSED, ERROR) and res["reason"]:
            print("            reason: %s" % res["reason"][:150])
        if args.pause:
            time.sleep(args.pause)

    counts: Dict[str, int] = {}
    for r in results:
        counts[r["verdict"]] = counts.get(r["verdict"], 0) + 1
    by_route: Dict[str, Dict[str, int]] = {}
    for r in results:
        # An unmeasured question would otherwise pile into "(no route)", which
        # is supposed to mean "the model could not attribute this" -- a finding
        # about the model, not about the box it runs on.
        if r["outcome"] == NOT_MEASURED:
            continue
        # A named route when there is one, otherwise how the general path
        # executed it. "(no route)" for four questions out of five says nothing
        # about where the model is weak; "flat" vs "temporal" vs nothing at all
        # does.
        # `execution_mode` is null even on a correct answer, so it groups
        # nothing. `parserMode` is present on every response and is the thing
        # that actually differs between a question the model understood and one
        # it did not.
        key = r["route"] or ("pit:" + str(
            ((r.get("spec") or {}).get("meta") or {}).get("parserMode") or "-"))
        by_route.setdefault(key, {})
        by_route[key][r["outcome"]] = by_route[key].get(r["outcome"], 0) + 1
    by_code: Dict[str, int] = {}
    for r in results:
        if r["outcome"] in (ANSWERED, NOT_MEASURED):
            continue
        by_code[str(r.get("error_code") or "(none)")] = by_code.get(
            str(r.get("error_code") or "(none)"), 0) + 1

    print("\n=== verdicts ===")
    for k in (REGRESSED, FIXED, UNCHANGED_OK, STILL_FAILING, WAS_MIXED,
              UNMEASURED):
        if counts.get(k):
            print("  %-14s %d" % (k, counts[k]))
    if counts.get(UNMEASURED):
        cut = sum(1 for r in results if r.get("gateway_cut"))
        print("\n  %d question(s) never reached the model (%d cut by the "
              "gateway at ~46s). Those are a capacity finding, not a model\n"
              "  finding, and are scored neither way." % (counts[UNMEASURED], cut))
    print("\n=== outcome by route (pit: = no named route, general path) ===")
    for route, oc in sorted(by_route.items()):
        print("  %-24s %s" % (route, oc))
    if by_code:
        print("\n=== why the rest did not answer ===")
        for code, n in sorted(by_code.items(), key=lambda kv: -kv[1]):
            print("  %-34s %d" % (code, n))
    defaulted = [r for r in results if r.get("metric_defaulted")]
    if defaulted:
        print("\n  %d question(s) were answered on a measure the reader did "
              "not name." % len(defaulted))

    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump({"generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ",
                                                 time.gmtime()),
                   "base": args.base, "portfolio": args.portfolio,
                   "lens": args.lens, "counts": counts,
                   "by_route": by_route, "by_error_code": by_code,
                   "results": results}, fh, indent=2)
    print("\nwrote %s" % args.out)

    if counts.get(REGRESSED):
        print("REGRESSIONS: %d question(s) answered before and do not now."
              % counts[REGRESSED])
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
