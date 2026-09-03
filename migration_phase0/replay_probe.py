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

#: Verdicts, and the only one that should stop a release.
FIXED = "FIXED"                 # did not answer before; answers now
REGRESSED = "REGRESSED"         # answered before; does not now
UNCHANGED_OK = "UNCHANGED_OK"   # answered before and after
STILL_FAILING = "STILL_FAILING"  # did not answer before or after
WAS_MIXED = "WAS_MIXED"         # prior runs disagreed; no single "before"

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

    corpus = []
    for q in order:
        counts = seen[q]
        # A single prior outcome is the baseline. Several means the question
        # was not deterministic BEFORE this run, so it has no "before" to have
        # regressed from — recorded, never averaged.
        prior = next(iter(counts)) if len(counts) == 1 else MIXED
        corpus.append({"question": q, "prior": prior, "prior_counts": counts})
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
    ok = bool(payload.get("ok"))
    code = payload.get("errorCode") or (payload.get("governance") or {}).get("errorCode")
    if transport or status == 0:
        outcome = ERROR
    elif ok:
        outcome = ANSWERED
    else:
        # The same split the telemetry makes: a capability decline is a
        # refusal, anything else is a failure.
        outcome = REFUSED if str(code or "").upper() in (
            "UNSUPPORTED_QUESTION", "AMBIGUOUS_QUESTION",
            "NO_MATCHING_RECORDS") else ERROR
    return {
        "outcome": outcome,
        "http": status,
        # Captured from the LIVE response, so route attribution does not wait
        # on the telemetry fix being deployed.
        "route": meta.get("route") or None,
        "error_code": code or None,
        "reason": _redact(payload.get("error")),
        "ms": elapsed,
        "transport_error": transport,
    }


def _verdict(prior: str, now: str) -> str:
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
    ap.add_argument("--portfolio", default="ERE/2026-06-30")
    ap.add_argument("--lens", default=None,
                    help="sourcePortfolioLens, e.g. direct. Omit for none.")
    ap.add_argument("--timeout", type=float, default=180.0)
    ap.add_argument("--pause", type=float, default=0.0,
                    help="seconds between questions (default 0)")
    ap.add_argument("--limit", type=int, default=0, help="first N only (0 = all)")
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
    print("target %s  portfolio %s  lens %s"
          % (args.base, args.portfolio, args.lens or "-"))

    results = []
    for i, item in enumerate(corpus, 1):
        res = _ask(args.base, token, item["question"], args.lens,
                   args.portfolio, args.timeout)
        verdict = _verdict(item["prior"], res["outcome"])
        row = {**item, **res, "verdict": verdict}
        results.append(row)
        flag = "  <-- REGRESSED" if verdict == REGRESSED else ""
        print("[%3d/%3d] %-13s %-13s %-22s %5dms  %s%s"
              % (i, len(corpus), item["prior"], res["outcome"],
                 (res["route"] or "-")[:22], res["ms"],
                 item["question"][:52], flag))
        if args.pause:
            time.sleep(args.pause)

    counts: Dict[str, int] = {}
    for r in results:
        counts[r["verdict"]] = counts.get(r["verdict"], 0) + 1
    by_route: Dict[str, Dict[str, int]] = {}
    for r in results:
        key = r["route"] or "(no route)"
        by_route.setdefault(key, {})
        by_route[key][r["outcome"]] = by_route[key].get(r["outcome"], 0) + 1

    print("\n=== verdicts ===")
    for k in (REGRESSED, FIXED, UNCHANGED_OK, STILL_FAILING, WAS_MIXED):
        if counts.get(k):
            print("  %-14s %d" % (k, counts[k]))
    print("\n=== outcome by route ===")
    for route, oc in sorted(by_route.items()):
        print("  %-24s %s" % (route, oc))

    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump({"generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ",
                                                 time.gmtime()),
                   "base": args.base, "portfolio": args.portfolio,
                   "lens": args.lens, "counts": counts,
                   "by_route": by_route, "results": results}, fh, indent=2)
    print("\nwrote %s" % args.out)

    if counts.get(REGRESSED):
        print("REGRESSIONS: %d question(s) answered before and do not now."
              % counts[REGRESSED])
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
