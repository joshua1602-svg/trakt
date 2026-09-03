#!/usr/bin/env python3
"""load_probe — how many concurrent users the MI dashboard can actually serve.

WHY THIS EXISTS. Every capacity claim made about this deployment so far has
been arithmetic on single-user timings: one dashboard load on 2026-09-03 took
16.5s for `query`, 15.7s for `portfolio-context`, 11.8s for `snapshot`, and the
auth layer in front of the app abandons a request at roughly 46 seconds. Divide
one into the other and you get "fewer than three users" — which is reasoning,
not evidence, and concurrency rarely divides that neatly.

WHAT IT MEASURES. A dashboard page load is not one request. The browser trace
shows NINE calls per user, several of them heavy, fired in a burst. This drives
that same burst from N users at once and reports how many complete, how many
the gateway kills, and where the time goes. The list is copied from the traces
(13:15 and the earlier load), never invented: a test firing one request per
user would measure a ninth of the real load and pronounce the box adequate.

IT GOES THROUGH THE FRONT DOOR. Driving `localhost:8000` from inside the
container would skip the auth layer — and the auth layer's timeout is the thing
that turned today's slow answers into 500s. A capacity test that cannot
reproduce the failure mode is measuring the wrong system, so this targets the
public host every client user reaches.

DIAGNOSTICS ONLY, THE STANDING RULE. It records the endpoint, the status code
and the elapsed time. It never reads, stores or prints a response body, so no
balance, count, category value or answer text can reach the output file. The
bearer token is read from the environment and is never logged, echoed or
written — the output is safe to paste into a chat window, which is the point.

USAGE
    export MI_BEARER='<paste your token>'      # from browser devtools
    python3 load_probe.py --ramp 1,2,4,6 --out load.json

    # a single point rather than a ramp
    python3 load_probe.py --users 6 --out load6.json
"""
from __future__ import annotations

import argparse
import json
import os
import ssl
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Dict, List, Optional, Tuple

#: The gateway's observed abandonment point. Measured 2026-09-03: four requests
#: fired together all failed at 46.74s with "Backend call failure" — the auth
#: container giving up, not the app erroring. A request slower than this is a
#: 500 to the user however healthy the app is, so it is the number that decides
#: capacity, and every summary below is expressed against it.
GATEWAY_TIMEOUT_S = 46.0

#: ONE DASHBOARD PAGE LOAD, as the browser actually issues it (network trace,
#: 2026-09-03 13:15). Reproduced rather than invented: a test that fires one
#: request per user would measure a ninth of the real load and call the box
#: adequate. `q` marks the MI question, the only POST and the expensive one.
SESSION: List[Tuple[str, str, Optional[Dict[str, Any]]]] = [
    ("GET", "/mi/portfolio-context", None),
    ("GET", "/mi/snapshots", None),
    ("GET", "/me", None),
    ("GET", "/mi/snapshot?portfolioId={pid}&portfolioContext=direct", None),
    ("GET", "/mi/decks?portfolioId={pid}", None),
    ("GET", "/mi/insights/weekly-brief?portfolioId={pid}&portfolioContext=direct", None),
    ("GET", "/mi/forecast/snapshot?portfolioId={pid}&portfolioContext=direct", None),
    ("GET", "/mi/concentration-tests?portfolioId={pid}&portfolioContext=direct", None),
    ("POST", "/mi/query", {"question": "Summarise the current pipeline.",
                           "sourcePortfolioLens": "direct"}),
]


def _label(path: str) -> str:
    """The endpoint without its query string — the unit results group by."""
    return path.split("?", 1)[0]


def _call(base: str, token: str, method: str, path: str,
          body: Optional[Dict[str, Any]], pid: str,
          timeout: float) -> Dict[str, Any]:
    """One request. Returns timing and status; never the response content.

    A body is read and discarded so the server measures a full round trip
    rather than a client that hung up early — the length is kept because a
    response size is a shape, not a figure, and a zero-length 200 is worth
    seeing. Nothing from the body itself is retained.
    """
    url = base.rstrip("/") + path.format(pid=urllib.parse.quote(pid, safe=""))
    data = json.dumps(body).encode("utf-8") if body is not None else None
    req = urllib.request.Request(url, data=data, method=method)
    req.add_header("Authorization", "Bearer " + token)
    req.add_header("Accept", "*/*")
    if data is not None:
        req.add_header("Content-Type", "application/json")

    t0 = time.time()
    status: Any = None
    size = 0
    error = ""
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            status = resp.status
            size = len(resp.read())
    except urllib.error.HTTPError as exc:
        status = exc.code
        try:
            size = len(exc.read())
        except Exception:  # noqa: BLE001
            size = 0
    except Exception as exc:  # noqa: BLE001 - a dead call is a data point
        # The CLASS only. An exception string can carry a URL with query
        # parameters, and those name portfolios.
        status = 0
        error = type(exc).__name__
    elapsed = time.time() - t0
    return {"endpoint": _label(path), "method": method, "status": status,
            "ms": int(elapsed * 1000), "bytes": size, "error": error,
            "over_gateway_timeout": elapsed > GATEWAY_TIMEOUT_S}


def _session(base: str, token: str, pid: str, user: int,
             timeout: float, sink: List[Dict[str, Any]],
             lock: threading.Lock) -> None:
    """One simulated user's page load, calls issued in the browser's order."""
    for method, path, body in SESSION:
        rec = _call(base, token, method, path, body, pid, timeout)
        rec["user"] = user
        with lock:
            sink.append(rec)


def _percentile(values: List[int], pct: float) -> int:
    if not values:
        return 0
    ordered = sorted(values)
    idx = min(len(ordered) - 1, int(round((pct / 100.0) * (len(ordered) - 1))))
    return ordered[idx]


def _summarise(records: List[Dict[str, Any]], users: int) -> Dict[str, Any]:
    ok = [r for r in records if r["status"] == 200]
    killed = [r for r in records if r["over_gateway_timeout"]]
    failed = [r for r in records if r["status"] not in (200, 304)]
    by_ep: Dict[str, List[int]] = {}
    for r in records:
        by_ep.setdefault(r["endpoint"], []).append(r["ms"])
    return {
        "users": users,
        "requests": len(records),
        "ok": len(ok),
        "failed": len(failed),
        "over_gateway_timeout": len(killed),
        "success_pct": round(100.0 * len(ok) / len(records), 1) if records else 0.0,
        "wall_ms": max((r["ms"] for r in records), default=0),
        "by_endpoint": {
            ep: {"n": len(v), "p50_ms": _percentile(v, 50),
                 "p90_ms": _percentile(v, 90), "max_ms": max(v)}
            for ep, v in sorted(by_ep.items())},
        "status_counts": _status_counts(records),
    }


def _status_counts(records: List[Dict[str, Any]]) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for r in records:
        key = str(r["status"]) if r["status"] else ("error:" + (r["error"] or "?"))
        out[key] = out.get(key, 0) + 1
    return out


def _run_round(base: str, token: str, pid: str, users: int,
               timeout: float) -> Dict[str, Any]:
    """N users, all starting together — the burst a shift change produces."""
    records: List[Dict[str, Any]] = []
    lock = threading.Lock()
    threads = [threading.Thread(target=_session,
                                args=(base, token, pid, i + 1, timeout,
                                      records, lock), daemon=True)
               for i in range(users)]
    t0 = time.time()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    summary = _summarise(records, users)
    summary["round_wall_s"] = round(time.time() - t0, 1)
    summary["records"] = records
    return summary


def _print_round(s: Dict[str, Any]) -> None:
    verdict = ("OK" if s["failed"] == 0 and s["over_gateway_timeout"] == 0
               else "DEGRADED")
    print(f"\n=== {s['users']} concurrent user(s) — {verdict} ===")
    print(f"  requests {s['requests']}  ok {s['ok']}  failed {s['failed']}"
          f"  past-{int(GATEWAY_TIMEOUT_S)}s {s['over_gateway_timeout']}"
          f"  round {s['round_wall_s']}s")
    print(f"  statuses: {s['status_counts']}")
    for ep, v in s["by_endpoint"].items():
        flag = "  <-- past gateway timeout" if v["max_ms"] > GATEWAY_TIMEOUT_S * 1000 else ""
        print(f"    {ep:<42} p50 {v['p50_ms']:>6}ms  p90 {v['p90_ms']:>6}ms"
              f"  max {v['max_ms']:>6}ms{flag}")


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--base", default="https://app.traktinfra.io/api",
                    help="API base URL (default: the public dashboard path)")
    ap.add_argument("--portfolio", default="ERE/2026-06-30",
                    help="portfolioId the dashboard is showing")
    ap.add_argument("--users", type=int, help="a single concurrency point")
    ap.add_argument("--ramp", default="1,2,4,6",
                    help="comma-separated concurrency steps (default 1,2,4,6)")
    ap.add_argument("--timeout", type=float, default=120.0,
                    help="client-side give-up, seconds (default 120)")
    ap.add_argument("--settle", type=float, default=20.0,
                    help="seconds to idle between ramp steps (default 20)")
    ap.add_argument("--out", default="load.json")
    args = ap.parse_args(argv)

    token = os.environ.get("MI_BEARER", "").strip()
    # The devtools header reads "Bearer eyJ...", and that whole string is what
    # gets copied. `_call` adds the scheme itself, so a token pasted with its
    # prefix would be sent as "Bearer Bearer eyJ..." and 401 every request --
    # losing the run to a formatting slip rather than to capacity, which is the
    # one thing a measurement must not do.
    if token.lower().startswith("bearer "):
        token = token[7:].strip()
    if not token:
        print("MI_BEARER is not set. Copy the Authorization header value from "
              "your browser's devtools (without the leading 'Bearer ') and:\n"
              "    export MI_BEARER='<token>'", file=sys.stderr)
        return 2

    steps = ([args.users] if args.users
             else [int(x) for x in args.ramp.split(",") if x.strip()])
    print(f"target {args.base}  portfolio {args.portfolio}")
    print(f"session = {len(SESSION)} calls per user; "
          f"gateway abandons at ~{int(GATEWAY_TIMEOUT_S)}s")

    rounds = []
    for i, users in enumerate(steps):
        if i:
            # Let the box return to idle, or a round inherits the previous
            # round's queue and every step after the first reads worse than
            # it is.
            time.sleep(args.settle)
        s = _run_round(args.base, token, args.portfolio, users, args.timeout)
        _print_round(s)
        rounds.append(s)

    payload = {"target": args.base, "portfolio": args.portfolio,
               "gateway_timeout_s": GATEWAY_TIMEOUT_S,
               "session_calls": [f"{m} {_label(p)}" for m, p, _ in SESSION],
               "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
               "rounds": rounds}
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
    print(f"\nwrote {args.out}")

    breaking = [r["users"] for r in rounds
                if r["failed"] or r["over_gateway_timeout"]]
    if breaking:
        print(f"FIRST DEGRADED AT: {min(breaking)} concurrent user(s)")
    else:
        print(f"No failures up to {max(r['users'] for r in rounds)} users.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
