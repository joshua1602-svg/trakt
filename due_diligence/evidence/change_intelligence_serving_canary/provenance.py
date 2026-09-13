#!/usr/bin/env python3
"""Prove WHICH COMMIT is serving, and that the bearer authenticates. No spend.

WHAT THIS IS FOR. Two of the operator's gates, and only those two:

    step 3  independently prove the running service corresponds to an exact SHA
    step 7  the minimum authenticated preflight, using the existing MI_BEARER

WHY IT COSTS NOTHING. Both are performed with GET requests against endpoints
that resolve no question: `/health` reports the deployed build, and the
catalogue endpoint reports the governed field registry. Neither reaches
`interpretation_v2`, so neither buys an Opus interpretation. The ten authorised
live calls stay unspent and the canary bank stays whole.

WHY A VERSION STRING IS NOT PROVENANCE. `mi_agent_api.build_info` exists because
`app.version` was hand-written and identical across every deploy this year, so it
could not tell one build from another. The deploy workflow stamps
`build_info.json` with `GITHUB_SHA` inside the artefact and asserts it is forty
characters before uploading; this reads that stamp back out of the running
process. `source: "unstamped"` is therefore a FAILURE, not a missing nicety — it
means the running code cannot say what it is, and an acceptance that cannot
establish the deployed commit has to stop.

WHAT IT DELIBERATELY DOES NOT DO. It does not read, set or infer
`MI_AGENT_PLAN_SERVE`. That is an App Service application setting requiring ARM
access, which this repository's mi-api automation does not hold, and guessing at
it is the one thing the operator's own sequencing forbids.

WHY IT INSPECTS THE TOKEN'S EXPIRY. A 401 has two causes with opposite owners: a
production auth defect, or a stored credential that has simply aged out.
`MI_BEARER` is a static repository secret holding an Entra access token, and those
live for tens of minutes, so "the bearer did not authenticate" is an unactionable
finding on its own. The `exp` claim is read WITHOUT verifying the signature — this
is not an authentication decision, it is a diagnosis — and only the expiry facts
are reported. An integer timestamp is not credential material; the token itself is
never printed, and no other claim is either.

SECRET HYGIENE. The bearer arrives through the environment only, never argv, and
is never printed — the report is rescanned for it before anything is written.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
TIMEOUT = 120


def _token_expiry(token: str) -> dict:
    """What the token says about its own validity window. Diagnosis, not auth.

    The signature is deliberately NOT verified: this establishes WHY a 401
    happened, and the service has already made the authentication decision. Only
    the time fields are returned, so nothing that could authenticate anything
    leaves this function.
    """
    import base64
    import time as _time

    try:
        payload = token.split(".")[1]
        payload += "=" * (-len(payload) % 4)
        claims = json.loads(base64.urlsafe_b64decode(payload))
    except Exception as exc:                                         # noqa: BLE001
        return {"readable": False, "why": f"{type(exc).__name__}"}
    exp = claims.get("exp")
    now = int(_time.time())
    if not isinstance(exp, int):
        return {"readable": True, "exp": None,
                "why": "the token carries no integer exp claim"}
    return {
        "readable": True,
        "exp": exp,
        "expires_at_utc": datetime.fromtimestamp(exp, timezone.utc).isoformat(),
        "expired": exp <= now,
        "seconds_past_expiry": max(0, now - exp),
        # Whether the allow-list CAN be satisfied at all: `principal_of` reads
        # `actor_id`, which on the bearer path is the token's subject, and
        # `plan_serving_canary` refuses the literal "unknown-principal".
        "carries_a_subject": bool(claims.get("oid") or claims.get("sub")),
    }


def _get(url: str, *, bearer: str = "") -> tuple:
    """``(status, body_or_error)``. Never raises for an HTTP status."""
    request = urllib.request.Request(url, method="GET")
    request.add_header("Accept", "application/json")
    if bearer:
        request.add_header("Authorization", f"Bearer {bearer}")
    try:
        with urllib.request.urlopen(request, timeout=TIMEOUT) as response:
            return response.status, response.read().decode("utf-8", "replace")
    except urllib.error.HTTPError as exc:                            # noqa: PERF203
        return exc.code, exc.read().decode("utf-8", "replace")
    except Exception as exc:                                         # noqa: BLE001
        return 0, f"{type(exc).__name__}: {exc}"


#: WHERE THE BUILD STAMP IS READ FROM, in this order, and the order is the point.
#:
#: `/` is the app's OWN designated liveness probe and its docstring says why:
#: "This route touches no data: it answers as soon as the process is up. `/health`
#: is a READINESS and diagnostic route — it reports the resolved data source,
#: which means it RESOLVES the data source, which on a cold process is a governed
#: tape download." Both carry `build`, so provenance costs nothing on `/` and can
#: fail on `/health` for a reason that has nothing to do with which commit is
#: deployed. That is exactly what happened here: `/health` came back unusable on
#: a cold process while `/mi/catalogue` answered 200, and the run stopped on a
#: provenance failure that was an instrument fault, not a deployment fact.
#:
#: `/health` is kept as a fallback because it carries the same stamp, and a
#: reading from either is the same evidence.
BUILD_PATHS = ("/", "/health")


def read_build(base: str) -> tuple:
    """``(build, attempts)`` — the deployed build stamp, from the first route
    that answers with one."""
    attempts = []
    for path in BUILD_PATHS:
        status, body = _get(f"{base}{path}" if path != "/" else base)
        attempt = {"path": path, "status": status}
        build = {}
        if status == 200:
            try:
                build = dict((json.loads(body).get("build") or {}))
            except Exception as exc:                                 # noqa: BLE001
                attempt["parse_error"] = f"{type(exc).__name__}: {exc}"
        else:
            attempt["error"] = body[:300]
        attempt["commit"] = build.get("commit")
        attempts.append(attempt)
        if build.get("commit"):
            return build, attempts
    return {}, attempts


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--expect-commit", required=True)
    parser.add_argument("--preflight-path", default="/mi/catalogue")
    parser.add_argument("--out", default=str(HERE / "provenance_result.json"))
    args = parser.parse_args()

    base = args.base_url.rstrip("/")
    expected = args.expect_commit.strip().lower()
    bearer = (os.environ.get("MI_BEARER") or "").strip()

    record = {
        "recorded_at": datetime.now(timezone.utc).isoformat(),
        "base_url": base,
        "expected_commit": expected,
        "live_model_calls": 0,
        "live_mi_query_calls": 0,
    }

    # ---- step 3: WHICH COMMIT IS SERVING -------------------------------- #
    build, attempts = read_build(base)
    record["build_attempts"] = attempts
    served = str(build.get("commit") or "").strip().lower()
    record["build"] = build
    record["served_commit"] = served or None

    if not served:
        record["provenance"] = "FAIL"
        record["provenance_detail"] = (
            f"no route reported a commit: "
            + "; ".join(f"{a['path']} -> {a['status']}" for a in attempts)
            + "; the deployed build cannot be established")
    elif served != expected:
        record["provenance"] = "FAIL"
        record["provenance_detail"] = (
            f"serving {served} — expected {expected}")
    else:
        record["provenance"] = "PASS"
        record["provenance_detail"] = (
            f"the running service reports {served}, stamped from the artefact "
            f"(source={build.get('source')!r}, read from "
            f"{attempts[-1]['path']})")

    # ---- step 7: THE MINIMUM AUTHENTICATED PREFLIGHT --------------------- #
    # Run whatever provenance said, so one report answers both questions and a
    # provenance failure is not also an unexplained auth silence.
    record["bearer_present"] = bool(bearer)
    record["bearer_length"] = len(bearer)
    if not bearer:
        record["auth"] = "FAIL"
        record["auth_detail"] = "MI_BEARER is not set in the environment"
    else:
        status, body = _get(f"{base}{args.preflight_path}", bearer=bearer)
        record["preflight_path"] = args.preflight_path
        record["preflight_status"] = status
        if status == 200:
            record["auth"] = "PASS"
            record["auth_detail"] = (
                f"GET {args.preflight_path} returned 200 with the bearer; the "
                f"authenticated path is ready")
        elif status in (401, 403):
            record["auth"] = "FAIL"
            record["auth_detail"] = (
                f"GET {args.preflight_path} returned {status}: the bearer did "
                f"not authenticate")
        else:
            record["auth"] = "FAIL"
            record["auth_detail"] = f"GET {args.preflight_path} returned {status}"
        # DIAGNOSE A 401 RATHER THAN REPORTING IT AS A MYSTERY. Read only on the
        # failing path, because on the passing path the token's own claims are
        # nobody's business.
        if record["auth"] == "FAIL":
            record["token_validity"] = _token_expiry(bearer)

    record["verdict"] = ("PASS" if record["provenance"] == "PASS"
                         and record["auth"] == "PASS" else "FAIL")

    payload = json.dumps(record, indent=1, sort_keys=True) + "\n"
    # THE REPORT IS RESCANNED FOR THE SECRET before it is written. A harness that
    # leaks a bearer into committed evidence has done more damage than the gate
    # it was proving is worth.
    if bearer and bearer in payload:
        print("::error::the report contained the bearer; refusing to write it")
        return 2
    Path(args.out).write_text(payload, encoding="utf-8")

    print(f"PROVENANCE   {record['provenance']}  {record['provenance_detail']}")
    print(f"AUTH         {record['auth']}  {record['auth_detail']}")
    validity = record.get("token_validity") or {}
    if validity.get("exp"):
        print(f"TOKEN        expires {validity['expires_at_utc']}  "
              f"expired={validity['expired']}  "
              f"{validity['seconds_past_expiry']}s past expiry  "
              f"carries_a_subject={validity['carries_a_subject']}")
    elif validity:
        print(f"TOKEN        {validity}")
    print(f"SPEND        model calls 0, /mi/query calls 0")
    print(f"VERDICT      {record['verdict']}")
    print(f"wrote {Path(args.out).name}")
    return 0 if record["verdict"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
