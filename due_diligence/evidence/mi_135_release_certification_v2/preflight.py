#!/usr/bin/env python3
"""The pre-run infrastructure gate. Nothing here costs a model call.

FIVE CHECKS, AND THE LAST ONE EXISTS BECAUSE OF N01. The output-contract canary's
first question returned HTTP 500 and no evidence record was ever written, and the
platform log showed why: changing `MI_AGENT_PLAN_SERVE` restarts App Service, and
the question was asked seventeen seconds into the new container's warm-up. So the
liveness probe must answer healthy TWICE, separated by a wait, before a single
interpretation is bought.

WHY `/` AND NOT `/health`. `/` is the app's own designated liveness probe — it
touches no data and answers as soon as the process is up. `/health` RESOLVES the
governed tape, which on a cold process is a tape download; reading it here would
be slower, would prove something else, and has already produced one false
provenance failure in this programme.

STDLIB ONLY. This runs on a bare runner before anything is installed, and an
acceptance harness that imports the product has already failed this estate once,
mid-bank, after live interpretations had been spent.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
_REPO = HERE.parents[2]
sys.path.insert(0, str(_REPO))
sys.path.insert(0, str(_REPO / "due_diligence" / "evidence"
                      / "deployed_acceptance_0399a315"))

from due_diligence.evidence.change_intelligence_serving_canary import (  # noqa: E402
    provenance as pv)

MANIFEST = HERE / "certification_manifest.json"
HASH = HERE / "certification_manifest.sha256"


def _get(url: str, *, bearer: str = "", timeout: float = 30.0):
    request = urllib.request.Request(url, method="GET")
    if bearer:
        request.add_header("Authorization", f"Bearer {bearer}")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.status, response.read().decode("utf-8", "replace")
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read().decode("utf-8", "replace")[:400]
    except Exception as exc:                                         # noqa: BLE001
        return 0, f"{type(exc).__name__}: {exc}"


def liveness(base: str, *, attempts: int, gap: float, timeout: float):
    """Healthy TWICE in a row, separated by a wait. Returns (ok, reads)."""
    reads, consecutive = [], 0
    for index in range(attempts):
        status, _body = _get(f"{base}/", timeout=timeout)
        stamp = datetime.now(timezone.utc).isoformat()
        # HTTP 200 is the whole question here: `/` is the app's own liveness
        # probe and answers as soon as the process is up. WHICH build is running
        # is a separate check above, with its own reader.
        healthy = status == 200
        reads.append({"at": stamp, "status": status, "healthy": healthy})
        consecutive = consecutive + 1 if healthy else 0
        print(f"  liveness {index + 1}/{attempts}: HTTP {status} "
              f"{'healthy' if healthy else 'NOT healthy'} "
              f"(consecutive {consecutive})")
        if consecutive >= 2:
            return True, reads
        if index < attempts - 1:
            time.sleep(gap)
    return False, reads


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="https://app.traktinfra.io/api")
    parser.add_argument("--expect-commit", default="")
    parser.add_argument("--evidence-path",
                        default="/home/LogFiles/mi-plan-shadow/evidence.jsonl")
    parser.add_argument("--liveness-attempts", type=int, default=10)
    parser.add_argument("--liveness-gap", type=float, default=15.0)
    parser.add_argument("--out", default=str(HERE / "preflight_result.json"))
    args = parser.parse_args()

    base = args.base_url.rstrip("/")
    bearer = (os.environ.get("MI_BEARER") or "").strip()
    profile = (os.environ.get("AZURE_MI_API_PUBLISH_PROFILE") or "").strip()
    out = {"recorded_at": datetime.now(timezone.utc).isoformat(),
           "base_url": base, "expect_commit": args.expect_commit,
           "live_model_calls": 0, "live_mi_query_calls": 0, "checks": {}}
    failures = []

    def check(name, ok, detail):
        out["checks"][name] = {"ok": bool(ok), "detail": detail}
        print(f"{'PASS' if ok else 'FAIL'}  {name:34} {detail}")
        if not ok:
            failures.append(name)

    # 1. the certification contract and everything it pins.
    digest = hashlib.sha256(MANIFEST.read_bytes()).hexdigest()
    pinned = HASH.read_text().split()[0].strip()
    check("certification manifest", digest == pinned, digest[:32])
    manifest = json.loads(MANIFEST.read_bytes())
    for name, entry in manifest["reuses_unmodified"].items():
        path = _REPO / entry["path"]
        got = hashlib.sha256(path.read_bytes()).hexdigest()
        check(f"unmodified: {name}", got == entry["sha256"], got[:32])
    check("question count", manifest["question_count"] == 135,
          str(manifest["question_count"]))

    # 2. the historical evidence must be untouched.
    historical = _REPO / manifest["historical_baseline"]["results"]
    check("historical results present", historical.exists(), historical.name)

    # 3. credentials.
    check("MI_BEARER present", bool(bearer), f"{len(bearer)} characters")
    check("publish profile present", bool(profile), f"{len(profile)} characters")

    # 4. THE DEPLOYED BUILD, READ BY THE ESTATE'S OWN READER.
    #
    #    This used to parse the body here and look for a top-level `commit`.
    #    The stamp lives at `build.commit`, so the read came back empty against a
    #    perfectly healthy HTTP 200 and the gate failed with "serving HTTP 200" —
    #    a harness defect wearing a deployment failure's clothes.
    #    `provenance.read_build` already knows the shape, already tries `/` then
    #    `/health`, and is stdlib-only like everything else here. Reimplementing
    #    it was the mistake; using it is the fix.
    build, attempts = pv.read_build(base)
    served = str(build.get("commit") or "")
    out["deployed_sha"] = served
    out["build_read_attempts"] = attempts
    check("deployed SHA",
          bool(args.expect_commit) and served.lower() == args.expect_commit.lower(),
          f"serving {served or 'no stamp'}")

    # 5. the bearer, against an endpoint that resolves no question.
    if bearer:
        status, _ = _get(f"{base}/mi/catalogue", bearer=bearer)
        check("authenticated preflight", status == 200, f"HTTP {status}")
    else:
        check("authenticated preflight", False, "no bearer")

    # 6. the evidence sink.
    if profile:
        import run_acceptance as ra
        scm, user, password = ra.publish_profile_credentials(profile)
        sink = ra.Sink(scm, user, password, args.evidence_path)
        rows, detail = sink.records()
        out["pre_existing_sink_records"] = None if rows is None else len(rows)
        check("evidence sink readable", rows is not None,
              f"{detail}; {0 if rows is None else len(rows)} existing records")
    else:
        check("evidence sink readable", False, "no publish profile")

    # 7. THE N01 GATE. Healthy twice in a row, separated by a wait.
    print("liveness — the app's own probe, which touches no data:")
    ok, reads = liveness(base, attempts=args.liveness_attempts,
                         gap=args.liveness_gap, timeout=30.0)
    out["liveness_reads"] = reads
    check("liveness twice in succession", ok,
          f"{sum(1 for r in reads if r['healthy'])} healthy of {len(reads)} reads")

    out["verdict"] = "PASS" if not failures else "FAIL"
    out["failed_checks"] = failures
    body = json.dumps(out, indent=2, sort_keys=True, default=str)
    for secret in (bearer, profile):
        if secret and secret in body:
            print("::error::a credential survived into the preflight report")
            return 2
    Path(args.out).write_text(body)
    print(f"\nPREFLIGHT {out['verdict']}   ({len(failures)} failed)")
    return 0 if not failures else 2


if __name__ == "__main__":
    raise SystemExit(main())
