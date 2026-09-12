#!/usr/bin/env python3
"""Finish the 135 bank: the 31 outstanding questions, ordinary production path.

WHY THIS EXISTS RATHER THAN ANOTHER CANARY RUN. The cost audit established that
NONE of the outstanding questions targets a capability migrated onto the
governed serving path at 00bb3e9d — they are forecast, limit_assessment,
borrowing_base and unbound analytical intents. With the canary on they could only
ever produce LEGACY_FALLBACK, and the reader would receive the legacy envelope
either way. With `MI_AGENT_PLAN_SERVE=off` the same user-facing answer costs no
interpretation call at all.

SO THIS MEASURES THE USER-FACING ANSWER, AND SAYS SO. Nothing here is evidence
about the governed path; every question it asks is recorded
`MIGRATION_OUTCOME = NOT_TESTED_KNOWN_UNMIGRATED`, never NEW and never
LEGACY_FALLBACK, because the canary was deliberately not invoked.

THREE THINGS IT WILL NOT DO, each learnt from the run it is completing:

  * IT DOES NOT POLL THE EVIDENCE SINK. The canary is off, so no governed record
    is written and waiting for one is waiting for nothing. The previous run spent
    93 of its 149 minutes doing exactly that.
  * IT STOPS DEAD ON HTTP 401. A bearer that has expired will not un-expire
    during the run; continuing cost the previous run 30 wasted questions and the
    evidence that went with them.
  * IT PREFLIGHTS THE BEARER WITHOUT SPENDING A QUESTION. The catalogue endpoint
    is an authenticated read that calls no model, so the credential is proved
    before the bank is touched, and never with a bank question.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parents[2]
for _p in (_REPO,
           _REPO / "due_diligence/evidence/deployed_acceptance_0399a315",
           _REPO / "due_diligence/evidence/slice1b_serving_canary",
           _REPO / "due_diligence/evidence/mi_api_certification",
           _REPO / "due_diligence/evidence/plan_temporal_slice2"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

QUESTIONS = _HERE / "questions.json"
ORIGINAL = _HERE / "raw_records.json"
PRODUCT_SHA = "00bb3e9d8175a395b6643772379866d6bc6169eb"
MIGRATED_CAPABILITIES = frozenset({"generic_analysis", "pipeline"})


def outstanding_ids() -> List[str]:
    """The missing set, DERIVED from the committed evidence, never assumed.

    A question is COMPLETED_GOOD when the request reached the service, a governed
    record was written, and the interpreter returned a reading. Everything else
    is outstanding — which is what makes this reproducible rather than a
    hand-kept list that drifts from the evidence it claims to describe.
    """
    bank = json.loads(QUESTIONS.read_text(encoding="utf-8"))["questions"]
    authoritative = [q["question_id"] for q in bank]
    good = set()
    for entry in json.loads(ORIGINAL.read_text(encoding="utf-8"))["records"]:
        record, envelope = entry.get("record"), entry.get("envelope") or {}
        if envelope.get("__transport_error__") or record is None:
            continue
        if record.get("disposition") == "INTERPRETER_FAILURE":
            continue
        good.add(entry["question_id"])
    return [q for q in authoritative if q not in good]


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="https://app.traktinfra.io/api")
    parser.add_argument("--path", default="/mi/query")
    parser.add_argument("--portfolio-id", default="ERE/2026-06-30")
    parser.add_argument("--expect-commit", default=PRODUCT_SHA)
    parser.add_argument("--out", default="raw_records_legacy_completion.json")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args(argv)

    if args.self_test:
        return self_test()

    import run_acceptance as ra
    import run_serving_acceptance as s1b
    from certify_mi_api import _live_asker

    bank = {q["question_id"]: q
            for q in json.loads(QUESTIONS.read_text(encoding="utf-8"))["questions"]}
    ids = outstanding_ids()
    bearer = os.environ.get("MI_BEARER", "").strip()
    secrets = [s for s in (bearer,) if len(s) >= ra.MIN_SECRET_LENGTH]

    report: Dict[str, Any] = {
        "what_this_is": "the 31 outstanding bank questions, asked through the "
                        "ordinary production MI path with the governed canary "
                        "OFF; user-facing evidence only",
        "product_sha_under_test": args.expect_commit,
        "run_mode": "LEGACY_ONLY_COMPLETION",
        "mi_agent_plan_serve_assumed_state": "off",
        "mi_agent_plan_serve_verified": False,
        "outstanding_count": len(ids), "outstanding_ids": ids,
        "stages": {}, "records": [],
        "http_acceptance_requests": 0, "transport_retries": 0,
    }

    def stop(stage: str, verdict: str, message: str) -> int:
        report["verdict"], report["stopped_at"] = verdict, stage
        report["stages"][stage] = message
        ra._save(report, args.out, secrets)
        print(f"::error::{stage}: {message}")
        return 2

    if len(ids) != 31:
        return stop("reconciliation", "NOT_EXECUTABLE",
                    f"the outstanding set is {len(ids)}, not 31; the evidence and "
                    f"the expectation disagree and nothing is asked until they do not")
    if not bearer:
        return stop("credentials", "NOT_EXECUTABLE", "MI_BEARER is empty")

    served = s1b.served_commit(args.base_url)
    report["stages"]["provenance"] = {"served_commit": served,
                                      "expected": args.expect_commit}
    if not served or not served.startswith(args.expect_commit[:7]):
        return stop("provenance", "FAIL",
                    f"serving {served!r} — expected {args.expect_commit}")
    print(f"PROVENANCE CONFIRMED  served={served}")

    # THE BEARER, PROVED WITHOUT SPENDING A BANK QUESTION. An authenticated read
    # that calls no model: if this 401s, nothing below is attempted.
    import run_temporal_acceptance as t2
    catalogue, detail = t2.live_catalogue(args.base_url,
                                          args.portfolio_id.split("/", 1)[0])
    report["stages"]["bearer_preflight"] = {"catalogue": catalogue, "detail": detail}
    if not catalogue:
        return stop("bearer_preflight", "NOT_EXECUTABLE",
                    f"the authenticated non-model check did not succeed: {detail}")
    print(f"BEARER PREFLIGHT OK   catalogue={catalogue}")

    ask = _live_asker(args.base_url, args.path, portfolio_id=args.portfolio_id)
    for index, qid in enumerate(ids, start=1):
        case = bank[qid]
        envelope = ask(case["question"])
        report["http_acceptance_requests"] += 1
        original_failure = None

        if envelope.get("__transport_error__"):
            status = envelope.get("__http_status__")
            if status == 401:
                report["records"].append({**case, "envelope": envelope,
                                          "run_mode": "LEGACY_ONLY_COMPLETION"})
                return stop("authentication", "STOPPED_ON_401",
                            f"HTTP 401 at question {index} ({qid}). The run is "
                            f"stopped rather than continued: a bearer does not "
                            f"un-expire mid-bank, and continuing spent 30 "
                            f"questions last time.")
            original_failure = dict(envelope)
            print(f"  [{index:2d}/{len(ids)}] {qid:16s} transport {status} — one retry")
            time.sleep(5.0)
            envelope = ask(case["question"])
            report["http_acceptance_requests"] += 1
            report["transport_retries"] += 1
            if envelope.get("__http_status__") == 401:
                return stop("authentication", "STOPPED_ON_401",
                            f"HTTP 401 on the retry at question {index} ({qid})")

        report["records"].append({
            **case,
            "run_mode": "LEGACY_ONLY_COMPLETION",
            "migration_outcome": "NOT_TESTED_KNOWN_UNMIGRATED",
            "governed_capability_migrated": False,
            "original_transport_failure": original_failure,
            "envelope": envelope,
        })
        print(f"  [{index:2d}/{len(ids)}] {qid:16s} ok={str(envelope.get('ok')):5s} "
              f"{case['question'][:50]}")

    report["verdict"] = "COLLECTED"
    ra._save(report, args.out, secrets)
    print(f"\n  COLLECTED  questions={len(ids)}  "
          f"http_requests={report['http_acceptance_requests']}  "
          f"transport_retries={report['transport_retries']}")
    print(f"  written {args.out}")
    return 0


def self_test() -> int:
    failures: List[str] = []

    def check(name: str, condition: bool) -> None:
        if not condition:
            failures.append(name)
        print(f"   {'ok  ' if condition else 'FAIL'} {name}")

    ids = outstanding_ids()
    check("the outstanding set derives to exactly 31", len(ids) == 31)
    check("every outstanding id is in the authoritative bank",
          set(ids) <= {q["question_id"] for q in
                       json.loads(QUESTIONS.read_text(encoding="utf-8"))["questions"]})
    check("no completed question is re-asked",
          not (set(ids) & {e["question_id"] for e in
                           json.loads(ORIGINAL.read_text(encoding="utf-8"))["records"]
                           if e.get("record") and not (e.get("envelope") or {}).get("__transport_error__")
                           and e["record"].get("disposition") != "INTERPRETER_FAILURE"}))

    # THE GATE THAT DECIDES THIS RUN IS LEGITIMATE AT ALL, applied without
    # importing the product: `expected_capability.json` is cut from
    # `expected_intents.yaml` with that file's sha256 recorded, for the same
    # reason the bank itself is — this self-test runs on a bare runner, and a
    # harness that reaches for yaml has already failed mid-bank in this estate.
    # Re-derived from the YAML and compared string for string wherever yaml is
    # importable.
    import hashlib
    caps_body = json.loads((_HERE / "expected_capability.json")
                           .read_text(encoding="utf-8"))
    expected = caps_body["expected"]
    check("the expectations file is the one this JSON was cut from",
          hashlib.sha256((_REPO / "mi_agent/interpretation_v2/banks/"
                          "expected_intents.yaml").read_bytes()).hexdigest()
          == caps_body["source_sha256"])
    caps = {str((expected.get(q) or {}).get("capability")) for q in ids}
    check("no outstanding question targets a migrated capability",
          not (caps & MIGRATED_CAPABILITIES))
    print(f"        expected capabilities outstanding: {sorted(caps)}")
    try:
        import yaml
    except ImportError:
        print("   skip  re-derivation of the capability map (no yaml here)")
    else:
        bank = yaml.safe_load((_REPO / "mi_agent/interpretation_v2/banks/"
                               "interpretation_bank_135.yaml").read_text())
        exp = yaml.safe_load((_REPO / "mi_agent/interpretation_v2/banks/"
                              "expected_intents.yaml").read_text())["expectations"]
        fresh = {v["source_id"]: ((exp.get(c["id"]) or {}).get("expected") or {})
                 .get("capability")
                 for c in bank["canonicals"] for v in c["variants"]}
        check("the capability map matches the YAML verbatim",
              all(fresh[q] == (expected.get(q) or {}).get("capability")
                  for q in fresh))

    print(f"\n  {len(failures)} failing rule(s)" if failures
          else "\n  every rule holds")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
