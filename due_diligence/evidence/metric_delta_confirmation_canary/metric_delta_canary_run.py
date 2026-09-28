#!/usr/bin/env python3
"""Run the pre-registered five-question confirmation canary. ONCE. No retries.

WHAT IT PROVES, and it is deliberately narrow: that `metric_delta` now serves
through its governed plan adapter, and that connecting it disturbed neither form
already serving.

THE GATE S03 TAUGHT. A metric delta answered only by legacy fallback is a FAIL
even when the number looks right, because S03's reading was flawless and it still
reached no owner. So every positive case pins the ROUTE, the CALCULATION OWNER
and the MODE, and `serving.response_served_from` must be NEW.

THE SCORER IS THE CORRECTED ONE. Temporal presence is read through
`serving_canary_run._temporal_route`, which now tests key MEMBERSHIP of
`interpreted_time_present` rather than the truthiness of a receipt the slice 1 and
slice 2 renderer fills with executor keys, and falls back to
`CandidateIntent.time.stated` rather than key presence in a dataclass `asdict`.
`--self-test` exercises both directions of that fault and costs nothing.

ONE ATTEMPT. The runner refuses if the result file already exists — and, unlike
the last bank, the result is committed to the repository immediately after the
run, so a fresh CI checkout carries the guard rather than starting clean.

NO NUMERIC PARITY IS CLAIMED. CI has no governed book. Each case records what a
reader with data access needs to reconcile it; offline parity against fixtures is
separate, passing evidence.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
_REPO_ROOT = HERE.parents[2]
sys.path.insert(0, str(_REPO_ROOT))

from due_diligence.evidence.deployed_acceptance_0399a315 import (  # noqa: E402
    run_acceptance as ra)
from due_diligence.evidence.change_intelligence_serving_canary import (  # noqa: E402
    provenance as pv, serving_canary_run as sc)

MANIFEST = HERE / "metric_delta_canary_bank_manifest.json"
HASH = HERE / "metric_delta_canary_bank_manifest.sha256"


def self_test() -> int:
    """The corrected temporal read, both directions of the S07/S08 fault."""
    def record(time_block):
        return {"interpretation": {"candidate_intent": {"time": time_block}},
                "model": {"raw_payload": {"time": time_block}}}

    slice2 = {"population_base": "funded", "aggregation": "sum"}
    change_form_absent = {"population_base": "funded",
                          "interpreted_time_present": False,
                          "interpreted_period_form": None}
    change_form_pair = {"population_base": "funded",
                        "interpreted_time_present": True,
                        "interpreted_period_form": "relative_pair"}
    cases = [
        ("an executor receipt must not be read as a temporal one",
         record({"form": "relative_pair", "stated": True}), slice2,
         "relative_pair"),
        ("...and a genuinely absent time still reads absent",
         record({"form": "current", "stated": False}), slice2, "ABSENT"),
        ("a change-form receipt saying absent is believed",
         record({"form": "current", "stated": False}), change_form_absent,
         "ABSENT"),
        ("a change-form receipt saying a pair is believed",
         record({"form": "relative_pair", "stated": True}), change_form_pair,
         "relative_pair"),
    ]
    failed = 0
    for label, rec, receipt, expected in cases:
        got = sc._temporal_route(rec, receipt)
        ok = got == expected
        failed += not ok
        print(f"{'ok ' if ok else 'BAD'} {label:56s} {got!r}")
    print("scorer self-test", "PASSED" if not failed else "FAILED")
    return 1 if failed else 0


def score(case: dict, record: dict, envelope: dict, status) -> dict:
    receipt = sc._receipt(envelope)
    route = ((envelope or {}).get("metadata") or {}).get("route")
    serving = (record or {}).get("serving") or {}
    provenance = serving.get("response_served_from")
    failures = []

    claimed, compiled = sc._claimed_form(record), sc._compiled_form(record)
    if claimed != case["change_form"]:
        failures.append(f"change_form: read {claimed!r}, pinned "
                        f"{case['change_form']!r}")
    if compiled and compiled != claimed:
        failures.append(f"the compiler derived {compiled!r} while the model "
                        f"claimed {claimed!r}")

    outcome = "ANSWER" if envelope.get("ok") else "REFUSE"
    if outcome != case["outcome"]:
        failures.append(f"outcome: {outcome}, pinned {case['outcome']}")

    # THE S03 GATE. Every case in this bank is positive and every one must be
    # served by the governed plan. Legacy fallback on a positive case is a
    # failure however good the number looks.
    owner = case["owner"]
    if provenance != "NEW":
        failures.append(f"served from {provenance!r}, not the governed plan — "
                        f"reason {serving.get('reason')!r}")
    if route != owner["route"]:
        failures.append(f"route: {route!r}, pinned {owner['route']!r}")
    if receipt.get("calculation_owner") != owner["calculation_owner"]:
        failures.append(f"calculation_owner: "
                        f"{receipt.get('calculation_owner')!r}, pinned "
                        f"{owner['calculation_owner']!r}")
    if receipt.get("composition_owner") != owner["composition_owner"]:
        failures.append(f"composition_owner: "
                        f"{receipt.get('composition_owner')!r}, pinned "
                        f"{owner['composition_owner']!r}")
    if owner["mode"] is not None and receipt.get("mode") != owner["mode"]:
        failures.append(f"mode: {receipt.get('mode')!r}, pinned "
                        f"{owner['mode']!r}")

    taken = sc._temporal_route(record, receipt)
    if taken not in case["temporal_accept"]:
        failures.append(f"temporal route {taken!r} is not in "
                        f"{case['temporal_accept']}")

    measures = sc._measures(record)
    if case.get("measure"):
        stray = [m for m in measures if m not in case["measure"]["concepts"]]
        if not measures:
            failures.append(f"no measure read; pinned family "
                            f"{case['measure']['family']}")
        elif stray:
            failures.append(f"measure {stray} outside "
                            f"{case['measure']['family']}")
    elif measures:
        failures.append(f"a measure was invented for a form that names none: "
                        f"{measures}")

    # THE FIELD THE PLAN NAMED MUST BE THE FIELD THE OWNER ANALYSED. A field the
    # owner excluded, with another standing in for it, is a different question
    # answered under the reader's words.
    if case["change_form"] == "metric_delta":
        requested = receipt.get("requested_fields") or []
        selected = receipt.get("selected_measures") or []
        missing = [f for f in requested if f not in selected]
        if not requested:
            failures.append("the receipt records no requested field")
        if missing:
            failures.append(f"the owner did not analyse {missing}; excluded "
                            f"{receipt.get('excluded_candidates')}")

    return {
        "case_id": case["id"], "question": case["question"],
        "http_status": status, "pass": not failures, "failures": failures,
        "read": {
            "change_form_claimed": claimed, "change_form_compiled": compiled,
            "capability": receipt.get("capability"),
            "operation": receipt.get("operation"), "mode": receipt.get("mode"),
            "measures": measures, "scope": receipt.get("direct_acquired_scope"),
            "time_stated": receipt.get("interpreted_time_present"),
            "period_form": receipt.get("interpreted_period_form"),
            "temporal_route": taken,
            "raw_payload_states_time": sc._raw_states_time(record),
            "period_from": receipt.get("period_from"),
            "period_to": receipt.get("period_to"),
            "serving_route": route, "serving_provenance": provenance,
            "calculation_owner": receipt.get("calculation_owner"),
            "composition_owner": receipt.get("composition_owner"),
            "outcome": outcome,
        },
        "receipt": {k: receipt.get(k) for k in (
            "plan_id", "population_base", "executed_scope",
            "temporal_default_applied", "temporal_default_method",
            "period_resolution", "requested_fields", "selected_measures",
            "excluded_candidates", "field_selection_policy",
            "metric_movements", "finding_count", "bridge_status",
            "bridge_reconciles") if k in receipt},
        "answer": str(envelope.get("answer") or "")[:600],
        "serving": serving,
        "seconds": envelope.get("__seconds__"),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--base-url", default="https://app.traktinfra.io/api")
    parser.add_argument("--path", default="/mi/query")
    parser.add_argument("--portfolio-id", default="ERE/2026-06-30")
    parser.add_argument("--expect-commit", default="")
    parser.add_argument("--evidence-path",
                        default="/home/LogFiles/mi-plan-shadow/evidence.jsonl")
    parser.add_argument("--poll-timeout", type=float, default=120.0)
    parser.add_argument("--poll-interval", type=float, default=5.0)
    parser.add_argument("--request-timeout", type=float, default=180.0)
    parser.add_argument("--out", default=str(HERE / "metric_delta_canary_result.json"))
    args = parser.parse_args()

    if args.self_test:
        return self_test()

    if Path(args.out).exists():
        print(f"::error::{Path(args.out).name} exists — this bank has been run. "
              f"It authorises one attempt with no retries.")
        return 2

    bearer = (os.environ.get("MI_BEARER") or "").strip()
    profile = (os.environ.get("AZURE_MI_API_PUBLISH_PROFILE") or "").strip()
    client_id = str(args.portfolio_id).split("/")[0]
    report = {"recorded_at": datetime.now(timezone.utc).isoformat(),
              "base_url": args.base_url, "portfolio_id": args.portfolio_id,
              "expect_commit": args.expect_commit}

    def stop(stage: str, detail: str, asked: int = 0) -> int:
        report.update({"verdict": "STOPPED", "stopped_at": stage,
                       "detail": detail, "questions_asked": asked})
        sc._write(report, args.out, [bearer, profile])
        print(f"STOPPED at {stage}: {detail}\nquestions asked: {asked}")
        return 2

    import hashlib
    body = MANIFEST.read_bytes()
    digest = hashlib.sha256(body).hexdigest()
    if digest != HASH.read_text().split()[0].strip():
        return stop("bank integrity", f"the bank was edited after hashing: {digest}")
    bank = json.loads(body)
    report.update({"bank_id": bank["bank_id"], "bank_sha256": digest})
    cases = bank["cases"]
    if len(cases) != bank["authorised_live_calls"]:
        return stop("bank integrity", f"{len(cases)} cases, "
                                      f"{bank['authorised_live_calls']} authorised")

    build, attempts = pv.read_build(args.base_url.rstrip("/"))
    served = str(build.get("commit") or "").strip()
    report.update({"served_commit": served, "build_attempts": attempts})
    if served.lower() != args.expect_commit.strip().lower():
        return stop("deployment provenance",
                    f"serving {served} — expected {args.expect_commit}")

    if not profile:
        return stop("evidence sink", "AZURE_MI_API_PUBLISH_PROFILE is not set")
    scm, user, password = ra.publish_profile_credentials(profile)
    sink = ra.Sink(scm, user, password, args.evidence_path)
    rows, detail = sink.records()
    report["sink_detail"] = detail
    if rows is None:
        return stop("evidence sink", f"the sink cannot be read ({detail})")
    already = {row.get("correlation_id") for row in rows}
    report["pre_existing_records"] = len(rows)

    # NO RECORD MAY ALREADY EXIST FOR THESE FIVE. A pre-existing one would let an
    # old answer satisfy a fresh case.
    asked_before = [case["id"] for case in cases
                    if any((row.get("request") or {}).get("question")
                           == case["question"] for row in rows)]
    if asked_before:
        return stop("bank freshness",
                    f"the sink already holds a record for {asked_before}")

    scored = []
    for index, case in enumerate(cases):
        envelope = sc.ask(args.base_url, args.path, args.portfolio_id,
                          case["question"], bearer=bearer,
                          timeout=args.request_timeout)
        status = envelope.get("__http_status__")
        if status in (401, 403):
            report["cases"] = scored
            return stop(f"authentication on {case['id']}",
                        f"HTTP {status}", asked=index)
        if envelope.get("__transport_error__"):
            report["cases"] = scored
            return stop(f"transport on {case['id']}",
                        str(envelope["__transport_error__"]), asked=index + 1)
        record, how = ra.poll_for(sink, case["question"], client_id,
                                  interval=args.poll_interval,
                                  timeout=args.poll_timeout, already=already)
        if record is not None:
            already.add(record.get("correlation_id"))
        verdict = score(case, record or {}, envelope, status)
        verdict.update({"evidence_lookup": how,
                        "record_found": record is not None})
        scored.append(verdict)
        print(f"{case['id']}  {'PASS' if verdict['pass'] else 'FAIL'}  "
              f"{verdict['read']['change_form_claimed']}  "
              f"{verdict['read']['serving_provenance']}  "
              f"route={verdict['read']['serving_route']}  "
              f"{'; '.join(verdict['failures'])[:140]}")

    report["cases"] = scored
    report["questions_asked"] = len(scored)
    report["passed"] = sum(1 for v in scored if v["pass"])
    report["verdict"] = ("PASS" if report["passed"] == len(cases) else "FAIL")
    report["numeric_parity"] = {
        "measured": False,
        "why": "CI has no governed book, so the independent owner run cannot "
               "happen here. Offline parity against fixtures is separate, "
               "passing evidence.",
    }
    sc._write(report, args.out, [bearer, profile])
    print(f"\n{report['bank_id']} = {report['verdict']}  "
          f"{report['passed']}/{len(cases)}")
    return 0 if report["verdict"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
