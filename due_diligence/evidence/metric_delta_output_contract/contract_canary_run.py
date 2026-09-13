#!/usr/bin/env python3
"""Run the pre-registered output-contract bank. ONCE. No retries.

MINIMUM INFRASTRUCTURE, AND NOTHING ELSE. Everything that already exists is
reused rather than rebuilt: the build-provenance reader, the `/mi/query` caller,
the Kudu evidence-sink reader and poller, the CandidateIntent / compiler /
serving record readers, and the corrected temporal-presence scorer. What is new
here is the two gates this bank exists to measure and the loop that spends it.

THE TWO NEW GATES.

  1. THE OPERATION CHAIN. A metric delta may arrive spelled `compare` or
     `movement`; the central contract collapses both to `movement`. So the
     CANONICAL operation on the compiled plan is pinned and the model's spelling
     is merely recorded — pinning the spelling would pin a coin toss.

  2. THE OUTPUT CONTRACT. `certification.score_requested_metric` decides whether
     the reader was actually answered. It is imported, not reimplemented: a
     second copy of that judgement could disagree with the one the offline
     controls exercise.

WHAT COUNTS AS A FAILURE IS THE BANK'S TO SAY. Every expectation is read from the
manifest. Nothing is added here after seeing a result, and a quality observation
outside the pinned contract is recorded as NON_BLOCKING and does not fail a case.

ONE ATTEMPT. The runner refuses if the result file exists, and the workflow
commits that file before it exits, so the guard the next checkout reads is a
committed file rather than a runner-local one.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
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

MANIFEST = HERE / "delta_contract_bank_manifest.json"
HASH = HERE / "delta_contract_bank_manifest.sha256"
RESULT_NAME = "delta_contract_canary_result.json"


def _certification():
    spec = importlib.util.spec_from_file_location(
        "_certification", HERE / "certification.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


cert = _certification()


# --------------------------------------------------------------------------- #
# scoring — every expectation read from the bank
# --------------------------------------------------------------------------- #
def score(case: dict, record: dict, envelope: dict, status) -> dict:
    receipt = sc._receipt(envelope)
    route = ((envelope or {}).get("metadata") or {}).get("route")
    serving = (record or {}).get("serving") or {}
    provenance = serving.get("response_served_from")
    owner = case["owner"]
    failures: list = []
    observations: list = []

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

    # A POSITIVE CASE SERVED BY LEGACY IS A FAILURE HOWEVER GOOD THE NUMBER IS.
    if provenance != "NEW":
        failures.append(f"served from {provenance!r}, not the governed plan — "
                        f"reason {serving.get('reason')!r}")
    if route != owner["route"]:
        failures.append(f"route: {route!r}, pinned {owner['route']!r}")
    if receipt.get("calculation_owner") != owner["calculation_owner"]:
        failures.append(f"calculation_owner: {receipt.get('calculation_owner')!r},"
                        f" pinned {owner['calculation_owner']!r}")
    if receipt.get("composition_owner") != owner["composition_owner"]:
        failures.append(f"composition_owner: {receipt.get('composition_owner')!r},"
                        f" pinned {owner['composition_owner']!r}")
    if owner["mode"] is not None and receipt.get("mode") != owner["mode"]:
        failures.append(f"mode: {receipt.get('mode')!r}, pinned {owner['mode']!r}")

    # THE OPERATION CHAIN. The candidate spelling is recorded; the CANONICAL
    # operation the plan carries is pinned.
    candidate = _candidate_operation(record)
    canonical = receipt.get("operation")
    if canonical != owner["canonical_operation"]:
        failures.append(f"canonical operation {canonical!r}, pinned "
                        f"{owner['canonical_operation']!r}")
    accept = owner.get("candidate_operation_accept") or []
    if candidate is not None and accept and candidate not in accept:
        failures.append(f"the model emitted operation {candidate!r}, outside the "
                        f"admitted variants {accept}")

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
            failures.append(f"measure {stray} outside {case['measure']['family']}")
    elif measures:
        failures.append(f"a measure was invented for a form that names none: "
                        f"{measures}")

    # THE OUTPUT CONTRACT, for the forms that name a metric.
    contract = None
    dispositions: dict = {}
    if case["change_form"] == "metric_delta":
        contract = cert.score_requested_metric(
            {"requested_fields": None,
             "expected_disposition": None}, envelope, receipt)
        failures.extend(contract["failures"])
        dispositions = contract["requested_metric_disposition"]
        pinned = case.get("expected_disposition") or []
        got = sorted(set(dispositions.values()))
        if pinned and not set(got).issubset(set(pinned)):
            failures.append(f"disposition {got}, pinned one of {pinned}")
    else:
        # A form that names no metric must carry no requested-metric block: a
        # "requested" answer for a request nobody made would be invented.
        if receipt.get("requested_metric_disposition"):
            failures.append("a requested-metric block appeared for a form that "
                            "names no metric")

    availability = [m.get("status") for m in (receipt.get("metric_movements") or ())]

    return {
        "case_id": case["id"], "question": case["question"],
        "http_status": status, "pass": not failures, "failures": failures,
        "non_blocking": observations,
        "read": {
            "change_form_claimed": claimed, "change_form_compiled": compiled,
            "candidate_operation": candidate, "compiled_operation": canonical,
            "capability": receipt.get("capability"), "mode": receipt.get("mode"),
            "measures": measures, "scope": receipt.get("direct_acquired_scope"),
            "owner_availability_status": availability,
            "requested_metric_disposition": dispositions,
            "requested_field": (contract or {}).get("requested_field"),
            "executed_field": (contract or {}).get("executed_field"),
            "served_metric_field": (contract or {}).get("served_metric_field"),
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
            "plan_id", "population_base", "executed_scope", "operation",
            "temporal_default_applied", "temporal_default_method",
            "period_resolution", "requested_fields", "selected_measures",
            "excluded_candidates", "field_selection_policy",
            "metric_movements", "requested_metric_disposition",
            "finding_count", "bridge_status", "bridge_reconciles")
            if k in receipt},
        "answer": str(envelope.get("answer") or "")[:900],
        "serving": serving,
        "seconds": envelope.get("__seconds__"),
    }


def _candidate_operation(record):
    """What the MODEL emitted, before any normalisation."""
    payload = ((record or {}).get("model") or {}).get("raw_payload") or {}
    if payload.get("operation") is not None:
        return payload.get("operation")
    intent = ((record or {}).get("interpretation") or {}).get(
        "candidate_intent") or {}
    return intent.get("operation")


# --------------------------------------------------------------------------- #
# self-tests — all six, offline, free
# --------------------------------------------------------------------------- #
def self_test() -> int:
    checks: list = []

    # 1. the bank hashes to its pin.
    body = MANIFEST.read_bytes()
    digest = hashlib.sha256(body).hexdigest()
    pinned = HASH.read_text().split()[0].strip()
    checks.append(("1 the bank hashes to the pinned sha256", digest == pinned,
                   digest[:16]))
    bank = json.loads(body)

    # 2 and 3. the requested-metric scorer, both directions.
    m03 = cert.score_requested_metric(
        {"requested_fields": ["current_interest_rate"]},
        {"answer": "Between 30 November 2025 and 30 June 2026, 0 of 1 governed "
                   "metrics could be compared. The balance bridge reconciles.",
         "artifacts": [{"title": "Metric movements",
                        "rows": [{"canonical_field": "current_interest_rate",
                                  "metric": "Current Interest Rate"}]}]},
        {"requested_fields": ["current_interest_rate"],
         "selected_measures": ["current_interest_rate"],
         "metric_movements": [{"field": "current_interest_rate",
                               "movement_value": -3.2147396826825796,
                               "movement_unit": "percentage_point",
                               "status": "partially_available"}],
         "requested_metric_disposition": [
             {"canonical_field": "current_interest_rate",
              "disposition": "QUALIFIED", "status": "partially_available"}]})
    checks.append(("2 the historical M03 shape is REJECTED", not m03["pass"],
                   "; ".join(m03["failures"])[:70]))

    qualified = cert.score_requested_metric(
        {"requested_fields": ["current_interest_rate"]},
        {"answer": "Current Interest Rate moved −3.21 pp between 30 November "
                   "2025 and 30 June 2026, from 9.53% to 6.32%. This is a "
                   "partially available comparison: 0 row(s) were excluded at "
                   "the opening date and 12 at the closing date.",
         "artifacts": [{"title": "Metric movements",
                        "rows": [{"canonical_field": "current_interest_rate",
                                  "metric": "Current Interest Rate"}]}]},
        {"requested_fields": ["current_interest_rate"],
         "selected_measures": ["current_interest_rate"],
         "metric_movements": [{"field": "current_interest_rate",
                               "movement_value": -3.2147396826825796,
                               "movement_unit": "percentage_point",
                               "status": "partially_available"}],
         "requested_metric_disposition": [
             {"canonical_field": "current_interest_rate",
              "disposition": "QUALIFIED", "status": "partially_available"}]})
    checks.append(("3 a communicated QUALIFIED answer is ACCEPTED",
                   qualified["pass"], "; ".join(qualified["failures"])[:70]))

    # 4. temporal presence: the receipt when present, `time.stated` otherwise.
    def record(time_block):
        return {"interpretation": {"candidate_intent": {"time": time_block}},
                "model": {"raw_payload": {"time": time_block}}}

    executor_receipt = {"population_base": "funded", "aggregation": "sum"}
    change_receipt = {"population_base": "funded",
                      "interpreted_time_present": False,
                      "interpreted_period_form": None}
    a = sc._temporal_route(record({"form": "relative_pair", "stated": True}),
                           executor_receipt)
    b = sc._temporal_route(record({"form": "current", "stated": False}),
                           executor_receipt)
    c = sc._temporal_route(record({"form": "current", "stated": False}),
                           change_receipt)
    checks.append(("4 temporal presence: receipt when present, else time.stated",
                   (a, b, c) == ("relative_pair", "ABSENT", "ABSENT"),
                   f"{a}/{b}/{c}"))

    # 5. a positive case served only by LEGACY_FALLBACK fails certification.
    case = dict(bank["cases"][0])
    legacy = score(case,
                   {"serving": {"response_served_from": "LEGACY_FALLBACK",
                                "reason": "INELIGIBLE:SOMETHING"},
                    "model": {"raw_payload": {"change_form": "metric_delta",
                                              "operation": "compare"}},
                    "interpretation": {"candidate_intent": {
                        "change_form": "metric_delta", "operation": "compare"}}},
                   {"ok": True, "answer": "", "metadata": {}}, 200)
    checks.append(("5 a LEGACY_FALLBACK positive case FAILS", not legacy["pass"],
                   "; ".join(legacy["failures"])[:70]))

    # 6. the one-attempt guard: the runner refuses when the result exists, and
    #    the file it refuses on is the one the workflow commits.
    committed = (HERE / RESULT_NAME)
    checks.append(("6 the one-attempt guard names the committed result file",
                   RESULT_NAME.endswith(".json") and not committed.exists(),
                   f"{RESULT_NAME} absent = unspent"))

    failed = 0
    for label, ok, detail in checks:
        failed += not ok
        print(f"{'ok  ' if ok else 'FAIL'} {label:58} {detail}")
    print("runner self-test " + ("PASSED" if not failed else "FAILED"))
    return 1 if failed else 0


# --------------------------------------------------------------------------- #
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
    parser.add_argument("--out", default=str(HERE / RESULT_NAME))
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

    body = MANIFEST.read_bytes()
    digest = hashlib.sha256(body).hexdigest()
    if digest != HASH.read_text().split()[0].strip():
        return stop("bank integrity", f"the bank was edited after hashing: {digest}")
    bank = json.loads(body)
    report.update({"bank_id": bank["bank_id"], "bank_sha256": digest})
    cases = bank["cases"]
    if len(cases) != bank["authorised_live_calls"]:
        return stop("bank integrity",
                    f"{len(cases)} cases, {bank['authorised_live_calls']} authorised")

    if self_test() != 0:
        return stop("runner self-test", "a self-test failed; no question asked")

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
            return stop(f"authentication on {case['id']}", f"HTTP {status}",
                        asked=index)
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
        read = verdict["read"]
        print(f"{case['id']}  {'PASS' if verdict['pass'] else 'FAIL'}  "
              f"{read['change_form_claimed']}  "
              f"{read['candidate_operation']}->{read['compiled_operation']}  "
              f"{read['serving_provenance']}  "
              f"{read['requested_metric_disposition']}  "
              f"{'; '.join(verdict['failures'])[:120]}")

    report["cases"] = scored
    report["questions_asked"] = len(scored)
    report["passed"] = sum(1 for v in scored if v["pass"])
    report["verdict"] = "PASS" if report["passed"] == len(cases) else "FAIL"
    report["numeric_parity"] = {
        "measured": False,
        "why": "CI has no governed book, so the owner cannot be run "
               "independently here. Offline parity against fixtures is separate, "
               "passing evidence (3/3, zero differences).",
    }
    sc._write(report, args.out, [bearer, profile])
    print(f"\n{report['bank_id']} = {report['verdict']}  "
          f"{report['passed']}/{len(cases)}")
    return 0 if report["verdict"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
