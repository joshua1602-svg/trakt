#!/usr/bin/env python3
"""Run the pre-registered ten-question serving canary. ONCE. No retries.

WHAT IT DOES, IN ORDER, AND WHY THE ORDER MATTERS.

    1. re-hash the bank              a bank that drifted is not pre-registered
    2. confirm the deployed commit   a wrong build makes every verdict meaningless
    3. read the evidence sink        so no PRE-EXISTING record can satisfy a case
    4. ask the ten, once each        no retry, no rephrase, no second attempt
    5. poll the sink per question    the record is the oracle for the plan
    6. score against the pinned bank and write the result

Steps 1-3 all fail BEFORE the first question, so a run that cannot be adjudicated
costs nothing.

THE ORACLE IS THE EVIDENCE RECORD, NOT THE ANSWER. The served envelope carries a
`metadata.governedPlan` block only when the governed plan path served it, so for
the two forms this sprint deliberately did not connect the envelope says nothing
about what was interpreted. `plan_serving_canary` writes a record for every
request its principal is allow-listed for, carrying the CandidateIntent and the
compiled plan, so the analytical form is read from there for all ten and from the
answer for none.

WHAT THIS RUN CANNOT DO, STATED HERE RATHER THAN GLOSSED. The bank says numeric
truth is established by running the deterministic owner independently over the
same governed snapshots. That is exactly what the offline parity gate does — and
it CANNOT be done from CI, which has no governed book. So this run records every
field a reader with data access needs to reconcile it (the resolved pair, the
snapshot references, the executed scope, and the bridge's own reconciliation
flag and residual) and claims NO independent numeric parity. The offline gate
proved parity against fixtures; this proves the arrow, the owner, the scope and
the temporal route live. Conflating the two would be claiming a measurement
nobody took.

SECRET HYGIENE. The bearer and the publish profile arrive through the environment
only, never argv. The report is rescanned for both and is not written if either
survived.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
_REPO_ROOT = HERE.parents[2]
sys.path.insert(0, str(_REPO_ROOT))

from due_diligence.evidence.deployed_acceptance_0399a315 import (  # noqa: E402
    run_acceptance as ra)
from due_diligence.evidence.slice1b_serving_canary import (  # noqa: E402
    run_serving_acceptance as sa)

MANIFEST = HERE / "serving_canary_bank_manifest.json"
HASH = HERE / "serving_canary_bank_manifest.sha256"

#: Routes this sprint added. A control case arriving through one of these means
#: a form that was supposed to be untouched has been claimed by a new owner.
CHANGE_FORM_ROUTES = {"governed_plan_material_summary", "governed_plan_attribution"}


# --------------------------------------------------------------------------- #
# transport
# --------------------------------------------------------------------------- #
def ask(base_url: str, path: str, portfolio_id: str, question: str,
        *, bearer: str, timeout: float) -> dict:
    """One question, once. Never raises; a transport fault is recorded as one."""
    body = json.dumps({"question": question,
                       "portfolioId": portfolio_id}).encode("utf-8")
    request = urllib.request.Request(
        f"{base_url.rstrip('/')}{path}", data=body, method="POST")
    request.add_header("Content-Type", "application/json")
    request.add_header("Accept", "application/json")
    request.add_header("Authorization", f"Bearer {bearer}")
    started = time.monotonic()
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            payload = json.loads(response.read().decode("utf-8", "replace"))
            payload["__http_status__"] = response.status
    except urllib.error.HTTPError as exc:
        payload = {"__http_status__": exc.code,
                   "__body__": exc.read().decode("utf-8", "replace")[:600]}
    except Exception as exc:                                         # noqa: BLE001
        payload = {"__transport_error__": f"{type(exc).__name__}: {exc}"}
    payload["__seconds__"] = round(time.monotonic() - started, 1)
    return payload


# --------------------------------------------------------------------------- #
# reading what happened
# --------------------------------------------------------------------------- #
def _plan_of(record: dict) -> dict:
    compiler = (record or {}).get("compiler") or {}
    return compiler.get("plan") or {}


def _claimed_form(record: dict) -> str:
    """The form the MODEL claimed, as v1-v4 scored it."""
    intent = ((record or {}).get("interpretation") or {}).get(
        "candidate_intent") or {}
    return str(intent.get("change_form") or "") or None


def _compiled_form(record: dict) -> str:
    """The form the COMPILER derived, which is what serving dispatches on."""
    binding = (((_plan_of(record).get("provenance") or {})
                .get("compiler_bindings") or {}).get("change_form") or {})
    return str(binding.get("form") or "") or None


def _measures(record: dict) -> list:
    intent = ((record or {}).get("interpretation") or {}).get(
        "candidate_intent") or {}
    out = []
    for output in (intent.get("outputs") or []):
        for measure in (output.get("measures") or []):
            out.append(str(measure.get("concept") or ""))
    for measure in (intent.get("measures") or []):
        out.append(str(measure.get("concept") or ""))
    return [m for m in out if m]


def _receipt(envelope: dict) -> dict:
    return (((envelope or {}).get("metadata") or {})
            .get("governedPlan") or {}).get("executed") or {}


def _temporal_route(record: dict, receipt: dict) -> str:
    """EXPLICIT_PAIR / EXPLICIT_ANCHOR / ABSENT, from structure not wording."""
    if receipt:
        if not receipt.get("interpreted_time_present"):
            return "ABSENT"
        form = receipt.get("interpreted_period_form")
        return "current" if form == "current" else str(form or "")
    intent = ((record or {}).get("interpretation") or {}).get(
        "candidate_intent") or {}
    time_block = intent.get("time")
    if not isinstance(time_block, dict) or "form" not in time_block:
        return "ABSENT"
    return str(time_block.get("form") or "")


# --------------------------------------------------------------------------- #
# scoring — strictly against the pinned bank
# --------------------------------------------------------------------------- #
def score(case: dict, record: dict, envelope: dict, status) -> dict:
    receipt = _receipt(envelope)
    route = ((envelope or {}).get("metadata") or {}).get("route")
    served_ok = bool(envelope.get("ok"))
    failures = []

    claimed, compiled = _claimed_form(record), _compiled_form(record)
    expected_form = case.get("change_form")
    if expected_form:
        if claimed != expected_form:
            failures.append(f"change_form: read {claimed!r}, pinned "
                            f"{expected_form!r}")
        if compiled and compiled != claimed:
            failures.append(f"the compiler derived {compiled!r} while the model "
                            f"claimed {claimed!r}")

    # OUTCOME. A REFUSE case is satisfied by a governed decline OR a
    # clarification; what it may never be is a confident answer.
    outcome = "ANSWER" if served_ok else "REFUSE"
    if case["outcome"] != outcome:
        failures.append(f"outcome: {outcome}, pinned {case['outcome']}")

    # EXECUTION OWNER. Pinned positively for the forms this sprint connected and
    # negatively for the two it did not touch.
    owner = case.get("owner") or {}
    if owner.get("serving_provenance") == "GOVERNED_PLAN":
        if route != owner.get("route"):
            failures.append(f"route: {route!r}, pinned {owner.get('route')!r}")
        if receipt.get("calculation_owner") != owner.get("calculation_owner"):
            failures.append(
                f"calculation_owner: {receipt.get('calculation_owner')!r}, "
                f"pinned {owner.get('calculation_owner')!r}")
        if receipt.get("composition_owner") != owner.get("composition_owner"):
            failures.append(
                f"composition_owner: {receipt.get('composition_owner')!r}, "
                f"pinned {owner.get('composition_owner')!r}")
    elif owner.get("route_must_not_be"):
        if route in owner["route_must_not_be"]:
            failures.append(f"a form this sprint did not touch arrived through "
                            f"{route!r}")

    # TEMPORAL ROUTE, valid per form from the pinned accept policy.
    taken = _temporal_route(record, receipt)
    if case.get("temporal_accept") and taken not in case["temporal_accept"]:
        failures.append(f"temporal route {taken!r} is not in "
                        f"{case['temporal_accept']}")

    # MEASURE SEMANTICS.
    measures = _measures(record)
    if case.get("measure"):
        permitted = set(case["measure"]["concepts"])
        stray = [m for m in measures if m not in permitted]
        if not measures:
            failures.append(f"no measure read; pinned family "
                            f"{case['measure']['family']}")
        elif stray:
            failures.append(f"measure {stray} outside {case['measure']['family']}")
    elif measures:
        failures.append(f"a measure was invented for a form that names none: "
                        f"{measures}")

    # SCOPE. The narrowing must reach the NUMBERS, not only the label.
    if case.get("scope_must_narrow"):
        ids = (receipt.get("executed_scope") or {}).get("portfolio_ids") or []
        if not ids:
            failures.append("the scoped question narrowed to no portfolio id")
    if case.get("must_not_answer") and served_ok:
        failures.append("a question naming an ungoverned portfolio was answered")

    return {
        "case_id": case["id"],
        "question": case["question"],
        "http_status": status,
        "pass": not failures,
        "failures": failures,
        "read": {
            "change_form_claimed": claimed,
            "change_form_compiled": compiled,
            "route": route,
            "outcome": outcome,
            "temporal_route": taken,
            "measures": measures,
        },
        # Everything a reader with data access needs to reconcile the numbers
        # independently. This run does not reconcile them; see the docstring.
        "receipt": {key: receipt.get(key) for key in (
            "plan_id", "capability", "operation", "mode", "population_base",
            "direct_acquired_scope", "source_scope", "executed_scope",
            "interpreted_time_present", "interpreted_period_form",
            "temporal_default_applied", "temporal_default_method",
            "temporal_default_owner", "period_from", "period_to",
            "snapshot_references", "calculation_owner", "composition_owner",
            "finding_count", "bridge_status", "bridge_reconciles",
            "bridge_residual") if key in receipt},
        "answer": str(envelope.get("answer") or "")[:600],
        "serving": (record or {}).get("serving"),
        "seconds": envelope.get("__seconds__"),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--path", default="/mi/query")
    parser.add_argument("--portfolio-id", required=True)
    parser.add_argument("--expect-commit", required=True)
    parser.add_argument("--evidence-path",
                        default="/home/LogFiles/mi-plan-shadow/evidence.jsonl")
    parser.add_argument("--poll-timeout", type=float, default=120.0)
    parser.add_argument("--poll-interval", type=float, default=5.0)
    parser.add_argument("--request-timeout", type=float, default=180.0)
    parser.add_argument("--out", default=str(HERE / "serving_canary_result.json"))
    args = parser.parse_args()

    # ONE ATTEMPT, ENFORCED BY THE FILESYSTEM. The bank authorises a single run
    # with `retries: 0`, and the workflow that fires it is push-triggered, so a
    # later edit to that file would otherwise ask the ten a second time. A result
    # that already exists means this bank has been spent.
    if Path(args.out).exists():
        print(f"::error::{Path(args.out).name} already exists — this bank has "
              f"been run. It authorises one attempt with no retries.")
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
        _write(report, args.out, [bearer, profile])
        print(f"STOPPED at {stage}: {detail}")
        print(f"questions asked: {asked}")
        return 2

    # -- 1. the bank ------------------------------------------------------- #
    import hashlib
    body = MANIFEST.read_bytes()
    digest = hashlib.sha256(body).hexdigest()
    pinned = HASH.read_text().split()[0].strip()
    if digest != pinned:
        return stop("bank integrity",
                    f"the bank was edited after hashing: {digest} != {pinned}")
    bank = json.loads(body)
    report["bank_id"] = bank["bank_id"]
    report["bank_sha256"] = digest
    cases = bank["cases"]
    if len(cases) != bank["authorised_live_calls"]:
        return stop("bank integrity",
                    f"{len(cases)} cases but {bank['authorised_live_calls']} "
                    f"authorised calls")

    # -- 2. the deployed commit -------------------------------------------- #
    served = sa.served_commit(args.base_url)
    report["served_commit"] = served
    if (served or "").lower() != args.expect_commit.strip().lower():
        return stop("deployment provenance",
                    f"serving {served} — expected {args.expect_commit}")

    # -- 3. the sink, BEFORE anything is asked ------------------------------ #
    if not profile:
        return stop("evidence sink",
                    "AZURE_MI_API_PUBLISH_PROFILE is not set, so which path "
                    "served cannot be proved")
    scm, user, password = ra.publish_profile_credentials(profile)
    sink = ra.Sink(scm, user, password, args.evidence_path)
    rows, detail = sink.records()
    report["sink_detail"] = detail
    if rows is None:
        return stop("evidence sink",
                    f"the sink cannot be read ({detail}); stopping BEFORE "
                    f"spending model calls")
    already = {row.get("correlation_id") for row in rows}
    report["pre_existing_records"] = len(rows)

    # -- 4-5. the ten, once each -------------------------------------------- #
    scored = []
    for index, case in enumerate(cases):
        envelope = ask(args.base_url, args.path, args.portfolio_id,
                       case["question"], bearer=bearer,
                       timeout=args.request_timeout)
        status = envelope.get("__http_status__")
        if status in (401, 403):
            report["cases"] = scored
            return stop(f"authentication on {case['id']}",
                        f"HTTP {status}; the credential was refused before any "
                        f"semantics ran", asked=index)
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
        verdict["evidence_lookup"] = how
        verdict["record_found"] = record is not None
        scored.append(verdict)
        print(f"{case['id']}  {'PASS' if verdict['pass'] else 'FAIL'}  "
              f"{verdict['read']['change_form_claimed']}  "
              f"route={verdict['read']['route']}  "
              f"{'; '.join(verdict['failures'])[:150]}")

    report["cases"] = scored
    report["questions_asked"] = len(scored)
    passed = sum(1 for v in scored if v["pass"])
    report["passed"] = passed
    report["verdict"] = "PASS" if passed == len(cases) else "FAIL"
    report["numeric_parity"] = {
        "measured": False,
        "why": "an independent owner run needs the governed book, which CI does "
               "not have. The receipt fields required to reconcile each figure "
               "are recorded per case. Offline parity against fixtures is the "
               "separate, passing gate.",
    }
    _write(report, args.out, [bearer, profile])
    print(f"\n{report['bank_id']} = {report['verdict']}  {passed}/{len(cases)}")
    return 0 if report["verdict"] == "PASS" else 1


def _write(report: dict, path: str, secrets: list) -> None:
    cleaned = ra.scrub(report, [s for s in secrets if s])
    payload = json.dumps(cleaned, indent=1, sort_keys=True) + "\n"
    for secret in secrets:
        if secret and secret in payload:
            print("::error::the report contained a secret; refusing to write it")
            return
    Path(path).write_text(payload, encoding="utf-8")
    print(f"wrote {Path(path).name}")


if __name__ == "__main__":
    raise SystemExit(main())
