#!/usr/bin/env python3
"""Read the M01 and M03 evidence records the confirmation canary already wrote.

READ ONLY. It asks no question, calls no model, touches no MI endpoint, changes
no configuration and writes nothing to the service. It opens the evidence sink
`plan_serving_canary` wrote during run 34756499729 and projects two records that
already exist. The bank is SPENT and is neither re-scored nor re-read for
expectations.

WHY THE SINK AND NOT THE CANARY ARTEFACT. The artefact carries the scorer's
projection, and the question Objective A has to answer is precisely what that
projection drops: what the model EMITTED as `operation`, what normalisation did
to it, and what the compiler BOUND. The record carries all three —
`raw_model_payload.operation`, `candidate_intent.operation`,
`governed_plan.operation` — plus the normalisation trace under
`governed_plan.provenance.compiler_bindings.normalisation`.

THE ANSWER IS NOT INFERRED FROM THE ENGLISH. No wording is read to reach it; the
three structured slots are printed side by side and the compatibility tables are
evaluated against them.

WHAT IS REDACTED. Anything that could be loan-level, plus both credentials.
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
    postmortem_read as pr)

BANK = (_REPO_ROOT / "due_diligence/evidence/metric_delta_confirmation_canary"
        / "metric_delta_canary_bank_manifest.json")

#: The two cases under diagnosis, by id. Read from the SPENT bank for IDENTITY
#: only — no expectation is read and nothing is scored here.
WANTED = ("M01", "M03")


def project(record):
    """The operation chain, the perimeter that refused, and what ran."""
    plan = record.get("governed_plan") or {}
    prov = plan.get("provenance") or {}
    bindings = prov.get("compiler_bindings") or {}
    claims = prov.get("intent_claims") or {}
    intent = record.get("candidate_intent") or {}
    raw = record.get("raw_model_payload") or {}
    execution = record.get("execution") or {}
    receipt = execution.get("receipt") or {}
    return {
        "correlation_id": record.get("correlation_id"),
        "disposition": record.get("disposition"),
        # THE OPERATION CHAIN, three separate facts kept apart.
        "operation_chain": {
            "candidate_raw_model_payload": raw.get("operation"),
            "intent_claims_at_compile": claims.get("operation"),
            "candidate_intent_after_normalise": intent.get("operation"),
            "compiled_plan": plan.get("operation"),
        },
        "normalisation_applied": bindings.get("normalisation"),
        "change_form": {
            "raw_model_payload": raw.get("change_form"),
            "candidate_intent": intent.get("change_form"),
            "compiler_binding": bindings.get("change_form"),
        },
        "capability": {
            "raw_model_payload": raw.get("capability"),
            "candidate_intent": intent.get("capability"),
            "compiled_plan": plan.get("capability"),
        },
        "compile": {k: (record.get("compile_result") or {}).get(k)
                    for k in ("outcome", "plan_id", "reason_codes")},
        "period": pr.redact(plan.get("period")),
        "eligibility": record.get("eligibility"),
        "serving": record.get("serving"),
        "execution_attempted": execution.get("attempted"),
        "execution_why_not": execution.get("why_not"),
        "mode": receipt.get("mode") or receipt.get("workflow_mode"),
        "receipt": pr.redact(receipt),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evidence-path",
                        default="/home/LogFiles/mi-plan-shadow/evidence.jsonl")
    parser.add_argument("--client-id", default="ERE")
    parser.add_argument("--out", default=str(HERE / "record_projection.json"))
    args = parser.parse_args()

    profile = (os.environ.get("AZURE_MI_API_PUBLISH_PROFILE") or "").strip()
    if not profile:
        print("::error::AZURE_MI_API_PUBLISH_PROFILE is not set")
        return 2
    scm, user, password = ra.publish_profile_credentials(profile)
    sink = ra.Sink(scm, user, password, args.evidence_path)
    rows, detail = sink.records()
    if rows is None:
        print(f"::error::the sink cannot be read ({detail})")
        return 2

    bank = json.loads(BANK.read_bytes())
    wanted = {case["id"]: case["question"] for case in bank["cases"]
              if case["id"] in WANTED}

    out = {"recorded_at": datetime.now(timezone.utc).isoformat(),
           "reads_only": True, "live_model_calls": 0, "live_mi_query_calls": 0,
           "sink_detail": detail, "sink_records_seen": len(rows), "cases": {}}
    for case_id, question in wanted.items():
        found = [r for r in rows
                 if (r.get("request") or {}).get("question") == question]
        out["cases"][case_id] = {
            "question": question,
            "records_for_this_question": len(found),
            # The LAST record, because the confirmation canary asked once and a
            # duplicate would mean an earlier bank used the same wording — which
            # pre-registration already proved it did not.
            "projection": project(found[-1]) if found else None,
        }

    body = json.dumps(out, indent=2, sort_keys=True, default=str)
    for secret in (profile, password or ""):
        if secret and secret in body:
            print("::error::a credential survived into the projection")
            return 2
    Path(args.out).write_text(body)
    print(body)
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
