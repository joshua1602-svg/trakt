#!/usr/bin/env python3
"""Read the FOUR already-written evidence records for S03, S07, S08, S09.

READ ONLY, AND THAT IS THE WHOLE CONTRACT. It asks no question, calls no model,
touches no MI endpoint, changes no configuration and writes nothing to the
service. It opens the evidence sink `plan_serving_canary` already wrote during
run 34752378644 and projects four records that exist. The canary bank is SPENT
and is neither read for expectations nor re-scored here.

WHY THE SINK AND NOT THE ARTEFACT. The run's artefact carries the SCORER's
projection — the subset my scoring function chose to keep — and the questions
this post-mortem has to answer are precisely the ones that subset omits: what the
CandidateIntent said about time, what the compiler bound, which runtime executed,
and what the returned groups actually were. The sink carries the whole record.

WHAT IS REDACTED. Anything that could be loan-level, plus both credentials. The
projection keeps governed group keys, period keys and aggregates — which is what
a period comparison IS — and drops any list of records long enough or shaped
enough to be rows. Both secrets are scrubbed and the output is rescanned for them
before it is written.
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

#: The four cases under classification, by their pre-registered question text.
#: Read from the SPENT bank for identity only — no expectation is read, and
#: nothing here scores anything.
WANTED = ("S03", "S07", "S08", "S09")

#: Keys that may carry loan-level detail. Dropped wholesale rather than sampled.
_ROW_BEARING = frozenset({
    "rows", "records", "loans", "data", "frame", "df", "loan_identifier",
    "identifiers", "raw_rows", "sample", "samples", "evidence_rows"})

#: Above this, a list is presumed to be rows rather than governed groups.
_MAX_GROUP_ROWS = 40


def redact(node, *, depth: int = 0):
    """Governed groups, periods and aggregates in; anything row-shaped out."""
    if isinstance(node, dict):
        out = {}
        for key, value in node.items():
            if str(key).lower() in _ROW_BEARING:
                out[key] = f"<redacted: {type(value).__name__}>"
                continue
            out[key] = redact(value, depth=depth + 1)
        return out
    if isinstance(node, list):
        if len(node) > _MAX_GROUP_ROWS:
            return [f"<redacted: {len(node)} entries, presumed rows>"]
        return [redact(item, depth=depth + 1) for item in node]
    if isinstance(node, str) and len(node) > 4000:
        return node[:4000] + f"… <truncated {len(node)} chars>"
    return node


def project(record: dict) -> dict:
    """The seven blocks the post-mortem asks for, and nothing else."""
    compiler = record.get("compiler") or {}
    interpretation = record.get("interpretation") or {}
    model = record.get("model") or {}
    return {
        "correlation_id": record.get("correlation_id"),
        "request": redact(record.get("request") or {}),
        # The model's own words are kept ONLY as the semantic subset: the raw
        # payload is what the interpreter emitted and is the one place a
        # `time` key's PRESENCE can be read directly rather than through the
        # dataclass's construction defaults.
        "raw_model_payload": redact(model.get("raw_payload")),
        "model_id": model.get("model_id"),
        "candidate_intent": redact(interpretation.get("candidate_intent")),
        "compile_result": {
            "outcome": compiler.get("outcome"),
            "reason_codes": compiler.get("reason_codes"),
            "reasons": redact(compiler.get("reasons")),
            "plan_id": compiler.get("plan_id"),
        },
        "governed_plan": redact(compiler.get("plan")),
        "eligibility": redact(record.get("eligibility") or {}),
        "execution": redact(record.get("execution") or {}),
        "serving": redact(record.get("serving") or {}),
        "disposition": record.get("disposition"),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evidence-path",
                        default="/home/LogFiles/mi-plan-shadow/evidence.jsonl")
    parser.add_argument("--client-id", default="ERE")
    parser.add_argument("--out", default=str(HERE / "postmortem_projection.json"))
    args = parser.parse_args()

    profile = (os.environ.get("AZURE_MI_API_PUBLISH_PROFILE") or "").strip()
    bearer = (os.environ.get("MI_BEARER") or "").strip()
    if not profile:
        print("::error::AZURE_MI_API_PUBLISH_PROFILE is not set; the sink "
              "cannot be read")
        return 2

    bank = json.loads((HERE / "serving_canary_bank_manifest.json").read_bytes())
    questions = {case["id"]: case["question"] for case in bank["cases"]
                 if case["id"] in WANTED}

    scm, user, password = ra.publish_profile_credentials(profile)
    sink = ra.Sink(scm, user, password, args.evidence_path)
    rows, detail = sink.records()
    if rows is None:
        print(f"::error::the sink could not be read: {detail}")
        return 2

    report = {
        "recorded_at": datetime.now(timezone.utc).isoformat(),
        "reads_only": True,
        "live_model_calls": 0,
        "live_mi_query_calls": 0,
        "sink_detail": detail,
        "sink_records_seen": len(rows),
        "cases": {},
    }

    for case_id, question in sorted(questions.items()):
        # LAST match, because the canary asked each question once and any older
        # record for the same text would be a different run.
        matches = [row for row in rows
                   if (row.get("request") or {}).get("question") == question]
        if not matches:
            report["cases"][case_id] = {"question": question,
                                        "record_found": False}
            print(f"{case_id}: NO RECORD for this question")
            continue
        projection = project(matches[-1])
        projection["question"] = question
        projection["record_found"] = True
        projection["duplicate_records_for_this_question"] = len(matches) - 1
        report["cases"][case_id] = projection
        print(f"{case_id}: record {projection['correlation_id']} "
              f"({len(matches)} match(es))")

    payload = json.dumps(report, indent=1, sort_keys=True, default=str) + "\n"
    for secret in (profile, bearer):
        if secret and secret in payload:
            print("::error::the projection contained a secret; refusing to write")
            return 2
    Path(args.out).write_text(payload, encoding="utf-8")
    print(f"wrote {Path(args.out).name}  ({len(payload)} chars)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
