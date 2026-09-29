#!/usr/bin/env python3
"""Read back the governed plan the canary recorded for each production-bank question.

READ ONLY, AND THAT IS THE WHOLE CONTRACT. It asks no question, calls no model,
touches no MI endpoint, changes no configuration and writes nothing to the
service. It opens the evidence sink `plan_serving_canary` writes on every request
it handles — `/home/LogFiles/mi-plan-shadow/evidence.jsonl` on the App Service —
and projects, for each question in `qb_questions.json`, the most recent record
of it.

WHY. The owner's production run (qb_full.txt) records answers, not plans, and
the next P0 steps depend on how the model encoded particular questions — which
capability, which operation, which period form. Those plans were written to the
sink when the questions were asked; this reads them instead of asking again.

WHAT IS REDACTED. The post-mortem reader's rules, imported rather than copied:
row-bearing keys are dropped wholesale, lists long enough to be rows are
presumed rows, long strings are truncated. The publish profile is scrubbed and
the output is rescanned for it before anything is written.

ONE TEXT, TWO IDS. "Show pipeline amount by product." is asked twice in the
bank (pipeline_020, pipeline_strat_001). Ids sharing a text are given that
text's most recent records, oldest first, in bank order; with fewer records
than ids, the earliest ids are reported as having none.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import OrderedDict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

HERE = Path(__file__).resolve().parent
_REPO_ROOT = HERE.parents[2]
sys.path.insert(0, str(_REPO_ROOT))

from due_diligence.evidence.change_intelligence_serving_canary.postmortem_read import (  # noqa: E402
    project, redact)
from due_diligence.evidence.deployed_acceptance_0399a315 import (  # noqa: E402
    run_acceptance as ra)


def load_bank(path: Path) -> List[Dict[str, Any]]:
    return list(json.loads(path.read_text(encoding="utf-8"))["questions"])


def summarise(record: Dict[str, Any]) -> Dict[str, Any]:
    """The one-line reading of a record: what the plan asked and what served."""
    compiler = record.get("compiler") or {}
    plan = compiler.get("plan") or {}
    output = (plan.get("outputs") or [{}])[0]
    period = plan.get("period") or {}
    serving = record.get("serving") or {}
    eligibility = record.get("eligibility") or {}
    filters = list(plan.get("filters") or ()) + list(output.get("filters") or ())
    # WHY THE LANGUAGE STEP FAILED, when it did: the interpreter's own reason
    # (code, subject, the client's error text). Without it an interpreter
    # failure reads the same whether the model timed out, was refused by the
    # provider, or emitted an intent the parser rejected.
    failure = (record.get("model") or {}).get("failure") or None
    usage = (record.get("model") or {}).get("usage") or {}
    return {
        # WHAT THE MODEL WAS SHOWN. 2.3.0 is the first vocabulary that defines
        # the forecast measures, so a comparison across runs must be able to
        # tell which vocabulary each reading was made against.
        "vocabulary_version": (plan.get("provenance") or {}).get(
            "vocabulary_version"),
        "compile_outcome": compiler.get("outcome"),
        "reason_codes": compiler.get("reason_codes"),
        "capability": plan.get("capability"),
        "operation": plan.get("operation"),
        "change_form": ((plan.get("provenance") or {}).get("intent_claims")
                        or {}).get("change_form"),
        "population_base": (plan.get("population") or {}).get("base"),
        "period": {k: period.get(k) for k in
                   ("form", "labels", "grain", "periods_back")},
        "measures": [m.get("concept") for m in output.get("measures") or ()],
        "dimensions": [d.get("canonical_field") or d.get("concept")
                       for d in output.get("dimensions") or ()],
        "filters": [[f.get("canonical_field") or f.get("concept"),
                     f.get("comparator"), f.get("value")] for f in filters],
        "target": plan.get("target"),
        "eligibility_reason": eligibility.get("reason"),
        "serving_decision": serving.get("decision"),
        "serving_reason": serving.get("reason"),
        "interpreter_failure": ({k: str(failure.get(k) or "")[:300]
                                 for k in ("code", "subject", "detail")}
                                if isinstance(failure, dict) else None),
        "model_usage": {k: usage.get(k) for k in
                        ("input_tokens", "output_tokens",
                         "cache_read_input_tokens",
                         "cache_creation_input_tokens", "model_calls",
                         "model_ms")
                        if k in usage},
    }


def why_lines(record: Dict[str, Any], summary: Dict[str, Any]) -> List[str]:
    """The reading and the reason a question was not served, in a few short
    lines: the plan's measures, axes and filters, the perimeter's own detail,
    and any ambiguity the model marked blocking. Plan-level fields only — no
    row, value or borrower detail — and redacted like the projection."""
    lines = [f"reading: measures={summary.get('measures')} "
             f"dimensions={summary.get('dimensions')} "
             f"filters={summary.get('filters')}"]
    failure = summary.get("interpreter_failure")
    if failure:
        lines.append(f"interpreter: {failure.get('code')}: "
                     f"{failure.get('subject')}: {failure.get('detail')}")
    detail = str((record.get("eligibility") or {}).get("detail") or "")
    if detail:
        lines.append(f"why: {detail[:300]}")
    payload = (record.get("raw_model_payload") or record.get("candidate_intent")
               or {})
    for item in (payload.get("ambiguity") or ()):
        if isinstance(item, dict) and item.get("blocking"):
            lines.append(f"clarify: {item.get('slot')}: "
                         f"{str(item.get('note') or '')[:300]}")
    return [str(redact(line)) for line in lines]


def select(rows: List[Dict[str, Any]], bank: List[Dict[str, Any]], *,
           since: Optional[str]) -> Dict[str, Optional[Dict[str, Any]]]:
    """`{question id: record or None}`, per the rule in the module docstring."""
    by_text: "OrderedDict[str, List[Dict[str, Any]]]" = OrderedDict()
    for case in bank:
        by_text.setdefault(case["question"], [])
    for row in rows:
        text = (row.get("request") or {}).get("question")
        if text not in by_text:
            continue
        if since and str(row.get("recorded_at") or "") < since:
            continue
        by_text[text].append(row)
    chosen: Dict[str, Optional[Dict[str, Any]]] = {}
    for text, records in by_text.items():
        records.sort(key=lambda r: str(r.get("recorded_at") or ""))
        ids = [case["id"] for case in bank if case["question"] == text]
        latest = records[-len(ids):] if records else []
        padded = [None] * (len(ids) - len(latest)) + latest
        for case_id, record in zip(ids, padded):
            chosen[case_id] = record
    return chosen


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence-path",
                        default="/home/LogFiles/mi-plan-shadow/evidence.jsonl")
    parser.add_argument("--bank", default=str(HERE / "qb_questions.json"))
    parser.add_argument("--since", default="",
                        help="ISO timestamp; ignore records written before it")
    parser.add_argument("--request", default="",
                        help="a READBACK_REQUEST.json whose `since` applies when "
                             "--since is empty")
    parser.add_argument("--out", default="qb_plan_readback.json")
    args = parser.parse_args(argv)

    profile = (os.environ.get("AZURE_MI_API_PUBLISH_PROFILE") or "").strip()
    if not profile:
        print("::error::AZURE_MI_API_PUBLISH_PROFILE is not set; the sink "
              "cannot be read")
        return 2
    if not args.since and args.request and Path(args.request).exists():
        request = json.loads(Path(args.request).read_text(encoding="utf-8"))
        args.since = str(request.get("since") or "")
        print(f"request: since={args.since or '(none)'} "
              f"label={request.get('label')!r}")
    bank = load_bank(Path(args.bank))
    scm, user, password = ra.publish_profile_credentials(profile)
    sink = ra.Sink(scm, user, password, args.evidence_path)
    rows, detail = sink.records()
    if rows is None:
        print(f"::error::the sink could not be read: {detail}")
        return 2

    chosen = select(rows, bank, since=args.since or None)
    report: Dict[str, Any] = {
        "recorded_at": datetime.now(timezone.utc).isoformat(),
        "reads_only": True, "live_model_calls": 0, "live_mi_query_calls": 0,
        "sink_detail": detail, "sink_records_seen": len(rows),
        "since": args.since or None,
        "found": sum(1 for r in chosen.values() if r is not None),
        "question_count": len(bank),
        "cases": {},
    }
    print(f"sink: {detail}; {len(rows)} record(s); "
          f"{report['found']} of {len(bank)} questions have one")
    for case in bank:
        record = chosen.get(case["id"])
        entry: Dict[str, Any] = {"n": case["n"], "category": case["category"],
                                 "question": case["question"],
                                 "record_found": record is not None}
        if record is not None:
            entry["recorded_at"] = record.get("recorded_at")
            entry["summary"] = redact(summarise(record))
            entry["record"] = project(record)
            s = entry["summary"]
            print(f"[{case['n']:>3}] {case['id']:<22} {s['capability']}/"
                  f"{s['operation']} base={s['population_base']} "
                  f"period={s['period']['form']} -> {s['serving_decision']} "
                  f"{s['serving_reason'] or ''}")
            if s["serving_decision"] != "NEW":
                # WHY IT FELL, readable in the job log itself — the artifact is
                # not always reachable from where the readback is read.
                for line in why_lines(record, s):
                    print(f"      {line}")
        else:
            print(f"[{case['n']:>3}] {case['id']:<22} NO RECORD")
        report["cases"][case["id"]] = entry

    payload = json.dumps(report, indent=1, sort_keys=True, default=str) + "\n"
    if profile in payload or password in payload:
        print("::error::the projection contained a credential; refusing to write")
        return 2
    Path(args.out).write_text(payload, encoding="utf-8")
    print(f"wrote {args.out} ({len(payload)} chars)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
