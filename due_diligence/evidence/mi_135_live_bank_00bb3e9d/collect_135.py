#!/usr/bin/env python3
"""Ask the frozen 135-question bank once each, and keep what came back.

THIS SCRIPT SCORES NOTHING. It asks, it waits for the governed evidence record,
and it writes both down. Judgement happens later, offline, in `score_135.py`,
against the SIGNED-OFF scorer (`interpretation_v2.equivalence`) rather than a
second rubric invented here — the historical run1..run8 baselines were produced
by that scorer, and a comparison against them is only meaningful if the same
owner does the scoring.

WHY THE SPLIT. This half runs on a bare GitHub runner with the standard library
alone. `mi_agent` reaches for yaml and pandas, and an acceptance harness that
imports the product has already failed this estate once, in the middle of a bank,
after live interpretations had been spent. So the bank arrives as JSON
(`questions.json`, materialised from the YAML with its sha256 recorded) and the
scorer stays where the dependencies are.

THE SPEND IS GUARDED. 135 live interpretations are expensive and a
misconfigured canary would waste every one of them. `--abort-after` questions are
asked first; if none of them was served by the governed path AND none matched the
canary principal, the run stops and says so instead of spending the rest. That
costs three calls to detect a wrong `MI_AGENT_PLAN_SERVE`, not a hundred and
thirty-five.

ONE QUESTION, ONE CALL. No repeats, no retries on a semantic result. A retry
happens only for an evidenced INFRASTRUCTURE failure — a transport error or an
HTTP status — and the original failure is kept in the evidence beside it.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))
for _helper in ("deployed_acceptance_0399a315", "slice1b_serving_canary",
                "mi_api_certification"):
    _p = _REPO / "due_diligence" / "evidence" / _helper
    if _p.is_dir() and str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

QUESTIONS = _HERE / "questions.json"
PRODUCT_SHA = "00bb3e9d8175a395b6643772379866d6bc6169eb"


def load_bank() -> Dict[str, Any]:
    body = json.loads(QUESTIONS.read_text(encoding="utf-8"))
    rows = body["questions"]
    if len(rows) != 135:
        raise SystemExit(f"the bank must be 135 questions; this one is {len(rows)}")
    if len({r["question"] for r in rows}) != 135:
        raise SystemExit("the bank carries a duplicate question; the evidence "
                         "poller matches on question text and could not tell "
                         "two identical questions apart")
    return body


def served_new(record: Optional[Mapping[str, Any]]) -> bool:
    return bool(((record or {}).get("serving") or {}).get("decision") == "NEW")


def principal_matched(record: Optional[Mapping[str, Any]]) -> bool:
    return ((record or {}).get("serving") or {}).get("principal_matched") is True


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="https://app.traktinfra.io/api")
    parser.add_argument("--path", default="/mi/query")
    parser.add_argument("--portfolio-id", default="ERE/2026-06-30")
    parser.add_argument("--expect-commit", default=PRODUCT_SHA)
    parser.add_argument("--evidence-path",
                        default="/home/LogFiles/mi-plan-shadow/evidence.jsonl")
    parser.add_argument("--poll-interval", type=float, default=3.0)
    parser.add_argument("--poll-timeout", type=float, default=180.0)
    parser.add_argument("--out", default="",
                        help="default: raw_records.json for a full run, and "
                             "raw_records_rerun.json for a targeted one")
    parser.add_argument("--only", default="", help="comma-separated question ids")
    parser.add_argument("--abort-after", type=int, default=3,
                        help="stop if none of the first N was served NEW and none "
                             "matched the canary principal")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args(argv)

    if args.self_test:
        return self_test()

    import run_acceptance as ra
    import run_serving_acceptance as s1b
    from certify_mi_api import _live_asker

    bank = load_bank()
    wanted = [t.strip() for t in args.only.split(",") if t.strip()]
    known = {r["question_id"] for r in bank["questions"]}
    unknown = [t for t in wanted if t not in known]
    if unknown:
        parser.error(f"unknown question id(s) {unknown}")
    cases = [r for r in bank["questions"]
             if not wanted or r["question_id"] in wanted]

    # A TARGETED RERUN MUST NOT OVERWRITE THE FULL COLLECTION. Both write an
    # evidence file and the workflow commits it, so a 31-question rerun sharing
    # the full run's filename would replace 135 records with 31 — destroying the
    # evidence it was meant to complete. Caught before it happened, by cancelling
    # a dispatched run; the filename is now decided by what was asked, not by a
    # flag the caller has to remember.
    out_path = args.out or ("raw_records_rerun.json" if wanted
                            else "raw_records.json")

    bearer = os.environ.get("MI_BEARER", "").strip()
    profile = os.environ.get("AZURE_MI_API_PUBLISH_PROFILE", "").strip()
    secrets = [s for s in (bearer, profile) if len(s) >= ra.MIN_SECRET_LENGTH]
    client_id = args.portfolio_id.split("/", 1)[0]

    report: Dict[str, Any] = {
        "what_this_is": "raw live evidence for the frozen 135 bank; unscored",
        "product_sha_under_test": args.expect_commit,
        "bank_path": bank["bank_path"], "bank_sha256": bank["bank_sha256"],
        "question_count": len(cases), "portfolio_id": args.portfolio_id,
        "stages": {}, "records": [],
    }

    def stop(stage: str, verdict: str, message: str) -> int:
        report["verdict"], report["stopped_at"] = verdict, stage
        report["stages"][stage] = message
        ra._save(report, out_path, secrets)
        print(f"::error::{stage}: {message}")
        return 2

    if not bearer or not profile:
        return stop("credentials", "NOT_EXECUTABLE",
                    "MI_BEARER and AZURE_MI_API_PUBLISH_PROFILE are both required")

    served = s1b.served_commit(args.base_url)
    report["stages"]["provenance"] = {"served_commit": served,
                                      "expected": args.expect_commit}
    if not served:
        return stop("provenance", "INCONCLUSIVE", "no build stamp at /health")
    if not served.startswith(args.expect_commit[:7]):
        return stop("provenance", "FAIL",
                    f"serving {served} — expected {args.expect_commit}; the bank "
                    f"is frozen against one build and will not run against another")
    print(f"PROVENANCE CONFIRMED  served={served}")

    try:
        scm_host, user, password = ra.publish_profile_credentials(profile)
    except Exception as exc:                                         # noqa: BLE001
        return stop("sink", "NOT_EXECUTABLE",
                    f"the publish profile could not be read: {type(exc).__name__}")
    sink = ra.Sink(scm_host, user, password, args.evidence_path)
    rows, detail = sink.records()
    if rows is None:
        return stop("sink", "NOT_EXECUTABLE", f"the evidence sink: {detail}")
    already = {r.get("correlation_id") for r in rows}
    print(f"evidence sink reachable ({len(rows)} existing records)")

    ask = _live_asker(args.base_url, args.path, portfolio_id=args.portfolio_id)
    live_calls = 0
    infra_retries: List[Dict[str, Any]] = []

    for index, case in enumerate(cases, start=1):
        question = case["question"]
        envelope = ask(question)
        live_calls += 1
        attempt: Dict[str, Any] = {}
        if envelope.get("__transport_error__"):
            # AN EVIDENCED INFRASTRUCTURE FAILURE, and the only thing that earns a
            # second call. The original is kept beside the retry, never replaced:
            # a harness that overwrites its own failures reports a cleaner run
            # than happened.
            attempt = {"transport_error": True,
                       "http_status": envelope.get("__http_status__"),
                       "answer": envelope.get("answer")}
            print(f"  [{index:3d}/{len(cases)}] {case['question_id']}  "
                  f"INFRA {attempt.get('http_status')} — retrying once")
            time.sleep(5.0)
            envelope = ask(question)
            live_calls += 1
            infra_retries.append({"question_id": case["question_id"],
                                  "original_failure": attempt})

        if envelope.get("__transport_error__"):
            # NOTHING TO WAIT FOR. A request rejected in the transport never
            # reached the interpreter, so no governed record will ever be
            # written for it. Polling anyway spent the full timeout on every
            # one of them: thirty-one questions rejected at authentication took
            # fifty-one minutes to fail instead of two.
            record, matched = None, ("no record is possible: the request failed "
                                     "in the transport")
        else:
            record, matched = ra.poll_for(sink, question, client_id,
                                          interval=args.poll_interval,
                                          timeout=args.poll_timeout,
                                          already=already)
        report["records"].append({
            "question_id": case["question_id"],
            "canonical_id": case["canonical_id"], "variant": case["variant"],
            "original_category": case["original_category"],
            "shape": case["shape"], "origin_bank": case["origin_bank"],
            "question": question,
            "retry_of_infrastructure_failure": attempt or None,
            "envelope": envelope,
            "record": record,
            "record_matched": matched,
        })
        decision = ((record or {}).get("serving") or {}).get("decision") or "NO_RECORD"
        disposition = (record or {}).get("disposition") or ""
        print(f"  [{index:3d}/{len(cases)}] {case['question_id']:6s} "
              f"{decision:16s} {disposition:22s} {question[:46]}")

        if index == args.abort_after and not wanted:
            head = report["records"]
            if not any(served_new(r["record"]) for r in head) and \
               not any(principal_matched(r["record"]) for r in head):
                report["live_model_calls"] = live_calls
                report["infra_retries"] = infra_retries
                return stop(
                    "canary", "NOT_EXECUTABLE",
                    f"none of the first {args.abort_after} questions was served by "
                    f"the governed path and none matched the canary principal — "
                    f"MI_AGENT_PLAN_SERVE / MI_AGENT_PLAN_SERVE_PRINCIPALS are not "
                    f"in the state this bank requires. Stopped after "
                    f"{live_calls} live calls instead of spending {len(cases)}.")
            print(f"  canary guard passed after {args.abort_after} questions")

    report["live_model_calls"] = live_calls
    report["infra_retries"] = infra_retries
    report["verdict"] = "COLLECTED"
    ra._save(report, out_path, secrets)
    served_count = sum(1 for r in report["records"] if served_new(r["record"]))
    missing = sum(1 for r in report["records"] if r["record"] is None)
    print(f"\n  COLLECTED  questions={len(cases)}  live_calls={live_calls}  "
          f"infra_retries={len(infra_retries)}  served_new={served_count}  "
          f"records_missing={missing}")
    print(f"  written {out_path}")
    return 0


def self_test() -> int:
    """Offline rules. No network, no model, standard library only."""
    failures: List[str] = []

    def check(name: str, condition: bool) -> None:
        if not condition:
            failures.append(name)
        print(f"   {'ok  ' if condition else 'FAIL'} {name}")

    bank = load_bank()
    rows = bank["questions"]
    check("the bank is exactly 135 questions", len(rows) == 135)
    check("every question is unique", len({r["question"] for r in rows}) == 135)
    check("every question id is unique",
          len({r["question_id"] for r in rows}) == 135)
    check("45 canonicals, three variants each",
          len({r["canonical_id"] for r in rows}) == 45
          and all(len([x for x in rows if x["canonical_id"] == c]) == 3
                  for c in {r["canonical_id"] for r in rows}))

    # THE BANK IS VERBATIM. When yaml is available the JSON is re-derived from
    # the committed YAML and must match string for string; the sha256 is checked
    # either way. This is the rule that stops the questions being "improved".
    import hashlib
    src = _REPO / bank["bank_path"]
    raw = src.read_bytes()
    check("the source bank is the one this JSON was cut from",
          hashlib.sha256(raw).hexdigest() == bank["bank_sha256"])
    try:
        import yaml
    except ImportError:
        print("   skip  verbatim re-derivation (no yaml on this interpreter)")
    else:
        y = yaml.safe_load(raw)
        expect = [(v["source_id"], v["question"])
                  for c in y["canonicals"] for v in c["variants"]]
        got = [(r["question_id"], r["question"]) for r in rows]
        check("every question string matches the YAML verbatim", expect == got)

    check("the product sha is pinned in the collector",
          PRODUCT_SHA == "00bb3e9d8175a395b6643772379866d6bc6169eb")

    # the guards that protect the spend
    check("a record with no serving block is not 'served NEW'",
          not served_new(None) and not served_new({}))
    check("a legacy fallback is not 'served NEW'",
          not served_new({"serving": {"decision": "LEGACY_FALLBACK"}}))
    check("NEW is recognised",
          served_new({"serving": {"decision": "NEW"}}))
    check("principal_matched is true only when it is literally true",
          principal_matched({"serving": {"principal_matched": True}})
          and not principal_matched({"serving": {"principal_matched": "yes"}})
          and not principal_matched(None))

    print(f"\n  {len(failures)} failing rule(s)" if failures
          else "\n  every rule holds")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
