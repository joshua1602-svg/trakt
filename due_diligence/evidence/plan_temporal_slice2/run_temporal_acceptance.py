#!/usr/bin/env python3
"""Slice 2 live temporal acceptance — did the NEW temporal result become the answer?

WHY THIS EXISTS AND THE SLICE 1 HARNESS DOES NOT SUFFICE.
`slice1b_serving_canary/run_serving_acceptance.py` reads `serving.decision`
correctly, but its bank is `[c for c in manifest["cases"] if
c["expected_slice1_eligible"]][:6]` from a sha256-pinned SLICE 1 manifest with no
bank parameter — it cannot ask a temporal question, and swapping the manifest
would break the integrity gate that is the point of pinning it.
`mi_api_certification/time_dimension_acceptance.py` asks a different fixed matrix
and contains no reference to `serving` or to `snapshot`, so it can tell neither
NEW from LEGACY nor which periods were selected. Neither reconciles a per-period
value, a grouped cell or a comparison against the execution receipt.

WHAT THIS ADDS, AND ONLY THIS: the six-question temporal bank, and the READING
of a temporal answer. Everything else is imported:

    transport / MI_BEARER    certify_mi_api._live_asker
    zero-cost provenance     run_serving_acceptance.served_commit  (GET /health)
    HTTP figure extraction   run_serving_acceptance.response_figures (pinned to)
    Kudu evidence sink       run_acceptance.Sink
    correlation-aware poll   run_acceptance.poll_for
    publish-profile creds    run_acceptance.publish_profile_credentials
    model-id gate            run_acceptance.is_required_model
    secret hygiene           run_acceptance.scrub / _save / MIN_SECRET_LENGTH

THE EXECUTION RECEIPT IS THE NUMERIC ORACLE. No portfolio figure is written
down here. Each case asserts that what the CALLER received equals what the
deterministic runtime recorded having computed, period by period and cell by
cell. A harness carrying its own expected balances would go stale the moment the
book moved and would be testing its own memory rather than the service.

ORDER, AND WHY. The served commit is confirmed FIRST by GET /health — a POST
would cost a live Opus interpretation against a canary that is switched on, and
a provenance mismatch must cost nothing. Then the sink is proved readable and
its existing correlation ids are recorded, so no pre-existing record can satisfy
a case. Only then is the bank asked, once each.

IT DEPLOYS NOTHING AND CONFIGURES NOTHING. `MI_AGENT_PLAN_SERVE` and its
principal allow-list are App Service settings needing ARM; this repository's
automation holds a publish profile, which grants Kudu and nothing else. The
canary must already be on when this runs.

    --self-test    exercises every reconciliation rule against synthetic records
                   and makes no network call of any kind. Stdlib only, so it runs
                   on the bare runner. Run it before pointing this at production.

    --shape-check  drives the REAL `plan_serving_canary.serve` offline (recorded
                   Opus payloads, replayed) and feeds the records and envelopes
                   the PRODUCT writes through this file's own adjudicator. No
                   network call and no model call. It answers the one question
                   the self-test cannot — whether the harness reads the shape the
                   product actually emits, rather than the shape it assumes —
                   and needs the product's dependencies, so it is not run in CI.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))
for _sibling in ("deployed_acceptance_0399a315", "slice1b_serving_canary",
                 "mi_api_certification"):
    _path = str(_REPO / "due_diligence" / "evidence" / _sibling)
    if _path not in sys.path:
        sys.path.insert(0, _path)

import run_acceptance as ra                                          # noqa: E402
import run_serving_acceptance as s1b                                 # noqa: E402
from certify_mi_api import _live_asker                               # noqa: E402

#: Agreement tolerance, the same 0.01 every other adjudicator in this estate uses.
TOLERANCE = 0.01

#: THE GOVERNED CATALOGUE THIS ACCEPTANCE RUNS AGAINST, as production holds it.
#:
#: Used for ONE thing: asserting WHICH periods were selected. No figure is taken
#: from it and none is written down anywhere in this file — the execution receipt
#: remains the numeric oracle. Snapshot IDENTITY is a different kind of claim
#: from a portfolio VALUE: it is a property of the catalogue rather than of the
#: book, it is stable between runs, and a harness that cannot say which periods
#: were expected cannot tell a correct selection from a fabricated one.
#:
#: The book is SPARSE — three reporting periods, with December 2025 to May 2026
#: absent. That is the point of asserting it: an answer that reports those six
#: months has invented them.
PRODUCTION_CATALOGUE: Tuple[str, ...] = ("2025-10-31", "2025-11-30", "2026-06-30")

#: The bank. FIXED — six questions, asked once each, never reworded and never
#: extended at run time. A bank a harness can edit is a bank that can be made to
#: pass.
#:
#: S2-P2 WAS CORRECTED AFTER THE FIRST PRODUCTION RUN. It asked for "the last 6
#: months", which this catalogue cannot answer: a six-month relative window ends
#: at 2026-06-30 and needs 2026-01-31 through 2026-05-31, none of which exist.
#: The live refusal was CORRECT, and the case was measuring the bank's own
#: assumption rather than the product. It now states an explicit range whose
#: bounds the catalogue does carry, so what it tests is what it was always meant
#: to test: that a bounded range returns the governed periods inside it and does
#: not fabricate the gap.
BANK: Tuple[Dict[str, Any], ...] = (
    {"case_id": "S2-P1", "kind": "series",
     "question": "What was funded balance each month?",
     "why": "whole available monthly series — grain-only resolution",
     "expect_snapshots": PRODUCTION_CATALOGUE},
    {"case_id": "S2-P2", "kind": "series",
     "question": "Show funded balance from October 2025 to June 2026.",
     "why": "a bounded explicit range returns the governed periods inside it "
            "and fabricates nothing for the months the book does not carry",
     "expect_snapshots": PRODUCTION_CATALOGUE},
    {"case_id": "S2-P3", "kind": "series",
     "question": "How many drawdown loans were there each month?",
     "why": "a governed filter survives every selected snapshot"},
    {"case_id": "S2-P4", "kind": "grouped",
     "question": "Show loan count by LTV bucket each month.",
     "why": "time x an existing governed dimension"},
    {"case_id": "S2-P5", "kind": "comparison",
     "question": "What was funded balance this month versus last month?",
     "why": "deterministic previous-period resolution and comparison"},
    {"case_id": "S2-N1", "kind": "negative",
     "question": "Show funded balance over the last 24 months.",
     "why": "history the governed catalogue does not contain — must fail closed"},
)

SERVED_NEW = "NEW"
REPORTING_DATE = "reporting_date"


# --------------------------------------------------------------------------- #
# reading a temporal record
# --------------------------------------------------------------------------- #

def temporal_block(record: Mapping[str, Any]) -> Dict[str, Any]:
    return dict(((record.get("execution") or {}).get("temporal") or {}))


#: `plan_runtime_adapter.value_column`, restated. NOT imported: this harness
#: runs on a bare runner with no `pip install`, and importing the adapter pulls
#: pandas in. A restatement can drift from its original, so `self_test` reads the
#: adapter's SOURCE and fails if either naming rule has moved.
_ROW_COUNT_FIELD = "loan_count"
_COUNT = "count"


def value_column(record: Mapping[str, Any]) -> str:
    """The column the executor named, derived from the spec it actually bound.

    Not guessed from the question and not hard-coded: `bound_spec` is what the
    accepted adapter produced, and `loan_count` is what the executor calls a row
    count. The adapter is the owner of that naming; this mirrors it exactly and
    is pinned to it by the drift guard in `self_test`.
    """
    spec = (record.get("execution") or {}).get("bound_spec") or {}
    if spec.get("aggregation") == _COUNT:
        return _ROW_COUNT_FIELD
    return f"{spec.get('metric')}_{spec.get('aggregation')}"


def http_rows(envelope: Mapping[str, Any]) -> List[Dict[str, Any]]:
    """The temporal rows as the CALLER received them.

    The rule the accepted end-to-end proof uses
    (`temporal_serving_integration.payload_series`): a grid row keyed by a
    reporting date, read out of the artefacts the response ships rather than out
    of the runtime's own object — a figure that reconciles inside the engine and
    is lost on the way out is the failure this exists to catch. It is widened
    here from that proof's FIRST matching artefact to EVERY artefact, so a
    two-artefact envelope whose series is not first cannot read as zero rows and
    report every period as dropped.

    `s1b.response_figures` is the estate's owner of "what figures did the
    response carry" and stays pinned to this reading by `self_test`, which
    asserts the two agree on a single-grid envelope. It is not merged into the
    sweep: it returns COPIES of the rows, so a merge cannot tell its rows from
    the originals and silently doubles every period — which is what the
    self-test caught when it was written that way.
    """
    rows: List[Dict[str, Any]] = []
    for artefact in (envelope.get("artifacts") or ()):
        for row in (artefact.get("rows") or artefact.get("data") or ()):
            if isinstance(row, Mapping) and REPORTING_DATE in row:
                rows.append(dict(row))
    return rows


def _close(left: Any, right: Any) -> bool:
    if left is None or right is None:
        return left is right
    try:
        return abs(float(left) - float(right)) <= TOLERANCE
    except (TypeError, ValueError):
        return False


# --------------------------------------------------------------------------- #
# the three reconciliations
# --------------------------------------------------------------------------- #

def reconcile_series(record: Mapping[str, Any], rows: Sequence[Mapping[str, Any]]
                     ) -> Tuple[List[str], int]:
    """Every period the runtime recorded must reach the caller with its figure."""
    problems: List[str] = []
    column = value_column(record)
    points = list(temporal_block(record).get("points") or ())
    if not points:
        return ["the execution evidence records no period"], 0
    checked = 0
    for point in points:
        date = str(point.get(REPORTING_DATE))
        served = [r for r in rows if str(r.get(REPORTING_DATE)) == date]
        if len(served) != 1:
            problems.append(f"{date}: {len(served)} rows served, expected 1")
            continue
        checked += 1
        if not _close(served[0].get(column), point.get("value")):
            problems.append(f"{date}: served {served[0].get(column)!r} != "
                            f"executed {point.get('value')!r}")
    extra = {str(r.get(REPORTING_DATE)) for r in rows} - {
        str(p.get(REPORTING_DATE)) for p in points}
    if extra:
        problems.append(f"the response carries periods the runtime did not "
                        f"execute: {sorted(extra)}")
    return problems, checked


def reconcile_grouped(record: Mapping[str, Any], rows: Sequence[Mapping[str, Any]]
                      ) -> Tuple[List[str], int]:
    """Every period/dimension cell, both ways — no missing cell and no extra one."""
    problems: List[str] = []
    column = value_column(record)
    spec = (record.get("execution") or {}).get("bound_spec") or {}
    dimensions = [str(d) for d in (spec.get("dimensions") or ())]
    if not dimensions:
        return ["the bound spec names no grouping dimension"], 0
    points = list(temporal_block(record).get("points") or ())
    if not points:
        return ["the execution evidence records no period"], 0

    checked = 0
    for point in points:
        date = str(point.get(REPORTING_DATE))
        executed = {tuple(str(cell.get(d)) for d in dimensions): cell.get("value")
                    for cell in (point.get("cells") or ())}
        served = {tuple(str(r.get(d)) for d in dimensions): r.get(column)
                  for r in rows if str(r.get(REPORTING_DATE)) == date}
        if set(served) != set(executed):
            problems.append(
                f"{date}: cell keys differ — served only "
                f"{sorted(set(served) - set(executed))}, executed only "
                f"{sorted(set(executed) - set(served))}")
        for key, figure in executed.items():
            checked += 1
            if not _close(served.get(key), figure):
                problems.append(f"{date} {key}: served {served.get(key)!r} != "
                                f"executed {figure!r}")
    return problems, checked


def reconcile_comparison(record: Mapping[str, Any],
                         rows: Sequence[Mapping[str, Any]]
                         ) -> Tuple[List[str], int]:
    """Both source periods, and the change the runtime computed from them."""
    problems: List[str] = []
    comparison = temporal_block(record).get("comparison") or {}
    if not comparison:
        return ["the execution evidence records no comparison"], 0

    series_problems, checked = reconcile_series(record, rows)
    problems.extend(series_problems)

    baseline = comparison.get("baseline_value")
    current = comparison.get("current_value")
    if baseline is None or current is None:
        problems.append("the comparison names no baseline or no current value")
        return problems, checked

    # The runtime's own arithmetic, re-derived from the two figures it reported.
    # This is not a second calculation owner: it checks that the change the
    # receipt carries is the change its own two values imply.
    expected_change = float(current) - float(baseline)
    if not _close(comparison.get("absolute_change"), expected_change):
        problems.append(f"absolute change {comparison.get('absolute_change')!r} "
                        f"does not follow from {baseline!r} -> {current!r}")
    if float(baseline) != 0.0:
        expected_pct = expected_change / float(baseline) * 100.0
        if comparison.get("percent_change") is None or abs(
                float(comparison["percent_change"]) - expected_pct) > 1e-6:
            problems.append(
                f"percent change {comparison.get('percent_change')!r} does not "
                f"follow from {baseline!r} -> {current!r}")
    elif comparison.get("percent_change") is not None:
        problems.append("a zero baseline reported a percentage change")
    return problems, checked + 2


def classify_negative(record: Mapping[str, Any],
                      rows: Sequence[Mapping[str, Any]]) -> List[str]:
    """The 24-month control. It passes by failing closed, and only that way.

    The two facts that decide it are that the temporal result was NOT served and
    that the runtime executed NO period — a narrowed window would show up as
    points the catalogue could honour, which is the substitution this control
    exists to catch.

    WHAT IS DELIBERATELY NOT ASSERTED: what the LEGACY envelope contains. Once
    the decision is a fallback, the caller is holding the answer the legacy path
    has always given, and whether that path ships dated rows of its own is not a
    slice 2 property and was not changed by slice 2. Failing this control on
    those rows would report a legacy behaviour as a temporal defect. The rows ARE
    a problem when the decision is NEW, which is the case above.
    """
    problems: List[str] = []
    serving = record.get("serving") or {}
    block = temporal_block(record)
    if serving.get("decision") == SERVED_NEW:
        problems.append("a temporal answer was SERVED for history the "
                        "catalogue does not contain")
        if rows:
            problems.append(f"and it carried {len(rows)} temporal rows")
    points = list(block.get("points") or ())
    if points:
        problems.append(f"the runtime executed {len(points)} periods for an "
                        f"unavailable 24-month window — a shortened series")
    reason = str(serving.get("reason") or "")
    if not reason:
        problems.append("no reason was recorded for the fallback")
    return problems


# --------------------------------------------------------------------------- #
# one case
# --------------------------------------------------------------------------- #

def adjudicate(case: Mapping[str, Any], envelope: Mapping[str, Any],
               record: Optional[Mapping[str, Any]],
               client_id: str = "",
               catalogue: Sequence[str] = PRODUCTION_CATALOGUE
               ) -> Dict[str, Any]:
    """Everything asserted about one case, from the response and the record."""
    out: Dict[str, Any] = {
        "case_id": case["case_id"], "question": case["question"],
        "kind": case["kind"], "why": case["why"], "problems": [],
        "values_reconciled": 0,
    }
    problems: List[str] = out["problems"]

    if envelope.get("__transport_error__"):
        problems.append(f"transport: HTTP {envelope.get('__http_status__')} "
                        f"{envelope.get('answer')}")
        out["verdict"] = "FAIL"
        return out
    if record is None:
        problems.append("no NEW evidence record matched this request")
        out["verdict"] = "FAIL"
        return out

    serving = record.get("serving") or {}
    eligibility = record.get("eligibility") or {}
    block = temporal_block(record)
    rows = http_rows(envelope)

    out.update({
        "http_ok": bool(envelope.get("ok")),
        "model_id": (record.get("model") or {}).get("model_id"),
        "principal_matched": bool(serving.get("principal_matched")),
        "eligible": bool(eligibility.get("eligible")),
        "perimeter": eligibility.get("perimeter"),
        "decision": serving.get("decision"),
        "response_served_from": serving.get("response_served_from"),
        "reason": serving.get("reason"),
        "shape": block.get("shape"), "basis": block.get("basis"),
        "snapshots": [p.get(REPORTING_DATE)
                      for p in (block.get("points") or ())],
        "snapshot_ids": [p.get("snapshot_id")
                         for p in (block.get("points") or ())],
        "http_rows": len(rows),
        "semantic_coverage_refused": envelope.get("semanticCoverageRefused"),
    })

    if not ra.is_required_model(out["model_id"]):
        problems.append(f"model_id {out['model_id']!r} is not the required model")

    if case["kind"] == "negative":
        problems.extend(classify_negative(record, rows))
        out["verdict"] = "PASS" if not problems else "FAIL"
        return out

    # -- positives ---------------------------------------------------------- #
    if not out["http_ok"]:
        problems.append(f"the response was not ok: {envelope.get('error')!r}")
    if not out["principal_matched"]:
        problems.append("the canary principal did not match")
    if not out["eligible"]:
        problems.append(f"ineligible: {eligibility.get('reason')!r}")
    if out["perimeter"] != "slice2_temporal":
        problems.append(f"perimeter {out['perimeter']!r} — the temporal runtime "
                        f"was not the one that answered")
    if out["decision"] != SERVED_NEW:
        problems.append(f"serving.decision = {out['decision']!r}")
    if out["response_served_from"] != SERVED_NEW:
        problems.append(f"response_served_from = {out['response_served_from']!r}")
    if envelope.get("semanticCoverageRefused") is True:
        problems.append("the coverage gate refused a served temporal result")
    if not out["snapshots"]:
        problems.append("no snapshot was selected")
    if any(not sid for sid in out["snapshot_ids"]):
        problems.append("a selected period carries no snapshot identity")

    # -- WHICH periods, not how many ---------------------------------------- #
    #
    # NO PERIOD MAY BE INVENTED. Every selected reporting date must be one the
    # governed catalogue actually holds. This is the assertion that separates
    # "returned the three periods the book has" from "returned a contiguous
    # nine-month series", and the latter is the failure the sparse catalogue
    # exists to expose here.
    # THE CATALOGUE IS A PARAMETER, not a global, so this one rule serves
    # whichever book the run is pointed at — production here, the fixture's own
    # history under `--shape-check`. An empty catalogue means "not asserted".
    selected = [str(d) for d in (out["snapshots"] or ())]
    known = [str(d) for d in (catalogue or ())]
    invented = [d for d in selected if d not in known] if known else []
    if invented:
        problems.append(f"period(s) the governed catalogue does not hold: "
                        f"{invented} — the catalogue is {known}")

    # NO OTHER CLIENT'S RUN. Applied to the production identity form `client/run`
    # only: an id carrying no client segment states nothing about ownership, and
    # reading the whole id as a client name would invent a finding.
    foreign = sorted({str(sid).split("/", 1)[0] for sid in out["snapshot_ids"]
                      if sid and "/" in str(sid)
                      and str(sid).split("/", 1)[0] != client_id})
    if client_id and foreign:
        problems.append(f"snapshot(s) belonging to another client: {foreign}")

    # And for the cases whose whole point is WHICH periods a window names.
    expected = case.get("expect_snapshots")
    if expected is not None and selected != list(expected):
        problems.append(f"selected {selected} — this window names "
                        f"{list(expected)}")
    if (record.get("execution") or {}).get("error"):
        problems.append(f"execution error: "
                        f"{(record.get('execution') or {}).get('error')!r}")

    reconcile = {"series": reconcile_series, "grouped": reconcile_grouped,
                 "comparison": reconcile_comparison}[case["kind"]]
    found, checked = reconcile(record, rows)
    problems.extend(found)
    out["values_reconciled"] = checked
    out["verdict"] = "PASS" if not problems else "FAIL"
    return out


# --------------------------------------------------------------------------- #
# the run
# --------------------------------------------------------------------------- #

def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="https://app.traktinfra.io/api")
    parser.add_argument("--path", default="/mi/query")
    parser.add_argument("--portfolio-id", default="ERE/2026-06-30")
    parser.add_argument("--expect-commit")
    parser.add_argument("--evidence-path",
                        default="/home/LogFiles/mi-plan-shadow/evidence.jsonl")
    parser.add_argument("--poll-interval", type=float, default=3.0)
    parser.add_argument("--poll-timeout", type=float, default=120.0)
    parser.add_argument("--json-out", default="slice2-temporal-acceptance.json")
    parser.add_argument("--self-test", action="store_true",
                        help="exercise the reconciliation rules offline; "
                             "makes no network call")
    parser.add_argument("--shape-check", action="store_true",
                        help="adjudicate records the real serving path wrote, "
                             "offline; makes no network or model call")
    args = parser.parse_args(argv)

    if args.self_test:
        return self_test()
    if args.shape_check:
        return shape_check()

    bearer = os.environ.get("MI_BEARER", "").strip()
    profile = os.environ.get("AZURE_MI_API_PUBLISH_PROFILE", "").strip()
    secrets = [s for s in (bearer, profile) if len(s) >= ra.MIN_SECRET_LENGTH]
    client_id = args.portfolio_id.split("/", 1)[0]
    report: Dict[str, Any] = {
        "what_this_is": "slice 2 live temporal acceptance — did the NEW "
                        "temporal result become the response, and does it "
                        "reconcile to the execution receipt",
        "bank": [c["case_id"] for c in BANK],
        "portfolio_id": args.portfolio_id, "expect_commit": args.expect_commit,
        "stages": {}, "cases": [],
    }

    def stop(stage: str, verdict: str, message: str) -> int:
        report["verdict"], report["stopped_at"] = verdict, stage
        report["stages"][stage] = message
        ra._save(report, args.json_out, secrets)
        print(f"::error::{stage}: {message}")
        return 2

    if not bearer or not profile:
        return stop("credentials", "NOT_EXECUTABLE",
                    "MI_BEARER and AZURE_MI_API_PUBLISH_PROFILE are both required")

    # -- 1. provenance, by GET, before a single question --------------------- #
    served = s1b.served_commit(args.base_url)
    report["stages"]["provenance"] = {"served_commit": served,
                                      "expected": args.expect_commit}
    if not served:
        return stop("provenance", "INCONCLUSIVE",
                    "no build stamp at /health; provenance cannot be established")
    if args.expect_commit and not served.startswith(args.expect_commit[:7]):
        return stop("provenance", "FAIL",
                    f"serving {served} — expected {args.expect_commit}")

    # -- 2. the sink, and everything already in it --------------------------- #
    try:
        scm_host, user, password = ra.publish_profile_credentials(profile)
    except Exception as exc:                                         # noqa: BLE001
        return stop("sink", "NOT_EXECUTABLE",
                    f"the publish profile could not be read: {type(exc).__name__}")
    sink = ra.Sink(scm_host, user, password, args.evidence_path)
    rows, detail = sink.records()
    if rows is None:
        return stop("sink", "NOT_EXECUTABLE", f"the evidence sink: {detail}")
    already = {row.get("correlation_id") for row in rows}
    report["stages"]["sink"] = {"readable": True, "detail": detail,
                                "pre_existing_records": len(rows)}

    # -- 3. the bank, once each ---------------------------------------------- #
    ask = _live_asker(args.base_url, args.path, [], args.portfolio_id)
    for case in BANK:
        envelope = ask(case["question"])
        record, matched = (None, "not polled")
        if not envelope.get("__transport_error__"):
            record, matched = ra.poll_for(
                sink, case["question"], client_id,
                interval=args.poll_interval, timeout=args.poll_timeout,
                already=already)
        adjudicated = adjudicate(case, envelope, record, client_id)
        adjudicated["evidence_match"] = matched
        report["cases"].append(adjudicated)
        print(f"  {adjudicated['verdict']} {case['case_id']:<6} "
              f"{case['question'][:46]:<48} "
              f"snapshots={len(adjudicated.get('snapshots') or ())} "
              f"reconciled={adjudicated['values_reconciled']}")
        for problem in adjudicated["problems"]:
            print(f"        {problem}")

    positives = [c for c in report["cases"] if c["kind"] != "negative"]
    negative = [c for c in report["cases"] if c["kind"] == "negative"]
    report["totals"] = {
        "positive_questions": len(positives),
        "positive_pass": sum(1 for c in positives if c["verdict"] == "PASS"),
        "served_new": sum(1 for c in positives if c.get("decision") == SERVED_NEW),
        "legacy_fallbacks": sum(1 for c in positives
                                if c.get("decision") and
                                c["decision"] != SERVED_NEW),
        "values_reconciled": sum(c["values_reconciled"] for c in report["cases"]),
        "negative_fail_closed": all(c["verdict"] == "PASS" for c in negative),
    }
    ok = (report["totals"]["positive_pass"] == len(positives)
          and report["totals"]["negative_fail_closed"])
    report["verdict"] = "PASS" if ok else "FAIL"
    ra._save(report, args.json_out, secrets)
    print(f"  VERDICT {report['verdict']}  {json.dumps(report['totals'])}")
    print("  REMINDER: turning the canary back off is an operator action.")
    return 0 if ok else 1


# --------------------------------------------------------------------------- #
# the self-test: every rule, offline
# --------------------------------------------------------------------------- #

def _record(points: List[Dict[str, Any]], *, aggregation: str = "sum",
            metric: str = "current_outstanding_balance",
            dimensions: Optional[List[str]] = None,
            comparison: Optional[Dict[str, Any]] = None,
            decision: str = SERVED_NEW, reason: str = "") -> Dict[str, Any]:
    return {
        "model": {"model_id": "claude-opus-5"},
        "serving": {"principal_matched": True, "decision": decision,
                    "response_served_from": decision, "reason": reason},
        "eligibility": {"eligible": True, "perimeter": "slice2_temporal"},
        "execution": {
            "bound_spec": {"metric": metric, "aggregation": aggregation,
                           "dimensions": dimensions or []},
            "temporal": {"shape": "time_series", "basis": "count",
                         "points": points, "comparison": comparison}},
    }


def _envelope(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    return {"ok": True, "artifacts": [{"rows": rows}]}


def self_test() -> int:
    """Prove every rule, including that each CATCHES its own failure."""
    checks: List[Tuple[str, bool, str]] = []

    def check(name: str, condition: bool, detail: str = "") -> None:
        checks.append((name, condition, detail))

    # bank
    check("bank loads six fixed cases", len(BANK) == 6)
    check("bank has five positives and one negative",
          sum(1 for c in BANK if c["kind"] != "negative") == 5
          and sum(1 for c in BANK if c["kind"] == "negative") == 1)
    check("bank questions are the specified wording",
          [c["question"] for c in BANK] == [
              "What was funded balance each month?",
              "Show funded balance from October 2025 to June 2026.",
              "How many drawdown loans were there each month?",
              "Show loan count by LTV bucket each month.",
              "What was funded balance this month versus last month?",
              "Show funded balance over the last 24 months."])

    # value column, from the bound spec rather than a guess
    check("value column follows the bound spec",
          value_column(_record([], aggregation="sum")) ==
          "current_outstanding_balance_sum")
    check("a count is the executor's own row-count column",
          value_column(_record([], aggregation="count")) == "loan_count")

    # series
    # The synthetic series uses dates the governed catalogue actually holds, so
    # the no-fabrication assertion is exercised rather than tripped.
    points = [{"reporting_date": "2025-11-30", "value": 10.0,
               "snapshot_id": "ERE/run_20251130"},
              {"reporting_date": "2026-06-30", "value": 20.0,
               "snapshot_id": "ERE/run_20260630"}]
    good = _envelope([{"reporting_date": "2025-11-30",
                       "current_outstanding_balance_sum": 10.0},
                      {"reporting_date": "2026-06-30",
                       "current_outstanding_balance_sum": 20.0}])
    found, checked = reconcile_series(_record(points), http_rows(good))
    check("series reconciles a faithful response", not found and checked == 2,
          str(found))
    drifted = _envelope([{"reporting_date": "2025-11-30",
                          "current_outstanding_balance_sum": 10.0},
                         {"reporting_date": "2026-06-30",
                          "current_outstanding_balance_sum": 999.0}])
    found, _ = reconcile_series(_record(points), http_rows(drifted))
    check("series CATCHES a drifted figure", bool(found), str(found))
    missing = _envelope([{"reporting_date": "2026-06-30",
                          "current_outstanding_balance_sum": 20.0}])
    found, _ = reconcile_series(_record(points), http_rows(missing))
    check("series CATCHES a dropped period", bool(found), str(found))
    extra = _envelope([{"reporting_date": "2025-10-31",
                        "current_outstanding_balance_sum": 5.0},
                       *good["artifacts"][0]["rows"]])
    found, _ = reconcile_series(_record(points), http_rows(extra))
    check("series CATCHES a period the runtime never executed",
          bool(found), str(found))

    # grouped
    gpoints = [{"reporting_date": "2026-06-30", "snapshot_id": "ERE/r",
                "cells": [{"ltv_bucket": "0-30%", "value": 3.0},
                          {"ltv_bucket": "60%+", "value": 7.0}]}]
    grecord = _record(gpoints, aggregation="count", metric=None,
                      dimensions=["ltv_bucket"])
    ggood = _envelope([{"reporting_date": "2026-06-30", "ltv_bucket": "0-30%",
                        "loan_count": 3.0},
                       {"reporting_date": "2026-06-30", "ltv_bucket": "60%+",
                        "loan_count": 7.0}])
    found, checked = reconcile_grouped(grecord, http_rows(ggood))
    check("grouped reconciles every cell", not found and checked == 2, str(found))
    gbad = _envelope([{"reporting_date": "2026-06-30", "ltv_bucket": "0-30%",
                       "loan_count": 3.0},
                      {"reporting_date": "2026-06-30", "ltv_bucket": "60%+",
                       "loan_count": 99.0}])
    found, _ = reconcile_grouped(grecord, http_rows(gbad))
    check("grouped CATCHES a wrong cell", bool(found), str(found))
    gshort = _envelope([{"reporting_date": "2026-06-30", "ltv_bucket": "0-30%",
                         "loan_count": 3.0}])
    found, _ = reconcile_grouped(grecord, http_rows(gshort))
    check("grouped CATCHES a dropped cell", bool(found), str(found))

    # comparison
    cpoints = [{"reporting_date": "2026-05-31", "value": 100.0,
                "snapshot_id": "ERE/a"},
               {"reporting_date": "2026-06-30", "value": 150.0,
                "snapshot_id": "ERE/b"}]
    comparison = {"baseline_value": 100.0, "current_value": 150.0,
                  "absolute_change": 50.0, "percent_change": 50.0}
    crecord = _record(cpoints, comparison=comparison)
    cgood = _envelope([{"reporting_date": "2026-05-31",
                        "current_outstanding_balance_sum": 100.0},
                       {"reporting_date": "2026-06-30",
                        "current_outstanding_balance_sum": 150.0}])
    found, checked = reconcile_comparison(crecord, http_rows(cgood))
    check("comparison reconciles both periods and the change",
          not found and checked == 4, str(found))
    bad_change = dict(comparison, absolute_change=10.0)
    found, _ = reconcile_comparison(_record(cpoints, comparison=bad_change),
                                    http_rows(cgood))
    check("comparison CATCHES a change its own values do not imply",
          bool(found), str(found))
    bad_pct = dict(comparison, percent_change=1.0)
    found, _ = reconcile_comparison(_record(cpoints, comparison=bad_pct),
                                    http_rows(cgood))
    check("comparison CATCHES a wrong percentage", bool(found), str(found))

    # the restated naming rule still matches the adapter that owns it
    adapter_source = (_REPO / "mi_agent" / "plan_runtime_adapter.py")
    if adapter_source.exists():
        text = adapter_source.read_text(encoding="utf-8")
        check("the row-count column is still what the adapter names it",
              f'_ROW_COUNT_FIELD = "{_ROW_COUNT_FIELD}"' in text)
        check("the aggregated column is still <metric>_<aggregation>",
              'f"{spec.metric}_{spec.aggregation}"' in text)
        check("a count is still the adapter's count sentinel",
              f'"{_COUNT}": COUNT,' in text and f'COUNT = "{_COUNT}"' in
              (_REPO / "mi_agent" / "query_plan.py").read_text(encoding="utf-8"))
    else:
        check("the adapter source was available to pin against", False,
              "run from the repository so the drift guard can read it")

    # the two readings of "what did the response carry" agree where they overlap
    _kpis, owner_rows = s1b.response_figures(good)
    check("the row reading agrees with the estate's figure extractor",
          [dict(r) for r in owner_rows if REPORTING_DATE in r] == http_rows(good))

    # a series that is not the last artefact is still read, and read ONCE
    two_artefacts = {"ok": True, "artifacts": [
        {"rows": good["artifacts"][0]["rows"]},
        {"rows": [{"ltv_bucket": "0-30%", "loan_count": 3}]}]}
    check("a non-series artefact contributes no temporal rows",
          len(http_rows(two_artefacts)) == 2)
    found, checked = reconcile_series(_record(points), http_rows(two_artefacts))
    check("a series artefact that is not last is still reconciled",
          not found and checked == 2, str(found))

    # negative
    refused = _record([], decision="LEGACY_FALLBACK",
                      reason="TEMPORAL_NOT_RESOLVED:PERIOD_NOT_AVAILABLE")
    check("negative passes when it fails closed",
          not classify_negative(refused, []))
    shortened = _record(points, decision="LEGACY_FALLBACK",
                        reason="TEMPORAL_NOT_RESOLVED:PERIOD_NOT_AVAILABLE")
    check("negative CATCHES a shortened series",
          bool(classify_negative(shortened, http_rows(good))))
    served = _record(points, decision=SERVED_NEW)
    check("negative CATCHES a served temporal answer",
          bool(classify_negative(served, http_rows(good))))
    check("negative does NOT fail on rows the LEGACY envelope happens to carry",
          not classify_negative(refused, http_rows(good)))

    # adjudication wiring
    generic = BANK[2]                      # S2-P3: no window to name
    verdict = adjudicate(generic, good, _record(points), "ERE")
    check("a faithful positive adjudicates PASS", verdict["verdict"] == "PASS",
          str(verdict["problems"]))
    check("the snapshot identities are recorded",
          verdict["snapshot_ids"] == ["ERE/run_20251130", "ERE/run_20260630"])
    legacy = _record(points, decision="LEGACY_FALLBACK")
    check("a legacy fallback adjudicates FAIL",
          adjudicate(generic, good, legacy, "ERE")["verdict"] == "FAIL")
    check("a missing evidence record adjudicates FAIL",
          adjudicate(generic, good, None, "ERE")["verdict"] == "FAIL")
    wrong_model = _record(points)
    wrong_model["model"]["model_id"] = "claude-3-5-sonnet"
    check("a wrong model adjudicates FAIL",
          adjudicate(generic, good, wrong_model, "ERE")["verdict"] == "FAIL")
    coverage = dict(good, semanticCoverageRefused=True)
    check("a coverage refusal on a served result adjudicates FAIL",
          adjudicate(generic, coverage, _record(points), "ERE")["verdict"] == "FAIL")
    check("a transport error adjudicates FAIL",
          adjudicate(generic, {"__transport_error__": True,
                              "__http_status__": 401}, None, "ERE")["verdict"] == "FAIL")

    # -- which periods, not how many ---------------------------------------- #
    check("the catalogue under acceptance is the sparse production one",
          PRODUCTION_CATALOGUE == ("2025-10-31", "2025-11-30", "2026-06-30"))
    fabricated = _record(points + [{"reporting_date": "2026-03-31", "value": 5.0,
                                    "snapshot_id": "ERE/run_20260331"}])
    invented = _envelope([*good["artifacts"][0]["rows"],
                          {"reporting_date": "2026-03-31",
                           "current_outstanding_balance_sum": 5.0}])
    check("a period the catalogue does not hold is CAUGHT",
          adjudicate(generic, invented, fabricated, "ERE")["verdict"] == "FAIL")
    foreign = _record([{"reporting_date": "2026-06-30", "value": 1.0,
                        "snapshot_id": "OTHER/run_20260630"}])
    check("another client's snapshot is CAUGHT",
          adjudicate(generic, _envelope([
              {"reporting_date": "2026-06-30",
               "current_outstanding_balance_sum": 1.0}]),
              foreign, "ERE")["verdict"] == "FAIL")
    windowed = dict(BANK[1])                       # S2-P2 names its three periods
    check("a window that names its periods CATCHES a short selection",
          adjudicate(windowed, good, _record(points), "ERE")["verdict"] == "FAIL")
    whole = [{"reporting_date": d, "value": float(i),
              "snapshot_id": f"ERE/run_{d.replace('-', '')}"}
             for i, d in enumerate(PRODUCTION_CATALOGUE)]
    whole_rows = _envelope([{"reporting_date": d,
                             "current_outstanding_balance_sum": float(i)}
                            for i, d in enumerate(PRODUCTION_CATALOGUE)])
    check("a window that names its periods PASSES when they are all served",
          adjudicate(windowed, whole_rows, _record(whole), "ERE")["verdict"]
          == "PASS",
          str(adjudicate(windowed, whole_rows, _record(whole), "ERE")["problems"]))

    failed = [c for c in checks if not c[1]]
    print("=== SLICE 2 TEMPORAL ACCEPTANCE — SELF TEST (no network)")
    for name, ok, detail in checks:
        print(f"  {'PASS' if ok else 'FAIL'} {name}"
              + (f"   {detail}" if not ok and detail else ""))
    print(f"  {len(checks) - len(failed)}/{len(checks)} rules proved")
    return 0 if not failed else 1


# --------------------------------------------------------------------------- #
# the shape check: the adjudicator against records the PRODUCT wrote
# --------------------------------------------------------------------------- #

#: The recorded case standing in for each acceptance KIND. These are the
#: questions the live temporal runs actually recorded, so the plans replayed here
#: are Opus's own. The acceptance bank's own wording cannot be replayed — no
#: recording of it exists and asking for one would be a model call — so what is
#: proved here is the SHAPE each kind produces and how this file reads it.
_SHAPE_STANDINS = (("T06", "series"), ("T07", "grouped"),
                   ("T11", "comparison"), ("T13", "negative"))

#: `ReplayClient` reports this as the model id. It is the one expected difference
#: between an offline record and a live one, and the only problem tolerated here.
_REPLAY_MODEL_PROBLEM = "is not the required model"


def shape_check() -> int:
    """Does this adjudicator read what `plan_serving_canary` actually writes?

    The self-test proves the rules are internally consistent against records this
    file constructs — which cannot catch this file assuming a key the product
    does not emit. So this drives the REAL serving path through the accepted
    offline integration harness and adjudicates its output with no special
    casing, reporting any problem other than the replay client's model sentinel.
    """
    import os
    import tempfile

    try:
        sys.path.insert(0, str(_HERE))
        import temporal_serving_integration as tsi                   # noqa: E402
        from mi_agent import plan_serving_canary as canary           # noqa: E402
        from mi_agent import plan_shadow_wiring as wiring            # noqa: E402
        from mi_agent.interpretation_v2.opus_interpreter import (     # noqa: E402
            OpusInterpreter, ReplayClient)
        from mi_agent.mi_query_validator import load_mi_semantics    # noqa: E402
        from mi_agent.tests import temporal_snapshot_fixture as fixture  # noqa: E402
    except Exception as exc:                                         # noqa: BLE001
        print(f"::error::the shape check needs the product's dependencies: "
              f"{type(exc).__name__}: {exc}")
        return 2

    payloads, questions = tsi.load_payloads()
    semantics = load_mi_semantics(str(tsi.REGISTRY))
    history = fixture.default_history()
    wiring.set_interpreter_factory(lambda: OpusInterpreter(ReplayClient(payloads)))
    os.environ[canary.SERVE_ENV_VAR] = canary.SERVE_CANARY
    os.environ[canary.PRINCIPALS_ENV_VAR] = tsi.CANARY_PRINCIPAL

    print("=== SLICE 2 TEMPORAL ACCEPTANCE — SHAPE CHECK "
          "(real serving path, no network, no model call)")
    failures = 0
    with tempfile.TemporaryDirectory() as tmp:
        store = fixture.build_store(Path(tmp) / "snapshots", history)
        book = store.load_loans(store.list_snapshots(
            fixture.CLIENT_ID, route=fixture.ROUTE)[-1].snapshot_id)
        for case, kind in _SHAPE_STANDINS:
            question = questions[case]
            payload, record = tsi.serve_case(
                question, store=store, semantics=semantics, frames=book,
                principal=tsi.CANARY_PRINCIPAL)
            envelope = dict(payload) if payload else {"ok": True, "artifacts": []}
            # The fixture's OWN history and client, because this mode proves
            # the adjudicator reads the product's shape — it is not pointed at
            # production and must not be judged against production's catalogue.
            out = adjudicate({"case_id": case, "kind": kind,
                              "question": question, "why": kind},
                             envelope, record,
                             client_id=fixture.CLIENT_ID,
                             catalogue=tuple(tsi.CATALOGUE))
            real = [p for p in out["problems"]
                    if _REPLAY_MODEL_PROBLEM not in p]
            failures += bool(real)
            print(f"  {'PASS' if not real else 'FAIL'} {case:<5} {kind:<11}"
                  f" decision={out.get('decision')}"
                  f" perimeter={out.get('perimeter')}"
                  f" periods={len(out.get('snapshots') or ())}"
                  f" rows={out.get('http_rows')}"
                  f" reconciled={out['values_reconciled']}"
                  f" column={value_column(record)!r}")
            for problem in real:
                print(f"        {problem}")
    print(f"  {len(_SHAPE_STANDINS) - failures}/{len(_SHAPE_STANDINS)} kinds "
          f"adjudicated against records the product wrote")
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
