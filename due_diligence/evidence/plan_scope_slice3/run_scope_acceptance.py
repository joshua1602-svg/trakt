#!/usr/bin/env python3
"""Slice 3 live scope acceptance — did the governed PORTFOLIO SCOPE serve?

WHY A THIRD HARNESS. The slice 1B and slice 2 banks are fixed and sha-pinned,
and neither can ask a scope question: `run_serving_acceptance` filters a slice 1
manifest, and `run_temporal_acceptance`'s six questions contain no role, no named
portfolio and no lifecycle phrase. Neither reads `population` off the plan at
all, so neither can tell a correct scope from an absent one.

WHAT THIS ADDS, AND ONLY THIS: the six-question scope bank, and the READING of a
scoped answer — which population the plan asked about, which scope predicates
the executor proved, and whether a named book resolved to a governed id the
model never saw. Everything else is imported from the harnesses that already own
it:

    transport / MI_BEARER    certify_mi_api._live_asker
    zero-cost provenance     run_serving_acceptance.served_commit  (GET /health)
    HTTP figure extraction   run_serving_acceptance.response_figures
    Kudu evidence sink       run_acceptance.Sink
    correlation-aware poll   run_acceptance.poll_for
    publish-profile creds    run_acceptance.publish_profile_credentials
    model-id gate            run_acceptance.is_required_model
    secret hygiene           run_acceptance.scrub / _save / MIN_SECRET_LENGTH
    temporal reconciliation  run_temporal_acceptance.reconcile_series

THE EXECUTION RECEIPT IS THE NUMERIC ORACLE, and the governed registry is the
IDENTITY oracle. No portfolio figure and no portfolio id is written down here.
A named case asserts that the id the plan bound is one the client's registry
declares — not a particular string this file remembers — so the bank does not go
stale when a client is onboarded or renamed.

IT DEPLOYS NOTHING AND CONFIGURES NOTHING. The canary must already be on.

    --self-test    exercises every adjudication rule against synthetic records.
                   Stdlib only, no network, no model. Run it before production.
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
                 "mi_api_certification", "plan_temporal_slice2"):
    _path = str(_REPO / "due_diligence" / "evidence" / _sibling)
    if _path not in sys.path:
        sys.path.insert(0, _path)

TOLERANCE = 0.01
SERVED_NEW = "NEW"
ROLE_FIELD = "source_portfolio_type"
SOURCE_FIELD = "source_portfolio_id"
SEASONING_FIELD = "seasoning_segment"

#: Capabilities that carry movement/attribution meaning. S3-N2 must land on one
#: of these or refuse; degrading to a scoped balance is the silent capability
#: loss this case exists to catch.
ATTRIBUTION_CAPABILITIES = frozenset({"period_movement", "funded_bridge",
                                      "pipeline_stage_movement"})
ATTRIBUTION_OPERATIONS = frozenset({"movement", "bridge", "transition",
                                    "arrivals", "departures", "reconciliation"})

#: Anything that looks like a physical handle rather than a reader's phrase.
_PHYSICAL_MARKERS = ("_id", "run_", "blob", "://", ".csv", ".parquet", "/")

#: THE BANK. FIXED — six questions, asked once each, never reworded and never
#: extended at run time. A bank a harness can edit is a bank that can be made to
#: pass.
BANK: Tuple[Dict[str, Any], ...] = (
    {"case_id": "S3-P1", "kind": "scalar",
     "question": "What is the back book balance?",
     "why": "the generic lifecycle phrase names the FUNDED population; it must "
            "not narrow to a seasoning cohort on either path",
     "expect_base": "funded", "forbid_seasoning": True,
     "expect_scope": []},
    {"case_id": "S3-P2", "kind": "scalar",
     "question": "How many acquired drawdown loans have LTV above 50%?",
     "why": "a role and two ordinary predicates, all three proven by the "
            "executor's own receipt",
     "expect_base": "funded", "expect_lens": "acquired",
     "expect_scope": [(ROLE_FIELD, "acquired")],
     "expect_proved_fields": ["erm_product_type", "current_loan_to_value",
                              ROLE_FIELD]},
    {"case_id": "S3-P3", "kind": "scalar",
     "question": "What is funded balance for the ALP back book?",
     "why": "a declared alias resolves to a governed id the model never saw, "
            "and the words inside the name create no other axis",
     "expect_base": "funded", "expect_named": True, "forbid_seasoning": True,
     "forbid_role": True},
    {"case_id": "S3-P4", "kind": "series",
     "question": "Show ALP Acquired Back Book funded balance each month.",
     "why": "a named source survives every selected snapshot, and the series "
            "reconciles period by period",
     "expect_base": "funded", "expect_named": True, "forbid_seasoning": True,
     "forbid_role": True},
    {"case_id": "S3-N1", "kind": "negative",
     "question": "What is in the front book?",
     "why": "the pipeline population has no deterministic owner on this "
            "runtime; it must fail closed rather than answer over funded rows",
     "expect_base": "pipeline", "must_not_execute": True},
    {"case_id": "S3-N2", "kind": "attribution",
     "question": "How much of this month's increase came from the acquired "
                 "back book?",
     "why": "scope must not cost the attribution capability; served or "
            "refused, it may not become a plain scoped balance",
     "expect_base": "funded", "expect_lens": "acquired",
     "forbid_seasoning": True},
)


# --------------------------------------------------------------------------- #
# reading a scoped record
# --------------------------------------------------------------------------- #

def plan_of(record: Mapping[str, Any]) -> Dict[str, Any]:
    return dict(((record.get("compiler") or {}).get("plan")) or {})


def population_of(record: Mapping[str, Any]) -> Dict[str, Any]:
    return dict(plan_of(record).get("population") or {})


def scope_of(record: Mapping[str, Any]) -> List[Tuple[str, Any]]:
    """The scope predicates the PLAN authorised, as (field, value) pairs."""
    return sorted((str(p.get("canonical_field") or ""), p.get("value"))
                  for p in (population_of(record).get("scope_predicates") or ()))


def receipts_of(record: Mapping[str, Any]) -> List[Dict[str, Any]]:
    """Every execution receipt, whether one snapshot or a series of them."""
    execution = record.get("execution") or {}
    series = execution.get("per_snapshot") or execution.get("snapshots")
    if isinstance(series, list) and series:
        return [dict(r) for r in series if isinstance(r, Mapping)]
    receipt = execution.get("receipt")
    return [dict(receipt)] if isinstance(receipt, Mapping) else []


def proved_fields(receipt: Mapping[str, Any]) -> List[str]:
    return [str(p.get("field") or p.get("field_key") or "")
            for p in (receipt.get("applied_predicates") or ())
            if isinstance(p, Mapping)]


def decision_of(record: Mapping[str, Any]) -> str:
    return str((record.get("serving") or {}).get("decision") or "")


def authored_physical(reference: Optional[str],
                      bound_id: Optional[str] = None) -> Optional[str]:
    """Did the MODEL write a physical handle where a reader's phrase belongs?

    TWO SIGNALS, AND THE SECOND IS THE ONE THAT MATTERS. The marker list catches
    a path, a run or a blob. It does NOT catch an opaque client id — production
    ids look like `alp_acquired`, which contains no marker at all — and that is
    precisely the handle the model must never author, because the id is the
    compiler's to produce from the registry and the model has never been shown
    one: `get_source_portfolios` returns labels and aliases only.

    So a reference IDENTICAL to the id the compiler bound is the signal. A
    reader says "the ALP back book"; only something that had seen the registry's
    internals would say "alp_acquired".

    Found by this harness's own self-test, which asserted the defect was caught
    and discovered it was not.
    """
    if not reference:
        return None
    low = str(reference).strip().lower()
    if any(m in low for m in _PHYSICAL_MARKERS):
        return reference
    if bound_id and low == str(bound_id).strip().lower():
        return reference
    return None


def scalar_of(envelope: Mapping[str, Any]) -> Optional[float]:
    import run_serving_acceptance as s1b

    figures, _rows = s1b.response_figures(envelope)
    values = [v for v in figures.values() if isinstance(v, (int, float))]
    return float(values[0]) if len(values) == 1 else None


# --------------------------------------------------------------------------- #
# adjudication
# --------------------------------------------------------------------------- #

def adjudicate(case: Mapping[str, Any], envelope: Mapping[str, Any],
               record: Optional[Mapping[str, Any]],
               declared_names: Sequence[str]) -> Dict[str, Any]:
    """One case. `(verdict, problems, everything that was read)`."""
    import run_acceptance as ra

    out: Dict[str, Any] = {
        "case_id": case["case_id"], "kind": case["kind"],
        "question": case["question"], "why": case["why"],
        "problems": [], "values_reconciled": 0,
        "http_status": envelope.get("__http_status__"),
        "transport_error": envelope.get("__transport_error__"),
        "silent_scope_drop": False, "silent_scope_widening": False,
        "silent_scope_narrowing": False,
        "model_authored_physical_source": False,
        "cross_client_source": False,
    }
    problems = out["problems"]

    if out["transport_error"]:
        problems.append(f"transport: {out['transport_error']}")
        out["verdict"] = "INCONCLUSIVE"
        return out
    if record is None:
        problems.append("no governed evidence record was found for this question")
        out["verdict"] = "INCONCLUSIVE"
        return out

    population = population_of(record)
    scope = scope_of(record)
    plan = plan_of(record)
    decision = decision_of(record)
    receipts = receipts_of(record)
    model_id = str((record.get("model") or {}).get("model_id") or "")
    serving = record.get("serving") or {}
    out.update({
        "decision": decision, "model_id": model_id,
        "population": population, "scope": scope,
        "capability": plan.get("capability"), "operation": plan.get("operation"),
        "eligibility": record.get("eligibility"),
        "disposition": record.get("disposition"),
        "principal_matched": serving.get("principal_matched"),
        "receipts": len(receipts),
        "execution_attempted": bool((record.get("execution") or {})
                                    .get("attempted")),
        "execution_error": (record.get("execution") or {}).get("error"),
        "orchestration_error": record.get("orchestration_error"),
    })

    if not ra.is_required_model(model_id):
        problems.append(f"model_id={model_id!r} is not the required model")
    if serving.get("principal_matched") is not True:
        problems.append("principal_matched was not true")
    if out["execution_error"]:
        problems.append(f"execution error: {out['execution_error']}")
    if out["orchestration_error"]:
        problems.append(f"orchestration error: {out['orchestration_error']}")

    base = str(population.get("base") or "")
    lens = str(population.get("lens") or "all")
    seasoning = str(population.get("seasoning") or "any")
    reference = population.get("source_reference")
    bound_id = population.get(SOURCE_FIELD)

    # -- the population the plan asked about -------------------------------- #
    if case.get("expect_base") and base != case["expect_base"]:
        problems.append(f"population.base={base!r}, expected "
                        f"{case['expect_base']!r}")

    # -- the negative: it must not have executed at all --------------------- #
    if case.get("must_not_execute"):
        out["served_a_figure"] = scalar_of(envelope) is not None
        if out["execution_attempted"]:
            problems.append("the governed runtime EXECUTED a pipeline plan")
        if decision == SERVED_NEW:
            problems.append("a pipeline plan was SERVED by the funded runtime")
        if receipts:
            problems.append(f"{len(receipts)} execution receipt(s) exist for a "
                            f"population this runtime does not execute")
        out["verdict"] = "PASS" if not problems else "FAIL"
        return out

    # -- scope predicates, plan side ---------------------------------------- #
    if case.get("expect_lens") and lens != case["expect_lens"]:
        problems.append(f"population.lens={lens!r}, expected "
                        f"{case['expect_lens']!r}")
        out["silent_scope_drop" if lens == "all"
            else "silent_scope_narrowing"] = True
    if case.get("forbid_role") and lens != "all":
        problems.append(f"a role was inferred that the question did not state: "
                        f"lens={lens!r}")
        out["silent_scope_narrowing"] = True
    if case.get("forbid_seasoning") and seasoning not in ("", "any"):
        problems.append(f"seasoning={seasoning!r} was inferred; it narrows the "
                        f"answer and the question named no vintage")
        out["silent_scope_narrowing"] = True
    if case.get("expect_scope") is not None and not case.get("expect_named"):
        expected = sorted(case["expect_scope"])
        if scope != expected:
            problems.append(f"scope predicates {scope}, expected {expected}")

    # -- a NAMED source: identity, not a string this file remembers ---------- #
    if case.get("expect_named"):
        out["source_reference"] = reference
        out["source_portfolio_id"] = bound_id
        if not reference:
            problems.append("the named book was not carried in source_reference")
            out["silent_scope_drop"] = True
        physical = authored_physical(reference, bound_id)
        if physical:
            out["model_authored_physical_source"] = True
            problems.append(f"the model authored a physical handle: {physical!r}")
        if not bound_id:
            problems.append("no governed source_portfolio_id was bound")
            out["silent_scope_widening"] = True
        elif declared_names and str(bound_id) not in set(declared_names):
            out["cross_client_source"] = True
            problems.append(f"source_portfolio_id={bound_id!r} is not declared "
                            f"by this client's governed registry")
        if bound_id and (SOURCE_FIELD, bound_id) not in scope:
            problems.append(f"the bound id {bound_id!r} is not in the plan's "
                            f"scope predicates {scope}")

    # -- the attribution control -------------------------------------------- #
    if case["kind"] == "attribution":
        capability = str(plan.get("capability") or "")
        operation = str(plan.get("operation") or "")
        kept = (capability in ATTRIBUTION_CAPABILITIES
                or operation in ATTRIBUTION_OPERATIONS)
        out["attribution_preserved"] = kept
        if not kept:
            problems.append(f"attribution intent lost: capability={capability!r} "
                            f"operation={operation!r} carries no movement "
                            f"meaning — the question became a scoped balance")
        # Served or refused are BOTH acceptable here. What is not acceptable is
        # answering a simpler question, and that is what `kept` decides.
        out["verdict"] = "PASS" if not problems else "FAIL"
        return out

    # -- from here the case must have SERVED NEW ---------------------------- #
    if decision != SERVED_NEW:
        problems.append(f"serving.decision={decision!r}, expected {SERVED_NEW}; "
                        f"reason={serving.get('reason')!r}")
        out["verdict"] = "FAIL"
        return out

    # -- every predicate the plan authorised is proved by every receipt ------ #
    wanted = list(case.get("expect_proved_fields") or ())
    if case.get("expect_named") and bound_id:
        wanted.append(SOURCE_FIELD)
    if not receipts and wanted:
        problems.append("no execution receipt to prove the predicates against")
    for field in wanted:
        for index, receipt in enumerate(receipts):
            if field not in proved_fields(receipt):
                problems.append(f"receipt {index} does not prove {field!r} "
                                f"(it proves {proved_fields(receipt)})")
    if case.get("forbid_seasoning"):
        for index, receipt in enumerate(receipts):
            if SEASONING_FIELD in proved_fields(receipt):
                out["silent_scope_narrowing"] = True
                problems.append(f"receipt {index} applied {SEASONING_FIELD!r}, "
                                f"which the question never asked for")

    # -- the caller received what the runtime recorded ----------------------- #
    if case["kind"] == "series":
        import run_temporal_acceptance as t2

        rows = t2.http_rows(envelope)
        series_problems, reconciled = t2.reconcile_series(record, rows)
        problems.extend(series_problems)
        out["values_reconciled"] = reconciled
        out["snapshots"] = [p.get("snapshot_id") for p in
                            (t2.temporal_block(record).get("points") or ())]
    else:
        served = scalar_of(envelope)
        executed = (record.get("execution") or {}).get("value")
        out["http_value"], out["executed_value"] = served, executed
        if served is None:
            problems.append("the response carried no single scalar to reconcile")
        elif not isinstance(executed, (int, float)):
            problems.append("the record carried no executed value to reconcile")
        elif abs(float(served) - float(executed)) > TOLERANCE:
            problems.append(f"the response said {served} and the runtime "
                            f"recorded {executed}")
        else:
            out["values_reconciled"] = 1

    # -- the coverage gate must not have refused a correct answer ----------- #
    meta = (envelope.get("metadata") or {}) if isinstance(envelope, Mapping) else {}
    if meta.get("semanticCoverageRefused"):
        problems.append("the answer was refused by the semantic coverage gate")
        out["semantic_coverage_refused"] = True
    out["parser_mode"] = meta.get("parserMode")

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
    parser.add_argument("--poll-timeout", type=float, default=150.0)
    parser.add_argument("--json-out", default="slice3-scope-acceptance.json")
    parser.add_argument("--self-test", action="store_true",
                        help="exercise the adjudication rules offline; makes "
                             "no network call and no model call")
    parser.add_argument("--provenance-only", action="store_true",
                        help="confirm WHICH build is serving and stop; one GET "
                             "per read, no question asked, no model call")
    parser.add_argument("--stable-reads", type=int, default=1)
    parser.add_argument("--stable-gap", type=float, default=15.0)
    args = parser.parse_args(argv)

    if args.self_test:
        return self_test()

    import run_acceptance as ra
    import run_serving_acceptance as s1b
    from certify_mi_api import _live_asker

    if args.provenance_only:
        import run_temporal_acceptance as t2
        return t2.provenance_only(args)

    bearer = os.environ.get("MI_BEARER", "").strip()
    profile = os.environ.get("AZURE_MI_API_PUBLISH_PROFILE", "").strip()
    secrets = [s for s in (bearer, profile) if len(s) >= ra.MIN_SECRET_LENGTH]
    client_id = args.portfolio_id.split("/", 1)[0]
    report: Dict[str, Any] = {
        "what_this_is": "slice 3 live scope acceptance — did the governed "
                        "portfolio scope become the response, and does it "
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
    # THE GOVERNED NAMES THIS CLIENT DECLARES, read from the records themselves
    # rather than written down here: a named case asserts that the id the plan
    # bound is one the registry governs, and the registry is the client's.
    declared: List[str] = []
    ask = _live_asker(args.base_url, args.path, [], args.portfolio_id)
    for case in BANK:
        envelope = ask(case["question"])
        if envelope.get("__http_status__") in (401, 403):
            return stop("auth", "AUTH / NOT_EXECUTABLE",
                        f"{case['case_id']} was rejected with HTTP "
                        f"{envelope.get('__http_status__')}; the remaining "
                        f"questions were not asked")
        record, matched = (None, "not polled")
        if not envelope.get("__transport_error__"):
            record, matched = ra.poll_for(
                sink, case["question"], client_id,
                interval=args.poll_interval, timeout=args.poll_timeout,
                already=already)
        if record is not None:
            for reason in ((record.get("compiler") or {}).get("reasons") or ()):
                detail_text = str((reason or {}).get("detail") or "")
                if "it governs [" in detail_text:
                    declared = sorted(set(declared) | _names_in(detail_text))
        adjudicated = adjudicate(case, envelope, record, declared)
        adjudicated["evidence_match"] = matched
        report["cases"].append(adjudicated)
        print(f"  {adjudicated['verdict']:<12} {case['case_id']:<6} "
              f"{case['question'][:44]:<46} "
              f"base={(adjudicated.get('population') or {}).get('base')!r} "
              f"decision={adjudicated.get('decision')!r} "
              f"reconciled={adjudicated['values_reconciled']}")
        for problem in adjudicated["problems"]:
            print(f"        {problem}")

    cases = report["cases"]
    report["totals"] = {
        "questions": len(cases),
        "passed": sum(1 for c in cases if c["verdict"] == "PASS"),
        "failed": sum(1 for c in cases if c["verdict"] == "FAIL"),
        "inconclusive": sum(1 for c in cases if c["verdict"] == "INCONCLUSIVE"),
        "served_new": sum(1 for c in cases if c.get("decision") == SERVED_NEW),
        "legacy_fallbacks": sum(1 for c in cases if c.get("decision")
                                and c["decision"] != SERVED_NEW),
        "values_reconciled": sum(c["values_reconciled"] for c in cases),
        "silent_scope_drops": sum(1 for c in cases if c.get("silent_scope_drop")),
        "silent_scope_widenings": sum(1 for c in cases
                                      if c.get("silent_scope_widening")),
        "silent_scope_narrowings": sum(1 for c in cases
                                       if c.get("silent_scope_narrowing")),
        "model_authored_physical_source_bindings": sum(
            1 for c in cases if c.get("model_authored_physical_source")),
        "cross_client_scope_selections": sum(1 for c in cases
                                             if c.get("cross_client_source")),
        "semantic_coverage_refusals": sum(1 for c in cases
                                          if c.get("semantic_coverage_refused")),
        "execution_errors": sum(1 for c in cases if c.get("execution_error")),
        "exception_escapes": sum(1 for c in cases
                                 if c.get("orchestration_error")),
    }
    ok = report["totals"]["passed"] == len(cases)
    report["verdict"] = "PASS" if ok else "FAIL"
    ra._save(report, args.json_out, secrets)
    print(f"  VERDICT {report['verdict']}  {json.dumps(report['totals'])}")
    print("  REMINDER: turning the canary back off is an operator action.")
    return 0 if ok else 1


def _names_in(detail: str) -> set:
    """The governed names a refusal listed, read back out of its own message."""
    start = detail.find("it governs [")
    if start < 0:
        return set()
    body = detail[start + len("it governs ["):]
    end = body.find("]")
    body = body if end < 0 else body[:end]
    return {part.strip().strip("'\"") for part in body.split(",") if part.strip()}


# --------------------------------------------------------------------------- #
# the self-test — every rule, offline
# --------------------------------------------------------------------------- #

def _record(**over: Any) -> Dict[str, Any]:
    body: Dict[str, Any] = {
        "model": {"model_id": "claude-opus-5"},
        "serving": {"decision": SERVED_NEW, "principal_matched": True},
        "compiler": {"plan": {"capability": "generic_analysis",
                              "operation": "point_in_time",
                              "population": {"base": "funded", "lens": "all",
                                             "seasoning": "any",
                                             "scope_predicates": []}}},
        "execution": {"attempted": True, "value": 100.0,
                      "receipt": {"applied_predicates": []}},
    }
    for key, value in over.items():
        if isinstance(value, Mapping) and isinstance(body.get(key), dict):
            body[key] = {**body[key], **value}
        else:
            body[key] = value
    return body


def _pop(record: Dict[str, Any], **fields: Any) -> Dict[str, Any]:
    record["compiler"]["plan"]["population"].update(fields)
    return record


def _envelope(value: Optional[float] = 100.0, **over: Any) -> Dict[str, Any]:
    body: Dict[str, Any] = {"ok": True, "artifacts": [], "metadata": {}}
    if value is not None:
        body["artifacts"] = [{"kpis": [{"label": "Balance", "rawValue": value}]}]
    body.update(over)
    return body


def self_test() -> int:
    """Every adjudication rule, against synthetic records. No network, no model."""
    failures: List[str] = []

    def check(name: str, condition: bool) -> None:
        if not condition:
            failures.append(name)
        print(f"   {'ok  ' if condition else 'FAIL'} {name}")

    names = ["alp_acquired", "alp_origination"]
    p1 = next(c for c in BANK if c["case_id"] == "S3-P1")
    p2 = next(c for c in BANK if c["case_id"] == "S3-P2")
    p3 = next(c for c in BANK if c["case_id"] == "S3-P3")
    n1 = next(c for c in BANK if c["case_id"] == "S3-N1")
    n2 = next(c for c in BANK if c["case_id"] == "S3-N2")

    # -- the bank itself ----------------------------------------------------- #
    check("the bank is six questions",
          len(BANK) == 6 and len({c["case_id"] for c in BANK}) == 6)
    check("no expected portfolio VALUE is written down anywhere",
          not any(isinstance(v, float) for c in BANK for v in c.values()))

    # -- P1: the lifecycle phrase -------------------------------------------- #
    good = adjudicate(p1, _envelope(), _record(), names)
    check("P1 passes on funded, unnarrowed, reconciled",
          good["verdict"] == "PASS" and good["values_reconciled"] == 1)
    bad = adjudicate(p1, _envelope(), _pop(_record(), seasoning="back_book"), names)
    check("P1 fails when a seasoning cohort was inferred",
          bad["verdict"] == "FAIL" and bad["silent_scope_narrowing"])
    bad = adjudicate(p1, _envelope(), _pop(_record(), base="pipeline"), names)
    check("P1 fails when the population is not funded", bad["verdict"] == "FAIL")
    bad = adjudicate(p1, _envelope(),
                     _record(serving={"decision": "LEGACY_FALLBACK",
                                      "principal_matched": True}), names)
    check("P1 fails when legacy served", bad["verdict"] == "FAIL")
    bad = adjudicate(p1, _envelope(value=999.0), _record(), names)
    check("P1 fails when the response and the receipt disagree",
          bad["verdict"] == "FAIL")
    bad = adjudicate(p1, _envelope(), _record(model={"model_id": "claude-3"}), names)
    check("P1 fails on the wrong model", bad["verdict"] == "FAIL")
    bad = adjudicate(p1, _envelope(),
                     _record(serving={"decision": SERVED_NEW,
                                      "principal_matched": False}), names)
    check("P1 fails when the principal did not match", bad["verdict"] == "FAIL")

    # -- P2: a role and two predicates, all proved --------------------------- #
    rec = _pop(_record(), lens="acquired",
               scope_predicates=[{"canonical_field": ROLE_FIELD,
                                  "value": "acquired"}])
    rec["execution"]["receipt"] = {"applied_predicates": [
        {"field": "erm_product_type"}, {"field": "current_loan_to_value"},
        {"field": ROLE_FIELD}]}
    check("P2 passes when all three predicates are proved",
          adjudicate(p2, _envelope(), rec, names)["verdict"] == "PASS")
    thin = json.loads(json.dumps(rec))
    thin["execution"]["receipt"]["applied_predicates"] = [
        {"field": "erm_product_type"}, {"field": ROLE_FIELD}]
    check("P2 fails when the LTV threshold is not in the receipt",
          adjudicate(p2, _envelope(), thin, names)["verdict"] == "FAIL")
    dropped = _pop(_record(), lens="all")
    dropped["execution"]["receipt"] = {"applied_predicates": [
        {"field": "erm_product_type"}, {"field": "current_loan_to_value"}]}
    out = adjudicate(p2, _envelope(), dropped, names)
    check("P2 records a silent scope DROP when the role vanished",
          out["verdict"] == "FAIL" and out["silent_scope_drop"])

    # -- P3: a named book ---------------------------------------------------- #
    named = _pop(_record(), source_reference="ALP back book",
                 source_portfolio_id="alp_acquired",
                 scope_predicates=[{"canonical_field": SOURCE_FIELD,
                                    "value": "alp_acquired"}])
    named["execution"]["receipt"] = {"applied_predicates": [{"field": SOURCE_FIELD}]}
    check("P3 passes on a resolved governed name",
          adjudicate(p3, _envelope(), named, names)["verdict"] == "PASS")
    physical = json.loads(json.dumps(named))
    physical["compiler"]["plan"]["population"]["source_reference"] = "alp_acquired"
    out = adjudicate(p3, _envelope(), physical, names)
    check("P3 fails when the model authored a physical handle",
          out["verdict"] == "FAIL" and out["model_authored_physical_source"])
    foreign = json.loads(json.dumps(named))
    foreign["compiler"]["plan"]["population"][SOURCE_FIELD] = "other_client_book"
    foreign["compiler"]["plan"]["population"]["scope_predicates"] = [
        {"canonical_field": SOURCE_FIELD, "value": "other_client_book"}]
    out = adjudicate(p3, _envelope(), foreign, names)
    check("P3 fails on a source this client's registry does not govern",
          out["verdict"] == "FAIL" and out["cross_client_source"])
    decomposed = json.loads(json.dumps(named))
    decomposed["compiler"]["plan"]["population"]["lens"] = "acquired"
    out = adjudicate(p3, _envelope(), decomposed, names)
    check("P3 fails when the book's NAME produced a role as well",
          out["verdict"] == "FAIL" and out["silent_scope_narrowing"])
    seasoned = json.loads(json.dumps(named))
    seasoned["execution"]["receipt"]["applied_predicates"].append(
        {"field": SEASONING_FIELD})
    check("P3 fails when the receipt applied a seasoning nobody asked for",
          adjudicate(p3, _envelope(), seasoned, names)["verdict"] == "FAIL")
    unbound = json.loads(json.dumps(named))
    unbound["compiler"]["plan"]["population"][SOURCE_FIELD] = None
    out = adjudicate(p3, _envelope(), unbound, names)
    check("P3 fails when no governed id was bound",
          out["verdict"] == "FAIL" and out["silent_scope_widening"])

    # -- N1: the pipeline population ----------------------------------------- #
    refused = _pop(_record(execution={"attempted": False}, serving={
        "decision": "LEGACY_FALLBACK", "principal_matched": True}),
        base="pipeline")
    refused["execution"] = {"attempted": False}
    out = adjudicate(n1, _envelope(value=None), refused, names)
    check("N1 passes when the pipeline plan never executed",
          out["verdict"] == "PASS")
    executed = _pop(_record(), base="pipeline")
    out = adjudicate(n1, _envelope(), executed, names)
    check("N1 FAILS when a pipeline plan executed over the funded runtime",
          out["verdict"] == "FAIL")
    served_pipeline = _pop(_record(execution={"attempted": False}), base="pipeline")
    served_pipeline["execution"] = {"attempted": False}
    served_pipeline["serving"] = {"decision": SERVED_NEW, "principal_matched": True}
    check("N1 fails when a pipeline plan was served at all",
          adjudicate(n1, _envelope(), served_pipeline, names)["verdict"] == "FAIL")
    wrong_base = _pop(_record(execution={"attempted": False}), base="funded")
    wrong_base["execution"] = {"attempted": False}
    check("N1 fails when the plan did not read the front book as pipeline",
          adjudicate(n1, _envelope(value=None), wrong_base, names)["verdict"]
          == "FAIL")

    # -- N2: attribution ------------------------------------------------------ #
    movement = _pop(_record(), lens="acquired")
    movement["compiler"]["plan"]["capability"] = "period_movement"
    movement["compiler"]["plan"]["operation"] = "movement"
    out = adjudicate(n2, _envelope(), movement, names)
    check("N2 passes when the movement capability survived",
          out["verdict"] == "PASS" and out["attribution_preserved"])
    degraded = _pop(_record(), lens="acquired")
    out = adjudicate(n2, _envelope(), degraded, names)
    check("N2 FAILS when the question became a plain scoped balance",
          out["verdict"] == "FAIL" and not out["attribution_preserved"])
    refused_movement = json.loads(json.dumps(movement))
    refused_movement["serving"] = {"decision": "LEGACY_FALLBACK",
                                   "principal_matched": True}
    refused_movement["execution"] = {"attempted": False}
    check("N2 passes on a REFUSAL that kept the capability",
          adjudicate(n2, _envelope(value=None), refused_movement, names)["verdict"]
          == "PASS")

    # -- inconclusive is not a pass ------------------------------------------- #
    check("a transport error is INCONCLUSIVE, never PASS",
          adjudicate(p1, {"__transport_error__": "boom"}, None, names)["verdict"]
          == "INCONCLUSIVE")
    check("a missing evidence record is INCONCLUSIVE, never PASS",
          adjudicate(p1, _envelope(), None, names)["verdict"] == "INCONCLUSIVE")

    # -- the registry names are read from the service, not written here ------- #
    check("declared names are parsed out of a governed refusal",
          _names_in("'x' is not a governed source portfolio for this book; it "
                    "governs ['alp_acquired', 'nbs_acquired']")
          == {"alp_acquired", "nbs_acquired"})

    print(f"\n  {len(failures)} failing rule(s)" if failures
          else "\n  every rule holds")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
