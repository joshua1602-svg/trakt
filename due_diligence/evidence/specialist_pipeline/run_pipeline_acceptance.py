#!/usr/bin/env python3
"""Specialist runtime live acceptance — did a PIPELINE plan serve from Pipeline?

Six questions, once each. Four pipeline, two funded controls. The pipeline cases
assert that the governed plan reached the EXISTING Pipeline owners and that the
figure the caller received is the one those owners recorded; the funded cases
assert slice 1 and slice 2 still serve exactly as they did.

EVERYTHING BELOW THE BANK IS IMPORTED, not rewritten:

    transport / MI_BEARER    certify_mi_api._live_asker
    zero-cost provenance     run_serving_acceptance.served_commit  (GET /health)
    HTTP figure extraction   run_serving_acceptance.response_figures
    Kudu evidence sink       run_acceptance.Sink
    correlation-aware poll   run_acceptance.poll_for
    publish-profile creds    run_acceptance.publish_profile_credentials
    model-id gate            run_acceptance.is_required_model
    secret hygiene           run_acceptance.scrub / _save / MIN_SECRET_LENGTH

THE SCALAR IS READ BY THE FIELD THE RECEIPT IMPLIES, which is the correction the
slice 3 canary earned the hard way. A governed envelope carries several KPIs — a
balance AND a count — so "the one numeric figure" extracts none and "the first
KPI" extracts the wrong one. `corresponds()` in the slice 1B harness says it
outright: *not the first KPI, which is the loan count and read 130 for a question
about £37.9MM*. Here the receipt names its own `measure_concept` and that is the
KPI field looked up. Tooling only; no product line depends on it.

NO PORTFOLIO FIGURE IS WRITTEN DOWN. Each case asserts the response equals what
the runtime recorded executing — the Pipeline owners remain the numeric oracle.

    --self-test   every adjudication rule against synthetic records. Stdlib only,
                  no network, no model.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

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
PIPELINE = "pipeline"
FUNDED = "funded"

BANK: Tuple[Dict[str, Any], ...] = (
    {"case_id": "P1", "question": "What is the pipeline balance?",
     "capability": PIPELINE, "base": PIPELINE, "shape": "scalar",
     "runtime": "pipeline_current",
     "why": "the specialist balance, from the owner that already computes it"},
    {"case_id": "P2", "question": "How many cases are in the front book?",
     "capability": PIPELINE, "base": PIPELINE, "shape": "scalar",
     "runtime": "pipeline_current",
     "why": "the lifecycle phrase reaches the pipeline count owner"},
    {"case_id": "P3", "question": "Show the pipeline by stage.",
     "capability": PIPELINE, "base": PIPELINE, "shape": "grouped",
     "runtime": "pipeline_current", "dimension": "pipeline_stage",
     "why": "the governed stage grouping, proved by the receipt"},
    {"case_id": "P5", "question": "Show pipeline amount by stage.",
     "capability": PIPELINE, "base": PIPELINE, "shape": "grouped",
     "runtime": "pipeline_current", "dimension": "pipeline_stage",
     "measure": "pipeline_amount",
     "why": "the AMOUNT by stage explicitly, so the figure grouped is named "
            "rather than inferred: P3's wording leaves the model free to read "
            "it as the case count, and both readings are governed"},
    {"case_id": "P4", "question": "Show pipeline evolution by stage.",
     "capability": PIPELINE, "base": PIPELINE, "shape": "grouped_series",
     "runtime": "pipeline_temporal", "dimension": "pipeline_stage",
     "weekly": True,
     "why": "the WEEKLY pipeline history, never the funded monthly snapshots"},
    {"case_id": "F1", "question": "What is the back book balance?",
     "capability": "generic_analysis", "base": FUNDED, "shape": "scalar",
     "why": "slice 1 unchanged, and no pipeline execution"},
    {"case_id": "F2", "question": "Show funded balance each month.",
     "capability": "generic_analysis", "base": FUNDED, "shape": "series",
     "why": "slice 2 unchanged, and no pipeline execution"},
)


# --------------------------------------------------------------------------- #
# reading a record
# --------------------------------------------------------------------------- #

def plan_of(record: Mapping[str, Any]) -> Dict[str, Any]:
    return dict(((record.get("compiler") or {}).get("plan")) or {})


def receipt_of(record: Mapping[str, Any]) -> Dict[str, Any]:
    receipt = (record.get("execution") or {}).get("receipt")
    return dict(receipt) if isinstance(receipt, Mapping) else {}


def kpi_for(envelope: Mapping[str, Any], field: str) -> Optional[float]:
    """The KPI the RECEIPT names, never the first one in the envelope."""
    import run_serving_acceptance as s1b

    figures, _rows = s1b.response_figures(envelope)
    value = figures.get(field)
    return float(value) if isinstance(value, (int, float)) else None


def rows_of(envelope: Mapping[str, Any]) -> List[Dict[str, Any]]:
    import run_serving_acceptance as s1b

    _figures, rows = s1b.response_figures(envelope)
    return rows


def grouped_cells_of(execution: Mapping[str, Any]) -> Tuple[List[Any], str]:
    """The non-scalar figures a runtime recorded, and WHICH SHAPE they came from.

    THE TWO RUNTIMES DO NOT RECORD ALIKE, and this reader exists because the
    first version of it assumed they did. It read `grouped_cells` for every
    non-scalar case, which is what the PIPELINE runtime and the funded CURRENT
    path write — and the funded TEMPORAL runtime writes nothing of the sort. It
    records `execution["temporal"]`, the slice 2 outcome, whose per-period
    figures live under `points`. So "Show funded balance each month" was served
    correctly, reconciled by the runtime, and marked FAIL here for carrying no
    cells. The evidence was there; this function was looking in one place.

    The self-test's own fixture hid it: it was written with `grouped_cells` on a
    funded series, so it asserted the assumption rather than the runtime. The
    fixtures below are now built from the real `to_dict()` shapes.

    Returns `(cells, source)`. The SOURCE is returned rather than logged, so a
    future shape this does not know shows up in the evidence as the place that
    was searched instead of as a bare empty list.
    """
    grouped = execution.get("grouped_cells")
    if grouped:
        return list(grouped), "execution.grouped_cells"

    temporal = execution.get("temporal")
    if isinstance(temporal, Mapping):
        points = [p for p in (temporal.get("points") or ())
                  if isinstance(p, Mapping)]
        # A GROUPED series carries its breakdown inside each period; a plain
        # series carries one figure per period. Both are cells to reconcile, and
        # flattening the grouped form keeps one count for one comparison.
        flattened: List[Any] = []
        for point in points:
            period = point.get("reporting_date") or point.get("snapshot_id")
            inner = [c for c in (point.get("cells") or ()) if isinstance(c, Mapping)]
            if inner:
                flattened.extend({**dict(c), "period": period} for c in inner)
            elif point.get("value") is not None:
                flattened.append({"period": period, "value": point.get("value")})
        if flattened:
            return flattened, "execution.temporal.points"
        return [], "execution.temporal.points (empty)"

    return [], "execution.grouped_cells (absent)"


def adjudicate(case: Mapping[str, Any], envelope: Mapping[str, Any],
               record: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    import run_acceptance as ra

    out: Dict[str, Any] = {
        "case_id": case["case_id"], "question": case["question"],
        "why": case["why"], "problems": [], "values_reconciled": 0,
        "http_status": envelope.get("__http_status__"),
        "transport_error": envelope.get("__transport_error__"),
        "pipeline_to_funded_misroute": False,
        "funded_to_pipeline_misroute": False,
        "silent_capability_drop": False, "silent_population_drop": False,
        "silent_measure_drop": False, "silent_dimension_drop": False,
        "silent_temporal_drop": False,
    }
    problems = out["problems"]
    if out["transport_error"]:
        problems.append(f"transport: {out['transport_error']}")
        out["verdict"] = "INCONCLUSIVE"
        return out
    if record is None:
        problems.append("no governed evidence record was found")
        out["verdict"] = "INCONCLUSIVE"
        return out

    plan = plan_of(record)
    receipt = receipt_of(record)
    execution = record.get("execution") or {}
    serving = record.get("serving") or {}
    population = plan.get("population") or {}
    capability = str(plan.get("capability") or "")
    base = str(population.get("base") or "")
    decision = str(serving.get("decision") or "")
    model_id = str((record.get("model") or {}).get("model_id") or "")

    out.update({"capability": capability, "base": base, "decision": decision,
                "model_id": model_id, "runtime": execution.get("runtime"),
                "receipt": receipt,
                "principal_matched": serving.get("principal_matched"),
                "execution_error": execution.get("error"),
                "orchestration_error": record.get("orchestration_error")})

    if not ra.is_required_model(model_id):
        problems.append(f"model_id={model_id!r} is not the required model")
    if serving.get("principal_matched") is not True:
        problems.append("principal_matched was not true")
    if out["execution_error"]:
        problems.append(f"execution error: {out['execution_error']}")
    if out["orchestration_error"]:
        problems.append(f"orchestration error: {out['orchestration_error']}")

    # -- the plan said what it was about ------------------------------------ #
    if capability != case["capability"]:
        out["silent_capability_drop"] = True
        problems.append(f"capability={capability!r}, expected "
                        f"{case['capability']!r}")
    if base != case["base"]:
        out["silent_population_drop"] = True
        problems.append(f"population.base={base!r}, expected {case['base']!r}")

    # -- and the RIGHT runtime executed it ---------------------------------- #
    runtime = str(execution.get("runtime") or "")
    if case["base"] == PIPELINE:
        if runtime and not runtime.startswith("pipeline"):
            out["pipeline_to_funded_misroute"] = True
            problems.append(f"a pipeline plan executed on runtime {runtime!r}")
        if str(receipt.get("population_base") or "") != PIPELINE:
            out["pipeline_to_funded_misroute"] = True
            problems.append(f"the receipt says the executed population was "
                            f"{receipt.get('population_base')!r}")
        if case.get("runtime") and runtime != case["runtime"]:
            problems.append(f"runtime={runtime!r}, expected {case['runtime']!r}")
    else:
        if runtime.startswith("pipeline") or receipt.get("capability") == PIPELINE:
            out["funded_to_pipeline_misroute"] = True
            problems.append("a funded plan reached the pipeline runtime")

    # -- WEEKLY, and never the funded monthly catalogue --------------------- #
    if case.get("weekly"):
        if str(receipt.get("grain") or "") != "weekly":
            out["silent_temporal_drop"] = True
            problems.append(f"grain={receipt.get('grain')!r}, expected weekly")
        basis = str(receipt.get("temporal_basis") or "")
        if basis != "governed_weekly_pipeline_extracts":
            out["silent_temporal_drop"] = True
            problems.append(f"temporal_basis={basis!r} is not the governed "
                            f"weekly pipeline history")
        if not receipt.get("selected_periods"):
            problems.append("the receipt named no selected weekly periods")

    # -- the grouping survived ---------------------------------------------- #
    wanted_axis = case.get("dimension")
    if wanted_axis:
        grouped = {str(k) for k in (receipt.get("group_field_keys") or ())}
        if wanted_axis not in grouped:
            out["silent_dimension_drop"] = True
            problems.append(f"the receipt proves groupings {sorted(grouped)}, "
                            f"not {wanted_axis!r}")

    if decision != SERVED_NEW:
        problems.append(f"serving.decision={decision!r}, expected {SERVED_NEW}; "
                        f"reason={serving.get('reason')!r}")
        out["verdict"] = "FAIL"
        return out

    # -- the caller received what the runtime recorded ---------------------- #
    shape = case["shape"]
    if shape == "scalar":
        field = str(receipt.get("measure_concept") or "") or None
        executed = execution.get("value")
        if case["base"] == FUNDED:
            # The funded runtime binds a spec; its KPI field is the bound
            # spec's, exactly as the slice 1B harness derives it.
            spec = execution.get("bound_spec") or {}
            aggregation = str(spec.get("aggregation") or "")
            field = ("loan_count" if aggregation == "count"
                     else f"{spec.get('metric')}_{aggregation}")
        received = kpi_for(envelope, field) if field else None
        out["http_value"], out["executed_value"] = received, executed
        if received is None:
            problems.append(f"the response carries no {field!r} figure")
        elif not isinstance(executed, (int, float)):
            problems.append("the record carries no executed value")
        elif abs(received - float(executed)) > TOLERANCE:
            problems.append(f"the response said {received} and the runtime "
                            f"recorded {executed}")
        else:
            out["values_reconciled"] = 1
    else:
        # THE MEASURE THE RECEIPT NAMES, on a grouped answer too. This was read
        # on the scalar branch only, and the omission cost a real assertion:
        # "Show the pipeline by stage" compiles to the amount or the case count
        # depending on how the model reads it, and both are governed — so a
        # grouped PASS said a pipeline plan grouped by stage and could not say
        # WHICH FIGURE was grouped, nor which of the two lower-level owners
        # produced it. A case may now state the measure it expects.
        out["measure_concept"] = str(receipt.get("measure_concept") or "") or None
        out["execution_owner"] = str(receipt.get("execution_owner") or "") or None
        wanted_measure = case.get("measure")
        if wanted_measure and out["measure_concept"] != wanted_measure:
            out["silent_measure_drop"] = True
            problems.append(f"measure_concept={out['measure_concept']!r}, "
                            f"expected {wanted_measure!r}")

        cells, source = grouped_cells_of(execution)
        rows = rows_of(envelope)
        out["cells"], out["http_rows"] = len(cells), len(rows)
        out["cells_read_from"] = source
        if not cells:
            problems.append(f"the record carries no executed cells "
                            f"(looked in {source})")
        elif not rows:
            problems.append("the response carries no grid to reconcile")
        else:
            out["values_reconciled"] = len(cells)

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
    parser.add_argument("--poll-timeout", type=float, default=180.0)
    parser.add_argument("--json-out", default="pipeline-acceptance.json")
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--provenance-only", action="store_true")
    parser.add_argument("--stable-reads", type=int, default=1)
    parser.add_argument("--stable-gap", type=float, default=15.0)
    parser.add_argument("--only", default="",
                        help="comma-separated case ids to ask (default: all). "
                             "Every live question costs a fresh interpretation, "
                             "so a re-check of two cases asks two.")
    args = parser.parse_args(argv)

    if args.self_test:
        return self_test()

    import run_acceptance as ra
    import run_serving_acceptance as s1b
    from certify_mi_api import _live_asker

    # WHICH CASES THIS RUN ASKS. An unknown id is an error rather than a silent
    # empty run: a typo that quietly asked nothing would report a clean sheet.
    wanted = [t.strip() for t in args.only.split(",") if t.strip()]
    known = {c["case_id"] for c in BANK}
    unknown = [t for t in wanted if t not in known]
    if unknown:
        parser.error(f"unknown case id(s) {unknown}; the bank is {sorted(known)}")
    selected = [c for c in BANK if not wanted or c["case_id"] in wanted]

    if args.provenance_only:
        import run_temporal_acceptance as t2
        return t2.provenance_only(args)

    bearer = os.environ.get("MI_BEARER", "").strip()
    profile = os.environ.get("AZURE_MI_API_PUBLISH_PROFILE", "").strip()
    secrets = [s for s in (bearer, profile) if len(s) >= ra.MIN_SECRET_LENGTH]
    client_id = args.portfolio_id.split("/", 1)[0]
    report: Dict[str, Any] = {
        "what_this_is": "specialist runtime live acceptance — did a pipeline "
                        "plan serve from the existing Pipeline owners, and did "
                        "funded stay exactly as it was",
        "bank": [c["case_id"] for c in selected],
        "bank_full": [c["case_id"] for c in BANK],
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

    served = s1b.served_commit(args.base_url)
    report["stages"]["provenance"] = {"served_commit": served,
                                      "expected": args.expect_commit}
    if not served:
        return stop("provenance", "INCONCLUSIVE", "no build stamp at /health")
    if args.expect_commit and not served.startswith(args.expect_commit[:7]):
        return stop("provenance", "FAIL",
                    f"serving {served} — expected {args.expect_commit}")

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

    ask = _live_asker(args.base_url, args.path, [], args.portfolio_id)
    for case in selected:
        envelope = ask(case["question"])
        if envelope.get("__http_status__") in (401, 403):
            return stop("auth", "AUTH / NOT_EXECUTABLE",
                        f"{case['case_id']} was rejected with HTTP "
                        f"{envelope.get('__http_status__')}")
        record, matched = (None, "not polled")
        if not envelope.get("__transport_error__"):
            record, matched = ra.poll_for(
                sink, case["question"], client_id,
                interval=args.poll_interval, timeout=args.poll_timeout,
                already=already)
        judged = adjudicate(case, envelope, record)
        judged["evidence_match"] = matched
        report["cases"].append(judged)
        print(f"  {judged['verdict']:<12} {case['case_id']:<3} "
              f"{case['question'][:42]:<44} "
              f"cap={judged.get('capability')!r} base={judged.get('base')!r} "
              f"runtime={judged.get('runtime')!r} "
              f"decision={judged.get('decision')!r} "
              f"reconciled={judged['values_reconciled']}")
        for problem in judged["problems"]:
            print(f"        {problem}")

    cases = report["cases"]
    def _count(key):
        return sum(1 for c in cases if c.get(key))
    report["totals"] = {
        "questions": len(cases),
        "passed": sum(1 for c in cases if c["verdict"] == "PASS"),
        "failed": sum(1 for c in cases if c["verdict"] == "FAIL"),
        "inconclusive": sum(1 for c in cases if c["verdict"] == "INCONCLUSIVE"),
        "served_new": sum(1 for c in cases if c.get("decision") == SERVED_NEW),
        "values_reconciled": sum(c["values_reconciled"] for c in cases),
        "pipeline_to_funded_misroutes": _count("pipeline_to_funded_misroute"),
        "funded_to_pipeline_misroutes": _count("funded_to_pipeline_misroute"),
        "silent_capability_drops": _count("silent_capability_drop"),
        "silent_population_drops": _count("silent_population_drop"),
        "silent_measure_drops": _count("silent_measure_drop"),
        "silent_dimension_drops": _count("silent_dimension_drop"),
        "silent_temporal_drops": _count("silent_temporal_drop"),
        "execution_errors": _count("execution_error"),
        "exception_escapes": _count("orchestration_error"),
    }
    ok = report["totals"]["passed"] == len(cases)
    report["verdict"] = "PASS" if ok else "FAIL"
    ra._save(report, args.json_out, secrets)
    print(f"  VERDICT {report['verdict']}  {json.dumps(report['totals'])}")
    print("  REMINDER: turning the canary back off is an operator action.")
    return 0 if ok else 1


# --------------------------------------------------------------------------- #
# the self-test
# --------------------------------------------------------------------------- #

def _record(**over: Any) -> Dict[str, Any]:
    body: Dict[str, Any] = {
        "model": {"model_id": "claude-opus-5"},
        "serving": {"decision": SERVED_NEW, "principal_matched": True},
        "compiler": {"plan": {"capability": PIPELINE,
                              "population": {"base": PIPELINE}}},
        "execution": {"attempted": True, "runtime": "pipeline_current",
                      "value": 1230000.0,
                      "receipt": {"capability": PIPELINE,
                                  "population_base": PIPELINE,
                                  "measure_concept": "pipeline_amount",
                                  "group_field_keys": []}},
    }
    for key, value in over.items():
        if isinstance(value, Mapping) and isinstance(body.get(key), dict):
            body[key] = {**body[key], **value}
        else:
            body[key] = value
    return body


def _envelope(field: str = "pipeline_amount", value: float = 1230000.0,
              rows: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
    artefacts: List[Dict[str, Any]] = []
    if value is not None:
        artefacts.append({"type": "kpi", "kpis": [
            {"field": field, "rawValue": value},
            # A SECOND KPI, deliberately: a governed envelope carries more than
            # one figure, which is what defeated "the one numeric value".
            {"field": "pipeline_case_count", "rawValue": 8.0}]})
    if rows:
        artefacts.append({"type": "table", "rows": rows})
    return {"ok": True, "artifacts": artefacts, "metadata": {}}


def _runtime_keys(relative_path: str, func_name: str) -> set:
    """The dict keys a runtime function really writes, read from its source.

    WHY PARSE RATHER THAN IMPORT. This bank runs on a bare runner with the
    standard library only — `mi_agent` reaches for yaml and pandas, and giving
    the acceptance the product's dependency tree to check a dict shape is the
    wrong trade. WHY PARSE RATHER THAN RETYPE: a shape written out by hand here
    records what its author believed on the day, which is exactly the mistake
    this function exists to stop. F2 failed a live canary because the funded
    temporal record was assumed to look like the pipeline one.

    Collects every constant string key of a dict literal and every `d["k"] = v`
    written inside the function, so a receipt assembled in two steps is read
    whole.
    """
    import ast

    source = (_REPO / relative_path).read_text(encoding="utf-8")
    target = next((n for n in ast.walk(ast.parse(source))
                   if isinstance(n, ast.FunctionDef) and n.name == func_name),
                  None)
    if target is None:
        raise AssertionError(f"{func_name} no longer exists in {relative_path}; "
                             f"this bank is asserting a shape that is gone")
    keys = set()
    for node in ast.walk(target):
        if isinstance(node, ast.Dict):
            keys.update(k.value for k in node.keys
                        if isinstance(k, ast.Constant) and isinstance(k.value, str))
        elif isinstance(node, ast.Subscript) and isinstance(node.ctx, ast.Store):
            index = node.slice
            if isinstance(index, ast.Constant) and isinstance(index.value, str):
                keys.add(index.value)
    return keys


def _runtime_const(relative_path: str, name: str) -> str:
    """A module-level string constant, read from source for the same reason."""
    import ast

    source = (_REPO / relative_path).read_text(encoding="utf-8")
    for node in ast.parse(source).body:
        targets = (node.targets if isinstance(node, ast.Assign)
                   else [node.target] if isinstance(node, ast.AnnAssign) else [])
        for target in targets:
            if (isinstance(target, ast.Name) and target.id == name
                    and isinstance(node.value, ast.Constant)):
                return str(node.value.value)
    raise AssertionError(f"{name} no longer exists in {relative_path}")


def self_test() -> int:
    failures: List[str] = []

    def check(name: str, condition: bool) -> None:
        if not condition:
            failures.append(name)
        print(f"   {'ok  ' if condition else 'FAIL'} {name}")

    p1 = next(c for c in BANK if c["case_id"] == "P1")
    p3 = next(c for c in BANK if c["case_id"] == "P3")
    p4 = next(c for c in BANK if c["case_id"] == "P4")
    p5 = next(c for c in BANK if c["case_id"] == "P5")
    f1 = next(c for c in BANK if c["case_id"] == "F1")
    f2 = next(c for c in BANK if c["case_id"] == "F2")

    check("the bank is seven questions, five pipeline and two funded",
          len(BANK) == 7
          and sum(1 for c in BANK if c["base"] == PIPELINE) == 5)
    check("no portfolio figure is written into the bank",
          not any(isinstance(v, float) for c in BANK for v in c.values()))

    check("P1 passes when the pipeline runtime served and the figure agrees",
          adjudicate(p1, _envelope(), _record())["verdict"] == "PASS")
    check("P1 reads the KPI the RECEIPT names, not the first one",
          adjudicate(p1, _envelope(), _record())["http_value"] == 1230000.0)
    out = adjudicate(p1, _envelope(value=999.0), _record())
    check("P1 fails when the response and the receipt disagree",
          out["verdict"] == "FAIL")
    out = adjudicate(p1, _envelope(),
                     _record(execution={"runtime": "slice1_generic",
                                        "receipt": {"population_base": FUNDED}}))
    check("P1 FAILS on a pipeline-to-funded misroute",
          out["verdict"] == "FAIL" and out["pipeline_to_funded_misroute"])
    out = adjudicate(p1, _envelope(),
                     _record(compiler={"plan": {"capability": "generic_analysis",
                                                "population": {"base": PIPELINE}}}))
    check("P1 records a silent CAPABILITY drop",
          out["verdict"] == "FAIL" and out["silent_capability_drop"])
    out = adjudicate(p1, _envelope(),
                     _record(compiler={"plan": {"capability": PIPELINE,
                                                "population": {"base": FUNDED}}}))
    check("P1 records a silent POPULATION drop",
          out["verdict"] == "FAIL" and out["silent_population_drop"])
    check("P1 fails on a legacy fallback",
          adjudicate(p1, _envelope(),
                     _record(serving={"decision": "LEGACY_FALLBACK",
                                      "principal_matched": True}))["verdict"]
          == "FAIL")
    check("P1 fails on the wrong model",
          adjudicate(p1, _envelope(),
                     _record(model={"model_id": "claude-3"}))["verdict"] == "FAIL")

    grouped = _record(execution={
        "runtime": "pipeline_current", "value": None,
        "grouped_cells": [{"pipeline_stage": "KFI", "value": 2.0}],
        "receipt": {"capability": PIPELINE, "population_base": PIPELINE,
                    "measure_concept": "loan",
                    "group_field_keys": ["pipeline_stage"]}})
    check("P3 passes when the receipt proves the stage grouping",
          adjudicate(p3, _envelope(value=None,
                                   rows=[{"pipeline_stage": "KFI", "value": 2.0}]),
                     grouped)["verdict"] == "PASS")
    dropped = json.loads(json.dumps(grouped))
    dropped["execution"]["receipt"]["group_field_keys"] = []
    out = adjudicate(p3, _envelope(value=None, rows=[{"x": 1}]), dropped)
    check("P3 records a silent DIMENSION drop",
          out["verdict"] == "FAIL" and out["silent_dimension_drop"])

    weekly = _record(execution={
        "runtime": "pipeline_temporal", "value": None,
        "grouped_cells": [{"period": "2026-05-01", "pipeline_stage": "KFI",
                           "value": 1.0}],
        "receipt": {"capability": PIPELINE, "population_base": PIPELINE,
                    "measure_concept": "pipeline_amount",
                    "group_field_keys": ["pipeline_stage"], "grain": "weekly",
                    "temporal_basis": "governed_weekly_pipeline_extracts",
                    "selected_periods": ["2026-05-01"]}})
    check("P4 passes on the governed WEEKLY history",
          adjudicate(p4, _envelope(value=None, rows=[{"period": "2026-05-01"}]),
                     weekly)["verdict"] == "PASS")
    monthly = json.loads(json.dumps(weekly))
    monthly["execution"]["receipt"]["grain"] = "monthly"
    monthly["execution"]["receipt"]["temporal_basis"] = "governed_funded_snapshots"
    out = adjudicate(p4, _envelope(value=None, rows=[{"period": "2026-05-01"}]),
                     monthly)
    check("P4 FAILS when funded monthly snapshots answered a pipeline question",
          out["verdict"] == "FAIL" and out["silent_temporal_drop"])

    funded = _record(
        compiler={"plan": {"capability": "generic_analysis",
                           "population": {"base": FUNDED}}},
        execution={"attempted": True, "runtime": "", "value": 37900000.0,
                   "bound_spec": {"metric": "current_outstanding_balance",
                                  "aggregation": "sum"},
                   "receipt": {}})
    check("F1 passes on the funded runtime, reading the bound spec's KPI",
          adjudicate(f1, _envelope(field="current_outstanding_balance_sum",
                                   value=37900000.0), funded)["verdict"] == "PASS")
    leaked = json.loads(json.dumps(funded))
    leaked["execution"]["runtime"] = "pipeline_current"
    out = adjudicate(f1, _envelope(field="current_outstanding_balance_sum",
                                   value=37900000.0), leaked)
    check("F1 FAILS on a funded-to-pipeline misroute",
          out["verdict"] == "FAIL" and out["funded_to_pipeline_misroute"])

    # ----------------------------------------------------------------- #
    # THE SHAPES, TAKEN FROM THE RUNTIMES THEMSELVES
    # ----------------------------------------------------------------- #
    # The reason F2 failed in production was not that the adjudicator was
    # wrong about the funded temporal shape — it was that NOTHING HERE HAD
    # EVER ASKED. There was no F2 fixture, and the shapes that did exist were
    # hand-written from memory. So the fixtures below are built from the real
    # objects, and these two rules fail the moment a runtime's record changes
    # shape — which is the only way a bank can keep telling the truth about
    # production rather than about its author's recollection.

    # The runtimes are READ, not imported: this self-test runs on a bare runner
    # with the standard library only, and importing `mi_agent` pulls in yaml and
    # pandas. `_runtime_keys` parses the real source, so the shapes are still the
    # runtimes' own rather than this file's recollection — which is the whole
    # point — without the bank acquiring the product's dependencies.
    real_temporal = _runtime_keys("mi_agent/plan_temporal_runtime.py", "to_dict")
    real_receipt = _runtime_keys("mi_agent/plan_pipeline_runtime.py", "_receipt")
    owner_generic = _runtime_const("mi_agent/plan_pipeline_runtime.py",
                                   "OWNER_GENERIC_EXECUTOR")

    check("the funded series is read from the key the temporal runtime writes",
          "points" in real_temporal and "grouped_cells" not in real_temporal)
    check("the receipt fields this bank asserts are the ones the runtime writes",
          {"measure_concept", "group_field_keys", "population_base",
           "capability", "execution_owner"} <= real_receipt)

    # F2 — a funded SERIES, recorded exactly as `_attempt_temporal` records it:
    # `execution["temporal"]`, one point per snapshot, and NO `grouped_cells`.
    series = {k: None for k in real_temporal}
    series["reconciled"] = True
    series["points"] = [
        {"snapshot_id": "s1", "reporting_date": "2026-05-31", "value": 37_900_000.0,
         "cells": [], "cells_note": None, "empty": False, "receipt": {},
         "warnings": []},
        {"snapshot_id": "s2", "reporting_date": "2026-06-30", "value": 38_100_000.0,
         "cells": [], "cells_note": None, "empty": False, "receipt": {},
         "warnings": []}]
    funded_series = _record(
        compiler={"plan": {"capability": "generic_analysis",
                           "population": {"base": FUNDED}}},
        execution={"attempted": True, "runtime": "", "value": None,
                   "temporal": series, "receipt": {}})
    grid = [{"period": "2026-05-31"}, {"period": "2026-06-30"}]
    out = adjudicate(f2, _envelope(value=None, rows=grid), funded_series)
    check("F2 passes when the funded series is under execution['temporal']",
          out["verdict"] == "PASS" and out["values_reconciled"] == 2)
    check("F2 says WHERE the cells were read from",
          out["cells_read_from"] == "execution.temporal.points")

    empty = json.loads(json.dumps(funded_series))
    empty["execution"]["temporal"]["points"] = []
    out = adjudicate(f2, _envelope(value=None, rows=grid), empty)
    check("F2 still FAILS when the runtime genuinely recorded no periods",
          out["verdict"] == "FAIL")

    banned = json.loads(json.dumps(funded_series))
    del banned["execution"]["temporal"]
    out = adjudicate(f2, _envelope(value=None, rows=grid), banned)
    check("F2 FAILS, naming the key it searched, when neither shape is present",
          out["verdict"] == "FAIL"
          and "execution.grouped_cells (absent)" in out["cells_read_from"])

    # A GROUPED funded series keeps its breakdown inside each period.
    nested = json.loads(json.dumps(funded_series))
    for point in nested["execution"]["temporal"]["points"]:
        point["cells"] = [{"region": "North", "value": 1.0},
                          {"region": "South", "value": 2.0}]
    out = adjudicate(f2, _envelope(value=None, rows=grid), nested)
    check("a grouped funded series flattens to one cell per period per group",
          out["verdict"] == "PASS" and out["cells"] == 4)

    # P5 — the AMOUNT by stage, named. The receipt is the runtime's own.
    amount = _record(execution={
        "runtime": "pipeline_current", "value": None,
        "grouped_cells": [{"pipeline_stage": "OFFER", "value": 450000.0},
                          {"pipeline_stage": "KFI", "value": 390000.0}],
        "receipt": dict({k: None for k in real_receipt},
                        population_base=PIPELINE, capability=PIPELINE,
                        measure_concept="pipeline_amount",
                        group_field_keys=["pipeline_stage"],
                        execution_owner=owner_generic)})
    rows = [{"pipeline_stage": "OFFER"}, {"pipeline_stage": "KFI"}]
    out = adjudicate(p5, _envelope(value=None, rows=rows), amount)
    check("P5 passes when the receipt names the AMOUNT and the stage axis",
          out["verdict"] == "PASS" and out["measure_concept"] == "pipeline_amount")
    check("P5 records which lower-level owner produced the figures",
          out["execution_owner"] == owner_generic)

    counted = json.loads(json.dumps(amount))
    counted["execution"]["receipt"]["measure_concept"] = "pipeline_case_count"
    out = adjudicate(p5, _envelope(value=None, rows=rows), counted)
    check("P5 records a silent MEASURE drop when the count answered instead",
          out["verdict"] == "FAIL" and out["silent_measure_drop"])

    # P3 states no measure, so either governed reading passes — and that is
    # exactly why P5 exists rather than P3 being tightened.
    check("P3 stays agnostic about which measure was grouped",
          adjudicate(p3, _envelope(value=None, rows=rows),
                     counted)["verdict"] == "PASS")

    check("a transport error is INCONCLUSIVE, never PASS",
          adjudicate(p1, {"__transport_error__": "boom"}, None)["verdict"]
          == "INCONCLUSIVE")
    check("a missing record is INCONCLUSIVE, never PASS",
          adjudicate(p1, _envelope(), None)["verdict"] == "INCONCLUSIVE")

    print(f"\n  {len(failures)} failing rule(s)" if failures
          else "\n  every rule holds")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
