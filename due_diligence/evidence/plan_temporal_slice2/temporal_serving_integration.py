#!/usr/bin/env python3
"""Slice 2 serving integration, proved through the REAL serving call path. No calls.

WHAT MAKES THIS DIFFERENT FROM THE OFFLINE ACCEPTANCE. `slice2_acceptance.py`
calls `plan_temporal_runtime.execute_temporal_plan` directly. That proves the
runtime and says nothing about whether a question can REACH it. This calls
`plan_serving_canary.serve` — the production entry point — with a canary
principal, and lets the real path do the rest:

    serve -> handles(context)            the real principal allow-list
          -> wiring.build_plan           the real interpreter seam, replayed
          -> DeterministicCompiler       the real compiler
          -> temporal.claims(plan)       the real dispatch
          -> check_temporal_eligibility  the real slice 2 perimeter
          -> execute_temporal_plan       the real runtime, per snapshot
          -> render / adapt_workflow_result   the real response contract
          -> the API payload a caller would receive

NO MODEL CALLS. `wiring.set_interpreter_factory` is the sanctioned offline seam;
the payloads replayed through it are the ones Opus actually emitted in the
recorded live runs, so the intents under test are the model's own.

THE ORACLE IMPORTS NOTHING FROM THE PRODUCT. Expected figures come from
`portfolio_truth_oracle`; expected snapshots are written out by hand below. Both
are compared against the FINAL PAYLOAD, not against the runtime's own return
value — an answer that reconciles internally and loses the figures on the way
out is exactly the failure a serving proof is for.

Run: `python due_diligence/evidence/plan_temporal_slice2/temporal_serving_integration.py`
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from mi_agent import plan_serving_canary as canary                     # noqa: E402
from mi_agent import plan_shadow_wiring as wiring                      # noqa: E402
from mi_agent import plan_temporal_runtime as temporal                 # noqa: E402
from mi_agent.interpretation_v2.opus_interpreter import (               # noqa: E402
    OpusInterpreter, ReplayClient)
from mi_agent.mi_query_validator import load_mi_semantics              # noqa: E402
from mi_agent.tests import portfolio_truth_oracle as truth             # noqa: E402
from mi_agent.tests import temporal_snapshot_fixture as fixture        # noqa: E402

HERE = Path(__file__).resolve().parent
OUT = HERE / "temporal_serving_integration.json"
REGISTRY = _REPO_ROOT / "mi_agent" / "mi_semantics_field_registry.yaml"

#: The recorded live runs. Payloads are taken from the RETEST first (it carries
#: the post-correction capability routing) and from the original run for the
#: cases the retest did not re-ask.
RETEST = HERE / "temporal_live_retest_result.json"
ORIGINAL = HERE / "temporal_live_run_result.json"

CANARY_PRINCIPAL = "00000000-1111-2222-3333-444444444444"
NON_CANARY_PRINCIPAL = "99999999-8888-7777-6666-555555555555"

CATALOGUE = ["2025-11-30", "2025-12-31", "2026-01-31", "2026-02-28",
             "2026-03-31", "2026-04-30", "2026-05-31", "2026-06-30"]


# --------------------------------------------------------------------------- #
# the bank: five positives and six controls, all through `serve`
# --------------------------------------------------------------------------- #

POSITIVES: Tuple[Dict[str, Any], ...] = (
    {"id": "S2-1", "case": "T01", "why": "funded balance over last N months",
     "snapshots": CATALOGUE[-6:],
     "oracle": {"how": "sum", "column": truth.BALANCE}},
    {"id": "S2-2", "case": "T04", "why": "loan count each month",
     "snapshots": list(CATALOGUE), "oracle": {"how": "count"}},
    {"id": "S2-3", "case": "T06", "why": "filtered temporal series",
     "snapshots": list(CATALOGUE),
     "oracle": {"how": "count",
                "predicates": [("erm_product_type", "eq", "Drawdown")]},
     "filters": ["erm_product_type"]},
    {"id": "S2-4", "case": "T07", "why": "time x one supported dimension",
     "snapshots": list(CATALOGUE),
     "oracle": {"how": "count", "group_by": ["ltv_bucket"]},
     "dimensions": ["ltv_bucket"]},
    {"id": "S2-5", "case": "T11", "why": "current versus previous period",
     "snapshots": CATALOGUE[-2:],
     "oracle": {"how": "sum", "column": truth.BALANCE},
     "comparison": True},
)

CONTROLS: Tuple[Dict[str, Any], ...] = (
    {"id": "N-1", "case": "T10", "why": "T10 — generic_analysis + movement",
     "expect": "LEGACY", "expect_reason_startswith": "INELIGIBLE:",
     "must_not_execute": True},
    {"id": "N-2", "case": "T13", "why": "unavailable requested period",
     "expect": "LEGACY", "expect_reason_startswith": "TEMPORAL_NOT_RESOLVED:",
     "must_not_execute": True},
    {"id": "N-3", "case": "T13", "why": "overlong period does not shorten",
     "expect": "LEGACY", "expect_reason_startswith": "TEMPORAL_NOT_RESOLVED:",
     "must_not_execute": True, "no_snapshots": True},
    {"id": "N-4", "case": "T14", "why": "movement attribution",
     "expect": "LEGACY", "expect_reason_startswith": "INELIGIBLE:",
     "must_not_execute": True},
    {"id": "N-5", "case": "T15", "why": "multi-output temporal request",
     "expect": "LEGACY", "expect_reason_startswith": "INELIGIBLE:",
     "must_not_execute": True},
    {"id": "N-6", "case": "T01", "why": "non-canary principal",
     "expect": "NOT_HANDLED", "non_canary": True},
)


def load_payloads() -> Dict[str, Dict[str, Any]]:
    """`{question: raw_payload}` from the recorded runs, retest winning."""
    payloads: Dict[str, Dict[str, Any]] = {}
    questions: Dict[str, str] = {}
    for path in (ORIGINAL, RETEST):                      # retest overwrites
        body = json.loads(path.read_text(encoding="utf-8"))
        for row in body["results"]:
            if row.get("raw_payload"):
                payloads[row["question"]] = row["raw_payload"]
                questions[row["id"]] = row["question"]
    return payloads, questions


def oracle(recipe: Mapping[str, Any], frame: Any) -> Any:
    predicates = list(recipe.get("predicates") or ())
    group_by = list(recipe.get("group_by") or ())
    how = recipe["how"]
    if group_by:
        return truth.grouped(frame, group_by, column=recipe.get("column"),
                             how=how, predicates=predicates)
    if how == "count":
        return float(truth.row_count(frame, predicates))
    return truth.total(frame, recipe["column"], predicates)


def payload_series(payload: Mapping[str, Any], column: str,
                   dimensions: Sequence[str]) -> List[Dict[str, Any]]:
    """The series as it reaches a CALLER, read back out of the API envelope.

    Deliberately read from the artifact rows the response ships, not from the
    runtime's own object: a figure that reconciles inside the engine and is lost
    on the way out is the failure this whole proof exists to catch.
    """
    for artifact in (payload.get("artifacts") or ()):
        rows = artifact.get("rows") or artifact.get("data")
        if not rows:
            continue
        first = rows[0]
        if isinstance(first, Mapping) and temporal.REPORTING_DATE in first:
            return [dict(r) for r in rows]
    return []


# --------------------------------------------------------------------------- #
# one case, through `serve`
# --------------------------------------------------------------------------- #

def serve_case(question: str, *, store: Any, semantics: Any, frames: Any,
               principal: str) -> Tuple[Optional[Dict[str, Any]], Dict[str, Any]]:
    """Call the real `serve`, and capture the evidence record it wrote."""
    captured: Dict[str, Any] = {}

    def _capture(record: Mapping[str, Any]) -> None:
        captured.update(record)

    original_write = evidence_write()
    try:
        set_evidence_write(_capture)
        context = SimpleNamespace(actor_id=principal)
        payload = canary.serve(
            question=question, context=context, client_id=fixture.CLIENT_ID,
            run_id="serving-integration", legacy_result={"ok": True, "value": 0.0},
            frame=frames, semantics=semantics, view="funded",
            portfolio_id=fixture.CLIENT_ID, as_of=None,
            snapshot_store=store, snapshot_client_id=fixture.CLIENT_ID,
            snapshot_route=fixture.ROUTE)
    finally:
        set_evidence_write(original_write)
    return payload, captured


def evidence_write():
    from mi_agent import plan_shadow_evidence as ev
    return ev.write


def set_evidence_write(fn) -> None:
    from mi_agent import plan_shadow_evidence as ev
    ev.write = fn


def main() -> int:
    payloads, questions = load_payloads()
    semantics = load_mi_semantics(str(REGISTRY))
    history = fixture.default_history()
    frames = dict(history)

    # The factory returns the INTERPRETER only; `wiring._compiler` builds the
    # real `DeterministicCompiler(CompilerContext())` itself, and this harness
    # deliberately does not replace it — the compiler under test must be the
    # production one.
    wiring.set_interpreter_factory(
        lambda: OpusInterpreter(ReplayClient(payloads)))
    os.environ[canary.SERVE_ENV_VAR] = canary.SERVE_CANARY
    os.environ[canary.PRINCIPALS_ENV_VAR] = CANARY_PRINCIPAL

    records: List[Dict[str, Any]] = []
    series_values = grouped_cells = comparisons = 0

    try:
        with tempfile.TemporaryDirectory() as tmp:
            store = fixture.build_store(Path(tmp) / "snapshots", history)
            book = store.load_loans(
                store.list_snapshots(fixture.CLIENT_ID,
                                     route=fixture.ROUTE)[-1].snapshot_id)

            for spec in POSITIVES:
                question = questions[spec["case"]]
                record: Dict[str, Any] = {
                    "id": spec["id"], "case": spec["case"],
                    "why": spec["why"], "question": question, "problems": []}
                payload, captured = serve_case(
                    question, store=store, semantics=semantics, frames=book,
                    principal=CANARY_PRINCIPAL)

                serving = captured.get("serving") or {}
                record["served_from"] = serving.get("decision")
                record["perimeter"] = (captured.get("eligibility") or {}).get(
                    "perimeter")
                temporal_block = ((captured.get("execution") or {})
                                  .get("temporal") or {})
                record["shape"] = temporal_block.get("shape")
                record["basis"] = temporal_block.get("basis")
                selected = [p["reporting_date"]
                            for p in (temporal_block.get("points") or ())]
                record["snapshots"] = selected

                if payload is None:
                    record["problems"].append(
                        f"the canary did not serve; reason="
                        f"{serving.get('reason')!r}")
                    record["verdict"] = "FAIL"
                    records.append(record)
                    continue
                if serving.get("decision") != canary.SERVED_NEW:
                    record["problems"].append(
                        f"served from {serving.get('decision')!r}")
                if record["perimeter"] != "slice2_temporal":
                    record["problems"].append(
                        f"perimeter {record['perimeter']!r} — the temporal "
                        f"runtime was not the one that answered")
                if selected != spec["snapshots"]:
                    record["problems"].append(
                        f"snapshots {selected} != {spec['snapshots']}")

                # --- governed evidence on the served envelope --------------- #
                governed = ((payload.get("metadata") or {}).get("governedPlan")
                            or {})
                executed = governed.get("executed") or {}
                requested = governed.get("requested") or {}
                record["requested_period_form"] = requested.get("period_form")
                record["evidence_snapshots"] = executed.get("snapshot_count")
                if not requested.get("period_form"):
                    record["problems"].append(
                        "the envelope states no requested temporal semantics")
                if executed.get("snapshot_count") != len(spec["snapshots"]):
                    record["problems"].append(
                        f"envelope evidence names "
                        f"{executed.get('snapshot_count')} snapshots")

                dimensions = list(spec.get("dimensions") or ())
                for entry in (executed.get("snapshots") or ()):
                    applied = {p.get("field")
                               for p in (entry.get("applied_predicates") or ())}
                    for wanted in (spec.get("filters") or ()):
                        if wanted not in applied:
                            record["problems"].append(
                                f"{entry['reporting_date']}: filter {wanted!r} "
                                f"absent from the served evidence")
                    grouped_keys = list(entry.get("group_field_keys") or ())
                    if sorted(grouped_keys) != sorted(dimensions):
                        record["problems"].append(
                            f"{entry['reporting_date']}: grouped on "
                            f"{grouped_keys}, expected {dimensions}")

                # --- the figures, read back off the PAYLOAD ----------------- #
                column = f"{payload.get('spec', {}).get('metric')}"
                bound = (captured.get("execution") or {}).get("bound_spec") or {}
                value_column = ("loan_count" if bound.get("aggregation") == "count"
                                else f"{bound.get('metric')}_{bound.get('aggregation')}")
                rows = payload_series(payload, value_column, dimensions)
                record["payload_rows"] = len(rows)
                if not rows:
                    record["problems"].append(
                        "the served envelope carries no temporal rows")
                for date in spec["snapshots"]:
                    frame = frames[date]
                    expected = oracle(spec["oracle"], frame)
                    period_rows = [r for r in rows
                                   if str(r.get(temporal.REPORTING_DATE)) == date]
                    if dimensions:
                        produced = {tuple(str(r[a]) for a in dimensions):
                                    r.get(value_column) for r in period_rows}
                        if set(produced) != set(expected):
                            record["problems"].append(
                                f"{date}: served group keys differ")
                        for key, figure in expected.items():
                            grouped_cells += 1
                            got = produced.get(key)
                            if got is None or abs(float(got) - figure) > 0.01:
                                record["problems"].append(
                                    f"{date} {key}: served {got} != {figure}")
                    else:
                        series_values += 1
                        if len(period_rows) != 1:
                            record["problems"].append(
                                f"{date}: {len(period_rows)} rows served")
                            continue
                        got = period_rows[0].get(value_column)
                        if got is None or abs(float(got) - expected) > 0.01:
                            record["problems"].append(
                                f"{date}: served {got} != {expected}")

                if spec.get("comparison"):
                    change = (executed.get("comparison") or {})
                    baseline = oracle(spec["oracle"], frames[spec["snapshots"][0]])
                    current = oracle(spec["oracle"], frames[spec["snapshots"][1]])
                    comparisons += 1
                    if abs((change.get("absolute_change") or 0.0)
                           - (current - baseline)) > 0.01:
                        record["problems"].append(
                            f"absolute change {change.get('absolute_change')} "
                            f"!= {current - baseline}")
                    expected_pct = (current - baseline) / baseline * 100.0
                    if abs((change.get("percent_change") or 0.0)
                           - expected_pct) > 1e-6:
                        record["problems"].append(
                            f"percent change {change.get('percent_change')} "
                            f"!= {expected_pct}")

                record["verdict"] = "PASS" if not record["problems"] else "FAIL"
                records.append(record)

            # --- the controls -------------------------------------------- #
            for spec in CONTROLS:
                question = questions[spec["case"]]
                record = {"id": spec["id"], "case": spec["case"],
                          "why": spec["why"], "question": question,
                          "problems": []}
                principal = (NON_CANARY_PRINCIPAL if spec.get("non_canary")
                             else CANARY_PRINCIPAL)
                payload, captured = serve_case(
                    question, store=store, semantics=semantics, frames=book,
                    principal=principal)
                serving = captured.get("serving") or {}
                reason = serving.get("reason") or ""
                record["served_from"] = serving.get("decision")
                record["reason"] = reason

                if spec["expect"] == "NOT_HANDLED":
                    if payload is not None:
                        record["problems"].append(
                            "a non-canary principal was served the new path")
                    if captured:
                        record["problems"].append(
                            "a record was written for a non-canary principal")
                else:
                    if payload is not None:
                        record["problems"].append(
                            "the canary served a case that must fail closed")
                    want = spec.get("expect_reason_startswith") or ""
                    if want and not reason.startswith(want):
                        record["problems"].append(
                            f"reason {reason!r} does not start {want!r}")
                    block = ((captured.get("execution") or {})
                             .get("temporal") or {})
                    if spec.get("must_not_execute") and block.get("points"):
                        record["problems"].append(
                            "a fail-closed case produced snapshot points")
                    if spec.get("no_snapshots") and block.get("points"):
                        record["problems"].append(
                            "an overlong request selected a shortened series")
                record["verdict"] = "PASS" if not record["problems"] else "FAIL"
                records.append(record)
    finally:
        wiring.set_interpreter_factory(None)
        os.environ.pop(canary.SERVE_ENV_VAR, None)
        os.environ.pop(canary.PRINCIPALS_ENV_VAR, None)

    verdicts = Counter(r["verdict"] for r in records)
    failures = [r for r in records if r["verdict"] != "PASS"]
    report = {
        "path_under_test": "plan_serving_canary.serve (the production entry point)",
        "model_calls": 0,
        "payload_source": "recorded live Opus output, replayed via ReplayClient",
        "positives": len(POSITIVES), "controls": len(CONTROLS),
        "verdicts": dict(verdicts),
        "series_values_reconciled": series_values,
        "grouped_cells_reconciled": grouped_cells,
        "period_comparisons_reconciled": comparisons,
        "snapshot_selection_errors": sum(
            1 for r in records if any("snapshots" in p for p in r["problems"])),
        "results": records,
    }
    OUT.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")

    print("=== SLICE 2 SERVING INTEGRATION — through plan_serving_canary.serve")
    print(f"  positives              {len(POSITIVES)}")
    print(f"  controls               {len(CONTROLS)}")
    for record in records:
        mark = "PASS" if record["verdict"] == "PASS" else "FAIL"
        extra = (f"snapshots={len(record.get('snapshots') or ())} "
                 f"rows={record.get('payload_rows')}"
                 if record.get("snapshots") else
                 f"served={record.get('served_from')} "
                 f"reason={record.get('reason', '')[:46]}")
        print(f"    {mark} {record['id']:<5} {record['why'][:38]:<40} {extra}")
        for problem in record["problems"]:
            print(f"          {problem}")
    print(f"  SERIES_VALUES          {series_values}")
    print(f"  GROUPED_CELLS          {grouped_cells}")
    print(f"  PERIOD_COMPARISONS     {comparisons}")
    print(f"  MODEL_CALLS            0")
    print(f"  WRITTEN                {OUT.relative_to(_REPO_ROOT)}")
    print(f"  RESULT                 {'PASS' if not failures else 'FAIL'}")
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
