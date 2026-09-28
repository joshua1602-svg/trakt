#!/usr/bin/env python3
"""The boundary correction, measured on the CAPTURED live payloads. No calls.

Every payload here is one Opus actually emitted in the 15-call run recorded in
`temporal_live_run_result.json`. Nothing is re-asked and nothing is re-worded:
the model's own output is replayed through the corrected deterministic layer, so
what this measures is the FIX, with the model held constant.

    The payloads were emitted at vocabulary 2.0.0 and are replayed against a
    compiler at 2.1.0. That is sound for this purpose — the compiler reads no
    capability-boundary block, which exists only in the orientation the model is
    SHOWN — and it is the reason the four capability cases must be re-asked
    live rather than replayed. A replay cannot tell you what a model would now
    say.

It goes further than the live harness, which stops at snapshot selection: every
case that resolves is EXECUTED through `execute_temporal_plan` and reconciled
against `portfolio_truth_oracle`, which imports nothing from the product.

Run: `python due_diligence/evidence/plan_temporal_slice2/temporal_boundary_regression.py`
"""
from __future__ import annotations

import json
import sys
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from mi_agent import plan_temporal_runtime as temporal                 # noqa: E402
from mi_agent.interpretation_v2.compiler import DeterministicCompiler   # noqa: E402
from mi_agent.interpretation_v2.intent import parse_candidate_intent    # noqa: E402
from mi_agent.mi_query_validator import load_mi_semantics              # noqa: E402
from mi_agent.tests import portfolio_truth_oracle as truth             # noqa: E402
from mi_agent.tests import temporal_snapshot_fixture as fixture        # noqa: E402

import temporal_live_run as harness                                    # noqa: E402

LIVE = HERE / "temporal_live_run_result.json"
MANIFEST = HERE / "temporal_bank_manifest.json"
OUT = HERE / "temporal_boundary_regression.json"
REGISTRY = _REPO_ROOT / "mi_agent" / "mi_semantics_field_registry.yaml"

#: How the independent oracle computes each positive. Written out here, never
#: read back from the product. The display form of a governed categorical value
#: is used, exactly as the slice 1 replay discloses.
RECIPES: Mapping[str, Dict[str, Any]] = {
    "T01": {"how": "sum", "column": truth.BALANCE},
    "T02": {"how": "sum", "column": truth.BALANCE},
    "T03": {"how": "sum", "column": truth.BALANCE},
    "T04": {"how": "count"},
    "T05": {"how": "avg", "column": truth.LTV},
    "T06": {"how": "count",
            "predicates": [("erm_product_type", "eq", "Drawdown")]},
    "T07": {"how": "count", "group_by": ["ltv_bucket"]},
    "T08": {"how": "count", "group_by": ["ltv_bucket", "erm_product_type"]},
    "T09": {"how": "sum", "column": truth.BALANCE},
    "T10": {"how": "sum", "column": truth.BALANCE},
    "T11": {"how": "sum", "column": truth.BALANCE},
}

#: Cases whose correct answer is a refusal, and what must refuse them.
CONTROLS = {
    "T12": "future period — a forward methodology this capability does not own",
    "T13": "twenty-four periods asked of an eight-period book",
    "T14": "movement attribution — why a balance moved",
    "T15": "three independent measures in one request",
}


def oracle(recipe: Mapping[str, Any], frame: Any) -> Any:
    predicates = list(recipe.get("predicates") or ())
    group_by = list(recipe.get("group_by") or ())
    how = recipe["how"]
    if group_by:
        return truth.grouped(frame, group_by, column=recipe.get("column"),
                             how=how, predicates=predicates)
    if how == "count":
        return float(truth.row_count(frame, predicates))
    if how == "sum":
        return truth.total(frame, recipe["column"], predicates)
    if how == "avg":
        rows = frame.loc[truth.mask_for(frame, predicates)]
        return float(rows[recipe["column"]].mean()) if len(rows) else None
    if how == "weighted_avg":
        return truth.weighted_average(frame, recipe["column"], truth.BALANCE,
                                      predicates)
    raise ValueError(f"no oracle rule for {how!r}")


def reconcile(record: Dict[str, Any], outcome: Any, recipe: Mapping[str, Any],
              frames: Mapping[str, Any]) -> None:
    """Every figure on every selected snapshot, against the independent truth."""
    group_by = list(recipe.get("group_by") or ())
    scalars = cells = 0
    for point in outcome.points:
        frame = frames.get(point.reporting_date)
        if frame is None:
            record["problems"].append(f"{point.reporting_date}: not a fixture period")
            continue
        expected = oracle(recipe, frame)
        if group_by:
            produced = {tuple(str(cell[axis]) for axis in group_by): cell["value"]
                        for cell in point.cells}
            if set(produced) != set(expected):
                record["problems"].append(
                    f"{point.reporting_date}: group keys differ")
            for key, figure in expected.items():
                cells += 1
                got = produced.get(key)
                if got is None or abs(got - figure) > 0.01:
                    record["problems"].append(
                        f"{point.reporting_date} {key}: {got} != {figure}")
        else:
            scalars += 1
            if point.value is None or abs(point.value - expected) > 0.01:
                record["problems"].append(
                    f"{point.reporting_date}: {point.value} != {expected}")
    record["scalars_reconciled"] = scalars
    record["cells_reconciled"] = cells


def main(argv: Sequence[str] = ()) -> int:
    # An alternate source file, so a LIVE RETEST result can be executed and
    # reconciled by this same owner rather than by a second copy of it.
    source = Path(argv[0]) if argv else LIVE
    if not source.is_absolute():
        source = HERE / source
    live = json.loads(source.read_text(encoding="utf-8"))
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    expected = {c["id"]: c for c in manifest["cases"]}
    was = {r["id"]: r["verdict"] for r in live["results"]}

    semantics = load_mi_semantics(str(REGISTRY))
    compiler = DeterministicCompiler()
    history = fixture.default_history()
    frames = dict(history)

    records: List[Dict[str, Any]] = []
    with tempfile.TemporaryDirectory() as tmp:
        store = fixture.build_store(Path(tmp) / "snapshots", history)
        for row in live["results"]:
            case = expected[row["id"]]
            record: Dict[str, Any] = {
                "id": row["id"], "question": row["question"],
                "category": case["category"],
                "verdict_before_correction": was[row["id"]],
                "problems": [],
            }
            compiled = compiler.compile(
                parse_candidate_intent(dict(row["raw_payload"])))
            record["compile_outcome"] = compiled.outcome
            record["compile_reasons"] = compiled.codes()

            if not compiled.is_plan:
                record["disposition"] = f"COMPILE_{compiled.outcome}"
                records.append(record)
                continue

            plan = compiled.plan.to_dict()
            record["plan_id"] = plan.get("plan_id")
            record["capability"] = plan.get("capability")
            record["operation"] = plan.get("operation")
            record["period_form"] = (plan.get("period") or {}).get("form")
            record["physical_binding"] = harness.physical_binding(plan)

            outcome = temporal.execute_temporal_plan(
                plan, store=store, client_id=fixture.CLIENT_ID,
                semantics=semantics, route=fixture.ROUTE)
            record["eligible"] = outcome.eligible
            record["reason"] = outcome.reason
            record["detail"] = outcome.detail[:200]
            record["shape"] = outcome.shape
            record["basis"] = outcome.basis
            record["snapshots"] = [p.reporting_date for p in outcome.points]

            if not outcome.eligible:
                record["disposition"] = f"INELIGIBLE:{outcome.reason}"
            elif outcome.reason:
                record["disposition"] = (
                    f"CLARIFY:{outcome.reason}" if outcome.clarifiable
                    else f"REFUSE:{outcome.reason}")
            else:
                record["disposition"] = "EXECUTED"
                want = list(case.get("expected_snapshots") or ())
                if record["snapshots"] != want:
                    record["problems"].append(
                        f"snapshots {record['snapshots']} != {want}")
                recipe = RECIPES.get(row["id"])
                if recipe:
                    reconcile(record, outcome, recipe, frames)
                for point in outcome.points:
                    applied = {e.get("field") for e in
                               (point.receipt.get("applied_predicates") or ())}
                    wanted = {f["field"] for f in
                              (outcome.requested.get("filters") or ())}
                    if wanted - applied:
                        record["problems"].append(
                            f"{point.reporting_date}: predicates "
                            f"{sorted(wanted - applied)} not applied")
            if record.get("physical_binding"):
                record["problems"].append(
                    f"physical binding: {record['physical_binding']}")
            records.append(record)

    # A SUBSET SOURCE IS A FIRST-CLASS INPUT. The retest file carries four cases,
    # so every cohort below is intersected with what is actually present rather
    # than indexed blindly — a KeyError here would report a harness limitation
    # as a failed correction.
    by_id = {r["id"]: r for r in records}
    subset = len(records) < len(expected)
    originally_passing = [r for r in records
                          if r["verdict_before_correction"] in harness._HELD]
    grain_only = [by_id[i] for i in ("T03", "T04", "T06", "T07") if i in by_id]
    capability = [by_id[i] for i in ("T01", "T05", "T10", "T11") if i in by_id]

    def holds(record: Mapping[str, Any]) -> bool:
        if record["problems"]:
            return False
        case_id = record["id"]
        if case_id in CONTROLS:
            return record["disposition"].split(":")[0] in (
                "INELIGIBLE", "CLARIFY", "REFUSE", "COMPILE_REFUSE",
                "COMPILE_CLARIFY")
        return record["disposition"] == "EXECUTED"

    original_pass = sum(1 for r in originally_passing if holds(r))
    grain_pass = sum(1 for r in grain_only if holds(r))

    report = {
        "source": source.name,
        "is_subset": subset,
        "payloads": "captured live Opus output, replayed; zero model calls",
        "vocabulary_note": ("payloads emitted at vocabulary 2.0.0, replayed "
                            "against a compiler at 2.1.0; the compiler reads no "
                            "capability-boundary block"),
        # On a subset source `verdict_before_correction` is that file's OWN
        # verdict, so this counts "held, and still holds" rather than the
        # fifteen-case regression. Named so it cannot be misread as the latter.
        "original_pass_replay": (f"{original_pass}/{len(originally_passing)}"
                                 if not subset else
                                 f"{original_pass}/{len(originally_passing)} "
                                 f"(subset: held-and-still-holds, not the "
                                 f"fifteen-case regression)"),
        "grain_only_replay": (f"{grain_pass}/{len(grain_only)}" if grain_only
                              else "not in this source"),
        "capability_cases_still_ineligible": [
            {"id": r["id"], "capability": r.get("capability"),
             "disposition": r["disposition"]} for r in capability],
        "controls": {r["id"]: r["disposition"] for r in records
                     if r["id"] in CONTROLS},
        "scalars_reconciled": sum(r.get("scalars_reconciled", 0) for r in records),
        "cells_reconciled": sum(r.get("cells_reconciled", 0) for r in records),
        "physical_bindings": sum(1 for r in records if r.get("physical_binding")),
        "dispositions": dict(Counter(r["disposition"] for r in records)),
        "results": records,
    }
    out = (OUT if source == LIVE
           else OUT.with_name(f"{source.stem}_reconciled.json"))
    out.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")

    print("=== SLICE 2 BOUNDARY CORRECTION — REPLAY OF CAPTURED PAYLOADS")
    print(f"  SOURCE                 {source.name}"
          f"{'  (SUBSET)' if subset else ''}")
    print(f"  ORIGINAL_PASS_REPLAY   {report['original_pass_replay']}")
    print(f"  GRAIN_ONLY_REPLAY      {report['grain_only_replay']}")
    for record in grain_only:
        print(f"    {record['id']}  {record['disposition']} basis={record.get('basis')} "
              f"snapshots={len(record.get('snapshots') or ())} "
              f"(was {record['verdict_before_correction']})")
    if capability:
        print("  CAPABILITY CASES")
        for record in capability:
            print(f"    {record['id']}  capability={record.get('capability')!r} "
                  f"operation={record.get('operation')!r} "
                  f"-> {record['disposition']}")
    if any(case_id in by_id for case_id in CONTROLS):
        print("  CONTROLS")
        for case_id, why in CONTROLS.items():
            if case_id in by_id:
                print(f"    {case_id}  {by_id[case_id]['disposition']:<34} {why}")
    print(f"  SCALARS_RECONCILED     {report['scalars_reconciled']}")
    print(f"  CELLS_RECONCILED       {report['cells_reconciled']}")
    print(f"  PHYSICAL_BINDINGS      {report['physical_bindings']}")
    failures = [r for r in records if r["problems"]]
    for record in failures:
        print(f"  FAIL {record['id']}: {record['problems'][:3]}")
    print(f"  WRITTEN                {out.relative_to(_REPO_ROOT)}")
    # WHAT COUNTS AS PASS DEPENDS ON WHAT THE SOURCE IS, and conflating the two
    # reported the fifteen-case regression as a failure for containing exactly
    # the pre-correction payloads it is supposed to contain.
    #
    #   the full run   payloads emitted BEFORE the correction. The four
    #                  capability cases are period_movement and are CORRECTLY
    #                  ineligible here; requiring them to execute would demand
    #                  that a replay change what the model said. They are
    #                  reported, not scored.
    #   a retest       payloads emitted AFTER it, by a re-ask. Every non-control
    #                  case must execute, because that is the whole question.
    capability_ids = {"T01", "T05", "T10", "T11"}
    must_execute = [r for r in records if r["id"] not in CONTROLS
                    and (subset or r["id"] not in capability_ids)]
    ok = (original_pass == len(originally_passing)
          and grain_pass == len(grain_only) and not failures
          and all(r["disposition"] == "EXECUTED" for r in must_execute))
    report["scored_cases"] = [r["id"] for r in must_execute]
    report["reported_not_scored"] = [
        r["id"] for r in records
        if r["id"] in capability_ids and r not in must_execute]
    print(f"  RESULT                 {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
