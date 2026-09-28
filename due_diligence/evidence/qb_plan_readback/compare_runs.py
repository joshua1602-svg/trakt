#!/usr/bin/env python3
"""Compare two production-bank runs' readings, question by question. Offline.

Takes the baseline run's recorded intents (`qb_recorded_intents.json`, the
2026-09-28 run at vocabulary 2.2.0) and a re-run's readback projection (the
`qb_plan_readback.json` artifact `qb-plan-readback.yml` produces), recompiles
BOTH runs' model payloads with today's compiler, puts each plan through today's
specialist perimeters, and reports:

  - which vocabulary each re-run reading was made against (expect 2.3.0);
  - every question whose READING changed between the runs;
  - for the two HELD forecast shapes, which questions land in them now — the
    evidence for lifting the hold is that only the readings that mean the
    held figure still do;
  - the newly admitted set under today's perimeter, for both runs.

It asks nothing, calls no model and reads no production data: both inputs are
files. `--write-intents` also writes the re-run as a slim recorded-intents file
(no answer, no figure) in the baseline's format, for the replay test.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2]))

from mi_agent import plan_forecast_runtime as forecast_rt  # noqa: E402
from mi_agent import plan_pipeline_runtime as pipeline_rt  # noqa: E402
from mi_agent import plan_runtime_adapter as adapter  # noqa: E402
from mi_agent import plan_stage_movement_runtime as stage_rt  # noqa: E402
from mi_agent.interpretation_v2.compiler import (  # noqa: E402
    CompilerContext, DeterministicCompiler)
from mi_agent.interpretation_v2.intent import parse_candidate_intent  # noqa: E402

RUNTIMES = (("pipeline", pipeline_rt), ("stage_movement", stage_rt),
            ("forecast", forecast_rt))


def reading(payload: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Today's compile of one model payload, and the perimeter's decision."""
    if not payload:
        return {"outcome": "NO_PAYLOAD"}
    try:
        result = DeterministicCompiler(CompilerContext()).compile(
            parse_candidate_intent(json.loads(json.dumps(payload))))
    except Exception as exc:                                         # noqa: BLE001
        return {"outcome": f"UNPARSEABLE: {type(exc).__name__}"}
    if result.plan is None:
        return {"outcome": str(result.outcome),
                "codes": [r.code for r in result.reasons]}
    plan = result.plan.to_dict()
    output = (plan.get("outputs") or [{}])[0]
    period = plan.get("period") or {}
    row: Dict[str, Any] = {
        "outcome": "PLAN",
        "shape": (plan.get("capability"), plan.get("operation"),
                  tuple(m.get("concept") for m in output.get("measures") or ()),
                  period.get("form"), period.get("grain"),
                  (plan.get("target") or {}).get("concept")),
        "route": None,
    }
    for name, runtime in RUNTIMES:
        if runtime.claims(plan):
            ok, why, _ = runtime.check_eligibility(plan)
            if ok:
                ok, why, _ = adapter.check_population_base(
                    plan, runtime.EXECUTION_POPULATION,
                    executable=runtime.EXECUTABLE_POPULATIONS)
            row["route"] = (name, ok, why)
            break
    return row


def held_shape(row: Dict[str, Any]) -> Optional[Tuple[str, str]]:
    shape = row.get("shape")
    if not shape:
        return None
    key = (shape[1], shape[2][0] if shape[2] else "")
    return key if key in forecast_rt.HELD_READINGS else None


def rerun_cases(projection: Dict[str, Any]) -> Dict[int, Dict[str, Any]]:
    out = {}
    for case_id, case in projection["cases"].items():
        record = case.get("record") or {}
        summary = case.get("summary") or {}
        out[int(case["n"])] = {
            "n": case["n"], "id": case_id, "category": case["category"],
            "question": case["question"], "recorded_at": case.get("recorded_at"),
            "raw_model_payload": record.get("raw_model_payload"),
            "vocabulary_version": summary.get("vocabulary_version"),
            "recorded": {"compile_outcome": summary.get("compile_outcome"),
                         "reason_codes": summary.get("reason_codes"),
                         "plan_id": (record.get("governed_plan") or {}).get("plan_id"),
                         "serving_decision": summary.get("serving_decision"),
                         "serving_reason": summary.get("serving_reason")},
        }
    return out


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", default=str(HERE / "qb_recorded_intents.json"))
    parser.add_argument("--rerun", required=True,
                        help="the qb_plan_readback.json artifact of the re-run")
    parser.add_argument("--write-intents", default="",
                        help="also write the re-run as a slim recorded-intents file")
    args = parser.parse_args(argv)

    base = {int(c["n"]): c for c in
            json.loads(Path(args.baseline).read_text())["cases"]}
    rerun = rerun_cases(json.loads(Path(args.rerun).read_text()))

    versions: Dict[str, int] = {}
    for case in rerun.values():
        key = str(case["vocabulary_version"])
        versions[key] = versions.get(key, 0) + 1
    print(f"re-run readings by vocabulary version: {versions}")

    before = {n: reading(c["raw_model_payload"]) for n, c in base.items()}
    after = {n: reading(c["raw_model_payload"]) for n, c in rerun.items()}

    print("\nREADINGS THAT CHANGED")
    for n in sorted(set(before) & set(after)):
        if before[n].get("shape") != after[n].get("shape") or \
                before[n]["outcome"] != after[n]["outcome"]:
            print(f"[{n:>3}] {base[n]['question'][:60]}\n"
                  f"      was {before[n].get('shape') or before[n]['outcome']}\n"
                  f"      now {after[n].get('shape') or after[n]['outcome']}"
                  f"  -> {after[n].get('route')}")

    print("\nWHO LANDS IN A HELD SHAPE")
    for label, rows in (("baseline", before), ("re-run", after)):
        held: Dict[Tuple[str, str], list] = {}
        for n, row in rows.items():
            key = held_shape(row)
            if key:
                held.setdefault(key, []).append(n)
        for key in forecast_rt.HELD_READINGS:
            print(f"  {label:8s} {key[0]}/{key[1]}: {sorted(held.get(key, []))}")

    print("\nNEWLY ADMITTED UNDER TODAY'S PERIMETER (was not served NEW)")
    for label, rows, cases in (("baseline", before, base), ("re-run", after, rerun)):
        admitted = sorted(n for n, row in rows.items()
                          if row.get("route") and row["route"][1]
                          and cases[n]["recorded"]["serving_decision"] != "NEW")
        print(f"  {label:8s} {admitted}")

    if args.write_intents:
        blob = {"what_this_is": ("A re-run of the owner's production bank, read back "
                                 "by qb-plan-readback.yml: the model payload and the "
                                 "recorded compile and serving decisions only — no "
                                 "answer, no executed figure, no row."),
                "question_count": len(rerun),
                "cases": [rerun[n] for n in sorted(rerun)]}
        Path(args.write_intents).write_text(json.dumps(blob, indent=1) + "\n")
        print(f"\nwrote {args.write_intents}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
