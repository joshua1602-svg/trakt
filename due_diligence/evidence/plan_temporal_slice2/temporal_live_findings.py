#!/usr/bin/env python3
"""What the live run found, separated into its two independent causes. No calls.

The live run reported 7/15. That number is correct and it is also two different
stories added together, which is not actionable. This reads the RECORDED live
payloads back and separates them.

ANALYSIS 1 — CAPABILITY ROUTING. Four cases were refused as
`CAPABILITY_NOT_GENERIC` because Opus routed them to `period_movement`. Were
those refusals about TIME, or only about which capability owns the question? The
recorded payload is re-compiled with `capability` forced to `generic_analysis`
(and a `movement` operation to the span operation it would need), and the result
says which.

    This is a COUNTERFACTUAL and is labelled as one everywhere it is reported.
    It is not a re-run, it buys no evidence about the model's routing, and its
    number never replaces the honest 7/15.

ANALYSIS 2 — THE COMPILER AND THE RESOLVER DISAGREE. Four cases were refused as
`PERIOD_LABEL_UNRESOLVED` on a payload carrying `form: series`, `grain: monthly`,
no labels and no count. The compiler accepts exactly that as a settled span and
emits a plan — `_bind_period` raises `AMBIGUOUS_PERIOD` only when a span has
*none* of labels, count or grain. So two governed layers disagree about what
counts as a stated span, and slice 2 introduced the disagreement. This measures
it, and measures what the whole-series reading would have selected.

Run: `python due_diligence/evidence/plan_temporal_slice2/temporal_live_findings.py`
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from mi_agent import plan_temporal_runtime as temporal                 # noqa: E402
from mi_agent.interpretation_v2.compiler import DeterministicCompiler   # noqa: E402
from mi_agent.interpretation_v2.intent import parse_candidate_intent    # noqa: E402
from mi_agent.states.selectors import SnapshotSelector                  # noqa: E402
from mi_agent.tests import temporal_snapshot_fixture as fixture         # noqa: E402

HERE = Path(__file__).resolve().parent
LIVE = HERE / "temporal_live_run_result.json"
MANIFEST = HERE / "temporal_bank_manifest.json"
OUT = HERE / "temporal_live_findings.json"


def main() -> int:
    live = json.loads(LIVE.read_text(encoding="utf-8"))
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    expected = {c["id"]: c for c in manifest["cases"]}
    compiler = DeterministicCompiler()

    routed: List[Dict[str, Any]] = []
    unresolved: List[Dict[str, Any]] = []

    with tempfile.TemporaryDirectory() as tmp:
        store = fixture.build_store(Path(tmp) / "snapshots",
                                    fixture.default_history())
        whole_series = [h.reporting_date for h in
                        SnapshotSelector.range(fixture.CLIENT_ID, None, None,
                                               route=fixture.ROUTE).resolve(store)]

        for record in live["results"]:
            want = list(expected[record["id"]].get("expected_snapshots") or ())
            intent = record.get("candidate_intent") or {}
            time_slot = dict(intent.get("time") or {})

            if record["verdict"] == "UNEXPECTED_INELIGIBLE":
                payload = dict(record["raw_payload"])
                row: Dict[str, Any] = {
                    "id": record["id"], "question": record["question"],
                    "model_capability": payload.get("capability"),
                    "model_operation": payload.get("operation"),
                    "model_time": time_slot,
                    "refused_as": record.get("ineligible_reason"),
                }
                payload["capability"] = "generic_analysis"
                if payload.get("operation") == "movement":
                    payload["operation"] = (
                        "series" if time_slot.get("form") in ("series", "range")
                        else "compare")
                row["counterfactual_operation"] = payload["operation"]
                result = compiler.compile(parse_candidate_intent(payload))
                if not result.is_plan:
                    row["counterfactual"] = f"COMPILE_{result.outcome}"
                    row["counterfactual_codes"] = result.codes()
                else:
                    plan = result.plan.to_dict()
                    ok, reason, _ = temporal.check_temporal_eligibility(plan)
                    if not ok:
                        row["counterfactual"] = f"INELIGIBLE:{reason}"
                    else:
                        resolution = temporal.resolve_temporal(
                            plan, store, client_id=fixture.CLIENT_ID,
                            route=fixture.ROUTE)
                        if not resolution.ok:
                            row["counterfactual"] = f"UNRESOLVED:{resolution.reason}"
                        else:
                            got = list(resolution.reporting_dates)
                            row["counterfactual"] = "RESOLVED"
                            row["counterfactual_snapshots"] = got
                            row["expected_snapshots"] = want
                            row["temporal_binding_sound"] = got == want
                routed.append(row)

            elif record["verdict"] == "PERIOD_UNRESOLVABLE":
                result = compiler.compile(
                    parse_candidate_intent(dict(record["raw_payload"])))
                unresolved.append({
                    "id": record["id"], "question": record["question"],
                    "model_time": time_slot,
                    "compiler_emitted_a_plan": result.is_plan,
                    "compiler_outcome": result.outcome,
                    "resolver_refused_as": record.get("resolution_reason"),
                    "whole_series_would_select": whole_series,
                    "expected_snapshots": want,
                    "whole_series_matches_expectation": whole_series == want,
                })

    sound = sum(1 for r in routed if r.get("temporal_binding_sound"))
    would_match = sum(1 for r in unresolved
                      if r["whole_series_matches_expectation"])
    findings = {
        "source_run": LIVE.name,
        "headline_live_result": {
            "boundary_held": live["boundary_held"], "of": live["of"],
            "live_calls_made": live["live_calls_made"],
        },
        "finding_1_capability_routing": {
            "cases": len(routed),
            "all_routed_to": sorted({r["model_capability"] for r in routed}),
            "counterfactual_temporal_binding_sound": sound,
            "counterfactual_is_not_evidence_about_routing": True,
            "rows": routed,
        },
        "finding_2_compiler_resolver_disagreement": {
            "cases": len(unresolved),
            "compiler_emitted_a_plan_for_all": all(
                r["compiler_emitted_a_plan"] for r in unresolved),
            "whole_series_reading_would_match": would_match,
            "rows": unresolved,
        },
    }
    OUT.write_text(json.dumps(findings, indent=2, default=str), encoding="utf-8")

    print("=== SLICE 2 LIVE RUN — THE TWO FINDINGS, SEPARATED")
    print(f"  headline               {live['boundary_held']}/{live['of']} held "
          f"on {live['live_calls_made']} live calls")
    print()
    print(f"  FINDING 1  capability routing: {len(routed)} cases -> "
          f"{sorted({r['model_capability'] for r in routed})}")
    for row in routed:
        print(f"    {row['id']}  {row['model_capability']}/{row['model_operation']}"
              f" -> counterfactual {row['counterfactual']}"
              f"{' / snapshots match' if row.get('temporal_binding_sound') else ''}")
    print(f"    temporal binding sound in {sound}/{len(routed)} "
          f"(COUNTERFACTUAL — says nothing about routing)")
    print()
    print(f"  FINDING 2  compiler/resolver disagreement: {len(unresolved)} cases")
    for row in unresolved:
        print(f"    {row['id']}  compiler emitted a plan: "
              f"{row['compiler_emitted_a_plan']}; resolver refused as "
              f"{row['resolver_refused_as']}")
    print(f"    whole-series reading would match in {would_match}/{len(unresolved)}")
    print()
    print(f"  WRITTEN                {OUT.relative_to(_REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
