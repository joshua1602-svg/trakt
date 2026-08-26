#!/usr/bin/env python3
"""migration_phase0/contract_plan_delta.py — what the LEVEL/MOVEMENT owner moved.

READ-ONLY. Sweeps every corpus question and records, per question:

  * the FULL contract, field by field, so "every question whose contract
    changes" is answerable exactly rather than for the one field that was
    expected to change;
  * the PLAN each route's builder produces from that contract, so "every route
    whose plan changes" is answerable the same way.

Run it in two trees and diff. Nothing here interprets anything; it records.

    python -m migration_phase0.contract_plan_delta --json out.json
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import warnings
from pathlib import Path
from typing import Any, Dict, List

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

CORPORA = ("question_interpretation/stage1_corpus.json",
           "question_interpretation/stage2_corpus.json")

#: The book's real columns. Dimension ALTERNATES only exist once the columns are
#: known, so a sweep with None would under-report the contract.
COLUMNS = ["loan_identifier", "current_outstanding_balance",
           "current_loan_to_value", "current_interest_rate",
           "youngest_borrower_age", "broker_channel",
           "geographic_region_obligor", "reporting_date"]


class DeltaError(RuntimeError):
    """The sweep could not be measured. Never absorbed into an empty diff."""


def _questions() -> List[str]:
    out, seen = [], set()
    for f in CORPORA:
        for row in json.loads((_REPO / f).read_text())["rows"]:
            q = row.get("question") or ""
            if q and q not in seen:
                seen.add(q)
                out.append(q)
    return out


def _plans(qi) -> Dict[str, Any]:
    """Every plan any route would build from this contract.

    Built from the CONTRACT ONLY — no question, no route identity — which is
    the property the compositional migration exists to establish, and is why a
    plan diff is a fair measure of what the contract change did.
    """
    from mi_agent_api import analytical_plan as plan

    out: Dict[str, Any] = {}
    builders = {
        "portfolio_summary": getattr(plan, "build_portfolio_summary_plan", None),
        "period_movement": getattr(plan, "build_period_movement_plan", None),
        "geo_exposure": getattr(plan, "build_geo_exposure_plan", None),
        "temporal_compare": getattr(plan, "build_temporal_compare_plan", None),
        "evolution": getattr(plan, "build_evolution_plan", None),
    }
    for name, fn in builders.items():
        if fn is None:
            continue
        try:
            built = fn(qi)
        except TypeError:
            # A builder needing extra route arguments is recorded as such
            # rather than skipped: a silent omission would read as "no change".
            out[name] = {"unbuildable": "requires route arguments"}
            continue
        except Exception as exc:  # noqa: BLE001 - recorded, never absorbed
            out[name] = {"error": f"{type(exc).__name__}: {exc}"}
            continue
        out[name] = [s.to_dict() for s in getattr(built, "steps", ())]
    return out


def run(plans_for=None) -> Dict[str, Any]:
    warnings.simplefilter("ignore")
    logging.disable(logging.WARNING)

    from mi_agent import execution_receipt as R
    from mi_agent.mi_query_validator import load_mi_semantics
    from mi_agent.parsed_question import ParsedQuestion
    from mi_agent_api.data_source import semantics_path
    from question_interpretation import projection as proj

    semantics = load_mi_semantics(semantics_path())
    if not (semantics.get("fields") or {}):
        raise DeltaError("SWEEP INVALID — governed MI semantics did not load")

    wanted = set(plans_for or ())
    rows: List[Dict[str, Any]] = []
    questions = _questions()
    for q in questions:
        spec = ParsedQuestion.parse(q, semantics).spec
        terms = R.requested_dimension_terms(q, semantics, COLUMNS)
        facets = R.detect_requested_facets(q, semantics, frame=None,
                                           requested_dimensions=terms)
        qi = proj.from_parts(q, spec=spec, facets=facets, dim_terms=terms,
                             semantics=semantics)
        rows.append({
            "question": q,
            "spec": {
                "temporal_mode": getattr(spec, "temporal_mode", None),
                "compare_periods": list(getattr(spec, "compare_periods", None) or []),
                "intent": getattr(spec, "intent", None),
                "metric": getattr(spec, "metric", None),
                "aggregation": getattr(spec, "aggregation", None),
            },
            "plans": (_plans(qi) if q in wanted else None),
            "contract": qi.to_dict() if hasattr(qi, "to_dict") else {
                "operation": qi.operation.as_dict(),
                "subject": qi.subject.as_dict(),
                "dimensions": [d.as_dict() for d in (qi.dimensions or [])],
                "time": qi.time.as_dict(),
                "dataset": qi.dataset.as_dict(),
            },
            # PLANS ARE NOT BUILT HERE. A plan is a function of the contract,
            # so a plan can only change where the contract changed — building
            # 5 plans for all 882 questions in both trees costs an hour and
            # measures nothing the contract diff does not already locate. They
            # are built for the changed set, by --plans-for.
        })
    if len(rows) != len(questions) or not rows:
        raise DeltaError(
            f"SWEEP INVALID — {len(rows)} readings for {len(questions)} questions")
    return {"rows": rows}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--json", type=Path, required=True)
    ap.add_argument("--plans-for", type=Path, default=None,
                    help="a JSON list of questions; build every route plan for "
                         "exactly those and record it")
    args = ap.parse_args(argv)
    result = run(plans_for=(json.loads(args.plans_for.read_text())
                            if args.plans_for else None))
    args.json.write_text(json.dumps(result, indent=2, default=str,
                                    sort_keys=True), encoding="utf-8")
    print(f"swept {len(result['rows'])} questions -> {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
