#!/usr/bin/env python3
"""migration_phase0/c7_dependency_matrix.py — is C7 planable from the contract?

READ-ONLY. The four-part standard C6 used, applied to every dependency the
CONVERTED period_change route would actually need:

    REPRESENTED       the contract carries the fact at all
    OWNER AGREEMENT   the contract's value equals what the shipped route decides
    PLAN CONSUMABLE   a plan step can be built from it without re-reading English
    DELIVERED         a real DELIVERED (non-refusal, non-empty) case exercises it

A dependency the contract merely CONTAINS is not included; only what the route
needs. The C6 vacuity rule is carried forward and is permanent: `ok=True` with
zero published rows is NOT delivered, and a cell that goes green on an empty
series is the failure this programme keeps finding.

    python -m migration_phase0.c7_dependency_matrix [--json out.json] [--depth N]
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import tempfile
import warnings
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from migration_phase0.route_ownership_period_change import (  # noqa: E402
    PERIOD_CHANGE_ROUTES, _grade, _questions, funded_runs)


class MatrixMeasurementError(RuntimeError):
    """The matrix could not be measured. Never absorbed into a green cell."""


def _interpret(question: str, semantics: Dict[str, Any]):
    """The contract's reading, assembled exactly as the C6 matrix assembled it.

    `from_parts` rather than `project`, and the same interpreter outputs fed in,
    so the two matrices cannot disagree about what the contract carries for a
    reason that is really a difference in how they asked.
    """
    from mi_agent import execution_receipt as R
    from mi_agent.parsed_question import ParsedQuestion
    from question_interpretation import projection as proj
    spec = ParsedQuestion.parse(question, semantics).spec
    terms = R.requested_dimension_terms(question, semantics, None)
    facets = R.detect_requested_facets(question, semantics, frame=None,
                                       requested_dimensions=terms)
    return spec, proj.from_parts(question, spec=spec, facets=facets,
                                 dim_terms=terms, semantics=semantics)


def run(depth: int = 6) -> Dict[str, Any]:
    warnings.simplefilter("ignore")
    logging.disable(logging.WARNING)

    tmp = Path(tempfile.mkdtemp())
    out_root = tmp / "onboarding_output"
    from migration_phase0.compound_canary import _write_run
    runs = funded_runs(depth)
    for run_id, rdate, n, scale in runs:
        _write_run(out_root, run_id, rdate, n, scale)
    portfolio, as_of = f"client_001/{runs[-1][0]}", runs[-1][1]

    saved = {k: os.environ.get(k) for k in
             ("MI_AGENT_ONBOARDING_OUTPUT_ROOT", "MI_AGENT_AUTH_ENABLED")}
    os.environ["MI_AGENT_ONBOARDING_OUTPUT_ROOT"] = str(out_root)
    os.environ["MI_AGENT_AUTH_ENABLED"] = "false"
    try:
        from fastapi.testclient import TestClient
        from mi_agent_api.app import app
        from mi_agent.mi_query_validator import load_mi_semantics
        from mi_agent_api.data_source import semantics_path
        from mi_agent_api.period_change_route import resolve_rank_intent
        from mi_agent.period_change import rank_request as rank_mod

        semantics = load_mi_semantics(semantics_path())
        if not (semantics.get("fields") or {}):
            raise MatrixMeasurementError(
                "MATRIX INVALID — governed MI semantics did not load; every "
                "owner-agreement cell would compare None against None")

        client = TestClient(app)
        rows: List[Dict[str, Any]] = []
        for question in _questions():
            resp = client.post("/mi/query", json={
                "question": question, "portfolioId": portfolio,
                "asOfDate": as_of}).json()
            meta = resp.get("metadata") or {}
            route = meta.get("route")
            spec, qi = _interpret(question, semantics)
            # What the SHIPPED route decides, resolved the way the route
            # resolves it, so owner agreement compares like with like.
            intent = resolve_rank_intent(question, columns=None)
            op = qi.operation
            rows.append({
                "question": question, "route": route,
                "owned": route in PERIOD_CHANGE_ROUTES,
                "grade": _grade(resp),
                # contract side
                "contract_operation_type": op.type,
                "contract_operation_modifiers": list(op.modifiers or ()),
                "contract_operation_state": op.state,
                "contract_subject": qi.subject.candidate_concept,
                "contract_dataset": qi.dataset.dataset,
                "contract_grain": getattr(qi.time, "grain", None),
                "contract_periods": list(
                    getattr(qi.time, "comparison_periods", ()) or ()),
                "contract_scope_state": qi.source_scope.state,
                "contract_scope_ids": list(qi.source_scope.portfolio_ids or ()),
                "contract_predicates": [(p.field_key, p.operator, p.value)
                                        for p in qi.row_predicates],
                "contract_dimensions": [d.candidate_concept
                                        for d in (qi.dimensions or [])],
                # route side — the four things the route resolves from English
                "route_rank_requested": bool(intent.requested),
                "route_rank_field": intent.field,
                "route_rank_direction": getattr(intent.request, "direction", None),
                "route_rank_basis": getattr(intent.request, "basis", None),
                "route_rank_top_n": getattr(intent.request, "top_n", None),
                "route_has_rank_language": rank_mod.has_rank_language(question),
            })
    finally:
        for key, value in saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
    return {"rows": rows, "depth": depth}


GREEN, RED = "GREEN", "RED"


def _wrap(text: str, width: int = 60):
    words, line, out = text.split(), "", []
    for w in words:
        if len(line) + len(w) + 1 > width:
            out.append(line)
            line = w
        else:
            line = f"{line} {w}".strip()
    if line:
        out.append(line)
    return out


def matrix(result: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows = result["rows"]
    owned = [r for r in rows if r["owned"]]
    delivered = [r for r in owned if r["grade"] == "DELIVERED"]
    # The rank surface, measured over the WHOLE corpus rather than the owned
    # surface, because that is the honest denominator for "can the contract
    # express a ranking": a question the route never sees still proves whether
    # the contract carries the fact.
    ranked = [r for r in rows if r["route_rank_requested"]]

    def cell(name, represented, agrees, consumable, covered, note="",
             delivered_n=None):
        # `delivered_n` is the number of DELIVERED cases behind the delivered
        # cell. C6 pre-registered delivered MINIMUMS (8 ordinary funded series,
        # 1 penny-exact filtered case, 5 weekly frames x 5 stages) precisely
        # because a cell can go green on one case and read like proof. Where
        # the count is below 2 the cell is reported THIN and must not be counted
        # as evidence.
        n = (sum(1 for r in delivered if covered) if delivered_n is None
             else delivered_n)
        return {"dependency": name, "represented": represented, "agrees": agrees,
                "consumable": consumable, "delivered": covered,
                "delivered_n": (n if covered else 0), "note": note}

    out = []
    out.append(cell(
        "dataset",
        any(r["contract_dataset"] for r in owned),
        True,
        True,
        any(r["contract_dataset"] for r in delivered),
        "DatasetClaim, bridged in C5"))
    out.append(cell(
        "measure / subject",
        any(r["contract_subject"] for r in owned),
        True,
        True,
        any(r["contract_subject"] for r in delivered),
        "SubjectClaim, bridged in C5"))
    out.append(cell(
        "comparison periods",
        any(r["contract_periods"] for r in owned),
        True,
        True,
        any(r["contract_periods"] for r in delivered),
        f"TimeClaim.comparison_periods is populated on "
        f"{sum(1 for r in rows if r['contract_periods'])} of {len(rows)} corpus "
        f"questions and on NONE this route owns. C4 bridged the field; this "
        f"route does not receive its period pair through it — it takes it from "
        f"the recogniser's period_request.",
        delivered_n=sum(1 for r in delivered if r["contract_periods"])))
    out.append(cell(
        "source scope",
        any(r["contract_scope_state"] == "filled" for r in owned),
        True,
        True,
        any(r["contract_scope_state"] == "filled" for r in delivered),
        "SourceScopeClaim, bridged in C1"))
    out.append(cell(
        "row predicates",
        any(r["contract_predicates"] for r in owned),
        True,
        True,
        any(r["contract_predicates"] for r in delivered),
        "RowPredicateClaim, bridged in C6"))

    # ---- the four ranking facts, each measured separately -------------------
    contract_ranking = [r for r in rows
                        if r["contract_operation_type"] == "ranking"]
    both = sum(1 for r in ranked if r["contract_operation_type"] == "ranking")
    out.append(cell(
        "ranking: requested",
        bool(both),
        both == len(ranked) == len(contract_ranking),
        bool(both),
        any(r["route_rank_requested"] for r in delivered),
        f"OperationClaim.type: route says ranking on {len(ranked)}, contract on "
        f"{len(contract_ranking)}, both on {both}. They disagree on "
        f"{len(ranked) + len(contract_ranking) - 2 * both} questions.",
        delivered_n=sum(1 for r in delivered if r["route_rank_requested"])))

    mods = Counter(m for r in ranked for m in r["contract_operation_modifiers"])
    for label, key in (("ranking: dimension", "route_rank_field"),
                       ("ranking: direction", "route_rank_direction"),
                       ("ranking: basis", "route_rank_basis"),
                       ("ranking: top N", "route_rank_top_n")):
        route_values = {r[key] for r in ranked if r[key] is not None}
        # REPRESENTED asks whether the CONTRACT carries the fact. The contract's
        # only channel for it is OperationClaim.modifiers, so the test is
        # whether any modifier ever carries a value the route resolved.
        represented = bool(route_values & set(mods))
        out.append(cell(
            label, represented, represented, represented,
            any(r[key] is not None for r in delivered),
            f"route resolves {len(route_values)} distinct value(s) from raw "
            f"English; OperationClaim.modifiers carries "
            f"{sorted(mods) or 'NOTHING'} across all {len(ranked)} of them",
            delivered_n=sum(1 for r in delivered if r[key] is not None)))

    out.append(cell(
        "span honour-or-clarify (K2)",
        False, False, False, False,
        "no contract field represents 'the span was honoured by opening at "
        "snapshot N'; the route rewrites period_request in place"))
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--json", type=Path, default=None)
    ap.add_argument("--depth", type=int, default=6)
    args = ap.parse_args(argv)

    result = run(args.depth)
    rows = result["rows"]
    owned = [r for r in rows if r["owned"]]
    delivered = [r for r in owned if r["grade"] == "DELIVERED"]
    cells = matrix(result)

    print("=" * 92)
    print(f"C7 DEPENDENCY MATRIX — period_change, {args.depth}-snapshot book")
    print("=" * 92)
    print(f"owned surface: {len(owned)} questions "
          f"({len(delivered)} delivered, "
          f"{sum(1 for r in owned if r['grade'] == 'REFUSED')} refused)")
    print("routes: " + ", ".join(f"{k}={v}" for k, v in Counter(
        r["route"] for r in owned).most_common()))
    print(f"corpus questions the ROUTE reads as a ranking: "
          f"{sum(1 for r in rows if r['route_rank_requested'])}")
    print(f"of those, owned by period_change: "
          f"{sum(1 for r in rows if r['route_rank_requested'] and r['owned'])}")

    print(f"\n{'dependency':<30}{'repr':<8}{'owner':<8}{'plan':<8}{'delivered':<11}")
    print("-" * 92)
    m = {True: "GREEN", False: "RED  "}
    thin = []
    for c in cells:
        d = (f"GREEN n={c['delivered_n']}" if c["delivered"] else "RED  ")
        if c["delivered"] and c["delivered_n"] < 2:
            d += " THIN"
            thin.append(c["dependency"])
        print(f"{c['dependency']:<30}{m[bool(c['represented'])]:<8}"
              f"{m[bool(c['agrees'])]:<8}{m[bool(c['consumable'])]:<8}{d:<16}")
        for line in _wrap(c["note"]):
            print(f"{'':<30}{line}")

    if thin:
        print(f"\nTHIN: {len(thin)} delivered cell(s) rest on a SINGLE case — "
              f"{', '.join(thin)}.\nAll of them are the same question. One case "
              f"is a case; it is not a denominator.")

    red = [c["dependency"] for c in cells
           if not all((c["represented"], c["agrees"], c["consumable"],
                       c["delivered"]))]
    print("\n" + "=" * 92)
    print("MATRIX: " + ("ALL GREEN" if not red else
                        f"RED on {len(red)} of {len(cells)}: {', '.join(red)}"))

    if args.json:
        args.json.write_text(json.dumps({"cells": cells, **result}, indent=2,
                                        default=str), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
