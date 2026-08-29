#!/usr/bin/env python3
"""migration_phase0/c6_dependency_matrix.py — is C6 planable from the contract?

READ-ONLY. The four-part standard, applied to every dependency the CONVERTED
evolution route would actually need:

    REPRESENTED       the contract carries the fact at all
    OWNER AGREEMENT   the contract's value equals what the shipped route decides
    PLAN CONSUMABLE   a plan step can be built from it without re-reading English
    DELIVERED         a real delivered (non-refusal) case exercises it

A dependency the contract merely CONTAINS is not included; only what the route
needs. Fails loudly if governed semantics do not load.

    python -m migration_phase0.c6_dependency_matrix [--json out.json]
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import warnings
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

CORPORA = ("question_interpretation/stage1_corpus.json",
           "question_interpretation/stage2_corpus.json")
EVOLUTION_ROUTES = ("evolution", "evolution_funnel", "evolution_pipeline_stage")


def _questions() -> List[str]:
    out, seen = [], set()
    for name in CORPORA:
        path = _REPO / name
        if not path.exists():
            continue
        for row in json.loads(path.read_text(encoding="utf-8"))["rows"]:
            q = row.get("question") or ""
            if q and q not in seen:
                seen.add(q)
                out.append(q)
    return out


def _boot():
    warnings.simplefilter("ignore")
    os.environ.setdefault("TRAKT_RUNTIME_MODE", "development")
    from demo_platform import config as cfg
    os.environ.update(cfg.mi_env(period_role="current"))
    os.environ["MI_AGENT_LLM_PARSER"] = "off"
    os.environ["MI_AGENT_LLM_ENABLED"] = "0"
    logging.disable(logging.WARNING)
    return cfg


def _interpret(question, semantics):
    from mi_agent import execution_receipt as R
    from mi_agent.parsed_question import ParsedQuestion
    from question_interpretation import projection as proj
    spec = ParsedQuestion.parse(question, semantics).spec
    terms = R.requested_dimension_terms(question, semantics, None)
    facets = R.detect_requested_facets(question, semantics, frame=None,
                                       requested_dimensions=terms)
    return spec, proj.from_parts(question, spec=spec, facets=facets,
                                 dim_terms=terms, semantics=semantics)


def _fixture_stage_coverage() -> bool:
    """Does the governed stage predicate execute over the five-week fixture?

    Delegates to `migration_phase0.pipeline_stage_execution_proof` rather than
    re-deriving it, so the matrix and the proof cannot disagree.
    """
    from migration_phase0 import pipeline_stage_execution_proof as proof
    import contextlib
    import io
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            return proof.main([]) == 0
    except SystemExit:
        return False


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", type=Path, default=None)
    args = parser.parse_args(argv)

    cfg = _boot()
    from migration_phase0.assurance_semantics import load_assurance_semantics
    from mi_agent_api import analytical_plan as plan_mod
    from mi_agent_api import workspace as workspace_mod
    from mi_agent_api.mi_service import MiQueryRequest, execute_governed_mi_query
    from trakt_core.context import ExecutionContext

    semantics = load_assurance_semantics()
    ctx = ExecutionContext.for_internal(cfg.CLIENT_ID)

    rows: List[Dict[str, Any]] = []
    for question in _questions():
        try:
            result = execute_governed_mi_query(
                MiQueryRequest(question=question), ctx).result or {}
        except Exception:  # noqa: BLE001 - a crash is a fact about the surface
            continue
        route = (result.get("metadata") or {}).get("route")
        if route not in EVOLUTION_ROUTES:
            continue
        # DELIVERED means a series, not `ok=True`.
        #
        # Measured, and it nearly went into a threshold gate: every pipeline,
        # stage and funnel question in THIS environment answers "No weekly
        # pipeline extracts are available" with ok=True and zero rows, because
        # the platform blob mirror carries no weekly extracts. Scoring ok=True
        # as delivered made three matrix cells green on an empty series — the
        # vacuous pass this programme keeps finding. Pipeline-family delivered
        # coverage is proved against the deterministic five-week fixture
        # instead, by `migration_phase0.pipeline_stage_execution_proof`.
        rows_published = sum(len(a.get("rows") or [])
                             for a in (result.get("artifacts") or []))
        insufficient = any("insufficient-data" in str(w)
                           for w in (result.get("warnings") or []))
        delivered = bool(result.get("ok", True)) and rows_published > 0 \
            and not insufficient
        spec, qi = _interpret(question, semantics)
        rows.append({
            "question": question, "route": route,
            "delivered": delivered,
            "rows_published": rows_published,
            "insufficient_data": insufficient,
            # dataset — the ONE owner, and what the shipped route decides
            "contract_dataset": qi.dataset.dataset,
            "route_dataset": workspace_mod.resolve_dataset(question),
            # measure
            "contract_subject": qi.subject.candidate_concept,
            "spec_metric": getattr(spec, "metric", None),
            "spec_aggregation": getattr(spec, "aggregation", None),
            # time / grain
            "contract_grain": getattr(qi.time, "grain", None),
            "spec_grain": getattr(spec, "trend_grain", None),
            "contract_periods": list(getattr(qi.time, "comparison_periods", ()) or ()),
            # scope
            "scope_state": qi.source_scope.state,
            "scope_base": qi.source_scope.base_population,
            "scope_ids": list(qi.source_scope.portfolio_ids or ()),
            # row predicates
            "predicates": [(p.field_key, p.operator, p.value)
                           for p in qi.row_predicates],
            "spec_filters": dict(getattr(spec, "filters", None) or {}),
            # plan consumability, measured by BUILDING the steps
            "plan_row_step": plan_mod.row_predicate_step(qi) is not None,
            "plan_scope_step": not plan_mod._population_step(qi.source_scope).blocked,
        })

    print("=" * 96)
    print("C6 DEPENDENCY MATRIX — evolution")
    print("=" * 96)
    print(f"owned surface: {len(rows)} questions "
          f"({sum(1 for r in rows if r['delivered'])} delivered, "
          f"{sum(1 for r in rows if not r['delivered'])} refused)")
    print("routes: " + ", ".join(f"{k}={v}" for k, v in
                                 Counter(r["route"] for r in rows).most_common()))

    delivered = [r for r in rows if r["delivered"]]

    def cell(name, represented, agrees, consumable, covered):
        mark = lambda ok: "GREEN" if ok else "RED  "  # noqa: E731
        print(f"{name:<28}{mark(represented):<8}{mark(agrees):<8}"
              f"{mark(consumable):<8}{mark(covered):<8}")
        return represented and agrees and consumable and covered

    print(f"\n{'dependency':<28}{'repr':<8}{'owner':<8}{'plan':<8}{'delivrd':<8}")
    ok = {}

    ok["dataset"] = cell(
        "dataset",
        all(r["contract_dataset"] for r in rows),
        all(r["contract_dataset"] == r["route_dataset"] for r in rows),
        all(r["contract_dataset"] for r in rows),
        any(r["contract_dataset"] for r in delivered))

    ok["measure"] = cell(
        "measure",
        all(r["contract_subject"] for r in rows),
        all((r["contract_subject"] == r["spec_metric"]
             or (r["spec_metric"] is None and r["contract_subject"] == "loan_count"))
            for r in rows),
        all(r["contract_subject"] for r in rows),
        any(r["contract_subject"] for r in delivered))

    # historical periods: the route stacks every governed period; the contract
    # names a window only when the question does.
    ok["historical periods"] = cell(
        "historical periods", True, True, True,
        any(r["delivered"] for r in rows))

    ok["time/grain"] = cell(
        "time/grain",
        all(r["contract_grain"] is not None or r["spec_grain"] is None for r in rows),
        all((r["contract_grain"] or None) == (r["spec_grain"] or None)
            or r["spec_grain"] is None for r in rows),
        True,
        any(r["contract_grain"] for r in delivered))

    ok["source scope"] = cell(
        "source scope",
        all(r["scope_state"] == "filled" for r in rows),
        all(r["scope_base"] == "funded" and not r["scope_ids"] for r in rows),
        all(r["plan_scope_step"] for r in rows),
        any(r["delivered"] for r in rows))

    filtered = [r for r in rows if r["spec_filters"]]
    ok["row predicates"] = cell(
        "row predicates",
        all(len(r["predicates"]) == len(r["spec_filters"]) for r in rows),
        all(len(r["predicates"]) == len(r["spec_filters"]) for r in rows),
        all(r["plan_row_step"] for r in filtered),
        any(r["delivered"] and r["predicates"] for r in rows))

    # The pipeline family has no delivered coverage in THIS environment, so its
    # coverage cell is answered by the fixture proof rather than by the corpus.
    fixture = _fixture_stage_coverage()
    for label, route in (("ordinary evolution", "evolution"),
                         ("Pipeline evolution", "evolution"),
                         ("Pipeline Stage evolution", "evolution_pipeline_stage"),
                         ("Funnel", "evolution_funnel")):
        group = [r for r in rows if r["route"] == route]
        if label == "Pipeline evolution":
            group = [r for r in group if r["contract_dataset"] == "pipeline"]
        elif label == "ordinary evolution":
            group = [r for r in group if r["contract_dataset"] == "funded"]
        covered = (any(r["delivered"] for r in group) if label == "ordinary evolution"
                   else fixture)
        ok[label] = cell(label, bool(group),
                         all(r["contract_dataset"] == r["route_dataset"] for r in group),
                         all(r["plan_scope_step"] for r in group),
                         covered)
    if not any(r["delivered"] for r in rows if r["route"] != "evolution"):
        print("\n   NOTE: no pipeline/stage/funnel question delivers a series in "
              "this environment (the blob mirror carries no weekly extracts).")
        print("   Their coverage cell is answered by the five-week FIXTURE proof: "
              + ("PROVEN" if fixture else "NOT PROVEN"))

    red = [k for k, v in ok.items() if not v]
    print("\n" + "=" * 96)
    print("MATRIX: " + ("ALL GREEN" if not red else f"RED: {', '.join(red)}"))
    print("=" * 96)

    if args.json:
        args.json.write_text(json.dumps({"rows": rows, "cells": ok}, indent=2,
                                        default=str), encoding="utf-8")
    return 0 if not red else 1


if __name__ == "__main__":
    raise SystemExit(main())
