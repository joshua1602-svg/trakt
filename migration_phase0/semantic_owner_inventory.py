#!/usr/bin/env python3
"""migration_phase0/semantic_owner_inventory.py — who decides meaning, and where?

READ-ONLY. Target-state closure §4, and the instrument that answers it.

The compositional target says interpretation owns meaning ONCE and nothing
downstream rereads the question. Every phase so far has discovered a violation
of that by converting a route and hitting it. This finds them all at once,
statically, before another conversion starts.

WHAT IT MEASURES. For each route handler, every call that passes the RAW
QUESTION onward. A handler that never passes `question` to anything cannot be
independently interpreting it; a handler that does is either reading meaning
(a second semantic owner) or passing it for presentation, and the two are
distinguished by what the callee is.

Deliberately AST-based and not a grep: `question` appears in strings, comments
and envelope payloads all over this module, and a text search cannot tell a
semantic read from an answer field. Only a call argument counts.

CLASSIFICATION, per the task's four buckets:

    B_SEMANTIC     the callee decides MEANING from the text — scope, role,
                   measure, population, comparison side, ranking. A duplicate
                   semantic owner, and the thing the target removes.
    C_PRESENTATION the callee puts the wording in front of the reader — an
                   echo, a label, a refusal that quotes what was asked.
    A_EXECUTION    the callee applies something already decided.
    D_SPECIALIST   a route the compositional layer does not claim.

The target is NOT zero downstream branches. It is zero B_SEMANTIC inside the
GENERIC funded-book path.

    python -m migration_phase0.semantic_owner_inventory [--out FILE]
"""
from __future__ import annotations

import argparse
import ast
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

#: The GENERIC funded-book estate — the routes the seven primitives claim.
GENERIC_ROUTES: Dict[str, str] = {
    "_route_portfolio_summary": "portfolio_summary",
    "_route_period_movement": "period_movement",
    "_route_compare": "temporal_compare",
    "_route_evolution": "evolution",
    "_route_bridge": "funded_bridge",
    "_route_geo": "geo_exposure",
    "_route_portfolio_comparison": "portfolio_risk_comparison",
    "_route_concentration": "concentration_analysis",
    # In `mi_agent_api/period_change_route.py`, not in `chat_routing`. Omitted
    # from the first draft of this inventory FOR THAT REASON ALONE, which is
    # exactly the failure mode the instrument exists to prevent: a route whose
    # handler lives elsewhere is not a route that decides less.
    "route_period_change": "period_change_analysis",
}

#: Deliberately OUTSIDE the compositional contract. Their semantics are not
#: forced into it, per §3.
SPECIALIST_ROUTES: Dict[str, str] = {
    "_route_scenario": "scenario",
    "_route_conversion": "cohort_conversion",
    "_route_forecast": "forecast_extrapolation",
    "_route_cohort_progression": "cohort_progression",
    "_route_risk": "risk_limits",
    "_route_concentration_tests": "concentration_tests",
}

#: Callees that DECIDE MEANING from raw text. Each is a semantic owner, and the
#: concept it owns is named so the coverage matrix can say whether the contract
#: carries it.
SEMANTIC_CALLEES: Dict[str, str] = {
    # source-portfolio scope
    "_resolve_lens": "source scope + caller precedence",
    "resolve_lens": "source scope",
    "resolve_lens_with_default": "source scope + caller precedence",
    "mentions_portfolio": "source scope explicitness",
    "names_governed_portfolio": "source scope explicitness",
    "names_total_scope": "source scope explicitness",
    "names_selected_scope": "source scope explicitness",
    "resolve_comparison_lenses": "comparison sides",
    "context_id": "source scope identity",
    "lens_from_term": "source scope",
    "disclaims_scope": "source scope disclaimer",
    # measures / statistics
    "_measure_hits": "measure selection",
    "detect_measure_set": "measure selection",
    "detect_measure_substitution": "measure substitution",
    "requested_statistic": "statistic",
    "executed_measure_concepts": "measure selection",
    # dimensions / grouping
    "_explicit_dimensions": "grouping dimensions",
    "requested_dimension_terms": "grouping dimensions",
    "detect_requested_facets": "requested facets (multi-concept)",
    # filters / population
    "_parse_filters": "row filters",
    "drill_population_facets": "row population",
    # time
    "requested_unit": "time grain",
    "requested_span": "time window",
    "time_axis_request": "time axis",
    "finer_than": "time grain",
    "requested_period_labels": "period selection",
    # intent / shape
    "is_comparative": "comparison intent",
    "classify": "analytical intent",
    "asked": "answer type",
    "detect_unranked_superlative": "ranking intent",
    "_is_portfolio_summary": "route shape",
    "_names_something_else": "route shape",
    "_is_period_movement": "route shape",
    # dataset / view. Its own docstring names it "THE SECOND OWNER".
    "_dataset_for": "dataset selection (funded vs pipeline)",
    "resolve_active_view": "dataset selection (funded vs pipeline)",
    "undisclaimed_mention": "dataset selection (funded vs pipeline)",
    # WHOLE-QUESTION DELEGATION. These hand the raw text to a workflow that
    # runs its own recognition predicates and rejection rules over it, so the
    # interpretation happening inside them is invisible to the contract. The
    # deepest form of duplicate ownership, and the easiest to miss: the call
    # site looks like execution.
    "run_portfolio_risk_comparison": "whole-question delegation to a workflow",
    "run_concentration_analysis": "whole-question delegation to a workflow",
    "rejection_reason": "route shape",
    # specialist parameter reads
    "_scenario_multiplier": "scenario magnitude",
    "_scenario_target": "scenario target",
    # ranking, in `period_change_route`
    "resolve_rank_intent": "ranking subject + direction",
    "_rank_subject": "ranking subject",
    "recognise_request": "route shape",
    "recognise": "route shape",
}

#: Callees that put the WORDING in front of a reader. Not semantic owners.
#: Route delegation and local names that are NOT semantic reads. Listed
#: explicitly rather than left to a default, so the audit is recorded.
PASSTHROUGH_CALLEES: Set[str] = {
    # `risk_limits` hands a concentration question to its own sibling handler;
    # the delegation carries the question, the decision was already made.
    "_route_concentration_tests",
    # A STRUCTURED REQUEST, not a resolver. Its own docstring: every field
    # "come[s] from the execution context, never from the question" — the
    # question travels on it for audit and prose, and the decisions were
    # already made. Audited rather than assumed.
    "PeriodChangeRequest",
    # CONVERSION 1. The contract CONSUMER: it takes the interpretation and
    # plans from it, and calls no resolver. The question travels on its
    # signature for the deferral path and for nothing else. Audited, not
    # assumed — `test_the_route_does_not_call_the_lens_resolver` pins it.
    "_summary_population",
    # A local variable named `answer`, called as a formatter. Not a resolver.
    "answer",
}

PRESENTATION_CALLEES: Set[str] = {
    "_envelope", "_error_envelope", "_summary_kpi_artifact", "_chart_artifact",
    "_table_artifact", "_risk_artifact", "_source", "_upper_first",
    "_rank_refusal_envelope", "build_rank_answer", "_failure_envelope", "_render",
    "_sentence_join", "clarification", "refusal_message",
}


def _callee_name(node: ast.Call) -> str:
    func = node.func
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        return func.attr
    return ""


def _passes_question(node: ast.Call) -> bool:
    """Whether this call hands the RAW QUESTION on."""
    names = []
    for arg in node.args:
        if isinstance(arg, ast.Name):
            names.append(arg.id)
    for kw in node.keywords:
        if isinstance(kw.value, ast.Name):
            names.append(kw.value.id)
        if isinstance(kw.value, ast.Attribute) and kw.value.attr == "question":
            names.append("question")
    return "question" in names


def scan(path: Path) -> List[Dict[str, Any]]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    out: List[Dict[str, Any]] = []
    for func in ast.walk(tree):
        if not isinstance(func, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        route = GENERIC_ROUTES.get(func.name) or SPECIALIST_ROUTES.get(func.name)
        if route is None:
            continue
        kind = "generic" if func.name in GENERIC_ROUTES else "specialist"
        for node in ast.walk(func):
            if not isinstance(node, ast.Call) or not _passes_question(node):
                continue
            callee = _callee_name(node)
            if callee in SEMANTIC_CALLEES:
                bucket = "D_SPECIALIST" if kind == "specialist" else "B_SEMANTIC"
                concept = SEMANTIC_CALLEES[callee]
            elif callee in PASSTHROUGH_CALLEES:
                bucket, concept = "A_EXECUTION", "route delegation; audited"
            elif callee in PRESENTATION_CALLEES:
                bucket, concept = "C_PRESENTATION", "wording shown to the reader"
            else:
                # DEFAULTS TO UNCLASSIFIED, NOT TO BENIGN. The first draft of
                # this instrument defaulted to A_EXECUTION and under-reported by
                # three concepts — `_dataset_for`, and two whole-question
                # delegations into workflows — every one of which reads as
                # execution at the call site. An inventory whose default is
                # "harmless" cannot find what it was built to find.
                bucket, concept = "UNCLASSIFIED", "NOT YET CLASSIFIED — audit"
            out.append({"route": route, "kind": kind, "handler": func.name,
                        "line": node.lineno, "callee": callee,
                        "bucket": bucket, "concept": concept})
    return out


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=str(
        _REPO / "migration_phase0" / "SEMANTIC_OWNER_INVENTORY.json"))
    args = ap.parse_args(argv)

    rows: List[Dict[str, Any]] = []
    for module in ("chat_routing.py", "period_change_route.py"):
        rows.extend(scan(_REPO / "mi_agent_api" / module))

    print("=" * 112)
    print("SEMANTIC OWNERS DOWNSTREAM OF INTERPRETATION")
    print("=" * 112)

    for kind, title in (("generic", "GENERIC FUNDED-BOOK ESTATE — the target"),
                        ("specialist", "SPECIALIST ROUTES — outside the contract")):
        print(f"\n{title}")
        print("-" * 112)
        routes = sorted({r["route"] for r in rows if r["kind"] == kind})
        covered = (sorted(GENERIC_ROUTES.values()) if kind == "generic"
                   else sorted(SPECIALIST_ROUTES.values()))
        for route in covered:
            hits = [r for r in rows if r["route"] == route]
            semantic = [r for r in hits if r["bucket"] == "B_SEMANTIC"]
            other = [r for r in hits if r["bucket"] != "B_SEMANTIC"]
            unk = [r for r in hits if r["bucket"] == "UNCLASSIFIED"]
            flag = "CLEAN" if not semantic else f"{len(semantic)} SEMANTIC"
            if unk:
                flag += f" +{len(unk)}?"
            print(f"  {route:28s} {flag:14s} "
                  f"(+{len(other) - len(unk)} presentation/pass-through)")
            for row in sorted(semantic, key=lambda r: r["line"]):
                print(f"       :{row['line']:<5d} {row['callee']:32s} "
                      f"-> {row['concept']}")
        if not routes:
            print("  (none found)")

    unclassified = [r for r in rows if r["bucket"] == "UNCLASSIFIED"]
    if unclassified:
        print("\n" + "!" * 112)
        print("UNCLASSIFIED CALL SITES — each must be judged before this "
              "inventory can be quoted as complete:")
        for row in sorted(unclassified, key=lambda r: (r["route"], r["line"])):
            print(f"   {row['route']:28s} :{row['line']:<5d} {row['callee']}")
        print("!" * 112)

    generic_semantic = [r for r in rows
                        if r["kind"] == "generic" and r["bucket"] == "B_SEMANTIC"]
    concepts = sorted({r["concept"] for r in generic_semantic})
    print("\n" + "=" * 112)
    print(f"B_SEMANTIC decisions inside the GENERIC path: {len(generic_semantic)}")
    print(f"distinct concepts re-decided downstream    : {len(concepts)}")
    for concept in concepts:
        owners = sorted({r["route"] for r in generic_semantic
                         if r["concept"] == concept})
        print(f"   {concept:38s} {owners}")
    print("=" * 112)

    Path(args.out).write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
    print(f"\nwrote {Path(args.out).relative_to(_REPO)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
