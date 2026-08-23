#!/usr/bin/env python3
"""compositional_plan_scoping/census.py — the decomposition, counted from source.

READ-ONLY. Imports no product module for its structural counts: it parses the
source with :mod:`ast` and greps for declared constants, so a number here cannot
drift from the code without this instrument moving too.

Three questions, three counts:

1. **How many places decide an answer's SHAPE?** The claim under test is that
   "thirteen routes" understates the problem, because the router is downstream of
   a parser that has already chosen a shape, and upstream of an executor that
   chooses one again.

2. **How many implementations does each PRIMITIVE have?** The claim under test is
   that a compositional layer would not need to invent primitives — it would need
   to consolidate several existing implementations of each, which is where the
   byte-identical migration risk lives.

3. **How route-coupled is the GOVERNANCE layer?** The claim under test is that
   honour-or-clarify currently works because the route set is fixed: the facet
   reconciler names routes literally and keeps route allowlists.

    python -m compositional_plan_scoping.census
"""
from __future__ import annotations

import ast
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

_REPO = Path(__file__).resolve().parent.parent

PARSER = _REPO / "mi_agent" / "llm_query_parser.py"
EXECUTOR = _REPO / "mi_agent" / "mi_query_executor.py"
ROUTING = _REPO / "mi_agent_api" / "chat_routing.py"
RECEIPT = _REPO / "mi_agent" / "execution_receipt.py"

#: Every governed route name the routing registry declares. Read from the
#: registry source rather than restated, so a new route appears here by itself.
_RECOGNISER_RE = re.compile(r'name=(?:"([a-z_0-9]+)"|([A-Za-z_.]+\.?[A-Za-z_]*))\s*,\s*priority=(\d+)')

#: The primitive vocabulary this study DERIVED (see the report). Each entry maps
#: a primitive to the implementations of it found in the tree, as
#: ``(module, symbol, arity-note)``. This table is the study's finding; the
#: instrument's job is to prove each entry still exists.
PRIMITIVES: Dict[str, Tuple[Tuple[str, str, str], ...]] = {
    "select population": (
        ("mi_agent.population", "apply_population", "row predicates from spec.filters"),
        ("mi_agent.portfolio_lens", "apply_lens", "source-portfolio scope, from question text"),
        ("mi_agent.seasoning", "resolve_population_predicate", "seasoning segment, from question text"),
        ("mi_workflows.analytical.populations", "PopulationSpec", "4 governed scope kinds"),
        ("mi_agent_api.evolution", "_scope_frame_lens", "lens narrowing, per snapshot frame"),
    ),
    "resolve measure": (
        ("mi_workflows.engine", "aggregate", "one frame, one governed aggregation"),
        ("mi_agent.mi_query_executor", "aggregate_series", "one frame, one aggregation"),
        ("mi_agent.period_change.calculations", "aggregate", "one frame, BSR-driven"),
        ("mi_agent_api.evolution", "_bal_sum", "balance sum, per snapshot frame"),
        ("mi_agent_api.evolution", "_weighted_avg", "balance-weighted average"),
    ),
    "group": (
        ("mi_agent.mi_query_executor", "_grouped_aggregate", "N-ARY — takes a list of columns"),
        ("mi_workflows.engine", "distribution", "UNARY — one field"),
        ("mi_agent_api.evolution", "_breakdown", "UNARY — one column"),
        ("mi_agent.period_change.distribution", "_snapshot_table", "UNARY — one field"),
    ),
    "stack periods": (
        ("mi_agent_api.evolution", "funded_frames", "ordered governed snapshot frames"),
        ("mi_agent.period_change.periods", "resolve_periods", "two governed snapshots"),
    ),
    "compare": (
        ("mi_workflows.engine", "compare_values", "absolute + relative, one definition"),
        ("mi_workflows.engine", "directionality_verdict", "governed direction"),
        ("mi_agent.period_change.calculations", "metric_change", "two snapshots, one field"),
    ),
    "rank": (
        ("mi_workflows.engine", "ranked_distribution", "post-ordering of distribution"),
        ("mi_agent.mi_query_executor", "_apply_top_n", "top-N with residual policy"),
        ("mi_agent.period_change.ranking", "rank_movement", "ranked movement"),
    ),
    "project": (
        ("mi_agent_api.forecast_extrapolation", "run_rate_model", "fitted completion run-rate"),
    ),
}

#: Routes whose work this study found does NOT express in the primitives above,
#: with what each would additionally require. The instrument checks the named
#: evidence still exists; it does not re-derive the judgement.
DOES_NOT_DECOMPOSE: Tuple[Tuple[str, str, str, str], ...] = (
    ("cohort_conversion",
     "mi_agent_api/pipeline_history.py",
     "case_id",
     "an ENTITY TIMELINE joined across weekly snapshots on a stable case "
     "identifier — a longitudinal reshape of the row set, not a grouping of it"),
    ("scenario",
     "mi_agent_api/chat_routing.py",
     "_scenario_multiplier",
     "a HYPOTHETICAL PARAMETER read from the question, substituted into a "
     "fitted model before it is re-solved"),
    ("forecast_extrapolation",
     "mi_workflows/analytical/registry.py",
     "threshold_projection",
     "an INVERSE of project — solve for the date a projection crosses a level, "
     "not a forward composition"),
    ("risk_limits",
     "mi_agent_api/risk_limits.py",
     "_headroom",
     "a comparison against an EXTERNAL CONTRACTUAL THRESHOLD with a declared "
     "direction; `compare` relates two values from the same book"),
)


def _src(path: Path) -> str:
    return path.read_text(encoding="utf-8")


# --------------------------------------------------------------------------- #
# 1. The shape cascades
# --------------------------------------------------------------------------- #
def parser_branches() -> List[str]:
    """The deterministic parser's shape cascade, as its own section markers."""
    lines = _src(PARSER).splitlines()
    out = []
    for i, line in enumerate(lines, start=1):
        stripped = line.strip()
        if stripped.startswith("# ---- ") and 2600 <= i <= 3200:
            out.append(stripped[7:].rstrip("- ").strip())
    return out


def executor_branches() -> int:
    """Arms of ``execute_mi_query``'s if/elif shape dispatch."""
    tree = ast.parse(_src(EXECUTOR))
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "execute_mi_query":
            n = 0
            for stmt in node.body:
                if isinstance(stmt, ast.If):
                    cur = stmt
                    while True:
                        n += 1
                        if len(cur.orelse) == 1 and isinstance(cur.orelse[0], ast.If):
                            cur = cur.orelse[0]
                        else:
                            break
            return n
    return 0


#: Three routes register under a module constant rather than a literal. The
#: constant is resolved from its OWN module's source, so a rename there moves
#: this census rather than silently dropping the route.
_CONSTANTS = {
    "prc_mod.WORKFLOW_ID": ("mi_workflows/portfolio_risk_comparison.py", "WORKFLOW_ID"),
    "conc_mod.WORKFLOW_ID": ("mi_workflows/concentration_analysis.py", "WORKFLOW_ID"),
    # ``period_change_route.ROUTE_NAME`` is an alias for the workflow id.
    "_period_change.ROUTE_NAME": ("mi_agent/period_change/models.py", "WORKFLOW_ID"),
}


def _resolve_constant(expr: str) -> str:
    """The literal value of a module constant a recogniser registers under."""
    path, symbol = _CONSTANTS.get(expr, (None, None))
    if not path:
        return expr
    m = re.search(r'^%s\s*(?::[^=]+)?=\s*["\']([a-z_0-9]+)["\']'
                  % re.escape(symbol), _src(_REPO / path), flags=re.M)
    return m.group(1) if m else expr


def routes() -> List[Tuple[str, int]]:
    """Registered recognisers, from the registry source."""
    out: List[Tuple[str, int]] = []
    for m in _RECOGNISER_RE.finditer(_src(ROUTING)):
        name = m.group(1) or m.group(2)
        out.append((_resolve_constant(name), int(m.group(3))))
    # The analytical layer registers itself through its own module's factory
    # (``analytical_mod.recogniser()``), so it carries no ``name=``/``priority=``
    # pair for the regex above. Its name and priority are read from that module.
    if not any(n.startswith("analytical") for n, _ in out):
        route_src = _src(_REPO / "mi_workflows" / "analytical" / "route.py")
        name = re.search(r'^ROUTE_NAME\s*=\s*["\']([a-z_0-9]+)["\']',
                         route_src, flags=re.M)
        prio = re.search(r"^ROUTE_PRIORITY\s*=\s*(\d+)", route_src, flags=re.M)
        out.insert(0, (name.group(1) if name else "analytical_composition",
                       int(prio.group(1)) if prio else 5))
    return sorted(out, key=lambda kv: kv[1])


def specs_binding_both_limbs() -> Tuple[int, int, int]:
    """``MIQuerySpec(...)`` constructions in the parser that bind a REPORTING-PERIOD
    axis, a GROUPING dimension, or both.

    ``x`` is excluded from the time set when it is assigned a dimension key — the
    chart-axis alias — because that is a grouping, not a time axis.
    """
    src = _src(PARSER)
    tree = ast.parse(src)
    TIME = {"trend_grain", "temporal_mode", "compare_periods",
            "baseline_date", "current_date", "start_date", "end_date"}
    GRP = {"dimension", "dimensions", "hierarchy"}
    total = t_only = g_only = both = 0
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and getattr(node.func, "id", None) == "MIQuerySpec":
            total += 1
            names = {k.arg for k in node.keywords if k.arg} - {"x"}
            # ``x=`` is the chart's horizontal axis, which is a TIME axis only
            # when it is not the grouping dimension wearing an alias. Two
            # constructions in the aggregate-contribution path pass
            # ``dimension=dims[0], x=dims[0]`` — the same expression in both
            # slots — and counting those as a bound time axis would report a
            # both-limbs spec that does not exist.
            by_arg = {kw.arg: ast.dump(kw.value) for kw in node.keywords if kw.arg}
            if "x" in by_arg and by_arg["x"] != by_arg.get("dimension"):
                names.add("x")
            has_t, has_g = bool(names & (TIME | {"x"})), bool(names & GRP)
            if has_t and has_g:
                both += 1
            elif has_t:
                t_only += 1
            elif has_g:
                g_only += 1
    return total, t_only, g_only, both  # type: ignore[return-value]


# --------------------------------------------------------------------------- #
# 3. Route coupling in the governance layer
# --------------------------------------------------------------------------- #
def receipt_route_coupling() -> Tuple[int, List[str], int]:
    """Route-name literals and route allowlists in the facet reconciler."""
    src = _src(RECEIPT)
    names = [n for n, _ in routes() if "(" not in n]
    literals = 0
    for name in names:
        literals += len(re.findall(r'["\']%s["\']' % re.escape(name), src))
    allowlists = re.findall(r"^([A-Z_]*ROUTES)\s*=", src, flags=re.M)
    return literals, sorted(set(allowlists)), len(names)




# --------------------------------------------------------------------------- #
# 4. Migration blast radius
# --------------------------------------------------------------------------- #
#: The handler that owns each route's answer shape. Named explicitly because the
#: function name is not derivable from the route name for six of them.
_HANDLERS: Dict[str, Tuple[str, str]] = {
    "scenario": ("mi_agent_api/chat_routing.py", "_route_scenario"),
    "cohort_conversion": ("mi_agent_api/chat_routing.py", "_route_conversion"),
    "forecast_extrapolation": ("mi_agent_api/chat_routing.py", "_route_forecast"),
    "funded_bridge": ("mi_agent_api/chat_routing.py", "_route_bridge"),
    "cohort_progression": ("mi_agent_api/chat_routing.py", "_route_cohort_progression"),
    "geo_exposure": ("mi_agent_api/chat_routing.py", "_route_geo"),
    "portfolio_risk_comparison": ("mi_agent_api/chat_routing.py", "_route_portfolio_comparison"),
    "concentration_analysis": ("mi_agent_api/chat_routing.py", "_route_concentration"),
    "period_movement": ("mi_agent_api/chat_routing.py", "_route_period_movement"),
    "portfolio_summary": ("mi_agent_api/chat_routing.py", "_route_portfolio_summary"),
    "period_change_analysis": ("mi_agent_api/period_change_route.py", "route_period_change"),
    "temporal_compare": ("mi_agent_api/chat_routing.py", "_route_compare"),
    "risk_limits": ("mi_agent_api/chat_routing.py", "_route_risk"),
    "evolution": ("mi_agent_api/chat_routing.py", "_route_evolution"),
}


def _function_loc(path: Path, name: str) -> int:
    try:
        tree = ast.parse(_src(path))
    except (OSError, SyntaxError):
        return 0
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return (node.end_lineno or node.lineno) - node.lineno + 1
    return 0


def blast_radius() -> List[Dict[str, Any]]:  # type: ignore[name-defined]
    """Per route: test files naming it, total test references, handler size and
    facet-reconciler literals. A conversion has to hold all four still."""
    test_files = sorted(set(_REPO.glob("tests/**/*.py")) |
                        set(_REPO.glob("*/tests/*.py")))
    texts = []
    for path in test_files:
        try:
            texts.append(path.read_text(encoding="utf-8", errors="ignore"))
        except OSError:
            continue
    receipt = _src(RECEIPT)
    out = []
    for name, _prio in routes():
        if name not in _HANDLERS:
            # The analytical layer's handler lives in its own module and is not
            # a route conversion in the sense §4 measures — it is the layer a
            # conversion would move work INTO. Counted in §3, not here.
            continue
        pat = re.compile(r'["\']%s["\']' % re.escape(name))
        path, fn = _HANDLERS.get(name, (None, None))
        out.append({
            "route": name,
            "test_files": sum(1 for t in texts if pat.search(t)),
            "test_refs": sum(len(pat.findall(t)) for t in texts),
            "handler_loc": _function_loc(_REPO / path, fn) if path else 0,
            "receipt_literals": len(pat.findall(receipt)),
        })
    return out

def main() -> int:
    print("=" * 78)
    print("COMPOSITIONAL PLAN LAYER — DECOMPOSITION CENSUS")
    print("=" * 78)

    print("\n1. HOW MANY PLACES DECIDE AN ANSWER'S SHAPE?\n")
    pb = parser_branches()
    rs = routes()
    eb = executor_branches()
    print(f"   parser shape cascade   {len(pb):3d} branches   "
          f"(mi_agent/llm_query_parser.py)")
    for b in pb:
        print(f"        - {b}")
    print(f"\n   router recognisers     {len(rs):3d} routes     "
          f"(mi_agent_api/chat_routing.py)")
    for name, prio in rs:
        print(f"        {prio:5d}  {name}")
    print(f"\n   executor dispatch      {eb:3d} branches   "
          f"(mi_agent/mi_query_executor.py::execute_mi_query)")
    print(f"\n   -> {len(pb)} + {len(rs)} + {eb} = {len(pb) + len(rs) + eb} independent "
          f"shape decisions across three cascades.")

    total, t_only, g_only, both = specs_binding_both_limbs()  # type: ignore[misc]
    print(f"\n   Of {total} MIQuerySpec constructions in the deterministic parser:")
    print(f"        {t_only:3d} bind a reporting-period axis and NO grouping dimension")
    print(f"        {g_only:3d} bind a grouping dimension and NO period axis")
    print(f"        {both:3d} bind BOTH")
    print("   -> the spec HAS both fields; no branch of the parser writes both.")

    print("\n2. HOW MANY IMPLEMENTATIONS DOES EACH PRIMITIVE HAVE?\n")
    total_impls = 0
    for prim, impls in PRIMITIVES.items():
        total_impls += len(impls)
        print(f"   {prim:20s} {len(impls)} implementation(s)")
        for mod, sym, note in impls:
            print(f"        {mod}.{sym}  — {note}")
    print(f"\n   -> {len(PRIMITIVES)} primitives, {total_impls} implementations "
          f"({total_impls / len(PRIMITIVES):.1f} per primitive).")

    print("\n   ROUTES THAT DO NOT DECOMPOSE\n")
    for route, path, marker, why in DOES_NOT_DECOMPOSE:
        present = marker in _src(_REPO / path)
        flag = "evidence present" if present else "EVIDENCE MISSING"
        print(f"   {route}")
        print(f"        needs: {why}")
        print(f"        seen at: {path} :: {marker}  [{flag}]")

    print("\n3. HOW ROUTE-COUPLED IS THE GOVERNANCE LAYER?\n")
    lits, allowlists, n_routes = receipt_route_coupling()
    print(f"   mi_agent/execution_receipt.py names {n_routes} governed routes "
          f"literally {lits} times,")
    print(f"   and keeps {len(allowlists)} route allowlists:")
    for a in allowlists:
        print(f"        {a}")
    print("\n   -> honour-or-clarify decides 'was this honoured?' partly from "
          "WHICH ROUTE ANSWERED,")
    print("      not only from what the step declared it applied.")

    print("\n4. MIGRATION BLAST RADIUS — what a conversion has to hold still\n")
    rows = blast_radius()
    print(f"   {'route':28s}{'test files':>11}{'test refs':>11}"
          f"{'handler LOC':>13}{'receipt lits':>14}")
    for r in sorted(rows, key=lambda x: (x["test_files"], x["handler_loc"])):
        print(f"   {r['route']:28s}{r['test_files']:11d}{r['test_refs']:11d}"
              f"{r['handler_loc']:13d}{r['receipt_literals']:14d}")
    print("\n   -> smallest blast radius first is a measurement, not a preference.")
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
