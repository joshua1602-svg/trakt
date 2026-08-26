#!/usr/bin/env python3
"""migration_phase0/analytical_meaning_census.py — who can still decide meaning?

READ-ONLY. Answers one question across the SEVEN migrated core routes:

    how many places can still independently decide measure, dataset,
    population, time, LEVEL/MOVEMENT, dimension, ranking or other
    analytical meaning?

"Independently" means the decision is taken there rather than read from the
governed contract. Two mechanisms count, and they are counted separately
because they are found by different means and fixed by different work:

  K1  RAW-QUESTION READ   the route hands the question's text to something
                          that decides meaning from it, or uses the text itself
                          for a decision of its own.
  K2  LOCAL VOCABULARY    the route matches text against business language
                          declared IN ITS OWN MODULE. A regex is not delegation:
                          the decision is taken here.

A third mechanism — choosing a default when the contract is SILENT — is NOT
counted statically here, because an AST cannot reliably tell a governed
fallback from an invented one. It is measured by execution instead
(`c7_independent_audit` check D, and the no-implicit-measure / no-implicit-
period tests), and the census reports that separation rather than hiding it.

    python -m migration_phase0.analytical_meaning_census [--json out.json]
"""
from __future__ import annotations

import argparse
import ast
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from migration_phase0.semantic_owner_inventory import (  # noqa: E402
    PASSTHROUGH_CALLEES, SEMANTIC_CALLEES)

CHAT = _REPO / "mi_agent_api" / "chat_routing.py"
PCR = _REPO / "mi_agent_api" / "period_change_route.py"

#: The seven migrated core routes, each with the handler that answers it and
#: the module that handler lives in. C6 converted THREE route identities behind
#: one handler; they are one conversion and one handler, and are named as such.
CONVERSIONS: Tuple[Tuple[str, str, str, Path], ...] = (
    ("C1", "portfolio_summary", "_route_portfolio_summary", CHAT),
    ("C2", "period_movement", "_route_period_movement", CHAT),
    ("C3", "geo_exposure", "_route_geo", CHAT),
    ("C4", "funded_bridge", "_route_bridge", CHAT),
    ("C5", "temporal_compare", "_route_compare", CHAT),
    ("C6", "evolution (+_funnel, +_pipeline_stage)", "_route_evolution", CHAT),
    ("C7", "period_change_analysis", "route_period_change", PCR),
    # NOT a core route, and reported anyway. `concentration_analysis` sits on
    # the GENERIC funded-book path and was the estate's last independent
    # whole-question interpreter; a census that stopped at the seven would have
    # scored 0 while it still read the sentence after the claim.
    ("Cx", "concentration_analysis", "_route_concentration", CHAT),
)

#: K0 — RECOGNITION. What each route's registered `recognise` predicate reads.
#: Counted SEPARATELY from K1/K2 because recognition-by-wording is the estate's
#: accepted routing model, not a migration defect: a recogniser returns a
#: boolean and decides which analysis runs, never what it means. It is reported
#: because "which analysis" is still analytical meaning, and a census that
#: silently omitted it would overstate the closure.
RECOGNISERS: Dict[str, Tuple[str, str]] = {
    "C1": ("_is_portfolio_summary(question, spec)", "raw question + spec"),
    "C2": ("_is_period_movement(question)", "raw question"),
    "C3": ("_is_geo_exposure(question, spec, view)", "raw question + spec + view"),
    "C4": ("spec.bridge_query and not _is_aggregate_contribution_question(question)",
           "raw question + spec"),
    "C5": ("spec.temporal_mode == 'compare'", "CONTRACT ONLY"),
    "C6": ("_is_evolution(question, spec)", "raw question + spec"),
    "C7": ("recognise_request -> recognise(question, spec, view, semantics)",
           "raw question + spec + view"),
    "Cx": ("_recognise_concentration -> is_concentration_question(question, spec)",
           "raw question + spec"),
}

#: Names that ARE the raw question.
QUESTION_NAMES = {"question", "q", "raw_question"}

#: Module-level constants encoding BUSINESS LANGUAGE, per module. Listed
#: explicitly, because a frozenset of strings is indistinguishable from a lookup
#: table until you read what is in it. A name that appears here and no longer
#: exists in the module is reported, so a stale entry cannot quietly shrink the
#: count.
#: Names retired by earlier conversions are NOT listed: the instrument reports a
#: listed-but-undefined name as stale, and a stale entry that stayed would read
#: as a decision site the estate had already removed. `_BRIDGE_MARKERS`,
#: `_COMPARE_MARKERS`, `_CONCENTRATION_MARKERS`, `_OVERVIEW_MARKERS`,
#: `_SCENARIO_MARKERS`, `TREND_MARKERS`, `_NARRATIVE_RANK_SUBJECTS`,
#: `_RANK_SUBJECT_LEAD_RE` and `_RANK_SUBJECT_SKIP` were all deleted by C1-C7
#: and are gone from the modules.
VOCABULARY: Dict[str, set] = {
    "chat_routing": {
        "_SUMMARY_MARKERS", "_MOVEMENT_MARKERS", "_GEO_MARKERS",
        "_EVOLUTION_MARKERS",
    },
    "period_change_route": set(),
}

#: What each semantic callee decides, mapped onto the concepts the question
#: names. Anything not mapped is reported under "other analytical meaning".
CONCEPTS: Dict[str, str] = {
    "measure selection": "measure", "measure substitution": "measure",
    "statistic": "measure",
    "dataset selection (funded vs pipeline)": "dataset",
    "row filters": "population", "row population": "population",
    "source scope": "population", "source scope + caller precedence": "population",
    "source scope explicitness": "population", "source scope identity": "population",
    "source scope disclaimer": "population", "comparison sides": "population",
    "time grain": "time", "time window": "time", "time axis": "time",
    "period selection": "time",
    "comparison intent": "LEVEL/MOVEMENT",
    "grouping dimensions": "dimension",
    "requested facets (multi-concept)": "dimension",
    "ranking intent": "ranking", "ranking subject": "ranking",
    "ranking subject + direction": "ranking",
    "route shape": "other analytical meaning",
    "analytical intent": "other analytical meaning",
    "answer type": "other analytical meaning",
    "whole-question delegation to a workflow": "other analytical meaning",
    "scenario magnitude": "other analytical meaning",
    "scenario target": "other analytical meaning",
}


class CensusError(RuntimeError):
    """The census could not be measured. Never absorbed into a clean zero."""


def _module_name(path: Path) -> str:
    return path.stem


def _functions(tree: ast.AST) -> Dict[str, ast.FunctionDef]:
    return {n.name: n for n in ast.walk(tree)
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}


def _vocabulary_for(path: Path) -> set:
    return VOCABULARY.get(_module_name(path), set())


def _is_local_vocabulary(func: ast.AST, vocab: set) -> bool:
    """True when the call applies THIS MODULE's own business language."""
    if isinstance(func, ast.Attribute):
        base = func.value
        if isinstance(base, ast.Name):
            return base.id == "re" or base.id in vocab
        return False
    if isinstance(func, ast.Name):
        return func.id in vocab
    return False


def _census_function(node: ast.FunctionDef, vocab: set) -> Dict[str, List[Any]]:
    """K1 and K2 sites inside one function body."""
    k1: List[Dict[str, Any]] = []
    k2: List[Dict[str, Any]] = []
    for sub in ast.walk(node):
        if not isinstance(sub, ast.Call):
            continue
        args = list(sub.args) + [k.value for k in sub.keywords]
        carries_question = any(
            isinstance(n, ast.Name) and n.id in QUESTION_NAMES
            for a in args for n in ast.walk(a))
        if not carries_question:
            continue
        if _is_local_vocabulary(sub.func, vocab):
            k2.append({"line": sub.lineno, "callee": ast.unparse(sub.func)})
            continue
        name = (sub.func.attr if isinstance(sub.func, ast.Attribute)
                else getattr(sub.func, "id", "…"))
        if name in PASSTHROUGH_CALLEES:
            continue
        decides = SEMANTIC_CALLEES.get(name)
        if decides:
            k1.append({"line": sub.lineno, "callee": name, "decides": decides,
                       "concept": CONCEPTS.get(decides, "other analytical meaning")})
    return {"K1": k1, "K2": k2}


def run() -> Dict[str, Any]:
    trees: Dict[Path, ast.AST] = {}
    for path in {p for *_, p in CONVERSIONS}:
        trees[path] = ast.parse(path.read_text(encoding="utf-8"))

    # A vocabulary name that no longer exists is REPORTED, never ignored: a
    # stale entry would silently shrink K2 and read as progress.
    stale: List[str] = []
    for path, tree in trees.items():
        assigned = {t.id for n in ast.walk(tree) if isinstance(n, ast.Assign)
                    for t in n.targets if isinstance(t, ast.Name)}
        for name in _vocabulary_for(path):
            if name not in assigned:
                stale.append(f"{_module_name(path)}.{name}")

    for key in RECOGNISERS:
        if key not in {c for c, *_ in CONVERSIONS}:
            raise CensusError(f"CENSUS INVALID — recogniser row {key!r} names no "
                              f"conversion")
    rows: List[Dict[str, Any]] = []
    for conversion, route, handler, path in CONVERSIONS:
        functions = _functions(trees[path])
        if handler not in functions:
            raise CensusError(
                f"CENSUS INVALID — handler {handler!r} not found in {path.name}; "
                f"a renamed handler must not read as a route that decides nothing")
        found = _census_function(functions[handler], _vocabulary_for(path))
        predicate, reads = RECOGNISERS[conversion]
        rows.append({"conversion": conversion, "route": route,
                     "handler": handler, "module": path.name,
                     "recogniser": predicate, "recogniser_reads": reads,
                     "K0": 0 if reads == "CONTRACT ONLY" else 1, **found})
    if len(rows) != len(CONVERSIONS):
        raise CensusError("CENSUS INVALID — not every conversion was measured")
    return {"rows": rows, "stale_vocabulary": sorted(stale)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--json", type=Path, default=None)
    args = ap.parse_args(argv)

    result = run()
    print("=" * 92)
    print("ANALYTICAL MEANING — independent decision sites across the seven "
          "migrated core routes")
    print("=" * 92)
    print(f"\n{'':<4}{'route':<40}{'K0 recognition':>15}"
          f"{'K1 raw-question':>16}{'K2 local vocab':>16}")
    print("-" * 92)
    k0_total = k1_total = k2_total = 0
    concepts: Dict[str, List[str]] = {}
    for row in result["rows"]:
        k0_total += row["K0"]
        k1_total += len(row["K1"])
        k2_total += len(row["K2"])
        print(f"{row['conversion']:<4}{row['route']:<40}"
              f"{row['K0']:>15}{len(row['K1']):>16}{len(row['K2']):>16}")
        print(f"      recognise: {row['recogniser']}")
        print(f"      reads    : {row['recogniser_reads']}")
        for site in row["K1"]:
            concepts.setdefault(site["concept"], []).append(
                f"{row['route']}:{site['line']} {site['callee']}")
        for site in row["K1"]:
            print(f"      :{site['line']:<6}{site['callee']:<32} -> "
                  f"{site['decides']}")
        for site in row["K2"]:
            print(f"      :{site['line']:<6}{site['callee']:<32} -> "
                  f"local vocabulary")
    print("-" * 92)
    print(f"{'':<4}{'TOTAL':<40}{k0_total:>15}{k1_total:>16}"
          f"{k2_total:>16}")
    print(f"\n    POST-CLAIM decision sites (K1 + K2): {k1_total + k2_total}")
    print(f"    RECOGNITION sites reading wording (K0): {k0_total} of "
          f"{len(result['rows'])}")

    print("\nBY CONCEPT")
    for concept in ("measure", "dataset", "population", "time",
                    "LEVEL/MOVEMENT", "dimension", "ranking",
                    "other analytical meaning"):
        sites = concepts.get(concept, [])
        print(f"   {concept:<28}{len(sites):>3}   {sites}")

    if result["stale_vocabulary"]:
        print("\nSTALE VOCABULARY ENTRIES (no longer defined; not counted):")
        for name in result["stale_vocabulary"]:
            print("   ", name)

    print("\nNOT COUNTED HERE, AND WHY")
    print("   implicit defaults when the contract is silent — an AST cannot "
          "tell a\n   governed fallback from an invented one. Measured by "
          "execution instead.")
    if args.json:
        args.json.write_text(json.dumps(result, indent=2, default=str),
                             encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
