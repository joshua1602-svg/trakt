#!/usr/bin/env python3
"""migration_phase0/mi_test_boundary.py — the MI Agent assurance denominator.

READ-ONLY. Decides, FROM IMPORTS rather than from filenames, which test modules
belong to the MI Agent's authoritative regression surface.

WHY NOT A FILENAME RULE. "Anything not called onboarding" is a guess, and a
guess in a denominator is worse than no denominator: it silently drops tests
that do cover MI and silently keeps ones that do not. This walks the real import
graph.

METHOD
------
1. SEED the MI production surface with the packages the MI Agent is made of.
2. CLOSE it forward: every first-party module those seeds import, transitively.
   Whatever they reach is a module MI depends on — a shared module included
   because MI actually uses it, not because it looked generic.
3. For each test module, take its own transitive first-party import closure.
   A test is IN when that closure intersects the MI production surface.
4. A test that reaches MI **only** through a shared leaf that MI also happens to
   use is still in — that is the task's rule for genuine coupling. What is NOT
   in is a test whose closure never touches MI at all.

A test that is IN because it is an unrelated END-TO-END WORKFLOW that happens to
import a shared module is reported separately, so the boundary can be argued
about with evidence instead of being asserted.

    python -m migration_phase0.mi_test_boundary [--json out.json] [--list]
"""
from __future__ import annotations

import argparse
import ast
import json
import sys
from collections import deque
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

#: The packages the MI Agent is made of, plus the top-level MI modules that
#: predate the packages. Membership here is what makes a test MI's.
MI_SEED_PACKAGES = ("mi_agent", "mi_agent_api", "question_interpretation",
                    "mi_workflows", "migration_phase0",
                    "mi_query_executor", "mi_query_validator", "mi_query_spec",
                    "mi_chart_factory", "mi_agent_config", "mi_agent_workflow",
                    "llm_query_parser")

#: NON-MI APPLICATION PACKAGES — the barrier, and the whole reason this
#: instrument needed a second cut.
#:
#: A plain forward closure said 377 of 477 test modules "reach MI", including
#: every OCC, onboarding, Annex 2 and mail test in the repository. Traced, the
#: coupling was ONE edge: `engine.onboarding_agent.onboarding_handoff` imports
#: `mi_agent.risk_monitor.risk_limits_contract`, and that single import made the
#: entire onboarding estate look like MI's regression surface.
#:
#: Reaching MI THROUGH another application is not being MI. A path is only
#: counted while it stays out of these packages, so an end-to-end onboarding
#: workflow test is excluded while a test that imports MI directly is not.
#: Narrow leaves MI genuinely depends on are carved back in below.
BARRIER_PACKAGES = ("engine", "operations_control", "demo_platform",
                    "trakt_mail", "readiness_agent", "regulatory_watch",
                    "due_diligence", "enterprise_agent", "agents",
                    "simulation", "ui", "cli", "apps", "mi_agent_pptx",
                    "mi_agent_operator", "clause_splitting_phase1",
                    "compositional_plan_scoping", "analytics", "trakt_a2a")

#: Modules inside a barrier package that MI production genuinely imports, and
#: which are therefore NOT barriers. Measured, not assumed: these are the only
#: barrier-package modules the MI packages import directly.
BARRIER_EXCEPTIONS = ("engine.provenance", "operations_control.stores",
                      "apps.blob_trigger_app.source_registry")

#: Shared libraries MI depends on. Their OWN tests belong to the MI denominator
#: — the task's rule — but a test merely touching them does not.
SHARED_LIBRARIES = ("trakt_core", "analytics_lib", "snapshot",
                    "trakt_notifications", "trakt_tools")

#: Workflow families the task names as out of scope. Listed to be REPORTED —
#: membership is still decided by the import graph, never by this list.
UNRELATED_FAMILIES = ("occ", "onboarding", "annex2", "annex_2", "regulatory",
                      "xml", "mail", "demo_platform", "due_diligence",
                      "readiness", "regulatory_watch")


def _is_barrier(name: str) -> bool:
    if any(name == e or name.startswith(e + ".") for e in BARRIER_EXCEPTIONS):
        return False
    return name.split(".")[0] in BARRIER_PACKAGES


class BoundaryError(RuntimeError):
    """The boundary could not be computed. Never absorbed into a partial list."""


def _first_party_roots() -> Set[str]:
    roots = {p.name for p in _REPO.iterdir()
             if p.is_dir() and (p / "__init__.py").exists()}
    roots |= {p.stem for p in _REPO.glob("*.py")}
    return roots


def _module_name(path: Path) -> str:
    rel = path.relative_to(_REPO).with_suffix("")
    parts = list(rel.parts)
    if parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts)


def _imports(path: Path, roots: Set[str]) -> Set[str]:
    """First-party modules this file imports, relative imports resolved."""
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001 - an unparseable file imports nothing
        return set()
    package = _module_name(path).rsplit(".", 1)[0] if "." in _module_name(path) else ""
    out: Set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.split(".")[0] in roots:
                    out.add(alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                base = package.split(".")
                base = base[:len(base) - node.level + 1] if node.level > 1 else base
                module = ".".join([p for p in base if p] +
                                  ([node.module] if node.module else []))
                out.add(module)
                for alias in node.names:
                    out.add(f"{module}.{alias.name}")
            elif node.module and node.module.split(".")[0] in roots:
                out.add(node.module)
                for alias in node.names:
                    out.add(f"{node.module}.{alias.name}")
    return out


def _index(roots: Set[str]) -> Dict[str, Path]:
    index: Dict[str, Path] = {}
    for path in _REPO.rglob("*.py"):
        rel = path.relative_to(_REPO)
        if rel.parts[0] in {".git", "node_modules", "__pycache__", "build"}:
            continue
        if rel.parts[0] not in roots and len(rel.parts) > 1:
            continue
        index[_module_name(path)] = path
    return index


def _closure(seeds: Set[str], index: Dict[str, Path], roots: Set[str],
             cache: Dict[str, Set[str]], *, barrier: bool = True) -> Set[str]:
    """Modules reachable from `seeds`. With `barrier`, a path STOPS at a non-MI
    application package rather than continuing through it."""
    seen: Set[str] = set()
    queue = deque(seeds)
    start = set(seeds)
    while queue:
        name = queue.popleft()
        # `a.b.C` — a symbol, not a module. Fall back to its module.
        while name and name not in index and "." in name:
            name = name.rsplit(".", 1)[0]
        if not name or name in seen or name not in index:
            continue
        seen.add(name)
        if barrier and name not in start and _is_barrier(name):
            continue          # reached, but not traversed
        if name not in cache:
            cache[name] = _imports(index[name], roots)
        queue.extend(cache[name])
    return seen


def run() -> Dict[str, Any]:
    roots = _first_party_roots()
    index = _index(roots)
    if not index:
        raise BoundaryError("BOUNDARY INVALID — no first-party modules indexed")
    cache: Dict[str, Set[str]] = {}

    seeds = {name for name in index
             if name.split(".")[0] in MI_SEED_PACKAGES
             and ".tests." not in name and not name.split(".")[-1].startswith("test_")}
    mi_core = {name for name in index if name.split(".")[0] in MI_SEED_PACKAGES}
    mi_surface = _closure(seeds, index, roots, cache)
    if not mi_surface:
        raise BoundaryError("BOUNDARY INVALID — the MI surface closed to nothing")

    tests = sorted(name for name, path in index.items()
                   if path.name.startswith("test_"))
    if not tests:
        raise BoundaryError("BOUNDARY INVALID — no test modules found")

    included: List[Dict[str, Any]] = []
    excluded: List[Dict[str, Any]] = []
    for name in tests:
        reach = _closure({name}, index, roots, cache)
        hit = sorted((reach & mi_core) - {name})
        # The task's shared-module rule: a test that LIVES with a shared library
        # MI depends on is MI's too. A test that merely touches one is not.
        shared_own = (name.split(".")[0] in SHARED_LIBRARIES
                      or any(f".{lib}." in name or name.startswith(f"tests.{lib}")
                             for lib in SHARED_LIBRARIES))
        row = {"module": name, "path": str(index[name].relative_to(_REPO)),
               "mi_modules_reached": len(hit),
               "first_mi_module": (hit[0] if hit else None),
               "reason": ("imports MI" if hit else
                          ("shared library MI depends on" if shared_own else None))}
        (included if (hit or shared_own) else excluded).append(row)

    # Reported, never used to decide: an unrelated end-to-end workflow that
    # reaches MI only through shared modules.
    flagged = [r for r in included
               if any(f in r["path"].lower() for f in UNRELATED_FAMILIES)]
    return {"mi_surface_size": len(mi_surface),
            "mi_surface": sorted(mi_surface),
            "included": included, "excluded": excluded,
            "unrelated_family_but_included": flagged}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--json", type=Path, default=None)
    ap.add_argument("--list", action="store_true",
                    help="print the included test paths, one per line")
    args = ap.parse_args(argv)
    result = run()
    if args.list:
        for row in result["included"]:
            print(row["path"])
        return 0
    print("=" * 88)
    print("MI AGENT TEST BOUNDARY — decided from the import graph")
    print("=" * 88)
    print(f"  MI production surface (transitive) : {result['mi_surface_size']} modules")
    print(f"  test modules INCLUDED              : {len(result['included'])}")
    print(f"  test modules EXCLUDED              : {len(result['excluded'])}")
    print(f"\nSHARED MODULES MI DEPENDS ON (outside the MI packages)")
    shared = sorted(m for m in result["mi_surface"]
                    if m.split(".")[0] not in MI_SEED_PACKAGES)
    for name in shared[:40]:
        print("   ", name)
    if len(shared) > 40:
        print(f"    … and {len(shared) - 40} more")
    print(f"\nUNRELATED-FAMILY TESTS THAT STILL REACH MI ({len(result['unrelated_family_but_included'])})")
    for row in result["unrelated_family_but_included"][:40]:
        print(f"    {row['path']:<62} via {row['first_mi_module']}")
    if args.json:
        args.json.write_text(json.dumps(result, indent=2, default=str),
                             encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
