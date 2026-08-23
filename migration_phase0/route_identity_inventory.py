#!/usr/bin/env python3
"""migration_phase0/route_identity_inventory.py — where route IDENTITY decides governance.

READ-ONLY. Parses the source; imports no product module. Answers Objective 3B:
every PRODUCTION site where the route name — rather than what execution declared
— decides whether a facet is applied, permitted, stamped, unavailable/lost, or
whether an answer may proceed.

    python -m migration_phase0.route_identity_inventory
"""
from __future__ import annotations

import ast
import re
import sys
from pathlib import Path
from typing import Dict, List, Set, Tuple

_REPO = Path(__file__).resolve().parent.parent

#: Production modules in the governance path. Tests are counted separately and
#: are NOT a migration cost — they are the bar the migration must keep clearing.
_PRODUCTION = (
    "mi_agent/execution_receipt.py",
    "mi_agent_api/chat_routing.py",
    "mi_agent_api/mi_service.py",
    "mi_agent_api/adapters.py",
    "mi_workflows/analytical/route.py",
)



#: A route name on a line is one of two things, and only one of them is a
#: migration cost.
#:
#: DECLARATION — the route naming ITSELF: ``route="evolution"`` on an envelope,
#:   ``name="evolution"`` in the recogniser registry, ``ROUTE_NAME = "..."``.
#:   These are the channel through which execution says what it was. A
#:   compositional layer replaces WHAT declares, not the fact of declaring, so
#:   these do not have to be removed.
#:
#: DECISION — a CONSUMER branching on which route answered: ``route in
#:   TEMPORAL_ROUTES``, ``_ROUTE_LABELS.get(route)``, ``route ==
#:   "geo_exposure"``. These are the sites where route identity, rather than
#:   what execution declared it applied, decides governance. These are the
#:   migration cost.
_DECLARATION_RE = re.compile(
    r'(?:^|[^_\w])(?:route|name|ROUTE_NAME|WORKFLOW_ID)\s*=\s*["\']'
    r'|Recogniser\(|route_owner=|"workflow":')


def classify(line: str) -> str:
    """``declaration`` (route names itself) or ``decision`` (consumer branches)."""
    stripped = line.strip()
    if stripped.startswith("#") or stripped.startswith("*"):
        return "comment"
    if _DECLARATION_RE.search(line):
        return "declaration"
    return "decision"


def route_names() -> List[str]:
    """The governed route names, resolved from the registry source."""
    src = (_REPO / "mi_agent_api" / "chat_routing.py").read_text(encoding="utf-8")
    names: List[str] = []
    for m in re.finditer(r'name=(?:"([a-z_0-9]+)"|([A-Za-z_.]+))\s*,\s*priority=\d+', src):
        names.append(m.group(1) or m.group(2))
    constants = {
        "prc_mod.WORKFLOW_ID": ("mi_workflows/portfolio_risk_comparison.py", "WORKFLOW_ID"),
        "conc_mod.WORKFLOW_ID": ("mi_workflows/concentration_analysis.py", "WORKFLOW_ID"),
        "_period_change.ROUTE_NAME": ("mi_agent/period_change/models.py", "WORKFLOW_ID"),
    }
    resolved = []
    for name in names:
        if name in constants:
            path, symbol = constants[name]
            text = (_REPO / path).read_text(encoding="utf-8")
            m = re.search(r'^%s\s*(?::[^=]+)?=\s*["\']([a-z_0-9]+)["\']' % symbol,
                          text, flags=re.M)
            resolved.append(m.group(1) if m else name)
        else:
            resolved.append(name)
    # The analytical layer registers through its own factory.
    route_src = (_REPO / "mi_workflows" / "analytical" / "route.py").read_text(encoding="utf-8")
    m = re.search(r'^ROUTE_NAME\s*=\s*["\']([a-z_0-9]+)["\']', route_src, flags=re.M)
    if m and m.group(1) not in resolved:
        resolved.insert(0, m.group(1))
    # Sub-route labels a handler emits that are not registered recognisers.
    for extra in ("evolution_funnel", "evolution_pipeline_stage"):
        if extra not in resolved:
            resolved.append(extra)
    return resolved


def allowlists(path: Path) -> Dict[str, List[str]]:
    """Module-level frozenset/dict constants keyed by route name."""
    out: Dict[str, List[str]] = {}
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        target = node.targets[0]
        name = getattr(target, "id", None)
        if not name or not name.isupper():
            continue
        members: List[str] = []
        for sub in ast.walk(node.value):
            if isinstance(sub, ast.Constant) and isinstance(sub.value, str):
                members.append(sub.value)
        # A single-valued constant (``ROUTE_NAME = "analytical_composition"``)
        # is a route DECLARING its own name, not an allowlist a consumer
        # consults. Only multi-member constants are governance allowlists.
        if len(members) > 1 and ("ROUTE" in name or "AXES" in name):
            out[name] = members
    return out



def consultation_re(path: Path) -> "re.Pattern":
    """Lines where a CONSUMER branches on route identity.

    Two forms, and only these two:

      * a route-keyed constant is CONSULTED — ``route in TEMPORAL_ROUTES``,
        ``_ROUTE_LABELS.get(route)``. The constant's own definition is not a
        decision site; consulting it is.
      * a route is compared DIRECTLY — ``route == "geo_exposure"``,
        ``route in ("evolution", ...)``.

    Declaration sites — ``route="evolution"`` on an envelope, ``name=`` in the
    recogniser registry — are excluded. They are the channel through which
    execution says what it was, and a compositional layer replaces WHAT
    declares rather than the fact of declaring.
    """
    constants = sorted(allowlists(path))
    parts = []
    if constants:
        parts.append(r"(?<![A-Za-z_])(?:%s)(?![A-Za-z_])" % "|".join(
            re.escape(c) for c in constants))
    parts.append(r'\broute\s*(?:==|!=|\bin\b)\s*[\("\']')
    parts.append(r'\broute\s*(?:==|!=)\s*[A-Za-z_]')
    return re.compile("|".join(parts))


def enclosing_function(tree: ast.Module, lineno: int) -> str:
    best, best_line = "<module>", -1
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.lineno <= lineno <= (node.end_lineno or 0):
            if node.lineno > best_line:
                best, best_line = node.name, node.lineno
    return best


def main() -> int:
    names = route_names()
    pattern = re.compile("|".join(r'["\']%s["\']' % re.escape(n) for n in names))

    print("=" * 78)
    print("ROUTE-IDENTITY GOVERNANCE INVENTORY")
    print("=" * 78)
    print(f"\n{len(names)} governed route names:\n  {', '.join(sorted(names))}")

    grand = 0
    per_file: List[Tuple[str, int]] = []
    print("\n--- PRODUCTION SITES ---")
    for rel in _PRODUCTION:
        path = _REPO / rel
        if not path.exists():
            continue
        text = path.read_text(encoding="utf-8")
        tree = ast.parse(text)
        consult = consultation_re(path)
        hits: Dict[str, List[int]] = {}
        for i, line in enumerate(text.splitlines(), start=1):
            stripped = line.strip()
            if stripped.startswith("#") or stripped.startswith("*"):
                continue
            if consult.search(line) and not re.match(
                    r"\s*[A-Za-z_]+\s*(?::[^=]+)?=\s*(?:frozenset|\{|\()", line):
                hits.setdefault(enclosing_function(tree, i), []).append(i)
        count = sum(len(v) for v in hits.values())
        grand += count
        per_file.append((rel, count))
        if not count:
            continue
        print(f"\n  {rel}   {count} route-name reference(s)")
        for fn, lines in sorted(hits.items(), key=lambda kv: -len(kv[1])):
            preview = ", ".join(str(x) for x in lines[:8])
            more = f" (+{len(lines) - 8})" if len(lines) > 8 else ""
            print(f"      {fn:38s} {len(lines):3d}   lines {preview}{more}")

    print("\n--- ROUTE ALLOWLISTS (module-level constants) ---")
    total_members = 0
    for rel in _PRODUCTION:
        path = _REPO / rel
        if not path.exists():
            continue
        for name, members in allowlists(path).items():
            total_members += len(members)
            print(f"  {rel}::{name}   {len(members)} member(s)")
            print(f"      {sorted(members)}")

    print("\n--- TOTALS ---")
    for rel, count in per_file:
        print(f"  {rel:44s} {count:4d}")
    print(f"  {'PRODUCTION DECISION SITES':44s} {grand:4d}")
    print("\n  (declaration sites — a route naming itself — are excluded: they are the\n   channel execution declares through, not a branch on identity.)")
    print(f"  {'allowlist members':44s} {total_members:4d}")
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
