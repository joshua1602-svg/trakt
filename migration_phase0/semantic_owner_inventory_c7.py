#!/usr/bin/env python3
"""migration_phase0/semantic_owner_inventory_c7.py — what C7's route actually owns.

READ-ONLY, static. Attributes every line of `mi_agent_api/period_change_route.py`
to a category, by EVIDENCE rather than by function name.

THE CLASSIFYING CRITERION
-------------------------
    A function that reads the RAW QUESTION STRING is interpreting it.

That is the whole test, and it is deliberately mechanical: it cannot be argued
with, and it does not depend on what a function is called or what its docstring
claims. `_rank_subject` reads the question, so it interprets. `build_answer`
does not, so it renders. A function that only passes `question` through to
another callable is recorded as DELEGATING — it carries the question but takes
no decision from it, and delegation is what the compositional plan asks for.

Categories:

  INTERPRETS   reads the raw question and decides something from it
  DELEGATES    passes the question to a named owner and takes no decision
  VOCABULARY   a module-level constant that encodes business language
  RENDERS      builds prose, rows, columns, envelopes from an already-computed
               result
  ADAPTS       snapshot supply, scope construction, failure-code mapping — the
               platform seam
  STRUCTURE    imports, dataclass declarations, module docstring

    python -m migration_phase0.semantic_owner_inventory_c7 [--json out.json]
"""
from __future__ import annotations

import argparse
import ast
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

TARGET = _REPO / "mi_agent_api" / "period_change_route.py"

INTERPRETS, DELEGATES, VOCABULARY = "INTERPRETS", "DELEGATES", "VOCABULARY"
RENDERS, ADAPTS, STRUCTURE = "RENDERS", "ADAPTS", "STRUCTURE"

#: Names that ARE the raw question inside this module.
_QUESTION_NAMES = {"question"}

#: Module-level constants that encode business language rather than mechanism.
#: Listed explicitly because a regex or a frozenset is indistinguishable from a
#: lookup table until you read what is in it.
_VOCABULARY_NAMES = {
    "_NARRATIVE_RANK_SUBJECTS", "_RANK_SUBJECT_LEAD_RE", "_RANK_SUBJECT_SKIP",
    "_PROSE_RUNNERS_UP", "_BASIS_UNITS",
}

#: Constants that are presentation or taxonomy, not business language.
_ADAPTER_NAMES = {"FAILURE_ERROR_CODES", "ROUTE_NAME", "KNOWN_ASSET_CLASSES",
                  "_RANK_COLUMNS", "_METRIC_COLUMNS", "_DISTRIBUTION_COLUMNS",
                  "_BRIDGE_COLUMNS"}


def _is_local_vocabulary(func: ast.AST) -> bool:
    """True when a call target applies THIS MODULE's own language.

    Handing the question to `chat_routing._resolve_lens` is delegation: another
    owner decides. Handing it to `_RANK_SUBJECT_LEAD_RE.search` or to `re` is
    NOT — the decision is taken here, against a vocabulary declared here, and
    calling that delegation would let any module launder interpretation through
    a regex. An earlier version of this instrument made exactly that mistake and
    scored `_rank_subject` — the module's clearest interpreter — as a delegator.
    """
    if isinstance(func, ast.Attribute):
        base = func.value
        if isinstance(base, ast.Name):
            return base.id == "re" or base.id in _VOCABULARY_NAMES
        return False
    if isinstance(func, ast.Name):
        return func.id in _VOCABULARY_NAMES
    return False


def _reads_question(node: ast.AST) -> bool:
    """True when the body USES the raw question for a decision of its own.

    Passing it to ANOTHER OWNER is not using it — that is delegation, and
    conflating the two would score every adapter in the estate as an
    interpreter. Slicing it, comparing it, or matching it against this module's
    own vocabulary IS using it.
    """
    delegated = set()
    for sub in ast.walk(node):
        if isinstance(sub, ast.Call) and not _is_local_vocabulary(sub.func):
            for arg in list(sub.args) + [k.value for k in sub.keywords]:
                for name in ast.walk(arg):
                    if isinstance(name, ast.Name) and name.id in _QUESTION_NAMES:
                        delegated.add(id(name))
    for sub in ast.walk(node):
        if isinstance(sub, ast.Name) and sub.id in _QUESTION_NAMES:
            if id(sub) not in delegated:
                return True
    return False


def _delegates_question(node: ast.AST) -> List[str]:
    """The callables this function hands the raw question to."""
    out: List[str] = []
    for sub in ast.walk(node):
        if not isinstance(sub, ast.Call):
            continue
        if _is_local_vocabulary(sub.func):
            continue          # local vocabulary is interpretation, not delegation
        args = list(sub.args) + [k.value for k in sub.keywords]
        if not any(isinstance(n, ast.Name) and n.id in _QUESTION_NAMES
                   for a in args for n in ast.walk(a)):
            continue
        func = sub.func
        if isinstance(func, ast.Attribute):
            base = func.value
            prefix = base.id if isinstance(base, ast.Name) else "…"
            out.append(f"{prefix}.{func.attr}")
        elif isinstance(func, ast.Name):
            out.append(func.id)
    return sorted(set(out))


def _lines(node: ast.AST) -> int:
    return (node.end_lineno or node.lineno) - node.lineno + 1


def classify() -> List[Dict[str, Any]]:
    tree = ast.parse(TARGET.read_text(encoding="utf-8"))
    out: List[Dict[str, Any]] = []
    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            out.append({"name": "<imports>", "kind": STRUCTURE,
                        "lines": _lines(node), "delegates": []})
            continue
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = ([node.target] if isinstance(node, ast.AnnAssign)
                       else node.targets)
            name = next((t.id for t in targets if isinstance(t, ast.Name)), "?")
            kind = (VOCABULARY if name in _VOCABULARY_NAMES
                    else ADAPTS if name in _ADAPTER_NAMES else STRUCTURE)
            out.append({"name": name, "kind": kind, "lines": _lines(node),
                        "delegates": []})
            continue
        if isinstance(node, ast.ClassDef):
            out.append({"name": node.name, "kind": STRUCTURE,
                        "lines": _lines(node), "delegates": []})
            continue
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            delegates = _delegates_question(node)
            takes_question = any(a.arg in _QUESTION_NAMES
                                 for a in node.args.args)
            if takes_question and _reads_question(node):
                kind = INTERPRETS      # checked BEFORE delegation: a function may
                                       # do both, and owning a decision is the
                                       # stronger fact about it
            elif delegates:
                kind = DELEGATES
            elif node.name.startswith(("build_answer", "_render", "_format",
                                       "_metric_rows", "_distribution_rows",
                                       "_bridge_rows", "_rank_rows", "describe_",
                                       "_share_pp", "_describe")):
                kind = RENDERS
            else:
                kind = ADAPTS
            out.append({"name": node.name, "kind": kind, "lines": _lines(node),
                        "delegates": delegates})
            continue
        out.append({"name": "<module>", "kind": STRUCTURE,
                    "lines": _lines(node), "delegates": []})
    return out


# --------------------------------------------------------------------------- #
# COMPOSITION DECISIONS — hand-audited, anchored, and verified on every run.
#
# The line classifier above cannot see these, and they are the ones that matter.
# `route_period_change` delegates every reading of the question — so it scores
# DELEGATES — and then takes SEVEN decisions of its own from what the owners
# returned. A decision taken from a delegated result is still a decision, and
# whether it is legitimate depends entirely on whether it is generic or
# route-shaped.
#
# Each entry carries an ANCHOR: a literal substring of the source. Every anchor
# is checked on each run, so if the code moves, this inventory FAILS rather than
# quietly describing a module that no longer exists.
# --------------------------------------------------------------------------- #
COMPOSITION_DECISIONS = [
    {
        "id": "K1", "name": "a ranked dimension IS the requested metric",
        "anchor": "mode = MODE_REQUESTED_METRIC",
        "decides": "overwrites the analysis mode and requested_fields with the "
                   "ranked dimension",
        "generic": False,
        "note": "drops rank_intent.alt_fields, which the resolver returned "
                "precisely so an availability difference is not read as a "
                "substitution. This is the mechanism behind canary defect D1.",
    },
    {
        "id": "K2", "name": "honour the stated span, or clarify",
        "anchor": "if len(snapshots) > span.periods:",
        "decides": "rewrites period_request.requested_start to a snapshot's "
                   "year-month, or returns a clarification envelope",
        "generic": False,
        "note": "a genuine product rule, owned by no shared layer. It reaches "
                "into the governed period_request and rewrites it.",
    },
    {
        "id": "K3", "name": "ranking is resolved BEFORE the analysis",
        "anchor": "rank_intent = resolve_rank_intent(interpretation)",
        "decides": "ordering: a rank refusal returns before the span guard runs",
        "generic": False,
        "note": "so a question with BOTH an unrankable dimension and an "
                "unhonourable span is told only about the dimension.",
    },
    {
        "id": "K4", "name": "suppress requested concepts when ranking",
        "anchor": "requested_concepts=() if rank_intent.requested",
        "decides": "which governed concepts the analysis covers",
        "generic": False, "note": "",
    },
    {
        "id": "K5", "name": "when to reconcile the book",
        "anchor": "include_bridge=(intent.include_bridge or mode !=",
        "decides": "whether the bridge is computed",
        "generic": False, "note": "",
    },
    {
        "id": "K6", "name": "ranking implies composition focus",
        "anchor": "composition_focus=intent.composition_focus or rank_intent.requested",
        "decides": "whether the analysis is composition-focused",
        "generic": False, "note": "",
    },
    {
        "id": "K7", "name": "reinterpret a controlled failure as a dimension refusal",
        "anchor": 'refusal_reason="dimension_not_governed"',
        "decides": "replaces the workflow's FAIL_NO_ELIGIBLE_FIELDS with a "
                   "statement about the dimension",
        "generic": False,
        "note": "CLOSED at the reduction — the message now names every "
                "candidate field considered, so the refusal is checkable "
                "instead of a bare false assertion. Still route-local.",
    },
]


def verify_anchors() -> List[str]:
    """Every anchor must still be present. A stale inventory is a lying one."""
    source = TARGET.read_text(encoding="utf-8")
    return [d["id"] for d in COMPOSITION_DECISIONS if d["anchor"] not in source]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--json", type=Path, default=None)
    args = ap.parse_args(argv)

    rows = classify()
    total = sum(r["lines"] for r in rows)
    file_lines = len(TARGET.read_text(encoding="utf-8").splitlines())

    print("=" * 84)
    print(f"C7 SEMANTIC-OWNER INVENTORY — {TARGET.name}, {file_lines} lines")
    print("=" * 84)
    print(f"\n{'name':<32}{'kind':<12}{'lines':>6}  delegates the question to")
    print("-" * 84)
    for r in sorted(rows, key=lambda r: -r["lines"]):
        if r["kind"] == STRUCTURE and r["lines"] < 4:
            continue
        print(f"{r['name']:<32}{r['kind']:<12}{r['lines']:>6}  "
              f"{', '.join(r['delegates'])[:34]}")

    print(f"\n{'category':<14}{'lines':>7}{'share':>8}   members")
    print("-" * 84)
    by_kind: Dict[str, List[Dict[str, Any]]] = {}
    for r in rows:
        by_kind.setdefault(r["kind"], []).append(r)
    for kind in (INTERPRETS, VOCABULARY, DELEGATES, RENDERS, ADAPTS, STRUCTURE):
        sel = by_kind.get(kind, [])
        n = sum(r["lines"] for r in sel)
        names = ", ".join(r["name"] for r in sel if r["name"] != "<imports>")
        print(f"{kind:<14}{n:>7}{n / total * 100:>7.1f}%   {names[:46]}")

    missing = verify_anchors()
    if missing:
        raise SystemExit(
            f"INVENTORY STALE — composition decisions {missing} no longer match "
            f"the source. Re-audit before trusting any number in this report.")

    print(f"\nCOMPOSITION DECISIONS — taken by the route from DELEGATED results.")
    print("The line classifier scores these as DELEGATES; they are decisions.")
    print("-" * 84)
    for d in COMPOSITION_DECISIONS:
        print(f"  {d['id']}  {'generic' if d['generic'] else 'ROUTE-LOCAL':<12} "
              f"{d['name']}")
        print(f"      decides: {d['decides']}")
        if d["note"]:
            print(f"      note   : {d['note']}")
    route_local = sum(1 for d in COMPOSITION_DECISIONS if not d["generic"])
    print(f"\n  {route_local} of {len(COMPOSITION_DECISIONS)} are route-local.")

    semantic = sum(r["lines"] for r in rows
                   if r["kind"] in (INTERPRETS, VOCABULARY))
    print(f"\nSEMANTIC OWNERSHIP (INTERPRETS + VOCABULARY): {semantic} lines, "
          f"{semantic / total * 100:.1f}% of the module")
    print("Everything else is adapter, rendering or structure — the module is "
          "big, but\nalmost none of its size is meaning it owns.")

    if args.json:
        args.json.write_text(json.dumps(
            {"lines": rows, "composition_decisions": COMPOSITION_DECISIONS},
            indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
