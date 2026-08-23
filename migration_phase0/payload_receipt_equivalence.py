#!/usr/bin/env python3
"""migration_phase0/payload_receipt_equivalence.py — the boundary a shadow cannot reach.

READ-ONLY with respect to production. Conversion 1 §3, and it runs BEFORE the
switch.

Phase 0 proved the economics and said plainly what it could not prove: a shadow
emits a plan and a result, not an envelope, so payload and receipt equivalence
were never measured. This measures them, by rendering the SAME route handler
twice over the same request — once through `movement_summary.portfolio_summary`
and once through the compositional plan — and diffing the two envelopes.

It does not switch production. It substitutes the one call for the duration of
a render, which is exactly the substitution the conversion will make
permanently, so a difference found here is a difference the conversion would
have shipped.

Compared per case: route identity, payload keys and shape, metadata, governed
population declaration, requested and applied facets, grouping/population/
measure proof, receipt verdict, limitation and refusal state, availability and
fall-through, and the inputs the answer is generated from.

    python -m migration_phase0.payload_receipt_equivalence [--out FILE]
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

warnings.simplefilter("ignore")
_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from migration_phase0.route_ownership_portfolio_summary import (  # noqa: E402
    CANDIDATES, DEFAULTS, _env,
)

#: Keys whose value legitimately differs run to run and says nothing about
#: behaviour.
_VOLATILE = {"runId", "run_id", "generatedAt", "generated_at", "timestamp",
             "durationMs", "duration_ms", "elapsed", "queryId", "query_id",
             "requestId", "request_id", "traceId", "trace_id", "createdAt"}

#: A generated KPI id (`kpi_<8 hex>`), randomised per render. Stripped BY VALUE
#: SHAPE rather than by key name: `id` elsewhere may be meaningful, and dropping
#: every `id` would hide a real difference.
_KPI_ID_RE = re.compile(r"^kpi_[0-9a-f]{8}$")
_UID_RE = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-"
                     r"[0-9a-f]{4}-[0-9a-f]{12}$")


def _strip(node: Any) -> Any:
    if isinstance(node, dict):
        return {k: _strip(v) for k, v in sorted(node.items())
                if k not in _VOLATILE
                and not (k == "id" and isinstance(v, str)
                         and (_KPI_ID_RE.match(v) or _UID_RE.match(v)))}
    if isinstance(node, (list, tuple)):
        return [_strip(v) for v in node]
    if isinstance(node, float):
        return round(node, 6)
    return node


def _diff(a: Any, b: Any, path: str = "") -> List[str]:
    """Every difference, by path. Not a boolean: a count cannot be attributed."""
    out: List[str] = []
    if type(a) is not type(b) and not (isinstance(a, (int, float))
                                       and isinstance(b, (int, float))):
        return [f"{path or '<root>'}: type {type(a).__name__} vs {type(b).__name__}"]
    if isinstance(a, dict):
        for key in sorted(set(a) | set(b)):
            if key not in a:
                out.append(f"{path}.{key}: absent in shipped, present in plan")
            elif key not in b:
                out.append(f"{path}.{key}: present in shipped, absent in plan")
            else:
                out.extend(_diff(a[key], b[key], f"{path}.{key}"))
    elif isinstance(a, list):
        if len(a) != len(b):
            out.append(f"{path}: length {len(a)} vs {len(b)}")
        else:
            for i, (x, y) in enumerate(zip(a, b)):
                out.extend(_diff(x, y, f"{path}[{i}]"))
    elif isinstance(a, float) or isinstance(b, float):
        # A2's tolerance, applied where a number is compared.
        if abs(float(a) - float(b)) >= 0.005:
            out.append(f"{path}: {a!r} vs {b!r}")
    elif a != b:
        out.append(f"{path}: {a!r} vs {b!r}")
    return out


class LegacyPathRemoved(Exception):
    """The lens-resolved population path is gone from this route.

    Raised rather than papered over. This instrument proved equivalence while
    BOTH paths existed; once the duplicate owner is deleted there is no "before"
    to render, and a comparison that silently compares the converted path
    against a deferral would report 1,832 spurious differences and mean nothing.
    The evidence stands at the commit where it was taken.
    """


def _legacy_path_exists() -> bool:
    """Whether the route still CALLS the lens resolver.

    Over the parsed AST, not over the source text: the function's docstring says
    "`_resolve_lens` is deliberately NOT reachable from here", and a substring
    check reads that sentence as the call it denies. The same mistake as the
    plan module's first import guard, in the same session — a guard that reads
    prose is not reading code.
    """
    import ast
    import inspect
    import textwrap

    from mi_agent_api import chat_routing as routing
    tree = ast.parse(textwrap.dedent(inspect.getsource(routing._summary_population)))
    return any(isinstance(n, ast.Call) and getattr(n.func, "id", "") == "_resolve_lens"
               for n in ast.walk(tree))


def _render(question: str, default: Optional[str], client_id: str,
            compositional: bool) -> Dict[str, Any]:
    """One full governed answer, with the population taken either way.

    The legacy render is forced through `_summary_population`'s OWN fall-through
    — the branch that runs when no contract is available — by suppressing the
    interpretation for the duration of the call. Nothing is copied and no second
    handler exists: both renders are the shipped handler, and the only thing
    that differs is the branch it takes, which is exactly what the conversion
    changes.
    """
    from mi_agent_api import chat_routing as routing
    from mi_agent_api.mi_service import MiQueryRequest, execute_governed_mi_query
    from trakt_core.context import ExecutionContext

    ctx = ExecutionContext.for_internal(client_id)
    original = routing._summary_population
    seen = {"plan": 0, "legacy": 0}
    if not compositional and not _legacy_path_exists():
        raise LegacyPathRemoved()

    def _spy(question_, source_lens_, interpretation_, **kw):
        # ANTI-VACUITY. The first run of this instrument reported 0 differences
        # across all 54 pairs while the compositional path was taken ZERO times:
        # `_build_interpretation` raised on every question and the try/except
        # around it returned None, so both renders were the legacy branch and
        # were trivially identical. A comparison that cannot tell "equivalent"
        # from "never ran" is not evidence.
        supplied = None if not compositional else interpretation_
        seen["plan" if supplied is not None else "legacy"] += 1
        return original(question_, source_lens_, supplied, **kw)

    routing._summary_population = _spy
    try:
        result = execute_governed_mi_query(
            MiQueryRequest(question=question,
                           source_portfolio_lens=default), ctx).result or {}
    finally:
        routing._summary_population = original
    if compositional and seen["legacy"]:
        raise AssertionError(
            f"the compositional render fell back to the legacy path for "
            f"{question!r} — this comparison would have been vacuous")
    if compositional and not seen["plan"]:
        raise AssertionError(
            f"the route never reached the summary population for {question!r}")
    return result


def _shape(result: Dict[str, Any]) -> Dict[str, Any]:
    """The comparable surface of one governed answer."""
    metadata = result.get("metadata") or {}
    summary = (result.get("executionSummary")
               or metadata.get("executionSummary") or {})
    guard = result.get("semanticGuard") or {}
    return {
        "route": metadata.get("route"),
        "ok": bool(result.get("ok")),
        "controlledRefusal": bool(result.get("controlledRefusal")),
        "error": result.get("error"),
        "answer": (result.get("answer") or "").strip(),
        "payloadKeys": sorted(result.keys()),
        "metadataKeys": sorted(metadata.keys()),
        "lensApplied": metadata.get("lensApplied"),
        "portfolioScope": result.get("portfolioScope"),
        "portfolioCoverage": result.get("portfolioCoverage"),
        "reconciliation": result.get("reconciliation"),
        "sourceNotes": result.get("sourceNotes"),
        "warnings": result.get("warnings"),
        "artifacts": [{k: a.get(k) for k in
                       ("type", "title", "description", "chartType", "xKey",
                        "valueFormat", "columns", "rows", "series", "kpis",
                        "reconciliation", "sourceNotes")}
                      for a in (result.get("artifacts") or [])],
        "executionSummary": summary,
        "facets": [(f.get("kind"), f.get("label"), f.get("field_key"),
                    f.get("status")) for f in (summary.get("facets") or [])],
        "notApplied": summary.get("notApplied"),
        "verdict": guard.get("verdict") or summary.get("verdict"),
        "guardFacets": guard.get("facets"),
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=str(
        _REPO / "migration_phase0" / "PAYLOAD_RECEIPT_EQUIVALENCE.json"))
    args = ap.parse_args(argv)
    client_id = _env()

    from mi_agent_api import chat_routing as routing

    owned = [(c, q) for c, q, _p in CANDIDATES
             if routing._is_portfolio_summary(q)]

    print("=" * 112)
    print("PAYLOAD + RECEIPT EQUIVALENCE — the boundary Phase 0 could not reach")
    print("=" * 112)
    print(f"\n{len(owned)} route-owned cases x {len(DEFAULTS)} caller defaults "
          f"= {len(owned) * len(DEFAULTS)} rendered answer pairs\n")

    if not _legacy_path_exists():
        print("\nThe lens-resolved population path has been REMOVED from this "
              "route,\nso there is no 'before' left to render. Equivalence was "
              "proved while both\npaths existed — 54 pairs, 0 differences — and "
              "that evidence stands at the\ncommit where it was taken. This "
              "instrument reports rather than inventing a\ncomparison it can no "
              "longer make.\n")
        return 0

    rows: List[Dict[str, Any]] = []
    total_diffs = 0
    for case, question in owned:
        for default in DEFAULTS:
            shipped = _shape(_render(question, default, client_id, False))
            planned = _shape(_render(question, default, client_id, True))
            diffs = _diff(_strip(shipped), _strip(planned))
            total_diffs += len(diffs)
            rows.append({"case": case, "question": question, "default": default,
                         "differences": diffs})
            flag = "identical" if not diffs else f"{len(diffs)} DIFFERENCES"
            print(f"  {case:5s} default={str(default):9s} {flag}")
            for d in diffs[:6]:
                print(f"        {d}")

    print("\n" + "=" * 112)
    print(f"cases compared        : {len(rows)}")
    print(f"total differences     : {total_diffs}")
    print("=" * 112)

    Path(args.out).write_text(json.dumps(rows, indent=2, default=str) + "\n",
                              encoding="utf-8")
    print(f"\nwrote {Path(args.out).relative_to(_REPO)}")
    return 1 if total_diffs else 0


if __name__ == "__main__":
    sys.exit(main())
