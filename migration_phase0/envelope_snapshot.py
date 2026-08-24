#!/usr/bin/env python3
"""migration_phase0/envelope_snapshot.py — freeze rendered envelopes for a route.

READ-ONLY. Payload/receipt equivalence, measured as a genuine BEFORE/AFTER.

Conversion 1 measured this by keeping both code paths alive and swapping between
them in-process. That worked, but it could only be run while the duplicate owner
still existed — and it reported a vacuous pass the first time, because both
renders silently took the same branch.

This takes the other approach: snapshot the rendered envelopes, run the same
snapshot at the pre-switch commit in a worktree, and diff the two files. There
is no branch to take wrongly, the "before" is a real commit rather than a
simulated one, and the comparison survives the duplicate owner being deleted.

    python -m migration_phase0.envelope_snapshot --route period_movement --out FILE
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
from typing import Any, Dict, List, Optional, Sequence

warnings.simplefilter("ignore")
_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

_VOLATILE = {"runId", "run_id", "generatedAt", "generated_at", "timestamp",
             "durationMs", "duration_ms", "elapsed", "queryId", "query_id",
             "requestId", "request_id", "traceId", "trace_id", "createdAt"}
_KPI_ID_RE = re.compile(r"^kpi_[0-9a-f]{8}$")
_UID_RE = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-"
                     r"[0-9a-f]{4}-[0-9a-f]{12}$")


def _strip(node: Any) -> Any:
    """Drop what legitimately differs run to run. By VALUE SHAPE for ids, not by
    key name: `id` elsewhere may be meaningful."""
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


def _cases(route: str):
    if route == "funded_bridge":
        # CONVERSION 4. The owned set comes from the ownership instrument's
        # declared surface, filtered by the SHIPPED recogniser, so a case the
        # route stops claiming drops out of the denominator loudly.
        from migration_phase0.route_ownership_funded_bridge import CASES, SCOPES
        from mi_agent.llm_query_parser import parse_with_repair
        from mi_agent.mi_query_validator import load_mi_semantics
        from mi_agent_api.data_source import semantics_path
        sem = load_mi_semantics(semantics_path())

        def _claims(q):
            spec, _m = parse_with_repair(q, sem, llm_enabled=False)
            return bool(getattr(spec, "bridge_query", False))
        return ([(c, q) for c, q, other in CASES
                 if other is None and _claims(q)], SCOPES)
    if route == "geo_exposure":
        # CONVERSION 3. The owned set comes from the ownership instrument's
        # declared surface, filtered by the SHIPPED recogniser — so a case the
        # route stops claiming drops out of the denominator loudly (the count
        # changes) rather than silently comparing two refusals.
        from migration_phase0.route_ownership_geo_exposure import CASES, SCOPES
        from mi_agent_api import chat_routing as routing
        return ([(c, q) for c, q, expected_other in CASES
                 if expected_other is None
                 and routing._is_geo_exposure(q)], SCOPES)
    if route == "period_movement":
        from migration_phase0.route_ownership_period_movement import (
            CANDIDATES, DEFAULTS)
        from mi_agent_api import chat_routing as routing
        return ([(c, q) for c, q, _p in CANDIDATES
                 if routing._is_period_movement(q)], DEFAULTS)
    from migration_phase0.route_ownership_portfolio_summary import (
        CANDIDATES, DEFAULTS)
    from mi_agent_api import chat_routing as routing
    return ([(c, q) for c, q, _p in CANDIDATES
             if routing._is_portfolio_summary(q)], DEFAULTS)


def _leaves(node: Any, path: str = ""):
    """Every scalar in a nested structure, with its full path.

    List LENGTH is yielded as its own leaf, so a truncated list is a difference
    rather than a shorter loop that compares fewer things and still says zero.
    """
    if isinstance(node, dict):
        for k, v in node.items():
            yield from _leaves(v, f"{path}.{k}")
    elif isinstance(node, (list, tuple)):
        yield f"{path}[len]", len(node)
        for i, v in enumerate(node):
            yield from _leaves(v, f"{path}[{i}]")
    else:
        yield path, node


def _diff(before_path: str, after_path: str, expect: Optional[int]) -> int:
    """Diff two snapshots, and PROVE the denominator before reporting zero.

    Conversion 2's first attempt at this keyed rows on a field that did not
    exist, silently collapsed 36 entries to 12, and reported "36 pairs, 0
    differences". Every assertion below exists because of that: the key is
    composite and checked for collisions, both sides must carry the same key
    set, the pair count is asserted against what the caller expected, and the
    number of leaf fields actually compared is printed rather than assumed.
    """
    before = json.loads(Path(before_path).read_text(encoding="utf-8"))
    after = json.loads(Path(after_path).read_text(encoding="utf-8"))

    def keyed(rows: List[Dict[str, Any]], side: str) -> Dict[Any, Dict[str, Any]]:
        out: Dict[Any, Dict[str, Any]] = {}
        for r in rows:
            k = (r["case"], str(r.get("default")))
            if k in out:
                raise SystemExit(f"DENOMINATOR UNSOUND: duplicate key {k} on "
                                 f"the {side} side — the key does not identify "
                                 f"a render, so a diff over it would compare "
                                 f"fewer pairs than it reports.")
            out[k] = r
        if len(out) != len(rows):
            raise SystemExit(f"DENOMINATOR UNSOUND: {len(rows)} rows collapsed "
                             f"to {len(out)} keys on the {side} side.")
        return out

    kb, ka = keyed(before, "before"), keyed(after, "after")
    if set(kb) != set(ka):
        only_b, only_a = sorted(set(kb) - set(ka)), sorted(set(ka) - set(kb))
        raise SystemExit(f"DENOMINATOR UNSOUND: the two sides do not cover the "
                         f"same renders.\n  only before: {only_b}\n"
                         f"  only after : {only_a}")
    if expect is not None and len(kb) != expect:
        raise SystemExit(f"DENOMINATOR UNSOUND: expected {expect} pairs, the "
                         f"snapshots carry {len(kb)}.")

    differences: List[str] = []
    compared = 0
    populated: Dict[str, int] = {}
    for k in sorted(kb):
        fb = dict(_leaves(kb[k]["envelope"]))
        fa = dict(_leaves(ka[k]["envelope"]))
        fields = sorted(set(fb) | set(fa))
        compared += len(fields)
        for f in fields:
            if fb.get(f) != fa.get(f):
                differences.append(f"{k} {f}: {fb.get(f)!r} -> {fa.get(f)!r}")
        for top in ("executionSummary", "facets", "reconciliation", "verdict",
                    "payloadKeys", "metadataKeys", "portfolioScope", "artifacts",
                    "answer", "route"):
            if kb[k]["envelope"].get(top) not in (None, [], {}, ""):
                populated[top] = populated.get(top, 0) + 1

    print("=" * 78)
    print("ENVELOPE EQUIVALENCE — with the denominator proved, not assumed")
    print("=" * 78)
    print(f"pairs compared            : {len(kb)}"
          + (f"  (expected {expect} — OK)" if expect else ""))
    print(f"envelope leaf fields      : {compared}")
    print(f"DIFFERENCES               : {len(differences)}")
    print("\nfields actually carrying content on the BEFORE side "
          "(a zero over empty structures proves nothing):")
    for top in sorted(populated):
        print(f"    {top:<20} non-empty in {populated[top]}/{len(kb)} pairs")
    if differences:
        print("\nevery difference:")
        for d in differences:
            print(f"  {d}")
    print("=" * 78)
    return 1 if differences else 0


def _env() -> str:
    os.environ.setdefault("TRAKT_RUNTIME_MODE", "development")
    from demo_platform import config as cfg
    os.environ.update(cfg.mi_env(period_role="current"))
    os.environ["MI_AGENT_LLM_PARSER"] = "off"
    os.environ["MI_AGENT_LLM_ENABLED"] = "0"
    logging.disable(logging.WARNING)
    return cfg.CLIENT_ID


def _shape(result: Dict[str, Any]) -> Dict[str, Any]:
    """The comparable surface of one governed answer — everything §6 lists."""
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
    ap.add_argument("--route", default="period_movement")
    ap.add_argument("--out")
    ap.add_argument("--diff", nargs=2, metavar=("BEFORE", "AFTER"),
                    help="diff two snapshots instead of taking one")
    ap.add_argument("--expect", type=int,
                    help="assert this many pairs; a diff that cannot prove its "
                         "denominator is refused rather than reported as clean")
    args = ap.parse_args(argv)
    if args.diff:
        return _diff(args.diff[0], args.diff[1], args.expect)
    if not args.out:
        ap.error("--out is required unless --diff is given")
    client_id = _env()

    from mi_agent_api.mi_service import MiQueryRequest, execute_governed_mi_query
    from trakt_core.context import ExecutionContext

    ctx = ExecutionContext.for_internal(client_id)
    owned, defaults = _cases(args.route)
    rows: List[Dict[str, Any]] = []
    for case, question in owned:
        for default in defaults:
            result = execute_governed_mi_query(
                MiQueryRequest(question=question,
                               source_portfolio_lens=default), ctx).result or {}
            rows.append({"case": case, "question": question,
                         "default": default,
                         "envelope": _strip(_shape(result))})
    if args.expect is not None and len(rows) != args.expect:
        raise SystemExit(f"SURFACE CHANGED: expected {args.expect} renders, "
                         f"took {len(rows)}. A snapshot whose denominator moved "
                         f"cannot be diffed against one taken earlier.")
    Path(args.out).write_text(json.dumps(rows, indent=2, default=str) + "\n",
                              encoding="utf-8")
    print(f"{len(rows)} envelopes ({len(owned)} cases x {len(defaults)} scopes) "
          f"-> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
