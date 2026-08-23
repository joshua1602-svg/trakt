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
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
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
    Path(args.out).write_text(json.dumps(rows, indent=2, default=str) + "\n",
                              encoding="utf-8")
    print(f"{len(rows)} envelopes -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
