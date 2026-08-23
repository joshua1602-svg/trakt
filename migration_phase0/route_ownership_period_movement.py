#!/usr/bin/env python3
"""migration_phase0/route_ownership_period_movement.py — Conversion 2's surface.

READ-ONLY. Conversion 2 §3 and §4.

Ownership is asked of the shipped recogniser, never assumed from a list — the
rule Phase 0 earned when two questions that read like portfolio summaries turned
out not to be claimed by that route. The bank therefore includes questions
expected NOT to be owned; a bank of only owned questions cannot detect drift.

`_is_period_movement` is deliberately narrow: an explicit CHANGE marker AND an
explicit PRIOR-PERIOD marker. A named two-period comparison or a single-metric
trend belongs to `temporal_compare` / `evolution` and must stay there.

It also records, per case, the two semantic facts this route re-decides
downstream today — the source scope and the time window — so §5 can show both
now come from the contract.

    python -m migration_phase0.route_ownership_period_movement [--out FILE]
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

#: (case, question, provenance)
CANDIDATES: Tuple[Tuple[str, str, str], ...] = (
    # Plain month-on-month, no source scope named.
    ("M1", "What has changed since last month?", "movement+prior"),
    ("M2", "What changed versus the prior month?", "movement+prior"),
    ("M3", "How has the portfolio changed since the previous month?", "movement+prior"),
    ("M4", "What has moved month on month?", "movement+prior"),
    ("M5", "How has the book changed since last period?", "movement+prior"),
    # With a governed source scope.
    ("S1", "What has changed since last month in the acquired book?", "scope"),
    ("S2", "What has changed since last month in the direct book?", "scope"),
    ("S3", "What changed versus the prior month for the ALP Origination Book?",
     "named portfolio"),
    ("S4", "What has changed since last month in the funded book?", "funded"),
    # With a stated window wider than one period — the span half of the route.
    ("W1", "What has changed since last month over the last 3 months", "span"),
    ("W2", "What has changed month on month this year?", "span"),
    # Unresolvable scope — must refuse, never widen.
    ("U1", "What has changed since last month in the acquired_001 book",
     "unresolvable"),
    # Expected NOT owned — kept so ownership drift is detectable.
    ("X1", "Summarise the portfolio", "no change marker"),
    ("X2", "How has the funded balance evolved over time?", "evolution"),
    ("X3", "Compare June 2026 with May 2026", "temporal_compare"),
    ("X4", "What has changed?", "no prior-period marker"),
    ("X5", "Show the balance by region", "point-in-time"),
)

DEFAULTS: Tuple[Optional[str], ...] = (None, "acquired", "direct")
_MOVE_RE = re.compile(r"(?:rose|fell|grew|declined|increased|decreased|moved|"
                      r"changed)\s+by\s+(\S+)", re.I)


def _env() -> str:
    os.environ.setdefault("TRAKT_RUNTIME_MODE", "development")
    from demo_platform import config as cfg
    os.environ.update(cfg.mi_env(period_role="current"))
    os.environ["MI_AGENT_LLM_PARSER"] = "off"
    os.environ["MI_AGENT_LLM_ENABLED"] = "0"
    logging.disable(logging.WARNING)
    return cfg.CLIENT_ID


def capture(client_id: str) -> Dict[str, Any]:
    from mi_agent import period_request as period_mod
    from mi_agent.mi_query_validator import load_mi_semantics
    from mi_agent_api import chat_routing as routing
    from mi_agent_api import portfolio_context as ctx_mod
    from mi_agent_api.datasets import semantics_path
    from mi_agent_api.mi_service import MiQueryRequest, execute_governed_mi_query
    from question_interpretation import projection
    from trakt_core.context import ExecutionContext

    semantics = load_mi_semantics(semantics_path())
    registry = ctx_mod.build_registry()
    ctx = ExecutionContext.for_internal(client_id)

    rows: List[Dict[str, Any]] = []
    for case, question, provenance in CANDIDATES:
        owned = bool(routing._is_period_movement(question))
        row: Dict[str, Any] = {"case": case, "question": question,
                               "provenance": provenance,
                               "recogniserClaims": owned, "byDefault": {}}
        for default in DEFAULTS:
            qi = projection.project(question, semantics=semantics,
                                    registry=registry, caller_scope=default)
            span = period_mod.requested_span(question)
            result = execute_governed_mi_query(
                MiQueryRequest(question=question,
                               source_portfolio_lens=default), ctx).result or {}
            metadata = result.get("metadata") or {}
            summary = (result.get("executionSummary")
                       or metadata.get("executionSummary") or {})
            answer = (result.get("answer") or "").strip().replace("\n", " ")
            move = _MOVE_RE.search(answer)
            row["byDefault"][str(default)] = {
                # THE CONTRACT — what a plan would have to work from.
                "scope": qi.source_scope.scope,
                "scopeState": qi.source_scope.state,
                "portfolioIds": list(qi.source_scope.portfolio_ids),
                "provenance": qi.source_scope.provenance,
                "windowPeriods": qi.time.window_periods,
                "windowGoverned": qi.time.window_governed,
                # THE OWNER the route reads today, for comparison.
                "ownerSpanPeriods": getattr(span, "periods", None),
                # PRODUCTION.
                "route": metadata.get("route"),
                "ok": bool(result.get("ok")),
                "controlledRefusal": bool(result.get("controlledRefusal")),
                "verdict": (result.get("semanticGuard") or {}).get("verdict"),
                "facets": [(f.get("kind"), f.get("label"), f.get("status"))
                           for f in (summary.get("facets") or [])],
                "movement": move.group(1) if move else None,
                "answer": answer[:240],
            }
        rows.append(row)
    return {"cases": rows}


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=str(
        _REPO / "migration_phase0" / "ROUTE_OWNERSHIP_PERIOD_MOVEMENT.json"))
    args = ap.parse_args(argv)

    data = capture(_env())
    rows = data["cases"]

    print("=" * 118)
    print("period_movement — ROUTE OWNERSHIP, verified against the recogniser")
    print("=" * 118)
    print(f"\n{'case':5s} {'claims':7s} {'route(None)':20s} {'scope':22s} "
          f"{'win':5s} {'movement':14s} question")
    print("-" * 118)
    owned_n = 0
    for row in rows:
        cell = row["byDefault"]["None"]
        owned_n += 1 if row["recogniserClaims"] else 0
        scope = f"{cell['scopeState']}/{cell['scope']}"
        if cell["portfolioIds"]:
            scope += "=" + ",".join(cell["portfolioIds"])
        print(f"{row['case']:5s} {str(row['recogniserClaims']):7s} "
              f"{str(cell['route']):20s} {scope[:22]:22s} "
              f"{str(cell['windowPeriods']):5s} "
              f"{str(cell['movement'])[:14]:14s} {row['question'][:38]}")
    print("-" * 118)
    print(f"{owned_n} of {len(rows)} candidates are claimed by "
          f"`_is_period_movement`.")

    # The contract must carry the window the OWNER reads, or the plan cannot be
    # built from it — S8.
    mismatched = [(r["case"], d, c["windowPeriods"], c["ownerSpanPeriods"])
                  for r in rows if r["recogniserClaims"]
                  for d, c in r["byDefault"].items()
                  if c["windowPeriods"] != c["ownerSpanPeriods"]]
    print(f"\nwindow: contract vs owner — {len(mismatched)} mismatch(es)")
    for case, default, contract, owner in mismatched:
        print(f"   {case} default={default}: contract={contract} owner={owner}")

    Path(args.out).write_text(json.dumps(data, indent=2, default=str) + "\n",
                              encoding="utf-8")
    print(f"\nwrote {Path(args.out).relative_to(_REPO)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
