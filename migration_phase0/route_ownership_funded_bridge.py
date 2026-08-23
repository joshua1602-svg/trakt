#!/usr/bin/env python3
"""migration_phase0/route_ownership_funded_bridge.py — what `funded_bridge` owns.

READ-ONLY. Enumerates candidate questions x caller scopes and records, from
EXECUTED routing, which the shipped route claims — and, for each, whether it
DELIVERS or REFUSES, and what the handler itself computed before the guard saw
it.

That last column is the point. Conversion 4 stopped on what this instrument
measures: the route's stated-dimension surface refuses, and the handler behind
the refusal returns a figure that is wrong for one dimension. The two defects
cancel, so neither is visible from the outside.

    python -m migration_phase0.route_ownership_funded_bridge
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

#: (case, question, expected_other_owner_or_None)
CASES: Tuple[Tuple[str, str, Any], ...] = (
    # --- no dimension named: the route defaults, and DELIVERS ------------
    ("B1", "funded balance bridge", None),
    ("B2", "Show me the funded balance bridge", None),
    ("B3", "What drove the change in funded balance?", None),
    ("B4", "waterfall of funded balance change", None),
    # --- a dimension NAMED: the route's reason for existing ---------------
    ("D1", "Funded balance bridge by region", None),
    ("D2", "Bridge the funded balance by product", None),
    ("D3", "balance bridge by LTV band", None),
    ("D4", "Bridge the funded balance by region since March 2026", None),
    # --- scoped -----------------------------------------------------------
    ("S1", "Funded balance bridge for the acquired book", None),
    ("S2", "Funded balance bridge for the direct book", None),
    ("S3", "Funded balance bridge for the ALP Origination Book", None),
    ("R1", "Funded balance bridge for the Highgate Mortgages Book", None),
    # --- NOT claimed, and named ------------------------------------------
    ("X1", "Show the balance by region", "an ordinary stratification"),
    ("X2", "What is the geographic exposure?", "geo_exposure"),
    ("X3", "Summarise the portfolio", "portfolio_summary"),
)

SCOPES: Tuple[Any, ...] = (None, "direct", "acquired")


def _env() -> str:
    import logging
    import warnings
    warnings.simplefilter("ignore")
    os.environ.setdefault("TRAKT_RUNTIME_MODE", "development")
    from demo_platform import config as cfg
    os.environ.update(cfg.mi_env(period_role="current"))
    os.environ["MI_AGENT_LLM_PARSER"] = "off"
    os.environ["MI_AGENT_LLM_ENABLED"] = "0"
    logging.disable(logging.WARNING)
    return cfg.CLIENT_ID


def main() -> int:
    client_id = _env()
    from mi_agent.llm_query_parser import parse_with_repair
    from mi_agent.mi_query_validator import load_mi_semantics
    from mi_agent_api import chat_routing as routing
    from mi_agent_api.data_source import semantics_path
    from mi_agent_api.mi_service import MiQueryRequest, execute_governed_mi_query
    from trakt_core.context import ExecutionContext

    semantics = load_mi_semantics(semantics_path())
    root = os.environ["MI_AGENT_ONBOARDING_OUTPUT_ROOT"]
    ctx = ExecutionContext.for_internal(client_id)

    print("=" * 78)
    print("funded_bridge — ROUTE OWNERSHIP, and what the refusal is hiding")
    print("=" * 78)

    records: List[Dict[str, Any]] = []
    claimed = delivered = refused = mismatched = 0
    for case, question, expected_other in CASES:
        by_scope: Dict[str, Any] = {}
        for scope in SCOPES:
            env = execute_governed_mi_query(
                MiQueryRequest(question=question,
                               source_portfolio_lens=scope), ctx).result or {}
            md = env.get("metadata") or {}
            by_scope[str(scope)] = {"route": md.get("route"), "ok": env.get("ok"),
                                    "answer": (env.get("answer") or "")[:200]}
        routes = {v["route"] for v in by_scope.values()}
        owns = routes == {"funded_bridge"}
        if owns:
            claimed += 1
        if owns != (expected_other is None):
            mismatched += 1

        # WHAT THE HANDLER ITSELF PRODUCED, before the guard.
        handler: Dict[str, Any] = {}
        if owns:
            spec, _m = parse_with_repair(question, semantics, llm_enabled=False)
            key, col, label = routing._bridge_dimension(spec, semantics)
            out = routing._route_bridge(
                question, spec, spec.to_dict(), client_id=client_id, run_id=None,
                output_root=root, portfolio_id=None, as_of=None,
                semantics=semantics)
            handler = {"dimensionKey": key, "dimensionLabel": label,
                       "ok": out.get("ok"),
                       "declaresGroupedBy": (out.get("metadata") or {}).get("groupedBy"),
                       "answer": (out.get("answer") or "")[:170]}
            if all(v["ok"] for v in by_scope.values()):
                delivered += 1
            else:
                refused += 1

        flag = "   " if owns == (expected_other is None) else "!! "
        print(f"\n{flag}{case}  {question!r}")
        print(f"      routes           : {sorted(r or '-' for r in routes)}")
        if owns:
            served = [s for s, v in by_scope.items() if v["ok"]]
            print(f"      DELIVERS for scopes: {served or 'NONE — refused everywhere'}")
            print(f"      handler dimension  : {handler.get('dimensionKey')!r} "
                  f"({handler.get('dimensionLabel')!r})")
            print(f"      handler declares   : groupedBy="
                  f"{handler.get('declaresGroupedBy')!r}")
            print(f"      handler answer     : {handler.get('answer')}")
            print(f"      user sees          : "
                  f"{by_scope['None']['answer'][:120]}")
        else:
            print(f"      expected NOT claimed: {expected_other}")
        records.append({"case": case, "question": question, "ownedByBridge": owns,
                        "expectedOther": expected_other, "byScope": by_scope,
                        "handler": handler})

    out_path = _REPO / "migration_phase0" / "ROUTE_OWNERSHIP_FUNDED_BRIDGE.json"
    out_path.write_text(json.dumps({"cases": records}, indent=2, default=str))

    print("\n" + "=" * 78)
    print(f"cases enumerated              : {len(CASES)} x {len(SCOPES)} scopes")
    print(f"claimed by funded_bridge      : {claimed}")
    print(f"  of which DELIVER            : {delivered}")
    print(f"  of which REFUSE every scope : {refused}")
    print(f"disagreeing with expectation  : {mismatched}")
    print(f"written                       : {out_path.relative_to(_REPO)}")
    print("=" * 78)
    return 1 if mismatched else 0


if __name__ == "__main__":
    sys.exit(main())
