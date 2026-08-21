#!/usr/bin/env python3
"""question_interpretation/hole_reachability.py

Is a hole in the stamping matrix REACHABLE, and by what?

The lesson of e35a01b is that "no surface reached it" and "it cannot be
reached" were treated as the same statement. They are not, and conflating them
hid a live refusal for a full commit. So reachability is established by
CONSTRUCTION here, never by inspection:

  1. `(point-in-time)/row_population` — the hole the role split lands a facet
     in. Probed on every book in the repository, through both entry points, on
     questions whose shape reaches it.
  2. The ROUTING DEPENDENCY. 61 of 64 unresolved facets never reach
     `reconcile_facets` because `chat_routing.try_route` claims their questions
     first. That is a property of routing, not of any fix. This forces routing
     off and reports what arrives and what happens to it — including whether the
     governed comparison alone holds for the 14 governed lending windows.

Nothing here is a fix. `--force-point-in-time` monkeypatches routing inside this
process only, to answer "what would arrive if routing changed".

    python -m question_interpretation.hole_reachability
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import warnings
from pathlib import Path
from typing import Any, Dict, List, Tuple

warnings.simplefilter("ignore")

_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parent
_EVIDENCE = _REPO_ROOT / "due_diligence" / "evidence" / "analytical_intent_v1"
for p in (str(_REPO_ROOT), str(_EVIDENCE)):
    if p not in sys.path:
        sys.path.insert(0, p)

_REAL_TAPE = (_REPO_ROOT / "demo_platform" / "workspace" / "store" / "processed"
              / "platform" / "alderbridge" / "2026-06-30"
              / "platform_canonical_typed.csv")

#: The seven questions whose facets are governed lending windows, from the
#: three-variant measurement. All 14 facets (7 x 2 books) sit in the unresolved
#: set and none of them reaches `reconcile_facets` today.
GOVERNED_WINDOW_QUESTIONS: List[Tuple[str, str]] = [
    ("Q1.1", "How has the profile of our new lending changed over the last few months?"),
    ("Q7.1", "How does the front book compare with our older lending from a risk perspective?"),
    ("Q7.3", "How different is the risk profile of recent originations versus the back book?"),
    ("Q7.4", "Compare the credit profile of the front book with our seasoned loans."),
    ("Q8.1", "How has lending to the front book changed compared with the back book?"),
    ("Q8.3", "Are the front book and the back book balances developing differently over time?"),
    ("Q8.4", "Compare how the the front book and the back book books have changed "
             "over the last few months."),
]

#: Questions that name a governed population and are NOT composite, so no route
#: claims them and they fall through to the point-in-time path — the shape that
#: reaches the population hole without any routing change at all.
NON_COMPOSITE_POPULATION_QUESTIONS: List[Tuple[str, str]] = [
    ("N1", "What is the balance of newly originated loans?"),
    ("N2", "What is the average LTV of the back book?"),
    ("N3", "What is the balance of the front book?"),
]


def _semantics():
    from mi_agent.mi_query_validator import load_mi_semantics
    return load_mi_semantics(_REPO_ROOT / "mi_agent" / "mi_semantics_field_registry.yaml")


# --------------------------------------------------------------------------- #
# Entry point 1 — the workflow directly (what the calibration bank uses)
# --------------------------------------------------------------------------- #
def via_workflow(questions) -> List[Dict[str, Any]]:
    """`run_mi_agent_query` — always the point-in-time path, no routing at all."""
    from mi_agent import mi_calibration as CAL
    semantics = _semantics()
    df = CAL.default_bank_frame(_REAL_TAPE)
    rows = []
    for qid, q in questions:
        res = CAL._run(q, df, semantics, False)
        guard = res.get("semantic_guard") or {}
        rows.append({
            "id": qid, "question": q, "entry": "run_mi_agent_query",
            "ok": res.get("ok"), "verdict": guard.get("verdict"),
            "facets": sorted((f.get("kind"), f.get("status"))
                             for f in (guard.get("facets") or [])),
            "answer": str(res.get("answer") or "")[:160],
        })
    return rows


# --------------------------------------------------------------------------- #
# Entry point 2 — the governed service (what every real caller uses)
# --------------------------------------------------------------------------- #
def via_service(questions, *, force_point_in_time: bool = False
                ) -> List[Dict[str, Any]]:
    from demo_platform import config as cfg
    os.environ.update(cfg.mi_env(period_role="current"))
    os.environ["MI_AGENT_LLM_PARSER"] = "off"
    os.environ["MI_AGENT_LLM_ENABLED"] = "0"
    import logging
    logging.disable(logging.WARNING)

    from mi_agent_api import chat_routing as chat_routing_mod
    from mi_agent_api.mi_service import MiQueryRequest, execute_governed_mi_query
    from trakt_core.context import ExecutionContext

    restore = None
    if force_point_in_time:
        # THE ROUTING DEPENDENCY, made testable. Every routed capability
        # declines, so every question falls through to the point-in-time path
        # and its facets reach `reconcile_facets` and the role split. This is
        # what a routing change would deliver, all at once.
        restore = chat_routing_mod.try_route
        chat_routing_mod.try_route = lambda *a, **kw: None

    ctx = ExecutionContext.for_internal(cfg.CLIENT_ID)
    rows = []
    try:
        for qid, q in questions:
            res = (execute_governed_mi_query(MiQueryRequest(question=q), ctx).result
                   or {})
            meta = res.get("metadata") or {}
            guard = (meta.get("semantic_guard") or res.get("semanticGuard") or {})
            rows.append({
                "id": qid, "question": q,
                "entry": "execute_governed_mi_query"
                         + ("/routing-off" if force_point_in_time else ""),
                "route": meta.get("route") or (res.get("metadata") or {}).get("route"),
                "ok": res.get("ok"), "verdict": guard.get("verdict"),
                "facets": sorted((f.get("kind"), f.get("status"))
                                 for f in (guard.get("facets") or [])),
                "answer": str(res.get("answer") or res.get("error") or "")[:200],
            })
    finally:
        if restore is not None:
            chat_routing_mod.try_route = restore
    return rows


def _print(title: str, rows: List[Dict[str, Any]]) -> None:
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)
    for r in rows:
        print("%-5s ok=%-5s verdict=%-9s route=%s"
              % (r["id"], r["ok"], r.get("verdict"), r.get("route", "-")))
        print("      facets: %s" % (r["facets"] or "none"))
        print("      %s" % r["answer"])


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--json", type=Path, default=None)
    ap.add_argument("--stage", choices=("workflow", "service", "routing-off"),
                    required=True,
                    help="one stage per process: the app binds its book and its "
                         "routing at import time, so two stages in one process "
                         "would measure the first one twice")
    args = ap.parse_args(argv)

    questions = NON_COMPOSITE_POPULATION_QUESTIONS + GOVERNED_WINDOW_QUESTIONS
    if args.stage == "workflow":
        rows = via_workflow(questions)
        _print("HOLE 1 — via run_mi_agent_query (no routing at all)", rows)
    elif args.stage == "service":
        rows = via_service(questions)
        _print("HOLE 1 — via the governed service, routing as shipped", rows)
    else:
        rows = via_service(questions, force_point_in_time=True)
        _print("ROUTING DEPENDENCY — every route declines, all facets arrive", rows)

    if args.json:
        args.json.write_text(json.dumps(rows, indent=2, default=str),
                             encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
