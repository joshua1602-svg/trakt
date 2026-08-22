#!/usr/bin/env python3
"""question_interpretation/stage5_reachability.py

Does a time-series question REACH the code Stage 5 would change?

Asked before building anything, because the last three changes each measured as
something other than what they were and the common cause was reach:

  * `e35a01b` read as harmless because its hole was unreachable on the tape the
    instruments used. It was live on the shipped path for ordinary questions.
  * the three-variant measurement predicted clarify would move three answers.
    Two rules added while applying it removed all three.
  * clarify as shipped is correct and entirely invisible: the questions it fires
    on are claimed by `risk_limits` and the evolution routes before
    `reconcile_facets` is reached.

Stage 5 — the time-axis carriage — is five changes, from §4 of the inventory:

  1. `period_request.requested_unit`'s reading reaching a facet. Consumed at ONE
     call site today, `chat_routing.py:1020`, and discarded everywhere else.
  2. A time axis distinguishable from a dimension axis, in the facet layer.
  3. The grouping/filter conflation resolved first (done — Stage 4).
  4. `trend_grain` set from the question rather than hard-coded.
  5. A facet raisable for a CORRECT request, not only for a problem.

This traces one question — "balance by month over the last 12 months" — through
`execute_governed_mi_query` with routing exactly as shipped, and reports which
route claims it, which of those sites execute, and what the reader gets.

    python -m question_interpretation.stage5_reachability
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import warnings
from pathlib import Path
from typing import Any, Dict, List

warnings.simplefilter("ignore")

_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

#: The question the brief names, plus the shapes either side of it. A single
#: phrasing would not distinguish "the route claims time-series questions" from
#: "the route claims THIS sentence".
QUESTIONS: List[Dict[str, str]] = [
    {"id": "T1", "question": "balance by month over the last 12 months",
     "note": "the briefed question: a time grouping AND a period filter"},
    {"id": "T2", "question": "funded balance by month",
     "note": "the time grouping alone, no period"},
    {"id": "T3", "question": "funded balance by quarter",
     "note": "a coarser grain the parser does not recognise as a time axis"},
    {"id": "T4", "question": "how many loans each month",
     "note": "'each month' — a grain phrasing with no 'by'"},
    {"id": "T5", "question": "balance by month by region",
     "note": "a time axis and a dimension axis in one sentence"},
    {"id": "T6", "question": "balance by week over the last 6 months",
     "note": "finer than the book's grain — the one case with a live consumer"},
    {"id": "T7", "question": "balance by region",
     "note": "control: no time axis at all"},
]

#: Every site the five changes would alter, with the change it belongs to.
_SITES = [
    ("period_request.requested_unit", "1 — the reading reaching a facet"),
    ("period_request.finer_than", "1 — the reading reaching a facet"),
    ("period_request.granularity_clarification", "1 — its only live consumer"),
    ("execution_receipt.detect_requested_facets", "2, 5 — where a time facet would be raised"),
    ("execution_receipt.granularity_disclosure", "2, 5 — the disclosure shape that exists"),
    ("execution_receipt.reconcile_facets", "5 — where a raised facet is stamped"),
    ("execution_receipt.reconcile_routed_facets", "5 — the routed adjudicator"),
    ("execution_receipt.assess", "5 — where a facet reaches the reader"),
    ("execution_receipt._split_named_dimension_roles", "3 — Stage 4's split"),
]


def _install_spies(seen: Dict[str, int]):
    """Wrap each site so the trace records what actually ran, not what should."""
    from mi_agent import execution_receipt as R
    from mi_agent import period_request as P

    undo = []

    def wrap(module, name, key):
        original = getattr(module, name, None)
        if original is None or not callable(original):
            return

        def spy(*a, **kw):
            seen[key] = seen.get(key, 0) + 1
            return original(*a, **kw)

        setattr(module, name, spy)
        undo.append((module, name, original))

    for site, _why in _SITES:
        mod_name, fn = site.split(".", 1)
        wrap({"period_request": P, "execution_receipt": R}[mod_name], fn, site)
    return undo


def _remove_spies(undo) -> None:
    for module, name, original in undo:
        setattr(module, name, original)


def trace(question: str) -> Dict[str, Any]:
    from demo_platform import config as cfg
    os.environ.update(cfg.mi_env(period_role="current"))
    os.environ["MI_AGENT_LLM_PARSER"] = "off"
    os.environ["MI_AGENT_LLM_ENABLED"] = "0"
    import logging
    logging.disable(logging.WARNING)

    from mi_agent_api.mi_service import MiQueryRequest, execute_governed_mi_query
    from trakt_core.context import ExecutionContext

    seen: Dict[str, int] = {}
    undo = _install_spies(seen)
    try:
        res = (execute_governed_mi_query(
            MiQueryRequest(question=question),
            ExecutionContext.for_internal(cfg.CLIENT_ID)).result or {})
    finally:
        _remove_spies(undo)

    meta = res.get("metadata") or {}
    guard = meta.get("semantic_guard") or res.get("semanticGuard") or {}
    spec = res.get("spec") or {}
    return {
        "question": question,
        "route": meta.get("route") or "(none — point-in-time)",
        "ok": res.get("ok"),
        "verdict": guard.get("verdict"),
        "facets": sorted((f.get("kind"), f.get("status"))
                         for f in (guard.get("facets") or [])),
        "spec_trend_grain": spec.get("trend_grain"),
        "spec_temporal_mode": spec.get("temporal_mode"),
        "spec_dimension": spec.get("dimension"),
        "spec_dimensions": spec.get("dimensions"),
        "spec_chart_type": spec.get("chart_type"),
        "artifacts": [a.get("type") for a in (res.get("artifacts") or [])],
        "answer": str(res.get("answer") or res.get("error") or "")[:300],
        "sites_reached": dict(seen),
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--json", type=Path, default=None)
    ap.add_argument("--only", default=None, help="one question id")
    args = ap.parse_args(argv)

    rows = []
    for case in QUESTIONS:
        if args.only and case["id"] != args.only:
            continue
        row = trace(case["question"])
        row["id"], row["note"] = case["id"], case["note"]
        rows.append(row)

    print("=" * 84)
    print("STAGE 5 REACHABILITY — does a time-series question reach the carriage?")
    print("=" * 84)
    for r in rows:
        print("\n%-4s %s" % (r["id"], r["question"]))
        print("     %s" % r["note"])
        print("     route      %s" % r["route"])
        print("     outcome    ok=%s verdict=%s artifacts=%s"
              % (r["ok"], r["verdict"], r["artifacts"]))
        print("     spec       trend_grain=%r temporal_mode=%r dimension=%r chart=%r"
              % (r["spec_trend_grain"], r["spec_temporal_mode"],
                 r["spec_dimension"], r["spec_chart_type"]))
        print("     facets     %s" % (r["facets"] or "none"))
        print("     answer     %s" % r["answer"][:160])
        print("     sites reached:")
        for site, why in _SITES:
            n = r["sites_reached"].get(site, 0)
            print("       %-6s %-52s %s"
                  % (("%dx" % n) if n else "  -", site, why))

    if args.json:
        args.json.write_text(json.dumps(rows, indent=2, default=str),
                             encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
