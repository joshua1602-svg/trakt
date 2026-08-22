#!/usr/bin/env python3
"""question_interpretation/route_carriage.py

Two questions about carriage, asked of every route that can answer.

1. CAN TIER THREE CERTIFY A WRONG BREAKDOWN?

   The routed grouping check has three tiers. One and two prove: the frame names
   the field (APPLIED) or carries no axis at all (LOST). Tier three stamps
   APPLIED on the strength that SOME breakdown happened, without establishing it
   was the requested one — narrower than the defect it replaced, and the same
   shape.

   The test is not whether tier three is imprecise; it is whether any route in
   it can group by a dimension OTHER than the one the question named. A route
   whose axis is FIXED certifies every dimension a question might name, and at
   most one of those can be right.

   Measured by pairing, per route, the field each grouping facet asked for with
   the axis the answer actually carried. A route pairing one axis with several
   different fields is certifying at least one of them wrongly.

2. WHICH ROUTES CAN RECEIVE A GOVERNED POPULATION?

   Governed populations have now failed to arrive at three separate routes. The
   concept resolves correctly and has since the first traces; the failure is
   carriage, and a different route each time. Rather than wait for the fourth,
   this reports for every route whether a governed population reaches it and
   what it does with the question if not.

    python -m question_interpretation.route_carriage
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import sys
import warnings
from pathlib import Path
from typing import Any, Dict, List, Set

warnings.simplefilter("ignore")

_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parent
_EVIDENCE = _REPO_ROOT / "due_diligence" / "evidence" / "analytical_intent_v1"
for p in (str(_REPO_ROOT), str(_EVIDENCE)):
    if p not in sys.path:
        sys.path.insert(0, p)

#: Questions naming a GOVERNED population, in wordings that reach different
#: routes. The population vocabulary is constant; only the route varies, which
#: is the whole point of asking it this way.
POPULATION_PROBES = [
    ("front book, simple", "What is the balance of the front book?"),
    ("front book, composite",
     "How does the front book compare with our older lending from a risk perspective?"),
    ("front book, over time",
     "How have the front book and the back book changed over time?"),
    ("new lending, simple", "What is the balance of new lending?"),
    ("new lending, run rate",
     "What is the run rate of new lending and is it accelerating?"),
    ("new lending, movement",
     "How has new lending moved since last month?"),
    ("front book, geography",
     "What is the geographic exposure of the front book?"),
    ("front book, limits",
     "Are any concentration limits breached for the front book?"),
    ("front book, forecast",
     "Forecast the balance of the front book for the next quarter."),
    ("back book, evolution", "Show the back book balance by month."),
]


def _ask():
    from demo_platform import config as cfg
    os.environ.update(cfg.mi_env(period_role="current"))
    os.environ["MI_AGENT_LLM_PARSER"] = "off"
    os.environ["MI_AGENT_LLM_ENABLED"] = "0"
    import logging
    logging.disable(logging.WARNING)

    from mi_agent_api.mi_service import MiQueryRequest, execute_governed_mi_query
    from trakt_core.context import ExecutionContext

    ctx = ExecutionContext.for_internal(cfg.CLIENT_ID)
    return lambda q: (execute_governed_mi_query(
        MiQueryRequest(question=q), ctx).result or {})


def _tier(facet: Dict[str, Any], axes: Set[str]) -> str:
    """Which tier of the routed grouping check stamped this facet."""
    field = str(facet.get("field") or "").strip().lower()
    if facet.get("status") != "applied":
        return "n/a"
    if field and field in axes:
        return "1 — named in the frame"
    if not axes:
        return "?? — applied with no axis at all"
    return "3 — an axis exists, unidentified"


def scan_tier_three(limit: int = 260) -> Dict[str, Any]:
    from mi_agent import execution_receipt as R
    from question_interpretation.lexical_decisions import corpus

    ask = _ask()
    pairs: Dict[str, Set] = collections.defaultdict(set)
    rows: List[Dict[str, Any]] = []
    for case in list(corpus())[:limit]:
        try:
            res = ask(case["question"])
        except Exception:                                     # pragma: no cover
            continue
        meta = res.get("metadata") or {}
        route = meta.get("route")
        if not route:
            continue                     # point-in-time uses the other branch
        guard = meta.get("semantic_guard") or res.get("semanticGuard") or {}
        axes = R.answer_axis_keys(res)
        for facet in (guard.get("facets") or []):
            if facet.get("kind") != R.KIND_GROUPING:
                continue
            tier = _tier(facet, axes)
            if not tier.startswith("3"):
                continue
            pairs[route].add((str(facet.get("field")), tuple(sorted(axes))))
            rows.append({"id": case["id"], "question": case["question"],
                         "route": route, "field": facet.get("field"),
                         "axes": sorted(axes)})
    suspect = {}
    for route, seen in pairs.items():
        by_axes: Dict[tuple, Set[str]] = collections.defaultdict(set)
        for field, axes in seen:
            by_axes[axes].add(field)
        # One axis paired with several requested fields: at most one is right.
        multi = {axes: sorted(fields) for axes, fields in by_axes.items()
                 if len(fields) > 1}
        if multi:
            suspect[route] = {str(list(a)): f for a, f in multi.items()}
    return {"tier_three_rows": rows,
            "routes_reaching_tier_three": sorted(pairs),
            "certifying_more_than_one_field_per_axis": suspect}


def scan_populations() -> List[Dict[str, Any]]:
    from mi_agent import execution_receipt as R

    ask = _ask()
    out = []
    for label, question in POPULATION_PROBES:
        res = ask(question)
        meta = res.get("metadata") or {}
        guard = meta.get("semantic_guard") or res.get("semanticGuard") or {}
        facets = [(str(f.get("kind")), str(f.get("field")), str(f.get("status")))
                  for f in (guard.get("facets") or [])]
        populations = [f for f in facets if f[0] == R.KIND_POPULATION]
        out.append({
            "label": label, "question": question,
            "route": meta.get("route") or "(none — point-in-time)",
            "ok": bool(res.get("ok")),
            "verdict": guard.get("verdict"),
            "population_facets": populations,
            "arrives": bool(populations),
            "applied": any(f[2] == "applied" for f in populations),
            "facets": facets,
            "answer": str(res.get("answer") or res.get("error") or "")[:170],
        })
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--json", type=Path, default=None)
    ap.add_argument("--limit", type=int, default=260)
    args = ap.parse_args(argv)

    tier = scan_tier_three(args.limit)
    print("=" * 84)
    print("1. CAN TIER THREE CERTIFY A WRONG BREAKDOWN?")
    print("=" * 84)
    print("routes reaching tier three: %s"
          % (", ".join(tier["routes_reaching_tier_three"]) or "none"))
    print("tier-three stamps observed : %d" % len(tier["tier_three_rows"]))
    suspect = tier["certifying_more_than_one_field_per_axis"]
    if not suspect:
        print("\nNo route paired one axis with more than one requested field.")
    else:
        print("\nROUTES CERTIFYING MORE THAN ONE FIELD FOR THE SAME AXIS:")
        for route, multi in suspect.items():
            for axes, fields in multi.items():
                print("   %-24s axis=%s" % (route, axes))
                print("      fields certified: %s" % ", ".join(fields))
    for row in tier["tier_three_rows"][:12]:
        print("   %-20s %-22s field=%-22s axes=%s"
              % (row["id"], row["route"], row["field"], row["axes"]))

    pops = scan_populations()
    print("\n" + "=" * 84)
    print("2. WHICH ROUTES CAN RECEIVE A GOVERNED POPULATION?")
    print("=" * 84)
    for row in pops:
        print("\n%-24s %s" % (row["label"], row["question"][:56]))
        print("   route=%-24s ok=%-5s verdict=%s"
              % (row["route"], row["ok"], row["verdict"]))
        print("   population facet: %s"
              % (row["population_facets"] or "NONE — it did not arrive"))
        print("   all facets      : %s" % (row["facets"] or "none"))
        print("   %s" % row["answer"][:150].replace("\n", " "))

    by_route: Dict[str, List[str]] = collections.defaultdict(list)
    for row in pops:
        state = ("applied" if row["applied"]
                 else "arrives, not applied" if row["arrives"]
                 else "DOES NOT ARRIVE")
        by_route[row["route"]].append("%s (%s)" % (state, row["label"]))
    print("\n" + "-" * 84)
    for route in sorted(by_route):
        print("%-26s %s" % (route, "; ".join(by_route[route])))

    if args.json:
        args.json.write_text(json.dumps({"tier_three": tier, "populations": pops},
                                        indent=2, default=str), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
