#!/usr/bin/env python3
"""question_interpretation/time_series_surface.py

THE TIME-SERIES SURFACE — eight shapes the capability summary marks as never
measured by a standing surface.

Ratings are PROVEN / PARTIAL / ABSENT, and the rule that decides them is the one
this programme has been enforcing all along:

    A WHOLE-BOOK SERIES RETURNED FOR A SEGMENTED REQUEST IS **ABSENT**,
    NOT PARTIAL, WHATEVER THE RECEIPT SAYS.

So this surface never trusts `dimensionsApplied`. It opens the ARTIFACT and
counts distinct values in the columns the question asked for. A receipt claiming
a breakdown over a table that carries one series is a false claim, and the
difference between the two is the whole point of looking.

    python -m question_interpretation.time_series_surface --book alderbridge
    python -m question_interpretation.time_series_surface --book kestrelmoor
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

warnings.simplefilter("ignore")

_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

PROVEN, PARTIAL, ABSENT, UNMEASURED = "PROVEN", "PARTIAL", "ABSENT", "UNMEASURED"

#: Column-name fragments that mean "this column is the TIME axis".
_TIME_HINTS = ("period", "month", "quarter", "date", "week", "as_of", "asof",
               "reporting", "snapshot", "vintage_year")

#: (shape id, name, [phrasings], expected dimension column fragments)
SHAPES: Tuple[Tuple[str, str, List[str], List[str]], ...] = (
    ("T1", "metric x time", [
        "Show me balance by month",
        "balance over time",
        "How has the funded balance moved over time?",
        "total balance by reporting period",
    ], []),
    ("T2", "metric x time x filter", [
        "Show me balance by month for loans over £150k",
        "balance over time for loans with LTV above 50%",
        "How has balance moved over time for the front book?",
        "balance by month where balance is above £150,000",
    ], []),
    ("T3", "metric x time x dimension", [
        "Show me balance by month by region",
        "balance over time by region",
        "How has balance moved over time, broken down by region?",
        "balance by month split by LTV band",
    ], ["region", "geograph", "ltv_bucket"]),
    ("T4", "metric x time x dimension x filter", [
        "Show me balance by month by region for loans over £150k",
        "balance over time by region for the front book",
        "balance by month by LTV band for loans above £150,000",
    ], ["region", "geograph", "ltv_bucket"]),
    ("T5", "metric x time x two dimensions", [
        "Show me balance by month by region and LTV band",
        "balance over time by region and ticket size",
        "balance by month broken down by LTV band and region",
    ], ["region", "geograph", "ltv_bucket", "ticket_bucket"]),
    ("T6", "period-over-period movement by segment", [
        "How has balance changed since last month by region?",
        "What moved between periods by region?",
        "month-on-month change in balance by region",
        "period on period movement by LTV band",
    ], ["region", "geograph", "ltv_bucket"]),
    ("T7", "ranked historical movement", [
        "Which region grew the most over the last three months?",
        "Which LTV band moved most between periods?",
        "Rank regions by balance growth over time",
        "Which region has grown fastest?",
    ], ["region", "geograph", "ltv_bucket"]),
    ("T8", "comparison of two historical segments", [
        "How has the front book moved over time compared with the back book?",
        "Compare balance over time for direct and acquired",
        "How have direct and acquired balances moved over the periods?",
    ], ["seasoning", "source_portfolio", "provenance"]),
)


def _client(book: str):
    if book == "alderbridge":
        from demo_platform import config as cfg
        os.environ.update(cfg.mi_env(period_role="current"))
        client_id = cfg.CLIENT_ID
    elif book == "kestrelmoor":
        from question_interpretation.run_robustness_deterministic import (
            _KESTRELMOOR_ROOT)
        from tests.analytical import second_book as bk
        if not _KESTRELMOOR_ROOT.exists():
            bk.build(_KESTRELMOOR_ROOT)
        os.environ.update(bk.mi_env(_KESTRELMOOR_ROOT))
        client_id = os.environ["MI_AGENT_CLIENT_ID"]
    else:
        raise ValueError("unknown book %r" % (book,))
    os.environ["MI_AGENT_LLM_PARSER"] = "off"
    os.environ["MI_AGENT_LLM_ENABLED"] = "0"
    import logging
    logging.disable(logging.WARNING)
    from mi_agent_api.mi_service import MiQueryRequest, execute_governed_mi_query
    from trakt_core.context import ExecutionContext
    ctx = ExecutionContext.for_internal(client_id)
    return lambda q: (execute_governed_mi_query(
        MiQueryRequest(question=q), ctx).result or {})


def _distinct(rows: List[Dict[str, Any]], key: str) -> int:
    return len({str(r.get(key)) for r in rows if r.get(key) is not None})


def inspect_artifacts(artifacts: List[Dict[str, Any]],
                      want_dims: List[str]) -> Dict[str, Any]:
    """What the RENDERED OBJECT actually carries. Never the receipt's word for it.

    Returns the time-axis column and its distinct count, and for each requested
    dimension the column that matched and how many distinct values it holds. A
    dimension column present with ONE value is not a breakdown.
    """
    best: Dict[str, Any] = {"artifact": None, "rows": 0, "columns": [],
                            "time_col": None, "time_points": 0, "dims": {}}
    for art in artifacts:
        rows = art.get("rows") or []
        if not rows:
            continue
        cols = list(rows[0].keys())
        time_col, time_points = None, 0
        for c in cols:
            if any(h in c.lower() for h in _TIME_HINTS):
                n = _distinct(rows, c)
                if n > time_points:
                    time_col, time_points = c, n
        dims: Dict[str, Any] = {}
        for frag in want_dims:
            for c in cols:
                if frag in c.lower():
                    dims[c] = _distinct(rows, c)
                    break
        score = (time_points > 1) + len([v for v in dims.values() if v > 1])
        prev = (best["time_points"] > 1) + len(
            [v for v in best["dims"].values() if v > 1])
        if best["artifact"] is None or score > prev:
            best = {"artifact": art.get("type"), "rows": len(rows),
                    "columns": cols, "time_col": time_col,
                    "time_points": time_points, "dims": dims}
    return best


def rate(ok: bool, want_dims: List[str], seen: Dict[str, Any],
         claimed_dims: List[str]) -> Tuple[str, str]:
    """PROVEN / PARTIAL / ABSENT for one run, and why.

    THE RULE: a whole-book series for a segmented request is ABSENT. A receipt
    claiming the breakdown while the artifact carries one series is worse than
    a refusal and is rated the same as one — with the false claim named.
    """
    if not ok:
        return ABSENT, "refused"
    real_dims = {k: v for k, v in seen["dims"].items() if v > 1}
    has_time = seen["time_points"] > 1
    if not want_dims:
        if has_time:
            return PROVEN, "%d time points in %s" % (seen["time_points"], seen["time_col"])
        return ABSENT, ("answered with no time axis in the artifact (%s)"
                        % (seen["columns"] or "no rows"))
    # a segmented request
    if has_time and real_dims:
        return PROVEN, ("%d time points x %s" % (seen["time_points"],
                        ", ".join("%s=%d" % (k, v) for k, v in real_dims.items())))
    if has_time and not real_dims:
        note = "WHOLE-BOOK SERIES for a segmented request"
        if claimed_dims:
            note += " — receipt CLAIMS %s" % (claimed_dims,)
        return ABSENT, note
    if real_dims and not has_time:
        # THE TIME AXIS WAS REQUESTED AND IS NOT IN THE ARTIFACT. Whether that
        # is honest depends entirely on whether anything said so.
        return ABSENT, ("TIME AXIS DROPPED — breakdown delivered over one period "
                        "(%s)" % ", ".join("%s=%d" % (k, v)
                                           for k, v in real_dims.items()))
    return ABSENT, "neither a time axis nor the requested breakdown"


def run(book: str) -> Dict[str, Any]:
    ask = _client(book)
    out: List[Dict[str, Any]] = []
    for sid, name, phrasings, want in SHAPES:
        for q in phrasings:
            res = ask(q)
            meta = res.get("metadata") or {}
            summary = res.get("executionSummary") or {}
            guard = meta.get("semantic_guard") or {}
            arts = res.get("artifacts") or []
            seen = inspect_artifacts(arts, want)
            claimed = list(summary.get("dimensionsApplied") or ())
            verdict, why = rate(bool(res.get("ok")), want, seen, claimed)
            # Was the loss DISCLOSED? A dropped axis with a facet, a warning or
            # a notApplied entry is an honest partial answer. One with ok=True,
            # no verdict, no facet and no note is a silent drop, and that is the
            # distinction this surface exists to draw.
            facets = guard.get("facets") or []
            disclosed = bool(facets or summary.get("notApplied")
                             or guard.get("verdict") in ("partial", "clarify", "refuse"))
            silent = (verdict == ABSENT and bool(res.get("ok"))
                      and not disclosed
                      and ("TIME AXIS DROPPED" in why or "WHOLE-BOOK SERIES" in why))
            if silent:
                why += "  ** SILENTLY — ok=True, no verdict, no facet, no note **"
            out.append({
                "shape": sid, "shape_name": name, "question": q, "book": book,
                "ok": bool(res.get("ok")), "route": meta.get("route"),
                "guard_verdict": guard.get("verdict"),
                "claimed_dimensions": claimed,
                "claimed_filters": list(summary.get("filtersApplied") or ()),
                "population": summary.get("population"),
                "artifact": seen, "rating": verdict, "why": why,
                "silent_drop": silent, "disclosed": disclosed,
                "facets": [(f.get("kind"), f.get("status")) for f in facets],
                "answer": str(res.get("answer") or res.get("error") or "")[:200],
            })
    return {"book": book, "rows": out}


def shape_rating(ratings: List[str]) -> str:
    """One rating per SHAPE from its phrasings.

    PROVEN only if every phrasing that was not refused proved it AND at least
    one did. PARTIAL where some phrasings prove it and others do not — a
    capability that works only in the wording nobody guesses is partial, not
    proven. ABSENT if none proved it.
    """
    if PROVEN in ratings and all(r == PROVEN for r in ratings):
        return PROVEN
    if PROVEN in ratings:
        return PARTIAL
    return ABSENT


def report(result: Dict[str, Any]) -> int:
    rows = result["rows"]
    print("=" * 92)
    print("TIME-SERIES SURFACE — book: %s" % result["book"])
    print("  A whole-book series returned for a segmented request is ABSENT,")
    print("  not partial, whatever the receipt says.")
    print("=" * 92)
    for sid, name, phrasings, _want in SHAPES:
        mine = [r for r in rows if r["shape"] == sid]
        overall = shape_rating([r["rating"] for r in mine])
        print()
        print("%-3s %-46s  %s" % (sid, name, overall))
        print("-" * 92)
        for r in mine:
            print("  %-8s %-58s route=%s" % (r["rating"], r["question"][:58], r["route"]))
            print("           %s" % r["why"])
            a = r["artifact"]
            if a["artifact"]:
                print("           artifact=%s rows=%s time=%s(%s) dims=%s"
                      % (a["artifact"], a["rows"], a["time_col"], a["time_points"],
                         a["dims"] or "{}"))
            if r["claimed_dimensions"]:
                print("           receipt claims dimensions: %s" % r["claimed_dimensions"])
    print()
    print("=" * 92)
    print("SUMMARY — %s" % result["book"])
    print("=" * 92)
    for sid, name, _p, _w in SHAPES:
        mine = [r for r in rows if r["shape"] == sid]
        print("  %-3s %-46s %-8s  (%d phrasings: %s)"
              % (sid, name, shape_rating([r["rating"] for r in mine]), len(mine),
                 ", ".join(r["rating"][:1] for r in mine)))
    silent = [r for r in rows if r.get("silent_drop")]
    print()
    print("  SILENT DROPS — ok=True, requested axis absent, NOTHING disclosed: %d"
          % len(silent))
    for r in silent:
        print("     %s  %s" % (r["shape"], r["question"][:70]))
        print("          %s" % r["why"][:96])
    refused = [r for r in rows if not r["ok"]]
    print()
    print("  HONEST REFUSALS (the loss was named): %d of %d runs"
          % (len(refused), len(rows)))
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--book", default="alderbridge",
                    choices=("alderbridge", "kestrelmoor"))
    ap.add_argument("--json", metavar="PATH")
    args = ap.parse_args(argv)
    result = run(args.book)
    if args.json:
        Path(args.json).write_text(json.dumps(result, indent=2, default=str),
                                   encoding="utf-8")
    return report(result)


if __name__ == "__main__":
    raise SystemExit(main())
