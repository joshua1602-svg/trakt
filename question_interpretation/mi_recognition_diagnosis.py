#!/usr/bin/env python3
"""question_interpretation/mi_recognition_diagnosis.py

WHERE THE 61 PHRASINGS FAIL — recognition, or capability?

A shape rated ABSENT does not say whether the product cannot do the thing or
cannot be asked for it. Those need different work and cost differently, and the
declared phrasing bank cannot tell them apart because it was written in the
vocabulary the system was built against.

This separates them for every phrasing, on evidence:

**RECOGNITION** — the sentence named something the parser did not resolve. Every
one of these shapes is a time-series shape, so a time axis is always named; a
measure is always named; shapes T3-T8 additionally name a breakdown dimension.
If the parser returns no metric, no time axis, or no dimension where the shape
requires one, the question was not understood — whatever happened downstream.

**CAPABILITY** — the parser resolved everything the sentence named, a route
claimed it, and the answer still did not carry it. That is the product's limit.

The distinction the brief asks for separately — **did any route claim it at
all** — is reported as its own column, because a question that reaches no route
never got as far as a capability.

    python -m question_interpretation.mi_recognition_diagnosis --bank combined
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import warnings
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

warnings.simplefilter("ignore")

_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from question_interpretation.time_series_surface import SHAPES  # noqa: E402
from question_interpretation.mi_capability_recontent import (  # noqa: E402
    _bank, _book_env, full_census, preflight, rate_content, dimension_domains,
    request_target)

#: Which element of the sentence the PARSE did not carry. Evidence, not a
#: verdict: a spec that drops a dimension it could resolve on its own is not the
#: same defect as one that never knew the word.
CAUSE_NO_METRIC = "metric not resolved"
CAUSE_NO_TIME = "time axis not resolved"
CAUSE_NO_DIMENSION = "dimension not resolved"
CAUSE_NONE = "everything named was resolved"

#: THE VERDICTS.
#:
#: An empty `spec.dimension` is NOT proof the word was unknown. "Show me balance
#: by region" resolves `collateral_geography`; "Show me balance by month by
#: region" resolves a time axis and drops the dimension — and the guard still
#: refuses by NAME ("I understood that you asked for region"). The vocabulary
#: knew the word; the spec could not hold both limbs. That is a representation
#: limit — precisely what the per-period-breakdown build is for — and counting
#: it as a recognition failure would inflate the cheap category with the
#: expensive one.
#:
#: So the discriminator is not the parse alone. It is whether the SAME REQUEST
#: is reachable by other words (the standing sibling rule: same shape AND same
#: target), and whether the system NAMED BACK the element it dropped.
DELIVERED = "DELIVERED"
#: The request is reachable — a sibling phrasing delivers it. This wording is
#: the only barrier. Cheap, client-facing, not a build.
WORDING = "WORDING"
#: Nothing in the sentence was understood well enough to name back, and no
#: sibling delivers. The words themselves are unknown.
UNPARSED = "UNPARSED"
#: Understood — resolved, or named back in the refusal — and still not
#: deliverable by any wording. The product's limit.
CAPABILITY = "CAPABILITY"


def _spec_fields(spec: Any) -> Dict[str, Any]:
    d = spec.model_dump() if hasattr(spec, "model_dump") else dict(spec.__dict__)
    return d


def resolved(spec: Any) -> Dict[str, bool]:
    """What the parser actually got out of the sentence.

    ``dimension`` is singular in the spec, and is the field a breakdown lands
    in. ``x`` carries the axis for a line chart; ``execution_mode='temporal'``
    and ``cohort_progression`` are the other two ways a time axis is expressed.
    """
    d = _spec_fields(spec)
    return {
        "metric": d.get("metric") is not None,
        "time": bool(d.get("x") or d.get("execution_mode") == "temporal"
                     or d.get("cohort_progression")),
        "dimension": d.get("dimension") is not None,
    }


def cause(want: List[str], got: Dict[str, bool]) -> str:
    """The FIRST unresolved element the sentence named."""
    if not got["metric"]:
        return CAUSE_NO_METRIC
    if not got["time"]:
        return CAUSE_NO_TIME
    if want and not got["dimension"]:
        return CAUSE_NO_DIMENSION
    return CAUSE_NONE


def _named_back(question: str, want: List[str], answer: str) -> bool:
    """Whether the system quoted back the thing it could not apply.

    `"I understood that you asked for region, but that could not be applied"`
    proves the word was understood. Matched against the requested dimension
    fragments and the target vocabulary, on the answer OR the refusal.
    """
    low = (answer or "").lower()
    if not low:
        return False
    if "i understood that you asked for" in low:
        return True
    return any(f.lower() in low for f in (want or []))


def run(book: str, bank: str, llm: bool = False) -> Dict[str, Any]:
    problems = preflight(book)
    if problems:
        return {"book": book, "blocked": problems}
    client_id = _book_env(book)
    os.environ["MI_AGENT_LLM_PARSER"] = "on" if llm else "off"
    os.environ["MI_AGENT_LLM_ENABLED"] = "1" if llm else "0"
    import logging
    logging.disable(logging.WARNING)

    from mi_agent import llm_query_parser as P
    from mi_agent.mi_query_validator import load_mi_semantics
    from mi_agent_api import data_source
    from mi_agent_api.mi_service import MiQueryRequest, execute_governed_mi_query
    from trakt_core.context import ExecutionContext

    semantics = load_mi_semantics(
        str(_REPO_ROOT / "mi_agent" / "mi_semantics_field_registry.yaml"))
    columns = set(data_source.get_dataframe().columns)
    ctx = ExecutionContext.for_internal(client_id)
    all_frags = sorted({f for _s, _n, _q, w in _bank("declared") for f in w})
    domains = dimension_domains(all_frags)

    rows: List[Dict[str, Any]] = []
    for sid, name, q, want in _bank(bank):
        spec, meta = P.parse_with_repair(q, semantics,
                                         available_columns=columns,
                                         llm_enabled=llm)
        res = (execute_governed_mi_query(MiQueryRequest(question=q), ctx).result
               or {})
        md = res.get("metadata") or {}
        route = md.get("route")
        ok = bool(res.get("ok"))
        census = full_census(res.get("artifacts") or [])
        rating, why, _ev = rate_content(ok, want, census, domains)
        got = resolved(spec)
        why_not = cause(want, got)
        delivered = rating == "PROVEN"
        answer_text = str(res.get("answer") or res.get("error") or "")
        # Did the system NAME BACK the element it dropped? The honour-or-clarify
        # refusal quotes what it could not apply, so this separates "did not
        # understand the word" from "understood it and could not carry it".
        named_back = _named_back(q, want, answer_text)
        rows.append({
            "named_back": named_back,
            "shape": sid, "question": q, "want": want,
            "target": request_target(q),
            "route": route, "claimed_by_a_route": route is not None,
            "ok": ok, "rating": rating, "delivered": delivered,
            "resolved": got, "cause": why_not,
            "parser_note": meta.get("note"),
            "parser_confidence": meta.get("parser_confidence"),
            "answer": str(res.get("answer") or res.get("error") or "")[:200],
        })
    return {"book": book, "bank": bank, "llm": llm,
            "rows": assign_verdicts(rows)}


def assign_verdicts(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Second pass: WORDING needs to know whether a SIBLING delivered.

    Sibling = same shape AND same target, per the standing rule in
    docs/mi_sibling_rule.md. Grouping by shape alone would count a phrasing
    reachable because a DIFFERENT question of the same shape works, which is the
    error that rule exists to prevent.
    """
    delivered_by_request = {
        (r["shape"], r["target"]) for r in rows if r["delivered"]}
    for r in rows:
        if r["delivered"]:
            r["verdict"] = DELIVERED
        elif (r["shape"], r["target"]) in delivered_by_request:
            r["verdict"] = WORDING
        elif r["cause"] != CAUSE_NONE and not r["named_back"]:
            r["verdict"] = UNPARSED
        else:
            r["verdict"] = CAPABILITY
    return rows


def by_shape(rows: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for sid, _name, _qs, _w in SHAPES:
        mine = [r for r in rows if r["shape"] == sid]
        if not mine:
            continue
        out[sid] = {
            "n": len(mine),
            "delivered": sum(1 for r in mine if r["verdict"] == DELIVERED),
            "wording": sum(1 for r in mine if r["verdict"] == WORDING),
            "unparsed": sum(1 for r in mine if r["verdict"] == UNPARSED),
            "capability": sum(1 for r in mine if r["verdict"] == CAPABILITY),
            "no_route": sum(1 for r in mine if not r["claimed_by_a_route"]),
            "causes": Counter(r["cause"] for r in mine
                              if r["verdict"] in (WORDING, UNPARSED)),
        }
    return out


def report(result: Dict[str, Any]) -> int:
    if result.get("blocked"):
        print("REFUSING TO MEASURE:")
        for p in result["blocked"]:
            print("  * %s" % p)
        return 4
    rows = result["rows"]
    shapes = by_shape(rows)
    n = len(rows)
    delivered = sum(1 for r in rows if r["verdict"] == DELIVERED)
    wording = sum(1 for r in rows if r["verdict"] == WORDING)
    unparsed = sum(1 for r in rows if r["verdict"] == UNPARSED)
    cap = sum(1 for r in rows if r["verdict"] == CAPABILITY)
    no_route = sum(1 for r in rows if not r["claimed_by_a_route"])

    print("=" * 96)
    print("RECOGNITION vs CAPABILITY — %s, %s bank, %d phrasings"
          % (result["book"], result["bank"], n))
    print("=" * 96)
    print()
    print("  DELIVERED                                            %3d  (%d%%)"
          % (delivered, round(100 * delivered / n)))
    print("  WORDING    — a sibling delivers this request         %3d  (%d%%)"
          % (wording, round(100 * wording / n)))
    print("  UNPARSED   — nothing understood, no sibling          %3d  (%d%%)"
          % (unparsed, round(100 * unparsed / n)))
    print("  CAPABILITY — understood, no wording reaches it       %3d  (%d%%)"
          % (cap, round(100 * cap / n)))
    print()
    print("  RECOGNITION total (WORDING + UNPARSED)               %3d  (%d%%)"
          % (wording + unparsed, round(100 * (wording + unparsed) / n)))
    print()
    print("  reached NO route at all                    %3d  (%d%%)"
          % (no_route, round(100 * no_route / n)))
    print()
    print("BY SHAPE")
    print("-" * 96)
    print("  %-4s %4s %10s %8s %9s %11s %9s"
          % ("", "n", "delivered", "wording", "unparsed", "capability",
             "no route"))
    for sid, st in shapes.items():
        print("  %-4s %4d %10d %8d %9d %11d %9d"
              % (sid, st["n"], st["delivered"], st["wording"], st["unparsed"],
                 st["capability"], st["no_route"]))
    print()
    print("WHICH ELEMENT THE PARSE DROPPED  (evidence, across WORDING+UNPARSED)")
    print("-" * 96)
    overall = Counter()
    for sid, st in shapes.items():
        for c, k in st["causes"].items():
            overall[c] += k
    for c, k in overall.most_common():
        print("  %-34s %3d" % (c, k))
    print()
    print("  per shape:")
    for sid, st in shapes.items():
        if st["causes"]:
            print("    %-4s %s" % (sid, dict(st["causes"])))
    print()
    print("EVERY RECOGNITION FAILURE  (WORDING and UNPARSED)")
    print("-" * 96)
    for r in rows:
        if r["verdict"] in (WORDING, UNPARSED):
            print("  %-9s %-4s %-52s %s"
                  % (r["verdict"], r["shape"], r["question"][:52], r["cause"]))
            print("            resolved=%s route=%s named_back=%s"
                  % (r["resolved"], r["route"], r["named_back"]))
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--book", default="alderbridge",
                    choices=("alderbridge", "kestrelmoor"))
    ap.add_argument("--bank", default="combined",
                    choices=("declared", "widened", "combined"))
    ap.add_argument("--llm", action="store_true")
    ap.add_argument("--json", metavar="PATH")
    args = ap.parse_args(argv)
    result = run(args.book, args.bank, args.llm)
    if args.json:
        Path(args.json).write_text(json.dumps(result, indent=2, default=str),
                                   encoding="utf-8")
    return report(result)


if __name__ == "__main__":
    raise SystemExit(main())
