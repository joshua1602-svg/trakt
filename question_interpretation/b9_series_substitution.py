#!/usr/bin/env python3
"""question_interpretation/b9_series_substitution.py

B9 — how many questions ask for a SERIES and receive a POINT?

"show me daily funded balance" returns a single whole-book figure. The question
asked for a run of values over time; the answer is one number, presented as the
answer, with nothing said. That is a substitution, and the most complete one on
the time axis: no time dimension survives at all.

Ruling on it means deciding that a point-in-time answer to a series question is
a substitution — which is defensible and would refuse a class. So the class is
measured before it is ruled on: every question on every available surface,
through the shipped service path, counted.

A question is IN the class when all three hold:

  * it names a time grain (`requested_unit`) or a series word ("trend", "over
    time", "evolution", "movement"), so it asked for a series;
  * the route that answered publishes no series — `route_time_grain` is None,
    which is exactly the condition under which `time_axis_disclosure` declines
    to raise a facet;
  * the answer SHIPPED (`ok`), so a figure was presented rather than refused.

The third clause matters: a question that already refuses is not in the class,
because nothing was substituted. Only shipped answers can carry a substitution.

    python -m question_interpretation.b9_series_substitution
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import warnings
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List

warnings.simplefilter("ignore")

_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parent
_EVIDENCE = _REPO_ROOT / "due_diligence" / "evidence" / "analytical_intent_v1"
for p in (str(_REPO_ROOT), str(_EVIDENCE)):
    if p not in sys.path:
        sys.path.insert(0, p)

#: Series words that ask for a run of values without naming a grain.
_SERIES_RE = re.compile(
    r"\bover time\b|\btrend(?:s|ing)?\b|\bevolution\b|\bevolve\b"
    r"|\bmovement\b|\bmoved\b|\bmoving\b|\bmonth on month\b|\bmonth-on-month\b"
    r"|\bover the (?:last|past)\b|\bsince\b|\bhistor(?:y|ical)\b"
    r"|\bdevelop(?:ing|ed|ment)\b|\bchanged?\b|\bchanging\b", re.I)


#: Compounds where a unit word names a DIMENSION, not a reporting grain.
#: "Show balance by origination year" is a bar chart by vintage cohort and is
#: correctly answered as one; the year is part of the field's name. The same
#: shape as "days in arrears", which is why `day` matches only grain
#: constructions. Excluded from the class rather than counted and then argued
#: away — a number that needs a caveat to be true is the wrong number.
_DIMENSION_COMPOUND_RE = re.compile(
    r"\borigination (?:year|month|quarter|date)\b|\bvintage year\b"
    r"|\bmaturity (?:year|month|date)\b|\byear of origination\b"
    # "over 80 years old" is an AGE. `year` has the same bare-noun defect `day`
    # had — a unit word that means a duration or an attribute in this domain and
    # a reporting level in others — except that `year` is already live and `day`
    # was caught before it shipped. Recorded as a finding, not fixed here: the
    # owner's `year` pattern is unchanged by this instrument.
    r"|\byears? old\b|\byears? of age\b|\baged? \d+\b", re.I)

#: Answers that shipped but state plainly that the data is absent. Nothing was
#: substituted — "No reporting periods are available to build a pipeline amount
#: trend" is the honest empty answer, not a point passed off as a series.
_DATA_ABSENT_RE = re.compile(
    r"\bno (?:reporting periods|weekly|governed)\b"
    r"|\bare available\b|\bavailable yet\b|\bnot available\b", re.I)

#: A forward-looking question answered in the present tense. "Likely to breach
#: over the next quarter" answered with today's limit status IS a substitution,
#: and a real one — but of the PROJECTION class, which has its own facet kind
#: and its own rule. Counted separately so B9 is not decided on it.
_FORWARD_RE = re.compile(
    r"\blikely to\b|\bdo we expect\b|\bat risk of\b|\bwill\b|\bforecast\b"
    r"|\bproject(?:ed|ion)?\b|\bnext (?:month|quarter|year)\b", re.I)


def asks_for_a_series(question: str) -> Dict[str, Any]:
    from mi_agent import period_request as P
    if _DIMENSION_COMPOUND_RE.search(question or ""):
        return {"unit": None, "series_words": False, "asks": False,
                "excluded": "the unit word names a dimension, not a grain"}
    unit = P.requested_unit(question)
    words = bool(_SERIES_RE.search(question or ""))
    return {"unit": unit, "series_words": words, "asks": bool(unit or words),
            "excluded": None}


def _payload_shape(res: Dict[str, Any]) -> str:
    """What the answer actually carries: a run, two dated points, or one point.

    Banded from the PAYLOAD, not the prose. A two-point dated comparison —
    "£1.36bn → £1.39bn" across two governed snapshots — is not a series, but it
    is not a whole-book point either, and lumping the two together would inflate
    the class with answers that did honour the question.
    """
    periods = set()
    for artifact in res.get("artifacts") or []:
        for row in (artifact.get("rows") or []):
            if isinstance(row, dict):
                key = row.get("period") or row.get("reporting_date") or row.get("date")
                if key is not None:
                    periods.add(str(key))
    if len(periods) >= 3:
        return "series"
    if len(periods) == 2:
        return "two dated points"
    text = str(res.get("answer") or "")
    if re.search(r"\d{4}-\d{2}-\d{2}.*(?:against|→|->|vs\b)", text) or \
            re.search(r"(?:against|→|->)\s*\d", text):
        return "two dated points"
    return "one point"


def scan() -> Dict[str, Any]:
    from demo_platform import config as cfg
    os.environ.update(cfg.mi_env(period_role="current"))
    os.environ["MI_AGENT_LLM_PARSER"] = "off"
    os.environ["MI_AGENT_LLM_ENABLED"] = "0"
    import logging
    logging.disable(logging.WARNING)

    from mi_agent import execution_receipt as R
    from mi_agent_api.mi_service import MiQueryRequest, execute_governed_mi_query
    from trakt_core.context import ExecutionContext
    from question_interpretation.lexical_decisions import corpus

    ctx = ExecutionContext.for_internal(cfg.CLIENT_ID)
    rows: List[Dict[str, Any]] = []
    for case in corpus():
        question = case["question"]
        read = asks_for_a_series(question)
        if not read["asks"]:
            continue
        try:
            res = (execute_governed_mi_query(
                MiQueryRequest(question=question), ctx).result or {})
        except Exception as exc:                              # pragma: no cover
            rows.append({"id": case["id"], "surface": case["surface"],
                         "question": question, "error": str(exc)[:120]})
            continue
        meta = res.get("metadata") or {}
        route = meta.get("route")
        publishes = R.route_time_grain(route)
        shipped = bool(res.get("ok"))
        shape = _payload_shape(res) if shipped else "refused"
        answer_text = str(res.get("answer") or "")
        data_absent = bool(shipped and _DATA_ABSENT_RE.search(answer_text))
        forward = bool(_FORWARD_RE.search(question))
        rows.append({
            "id": case["id"], "surface": case["surface"], "question": question,
            "unit": read["unit"], "series_words": read["series_words"],
            "route": route or "(none — point-in-time)",
            "route_publishes_series": bool(publishes),
            "shipped": shipped, "payload": shape,
            "data_absent": data_absent, "forward_looking": forward,
            # THE HARD CORE: a series was asked for, ONE POINT was shipped, and
            # nothing was said. An answer that states the data is absent
            # substituted nothing; a forward-looking question answered in the
            # present tense is a substitution of the PROJECTION class, which has
            # its own kind and its own rule, and B9 must not be decided on it.
            "in_class": bool(shipped and shape == "one point"
                             and not data_absent and not forward),
            "forward_substitution": bool(shipped and shape == "one point"
                                         and not data_absent and forward),
            "partial": bool(shipped and shape == "two dated points"),
            "answer": str(res.get("answer") or res.get("error") or "")[:140],
        })

    in_class = [r for r in rows if r.get("in_class")]
    return {
        "asked_for_a_series": len(rows),
        "by_payload": dict(Counter(r.get("payload") for r in rows)),
        "two_dated_points": [r for r in rows if r.get("partial")],
        "honest_data_absence": [r for r in rows if r.get("data_absent")],
        "forward_substitutions": [r for r in rows
                                  if r.get("forward_substitution")],
        "in_class": in_class,
        "by_surface": dict(Counter(r["surface"] for r in in_class)),
        "by_route": dict(Counter(r["route"] for r in in_class)),
        "named_a_grain": sum(1 for r in in_class if r.get("unit")),
        "series_words_only": sum(1 for r in in_class if not r.get("unit")),
        "rows": rows,
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--json", type=Path, default=None)
    ap.add_argument("--show", type=int, default=25)
    args = ap.parse_args(argv)

    result = scan()
    print("=" * 84)
    print("B9 — questions that ask for a SERIES and receive a POINT")
    print("=" * 84)
    print("asked for a series (grain or series words)  %4d"
          % result["asked_for_a_series"])
    print("what the payload carried: %s" % result["by_payload"])
    print("  two dated points (a comparison, not a series) %4d"
          % len(result["two_dated_points"]))
    print("  shipped, but stated the data is absent        %4d"
          % len(result["honest_data_absence"]))
    print("  forward-looking, answered in the present      %4d   (projection class)"
          % len(result["forward_substitutions"]))
    print("OF THOSE, IN THE CLASS                     %4d"
          % len(result["in_class"]))
    print("   named a time grain                      %4d" % result["named_a_grain"])
    print("   series words only, no grain             %4d" % result["series_words_only"])
    print("\nby surface: %s" % result["by_surface"])
    print("by route  : %s" % result["by_route"])
    print("\nthe class, %d shown:" % min(args.show, len(result["in_class"])))
    for r in result["in_class"][:args.show]:
        print("   %-18s unit=%-6s %s" % (r["id"], r["unit"], r["question"][:56]))
        print("      %s" % r["answer"][:110].replace("\n", " "))

    if args.json:
        args.json.write_text(json.dumps(result, indent=2, default=str),
                             encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
