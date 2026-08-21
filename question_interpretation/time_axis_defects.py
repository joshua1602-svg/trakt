#!/usr/bin/env python3
"""question_interpretation/time_axis_defects.py

Two defects on the time axis, measured SEPARATELY because they look alike and
are not alike.

    COVERAGE LIMIT   the question names a window longer than the book carries.
                     "balance by month over the last 12 months" against three
                     governed snapshots. Nothing was substituted: the answer
                     covers the whole series that exists, which is all there is.
                     What the reader is owed is that the window was shorter than
                     asked. A DISCLOSURE.

    GRAIN SUBSTITUTION
                     the question names a unit the series cannot express.
                     "balance by week over the last 6 months" against month-end
                     snapshots returns a MONTHLY series, presented as the answer,
                     with `comparison_period` stamped applied. A different thing
                     was measured from the one asked for. A SUBSTITUTION, and the
                     same class honour-or-clarify closes everywhere else in this
                     programme.

Conflating them would produce one rule that is too weak for the second or too
strict for the first — disclosing a substitution is what Tranche D rejected for
periods and this programme rejected for populations, and refusing a coverage
limit would reject an answer that is the best the book can give and honestly so.

Both readings already exist and are correct. `requested_span` returns the window
and `clarification` the sentence for it; `requested_unit`/`finer_than` return the
grain and `granularity_clarification` the sentence for it. Neither reaches the
`evolution` route — `clarification` is consumed by `period_movement` and the
period-change route, `granularity_clarification` by `forecast_extrapolation`
alone. This measures the gap, it does not close it.

    python -m question_interpretation.time_axis_defects
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional

warnings.simplefilter("ignore")

_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

#: The grain the governed funded series is published at. Month-end snapshots.
BOOK_GRAIN = "month"

QUESTIONS: List[Dict[str, str]] = [
    {"id": "C1", "question": "balance by month over the last 12 months",
     "expect": "coverage", "note": "12 months asked, 3 carried"},
    {"id": "C2", "question": "funded balance over the last 6 months",
     "expect": "coverage", "note": "6 months asked, 3 carried"},
    {"id": "C3", "question": "balance by month over the last 3 months",
     "expect": "neither", "note": "3 asked, 3 carried — the boundary"},
    {"id": "C4", "question": "funded balance by month",
     "expect": "neither", "note": "no window named"},
    {"id": "S1", "question": "balance by week over the last 6 months",
     "expect": "both", "note": "weekly grain AND a 6-month window"},
    {"id": "S2", "question": "funded balance by week",
     "expect": "grain", "note": "weekly grain, no window"},
    {"id": "S3", "question": "show me daily funded balance",
     "expect": "grain", "note": "daily grain"},
    {"id": "N1", "question": "funded balance by quarter",
     "expect": "neither", "note": "coarser than the book — expressible"},
    {"id": "N2", "question": "balance by region",
     "expect": "neither", "note": "control: no time axis"},
]


def classify(question: str, available_periods: int) -> Dict[str, Any]:
    """Which of the two defects this question is exposed to, from the readers."""
    from mi_agent import period_request as P

    span = P.requested_span(question)
    unit = P.requested_unit(question)
    grain_substituted = P.finer_than(unit, BOOK_GRAIN)
    coverage_short = bool(span is not None and span.periods > available_periods)
    if coverage_short and grain_substituted:
        kind = "both"
    elif grain_substituted:
        kind = "grain"
    elif coverage_short:
        kind = "coverage"
    else:
        kind = "neither"
    return {
        "requested_span": (None if span is None
                           else {"label": span.label, "periods": span.periods,
                                 "governed": span.governed}),
        "requested_unit": unit,
        "coverage_short": coverage_short,
        "grain_substituted": grain_substituted,
        "kind": kind,
        "owed_sentence": (
            P.clarification(span, available_periods) if coverage_short and span
            else P.granularity_clarification(unit, BOOK_GRAIN,
                                             "month-end funded snapshots")
            if grain_substituted else None),
    }


def observe(question: str) -> Dict[str, Any]:
    """What the shipped service path actually returns."""
    from demo_platform import config as cfg
    os.environ.update(cfg.mi_env(period_role="current"))
    os.environ["MI_AGENT_LLM_PARSER"] = "off"
    os.environ["MI_AGENT_LLM_ENABLED"] = "0"
    import logging
    logging.disable(logging.WARNING)

    from mi_agent_api.mi_service import MiQueryRequest, execute_governed_mi_query
    from trakt_core.context import ExecutionContext

    res = (execute_governed_mi_query(
        MiQueryRequest(question=question),
        ExecutionContext.for_internal(cfg.CLIENT_ID)).result or {})
    meta = res.get("metadata") or {}
    guard = meta.get("semantic_guard") or res.get("semanticGuard") or {}
    answer = str(res.get("answer") or res.get("error") or "")
    return {
        "route": meta.get("route") or "(none — point-in-time)",
        "ok": res.get("ok"),
        "verdict": guard.get("verdict"),
        "facets": sorted((f.get("kind"), f.get("status"))
                         for f in (guard.get("facets") or [])),
        "answer": answer[:200],
        "warnings": [str(w)[:120] for w in (res.get("warnings") or [])],
    }


def _blob(observed: Dict[str, Any]) -> str:
    return " ".join([observed["answer"]] + observed["warnings"]).lower()


def _coverage_disclosed(observed: Dict[str, Any], span) -> bool:
    """Is the reader told the WINDOW THEY ASKED FOR could not be covered?

    Two false positives before this was right, both caught by reading the rows
    rather than the totals, and both worth recording because they are the same
    mistake in different clothes — a detector that matches something the output
    happens to contain rather than the thing being claimed.

      1. looked for the substring "period(s)", which "Funded balance over 3
         period(s)" satisfies. Stating what was USED is not disclosing what was
         ASKED.
      2. looked for the requested period count as a substring. "the last 6
         months" against "latest £1.96bn" matched on the 6 in the balance.

    The bar now: the output must name the requested span IN WORDS, or say in so
    many words that the window could not be reached. Both are phrases only a
    deliberate disclosure produces.
    """
    if span is None:
        return False
    blob = _blob(observed)
    return str(span["label"]).lower() in blob or "furthest back" in blob


def _grain_disclosed(observed: Dict[str, Any], unit: Optional[str]) -> bool:
    """Is the reader told the GRAIN they asked for could not be expressed?"""
    if not unit:
        return False
    blob = _blob(observed)
    return unit in blob and ("finest" in blob or "cannot" in blob
                             or "not answered" in blob)


def run(available_periods: int = 3) -> List[Dict[str, Any]]:
    rows = []
    for case in QUESTIONS:
        row = dict(case)
        row["read"] = classify(case["question"], available_periods)
        row["observed"] = observe(case["question"])
        owed = row["read"]["kind"]
        # Measured SEPARATELY even where one question carries both, because the
        # two are owed different sentences and one can be given without the
        # other.
        row["coverage_disclosed"] = (
            _coverage_disclosed(row["observed"], row["read"]["requested_span"])
            if row["read"]["coverage_short"] else None)
        row["grain_disclosed"] = (
            _grain_disclosed(row["observed"], row["read"]["requested_unit"])
            if row["read"]["grain_substituted"] else None)
        rows.append(row)
    return rows


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--json", type=Path, default=None)
    ap.add_argument("--available-periods", type=int, default=3)
    args = ap.parse_args(argv)

    rows = run(args.available_periods)
    print("=" * 86)
    print("TIME AXIS — a coverage LIMIT and a grain SUBSTITUTION, measured apart")
    print("=" * 86)
    print("book grain = %s; governed periods available = %d\n"
          % (BOOK_GRAIN, args.available_periods))
    for r in rows:
        read, obs = r["read"], r["observed"]
        flag = "" if read["kind"] == r["expect"] else "   <-- NOT AS EXPECTED"
        print("%-4s %-44s read=%-9s expected=%-9s%s"
              % (r["id"], r["question"][:44], read["kind"], r["expect"], flag))
        print("      span=%-28s unit=%-8s route=%s"
              % (read["requested_span"] and read["requested_span"]["label"],
                 read["requested_unit"], obs["route"]))
        print("      returned  ok=%-5s verdict=%-8s facets=%s"
              % (obs["ok"], obs["verdict"], obs["facets"] or "none"))
        print("      answer    %s" % obs["answer"][:130].replace("\n", " "))
        if r["coverage_disclosed"] is not None:
            print("      coverage  disclosed=%s"
                  % ("yes" if r["coverage_disclosed"] else "NO"))
        if r["grain_disclosed"] is not None:
            print("      grain     disclosed=%s"
                  % ("yes" if r["grain_disclosed"] else "NO"))
        if read["owed_sentence"]:
            print("      OWED      %s" % read["owed_sentence"][:130])
        print()

    coverage = [r for r in rows if r["read"]["coverage_short"]]
    grain = [r for r in rows if r["read"]["grain_substituted"]]
    print("-" * 86)
    print("COVERAGE LIMITS      %d, disclosed %d   (a disclosure is owed)"
          % (len(coverage), sum(1 for r in coverage if r["coverage_disclosed"])))
    print("GRAIN SUBSTITUTIONS  %d, disclosed %d   (honour-or-clarify is owed)"
          % (len(grain), sum(1 for r in grain if r["grain_disclosed"])))
    print("misread by the classifier: %s"
          % ([r["id"] for r in rows if r["read"]["kind"] != r["expect"]] or "none"))

    if args.json:
        args.json.write_text(json.dumps(rows, indent=2, default=str),
                             encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
