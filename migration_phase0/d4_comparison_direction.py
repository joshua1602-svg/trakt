#!/usr/bin/env python3
"""migration_phase0/d4_comparison_direction.py — the D4 blast, measured.

READ-ONLY. D4 is canary I7: a two-period comparison reported from the LATER
date to the EARLIER one, so a fall is narrated as a rise.

THE SINGLE OWNER OF COMPARISON DIRECTION
----------------------------------------
Traced end to end, the chain has exactly one place where the ORDER is decided:

    llm_query_parser._compare_recognizer   -> spec.compare_periods   (ORDER SET HERE)
    projection._time                       -> time.comparison_periods (copied verbatim)
    analytical_plan.comparison_periods     -> (str, str)              (read, not reordered)
    analytical_plan.build_temporal_compare_plan
        STACK_PERIODS take=named_pair, COMPARE direction="b relative to a"
    temporal_compare.compare_periods       -> abs_delta = vb - va
    chat_routing._route_compare            -> "moved from {periodA} to {periodB}"

Everything downstream of the parser is faithful: the receipt reports what
executed and the prose reports the receipt. So the fix belongs at the ORDER,
and patching the narration to compensate would leave `absoluteDelta`,
`percentageDelta` and `direction` still inverted on the receipt while the
sentence read correctly — a worse defect than the one it hid.

    python -m migration_phase0.d4_comparison_direction [--json out.json]
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import warnings
from pathlib import Path
from typing import Any, Dict, List

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

CORPORA = ("question_interpretation/stage1_corpus.json",
           "question_interpretation/stage2_corpus.json")

#: The extra questions the corpus does not contain but the canary does, plus the
#: shapes that exercise each branch of the recogniser. Listed so the blast is
#: measured over the branch, not only over what the corpus happens to ask.
PROBES = (
    "How did the book change since last month?",
    "How did LTV change since last month?",
    "How has the book changed since last week?",
    "How did the book change compared to the prior month?",
    "Compare October and November.",
    "Compare November and October.",
    "How did the book change from October to November?",
    "How did the book change from November to October?",
    "Compare October to last month.",
)


def _questions() -> List[str]:
    out, seen = [], set()
    for f in CORPORA:
        for row in json.loads((_REPO / f).read_text())["rows"]:
            q = row.get("question") or ""
            if q and q not in seen:
                seen.add(q)
                out.append(q)
    return out


def run() -> Dict[str, Any]:
    warnings.simplefilter("ignore")
    logging.disable(logging.WARNING)
    from mi_agent.mi_query_validator import load_mi_semantics
    from mi_agent_api.data_source import semantics_path
    from mi_agent.parsed_question import ParsedQuestion
    from mi_agent import llm_query_parser as P

    semantics = load_mi_semantics(semantics_path())
    if not (semantics.get("fields") or {}):
        raise RuntimeError("D4 BLAST INVALID — governed MI semantics did not load")

    rows: List[Dict[str, Any]] = []
    for question in list(_questions()) + list(PROBES):
        spec = ParsedQuestion.parse(question, semantics).spec
        periods = list(getattr(spec, "compare_periods", None) or [])
        if len(periods) < 2:
            continue
        # LOWERCASED, because `_MONTH_NAMES` are lowercase and `_detect_periods`
        # matches case-sensitively — the recogniser hands it an already-lowered
        # question. Passing the original case made every explicit month pair
        # classify as `other`, which would have published a wrong branch table.
        lowered = question.lower()
        explicit = P._detect_periods(lowered)
        rel = next((r for r in P._RELATIVE_PERIOD_TERMS if r in lowered), None)
        # WHICH BRANCH of the recogniser produced this pair. The branch is the
        # unit of the fix, so it is the unit of the blast.
        if len(explicit) >= 2:
            branch = "explicit_pair"
        elif explicit and rel:
            branch = "explicit_plus_relative"
        elif rel:
            branch = "latest_plus_relative"
        else:
            branch = "other"
        rows.append({
            "question": question, "periods": periods, "branch": branch,
            "explicit": explicit, "relative": rel,
            "in_corpus": question in set(_questions()),
            # The defect, stated as a property rather than as a list of cases:
            # the fallback hard-codes "latest" FIRST, and "latest" is by
            # definition the closing period, so the pair opens at the close.
            "reversed": branch == "latest_plus_relative" and periods[0] == "latest",
        })
    return {"rows": rows}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--json", type=Path, default=None)
    args = ap.parse_args(argv)

    result = run()
    rows = result["rows"]
    corpus = [r for r in rows if r["in_corpus"]]
    reversed_rows = [r for r in rows if r["reversed"]]

    print("=" * 84)
    print("D4 BLAST — questions that produce an ordered comparison pair")
    print("=" * 84)
    print(f"\ntotal pairs produced : {len(rows)}  "
          f"(corpus {len(corpus)}, probes {len(rows) - len(corpus)})")
    from collections import Counter
    print("by recogniser branch : " + ", ".join(
        f"{k}={v}" for k, v in Counter(r["branch"] for r in rows).most_common()))
    print(f"\nREVERSED (opens at the closing period): {len(reversed_rows)}")
    for r in reversed_rows:
        tag = "corpus" if r["in_corpus"] else "probe "
        print(f"   [{tag}] {r['periods']}  {r['question']}")

    print("\nNOT REVERSED — the pair the reader stated, in the order stated")
    for r in rows:
        if not r["reversed"]:
            tag = "corpus" if r["in_corpus"] else "probe "
            print(f"   [{tag}] {r['branch']:<22} {r['periods']}  {r['question'][:52]}")

    if args.json:
        args.json.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
