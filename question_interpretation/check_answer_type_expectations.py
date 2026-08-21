#!/usr/bin/env python3
"""question_interpretation/check_answer_type_expectations.py

A bounded pre-Stage-2 check, run once and reported.

All 252 calibration expectations carry `expected_answer_type`, and those values
were DERIVED from `answer_type.asked()` — a classifier that disagrees with the
parser on 46 questions and that no production path calls. The bank grades via
`of_measure`/`satisfies` and passes. But an expectation authored from a reading
production never produces would pass anyway, which is the right-for-wrong-reason
class that has caught this programme four times.

Three questions, answered per case:

  1. Does `asked(question)` still equal the stored `expected_answer_type`?
     Where it does not, the stored value was overridden or has gone stale.
  2. Is the answer-type check REACHED at all? `evaluate_case` returns early for
     parse_only, refuse/clarify, missing-required-field and not-ok cases, so a
     stored expectation on one of those is never evaluated by anything.
  3. Where it IS reached, does the case pass STRICTLY (observed == expected) or
     only through `_SATISFIES` permissiveness (RATE accepts CURRENCY; DATE
     accepts ANY/CURRENCY/COUNT; MIXED accepts almost anything; ANY and NONE
     accept everything)?

Read-only. Changes nothing.
"""
from __future__ import annotations

import argparse
import json
import sys
import warnings
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List

warnings.simplefilter("ignore")

_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

_REAL_TAPE = (_REPO_ROOT / "demo_platform" / "workspace" / "store" / "processed"
              / "platform" / "alderbridge" / "2026-06-30"
              / "platform_canonical_typed.csv")

#: Expectations that `_SATISFIES` lets a DIFFERENT observed type satisfy.
_PERMISSIVE = {"rate", "date", "mixed", "any", "none"}

#: parser aggregation -> the operation class it implies (same mapping the
#: Stage 1 projection uses).
_AGG_TO_OP = {"count": "count", "count_distinct": "count", "sum": "amount",
              "balance_sum": "amount", "avg": "average", "weighted_avg": "average",
              "median": "average", "distribution": "neutral", "loan_level": "neutral"}
#: answer_type vocabulary -> the same operation classes, where they mean the same.
_ASKED_TO_OP = {"count": "count", "currency": "amount", "rate": "average"}


def analyse() -> Dict[str, Any]:
    from mi_agent import answer_type as AT
    from mi_agent import mi_calibration as CAL
    from mi_agent.llm_query_parser import _deterministic_parse
    from mi_agent.mi_query_validator import load_mi_semantics

    semantics = load_mi_semantics(_REPO_ROOT / "mi_agent" / "mi_semantics_field_registry.yaml")
    df = CAL.default_bank_frame(_REAL_TAPE)
    cases = CAL.load_bank()

    rows: List[Dict[str, Any]] = []
    for case in cases:
        q = case["question"]
        expected = case.get("expected_answer_type")
        asked = AT.asked(q)

        # Is this case in the 46-question disagreement set?
        spec_parse, _ = _deterministic_parse(q, semantics)
        parser_op = _AGG_TO_OP.get(spec_parse.aggregation)
        asked_op = _ASKED_TO_OP.get(asked)
        in_46 = bool(parser_op and asked_op and parser_op != asked_op)

        row: Dict[str, Any] = {
            "id": case["id"], "category": case["category"], "question": q,
            "expected_answer_type": expected,
            "asked_now": asked,
            "derivation_matches": (asked == expected),
            "in_46_disagreement": in_46,
            "expected_status": case.get("expected_status", "answer"),
            "execution": case.get("execution", "full"),
            "known_gap": bool(case.get("known_gap")),
        }

        # Does the answer-type check get REACHED? Mirror evaluate_case's gates.
        reached = True
        why_not = None
        if row["execution"] == "parse_only":
            reached, why_not = False, "parse_only returns before the type check"
        elif CAL._absent_required_fields(case, df, semantics):
            reached, why_not = False, "required field absent on this book"
        elif row["expected_status"] in ("refuse", "clarify"):
            reached, why_not = False, "refuse/clarify returns before the type check"
        elif not expected:
            reached, why_not = False, "no expected_answer_type declared"

        observed = None
        if reached:
            res = CAL._run(q, df, semantics, False)
            if not res.get("ok"):
                reached, why_not = False, "not ok — returns before the type check"
            else:
                spec = res.get("spec") or {}
                observed = AT.of_measure(spec.get("metric"),
                                         spec.get("aggregation"), semantics)

        row["type_check_reached"] = reached
        row["type_check_skipped_because"] = why_not
        row["observed_answer_type"] = observed
        if reached and observed is not None:
            row["passes_strictly"] = (observed == expected)
            row["passes_via_permissiveness"] = (
                observed != expected and AT.satisfies(expected, observed))
            row["expectation_is_permissive"] = expected in _PERMISSIVE
        else:
            row["passes_strictly"] = None
            row["passes_via_permissiveness"] = None
            row["expectation_is_permissive"] = (expected in _PERMISSIVE) if expected else None
        rows.append(row)
    return {"rows": rows}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--json", type=Path, default=None)
    args = ap.parse_args(argv)

    data = analyse()
    rows = data["rows"]
    n = len(rows)
    print("calibration cases: %d\n" % n)

    stale = [r for r in rows if not r["derivation_matches"]]
    print("1. stored expectation vs asked() today")
    print("   agree                    %3d" % (n - len(stale)))
    print("   DISAGREE                 %3d" % len(stale))

    reached = [r for r in rows if r["type_check_reached"]]
    skipped = [r for r in rows if not r["type_check_reached"]]
    print("\n2. is the answer-type check reached?")
    print("   reached                  %3d" % len(reached))
    print("   never evaluated          %3d" % len(skipped))
    for why, c in Counter(r["type_check_skipped_because"] for r in skipped).most_common():
        print("      %-46s %3d" % (why, c))

    strict = [r for r in reached if r["passes_strictly"]]
    lenient = [r for r in reached if r["passes_via_permissiveness"]]
    neither = [r for r in reached
               if not r["passes_strictly"] and not r["passes_via_permissiveness"]]
    print("\n3. of the %d reached, how do they pass?" % len(reached))
    print("   strictly (observed == expected)      %3d" % len(strict))
    print("   only via _SATISFIES permissiveness   %3d" % len(lenient))
    print("   neither — would FAIL                 %3d" % len(neither))

    in46 = [r for r in rows if r["in_46_disagreement"]]
    print("\n4. calibration cases inside the 46-question disagreement set: %d" % len(in46))
    if in46:
        print("   of those: reached=%d  never evaluated=%d"
              % (sum(1 for r in in46 if r["type_check_reached"]),
                 sum(1 for r in in46 if not r["type_check_reached"])))
        print("   of the reached: strict=%d  permissive-only=%d"
              % (sum(1 for r in in46 if r["passes_strictly"]),
                 sum(1 for r in in46 if r["passes_via_permissiveness"])))

    findings = lenient + neither + [r for r in in46 if r["passes_via_permissiveness"]]
    print("\n%s" % ("-" * 70))
    if lenient or neither or stale:
        print("FINDINGS PRESENT — see the report. Stage 2 does not start.")
    else:
        print("No case passes for a reason other than the one its expectation states.")

    if args.json:
        args.json.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")
        print("wrote %s" % args.json)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
