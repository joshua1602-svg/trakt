#!/usr/bin/env python3
"""question_interpretation/lexical_decisions.py

The per-conversion instrument for Stage 3.

Each conversion moves one consumer off re-reading the raw question and onto the
carried object. The contract asks, per conversion, for "the diff in decisions
across the corpus" — so the decision each consumer makes is snapshotted here,
verbatim, before and after.

This is finer-grained than the answer-text diff on purpose. The answer diff
proves nothing a reader sees moved; this proves the FUNCTION's own output did
not move either. A conversion that changed a consumer's decision but happened
not to reach the answer would pass the first and fail this.

    python -m question_interpretation.lexical_decisions --record before.json
    python -m question_interpretation.lexical_decisions --against before.json

The three subject-side splitters, which the inventory found implemented three
times from vocabularies that agree by maintenance rather than construction:

    answer_type.subject_side          splits at `by` first, then truncates at a
                                      condition opener followed by a digit
    llm_query_parser._metric_slot     truncates only; grouping handled upstream
    execution_receipt._is_filter_subject
                                      32-character windows either side of a
                                      measure word, a different vocabulary
"""
from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path
from typing import Any, Dict, List

import yaml

warnings.simplefilter("ignore")

_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parent
_EVIDENCE = _REPO_ROOT / "due_diligence" / "evidence" / "analytical_intent_v1"
for p in (str(_REPO_ROOT), str(_EVIDENCE)):
    if p not in sys.path:
        sys.path.insert(0, p)

_BANKS = {
    "ere_mi_calibration_250": _REPO_ROOT / "config" / "mi" / "golden_questions" / "ere_mi_calibration_250.yaml",
    "ere_mi_questions": _REPO_ROOT / "config" / "mi" / "golden_questions" / "ere_mi_questions.yaml",
}


def corpus() -> List[Dict[str, str]]:
    out: List[Dict[str, str]] = []
    for bank, path in _BANKS.items():
        doc = yaml.safe_load(path.read_text(encoding="utf-8"))
        for q in doc.get("questions", []):
            if q.get("question"):
                out.append({"surface": bank, "id": q["id"], "question": q["question"]})
    import nl_bank
    for book in ("alderbridge", "kestrelmoor"):
        for row in nl_bank.bank(book):
            pair = row.get("pair")
            ident = "%s.%s" % (row["intent"], row["variation"])
            if pair:
                ident = "%s[%s]" % (ident, pair if isinstance(pair, str)
                                    else "|".join(str(x) for x in pair))
            out.append({"surface": "nl_robustness_%s" % book, "id": ident,
                        "question": row["question"]})
    return out


def decisions_for(question: str) -> Dict[str, Any]:
    """Every lexical decision Stage 3 will move, for one question."""
    from mi_agent import answer_type as AT
    from mi_agent import period_request as PR
    from mi_agent import execution_receipt as R
    from mi_agent.llm_query_parser import _metric_slot

    out: Dict[str, Any] = {
        "answer_type.subject_side": AT.subject_side(question),
        "answer_type.asked": AT.asked(question),
        "llm_query_parser._metric_slot": _metric_slot(question),
        "period_request.requested_unit": PR.requested_unit(question),
    }
    span = PR.requested_span(question)
    out["period_request.requested_span"] = (
        None if span is None else {"label": getattr(span, "label", None),
                                   "periods": getattr(span, "periods", None),
                                   "governed": getattr(span, "governed", None)})
    # _is_filter_subject is positional; probe it at every measure word the
    # receipt layer itself would find, so the record is what it actually decides.
    lowered = question.lower()
    probes = []
    for concept in R.named_measure_concepts(question) or []:
        idx = lowered.find(str(concept).lower())
        if idx >= 0:
            probes.append({"at": concept,
                           "is_filter_subject":
                               R._is_filter_subject(lowered, idx,
                                                    idx + len(str(concept)))})
    out["execution_receipt._is_filter_subject"] = probes
    return out


def collect() -> List[Dict[str, Any]]:
    return [{**case, "decisions": decisions_for(case["question"])}
            for case in corpus()]


def compare(before: List[Dict[str, Any]],
            after: List[Dict[str, Any]]) -> Dict[str, Any]:
    b = {(r["surface"], r["id"]): r for r in before}
    a = {(r["surface"], r["id"]): r for r in after}
    shared = sorted(set(b) & set(a))
    moved: List[Dict[str, Any]] = []
    per_function: Dict[str, int] = {}
    for key in shared:
        bd, ad = b[key]["decisions"], a[key]["decisions"]
        changed = {k: {"before": bd.get(k), "after": ad.get(k)}
                   for k in sorted(set(bd) | set(ad)) if bd.get(k) != ad.get(k)}
        if changed:
            moved.append({"surface": key[0], "id": key[1],
                          "question": b[key]["question"], "changed": changed})
            for k in changed:
                per_function[k] = per_function.get(k, 0) + 1
    return {"compared": len(shared), "moved": moved,
            "per_function": per_function,
            "identical": len(shared) - len(moved),
            "only_before": sorted("%s/%s" % k for k in set(b) - set(a)),
            "only_after": sorted("%s/%s" % k for k in set(a) - set(b))}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--record", type=Path)
    ap.add_argument("--against", type=Path)
    args = ap.parse_args(argv)

    records = collect()
    if args.record:
        args.record.write_text(json.dumps({"records": records}, indent=2,
                                          default=str), encoding="utf-8")
        print("recorded %d questions x %d decisions -> %s"
              % (len(records), len(records[0]["decisions"]), args.record))
        return 0
    if not args.against:
        ap.error("one of --record or --against is required")

    before = json.loads(args.against.read_text(encoding="utf-8"))["records"]
    result = compare(before, records)
    print("=" * 72)
    print("LEXICAL DECISION DIFF — per-consumer, finer than the answer diff")
    print("=" * 72)
    print("compared   %4d" % result["compared"])
    print("identical  %4d" % result["identical"])
    print("MOVED      %4d" % len(result["moved"]))
    for fn, n in sorted(result["per_function"].items(), key=lambda t: -t[1]):
        print("   %-42s %4d" % (fn, n))
    for entry in result["moved"][:20]:
        print("\n  %s/%s\n    %s" % (entry["surface"], entry["id"],
                                     entry["question"][:88]))
        for fn, move in entry["changed"].items():
            print("      %s\n        before: %r\n        after:  %r"
                  % (fn, move["before"], move["after"]))
    print("\n%s" % ("-" * 72))
    print("PASS — no lexical decision moved" if not result["moved"]
          else "MOVED — attribute every one before proceeding")
    return 0 if not result["moved"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
