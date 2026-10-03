#!/usr/bin/env python3
"""Score a read-back bank run against the question buckets (D13).

The pass mark for the proof stage is every MUST ANSWER question answered
correctly and no question answered wrongly — not 135/135
(`qb_question_buckets.json`, owner decision D13). This reads a read-back
projection (`qb_plan_readback.json`, as `qb-plan-readback.yml` publishes it)
and states, per bucket, how many questions the governed path served, and names
every must-answer question it did not — with the reason the evidence records.

Correctness is not decided here: whether a served figure is RIGHT is checked
against the dashboard (the golden answers). This says what was served.

    python due_diligence/evidence/qb_plan_readback/score_against_buckets.py \\
        --rerun qb_plan_readback.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

HERE = Path(__file__).resolve().parent
ORDER = ("must_answer", "nice_to_have", "fine_to_decline")
TITLE = {"must_answer": "must answer", "nice_to_have": "nice to have",
         "fine_to_decline": "fine to decline / ask back"}


def score(projection: Dict[str, Any], buckets: Dict[str, Any]) -> Dict[str, Any]:
    cases = projection.get("cases") or {}
    out: Dict[str, Any] = {}
    for bucket in ORDER:
        rows: List[Dict[str, Any]] = []
        for cid in buckets["buckets"][bucket]:
            case = cases.get(cid) or {}
            summary = case.get("summary") or {}
            rows.append({
                "id": cid, "n": buckets["questions"][cid]["n"],
                "question": buckets["questions"][cid]["question"],
                "asked": bool(case.get("record_found")),
                "served": summary.get("serving_decision") == "NEW",
                "reason": summary.get("serving_reason") or "",
            })
        asked = [r for r in rows if r["asked"]]
        out[bucket] = {"questions": len(rows), "asked": len(asked),
                       "served_new": sum(r["served"] for r in asked),
                       "not_served": [r for r in asked if not r["served"]],
                       "not_asked": [r["id"] for r in rows if not r["asked"]]}
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--rerun", required=True, help="a read-back projection")
    parser.add_argument("--buckets", default=str(HERE / "qb_question_buckets.json"))
    args = parser.parse_args()
    result = score(json.loads(Path(args.rerun).read_text()),
                   json.loads(Path(args.buckets).read_text()))
    print(f"{'bucket':<28} {'questions':>9} {'asked':>6} {'new path':>9}")
    for bucket in ORDER:
        r = result[bucket]
        print(f"{TITLE[bucket]:<28} {r['questions']:>9} {r['asked']:>6} "
              f"{r['served_new']:>9}")
    print()
    for bucket in ORDER[:2]:
        missing = result[bucket]["not_served"]
        if missing:
            print(f"{TITLE[bucket].upper()} — not served by the new path ({len(missing)}):")
            for r in missing:
                print(f"  [{r['n']:>3}] {r['id']:<24} {r['reason'] or '-':<45} "
                      f"{r['question']}")
            print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
