#!/usr/bin/env python3
"""Do reworded questions get the bank question's answer? (P0 design §25)

Owner direction, 2026-09-30: fixes "MUST be supportive of other natural
language variants". The held-out variants
(`config/mi/golden_questions/holdout_variants_20260930b.yaml`) each ask
exactly what one bank question asks, in other words. So the test is
paraphrase invariance: a variant must reach the same outcome, by the same path,
with the same headline figure as its bank question on the same deploy.

    SAME       both answered by the same path with the same headline figure
    DIFFERENT  both answered, different headline figure       -> finding
    PATH       same outcome, but by a different path (governed vs legacy)
                                                               -> finding
    LOST       the bank question was answered, the variant not -> finding
    GAINED     the variant was answered, the bank question not -> review
    DECLINED   both declined (right for fine-to-decline; a bank finding
               already for the other buckets)

Whether a figure is RIGHT is the golden answers' job; this says whether the
wording changed it.

    python due_diligence/evidence/qb_plan_readback/score_variants.py \\
        --variants qb_variants.jsonl --bank qb_plan_rerun.jsonl
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import yaml

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
HOLDOUT = ROOT / "config/mi/golden_questions/holdout_variants_20260930b.yaml"

FINDINGS = ("DIFFERENT", "PATH", "LOST")

#: A figure as the answer standard writes one: money, a percentage, a date, a
#: month, or a count. The headline is chosen by `headline`.
_FIGURE = re.compile(
    r"£-?[\d,]+(?:\.\d+)?[mkbn]*"            # £94.1m, £866k, £1,234
    r"|-?[\d,]*\.?\d+%"                        # 12.5%
    r"|\b\d{4}-\d{2}(?:-\d{2})?\b"             # 2026-06-08, 2026-10
    r"|\b\d{1,3}(?:,\d{3})+\b|\b\d+\b")        # 1,330 / 579


def figures(answer: str) -> List[str]:
    return [m.group(0).replace(",", "") for m in _FIGURE.finditer(answer or "")]


def headline(answer: str) -> Optional[str]:
    """The figure the answer is ABOUT: the first amount or percentage, else
    the first date, else the first count. A month that only says WHICH month
    ("next month (2026-10) is £4.8m") is not the answer — the 13:49 check
    scored £4.8m and £7.8m the same because both said 2026-10 first."""
    found = figures(answer)
    for wanted in (lambda f: f.startswith("£") or f.endswith("%"),
                   lambda f: re.fullmatch(r"\d{4}-\d{2}(?:-\d{2})?", f),
                   lambda f: True):
        for figure in found:
            if wanted(figure):
                return figure
    return None


def _answered(rec: Dict[str, Any]) -> bool:
    return rec.get("outcome") == "ANSWERED"


def _path(rec: Dict[str, Any]) -> str:
    return "governed" if rec.get("served") == "NEW" else "legacy"


def verdict(variant: Dict[str, Any], bank: Optional[Dict[str, Any]]) -> str:
    if bank is None:
        return "NO_BANK_RECORD"
    if _answered(bank) and not _answered(variant):
        return "LOST"
    if _answered(variant) and not _answered(bank):
        return "GAINED"
    if not _answered(variant):
        return "DECLINED"
    if _path(variant) != _path(bank):
        return "PATH"
    head_v, head_b = headline(variant["answer"]), headline(bank["answer"])
    return "SAME" if head_v == head_b else "DIFFERENT"


def _jsonl(paths: Iterable[Path]) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for path in paths:
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rec = json.loads(line)
                out[rec["id"]] = rec          # the latest record for an id wins
    return out


def score(variants: Dict[str, Dict[str, Any]], bank: Dict[str, Dict[str, Any]],
          holdout: Path = HOLDOUT) -> Dict[str, Any]:
    rows = yaml.safe_load(holdout.read_text(encoding="utf-8"))["questions"]
    results = []
    for row in rows:
        rec = variants.get(row["id"])
        if rec is None:
            continue
        original = bank.get(row["variant_of"])
        results.append({
            "id": row["id"], "variant_of": row["variant_of"],
            "bucket": row["bucket"], "category": row["category"],
            "verdict": verdict(rec, original),
            "variant": {"question": row["question"],
                        "outcome": rec.get("outcome"), "path": _path(rec),
                        "headline": headline(rec.get("answer", "")) if _answered(rec) else None,
                        "answer": (rec.get("answer") or "")[:240]},
            "bank": None if original is None else {
                "question": original.get("question"),
                "outcome": original.get("outcome"), "path": _path(original),
                "headline": (headline(original.get("answer", ""))
                             if _answered(original) else None),
                "answer": (original.get("answer") or "")[:240]},
        })
    tally: Dict[str, int] = {}
    for r in results:
        tally[r["verdict"]] = tally.get(r["verdict"], 0) + 1
    return {"results": results, "tally": tally,
            "findings": [r for r in results if r["verdict"] in FINDINGS]}


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--variants", type=Path, nargs="+", required=True)
    ap.add_argument("--bank", type=Path, nargs="+", required=True)
    ap.add_argument("--json", type=Path, default=None)
    args = ap.parse_args(argv)
    out = score(_jsonl(args.variants), _jsonl(args.bank))
    print("VERDICTS  " + "  ".join(f"{k} {v}" for k, v in sorted(out["tally"].items())))
    for r in out["findings"]:
        v, b = r["variant"], r["bank"] or {}
        print(f"\n{r['verdict']:<9} {r['id']}  ({r['bucket']})\n"
              f"  variant: {v['question']!r} -> {v['outcome']} {v['path']} "
              f"{v['headline']}\n"
              f"  bank:    {b.get('question')!r} -> {b.get('outcome')} "
              f"{b.get('path')} {b.get('headline')}")
    if args.json:
        args.json.write_text(json.dumps(out, indent=1), encoding="utf-8")
    return 1 if out["findings"] else 0


if __name__ == "__main__":
    sys.exit(main())
