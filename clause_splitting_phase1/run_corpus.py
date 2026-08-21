#!/usr/bin/env python3
"""clause_splitting_phase1/run_corpus.py

Corpus validation — §4.2. Runs the rules read-only over every question in the
banks and reports what they cannot classify.

    python -m clause_splitting_phase1.run_corpus --out docs/…report.md

Read-only. Parses questions. Executes nothing. Modifies nothing.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

import yaml

_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from clause_splitting_phase1.reference_splitter import ReferenceSplitter  # noqa: E402
from clause_splitting_phase1.resolve import resolve  # noqa: E402
from clause_splitting_phase1.span_model import SPAN_NAMES, UNRESOLVABLE  # noqa: E402
from clause_splitting_phase1.trees import load_semantics, rc_tree  # noqa: E402

_BANKS = {
    "ere_mi_calibration_250": _REPO_ROOT / "config" / "mi" / "golden_questions" / "ere_mi_calibration_250.yaml",
    "ere_mi_questions": _REPO_ROOT / "config" / "mi" / "golden_questions" / "ere_mi_questions.yaml",
}


def load_corpus(include_generated: bool = True) -> List[Dict[str, Any]]:
    out = []
    for bank, path in _BANKS.items():
        doc = yaml.safe_load(path.read_text(encoding="utf-8"))
        for q in doc.get("questions", []):
            out.append({
                "bank": bank,
                "id": q.get("id"),
                "category": q.get("category"),
                "question": q.get("question"),
                "known_gap": q.get("known_gap"),
            })
    if include_generated:
        # The registry-driven harness. Its phrasings are MACHINE-GENERATED from
        # registry names, so it is reported separately: a rule set that reads
        # registry names back to itself proves nothing about client wording.
        from mi_agent.mi_query_harness import (build_fixture, generate_cases,
                                               probe_usable_dimensions)
        from clause_splitting_phase1.trees import load_semantics as _ls
        sem = _ls()
        df = build_fixture()
        for c in generate_cases(df, sem, usable_dims=probe_usable_dimensions(df, sem)):
            out.append({"bank": "generated_harness", "id": c.id,
                        "category": c.kind, "question": c.query,
                        "known_gap": None})
    return out


def _norm_wording(text: str) -> str:
    """Group unmatched wording by SHAPE, not by the values inside it, so the
    report lists missing rules rather than listing every question again."""
    t = re.sub(r"[£$€]?\d[\d,.]*\s*(?:k|m|bn|mm|%)?", "<n>", text.lower())
    t = re.sub(r"\s+", " ", t).strip(" ,.?")
    words = t.split()
    return " ".join(words[:3]) if len(words) > 3 else t


def analyse(limit: int = 0, include_generated: bool = True) -> Dict[str, Any]:
    corpus = load_corpus(include_generated=include_generated)
    if limit:
        corpus = corpus[:limit]
    sem = load_semantics()
    splitter = ReferenceSplitter()
    rc = rc_tree(sem)

    rules_doc = yaml.safe_load(
        (_HERE / "rules" / "clause_rules.yaml").read_text(encoding="utf-8"))
    rules = {r["id"]: r for r in rules_doc["rules"]}

    unresolvable_rows: List[Dict[str, Any]] = []
    competition_rows: List[Dict[str, Any]] = []
    disagreements: List[Dict[str, Any]] = []
    rule_fire_counts: Counter = Counter()
    as_per: List[Dict[str, str]] = []
    operation_empty: List[Dict[str, Any]] = []

    for case in corpus:
        q = case["question"]
        if not q:
            continue
        if re.search(r"\bas per\b", q, re.I):
            as_per.append({"id": case["id"], "question": q})

        split = splitter.split(q)
        spans = resolve(split, sem)
        rc_spans = rc(q)

        for rid in set(split.fired):
            rule_fire_counts[rid] += 1

        # (1) unresolvable spans, grouped by the unmatched wording
        for name in SPAN_NAMES:
            sp = getattr(spans, name)
            if sp.state == UNRESOLVABLE and sp.unmatched:
                unresolvable_rows.append({
                    "id": case["id"], "bank": case["bank"], "question": q,
                    "span": name, "unmatched": sp.unmatched,
                    "bank_kind": ("generated" if case["bank"] == "generated_harness"
                                  else "natural"),
                    "shape": _norm_wording(sp.unmatched),
                })
        for residue in spans.unresolved_residue:
            unresolvable_rows.append({
                "id": case["id"], "bank": case["bank"], "question": q,
                "span": "residue(rule 35)", "unmatched": residue,
                "bank_kind": ("generated" if case["bank"] == "generated_harness"
                              else "natural"),
                "shape": _norm_wording(residue),
            })
        if not spans.concepts_of("operation"):
            operation_empty.append({"id": case["id"], "question": q,
                                    "bank_kind": ("generated" if case["bank"] == "generated_harness"
                                                  else "natural")})

        # (2) competing openers
        for loser, winner, wording, precedence in split.competitions:
            competition_rows.append({
                "id": case["id"], "question": q, "wording": wording,
                "loser": loser, "winner": winner, "precedence_rule": precedence,
            })

        # (3) disagreement against the existing parser, span by span
        for name in ("subject", "grouping", "filter", "target"):
            a = spans.concepts_of(name)
            b = rc_spans.concepts_of(name)
            if a == b:
                continue
            disagreements.append({
                "id": case["id"], "bank": case["bank"], "question": q,
                "span": name, "splitter": a, "rc": b,
                "class": _classify(name, a, b, spans, rc_spans),
                "bank_kind": ("generated" if case["bank"] == "generated_harness"
                              else "natural"),
            })

    return {
        "corpus_size": len(corpus),
        "banks": dict(Counter(c["bank"] for c in corpus)),
        "unresolvable": unresolvable_rows,
        "competitions": competition_rows,
        "disagreements": disagreements,
        "rule_fire_counts": dict(rule_fire_counts),
        "as_per": as_per,
        "operation_empty": operation_empty,
        "rules": rules,
    }


def _classify(span: str, splitter: List[str], rc: List[str],
              s_spans, rc_spans) -> str:
    """Every disagreement is either a defect the splitter found or a rule it
    is missing. Bucket mechanically; the report classifies the buckets."""
    s_all = {n: set(s_spans.concepts_of(n)) for n in SPAN_NAMES}
    r_all = {n: set(rc_spans.concepts_of(n)) for n in SPAN_NAMES}
    moved = set(rc) - set(splitter)
    for other in SPAN_NAMES:
        if other == span:
            continue
        if moved and moved & s_all[other]:
            return "span_moved"          # same concept, different span
    moved_back = set(splitter) - set(rc)
    for other in SPAN_NAMES:
        if other == span:
            continue
        if moved_back and moved_back & r_all[other]:
            return "span_moved"
    if splitter and not rc:
        return "splitter_only"
    if rc and not splitter:
        return "rc_only"
    return "different_concept"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--json", type=Path, default=None)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--no-generated", action="store_true",
                    help="natural-language banks only")
    args = ap.parse_args(argv)

    data = analyse(limit=args.limit, include_generated=not args.no_generated)
    print("corpus: %d questions across %s" % (data["corpus_size"], data["banks"]))
    print("unresolvable rows: %d over %d distinct shapes"
          % (len(data["unresolvable"]),
             len({r["shape"] for r in data["unresolvable"]})))
    print("competitions: %d" % len(data["competitions"]))
    print("disagreements: %d  %s"
          % (len(data["disagreements"]),
             dict(Counter(d["class"] for d in data["disagreements"]))))
    print("'as per' occurrences: %d" % len(data["as_per"]))
    print("questions with an EMPTY operation span: %d" % len(data["operation_empty"]))

    top = Counter(r["shape"] for r in data["unresolvable"]).most_common(25)
    print("\ntop unmatched shapes:")
    for shape, n in top:
        print("  %5d  %s" % (n, shape))

    print("\ncompetitions by precedence rule:")
    for rule, n in Counter(c["precedence_rule"] for c in data["competitions"]).most_common():
        print("  rule %-3s %5d" % (rule, n))

    if args.json:
        args.json.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")
        print("\nwrote %s" % args.json)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
