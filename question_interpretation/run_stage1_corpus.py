#!/usr/bin/env python3
"""question_interpretation/run_stage1_corpus.py — Stage 1 deliverable.

Runs every question in the corpus through the read-only projection and reports
what does not fit: slots no existing interpreter can fill, and slots two
interpreters fill inconsistently.

    python -m question_interpretation.run_stage1_corpus --json out.json

Read-only. Parses questions; executes nothing; changes no production code.

Surfaces:
    ere_mi_calibration_250   252   the calibration bank
    ere_mi_questions         350   the ERE golden library
    generated_harness        249   registry-driven, machine-generated phrasings
    nl_robustness_44          88   44 variations x 2 books — the second surface
                                   standing condition 1 requires
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List

import yaml

_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

_REAL_TAPE = (_REPO_ROOT / "demo_platform" / "workspace" / "store" / "processed"
              / "platform" / "alderbridge" / "2026-06-30"
              / "platform_canonical_typed.csv")
_BANKS = {
    "ere_mi_calibration_250": _REPO_ROOT / "config" / "mi" / "golden_questions" / "ere_mi_calibration_250.yaml",
    "ere_mi_questions": _REPO_ROOT / "config" / "mi" / "golden_questions" / "ere_mi_questions.yaml",
}


def load_corpus() -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for bank, path in _BANKS.items():
        doc = yaml.safe_load(path.read_text(encoding="utf-8"))
        for q in doc.get("questions", []):
            if q.get("question"):
                out.append({"surface": bank, "id": q.get("id"),
                            "question": q["question"]})
    from mi_agent.mi_query_harness import (build_fixture, generate_cases,
                                           probe_usable_dimensions)
    from mi_agent.mi_query_validator import load_mi_semantics
    sem = load_mi_semantics(_REPO_ROOT / "mi_agent" / "mi_semantics_field_registry.yaml")
    synth = build_fixture()
    for c in generate_cases(synth, sem, usable_dims=probe_usable_dimensions(synth, sem)):
        out.append({"surface": "generated_harness", "id": c.id, "question": c.query})

    sys.path.insert(0, str(_REPO_ROOT / "due_diligence" / "evidence" / "analytical_intent_v1"))
    import nl_bank
    for book in ("alderbridge", "kestrelmoor"):
        for row in nl_bank.bank(book):
            out.append({"surface": "nl_robustness_44",
                        "id": "%s/%s.%s" % (book, row["intent"], row["variation"]),
                        "question": row["question"]})
    return out


def analyse() -> Dict[str, Any]:
    from mi_agent.mi_calibration import default_bank_frame
    from mi_agent.mi_query_validator import load_mi_semantics
    from question_interpretation.projection import project

    sem = load_mi_semantics(_REPO_ROOT / "mi_agent" / "mi_semantics_field_registry.yaml")
    df = default_bank_frame(_REAL_TAPE)

    rows: List[Dict[str, Any]] = []
    for case in load_corpus():
        qi = project(case["question"], semantics=sem, frame=df)
        d = qi.as_dict()
        rows.append({
            "surface": case["surface"], "id": case["id"],
            "question": case["question"],
            "operation": d["operation"]["type"],
            "operation_state": d["operation"]["state"],
            "subject": d["subject"]["candidate_concept"],
            "subject_state": d["subject"]["state"],
            "dimension_roles": [x["role"] for x in d["dimensions"]],
            "dimensions": [(x["raw_text"], x["role"], x["candidate_concept"])
                           for x in d["dimensions"]],
            "filters": [(x["raw_text"], x["operator"], x["source"])
                        for x in d["filters"]],
            "filters_without_field": sum(
                1 for x in d["filters"] if x["source"].startswith("facet.")),
            "filters_without_text": sum(
                1 for x in d["filters"] if x["source"].startswith("parser.")),
            "grain": d["time"]["grain"],
            "trend_window_state": d["time"]["trend_window"]["state"],
            "target": d["target"]["value"],
            "target_state": d["target"]["state"],
            "spans_missing": sum(1 for x in d["dimensions"] if x["span"] is None)
                             + sum(1 for x in d["filters"] if x["span"] is None),
            "notes": d["notes"],
        })
    return {"rows": rows}


_NOTE_BUCKETS = (
    ("operation: no source supplies it", "operation — no interpreter supplies it"),
    ("subject: no source supplies it", "subject — no interpreter supplies it"),
    ("no role from any source", "dimension role — named but no role available"),
    ("no field from the facet layer", "filter — clause identified, field not"),
    ("no raw text", "filter/dimension — field known, wording not"),
    ("NOT carried on the spec", "time grain — read, but not carried"),
    ("operation DISAGREEMENT", "operation — two interpreters disagree"),
    ("configured sense present", "target — configured sense unsupplied"),
    ("outside the controlled vocabulary", "time grain — outside the vocabulary"),
)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--json", type=Path, default=None)
    args = ap.parse_args(argv)

    data = analyse()
    rows = data["rows"]
    by_surface = Counter(r["surface"] for r in rows)
    print("corpus: %d questions" % len(rows))
    for s, n in by_surface.most_common():
        print("   %-26s %4d" % (s, n))

    print("\n--- slots no existing interpreter can fill ---")
    buckets: Dict[str, int] = defaultdict(int)
    for r in rows:
        for note in r["notes"]:
            for needle, label in _NOTE_BUCKETS:
                if needle in note:
                    buckets[label] += 1
                    break
    for label, n in sorted(buckets.items(), key=lambda t: -t[1]):
        print("   %-46s %4d" % (label, n))

    print("\n--- slot fill rates ---")
    tot = len(rows)
    print("   operation filled       %4d / %d" % (sum(1 for r in rows if r["operation"]), tot))
    print("   subject filled         %4d / %d" % (sum(1 for r in rows if r["subject"]), tot))
    print("   any dimension          %4d / %d" % (sum(1 for r in rows if r["dimensions"]), tot))
    print("   any filter             %4d / %d" % (sum(1 for r in rows if r["filters"]), tot))
    print("   time grain             %4d / %d" % (sum(1 for r in rows if r["grain"]), tot))
    print("   target                 %4d / %d" % (sum(1 for r in rows if r["target"]), tot))

    unresolved = [r for r in rows if "unresolved" in r["dimension_roles"]]
    print("\n--- dimension role ---")
    print("   questions with an UNRESOLVED role  %4d" % len(unresolved))
    print("   roles across the corpus            %s"
          % dict(Counter(x for r in rows for x in r["dimension_roles"])))

    print("\n--- spans ---")
    print("   claims with no recoverable span    %4d"
          % sum(r["spans_missing"] for r in rows))

    if args.json:
        args.json.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")
        print("\nwrote %s" % args.json)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
