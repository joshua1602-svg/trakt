#!/usr/bin/env python3
"""question_interpretation/unresolved_role_variants.py

Measure the three candidate defaults for an UNRESOLVED role, side by side.

`_split_named_dimension_roles` gives a named dimension the role the sentence
gave it. Where no source supplies one — the field is neither in `spec.filters`
nor on an axis — three things could be done, and they differ in what a reader
receives:

    grouping     leave it. An unapplied grouping is a SHAPE facet: the answer
                 stands, the dimension is named as not applied. A whole-book
                 figure with a disclosure attached.
    population   let it fall to the blocking side, as 32c263a did. Refuses.
    clarify      ask which was meant. No figure, and no refusal.

This runs each variant over both parse paths and reports, per variant:

  * which facets move, by kind and by field
  * how many questions block, and which
  * how many questions clarify, and which
  * whether the seasoning families hold
  * whether the 32c263a mechanism can recur — i.e. whether any GOVERNED
    population is sitting in the unresolved set and would be re-read as one

TWO LIMITATIONS, which must travel with any number quoted from here
------------------------------------------------------------------
1. The role split is INERT on this tape. The alderbridge book carries no
   `borrower_type` column, so the parser correctly drops that filter and there
   is nothing to reclassify on the production parse. The effect therefore rests
   on the UNFILTERED parse — the same corpus parsed without column filtering —
   rather than on an end-to-end run. Both are reported; do not quote the
   unfiltered figures as production behaviour.
2. The B5 scanner initially watched detection-time facets while the split
   happens at reconcile, so it would have missed exactly the facets it exists
   to guard. That is the SEVENTH instance in this programme of an instrument
   carrying the defect it was built to find. It is recorded in the standing
   rules as a pattern, not as an incident.

    python -m question_interpretation.unresolved_role_variants --json out.json
"""
from __future__ import annotations

import argparse
import collections
import json
import sys
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

warnings.simplefilter("ignore")

_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parent
_EVIDENCE = _REPO_ROOT / "due_diligence" / "evidence" / "analytical_intent_v1"
for p in (str(_REPO_ROOT), str(_EVIDENCE)):
    if p not in sys.path:
        sys.path.insert(0, p)

_REAL_TAPE = (_REPO_ROOT / "demo_platform" / "workspace" / "store" / "processed"
              / "platform" / "alderbridge" / "2026-06-30"
              / "platform_canonical_typed.csv")

VARIANTS = ("grouping", "population", "clarify")


# --------------------------------------------------------------------------- #
# Is this unresolved facet actually a GOVERNED population?
# --------------------------------------------------------------------------- #
def _unresolved_facets(facets, spec, R) -> List[Any]:
    """The set the variant switch decides the fate of."""
    if isinstance(spec, dict):
        filters = dict(spec.get("filters") or {})
        axes = list(spec.get("dimensions") or [])
        if spec.get("dimension"):
            axes.append(spec.get("dimension"))
    else:
        filters = dict(getattr(spec, "filters", None) or {})
        axes = list(getattr(spec, "dimensions", None) or [])
        if getattr(spec, "dimension", None):
            axes.append(getattr(spec, "dimension"))
    axes = {str(a) for a in axes if a}
    out = []
    for f in facets:
        field = getattr(f, "field_key", None)
        if f.kind == R.KIND_GROUPING and field and field not in filters \
                and field not in axes:
            out.append(f)
    return out


def _is_governed_window(facet, semantics, R) -> Optional[str]:
    """The governed predicate this facet's WORDING resolves to, if any.

    Deliberately the SAME route the population check uses since the governed
    comparison went in — `_governed_population_predicates`, keyed on the facet's
    wording rather than its field key. Asking the question any other way would
    make this instrument disagree with the code it is measuring.
    """
    try:
        described = R._governed_population_predicates(facet)
    except Exception:
        return None
    return "; ".join(described) if described else None


# --------------------------------------------------------------------------- #
# Per-variant scan
# --------------------------------------------------------------------------- #
def scan(variant: str, *, filtered: bool) -> Dict[str, Any]:
    from mi_agent import execution_receipt as R
    from mi_agent import mi_calibration as CAL
    from mi_agent.llm_query_parser import _deterministic_parse
    from mi_agent.mi_query_validator import load_mi_semantics
    from question_interpretation.lexical_decisions import corpus

    semantics = load_mi_semantics(
        _REPO_ROOT / "mi_agent" / "mi_semantics_field_registry.yaml")
    df = CAL.default_bank_frame(_REAL_TAPE)
    columns = list(df.columns)

    previous = R.UNRESOLVED_ROLE_DEFAULT
    R.UNRESOLVED_ROLE_DEFAULT = variant
    try:
        moved: List[Dict[str, Any]] = []
        by_field: Dict[str, int] = collections.Counter()
        governed_in_set: List[Dict[str, Any]] = []
        unresolved_total = 0
        b5_offenders: List[Dict[str, Any]] = []

        for case in corpus():
            q = case["question"]
            spec, _meta = _deterministic_parse(
                q, semantics,
                available_columns=(set(columns) if filtered else None))
            dims = R.requested_dimension_terms(q, semantics, columns)
            detected = R.detect_requested_facets(q, semantics, frame=df,
                                                 requested_dimensions=dims)
            unresolved = _unresolved_facets(detected, spec, R)
            unresolved_total += len(unresolved)
            for f in unresolved:
                by_field[f.field_key] += 1
                governed = _is_governed_window(f, semantics, R)
                if governed:
                    governed_in_set.append(
                        {"id": case["id"], "question": q, "field": f.field_key,
                         "label": f.label, "governed_as": governed})

            after = R._split_named_dimension_roles(detected, spec)
            for before_f, after_f in zip(detected, after):
                if before_f.kind != after_f.kind or before_f.label != after_f.label:
                    moved.append({"id": case["id"], "question": q,
                                  "field": before_f.field_key,
                                  "from": before_f.kind, "to": after_f.kind,
                                  "label_from": before_f.label,
                                  "label_to": after_f.label})
            for f in after:
                if f.kind == R.KIND_POPULATION and getattr(f, "field_key", None):
                    fk = str(f.field_key).lower()
                    if fk not in str(f.label or "").lower():
                        b5_offenders.append({"id": case["id"], "field": f.field_key,
                                             "label": f.label})
    finally:
        R.UNRESOLVED_ROLE_DEFAULT = previous

    return {"variant": variant, "parse": "filtered" if filtered else "unfiltered",
            "unresolved_facets": unresolved_total,
            "unresolved_by_field": dict(by_field),
            "governed_windows_in_unresolved_set": governed_in_set,
            "facets_moved": moved,
            "b5_offenders": b5_offenders}


# --------------------------------------------------------------------------- #
# End-to-end: what does the READER receive?
# --------------------------------------------------------------------------- #
def outcomes(variant: str) -> Dict[str, Any]:
    """Verdict per calibration case, on the production (filtered) path."""
    from mi_agent import execution_receipt as R
    from mi_agent import mi_calibration as CAL
    from mi_agent.mi_query_validator import load_mi_semantics

    semantics = load_mi_semantics(
        _REPO_ROOT / "mi_agent" / "mi_semantics_field_registry.yaml")
    df = CAL.default_bank_frame(_REAL_TAPE)

    previous = R.UNRESOLVED_ROLE_DEFAULT
    R.UNRESOLVED_ROLE_DEFAULT = variant
    try:
        verdicts: Dict[str, str] = {}
        answers: Dict[str, str] = {}
        for case in CAL.load_bank():
            res = CAL._run(case["question"], df, semantics, False)
            guard = res.get("semantic_guard") or {}
            verdicts[case["id"]] = str(guard.get("verdict") or "none")
            answers[case["id"]] = str(res.get("answer") or "")
    finally:
        R.UNRESOLVED_ROLE_DEFAULT = previous
    return {"verdicts": verdicts, "answers": answers,
            "counts": dict(collections.Counter(verdicts.values()))}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--json", type=Path, default=None)
    ap.add_argument("--skip-outcomes", action="store_true")
    args = ap.parse_args(argv)

    report: Dict[str, Any] = {"scans": {}, "outcomes": {}}
    for variant in VARIANTS:
        for filtered in (True, False):
            key = "%s/%s" % (variant, "filtered" if filtered else "unfiltered")
            report["scans"][key] = scan(variant, filtered=filtered)
    if not args.skip_outcomes:
        for variant in VARIANTS:
            report["outcomes"][variant] = outcomes(variant)

    print("=" * 78)
    print("UNRESOLVED-ROLE DEFAULT — three variants, side by side")
    print("=" * 78)
    for key, s in report["scans"].items():
        print("\n%-24s unresolved=%d  moved=%d  b5_offenders=%d"
              % (key, s["unresolved_facets"], len(s["facets_moved"]),
                 len(s["b5_offenders"])))
        for field, n in sorted(s["unresolved_by_field"].items(),
                               key=lambda kv: -kv[1]):
            print("      %-26s %3d" % (field, n))
        gov = s["governed_windows_in_unresolved_set"]
        if gov:
            print("      GOVERNED WINDOWS IN THE UNRESOLVED SET: %d" % len(gov))
            for g in gov[:6]:
                print("        %-12s %-24s -> %s"
                      % (g["id"], g["field"], g["governed_as"]))
    for variant, o in report["outcomes"].items():
        print("\noutcomes %-12s %s" % (variant, o["counts"]))

    base = report["outcomes"].get("grouping", {}).get("answers") or {}
    for variant, o in report["outcomes"].items():
        if variant == "grouping" or not base:
            continue
        diff = [k for k, v in o["answers"].items() if base.get(k) != v]
        print("answer text moved vs grouping: %-12s %d of %d"
              % (variant, len(diff), len(base)))
        report["outcomes"][variant]["answer_text_moved"] = diff

    if args.json:
        args.json.write_text(json.dumps(report, indent=2, default=str),
                             encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
