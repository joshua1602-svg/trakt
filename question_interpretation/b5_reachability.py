#!/usr/bin/env python3
"""question_interpretation/b5_reachability.py

Is B5 reachable? Run this before and after any change that alters which facets
are built or how they are labelled.

B5: `_analytical_population_satisfies` derives the value it wants by splitting
the facet's label on its field name. Where the label does not contain the field
name the value comes out empty, and the literal comparison then accepts ANY
predicate naming that field — so a facet for FRONT BOOK is accepted against a
declared `seasoning_segment = Back Book`.

It is unreachable today only because of how the receipt builds these labels:

    label     = f"the population {predicate.describe()}"
    field_key = predicate.field

and `Predicate.describe()` always leads with the field name. That is a property
of the label construction, not a property of the guard — so it holds only until
the label construction changes.

The role split is exactly such a change. This asserts the property directly:

    every KIND_POPULATION facet the receipt builds, that carries a field_key,
    has a label containing that field name.

    python -m question_interpretation.b5_reachability
"""
from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path
from typing import Any, Dict, List

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


def label_omits_its_field(facet) -> bool:
    """The B5 precondition, exactly as the guard computes it."""
    field = str(getattr(facet, "field_key", None) or "").strip().lower()
    if not field:
        return False                     # no field key: the guard exits early
    return field not in str(getattr(facet, "label", "") or "").lower()


def scan() -> Dict[str, Any]:
    """Every population facet the receipt builds across both corpora."""
    from mi_agent import execution_receipt as R
    from mi_agent import mi_calibration as CAL
    from mi_agent.llm_query_parser import _deterministic_parse
    from mi_agent.mi_query_validator import load_mi_semantics
    from question_interpretation.lexical_decisions import corpus

    semantics = load_mi_semantics(_REPO_ROOT / "mi_agent" / "mi_semantics_field_registry.yaml")
    df = CAL.default_bank_frame(_REAL_TAPE)
    columns = list(df.columns)

    offenders: List[Dict[str, Any]] = []
    population_facets = 0
    with_field = 0

    for case in corpus():
        q = case["question"]
        spec, _meta = _deterministic_parse(q, semantics, available_columns=set(columns))
        spec_dict = spec.to_dict() if hasattr(spec, "to_dict") else {
            "filters": getattr(spec, "filters", {})}

        built: List[Any] = []
        # Both routes that construct population facets.
        try:
            built.extend(R.population_facets(spec_dict, semantics))
        except AttributeError:
            pass
        dims = R.requested_dimension_terms(q, semantics, columns)
        built.extend(f for f in R.detect_requested_facets(
            q, semantics, frame=df, requested_dimensions=dims)
            if f.kind == R.KIND_POPULATION)

        for facet in built:
            if facet.kind != R.KIND_POPULATION:
                continue
            population_facets += 1
            if getattr(facet, "field_key", None):
                with_field += 1
            if label_omits_its_field(facet):
                offenders.append({"id": case["id"], "question": q,
                                  "label": facet.label,
                                  "field_key": facet.field_key})

    return {"population_facets": population_facets,
            "carrying_a_field_key": with_field,
            "offenders": offenders}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--json", type=Path, default=None)
    args = ap.parse_args(argv)

    result = scan()
    print("=" * 72)
    print("B5 REACHABILITY — can a population label omit its own field name?")
    print("=" * 72)
    print("population facets built      %4d" % result["population_facets"])
    print("  carrying a field_key       %4d" % result["carrying_a_field_key"])
    print("  LABEL OMITS ITS FIELD      %4d" % len(result["offenders"]))
    for o in result["offenders"][:15]:
        print("    %-12s field=%-24s label=%r" % (o["id"], o["field_key"], o["label"]))
    print("\n%s" % ("-" * 72))
    print("B5 UNREACHABLE — every population label names its field"
          if not result["offenders"]
          else "B5 REACHABLE — fix B5 before proceeding")

    if args.json:
        args.json.write_text(json.dumps(result, indent=2, default=str),
                             encoding="utf-8")
    return 0 if not result["offenders"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
