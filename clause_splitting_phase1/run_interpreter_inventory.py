#!/usr/bin/env python3
"""clause_splitting_phase1/run_interpreter_inventory.py

Inventory, not a design. Runs the question corpus through the interpreters that
independently read the raw sentence, and reports where they disagree.

    python -m clause_splitting_phase1.run_interpreter_inventory --json out.json

Compared, per question:

  parser    mi_agent.llm_query_parser._deterministic_parse — the spec slots
  facets    mi_agent.execution_receipt.detect_requested_facets — the facet
            layer's OWN reading, taken BEFORE reconciliation, so this is
            interpreter against interpreter rather than interpreter against
            execution
  answer    mi_agent.answer_type.asked — a third reading, "from its wording
            alone", with its own subject-side clause split

Read-only. Parses questions; executes nothing; writes nothing unless --json.
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

_BANKS = {
    "ere_mi_calibration_250": _REPO_ROOT / "config" / "mi" / "golden_questions" / "ere_mi_calibration_250.yaml",
    "ere_mi_questions": _REPO_ROOT / "config" / "mi" / "golden_questions" / "ere_mi_questions.yaml",
}
_REAL_TAPE = (_REPO_ROOT / "demo_platform" / "workspace" / "store" / "processed"
              / "platform" / "alderbridge" / "2026-06-30"
              / "platform_canonical_typed.csv")

#: Facet kinds that make a FILTER claim about the sentence.
_FILTER_KINDS = {"geographic_scope", "row_population", "threshold"}
#: Facet kinds that make a GROUPING claim.
_GROUPING_KINDS = {"grouping_dimension"}
#: Facet kinds that make a SUBJECT / measure claim.
_SUBJECT_KINDS = {"multi_measure", "unresolved_measure", "relationship"}


def load_corpus() -> List[Dict[str, Any]]:
    out = []
    for bank, path in _BANKS.items():
        doc = yaml.safe_load(path.read_text(encoding="utf-8"))
        for q in doc.get("questions", []):
            if q.get("question"):
                out.append({"bank": bank, "id": q.get("id"),
                            "category": q.get("category"),
                            "question": q["question"]})
    return out


def analyse() -> Dict[str, Any]:
    from mi_agent import answer_type as AT
    from mi_agent import execution_receipt as R
    from mi_agent.llm_query_parser import _deterministic_parse
    from mi_agent.mi_calibration import default_bank_frame
    from mi_agent.mi_query_validator import load_mi_semantics

    sem = load_mi_semantics(_REPO_ROOT / "mi_agent" / "mi_semantics_field_registry.yaml")
    df = default_bank_frame(_REAL_TAPE)
    cols = list(df.columns)

    rows: List[Dict[str, Any]] = []
    for case in load_corpus():
        q = case["question"]
        spec, _meta = _deterministic_parse(q, sem)
        dims = R.requested_dimension_terms(q, sem, cols)
        facets = R.detect_requested_facets(q, sem, frame=df, requested_dimensions=dims)

        parser_filters = sorted((spec.filters or {}).keys())
        parser_groups = sorted({d for d in ([spec.dimension] + list(spec.dimensions or []))
                                if d})
        parser_metric = spec.metric

        # A facet may identify a filter CLAUSE without resolving the FIELD it
        # binds: every `threshold` facet carries field=None. So the clause-level
        # and field-level readings are recorded separately — conflating them
        # reports the facet layer as blind to filters it can plainly see.
        facet_filter_clauses = [f.kind for f in facets if f.kind in _FILTER_KINDS]
        facet_filters_unfielded = [f.label for f in facets
                                   if f.kind in _FILTER_KINDS and not f.field_key]
        facet_filters = sorted({f.field_key for f in facets
                                if f.kind in _FILTER_KINDS and f.field_key})
        facet_groups = sorted({f.field_key for f in facets
                               if f.kind in _GROUPING_KINDS and f.field_key})
        facet_subject_claims = sorted({f.kind for f in facets if f.kind in _SUBJECT_KINDS})
        facet_kinds = sorted({f.kind for f in facets})

        # A facet layer claim of a filter/grouping that the parser did not make,
        # or the reverse, is the two interpreters reading the same sentence
        # differently.
        rows.append({
            "id": case["id"], "bank": case["bank"], "category": case["category"],
            "question": q,
            "parser_filters": parser_filters,
            "facet_filters": facet_filters,
            "facet_filter_clauses": facet_filter_clauses,
            "facet_filters_unfielded": facet_filters_unfielded,
            "parser_groups": parser_groups,
            "facet_groups": facet_groups,
            "parser_metric": parser_metric,
            "parser_aggregation": spec.aggregation,
            "facet_kinds": facet_kinds,
            "facet_subject_claims": facet_subject_claims,
            "answer_type_asked": AT.asked(q),
            "answer_type_of_result": AT.of_measure(spec.metric, spec.aggregation, sem),
            "parser_subject_side": None,
            "filter_disagree": parser_filters != facet_filters,
            "filter_clause_seen_both": bool(parser_filters) and bool(facet_filter_clauses),
            "filter_field_only_parser": bool(parser_filters) and not facet_filters,
            "group_disagree": parser_groups != facet_groups,
            "type_disagree": None,
        })
        r = rows[-1]
        asked, got = r["answer_type_asked"], r["answer_type_of_result"]
        r["type_disagree"] = (asked not in (AT.ANY, AT.NONE, AT.MIXED)
                              and got not in (AT.ANY, AT.NONE, AT.MIXED)
                              and asked != got)
    return {"rows": rows}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--json", type=Path, default=None)
    args = ap.parse_args(argv)

    data = analyse()
    rows = data["rows"]
    print("corpus: %d questions" % len(rows))
    print("  filter-slot disagreements  : %d" % sum(r["filter_disagree"] for r in rows))
    print("  grouping-slot disagreements: %d" % sum(r["group_disagree"] for r in rows))
    print("  answer-type disagreements  : %d" % sum(bool(r["type_disagree"]) for r in rows))

    print("\nfacet kinds raised across the corpus:")
    for kind, n in Counter(k for r in rows for k in r["facet_kinds"]).most_common():
        print("  %-26s %4d" % (kind, n))

    print("\nquestions raising NO facet at all: %d"
          % sum(1 for r in rows if not r["facet_kinds"]))

    if args.json:
        args.json.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")
        print("\nwrote %s" % args.json)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
