#!/usr/bin/env python3
"""The canonical semantic census: what every question in the corpus MEANS.

WHY A CENSUS AND NOT A TEST SUITE. A test bank asserts the answers somebody
thought to write down. The census records the INTERPRETATION of every question
the estate knows about — 882 of them — so a change that moves a meaning nobody
wrote a test for is still visible. Twice now it has been the only instrument
that saw a defect:

  * a unit resolver matched "year" inside the synonym "one year pd" and bound
    `probability_of_default > 80` for "loans over 80 years old". It survived
    three full regressions — roughly 3,600 tests — and the census caught it.

  * consolidating the population owner silently retired an "unknown category"
    disclosure on one question. Every bank stayed green; the census showed the
    one movement.

WHAT IT RECORDS, and the list is deliberate. An earlier version recorded FILTERS
only, and after a change to what "by" means it reported zero movements — which
was not evidence of anything, because the change moved DIMENSIONS. A census that
cannot see the axis a question is answered along is not a census. The seven
facets below are what decides an answer; `test_semantic_census` asserts that
they are all recorded, so the instrument cannot be quietly narrowed again.

RESOLVED AGAINST THE REAL BOOK. An earlier version parsed with no columns and no
value catalogue, so every categorical narrowing resolved to nothing and a whole
class of movement was invisible to it. This supplies exactly what the live path
supplies — `book_columns` and `book_values` over the funded portfolio.

A ZERO IS NOT AUTOMATICALLY A PASS. The corpus is a fixed body of language, and
a construction it does not contain cannot move in it. When a change reports zero
movements, the honest reading is "the corpus does not exercise this", and the
right follow-up is to measure whether it exercises it at all — not to record
safety.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

#: The book the census resolves against — the estate's canonical funded
#: portfolio, the same file the live path reads.
BOOK_CSV = _REPO_ROOT / (
    "ERE_Portfolio_122025_ESMA_Annex2_canonical_ESMA_Annex2_typed.csv")

SEMANTICS_YAML = _REPO_ROOT / "mi_agent" / "mi_semantics_field_registry.yaml"

#: Every facet that decides an answer. Narrowing this list is how a census stops
#: being one; `test_semantic_census` pins it.
FACETS = ("filters", "dimensions", "metric", "aggregation", "measures",
          "intent", "unavailable")

#: The artifact this census is compared against, checked in beside it so a
#: movement appears in the diff of the commit that caused it.
BASELINE = Path(__file__).with_name("semantic_census_baseline.json")


def corpus() -> List[str]:
    """The questions, in corpus order. One owner: the migration trace's."""
    import migration_phase0.filter_ownership_trace as trace

    return list(trace._corpus())


def _book_context():
    import pandas as pd

    import mi_agent.execution_receipt as receipt
    from mi_agent.mi_query_validator import load_mi_semantics

    semantics = load_mi_semantics(str(SEMANTICS_YAML))
    frame = pd.read_csv(BOOK_CSV, low_memory=False)
    return semantics, receipt.book_columns(frame), receipt.book_values(
        frame, semantics)


def interpretation(spec) -> Dict[str, Any]:
    """One spec reduced to the seven facets, in a canonical order.

    Ordering is canonicalised — mappings sorted by key, sequences kept in
    question order — so a diff shows a MEANING that moved and never an
    iteration order that did.
    """
    if spec is None:
        return {"__none__": True}
    dimensions = list(spec.dimensions or []) or (
        [spec.dimension] if spec.dimension else [])
    return {
        "aggregation": spec.aggregation,
        "dimensions": dimensions,
        "filters": {k: spec.filters[k] for k in sorted(spec.filters or {})},
        "intent": spec.intent,
        "measures": [[m.get("field"), m.get("aggregation")]
                     for m in (spec.measures or [])],
        "metric": spec.metric,
        "unavailable": sorted(str(u) for u in (spec.unavailable_filters or [])),
    }


def census() -> Dict[str, Dict[str, Any]]:
    """``{question: interpretation}`` for the whole corpus."""
    from mi_agent.llm_query_parser import _deterministic_parse

    semantics, columns, values = _book_context()
    out: Dict[str, Dict[str, Any]] = {}
    for question in corpus():
        try:
            spec, _meta = _deterministic_parse(
                question, semantics, available_columns=columns,
                available_values=values)
            out[question] = interpretation(spec)
        except Exception as exc:  # noqa: BLE001 - a raise IS a movement
            out[question] = {"__error__": f"{type(exc).__name__}: {exc}"}
    return out


def canonical_json(data: Dict[str, Any]) -> str:
    """The persisted form. Sorted keys, one fact per line, stable across runs."""
    return json.dumps(data, indent=1, sort_keys=True, default=str) + "\n"


def movements(before: Dict[str, Any], after: Dict[str, Any]) -> List[str]:
    """Human-readable lines for every question whose meaning differs."""
    lines: List[str] = []
    for question in sorted(set(before) | set(after)):
        was, now = before.get(question), after.get(question)
        if was == now:
            continue
        lines.append(f"--- {question}")
        lines.append(f"    before: {json.dumps(was, sort_keys=True, default=str)}")
        lines.append(f"    after : {json.dumps(now, sort_keys=True, default=str)}")
    return lines


def main(argv: List[str]) -> int:
    """``python -m mi_agent.tests.semantic_census [--write]``."""
    current = census()
    if "--write" in argv:
        BASELINE.write_text(canonical_json(current), encoding="utf-8")
        print(f"wrote {len(current)} interpretations to {BASELINE}")
        return 0
    if not BASELINE.exists():
        print("no baseline; run with --write")
        return 1
    moved = movements(json.loads(BASELINE.read_text(encoding="utf-8")), current)
    print("\n".join(moved) if moved else "no movements")
    print(f"corpus: {len(current)}  movements: {len(moved) // 3}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
