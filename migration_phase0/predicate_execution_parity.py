#!/usr/bin/env python3
"""migration_phase0/predicate_execution_parity.py — one Predicate, one meaning.

READ-ONLY. The assurance instrument for the predicate-execution parity
invariant:

    A governed Predicate(field, op, value) has ONE deterministic execution
    meaning everywhere in Trakt. For any frame and predicate, the shipped
    filter executor and the reusable population executor must either select
    the same rows, or fail in the same governed way. There must be no case
    where one narrows while another silently widens.

Three sections, and the second is the one that matters:

  CORPUS   the 119 filtered questions, through the PRODUCTION assembly path,
           comparing `_apply_filters` row sets against `apply_population` row
           sets. This is what the C6 plan primitive would rely on.
  PROBE    the divergence classes directly — identical predicate, both
           executors, on frames built to exercise one concern each. The corpus
           only ever exercises the predicates the corpus happens to contain;
           three of the five known classes were invisible to it.
  NEGATIVE the controls: missing field, unsupported operator, invalid numeric
           value, categorical mismatch, null-heavy column, percent field,
           non-percent numeric field. Every one must refuse rather than widen.

Fails loudly if governed semantics do not load.

    python -m migration_phase0.predicate_execution_parity [--json out.json]
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import warnings
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

CORPORA = ("question_interpretation/stage1_corpus.json",
           "question_interpretation/stage2_corpus.json")

#: The seven filter families the binding has to be generic across.
_FAMILIES = (("loan_to_value", "LTV"), ("age", "borrower age"),
             ("outstanding", "balance"), ("interest", "interest rate"),
             ("geograph", "geography"), ("months_on_book", "months on book"),
             ("borrower_type", "borrower type"),
             ("number_of_borrowers", "borrower type"),
             ("pipeline_stage", "pipeline stage"))


def _family(field: str) -> str:
    f = (field or "").lower()
    for needle, name in _FAMILIES:
        if needle in f:
            return name
    return f"other:{field}"


def _corpus() -> List[str]:
    out, seen = [], set()
    for name in CORPORA:
        path = _REPO / name
        if not path.exists():
            continue
        for row in json.loads(path.read_text(encoding="utf-8"))["rows"]:
            question = row.get("question") or ""
            if question and question not in seen:
                seen.add(question)
                out.append(question)
    return out


def _boot() -> None:
    warnings.simplefilter("ignore")
    os.environ.setdefault("TRAKT_RUNTIME_MODE", "development")
    from demo_platform import config as cfg
    os.environ.update(cfg.mi_env(period_role="current"))
    os.environ["MI_AGENT_LLM_PARSER"] = "off"
    os.environ["MI_AGENT_LLM_ENABLED"] = "0"
    logging.disable(logging.WARNING)


# --------------------------------------------------------------------------- #
# The two executions, expressed so a single caller can compare them
# --------------------------------------------------------------------------- #
def _shipped(frame, filters: Dict[str, Any], semantics) -> Tuple[str, Any]:
    """`_apply_filters`, the point-in-time executor, from a spec."""
    from mi_agent.mi_query_executor import _apply_filters, MIQueryExecutionError
    from mi_agent.mi_query_spec import MIQuerySpec
    try:
        out = _apply_filters(frame.copy(), MIQuerySpec(filters=dict(filters)),
                             semantics, [], [])
        return "rows", sorted(out.index)
    except MIQueryExecutionError as exc:
        return "refused", str(exc)


def _reusable(frame, predicates, semantics) -> Tuple[str, Any]:
    """`apply_population`, the reusable executor, from Predicates."""
    from mi_agent.population import apply_population
    out, evidence = apply_population(frame, predicates, semantics)
    if evidence.unavailable:
        # Parity requires this to be a REFUSAL, not a wider frame. A frame that
        # comes back non-None here is the silent-widening defect itself.
        return ("refused" if out is None else "WIDENED",
                "; ".join(evidence.unavailable))
    return "rows", sorted(out.index) if out is not None else []


def _agree(a: Tuple[str, Any], b: Tuple[str, Any]) -> bool:
    """Same rows, or the same governed failure. Nothing else counts."""
    if a[0] == "rows" and b[0] == "rows":
        return a[1] == b[1]
    return a[0] == "refused" and b[0] == "refused"


def _predicates(filters, semantics):
    from mi_agent.population import material_predicates
    return list(material_predicates(filters, semantics))


# --------------------------------------------------------------------------- #
def _corpus_census(frame, semantics) -> Dict[str, Any]:
    from mi_agent import execution_receipt as R
    from mi_agent.parsed_question import ParsedQuestion
    from question_interpretation import projection as proj

    rows: List[Dict[str, Any]] = []
    fam_ok, fam_bad = Counter(), Counter()
    contract_ok = Counter()
    predicates_seen = 0
    columns = list(frame.columns)

    for question in _corpus():
        # THE PRODUCTION SHAPE. `projection.project` calls `_deterministic_parse`
        # directly and so never sees `resolve_seasoning_role`, which runs inside
        # `parse_with_repair` and is where "new lending" becomes
        # `months_on_book <= 1`. Measuring that path understates the corpus by
        # exactly the seasoning family.
        spec = ParsedQuestion.parse(question, semantics).spec
        filters = dict(getattr(spec, "filters", None) or {})
        if not filters:
            continue
        dim_terms = R.requested_dimension_terms(question, semantics, columns)
        facets = R.detect_requested_facets(question, semantics, frame=frame,
                                           requested_dimensions=dim_terms)
        qi = proj.from_parts(question, spec=spec, facets=facets,
                             dim_terms=dim_terms, semantics=semantics)
        claims = list(getattr(qi, "row_predicates", None) or [])

        # CONTRACT agreement: does the claim describe the spec entry the
        # executor was handed?
        for claim in claims:
            predicates_seen += 1
            raw = filters.get(claim.field_key)
            contract_ok["field"] += int(claim.field_key in filters)
            if isinstance(raw, dict):
                contract_ok["operator"] += int(str(raw.get("op")) == str(claim.operator))
                contract_ok["value"] += int(raw.get("value") == claim.value)
            else:
                contract_ok["operator"] += int(claim.operator in ("eq", "in"))
                contract_ok["value"] += int(
                    raw == claim.value
                    or (isinstance(raw, (list, tuple, set)) and list(raw) == claim.value))

        # EXECUTION agreement: do the claims, executed, select what the spec
        # executed selects?
        from mi_agent.population import Predicate
        contract_predicates = [Predicate(c.field_key, c.operator, c.value)
                               for c in claims]
        shipped = _shipped(frame, filters, semantics)
        reusable = _reusable(frame, contract_predicates, semantics)
        same = _agree(shipped, reusable)
        for claim in claims:
            (fam_ok if same else fam_bad)[_family(claim.field_key)] += 1

        rows.append({
            "question": question, "spec_filters": filters,
            "claims": [{"field": c.field_key, "op": c.operator, "value": c.value}
                       for c in claims],
            "shipped": [shipped[0], shipped[1] if isinstance(shipped[1], str)
                        else len(shipped[1])],
            "reusable": [reusable[0], reusable[1] if isinstance(reusable[1], str)
                         else len(reusable[1])],
            "agree": same,
        })

    return {"rows": rows, "families_ok": fam_ok, "families_bad": fam_bad,
            "contract": contract_ok, "predicates": predicates_seen}


# --------------------------------------------------------------------------- #
def _probe_frame():
    import pandas as pd
    return pd.DataFrame({
        "current_loan_to_value": [0.30, 0.55, 0.72, None, 0.95],
        "youngest_borrower_age": [55, 68, 75, 82, None],
        "collateral_geography": ["South East", "London", "South West", None,
                                 "south east"],
        "borrower_type": ["Single", "Joint", "Single", "Joint", None],
        "current_interest_rate": [3.5, 4.25, 5.0, 6.75, None],
        "months_on_book": [1, 3, 12, 36, None],
    })


#: (label, class, filters, expected). The class names the divergence each case
#: exercises, so a regression reports WHICH semantic owner drifted. `expected`
#: is the ABSOLUTE outcome, and it is the load-bearing half.
#:
#: Why an absolute is needed at all: once both paths go through one owner, a
#: mutation INSIDE that owner moves them together and they still agree. A pure
#: agreement check therefore proves parity and says nothing about correctness —
#: measured, not assumed: removing the percent normalisation left this census at
#: 119/119 until these expectations were added. Parity is what the invariant
#: asks for; correctness is what the reader gets.
PROBES: Tuple[Tuple[str, str, Dict[str, Any], Any], ...] = (
    ("percent threshold in points", "percent",
     {"current_loan_to_value": {"op": "gt", "value": 50}}, 3),
    ("percent range in points", "percent",
     {"current_loan_to_value": {"op": "between", "value": [40, 60]}}, 1),
    ("percent field, fraction-scale operand", "percent",
     {"current_loan_to_value": {"op": "gt", "value": 0.5}}, 3),
    ("non-percent numeric must NOT rescale", "no-rescale",
     {"youngest_borrower_age": {"op": "gt", "value": 70}}, 2),
    ("percent_points field must NOT rescale", "no-rescale",
     {"current_interest_rate": {"op": "gt", "value": 4}}, 3),
    ("operator alias '>'", "operator", {"youngest_borrower_age": {"op": ">", "value": 70}}, 2),
    ("operator alias 'above'", "operator",
     {"youngest_borrower_age": {"op": "above", "value": 70}}, 2),
    ("operator alias 'gte'", "operator",
     {"youngest_borrower_age": {"op": "gte", "value": 70}}, 2),
    ("operator alias 'greater_than_or_equal'", "operator",
     {"youngest_borrower_age": {"op": "greater_than_or_equal", "value": 70}}, 2),
    ("categorical bare string", "categorical", {"borrower_type": "Joint"}, 2),
    ("categorical dict shape", "categorical",
     {"borrower_type": {"op": "eq", "value": "Joint"}}, 2),
    ("categorical case-insensitive", "categorical",
     {"collateral_geography": "south east"}, 2),
    ("membership in []", "categorical", {"borrower_type": ["Joint"]}, 2),
    ("null-heavy column, wide bound", "nulls",
     {"youngest_borrower_age": {"op": "le", "value": 200}}, 4),
)

#: Negative controls. Each must REFUSE on both paths — never widen, never
#: silently empty.
NEGATIVE: Tuple[Tuple[str, Dict[str, Any]], ...] = (  # expected: refusal
    ("missing field (column absent)", {"borrower_type_missing": "Joint"}),
    ("unknown semantic field", {"not_a_governed_field": {"op": "gt", "value": 1}}),
    ("invalid numeric value", {"youngest_borrower_age": {"op": "gt", "value": "abc"}}),
    ("invalid range bounds", {"youngest_borrower_age": {"op": "between", "value": ["a", "b"]}}),
)


def _probe_section(semantics, cases, *, drop: Optional[str] = None):
    frame = _probe_frame()
    out = []
    for entry in cases:
        if len(entry) == 4:
            label, klass, filters, expected = entry
        else:
            label, klass, filters, expected = entry[0], "negative", entry[1], "refused"
        work = frame
        key = next(iter(filters))
        if drop == "auto" and key.endswith("_missing"):
            filters = {key.replace("_missing", ""): filters[key]}
            work = frame.drop(columns=[key.replace("_missing", "")])
        shipped = _shipped(work, filters, semantics)
        reusable = _reusable(work, _predicates(filters, semantics), semantics)
        if expected == "refused":
            correct = shipped[0] == "refused" and reusable[0] == "refused"
        else:
            correct = (shipped[0] == "rows" and reusable[0] == "rows"
                       and len(shipped[1]) == expected == len(reusable[1]))
        out.append({"label": label, "class": klass, "filters": filters,
                    "expected": expected,
                    "shipped": [shipped[0], shipped[1] if isinstance(shipped[1], str)
                                else len(shipped[1])],
                    "reusable": [reusable[0], reusable[1] if isinstance(reusable[1], str)
                                 else len(reusable[1])],
                    "agree": _agree(shipped, reusable),
                    "correct": correct})
    return out


def _print_probes(title, rows) -> int:
    print(f"\n{title}")
    bad = 0
    for row in rows:
        ok = row["agree"] and row["correct"]
        bad += 0 if ok else 1
        flag = "OK " if ok else ("WRONG" if row["agree"] else "DIV")
        s, r = row["shipped"], row["reusable"]
        print(f"  {flag:<5}[{row['class']:<11}] {row['label']:<40}"
              f" want={str(row['expected']):<8} shipped={s[0]}:{str(s[1])[:20]:<22}"
              f" reusable={r[0]}:{str(r[1])[:20]}")
    return bad


# --------------------------------------------------------------------------- #
def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", type=Path, default=None)
    args = parser.parse_args(argv)

    _boot()
    from demo_platform import config as cfg
    from migration_phase0.assurance_semantics import load_assurance_semantics
    from mi_agent_api import evolution as evolution_mod

    semantics = load_assurance_semantics()
    frames = evolution_mod.funded_frames(
        os.environ["MI_AGENT_ONBOARDING_OUTPUT_ROOT"], cfg.CLIENT_ID, None)
    if not frames:
        raise SystemExit("ASSURANCE INVALID - no governed funded frames")
    frame = frames[-1]["df"]
    if frame is None or not len(frame):
        raise SystemExit("ASSURANCE INVALID - the governed frame is empty")

    print("=" * 100)
    print("PREDICATE EXECUTION PARITY — one governed Predicate, one meaning")
    print("=" * 100)
    print(f"frame: {len(frame):,} rows, {len(frame.columns)} columns")

    census = _corpus_census(frame, semantics)
    rows = census["rows"]
    n = len(rows)
    agree = sum(1 for r in rows if r["agree"])
    p = census["predicates"]
    c = census["contract"]
    print(f"\nCONTRACT agreement (parser -> RowPredicateClaim)")
    print(f"   field    : {c['field']}/{p}")
    print(f"   operator : {c['operator']}/{p}")
    print(f"   value    : {c['value']}/{p}")
    print(f"\nEXECUTION agreement (same rows, or the same governed failure)")
    print(f"   {agree}/{n}")

    print(f"\n{'family':<20}{'agree':>7}{'disagree':>10}")
    for fam in sorted(set(census["families_ok"]) | set(census["families_bad"])):
        print(f"{fam:<20}{census['families_ok'][fam]:>7}"
              f"{census['families_bad'][fam]:>10}")

    bad = [r for r in rows if not r["agree"]]
    if bad:
        print(f"\nEXECUTION DISAGREEMENTS ({len(bad)}):")
        for r in bad[:12]:
            print(f"   shipped={r['shipped']} reusable={r['reusable']}  "
                  f"{r['claims']} | {r['question'][:42]}")

    probes = _probe_section(semantics, PROBES)
    negatives = _probe_section(semantics, NEGATIVE, drop="auto")
    probe_bad = _print_probes("DIVERGENCE-CLASS PROBES", probes)
    neg_bad = _print_probes("NEGATIVE CONTROLS (both paths must refuse)", negatives)
    widened = [r for r in probes + negatives if r["reusable"][0] == "WIDENED"]
    print(f"\n   silent widening (reusable returned a frame it could not narrow): "
          f"{len(widened)}")

    ok = (agree == n and probe_bad == 0 and neg_bad == 0 and not widened
          and c["field"] == p and c["operator"] == p and c["value"] == p)
    print("\n" + ("=" * 100))
    print("VERDICT: " + ("PARITY HOLDS" if ok else "PARITY DOES NOT HOLD"))
    print("=" * 100)

    if args.json:
        args.json.write_text(json.dumps(
            {"census": rows, "probes": probes, "negatives": negatives},
            indent=2, default=str), encoding="utf-8")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
