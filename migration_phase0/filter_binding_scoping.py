#!/usr/bin/env python3
"""migration_phase0/filter_binding_scoping.py — who binds a predicate to a field.

READ-ONLY. Scoping evidence for the last C6 filter prerequisite.

The question is where "LTV above 50%" becomes `current_loan_to_value gt 50.0`,
whether that binding is already governed and single-owner, and whether the
interpretation contract can carry it without a second resolver.

Measures four things over the governed corpus:

  1. the filter FAMILIES actually present, and the resolver each one reaches;
  2. numeric vs categorical - one mechanism or two;
  3. whether the parser's existing (but unused) `spans` output can join a
     `FilterClaim` to the governed field the parser already resolved;
  4. the fail-closed path already in production for an unbindable threshold.

Fails loudly if governed semantics do not load.

    python -m migration_phase0.filter_binding_scoping [--json out.json]
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

#: The delivered filtered-evolution cases from the threshold receipt fix. These
#: are the non-vacuous anchors: whatever representation is proposed must be able
#: to describe exactly what these executed.
DELIVERED = (
    ("balance, LTV > 50", "Show funded balance evolution by month for loans above 50% LTV."),
    ("balance, age > 75", "Show funded balance evolution by month for borrowers over 75."),
    ("balance, loan > 200k", "Show funded balance evolution by month for loans above 200000."),
    ("count, LTV > 50", "Show loan count evolution by month for loans above 50% LTV."),
)


def _corpus() -> List[str]:
    out, seen = [], set()
    for f in CORPORA:
        p = _REPO / f
        if not p.exists():
            continue
        for row in json.loads(p.read_text(encoding="utf-8"))["rows"]:
            q = row.get("question") or ""
            if q and q not in seen:
                seen.add(q)
                out.append(q)
    return out


def _env():
    warnings.simplefilter("ignore")
    os.environ.setdefault("TRAKT_RUNTIME_MODE", "development")
    from demo_platform import config as cfg
    os.environ.update(cfg.mi_env(period_role="current"))
    os.environ["MI_AGENT_LLM_PARSER"] = "off"
    os.environ["MI_AGENT_LLM_ENABLED"] = "0"
    logging.disable(logging.WARNING)
    from migration_phase0.assurance_semantics import load_assurance_semantics
    return load_assurance_semantics()


def _observe_parse(P, ParsedQuestion, question: str, semantics: dict):
    """Run the real parse, recording the args production hands `_parse_filters`.

    Returns `(spans, unresolved, filters)` from the call that produced the spec's
    filters — the last invocation whose result matches `spec.filters`.
    """
    seen: List[Tuple[Dict[str, Any], List[str], Dict[str, Any]]] = []
    original = P._parse_filters

    def spy(q, sem, available_columns=None, unresolved=None, spans=None):
        spans = {} if spans is None else spans
        unresolved = [] if unresolved is None else unresolved
        out = original(q, sem, available_columns, unresolved=unresolved, spans=spans)
        seen.append((dict(spans), list(unresolved), dict(out)))
        return out

    P._parse_filters = spy
    try:
        spec = ParsedQuestion.parse(question, semantics).spec
    finally:
        P._parse_filters = original
    produced = dict(getattr(spec, "filters", None) or {})
    for spans, unresolved, out in reversed(seen):
        if out == produced:
            return spans, unresolved, produced
    return ({}, [], produced)


def _family(field: str) -> str:
    f = (field or "").lower()
    if "loan_to_value" in f:
        return "LTV"
    if "age" in f:
        return "borrower age"
    if "balance" in f or "outstanding" in f:
        return "balance"
    if "interest" in f or "rate" in f:
        return "interest rate"
    if "geograph" in f or "region" in f:
        return "geography"
    if "months_on_book" in f or "seasoning" in f:
        return "months on book"
    if "borrower_type" in f or "number_of_borrowers" in f:
        return "borrower type"
    if "pipeline_stage" in f:
        return "pipeline stage"
    return f"other:{field}"


def _claim_span(claim) -> Optional[Tuple[int, int]]:
    """A `Span` is a frozen dataclass with `.start`/`.end`, not a tuple."""
    span = getattr(claim, "span", None)
    if span is None:
        return None
    return (int(span.start), int(span.end))


def _overlap(a: Optional[Tuple[int, int]], b: Optional[Tuple[int, int]]) -> bool:
    if not a or not b:
        return False
    return not (a[1] <= b[0] or b[1] <= a[0])


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--json", type=Path, default=None)
    args = ap.parse_args(argv)

    semantics = _env()
    from mi_agent import llm_query_parser as P
    from mi_agent.parsed_question import ParsedQuestion
    from question_interpretation import projection as proj

    print("=" * 92)
    print("C6 FILTER BINDING — where a predicate acquires its governed field")
    print("=" * 92)

    # ---- 1. Ownership: is the binder reachable from any route? ------------ #
    print("\n1. THE BINDING OWNER")
    print("   parser      : llm_query_parser._filter_field_of  (anchor-aware, "
          "proximity rule)")
    print("   vocabulary  : lexical.THRESHOLD_SUBJECT_PATTERNS  (the single "
          "lexical owner)")
    print("   kind->field : llm_query_parser._resolve_subject -> "
          "_ltv/_age/_rate/_balance_metric, find_field")
    print("   output      : spec.filters, ALREADY keyed by governed field")
    route_src = (_REPO / "mi_agent_api" / "chat_routing.py").read_text(encoding="utf-8")
    for name in ("_filter_field_of", "_resolve_subject", "THRESHOLD_SUBJECT_PATTERNS"):
        print(f"   route calls {name:<28}: {'YES' if name in route_src else 'no'}")

    # ---- 2/3. Corpus families, numeric vs categorical, span join ---------- #
    rows: List[Dict[str, Any]] = []
    fam = Counter()
    shape = Counter()
    joinable = unjoinable = no_span = 0
    filtered_questions = 0

    for q in _corpus():
        # OBSERVE production rather than re-implement it. Calling `_parse_filters`
        # directly gave a DIFFERENT answer: production parses the REMAINDER left
        # after `_mask_spans` blanks the measure spans, so "funded balance ...
        # above 50% LTV" binds LTV only because "funded balance" was masked
        # first. The proxy bound it to `current_outstanding_balance` and reported
        # no LTV family at all. The binding is not a pure function of the clause;
        # it depends on measure extraction having already run.
        spans, unresolved, filters = _observe_parse(P, ParsedQuestion, q, semantics)
        if not filters:
            continue
        filtered_questions += 1
        qi = proj.project(q, semantics=semantics, frame=None)
        claims = [c for c in (getattr(qi, "filters", None) or [])]

        for field, cond in filters.items():
            numeric = isinstance(cond, dict)
            shape["numeric" if numeric else "categorical"] += 1
            fam[_family(field)] += 1
            pspan = spans.get(field)
            # Can a FilterClaim be joined to this governed field by span?
            hits = [c for c in claims
                    if _overlap(_claim_span(c), pspan)]
            if pspan is None:
                no_span += 1
                state = "NO PARSER SPAN"
            elif len(hits) == 1:
                joinable += 1
                state = "JOINABLE (1 claim)"
            else:
                unjoinable += 1
                state = f"AMBIGUOUS ({len(hits)} claims)"
            rows.append({"question": q, "field": field, "family": _family(field),
                         "numeric": numeric, "condition": cond,
                         "parser_span": pspan, "claims": len(claims),
                         "join": state, "unresolved": list(unresolved)})

    print(f"\n2. CORPUS FILTERS — {filtered_questions} questions carrying filters")
    print(f"   total filter predicates: {len(rows)}")
    print(f"\n   {'family':<22}{'count':>6}")
    for k, n in fam.most_common():
        print(f"   {k:<22}{n:>6}")
    print(f"\n   numeric     {shape['numeric']}")
    print(f"   categorical {shape['categorical']}")

    print("\n3. CAN A FilterClaim BE JOINED TO THE GOVERNED FIELD BY SPAN?")
    print(f"   JOINABLE (exactly one claim overlaps): {joinable}")
    print(f"   AMBIGUOUS (0 or >1 claims overlap)   : {unjoinable}")
    print(f"   parser produced no span              : {no_span}")

    # ---- 4. Fail-closed path already in production ------------------------ #
    print("\n4. THE EXISTING FAIL-CLOSED PATH (unbindable threshold)")
    for probe in ("Show funded balance for loans above 50 zorkmids.",
                  "Show funded balance over 50.",
                  "Show funded balance evolution by month for loans above 50% LTV."):
        _sp, un, f = _observe_parse(P, ParsedQuestion, probe, semantics)
        print(f"   filters={str(f)[:52]:<54} unresolved={un}  | {probe[:44]}")

    # ---- 5. Delivered anchors -------------------------------------------- #
    print("\n5. DELIVERED CASES — can the representation describe what executed?")
    for label, q in DELIVERED:
        spans2, _un2, f = _observe_parse(P, ParsedQuestion, q, semantics)
        for field, cond in f.items():
            op = cond.get("op") if isinstance(cond, dict) else "eq"
            val = cond.get("value") if isinstance(cond, dict) else cond
            print(f"   {label:<22} field={field:<28} op={str(op):<6} "
                  f"value={val}  span={spans2.get(field)}")

    if args.json:
        args.json.write_text(json.dumps(rows, indent=2, default=str), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
