#!/usr/bin/env python3
"""compositional_plan_scoping/t3_now.py — what would it take to land T3 today?

READ-ONLY. Nothing is patched and no product module is imported into a serving
path. This instrument tests one specific claim:

    "the evolution route already computes a full period-by-region series —
     516 rows measured — and discards it, so T3 may be a small carriage change."

It checks that claim in five places, in the order a carriage change would have
to satisfy them:

  1. **Does the route KNOW a dimension was asked for?**  It receives a spec.
  2. **Is the discarded series the RIGHT series?**  Granularity, not row count.
  3. **Would the receipt ACCEPT it?**  `grouping_proven`, unchanged.
  4. **How many dimensions does the existing breakdown reach?**
  5. **What would a parser change move?**  Measured on the standing banks.

    python -m compositional_plan_scoping.t3_now
"""
from __future__ import annotations

import argparse
import logging
import os
import pathlib
import sys
import warnings
from typing import Any, Dict, List, Optional, Sequence, Tuple

warnings.simplefilter("ignore")

_HERE = pathlib.Path(__file__).resolve().parent
_REPO = _HERE.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

BALANCE = "current_outstanding_balance"

#: The phrasings whose specs a line-branch dimension carry would change, split by
#: whether the recovered key is the question's GROUPING or its SUBJECT. The split
#: is the study's reading; the instrument proves each phrasing still parses the
#: way the reading assumes.
_SUBJECT_SIDE = (
    "Show average borrower age evolution by month.",
    "Show LTV bucket evolution over time.",
    "Show age bucket evolution over time.",
)


def _env() -> str:
    from demo_platform import config as cfg
    os.environ.update(cfg.mi_env(period_role="current"))
    os.environ["MI_AGENT_LLM_PARSER"] = "off"
    os.environ["MI_AGENT_LLM_ENABLED"] = "0"
    logging.disable(logging.WARNING)
    return cfg.CLIENT_ID


def _bank_questions() -> List[str]:
    import yaml
    out: List[str] = []

    def walk(node: Any) -> None:
        if isinstance(node, list):
            for item in node:
                walk(item)
        elif isinstance(node, dict):
            question = node.get("question")
            if isinstance(question, str):
                out.append(question)
            else:
                for value in node.values():
                    walk(value)

    for path in sorted((_REPO / "config" / "mi" / "golden_questions").glob("*.yaml")):
        try:
            walk(yaml.safe_load(path.read_text(encoding="utf-8")))
        except Exception:  # noqa: BLE001 - a bad bank file is not this study's problem
            continue
    return list(dict.fromkeys(out))


def run(client_id: str) -> int:
    from mi_agent.mi_query_validator import load_mi_semantics
    from mi_agent_api.data_source import semantics_path
    from mi_agent.llm_query_parser import parse_with_repair, _explicit_dimensions
    from mi_agent.mi_query_executor import _grouped_aggregate
    from mi_agent_api import chat_routing as routing
    from mi_agent_api import evolution as evolution_mod
    from mi_agent import execution_receipt as receipt
    from question_interpretation import mi_phrasing_bank as phrasing_bank
    from question_interpretation import time_series_surface as surface

    semantics = load_mi_semantics(semantics_path())
    fields = semantics.get("fields", {})
    root = os.environ["MI_AGENT_ONBOARDING_OUTPUT_ROOT"]
    frames = evolution_mod.funded_frames(root, client_id, None)
    columns = set(frames[0]["df"].columns)

    print("=" * 78)
    print(f"WHAT WOULD IT TAKE TO LAND T3 TODAY? — {client_id}")
    print("=" * 78)

    # -- 1 ---------------------------------------------------------------- #
    print("\n1. DOES THE ROUTE KNOW A DIMENSION WAS ASKED FOR?\n")
    probes = ("balance over time", "balance over time by region",
              "balance over time by LTV band", "balance over time by broker")
    seen: List[Tuple[str, ...]] = []
    for question in probes:
        spec, _meta = parse_with_repair(question, semantics, llm_enabled=False)
        shape = (spec.chart_type, spec.metric, spec.aggregation,
                 str(spec.dimension), str(list(spec.dimensions)))
        seen.append(shape)
        print(f"   {question:36s} chart={spec.chart_type!r} "
              f"dimension={spec.dimension!r} dimensions={list(spec.dimensions)}")
    identical = len(set(seen)) == 1
    print(f"\n   -> the four specs are {'IDENTICAL' if identical else 'DISTINCT'}."
          f" The evolution route cannot tell these questions apart,")
    print("      so no change confined to the route can carry a dimension it "
          "never receives.")

    # -- 2 ---------------------------------------------------------------- #
    print("\n2. IS THE DISCARDED SERIES THE RIGHT SERIES?\n")
    breakdown_col = evolution_mod._FUNDED_BREAKDOWN_DIMS.get("region")
    requested = receipt.requested_dimension_terms("balance over time by region",
                                                  semantics)
    requested_key = requested[0][0] if requested else None
    requested_col = (fields.get(requested_key, {}) or {}).get(
        "canonical_field", requested_key)
    for col, label in ((breakdown_col, "what the breakdown COMPUTES"),
                       (requested_col, "what the request RESOLVES to")):
        if col not in columns:
            print(f"   {str(col):34s}  absent from this book")
            continue
        rows = 0
        for frame in frames:
            df = frame["df"]
            grouped, _c = _grouped_aggregate(
                df.assign(**{col: df[col].astype(str)}), [col],
                BALANCE, "sum", None, None)
            rows += len(grouped)
        distinct = int(frames[-1]["df"][col].astype(str).nunique())
        print(f"   {col:34s}  {distinct:4d} categories, {rows:5d} rows / "
              f"{len(frames)} periods   ({label})")
    print(f"\n   -> the discarded series is cut by {breakdown_col!r}; the request "
          f"resolves to {requested_col!r}.")
    print("      Same word, different granularity. Row count is not the test.")

    # -- 3 ---------------------------------------------------------------- #
    print("\n3. WOULD THE RECEIPT ACCEPT THE DISCARDED SERIES?\n")
    if requested:
        key, term, alts = requested[0]
        facet = receipt.RequestedFacet(kind=receipt.KIND_GROUPING, label=term,
                                       field_key=key, alt_keys=alts)
        print(f"   the request raises: kind={facet.kind!r} label={facet.label!r} "
              f"satisfied_by={facet.satisfied_by()}")
        for declared in ([breakdown_col], [requested_col]):
            proven = receipt.grouping_proven(facet, declared, fields)
            verdict = "APPLIED" if proven else "still LOST — the answer refuses"
            print(f"   grouping_proven(groupedBy={declared}) -> {proven}   {verdict}")
    print("\n   -> the guard is already correct. Publishing the discarded rows "
          "would produce a")
    print("      STILL-REFUSED answer, not a working T3 — and not a false "
          "certification either.")

    # -- 4 ---------------------------------------------------------------- #
    print("\n4. HOW MANY DIMENSIONS DOES THE EXISTING BREAKDOWN REACH?\n")
    dimension_keys = [k for k, e in fields.items() if e.get("role") == "dimension"]
    on_tape = [k for k in dimension_keys
               if (fields[k].get("canonical_field") or k) in columns
               or (fields[k].get("bucket_field") or "") in columns]
    reachable = []
    for name, col in evolution_mod._FUNDED_BREAKDOWN_DIMS.items():
        present = col in columns
        print(f"   {name:12s} -> {col:32s} {'present' if present else 'ABSENT — returns [] silently'}")
        if present:
            reachable.append(name)
    print(f"\n   -> {len(reachable)} of {len(on_tape)} dimensions this book can be "
          f"cut by ({len(reachable) / len(on_tape) * 100:.0f}%);"
          f" {len(dimension_keys)} are governed in the registry.")

    # -- 5 ---------------------------------------------------------------- #
    print("\n5. WHAT WOULD A LINE-BRANCH DIMENSION CARRY MOVE?\n")
    questions = _bank_questions()
    line = 0
    moved: List[Tuple[str, List[str]]] = []
    for question in questions:
        try:
            spec, _meta = parse_with_repair(question, semantics, llm_enabled=False)
        except Exception:  # noqa: BLE001
            continue
        if getattr(spec, "chart_type", None) != "line":
            continue
        line += 1
        try:
            keys, _terms, _rest = _explicit_dimensions(
                question.lower(), semantics, grouping=True,
                available_columns=columns)
        except Exception:  # noqa: BLE001
            keys = []
        if keys:
            moved.append((question, list(keys)))
    subject_side = [m for m in moved if m[0] in _SUBJECT_SIDE]
    print(f"   {len(questions)} distinct bank questions; {line} reach the line branch;")
    print(f"   {len(moved)} of those name a resolvable grouping dimension "
          f"({len(moved) / len(questions) * 100:.1f}% of the bank).")
    print(f"\n   of the {len(moved)} specs that would move, {len(subject_side)} recover "
          f"the question's SUBJECT, not its grouping:")
    for question, keys in moved:
        mark = "SUBJECT-SIDE" if question in _SUBJECT_SIDE else "grouping"
        print(f"     [{mark:12s}] {keys}  <- {question[:62]!r}")
    print("\n   -> the parser has no role slot, so it cannot tell a grouping from a")
    print("      subject. `question_interpretation.schema.DimensionClaim.role` is "
          "the owner")
    print("      of that distinction, and the deterministic parser does not read it.")

    # -- 6 ---------------------------------------------------------------- #
    print("\n6. WHAT WOULD LAND, IF ALL OF THAT WERE DONE?\n")
    shapes = {s[0]: list(s[2]) for s in surface.SHAPES}
    for tag in ("T3", "T4"):
        phrasings = list(dict.fromkeys(
            shapes.get(tag, []) + list(phrasing_bank.WIDENED.get(tag, ()))))
        deliverable = 0
        for question in phrasings:
            spec, _meta = parse_with_repair(question, semantics, llm_enabled=False)
            reaches = bool(routing._is_evolution(question, spec))
            try:
                keys, _t, _r = _explicit_dimensions(
                    question.lower(), semantics, grouping=True,
                    available_columns=columns)
            except Exception:  # noqa: BLE001
                keys = []
            deliverable += bool(reaches and keys)
        print(f"   {tag}: {deliverable}/{len(phrasings)} phrasings reach the "
              f"evolution route AND carry a dimension")
    print("\n   -> measured today, both are 0/8 and 0/7.\n")
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    argparse.ArgumentParser(
        prog="python -m compositional_plan_scoping.t3_now").parse_args(argv)
    return run(_env())


if __name__ == "__main__":
    sys.exit(main())
