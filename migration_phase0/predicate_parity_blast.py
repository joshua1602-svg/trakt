#!/usr/bin/env python3
"""migration_phase0/predicate_parity_blast.py — before/after over the whole corpus.

READ-ONLY. The blast census for the predicate-execution parity change.

Runs every distinct Stage 1 + Stage 2 corpus question through the real chat
entry point and records a stable fingerprint of what the reader would receive.
Run it once before the change and once after, then `--diff` the two files.

The fingerprint deliberately separates the axes the pre-registration authorises
from the axes it does not, so a movement is classified rather than merely
counted:

    interpretation  the contract's own claims — must NOT move at all, because
                    parsing and contract semantics are untouched
    dataset         which governed tape answered
    route           which capability answered
    predicates      the resolved row predicates the contract carries
    answer          the rendered answer / refusal the reader sees
    population      the narrowing ledger and the facet statuses

    python -m migration_phase0.predicate_parity_blast --out before.json
    python -m migration_phase0.predicate_parity_blast --diff before.json after.json
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
import warnings
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

CORPORA = ("question_interpretation/stage1_corpus.json",
           "question_interpretation/stage2_corpus.json")

#: Anything that legitimately differs between two runs of the same code.
_VOLATILE = re.compile(r"(run[_-]?id|generated[_-]?at|timestamp|duration|"
                       r"query[_-]?id|request[_-]?id|trace[_-]?id)", re.I)

AXES = ("interpretation", "dataset", "route", "predicates", "answer", "population")


def _questions() -> List[str]:
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


def _scrub(text: Optional[str]) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def _fingerprint(question: str, result: Dict[str, Any],
                 interpretation: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    metadata = result.get("metadata") or {}
    summary = metadata.get("executionSummary") or {}
    facets = [(f.get("kind"), f.get("field"), f.get("status"))
              for f in (summary.get("facets") or [])]
    claims = (interpretation or {})
    return {
        "question": question,
        "route": metadata.get("route"),
        "dataset": (claims.get("dataset") or {}).get("view")
                   or metadata.get("view") or metadata.get("dataset"),
        "answer": _scrub(result.get("answer")),
        "ok": bool(result.get("ok", True)),
        "population": {
            "ledger": metadata.get("populationApplied"),
            "facets": sorted(str(f) for f in facets),
        },
        "predicates": sorted(
            f"{p.get('field_key')} {p.get('operator')} {p.get('value')}"
            for p in (claims.get("row_predicates") or [])),
        "interpretation": {
            key: claims.get(key) for key in
            ("operation", "subject", "dimensions", "filters", "time", "target",
             "population", "source_scope", "residue")
        } if claims else None,
    }


def capture() -> List[Dict[str, Any]]:
    warnings.simplefilter("ignore")
    os.environ.setdefault("TRAKT_RUNTIME_MODE", "development")
    from demo_platform import config as cfg
    os.environ.update(cfg.mi_env(period_role="current"))
    os.environ["MI_AGENT_LLM_PARSER"] = "off"
    os.environ["MI_AGENT_LLM_ENABLED"] = "0"
    logging.disable(logging.WARNING)

    from migration_phase0.assurance_semantics import load_assurance_semantics
    from mi_agent import execution_receipt as R
    from mi_agent.parsed_question import ParsedQuestion
    from mi_agent_api.mi_service import MiQueryRequest, execute_governed_mi_query
    from question_interpretation import projection as proj
    from trakt_core.context import ExecutionContext

    semantics = load_assurance_semantics()
    ctx = ExecutionContext.for_internal(cfg.CLIENT_ID)
    out: List[Dict[str, Any]] = []
    for question in _questions():
        try:
            result = execute_governed_mi_query(
                MiQueryRequest(question=question), ctx).result or {}
        except Exception as exc:  # noqa: BLE001 - a crash is itself a movement
            result = {"answer": f"__RAISED__ {type(exc).__name__}", "ok": False}
        try:
            spec = ParsedQuestion.parse(question, semantics).spec
            terms = R.requested_dimension_terms(question, semantics, None)
            facets = R.detect_requested_facets(question, semantics, frame=None,
                                               requested_dimensions=terms)
            interpretation = proj.from_parts(question, spec=spec, facets=facets,
                                             dim_terms=terms,
                                             semantics=semantics).as_dict()
        except Exception as exc:  # noqa: BLE001
            interpretation = {"__raised__": f"{type(exc).__name__}: {exc}"}
        out.append(_fingerprint(question, result, interpretation))
    return out


def diff(before: List[Dict[str, Any]], after: List[Dict[str, Any]]) -> int:
    index = {row["question"]: row for row in before}
    moved: Dict[str, List[str]] = {axis: [] for axis in AXES}
    unmatched = 0
    for row in after:
        was = index.get(row["question"])
        if was is None:
            unmatched += 1
            continue
        for axis in AXES:
            if json.dumps(was.get(axis), sort_keys=True, default=str) != \
                    json.dumps(row.get(axis), sort_keys=True, default=str):
                moved[axis].append(row["question"])

    print("=" * 92)
    print(f"PREDICATE PARITY BLAST — {len(after)} questions")
    print("=" * 92)
    for axis in AXES:
        print(f"   {axis:<16} changed: {len(moved[axis])}")
    if unmatched:
        print(f"   questions with no 'before' row: {unmatched}")

    for axis in AXES:
        if not moved[axis]:
            continue
        print(f"\n{axis.upper()} MOVEMENTS ({len(moved[axis])}):")
        for question in moved[axis][:20]:
            was, now = index[question], next(r for r in after
                                             if r["question"] == question)
            print(f"   {question[:60]}")
            print(f"      before: {json.dumps(was.get(axis), default=str)[:150]}")
            print(f"      after : {json.dumps(now.get(axis), default=str)[:150]}")

    print("\nEXPECTED (pre-registered): interpretation 0, dataset 0, route 0, "
          "predicates 0.")
    print("Every answer/population movement must be explained by the "
          "percent, alias, domain or fail-closed class.")
    return 0 if not (moved["interpretation"] or moved["dataset"]
                     or moved["route"] or moved["predicates"]) else 1


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--diff", nargs=2, metavar=("BEFORE", "AFTER"), type=Path)
    args = parser.parse_args(argv)

    if args.diff:
        before = json.loads(args.diff[0].read_text(encoding="utf-8"))
        after = json.loads(args.diff[1].read_text(encoding="utf-8"))
        return diff(before, after)

    rows = capture()
    print(f"captured {len(rows)} questions")
    print("routes: " + ", ".join(
        f"{k}={v}" for k, v in Counter(r["route"] for r in rows).most_common()))
    if args.out:
        args.out.write_text(json.dumps(rows, indent=2, default=str),
                            encoding="utf-8")
        print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
