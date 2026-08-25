#!/usr/bin/env python3
"""migration_phase0/contract_role_census.py — before/after interpretation census.

READ-ONLY. Projects every question in the calibration + question corpora and
records each dimension claim's (concept, role). Run it at the base commit and at
HEAD, then diff the two JSON files: the ONLY differences permitted are bridge
questions with a populated ``spec.bridge_dimension`` whose matching claim moved
to ``grouping``. Anything else is a blast-radius violation.

    python -m migration_phase0.contract_role_census --out FILE
    python -m migration_phase0.contract_role_census --diff BEFORE AFTER
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

_CORPORA = (
    "config/mi/golden_questions/ere_mi_calibration_250.yaml",
    "config/mi/golden_questions/ere_mi_questions.yaml",
    "config/mi/golden_questions/business_semantic_questions.yaml",
)


def _questions() -> List[str]:
    import yaml
    out: List[str] = []
    seen = set()

    def walk(o: Any) -> None:
        if isinstance(o, dict):
            q = o.get("question")
            if isinstance(q, str) and q.strip() and q not in seen:
                seen.add(q)
                out.append(q)
            for v in o.values():
                walk(v)
        elif isinstance(o, list):
            for v in o:
                walk(v)

    for rel in _CORPORA:
        p = _REPO / rel
        if p.exists():
            walk(yaml.safe_load(p.read_text()))
    # Bridge questions the corpora do not carry, so the census exercises the
    # intended promotion (and proves it is the ONLY delta) rather than only the
    # untouched majority. `spec.bridge_dimension` is populated for exactly these.
    for q in _BRIDGE_PROBES:
        if q not in seen:
            seen.add(q)
            out.append(q)
    return out


#: Bridge attribution questions, added to the census corpus explicitly.
_BRIDGE_PROBES = (
    "Funded balance bridge by region",
    "Bridge the funded balance by product",
    "balance bridge by LTV band",
    "Bridge the funded balance by region for joint borrowers",
    "Bridge the funded balance by region since March 2026",
)


def _env() -> None:
    import logging
    import warnings
    warnings.simplefilter("ignore")
    os.environ.setdefault("TRAKT_RUNTIME_MODE", "development")
    from demo_platform import config as cfg
    os.environ.update(cfg.mi_env(period_role="current"))
    os.environ["MI_AGENT_LLM_PARSER"] = "off"
    os.environ["MI_AGENT_LLM_ENABLED"] = "0"
    logging.disable(logging.WARNING)


def _snapshot(out_path: str) -> int:
    _env()
    from mi_agent.llm_query_parser import parse_with_repair
    from mi_agent.mi_query_validator import load_mi_semantics
    from mi_agent_api.data_source import semantics_path
    from question_interpretation import projection

    from migration_phase0.assurance_semantics import measurement_failed

    sem = load_mi_semantics(semantics_path())
    rows: List[Dict[str, Any]] = []
    for q in _questions():
        # A parse fault used to be recorded as a row and stepped over, on the
        # reasoning that "a parse fault is data, not a stop". Measured, this
        # corpus produces ZERO faults in normal operation, so a fault is not
        # data — it is the measurement failing. And the error row carried no
        # `dimensions` key, which `_diff` reads through `.get("dimensions", [])`
        # as *zero dimensions*: injecting a parse fault made all 645 questions
        # error, and the census still printed "questions compared: 645 /
        # ILLEGAL deltas (blast): 0".
        #
        # A question that legitimately has no dimensions is a different thing
        # and still records `[]` — 275 of the 645 do.
        try:
            spec, _m = parse_with_repair(q, sem, llm_enabled=False)
            qi = projection.project(q, semantics=sem)
        except Exception as exc:  # noqa: BLE001 - re-raised, never absorbed
            raise measurement_failed("contract_role_census", q, exc) from exc
        rows.append({
            "question": q,
            "bridgeDimension": getattr(spec, "bridge_dimension", None),
            "dimensions": [{"concept": d.candidate_concept, "role": d.role,
                            "source": d.source} for d in qi.dimensions],
            "filters": [{"raw": f.raw_text} for f in qi.filters],
        })
    Path(out_path).write_text(json.dumps(rows, indent=2, default=str) + "\n")
    print(f"{len(rows)} questions projected -> {out_path}")
    return 0


def _diff(before: str, after: str) -> int:
    b = {r["question"]: r for r in json.loads(Path(before).read_text())}
    a = {r["question"]: r for r in json.loads(Path(after).read_text())}
    if set(b) != set(a):
        raise SystemExit(f"CENSUS UNSOUND: corpora differ — "
                         f"only before {sorted(set(b) - set(a))[:3]}, "
                         f"only after {sorted(set(a) - set(b))[:3]}")
    # A census file written by an older build could still carry error rows, and
    # those read as "zero dimensions" here. Refuse to compare them rather than
    # silently counting an un-measured question as an unchanged one.
    unmeasured = sorted([q for q, r in list(b.items()) + list(a.items())
                         if "error" in r])
    if unmeasured:
        raise SystemExit(
            "CENSUS UNSOUND: %d question(s) carry a measurement error and cannot "
            "be compared as evidence, e.g. %s" % (len(unmeasured), unmeasured[:3]))
    if not b:
        raise SystemExit("CENSUS UNSOUND: the census is empty")

    changed: List[Dict[str, Any]] = []
    illegal: List[str] = []
    for q in sorted(b):
        rb, ra = b[q], a[q]
        if rb == ra:
            continue
        # Characterise the delta.
        bd = ra.get("bridgeDimension")
        db = {(d["concept"]): d["role"] for d in rb.get("dimensions", [])}
        da = {(d["concept"]): d["role"] for d in ra.get("dimensions", [])}
        moved = {k: (db.get(k), da.get(k)) for k in set(db) | set(da)
                 if db.get(k) != da.get(k)}
        # Anything other than the dimensions list changing is illegal.
        for field in ("filters", "bridgeDimension"):
            if rb.get(field) != ra.get(field):
                illegal.append(f"{q}: field {field!r} moved "
                               f"{rb.get(field)!r} -> {ra.get(field)!r}")
        # Every moved role must be the bridge dimension, unresolved -> grouping.
        for concept, (old, new) in moved.items():
            legal = (bd is not None and concept == bd
                     and old == "unresolved" and new == "grouping")
            if not legal:
                illegal.append(f"{q}: claim {concept!r} moved {old!r} -> {new!r}"
                               f" (bridgeDimension={bd!r}) — NOT a bridge-role promotion")
        changed.append({"question": q, "bridgeDimension": bd, "rolesMoved": moved})

    print("=" * 78)
    print("CONTRACT-ROLE CENSUS — before vs after")
    print("=" * 78)
    print(f"questions compared          : {len(b)}")
    print(f"questions with any delta    : {len(changed)}")
    print(f"ILLEGAL deltas (blast)      : {len(illegal)}")
    print()
    for c in changed:
        print(f"  CHANGED {c['question']!r}")
        print(f"          bridgeDimension={c['bridgeDimension']!r} "
              f"rolesMoved={c['rolesMoved']}")
    if illegal:
        print("\nBLAST-RADIUS VIOLATIONS:")
        for x in illegal:
            print(f"  {x}")
    print("=" * 78)
    if illegal:
        print("STOP — BLAST RADIUS")
        return 1
    print("Every delta is a bridge question whose matching dimension claim moved "
          "unresolved -> grouping. Nothing else moved.")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out")
    ap.add_argument("--diff", nargs=2, metavar=("BEFORE", "AFTER"))
    args = ap.parse_args(argv)
    if args.diff:
        return _diff(*args.diff)
    if not args.out:
        ap.error("--out or --diff required")
    return _snapshot(args.out)


if __name__ == "__main__":
    sys.exit(main())
