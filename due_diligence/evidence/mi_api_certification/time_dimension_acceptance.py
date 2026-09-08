#!/usr/bin/env python3
"""TIME x DIMENSION live acceptance — evidence collection, nothing else.

WHY THIS IS NOT A SECOND CERTIFICATION CLIENT. It asks its questions through
`certify_mi_api._live_asker` and establishes the build through
`certify_mi_api.preflight`, so this repository keeps ONE live transport, ONE
`MI_BEARER` convention, ONE reachability/auth classifier and ONE deployed-SHA
provenance check. What it adds is READING: the core suite scores a question as
answered or refused, which cannot adjudicate whether the grouped dimension was
region, whether a predicate ran in every period, or whether the latest temporal
period equals the ordinary current-period answer cell for cell.

It changes nothing and asserts nothing about the service. It prints what came
back, so a human verdict rests on structural execution evidence rather than on
answer prose.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

_HERE = Path(__file__).resolve()
_REPO = _HERE.parent.parent.parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from due_diligence.evidence.mi_api_certification.certify_mi_api import (  # noqa: E402
    _live_asker, preflight,
)

#: The acceptance matrix. `expect` is what the BRIEF requires, and is recorded
#: beside the observation so a reader can see the expectation that was set
#: before the run rather than one fitted to it afterwards.
MATRIX = (
    ("T01", "Show funded balance over time", "ANSWER"),
    ("T02", "Show funded balance by region over time", "ANSWER"),
    ("T03", "Show loan count by LTV bucket over time", "ANSWER"),
    ("T04", "Show funded balance by region over time where LTV is over 50%", "ANSWER"),
    ("T05", "Show funded balance over time for Scotland", "ANSWER"),
    ("T06", "Show funded balance by region over time for the front book", "ANSWER"),
    ("T07", "Show funded balance by property region over time", "ANSWER"),
    ("T08", "Show funded balance by borrower region over time", "REFUSAL-OR-PROVEN-BORROWER"),
    ("T09", "Show platinum balance by region over time", "REFUSAL"),
    ("T10", "Show funded balance by region as at 31 May 2026", "REFUSAL-IF-UNSUPPORTED"),
)

#: (temporal, current) pairs for the latest-period parity gate.
PARITY = (
    ("P01", "Show funded balance by region over time",
     "Show funded balance by region"),
    ("P02", "Show loan count by LTV bucket over time",
     "Show loan count by LTV bucket"),
    ("P03", "Show funded balance over time for Scotland",
     "Show funded balance for Scotland"),
    ("P04", "Show funded balance by region over time where LTV is over 50%",
     "Show funded balance by region where LTV is over 50%"),
)


def _artifacts(env: Dict[str, Any], kind: str) -> List[Dict[str, Any]]:
    return [a for a in (env.get("artifacts") or []) if a.get("type") == kind]


def _meta(env: Dict[str, Any]) -> Dict[str, Any]:
    return env.get("metadata") or {}


def _periods(env: Dict[str, Any]) -> List[str]:
    """Period labels, in published order, from whichever artifact carries them."""
    for art in (env.get("artifacts") or []):
        rows = art.get("rows") or []
        labels = [str(r.get("period")) for r in rows if r.get("period") is not None]
        if labels:
            seen: List[str] = []
            for lab in labels:
                if lab not in seen:
                    seen.append(lab)
            return seen
    return []


def _grouped_latest(env: Dict[str, Any], group_key: str) -> Dict[str, float]:
    """``{category: value}`` for the LAST published period of a grouped series."""
    tables = _artifacts(env, "table")
    if not tables or not group_key:
        return {}
    rows = tables[0].get("rows") or []
    periods = _periods(env)
    if not periods:
        return {}
    last = periods[-1]
    out: Dict[str, float] = {}
    for row in rows:
        if str(row.get("period")) != last:
            continue
        cat = row.get(group_key)
        val = row.get("value")
        if cat is not None and val is not None:
            out[str(cat)] = float(val)
    return out


def _current_grouped(env: Dict[str, Any]) -> Dict[str, float]:
    """``{category: value}`` from an ordinary current-period grouped answer."""
    tables = _artifacts(env, "table")
    if not tables:
        return {}
    table = tables[0]
    cols = [c.get("key") for c in (table.get("columns") or [])]
    if not cols:
        rows = table.get("rows") or []
        cols = list(rows[0]) if rows else []
    group_key = cols[0] if cols else None
    value_key = next((c for c in cols
                      if isinstance(c, str)
                      and (c.endswith("_sum") or c == "loan_count")
                      and c != group_key), None)
    if not (group_key and value_key):
        return {}
    return {str(r[group_key]): float(r[value_key])
            for r in (table.get("rows") or [])
            if r.get(group_key) is not None and r.get(value_key) is not None}


def _series_values(env: Dict[str, Any]) -> List[Optional[float]]:
    for art in (env.get("artifacts") or []):
        rows = art.get("rows") or []
        if rows and all("value" in r for r in rows) and "period" in (rows[0] or {}):
            return [None if r.get("value") is None else float(r["value"]) for r in rows]
    return []


def _scalar(env: Dict[str, Any]) -> Optional[float]:
    """The single figure a current-period scalar answer carries."""
    recon = env.get("reconciliation") or {}
    for key in ("balance_after_filters", "balance_included"):
        if recon.get(key) is not None:
            return float(recon[key])
    return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", required=True)
    ap.add_argument("--path", default="/mi/query")
    ap.add_argument("--portfolio-id", default=None)
    ap.add_argument("--expect-commit", default=None)
    ap.add_argument("--json-out", default=None)
    args = ap.parse_args()

    ask = _live_asker(args.base_url, args.path, [], args.portfolio_id)
    reached, authorised, commit = preflight(
        args.base_url, args.path, [], args.portfolio_id)
    print("=" * 74)
    print("TIME x DIMENSION — LIVE ACCEPTANCE")
    print(f"target      : {args.base_url.rstrip('/')}{args.path}")
    print(f"portfolio   : {args.portfolio_id}")
    print(f"reached     : {reached}")
    print(f"authorised  : {authorised}")
    print(f"deployed    : {commit or 'NOT ESTABLISHED'}")
    print(f"expected    : {args.expect_commit or '(not supplied)'}")
    match = bool(commit and args.expect_commit and commit == args.expect_commit)
    print(f"SHA match   : {'YES' if match else 'NO'}")
    print("=" * 74)
    if not str(authorised).startswith("YES"):
        print("VERDICT: NOT EXECUTABLE — auth")
        return 2
    if not match:
        print("VERDICT: NOT EXECUTABLE — provenance")
        return 2

    record: Dict[str, Any] = {"deployed": commit, "matrix": {}, "parity": {}}
    latencies: List[float] = []

    for qid, question, expect in MATRIX:
        started = time.perf_counter()
        env = ask(question)
        elapsed = time.perf_counter() - started
        meta = _meta(env)
        grouped_by = meta.get("groupedBy") or []
        group_key = str(grouped_by[0]) if grouped_by else ""
        pop = meta.get("populationApplied") or {}
        row = {
            "question": question, "expect": expect,
            "ok": env.get("ok"), "route": meta.get("route"),
            "latency_s": round(elapsed, 3),
            "answer": (env.get("answer") or env.get("error") or "")[:200],
            "groupedBy": grouped_by,
            "geographyBasis": meta.get("geographyBasis"),
            "periods": _periods(env),
            "executed": [{k: e.get(k) for k in
                          ("field", "canonical_field", "op", "values")}
                         for e in (pop.get("executed") or [])],
            "populationApplied": pop.get("applied"),
            "latest": _grouped_latest(env, group_key) if group_key else None,
            "series": None if group_key else _series_values(env),
        }
        record["matrix"][qid] = row
        if env.get("ok"):
            latencies.append(elapsed)
        print(f"\n--- {qid}  {question}")
        print(f"    expect={expect}  ok={row['ok']}  route={row['route']}  "
              f"{row['latency_s']}s")
        print(f"    answer: {row['answer'][:160]}")
        print(f"    groupedBy={row['groupedBy']}  basis={row['geographyBasis']}")
        print(f"    periods={row['periods']}")
        print(f"    executed={json.dumps(row['executed'])}")
        if row["latest"] is not None:
            print(f"    latest={json.dumps(row['latest'], sort_keys=True)}")
        if row["series"]:
            print(f"    series={row['series']}")

    print("\n" + "=" * 74)
    print("LATEST-PERIOD PARITY")
    for pid, temporal_q, current_q in PARITY:
        t_env = ask(temporal_q)
        c_env = ask(current_q)
        t_meta, c_meta = _meta(t_env), _meta(c_env)
        grouped_by = t_meta.get("groupedBy") or []
        group_key = str(grouped_by[0]) if grouped_by else ""
        if group_key:
            t_vals = _grouped_latest(t_env, group_key)
            c_vals = _current_grouped(c_env)
        else:
            series = _series_values(t_env)
            t_vals = {"__scalar__": series[-1]} if series else {}
            c_scalar = _scalar(c_env)
            c_vals = {"__scalar__": c_scalar} if c_scalar is not None else {}
        keys = sorted(set(t_vals) | set(c_vals))
        mismatches = []
        for key in keys:
            tv, cv = t_vals.get(key), c_vals.get(key)
            if tv is None or cv is None or abs(float(tv) - float(cv)) > 0.011:
                mismatches.append({"category": key, "temporal": tv, "current": cv})
        record["parity"][pid] = {
            "temporal": temporal_q, "current": current_q,
            "temporal_ok": t_env.get("ok"), "current_ok": c_env.get("ok"),
            "cells": len(keys), "mismatches": mismatches,
            "temporal_values": t_vals, "current_values": c_vals,
        }
        print(f"\n--- {pid}  cells={len(keys)}  mismatches={len(mismatches)}  "
              f"temporal_ok={t_env.get('ok')} current_ok={c_env.get('ok')}")
        print(f"    temporal={json.dumps(t_vals, sort_keys=True)}")
        print(f"    current ={json.dumps(c_vals, sort_keys=True)}")
        if mismatches:
            print(f"    MISMATCH {json.dumps(mismatches)}")

    if latencies:
        ordered = sorted(latencies)
        median = ordered[len(ordered) // 2]
        print(f"\nlatency: median {median:.2f}s  slowest {max(ordered):.2f}s  "
              f"n={len(ordered)}")
        record["latency"] = {"median_s": round(median, 3),
                             "slowest_s": round(max(ordered), 3),
                             "n": len(ordered)}
    if args.json_out:
        Path(args.json_out).write_text(json.dumps(record, indent=1, default=str),
                                       encoding="utf-8")
    print("\nEVIDENCE COLLECTED — verdict is adjudicated by the operator.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
