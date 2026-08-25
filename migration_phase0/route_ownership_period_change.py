#!/usr/bin/env python3
"""migration_phase0/route_ownership_period_change.py — C7's owned surface, executed.

READ-ONLY. Runs every distinct Stage 1 + Stage 2 corpus question through the LIVE
`/mi/query` path and records, from EXECUTED ROUTING rather than from wording,
which questions the shipped `period_change` family claims and which of those
actually deliver.

The C6 non-vacuity rule is carried forward unchanged and is permanent:

  REFUSED   ok=False — the route declined
  EMPTY     ok=True but no artifact carries a row, or an insufficient-data
            warning stands — a controlled non-answer
  DELIVERED ok=True and at least one artifact carries rows — real numbers

Only DELIVERED counts as coverage.

TWO ROUTE LABELS, ONE ROUTE. `period_change_route` publishes `period_change_analysis`
(ROUTE_NAME = WORKFLOW_ID) from every ordinary path, and the bare string
`period_change` from the span-clarification envelope. Both are this route, and a
census that counted only the first would undercount the owned surface and miss
the entire clarification partition.

    python -m migration_phase0.route_ownership_period_change [--json out.json]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

CORPORA = ("question_interpretation/stage1_corpus.json",
           "question_interpretation/stage2_corpus.json")

#: Both labels the one route publishes. See the module docstring.
PERIOD_CHANGE_ROUTES = ("period_change_analysis", "period_change")

#: Month-end governed runs. The census is measured at TWO depths on purpose.
#:
#: A two-snapshot book is the production-shaped case and is what C6's fixture
#: ruling calls the honest default. A six-snapshot book is the control: it
#: separates "this route refused because the book is thin" from "this route
#: refused because it cannot answer the question", and C6's recorded correction
#: — the denominator was wrong, not the measurement — is why that control is
#: taken before any vacuity is claimed.
_MONTH_ENDS = [("mi_2026_01", "2026-01-31"), ("mi_2026_02", "2026-02-28"),
               ("mi_2026_03", "2026-03-31"), ("mi_2026_04", "2026-04-30"),
               ("mi_2026_05", "2026-05-31"), ("mi_2026_06", "2026-06-30")]


def funded_runs(depth: int):
    """The last ``depth`` month-end runs, oldest first."""
    chosen = _MONTH_ENDS[-depth:]
    return tuple((rid, date, 60 + 4 * i, round(1.0 + 0.06 * i, 4))
                 for i, (rid, date) in enumerate(chosen))


class CensusMeasurementError(RuntimeError):
    """The census could not be measured. Never absorbed into a zero."""


def _questions() -> List[str]:
    out: List[str] = []
    seen = set()
    for f in CORPORA:
        for row in json.loads((_REPO / f).read_text())["rows"]:
            q = row.get("question") or ""
            if q and q not in seen:
                seen.add(q)
                out.append(q)
    return out


def _grade(resp: Dict[str, Any]) -> str:
    if not resp.get("ok"):
        return "REFUSED"
    for warning in resp.get("warnings") or []:
        if "insufficient" in str(warning).lower():
            return "EMPTY"
    for a in resp.get("artifacts") or []:
        if a.get("rows"):
            return "DELIVERED"
    return "EMPTY"


def run(depth: int = 2) -> List[Dict[str, Any]]:
    import logging
    import warnings
    warnings.simplefilter("ignore")
    logging.disable(logging.WARNING)

    tmp = Path(tempfile.mkdtemp())
    out_root = tmp / "onboarding_output"
    from migration_phase0.compound_canary import _write_run
    runs = funded_runs(depth)
    for run_id, rdate, n, scale in runs:
        _write_run(out_root, run_id, rdate, n, scale)
    portfolio = f"client_001/{runs[-1][0]}"
    as_of = runs[-1][1]

    # MUTATED AND RESTORED — an assurance instrument that leaves the governed
    # roots repointed corrupts every test that runs after it in the process.
    saved = {k: os.environ.get(k) for k in
             ("MI_AGENT_ONBOARDING_OUTPUT_ROOT", "MI_AGENT_AUTH_ENABLED")}
    os.environ["MI_AGENT_ONBOARDING_OUTPUT_ROOT"] = str(out_root)
    os.environ["MI_AGENT_AUTH_ENABLED"] = "false"
    try:
        from fastapi.testclient import TestClient
        from mi_agent_api.app import app
        from mi_agent import period_request as period_request_mod
        from mi_agent.period_change import rank_request as rank_mod
        from mi_agent_api.period_change_route import _rank_subject

        client = TestClient(app)
        questions = _questions()
        rows: List[Dict[str, Any]] = []
        for q in questions:
            # A faulting query used to carry route=None, which makes `owned`
            # False, which drops the question out of the denominator without
            # trace. A REFUSED answer is a measurement; an exception is not.
            try:
                resp = client.post("/mi/query", json={
                    "question": q, "portfolioId": portfolio,
                    "asOfDate": as_of}).json()
            except Exception as exc:  # noqa: BLE001 - re-raised, never absorbed
                raise CensusMeasurementError(
                    f"census failed on {q!r}: {exc!r}") from exc
            meta = resp.get("metadata") or {}
            route = meta.get("route")
            rm = meta.get("rankedMovement") or {}
            span = period_request_mod.requested_span(q)
            rows.append({
                "question": q,
                "route": route,
                "owned": route in PERIOD_CHANGE_ROUTES,
                "grade": _grade(resp),
                # The three route-local reads, recorded for the semantic-owner
                # inventory so the census and the inventory share one denominator.
                "has_rank_language": rank_mod.has_rank_language(q),
                "rank_subject": _rank_subject(q),
                "requested_span": (span.label if span else None),
                "ranked_applied": bool(rm.get("applied")),
                "ranked_field": rm.get("canonicalField"),
                "refusal_reason": rm.get("reason"),
                "lens_applied": meta.get("lensApplied"),
                "rows": max([len(a.get("rows") or [])
                             for a in (resp.get("artifacts") or [])] or [0]),
            })
    finally:
        for key, value in saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value

    if len(rows) != len(questions) or not rows:
        raise CensusMeasurementError(
            f"CENSUS INVALID — {len(rows)} reading(s) for {len(questions)} question(s)")
    return rows


def _tally(rows, pred) -> str:
    sel = [r for r in rows if pred(r)]
    d = sum(1 for r in sel if r["grade"] == "DELIVERED")
    e = sum(1 for r in sel if r["grade"] == "EMPTY")
    f = sum(1 for r in sel if r["grade"] == "REFUSED")
    return f"{len(sel):>6}{d:>11}{e:>7}{f:>9}"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--json", type=Path, default=None)
    ap.add_argument("--depth", type=int, default=2,
                    help="governed month-end snapshots in the book (2 or 6)")
    args = ap.parse_args(argv)

    rows = run(args.depth)
    owned = [r for r in rows if r["owned"]]
    print("=" * 84)
    print(f"C7 OWNED SURFACE — {len(rows)} corpus questions executed against a "
          f"{args.depth}-snapshot book")
    print("=" * 84)
    print(f"\nOWNED BY THE PERIOD_CHANGE FAMILY: {len(owned)}")
    print(f"\n{'partition':<38}{'owned':>6}{'DELIVERED':>11}{'EMPTY':>7}{'REFUSED':>9}")
    print("-" * 71)
    for name, pred in (
        ("all owned", lambda r: True),
        ("  label=period_change_analysis",
         lambda r: r["route"] == "period_change_analysis"),
        ("  label=period_change (clarify)",
         lambda r: r["route"] == "period_change"),
        ("  rank language present", lambda r: r["has_rank_language"]),
        ("    ranking APPLIED", lambda r: r["ranked_applied"]),
        ("    ranking requested, refused",
         lambda r: r["has_rank_language"] and not r["ranked_applied"]),
        ("  narrative (no rank language)", lambda r: not r["has_rank_language"]),
        ("  names a span", lambda r: bool(r["requested_span"])),
        ("  lens applied", lambda r: bool(r["lens_applied"])),
    ):
        print(f"{name:<38}{_tally(owned, pred)}")

    print("\nROUTES THAT TOOK RANK-LANGUAGE QUESTIONS AWAY FROM period_change")
    away: Dict[Any, int] = {}
    for r in rows:
        if r["has_rank_language"] and not r["owned"]:
            away[r["route"]] = away.get(r["route"], 0) + 1
    for route, n in sorted(away.items(), key=lambda kv: -kv[1]):
        print(f"   {str(route):<32} {n}")

    print("\nREFUSAL REASONS ON THE OWNED SURFACE")
    reasons: Dict[Any, int] = {}
    for r in owned:
        if r["refusal_reason"]:
            reasons[r["refusal_reason"]] = reasons.get(r["refusal_reason"], 0) + 1
    for reason, n in sorted(reasons.items(), key=lambda kv: -kv[1]):
        print(f"   {str(reason):<32} {n}")

    if args.json:
        args.json.write_text(json.dumps(rows, indent=2, default=str),
                             encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
