#!/usr/bin/env python3
"""migration_phase0/route_ownership_evolution.py — C6's owned surface, executed.

READ-ONLY. Runs every distinct Stage 1 + Stage 2 corpus question through the
LIVE `/mi/query` path and records, from executed routing rather than from
wording, which questions the shipped `evolution` recogniser family claims, and
which of those actually DELIVER a series with rows in it.

Why this exists
---------------
The C6 four-part matrix failed on delivered coverage: seven of thirty-two owned
questions and two of three route identities could not be exercised at all,
because `pipeline_evolution` returned zero periods against the demo store. An
equivalence measured there is a refusal compared against a refusal.

The five-week fixture supplies the missing data. This instrument re-measures the
same surface against it, so the coverage claim in the C6 matrix is a fresh
number rather than a repeated one.

Coverage is reported in three grades, because "ok" is not "exercised":

  REFUSED   ok=False — the route declined
  EMPTY     ok=True but no artifact carries a row — a controlled non-answer
  DELIVERED ok=True and at least one artifact carries rows — real numbers

Only DELIVERED counts as coverage.

    python -m migration_phase0.route_ownership_evolution [--json out.json]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

CORPORA = ("question_interpretation/stage1_corpus.json",
           "question_interpretation/stage2_corpus.json")

#: The five-week pipeline history built as C6 prerequisite 1.
FIXTURE = _REPO / "tests" / "fixtures" / "pipeline_history_5w"

#: Two governed month-end funded runs straddling the fixture's weeks, so a
#: funded series has two points and the pipeline series five.
FUNDED_RUNS = (("mi_2026_04", "2026-04-30", 60, 1.0),
               ("mi_2026_05", "2026-05-31", 70, 1.15))

EVOLUTION_ROUTES = ("evolution", "evolution_funnel", "evolution_pipeline_stage")


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


def _write_run(root: Path, run_id: str, reporting_date: str, n: int,
               scale: float) -> None:
    import numpy as np
    import pandas as pd
    rng = np.random.default_rng(sum(ord(c) for c in run_id))
    d = root / "client_001" / run_id / "output" / "central"
    d.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({
        "loan_identifier": [f"{run_id}_{i}" for i in range(n)],
        "current_outstanding_balance": (rng.uniform(120_000, 280_000, n) * scale).round(2),
        "current_loan_to_value": rng.uniform(20, 55, n).round(1),
        "current_interest_rate": rng.uniform(3, 8, n).round(2),
        "youngest_borrower_age": rng.integers(62, 88, n),
        "broker_channel": rng.choice(["Alpha", "Beta", "Gamma", "Delta"], n),
        "geographic_region_obligor": rng.choice(["London", "South East", "Scotland"], n),
        "reporting_date": [reporting_date] * n,
    }).to_csv(d / "18_central_lender_tape.csv", index=False)


def _grade(resp: Dict[str, Any]) -> str:
    if not resp.get("ok"):
        return "REFUSED"
    for a in resp.get("artifacts") or []:
        if a.get("rows"):
            return "DELIVERED"
    return "EMPTY"


def run() -> List[Dict[str, Any]]:
    import logging
    import warnings
    warnings.simplefilter("ignore")
    logging.disable(logging.WARNING)

    tmp = Path(tempfile.mkdtemp())
    out_root = tmp / "onboarding_output"
    for run_id, rdate, n, scale in FUNDED_RUNS:
        _write_run(out_root, run_id, rdate, n, scale)
    # MUTATED AND RESTORED. This instrument repoints the governed roots at its
    # own fixture, and it used to leave them repointed: any test that ran after
    # it in the same process saw the five-week fixture as production, which is
    # how `test_stage_temporal_execution_is_fixture_proven_only` came to measure
    # 5 weekly extracts and report that production had acquired data. An
    # assurance instrument that corrupts the environment it measures is not a
    # measurement.
    _saved_env = {k: os.environ.get(k) for k in
                  ("MI_AGENT_ONBOARDING_OUTPUT_ROOT", "MI_AGENT_PIPELINE_ROOT",
                   "MI_AGENT_AUTH_ENABLED")}
    os.environ["MI_AGENT_ONBOARDING_OUTPUT_ROOT"] = str(out_root)
    os.environ["MI_AGENT_PIPELINE_ROOT"] = str(FIXTURE)
    os.environ["MI_AGENT_AUTH_ENABLED"] = "false"

    from fastapi.testclient import TestClient
    from mi_agent_api.app import app
    from mi_agent_api.workspace import resolve_dataset
    from question_interpretation.lexical import pipeline_stage_request

    from migration_phase0.assurance_semantics import (AssuranceMeasurementError,
                                                      measurement_failed)

    client = TestClient(app)
    questions = _questions()
    rows: List[Dict[str, Any]] = []
    for q in questions:
        # An error row carried `route=None`, which makes `owned` False, which
        # drops the question out of the owned-surface denominator without trace.
        # With every query faulting, this printed "OWNED BY THE EVOLUTION FAMILY:
        # 0" and a table of zeros — indistinguishable from a route that owns
        # nothing. A REFUSED answer is a legitimate measurement and still counts;
        # an exception is not an answer.
        try:
            r = client.post("/mi/query", json={
                "question": q, "portfolioId": "client_001/mi_2026_05",
                "asOfDate": "2026-05-31"}).json()
        except Exception as exc:  # noqa: BLE001 - re-raised, never absorbed
            raise measurement_failed("route_ownership_evolution", q, exc) from exc
        route = (r.get("metadata") or {}).get("route")
        low = q.lower()
        rows.append({
            "question": q,
            "route": route,
            "owned": route in EVOLUTION_ROUTES,
            "grade": _grade(r),
            "dataset": resolve_dataset(q),
            # THE GOVERNED READER, not the retired five-substring map. C6
            # deleted `chat_routing._FUNNEL_KEYWORDS`; this instrument was its
            # last consumer, and kept importing it. `pipeline_stage_request` is
            # the one place a stage is read from a question, and it also answers
            # the axis question, so the two hard-coded phrase lists go with it.
            "funnel_word": pipeline_stage_request(q)[0],
            "by_stage_word": ("by stage" if pipeline_stage_request(q)[1] else None),
            "rows": max([len(a.get("rows") or []) for a in (r.get("artifacts") or [])]
                        or [0]),
        })
    # The denominator, asserted: every corpus question must have produced a
    # reading. Zero owned cases is a legitimate finding; zero READINGS is not.
    for key, value in _saved_env.items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value
    if len(rows) != len(questions) or not rows:
        raise AssuranceMeasurementError(
            "ASSURANCE INVALID - measurement failed in route_ownership_evolution: "
            "%d reading(s) for %d corpus question(s)" % (len(rows), len(questions)))
    return rows


def _tally(rows, pred) -> str:
    sel = [r for r in rows if pred(r)]
    d = sum(1 for r in sel if r.get("grade") == "DELIVERED")
    e = sum(1 for r in sel if r.get("grade") == "EMPTY")
    f = sum(1 for r in sel if r.get("grade") == "REFUSED")
    return f"{len(sel):>5} {d:>10} {e:>7} {f:>9}"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--json", type=Path, default=None)
    args = ap.parse_args(argv)

    rows = run()
    owned = [r for r in rows if r.get("owned")]
    print("=" * 84)
    print(f"C6 OWNED SURFACE — {len(rows)} corpus questions executed against the "
          "five-week fixture")
    print("=" * 84)
    print(f"\nOWNED BY THE EVOLUTION FAMILY: {len(owned)}")
    print(f"\n{'partition':<38}{'owned':>5}{'DELIVERED':>11}{'EMPTY':>7}{'REFUSED':>9}")
    print("-" * 70)
    for name, pred in (
        ("all owned", lambda r: True),
        ("  route=evolution", lambda r: r["route"] == "evolution"),
        ("  route=evolution_funnel", lambda r: r["route"] == "evolution_funnel"),
        ("  route=evolution_pipeline_stage",
         lambda r: r["route"] == "evolution_pipeline_stage"),
        ("  dataset=funded", lambda r: r["dataset"] != "pipeline"),
        ("  dataset=pipeline", lambda r: r["dataset"] == "pipeline"),
        ("  funnel-stage vocabulary", lambda r: bool(r["funnel_word"])),
        ("  by-stage vocabulary", lambda r: bool(r["by_stage_word"])),
    ):
        print(f"{name:<38}{_tally(owned, pred)}")

    print("\nDELIVERED PIPELINE-SIDE QUESTIONS (the partition C6 could not exercise)")
    for r in owned:
        if r["dataset"] == "pipeline" and r["grade"] == "DELIVERED":
            print(f"   [{r['route']:<24}] rows={r['rows']}  {r['question']}")
    print("\nSTILL NOT DELIVERED, PIPELINE SIDE")
    for r in owned:
        if r["dataset"] == "pipeline" and r["grade"] != "DELIVERED":
            print(f"   [{r['grade']:<9}] {r['question']}")

    if args.json:
        args.json.write_text(json.dumps(rows, indent=2, default=str), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
