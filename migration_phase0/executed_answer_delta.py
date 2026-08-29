#!/usr/bin/env python3
"""migration_phase0/executed_answer_delta.py — every answer and refusal, executed.

READ-ONLY. Runs every corpus question through the LIVE `/mi/query` path and
records route, grade and answer text. Run it in two trees and diff to get the
answer/refusal movement exactly, rather than inferring it from the contract.

Inferring it would be wrong here: the LEVEL/MOVEMENT delegation changed four
readers, and three of them (the parser's compare trigger, the deterministic
interpreter, the concentration intent classifier) steer ROUTING directly rather
than through the contract. An answer can therefore move on a question whose
contract did not.

    python -m migration_phase0.executed_answer_delta --json out.json [--depth N]
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import tempfile
import warnings
from pathlib import Path
from typing import Any, Dict, List

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from migration_phase0.route_ownership_period_change import (  # noqa: E402
    _grade, _questions, funded_runs)


class DeltaError(RuntimeError):
    """The sweep could not be measured. Never absorbed into an empty diff."""


def run(depth: int = 6, only=None) -> Dict[str, Any]:
    warnings.simplefilter("ignore")
    logging.disable(logging.WARNING)

    tmp = Path(tempfile.mkdtemp())
    out_root = tmp / "onboarding_output"
    from migration_phase0.compound_canary import _write_run
    runs = funded_runs(depth)
    for run_id, rdate, n, scale in runs:
        _write_run(out_root, run_id, rdate, n, scale)
    portfolio, as_of = f"client_001/{runs[-1][0]}", runs[-1][1]

    saved = {k: os.environ.get(k) for k in
             ("MI_AGENT_ONBOARDING_OUTPUT_ROOT", "MI_AGENT_AUTH_ENABLED")}
    os.environ["MI_AGENT_ONBOARDING_OUTPUT_ROOT"] = str(out_root)
    os.environ["MI_AGENT_AUTH_ENABLED"] = "false"
    try:
        from fastapi.testclient import TestClient
        from mi_agent_api.app import app

        client = TestClient(app)
        rows: List[Dict[str, Any]] = []
        questions = _questions()
        if only is not None:
            wanted = set(only)
            questions = [q for q in questions if q in wanted]
        for q in questions:
            try:
                resp = client.post("/mi/query", json={
                    "question": q, "portfolioId": portfolio,
                    "asOfDate": as_of}).json()
            except Exception as exc:  # noqa: BLE001 - re-raised, never absorbed
                raise DeltaError(f"execution failed on {q!r}: {exc!r}") from exc
            meta = resp.get("metadata") or {}
            rows.append({
                "question": q,
                "route": meta.get("route"),
                "ok": bool(resp.get("ok")),
                "grade": _grade(resp),
                "answer": str(resp.get("answer") or "")[:300],
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
        raise DeltaError(
            f"SWEEP INVALID — {len(rows)} readings for {len(questions)} questions")
    return {"rows": rows, "depth": depth}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--json", type=Path, required=True)
    ap.add_argument("--depth", type=int, default=6)
    ap.add_argument("--only", type=Path, default=None,
                    help="a JSON list of questions to execute; the complete set "
                         "of questions that CAN move, computed from the reader "
                         "and contract deltas")
    args = ap.parse_args(argv)
    result = run(args.depth,
                 only=(json.loads(args.only.read_text()) if args.only else None))
    args.json.write_text(json.dumps(result, indent=2, default=str,
                                    sort_keys=True), encoding="utf-8")
    print(f"executed {len(result['rows'])} questions -> {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
