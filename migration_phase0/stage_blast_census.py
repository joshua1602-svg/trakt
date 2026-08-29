#!/usr/bin/env python3
"""migration_phase0/stage_blast_census.py — executed blast radius of the stage claim.

READ-ONLY. Adding a claim to the interpretation is not a free act: the receipt
reconciles every dimension claim, and a claim stamped FILTER makes an answer look
NARROWED. A question that acquires an unsatisfiable narrowing is refused. So the
blast of a contract change is measured on EXECUTED ANSWERS, not on the claims.

Run before and after; compare by exact question.

    python -m migration_phase0.stage_blast_census --json out.json
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

CORPORA = ("question_interpretation/stage1_corpus.json",
           "question_interpretation/stage2_corpus.json")
FIXTURE = _REPO / "tests" / "fixtures" / "pipeline_history_5w"
FUNDED_RUNS = (("mi_2026_04", "2026-04-30", 60, 1.0),
               ("mi_2026_05", "2026-05-31", 70, 1.15))


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


def _write_run(root: Path, run_id: str, rdate: str, n: int, scale: float) -> None:
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
        "reporting_date": [rdate] * n,
    }).to_csv(d / "18_central_lender_tape.csv", index=False)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--json", type=Path, required=True)
    args = ap.parse_args(argv)

    warnings.simplefilter("ignore")
    logging.disable(logging.WARNING)
    tmp = Path(tempfile.mkdtemp())
    out_root = tmp / "onboarding_output"
    for a in FUNDED_RUNS:
        _write_run(out_root, *a)
    os.environ["MI_AGENT_ONBOARDING_OUTPUT_ROOT"] = str(out_root)
    os.environ["MI_AGENT_PIPELINE_ROOT"] = str(FIXTURE)
    os.environ["MI_AGENT_AUTH_ENABLED"] = "false"

    from fastapi.testclient import TestClient
    from mi_agent_api.app import app
    from mi_agent_api.workspace import resolve_dataset
    client = TestClient(app)

    rows: List[Dict[str, Any]] = []
    for q in _corpus():
        r = client.post("/mi/query", json={
            "question": q, "portfolioId": "client_001/mi_2026_05",
            "asOfDate": "2026-05-31"}).json()
        md = r.get("metadata") or {}
        rows.append({
            "question": q,
            "ok": r.get("ok"),
            "route": md.get("route"),
            "dataset": resolve_dataset(q),
            "answer_head": (r.get("answer") or "")[:160],
            "warnings": [str(w)[:120] for w in (r.get("warnings") or [])],
            "n_rows": max([len(a.get("rows") or []) for a in (r.get("artifacts") or [])] or [0]),
        })
    args.json.write_text(json.dumps(rows, indent=2, default=str), encoding="utf-8")
    print(f"captured {len(rows)} executed answers -> {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
