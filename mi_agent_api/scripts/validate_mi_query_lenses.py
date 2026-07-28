#!/usr/bin/env python3
"""Manual validation: the MI Query Agent across Direct, Acquired and Total.

Drives the REAL ``POST /mi/query`` endpoint over a multi-portfolio provenance
frame and prints, per lens, what the agent answered and what scope it declared —
so the two can be checked against each other by eye rather than only by
assertion.

    python -m mi_agent_api.scripts.validate_mi_query_lenses

It also proves the 404 fix: every question is asked twice, once on the bare path
and once under the gateway prefix, and the two answers must agree.

Read-only. Uses the governed synthetic pack with provenance stamped on, so it is
safe to run anywhere the test suite runs (``TRAKT_RUNTIME_MODE=test``).
"""

from __future__ import annotations

import os
import sys
from typing import Any, Dict, List, Optional

os.environ.setdefault("TRAKT_RUNTIME_MODE", "test")
os.environ.setdefault("MI_AGENT_AUTH_ENABLED", "false")

PORTFOLIOS = [
    ("direct_001", "direct", "Direct Book One"),
    ("direct_002", "direct", "Direct Book Two"),
    ("direct_003", "direct", "Direct Book Three"),
    ("acquired_001", "acquired", "Acquired Book One"),
    ("acquired_002", "acquired", "Acquired Book Two"),
]

QUESTIONS = [
    "portfolio summary",
    "what is the total current balance?",
    "what is the weighted average LTV?",
    "show balance by region",
    "where is the largest geographic concentration?",
    "show top 5 regions by balance",
    "show balance over time",
    "how many loans are there?",
    "are we within our risk limits?",
    "what is in the pipeline?",
    "how much protected equity is there?",
]

LENSES: List[Optional[str]] = [
    "total", "direct", "acquired", "direct_001", "acquired_002", "direct_999",
]


def _install_frame():
    from mi_agent_api import data_source as ds_mod
    from mi_agent_api import datasets as datasets_mod

    frame = ds_mod.get_dataframe().copy()
    n = len(frame)
    frame["source_portfolio_id"] = [PORTFOLIOS[i % len(PORTFOLIOS)][0] for i in range(n)]
    frame["source_portfolio_type"] = [PORTFOLIOS[i % len(PORTFOLIOS)][1] for i in range(n)]
    frame["source_portfolio_label"] = [PORTFOLIOS[i % len(PORTFOLIOS)][2] for i in range(n)]
    datasets_mod.get_dataframe = lambda *a, **k: frame
    ds_mod.get_dataframe = lambda *a, **k: frame
    return frame


def _scope(body: Dict[str, Any]) -> str:
    coverage = body.get("portfolioCoverage") or {}
    ids = coverage.get("portfolios_used")
    if ids is None:
        ids = (body.get("portfolioScope") or {}).get("portfolio_ids") or []
    return ", ".join(ids) or "—"


def main() -> int:
    frame = _install_frame()
    from fastapi.testclient import TestClient

    from mi_agent_api import gateway
    from mi_agent_api.app import app

    client = TestClient(app)
    prefix = gateway.api_prefix()

    print(f"MI Query Agent — manual lens validation")
    print(f"rows={len(frame)}  portfolios={[p[0] for p in PORTFOLIOS]}")
    print(f"paths verified: /mi/query and {prefix}/mi/query\n")

    mismatches = 0
    for question in QUESTIONS:
        print("=" * 100)
        print(f"Q: {question}")
        for lens in LENSES:
            body = {"question": question, "sourcePortfolioLens": lens}
            bare = client.post("/mi/query", json=body)
            prefixed = client.post(f"{prefix}/mi/query", json=body)
            answer = bare.json()
            if prefixed.json().get("answer") != answer.get("answer"):
                mismatches += 1
                print(f"  !! {lens}: bare and prefixed paths disagree")
            meta = answer.get("metadata") or {}
            status = "ok " if answer.get("ok") else "REF"
            route = meta.get("route") or "point-in-time"
            applied = meta.get("lensApplied")
            print(f"  [{status}] {lens:<13} route={route:<16} "
                  f"lensApplied={str(applied):<5} scope=({_scope(answer)})")
            text = (answer.get("answer") or answer.get("error") or "").strip()
            print(f"        {text[:150]}")
            for warning in answer.get("warnings") or []:
                if any(k in warning for k in ("Scope not narrowed", "not in the governed",
                                              "portfolio scope applied")):
                    print(f"        · {warning[:150]}")
        print()

    print("=" * 100)
    print(f"path-parity mismatches: {mismatches}")
    return 1 if mismatches else 0


if __name__ == "__main__":
    sys.exit(main())
