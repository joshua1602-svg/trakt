#!/usr/bin/env python3
"""PHASE 7 — VERTICAL NON-REGRESSION, on the ROUTED path, offline.

The corpus census proves what a question is UNDERSTOOD as. This proves what
the estate ANSWERS, end to end, through the real FastAPI app against the demo
book — for the five verticals the recovery sprint names:

    TIME x DIMENSION      a measure, an axis and a period, in every combination
    BORROWING_BASE        the facility capability, incl. its no-facility refusal
    PORTFOLIO_COMPARISON  two governed populations set against each other
    PIPELINE_MOVEMENT     stage transitions, arrivals, departures, stayers
    RISK_CONCENTRATION    governed limits and concentration tests

Every question here is AUTHORED FOR THIS SUITE. None is copied from the frozen
135-question bank, and no expected answer is recorded: the suite captures what
the estate does and the DIFF between two revisions is the evidence. A movement
is not a failure — it is something that must be explained.

    python vertical_suites.py run --out before.json
    python vertical_suites.py diff before.json after.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import warnings
from pathlib import Path
from typing import Any, Dict, List

SUITES: Dict[str, List[str]] = {
    "TIME_X_DIMENSION": [
        "What is the total funded balance?",
        "What is the funded balance by region?",
        "What is the weighted average LTV by region?",
        "Show the funded balance by month.",
        "Show the loan count by region over time.",
        "How has the funded balance moved?",
        "How has the weighted average LTV changed month on month?",
        "Bridge the movement in the funded balance by region.",
        "What was the funded balance at the previous reporting date?",
        "Compare the funded balance between October and November.",
        "Show the weighted average interest rate by broker channel.",
        "What is the average borrower age by region?",
        "Which region has the largest funded balance?",
        "Which region grew the most since the prior reporting date?",
        "Show the LTV distribution of the book.",
        "What is the balance by LTV band?",
    ],
    "BORROWING_BASE": [
        "What is the borrowing base?",
        "How much headroom is there under the borrowing base?",
        "What is the facility utilisation?",
        "How much collateral value are we able to borrow against?",
        "How much can we draw against the facility?",
        "Why are loans ineligible?",
        "What is the eligible collateral balance?",
        "Show the borrowing base by region.",
        "How has the borrowing base moved since the prior reporting date?",
    ],
    "PORTFOLIO_COMPARISON": [
        "Summarise the direct book.",
        "Summarise the acquired book.",
        "How does the direct book compare with the acquired book?",
        "What is the funded balance for the direct book?",
        "Compare the weighted average LTV of the direct and acquired books.",
        "How does the front book compare with the back book?",
        "Which source portfolio has the highest weighted average LTV?",
        "What is the balance by source portfolio?",
    ],
    "PIPELINE_MOVEMENT": [
        "How many cases are in the pipeline?",
        "What is the pipeline balance by stage?",
        "How many cases moved from KFI into Application?",
        "How many cases arrived into Offer?",
        "How many cases left the pipeline?",
        "How many cases stayed at Offer?",
        "What is the conversion rate from Offer to Completion?",
        "Show the pipeline case count over time.",
    ],
    "RISK_CONCENTRATION": [
        "Which concentration limits are we closest to?",
        "Are we in breach of any limits?",
        "What is the geographic concentration?",
        "Show the single largest loan against the large-loan limit.",
        "What is the broker concentration?",
        "How much of the book is in the top 3 regions?",
        "What is the weighted average LTV against the LTV limit?",
    ],
}


def _env() -> None:
    warnings.simplefilter("ignore")
    os.environ["TRAKT_RUNTIME_MODE"] = "development"
    from demo_platform import config as cfg
    os.environ.update(cfg.mi_env(period_role="current"))
    os.environ["MI_AGENT_LLM_PARSER"] = "off"
    os.environ["MI_AGENT_LLM_ENABLED"] = "0"
    os.environ["MI_AGENT_AUTH_ENABLED"] = "false"


def _shape(answer: str) -> str:
    """The answer's SHAPE, not its wording: a figure's presence and kind. Two
    revisions that both say "£5.4MM · 36 loans" agree; a revision that starts
    naming a different unit has moved, and the diff must show it."""
    import re
    out = []
    if re.search(r"£[\d,.]+", answer or ""):
        out.append("currency")
    if re.search(r"\d+(?:\.\d+)?%", answer or ""):
        out.append("percent")
    if re.search(r"\b\d[\d,]*\s+loans?\b", answer or ""):
        out.append("count")
    return "+".join(out) or "text"


def run() -> List[Dict[str, Any]]:
    _env()
    from fastapi.testclient import TestClient
    from mi_agent_api.app import app

    client = TestClient(app)
    rows: List[Dict[str, Any]] = []
    for suite, questions in SUITES.items():
        view = "pipeline" if suite == "PIPELINE_MOVEMENT" else "funded"
        for i, question in enumerate(questions, start=1):
            try:
                r = client.post("/mi/query",
                                json={"question": question, "view": view}).json()
            except Exception as exc:  # noqa: BLE001 - a fault is a result
                r = {"ok": False, "error": "%s: %s" % (type(exc).__name__, exc)}
            meta = r.get("metadata") or {}
            spec = r.get("spec") or {}
            answer = str(r.get("answer") or r.get("error") or "")
            rows.append({
                "suite": suite, "id": "%s_%02d" % (suite[:4], i),
                "question": question, "view": view,
                "ok": bool(r.get("ok")),
                "route": meta.get("route"),
                "controlled_refusal": bool(r.get("controlledRefusal")),
                "error_code": meta.get("errorCode"),
                "metric": spec.get("metric"),
                "aggregation": spec.get("aggregation"),
                "weight_field": spec.get("weight_field"),
                "dimension": spec.get("dimension"),
                "filters": sorted((spec.get("filters") or {}).keys()),
                "unavailable_filters": list(spec.get("unavailable_filters") or []),
                "artifacts": len(r.get("artifacts") or []),
                "answer_shape": _shape(answer),
                "answer": answer[:400],
            })
    return rows


_COMPARED = ("ok", "route", "controlled_refusal", "metric", "aggregation",
             "weight_field", "dimension", "filters", "unavailable_filters",
             "artifacts", "answer_shape", "answer")


def diff(before: List[Dict[str, Any]], after: List[Dict[str, Any]]) -> int:
    by_id = {r["id"]: r for r in before}
    moved = 0
    for row in after:
        old = by_id.get(row["id"])
        if old is None:
            continue
        changes = {k: (old.get(k), row.get(k)) for k in _COMPARED
                   if old.get(k) != row.get(k)}
        if not changes:
            continue
        moved += 1
        print("  %s [%s] %s" % (row["id"], row["suite"], row["question"]))
        for key, (a, b) in changes.items():
            if key == "answer":
                print("      answer: %r" % (a,))
                print("           -> %r" % (b,))
            else:
                print("      %s: %r -> %r" % (key, a, b))
    print("compared %d, moved %d" % (len(after), moved))
    return moved


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run"); r.add_argument("--out", required=True)
    d = sub.add_parser("diff"); d.add_argument("before"); d.add_argument("after")
    args = ap.parse_args(argv)
    if args.cmd == "run":
        rows = run()
        Path(args.out).write_text(json.dumps(rows, indent=1, default=str),
                                  encoding="utf-8")
        ok = sum(1 for x in rows if x["ok"])
        print("vertical suites: %d questions -> %s (%d answered, %d refused)"
              % (len(rows), args.out, ok, len(rows) - ok))
        return 0
    diff(json.loads(Path(args.before).read_text(encoding="utf-8")),
         json.loads(Path(args.after).read_text(encoding="utf-8")))
    return 0


if __name__ == "__main__":
    sys.exit(main())
