#!/usr/bin/env python3
"""question_interpretation/series_emission.py

Does anything in the system return a SERIES — the run of values — or only a
sentence about one?

Every successful time-series question answers with "Funded balance over N
period(s): latest £X (up over the window)". That is one number and a direction.
If the run of values exists nowhere, then trends as a capability does not exist,
Stage 5's five changes were never scoped to build it, and the refusal
improvements must not be allowed to stand in for it.

The answer text is not the whole payload, so this looks at the whole payload:
every artifact a routed answer carries, its row count, and whether those rows
carry a period and a value. An artifact with three `{period, value}` rows IS the
series, whatever the prose says.

    python -m question_interpretation.series_emission
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import warnings
from pathlib import Path
from typing import Any, Dict, List

warnings.simplefilter("ignore")

_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

QUESTIONS: List[Dict[str, str]] = [
    {"id": "E1", "question": "funded balance by month", "expect_route": "evolution"},
    {"id": "E2", "question": "balance by month over the last 3 months", "expect_route": "evolution"},
    {"id": "E3", "question": "balance by month by region", "expect_route": "evolution"},
    {"id": "E4", "question": "show me the funded balance trend", "expect_route": "evolution"},
    {"id": "M1", "question": "how has the funded balance moved since last month", "expect_route": "period_movement"},
    {"id": "F1", "question": "run-rate forecast for the next quarter", "expect_route": "forecast_extrapolation"},
    {"id": "P1", "question": "show the pipeline funnel over time", "expect_route": "evolution_funnel"},
    {"id": "C1", "question": "how have the front book and back book changed over time", "expect_route": "analytical_composition"},
]


def _series_rows(artifact: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Rows that carry BOTH a period and a value — the run of values itself."""
    out = []
    for row in (artifact.get("rows") or []):
        if not isinstance(row, dict):
            continue
        period = row.get("period") or row.get("reporting_date") or row.get("date")
        if period is None:
            continue
        values = [k for k, v in row.items()
                  if k not in ("period", "reporting_date", "date")
                  and isinstance(v, (int, float))]
        if values:
            out.append(row)
    return out


def observe(question: str) -> Dict[str, Any]:
    from demo_platform import config as cfg
    os.environ.update(cfg.mi_env(period_role="current"))
    os.environ["MI_AGENT_LLM_PARSER"] = "off"
    os.environ["MI_AGENT_LLM_ENABLED"] = "0"
    import logging
    logging.disable(logging.WARNING)

    from mi_agent_api.mi_service import MiQueryRequest, execute_governed_mi_query
    from trakt_core.context import ExecutionContext

    res = (execute_governed_mi_query(
        MiQueryRequest(question=question),
        ExecutionContext.for_internal(cfg.CLIENT_ID)).result or {})
    meta = res.get("metadata") or {}
    artifacts = []
    for a in (res.get("artifacts") or []):
        rows = _series_rows(a)
        artifacts.append({
            "type": a.get("type"),
            "chart_type": a.get("chartType"),
            "total_rows": len(a.get("rows") or []),
            "series_rows": len(rows),
            "sample": rows[:3],
            "columns": a.get("columns"),
        })
    return {
        "route": meta.get("route") or "(none — point-in-time)",
        "ok": res.get("ok"),
        "answer": str(res.get("answer") or res.get("error") or "")[:180],
        "artifacts": artifacts,
        "emits_series": any(a["series_rows"] >= 2 for a in artifacts),
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--json", type=Path, default=None)
    args = ap.parse_args(argv)

    rows = []
    for case in QUESTIONS:
        row = observe(case["question"])
        row.update(id=case["id"], question=case["question"],
                   expect_route=case["expect_route"])
        rows.append(row)

    print("=" * 84)
    print("DOES ANYTHING RETURN A SERIES?")
    print("=" * 84)
    for r in rows:
        print("\n%-4s %s" % (r["id"], r["question"]))
        print("     route=%s ok=%s  EMITS SERIES: %s"
              % (r["route"], r["ok"], "YES" if r["emits_series"] else "no"))
        print("     answer: %s" % r["answer"].replace("\n", " ")[:140])
        for a in r["artifacts"]:
            print("     artifact type=%-8s chart=%-6s rows=%-3d series_rows=%d"
                  % (a["type"], a["chart_type"], a["total_rows"], a["series_rows"]))
            for sample in a["sample"]:
                print("        %s" % sample)
    emitting = [r for r in rows if r["emits_series"]]
    print("\n" + "-" * 84)
    print("questions whose payload carries the run of values: %d of %d"
          % (len(emitting), len(rows)))
    print("routes that emit a series: %s"
          % (sorted({r["route"] for r in emitting}) or "NONE"))

    if args.json:
        args.json.write_text(json.dumps(rows, indent=2, default=str),
                             encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
