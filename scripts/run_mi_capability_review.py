#!/usr/bin/env python3
"""scripts/run_mi_capability_review.py

Run the business-semantic MI question bank through THE governed MI capability —
``mi_agent_api.mi_service.execute_governed_mi_query`` — which is the same
entrypoint POST /mi/query (React MI Agent) and the Microsoft 365 Copilot
``askTraktMi`` action both call. No stubs and no re-implementation: the review
exercises the production parse -> route -> execute -> adapt path.

The book under test is the synthetic ``demo_platform`` platform (two source
portfolios — one direct, one acquired — across three month-end snapshots), so the
temporal, forecast and Schedule 8 limit routes all have genuine governed inputs.

Usage
-----
    # once — build the synthetic platform (~90s)
    python -m demo_platform.run_demo --generate --orchestrate

    # then — run the bank
    python scripts/run_mi_capability_review.py --out out/mi_capability_review

Outputs ``transcript.json`` (full envelopes, artifact-summarised) and
``summary.md`` (one line per question) under ``--out``.

Notes
-----
* With no ``ANTHROPIC_API_KEY`` configured this measures the DETERMINISTIC
  parser, which is also the fallback whenever the LLM parser is unavailable.
  Pass ``--llm`` to let the configured LLM parser participate.
* Nothing here writes to the repository or to any client data source.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import traceback
from pathlib import Path
from typing import Any, Dict, List, Tuple

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import yaml  # noqa: E402

_BANK = _REPO_ROOT / "config" / "mi" / "golden_questions" / "business_semantic_questions.yaml"


def load_questions() -> List[Tuple[str, str]]:
    """``[(id, question)]`` for both groups of the business-semantic bank."""
    bank = yaml.safe_load(_BANK.read_text())
    out: List[Tuple[str, str]] = []
    for section in ("submitted", "questions"):
        for entry in bank.get(section) or []:
            out.append((entry["id"], entry["question"]))
    return out


def summarise_artifact(a: Dict[str, Any]) -> Dict[str, Any]:
    """A compact, reviewable view of one artifact (rows truncated)."""
    kind = a.get("type")
    out: Dict[str, Any] = {"type": kind, "title": a.get("title")}
    rows = a.get("rows") or a.get("data") or []
    if kind == "chart":
        out["chartType"] = a.get("chartType") or a.get("chart_type")
        out["xKey"] = a.get("xKey")
        out["series"] = [s.get("key") if isinstance(s, dict) else s
                         for s in (a.get("series") or [])]
    elif kind == "table":
        out["columns"] = [c.get("key") if isinstance(c, dict) else c
                          for c in (a.get("columns") or [])]
    elif kind == "kpi":
        out["items"] = [{k: v for k, v in (it or {}).items()
                         if k in ("label", "value", "delta")}
                        for it in (a.get("items") or a.get("kpis") or [])]
    out["rowCount"] = len(rows)
    out["sampleRows"] = rows[:6]
    return out


def run_one(qid: str, question: str, ctx) -> Dict[str, Any]:
    from mi_agent_api.mi_service import MiQueryRequest, execute_governed_mi_query

    rec: Dict[str, Any] = {"id": qid, "question": question}
    try:
        res = execute_governed_mi_query(MiQueryRequest(question=question), ctx)
    except Exception:  # noqa: BLE001 - a crash IS a review finding
        rec["exception"] = traceback.format_exc()[-2000:]
        return rec
    env = res.result or {}
    meta = env.get("metadata") or {}
    spec = env.get("spec") if isinstance(env.get("spec"), dict) else {}
    rec.update({
        "governedStatus": res.status,
        "httpStatus": res.http_status,
        "ok": env.get("ok"),
        "error": env.get("error"),
        "answer": env.get("answer"),
        "route": meta.get("route"),
        "specMetric": (spec or {}).get("metric"),
        "specDimension": (spec or {}).get("dimension"),
        "specDimensions": (spec or {}).get("dimensions"),
        "specFilters": (spec or {}).get("filters"),
        "spec": spec,
        "warnings": env.get("warnings"),
        "rejectedDimensions": meta.get("rejected_dimensions") or meta.get("rejectedDimensions"),
        "artifacts": [summarise_artifact(a) for a in (env.get("artifacts") or [])],
    })
    return rec


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="out/mi_capability_review",
                    help="output directory for transcript.json + summary.md")
    ap.add_argument("--llm", action="store_true",
                    help="allow the configured LLM parser to participate "
                         "(default: deterministic only)")
    ap.add_argument("--period", default="current",
                    help="demo_platform reporting period role to answer from")
    args = ap.parse_args()

    from demo_platform import config as cfg

    os.environ.update(cfg.mi_env(period_role=args.period))
    os.environ["MI_AGENT_DATA_CACHE_TTL"] = "0"
    if args.llm:
        os.environ["MI_AGENT_LLM_PARSER"] = "on"
    else:
        os.environ["MI_AGENT_LLM_PARSER"] = "off"
        os.environ.pop("ANTHROPIC_API_KEY", None)

    logging.disable(logging.WARNING)

    from trakt_core.context import ExecutionContext
    from mi_agent_api import data_source

    data_source.reset_cache()
    ctx = ExecutionContext.for_internal(cfg.CLIENT_ID)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    transcript: List[Dict[str, Any]] = []
    for qid, question in load_questions():
        rec = run_one(qid, question, ctx)
        transcript.append(rec)
        print("%-5s ok=%-5s route=%-24s %s" % (
            qid, rec.get("ok"), rec.get("route") or "-",
            (rec.get("answer") or rec.get("error") or "")[:110].replace("\n", " ")),
            flush=True)

    (out_dir / "transcript.json").write_text(
        json.dumps(transcript, indent=2, default=str))

    lines = ["# MI Query Agent — capability review transcript", "",
             f"Questions: {len(transcript)}  ·  parser: "
             f"{'llm-enabled' if args.llm else 'deterministic'}", "",
             "| id | ok | route | answer |", "| --- | --- | --- | --- |"]
    for rec in transcript:
        answer = (rec.get("answer") or rec.get("error") or "").replace("\n", " ")
        lines.append("| %s | %s | %s | %s |" % (
            rec["id"], rec.get("ok"), rec.get("route") or "—", answer[:160]))
    (out_dir / "summary.md").write_text("\n".join(lines) + "\n")

    print(f"\nwrote {out_dir/'transcript.json'} and {out_dir/'summary.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
