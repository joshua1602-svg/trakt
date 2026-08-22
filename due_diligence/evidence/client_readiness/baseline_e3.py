"""E3 baseline: the nine acceptance questions, on the client-shaped fixture.

Run BEFORE any Tranche F or G change lands. Its purpose is not to grade — it is
to record what the shipped system says on a fixture with fifty-two weeks of
pipeline and thirteen months of funded history, so that a later change can be
attributed. A measurement with no before is not evidence of anything.

Both books are run. kestrelmoor carries the declining-KFI stress profile,
alderbridge is stationary, so the same nine questions are asked of a book whose
intake is falling and one whose intake is not — and any answer that is identical
across the two is, by that fact, not reading the pipeline.

    python3 due_diligence/evidence/client_readiness/baseline_e3.py --root <dir>

``--root`` reuses an already-built pack; omitted, the pack is built into a temp
directory and removed afterwards.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import sys
import tempfile
import warnings
from pathlib import Path
from typing import Any, Dict, List

_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parents[2]
sys.path.insert(0, str(_REPO_ROOT))

# The nine commercial acceptance questions, verbatim from the acceptance suite.
BANK: List[str] = [
    "How has the profile of new originations changed in the last few months?",
    "When are we likely to reach £100m of loans?",
    "What is the current value of outstanding offers in the pipeline, how "
    "much do we expect to complete, and when?",
    "What is the current completion run rate?",
    "Are we likely to breach any concentration limits in the next three months?",
    "Which limits are closest to breaching?",
    "How does the risk profile of older vintages compare with the front book?",
    "How has the balance from direct lending changed relative to acquired lending?",
    "Based on the current pipeline, what is the forecast funded balance?",
]


def _ask(root: Path, client_id: str, reporting_date: str) -> List[Dict[str, Any]]:
    """Ask the nine, through the real governed /mi/query path."""
    from tests.analytical import client_pack

    before = dict(os.environ)
    warnings.simplefilter("ignore")
    logging.disable(logging.WARNING)
    cwd = os.getcwd()
    os.chdir(_REPO_ROOT)
    try:
        os.environ.update(client_pack.mi_env(root, client_id, reporting_date))
        os.environ["MI_AGENT_LLM_PARSER"] = "off"      # deterministic parse only
        os.environ.pop("ANTHROPIC_API_KEY", None)

        from fastapi.testclient import TestClient
        from mi_agent_api import data_source
        from mi_agent_api.app import app

        data_source.reset_cache()
        client = TestClient(app)
        out: List[Dict[str, Any]] = []
        for question in BANK:
            response = client.post("/mi/query", json={
                "question": question,
                "portfolioId": f"{client_id}/{reporting_date}",
                "datasetContext": "funded"})
            body = response.json()
            meta = body.get("metadata") or {}
            findings = meta.get("analyticalFindings") or []
            summary = body.get("executionSummary") or {}
            # The PERIOD PAIR each movement finding was computed over. Recorded
            # explicitly because it is the axis a longer history moves: on a
            # three-snapshot fixture almost any span resolves to the same two
            # dates, so a span defect cannot show.
            spans = sorted({
                f"{f.get('priorPeriod') or f.get('comparisonPeriod') or ''}"
                f" -> {f.get('period') or ''}"
                for f in findings if f.get("kind") in ("movement", "comparison")
            } - {" -> "})
            out.append({
                "question": question,
                "status": response.status_code,
                "answer": body.get("answer"),
                "route": meta.get("route"),
                "findingKinds": sorted({f.get("kind") for f in findings if f.get("kind")}),
                "findingCount": len(findings),
                "periodSpans": spans,
                "period": summary.get("period"),
                "comparisonPeriod": summary.get("comparisonPeriod"),
                "population": summary.get("populationLabel"),
                "warnings": body.get("warnings") or [],
                "refused": not bool(findings),
            })
        return out
    finally:
        os.chdir(cwd)
        logging.disable(logging.NOTSET)
        os.environ.clear()
        os.environ.update(before)
        from mi_agent_api import data_source as _ds
        _ds.reset_cache()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", help="an already-built pack root")
    parser.add_argument("--out", default=str(_HERE / "baseline_e3.json"))
    args = parser.parse_args(argv)

    from tests.analytical import client_funded, client_pack

    temp = None
    if args.root:
        root = Path(args.root)
    else:
        temp = Path(tempfile.mkdtemp(prefix="client_pack_"))
        client_pack.build(temp)
        root = temp
    try:
        reporting_date = client_funded.REPORTING_DATES[-1]
        result = {"reportingDate": reporting_date,
                  "lastPipelineWeek": client_pack.LAST_PIPELINE_WEEK.isoformat(),
                  "stressBook": client_pack.STRESS_BOOK,
                  "books": {}}
        for client_id in ("alderbridge", "kestrelmoor"):
            answers = _ask(root, client_id, reporting_date)
            result["books"][client_id] = answers
            answered = sum(1 for a in answers if not a["refused"] and a["findingCount"])
            print(f"\n=== {client_id} "
                  f"({'stress' if client_id == client_pack.STRESS_BOOK else 'control'}) "
                  f"— {answered}/9 answered with findings ===")
            for i, a in enumerate(answers, 1):
                state = ("NO FINDINGS" if a["refused"]
                         else f"{a['findingCount']} findings {a['findingKinds']}")
                span = f"  spans={a['periodSpans']}" if a["periodSpans"] else ""
                print(f"  Q{i} [{a['status']}] {state}{span}")
                print(f"      {(a['answer'] or '')[:200]}")
        Path(args.out).write_text(json.dumps(result, indent=2, ensure_ascii=False)
                                  + "\n", encoding="utf-8")
        print(f"\nwrote {args.out}")
        return 0
    finally:
        if temp is not None:
            shutil.rmtree(temp, ignore_errors=True)


if __name__ == "__main__":                  # pragma: no cover - CLI
    raise SystemExit(main())
