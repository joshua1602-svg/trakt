#!/usr/bin/env python3
"""question_interpretation/run_robustness_deterministic.py

The SECOND measurement surface standing condition 1 requires, in the only arm
this environment can run.

Standing condition 1 names the calibration bank and the 44-variation robustness
bank. The robustness bank's recorded 752-run figures are an LLM-arm
measurement: `nl_harness.py` asserts ANTHROPIC_API_KEY, which is not set here,
so those numbers cannot be reproduced or re-measured. What CAN be run is the
same bank, the same endpoint and the same frozen scorer with the LLM parser
off — the deterministic arm.

    python -m question_interpretation.run_robustness_deterministic --json out.json

This is a strictly weaker surface than the recorded one and is labelled as such
everywhere it is reported. It is not a substitute for the LLM arm; it is the
part of standing condition 1 that is honourable without a key.

Frozen artefacts are IMPORTED, never modified: `nl_bank.bank` supplies the
questions and `nl_score.grade` supplies the verdict, so this runner cannot
grade more leniently than the recorded harness did.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import warnings
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List

warnings.simplefilter("ignore")

_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parent
_EVIDENCE = _REPO_ROOT / "due_diligence" / "evidence" / "analytical_intent_v1"
for p in (str(_REPO_ROOT), str(_EVIDENCE)):
    if p not in sys.path:
        sys.path.insert(0, p)

#: The intents the Stage 4 acceptance condition names by name. The 160-run
#: regression was entirely Q1.1 and the Q7/Q8 seasoning family, so these are
#: reported separately and never folded into an aggregate.
SEASONING_INTENTS = ("Q1", "Q7", "Q8")


def _capture(response: Dict[str, Any]) -> Dict[str, Any]:
    """The subset of `nl_harness.capture` that `nl_score.grade` reads.

    Deliberately a subset: anything else would invite grading on a field the
    recorded harness did not record.
    """
    meta = response.get("metadata") or {}
    plan = meta.get("analyticalPlan") or {}
    return {
        "ok": response.get("ok"),
        "route": meta.get("route"),
        "answer": (response.get("answer") or "").strip(),
        "intent": plan.get("intent"),
        "capabilities": [c.get("capability") for c in (plan.get("calls") or [])],
        "analyticalIntent": meta.get("analyticalIntent"),
        "spec": {k: v for k, v in (response.get("spec") or {}).items()
                 if k in ("intent", "metric", "dimension", "aggregation",
                          "filters", "chart_type", "temporal_mode",
                          "forecast_mode", "risk_limit_query", "top_n",
                          "measures", "execution_mode", "cohort_progression",
                          "bridge_query")},
        "controlledRefusal": response.get("controlledRefusal", False),
        "artifacts": [{"type": a.get("type"), "rows": len(a.get("rows") or [])}
                      for a in (response.get("artifacts") or [])],
        "warnings": response.get("warnings") or [],
    }


#: Where the synthetic second book is materialised. Built on demand and
#: reproducible from its own seed, so it is not committed.
_KESTRELMOOR_ROOT = _HERE / ".kestrelmoor_store"


def _client_for(book: str):
    """One book per process, as the recorded harness does.

    The app reads its book from the environment at import time, so two books in
    one process would measure the first one twice.
    """
    if book == "alderbridge":
        from demo_platform import config as cfg
        os.environ.update(cfg.mi_env(period_role="current"))
    elif book == "kestrelmoor":
        from tests.analytical import second_book as bk
        if not _KESTRELMOOR_ROOT.exists():
            bk.build(_KESTRELMOOR_ROOT)
        os.environ.update(bk.mi_env(_KESTRELMOOR_ROOT))
    else:
        raise ValueError("unknown book %r" % (book,))
    portfolio = "%s/%s" % (os.environ["MI_AGENT_CLIENT_ID"],
                           os.environ["MI_AGENT_RUN_ID"])
    # The one documented difference from the recorded harness.
    os.environ["MI_AGENT_LLM_PARSER"] = "off"
    os.environ["MI_AGENT_LLM_ENABLED"] = "0"
    os.environ["MI_AGENT_DATA_CACHE_TTL"] = "0"
    logging.disable(logging.WARNING)

    from fastapi.testclient import TestClient
    from mi_agent_api import data_source
    from mi_agent_api.app import app
    data_source.reset_cache()
    return TestClient(app), portfolio


def run(book: str = "alderbridge") -> Dict[str, Any]:
    import nl_bank
    import nl_score

    client, portfolio = _client_for(book)
    rows: List[Dict[str, Any]] = []
    for entry in nl_bank.bank(book):
        response = client.post("/mi/query", json={
            "question": entry["question"], "portfolioId": portfolio,
            "datasetContext": "funded"}).json()
        run_record = _capture(response)
        outcome, causes, note = nl_score.grade(entry["intent"], run_record)
        rows.append({
            "book": book, "intent": entry["intent"],
            "variation": entry["variation"], "question": entry["question"],
            "outcome": outcome, "causes": list(causes), "note": note,
            "route": run_record["route"], "ok": run_record["ok"],
            "analytical_intent": run_record["intent"],
        })
    return {"book": book, "arm": "deterministic", "rows": rows}


def summarise(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "total": len(rows),
        "by_outcome": dict(Counter(r["outcome"] for r in rows)),
        "by_intent": {
            intent: dict(Counter(r["outcome"] for r in rows if r["intent"] == intent))
            for intent in sorted({r["intent"] for r in rows})
        },
        "seasoning_family": {
            intent: dict(Counter(r["outcome"] for r in rows if r["intent"] == intent))
            for intent in SEASONING_INTENTS
        },
    }


def _run_both_books() -> Dict[str, Any]:
    """Both books, one subprocess each.

    Not a loop in this process: the app binds its book from the environment at
    import time, so a second book in the same process would silently re-measure
    the first.
    """
    import subprocess
    import tempfile
    books = {}
    for book in ("alderbridge", "kestrelmoor"):
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as fh:
            out = Path(fh.name)
        proc = subprocess.run(
            [sys.executable, "-m", __spec__.name, "--book", book, "--json", str(out)],
            cwd=str(_REPO_ROOT), capture_output=True, text=True)
        if proc.returncode != 0:
            raise RuntimeError("book %s failed:\n%s" % (book, proc.stdout + proc.stderr))
        books[book] = json.loads(out.read_text(encoding="utf-8"))
        out.unlink(missing_ok=True)
    return books


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--book", default="alderbridge")
    ap.add_argument("--all-books", action="store_true",
                    help="run both books, one subprocess each, and compare")
    ap.add_argument("--json", type=Path, default=None)
    args = ap.parse_args(argv)

    if args.all_books:
        books = _run_both_books()
        print("=" * 74)
        print("44-variation robustness bank — DETERMINISTIC ARM, BOTH BOOKS")
        print("NOT the recorded 752-run LLM measurement")
        print("=" * 74)
        for book, payload in books.items():
            print("\n%s: %s" % (book, payload["summary"]["by_outcome"]))
            print("   seasoning family: %s" % payload["summary"]["seasoning_family"])
        a, b = (books["alderbridge"]["rows"], books["kestrelmoor"]["rows"])
        same = sum(1 for x, y in zip(a, b) if x["outcome"] == y["outcome"])
        print("\nsame verdict on both books: %d / %d" % (same, len(a)))
        if args.json:
            args.json.write_text(json.dumps(books, indent=2, default=str),
                                 encoding="utf-8")
            print("wrote %s" % args.json)
        return 0

    data = run(args.book)
    rows = data["rows"]
    s = summarise(rows)
    data["summary"] = s

    print("=" * 74)
    print("44-variation robustness bank — DETERMINISTIC ARM (LLM parser off)")
    print("book: %s   NOT the recorded 752-run LLM measurement" % args.book)
    print("=" * 74)
    print("\n%d variations" % s["total"])
    for outcome, n in sorted(s["by_outcome"].items(), key=lambda t: -t[1]):
        print("   %-38s %3d" % (outcome, n))

    print("\nby intent:")
    for intent in sorted(s["by_intent"]):
        marker = "  <- seasoning family" if intent in SEASONING_INTENTS else ""
        print("   %-4s %s%s" % (intent, s["by_intent"][intent], marker))

    if args.json:
        args.json.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")
        print("\nwrote %s" % args.json)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
