"""NL bank harness, DETERMINISTIC PARSE ONLY — the reproducible A/B instrument.

    python3 nl_harness_det.py <book> <arm> <out.json>

A SEPARATE instrument from nl_harness.py, which is pinned in the manifest and is
NOT modified. The difference is one thing: the LLM parser is off, so the parse is
a pure function of the question text.

Why it exists. The LLM arm's parser mix is not reproducible — the same question
can parse by LLM on one run and fall back to deterministic on the next, and the
two paths produce different specs. Comparing two code revisions across such a
pair measures the parser mix as much as the code. With the LLM off, every run of
a given question produces the same spec on both revisions by construction, so any
difference in the answer is attributable to the code and to nothing else.

What it does NOT do: exercise the LLM parse path. It is the controlled comparator,
not a replacement for the LLM arm.
"""
from __future__ import annotations

import json
import logging
import os
import sys
import time
import warnings
from pathlib import Path

warnings.simplefilter("ignore")
REPO = Path("/home/user/trakt")
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(Path(__file__).resolve().parent))
os.chdir(REPO)

BOOK, ARM, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
SCRATCH = Path(__file__).resolve().parent

if BOOK == "alderbridge":
    from demo_platform import config as cfg
    os.environ.update(cfg.mi_env(period_role="current"))
    PORTFOLIO = f"{os.environ['MI_AGENT_CLIENT_ID']}/{os.environ['MI_AGENT_RUN_ID']}"
else:
    from tests.analytical import second_book as bk
    os.environ.update(bk.mi_env(SCRATCH / "kestrelmoor_store"))
    PORTFOLIO = f"{bk.CLIENT_ID}/{bk.REPORTING_DATES[-1]}"

os.environ["MI_AGENT_PIPELINE_ROOT"] = str(SCRATCH / "pipeline_root")
os.environ["MI_AGENT_DATA_CACHE_TTL"] = "0"
# The LLM parser OFF — the whole point of THIS instrument.
os.environ["MI_AGENT_LLM_PARSER"] = "off"
os.environ["MI_AGENT_LLM_ENABLED"] = "0"
os.environ.pop("ANTHROPIC_API_KEY", None)
logging.disable(logging.WARNING)

from mi_agent.parsed_question import ParsedQuestion  # noqa: E402

if ARM == "forced_llm":
    _original = ParsedQuestion.parse.__func__

    def _forced(cls, question, semantics, **kw):
        kw["zero_cost_first"] = False
        return _original(cls, question, semantics, **kw)

    ParsedQuestion.parse = classmethod(_forced)

from mi_agent_api import data_source  # noqa: E402
data_source.reset_cache()
from fastapi.testclient import TestClient  # noqa: E402
from mi_agent_api.app import app  # noqa: E402
from nl_bank import bank, runs_for  # noqa: E402

client = TestClient(app)


def capture(response: dict) -> dict:
    meta = response.get("metadata") or {}
    provenance = meta.get("parserProvenance") or {}
    plan = meta.get("analyticalPlan") or {}
    return {
        "ok": response.get("ok"),
        "route": meta.get("route"),
        "answer": (response.get("answer") or "").strip(),
        # The governed provenance contract (mi_service._parser_provenance):
        #   parser_used = llm | deterministic | deterministic_fallback_after_llm_failure
        # Only "llm" counts as a genuine LLM parse per the brief's §4.
        "parserUsed": provenance.get("parser_used"),
        "parserDetail": provenance.get("parser_mode_detail"),
        "llmFailure": provenance.get("llm_failure"),
        "parserProvenance": provenance,
        "llm": meta.get("llm") or {},
        "intent": plan.get("intent"),
        "capabilities": [c.get("capability") for c in (plan.get("calls") or [])],
        # RECORDED, not derived later. The boundary's own classification and the
        # plan it produced, captured at measurement time. Before this, a trace
        # had to re-run intent.classify afterwards and assert it was a pure
        # function of the question — an assertion, not a control. Now the run
        # file carries what the running system actually decided.
        "analyticalIntent": meta.get("analyticalIntent"),
        "planCalls": [{"capability": c.get("capability"),
                       "because": c.get("because"),
                       "inputs": {k: (v.get("label") if isinstance(v, dict) and "label" in v else v)
                                  for k, v in (c.get("inputs") or {}).items()}}
                      for c in (plan.get("calls") or [])],
        "requiredKinds": plan.get("requiredKinds"),
        "planOrigin": plan.get("origin"),
        "composition": meta.get("analyticalComposition") or {},
        "execution": meta.get("analyticalExecution") or [],
        "findings": meta.get("analyticalFindings") or [],
        "populationApplied": meta.get("populationApplied"),
        # The spec the parse produced — needed to trace WHY a route was chosen
        # and whether a degenerate dimension was used.
        "spec": {k: v for k, v in (response.get("spec") or {}).items()
                 if k in ("intent", "metric", "dimension", "aggregation",
                          "filters", "chart_type", "temporal_mode",
                          "forecast_mode", "risk_limit_query", "top_n",
                          "measures", "execution_mode", "cohort_progression",
                          "bridge_query")},
        "semanticGuard": (response.get("semanticGuard") or {}).get("verdict"),
        "guardFacets": [{"kind": f.get("kind"), "status": f.get("status"),
                         "label": f.get("label")}
                        for f in ((response.get("semanticGuard") or {})
                                  .get("facets") or [])],
        "reconciliation": response.get("reconciliation"),
        "executionSummary": response.get("executionSummary"),
        "controlledRefusal": response.get("controlledRefusal", False),
        "warnings": response.get("warnings") or [],
        "artifacts": [{"type": a.get("type"), "title": a.get("title"),
                       "rows": len(a.get("rows") or [])}
                      for a in (response.get("artifacts") or [])],
    }


rows = bank(BOOK)
out = []
started = time.time()
for index, entry in enumerate(rows, start=1):
    repeats = runs_for(entry["intent"])
    runs = []
    for attempt in range(repeats):
        try:
            response = client.post("/mi/query", json={
                "question": entry["question"], "portfolioId": PORTFOLIO,
                "datasetContext": "funded"}).json()
            runs.append(capture(response))
        except Exception as exc:  # noqa: BLE001 - a crash IS a hard failure
            runs.append({"ok": False, "route": None,
                         "answer": f"HARD_FAILURE {type(exc).__name__}: {exc}",
                         "hardFailure": True})
    out.append({**entry, "book": BOOK, "arm": ARM, "runs": runs})
    genuine = sum(1 for r in runs if r.get("parserUsed") == "llm")
    print(f"[{index:>2}/{len(rows)}] {entry['intent']}.{entry['variation']} "
          f"{'/' + entry['pair'] if entry['pair'] else '':14s} "
          f"genuine_llm={genuine}/{repeats} "
          f"routes={sorted({str(r.get('route')) for r in runs})} "
          f"ok={sorted({str(r.get('ok')) for r in runs})}", flush=True)

Path(OUT).write_text(json.dumps(out, indent=1, default=str))
print(f"WROTE {OUT} — {len(out)} variations in {time.time() - started:.0f}s")
