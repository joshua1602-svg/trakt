"""Capture the TYPE of what each bank question returned, structurally.

Reads the receipt and the spec rather than the prose, so nothing here depends
on how an answer is worded.
"""
from __future__ import annotations
import json, logging, os, sys, warnings
from pathlib import Path
warnings.simplefilter("ignore")
REPO = Path("/home/user/trakt"); S = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(S)); os.chdir(REPO)
from demo_platform import config as cfg
os.environ.update(cfg.mi_env(period_role="current"))
os.environ["MI_AGENT_PIPELINE_ROOT"] = str(S / "pipeline_root")
os.environ["MI_AGENT_DATA_CACHE_TTL"] = "0"
os.environ["MI_AGENT_LLM_PARSER"] = "off"; os.environ["MI_AGENT_LLM_ENABLED"] = "0"
os.environ.pop("ANTHROPIC_API_KEY", None)
logging.disable(logging.WARNING)
from mi_agent_api import data_source; data_source.reset_cache()
from fastapi.testclient import TestClient
from mi_agent_api.app import app
c = TestClient(app)
CLIENT, RUN = os.environ["MI_AGENT_CLIENT_ID"], os.environ["MI_AGENT_RUN_ID"]

questions = [l.strip() for l in Path(sys.argv[1]).read_text().splitlines() if l.strip()]
out = []
for q in questions:
    r = c.post("/mi/query", json={"question": q, "portfolioId": f"{CLIENT}/{RUN}",
                                  "datasetContext": "funded"}).json()
    m = r.get("metadata") or {}
    spec = r.get("spec") or {}
    out.append({
        "q": q,
        "ok": r.get("ok"),
        "refusal": r.get("controlledRefusal", False),
        "metric": spec.get("metric"),
        "aggregation": spec.get("aggregation"),
        "dimension": spec.get("dimension"),
        "dimensions": spec.get("dimensions") or [],
        "chart_type": spec.get("chart_type"),
        "intent": spec.get("intent"),
        "filters": spec.get("filters") or {},
        "artifacts": [a.get("type") for a in (r.get("artifacts") or [])],
        "executionSummary": r.get("executionSummary"),
        "answer": (r.get("answer") or "").strip(),
    })
Path(sys.argv[2]).write_text(json.dumps(out, indent=1, default=str))
print(f"{len(out)} captured -> {sys.argv[2]}")
