"""D4 amendment 2 — what does the DETERMINISTIC path resolve for the vague
phrase, and does the frozen expectation agree?

Rejecting a fabricated bound is only an improvement if what replaces it is
correct or a refusal. A different unstated assumption would be the same defect
with a different author.
"""
from __future__ import annotations
import json, logging, os, sys, warnings
from pathlib import Path
warnings.simplefilter("ignore")
REPO = Path("/home/user/trakt"); S = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(S)); os.chdir(REPO)
import yaml
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
from nl_bank import bank

EXP = yaml.safe_load((REPO / "due_diligence/evidence/forecast_composition_hardening/"
                      "frozen_expectations.yaml").read_text())
TARGETS = {"Q1.3", "Q3.3", "Q3.4", "Q4.4", "Q6.3"}
c = TestClient(app)
PORT = f"{os.environ['MI_AGENT_CLIENT_ID']}/{os.environ['MI_AGENT_RUN_ID']}"

for e in bank("alderbridge"):
    key = f"{e['intent']}.{e['variation']}"
    if key not in TARGETS:
        continue
    r = c.post("/mi/query", json={"question": e["question"], "portfolioId": PORT,
                                  "datasetContext": "funded"}).json()
    m = r.get("metadata") or {}
    exp = EXP.get(key) or {}
    preds, labels = set(), set()
    for f in (m.get("analyticalFindings") or []):
        p = f.get("population") or {}
        if p.get("predicate"): preds.add(p["predicate"])
        if p.get("label"): labels.add(p["label"])
    applied = (m.get("populationApplied") or {}).get("applied") or []
    print(f"\n=== {key}  {e['question']}")
    print(f"  frozen expectation population : {exp.get('population')}")
    print(f"  deterministic resolved to     : predicates={sorted(preds) or '-'}")
    print(f"                                  labels={sorted(labels) or '-'}")
    print(f"                                  populationApplied={applied or '-'}")
    print(f"  ok={r.get('ok')} refusal={r.get('controlledRefusal')}")
    print(f"  window disclosed in answer    : {str(r.get('answer'))[:210]}")
