#!/usr/bin/env python3
"""READ-ONLY: which recogniser CLAIMS each corpus question, offline, no data roots."""
import json, os, sys, collections
sys.path.insert(0, os.getcwd())
from mi_agent.llm_query_parser import parse_with_repair
from mi_agent.mi_query_validator import load_mi_semantics
from mi_agent.query_plan_adapter import plan_from_spec
from mi_agent_api.recogniser_registry import REGISTRY, RouteRequest
import mi_agent_api.chat_routing as cr  # registers the default recognisers

SEM = load_mi_semantics("mi_agent/mi_semantics_field_registry.yaml")
qs, seen = [], set()
for f in ("question_interpretation/stage1_corpus.json",
          "question_interpretation/stage2_corpus.json"):
    for r in json.load(open(f))["rows"]:
        q = (r.get("question") or "").strip()
        if q and q not in seen:
            seen.add(q); qs.append(q)

out = []
for q in qs:
    try:
        spec, meta = parse_with_repair(q, SEM, llm_enabled=False)
    except Exception as e:
        out.append({"q": q, "claim": "<parse-error>", "err": repr(e)}); continue
    sd = spec.to_dict() if hasattr(spec, "to_dict") else {}
    req = RouteRequest(question=q, spec=spec, spec_dict=sd, semantics=SEM,
                       view="funded", client_id="c", run_id=None, portfolio_id=None)
    claims = []
    for rec in REGISTRY.ordered():
        try:
            verdict = rec.recognise(req.for_recognition())
        except Exception:
            continue
        ok = getattr(verdict, "matched", verdict)
        if ok:
            claims.append(rec.name)
    out.append({"q": q, "claims": claims,
                "claim": claims[0] if claims else "<generic_mi_workflow>",
                "liftable": plan_from_spec(spec) is not None,
                "agg": spec.aggregation, "metric": spec.metric,
                "intent": spec.intent, "chart": spec.chart_type})
print(json.dumps(out, default=str))
