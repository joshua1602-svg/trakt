#!/usr/bin/env python3
"""READ-ONLY census probe: corpus -> deterministic parse -> route claim + plan liftability.

No network, no LLM, no product file touched. Runs offline with the committed corpora.
"""
import json, os, sys, collections
sys.path.insert(0, os.getcwd())

from mi_agent.llm_query_parser import parse_with_repair
from mi_agent.query_plan_adapter import plan_from_spec, MODELLED_FIELDS
from mi_agent.mi_query_validator import load_mi_semantics

# semantics registry
import glob
cands = ["mi_agent/mi_semantics_field_registry.yaml"]
sem_path = next((c for c in cands if os.path.exists(c)), None)
print("SEMANTICS:", sem_path, file=sys.stderr)
SEM = load_mi_semantics(sem_path)

qs = []
seen = set()
for f in ("question_interpretation/stage1_corpus.json",
          "question_interpretation/stage2_corpus.json"):
    for r in json.load(open(f))["rows"]:
        q = (r.get("question") or "").strip()
        if q and q not in seen:
            seen.add(q); qs.append(q)
print("DISTINCT QUESTIONS:", len(qs), file=sys.stderr)

import dataclasses
from mi_agent.mi_query_spec import MIQuerySpec
DEF = {}
for fl in dataclasses.fields(MIQuerySpec):
    if fl.default is not dataclasses.MISSING: DEF[fl.name] = fl.default
    elif fl.default_factory is not dataclasses.MISSING: DEF[fl.name] = fl.default_factory()

rows = []
liftable = 0
nonmodelled_counter = collections.Counter()
for q in qs:
    try:
        spec, meta = parse_with_repair(q, SEM, llm_enabled=False)
    except Exception as e:
        rows.append({"q": q, "error": repr(e)}); continue
    plan = None
    try:
        plan = plan_from_spec(spec)
    except Exception:
        plan = None
    lift = plan is not None
    liftable += lift
    blockers = []
    for name, default in DEF.items():
        if name in MODELLED_FIELDS: continue
        if getattr(spec, name, default) != default:
            blockers.append(name)
    if not lift:
        for b in (blockers or ["<axis-or-contract>"]):
            nonmodelled_counter[b] += 1
    rows.append({"q": q, "liftable": lift, "blockers": blockers,
                 "intent": spec.intent, "chart": spec.chart_type,
                 "metric": spec.metric, "agg": spec.aggregation,
                 "dims": list(spec.dimensions or []) + ([spec.dimension] if spec.dimension else []),
                 "filters": sorted((spec.filters or {}).keys()),
                 "note": (meta or {}).get("note")})

print(json.dumps({"total": len(qs), "liftable": liftable,
                  "not_liftable": len(qs) - liftable,
                  "blocker_frequency": nonmodelled_counter.most_common(40),
                  "rows": rows}, default=str))
