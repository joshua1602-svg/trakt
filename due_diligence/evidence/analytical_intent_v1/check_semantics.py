"""Compare the semantic EXPECTATION against what the agent actually resolved.

The expectation stands. Where they disagree the disagreement is reported; the
expectation is never adjusted toward the agent.
"""
from __future__ import annotations
import json, sys, yaml
from pathlib import Path
S = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(S)); sys.path.insert(0, "/home/user/trakt")
from mi_workflows.analytical import intent as intent_mod

EXP = yaml.safe_load((Path(__file__).parent / "expected_semantics.yaml").read_text())
SOURCE = {
    "Q1.1": ("alderbridge", "production", "Q1", 1, ""),
    "Q1.2": ("alderbridge", "forced_llm", "Q1", 2, ""),
    "Q3.1": ("alderbridge", "production", "Q3", 1, ""),
    "Q4.2": ("alderbridge", "production", "Q4", 2, ""),
    "Q6.2": ("kestrelmoor", "production", "Q6", 2, ""),
    "Q7.2": ("alderbridge", "production", "Q7", 2, ""),
    "Q8.3/provenance": ("kestrelmoor", "production", "Q8", 3, "provenance"),
    "Q8.3/seasoning": ("kestrelmoor", "production", "Q8", 3, "seasoning"),
    "Q8.3/dimension_value": ("kestrelmoor", "production", "Q8", 3, "dimension_value"),
    "Q9.3": ("kestrelmoor", "forced_llm", "Q9", 3, ""),
}
_cache = {}


def run_for(key):
    book, arm, tag, var, pair = SOURCE[key]
    f = f"v1_nl_{book}_{arm}.json"
    if f not in _cache:
        _cache[f] = json.load(open(S / f))
    for e in _cache[f]:
        if e["intent"] == tag and e["variation"] == var and (e.get("pair") or "") == pair:
            return e, e["runs"][0]
    raise KeyError(key)


rows = []
for key, exp in EXP.items():
    entry, run = run_for(key)
    reading = intent_mod.classify(entry["question"])
    pop = run.get("populationApplied") or {}
    applied = pop.get("applied") or []
    # what the answer was actually computed over, in governed terms
    got_pop = "; ".join(applied) if applied else None
    if not got_pop:
        preds = []
        for f in run.get("findings") or []:
            p = (f.get("population") or {})
            if p.get("predicate"):
                preds.append(p["predicate"])
        got_pop = "; ".join(sorted(set(preds))) or "(whole book / none recorded)"

    fam_exp, fam_got = set(exp["family"]), set(reading.families)
    op_exp, op_got = set(exp["operation"]), set(reading.operations)
    fam_ok = fam_exp <= fam_got
    op_ok = bool(op_exp & op_got)
    rows.append({"key": key, "fam_exp": sorted(fam_exp), "fam_got": sorted(fam_got),
                 "fam_ok": fam_ok, "op_exp": sorted(op_exp), "op_got": sorted(op_got),
                 "op_ok": op_ok, "pop_exp": exp["population"], "pop_got": got_pop,
                 "tension": exp.get("tension"),
                 "expected_outcome": exp.get("expected_outcome"),
                 "ok_flag": run.get("ok")})

print(f"{'question':22s} {'family':8s} {'operation':10s}  population")
print("-" * 110)
for r in rows:
    print(f"{r['key']:22s} {'agree' if r['fam_ok'] else 'DIFFER':8s} "
          f"{'agree' if r['op_ok'] else 'DIFFER':10s}  expected: {r['pop_exp']}")
    print(f"{'':22s} {'':8s} {'':10s}  resolved: {r['pop_got']}")
    if not r["fam_ok"]:
        print(f"{'':44s}  family expected {r['fam_exp']} got {r['fam_got']}")
    if r["expected_outcome"] == "refusal":
        print(f"{'':44s}  expected outcome REFUSAL — agent ok={r['ok_flag']} "
              f"({'agrees' if r['ok_flag'] is False else 'DIFFERS'})")
print()
print("RECORDED TENSIONS (question supports another reading; not resolved here):")
for r in rows:
    if r["tension"]:
        print(f"  {r['key']}: {' '.join(r['tension'].split())[:190]}…")
