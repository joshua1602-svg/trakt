"""TYPE-CONFORMANCE SWEEP over the 44-variation NL bank.

The frozen expectation file declares a MEASURE and an OPERATION per variation.
This joins the type of the measure each run actually reported against the type
the QUESTION asks for, and separately reports which variations declare no
measure at all -- those rest on the frozen scorer's outcome grade and nothing
else.
"""
from __future__ import annotations
import collections, json, sys, warnings
from pathlib import Path
warnings.simplefilter("ignore")
sys.path.insert(0, "/home/user/trakt")
sys.path.insert(0, str(Path(__file__).resolve().parent))
import yaml
from type_sweep import asked_type, metric_type, _OK

EXP = yaml.safe_load(Path("/home/user/trakt/due_diligence/evidence/"
                          "forecast_composition_hardening/frozen_expectations.yaml").read_text())
ARMS = ["nl_alderbridge_production.json", "nl_alderbridge_forced_llm.json",
        "nl_kestrelmoor_production.json", "nl_kestrelmoor_forced_llm.json"]
root = Path(sys.argv[1])


def key_for(e):
    pair = e.get("pair") or ""
    return f"{e['intent']}.{e['variation']}/{pair}" if pair else f"{e['intent']}.{e['variation']}"


findings, no_measure, checked = [], [], 0
seen_variation = set()
for p in ARMS:
    for e in json.load(open(root / p)):
        k = key_for(e)
        exp = EXP.get(k) or {}
        if k not in seen_variation:
            seen_variation.add(k)
            if not exp.get("measure"):
                no_measure.append((k, e["question"]))
        want = asked_type(e["question"])
        for r in e.get("runs") or []:
            if r.get("controlledRefusal") or r.get("ok") is False:
                continue
            for f in (r.get("findings") or []):
                if f.get("kind") not in ("measure", "movement", "comparison"):
                    continue
                checked += 1
                got = metric_type(f.get("metric"), f.get("aggregation"))
                allowed = _OK.get(want)
                if allowed is not None and got not in allowed:
                    findings.append((k, e["question"], want, got, f.get("metric")))

print(f"variations: {len(seen_variation)}   findings checked: {checked}")
print(f"\n--- TYPE findings: {len(findings)}")
agg = collections.Counter((k, w, g, m) for k, _q, w, g, m in findings)
for (k, w, g, m), n in sorted(agg.items()):
    q = next(f[1] for f in findings if f[0] == k)
    print(f"  {k:22s} x{n:<4d} asked {w:9s} returned {g:9s} ({m})")
    print(f"  {'':22s}       {q[:88]}")
print(f"\n--- variations whose frozen expectation declares NO measure: {len(no_measure)}")
for k, q in sorted(no_measure):
    print(f"  {k:22s} {q[:84]}")
