"""One command: score all four arms, diff against the frozen baseline,
reconcile every finding, and print the §12 gate."""
from __future__ import annotations
import collections, json, os, subprocess, sys
from pathlib import Path
S = Path(__file__).resolve().parent
sys.path.insert(0, str(S))
from nl_score import grade, EXPECTED  # noqa: E402

UNSAFE = {"INCORRECT_SUCCESSFUL", "SILENT_SEMANTIC_ERROR", "HARD_FAILURE"}
OK = {"CORRECT", "CORRECT_WITH_DISCLOSED_LIMITATION"}
BASE = ["nl_alderbridge_production.json", "nl_alderbridge_forced_llm.json",
        "nl_kestrelmoor_production.json", "nl_kestrelmoor_forced_llm.json"]
NEW = ["v1_" + p for p in BASE]


def load(paths):
    rows = []
    for path in paths:
        for entry in json.load(open(S / path)):
            for i, run in enumerate(entry.get("runs") or []):
                outcome, causes, note = grade(entry["intent"], run)
                rows.append({"book": entry["book"], "arm": entry["arm"],
                             "vkey": (entry["intent"], entry["variation"],
                                      entry.get("pair") or ""),
                             "repeat": i, "question": entry["question"],
                             "outcome": outcome, "causes": causes, "note": note,
                             "run": run})
    return rows


def dist(rows, label):
    n = len(rows); d = collections.Counter(r["outcome"] for r in rows)
    print(f"\n### {label} — {n} runs")
    for k in ("CORRECT", "CORRECT_WITH_DISCLOSED_LIMITATION", "HONEST_PARTIAL",
              "SAFE_REFUSAL", "INCORRECT_SUCCESSFUL", "SILENT_SEMANTIC_ERROR",
              "HARD_FAILURE"):
        print(f"  {k:36s} {d[k]:4d}  {100*d[k]/n:5.1f}%")
    good = sum(d[k] for k in OK); bad = sum(d[k] for k in UNSAFE)
    print(f"  {'CORRECT + DISCLOSED':36s} {good:4d}  {100*good/n:5.1f}%")
    print(f"  {'UNSAFE':36s} {bad:4d}  {100*bad/n:5.1f}%")
    llm = sum(1 for r in rows if (r["run"] or {}).get("parserUsed") == "llm")
    print(f"  genuine LLM parses {llm}/{n} ({100*llm/n:.0f}%)")
    return d


base, new = load(BASE), load(NEW)
db, dn = dist(base, "BASELINE (frozen, 104c89d)"), dist(new, "V1")

print("\n### Per intent")
print(f"  {'':4s} {'ok':>13s} {'partial':>10s} {'refusal':>10s} {'UNSAFE':>11s}")
f = lambda rows, s: sum(1 for r in rows if r["outcome"] in s)
for q in sorted(EXPECTED):
    b = [r for r in base if r["vkey"][0] == q]
    v = [r for r in new if r["vkey"][0] == q]
    print(f"  {q}  {f(b,OK):5d} -> {f(v,OK):<4d} {f(b,{'HONEST_PARTIAL'}):5d} -> "
          f"{f(v,{'HONEST_PARTIAL'}):<3d} {f(b,{'SAFE_REFUSAL'}):5d} -> "
          f"{f(v,{'SAFE_REFUSAL'}):<3d} {f(b,UNSAFE):6d} -> {f(v,UNSAFE):<3d}")

gb, gn = collections.defaultdict(list), collections.defaultdict(list)
for r in base: gb[r["vkey"]].append(r)
for r in new: gn[r["vkey"]].append(r)
maj = lambda rows: collections.Counter(r["outcome"] for r in rows).most_common(1)[0][0]

print("\n### Per variation, baseline -> V1")
moved = []
for k in sorted(gb):
    mb, mn = maj(gb[k]), maj(gn[k])
    routes = sorted({str((r["run"] or {}).get("route")) for r in gn[k]})
    mark = "" if mb == mn else "   <-- moved"
    if mb != mn: moved.append((k, mb, mn))
    print(f"  {k[0]}.{k[1]}{('/'+k[2]) if k[2] else '':16s} {mb:34s} -> {mn:34s} {routes}{mark}")

print(f"\n{len(moved)} of {len(gb)} variations changed verdict.")
lost = [(k,a,b) for k,a,b in moved if a in OK and b not in OK]
regressed = [(k,a,b) for k,a,b in moved if b in UNSAFE]
print("Regressions into unsafe: " + ("NONE" if not regressed else str(regressed)))
print("Lost capability (ok -> not ok): " + ("NONE" if not lost else str(lost)))

print("\n### Cross-book agreement")
agree = 0; keys = sorted({k for k in gn})
for k in keys:
    per = collections.defaultdict(list)
    for r in gn[k]: per[r["book"]].append(r["outcome"])
    verdicts = {b: collections.Counter(v).most_common(1)[0][0] for b, v in per.items()}
    if len(set(verdicts.values())) == 1: agree += 1
    else: print(f"  DIVERGES {k}: {verdicts}")
print(f"  {agree}/{len(keys)} variations agree across both books")

print("\n### Repeat consistency")
groups = collections.defaultdict(list)
for r in new: groups[(r["book"], r["arm"]) + r["vkey"]].append(r["outcome"])
ident = sum(1 for v in groups.values() if len(set(v)) == 1)
print(f"  {ident}/{len(groups)} variation x book x arm groups identical across repeats")

print("\n### Remaining unsafe runs")
rem = [r for r in new if r["outcome"] in UNSAFE]
print(f"  {len(rem)}")
for r in rem[:30]:
    print(f"   [{r['book']}/{r['arm']}] {r['vkey']} r{r['repeat']} {r['outcome']} :: {r['note']}")
    print(f"      Q: {r['question']}")
    print(f"      A: {(r['run'].get('answer') or '')[:200]}")

print("\n### §12 GATE")
n = len(new)
good = sum(1 for r in new if r["outcome"] in OK)
print(f"  INCORRECT_SUCCESSFUL   = {sum(1 for r in new if r['outcome']=='INCORRECT_SUCCESSFUL')}   (required 0)")
print(f"  SILENT_SEMANTIC_ERROR  = {sum(1 for r in new if r['outcome']=='SILENT_SEMANTIC_ERROR')}   (required 0)")
print(f"  HARD_FAILURE           = {sum(1 for r in new if r['outcome']=='HARD_FAILURE')}   (required 0)")
print(f"  CORRECT + DISCLOSED    = {good}/{n} = {100*good/n:.1f}%   (commercial target >= 80%)")
rest = [r for r in new if r["outcome"] not in OK]
bad_rest = [r for r in rest if r["outcome"] not in ("HONEST_PARTIAL", "SAFE_REFUSAL")]
print(f"  remainder honest partial / safe refusal: {len(rest)-len(bad_rest)}/{len(rest)}")
