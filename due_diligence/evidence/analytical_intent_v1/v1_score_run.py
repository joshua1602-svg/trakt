"""Apply the FROZEN scorer to a set of run files and emit the distribution.

    python3 v1_score_run.py <out.json> <runs.json> ...

nl_score.py is not modified: §11 requires the same scoring as the baseline.
"""
from __future__ import annotations
import json, sys, collections
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from nl_score import grade, EXPECTED   # noqa: E402

OUT, PATHS = sys.argv[1], sys.argv[2:]
scored = []
for path in PATHS:
    for entry in json.load(open(path)):
        for i, run in enumerate(entry.get("runs") or []):
            outcome, causes, note = grade(entry["intent"], run)
            scored.append({
                "book": entry["book"], "arm": entry["arm"],
                "intent": entry["intent"], "variation": entry["variation"],
                "pair": entry.get("pair"), "repeat": i,
                "question": entry["question"], "outcome": outcome,
                "causes": causes, "note": note, "route": run.get("route"),
                "planIntent": run.get("intent"),
                "parserUsed": run.get("parserUsed"),
                "guard": run.get("semanticGuard"),
                "ok": run.get("ok"),
                "answer": (run.get("answer") or "")[:400],
            })
Path(OUT).write_text(json.dumps(scored, indent=1))

UNSAFE = {"INCORRECT_SUCCESSFUL", "SILENT_SEMANTIC_ERROR", "HARD_FAILURE"}
OK = {"CORRECT", "CORRECT_WITH_DISCLOSED_LIMITATION"}
n = len(scored)
dist = collections.Counter(s["outcome"] for s in scored)
print(f"runs {n}")
for k, v in sorted(dist.items(), key=lambda kv: -kv[1]):
    print(f"  {k:36s} {v:4d}  {100*v/n:5.1f}%")
print(f"  {'-- CORRECT + DISCLOSED':36s} {sum(dist[k] for k in OK):4d}  "
      f"{100*sum(dist[k] for k in OK)/n:5.1f}%")
print(f"  {'-- UNSAFE':36s} {sum(dist[k] for k in UNSAFE):4d}  "
      f"{100*sum(dist[k] for k in UNSAFE)/n:5.1f}%")
print(f"  genuine LLM parses {sum(1 for s in scored if s['parserUsed']=='llm')}/{n}")

print("\nby intent:")
for q in sorted(EXPECTED):
    rows = [s for s in scored if s["intent"] == q]
    unsafe = sum(1 for s in rows if s["outcome"] in UNSAFE)
    good = sum(1 for s in rows if s["outcome"] in OK)
    ref = sum(1 for s in rows if s["outcome"] == "SAFE_REFUSAL")
    part = sum(1 for s in rows if s["outcome"] == "HONEST_PARTIAL")
    print(f"  {q}  runs={len(rows):3d}  ok={good:3d}  partial={part:3d}  "
          f"refusal={ref:3d}  UNSAFE={unsafe:3d}")

print("\nby variation (production arm, majority outcome):")
key = lambda s: (s["intent"], s["variation"], s["pair"] or "")
groups = collections.defaultdict(list)
for s in scored:
    groups[key(s)].append(s)
for k in sorted(groups):
    rows = groups[k]
    maj = collections.Counter(r["outcome"] for r in rows).most_common(1)[0]
    routes = sorted({str(r["route"]) for r in rows})
    print(f"  {k[0]}.{k[1]}{'/'+k[2] if k[2] else '':16s} {maj[0]:34s} "
          f"({maj[1]}/{len(rows)}) routes={routes}")

if any(s["outcome"] in UNSAFE for s in scored):
    print("\nUNSAFE RUNS:")
    for s in scored:
        if s["outcome"] in UNSAFE:
            print(f"  [{s['book']}/{s['arm']}] {s['intent']}.{s['variation']}"
                  f"{'/'+s['pair'] if s['pair'] else ''} r{s['repeat']} "
                  f"{s['outcome']} :: {s['note']}")
            print(f"      Q: {s['question']}")
            print(f"      A: {s['answer'][:200]}")
