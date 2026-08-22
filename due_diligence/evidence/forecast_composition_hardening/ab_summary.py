"""The like-for-like A/B: same bank, same parser, two code revisions.

Prints the frozen scorer's grades, the substance partition and the parser mix
for each revision, then every run whose grade or answer text differs.
"""
from __future__ import annotations
import collections, json, sys
from pathlib import Path
#: nl_score.py is the FROZEN control and lives with the V1 pack; it is imported,
#: never copied, so there can only ever be one of it.
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "analytical_intent_v1"))
from nl_score import grade                              # frozen control, unmodified
from classify_substance_guarded import substance        # structural, reads no prose

UNSAFE = {"INCORRECT_SUCCESSFUL", "SILENT_SEMANTIC_ERROR", "HARD_FAILURE"}
OK = {"CORRECT", "CORRECT_WITH_DISCLOSED_LIMITATION"}
ARMS = ["nl_alderbridge_production.json", "nl_alderbridge_forced_llm.json",
        "nl_kestrelmoor_production.json", "nl_kestrelmoor_forced_llm.json"]


def load(d):
    out = {}
    for p in ARMS:
        for e in json.load(open(Path(d) / p)):
            for i, r in enumerate(e.get("runs") or []):
                out[(e["book"], e["arm"], e["intent"], e["variation"],
                     e.get("pair") or "", i)] = (grade(e["intent"], r)[0], substance(r),
                                                 (r.get("answer") or "").strip(), r, e["question"])
    return out


def report(label, m):
    n = len(m)
    c = collections.Counter(v[0] for v in m.values())
    s = collections.Counter(v[1] for v in m.values())
    p = collections.Counter(v[3].get("parserUsed") for v in m.values())
    print(f"== {label}  ({n} runs)")
    print(f"   grades          : {dict(c.most_common())}")
    print(f"   unsafe          : {sum(c[k] for k in UNSAFE)}")
    print(f"   correct/disclosed: {sum(c[k] for k in OK)} ({100*sum(c[k] for k in OK)/n:.1f}%)")
    for k in ("SUBSTANTIVE_CALCULATED", "INFORMATIONAL_NO_COMPUTE", "CONTROLLED_REFUSAL"):
        print(f"   {k:24s}: {s[k]:4d} ({100*s[k]/n:.1f}%)")
    print(f"   parser mix      : {dict(p)}")


a, b = load(sys.argv[1]), load(sys.argv[2])
report(f"BEFORE  {sys.argv[1]}", a)
report(f"AFTER   {sys.argv[2]}", b)
common = sorted(set(a) & set(b))
gd = [k for k in common if a[k][0] != b[k][0]]
ad = [k for k in common if a[k][2] != b[k][2]]
print(f"\nmatched runs {len(common)}   grade differences {len(gd)}   answer-text differences {len(ad)}")
print("grade transitions:", dict(collections.Counter((a[k][0], b[k][0]) for k in gd)))
print("text differences by variation:",
      dict(sorted(collections.Counter(f"{k[2]}.{k[3]}{'/'+k[4] if k[4] else ''}" for k in ad).items())))
for k in gd:
    print(f"\n--- {k[2]}.{k[3]}{'/'+k[4] if k[4] else ''} {k[0]}/{k[1]} repeat {k[5]}: {a[k][0]} -> {b[k][0]}")
    print(f"    parser before={a[k][3].get('parserUsed')} after={b[k][3].get('parserUsed')}")
    print(f"    before: {a[k][2][:220]}")
    print(f"    after : {b[k][2][:220]}")
