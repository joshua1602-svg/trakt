"""How much does the SAME question, on the SAME code, disagree with itself?

The bank runs each variation several times. Within one run FILE — one book, one
arm, one commit — those repeats should agree. Where they do not, the disagreement
is the measurement's own noise floor, and any before/after difference smaller
than it means nothing.

This exists because the LLM arm's before/after comparison showed 31 grade
differences running in BOTH directions. That is the signature of noise, not of a
code change, and this script is how that was established rather than asserted.
"""
from __future__ import annotations
import collections, json, sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from nl_score import grade   # frozen control, unmodified

ARMS = ["nl_alderbridge_production.json", "nl_alderbridge_forced_llm.json",
        "nl_kestrelmoor_production.json", "nl_kestrelmoor_forced_llm.json"]


def variance(d, pre=""):
    unstable, cells, detail = 0, 0, []
    for p in ARMS:
        for e in json.load(open(Path(d) / (pre + p))):
            grades = {grade(e["intent"], r)[0] for r in e.get("runs") or []}
            cells += 1
            if len(grades) > 1:
                unstable += 1
                key = f"{e['intent']}.{e['variation']}" + (f"/{e['pair']}" if e.get("pair") else "")
                detail.append((key, e["book"], e["arm"], sorted(grades)))
    return unstable, cells, detail


for spec in sys.argv[1:]:
    label, _, path = spec.partition("=")
    d, _, pre = path.partition(":")
    u, c, detail = variance(d, pre)
    print(f"{label:34s} cells whose repeats DISAGREE: {u:3d}/{c}  ({100*u/c:.1f}%)")
    for k in detail:
        print(f"      {k[0]:10s} {k[1]:12s} {k[2]:12s} {k[3]}")
    print()
