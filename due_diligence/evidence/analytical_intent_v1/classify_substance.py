"""Mechanical partition of every response by SUBSTANCE, not by reading prose.

The rule below is structural and applied by script. It never inspects answer
text, so it cannot be tuned to a phrasing.

  CONTROLLED_REFUSAL        ok is False.
                            The system declined and said why.

  INFORMATIONAL_NO_COMPUTE  ok is True, but the response carries NO artifact,
                            NO structured finding and NO reconciliation block.
                            Nothing was computed from the portfolio: the route
                            ran and reported that an input it needs is absent.

  SUBSTANTIVE_CALCULATED    ok is True and the response carries at least one
                            artifact or structured finding — i.e. a governed
                            figure was computed from the data.

  OTHER                     anything the three rules above do not cover.
"""
from __future__ import annotations
import collections, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from nl_score import grade   # frozen scorer, unmodified

REFUSAL, INFO, SUBSTANTIVE, OTHER = (
    "CONTROLLED_REFUSAL", "INFORMATIONAL_NO_COMPUTE",
    "SUBSTANTIVE_CALCULATED", "OTHER")
OK_GRADES = {"CORRECT", "CORRECT_WITH_DISCLOSED_LIMITATION"}


def substance(run: dict) -> str:
    if not run.get("ok"):
        return REFUSAL
    artifacts = run.get("artifacts") or []
    findings = run.get("findings") or []
    recon = run.get("reconciliation")
    if not artifacts and not findings and recon is None:
        return INFO
    if artifacts or findings:
        return SUBSTANTIVE
    return OTHER


SETS = {
    "V1": ["v1_nl_alderbridge_production.json", "v1_nl_alderbridge_forced_llm.json",
           "v1_nl_kestrelmoor_production.json", "v1_nl_kestrelmoor_forced_llm.json"],
    "BASELINE": ["nl_alderbridge_production.json", "nl_alderbridge_forced_llm.json",
                 "nl_kestrelmoor_production.json", "nl_kestrelmoor_forced_llm.json"],
}
S = Path(__file__).resolve().parent.parent

for name, files in SETS.items():
    rows = []
    for p in files:
        for e in json.load(open(S / p)):
            for r in e["runs"]:
                rows.append({"book": e["book"], "arm": e["arm"], "intent": e["intent"],
                             "variation": e["variation"], "pair": e.get("pair") or "",
                             "grade": grade(e["intent"], r)[0], "substance": substance(r)})
    n = len(rows)
    print("=" * 96)
    print(f"{name} — {n} runs")
    c = collections.Counter(r["substance"] for r in rows)
    for k in (SUBSTANTIVE, INFO, REFUSAL, OTHER):
        print(f"   {k:26s} {c[k]:4d}  {100*c[k]/n:5.1f}%")
    ok = [r for r in rows if r["grade"] in OK_GRADES]
    ok_sub = [r for r in ok if r["substance"] == SUBSTANTIVE]
    ok_info = [r for r in ok if r["substance"] == INFO]
    print(f"\n   graded CORRECT/DISCLOSED        {len(ok):4d}  {100*len(ok)/n:5.1f}%   <- headline")
    print(f"      of which SUBSTANTIVE          {len(ok_sub):4d}  {100*len(ok_sub)/n:5.1f}%   <- corrected headline")
    print(f"      of which INFORMATIONAL        {len(ok_info):4d}  {100*len(ok_info)/n:5.1f}%")
    print("\n   per book / arm (substantive-correct rate):")
    for key in sorted({(r["book"], r["arm"]) for r in rows}):
        sub = [r for r in rows if (r["book"], r["arm"]) == key]
        s_ok = sum(1 for r in sub if r["grade"] in OK_GRADES and r["substance"] == SUBSTANTIVE)
        h_ok = sum(1 for r in sub if r["grade"] in OK_GRADES)
        print(f"      {key[0]:12s} {key[1]:12s} headline {100*h_ok/len(sub):5.1f}%   "
              f"substantive {100*s_ok/len(sub):5.1f}%   (n={len(sub)})")
    info_by = collections.Counter(
        f"{r['intent']}.{r['variation']}{'/'+r['pair'] if r['pair'] else ''} [{r['book']}]"
        for r in ok_info)
    if info_by:
        print("\n   informational-but-counted-correct, by variation:")
        for k, v in sorted(info_by.items()):
            print(f"      {k:34s} {v:3d}")
