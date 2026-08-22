"""A5 — the numerical neutrality proof for Tranche A.

Compares the FIGURES in every bank answer before and after, and classifies each
difference. The gate:

  FAIL  a figure that was printed before now prints a different value
  FAIL  a figure appears that no finding holds
  OK    a figure is withheld because A2 refused a competing measure
  OK    a figure is withheld because A3 declines to narrate a rolling-cohort
        delta as a movement (the finding still holds it; it is not narrated)
  OK    a figure is newly printed and a finding already held it
"""
from __future__ import annotations
import json, re, sys

#: Money / percent / count tokens. Dates are excluded explicitly: an ISO date
#: fragment is not a figure, and letting one through makes the proof cry wolf.
TOKEN = re.compile(r"[-+−]?£?\d[\d,]*\.?\d*\s?(?:bn|m|k|pp|%)?")
DATE = re.compile(r"\b\d{4}-\d{2}(-\d{2})?\b")


def figures(text: str) -> list[str]:
    text = DATE.sub(" ", text or "")
    out = []
    for t in TOKEN.findall(text):
        t = t.strip()
        if any(c.isdigit() for c in t):
            out.append(t.replace("−", "-").replace(" ", ""))
    return out


def held(findings: list) -> set:
    """Every numeric a finding holds, in the shapes the narrator could print."""
    vals = set()

    def rec(v):
        if v is None:
            return
        try:
            f = float(v)
        except (TypeError, ValueError):
            return
        vals.add(round(abs(f), 2))
        vals.add(round(abs(f) / 1e6, 1))
        vals.add(round(abs(f) / 1e9, 2))
        vals.add(round(abs(f), 1))
        vals.add(round(abs(f) * 100, 1))
        vals.add(round(abs(f) * 100, 2))
    for fd in findings:
        for k in ("value", "priorValue", "change", "relativeChange",
                  "comparandValue", "forecastValue", "share", "priorShare",
                  "limit", "headroom", "rank"):
            rec(fd.get(k))
        for side in ("population", "comparand"):
            p = fd.get(side) or {}
            for k in ("rows", "rowsBefore", "rowsPrior", "exposure"):
                rec(p.get(k))
    return vals


def numeric(tok: str):
    t = tok.replace("£", "").replace(",", "").replace("+", "")
    mult = 1.0
    for suf, m in (("bn", 1e9), ("m", 1e6), ("k", 1e3), ("pp", 1), ("%", 1)):
        if t.endswith(suf):
            t = t[: -len(suf)]; mult = m; break
    try:
        return abs(float(t)) * mult
    except ValueError:
        return None


before = {r["q"]: r for r in json.load(open("tA_before2.json"))}
after = {r["q"]: r for r in json.load(open("tA_after2.json"))}

value_changes, unheld, a2, a3, added_ok, identical = [], [], [], [], [], 0
for q in before:
    fb, fa = figures(before[q]["answer"]), figures(after[q]["answer"])
    if fb == fa:
        identical += 1
        continue
    gone, new = [x for x in fb if x not in fa], [x for x in fa if x not in fb]
    ans_after = after[q]["answer"]
    held_after = held(after[q]["findings"])
    held_before = held(before[q]["findings"])

    for tok in new:
        v = numeric(tok)
        if v is None:
            continue
        if any(abs(v - h) < max(0.02, abs(h) * 0.005) for h in held_before | held_after):
            added_ok.append((q, tok))
        else:
            unheld.append((q, tok))
    for tok in gone:
        if "could not be reported" in ans_after:
            a2.append((q, tok))
        elif "rolling cohorts" in ans_after:
            a3.append((q, tok))
        else:
            value_changes.append((q, tok))

print(f"questions                      : {len(before)}")
print(f"figure sets identical          : {identical}")
print()
print(f"FAIL  value changed            : {len(value_changes)}")
for q, t in value_changes:
    print(f"        {t:12s} {q[:70]}")
print(f"FAIL  printed but no finding   : {len(unheld)}")
for q, t in unheld:
    print(f"        {t:12s} {q[:70]}")
print()
print(f"OK    withheld by A2 refusal   : {len(a2)}  {sorted({t for _, t in a2})}")
print(f"OK    delta not narrated (A3)  : {len(a3)}  {sorted({t for _, t in a3})}")
print(f"OK    newly printed, held      : {len(added_ok)}  {sorted({t for _, t in added_ok})}")
print()
verdict = "PASS" if not value_changes and not unheld else "FAIL"
print(f"TRANCHE A NUMERICAL NEUTRALITY: {verdict}")
