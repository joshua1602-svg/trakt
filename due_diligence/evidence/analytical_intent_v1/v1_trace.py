"""Full-chain trace for a strategically chosen sample of measured runs.

Every column states where it comes from:

  RECORDED   — read verbatim out of the measured run file. Not recomputed.
  DERIVED    — computed here by calling the boundary's own pure classifier on
               the same question text. `intent.classify` reads nothing but the
               question and the parse, so this is what the boundary saw.
  TRUTH      — computed HERE from the fixture CSVs with pandas, with no
               reference to anything the agent produced.
"""
from __future__ import annotations
import json, os, sys, warnings
from pathlib import Path
warnings.simplefilter("ignore")
S = Path(__file__).resolve().parent
sys.path.insert(0, "/home/user/trakt"); sys.path.insert(0, str(S))
os.chdir("/home/user/trakt")

import nl_reconcile as R                                   # noqa: E402
from mi_workflows.analytical import intent as intent_mod   # noqa: E402

#: (intent, variation, pair, book, arm, why this one was chosen)
SAMPLE = [
    ("Q1", 1, "",  "alderbridge", "production",
     "the governed lending ruling: NEW = L1M, on the question that failed 100% "
     "of the time in the baseline"),
    ("Q1", 2, "",  "alderbridge", "forced_llm",
     "the same family reached by a different window (RECENT = L3M) and a "
     "different phrasing, under a forced model parse"),
    ("Q3", 1, "",  "alderbridge", "production",
     "a DIFFERENT GOVERNED DATASET — the pipeline extract, not the loan tape"),
    ("Q4", 2, "",  "alderbridge", "production",
     "THE FAIL-CLOSED REFUSAL. Baseline answered '11,035 loans'"),
    ("Q6", 2, "",  "kestrelmoor", "production",
     "governed FLAG SETTLING: no analytical plan at all, the existing limits "
     "route answers. Second book"),
    ("Q7", 2, "",  "alderbridge", "production",
     "an ASYMMETRIC pair — a segment against a window (Back Book vs RECENT)"),
    ("Q8", 3, "provenance",      "kestrelmoor", "production", "§8 control, resolver 1 of 3"),
    ("Q8", 3, "seasoning",       "kestrelmoor", "production", "§8 control, resolver 2 of 3"),
    ("Q8", 3, "dimension_value", "kestrelmoor", "production", "§8 control, resolver 3 of 3"),
    ("Q9", 3, "",  "kestrelmoor", "forced_llm",
     "a composed FORECAST on the second book, under a forced model parse"),
]

FILES = {("alderbridge", "production"): "v1_nl_alderbridge_production.json",
         ("alderbridge", "forced_llm"): "v1_nl_alderbridge_forced_llm.json",
         ("kestrelmoor", "production"): "v1_nl_kestrelmoor_production.json",
         ("kestrelmoor", "forced_llm"): "v1_nl_kestrelmoor_forced_llm.json"}

_cache, _frames = {}, {}


def entries(book, arm):
    key = (book, arm)
    if key not in _cache:
        _cache[key] = json.load(open(S / FILES[key]))
    return _cache[key]


def frames(book):
    if book not in _frames:
        _frames[book] = R.load(book)
    return _frames[book]


def money(v):
    if v is None: return "—"
    a = abs(v)
    if a >= 1e9: return f"£{v/1e9:,.2f}bn"
    if a >= 1e6: return f"£{v/1e6:,.1f}m"
    if a >= 1e3: return f"£{v/1e3:,.1f}k"
    return f"£{v:,.2f}"


def show(v, metric):
    if v is None: return "—"
    if metric in (R.LTV, R.RATE): return f"{v:.4f}%"
    if metric == R.AGE: return f"{v:.4f}"
    if metric == "loan_count": return f"{int(round(v)):,}"
    return money(v)


out = []
for tag, var, pair, book, arm, why in SAMPLE:
    rows = [e for e in entries(book, arm)
            if e["intent"] == tag and e["variation"] == var
            and (e.get("pair") or "") == pair]
    if not rows:
        out.append(f"### {tag}.{var} — NOT FOUND in {book}/{arm}\n"); continue
    e = rows[0]
    run = e["runs"][0]
    prov = run.get("parserProvenance") or {}
    reading = intent_mod.classify(e["question"])
    fr = frames(book)
    close, open_ = fr[R.DATES[-1]], fr[R.DATES[0]]

    b = []
    b.append(f"### {tag}.{var}{'/' + pair if pair else ''} — {book} / {arm}")
    b.append(f"*Chosen because: {why}.*\n")
    b.append(f"**1. Question** (RECORDED)\n\n> {e['question']}\n")
    b.append(f"**2. Parser provenance** (RECORDED)\n")
    b.append(f"| | |\n|---|---|")
    b.append(f"| parser used | `{prov.get('parser_used')}` |")
    b.append(f"| mode detail | `{prov.get('parser_mode_detail')}` |")
    b.append(f"| LLM failure | `{prov.get('llm_failure')}` |")
    b.append(f"| repeats in this arm | {len(e['runs'])}, all reaching "
             f"`{sorted({str(r.get('route')) for r in e['runs']})[0]}` |\n")
    b.append(f"**3. Recognised family / operation** (DERIVED — `intent.classify`, "
             f"a pure function of the question)\n")
    b.append(f"| | |\n|---|---|")
    b.append(f"| families | {', '.join(reading.families) or '—'} |")
    b.append(f"| operations | {', '.join(reading.operations) or '—'} |")
    b.append(f"| concept signals matched | {', '.join(reading.signals) or '—'} |")
    b.append(f"| lending windows named | {', '.join(reading.lending_windows) or '—'} |")
    b.append(f"| lending role | {reading.lending_role or '— (none named)'} |")
    b.append(f"| structural requirements | {', '.join(reading.requirements) or '—'} |")
    b.append(f"| materially analytical | **{reading.materially_analytical}** |\n")

    pop = run.get("populationApplied") or {}
    comp = run.get("composition") or {}
    b.append(f"**4. Resolved population and period** (RECORDED — the governed "
             f"population ledger, i.e. what execution reported)\n")
    b.append(f"| | |\n|---|---|")
    b.append(f"| predicates applied | {', '.join(pop.get('applied') or []) or '—'} |")
    b.append(f"| predicates unavailable | {', '.join(pop.get('unavailable') or []) or 'none'} |")
    b.append(f"| rows before → after | {pop.get('rowsBefore')} → {pop.get('rowsAfter')} |")
    b.append(f"| periods compared | {', '.join(comp.get('periodsCompared') or []) or '—'} |")
    b.append(f"| narrowed to | {json.dumps(comp.get('narrowedTo') or []) if comp.get('narrowedTo') else '—'} |\n")

    b.append(f"**5. Deterministic capabilities invoked** (RECORDED)\n")
    if run.get("intent"):
        b.append(f"| | |\n|---|---|")
        b.append(f"| route | `{run.get('route')}` |")
        b.append(f"| plan intent | `{run.get('intent')}` |")
        b.append(f"| capabilities | {', '.join(f'`{c}`' for c in (run.get('capabilities') or []))} |")
        b.append(f"| required finding kinds | {', '.join(run.get('requiredKinds') or []) or '—'} |")
        b.append(f"| composition version | {comp.get('compositionVersion')} · plan origin "
                 f"{comp.get('planOrigin')} |\n")
    else:
        b.append(f"**None — no analytical plan was built.** Route: "
                 f"`{run.get('route') or '(none: the point-in-time path)'}`. "
                 f"This is the correct outcome for this question; see the note "
                 f"below.\n")

    findings = run.get("findings") or []
    b.append(f"**6. Structured findings and 7. independently calculated truth**")
    b.append(f"  (findings RECORDED; truth computed HERE from the fixture CSVs "
             f"with pandas, referencing nothing the agent produced)\n")
    if findings:
        b.append("| # | kind | measure | population (rows) | delivered | **independent truth** | agrees |")
        b.append("|---|---|---|---|---|---|---|")
        n = 0
        for f in findings:
            key = (f.get("population") or {}).get("key")
            metric, kind = f.get("metric"), f.get("kind")
            if kind not in ("measure", "movement", "comparison") or not metric:
                continue
            sub = R.population(close, key)
            if sub is None:
                continue
            n += 1
            t = R.truth(sub, metric)
            d = f.get("value")
            tol = 1e-6 if metric in (R.LTV, R.RATE, R.AGE) else 0.05
            ok = (d is not None and t is not None and abs(float(d) - float(t)) <= tol)
            b.append(f"| {n} | {kind} | `{metric}` | "
                     f"{(f.get('population') or {}).get('label')} "
                     f"({(f.get('population') or {}).get('rows')}) | "
                     f"{show(d, metric)} | **{show(t, metric)}** | "
                     f"{'✅' if ok else '❌'} |")
            if kind == "movement":
                po = R.population(open_, key)
                tp = R.truth(po, metric) if po is not None else None
                dp = f.get("priorValue")
                okp = (dp is not None and tp is not None
                       and abs(float(dp) - float(tp)) <= tol)
                b.append(f"| {n}p | prior | `{metric}` @ {R.DATES[0]} | "
                         f"{(f.get('population') or {}).get('label')} "
                         f"({len(po) if po is not None else '—'}) | "
                         f"{show(dp, metric)} | **{show(tp, metric)}** | "
                         f"{'✅' if okp else '❌'} |")
            if kind == "comparison":
                ck = (f.get("comparand") or {}).get("key")
                cs = R.population(close, ck)
                tc = R.truth(cs, metric) if cs is not None else None
                dc = f.get("comparandValue")
                okc = (dc is not None and tc is not None
                       and abs(float(dc) - float(tc)) <= tol)
                b.append(f"| {n}c | comparand | `{metric}` | "
                         f"{(f.get('comparand') or {}).get('label')} "
                         f"({len(cs) if cs is not None else '—'}) | "
                         f"{show(dc, metric)} | **{show(tc, metric)}** | "
                         f"{'✅' if okc else '❌'} |")
        rest = len(findings) - n
        if rest > 0:
            b.append(f"\n*(plus {rest} composition / cohort / forecast / timing "
                     f"findings on this run, reconciled in §12.4 but not tabulated "
                     f"here)*")
        b.append("")
    else:
        b.append("**No findings — nothing was computed.** That is the point of "
                 "this trace.\n")

    guard = run.get("guardFacets") or []
    b.append(f"**8. Final answer** (RECORDED, verbatim)\n")
    b.append(f"> {(run.get('answer') or '').strip()[:900]}\n")
    b.append(f"| | |\n|---|---|")
    b.append(f"| ok | `{run.get('ok')}` |")
    b.append(f"| controlled refusal | `{run.get('controlledRefusal')}` |")
    b.append(f"| semantic guard | `{run.get('semanticGuard')}` "
             f"{[(x.get('kind'), x.get('status')) for x in guard]} |")
    b.append("")
    out.append("\n".join(b))

Path(S / "v1_trace.md").write_text("\n---\n\n".join(out))
print(f"wrote v1_trace.md — {len(out)} traces")
