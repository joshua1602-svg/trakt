"""TRANCHE C — INVESTIGATION ONLY. No production code path reads this file.

Is the empirical stage completion rate biased by right-censoring, and by how
much?

The shipped estimator (mi_agent_api/pipeline_history.py) computes, per stage:

    rate(S) = |cases ever seen at S that were ever seen COMPLETED|
              -----------------------------------------------------
              |cases ever seen at S|

over whatever weekly extracts exist. Every case ever seen at S is in the
denominator, including one first seen at S in the final week, which has had days
rather than months to complete. If completion takes time — and the governed
config says it takes 30 to 90 days — that denominator contains cases that could
not yet have converted, and the rate is biased DOWN.

This script recomputes the same quantity three ways from the raw weekly extracts,
importing nothing from the forecast modules:

  1. NAIVE          the shipped definition, reproduced as a control
  2. MATURED-COHORT restricted to cases first seen at S at least H days before
                    the last extract, so every case in the denominator has had a
                    full H days to convert
  3. KAPLAN-MEIER   cumulative incidence of completion by day H, treating a case
                    still open at the last extract as censored at that point and
                    a WITHDRAWN case as a competing failure that never completes

Nothing here is proposed for production. The output is evidence for a decision.
"""
from __future__ import annotations
import sys
from pathlib import Path
import pandas as pd

#: The weekly pipeline extracts. Every one of them is hashed in the manifest
#: under fixture:pipeline:*, so a reader can check this reads the same bytes the
#: agent read. Override with MI_AGENT_PIPELINE_ROOT if the pack is relocated.
import os
ROOT = Path(os.environ.get(
    "MI_AGENT_PIPELINE_ROOT",
    "/tmp/claude-0/-home-user-trakt/8fa44461-4d82-5a00-b9a3-1df72087ab19/"
    "scratchpad/pipeline_root"))
# mi_agent_api/pipeline_prep.py :: _STAGE_ALIASES, restricted to the Status
# values these extracts carry.
STATUS_TO_STAGE = {"KFI ISSUED": "KFI", "APPLICATION": "APPLICATION",
                   "OFFER ISSUED": "OFFER", "COMPLETED": "COMPLETED",
                   "WITHDRAWN": "WITHDRAWN"}
ACTIVE = ("KFI", "APPLICATION", "OFFER")
# config/client/pipeline_expected_funding.yaml :: stage_days_to_fund
CONFIG_LAG = {"KFI": 90, "APPLICATION": 60, "OFFER": 30}
CONFIG_RATE = {"KFI": 0.20, "APPLICATION": 0.45, "OFFER": 0.75}


def timelines(book: str):
    """case_id -> {stage: first_seen_date}, plus terminal state and its date."""
    weeks = sorted((ROOT / book / "pipeline").iterdir())
    seen, done, drop = {}, {}, {}
    for folder in weeks:
        d = pd.Timestamp(folder.name)
        src = next(folder.glob("*.csv"))
        raw = pd.read_csv(src, low_memory=False)
        stage = raw["Status"].astype(str).str.strip().str.upper().map(STATUS_TO_STAGE)
        ident = raw["Account Number"].astype(str).str.strip()
        for cid, st in zip(ident, stage):
            if not isinstance(st, str):
                continue
            t = seen.setdefault(cid, {})
            t.setdefault(st, d)
            if st == "COMPLETED":
                done.setdefault(cid, d)
            if st == "WITHDRAWN":
                drop.setdefault(cid, d)
    last = pd.Timestamp(weeks[-1].name)
    first = pd.Timestamp(weeks[0].name)
    return seen, done, drop, first, last


def naive(seen, done):
    out = {}
    for s in ACTIVE:
        obs = [c for c, t in seen.items() if s in t]
        comp = [c for c in obs if c in done]
        out[s] = (len(comp), len(obs), (len(comp) / len(obs)) if obs else None)
    return out


def matured(seen, done, last, horizon):
    """Only cases with a FULL `horizon` days of observation after entering S."""
    out = {}
    for s in ACTIVE:
        obs = [c for c, t in seen.items()
               if s in t and (last - t[s]).days >= horizon]
        comp = [c for c in obs if c in done and (done[c] - seen[c][s]).days <= horizon]
        out[s] = (len(comp), len(obs), (len(comp) / len(obs)) if obs else None)
    return out


def km_curve(seen, done, drop, last, stage, horizon):
    """Return (incidence, n, at_risk_at_horizon, events_seen) so the estimate can
    be judged rather than quoted. A product-limit estimate whose risk set has
    emptied before the horizon is not an estimate — it is an artefact."""
    rows = []
    for c, t in seen.items():
        if stage not in t:
            continue
        entry = t[stage]
        if c in done:
            rows.append(((done[c] - entry).days, 1))
        elif c in drop:
            rows.append((horizon + 1, 0))
        else:
            rows.append(((last - entry).days, 0))
    rows = sorted((max(d, 0), e) for d, e in rows)
    n_at_risk, surv, i, events_seen = len(rows), 1.0, 0, 0
    at_risk_at_h = len(rows)
    while i < len(rows):
        day = rows[i][0]
        if day > horizon:
            break
        j, events = i, 0
        while j < len(rows) and rows[j][0] == day:
            events += rows[j][1]
            j += 1
        if events:
            surv *= (1 - events / n_at_risk)
            events_seen += events
        n_at_risk -= (j - i)
        at_risk_at_h = n_at_risk
        i = j
    return 1 - surv, len(rows), at_risk_at_h, events_seen


def kaplan_meier(seen, done, drop, last, horizon):
    """Cumulative incidence of completion by `horizon` days after entering S.

    Completion is the event. A WITHDRAWN case is a competing failure and stays in
    the risk set as a case that will never complete — removing it would inflate
    the rate. A case still open at the last extract is censored there.
    """
    out = {}
    for s in ACTIVE:
        rows = []
        for c, t in seen.items():
            if s not in t:
                continue
            entry = t[s]
            if c in done:
                rows.append(((done[c] - entry).days, 1))
            elif c in drop:
                # never completes; contributes to the risk set for the whole
                # horizon and never produces an event
                rows.append((horizon + 1, 0))
            else:
                rows.append(((last - entry).days, 0))
        rows = [(max(d, 0), e) for d, e in rows]
        rows.sort()
        n_at_risk = len(rows)
        surv, i = 1.0, 0
        while i < len(rows):
            day = rows[i][0]
            if day > horizon:
                break
            j = i
            events = 0
            while j < len(rows) and rows[j][0] == day:
                events += rows[j][1]
                j += 1
            if events:
                surv *= (1 - events / n_at_risk)
            n_at_risk -= (j - i)
            i = j
        out[s] = (1 - surv, len(rows))
    return out


for book in ("alderbridge", "kestrelmoor"):
    seen, done, drop, first, last = timelines(book)
    window = (last - first).days
    print("=" * 96)
    print(f"{book.upper()}   {len(seen)} cases   window {first.date()} -> {last.date()} "
          f"= {window} days")
    n = naive(seen, done)
    print(f"\n  {'stage':12s} {'config':>7s} {'NAIVE':>18s} {'MATURED @ config lag':>28s} "
          f"{'KM @ config lag':>18s}")
    for s in ACTIVE:
        h = CONFIG_LAG[s]
        m = matured(seen, done, last, h)[s]
        k = kaplan_meier(seen, done, drop, last, h)[s]
        nn = n[s]
        mat = (f"{m[2]:.4f} ({m[0]}/{m[1]})" if m[2] is not None else
               f"NO MATURED SAMPLE (0/{m[1]})")
        print(f"  {s:12s} {CONFIG_RATE[s]:7.2f} {nn[2]:8.4f} ({nn[0]:>3}/{nn[1]:<4}) "
              f"{mat:>28s} {k[0]:11.4f} (n={k[1]})")
    print(f"\n  maturity check — cases with a FULL config lag of observation:")
    for s in ACTIVE:
        h = CONFIG_LAG[s]
        m = matured(seen, done, last, h)[s]
        verdict = ("INSUFFICIENT — window shorter than the lag" if window < h
                   else f"{m[1]} matured of {n[s][1]} observed")
        print(f"    {s:12s} lag {h:3d}d   {verdict}")

    print(f"\n  is the KM estimate identified? risk set remaining AT the horizon:")
    print(f"    {'stage':12s} {'horizon':>8s} {'n':>6s} {'at risk':>8s} {'events':>7s}  estimate")
    for s in ACTIVE:
        for h in (14, 28, 42, 56, CONFIG_LAG[s]):
            inc, nn, risk, ev = km_curve(seen, done, drop, last, s, h)
            flag = "" if risk >= 20 else "   <- risk set exhausted; not identified"
            print(f"    {s:12s} {h:6d}d  {nn:6d} {risk:8d} {ev:7d}  {inc:.4f}{flag}")

    print(f"\n  observed days from first sighting at a stage to completion:")
    for s in ACTIVE:
        days = sorted((done[c] - t[s]).days for c, t in seen.items()
                      if s in t and c in done)
        if days:
            mid = days[len(days) // 2]
            print(f"    {s:12s} n={len(days):3d}  min={days[0]:3d}  median={mid:3d}  "
                  f"max={days[-1]:3d}   (config assumes {CONFIG_LAG[s]}d)")
