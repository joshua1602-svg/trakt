"""TRUTH — independent recomputation of the pipeline conversion forecast.

Imports NOTHING from the forecast modules. Every assumption below is an explicit
constant, with a comment naming where the value came from. If a value could not
be recovered from code or config it is not substituted — the script stops.

Recomputes, from the raw weekly M2L extracts alone:
  * per-stage empirical completion rates (the historical model)
  * per-stage expected amounts, at the as-of extract
  * the expected landing month per stage
  * the open-pipeline expectation and the whole-extract expectation
"""
from __future__ import annotations
import sys
from pathlib import Path
import pandas as pd

# --------------------------------------------------------------------------- #
# ASSUMPTIONS — each names its source. Nothing here is inferred from output.
# --------------------------------------------------------------------------- #
# config/client/pipeline_expected_funding.yaml :: stage_probabilities
CONFIG_STAGE_PROBABILITY = {"KFI": 0.20, "APPLICATION": 0.45,
                            "OFFER": 0.75, "COMPLETED": 1.00}
# config/client/pipeline_expected_funding.yaml :: stage_days_to_fund
CONFIG_STAGE_DAYS_TO_FUND = {"KFI": 90, "APPLICATION": 60, "OFFER": 30, "COMPLETED": 0}
# config/client/pipeline_expected_funding.yaml :: include_stages
CONFIG_INCLUDE_STAGES = ("KFI", "APPLICATION", "OFFER")
# config/client/pipeline_expected_funding.yaml :: exclude_stages
CONFIG_EXCLUDE_STAGES = ("WITHDRAWN", "COMPLETED")
# mi_agent_api/pipeline_prep.py :: ACTIVE_STAGES  (line 369)
ACTIVE_STAGES = ("KFI", "APPLICATION", "OFFER")
# mi_agent_api/pipeline_history.py :: MIN_OBSERVATIONS  (line 31)
MIN_OBSERVATIONS = 12
# mi_agent_api/pipeline_prep.py :: _STAGE_ALIASES  (lines 52-58), restricted to
# the Status values this extract actually carries.
STATUS_TO_STAGE = {"KFI ISSUED": "KFI", "APPLICATION": "APPLICATION",
                   "OFFER ISSUED": "OFFER", "COMPLETED": "COMPLETED",
                   "WITHDRAWN": "WITHDRAWN"}
# mi_agent_api/pipeline_prep.py :: case_stage_frame (line 372) — case identity is
# the pipeline case identifier, i.e. the account number in the M2L KFI extract;
# completion date is the funds-released date.
CASE_ID_COLUMN = "Account Number"
STAGE_COLUMN = "Status"
COMPLETION_DATE_COLUMN = "Date Funds Released"
AMOUNT_COLUMN = "Loan Amount"

SCRATCH = Path(__file__).resolve().parent.parent


def stage_of(series: pd.Series) -> pd.Series:
    s = series.astype(str).str.strip().str.upper()
    unmapped = sorted(set(s) - set(STATUS_TO_STAGE))
    if unmapped:
        sys.exit(f"STOP: unmapped Status value(s) {unmapped} — assumption not "
                 f"recoverable, not substituting a plausible mapping.")
    return s.map(STATUS_TO_STAGE)


def weekly_files(book: str) -> list[tuple[str, Path]]:
    root = SCRATCH / "pipeline_root" / book / "pipeline"
    out = []
    for d in sorted(p for p in root.iterdir() if p.is_dir()):
        csv = next(iter(sorted(d.glob("*.csv"))), None)
        if csv:
            out.append((d.name, csv))
    return out


def historical_rates(book: str) -> tuple[dict, dict]:
    """Empirical per-stage completion rate, tracked case-by-case across weeks."""
    files = weekly_files(book)
    timelines: dict[str, dict] = {}
    for _date, csv in files:
        df = pd.read_csv(csv, low_memory=False)
        stage = stage_of(df[STAGE_COLUMN])
        cid = df[CASE_ID_COLUMN].astype(str).str.strip()
        for c, st in zip(cid, stage):
            t = timelines.setdefault(c, set())
            t.add(st)
    observed = {s: 0 for s in ACTIVE_STAGES}
    completed = {s: 0 for s in ACTIVE_STAGES}
    for _c, seen in timelines.items():
        ever_completed = "COMPLETED" in seen
        for s in ACTIVE_STAGES:
            if s in seen:
                observed[s] += 1
                if ever_completed:
                    completed[s] += 1
    rates, detail = {}, {}
    for s in ACTIVE_STAGES:
        obs, comp = observed[s], completed[s]
        rate = round(comp / obs, 4) if obs else None
        sufficient = obs >= MIN_OBSERVATIONS and rate is not None
        detail[s] = {"observed": obs, "completed": comp, "rate": rate,
                     "sufficient": sufficient}
        if sufficient:
            rates[s] = rate
    return rates, {"weeks": len(files), "cases_tracked": len(timelines),
                   "per_stage": detail}


def recompute(book: str) -> dict:
    files = weekly_files(book)
    as_of, latest = files[-1]
    df = pd.read_csv(latest, low_memory=False)
    df["_stage"] = stage_of(df[STAGE_COLUMN])
    df["_amount"] = pd.to_numeric(df[AMOUNT_COLUMN], errors="coerce").fillna(0.0)
    rates, hist = historical_rates(book)

    rows = []
    for s in sorted(set(df["_stage"])):
        sub = df[df["_stage"] == s]
        if s in rates:
            prob, basis = rates[s], f"historical ({rates[s]:.4f})"
        elif s in CONFIG_STAGE_PROBABILITY:
            prob, basis = CONFIG_STAGE_PROBABILITY[s], "config"
        else:
            sys.exit(f"STOP: no probability recoverable for stage {s!r}.")
        land = (pd.Timestamp(as_of) +
                pd.Timedelta(days=CONFIG_STAGE_DAYS_TO_FUND.get(s, 0))).strftime("%Y-%m")
        rows.append({"stage": s, "cases": len(sub),
                     "gross": round(float(sub["_amount"].sum()), 2),
                     "probability": prob, "basis": basis,
                     "expected": round(float(sub["_amount"].sum()) * prob, 2),
                     "lands": land})
    open_rows = [r for r in rows if r["stage"] in CONFIG_INCLUDE_STAGES]
    excl_rows = [r for r in rows if r["stage"] in CONFIG_EXCLUDE_STAGES]
    return {"book": book, "as_of": as_of, "rows": rows, "hist": hist,
            "open_expected": round(sum(r["expected"] for r in open_rows), 2),
            "open_gross": round(sum(r["gross"] for r in open_rows), 2),
            "all_expected": round(sum(r["expected"] for r in rows), 2),
            "all_gross": round(sum(r["gross"] for r in rows), 2),
            "excluded_expected": round(sum(r["expected"] for r in excl_rows), 2)}


for book in ("alderbridge", "kestrelmoor"):
    R = recompute(book)
    print("=" * 96)
    print(f"{book.upper()}  as-of {R['as_of']}   "
          f"({R['hist']['weeks']} weekly extracts, {R['hist']['cases_tracked']} cases tracked)")
    print("\n  empirical stage rates (independent):")
    for s, d in R["hist"]["per_stage"].items():
        print(f"    {s:12s} observed={d['observed']:4d} completed={d['completed']:4d} "
              f"rate={d['rate']}  sufficient={d['sufficient']}")
    print(f"\n  {'stage':12s} {'cases':>6s} {'gross':>16s} {'prob':>8s} "
          f"{'basis':>22s} {'expected':>15s}  lands")
    for r in R["rows"]:
        print(f"    {r['stage']:12s} {r['cases']:>4d} {r['gross']:>18,.2f} "
              f"{r['probability']:>8.4f} {r['basis']:>22s} {r['expected']:>15,.2f}  {r['lands']}")
    print(f"\n  OPEN pipeline (include_stages)   gross {R['open_gross']:>16,.2f}   "
          f"expected {R['open_expected']:>14,.2f}")
    print(f"  WHOLE extract                    gross {R['all_gross']:>16,.2f}   "
          f"expected {R['all_expected']:>14,.2f}")
    print(f"  excluded-stage component (config says EXCLUDE) "
          f"{R['excluded_expected']:>14,.2f}")
