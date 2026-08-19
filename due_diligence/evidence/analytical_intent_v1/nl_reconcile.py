"""Independently reconcile every numeric finding the NL runs produced.

Truth comes from the fixture CSVs read with pandas. The planner and the
narrative are never the oracle (brief §7).
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import pandas as pd

BOOKS = {
    "alderbridge": Path("demo_platform/workspace/store/processed/platform/alderbridge"),
    "kestrelmoor": Path(__file__).resolve().parent
                   / "kestrelmoor_store/processed/platform/kestrelmoor",
}
DATES = ["2026-04-30", "2026-05-31", "2026-06-30"]
BAL, LTV, RATE, AGE = ("current_outstanding_balance", "current_loan_to_value",
                       "current_interest_rate", "youngest_borrower_age")


def num(s):
    return pd.to_numeric(s, errors="coerce")


def load(book):
    frames = {}
    for d in DATES:
        f = pd.read_csv(BOOKS[book] / d / "platform_canonical_typed.csv",
                        low_memory=False)
        o = pd.to_datetime(f["origination_date"], errors="coerce")
        t = pd.Timestamp(d)
        m = (t.year - o.dt.year) * 12 + (t.month - o.dt.month)
        f["_mob"] = m
        f["_segment"] = ["Front Book" if x == x and x <= 12 else "Back Book"
                         for x in m]
        f["_vintage"] = o.dt.year
        frames[d] = f
    return frames


def wavg(v, w):
    v, w = num(v), num(w)
    k = v.notna() & w.notna() & (w > 0)
    return float((v[k] * w[k]).sum() / w[k].sum()) if k.any() else None


def population(frame, key):
    # The governed LENDING WINDOWS, recomputed here from origination_date and the
    # snapshot date so the truth is independent of the production derivation.
    if key == "new":
        return frame[frame["_mob"] <= 1]
    if key == "recent":
        return frame[frame["_mob"] <= 3]
    if key == "front_book":
        return frame[frame["_segment"] == "Front Book"]
    if key == "back_book":
        return frame[frame["_segment"] == "Back Book"]
    if key in ("direct", "acquired"):
        return frame[frame["source_portfolio_type"] == key]
    if key == "total":
        return frame
    if key and key.startswith("collateral_geography="):
        return frame[frame["collateral_geography"].astype(str)
                     == key.split("=", 1)[1]]
    return None


def truth(frame, metric):
    if metric == BAL:
        return float(num(frame[BAL]).sum())
    if metric == LTV:
        return wavg(frame[LTV], frame[BAL])
    if metric == RATE:
        return wavg(frame[RATE], frame[BAL])
    if metric == AGE:
        return float(num(frame[AGE]).mean())
    if metric == "loan_count":
        return float(len(frame))
    return None


def reconcile(path, book):
    frames = load(book)
    close, open_ = frames[DATES[-1]], frames[DATES[0]]
    rows = []

    def add(label, delivered, expected, tol):
        if delivered is None or expected is None:
            return
        ok = abs(float(delivered) - float(expected)) <= tol
        rows.append((label, float(delivered), float(expected), ok))

    for entry in json.load(open(path)):
        for i, run in enumerate(entry.get("runs") or []):
            if run.get("route") != "analytical_composition":
                continue
            tag = f"{entry['intent']}.{entry['variation']}r{i}"
            for f in run.get("findings") or []:
                key = (f.get("population") or {}).get("key")
                metric, kind = f.get("metric"), f.get("kind")
                if kind in ("measure", "movement", "comparison") and metric:
                    pop = population(close, key)
                    if pop is None:
                        continue
                    tol = 1e-6 if metric in (LTV, RATE, AGE) else 0.05
                    add(f"{tag} {key}.{metric}", f.get("value"),
                        truth(pop, metric), tol)
                    if kind == "movement":
                        add(f"{tag} {key}.{metric}.prior", f.get("priorValue"),
                            truth(population(open_, key), metric), tol)
                    if kind == "comparison":
                        ck = (f.get("comparand") or {}).get("key")
                        cpop = population(close, ck)
                        if cpop is not None:
                            add(f"{tag} {ck}.{metric}", f.get("comparandValue"),
                                truth(cpop, metric), tol)
                    if (f.get("population") or {}).get("rows") is not None:
                        add(f"{tag} {key}.rows",
                            (f.get("population") or {})["rows"],
                            float(len(pop)), 0)
                if kind == "cohort":
                    sub = close[close["_vintage"] == int(f["label"])]
                    add(f"{tag} vintage {f['label']}", f.get("value"),
                        float(num(sub[BAL]).sum()), 0.05)
    return rows


if __name__ == "__main__":
    import os
    os.chdir("/home/user/trakt")
    total = failed = 0
    for path in sys.argv[1:]:
        book = "kestrelmoor" if "kestrelmoor" in path else "alderbridge"
        rows = reconcile(path, book)
        bad = [r for r in rows if not r[3]]
        total += len(rows); failed += len(bad)
        print(f"{Path(path).name}: {len(rows) - len(bad)}/{len(rows)} reconciled")
        for label, d, t, _ in bad[:15]:
            print(f"   MISMATCH {label}: delivered={d!r} truth={t!r}")
    print(f"\nTOTAL RECONCILED {total - failed}/{total}  MISMATCHES {failed}")
