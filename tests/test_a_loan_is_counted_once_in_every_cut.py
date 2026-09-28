"""A loan is counted once however each reporting cut happens to key it.

Reported defect: the 2025-12 vintage showed 114 loans entering (£17.4MM) while
its own static pool held 57 (£8.7MM) — exactly half. Every 2025 vintage was
doubled; 2026 vintages were right. The 57 loans were keyed one way in the 2025
cuts and another way from 2026-01, so formation counted each of them twice and
the static pool reported all 57 as exits in 2026-01 ("57 (0 cum.)").

Two ways a cut re-keys a loan, both covered here:
  * the id column is read as float in one cut (a single blank is enough), so
    ``1000`` becomes ``"1000.0"``;
  * a second id column is populated only from some month on, and the key was
    chosen per cut, so it switched column mid-series.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from mi_agent_api import cohorts as C

BALANCE = "current_outstanding_balance"
N = 57


def _cut(ids, *, extra=None):
    df = pd.DataFrame({
        "loan_identifier": ids,
        BALANCE: [150_000.0] * len(ids),
        "origination_date": pd.to_datetime(["2025-12-10"] * len(ids)),
    })
    for col, values in (extra or {}).items():
        df[col] = values
    return df


def _float_read_series():
    ints = list(range(1000, 1000 + N))
    # One blank id in the later cuts makes the column float: 1000 -> 1000.0.
    floats = [float(i) for i in ints] + [np.nan]
    later = _cut(floats)
    later.loc[later.index[-1], "origination_date"] = pd.Timestamp("2026-01-05")
    return [
        {"reporting_date": "2025-12-31", "df": _cut(ints)},
        {"reporting_date": "2026-01-31", "df": later},
        {"reporting_date": "2026-02-28", "df": later.copy()},
    ]


def _switching_key_series():
    ids = [f"L{i:03d}" for i in range(N)]
    return [
        {"reporting_date": "2025-12-31", "df": _cut(ids)},
        # From 2026-01 the cut also carries a differently-spelt `loan_id`.
        {"reporting_date": "2026-01-31",
         "df": _cut(ids, extra={"loan_id": [f"ERE-{i}" for i in ids]})},
    ]


def _formation_row(frames, vintage="2025-12"):
    out = C.cohort_formation(frames, grain="M")
    return next(r for r in out["vintages"] if r["vintage"] == vintage)


def test_a_float_read_cut_does_not_double_the_vintage():
    row = _formation_row(_float_read_series())
    assert row["originalLoanCount"] == N
    assert row["originalBalance"] == N * 150_000.0


def test_a_float_read_cut_is_not_a_mass_exit():
    pool = C.cohort_static_pool(_float_read_series(), vintage="2025-12", grain="M")
    assert pool["originalLoanCount"] == N
    jan = next(p for p in pool["periods"] if p["period"] == "2026-01")
    assert jan["survivingLoanCount"] == N
    assert jan["exitsInPeriod"] == 0
    assert jan["cumulativeExits"] == 0


def test_the_key_column_does_not_switch_mid_series():
    frames = _switching_key_series()
    assert _formation_row(frames)["originalLoanCount"] == N
    pool = C.cohort_static_pool(frames, vintage="2025-12", grain="M")
    assert all(p["exitsInPeriod"] == 0 for p in pool["periods"])


def test_a_loan_repeated_within_a_cut_enters_once():
    ids = [f"L{i:03d}" for i in range(N)]
    frames = [{"reporting_date": "2025-12-31", "df": _cut(ids + ids[:5])}]
    assert _formation_row(frames)["originalLoanCount"] == N


def test_cumulative_exits_agree_with_period_exits():
    ids = [f"L{i:03d}" for i in range(N)]
    frames = [
        {"reporting_date": "2025-12-31", "df": _cut(ids)},
        {"reporting_date": "2026-01-31", "df": _cut(ids[3:])},
        {"reporting_date": "2026-02-28", "df": _cut(ids[5:])},
    ]
    pool = C.cohort_static_pool(frames, vintage="2025-12", grain="M")
    by = {p["period"]: p for p in pool["periods"]}
    assert (by["2026-01"]["exitsInPeriod"], by["2026-01"]["cumulativeExits"]) == (3, 3)
    assert (by["2026-02"]["exitsInPeriod"], by["2026-02"]["cumulativeExits"]) == (2, 5)


def _odd_cut_series():
    """The live pattern: 2025-12 alone keys its loans differently.

    Seen on the 2025-10 static pool: 33 exits at 2025-12, then 33 exits again
    at 2026-01 with 0 cumulative — the 2025-12 cut's loan column held other
    values, while the same loans' ids sat in another identifier column.
    """
    ref = [f"{776000 + i}01" for i in range(33)]
    odd = _cut([f"{776000 + i}" for i in range(33)],
               extra={"original_underlying_exposure_identifier": ref})
    for fr in (odd,):
        fr["origination_date"] = pd.Timestamp("2025-10-10")
    def ok():
        df = _cut(ref)
        df["origination_date"] = pd.Timestamp("2025-10-10")
        return df
    return [
        {"reporting_date": "2025-10-31", "df": ok()},
        {"reporting_date": "2025-11-30", "df": ok()},
        {"reporting_date": "2025-12-31", "df": odd},
        {"reporting_date": "2026-01-31", "df": ok()},
    ]


def test_a_cut_keyed_in_another_column_is_followed():
    frames = _odd_cut_series()
    assert _formation_row(frames, "2025-10")["originalLoanCount"] == 33
    pool = C.cohort_static_pool(frames, vintage="2025-10", grain="M")
    assert all(p["survivingLoanCount"] == 33 for p in pool["periods"])
    assert all(p["exitsInPeriod"] == 0 and p["cumulativeExits"] == 0
               for p in pool["periods"])
    dec = next(l for l in pool["idLinkage"] if l["reportingDate"] == "2025-12-31")
    assert dec["idColumn"] == "original_underlying_exposure_identifier"
    assert dec["linkedFromPriorPct"] == 100.0


def test_a_cut_that_cannot_be_joined_says_so():
    frames = _odd_cut_series()
    frames[2]["df"] = frames[2]["df"].drop(
        columns="original_underlying_exposure_identifier")
    out = C.cohort_formation(frames, grain="M")
    dec = next(l for l in out["idLinkage"] if l["reportingDate"] == "2025-12-31")
    assert dec["linkedFromPriorPct"] == 0.0


def _live_series():
    """The live book: 33 loans originated in 2025-10, 40 in 2025-11, 57 in
    2025-12, and a 2025-12 cut whose ids match NO other cut in any column.

    Before the fix formation read 66 / 80 / 114 — every 2025 vintage doubled —
    and the 2025-11 pool showed "40 (40 cum.)" exits at 2025-12, then "40 (0
    cum.)" at 2026-01.
    """
    plan = [("2025-10-15", 33, 0), ("2025-11-15", 40, 100), ("2025-12-15", 57, 200)]

    def cut(through, rekey=False):
        ids, dates = [], []
        for d, n, base in plan:
            if d[:7] <= through:
                ids += [f"{'X' if rekey else 'L'}{base + i}" for i in range(n)]
                dates += [d] * n
        df = pd.DataFrame({"loan_identifier": ids,
                           BALANCE: [100_000.0] * len(ids),
                           "origination_date": pd.to_datetime(dates)})
        return df

    return [
        {"reporting_date": "2025-10-31", "df": cut("2025-10")},
        {"reporting_date": "2025-11-30", "df": cut("2025-11")},
        {"reporting_date": "2025-12-31", "df": cut("2025-12", rekey=True)},
        {"reporting_date": "2026-01-31", "df": cut("2025-12")},
        {"reporting_date": "2026-02-28", "df": cut("2025-12")},
    ]


def test_a_rekeyed_cut_does_not_double_any_vintage():
    out = C.cohort_formation(_live_series(), grain="M")
    counts = {r["vintage"]: r["originalLoanCount"] for r in out["vintages"]}
    assert counts == {"2025-10": 33, "2025-11": 40, "2025-12": 57}
    bal = {r["vintage"]: r["originalBalance"] for r in out["vintages"]}
    assert bal["2025-11"] == 40 * 100_000.0
    assert out["totalLoanCount"] == 130


def test_a_rekeyed_cut_is_not_read_as_the_pool_leaving():
    pool = C.cohort_static_pool(_live_series(), vintage="2025-11", grain="M")
    assert pool["originalLoanCount"] == 40
    by = {p["period"]: p for p in pool["periods"]}
    assert all(p["survivingLoanCount"] == 40 for p in pool["periods"])
    assert all(p["exitsInPeriod"] == 0 and p["cumulativeExits"] == 0
               for p in pool["periods"])
    assert by["2025-12"]["idsRekeyed"] and by["2026-01"]["idsRekeyed"]
    assert not by["2026-02"]["idsRekeyed"]


def test_a_year_vintage_is_the_whole_year():
    out = C.cohort_formation(_live_series(), grain="Y")
    assert [(r["vintage"], r["originalLoanCount"]) for r in out["vintages"]] == [("2025", 130)]
