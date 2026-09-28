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
