"""Vintages side by side by months on book, and a vintage profile that does
not go blank because one cut lacks a column.

The live 2025-12 vintage showed "—" for both LTVs: it is measured in the
2025-12 cut, which carries no LTV. The profile now reads the next cut holding
the same vintage. The matrix is the market-standard static-pool grid: rows are
vintages, columns months on book.
"""
from __future__ import annotations

import pandas as pd
import pytest

from mi_agent_api import cohorts as C

BAL = "current_outstanding_balance"


def _cut(date, rows, *, ltv=True, advance=False):
    """rows: [(loan_id, origination_date, balance)]"""
    df = pd.DataFrame(rows, columns=["loan_identifier", "origination_date", BAL])
    df["origination_date"] = pd.to_datetime(df["origination_date"])
    if ltv:
        df["current_loan_to_value"] = 0.40
        df["original_loan_to_value"] = 0.38
    if advance:
        df["original_principal_balance"] = 100_000.0
    df["youngest_borrower_age"] = 70
    return {"reporting_date": date, "df": df}


def _series():
    oct_loans = [(f"A{i}", "2025-10-10", 100_000.0) for i in range(10)]
    nov_loans = [(f"B{i}", "2025-11-10", 100_000.0) for i in range(4)]
    return [
        _cut("2025-10-31", oct_loans, advance=True),
        _cut("2025-11-30", [(i, d, b * 1.01) for i, d, b in oct_loans] + nov_loans,
             ltv=False, advance=True),
        # One October loan redeemed.
        _cut("2025-12-31", [(i, d, b * 1.02) for i, d, b in oct_loans[1:]]
             + [(i, d, b * 1.01) for i, d, b in nov_loans], advance=True),
    ]


def test_a_cut_without_ltv_does_not_blank_the_vintage_profile():
    out = C.cohort_formation(_series(), grain="M")
    nov = next(v for v in out["vintages"] if v["vintage"] == "2025-11")
    assert nov["measuredAt"] == "2025-11-30"
    assert nov["waEntryLtv"] == pytest.approx(0.40)  # fractions, as the table formats them
    assert nov["waOriginalLtv"] == pytest.approx(0.38)
    assert nov["profileFrom"]["waEntryLtv"] == "2025-12-31"
    assert nov["waEntryAge"] == pytest.approx(70.0)


def test_the_matrix_lines_vintages_up_by_months_on_book():
    m = C.cohort_matrix(_series(), grain="M")
    assert m["available"]
    rows = {r["vintage"]: r for r in m["vintages"]}
    oct_ = rows["2025-10"]
    assert oct_["basis"] == "original_advance" and oct_["base"] == 1_000_000.0
    assert oct_["cells"]["0"]["balanceFactor"] == pytest.approx(1.0)
    assert oct_["cells"]["2"]["survivingLoanCount"] == 9
    assert oct_["cells"]["2"]["cumulativeExitRate"] == pytest.approx(0.1)
    assert oct_["cells"]["2"]["balanceFactor"] == pytest.approx(9 * 102_000 / 1_000_000)
    assert set(rows["2025-11"]["cells"]) == {"0", "1"}
    assert m["monthsOnBook"] == [0, 1, 2]


def test_without_an_original_advance_the_base_is_the_fixed_pool_balance():
    series = _series()
    for fr in series:
        fr["df"] = fr["df"].drop(columns="original_principal_balance")
    oct_ = next(r for r in C.cohort_matrix(series, grain="M")["vintages"]
                if r["vintage"] == "2025-10")
    assert oct_["basis"] == "balance_when_pool_fixed"
    assert oct_["base"] == 1_000_000.0
