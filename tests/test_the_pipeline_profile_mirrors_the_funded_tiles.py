"""The pipeline snapshot carries the funded tiles' credit profile.

The Pipeline page's tiles were cases / amount / weighted / average plus a
"Top stage by amount" tile (always KFI — no information). The funded page
shows LTV, rate, borrower age, single-borrower share and property value. The
pipeline snapshot now carries the same measures, on the same definitions
(amount-weighted, as the funded tiles are balance-weighted), so the two
lenses read alike.
"""
from __future__ import annotations

import pandas as pd
import pytest

from mi_agent_api import pipeline_contract as pc
from mi_agent_api.pipeline_prep import prepare_pipeline_mi_dataset


def _snapshot(**extra):
    raw = pd.DataFrame({
        "Account Number": ["A1", "A2", "A3", "A4"],
        "Status": ["Offer", "Application", "KFI", "Completed"],
        "Loan Amount": [100_000, 50_000, 30_000, 80_000],
        "Estimated Value": [400_000, 100_000, 300_000, 200_000],
        **extra,
    })
    df, report = prepare_pipeline_mi_dataset(raw, as_of_date="2026-09-24")
    return pc.compute_pipeline_snapshot(df, report, {}, client_id="ERE", run_id="x")


def test_profile_is_amount_weighted_like_the_funded_tiles():
    prof = _snapshot(**{
        "Interest Rate": [6.5, 7.0, 7.1, 6.9],
        "DOB App 1": ["01/01/1950", "01/01/1955", "01/01/1960", "01/01/1945"],
        "DOB App 2": ["01/01/1952", None, None, None],
    })["profile"]
    total = 260_000
    # LTV = loan amount / estimated value, weighted by loan amount.
    ltv = (100_000 * 25 + 50_000 * 50 + 30_000 * 10 + 80_000 * 40) / total
    assert prof["waLtvPct"] == pytest.approx(ltv, abs=0.01)
    rate = (100_000 * 6.5 + 50_000 * 7.0 + 30_000 * 7.1 + 80_000 * 6.9) / total
    assert prof["waInterestRatePct"] == pytest.approx(rate, abs=0.01)
    assert prof["waPropertyValue"] == pytest.approx(
        (100_000 * 400_000 + 50_000 * 100_000 + 30_000 * 300_000 + 80_000 * 200_000) / total,
        abs=0.01)
    assert prof["waYoungestAge"] is not None
    assert (prof["singleBorrowerCount"], prof["borrowerTypeKnownCount"]) == (3, 4)
    assert prof["singleBorrowerPct"] == 75.0


def test_a_fractional_rate_reads_in_points():
    prof = _snapshot(**{"Interest Rate": [0.065, 0.07, 0.071, 0.069]})["profile"]
    assert 6.5 <= prof["waInterestRatePct"] <= 7.1


def test_a_measure_the_extract_cannot_supply_is_none():
    prof = _snapshot()["profile"]
    assert prof["waInterestRatePct"] is None
    assert prof["waYoungestAge"] is None
    assert prof["singleBorrowerPct"] is None
