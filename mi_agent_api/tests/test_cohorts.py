#!/usr/bin/env python3
"""mi_agent_api/tests/test_cohorts.py

Funded origination-vintage (static-pool) cohort analysis (Task 2). Verifies that
per-vintage balance / count / share and balance-weighted LTV, rate and MOB are
computed from the funded tape, that only computed metrics are surfaced (no
fabricated redemption curves), and that a tape with no vintage degrades to an
honest ``available=false`` state.
"""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import pandas as pd

from mi_agent_api import cohorts as cohorts_mod


def _funded_df() -> pd.DataFrame:
    return pd.DataFrame({
        "origination_date": ["2021-03-01", "2021-08-01", "2022-05-01",
                             "2022-06-01", "2023-01-01"],
        "current_outstanding_balance": [100_000, 300_000, 200_000, 200_000, 50_000],
        "current_loan_to_value": [0.60, 0.80, 0.50, 0.70, 0.40],
        "current_interest_rate": [0.04, 0.05, 0.045, 0.055, 0.03],
        "months_on_book": [40, 35, 20, 19, 12],
    })


def test_cohort_analysis_groups_by_vintage_year():
    out = cohorts_mod.cohort_analysis(_funded_df(), client_id="client_001",
                                      portfolio_id="client_001/mi_2023_01")
    assert out["available"] is True
    vintages = [c["vintage"] for c in out["cohorts"]]
    assert vintages == ["2021", "2022", "2023"]  # ascending
    by = {c["vintage"]: c for c in out["cohorts"]}
    assert by["2021"]["loanCount"] == 2
    assert by["2021"]["balance"] == 400_000.0
    assert by["2022"]["balance"] == 400_000.0
    # Book share reconciles to 100% across cohorts.
    assert round(sum(c["sharePct"] for c in out["cohorts"]), 1) == 100.0
    assert out["totalLoanCount"] == 5
    assert out["totalBalance"] == 850_000.0


def test_balance_weighted_metrics():
    out = cohorts_mod.cohort_analysis(_funded_df())
    by = {c["vintage"]: c for c in out["cohorts"]}
    # 2021 WA LTV = (0.60*100k + 0.80*300k) / 400k = 0.75.
    assert by["2021"]["waLtv"] == 0.75
    # 2022 WA rate = (0.045*200k + 0.055*200k)/400k = 0.05.
    assert by["2022"]["waRate"] == 0.05
    assert "waMonthsOnBook" in by["2021"]
    assert set(out["metricsAvailable"]) >= {
        "balance", "loanCount", "waLtv", "waRate", "waMonthsOnBook"}


def _wave1_df() -> pd.DataFrame:
    return pd.DataFrame({
        "origination_date": ["2021-03-01", "2022-05-01", "2022-06-01", "2023-01-01"],
        "current_outstanding_balance": [100_000, 300_000, 200_000, 50_000],
        "current_loan_to_value": [0.60, 0.80, 0.50, 0.40],
        "age_bucket": ["65–69", "70–74", "65–69", "80–84"],
        "original_loan_to_value": [0.55, 0.75, 0.48, 0.38],
        "origination_channel": ["Broker A", "Broker B", "Broker A", "Direct"],
    })


def test_available_dimensions_reflect_the_tape():
    out = cohorts_mod.cohort_analysis(_wave1_df())
    assert out["availableDimensions"] == ["vintage", "age", "ltv", "channel"]
    # The slim tape has vintage + current LTV (ltv falls back to current_loan_to_value),
    # but no age/channel fields.
    slim = cohorts_mod.cohort_analysis(_funded_df())
    assert slim["availableDimensions"] == ["vintage", "ltv"]


def test_cohort_by_borrower_age():
    out = cohorts_mod.cohort_analysis(_wave1_df(), dimension="age")
    assert out["available"] is True
    assert out["dimension"] == "age" and out["dimensionLabel"] == "Borrower age"
    by = {c["cohort"]: c for c in out["cohorts"]}
    assert by["65–69"]["loanCount"] == 2  # two loans in that band
    assert [c["cohort"] for c in out["cohorts"]] == ["65–69", "70–74", "80–84"]  # by age


def test_cohort_by_ltv_band_and_channel():
    ltv = cohorts_mod.cohort_analysis(_wave1_df(), dimension="ltv")
    assert ltv["dimensionLabel"] == "LTV band"
    # Bands ordered by their leading number.
    labels = [c["cohort"] for c in ltv["cohorts"]]
    assert labels == sorted(labels, key=lambda s: int(__import__("re").search(r"\d+", s).group()))
    ch = cohorts_mod.cohort_analysis(_wave1_df(), dimension="channel")
    assert ch["dimensionLabel"] == "Origination channel"
    # Channels ranked by balance (Broker A = 300k leads).
    assert ch["cohorts"][0]["cohort"] == "Broker A"


def test_dimension_unavailable_is_honest():
    # No channel/age/ltv fields -> those dimensions are unavailable, vintage still works.
    out = cohorts_mod.cohort_analysis(_funded_df(), dimension="channel")
    assert out["available"] is False
    assert "origination channel" in out["reason"].lower()
    assert "channel" not in out["availableDimensions"]


def test_no_fabricated_curves():
    out = cohorts_mod.cohort_analysis(_funded_df())
    for c in out["cohorts"]:
        # Only computed aggregates — no redemption/completion/performance curves.
        assert not any(k for k in c
                       if "curve" in k.lower() or "redempt" in k.lower())


def test_missing_vintage_is_unavailable():
    df = pd.DataFrame({"current_outstanding_balance": [100_000, 200_000]})
    out = cohorts_mod.cohort_analysis(df)
    assert out["available"] is False
    assert "vintage" in out["reason"] or "origination" in out["reason"]
    assert out["cohorts"] == []


def test_unknown_vintage_bucketed_not_dropped():
    df = _funded_df()
    df.loc[len(df)] = ["not-a-date", 25_000, 0.5, 0.04, 10]
    out = cohorts_mod.cohort_analysis(df)
    labels = [c["vintage"] for c in out["cohorts"]]
    assert "Unknown" in labels
    assert labels[-1] == "Unknown"  # sinks to the end
    assert out["totalLoanCount"] == 6


def test_mixed_iso_and_uk_dates_all_parse():
    """Regression: an origination_date column that MIXES ISO (YYYY-MM-DD) and UK
    (DD/MM/YYYY) rows must parse per element — previously pandas inferred a single
    format from the first value and NaT'd the rest into a spurious 'Unknown'
    bucket (59 of 73 loans on the live November book)."""
    df = pd.DataFrame({
        # 3 ISO + 3 UK, all genuine 2025 dates; UK days > 12 to prove dayfirst.
        "origination_date": ["2025-01-15", "28/11/2025", "2025-06-30",
                             "03/10/2025", "2025-09-01", "17/07/2025"],
        "current_outstanding_balance": [100_000, 200_000, 150_000,
                                       250_000, 120_000, 180_000],
        "current_loan_to_value": [0.55, 0.60, 0.50, 0.65, 0.58, 0.62],
    })
    out = cohorts_mod.cohort_analysis(df)
    labels = [c["vintage"] for c in out["cohorts"]]
    # Every row is a valid 2025 date — nothing may leak into 'Unknown'.
    assert labels == ["2025"]
    assert "Unknown" not in labels
    assert out["cohorts"][0]["loanCount"] == 6


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-q"]))


def test_points_scaled_rate_is_normalised_to_fraction():
    # The ERE tape stores LTV as a fraction (0.40) but the interest rate in
    # POINTS (9.55). A single ×100 UI formatter turned 9.55% into 955%. The
    # endpoint must emit both as fractions so the UI renders them correctly.
    df = pd.DataFrame({
        "vintage_year": [2025, 2025, 2024, 2024],
        "current_outstanding_balance": [100_000, 100_000, 100_000, 100_000],
        "current_loan_to_value": [0.40, 0.38, 0.42, 0.30],   # fraction
        "current_interest_rate": [9.55, 9.52, 9.56, 9.29],    # points
    })
    out = cohorts_mod.cohort_analysis(df, client_id="c")
    by = {c["vintage"]: c for c in out["cohorts"]}
    # LTV already a fraction — unchanged; rate normalised points→fraction.
    assert 0.30 <= by["2025"]["waLtv"] <= 0.45
    assert 0.09 <= by["2025"]["waRate"] <= 0.10, by["2025"]["waRate"]
    assert 0.09 <= by["2024"]["waRate"] <= 0.10


def test_finer_vintage_grain_quarter_and_month():
    # A young book all originated in 2025 collapses to one 'Y' bucket; finer
    # grain reveals the seasoning spread.
    df = pd.DataFrame({
        "origination_date": ["2025-01-15", "2025-02-10", "2025-05-20", "2025-08-01"],
        "current_outstanding_balance": [100_000, 100_000, 100_000, 100_000],
        "current_loan_to_value": [0.40, 0.42, 0.38, 0.30],
    })
    y = cohorts_mod.cohort_analysis(df, client_id="c", grain="Y")
    assert {c["vintage"] for c in y["cohorts"]} == {"2025"}
    q = cohorts_mod.cohort_analysis(df, client_id="c", grain="Q")
    labels = [c["vintage"] for c in q["cohorts"]]
    assert labels == sorted(labels)  # chronological
    assert set(labels) == {"2025-Q1", "2025-Q2", "2025-Q3"}
    m = cohorts_mod.cohort_analysis(df, client_id="c", grain="M")
    assert "2025-01" in {c["vintage"] for c in m["cohorts"]}
