#!/usr/bin/env python3
"""A production-shaped portfolio built for readiness assessment.

The Sprint 2 planted portfolio (``tests/planted_portfolio.py``) is twelve loans
sized for valuation-selection cases, and it breaches every concentration
threshold at once — useful for proving a metric fires, useless for proving that
different rulebooks reach *different* verdicts on the same number.

This book is 86 loans, arranged so the interesting outcomes are separable:

    largest region share = 28.0%
        EXAMPLE_WAREHOUSE      <= 35%   PASS
        EXAMPLE_SECURITISATION <= 27%   BREACH
        TRAKT_SCREENING        <= 25%   FLAG

One measured fact, three rulebooks, three verdicts, and no recalculation. That
is the architectural claim Sprint 2.5 exists to make, and it cannot be
demonstrated on a book that fails everything.

The planted cases, and what each is for:

    clean                    the majority: nothing to find
    high-LTV cohort          above 80%, sized to sit between rulebooks
    extreme LTV              a single loan over 100% — outlier, not a trend
    geographic concentration 28%, deliberately between the thresholds
    large-loan concentration top-10 at 14%: warehouse pass, screen flag
    stale valuations         older than the 24-month policy limit
    recent valuations        the control group
    missing valuations       absent from every LTV metric, and must be seen
    multiple observations    full + indexed + purchase price on one collateral
    arrears                  30+ DPD
    90+ DPD                  usually an outright exclusion
    deteriorating cohort     a region whose arrears are concentrated
    stable cohort            the comparison that makes the above meaningful
    clean data / weak economics   high LTV, high arrears, perfect data
    strong economics / weak data  low LTV, no arrears, missing critical fields

Known truth is stated below as literals worked out by hand. Nothing in this
module calls the code under test to decide what the answer should be.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

REPORTING_DATE = "2026-07-31"
BOOK = "direct_001"
SNAPSHOT_ID = "snap_readiness_0001"

# --------------------------------------------------------------------------- #
# The truth table — hand arithmetic, written down
# --------------------------------------------------------------------------- #
TOTAL_BALANCE = 20_000_000.0
LOAN_COUNT = 86

#: region -> (loan count, balance each, total balance, share %)
REGION_PLAN = {
    "London":     (20, 280_000.0, 5_600_000.0, 28.0),
    "South East": (24, 225_000.0, 5_400_000.0, 27.0),
    "North West": (16, 275_000.0, 4_400_000.0, 22.0),
    "Wales":      (14, 200_000.0, 2_800_000.0, 14.0),
    "Scotland":   (12, 150_000.0, 1_800_000.0,  9.0),
}

#: The number that carries the whole demonstration.
LARGEST_REGION = "London"
LARGEST_REGION_SHARE = 28.0

#: The ten largest loans are the twenty London loans at 280,000 each.
TOP_10_BALANCE = 2_800_000.0
TOP_10_SHARE = 14.0

#: Expected verdicts on LARGEST_REGION_SHARE, stated independently of the packs.
GEOGRAPHIC_VERDICTS = {
    "EXAMPLE_WAREHOUSE@v1": ("pass", 35.0),
    "EXAMPLE_PROPOSED_SECURITISATION@v1": ("breach", 27.0),
    "TRAKT_SCREENING@v1": ("flag", 25.0),
}

#: Same shape for the granularity metric.
TOP_10_VERDICTS = {
    "EXAMPLE_WAREHOUSE@v1": ("pass", 15.0),
    "EXAMPLE_PROPOSED_SECURITISATION@v1": ("breach", 12.0),
    "TRAKT_SCREENING@v1": ("flag", 10.0),
}

#: Planted cohorts, by index range within each region's block.
#: Balance figures are exact because every loan in a region has one size.
HIGH_LTV_LOANS = 12          # LTV 85 - a cohort, not a tail
EXTREME_LTV_LOANS = 1        # LTV 108 - one loan, above 100
ARREARS_30_LOANS = 6         # 45 days
ARREARS_90_LOANS = 2         # 120 days
STALE_VALUATION_LOANS = 8    # valued 2021 - older than the 24-month limit
MISSING_VALUATION_LOANS = 3  # no valuation at all
MULTI_OBSERVATION_LOANS = 4  # full + indexed + purchase price
WEAK_DATA_LOANS = 5          # strong economics, critical fields blank


def _region_blocks() -> List[Dict[str, Any]]:
    """Region assignment for each loan index, in a fixed order."""
    out: List[Dict[str, Any]] = []
    for region, (count, size, _total, _share) in REGION_PLAN.items():
        for _ in range(count):
            out.append({"region": region, "balance": size})
    return out


def readiness_frame() -> pd.DataFrame:
    """The 86-loan readiness portfolio."""
    blocks = _region_blocks()
    assert len(blocks) == LOAN_COUNT, f"{len(blocks)} != {LOAN_COUNT}"

    rows: List[Dict[str, Any]] = []
    for i, block in enumerate(blocks):
        balance = block["balance"]
        region = block["region"]

        # ---- planted characteristics, assigned by position -------------- #
        high_ltv = i < HIGH_LTV_LOANS
        extreme_ltv = i == HIGH_LTV_LOANS                      # exactly one
        arrears_30 = HIGH_LTV_LOANS + 1 <= i < HIGH_LTV_LOANS + 1 + ARREARS_30_LOANS
        arrears_90 = 30 <= i < 30 + ARREARS_90_LOANS
        stale_valuation = 40 <= i < 40 + STALE_VALUATION_LOANS
        missing_valuation = 50 <= i < 50 + MISSING_VALUATION_LOANS
        multi_observation = 60 <= i < 60 + MULTI_OBSERVATION_LOANS
        weak_data = 70 <= i < 70 + WEAK_DATA_LOANS

        if extreme_ltv:
            ltv = 108.0
        elif high_ltv:
            ltv = 85.0
        elif weak_data:
            ltv = 45.0                    # strong economics, weak data
        else:
            ltv = 55.0 + (i % 7)          # 55 - 61, comfortably clean

        days = 120 if arrears_90 else (45 if arrears_30 else 0)
        arrears_balance = round(balance * 0.03, 2) if days else 0.0
        status = "arrears" if days else "performing"

        valuation = round(balance / (ltv / 100.0), 2)
        if missing_valuation:
            valuation_amount = None
            valuation_date = None
            valuation_method = None
        elif stale_valuation:
            valuation_amount = valuation
            valuation_date = "2021-06-30"           # 61 months at 2026-07-31
            valuation_method = "full valuation"
        else:
            valuation_amount = valuation
            valuation_date = "2026-04-30"           # 3 months
            valuation_method = "full valuation"

        row: Dict[str, Any] = {
            "loan_identifier": f"RD{i:05d}",
            "borrower_identifier": f"BW{i:05d}",     # one borrower per loan
            "source_portfolio_id": BOOK,
            "source_portfolio_type": "direct",
            "portfolio_cohort": "2026H1",
            "account_status": status,
            "current_principal_balance": balance,
            "current_outstanding_balance": balance,
            "original_principal_balance": round(balance * 1.15, 2),
            "current_loan_to_value": None if missing_valuation else ltv,
            "indexed_loan_to_value": None,
            "current_interest_rate": 5.0 + (i % 5) * 0.25,
            "arrears_balance": arrears_balance,
            "number_of_days_in_arrears": days,
            "ifrs9_stage": "2" if days else "1",
            "origination_date": "2021-05-01" if i % 2 else "2023-09-01",
            "maturity_date": "2046-05-01",
            "current_valuation_amount": valuation_amount,
            "current_valuation_date": valuation_date,
            "current_valuation_method": valuation_method,
            "current_valuation_basis": "market_value" if valuation_amount else None,
            "original_valuation_amount": None,
            "date_of_valuation_at_securitisation": None,
            "indexed_value": None,
            "index_determination_date": None,
            "collateral_type": "residential_property",
            "property_type": "detached_house",
            "occupancy_type": "owner_occupied",
            "geographic_region_obligor": region,
            "geographic_region_collateral": region,
            "borrower_type": "individual",
            "employment_status": "employed",
            "exposure_currency_denomination": "GBP",
            "collateral_currency": "GBP",
            "reporting_date": REPORTING_DATE,
        }

        if multi_observation:
            # Three observations on one collateral, so the selection policy has
            # something to choose between and reject.
            row["original_valuation_amount"] = round(valuation * 0.85, 2)
            row["date_of_valuation_at_securitisation"] = "2019-03-01"
            row["indexed_value"] = round(valuation * 1.05, 2)
            row["index_determination_date"] = "2026-06-30"

        if weak_data:
            # Strong economics, weak data: the critical fields the screening
            # rule watches are blank even though the loan itself is sound.
            row["geographic_region_collateral"] = region   # kept, drives shares
            row["origination_date"] = None
            row["current_valuation_method"] = None

        rows.append(row)

    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Stated expectations
# --------------------------------------------------------------------------- #
#: Balances are uniform within a region, so cohort balances are exact products.
#: The first 12 loans are London at 280,000 each.
HIGH_LTV_BALANCE = HIGH_LTV_LOANS * 280_000.0          # 3,360,000
HIGH_LTV_SHARE = round(HIGH_LTV_BALANCE / TOTAL_BALANCE * 100, 6)   # 16.8%

EXTREME_LTV_BALANCE = EXTREME_LTV_LOANS * 280_000.0    # 280,000
NEGATIVE_EQUITY_SHARE = round(EXTREME_LTV_BALANCE / TOTAL_BALANCE * 100, 6)  # 1.4%

#: Loans 13-18 are London (280,000); loans 30-31 are South East (225,000).
ARREARS_30_BALANCE = ARREARS_30_LOANS * 280_000.0      # 1,680,000
ARREARS_90_BALANCE = ARREARS_90_LOANS * 225_000.0      # 450,000
#: perf_arrears_share(min_days=30) counts BOTH bands.
ARREARS_30_PLUS_SHARE = round((ARREARS_30_BALANCE + ARREARS_90_BALANCE)
                              / TOTAL_BALANCE * 100, 6)              # 10.65%
ARREARS_90_PLUS_SHARE = round(ARREARS_90_BALANCE / TOTAL_BALANCE * 100, 6)  # 2.25%

#: Region blocks by loan index, in the order REGION_PLAN declares them:
#:     London     0-19   (20 @ 280,000)
#:     South East 20-43  (24 @ 225,000)
#:     North West 44-59  (16 @ 275,000)
#:     Wales      60-73  (14 @ 200,000)
#:     Scotland   74-85  (12 @ 150,000)
#: Every cohort balance below is read off those blocks by hand.

#: Loans 40-47 straddle two regions: 40-43 South East (4 x 225,000 = 900,000)
#: and 44-47 North West (4 x 275,000 = 1,100,000). Valued 2021-06-30, which is
#: 61 months before the 2026-07-31 reporting date.
STALE_BALANCE = 900_000.0 + 1_100_000.0                # 2,000,000
STALE_SHARE = round(STALE_BALANCE / TOTAL_BALANCE * 100, 6)          # 10.0%

#: Loans 50-52 are all North West (3 x 275,000), no valuation at all.
MISSING_VALUATION_BALANCE = MISSING_VALUATION_LOANS * 275_000.0      # 825,000
MISSING_VALUATION_SHARE = round(MISSING_VALUATION_BALANCE
                                / TOTAL_BALANCE * 100, 6)            # 4.125%

TRUTH: Dict[str, Any] = {
    "loan_count": LOAN_COUNT,
    "total_balance": TOTAL_BALANCE,
    "largest_region": LARGEST_REGION,
    "largest_region_share": LARGEST_REGION_SHARE,
    "top_10_share": TOP_10_SHARE,
    "high_ltv_share": HIGH_LTV_SHARE,
    "negative_equity_share": NEGATIVE_EQUITY_SHARE,
    "arrears_30_plus_share": ARREARS_30_PLUS_SHARE,
    "arrears_90_plus_share": ARREARS_90_PLUS_SHARE,
    "stale_valuation_share": STALE_SHARE,
    "missing_valuation_share": MISSING_VALUATION_SHARE,
    "region_shares": {r: plan[3] for r, plan in REGION_PLAN.items()},
}

#: The point of the whole fixture: a portfolio where the warehouse is satisfied
#: and Trakt's screen is not. Neither verdict is wrong; they are different
#: questions, and an agent must not report the flag as a breach.
PASS_BUT_FLAGGED = ("CONC_GEOGRAPHIC", "CONC_TOP_N_LOANS")
