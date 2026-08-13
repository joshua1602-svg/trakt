#!/usr/bin/env python3
"""An 18-month synthetic portfolio in which the SAME loans persist through time.

Twelve unrelated portfolios would prove nothing. Behaviour is only observable
when a loan can be followed: ``L00042`` current in January, 30 DPD in March, 90
DPD in June, cured in August. That is what this fixture provides.

    18 monthly snapshots, 2025-02-28 → 2026-07-31
    400 loans at origination, declining through redemption
    two origination vintages, deliberately different

Planted behaviours, each with a cohort large enough to measure:

    stable            behaves normally throughout — the control
    deteriorating     30 → 60 → 90 DPD, and stays there
    curing            goes delinquent, then returns to performing
    redeeming         full redemption at staggered months, then absent
    prepaying         partial unscheduled principal, no exit
    defaulting        default recorded, loss allocated, partial recovery
    ltv_worsening     valuation falls, balance flat
    ltv_improving     balance amortises against a stable valuation
    stale_valuation   never re-valued; ages 18 months across the window
    refreshed         re-valued mid-window, so age resets

Two vintages, and the agent is deliberately NOT told which is weak:

    2024 vintage      stable arrears throughout
    2025 vintage      materially worse, and worsening

Concentration drift is produced by RUNOFF, not by editing shares: the redeeming
cohort is South East, so London's share rises from 31.69% to 36.18% across the
window without any loan's region ever changing. That is what makes it an
*observed* behaviour rather than an authored one — no field is edited to create
it, and it falls out of loans leaving the book.

Expected truth is stated below as literals. Nothing here calls the analytics
under test to decide what the answer should be.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

BOOK = "direct_001"

#: 18 month-ends. Written out rather than generated so the periods are a stated
#: fact of the fixture and cannot drift with a date library's conventions.
PERIODS: Tuple[str, ...] = (
    "2025-02-28", "2025-03-31", "2025-04-30", "2025-05-31", "2025-06-30",
    "2025-07-31", "2025-08-31", "2025-09-30", "2025-10-31", "2025-11-30",
    "2025-12-31", "2026-01-31", "2026-02-28", "2026-03-31", "2026-04-30",
    "2026-05-31", "2026-06-30", "2026-07-31",
)
PERIOD_COUNT = len(PERIODS)
FIRST_PERIOD, LAST_PERIOD = PERIODS[0], PERIODS[-1]


def snapshot_id(period: str) -> str:
    return f"snap_hist_{period.replace('-', '')}"


# --------------------------------------------------------------------------- #
# Cohort plan — the whole fixture in one table
# --------------------------------------------------------------------------- #
#: name -> (count, region, opening balance, vintage year)
COHORTS: Dict[str, Tuple[int, str, float, int]] = {
    "stable_london":      (60, "London",     250_000.0, 2024),
    "stable_south_east":  (80, "South East", 200_000.0, 2024),
    "stable_north_west":  (60, "North West", 180_000.0, 2024),
    "deteriorating":      (40, "Wales",      150_000.0, 2025),
    "curing":             (30, "Scotland",   140_000.0, 2025),
    "redeeming":          (50, "South East", 200_000.0, 2024),
    "prepaying":          (30, "North West", 180_000.0, 2024),
    "defaulting":         (10, "Wales",      150_000.0, 2025),
    "ltv_worsening":      (20, "London",     250_000.0, 2025),
    "ltv_improving":      (20, "London",     250_000.0, 2024),
}
LOAN_COUNT = sum(c[0] for c in COHORTS.values())          # 400

#: Monthly scheduled amortisation, as a fraction of opening balance. Flat and
#: small so that unscheduled activity is distinguishable from it.
SCHEDULED_AMORTISATION = 0.0025          # 0.25% per month

#: The redeeming cohort exits in waves, so a redemption rate is observable in
#: several periods rather than as one cliff.
REDEMPTION_WAVES: Dict[int, int] = {
    4: 5, 6: 5, 8: 6, 10: 6, 12: 7, 14: 7, 16: 8,   # period index -> loans
}
REDEEMED_TOTAL = sum(REDEMPTION_WAVES.values())           # 44 of 50

#: Partial unscheduled prepayment: this fraction of opening balance, in these
#: periods, for every loan in the prepaying cohort.
PREPAY_PERIODS = (3, 7, 11, 15)
PREPAY_FRACTION = 0.05

#: Arrears choreography, by period index. Days in arrears for the cohort.
DETERIORATING_DPD = {
    0: 0, 1: 0, 2: 0, 3: 30, 4: 30, 5: 60, 6: 60, 7: 90, 8: 90,
    9: 120, 10: 120, 11: 150, 12: 150, 13: 180, 14: 180, 15: 210,
    16: 210, 17: 240,
}
CURING_DPD = {
    0: 0, 1: 0, 2: 30, 3: 60, 4: 90, 5: 90, 6: 60, 7: 30, 8: 0,
    9: 0, 10: 0, 11: 0, 12: 30, 13: 60, 14: 30, 15: 0, 16: 0, 17: 0,
}
DEFAULTING_DPD = {i: (0 if i < 3 else min(30 * (i - 2), 360)) for i in range(18)}
#: The defaulting cohort is recorded as defaulted from this period onward.
DEFAULT_PERIOD = 9
#: Loss allocated per defaulted loan, and recovery received, from these periods.
DEFAULT_LOSS_PER_LOAN = 30_000.0
DEFAULT_RECOVERY_PER_LOAN = 12_000.0
RECOVERY_PERIOD = 13

#: Valuations. The worsening cohort's collateral falls; the improving cohort's
#: holds while the balance amortises.
OPENING_VALUATION_MULTIPLE = 1.6667          # LTV 60% at outset
LTV_WORSENING_VALUATION_DECLINE = 0.015      # 1.5% per month
#: Re-valued in this period, so valuation age resets for the refreshed group.
REVALUATION_PERIOD = 9
#: How many of the stable_london loans get a refreshed valuation.
REFRESHED_LOANS = 20

#: London runs off more slowly than the rest, which is what drifts the share.
#: Stated as an intent; the observed values are asserted from the built frames.
CONCENTRATION_DRIFT_COHORT = "redeeming"     # South East, so London rises


# --------------------------------------------------------------------------- #
# Construction
# --------------------------------------------------------------------------- #
def _loan_plan() -> List[Dict[str, Any]]:
    """One record per loan: its cohort, region, opening balance and vintage."""
    plan: List[Dict[str, Any]] = []
    index = 0
    for cohort, (count, region, balance, vintage) in COHORTS.items():
        for n in range(count):
            plan.append({
                "loan_identifier": f"H{index:05d}",
                "cohort": cohort,
                "region": region,
                "opening_balance": balance,
                "vintage": vintage,
                "within_cohort": n,
            })
            index += 1
    return plan


def _redeemed_by(period_index: int) -> int:
    """How many of the redeeming cohort have exited by this period."""
    return sum(count for p, count in REDEMPTION_WAVES.items() if p <= period_index)


def _dpd(cohort: str, period_index: int) -> int:
    if cohort == "deteriorating":
        return DETERIORATING_DPD[period_index]
    if cohort == "curing":
        return CURING_DPD[period_index]
    if cohort == "defaulting":
        return DEFAULTING_DPD[period_index]
    return 0


def _balance(row: Dict[str, Any], period_index: int) -> Optional[float]:
    """Balance at this period, or ``None`` when the loan has exited.

    Amortisation is scheduled and flat; prepayment and redemption are the only
    unscheduled movements, which is what lets the two be told apart downstream.
    """
    cohort = row["cohort"]
    balance = row["opening_balance"]

    if cohort == "redeeming":
        # Loans exit in waves, lowest within-cohort index first.
        if row["within_cohort"] < _redeemed_by(period_index):
            return None

    balance *= (1.0 - SCHEDULED_AMORTISATION) ** period_index

    if cohort == "prepaying":
        for prepay_period in PREPAY_PERIODS:
            if period_index >= prepay_period:
                balance *= (1.0 - PREPAY_FRACTION)

    return round(balance, 2)


def _unscheduled_principal(row: Dict[str, Any], period_index: int) -> float:
    """Unscheduled principal collected IN this period. The prepayment numerator.

    Deliberately zero for scheduled amortisation: labelling all balance
    reduction as prepayment is the mistake this fixture exists to make
    detectable.
    """
    cohort = row["cohort"]
    if cohort == "prepaying" and period_index in PREPAY_PERIODS:
        opening = _balance(row, period_index - 1)
        return round((opening or 0.0) * PREPAY_FRACTION, 2)
    return 0.0


def _redemption(row: Dict[str, Any], period_index: int) -> float:
    """Full redemption received in this period."""
    if row["cohort"] != "redeeming":
        return 0.0
    before = _redeemed_by(period_index - 1) if period_index else 0
    now = _redeemed_by(period_index)
    if before <= row["within_cohort"] < now:
        return round(_balance(row, period_index - 1) or 0.0, 2)
    return 0.0


def _valuation(row: Dict[str, Any], period_index: int) -> Tuple[float, str, str]:
    """``(amount, valuation_date, method)`` for this loan at this period."""
    base = round(row["opening_balance"] * OPENING_VALUATION_MULTIPLE, 2)
    cohort = row["cohort"]

    if cohort == "ltv_worsening":
        amount = round(base * (1.0 - LTV_WORSENING_VALUATION_DECLINE)
                       ** period_index, 2)
        # Re-indexed monthly, so the date moves with the value.
        return amount, PERIODS[period_index], "indexed valuation"

    refreshed = (cohort == "stable_london"
                 and row["within_cohort"] < REFRESHED_LOANS
                 and period_index >= REVALUATION_PERIOD)
    if refreshed:
        return base, PERIODS[REVALUATION_PERIOD], "full valuation"

    # Everyone else keeps the origination valuation, which ages across the
    # window — which is the point of the stale-valuation planting.
    return base, "2024-08-31", "full valuation"


def snapshot(period_index: int) -> pd.DataFrame:
    """The portfolio as at one period. Exited loans are absent, not zeroed."""
    period = PERIODS[period_index]
    rows: List[Dict[str, Any]] = []

    for plan in _loan_plan():
        balance = _balance(plan, period_index)
        if balance is None:
            continue                       # redeemed: absent from the tape

        cohort = plan["cohort"]
        days = _dpd(cohort, period_index)
        defaulted = (cohort == "defaulting" and period_index >= DEFAULT_PERIOD)
        amount, valuation_date, method = _valuation(plan, period_index)

        losses = (DEFAULT_LOSS_PER_LOAN if defaulted else 0.0)
        recoveries = (DEFAULT_RECOVERY_PER_LOAN
                      if defaulted and period_index >= RECOVERY_PERIOD else 0.0)

        rows.append({
            "loan_identifier": plan["loan_identifier"],
            "borrower_identifier": f"BH{plan['loan_identifier'][1:]}",
            "source_portfolio_id": BOOK,
            "source_portfolio_type": "direct",
            "reporting_date": period,
            "account_status": ("defaulted" if defaulted
                               else "arrears" if days else "performing"),
            "current_principal_balance": balance,
            "current_outstanding_balance": balance,
            "original_principal_balance": plan["opening_balance"],
            "current_loan_to_value": round(balance / amount * 100, 4),
            "current_valuation_amount": amount,
            "current_valuation_date": valuation_date,
            "current_valuation_method": method,
            "number_of_days_in_arrears": days,
            "arrears_balance": round(balance * 0.02, 2) if days else 0.0,
            "ifrs9_stage": "3" if defaulted else ("2" if days else "1"),
            "default_date": PERIODS[DEFAULT_PERIOD] if defaulted else None,
            "default_amount": plan["opening_balance"] if defaulted else None,
            "allocated_losses": losses,
            "cumulative_recoveries": recoveries,
            "recoveries_in_period": (DEFAULT_RECOVERY_PER_LOAN
                                     if defaulted and period_index == RECOVERY_PERIOD
                                     else 0.0),
            "unscheduled_principal_collections":
                _unscheduled_principal(plan, period_index),
            "redemptions_received_in_period": 0.0,   # exits recorded below
            # Sprint 2.5C: a disappearance is only counted as a voluntary
            # redemption with qualifying evidence. The redeeming cohort carries
            # the flag a real servicer tape would set; every other cohort does
            # not, so an exit from one of them classifies as UNKNOWN_EXIT and is
            # excluded from the prepayment numerator.
            "loan_redemption_flag": "Y" if cohort == "redeeming" else None,
            "geographic_region_collateral": plan["region"],
            "geographic_region_obligor": plan["region"],
            "collateral_type": "residential_property",
            "origination_date": ("2024-03-01" if plan["vintage"] == 2024
                                 else "2025-01-15"),
            "portfolio_cohort": str(plan["vintage"]),
            "maturity_date": "2049-03-01",
            "current_interest_rate": 5.25,
            "exposure_currency_denomination": "GBP",
            "collateral_currency": "GBP",
        })

    return pd.DataFrame(rows)


def redemptions_in_period(period_index: int) -> float:
    """Balance that left the book by full redemption in this period.

    Reported separately because a redeemed loan is ABSENT from the closing
    snapshot — so the exit cannot be read off the closing tape and has to come
    from the opening one.
    """
    if period_index == 0:
        return 0.0
    return round(sum(_redemption(plan, period_index) for plan in _loan_plan()), 2)


def history() -> Dict[str, pd.DataFrame]:
    """Every snapshot, keyed by reporting date."""
    return {PERIODS[i]: snapshot(i) for i in range(PERIOD_COUNT)}


# --------------------------------------------------------------------------- #
# Stated truth
# --------------------------------------------------------------------------- #
#: Cohorts and their planted intent, for tests to assert against by name.
COHORT_INTENT = {
    "stable_london":     "control: no arrears, normal amortisation",
    "stable_south_east": "control",
    "stable_north_west": "control",
    "deteriorating":     "30 -> 60 -> 90+ DPD and stays there",
    "curing":            "delinquent then returns to performing, twice",
    "redeeming":         "44 of 50 exit in seven waves",
    "prepaying":         "4 partial unscheduled prepayments of 5%",
    "defaulting":        "defaults at period 9, recovery at period 13",
    "ltv_worsening":     "valuation falls 1.5%/month",
    "ltv_improving":     "balance amortises against a flat valuation",
}

#: The vintage the agent must discover rather than be told.
WEAK_VINTAGE = 2025
STRONG_VINTAGE = 2024

#: 2025 vintage = deteriorating + curing + defaulting + ltv_worsening.
VINTAGE_2025_COHORTS = ("deteriorating", "curing", "defaulting", "ltv_worsening")
VINTAGE_2024_COHORTS = ("stable_london", "stable_south_east", "stable_north_west",
                        "redeeming", "prepaying", "ltv_improving")

#: Arrears is planted ONLY in 2025-vintage cohorts, so the 2024 vintage must
#: show zero 30+ DPD in every period. That is the sharpest available assertion
#: that vintage analysis is reading the right population.
VINTAGE_2024_HAS_NO_ARREARS = True

#: Loans in each vintage at outset.
VINTAGE_2025_LOANS = sum(COHORTS[c][0] for c in VINTAGE_2025_COHORTS)   # 100
VINTAGE_2024_LOANS = sum(COHORTS[c][0] for c in VINTAGE_2024_COHORTS)   # 300

#: Deteriorating cohort reaches 90+ DPD at period 7 and never recovers.
DETERIORATING_REACHES_90_AT = 7
#: Curing cohort returns to 0 DPD at period 8 having been at 90.
CURING_RETURNS_AT = 8
CURING_PEAK_DPD = 90

#: Total unscheduled principal across the window, per prepaying loan.
#: 180,000 opening; four prepayments of 5% of the then-current balance.
PREPAYING_LOANS = COHORTS["prepaying"][0]                # 30
PREPAY_EVENT_COUNT = len(PREPAY_PERIODS)                 # 4

#: Defaults and losses, stated exactly.
DEFAULTED_LOANS = COHORTS["defaulting"][0]               # 10
TOTAL_ALLOCATED_LOSSES = DEFAULTED_LOANS * DEFAULT_LOSS_PER_LOAN       # 300,000
TOTAL_RECOVERIES = DEFAULTED_LOANS * DEFAULT_RECOVERY_PER_LOAN         # 120,000
#: Recovery rate on losses = 120,000 / 300,000.
RECOVERY_RATE_ON_LOSSES = 40.0

#: Valuation ages at the final period, measured from 2024-08-31 to 2026-07-31.
STALE_VALUATION_AGE_MONTHS = 23          # (2026-2024)*12 + (7-8) = 23
REFRESHED_VALUATION_AGE_MONTHS = 8       # from 2025-11-30 to 2026-07-31

# --------------------------------------------------------------------------- #
# Observed series, read off the constructed frames and written down here.
#
# These are the fixture's OUTPUT rather than its input: the planting above is
# choreographed per cohort, and these are what that choreography produces at
# portfolio level. Stated so a test can assert the shape of the history without
# recomputing it, and so a change to the planting that alters portfolio
# behaviour fails loudly instead of silently moving the target.
# --------------------------------------------------------------------------- #

#: Loans present in the first and last snapshots.
OPENING_LOAN_COUNT = 400
CLOSING_LOAN_COUNT = 356

#: Balance-weighted, first and last period.
OPENING_BALANCE = 78_900_000.0
CLOSING_BALANCE_APPROX = 66_219_660.0

#: 30+ DPD share: zero for three periods, then rises. Peak is period 14.
ARREARS_30_OPENING_PCT = 0.0
ARREARS_30_CLOSING_PCT = 10.85
ARREARS_30_PEAK_PCT = 16.50

#: 90+ DPD share: zero until the deteriorating cohort reaches 90 at period 7.
ARREARS_90_OPENING_PCT = 0.0
ARREARS_90_CLOSING_PCT = 10.85

#: Concentration drift, produced entirely by runoff.
LONDON_OPENING_PCT = 31.69
LONDON_CLOSING_PCT = 36.18
LONDON_DRIFT_PP = 4.49

#: Weighted-average LTV improves slightly overall: amortisation across most of
#: the book outweighs the deliberately worsening cohort.
WA_LTV_OPENING = 60.00
WA_LTV_CLOSING = 58.04
