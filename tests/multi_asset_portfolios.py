#!/usr/bin/env python3
"""Four portfolios with materially different economics, in canonical shape.

The point of these is to prove one claim: **same registry, same analytics,
different governed data, appropriate capabilities.** No portfolio here has a
bespoke code path, and none is named in the resolver. If the capability matrix
below comes out right, the architecture is asset-agnostic; if it needed an
``if asset_class ==`` anywhere to come out right, it is not.

    A  equity release        collateralised, RREL35 = OTHR, contingent repayment
    B  fixed-rate amortising collateralised, RREL35 = FIXE, RREL42 = FXRL
    C  floating conventional collateralised, RREL35 = FRXX, RREL42 = FLIF
    D  unsecured credit      NO collateral, RREL35 = FIXE, RREL42 = FXRL

D is the important one for asset agnosticism. It is not a mortgage at all, it
carries no valuation of any kind, and it should still get its credit and
cash-flow capabilities — losing only the ones that need a collateral value.
Nothing about it was special-cased to make that happen.

C is the important one for the determinism model: it is floating, so its yield
is unknowable, and it amortises on a constant TOTAL instalment, so its
principal moves with the rate too — unlike a floating bullet, which keeps its
WAL.
"""

from __future__ import annotations

from typing import Dict

import pandas as pd

CUT_OFF = "2026-01-31"


def _base(prefix: str, count: int) -> Dict[str, list]:
    return {
        "loan_identifier": [f"{prefix}{i:04d}" for i in range(count)],
        "data_cut_off_date": [CUT_OFF] * count,
    }


def portfolio_a_equity_release(count: int = 40) -> pd.DataFrame:
    """Lifetime mortgages. Collateralised, interest rolls up, repayment falls
    due on death, sale or long-term care — which Trakt reports as RREL35 =
    OTHR, per ``config/asset/product_defaults_ERM.yaml``."""
    frame = pd.DataFrame(_base("ERM", count))
    frame["current_principal_balance"] = [
        120_000 + (i % 10) * 15_000 for i in range(count)]
    frame["original_principal_balance"] = [
        80_000 + (i % 10) * 10_000 for i in range(count)]
    frame["current_valuation_amount"] = [
        400_000 + (i % 8) * 50_000 for i in range(count)]
    frame["original_valuation_amount"] = [
        350_000 + (i % 8) * 50_000 for i in range(count)]
    frame["current_valuation_date"] = "2025-06-30"
    frame["collateral_geography"] = [
        ["London", "South East", "North West", "Scotland"][i % 4]
        for i in range(count)]
    frame["amortisation_type"] = "OTHR"          # contingent repayment
    frame["interest_rate_type"] = "FXRL"         # fixed for life, and roll-up
    frame["current_interest_rate"] = 6.4
    frame["maturity_date"] = "2075-01-31"        # legal long-stop only
    frame["scheduled_principal_payment_frequency"] = "OTHR"
    frame["scheduled_interest_payment_frequency"] = "OTHR"
    frame["origination_date"] = [
        f"20{15 + (i % 8):02d}-04-01" for i in range(count)]
    frame["number_of_days_in_arrears"] = 0
    frame["arrears_balance"] = 0.0
    frame["account_status"] = "Performing"
    frame["youngest_borrower_age"] = [68 + (i % 20) for i in range(count)]
    frame["purchase_price"] = 100.0
    return frame


def portfolio_b_fixed_amortising(count: int = 40) -> pd.DataFrame:
    """Conventional amortising mortgages: constant principal, fixed for life."""
    frame = pd.DataFrame(_base("FIX", count))
    frame["current_principal_balance"] = [
        150_000 + (i % 12) * 20_000 for i in range(count)]
    frame["original_principal_balance"] = [
        200_000 + (i % 12) * 20_000 for i in range(count)]
    frame["current_valuation_amount"] = [
        300_000 + (i % 9) * 40_000 for i in range(count)]
    frame["current_valuation_date"] = "2025-11-30"
    frame["collateral_geography"] = [
        ["London", "Midlands", "Wales", "North East"][i % 4]
        for i in range(count)]
    frame["amortisation_type"] = "FIXE"
    frame["interest_rate_type"] = "FXRL"
    frame["current_interest_rate"] = [4.0 + (i % 5) * 0.25 for i in range(count)]
    frame["maturity_date"] = [
        f"20{32 + (i % 6):02d}-01-31" for i in range(count)]
    frame["scheduled_principal_payment_frequency"] = "MNTH"
    frame["scheduled_interest_payment_frequency"] = "MNTH"
    frame["origination_date"] = [
        f"20{18 + (i % 6):02d}-09-01" for i in range(count)]
    frame["number_of_days_in_arrears"] = [
        [0, 0, 0, 35, 95][i % 5] for i in range(count)]
    frame["arrears_balance"] = [
        [0.0, 0.0, 0.0, 900.0, 3_100.0][i % 5] for i in range(count)]
    frame["account_status"] = [
        "Performing" if i % 5 < 3 else "Arrears" for i in range(count)]
    frame["purchase_price"] = 99.0
    frame["balloon_amount"] = 0.0
    # B is the only portfolio carrying loss and prepayment evidence, so the
    # matrix proves AVAILABLE is reachable for those capabilities and is not
    # merely UNAVAILABLE everywhere for want of a fixture.
    frame["allocated_losses"] = [
        0.0 if i % 10 else 12_000.0 for i in range(count)]
    frame["cumulative_recoveries"] = [
        0.0 if i % 10 else 4_000.0 for i in range(count)]
    frame["default_amount"] = [
        0.0 if i % 10 else 40_000.0 for i in range(count)]
    frame["unscheduled_principal_collections"] = [
        0.0 if i % 6 else 5_000.0 for i in range(count)]
    frame["loan_redemption_flag"] = "N"
    return frame


def portfolio_c_floating_conventional(count: int = 40) -> pd.DataFrame:
    """Floating-rate French-amortising loans.

    Both determinism flags fail here, and for different reasons: the rate is
    floating (so interest is unknowable) *and* the amortisation type fixes the
    total instalment (so principal moves with the rate too). A floating BULLET
    would keep its WAL; this does not.
    """
    frame = pd.DataFrame(_base("FLT", count))
    frame["current_principal_balance"] = [
        250_000 + (i % 10) * 30_000 for i in range(count)]
    frame["original_principal_balance"] = [
        300_000 + (i % 10) * 30_000 for i in range(count)]
    frame["current_valuation_amount"] = [
        500_000 + (i % 7) * 75_000 for i in range(count)]
    frame["current_valuation_date"] = "2025-09-30"
    frame["collateral_geography"] = [
        ["London", "South West", "Yorkshire"][i % 3] for i in range(count)]
    frame["amortisation_type"] = "FRXX"
    frame["interest_rate_type"] = "FLIF"
    frame["current_interest_rate"] = [5.5 + (i % 4) * 0.3 for i in range(count)]
    frame["current_interest_rate_margin"] = 2.0
    frame["current_interest_rate_index"] = "SONIA"
    frame["interest_rate_reset_interval"] = 3
    frame["maturity_date"] = [
        f"20{34 + (i % 5):02d}-06-30" for i in range(count)]
    frame["scheduled_principal_payment_frequency"] = "QUTR"
    frame["scheduled_interest_payment_frequency"] = "QUTR"
    frame["origination_date"] = [
        f"20{19 + (i % 5):02d}-06-30" for i in range(count)]
    frame["number_of_days_in_arrears"] = [
        [0, 0, 45, 0][i % 4] for i in range(count)]
    frame["arrears_balance"] = [
        [0.0, 0.0, 2_400.0, 0.0][i % 4] for i in range(count)]
    frame["account_status"] = "Performing"
    frame["purchase_price"] = 100.0
    return frame


def portfolio_d_unsecured(count: int = 40) -> pd.DataFrame:
    """Unsecured credit. No valuation of any kind, no geography.

    The asset-agnosticism proof. Credit performance and contractual cash-flow
    capabilities should survive; only the ones that genuinely need a collateral
    VALUE should drop out, and they should drop out as NOT_APPLICABLE rather
    than as a missing field — because no field could be supplied to fix them.
    """
    frame = pd.DataFrame(_base("UNS", count))
    frame["current_principal_balance"] = [
        5_000 + (i % 15) * 1_200 for i in range(count)]
    frame["original_principal_balance"] = [
        8_000 + (i % 15) * 1_200 for i in range(count)]
    frame["amortisation_type"] = "FIXE"
    frame["interest_rate_type"] = "FXRL"
    frame["current_interest_rate"] = [9.9 + (i % 6) * 0.5 for i in range(count)]
    frame["maturity_date"] = [
        f"20{28 + (i % 4):02d}-03-31" for i in range(count)]
    frame["scheduled_principal_payment_frequency"] = "MNTH"
    frame["scheduled_interest_payment_frequency"] = "MNTH"
    frame["origination_date"] = [
        f"20{22 + (i % 4):02d}-03-31" for i in range(count)]
    frame["number_of_days_in_arrears"] = [
        [0, 0, 0, 0, 15, 65, 120][i % 7] for i in range(count)]
    frame["arrears_balance"] = [
        [0.0, 0.0, 0.0, 0.0, 90.0, 310.0, 640.0][i % 7] for i in range(count)]
    frame["account_status"] = [
        "Performing" if i % 7 < 4 else "Arrears" for i in range(count)]
    frame["purchase_price"] = 97.5
    return frame


#: Name -> builder, so a test can iterate the whole set without naming them
#: individually and quietly skipping one.
PORTFOLIOS = {
    "A_equity_release": portfolio_a_equity_release,
    "B_fixed_amortising": portfolio_b_fixed_amortising,
    "C_floating_conventional": portfolio_c_floating_conventional,
    "D_unsecured": portfolio_d_unsecured,
}


def with_history(builder, periods: int = 3) -> Dict[str, pd.DataFrame]:
    """The same portfolio across consecutive month-ends, amortising slightly,
    for the capabilities that need two or more snapshots."""
    frames: Dict[str, pd.DataFrame] = {}
    for index, period in enumerate(
            ["2025-11-30", "2025-12-31", "2026-01-31"][-periods:]):
        frame = builder()
        frame["data_cut_off_date"] = period
        factor = 1.0 + (periods - index - 1) * 0.01
        frame["current_principal_balance"] = (
            frame["current_principal_balance"] * factor).round(2)
        frames[period] = frame
    return frames
