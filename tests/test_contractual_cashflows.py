#!/usr/bin/env python3
"""Contractual cash flows, WAL and YTM — against arithmetic done by hand.

Every expected value here is worked out independently of the code under test.
Where a schedule is short enough to write down, it is written down.

The structure follows the one distinction the module exists to hold: **WAL is
principal-weighted, YTM is not.** Under BLLT and FIXE the contract fixes the
principal series directly, so their WAL survives a floating rate while their
YTM does not. Under FRXX and DEXX the contract fixes the *total* instalment, so
the principal split is rate-dependent and both fall together. Several tests
below exist only to assert that those two cases stay apart.
"""

from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

import pandas as pd
import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from analytics_lib.contractual import (
    ASSUMPTION_REQUIRED,
    FIELD_GAP,
    NOT_APPLICABLE,
    SUPPORTED,
    WAL_METHOD,
    YTM_METHOD,
    build_schedule,
    contractual_wal,
    contractual_ytm,
    frequency_months,
    normalise_amortisation,
    normalise_rate_type,
    portfolio_wal,
    schedule_frame,
)

AS_OF = date(2026, 1, 31)


def _loan(**overrides):
    base = {
        "loan_identifier": "L1",
        "current_principal_balance": 1_000_000.0,
        "maturity_date": "2031-01-31",
        "amortisation_type": "BLLT",
        "interest_rate_type": "FXRL",
        "current_interest_rate": 5.0,
        "scheduled_principal_payment_frequency": "MNTH",
        "scheduled_interest_payment_frequency": "MNTH",
        "data_cut_off_date": "2026-01-31",
        "purchase_price": 100.0,
    }
    base.update(overrides)
    return base


# =========================================================================== #
# Enum normalisation — reused from the delivery rules, not restated
# =========================================================================== #
def test_the_repositorys_own_enum_map_is_what_normalises_rrel35():
    """`annex2_delivery_rules.yaml` already maps "French" to FRXX for the
    delivery path. A second mapping in Python would be a second definition of
    one thing, and they would drift the first time a synonym was added to only
    one of them."""
    assert normalise_amortisation("French") == "FRXX"
    assert normalise_amortisation("Bullet") == "BLLT"
    assert normalise_amortisation("Interest roll-up") == "OTHR"
    assert normalise_amortisation("BLLT") == "BLLT"


def test_an_unrecognised_amortisation_label_is_none_not_other():
    """"We do not recognise this label" and "the contract says OTHR" are
    different statements, and only one of them is a fact about the loan.
    Collapsing the first into the second would turn a parsing failure into a
    silent NOT_APPLICABLE."""
    assert normalise_amortisation("wibble") is None
    assert normalise_rate_type("wibble") is None


def test_an_other_frequency_is_not_assumed_to_be_monthly():
    assert frequency_months("MNTH") == 1
    assert frequency_months("QUTR") == 3
    assert frequency_months("SEMI") == 6
    assert frequency_months("YEAR") == 12
    assert frequency_months("OTHR") is None


# =========================================================================== #
# BLLT
# =========================================================================== #
def test_a_bullet_repays_all_principal_at_maturity_and_nothing_before():
    schedule = build_schedule(_loan(), as_of=AS_OF)
    assert schedule.status == SUPPORTED
    assert schedule.principal[-1] == pytest.approx(1_000_000.0)
    assert sum(schedule.principal[:-1]) == 0.0
    assert schedule.dates[-1] == date(2031, 1, 31)


def test_bullet_wal_is_exactly_the_time_to_maturity():
    """Five years to the day from 2026-01-31 to 2031-01-31 is 1,826 days
    (2028 is a leap year). On ACT/365F that is 1826/365 = 5.002740 years.
    Worked by hand; the single principal flow makes WAL the term itself."""
    result = contractual_wal(build_schedule(_loan(), as_of=AS_OF))
    assert result.status == SUPPORTED
    assert (date(2031, 1, 31) - AS_OF).days == 1826
    assert result.wal_years == pytest.approx(1826 / 365.0, abs=1e-6)
    assert result.wal_years == pytest.approx(5.00274, abs=1e-5)


def test_bullet_ytm_at_par_is_the_coupon():
    """A bullet bought at par pays 5% and returns principal at maturity, so its
    yield is its coupon. Monthly periodic yield 5/12 = 0.4166667%; annualised
    (1 + 0.00416667)^12 − 1 = 5.116190%. Both worked by hand."""
    schedule = build_schedule(_loan(), as_of=AS_OF)
    result = contractual_ytm(schedule, loan=_loan())

    assert result.status == SUPPORTED
    assert result.periodic_yield_pct == pytest.approx(5.0 / 12.0, abs=1e-6)
    expected = ((1 + 0.05 / 12) ** 12 - 1) * 100
    assert expected == pytest.approx(5.116190, abs=1e-5)
    assert result.ytm_pct == pytest.approx(expected, abs=1e-5)


def test_a_bullet_bought_at_a_discount_yields_more_than_its_coupon():
    """RREL34 is a price relative to par, so 98 means a discount. The yield
    must exceed the 5% coupon — the direction is the assertion, because a sign
    error here would be invisible in a single number."""
    at_par = contractual_ytm(build_schedule(_loan(), as_of=AS_OF),
                             price_percent_of_par=100.0)
    discounted = contractual_ytm(build_schedule(_loan(), as_of=AS_OF),
                                 price_percent_of_par=98.0)
    premium = contractual_ytm(build_schedule(_loan(), as_of=AS_OF),
                              price_percent_of_par=102.0)

    assert discounted.ytm_pct > at_par.ytm_pct > premium.ytm_pct


def test_a_floating_bullet_keeps_its_wal_and_loses_its_ytm():
    """The whole point of separating principal- from interest-determinism. A
    bullet's principal is one flow at maturity whatever the rate does, so WAL
    is contractual. The yield is not, and must refuse rather than use today's
    rate as if it were fixed."""
    floating = _loan(interest_rate_type="FLIF")
    schedule = build_schedule(floating, as_of=AS_OF)

    assert schedule.status == SUPPORTED
    assert schedule.principal_deterministic is True
    assert schedule.interest_deterministic is False

    wal = contractual_wal(schedule)
    assert wal.status == SUPPORTED
    assert wal.wal_years == pytest.approx(1826 / 365.0, abs=1e-6)

    ytm = contractual_ytm(schedule, loan=floating)
    assert ytm.status == ASSUMPTION_REQUIRED
    assert ytm.ytm_pct is None
    assert "rate assumption" in ytm.reason


def test_a_rate_fixed_only_to_a_revision_date_inside_the_schedule_refuses():
    """FXPR is fixed *with future periodic resets*. RREL51 dates the reset;
    RREL50 gives the new margin, not the new index level. So the yield beyond
    that date is not a contractual fact."""
    loan = _loan(interest_rate_type="FXPR",
                 interest_revision_date_1="2028-01-31")
    schedule = build_schedule(loan, as_of=AS_OF)
    assert schedule.interest_deterministic is False
    assert contractual_ytm(schedule, loan=loan).status == ASSUMPTION_REQUIRED


def test_a_revision_date_beyond_the_final_payment_is_not_a_blocker():
    """A reset that never happens inside the schedule cannot make the schedule
    uncertain. Refusing here would be conservative to the point of wrong."""
    loan = _loan(interest_rate_type="FXPR",
                 interest_revision_date_1="2040-01-31")
    schedule = build_schedule(loan, as_of=AS_OF)
    assert schedule.interest_deterministic is True
    assert contractual_ytm(schedule, loan=loan).status == SUPPORTED


# =========================================================================== #
# FIXE
# =========================================================================== #
def test_fixe_repays_constant_principal_and_its_wal_is_the_midpoint():
    """Four annual instalments of £250,000 on a £1,000,000 balance, at
    1, 2, 3 and 4 years. WAL = (1+2+3+4)/4 = 2.5 years on an exact-year basis.

    The dates are 2027-01-31 … 2030-01-31, so on ACT/365F the elapsed days are
    365, 730, 1096 and 1461 (2028 is a leap year), giving
    (365+730+1096+1461)/4/365 = 913/365 = 2.501370 years. Worked by hand.
    """
    loan = _loan(amortisation_type="FIXE", maturity_date="2030-01-31",
                 scheduled_principal_payment_frequency="YEAR",
                 scheduled_interest_payment_frequency="YEAR")
    schedule = build_schedule(loan, as_of=AS_OF)

    assert schedule.status == SUPPORTED
    assert len(schedule.principal) == 4
    assert all(p == pytest.approx(250_000.0) for p in schedule.principal)

    days = [(d - AS_OF).days for d in schedule.dates]
    assert days == [365, 730, 1096, 1461]
    assert contractual_wal(schedule).wal_years == pytest.approx(
        sum(days) / 4 / 365.0, abs=1e-6)
    assert contractual_wal(schedule).wal_years == pytest.approx(2.50137, abs=1e-5)


def test_fixe_derives_its_constant_only_because_rrel35_says_it_is_constant():
    """`regular_principal_instalment` carries no regime code on any annex, so
    it is often absent. Recovering it as (balance − balloon) ÷ periods is
    sound *only* because RREL35 = FIXE has already asserted the per-instalment
    principal is the same. The note has to say so — outside FIXE the identical
    arithmetic would be a fabrication."""
    loan = _loan(amortisation_type="FIXE", maturity_date="2030-01-31",
                 scheduled_principal_payment_frequency="YEAR")
    schedule = build_schedule(loan, as_of=AS_OF)
    assert any("RREL35 = FIXE" in note for note in schedule.notes)


def test_fixe_with_a_balloon_puts_it_at_maturity_per_rrel41():
    """£1,000,000 with a £400,000 balloon over four annual periods: three
    instalments of £200,000 and £400,000 at maturity.

    WAL = (200k×1 + 200k×2 + 200k×3 + 400k×4) / 1,000,000
        = (200 + 400 + 600 + 1600) / 1000 = 2.8 years on an exact-year basis.
    The balloon pulls the WAL out by 0.3 years against the level case, which is
    the whole reason RREL41 has to land on the maturity date.
    """
    loan = _loan(amortisation_type="FIXE", maturity_date="2030-01-31",
                 scheduled_principal_payment_frequency="YEAR",
                 balloon_amount=400_000.0)
    schedule = build_schedule(loan, as_of=AS_OF)

    assert list(schedule.principal) == pytest.approx(
        [200_000.0, 200_000.0, 200_000.0, 400_000.0])
    assert schedule.dates[-1] == date(2030, 1, 31)

    days = [(d - AS_OF).days for d in schedule.dates]
    expected = sum(d * p for d, p in zip(days, schedule.principal)) / 1_000_000
    assert contractual_wal(schedule).wal_years == pytest.approx(
        expected / 365.0, abs=1e-6)
    # And materially longer than the level-pay case above.
    assert contractual_wal(schedule).wal_years > 2.79


def test_a_floating_fixe_keeps_its_wal_too():
    """Constant principal is rate-independent, so the FIXE case behaves like
    the bullet: WAL contractual, YTM not."""
    loan = _loan(amortisation_type="FIXE", maturity_date="2030-01-31",
                 scheduled_principal_payment_frequency="YEAR",
                 interest_rate_type="CAPP")
    schedule = build_schedule(loan, as_of=AS_OF)

    assert schedule.principal_deterministic is True
    assert schedule.interest_deterministic is False
    assert contractual_wal(schedule).status == SUPPORTED
    assert contractual_ytm(schedule, loan=loan).status == ASSUMPTION_REQUIRED


# =========================================================================== #
# FRXX — the case the earlier sprint wrongly called a modelling gap
# =========================================================================== #
def test_frxx_splits_a_constant_instalment_into_interest_and_principal():
    """Hand-worked, four annual periods, £100,000 at 10% with a £30,000
    constant instalment supplied as payment_due (RREL39):

      period 1: interest 10,000.00, principal 20,000.00, balance 80,000.00
      period 2: interest  8,000.00, principal 22,000.00, balance 58,000.00
      period 3: interest  5,800.00, principal 24,200.00, balance 33,800.00
      period 4: final instalment closes the loan — principal 33,800.00,
                interest 3,380.00

    Every line above is arithmetic anyone can check, and the rising principal
    share is the property that makes FRXX rate-dependent.
    """
    loan = _loan(amortisation_type="FRXX", current_principal_balance=100_000.0,
                 maturity_date="2030-01-31", current_interest_rate=10.0,
                 scheduled_principal_payment_frequency="YEAR",
                 payment_due=30_000.0)
    schedule = build_schedule(loan, as_of=AS_OF)

    assert schedule.status == SUPPORTED
    assert list(schedule.interest) == pytest.approx(
        [10_000.0, 8_000.0, 5_800.0, 3_380.0])
    assert list(schedule.principal) == pytest.approx(
        [20_000.0, 22_000.0, 24_200.0, 33_800.0])
    # The schedule closes exactly on the balance — no residue, no overshoot.
    assert sum(schedule.principal) == pytest.approx(100_000.0)


def test_frxx_wal_uses_the_rising_principal_series():
    """Continuing the hand-worked book above:
    (20,000×1 + 22,000×2 + 24,200×3 + 33,800×4) / 100,000
      = (20,000 + 44,000 + 72,600 + 135,200) / 100,000 = 2.718 years
    on an exact-year basis."""
    loan = _loan(amortisation_type="FRXX", current_principal_balance=100_000.0,
                 maturity_date="2030-01-31", current_interest_rate=10.0,
                 scheduled_principal_payment_frequency="YEAR",
                 payment_due=30_000.0)
    schedule = build_schedule(loan, as_of=AS_OF)
    days = [(d - AS_OF).days for d in schedule.dates]
    expected = sum(d * p for d, p in zip(days, schedule.principal)) / 100_000
    result = contractual_wal(schedule)

    assert result.wal_years == pytest.approx(expected / 365.0, abs=1e-6)
    # Sanity against the exact-year arithmetic in the docstring.
    assert result.wal_years == pytest.approx(2.718, abs=0.01)


def test_frxx_uses_payment_due_rrel39_as_the_constant_instalment():
    """RREL39 is "the next contractual payment due ... according to the payment
    frequency", which for a constant-total structure IS that constant — and it
    is carried on all five annexes. Sprint 2.5C missed it and called fixed-rate
    FRXX a modelling gap on that basis."""
    with_stated = build_schedule(
        _loan(amortisation_type="FRXX", current_principal_balance=100_000.0,
              maturity_date="2030-01-31", current_interest_rate=10.0,
              scheduled_principal_payment_frequency="YEAR",
              payment_due=30_000.0), as_of=AS_OF)
    assert with_stated.interest[0] + with_stated.principal[0] == pytest.approx(
        30_000.0)
    assert not any("annuity factor" in n for n in with_stated.notes)


def test_frxx_derives_the_instalment_from_the_annuity_factor_when_absent():
    """Still contractual arithmetic — balance, rate and term — with no
    behavioural input. The note must say the instalment was derived, because a
    reader comparing two loans needs to know which one came off the tape."""
    schedule = build_schedule(
        _loan(amortisation_type="FRXX", current_principal_balance=100_000.0,
              maturity_date="2030-01-31", current_interest_rate=10.0,
              scheduled_principal_payment_frequency="YEAR"), as_of=AS_OF)

    assert schedule.status == SUPPORTED
    assert any("annuity factor" in n for n in schedule.notes)
    # £100k at 10% over 4 annual periods: instalment = 100,000 × 0.1 × 1.1^4
    # / (1.1^4 − 1) = 31,547.08, worked by hand.
    instalment = schedule.principal[0] + schedule.interest[0]
    assert instalment == pytest.approx(31_547.08, abs=0.01)
    assert sum(schedule.principal) == pytest.approx(100_000.0)


def test_a_floating_frxx_refuses_outright_because_principal_moves_with_the_rate():
    """The asymmetry against BLLT/FIXE, asserted. Here the contract fixes the
    TOTAL, so an unknown rate makes the *principal* split unknown too — WAL
    falls with YTM rather than surviving it."""
    loan = _loan(amortisation_type="FRXX", maturity_date="2030-01-31",
                 scheduled_principal_payment_frequency="YEAR",
                 interest_rate_type="FLIF")
    schedule = build_schedule(loan, as_of=AS_OF)

    assert schedule.status == ASSUMPTION_REQUIRED
    assert "rate-dependent" in schedule.reason
    assert contractual_wal(schedule).status == ASSUMPTION_REQUIRED
    assert contractual_wal(schedule).wal_years is None


# =========================================================================== #
# DEXX
# =========================================================================== #
def test_dexx_allocates_no_principal_in_the_interest_only_first_instalment():
    """RREL35: German is "amortisation in which the first instalment is
    interest-only and the remaining instalments are constant". £100,000 at 10%
    annually: period 1 pays £10,000 of interest and no principal, and the
    balance is unchanged going into period 2."""
    loan = _loan(amortisation_type="DEXX", current_principal_balance=100_000.0,
                 maturity_date="2030-01-31", current_interest_rate=10.0,
                 scheduled_principal_payment_frequency="YEAR")
    schedule = build_schedule(loan, as_of=AS_OF)

    assert schedule.status == SUPPORTED
    assert schedule.principal[0] == 0.0
    assert schedule.interest[0] == pytest.approx(10_000.0)
    assert schedule.interest[1] == pytest.approx(10_000.0), (
        "the balance is untouched by an interest-only period, so period 2 "
        "accrues on the full £100,000")
    assert sum(schedule.principal) == pytest.approx(100_000.0)


def test_dexx_has_a_longer_wal_than_the_same_loan_amortising_from_period_one():
    """Deferring principal by one period must lengthen the life. If it did not,
    the interest-only phase is not being applied."""
    common = dict(current_principal_balance=100_000.0,
                  maturity_date="2030-01-31", current_interest_rate=10.0,
                  scheduled_principal_payment_frequency="YEAR")
    german = contractual_wal(build_schedule(
        _loan(amortisation_type="DEXX", **common), as_of=AS_OF))
    french = contractual_wal(build_schedule(
        _loan(amortisation_type="FRXX", **common), as_of=AS_OF))

    assert german.status == french.status == SUPPORTED
    assert german.wal_years > french.wal_years


def test_a_floating_dexx_refuses_like_frxx():
    loan = _loan(amortisation_type="DEXX", maturity_date="2030-01-31",
                 scheduled_principal_payment_frequency="YEAR",
                 interest_rate_type="FLFL")
    assert build_schedule(loan, as_of=AS_OF).status == ASSUMPTION_REQUIRED


# =========================================================================== #
# ERM — the case that must never produce a number
# =========================================================================== #
def _erm_loan():
    """A lifetime mortgage as Trakt reports it. `product_defaults_ERM.yaml`
    maps both "Bullet" and "Interest roll-up" to OTHR, on the stated ground
    that repayment at death or sale is not a scheduled bullet."""
    return _loan(amortisation_type="OTHR", maturity_date="2075-01-31",
                 interest_rate_type="FXRL", current_interest_rate=6.5)


def test_erm_has_a_legal_maturity_and_no_contractual_wal():
    """The three things that must not be run together. A legal long-stop of
    2075 exists on the tape; a contractual WAL does not exist at all; and an
    expected WAL would need mortality. Publishing 49 years because the
    long-stop says so would be confidently wrong by decades."""
    loan = _erm_loan()
    assert loan["maturity_date"] == "2075-01-31", "legal maturity is present"

    schedule = build_schedule(loan, as_of=AS_OF)
    assert schedule.status == NOT_APPLICABLE

    wal = contractual_wal(schedule)
    assert wal.status == NOT_APPLICABLE
    assert wal.wal_years is None, "no number, not even a defensible-looking one"
    assert "contingent on death, sale or long-term care" in wal.reason
    assert "legal maturity is not a contractual WAL" in wal.reason


def test_erm_produces_no_ytm_even_though_price_and_maturity_exist():
    """Price (RREL34) and a legal maturity are both present, which is exactly
    the trap: the inputs of a YTM are there and the *meaning* is not, because
    the payment timing itself is contingent."""
    loan = _erm_loan()
    result = contractual_ytm(build_schedule(loan, as_of=AS_OF), loan=loan)
    assert result.status == NOT_APPLICABLE
    assert result.ytm_pct is None


def test_the_erm_refusal_says_why_rather_than_returning_a_bare_null():
    """An agent receiving null cannot tell "not applicable" from "failed", and
    will either drop the metric silently or retry forever. The status and the
    reason are the deliverable here, not the absence of a number."""
    payload = contractual_wal(build_schedule(_erm_loan(), as_of=AS_OF)).to_dict()
    assert payload["metric"] == "contractual_wal"
    assert payload["status"] == NOT_APPLICABLE
    assert payload["reason"]
    assert payload["wal_years"] is None
    assert "mortality" in payload["reason"]


# =========================================================================== #
# Refusals that name the missing field
# =========================================================================== #
def test_an_other_frequency_refuses_rather_than_assuming_monthly():
    loan = _loan(scheduled_principal_payment_frequency="OTHR",
                 scheduled_interest_payment_frequency="OTHR")
    schedule = build_schedule(loan, as_of=AS_OF)
    assert schedule.status == FIELD_GAP
    assert "not assumed to be monthly" in schedule.reason


def test_a_missing_maturity_refuses():
    schedule = build_schedule(_loan(maturity_date=None), as_of=AS_OF)
    assert schedule.status == FIELD_GAP
    assert "RREL24" in schedule.reason


def test_a_missing_amortisation_type_refuses_rather_than_defaulting():
    schedule = build_schedule(_loan(amortisation_type=None), as_of=AS_OF)
    assert schedule.status == FIELD_GAP
    assert "RREL35" in schedule.reason


def test_a_missing_price_does_not_become_par():
    """Assuming par would turn every yield into the coupon and look plausible
    doing it.

    The schedule is built from the SAME priceless loan. An earlier version of
    this test built the schedule from a priced loan and then passed a
    priceless one to the yield, which stopped being a meaningful case once
    Sprint 3 made the schedule carry its own RREL34: the two inputs described
    different loans, and the schedule's price — correctly — won.
    """
    priceless = _loan(purchase_price=None)
    schedule = build_schedule(priceless, as_of=AS_OF)
    assert schedule.price_percent_of_par is None

    result = contractual_ytm(schedule, loan=priceless)
    assert result.status == FIELD_GAP
    assert "not assumed to be par" in result.reason


def test_a_schedule_carries_its_own_price_so_a_caller_need_not_refetch_it():
    """Sprint 3 found the governed tool silently dropping RREL34: the schedule
    did not retain it, so ``contractual_analytics`` could only price a
    portfolio when the caller passed an override — and a 400-loan book whose
    every loan carried a purchase price reported ASSUMPTION_REQUIRED for all
    of them. Only executing the real path showed it."""
    schedule = build_schedule(_loan(purchase_price=98.0), as_of=AS_OF)
    assert schedule.price_percent_of_par == pytest.approx(98.0)

    from_schedule = contractual_ytm(schedule)
    explicit = contractual_ytm(schedule, price_percent_of_par=98.0)
    assert from_schedule.status == SUPPORTED
    assert from_schedule.ytm_pct == pytest.approx(explicit.ytm_pct)


def test_the_ytm_explanation_refuses_to_claim_day_count_exactness():
    """`day_count_convention` is CREL122 — Annex 3 only. The methodology name
    and the explanation both have to say the yield is periodic, or a reader
    will assume the stronger claim."""
    result = contractual_ytm(build_schedule(_loan(), as_of=AS_OF),
                             loan=_loan())
    assert result.method == YTM_METHOD == "CONTRACTUAL_YTM_PERIODIC@v1"
    assert "NOT APPLIED" in result.explanation["day_count"]
    assert "CREL122" in result.explanation["day_count"]


# =========================================================================== #
# Anchoring
# =========================================================================== #
def test_dates_are_counted_back_from_maturity_so_the_balloon_lands_on_it():
    """RREL41 says the balloon is "to be paid at the maturity date". Counting
    forward from origination leaves a short final period and puts it somewhere
    else, so the anchor is stated and asserted."""
    loan = _loan(amortisation_type="FIXE", maturity_date="2030-03-15",
                 scheduled_principal_payment_frequency="YEAR",
                 balloon_amount=400_000.0)
    schedule = build_schedule(loan, as_of=AS_OF)

    assert schedule.anchor_basis == "counted_back_from_maturity"
    assert schedule.dates[-1] == date(2030, 3, 15)
    assert any("stub period is therefore at the front" in n
               for n in schedule.notes)


def test_an_explicit_next_payment_date_is_preferred_where_the_regime_gives_one():
    """CREL104 exists on Annex 3. Where the tape states the next payment date,
    deriving one instead would be substituting arithmetic for a fact."""
    loan = _loan(amortisation_type="FIXE", maturity_date="2030-01-31",
                 scheduled_principal_payment_frequency="YEAR",
                 next_payment_date="2026-06-30")
    schedule = build_schedule(loan, as_of=AS_OF)
    assert schedule.anchor_basis == "explicit_next_payment_date"
    assert schedule.dates[0] == date(2026, 6, 30)


# =========================================================================== #
# Portfolio level
# =========================================================================== #
def test_portfolio_wal_aggregates_cash_flows_rather_than_averaging_wals():
    """Two bullets, £1m at 5 years and £3m at 1 year. Aggregating principal:
    (1m × 5.00274 + 3m × 1.00274) / 4m = 2.00274 years.

    A *balance*-weighted average of the two loan WALs gives the same answer
    here only because a bullet's principal equals its balance. The aggregation
    form is used because it stays correct when they differ — a book with
    balloons is exactly where an average would drift.
    """
    frame = pd.DataFrame([
        _loan(loan_identifier="A", current_principal_balance=1_000_000.0,
              maturity_date="2031-01-31"),
        _loan(loan_identifier="B", current_principal_balance=3_000_000.0,
              maturity_date="2027-01-31"),
    ])
    result = portfolio_wal(schedule_frame(frame, as_of=AS_OF))

    assert result.status == SUPPORTED
    long_years = (date(2031, 1, 31) - AS_OF).days / 365.0
    short_years = (date(2027, 1, 31) - AS_OF).days / 365.0
    expected = (1_000_000 * long_years + 3_000_000 * short_years) / 4_000_000
    assert result.wal_years == pytest.approx(expected, abs=1e-6)
    assert result.explanation["loans_included"] == 2


def test_portfolio_wal_excludes_unsupported_loans_and_says_how_many():
    """A portfolio number that silently dropped the ERM loans would be a
    different portfolio's WAL. The exclusion count is part of the answer."""
    frame = pd.DataFrame([
        _loan(loan_identifier="A"),
        _erm_loan() | {"loan_identifier": "ERM"},
        _loan(loan_identifier="F", scheduled_principal_payment_frequency="OTHR",
              scheduled_interest_payment_frequency="OTHR"),
    ])
    result = portfolio_wal(schedule_frame(frame, as_of=AS_OF))

    assert result.status == SUPPORTED
    assert result.explanation["loans_included"] == 1
    assert result.explanation["excluded_by_status"] == {
        NOT_APPLICABLE: 1, FIELD_GAP: 1}


def test_an_all_erm_portfolio_returns_not_applicable_not_zero():
    frame = pd.DataFrame([_erm_loan() | {"loan_identifier": "E1"},
                          _erm_loan() | {"loan_identifier": "E2"}])
    result = portfolio_wal(schedule_frame(frame, as_of=AS_OF))
    assert result.status == NOT_APPLICABLE
    assert result.wal_years is None


# =========================================================================== #
# Explainability
# =========================================================================== #
def test_a_wal_result_explains_itself_without_an_llm():
    """"Why is contractual WAL 4.2 years?" has to be answerable from the
    payload alone."""
    result = contractual_wal(build_schedule(
        _loan(amortisation_type="FIXE", maturity_date="2030-01-31",
              scheduled_principal_payment_frequency="YEAR",
              balloon_amount=400_000.0), as_of=AS_OF))
    explanation = result.explanation

    for key in ("amortisation_type", "rate_type", "schedule_method",
                "anchor_basis", "payment_frequency_months", "as_of",
                "time_basis", "total_principal", "balloon_treatment"):
        assert explanation.get(key) is not None, f"{key} missing"
    assert result.method == WAL_METHOD
    assert explanation["time_basis"] == "ACT/365F"
    assert "RREL41" in explanation["balloon_treatment"]


def test_a_ytm_result_explains_its_price_and_frequency_basis():
    result = contractual_ytm(build_schedule(_loan(), as_of=AS_OF),
                             loan=_loan(purchase_price=98.0))
    explanation = result.explanation

    for key in ("price_basis", "interest_rate_basis", "date_frequency_basis",
                "irr_method", "day_count", "schedule_method", "periods"):
        assert explanation.get(key), f"{key} missing"
    assert "RREL34" in explanation["price_basis"]
    assert "RREL43" in explanation["interest_rate_basis"]


# =========================================================================== #
# The SMM boundary — §19 of the brief
# =========================================================================== #
def test_this_module_never_touches_the_smm_denominator():
    """RREL39 is principal PLUS interest. It is used here as the constant
    instalment of an annuity, which is exactly what it is — and it must never
    reach the SMM denominator, which needs scheduled principal alone. The two
    modules are asserted to stay apart at the import level."""
    import analytics_lib.contractual as contractual
    import analytics_lib.history as history

    assert history.SCHEDULED_PRINCIPAL_FIELD == "regular_principal_instalment"
    assert history.SCHEDULED_PRINCIPAL_FIELD != contractual.PAYMENT_DUE_FIELD
    source = Path(history.__file__).read_text(encoding="utf-8")
    assert "payment_due" not in source, (
        "history.py must not reference RREL39: using a principal-plus-interest "
        "figure as scheduled principal would inflate the SMM deduction and "
        "overstate every published prepayment rate")
    assert "analytics_lib.contractual" not in source, (
        "history.py must not import the contractual stack: OBSERVED and "
        "CONTRACTUAL are separate vocabularies and a schedule must never "
        "silently change a measured prepayment rate")
