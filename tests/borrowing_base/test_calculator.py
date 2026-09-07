"""The borrowing-base arithmetic, against figures worked out by hand.

Every expected number in this file is a literal or an expression written out in
the test. Nothing calls the calculator to establish what the calculator should
have said.
"""

from __future__ import annotations

import pandas as pd
import pytest

from mi_agent.borrowing_base import calculator as calc
from mi_agent.borrowing_base.eligibility import derive_eligibility
from mi_agent.borrowing_base.models import (
    ELIGIBLE,
    INELIGIBLE,
    NOT_CALCULABLE,
    UNDETERMINED,
    EligibilityRule,
    FacilityConfiguration,
)

from .conftest import (
    ADVANCE_RATE,
    COMMITMENT,
    DENOMINATOR_FLOOR,
    facility,
    frame,
    production_facility,
)


def book(*balances: float) -> pd.DataFrame:
    return frame([{"loan_id": f"L{i}", "current_outstanding_balance": b}
                  for i, b in enumerate(balances)])


def result(df: pd.DataFrame, fac: FacilityConfiguration, **kw):
    derive_eligibility(df, fac)
    return calc.calculate(df, fac, **kw)


# --------------------------------------------------------------------------- #
# The Concentration Limit Denominator
# --------------------------------------------------------------------------- #
class TestConcentrationLimitDenominator:
    """MAX(£33,000,000, Current Balance of all Eligible Mortgage Loans)."""

    def test_the_floor_binds_below_33m(self):
        r = result(book(20_000_000.0), facility())
        assert r.eligibility.eligible_current_balance == 20_000_000.0
        assert r.concentration_limit_denominator == 33_000_000.0
        assert r.concentration_denominator_floor_binding is True

    def test_the_eligible_balance_binds_above_33m(self):
        r = result(book(40_000_000.0), facility())
        assert r.concentration_limit_denominator == 40_000_000.0
        assert r.concentration_denominator_floor_binding is False

    def test_exactly_at_the_floor_neither_side_wins_and_the_answer_is_33m(self):
        r = result(book(33_000_000.0), facility())
        assert r.concentration_limit_denominator == 33_000_000.0
        # 33m is not GREATER than 33m, so the floor is not what bound.
        assert r.concentration_denominator_floor_binding is False

    def test_a_penny_above_the_floor_uses_the_balance(self):
        r = result(book(33_000_000.01), facility())
        assert r.concentration_limit_denominator == 33_000_000.01

    def test_an_empty_book_still_yields_the_floor_not_zero(self):
        r = result(book(), facility())
        assert r.eligibility.eligible_current_balance == 0.0
        assert r.concentration_limit_denominator == DENOMINATOR_FLOOR

    def test_no_configured_floor_is_not_calculable_rather_than_unfloored(self):
        r = result(book(20_000_000.0),
                   facility(concentration_denominator_floor=None))
        assert r.concentration_limit_denominator == NOT_CALCULABLE
        assert "concentration_denominator_floor" in r.missing_inputs


# --------------------------------------------------------------------------- #
# The advance rate
# --------------------------------------------------------------------------- #
class TestAdvanceRate:
    """gross borrowing base = eligible current balance x 103%."""

    def test_100m_of_eligible_collateral_produces_103m(self):
        r = result(book(100_000_000.0), facility())
        assert r.gross_borrowing_base == 103_000_000.0

    def test_the_rate_is_applied_to_the_ELIGIBLE_balance_only(self):
        # £60m eligible + £40m ineligible. 60m x 1.03 = 61.8m, not 103m.
        df = book(60_000_000.0, 40_000_000.0)
        fac = production_facility([EligibilityRule(
            rule_id="max_ltv", field="original_loan_to_value",
            operator="max", value=50.0, reason_code="ltv_too_high")])
        df["original_loan_to_value"] = [40.0, 90.0]
        derive_eligibility(df, fac)
        r = calc.calculate(df, fac)
        assert r.eligibility.eligible_current_balance == 60_000_000.0
        assert r.eligibility.ineligible_current_balance == 40_000_000.0
        assert r.gross_borrowing_base == 61_800_000.0

    def test_a_zero_book_gives_a_zero_base_which_is_a_real_zero(self):
        r = result(book(0.0), facility())
        assert r.gross_borrowing_base == 0.0

    def test_no_configured_advance_rate_is_not_calculable(self):
        r = result(book(100_000_000.0), facility(advance_rate=None))
        assert r.gross_borrowing_base == NOT_CALCULABLE
        assert r.available_borrowing_base == NOT_CALCULABLE
        assert "advance_rate" in r.missing_inputs


# --------------------------------------------------------------------------- #
# The facility cap
# --------------------------------------------------------------------------- #
class TestFacilityCap:
    """available borrowing base = MIN(gross borrowing base, £250,000,000)."""

    def test_below_the_cap_the_gross_base_stands(self):
        r = result(book(100_000_000.0), facility())
        assert r.available_borrowing_base == 103_000_000.0
        assert r.facility_cap_binding is False

    def test_above_the_cap_the_commitment_binds(self):
        # £300m x 1.03 = £309m gross, capped to the £250m commitment.
        r = result(book(300_000_000.0), facility())
        assert r.gross_borrowing_base == 309_000_000.0
        assert r.available_borrowing_base == 250_000_000.0
        assert r.facility_cap_binding is True

    def test_exactly_at_the_cap_the_cap_does_not_bind(self):
        # £250m / 1.03 to the penny is not round, so use the exact inverse:
        # an eligible balance whose gross base is exactly the commitment.
        eligible = COMMITMENT / ADVANCE_RATE
        r = result(book(eligible), facility())
        assert r.available_borrowing_base == pytest.approx(COMMITMENT, abs=0.01)
        assert r.facility_cap_binding is False

    def test_no_commitment_shows_the_base_uncapped_and_says_so(self):
        r = result(book(300_000_000.0), facility(commitment=None))
        assert r.available_borrowing_base == 309_000_000.0
        assert r.facility_cap_binding is None
        assert "facility_commitment" in r.missing_inputs


# --------------------------------------------------------------------------- #
# Headroom, deficiency, utilisation
# --------------------------------------------------------------------------- #
class TestHeadroom:
    """headroom = available borrowing base - current drawn amount."""

    def test_borrowing_base_above_drawn_gives_positive_headroom(self):
        # £100m eligible -> £103m base; £80m drawn -> £23m headroom.
        r = result(book(100_000_000.0),
                   facility(current_drawn_amount=80_000_000.0))
        assert r.available_borrowing_base == 103_000_000.0
        assert r.borrowing_base_headroom == 23_000_000.0
        assert r.borrowing_base_deficiency == 0.0

    def test_borrowing_base_equal_to_drawn_gives_exactly_zero_headroom(self):
        r = result(book(100_000_000.0),
                   facility(current_drawn_amount=103_000_000.0))
        assert r.borrowing_base_headroom == 0.0
        assert r.borrowing_base_deficiency == 0.0
        assert r.borrowing_base_utilisation_pct == 100.0

    def test_borrowing_base_below_drawn_keeps_the_headroom_NEGATIVE(self):
        # £103m base, £120m drawn. The governed headroom is -£17m; it is NOT
        # floored to zero, and the deficiency is its absolute value.
        r = result(book(100_000_000.0),
                   facility(current_drawn_amount=120_000_000.0))
        assert r.borrowing_base_headroom == -17_000_000.0
        assert r.borrowing_base_deficiency == 17_000_000.0

    def test_a_missing_drawn_amount_is_not_calculable_never_zero(self):
        r = result(book(100_000_000.0), facility(current_drawn_amount=None))
        assert r.available_borrowing_base == 103_000_000.0   # unaffected
        assert r.borrowing_base_headroom == NOT_CALCULABLE
        assert r.borrowing_base_deficiency == NOT_CALCULABLE
        assert r.borrowing_base_utilisation_pct == NOT_CALCULABLE
        assert r.facility_utilisation_pct == NOT_CALCULABLE
        assert r.current_drawn_amount == NOT_CALCULABLE
        assert r.missing_inputs == ["current_drawn_amount"]

    def test_a_zero_drawn_amount_is_a_real_zero_not_a_missing_input(self):
        r = result(book(100_000_000.0), facility(current_drawn_amount=0.0))
        assert r.current_drawn_amount == 0.0
        assert r.borrowing_base_headroom == 103_000_000.0
        assert "current_drawn_amount" not in r.missing_inputs


class TestUtilisation:
    def test_borrowing_base_utilisation_is_drawn_over_the_available_base(self):
        # £80m drawn / £103m base = 77.6699...%
        r = result(book(100_000_000.0),
                   facility(current_drawn_amount=80_000_000.0))
        assert r.borrowing_base_utilisation_pct == pytest.approx(
            80_000_000.0 / 103_000_000.0 * 100.0, abs=1e-4)

    def test_facility_utilisation_is_drawn_over_the_commitment(self):
        # £80m drawn / £250m commitment = 32%.
        r = result(book(100_000_000.0),
                   facility(current_drawn_amount=80_000_000.0))
        assert r.facility_utilisation_pct == 32.0

    def test_the_two_utilisations_differ_and_both_are_reported(self):
        r = result(book(100_000_000.0),
                   facility(current_drawn_amount=80_000_000.0))
        assert r.borrowing_base_utilisation_pct != r.facility_utilisation_pct

    def test_utilisation_against_a_zero_base_is_not_calculable(self):
        r = result(book(0.0), facility(current_drawn_amount=1_000_000.0))
        assert r.available_borrowing_base == 0.0
        assert r.borrowing_base_utilisation_pct == NOT_CALCULABLE


# --------------------------------------------------------------------------- #
# Reconciliation invariants
# --------------------------------------------------------------------------- #
class TestInvariants:
    def test_the_three_statuses_partition_the_financing_portfolio(self):
        df = book(10_000_000.0, 20_000_000.0, 30_000_000.0)
        df["original_loan_to_value"] = [40.0, 90.0, None]
        fac = production_facility([EligibilityRule(
            rule_id="max_ltv", field="original_loan_to_value",
            operator="max", value=50.0)])
        derive_eligibility(df, fac)
        r = calc.calculate(df, fac)
        s = r.eligibility
        assert s.eligible_current_balance == 10_000_000.0
        assert s.ineligible_current_balance == 20_000_000.0
        assert s.undetermined_current_balance == 30_000_000.0
        assert s.financing_portfolio_balance == 60_000_000.0
        assert (s.eligible_loan_count + s.ineligible_loan_count
                + s.undetermined_loan_count) == s.financing_portfolio_loan_count == 3
        assert r.reconciles

    def test_every_invariant_states_the_numbers_that_prove_it(self):
        r = result(book(10_000_000.0), facility())
        for invariant in r.eligibility.invariants:
            assert invariant["holds"] is True
            assert {"invariant", "statement", "expected", "actual"} <= set(invariant)

    def test_no_loan_can_hold_two_statuses(self):
        r = result(book(1_000_000.0, 2_000_000.0), facility())
        overlap = next(i for i in r.eligibility.invariants
                       if i["invariant"] == "no_loan_holds_two_eligibility_statuses")
        assert overlap["actual"] == 0

    def test_a_broken_population_FAILS_CLOSED_rather_than_reporting_a_base(self):
        # A frame whose status column has been corrupted so a loan belongs to
        # the facility but holds no status: the partition no longer holds.
        df = book(10_000_000.0, 20_000_000.0)
        fac = facility()
        derive_eligibility(df, fac)
        df.loc[0, "borrowing_base_eligibility_status"] = "SOMETHING_ELSE"
        with pytest.raises(calc.ReconciliationError):
            calc.calculate(df, fac)

    def test_a_diagnostic_caller_can_see_the_failure_instead_of_being_stopped(self):
        df = book(10_000_000.0, 20_000_000.0)
        fac = facility()
        derive_eligibility(df, fac)
        df.loc[0, "borrowing_base_eligibility_status"] = "SOMETHING_ELSE"
        r = calc.calculate(df, fac, strict=False)
        assert r.reconciles is False
        assert any(not i["holds"] for i in r.eligibility.invariants)


# --------------------------------------------------------------------------- #
# The concentration-breach seam
# --------------------------------------------------------------------------- #
BREACHED = [
    {"testId": "t_region_east", "displayName": "East of England",
     "status": "breach", "currentValue": 30.0, "threshold": 25.0,
     "unit": "percent", "utilization": 120.0, "breachAmount": 5.0,
     "headroom": -5.0, "denominatorValue": 100_000_000.0},
    {"testId": "t_region_wales", "displayName": "Wales", "status": "breach",
     "currentValue": 11.0, "threshold": 10.0, "unit": "percent",
     "utilization": 110.0, "breachAmount": 1.0, "headroom": -1.0,
     "denominatorValue": 100_000_000.0},
    {"testId": "t_region_scotland", "displayName": "Scotland", "status": "pass",
     "currentValue": 9.0, "threshold": 10.0, "unit": "percent",
     "utilization": 90.0, "breachAmount": None, "headroom": 1.0,
     "denominatorValue": 100_000_000.0},
]


class TestConcentrationTreatment:
    def test_a_breach_deducts_NOTHING_from_the_borrowing_base(self):
        clean = result(book(100_000_000.0), facility())
        breached = result(book(100_000_000.0), facility(),
                          concentration_results=BREACHED)
        assert breached.available_borrowing_base == clean.available_borrowing_base
        assert breached.concentration_adjustment["amount"] == 0.0

    def test_the_excess_is_still_measured_and_reported(self):
        r = result(book(100_000_000.0), facility(),
                   concentration_results=BREACHED)
        assert r.concentration_adjustment["breachedTestCount"] == 2
        # 5.0pp + 1.0pp of excess, measured but not deducted.
        assert r.concentration_adjustment["excessMeasuredPct"] == 6.0

    def test_an_unimplemented_treatment_REFUSES_rather_than_guessing(self):
        with pytest.raises(calc.ReconciliationError):
            result(book(100_000_000.0),
                   facility(borrowing_base_treatment="exclude_excess"),
                   concentration_results=BREACHED)


class TestNearestConcentration:
    def test_the_largest_utilisation_binds_when_limits_are_breached(self):
        out = calc.nearest_concentration(BREACHED)
        assert out["nearest_concentration_limit"] == "East of England"
        assert out["nearest_concentration_utilisation_pct"] == 120.0
        assert out["breached_concentration_count"] == 2

    def test_every_breach_is_retained_ordered_most_severe_first(self):
        out = calc.nearest_concentration(BREACHED)
        assert [b["displayName"] for b in out["breached_concentrations"]] == [
            "East of England", "Wales"]

    def test_the_smallest_headroom_binds_when_nothing_is_breached(self):
        passing = [t for t in BREACHED if t["status"] == "pass"] + [
            {"testId": "t_wide", "displayName": "Wide", "status": "pass",
             "currentValue": 1.0, "threshold": 25.0, "unit": "percent",
             "utilization": 4.0, "breachAmount": None, "headroom": 24.0,
             "denominatorValue": 100_000_000.0}]
        out = calc.nearest_concentration(passing)
        assert out["nearest_concentration_limit"] == "Scotland"
        assert out["nearest_concentration_headroom_pct"] == 1.0

    def test_headroom_converts_to_currency_at_the_tests_own_denominator(self):
        # 1.0pp of a £100,000,000 denominator is £1,000,000.
        passing = [t for t in BREACHED if t["status"] == "pass"]
        out = calc.nearest_concentration(passing)
        assert out["nearest_concentration_headroom_amount"] == 1_000_000.0

    def test_a_test_with_no_denominator_has_no_currency_headroom(self):
        out = calc.nearest_concentration([
            {"testId": "t_avg", "displayName": "Average balance",
             "status": "pass", "currentValue": 250_000.0,
             "threshold": 300_000.0, "unit": "currency", "utilization": 83.3,
             "headroom": 50_000.0, "breachAmount": None,
             "denominatorValue": None}])
        assert out["nearest_concentration_headroom_amount"] is None

    def test_with_no_results_at_all_the_answer_is_not_calculable(self):
        out = calc.nearest_concentration([])
        assert out["nearest_concentration_limit"] == NOT_CALCULABLE
        assert out["breached_concentration_count"] == 0

    def test_ties_break_deterministically_on_test_id(self):
        tied = [
            {"testId": "t_b", "displayName": "B", "status": "breach",
             "currentValue": 11.0, "threshold": 10.0, "unit": "percent",
             "utilization": 110.0, "breachAmount": 1.0, "headroom": -1.0,
             "denominatorValue": 100.0},
            {"testId": "t_a", "displayName": "A", "status": "breach",
             "currentValue": 11.0, "threshold": 10.0, "unit": "percent",
             "utilization": 110.0, "breachAmount": 1.0, "headroom": -1.0,
             "denominatorValue": 100.0},
        ]
        assert calc.nearest_concentration(tied)["nearest_concentration_limit"] == "A"
        assert calc.nearest_concentration(list(reversed(tied)))[
            "nearest_concentration_limit"] == "A"


class TestThePrototypeAssumptionIsAlwaysDeclared:
    """It must reach the envelope from the FACILITY, not from a receipt.

    A disclosure that depends on some caller remembering to pass a derivation
    receipt is a disclosure that will eventually go missing — and this is the
    one that must not, because every figure beside it rests on it.
    """

    def test_the_calculation_declares_it_with_no_receipt_plumbed_in(self):
        r = result(book(100_000_000.0), facility())
        assert r.prototype_assumptions_used
        assert "PROTOTYPE ASSUMPTION" in r.prototype_assumptions_used[0]

    def test_it_reaches_the_serialised_envelope(self):
        r = result(book(100_000_000.0), facility())
        assert r.to_dict()["prototypeAssumptionsUsed"]

    def test_a_governed_facility_declares_nothing(self):
        fac = production_facility([EligibilityRule(
            rule_id="max_ltv", field="original_loan_to_value",
            operator="max", value=50.0)])
        df = book(100_000_000.0)
        df["original_loan_to_value"] = [40.0]
        derive_eligibility(df, fac)
        assert calc.calculate(df, fac).prototype_assumptions_used == []

    def test_the_derivation_and_the_calculation_use_the_SAME_words(self):
        from mi_agent.borrowing_base.models import PROTOTYPE_ASSUMPTION_NOTE
        df = book(100_000_000.0)
        derivation = derive_eligibility(df, facility())
        r = calc.calculate(df, facility())
        assert derivation["prototype_assumptions_used"] == [PROTOTYPE_ASSUMPTION_NOTE]
        assert r.prototype_assumptions_used == [PROTOTYPE_ASSUMPTION_NOTE]
