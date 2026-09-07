"""Governed loan-level eligibility: the tri-state, and what it refuses to do.

The rule this whole module exists to enforce: an Eligible Mortgage Loan is a
CONTRACTUAL determination. With no approved criteria, a production facility
reports UNDETERMINED for every loan — it does not report ELIGIBLE, and it does
not infer eligibility from whether the portfolio passes its concentration
tests.
"""

from __future__ import annotations

import pandas as pd
import pytest

from mi_agent.borrowing_base.eligibility import (
    derive_eligibility,
    eligibility_available,
    eligible_mask,
    financing_portfolio_mask,
    in_financing_portfolio_mask,
    status_mask,
)
from mi_agent.borrowing_base.models import (
    ELIGIBLE,
    FIELD_ELIGIBILITY_REASON,
    FIELD_ELIGIBILITY_STATUS,
    FIELD_ELIGIBLE,
    FIELD_FACILITY_ID,
    INELIGIBLE,
    REASON_NO_APPROVED_RULES,
    REASON_OUTSIDE_FINANCING_PORTFOLIO,
    REASON_PROTOTYPE_ASSUMPTION,
    REASON_RULES_SATISFIED,
    UNDETERMINED,
    EligibilityRule,
)

from .conftest import facility, frame, production_facility


def book(n: int = 3, **cols) -> pd.DataFrame:
    df = frame([{"loan_id": f"L{i}"} for i in range(n)])
    for name, values in cols.items():
        df[name] = values
    return df


def statuses(df: pd.DataFrame) -> list:
    return list(df[FIELD_ELIGIBILITY_STATUS])


# --------------------------------------------------------------------------- #
# The prototype assumption
# --------------------------------------------------------------------------- #
class TestPrototypeAssumption:
    def test_enabled_every_financing_portfolio_loan_is_eligible(self):
        df = book(3)
        receipt = derive_eligibility(df, facility())
        assert statuses(df) == [ELIGIBLE] * 3
        assert list(df[FIELD_ELIGIBILITY_REASON]) == [
            REASON_PROTOTYPE_ASSUMPTION] * 3
        assert receipt["status_counts"] == {ELIGIBLE: 3, INELIGIBLE: 0,
                                            UNDETERMINED: 0}

    def test_enabled_the_receipt_says_it_is_an_assumption_not_a_determination(self):
        df = book(2)
        receipt = derive_eligibility(df, facility())
        [note] = receipt["prototype_assumptions_used"]
        assert "PROTOTYPE ASSUMPTION" in note
        assert "not a contractual eligibility determination" in note

    def test_disabled_every_loan_is_undetermined_with_the_reason_named(self):
        df = book(3)
        receipt = derive_eligibility(
            df, facility(prototype_assume_financing_portfolio_eligible=False))
        assert statuses(df) == [UNDETERMINED] * 3
        assert set(df[FIELD_ELIGIBILITY_REASON]) == {REASON_NO_APPROVED_RULES}
        assert receipt["prototype_assumptions_used"] == []

    def test_it_is_IGNORED_outside_a_prototype_environment(self):
        # The same switch, on a production facility. Configuring it is not
        # enough: it can never become the production default by a copy-paste.
        fac = production_facility()
        fac.prototype_assume_financing_portfolio_eligible = True
        assert fac.prototype_assumption_active is False
        df = book(3)
        derive_eligibility(df, fac)
        assert statuses(df) == [UNDETERMINED] * 3

    def test_configuring_it_on_a_production_facility_is_flagged_as_a_problem(self):
        fac = production_facility()
        fac.prototype_assume_financing_portfolio_eligible = True
        [problem] = fac.validate()
        assert "will NOT be honoured" in problem


class TestProductionFailsClosed:
    def test_a_production_facility_with_no_rules_never_reports_eligible(self):
        df = book(5)
        derive_eligibility(df, production_facility())
        assert statuses(df) == [UNDETERMINED] * 5
        assert eligible_mask(df).sum() == 0

    def test_passing_every_concentration_test_does_not_make_a_loan_eligible(self):
        # A perfectly diversified, perfectly compliant book. Eligibility is
        # still unknown, because Schedule 8 does not define it.
        df = frame([
            {"loan_id": "L0", "collateral_geography": "Scotland"},
            {"loan_id": "L1", "collateral_geography": "Wales"},
            {"loan_id": "L2", "collateral_geography": "South West"},
        ])
        derive_eligibility(df, production_facility())
        assert statuses(df) == [UNDETERMINED] * 3


# --------------------------------------------------------------------------- #
# Approved contractual rules
# --------------------------------------------------------------------------- #
LTV_RULE = EligibilityRule(
    rule_id="max_original_ltv", description="Original LTV must not exceed 50%",
    field="original_loan_to_value", operator="max", value=50.0,
    reason_code="original_ltv_above_facility_limit")

ARREARS_RULE = EligibilityRule(
    rule_id="no_arrears", description="No more than 30 days past due",
    field="days_past_due", operator="max", value=30.0,
    reason_code="in_arrears")


class TestApprovedRules:
    def test_a_loan_meeting_every_rule_is_eligible(self):
        df = book(1, original_loan_to_value=[40.0])
        derive_eligibility(df, production_facility([LTV_RULE]))
        assert statuses(df) == [ELIGIBLE]
        assert list(df[FIELD_ELIGIBILITY_REASON]) == [REASON_RULES_SATISFIED]

    def test_a_loan_breaching_a_rule_is_ineligible_with_that_rules_reason(self):
        df = book(1, original_loan_to_value=[90.0])
        derive_eligibility(df, production_facility([LTV_RULE]))
        assert statuses(df) == [INELIGIBLE]
        assert list(df[FIELD_ELIGIBILITY_REASON]) == [
            "original_ltv_above_facility_limit"]

    def test_exactly_at_the_threshold_passes_a_max_rule(self):
        df = book(1, original_loan_to_value=[50.0])
        derive_eligibility(df, production_facility([LTV_RULE]))
        assert statuses(df) == [ELIGIBLE]

    def test_a_missing_input_makes_the_loan_UNDETERMINED_not_ineligible(self):
        df = book(1, original_loan_to_value=[None])
        derive_eligibility(df, production_facility([LTV_RULE]))
        assert statuses(df) == [UNDETERMINED]
        assert df[FIELD_ELIGIBILITY_REASON][0].startswith(
            "eligibility_rule_input_missing")

    def test_a_column_the_book_does_not_carry_makes_every_loan_undetermined(self):
        df = book(3)  # no original_loan_to_value column at all
        receipt = derive_eligibility(df, production_facility([LTV_RULE]))
        assert statuses(df) == [UNDETERMINED] * 3
        assert receipt["rules_applied"][0]["input_available"] is False

    def test_rules_are_conjunctive_one_failure_is_enough(self):
        df = book(1, original_loan_to_value=[40.0], days_past_due=[90.0])
        derive_eligibility(df, production_facility([LTV_RULE, ARREARS_RULE]))
        assert statuses(df) == [INELIGIBLE]
        assert list(df[FIELD_ELIGIBILITY_REASON]) == ["in_arrears"]

    def test_a_definite_failure_beats_a_missing_input(self):
        # Breaches the LTV rule; the arrears input is absent. A loan that
        # breaches an approved criterion is ineligible whatever else is unknown.
        df = book(1, original_loan_to_value=[90.0], days_past_due=[None])
        derive_eligibility(df, production_facility([LTV_RULE, ARREARS_RULE]))
        assert statuses(df) == [INELIGIBLE]

    @pytest.mark.parametrize("operator,value,cell,expected", [
        ("min", 100_000.0, 150_000.0, ELIGIBLE),
        ("min", 100_000.0, 50_000.0, INELIGIBLE),
        ("equals", "GBP", "GBP", ELIGIBLE),
        ("equals", "GBP", "EUR", INELIGIBLE),
        ("not_equals", "DEFAULT", "PERFORMING", ELIGIBLE),
        ("not_equals", "DEFAULT", "DEFAULT", INELIGIBLE),
        ("in", ["ENGLAND", "WALES"], "wales", ELIGIBLE),
        ("in", ["ENGLAND", "WALES"], "SCOTLAND", INELIGIBLE),
        ("not_in", ["DEFAULT"], "PERFORMING", ELIGIBLE),
        ("not_in", ["DEFAULT"], "default", INELIGIBLE),
        ("present", None, "anything", ELIGIBLE),
    ])
    def test_every_supported_operator(self, operator, value, cell, expected):
        rule = EligibilityRule(rule_id="r", field="probe", operator=operator,
                               value=value)
        df = book(1, probe=[cell])
        derive_eligibility(df, production_facility([rule]))
        assert statuses(df) == [expected]

    def test_an_invalid_rule_is_reported_not_silently_skipped(self):
        bad = EligibilityRule(rule_id="r", field="probe", operator="nonsense")
        problems = production_facility([bad]).validate()
        assert any("unknown operator" in p for p in problems)


# --------------------------------------------------------------------------- #
# The Financing Portfolio boundary
# --------------------------------------------------------------------------- #
class TestFinancingPortfolioScope:
    def test_an_empty_selector_means_the_whole_book(self):
        df = book(3)
        assert financing_portfolio_mask(df, facility()).all()

    def test_a_selector_narrows_to_the_named_source_portfolios(self):
        df = book(3)
        df["source_portfolio_id"] = ["direct_001", "acquired_002", "direct_001"]
        fac = facility(financing_portfolio={"source_portfolio_ids": ["direct_001"]})
        derive_eligibility(df, fac)
        attributed = [None if pd.isna(v) else v for v in df[FIELD_FACILITY_ID]]
        assert attributed == [fac.facility_id, None, fac.facility_id]
        assert [None if pd.isna(v) else v for v in df[FIELD_ELIGIBILITY_STATUS]] \
            == [ELIGIBLE, None, ELIGIBLE]
        assert in_financing_portfolio_mask(df, fac).sum() == 2

    def test_a_loan_outside_the_portfolio_says_why_it_has_no_status(self):
        df = book(2)
        df["source_portfolio_id"] = ["direct_001", "acquired_002"]
        fac = facility(financing_portfolio={"source_portfolio_ids": ["direct_001"]})
        derive_eligibility(df, fac)
        assert df[FIELD_ELIGIBILITY_REASON][1] == REASON_OUTSIDE_FINANCING_PORTFOLIO

    def test_an_UNENFORCEABLE_narrowing_selects_nothing_rather_than_everything(self):
        # The facility restricts to one source portfolio; the frame carries no
        # source-portfolio column at all. Selecting the whole book would be a
        # fail-open on the exact narrowing that was asked for.
        df = book(3)
        fac = facility(financing_portfolio={"source_portfolio_ids": ["direct_001"]})
        assert financing_portfolio_mask(df, fac).sum() == 0


# --------------------------------------------------------------------------- #
# The canonical columns
# --------------------------------------------------------------------------- #
class TestCanonicalColumns:
    def test_all_four_governed_columns_are_produced(self):
        df = book(2)
        receipt = derive_eligibility(df, facility())
        for column in (FIELD_ELIGIBLE, FIELD_ELIGIBILITY_STATUS,
                       FIELD_ELIGIBILITY_REASON, FIELD_FACILITY_ID):
            assert column in df.columns
        assert set(receipt["derived_fields"]) == {
            FIELD_ELIGIBLE, FIELD_ELIGIBILITY_STATUS,
            FIELD_ELIGIBILITY_REASON, FIELD_FACILITY_ID}

    def test_the_boolean_flag_is_NULL_for_undetermined_not_false(self):
        # Flattening a tri-state into a boolean turns "we cannot tell" into
        # "no". The flag stays null so a caller cannot make that mistake.
        df = book(3, original_loan_to_value=[40.0, 90.0, None])
        derive_eligibility(df, production_facility([LTV_RULE]))
        flags = df[FIELD_ELIGIBLE]
        assert flags[0] is True or flags[0] == True   # noqa: E712
        assert flags[1] == False                       # noqa: E712
        assert pd.isna(flags[2])

    def test_the_eligible_mask_reads_the_STATUS_not_the_flag(self):
        df = book(2, original_loan_to_value=[40.0, None])
        derive_eligibility(df, production_facility([LTV_RULE]))
        assert list(eligible_mask(df)) == [True, False]
        assert list(status_mask(df, UNDETERMINED)) == [False, True]

    def test_the_mask_is_scoped_so_two_facilities_cannot_bleed_together(self):
        df = book(2)
        derive_eligibility(df, facility())
        other = facility(facility_id="OTHER_FACILITY")
        assert eligible_mask(df, facility()).sum() == 2
        assert eligible_mask(df, other).sum() == 0

    def test_a_frame_the_derivation_never_ran_on_reports_no_eligibility(self):
        assert eligibility_available(book(2)) is False

    def test_the_receipt_carries_the_configuration_it_was_derived_under(self):
        fac = facility()
        receipt = derive_eligibility(book(2), fac)
        assert receipt["facility_id"] == fac.facility_id
        assert receipt["eligibility_rule_version"] == fac.eligibility_rule_version
        assert receipt["config_hash"] == fac.content_hash()
        assert receipt["eligibility_governed"] is False
