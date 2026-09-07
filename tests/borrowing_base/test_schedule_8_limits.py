"""Every supplied Schedule 8 limit, at its boundary, on the eligible population.

The whole schedule is activated through the PRODUCTION path — deterministic
extraction, operator approval, immutable activation, governed evaluation — and
then measured against a synthetic book whose answers are readable by eye:
£100,000,000 of eligible collateral, so a £15,000,000 slice is 15.00% of the
Concentration Limit Denominator and nothing needs working out.

Each limit is tested immediately below, exactly at, and immediately above its
threshold. "Exactly at" must PASS: every clause reads "does not exceed" or
"shall not exceed", so the limit itself is inside the limit.

Expected values are literals. Nothing here asks the evaluator what the answer
should be and then asserts that it said so.
"""

from __future__ import annotations

from typing import Any, Dict, List

import pandas as pd
import pytest

from mi_agent.concentration_tests.evaluation import evaluate_active_tests
from mi_agent.concentration_tests.matching import build_proposals
from mi_agent.concentration_tests.models import (
    ActiveConfiguration,
    ActiveTest,
    ApprovalRecord,
    MATCH_AMBIGUOUS,
    STATUS_BREACH,
    STATUS_PASS,
    STATUS_UNAVAILABLE,
    STATUS_WARNING,
)

from .conftest import (
    PORTFOLIO_TOTAL,
    UNLIMITED_REGION,
    facility,
    frame,
    two_loan_book,
)

#: The Schedule 8 regional table, verbatim: (region label, limit %).
REGION_LIMITS = [
    ("London", 50.0), ("South East", 50.0),      # UKI + UKJ, ONE combined test
    ("North East", 10.0),                        # UKC
    ("North West", 10.0),                        # UKD
    ("Yorkshire And The Humber", 15.0),          # UKE
    ("East Midlands", 15.0),                     # UKF
    ("West Midlands", 15.0),                     # UKG
    ("East Of England", 25.0),                   # UKH
    ("South West", 20.0),                        # UKK
    ("Scotland", 10.0),                          # UKM
    ("Wales", 10.0),                             # UKL
]


@pytest.fixture(scope="module")
def config(schedule_8_text, lib) -> ActiveConfiguration:
    """The whole schedule, approved and activated exactly as OCC would.

    Only proposals with no outstanding concern are approved — which leaves Net
    WAC out, because the schedule defines it against a Series Fixed Rate that
    lives elsewhere in the facility. That exclusion is the point of
    :class:`TestPortfolioNetWac` below, not an oversight here.
    """
    proposals = build_proposals(lib=lib, client_id="test_client",
                                source_reference="Warehouse Facility Schedule 8",
                                text=schedule_8_text)
    tests: List[ActiveTest] = []
    for p in proposals:
        if p.concern_codes:
            continue
        metric = lib.get(p.proposed_metric_id)
        tests.append(ActiveTest(
            test_id=f"ct_{len(tests):02d}",
            metric_id=p.proposed_metric_id,
            display_name=metric.display_name if metric else p.extracted_test_name,
            category=metric.category if metric else "",
            operator=p.extracted_operator,
            threshold=p.extracted_threshold,
            unit=p.extracted_unit,
            parameters=dict(p.parameters),
            population=p.population,
            approval=ApprovalRecord(decision="approved", operator="test"),
        ))
    return ActiveConfiguration(client_id="test_client", tests=tests,
                               library_version=lib.library_version)


def evaluate(df: pd.DataFrame, config: ActiveConfiguration, lib
             ) -> Dict[str, Dict[str, Any]]:
    """Evaluate the schedule over the ELIGIBLE population of ``df``.

    The eligible frame is what a Schedule 8 clause means by "Eligible Mortgage
    Loans", and the whole schedule declares that population, so it is supplied
    here exactly as the API supplies it.
    """
    from mi_agent.borrowing_base.eligibility import derive_eligibility, eligible_mask
    fac = facility()
    derive_eligibility(df, fac)
    eligible = df[eligible_mask(df, fac)]
    out = evaluate_active_tests(
        df, None, config, lib, reporting_date="2025-11-30",
        populations={"eligible_mortgage_loans": eligible})
    return {t["testId"]: t for t in out["tests"]}


def one(results: Dict[str, Dict[str, Any]], metric_id: str,
        **params: Any) -> Dict[str, Any]:
    """The single result for a metric, optionally narrowed by parameters."""
    hits = [t for t in results.values() if t["metricId"] == metric_id
            and all(t["parameters"].get(k) == v for k, v in params.items())]
    assert len(hits) == 1, f"{metric_id} {params} matched {len(hits)} tests"
    return hits[0]


# --------------------------------------------------------------------------- #
# The schedule as a whole
# --------------------------------------------------------------------------- #
class TestTheScheduleIsComplete:
    def test_sixteen_calculable_limits_are_active(self, config):
        # Seventeen stated limits, minus Net WAC, which cannot be calculated.
        assert len(config.tests) == 16

    def test_every_active_test_is_measured_over_eligible_mortgage_loans(self, config):
        assert {t.population for t in config.tests} == {"eligible_mortgage_loans"}

    def test_every_share_test_carries_the_33m_denominator_floor(self, config):
        shares = [t for t in config.tests if t.unit == "percent"]
        assert len(shares) == 15   # everything but the average-balance test
        assert all(t.parameters["denominator_floor"] == 33_000_000.0
                   for t in shares)


# --------------------------------------------------------------------------- #
# Geographic concentration
# --------------------------------------------------------------------------- #
class TestGeographicConcentration:
    """Current Balance of Eligible Mortgage Loans in each region ÷ the
    Concentration Limit Denominator."""

    def test_london_and_south_east_are_ONE_combined_test(self, config):
        regional = [t for t in config.tests if t.metric_id == "geo_region_share"]
        combined = [t for t in regional
                    if t.parameters["regions"] == ["London", "South East"]]
        assert len(combined) == 1
        assert combined[0].threshold == 50.0
        # ...and there is no separate London or South East limit beside it.
        assert not [t for t in regional
                    if t.parameters["regions"] in (["London"], ["South East"])]

    @pytest.mark.parametrize("region,limit", REGION_LIMITS)
    def test_immediately_below_the_limit_passes(self, config, lib, region, limit):
        # limit% of £100m, less £1.
        balance = limit * 1_000_000.0 - 1.0
        results = evaluate(two_loan_book(subject_balance=balance,
                                         collateral_geography=region), config, lib)
        row = one(results, "geo_region_share",
                  regions=(["London", "South East"] if limit == 50.0 else [region]))
        assert row["denominatorValue"] == PORTFOLIO_TOTAL
        assert row["currentValue"] == pytest.approx(limit, abs=0.01)
        assert row["status"] in (STATUS_PASS, STATUS_WARNING)

    @pytest.mark.parametrize("region,limit", REGION_LIMITS)
    def test_exactly_at_the_limit_passes(self, config, lib, region, limit):
        balance = limit * 1_000_000.0
        results = evaluate(two_loan_book(subject_balance=balance,
                                         collateral_geography=region), config, lib)
        row = one(results, "geo_region_share",
                  regions=(["London", "South East"] if limit == 50.0 else [region]))
        assert row["currentValue"] == limit
        assert row["status"] != STATUS_BREACH

    @pytest.mark.parametrize("region,limit", REGION_LIMITS)
    def test_immediately_above_the_limit_breaches(self, config, lib, region, limit):
        balance = limit * 1_000_000.0 + 10_000.0     # +0.01pp
        results = evaluate(two_loan_book(subject_balance=balance,
                                         collateral_geography=region), config, lib)
        row = one(results, "geo_region_share",
                  regions=(["London", "South East"] if limit == 50.0 else [region]))
        assert row["currentValue"] == pytest.approx(limit + 0.01, abs=0.001)
        assert row["status"] == STATUS_BREACH

    def test_the_combined_test_adds_london_AND_south_east_together(self, config, lib):
        # £30m London + £25m South East = £55m = 55%, over the 50% combined
        # limit, even though neither alone would reach it.
        df = frame([
            {"loan_id": "LON", "current_outstanding_balance": 30_000_000.0,
             "collateral_geography": "London"},
            {"loan_id": "SE", "current_outstanding_balance": 25_000_000.0,
             "collateral_geography": "South East"},
            {"loan_id": "REST", "current_outstanding_balance": 45_000_000.0},
        ])
        row = one(evaluate(df, config, lib), "geo_region_share",
                  regions=["London", "South East"])
        assert row["currentValue"] == 55.0
        assert row["status"] == STATUS_BREACH

    def test_the_floor_binds_on_a_small_book_and_shrinks_the_percentage(self,
                                                                       config, lib):
        # £10m of Scotland in a £20m book is 50% of the BOOK but only
        # 30.30% of the £33m contractual denominator.
        df = frame([
            {"loan_id": "SCO", "current_outstanding_balance": 10_000_000.0,
             "collateral_geography": "Scotland"},
            {"loan_id": "REST", "current_outstanding_balance": 10_000_000.0},
        ])
        row = one(evaluate(df, config, lib), "geo_region_share",
                  regions=["Scotland"])
        assert row["denominatorValue"] == 33_000_000.0
        assert row["currentValue"] == pytest.approx(30.30, abs=0.01)


# --------------------------------------------------------------------------- #
# Property value
# --------------------------------------------------------------------------- #
class TestPropertyValue:
    """Original Valuation > £1,500,000 <= 10%; < £150,000 <= 10%."""

    @pytest.mark.parametrize("balance,expected,breaches", [
        (9_999_000.0, 9.999, False),
        (10_000_000.0, 10.0, False),
        (10_010_000.0, 10.01, True),
    ])
    def test_high_value_boundary(self, config, lib, balance, expected, breaches):
        df = two_loan_book(subject_balance=balance,
                           original_valuation_amount=1_500_001.0)
        row = one(evaluate(df, config, lib), "property_value_above_share")
        assert row["threshold"] == 10.0
        assert row["currentValue"] == pytest.approx(expected, abs=0.01)
        assert (row["status"] == STATUS_BREACH) is breaches

    def test_a_property_valued_at_exactly_1_5m_is_NOT_above_1_5m(self, config, lib):
        df = two_loan_book(subject_balance=50_000_000.0,
                           original_valuation_amount=1_500_000.0)
        row = one(evaluate(df, config, lib), "property_value_above_share")
        assert row["currentValue"] == 0.0

    @pytest.mark.parametrize("balance,expected,breaches", [
        (9_999_000.0, 9.999, False),
        (10_000_000.0, 10.0, False),
        (10_010_000.0, 10.01, True),
    ])
    def test_low_value_boundary(self, config, lib, balance, expected, breaches):
        df = two_loan_book(subject_balance=balance,
                           original_valuation_amount=149_999.0)
        row = one(evaluate(df, config, lib), "property_value_below_share")
        assert row["threshold"] == 10.0
        assert row["currentValue"] == pytest.approx(expected, abs=0.01)
        assert (row["status"] == STATUS_BREACH) is breaches

    def test_a_property_valued_at_exactly_150k_is_NOT_below_150k(self, config, lib):
        df = two_loan_book(subject_balance=50_000_000.0,
                           original_valuation_amount=150_000.0)
        row = one(evaluate(df, config, lib), "property_value_below_share")
        assert row["currentValue"] == 0.0


# --------------------------------------------------------------------------- #
# Principal balance
# --------------------------------------------------------------------------- #
class TestBorrowerAggregatePrincipal:
    """Loans to any Borrower whose AGGREGATE initial principal balance exceeds
    £1,000,000, as a percentage of the Concentration Limit Denominator <= 10%.

    The aggregate is struck per borrower. A borrower with five £250,000 loans
    qualifies; one £250,000 loan does not. Substituting the loan's own initial
    balance for the borrower's aggregate is the error these tests catch."""

    @staticmethod
    def _book(loans_per_borrower: int, each_original: float,
              current_each: float) -> pd.DataFrame:
        rows = [{"loan_id": f"BIG{i}", "borrower_identifier": "BIG_BORROWER",
                 "original_principal_balance": each_original,
                 "current_outstanding_balance": current_each}
                for i in range(loans_per_borrower)]
        rest = PORTFOLIO_TOTAL - current_each * loans_per_borrower
        rows.append({"loan_id": "REST", "borrower_identifier": "SMALL",
                     "original_principal_balance": 250_000.0,
                     "current_outstanding_balance": rest})
        return frame(rows)

    def test_aggregation_is_per_BORROWER_not_per_loan(self, config, lib):
        # Five loans of £250,000 each: no single loan is over £1m, but the
        # borrower's aggregate initial principal is £1.25m, which is.
        df = self._book(5, 250_000.0, 1_000_000.0)
        row = one(evaluate(df, config, lib), "borrower_aggregate_balance_share")
        assert row["parameters"]["aggregate_basis"] == "original"
        assert row["loansInNumerator"] == 5
        assert row["currentValue"] == 5.0        # £5m of £100m

    def test_a_borrower_at_exactly_1m_aggregate_is_not_above_1m(self, config, lib):
        df = self._book(4, 250_000.0, 1_000_000.0)     # aggregate exactly £1m
        row = one(evaluate(df, config, lib), "borrower_aggregate_balance_share")
        assert row["currentValue"] == 0.0

    @pytest.mark.parametrize("current_each,expected,breaches", [
        (1_999_800.0, 9.999, False),
        (2_000_000.0, 10.0, False),
        (2_002_000.0, 10.01, True),
    ])
    def test_the_10pc_boundary(self, config, lib, current_each, expected, breaches):
        df = self._book(5, 250_000.0, current_each)
        row = one(evaluate(df, config, lib), "borrower_aggregate_balance_share")
        assert row["currentValue"] == pytest.approx(expected, abs=0.01)
        assert (row["status"] == STATUS_BREACH) is breaches


class TestAverageInitialPrincipal:
    """The AVERAGE initial principal balance of Eligible Mortgage Loans shall
    not be greater than £300,000. A currency limit, not a share — it has no
    denominator and no percentage."""

    @staticmethod
    def _book(*originals: float) -> pd.DataFrame:
        return frame([{"loan_id": f"L{i}",
                       "original_principal_balance": o,
                       "current_outstanding_balance": 1_000_000.0}
                      for i, o in enumerate(originals)])

    def test_the_test_is_a_currency_amount_with_no_denominator(self, config, lib):
        row = one(evaluate(self._book(250_000.0), config, lib), "balance_average")
        assert row["unit"] == "currency"
        assert row["threshold"] == 300_000.0

    @pytest.mark.parametrize("originals,expected,breaches", [
        ((299_999.0, 299_999.0), 299_999.0, False),
        ((300_000.0, 300_000.0), 300_000.0, False),
        ((300_001.0, 300_001.0), 300_001.0, True),
        # An average, not a maximum: one £1m loan among three small ones
        # averages £362,500 and breaches, even though most loans are tiny.
        ((1_000_000.0, 150_000.0, 150_000.0, 150_000.0), 362_500.0, True),
    ])
    def test_the_300k_boundary(self, config, lib, originals, expected, breaches):
        row = one(evaluate(self._book(*originals), config, lib), "balance_average")
        assert row["currentValue"] == pytest.approx(expected, abs=1.0)
        assert (row["status"] == STATUS_BREACH) is breaches

    def test_it_reads_the_INITIAL_principal_not_the_current_balance(self, config, lib):
        # Initial £250,000 each, current £1,000,000 each. Reading the current
        # balance would give £1m and a breach; the schedule says initial.
        row = one(evaluate(self._book(250_000.0, 250_000.0), config, lib),
                  "balance_average")
        assert row["currentValue"] == 250_000.0
        assert row["status"] != STATUS_BREACH


# --------------------------------------------------------------------------- #
# Age
# --------------------------------------------------------------------------- #
class TestAge:
    """Sole borrower, or youngest borrower, below 55 years of age: <= 0%.

    A zero limit means any exposure at all is a breach."""

    def test_a_book_with_nobody_under_55_passes_at_zero(self, config, lib):
        df = two_loan_book(subject_balance=50_000_000.0,
                           youngest_borrower_age=55.0)
        row = one(evaluate(df, config, lib), "borrower_age_share")
        assert row["threshold"] == 0.0
        assert row["currentValue"] == 0.0
        assert row["status"] == STATUS_PASS

    def test_exactly_55_is_NOT_below_55(self, config, lib):
        df = two_loan_book(subject_balance=99_999_999.0,
                           youngest_borrower_age=55.0)
        row = one(evaluate(df, config, lib), "borrower_age_share")
        assert row["currentValue"] == 0.0

    def test_any_exposure_at_all_breaches_a_zero_limit(self, config, lib):
        # £1,000,000 of a £100,000,000 book: 1.00%, against a limit of 0%.
        df = two_loan_book(subject_balance=1_000_000.0, youngest_borrower_age=54.0)
        row = one(evaluate(df, config, lib), "borrower_age_share")
        assert row["currentValue"] == 1.0
        assert row["status"] == STATUS_BREACH

    def test_an_exposure_too_small_to_ROUND_above_zero_still_breaches(self,
                                                                     config, lib):
        # £1 of a £100,000,000 book is 0.000001%, which the metric reports as
        # 0.00 at its output precision. A zero limit permits nothing, so the
        # CONTRIBUTING POPULATION decides the status, not the rounded share.
        # The reported value stays 0.00 — it is the honest rounded figure —
        # while the status is the honest contractual answer.
        df = two_loan_book(subject_balance=1.0, youngest_borrower_age=54.0)
        row = one(evaluate(df, config, lib), "borrower_age_share")
        assert row["currentValue"] == 0.0
        assert row["loansInNumerator"] == 1
        assert row["status"] == STATUS_BREACH

    def test_a_zero_limit_with_an_EMPTY_numerator_still_passes(self, config, lib):
        df = two_loan_book(subject_balance=1.0, youngest_borrower_age=70.0)
        row = one(evaluate(df, config, lib), "borrower_age_share")
        assert row["loansInNumerator"] == 0
        assert row["status"] == STATUS_PASS

    def test_it_reads_the_YOUNGEST_borrower(self, config, lib):
        row = one(evaluate(two_loan_book(subject_balance=1_000_000.0,
                                         youngest_borrower_age=40.0),
                           config, lib), "borrower_age_share")
        assert row["parameters"]["age_basis"] == "youngest"
        assert row["status"] == STATUS_BREACH

    def test_it_uses_the_governed_youngest_age_field(self, config, lib):
        df = two_loan_book(subject_balance=1_000_000.0, youngest_borrower_age=40.0)
        row = one(evaluate(df, config, lib), "borrower_age_share")
        assert row["resolvedColumns"] == {"borrower_age_youngest":
                                          "youngest_borrower_age"}


# --------------------------------------------------------------------------- #
# Other
# --------------------------------------------------------------------------- #
class TestTwoBorrowers:
    """Eligible Mortgage Loans with TWO Borrowers <= 90% of the denominator."""

    @pytest.mark.parametrize("balance,expected,breaches", [
        (89_990_000.0, 89.99, False),
        (90_000_000.0, 90.0, False),
        (90_010_000.0, 90.01, True),
    ])
    def test_the_90pc_boundary(self, config, lib, balance, expected, breaches):
        df = two_loan_book(subject_balance=balance, number_of_borrowers=2)
        row = one(evaluate(df, config, lib), "borrower_joint_share")
        assert row["parameters"]["joint_basis"] == "borrower_count"
        assert row["currentValue"] == pytest.approx(expected, abs=0.01)
        assert (row["status"] == STATUS_BREACH) is breaches

    def test_a_sole_borrower_loan_is_not_in_the_numerator(self, config, lib):
        df = two_loan_book(subject_balance=90_000_000.0, number_of_borrowers=1)
        row = one(evaluate(df, config, lib), "borrower_joint_share")
        assert row["currentValue"] == 0.0


class TestPortfolioNetWac:
    """Portfolio Net WAC >= 3.75% per annum.

    Schedule 8 defines it as the weighted average fixed rate of all Eligible
    Mortgage Loans (weighted by Current Balance AND EXPECTED DURATION) minus
    the SERIES FIXED RATE from the Final Maturity Date onwards. Neither the
    expected duration nor the Series Fixed Rate exists in the canonical data or
    the facility configuration, so the test is NOT CALCULABLE. It is not
    approximated with a balance-only weighted average, which would be a
    different number reported under the contractual test's name.
    """

    def test_it_is_extracted_with_its_threshold_and_direction(
            self, schedule_8_text, lib):
        proposals = build_proposals(lib=lib, client_id="c",
                                    source_reference="S8", text=schedule_8_text)
        [wac] = [p for p in proposals if p.proposed_metric_id == "rate_net_wac"]
        assert wac.extracted_threshold == 3.75
        assert wac.extracted_operator == "min"

    def test_it_is_held_back_from_approval_with_the_missing_input_named(
            self, schedule_8_text, lib):
        proposals = build_proposals(lib=lib, client_id="c",
                                    source_reference="S8", text=schedule_8_text)
        [wac] = [p for p in proposals if p.proposed_metric_id == "rate_net_wac"]
        assert wac.match_outcome == MATCH_AMBIGUOUS
        assert wac.concern_codes == ["net_wac_definition_uncertain"]
        assert wac.confirmation_questions

    def test_it_is_therefore_not_in_the_active_configuration(self, config):
        assert "rate_net_wac" not in {t.metric_id for t in config.tests}

    def test_activating_it_anyway_yields_UNAVAILABLE_not_a_number(self, lib):
        # Belt and braces: even if an operator forced it through with no
        # confirmed deduction, the evaluator refuses to produce a figure.
        forced = ActiveConfiguration(
            client_id="c", library_version=lib.library_version,
            tests=[ActiveTest(test_id="ct_wac", metric_id="rate_net_wac",
                              display_name="Net WAC", operator="min",
                              threshold=3.75, unit="percent", parameters={},
                              population="eligible_mortgage_loans")])
        df = two_loan_book(subject_balance=50_000_000.0)
        results = evaluate(df, forced, lib)
        [row] = list(results.values())
        assert row["currentValue"] is None
        assert row["status"] == STATUS_UNAVAILABLE


# --------------------------------------------------------------------------- #
# The population itself
# --------------------------------------------------------------------------- #
class TestTheEligiblePopulation:
    def test_an_ineligible_loan_leaves_BOTH_numerator_and_denominator(self,
                                                                     config, lib):
        """The one property that makes Schedule 8 mean what it says.

        £20m of Scotland and £80m elsewhere, but the £80m is ineligible. The
        answer is not 20% of £100m: it is 100% of the £20m of eligible
        collateral — and, because that is under the floor, 60.61% of £33m.
        """
        from mi_agent.borrowing_base.eligibility import derive_eligibility, eligible_mask
        from mi_agent.borrowing_base.models import EligibilityRule
        from .conftest import production_facility

        df = frame([
            {"loan_id": "SCO", "current_outstanding_balance": 20_000_000.0,
             "collateral_geography": "Scotland"},
            {"loan_id": "REST", "current_outstanding_balance": 80_000_000.0},
        ])
        df["days_past_due"] = [0.0, 400.0]
        fac = production_facility([EligibilityRule(
            rule_id="no_arrears", field="days_past_due", operator="max",
            value=30.0)])
        derive_eligibility(df, fac)
        eligible = df[eligible_mask(df, fac)]
        out = evaluate_active_tests(
            df, None, config, lib, reporting_date="2025-11-30",
            populations={"eligible_mortgage_loans": eligible})
        row = one({t["testId"]: t for t in out["tests"]},
                  "geo_region_share", regions=["Scotland"])
        assert row["denominatorValue"] == 33_000_000.0
        assert row["numeratorValue"] == 20_000_000.0
        assert row["currentValue"] == pytest.approx(60.61, abs=0.01)

    def test_a_caller_that_supplies_no_population_gets_UNAVAILABLE_not_the_book(
            self, config, lib):
        # Silently widening to the whole portfolio would answer a different
        # question under the contractual test's name.
        df = two_loan_book(subject_balance=50_000_000.0,
                           collateral_geography="Scotland")
        out = evaluate_active_tests(df, None, config, lib,
                                    reporting_date="2025-11-30")
        row = one({t["testId"]: t for t in out["tests"]},
                  "geo_region_share", regions=["Scotland"])
        assert row["status"] == STATUS_UNAVAILABLE
        assert "Eligible Mortgage Loans" in row["notes"]
