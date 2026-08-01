"""Deterministic metric evaluators: hand-checkable values, honest failure
states, and a drill mask that reconciles to the numerator by construction."""

from __future__ import annotations

import pandas as pd
import pytest

from mi_agent.concentration_tests.metrics import evaluate_metric
from mi_agent.concentration_tests.models import (
    DATA_EXTERNAL_UNCONFIGURED,
    DATA_MISSING,
    DATA_OK,
)


def run(lib, df, metric_id, params, **kw):
    return evaluate_metric(df, lib, lib.get(metric_id), params, **kw)


class TestGeography:
    def test_single_region_share(self, lib, funded_frame):
        # East Anglia = 100k + 200k + 50k = 350k of 1,000k → 35%
        r = run(lib, funded_frame, "geo_region_share",
                {"regions": ["East Anglia"]})
        assert r.value == 35.0
        assert r.numerator_value == 350_000
        assert r.denominator_value == 1_000_000
        assert r.loans_in_numerator == 3

    def test_combined_region_share(self, lib, funded_frame):
        # London 250k + South East 110k = 360k → 36%
        r = run(lib, funded_frame, "geo_region_share",
                {"regions": ["London", "South East"]})
        assert r.value == 36.0

    def test_region_matching_is_case_insensitive(self, lib, funded_frame):
        r = run(lib, funded_frame, "geo_region_share",
                {"regions": ["east anglia"]})
        assert r.value == 35.0

    def test_invalid_region_yields_zero_share_not_error(self, lib, funded_frame):
        r = run(lib, funded_frame, "geo_region_share",
                {"regions": ["Atlantis"]})
        assert r.value == 0.0
        assert r.data_status == DATA_OK
        assert r.loans_in_numerator == 0

    def test_largest_region_with_exclusions(self, lib, funded_frame):
        r = run(lib, funded_frame, "geo_largest_region_share",
                {"exclude_values": ["East Anglia", "London"]})
        # Next largest is Scotland (120k) → 12%
        assert r.value == 12.0
        assert "Scotland" in r.notes

    def test_region_count(self, lib, funded_frame):
        r = run(lib, funded_frame, "geo_region_count", {})
        assert r.value == 6.0

    def test_missing_region_column_is_data_missing(self, lib, funded_frame):
        df = funded_frame.drop(columns=["collateral_geography"])
        r = run(lib, df, "geo_region_share", {"regions": ["London"]})
        assert r.value is None
        assert r.data_status == DATA_MISSING
        assert "collateral_geography" in r.missing_fields


class TestBalancesAndValuations:
    def test_original_vs_current_basis_differ(self, lib, funded_frame):
        current = run(lib, funded_frame, "balance_above_share",
                      {"amount": 150_000, "balance_basis": "current"})
        original = run(lib, funded_frame, "balance_above_share",
                       {"amount": 150_000, "balance_basis": "original"})
        # current: 200k loan only (>150k) → 20%; original: L2 (160k) + L4 (1.2m)
        # → their CURRENT balances 150k? no — numerator sums the DENOMINATOR
        # basis balance for matching loans: L2 (150k) + L4 (200k) + L7? 130k>150k? no.
        assert current.value == 20.0
        assert original.value == 35.0

    def test_valuation_below_share(self, lib, funded_frame):
        # < 100k original valuation: L3 (90k) + L8 (95k) → 50k + 60k = 110k → 11%
        r = run(lib, funded_frame, "property_value_below_share",
                {"amount": 100_000, "value_basis": "original",
                 "comparison": "below"})
        assert r.value == 11.0

    def test_valuation_above_share_exact_threshold_excluded(self, lib, funded_frame):
        # strict > comparison: a loan AT the threshold does not count.
        at = run(lib, funded_frame, "property_value_above_share",
                 {"amount": 2_000_000, "value_basis": "original",
                  "comparison": "above"})
        assert at.value == 0.0
        just_below = run(lib, funded_frame, "property_value_above_share",
                         {"amount": 1_999_999, "value_basis": "original",
                          "comparison": "above"})
        assert just_below.value == 15.0  # L2 = 150k of 1m

    def test_average_balance(self, lib, funded_frame):
        r = run(lib, funded_frame, "balance_average",
                {"balance_basis": "current"})
        assert r.value == 100_000
        assert r.denominator_basis == "loan_count"

    def test_weighted_average_valuation(self, lib, funded_frame):
        r = run(lib, funded_frame, "property_wa_valuation",
                {"value_basis": "original", "weighting": "current_balance"})
        expected = (funded_frame.original_valuation_amount
                    * funded_frame.current_outstanding_balance).sum() \
            / funded_frame.current_outstanding_balance.sum()
        assert r.value == round(float(expected), 0)

    def test_maximum_balance(self, lib, funded_frame):
        r = run(lib, funded_frame, "balance_maximum",
                {"balance_basis": "original"})
        assert r.value == 1_200_000
        assert r.loans_in_numerator == 1

    def test_top_n_share(self, lib, funded_frame):
        # top 2 by current balance: 200k + 150k = 350k → 35%
        r = run(lib, funded_frame, "top_n_loans_share", {"n": 2})
        assert r.value == 35.0
        assert r.loans_in_numerator == 2

    def test_zero_denominator_is_not_zero_percent(self, lib, funded_frame):
        df = funded_frame.copy()
        df["current_outstanding_balance"] = 0
        r = run(lib, df, "geo_region_share", {"regions": ["London"]})
        assert r.value is None
        assert r.data_status == DATA_MISSING

    def test_null_values_are_excluded_not_guessed(self, lib, funded_frame):
        df = funded_frame.copy()
        df.loc[0, "original_valuation_amount"] = None
        r = run(lib, df, "property_value_below_share",
                {"amount": 100_000, "value_basis": "original",
                 "comparison": "below"})
        assert r.value == 11.0  # unchanged: nulls never satisfy a comparison

    def test_mixed_units_parse_through_coerce_numeric(self, lib, funded_frame):
        df = funded_frame.copy()
        df["current_outstanding_balance"] = [
            "£100,000", "150000", "50,000", "£200,000", "100000",
            "80,000", "120000", "60,000", "90,000", "50000"]
        r = run(lib, df, "geo_region_share", {"regions": ["East Anglia"]})
        assert r.value == 35.0

    def test_empty_frame_is_insufficient(self, lib):
        r = run(lib, pd.DataFrame(), "geo_region_share",
                {"regions": ["London"]})
        assert r.value is None


class TestBorrowers:
    def test_duplicate_borrower_grouping(self, lib, funded_frame):
        # B1 holds L1 + L2 + L10 = 100k + 150k + 50k = 300k → 30%
        r = run(lib, funded_frame, "borrower_largest_share", {})
        assert r.value == 30.0
        assert r.loans_in_numerator == 3

    def test_max_loans_per_borrower(self, lib, funded_frame):
        r = run(lib, funded_frame, "borrower_max_loans", {})
        assert r.value == 3.0

    def test_multi_loan_borrower_share(self, lib, funded_frame):
        r = run(lib, funded_frame, "borrower_multi_loan_share",
                {"min_loans": 3})
        assert r.value == 30.0

    def test_joint_share_by_structure_flag(self, lib, funded_frame):
        # Joint: L1, L2, L4, L7 → 100+150+200+120 = 570k → 57%
        r = run(lib, funded_frame, "borrower_joint_share",
                {"joint_basis": "structure_flag"})
        assert r.value == 57.0

    def test_single_share_is_the_inverse(self, lib, funded_frame):
        r = run(lib, funded_frame, "borrower_joint_share",
                {"joint_basis": "structure_flag", "invert": True})
        assert r.value == 43.0

    def test_age_below_share(self, lib, funded_frame):
        # youngest < 55: L2 (54) → 150k → 15%
        r = run(lib, funded_frame, "borrower_age_share",
                {"age": 55, "comparison": "below", "age_basis": "youngest"})
        assert r.value == 15.0


class TestRatesAndLtv:
    def test_gross_wac(self, lib, funded_frame):
        w = funded_frame.current_outstanding_balance
        expected = round(float((funded_frame.current_interest_rate * w).sum()
                               / w.sum()), 2)
        r = run(lib, funded_frame, "rate_gross_wac", {})
        assert r.value == expected

    def test_net_wac_subtracts_the_confirmed_deduction(self, lib, funded_frame):
        gross = run(lib, funded_frame, "rate_gross_wac", {})
        net = run(lib, funded_frame, "rate_net_wac",
                  {"deduction_percent": 0.5,
                   "deduction_basis": "servicing_fee_only"})
        assert net.value == round(gross.value - 0.5, 2)

    def test_variable_rate_share(self, lib, funded_frame):
        # Variable: L2, L4, L7 → 150+200+120 = 470k → 47%
        r = run(lib, funded_frame, "rate_type_share",
                {"values": ["variable", "floating"]})
        assert r.value == 47.0

    def test_ltv_fraction_normalised_to_percent(self, lib, funded_frame):
        # ratios (0-1) must be read as percent: > 50% → L3 (0.51), L8 (0.60)
        r = run(lib, funded_frame, "ltv_above_share",
                {"ltv_percent": 50, "ltv_basis": "current"})
        assert r.value == 11.0  # 50k + 60k

    def test_wa_ltv(self, lib, funded_frame):
        r = run(lib, funded_frame, "ltv_weighted_average",
                {"ltv_basis": "current"})
        w = funded_frame.current_outstanding_balance
        expected = round(float((funded_frame.current_loan_to_value * 100 * w).sum()
                               / w.sum()), 2)
        assert r.value == expected

    def test_indexed_ltv_without_field_is_missing(self, lib, funded_frame):
        r = run(lib, funded_frame, "ltv_above_share",
                {"ltv_percent": 50, "ltv_basis": "indexed"})
        assert r.value is None
        assert r.data_status == DATA_MISSING


class TestPerformanceAndComposition:
    def test_dpd_share(self, lib, funded_frame):
        # > 30 days: L3 (45d, 50k) + L6 (95d, 80k) → 13%
        r = run(lib, funded_frame, "perf_arrears_share", {"min_days": 30})
        assert r.value == 13.0

    def test_vintage_share(self, lib, funded_frame):
        r = run(lib, funded_frame, "composition_vintage_share",
                {"years": ["2024"]})
        # 2024 originations: L2, L4, L7, L10 → 150+200+120+50 = 520k → 52%
        assert r.value == 52.0

    def test_dimension_share_missing_column(self, lib, funded_frame):
        r = run(lib, funded_frame, "composition_dimension_share",
                {"dimension": "servicer", "values": ["Acme"]})
        assert r.data_status == DATA_MISSING


class TestExternalIndex:
    def test_unconfigured_index_is_never_simulated(self, lib, funded_frame):
        r = run(lib, funded_frame, "index_hpi_ratio",
                {"index_source": "ONS", "index_series": "UK-all",
                 "reference_date": "2024-01-31"})
        assert r.value is None
        assert r.data_status == DATA_EXTERNAL_UNCONFIGURED

    def test_a_configured_provider_evaluates_the_ratio(self, lib, funded_frame):
        class Provider:
            def index_value(self, source, series, on_date):
                return {"2024-01-31": 100.0, "2025-11-30": 92.5}.get(on_date)
        r = run(lib, funded_frame, "index_hpi_ratio",
                {"index_source": "ONS", "index_series": "UK-all",
                 "reference_date": "2024-01-31",
                 "_reporting_date": "2025-11-30"},
                external=Provider())
        assert r.value == 92.5


class TestDrillMask:
    @pytest.mark.parametrize("metric_id,params", [
        ("geo_region_share", {"regions": ["East Anglia"]}),
        ("balance_above_share", {"amount": 90_000, "balance_basis": "current"}),
        ("borrower_joint_share", {"joint_basis": "structure_flag"}),
        ("top_n_loans_share", {"n": 3}),
        ("perf_arrears_share", {"min_days": 30}),
    ])
    def test_mask_reconciles_exactly_to_the_numerator(self, lib, funded_frame,
                                                      metric_id, params):
        r = run(lib, funded_frame, metric_id, params)
        subset = funded_frame[r.mask]
        assert len(subset) == r.loans_in_numerator
        assert round(float(subset.current_outstanding_balance.sum()), 2) == \
            r.numerator_value
