"""Temporality, aggregation, units and directionality — the calculation core.

These are the tests that make the workflow trustworthy: they prove that a
governed classification in the Business Semantics Registry actually changes the
arithmetic, and that a calculation which cannot be performed honestly is reported
as such rather than approximated.
"""

from __future__ import annotations

import pandas as pd
import pytest

from mi_agent.business_semantics import BusinessSemanticsEntry
from mi_agent.period_change import models as m
from mi_agent.period_change.calculations import (
    aggregate,
    build_pair_context,
    interpret,
    metric_change,
)
from mi_agent.period_change.units import (
    SOURCE_AGGREGATION_DEFAULT,
    SOURCE_MI_SEMANTICS,
    resolve_unit,
)


def entry(field="metric", *, role="measure", temporality="point_in_time",
          aggregation="sum", weight_field=None, share_basis=None,
          directionality="neutral", concept="exposure", confidence="high",
          assets=("cross_asset",), tags=("period_change",)) -> BusinessSemanticsEntry:
    return BusinessSemanticsEntry(
        field=field, source_field=field, display_name=field.replace("_", " ").title(),
        analytical_concept=concept, analytical_role=role, temporality=temporality,
        categories=(concept,), workflow_tags=tuple(tags),
        directionality=directionality, default_aggregation=aggregation,
        weight_field=weight_field, share_basis=share_basis,
        portfolio_comparability="comparable", supports_materiality_assessment=True,
        asset_applicability=tuple(assets), confidence=confidence,
        rationale="test entry")


def snap(df, snapshot_id="s", reporting_date="2026-06-30") -> m.SnapshotFrame:
    return m.SnapshotFrame(snapshot_id=snapshot_id, reporting_date=reporting_date,
                           frame=df, row_count=None if df is None else len(df))


def change(e, start_df, end_df) -> m.MetricChange:
    return metric_change(e, snap(start_df, "s0", "2026-03-31"),
                         snap(end_df, "s1", "2026-06-30"))


# --------------------------------------------------------------------------- #
# Aggregation — §8
# --------------------------------------------------------------------------- #
class TestAggregation:
    def test_sum_aggregates_valid_numeric_values(self):
        e = entry("amount", aggregation="sum")
        df = pd.DataFrame({"amount": [100.0, 200.0, 300.0]})
        out = aggregate(df, build_pair_context(e, [df]))
        assert out.value == 600.0
        assert out.status == m.STATUS_AVAILABLE
        assert (out.valid_population, out.excluded_population) == (3, 0)

    def test_sum_parses_accounting_formatted_amounts(self):
        """The platform's single deterministic coercion is used, so a tape that
        stores "1,234.50" and "(500)" is not silently summed to zero."""
        e = entry("amount", aggregation="sum")
        df = pd.DataFrame({"amount": ["1,234.50", "(500)", "£100"]})
        out = aggregate(df, build_pair_context(e, [df]))
        assert out.value == pytest.approx(834.50)

    def test_invalid_values_are_excluded_and_counted(self):
        e = entry("amount", aggregation="sum")
        df = pd.DataFrame({"amount": [100.0, None, "not a number"]})
        out = aggregate(df, build_pair_context(e, [df]))
        assert out.value == 100.0
        assert out.status == m.STATUS_PARTIALLY_AVAILABLE
        assert (out.valid_population, out.excluded_population) == (1, 2)

    def test_average_uses_the_valid_population_as_its_denominator(self):
        e = entry("rate", aggregation="average")
        df = pd.DataFrame({"rate": [2.0, 4.0, None]})
        out = aggregate(df, build_pair_context(e, [df]))
        assert out.value == 3.0
        assert out.denominator == 2.0

    def test_a_field_absent_from_the_snapshot_is_not_available(self):
        e = entry("missing")
        df = pd.DataFrame({"other": [1]})
        out = aggregate(df, build_pair_context(e, [df]))
        assert out.value is None
        assert out.status == m.STATUS_NOT_AVAILABLE


class TestWeightedAverage:
    def _entry(self):
        return entry("current_loan_to_value", aggregation="weighted_average",
                     weight_field="current_outstanding_balance")

    def test_weights_by_the_registry_weight_field(self):
        e = self._entry()
        df = pd.DataFrame({"current_loan_to_value": [40.0, 60.0],
                           "current_outstanding_balance": [300.0, 100.0]})
        out = aggregate(df, build_pair_context(e, [df]))
        assert out.value == pytest.approx(45.0)          # not the 50.0 mean
        assert out.weight_field == "current_outstanding_balance"
        assert out.total_weight == 400.0
        assert out.numerator == pytest.approx(18000.0)
        assert out.denominator == 400.0

    def test_a_missing_weight_column_never_falls_back_to_an_unweighted_mean(self):
        """An unweighted answer to a weighted question is a different number
        wearing the same label."""
        e = self._entry()
        df = pd.DataFrame({"current_loan_to_value": [40.0, 60.0]})
        out = aggregate(df, build_pair_context(e, [df]))
        assert out.value is None
        assert out.status == m.STATUS_INVALID_WEIGHT_POPULATION
        assert any("not substituted" in n for n in out.notes)

    def test_a_registry_entry_with_no_weight_field_is_an_invalid_population(self):
        e = entry("ratio", aggregation="weighted_average", weight_field=None)
        df = pd.DataFrame({"ratio": [1.0], "current_outstanding_balance": [1.0]})
        out = aggregate(df, build_pair_context(e, [df]))
        assert out.status == m.STATUS_INVALID_WEIGHT_POPULATION

    def test_zero_total_weight_is_a_zero_denominator_not_a_mean(self):
        e = self._entry()
        df = pd.DataFrame({"current_loan_to_value": [40.0, 60.0],
                           "current_outstanding_balance": [0.0, 0.0]})
        out = aggregate(df, build_pair_context(e, [df]))
        assert out.value is None
        assert out.status == m.STATUS_ZERO_DENOMINATOR
        assert out.total_weight == 0.0

    def test_rows_missing_either_value_or_weight_are_excluded_and_recorded(self):
        e = self._entry()
        df = pd.DataFrame({"current_loan_to_value": [40.0, 60.0, None],
                           "current_outstanding_balance": [100.0, None, 100.0]})
        out = aggregate(df, build_pair_context(e, [df]))
        assert out.value == pytest.approx(40.0)
        assert (out.valid_population, out.excluded_population) == (1, 2)
        assert out.status == m.STATUS_PARTIALLY_AVAILABLE

    def test_no_valid_weighted_population_at_all(self):
        e = self._entry()
        df = pd.DataFrame({"current_loan_to_value": [None, None],
                           "current_outstanding_balance": [1.0, 2.0]})
        assert aggregate(df, build_pair_context(e, [df])).status == \
            m.STATUS_INVALID_WEIGHT_POPULATION


class TestShare:
    def _entry(self, basis="count"):
        return entry("interest_in_arrears", aggregation="share", share_basis=basis,
                     directionality="higher_is_worse", concept="payment_performance")

    def test_a_count_share_states_its_numerator_and_denominator(self):
        e = self._entry()
        df = pd.DataFrame({"interest_in_arrears": ["Y", "N", "N", "Y"]})
        out = aggregate(df, build_pair_context(e, [df]))
        assert out.value == pytest.approx(50.0)          # points
        assert (out.numerator, out.denominator) == (2.0, 4.0)
        assert out.share_basis == "count"

    def test_a_count_share_is_never_silently_turned_into_a_balance_share(self):
        e = self._entry()
        df = pd.DataFrame({"interest_in_arrears": ["Y", "N"],
                           "current_outstanding_balance": [1_000_000.0, 1.0]})
        out = aggregate(df, build_pair_context(e, [df]))
        assert out.value == pytest.approx(50.0)          # 1 of 2 loans, not 99.9%

    def test_an_unrecognised_flag_value_is_excluded_from_both_sides(self):
        """Counting an unknown code as "no" would understate every flag share on
        a tape with a local convention."""
        e = self._entry()
        df = pd.DataFrame({"interest_in_arrears": ["Y", "N", "MAYBE"]})
        out = aggregate(df, build_pair_context(e, [df]))
        assert (out.numerator, out.denominator) == (1.0, 2.0)
        assert out.status == m.STATUS_SOURCE_CONVENTION_UNCERTAIN

    def test_a_zero_denominator_is_reported_not_divided(self):
        e = self._entry()
        df = pd.DataFrame({"interest_in_arrears": [None, None]})
        out = aggregate(df, build_pair_context(e, [df]))
        assert out.value is None
        assert out.status == m.STATUS_ZERO_DENOMINATOR

    def test_a_balance_share_basis_uses_the_governed_balance_field(self):
        e = self._entry("balance")
        df = pd.DataFrame({"interest_in_arrears": ["Y", "N"],
                           "current_outstanding_balance": [300.0, 100.0]})
        out = aggregate(df, build_pair_context(e, [df]))
        assert out.value == pytest.approx(75.0)

    def test_an_unsupported_share_basis_is_refused_not_downgraded(self):
        e = self._entry("weighted_exposure")
        df = pd.DataFrame({"interest_in_arrears": ["Y", "N"]})
        out = aggregate(df, build_pair_context(e, [df]))
        assert out.value is None
        assert out.status == m.STATUS_NOT_AVAILABLE


def test_amounts_are_not_aggregated_across_currencies():
    """§17: without a governed currency-normalisation capability, a mixed-currency
    snapshot yields no monetary total at all."""
    e = entry("current_outstanding_balance", aggregation="sum")
    df = pd.DataFrame({"current_outstanding_balance": [100.0, 100.0],
                       "exposure_currency_denomination": ["GBP", "EUR"]})
    out = aggregate(df, build_pair_context(e, [df]))
    assert out.value is None
    assert out.status == m.STATUS_NOT_COMPARABLE


# --------------------------------------------------------------------------- #
# Temporality — §7
# --------------------------------------------------------------------------- #
class TestTemporality:
    def test_point_in_time_compares_the_two_aggregates(self):
        e = entry("current_outstanding_balance", temporality="point_in_time")
        out = change(e, pd.DataFrame({"current_outstanding_balance": [100.0]}),
                     pd.DataFrame({"current_outstanding_balance": [150.0]}))
        assert out.movement_basis == m.BASIS_POINT_IN_TIME_DIFFERENCE
        assert (out.start_value, out.end_value, out.movement_value) == (100.0, 150.0, 50.0)
        assert out.relative_change == pytest.approx(0.5)

    def test_period_flows_are_compared_as_independent_period_totals(self):
        """March recoveries versus April recoveries — NOT a cumulative
        difference. The basis says so, and the note says so in words."""
        e = entry("recoveries_in_period", temporality="period_flow")
        out = change(e, pd.DataFrame({"recoveries_in_period": [100.0, 100.0]}),
                     pd.DataFrame({"recoveries_in_period": [300.0]}))
        assert out.movement_basis == m.BASIS_FLOW_LEVEL_COMPARISON
        assert out.movement_basis != m.BASIS_CUMULATIVE_DIFFERENCE
        assert (out.start_value, out.end_value) == (200.0, 300.0)
        assert m.FLOW_COMPARISON_NOTE in out.notes

    def test_cumulative_values_are_differenced(self):
        e = entry("cumulative_recoveries", temporality="cumulative")
        out = change(e, pd.DataFrame({"cumulative_recoveries": [1000.0]}),
                     pd.DataFrame({"cumulative_recoveries": [1500.0]}))
        assert out.movement_basis == m.BASIS_CUMULATIVE_DIFFERENCE
        assert out.movement_value == 500.0

    def test_a_falling_cumulative_value_warns_and_is_not_floored_at_zero(self):
        e = entry("cumulative_recoveries", temporality="cumulative")
        out = change(e, pd.DataFrame({"cumulative_recoveries": [1500.0]}),
                     pd.DataFrame({"cumulative_recoveries": [1000.0]}))
        assert out.movement_value == -500.0              # retained, not floored
        assert out.status == m.STATUS_SOURCE_CONVENTION_UNCERTAIN
        assert m.CUMULATIVE_DECLINE_NOTE in out.notes

    def test_a_static_baseline_receives_no_change_calculation(self):
        e = entry("original_loan_to_value", temporality="static_baseline")
        out = change(e, pd.DataFrame({"original_loan_to_value": [50.0]}),
                     pd.DataFrame({"original_loan_to_value": [55.0]}))
        assert out.movement_value is None
        assert out.movement_basis == m.BASIS_STATIC_BASELINE_REFERENCE
        assert out.status == m.STATUS_NOT_COMPARABLE
        assert out.start_value == 50.0 and out.end_value == 55.0


# --------------------------------------------------------------------------- #
# Availability — §9
# --------------------------------------------------------------------------- #
class TestAvailability:
    def test_a_field_present_at_only_one_date_is_marked_not_comparable(self):
        e = entry("amount")
        out = change(e, pd.DataFrame({"other": [1]}),
                     pd.DataFrame({"amount": [100.0]}))
        assert out.status == m.STATUS_NOT_COMPARABLE_DUE_TO_AVAILABILITY
        assert out.movement_value is None
        assert out.end_value == 100.0
        assert any("not substituted with a related field" in n for n in out.notes)

    def test_a_field_present_at_neither_date_is_not_available(self):
        e = entry("amount")
        out = change(e, pd.DataFrame({"other": [1]}), pd.DataFrame({"other": [1]}))
        assert out.status == m.STATUS_NOT_AVAILABLE
        assert out.movement_value is None


# --------------------------------------------------------------------------- #
# Units and expressions — §11
# --------------------------------------------------------------------------- #
class TestUnitsAndExpressions:
    def test_a_currency_field_moves_by_an_amount_and_a_percentage(self):
        e = entry("current_outstanding_balance", aggregation="sum")
        unit = resolve_unit(e)
        assert unit.unit == m.UNIT_CURRENCY
        assert unit.source == SOURCE_MI_SEMANTICS
        out = change(e, pd.DataFrame({"current_outstanding_balance": [200.0]}),
                     pd.DataFrame({"current_outstanding_balance": [250.0]}))
        assert out.relative_change == pytest.approx(0.25)
        assert out.basis_point_change is None

    def test_a_rate_moves_by_percentage_points_and_basis_points_not_a_percentage(self):
        """A rate moving 4% → 5% has risen by one point; calling that "25%
        higher" invites exactly the misreading this rule prevents."""
        e = entry("current_interest_rate", aggregation="weighted_average",
                  weight_field="current_outstanding_balance")
        out = change(e,
                     pd.DataFrame({"current_interest_rate": [4.0],
                                   "current_outstanding_balance": [100.0]}),
                     pd.DataFrame({"current_interest_rate": [5.0],
                                   "current_outstanding_balance": [100.0]}))
        assert out.movement_unit == m.UNIT_PERCENTAGE_POINT
        assert out.movement_value == pytest.approx(1.0)
        assert out.basis_point_change == pytest.approx(100.0)
        assert out.relative_change is None

    def test_a_percent_stored_as_a_fraction_is_reported_in_points(self):
        e = entry("current_loan_to_value", aggregation="weighted_average",
                  weight_field="current_outstanding_balance")
        out = change(e,
                     pd.DataFrame({"current_loan_to_value": [0.40],
                                   "current_outstanding_balance": [100.0]}),
                     pd.DataFrame({"current_loan_to_value": [0.45],
                                   "current_outstanding_balance": [100.0]}))
        assert out.start_value == pytest.approx(40.0)
        assert out.movement_value == pytest.approx(5.0)

    def test_the_percent_scale_is_decided_once_across_both_snapshots(self):
        """Deciding the scale per snapshot could read the start as a fraction and
        the end as points, manufacturing a 5,000-point movement."""
        e = entry("current_loan_to_value", aggregation="weighted_average",
                  weight_field="current_outstanding_balance")
        start = pd.DataFrame({"current_loan_to_value": [0.40] * 10,
                              "current_outstanding_balance": [100.0] * 10})
        end = pd.DataFrame({"current_loan_to_value": [0.42] * 10,
                            "current_outstanding_balance": [100.0] * 10})
        ctx = build_pair_context(e, [start, end])
        assert ctx.multiplier == 100.0
        assert aggregate(start, ctx).value == pytest.approx(40.0)
        assert aggregate(end, ctx).value == pytest.approx(42.0)

    def test_a_zero_start_value_yields_no_percentage_change(self):
        e = entry("amount", aggregation="sum")
        out = change(e, pd.DataFrame({"amount": [0.0]}),
                     pd.DataFrame({"amount": [100.0]}))
        assert out.movement_value == 100.0
        assert out.relative_change is None
        assert any("percentage change is undefined" in n for n in out.notes)

    def test_an_unknown_field_falls_back_to_the_governed_aggregation_not_a_name(self):
        e = entry("a_field_no_registry_knows", aggregation="sum")
        unit = resolve_unit(e)
        assert unit.source == SOURCE_AGGREGATION_DEFAULT
        assert unit.unit == m.UNIT_CURRENCY


# --------------------------------------------------------------------------- #
# Directionality — §12, all six controlled values
# --------------------------------------------------------------------------- #
class TestDirectionality:
    @pytest.mark.parametrize("directionality,rise,fall", [
        ("higher_is_better", m.INTERPRETATION_IMPROVEMENT, m.INTERPRETATION_DETERIORATION),
        ("higher_is_worse", m.INTERPRETATION_DETERIORATION, m.INTERPRETATION_IMPROVEMENT),
        ("lower_is_better", m.INTERPRETATION_DETERIORATION, m.INTERPRETATION_IMPROVEMENT),
        ("lower_is_worse", m.INTERPRETATION_IMPROVEMENT, m.INTERPRETATION_DETERIORATION),
        ("neutral", m.INTERPRETATION_NOT_ASSESSED, m.INTERPRETATION_NOT_ASSESSED),
        ("context_dependent", m.INTERPRETATION_NOT_ASSESSED, m.INTERPRETATION_NOT_ASSESSED),
    ])
    def test_every_controlled_directionality_value(self, directionality, rise, fall):
        assert interpret(directionality, 10.0) == rise
        assert interpret(directionality, -10.0) == fall

    def test_no_movement_is_neither_improvement_nor_deterioration(self):
        assert interpret("higher_is_worse", 0.0) == m.INTERPRETATION_NO_MOVEMENT

    def test_an_uncalculated_movement_is_not_assessed(self):
        assert interpret("higher_is_worse", None) == m.INTERPRETATION_NOT_ASSESSED

    def test_a_context_dependent_rate_rise_is_not_called_good_or_bad(self):
        e = entry("current_interest_rate", directionality="context_dependent",
                  aggregation="weighted_average",
                  weight_field="current_outstanding_balance")
        out = change(e,
                     pd.DataFrame({"current_interest_rate": [4.0],
                                   "current_outstanding_balance": [100.0]}),
                     pd.DataFrame({"current_interest_rate": [6.0],
                                   "current_outstanding_balance": [100.0]}))
        assert out.movement_value == pytest.approx(2.0)
        assert out.interpretation == m.INTERPRETATION_NOT_ASSESSED


# --------------------------------------------------------------------------- #
# The metric-change record — §15
# --------------------------------------------------------------------------- #
def test_a_metric_change_carries_its_full_semantic_provenance():
    e = entry("current_loan_to_value", aggregation="weighted_average",
              weight_field="current_outstanding_balance",
              directionality="higher_is_worse", concept="leverage")
    out = change(e,
                 pd.DataFrame({"current_loan_to_value": [40.0],
                               "current_outstanding_balance": [100.0]}),
                 pd.DataFrame({"current_loan_to_value": [45.0],
                               "current_outstanding_balance": [100.0]}))
    data = out.to_dict()
    for key in ("canonical_field", "display_name", "analytical_concept",
                "analytical_role", "temporality", "aggregation", "weight_field",
                "share_basis", "start_value", "end_value", "movement_value",
                "movement_unit", "relative_change", "directionality",
                "interpretation", "valid_population", "excluded_population",
                "evidence", "confidence", "rationale"):
        assert key in data, key
    assert data["weight_field"] == "current_outstanding_balance"
    assert data["rationale"] == "test entry"
    assert len(data["evidence"]) == 2
    # Evidence references the snapshot, never a loan-level value.
    assert data["evidence"][0]["snapshot_id"] == "s0"
    assert "loan_identifier" not in str(data["evidence"])
