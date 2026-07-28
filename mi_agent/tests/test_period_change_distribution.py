"""Composition shift for governed dimensions — §13.

Shares are computed against each snapshot's OWN denominator, so a portfolio that
grew does not manufacture a share movement; categories that appear or disappear
are reported as such rather than dropped; and a missing value is a named
``Unknown`` bucket, not a silent exclusion.
"""

from __future__ import annotations

import pandas as pd
import pytest

from analytics_lib.stratify import UNKNOWN_LABEL

from mi_agent.period_change import models as m
from mi_agent.period_change.distribution import (
    PRESENCE_BOTH,
    PRESENCE_END_ONLY,
    PRESENCE_START_ONLY,
    distribution_change,
)

from .test_period_change_calculations import entry as make_entry, snap


def dimension(field="collateral_geography"):
    return make_entry(field, role="dimension", aggregation="distribution",
                      temporality="point_in_time", concept="geography",
                      directionality="context_dependent")


def frame(categories, balances=None):
    data = {"collateral_geography": list(categories)}
    if balances is not None:
        data["current_outstanding_balance"] = list(balances)
    return pd.DataFrame(data)


def shift(result, category):
    return next(c for c in result.categories if c.category == category)


def run(start, end):
    return distribution_change(dimension(), snap(start, "s0", "2026-03-31"),
                               snap(end, "s1", "2026-06-30"))


# --------------------------------------------------------------------------- #
# Counts and shares
# --------------------------------------------------------------------------- #
class TestCountShares:
    def test_reports_counts_and_count_shares_at_both_dates(self):
        out = run(frame(["London", "London", "Wales", "Wales"]),
                  frame(["London", "London", "London", "Wales"]))
        london = shift(out, "London")
        assert (london.start_count, london.end_count) == (2, 3)
        assert london.start_count_share == pytest.approx(0.5)
        assert london.end_count_share == pytest.approx(0.75)
        assert london.count_share_movement == pytest.approx(0.25)
        assert london.presence == PRESENCE_BOTH

    def test_shares_are_relative_to_each_snapshots_own_total(self):
        """The book doubled but the mix is unchanged, so no share moved."""
        out = run(frame(["London", "Wales"]),
                  frame(["London", "London", "Wales", "Wales"]))
        for category in ("London", "Wales"):
            assert shift(out, category).count_share_movement == pytest.approx(0.0)
        assert (out.start_total_count, out.end_total_count) == (2, 4)
        assert any("population changed" in n for n in out.notes)

    def test_shares_sum_to_one_within_each_snapshot(self):
        out = run(frame(["A", "B", "C", "C"]), frame(["A", "B", "B", "C"]))
        assert sum(c.start_count_share for c in out.categories) == pytest.approx(1.0)
        assert sum(c.end_count_share for c in out.categories) == pytest.approx(1.0)

    def test_ranks_the_largest_positive_and_negative_shifts(self):
        out = run(frame(["A", "A", "B", "C"]), frame(["A", "B", "B", "B"]))
        assert out.largest_increases[0] == "B"
        assert out.largest_decreases[0] == "A"


# --------------------------------------------------------------------------- #
# Balance shares
# --------------------------------------------------------------------------- #
class TestBalanceShares:
    def test_balance_shares_where_the_governed_balance_field_exists(self):
        out = run(frame(["London", "Wales"], [300.0, 100.0]),
                  frame(["London", "Wales"], [100.0, 300.0]))
        london = shift(out, "London")
        assert london.start_balance_share == pytest.approx(0.75)
        assert london.end_balance_share == pytest.approx(0.25)
        assert london.balance_share_movement == pytest.approx(-0.5)
        assert out.balance_field == "current_outstanding_balance"

    def test_balance_shares_are_omitted_and_explained_when_unsupported(self):
        out = run(frame(["London", "Wales"]), frame(["London", "Wales"]))
        assert out.balance_field is None
        assert all(c.balance_share_movement is None for c in out.categories)
        assert any("Balance shares are not reported" in n for n in out.notes)

    def test_count_share_and_balance_share_can_move_in_opposite_directions(self):
        """Which is exactly why both are reported rather than one standing in
        for the other."""
        out = run(frame(["London", "Wales", "Wales"], [900.0, 50.0, 50.0]),
                  frame(["London", "Wales", "Wales", "Wales"], [100.0, 300.0, 300.0, 300.0]))
        london = shift(out, "London")
        assert london.count_share_movement < 0
        assert london.balance_share_movement < 0
        wales = shift(out, "Wales")
        assert wales.count_share_movement > 0 and wales.balance_share_movement > 0


# --------------------------------------------------------------------------- #
# Category churn and missing values
# --------------------------------------------------------------------------- #
class TestCategoryChurn:
    def test_a_category_appearing_only_in_the_closing_snapshot(self):
        out = run(frame(["London"]), frame(["London", "Scotland"]))
        scotland = shift(out, "Scotland")
        assert scotland.presence == PRESENCE_END_ONLY
        assert (scotland.start_count, scotland.end_count) == (0, 2 - 1)
        assert scotland.start_count_share == pytest.approx(0.0)
        assert any("only in the closing snapshot" in n for n in out.notes)

    def test_a_category_disappearing_after_the_opening_snapshot(self):
        out = run(frame(["London", "Scotland"]), frame(["London"]))
        scotland = shift(out, "Scotland")
        assert scotland.presence == PRESENCE_START_ONLY
        assert scotland.end_count == 0
        assert scotland.count_share_movement == pytest.approx(-0.5)
        assert any("only in the opening snapshot" in n for n in out.notes)

    def test_missing_values_become_a_named_unknown_bucket_not_a_silent_drop(self):
        out = run(frame(["London", None]), frame(["London", ""]))
        unknown = shift(out, UNKNOWN_LABEL)
        assert (unknown.start_count, unknown.end_count) == (1, 1)
        assert out.status == m.STATUS_PARTIALLY_AVAILABLE
        assert any("carry no value for this dimension" in n for n in out.notes)

    def test_blank_and_null_are_the_same_category_not_two(self):
        """A value that is blank at one date and null at the other must not read
        as a category that appeared."""
        out = run(frame(["London", None]), frame(["London", "  "]))
        assert [c.category for c in out.categories] == ["London", UNKNOWN_LABEL]
        assert shift(out, UNKNOWN_LABEL).presence == PRESENCE_BOTH

    def test_surrounding_whitespace_does_not_split_a_category(self):
        out = run(frame(["London", " London "]), frame(["London", "London"]))
        assert [c.category for c in out.categories] == ["London"]


# --------------------------------------------------------------------------- #
# Availability and discipline
# --------------------------------------------------------------------------- #
class TestAvailabilityAndDiscipline:
    def test_a_dimension_present_at_only_one_date_is_not_comparable(self):
        out = distribution_change(
            dimension(), snap(pd.DataFrame({"other": [1]}), "s0", "2026-03-31"),
            snap(frame(["London"]), "s1", "2026-06-30"))
        assert out.status == m.STATUS_NOT_COMPARABLE_DUE_TO_AVAILABILITY
        assert out.categories == ()

    def test_a_dimension_present_at_neither_date_is_not_available(self):
        empty = pd.DataFrame({"other": [1]})
        out = distribution_change(dimension(), snap(empty, "s0"), snap(empty, "s1"))
        assert out.status == m.STATUS_NOT_AVAILABLE

    def test_no_concentration_index_is_invented(self):
        """§13 explicitly: concentration belongs to the existing risk-monitor
        service, and a second implementation would produce a second answer."""
        out = run(frame(["A", "B"]), frame(["A", "B"]))
        data = out.to_dict()
        assert not any("concentration" in key.lower() for key in data)

    def test_a_category_increase_is_not_labelled_deterioration(self):
        """No governed ordering exists here, so the shift is reported and left
        as an observation."""
        out = run(frame(["Performing", "Arrears"]),
                  frame(["Performing", "Arrears", "Arrears"]))
        data = out.to_dict()
        assert "interpretation" not in data
        assert not any("deterioration" in str(v) for v in data.values())

    def test_the_result_records_its_evidence_without_loan_level_data(self):
        out = run(frame(["London"], [100.0]), frame(["London"], [200.0]))
        assert len(out.evidence) == 2
        assert all("snapshot_id" in e for e in out.evidence)
        assert "loan_identifier" not in str(out.evidence)
