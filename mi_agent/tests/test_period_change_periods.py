"""Governed period resolution: two snapshots, or an explicit failure.

The platform's reporting dates are irregular, so the resolver must never assume a
calendar month end, and must never silently compare the current snapshot with
itself. Every failure case in §5 of the workflow specification is exercised here.
"""

from __future__ import annotations

from datetime import date

import pytest

from mi_agent.period_change import models as m
from mi_agent.period_change.periods import (
    GRANULARITY_DAY,
    GRANULARITY_MONTH,
    PeriodRequest,
    order_snapshots,
    parse_period_token,
    resolve_periods,
    shift_months,
)


def snap(snapshot_id: str, reporting_date: str | None, *, frame=None,
         portfolios=()) -> m.SnapshotFrame:
    return m.SnapshotFrame(snapshot_id=snapshot_id, reporting_date=reporting_date,
                           frame=frame, portfolio_ids=tuple(portfolios))


#: A deliberately IRREGULAR series: a mid-month cut, a quarter end, a month end.
IRREGULAR = [
    snap("r1", "2025-06-30"), snap("r2", "2025-09-30"),
    snap("r3", "2025-12-31"), snap("r4", "2026-03-31"),
    snap("r5", "2026-06-15"),
]

MONTHLY = [snap(f"m{i}", d) for i, d in enumerate(
    ["2025-10-31", "2025-11-30", "2025-12-31", "2026-01-31", "2026-02-28",
     "2026-03-31", "2026-04-30", "2026-05-31", "2026-06-30"])]


# --------------------------------------------------------------------------- #
# Date helpers
# --------------------------------------------------------------------------- #
class TestDateHelpers:
    def test_a_month_end_stays_a_month_end_when_shifted(self):
        """31 March back one month is 28 February, not 28 March — otherwise a
        month-on-month request resolves to the wrong snapshot every long month."""
        assert shift_months(date(2026, 3, 31), 1) == date(2026, 2, 28)
        assert shift_months(date(2026, 5, 31), 1) == date(2026, 4, 30)
        assert shift_months(date(2026, 3, 31), 3) == date(2025, 12, 31)
        assert shift_months(date(2026, 3, 31), 12) == date(2025, 3, 31)

    def test_a_mid_month_date_keeps_its_day(self):
        assert shift_months(date(2026, 6, 15), 1) == date(2026, 5, 15)

    @pytest.mark.parametrize("token,expected,grain", [
        ("2026-03-31", date(2026, 3, 31), GRANULARITY_DAY),
        ("2026-03", date(2026, 3, 31), GRANULARITY_MONTH),
        ("31 March 2026", date(2026, 3, 31), GRANULARITY_DAY),
        ("31/03/2026", date(2026, 3, 31), GRANULARITY_DAY),
        ("March 2026", date(2026, 3, 31), GRANULARITY_MONTH),
        ("March", date(2026, 3, 31), GRANULARITY_MONTH),
        ("Jun", date(2026, 6, 30), GRANULARITY_MONTH),
    ])
    def test_period_tokens_resolve_to_a_target_date(self, token, expected, grain):
        got, got_grain = parse_period_token(token, reference=date(2026, 6, 15))
        assert (got, got_grain) == (expected, grain)

    def test_an_unparseable_token_resolves_to_nothing(self):
        assert parse_period_token("sometime last spring")[0] is None

    def test_snapshots_are_ordered_by_reporting_date_not_caller_order(self):
        shuffled = [IRREGULAR[3], IRREGULAR[0], IRREGULAR[4], IRREGULAR[1]]
        assert [s.snapshot_id for s in order_snapshots(shuffled)] == [
            "r1", "r2", "r4", "r5"]


# --------------------------------------------------------------------------- #
# Relative resolution
# --------------------------------------------------------------------------- #
class TestRelativeResolution:
    def test_no_request_resolves_to_the_latest_available_pair(self):
        out = resolve_periods(MONTHLY, PeriodRequest())
        assert out.resolution_method == m.METHOD_LATEST_AVAILABLE_PAIR
        assert (out.start_snapshot.reporting_date,
                out.end_snapshot.reporting_date) == ("2026-05-31", "2026-06-30")

    def test_current_versus_previous_uses_the_previous_snapshot_not_a_calendar(self):
        """On an irregular series "previous" means the previous GOVERNED snapshot,
        so no arithmetic can put it on a date the platform never reported."""
        out = resolve_periods(IRREGULAR, PeriodRequest(
            relative_mode=m.METHOD_CURRENT_VS_PREVIOUS))
        assert out.resolution_method == m.METHOD_CURRENT_VS_PREVIOUS
        assert out.start_snapshot.snapshot_id == "r4"
        assert out.end_snapshot.snapshot_id == "r5"
        assert not out.any_adjusted

    def test_month_on_month(self):
        out = resolve_periods(MONTHLY, PeriodRequest(
            relative_mode=m.METHOD_MONTH_ON_MONTH))
        assert (out.start_snapshot.reporting_date,
                out.end_snapshot.reporting_date) == ("2026-05-31", "2026-06-30")
        assert not out.start_adjusted

    def test_quarter_on_quarter(self):
        out = resolve_periods(MONTHLY, PeriodRequest(
            relative_mode=m.METHOD_QUARTER_ON_QUARTER))
        assert (out.start_snapshot.reporting_date,
                out.end_snapshot.reporting_date) == ("2026-03-31", "2026-06-30")

    def test_year_on_year(self):
        series = MONTHLY + [snap("y", "2026-07-31")]
        series = [snap("y0", "2025-07-31")] + series
        out = resolve_periods(series, PeriodRequest(
            relative_mode=m.METHOD_YEAR_ON_YEAR))
        assert (out.start_snapshot.reporting_date,
                out.end_snapshot.reporting_date) == ("2025-07-31", "2026-07-31")

    def test_year_to_date_opens_at_the_prior_year_end(self):
        out = resolve_periods(MONTHLY, PeriodRequest(
            relative_mode=m.METHOD_YEAR_TO_DATE))
        assert out.resolution_method == m.METHOD_YEAR_TO_DATE
        assert out.start_snapshot.reporting_date == "2025-12-31"
        assert out.end_snapshot.reporting_date == "2026-06-30"
        assert not out.start_adjusted

    def test_year_to_date_says_so_when_the_year_opening_is_missing(self):
        """A partial year must not be reported as a full one."""
        in_year_only = [s for s in MONTHLY if str(s.reporting_date).startswith("2026")]
        out = resolve_periods(in_year_only, PeriodRequest(
            relative_mode=m.METHOD_YEAR_TO_DATE))
        assert out.start_snapshot.reporting_date == "2026-01-31"
        assert out.start_adjusted
        assert any("part of the year only" in n for n in out.adjustment_notes)

    def test_a_relative_request_adjusts_to_the_nearest_irregular_snapshot(self):
        """2026-06-15 less one month is 2026-05-15, which the platform never
        reported; the nearest snapshot is used AND the adjustment is recorded."""
        out = resolve_periods(IRREGULAR, PeriodRequest(
            relative_mode=m.METHOD_MONTH_ON_MONTH))
        assert out.start_snapshot.snapshot_id == "r4"
        assert out.start_adjusted
        assert out.adjustment_notes


# --------------------------------------------------------------------------- #
# Explicit resolution
# --------------------------------------------------------------------------- #
class TestExplicitResolution:
    def test_explicit_iso_dates(self):
        out = resolve_periods(MONTHLY, PeriodRequest(
            requested_start="2025-12-31", requested_end="2026-03-31"))
        assert out.resolution_method == m.METHOD_EXPLICIT_DATES
        assert (out.start_snapshot.reporting_date,
                out.end_snapshot.reporting_date) == ("2025-12-31", "2026-03-31")
        assert not out.any_adjusted
        assert out.requested_start == "2025-12-31"

    def test_explicit_month_names_resolve_inside_the_latest_year(self):
        out = resolve_periods(MONTHLY, PeriodRequest(
            requested_start="March", requested_end="June"))
        assert (out.start_snapshot.reporting_date,
                out.end_snapshot.reporting_date) == ("2026-03-31", "2026-06-30")

    def test_a_requested_date_the_platform_never_reported_is_adjusted_and_recorded(self):
        out = resolve_periods(MONTHLY, PeriodRequest(
            requested_start="2026-01-20", requested_end="2026-06-30"))
        assert out.start_snapshot.reporting_date == "2026-01-31"
        assert out.start_adjusted is True
        assert out.end_adjusted is False
        assert any("adjusted to the nearest available snapshot" in n
                   for n in out.adjustment_notes)

    def test_only_an_end_date_compares_against_the_snapshot_before_it(self):
        out = resolve_periods(MONTHLY, PeriodRequest(requested_end="2026-03-31"))
        assert (out.start_snapshot.reporting_date,
                out.end_snapshot.reporting_date) == ("2026-02-28", "2026-03-31")

    def test_a_month_containing_several_snapshots_resolves_to_the_month_end(self):
        series = MONTHLY + [snap("mid", "2026-06-15")]
        out = resolve_periods(series, PeriodRequest(
            requested_start="March", requested_end="June"))
        assert out.end_snapshot.reporting_date == "2026-06-30"
        assert any("snapshots fall inside the requested end month"
                   in n for n in out.adjustment_notes)


# --------------------------------------------------------------------------- #
# Failure behaviour — §5
# --------------------------------------------------------------------------- #
class TestFailures:
    def test_fewer_than_two_snapshots_fails_explicitly(self):
        with pytest.raises(m.PeriodChangeFailure) as err:
            resolve_periods([snap("only", "2026-06-30")], PeriodRequest())
        assert err.value.reason == m.FAIL_INSUFFICIENT_SNAPSHOTS

    def test_identical_resolved_snapshots_never_compare_a_period_with_itself(self):
        with pytest.raises(m.PeriodChangeFailure) as err:
            resolve_periods(MONTHLY, PeriodRequest(
                requested_start="2026-06-30", requested_end="2026-06-30"))
        assert err.value.reason == m.FAIL_IDENTICAL_SNAPSHOTS

    def test_a_reversed_range_fails_rather_than_being_swapped(self):
        with pytest.raises(m.PeriodChangeFailure) as err:
            resolve_periods(MONTHLY, PeriodRequest(
                requested_start="2026-06-30", requested_end="2025-12-31"))
        assert err.value.reason == m.FAIL_REVERSED_PERIOD_RANGE

    def test_an_unparseable_period_is_ambiguous_not_guessed(self):
        with pytest.raises(m.PeriodChangeFailure) as err:
            resolve_periods(MONTHLY, PeriodRequest(requested_start="whenever"))
        assert err.value.reason == m.FAIL_AMBIGUOUS_PERIOD_RANGE

    def test_an_equidistant_day_precise_request_is_ambiguous(self):
        series = [snap("a", "2026-01-10"), snap("b", "2026-01-20"),
                  snap("c", "2026-06-30")]
        with pytest.raises(m.PeriodChangeFailure) as err:
            resolve_periods(series, PeriodRequest(
                requested_start="2026-01-15", requested_end="2026-06-30"))
        assert err.value.reason == m.FAIL_AMBIGUOUS_PERIOD_RANGE
        assert err.value.detail["equidistant"] == ["2026-01-10", "2026-01-20"]

    def test_a_portfolio_absent_at_one_date_fails_explicitly(self):
        series = [snap("a", "2026-05-31", portfolios=("p1",)),
                  snap("b", "2026-06-30", portfolios=("p1", "p2"))]
        with pytest.raises(m.PeriodChangeFailure) as err:
            resolve_periods(series, PeriodRequest(), scope=m.PortfolioScopeRef(
                portfolio_ids=("p1", "p2")))
        assert err.value.reason == m.FAIL_PORTFOLIO_ABSENT_AT_PERIOD
        assert err.value.detail["missing_at_start"] == ["p2"]

    def test_absent_provenance_is_not_reported_as_a_missing_portfolio(self):
        """A dataset with no source-portfolio provenance is an absence of
        evidence, not evidence that the portfolio is missing."""
        series = [snap("a", "2026-05-31"), snap("b", "2026-06-30")]
        out = resolve_periods(series, PeriodRequest(),
                              scope=m.PortfolioScopeRef(portfolio_ids=("p1",)))
        assert out.end_snapshot.snapshot_id == "b"


# --------------------------------------------------------------------------- #
# The resolution record — §5
# --------------------------------------------------------------------------- #
def test_the_resolution_records_everything_needed_to_reproduce_it():
    out = resolve_periods(MONTHLY, PeriodRequest(
        requested_start="2026-01-20", requested_end="2026-06-30"),
        scope=m.PortfolioScopeRef(tenant_id="t", label="Total"))
    data = out.to_dict()
    assert data["requested_start_period"] == "2026-01-20"
    assert data["requested_end_period"] == "2026-06-30"
    assert data["resolution_method"] == m.METHOD_EXPLICIT_DATES
    assert data["resolved_start_snapshot"]["reporting_date"] == "2026-01-31"
    assert data["resolved_end_snapshot"]["reporting_date"] == "2026-06-30"
    assert data["start_adjusted_to_available_snapshot"] is True
    assert data["end_adjusted_to_available_snapshot"] is False
    assert data["portfolio_scope"]["tenant_id"] == "t"
    assert len(data["available_snapshots"]) == len(MONTHLY)
