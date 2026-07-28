"""The Period Change Analysis workflow end to end.

Covers the orchestration the individual modules cannot: mode behaviour, ranking,
the deterministic summary, governance and tenancy, the audit contract, and the
worked examples the documentation quotes.
"""

from __future__ import annotations

import pandas as pd
import pytest

from mi_agent.business_semantics import BusinessSemanticsRegistry
from mi_agent.period_change import models as m
from mi_agent.period_change.periods import PeriodRequest
from mi_agent.period_change.workflow import (
    PeriodChangeRequest,
    check_scope_access,
    run_period_change_analysis,
)

from .test_period_change_calculations import entry as make_entry


# --------------------------------------------------------------------------- #
# Fixtures — a small funded book with the columns the governed metrics read
# --------------------------------------------------------------------------- #
def book(*, loans, balance, ltv, arrears, status, recoveries=100.0,
         cumulative=1000.0, region="London") -> pd.DataFrame:
    return pd.DataFrame({
        "loan_identifier": [f"L{i:04d}" for i in range(loans)],
        "current_outstanding_balance": [balance / loans] * loans,
        "current_loan_to_value": [ltv] * loans,
        "current_interest_rate": [5.0] * loans,
        "interest_in_arrears": [("Y" if i < arrears else "N") for i in range(loans)],
        "account_status": [status[i % len(status)] for i in range(loans)],
        "collateral_geography": [region] * loans,
        "recoveries_in_period": [recoveries / loans] * loans,
        "cumulative_recoveries": [cumulative / loans] * loans,
    })


@pytest.fixture
def snapshots():
    start = book(loans=10, balance=1_000_000.0, ltv=0.40, arrears=1,
                 status=["Performing", "Arrears"], recoveries=1_000.0,
                 cumulative=10_000.0)
    end = book(loans=12, balance=1_200_000.0, ltv=0.44, arrears=3,
               status=["Performing", "Arrears"], recoveries=1_500.0,
               cumulative=12_000.0)
    return [
        m.SnapshotFrame("mi_2026_03", "2026-03-31", start,
                        dataset_label="18_central_lender_tape.csv",
                        dataset_reference="mi_2026_03", row_count=10),
        m.SnapshotFrame("mi_2026_06", "2026-06-30", end,
                        dataset_label="18_central_lender_tape.csv",
                        dataset_reference="mi_2026_06", row_count=12),
    ]


def request(**kw) -> PeriodChangeRequest:
    kw.setdefault("scope", m.PortfolioScopeRef(
        tenant_id="tenant_a", label="Total", asset_classes=("cross_asset",)))
    return PeriodChangeRequest(**kw)


def metric(result, field):
    return next(x for x in result.metric_changes if x.field == field)


# --------------------------------------------------------------------------- #
# Modes — §10
# --------------------------------------------------------------------------- #
class TestModes:
    def test_portfolio_overview_reports_a_governed_spread(self, snapshots):
        out = run_period_change_analysis(request(), snapshots)
        assert out.workflow_id == m.WORKFLOW_ID
        assert len(out.metric_changes) >= 4
        concepts = {x.analytical_concept for x in out.metric_changes}
        assert len(concepts) >= 3
        assert out.distribution_changes

    def test_requested_metric_mode_analyses_only_what_was_asked(self, snapshots):
        out = run_period_change_analysis(request(
            mode=m.MODE_REQUESTED_METRIC,
            requested_fields=("current_loan_to_value",)), snapshots)
        assert [x.field for x in out.metric_changes] == ["current_loan_to_value"]

    def test_concept_mode_selects_under_the_requested_concept(self, snapshots):
        out = run_period_change_analysis(request(
            mode=m.MODE_CONCEPT,
            requested_concepts=("payment_performance",)), snapshots)
        assert out.metric_changes
        assert {x.analytical_concept for x in out.metric_changes} == \
            {"payment_performance"}

    def test_an_unavailable_requested_metric_fails_rather_than_substituting(
            self, snapshots):
        """§2: never silently substitute another metric."""
        with pytest.raises(m.PeriodChangeFailure) as err:
            run_period_change_analysis(request(
                mode=m.MODE_REQUESTED_METRIC,
                requested_fields=("debt_to_income_ratio",)), snapshots)
        assert err.value.reason == m.FAIL_NO_ELIGIBLE_FIELDS


# --------------------------------------------------------------------------- #
# Worked examples — the eight cases the documentation quotes
# --------------------------------------------------------------------------- #
class TestWorkedExamples:
    @pytest.fixture
    def result(self, snapshots):
        return run_period_change_analysis(request(), snapshots)

    def test_current_balance_is_a_point_in_time_stock(self, result):
        change = metric(result, "current_outstanding_balance")
        assert change.temporality == "point_in_time"
        assert change.movement_basis == m.BASIS_POINT_IN_TIME_DIFFERENCE
        assert change.start_value == pytest.approx(1_000_000.0)
        assert change.end_value == pytest.approx(1_200_000.0)
        assert change.movement_value == pytest.approx(200_000.0)
        assert change.relative_change == pytest.approx(0.2)
        assert change.movement_unit == m.UNIT_CURRENCY

    def test_recoveries_in_period_is_a_flow_comparison(self, result):
        change = metric(result, "recoveries_in_period")
        assert change.movement_basis == m.BASIS_FLOW_LEVEL_COMPARISON
        assert change.start_value == pytest.approx(1_000.0)
        assert change.end_value == pytest.approx(1_500.0)
        assert m.FLOW_COMPARISON_NOTE in change.notes

    def test_cumulative_recoveries_is_a_differenced_cumulative(self, snapshots):
        """Cumulative recoveries are not two independent period totals: the
        movement IS the difference of the two cumulative positions."""
        out = run_period_change_analysis(request(
            mode=m.MODE_REQUESTED_METRIC,
            requested_fields=("cumulative_recoveries",)), snapshots)
        change = metric(out, "cumulative_recoveries")
        assert change.temporality == "cumulative"
        assert change.movement_basis == m.BASIS_CUMULATIVE_DIFFERENCE
        assert change.start_value == pytest.approx(10_000.0)
        assert change.end_value == pytest.approx(12_000.0)
        assert change.movement_value == pytest.approx(2_000.0)

    def test_current_ltv_is_a_balance_weighted_average_in_points(self, result):
        change = metric(result, "current_loan_to_value")
        assert change.aggregation == "weighted_average"
        assert change.weight_field == "current_outstanding_balance"
        assert change.start_value == pytest.approx(40.0)
        assert change.end_value == pytest.approx(44.0)
        assert change.movement_value == pytest.approx(4.0)
        assert change.movement_unit == m.UNIT_PERCENTAGE_POINT
        assert change.basis_point_change == pytest.approx(400.0)

    def test_the_arrears_flag_is_a_count_share(self, result):
        change = metric(result, "interest_in_arrears")
        assert change.aggregation == "share"
        assert change.share_basis == "count"
        assert change.start_value == pytest.approx(10.0)     # 1 of 10
        assert change.end_value == pytest.approx(25.0)       # 3 of 12
        assert change.start.numerator == 1.0 and change.start.denominator == 10.0
        assert change.interpretation == m.INTERPRETATION_DETERIORATION

    def test_a_dimension_produces_a_distribution_shift(self, result):
        dist = next(d for d in result.distribution_changes
                    if d.field == "account_status")
        assert {c.category for c in dist.categories} == {"Performing", "Arrears"}
        assert all(c.count_share_movement is not None for c in dist.categories)
        assert dist.balance_field == "current_outstanding_balance"

    def test_the_balance_bridge_reconciles(self, result):
        bridge = result.balance_bridge
        assert bridge.status == m.BRIDGE_STATUS_AVAILABLE
        assert bridge.new_loan_count == 2
        assert bridge.exited_loan_count == 0
        assert bridge.continuing_loan_count == 10
        assert bridge.reconciles is True

    def test_an_original_balance_static_baseline_is_not_ranked(self, snapshots):
        """The committed registry tags no static baseline for period change, so
        the rule is proved against a purpose-built registry."""
        baseline = make_entry("original_loan_to_value", temporality="static_baseline",
                              concept="leverage")
        registry = BusinessSemanticsRegistry(
            version="t", schema_version=2, source_registry="t",
            entries={baseline.field: baseline}, taxonomy={})
        for snapshot in snapshots:
            snapshot.frame["original_loan_to_value"] = 0.5
        with pytest.raises(m.PeriodChangeFailure) as err:
            run_period_change_analysis(request(), snapshots, registry=registry)
        assert err.value.reason == m.FAIL_NO_ELIGIBLE_FIELDS


# --------------------------------------------------------------------------- #
# Ranking and significance — §11
# --------------------------------------------------------------------------- #
class TestRankingAndSignificance:
    @pytest.fixture
    def result(self, snapshots):
        return run_period_change_analysis(request(), snapshots)

    def test_movements_are_ranked_within_their_unit_never_across_units(self, result):
        """A currency movement and a percentage-point movement have no common
        scale. Each unit gets its own 1..n sequence."""
        from collections import defaultdict

        by_unit = defaultdict(list)
        for change in result.metric_changes:
            if change.movement_rank is not None:
                by_unit[change.movement_unit].append(change.movement_rank)
        assert len(by_unit) > 1, "fixture must exercise more than one unit"
        for unit, ranks in by_unit.items():
            assert sorted(ranks) == list(range(1, len(ranks) + 1)), unit
        for change in result.metric_changes:
            if change.movement_rank is not None:
                assert change.rank_population == len(by_unit[change.movement_unit])

    def test_a_tiny_currency_move_does_not_outrank_a_large_rate_move(self):
        """The defect this replaced: a +0.1% balance drift was ranked above a
        +15-point arrears rise and labelled the largest observed increase."""
        start = book(loans=100, balance=1_000_000.0, ltv=0.40, arrears=2,
                     status=["Performing"])
        end = book(loans=100, balance=1_001_000.0, ltv=0.48, arrears=17,
                   status=["Performing"])
        out = run_period_change_analysis(request(), [
            m.SnapshotFrame("a", "2026-03-31", start),
            m.SnapshotFrame("b", "2026-06-30", end)])

        balance = metric(out, "current_outstanding_balance")
        arrears = metric(out, "interest_in_arrears")
        assert balance.movement_unit == m.UNIT_CURRENCY
        assert arrears.movement_unit == m.UNIT_PERCENTAGE_POINT
        # Each is the largest observed increase IN ITS OWN UNIT, and both say so.
        assert balance.significance == m.SIGNIFICANCE_LARGEST_INCREASE
        assert arrears.significance == m.SIGNIFICANCE_LARGEST_INCREASE
        assert balance.movement_rank == 1 and arrears.movement_rank == 1
        increases = {row["canonical_field"]
                     for row in out.summary["largest_observed_increases"]}
        assert {"current_outstanding_balance", "interest_in_arrears"} <= increases

    def test_the_largest_increase_and_decrease_are_named_per_unit(self, result):
        significances = {x.significance for x in result.metric_changes}
        assert m.SIGNIFICANCE_LARGEST_INCREASE in significances
        for row in result.summary["largest_observed_increases"]:
            assert row["movement_rank_scope"] == row["movement_unit"]
        assert m.RANK_SCOPE_NOTE == result.summary["rank_scope"]

    def test_no_movement_is_called_material_or_a_breach(self, result):
        # The disclaimer itself names the words it refuses to use, so it is
        # removed before the check rather than weakening the check.
        text = str(result.to_dict()).lower().replace(
            m.MATERIALITY_DISCLAIMER.lower(), "")
        for forbidden in ("is material", "materially significant", "breach",
                          "high risk", "significant movement"):
            assert forbidden not in text
        assert m.MATERIALITY_DISCLAIMER in result.limitations

    def test_the_significance_vocabulary_is_controlled(self, result):
        allowed = {m.SIGNIFICANCE_LARGEST_INCREASE, m.SIGNIFICANCE_LARGEST_DECREASE,
                   m.SIGNIFICANCE_NOTABLE_BY_RANK, m.SIGNIFICANCE_RELATIVELY_STABLE,
                   m.SIGNIFICANCE_MAIN_OBSERVED_MOVEMENT, m.SIGNIFICANCE_NO_BASIS}
        assert {x.significance for x in result.metric_changes} <= allowed

    def test_currency_movements_state_their_contribution_to_the_balance_change(
            self, result):
        balance = metric(result, "current_outstanding_balance")
        assert balance.contribution_to_balance_change == pytest.approx(1.0)


# --------------------------------------------------------------------------- #
# The deterministic summary — §15
# --------------------------------------------------------------------------- #
class TestSummary:
    @pytest.fixture
    def result(self, snapshots):
        return run_period_change_analysis(request(), snapshots)

    def test_the_summary_only_restates_calculated_values(self, result):
        calculated = {x.field: x.movement_value for x in result.metric_changes}
        for rows in result.summary["top_movements_by_unit"].values():
            for row in rows:
                assert row["canonical_field"] in calculated
                assert row["movement_value"] == calculated[row["canonical_field"]]

    def test_top_movements_are_grouped_by_unit_not_flattened(self, result):
        by_unit = result.summary["top_movements_by_unit"]
        assert len(by_unit) > 1
        for unit, rows in by_unit.items():
            assert all(row["movement_unit"] == unit for row in rows)

    def test_improvements_and_deteriorations_come_from_directionality(self, result):
        improvements = {x.field for x in result.metric_changes
                        if x.interpretation == m.INTERPRETATION_IMPROVEMENT}
        assert set(result.summary["improvements"]) == improvements

    def test_the_summary_states_the_resolved_period(self, result):
        assert result.summary["period"]["start"] == "2026-03-31"
        assert result.summary["period"]["end"] == "2026-06-30"
        assert result.summary["period"]["resolution_method"] == \
            m.METHOD_LATEST_AVAILABLE_PAIR

    def test_the_summary_carries_the_materiality_boundary(self, result):
        assert result.summary["materiality"] == m.MATERIALITY_DISCLAIMER


# --------------------------------------------------------------------------- #
# Governance and tenancy — §20
# --------------------------------------------------------------------------- #
class TestGovernance:
    def test_the_scope_is_taken_from_the_execution_context(self, snapshots):
        out = run_period_change_analysis(request(
            scope=m.PortfolioScopeRef(tenant_id="tenant_a", label="Direct",
                                      asset_classes=("cross_asset",))), snapshots)
        assert out.portfolio_scope.tenant_id == "tenant_a"
        assert out.to_dict()["portfolio_scope"]["tenant_id"] == "tenant_a"

    def test_a_caller_may_narrow_within_the_authorised_scope(self):
        scope = check_scope_access(PeriodChangeRequest(
            scope=m.PortfolioScopeRef(portfolio_ids=("p1",)),
            authorised_portfolio_ids=("p1", "p2")))
        assert scope.portfolio_ids == ("p1",)

    def test_a_caller_may_not_widen_beyond_the_authorised_scope(self):
        with pytest.raises(m.PeriodChangeFailure) as err:
            check_scope_access(PeriodChangeRequest(
                scope=m.PortfolioScopeRef(portfolio_ids=("p1", "p3")),
                authorised_portfolio_ids=("p1", "p2")))
        assert err.value.reason == m.FAIL_CROSS_TENANT_ACCESS
        assert err.value.detail["requested_outside_scope"] == ["p3"]

    def test_an_unnarrowed_request_defaults_to_the_authorised_scope(self):
        scope = check_scope_access(PeriodChangeRequest(
            authorised_portfolio_ids=("p2", "p1")))
        assert scope.portfolio_ids == ("p1", "p2")

    def test_a_conflicting_scope_fails_closed_rather_than_intersecting(self):
        """Quietly answering for the intersection would give the caller a
        different answer to the one they asked for, without saying so."""
        with pytest.raises(m.PeriodChangeFailure):
            check_scope_access(PeriodChangeRequest(
                scope=m.PortfolioScopeRef(portfolio_ids=("other_tenant_p",)),
                authorised_portfolio_ids=("p1",)))


# --------------------------------------------------------------------------- #
# Audit and evidence — §16
# --------------------------------------------------------------------------- #
class TestAudit:
    @pytest.fixture
    def payload(self, snapshots):
        return run_period_change_analysis(request(), snapshots).to_dict()

    def test_the_result_contract_carries_every_required_block(self, payload):
        for key in ("workflow_id", "request_interpretation", "portfolio_scope",
                    "period_resolution", "dataset_provenance", "summary",
                    "metric_changes", "distribution_changes", "balance_bridge",
                    "warnings", "limitations", "evidence", "audit"):
            assert key in payload, key

    def test_the_audit_identifies_the_registry_and_calculation_versions(self, payload):
        audit = payload["audit"]
        assert audit["business_semantics"]["registry_version"] == "0.2.0"
        assert audit["business_semantics"]["schema_version"] == 2
        assert audit["calculation_version"] == m.CALCULATION_VERSION
        assert audit["selection_policy"]["name"]

    def test_the_audit_records_selected_and_excluded_entries_with_reasons(self, payload):
        audit = payload["audit"]
        assert audit["selected_entries"]
        assert audit["excluded_candidates"]
        assert all(e["reason"] for e in audit["excluded_candidates"])

    def test_the_audit_records_the_numerators_denominators_and_row_counts(self, payload):
        entry = next(e for e in payload["audit"]["selected_entries"]
                     if e["canonical_field"] == "interest_in_arrears")
        assert entry["numerator"]["start"] == 1.0
        assert entry["denominator"]["start"] == 10.0
        assert entry["valid_rows"]["end"] == 12

    def test_the_audit_records_the_resolved_snapshots_and_data_sources(self, payload):
        audit = payload["audit"]
        assert audit["resolved_snapshots"]["start"]["snapshot_id"] == "mi_2026_03"
        assert audit["resolved_snapshots"]["end"]["snapshot_id"] == "mi_2026_06"
        assert audit["data_source_identifiers"] == ["mi_2026_03", "mi_2026_06"]

    def test_no_loan_level_data_appears_in_the_audit_or_evidence(self, payload):
        """§16: the audit reproduces a result; it does not carry the book."""
        text = str(payload["audit"]) + str(payload["evidence"])
        assert "L0000" not in text
        assert "loan_identifier" not in text.replace(
            "'identifier_field'", "")  # the bridge names its KEY, not its values

    def test_every_metric_carries_evidence_referencing_both_snapshots(self, payload):
        for change in payload["metric_changes"]:
            snapshots = {e["snapshot_id"] for e in change["evidence"]}
            assert snapshots == {"mi_2026_03", "mi_2026_06"}


# --------------------------------------------------------------------------- #
# Source overrides — §18
# --------------------------------------------------------------------------- #
def test_a_governed_source_override_changes_temporality_without_code_changes(
        snapshots):
    """``allocated_losses`` is governed as cumulative under the ESMA reading. A
    source that reports it as a period figure is corrected by CONFIGURATION."""
    from mi_agent.business_semantics import load_business_semantics

    for snapshot in snapshots:
        snapshot.frame["allocated_losses"] = 500.0

    base = load_business_semantics()
    assert base.get("allocated_losses").temporality == "cumulative"

    overridden = base.for_source(
        "client_x", {"client_x": {"allocated_losses": {"temporality": "period_flow"}}})
    out = run_period_change_analysis(
        request(mode=m.MODE_REQUESTED_METRIC,
                requested_fields=("allocated_losses",)),
        snapshots, registry=overridden)
    change = metric(out, "allocated_losses")
    assert change.movement_basis == m.BASIS_FLOW_LEVEL_COMPARISON
    assert out.audit["business_semantics"]["overridden_fields"] == ["allocated_losses"]


# --------------------------------------------------------------------------- #
# Failure behaviour
# --------------------------------------------------------------------------- #
class TestFailures:
    def test_a_single_snapshot_fails_explicitly(self, snapshots):
        with pytest.raises(m.PeriodChangeFailure) as err:
            run_period_change_analysis(request(), snapshots[:1])
        assert err.value.reason == m.FAIL_INSUFFICIENT_SNAPSHOTS

    def test_a_reversed_explicit_range_fails(self, snapshots):
        with pytest.raises(m.PeriodChangeFailure) as err:
            run_period_change_analysis(request(period_request=PeriodRequest(
                requested_start="2026-06-30", requested_end="2026-03-31")),
                snapshots)
        assert err.value.reason == m.FAIL_REVERSED_PERIOD_RANGE

    def test_the_current_snapshot_is_never_compared_with_itself(self, snapshots):
        with pytest.raises(m.PeriodChangeFailure) as err:
            run_period_change_analysis(request(period_request=PeriodRequest(
                requested_start="2026-06-30", requested_end="2026-06-30")),
                snapshots)
        assert err.value.reason == m.FAIL_IDENTICAL_SNAPSHOTS

    def test_every_failure_reason_has_governed_wording(self):
        for reason, message in m.FAILURE_MESSAGES.items():
            assert message and reason
            assert m.PeriodChangeFailure(reason).message == message


# --------------------------------------------------------------------------- #
# Data-quality behaviour — §17
# --------------------------------------------------------------------------- #
class TestDataQuality:
    def test_a_falling_cumulative_value_surfaces_a_workflow_warning(self, snapshots):
        snapshots[1].frame["cumulative_recoveries"] = 1.0    # below the opening
        out = run_period_change_analysis(request(
            mode=m.MODE_REQUESTED_METRIC,
            requested_fields=("cumulative_recoveries",)), snapshots)
        assert any("cumulative value fell" in w for w in out.warnings)
        assert metric(out, "cumulative_recoveries").movement_value < 0

    def test_a_metric_available_at_only_one_date_is_reported_as_such(self, snapshots):
        snapshots[1].frame["debt_to_income_ratio"] = 3.0
        out = run_period_change_analysis(request(
            mode=m.MODE_REQUESTED_METRIC,
            requested_fields=("debt_to_income_ratio",)), snapshots)
        change = metric(out, "debt_to_income_ratio")
        assert change.status == m.STATUS_NOT_COMPARABLE_DUE_TO_AVAILABILITY
        assert any("only one of the two dates" in l for l in out.limitations)

    def test_a_period_adjustment_becomes_a_warning(self, snapshots):
        out = run_period_change_analysis(request(period_request=PeriodRequest(
            requested_start="2026-04-15", requested_end="2026-06-30")), snapshots)
        assert out.period_resolution.start_adjusted
        assert out.period_resolution.start_gap_days == 15
        assert any("at or before it" in w for w in out.warnings)

    def test_a_period_flow_over_unequal_reporting_periods_is_not_compared(self):
        """A monthly flow and a quarterly flow are not two comparable period
        totals; differencing them reports a change in the length of the window."""
        series = [
            m.SnapshotFrame("q", "2026-01-31", book(
                loans=10, balance=100.0, ltv=0.4, arrears=0,
                status=["Performing"], recoveries=100.0)),
            m.SnapshotFrame("a", "2026-04-30", book(          # 89-day period
                loans=10, balance=100.0, ltv=0.4, arrears=0,
                status=["Performing"], recoveries=300.0)),
            m.SnapshotFrame("b", "2026-05-31", book(          # 31-day period
                loans=10, balance=100.0, ltv=0.4, arrears=0,
                status=["Performing"], recoveries=120.0)),
        ]
        out = run_period_change_analysis(request(
            mode=m.MODE_REQUESTED_METRIC,
            requested_fields=("recoveries_in_period",),
            period_request=PeriodRequest(requested_start="2026-04-30",
                                         requested_end="2026-05-31")), series)
        change = metric(out, "recoveries_in_period")
        assert change.status == m.STATUS_NOT_COMPARABLE_PERIOD_BASIS
        assert change.movement_value is None
        assert change.start_value == 300.0 and change.end_value == 120.0
        assert change.flow_basis.start_period_days == 89
        assert change.flow_basis.end_period_days == 31
        assert m.FLOW_BASIS_MISMATCH_NOTE in change.notes

    def test_equal_reporting_periods_leave_a_flow_comparable(self):
        series = [
            m.SnapshotFrame("a", "2026-03-31", book(
                loans=10, balance=100.0, ltv=0.4, arrears=0,
                status=["Performing"], recoveries=100.0)),
            m.SnapshotFrame("b", "2026-04-30", book(
                loans=10, balance=100.0, ltv=0.4, arrears=0,
                status=["Performing"], recoveries=200.0)),
            m.SnapshotFrame("c", "2026-05-31", book(
                loans=10, balance=100.0, ltv=0.4, arrears=0,
                status=["Performing"], recoveries=300.0)),
        ]
        out = run_period_change_analysis(request(
            mode=m.MODE_REQUESTED_METRIC,
            requested_fields=("recoveries_in_period",)), series)
        change = metric(out, "recoveries_in_period")
        assert change.flow_basis.comparable is True
        assert change.movement_value == 100.0
        assert change.status == m.STATUS_AVAILABLE

    def test_an_unknown_period_length_states_the_uncertainty_without_disqualifying(self):
        """With only two snapshots the opening period length cannot be known.
        That is reported, not treated as a defect."""
        series = [
            m.SnapshotFrame("a", "2026-03-31", book(
                loans=10, balance=100.0, ltv=0.4, arrears=0,
                status=["Performing"], recoveries=100.0)),
            m.SnapshotFrame("b", "2026-04-30", book(
                loans=10, balance=100.0, ltv=0.4, arrears=0,
                status=["Performing"], recoveries=200.0)),
        ]
        out = run_period_change_analysis(request(
            mode=m.MODE_REQUESTED_METRIC,
            requested_fields=("recoveries_in_period",)), series)
        change = metric(out, "recoveries_in_period")
        assert change.flow_basis.comparable is None
        assert change.movement_value == 100.0
        assert m.FLOW_BASIS_UNKNOWN_NOTE in change.notes

    def test_an_overview_states_that_it_is_a_governed_subset(self, snapshots):
        out = run_period_change_analysis(request(), snapshots)
        assert any("governed subset of the registry" in l for l in out.limitations)


# --------------------------------------------------------------------------- #
# Determinism
# --------------------------------------------------------------------------- #
def test_the_same_inputs_always_produce_the_same_result(snapshots):
    first = run_period_change_analysis(request(), snapshots).to_dict()
    second = run_period_change_analysis(request(), snapshots).to_dict()
    assert first == second


# --------------------------------------------------------------------------- #
# Mixed currency — §17. Suppression must reach every monetary output.
# --------------------------------------------------------------------------- #
def test_mixed_currency_suppresses_every_monetary_output_not_just_metrics():
    """The defect this replaced: monetary metrics were suppressed while the
    balance bridge happily added GBP to EUR and reported "reconciles: true"."""
    def mixed(n, balance, ltv, arrears):
        frame = book(loans=n, balance=balance, ltv=ltv, arrears=arrears,
                     status=["Performing", "Arrears"])
        frame["exposure_currency_denomination"] = [
            ("GBP" if i % 2 else "EUR") for i in range(n)]
        return frame

    out = run_period_change_analysis(request(), [
        m.SnapshotFrame("a", "2026-03-31", mixed(10, 1_000_000.0, 0.40, 1)),
        m.SnapshotFrame("b", "2026-06-30", mixed(12, 1_200_000.0, 0.44, 3))])

    balance = metric(out, "current_outstanding_balance")
    assert balance.status == m.STATUS_NOT_COMPARABLE_MIXED_CURRENCY
    assert balance.movement_value is None

    assert out.balance_bridge.status == m.BRIDGE_STATUS_UNAVAILABLE_MIXED_CURRENCY
    assert out.balance_bridge.opening_balance is None
    assert out.balance_bridge.reconciles is None

    for dist in out.distribution_changes:
        assert dist.balance_field is None
        assert all(c.balance_share_movement is None for c in dist.categories)

    # Non-monetary metrics remain valid and are still answered.
    arrears = metric(out, "interest_in_arrears")
    assert arrears.status in (m.STATUS_AVAILABLE, m.STATUS_PARTIALLY_AVAILABLE)
    assert arrears.movement_value is not None


# --------------------------------------------------------------------------- #
# Summary language — §15. Coincidence is never reported as cause.
# --------------------------------------------------------------------------- #
def test_the_summary_never_asserts_causation(snapshots):
    """Only the balance bridge can attribute a balance movement. Nothing in the
    summary may link a metric movement to a composition shift."""
    out = run_period_change_analysis(request(), snapshots)
    # The fixed methodological notes explain the METHOD ("ranked within a unit
    # because they have no common scale"); they make no claim about the data, so
    # they are removed before the check rather than weakening it.
    text = str(out.summary).lower()
    for boilerplate in (m.RANK_SCOPE_NOTE, m.MATERIALITY_DISCLAIMER):
        text = text.replace(boilerplate.lower(), "")
    for connective in ("because", "driven by", "due to", "caused by", "led to",
                       "as a result of", "attributable to", "explained by",
                       "thanks to", "owing to", "resulted in"):
        assert connective not in text, connective
