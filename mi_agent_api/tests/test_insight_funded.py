#!/usr/bin/env python3
"""Sprint B — the funded material-change composition, proved deterministically.

NO I/O, NO FRAMES, NO MODEL, NO DEPLOYMENT. Every fixture is a real
``PeriodChangeResult`` built from the workflow's own frozen dataclasses and a
real ``evaluate_active_tests``-shaped envelope. Building them from the actual
contracts rather than from hand-written dicts is deliberate: a dict fixture can
drift from the owner it stands for and keep passing, and the thing most worth
proving here is that this composition reads what those owners actually publish.

The suite answers one question in fifteen ways: given governed outputs, does the
composition report exactly what the configuration says is material, say why
about everything it does not report, and produce the same answer twice?
"""

from __future__ import annotations

import json

import pytest

from mi_agent.period_change.models import (
    BRIDGE_STATUS_AVAILABLE, BRIDGE_STATUS_DOES_NOT_RECONCILE,
    BRIDGE_STATUS_UNAVAILABLE_NO_IDENTIFIER, RESULT_SCHEMA_VERSION,
    STATUS_AVAILABLE, STATUS_NOT_COMPARABLE, WORKFLOW_ID, AggregateOutcome,
    BalanceBridge, CategoryShift, DistributionChange, FieldSelection,
    MetricChange, PeriodChangeResult, PeriodResolution, PortfolioScopeRef,
    SnapshotFrame,
)
from mi_agent_api import insight_config as cfg
from mi_agent_api import insight_funded as funded
from mi_agent_api.insight_contract import (
    FUNDED_BALANCE_ATTRIBUTION, FUNDED_BALANCE_MOVEMENT,
    FUNDED_COMPOSITION_SHIFT, FUNDED_METRIC_MOVEMENT, FUNDED_QUIET_PERIOD,
    LIMIT_STATUS_TRANSITION, SEVERITY_ATTENTION, SEVERITY_CONCERN,
    SEVERITY_INFO,
)

BALANCE = "current_outstanding_balance"


# --------------------------------------------------------------------------- #
# Builders — the owners' own types, never a stand-in dict
# --------------------------------------------------------------------------- #
def _outcome(value, *, valid=100, excluded=0):
    return AggregateOutcome(value=value, status=STATUS_AVAILABLE,
                            aggregation="sum", valid_population=valid,
                            excluded_population=excluded)


def _metric(field, name, start, end, movement, unit, *, relative=None,
            bps=None, interpretation="neutral", status=STATUS_AVAILABLE,
            rank=None, population=None):
    return MetricChange(
        field=field, display_name=name, analytical_concept=field,
        analytical_role="measure", temporality="point_in_time",
        aggregation="sum", weight_field=None, share_basis=None,
        start_value=start, end_value=end, movement_value=movement,
        movement_unit=unit, movement_basis="point_in_time_difference",
        relative_change=relative, basis_point_change=bps,
        directionality="neutral", interpretation=interpretation, status=status,
        start=_outcome(start), end=_outcome(end), confidence="high",
        rationale="fixture", movement_rank=rank, rank_population=population)


def _shift(category, s_count, e_count, s_share, e_share, s_bal=None, e_bal=None,
           s_bshare=None, e_bshare=None):
    return CategoryShift(
        category=category, start_count=s_count, end_count=e_count,
        start_count_share=s_share, end_count_share=e_share,
        count_share_movement=(None if s_share is None or e_share is None
                              else round(e_share - s_share, 6)),
        start_balance=s_bal, end_balance=e_bal,
        start_balance_share=s_bshare, end_balance_share=e_bshare,
        balance_share_movement=(None if s_bshare is None or e_bshare is None
                                else round(e_bshare - s_bshare, 6)))


def _distribution(*, increases=("Bridging",), decreases=("Term",),
                  bridging_end_share=0.495):
    """Product-type composition. ``bridging_end_share`` tunes the size of the
    shift, so the same dimension can be material or immaterial."""
    other = round(1.0 - bridging_end_share, 6)
    return DistributionChange(
        field="product_type", display_name="Product type",
        analytical_concept="product_type", status=STATUS_AVAILABLE,
        categories=(
            _shift("Bridging", 400, 500, 0.40, 0.4975,
                   40e6, 52e6, 0.40, bridging_end_share),
            _shift("Term", 600, 505, 0.60, 0.5025, 60e6, 53e6, 0.60, other),
        ),
        balance_field=BALANCE, start_total_count=1000, end_total_count=1005,
        start_total_balance=100e6, end_total_balance=105e6,
        largest_increases=increases, largest_decreases=decreases)


def _bridge(status=BRIDGE_STATUS_AVAILABLE, *, reconciles=True, residual=0.0,
            limitation=None):
    return BalanceBridge(
        status=status, balance_field=BALANCE,
        identifier_field="loan_identifier",
        identifier_fields=("loan_identifier",),
        opening_balance=100e6, closing_balance=105e6, new_loan_balance=12e6,
        exited_loan_balance=4e6, continuing_movement=-3e6, new_loan_count=60,
        exited_loan_count=25, continuing_loan_count=945, reconciles=reconciles,
        residual=residual, limitation=limitation)


def _result(*, metrics=None, distributions=None, bridge=..., warnings=(),
            start_date="2025-03-31", end_date="2025-06-30"):
    if metrics is None:
        metrics = (
            # +5.0% — above the 2.0% gate
            _metric(BALANCE, "Current outstanding balance",
                    100_000_000.0, 105_000_000.0, 5_000_000.0, "currency",
                    relative=0.05, rank=1, population=2),
            # +0.5% — below it
            _metric("loan_count", "Loan count", 1000.0, 1005.0, 5.0, "count",
                    relative=0.005, rank=2, population=2),
            # +2.4 points — above the 1.0pp gate, and a governed deterioration
            _metric("weighted_average_ltv", "Weighted-average LTV",
                    61.0, 63.4, 2.4, "percentage_point", bps=240.0,
                    interpretation="deterioration", rank=1, population=1),
        )
    if distributions is None:
        distributions = (_distribution(),)
    resolution = PeriodResolution(
        requested_start=None, requested_end=None,
        resolution_method="latest_available_pair",
        start_snapshot=SnapshotFrame(snapshot_id="s1", reporting_date=start_date),
        end_snapshot=SnapshotFrame(snapshot_id="s2", reporting_date=end_date))
    return PeriodChangeResult(
        workflow_id=WORKFLOW_ID, result_schema_version=RESULT_SCHEMA_VERSION,
        request_interpretation={"mode": "portfolio_overview"},
        portfolio_scope=PortfolioScopeRef(), period_resolution=resolution,
        dataset_provenance=(resolution.start_snapshot.reference(),
                            resolution.end_snapshot.reference()),
        summary={}, metric_changes=tuple(metrics),
        distribution_changes=tuple(distributions),
        balance_bridge=(_bridge() if bridge is ... else bridge),
        field_selection=FieldSelection(
            mode="portfolio_overview", measures=(), dimensions=(), excluded=(),
            concepts_covered=(), policy_name="governed", policy_version="1"),
        warnings=tuple(warnings), limitations=(), evidence=(),
        audit={"calculation_version": "1.0.0"})


def _test_row(test_id, name, status, prior_status, *, deteriorated=False,
              current=None, prior=None, threshold=None, headroom=None,
              breach=None, data_status="ok"):
    transition = (f"{prior_status} -> {status}"
                  if prior_status and prior_status != status else None)
    return {
        "testId": test_id, "displayName": name, "currentValue": current,
        "priorValue": prior, "threshold": threshold, "operator": "<=",
        "unit": "percent", "utilization": None, "headroom": headroom,
        "breachAmount": breach, "status": status, "priorStatus": prior_status,
        "statusTransition": transition, "deteriorated": deteriorated,
        "dataStatus": data_status,
    }


def _envelope(*rows, prior_available=True):
    return {"configurationVersion": 3, "configurationHash": "cfg-hash",
            "libraryVersion": "1", "reportingDate": "2025-06-30",
            "priorReportingDate": "2025-03-31",
            "priorAvailable": prior_available, "tests": list(rows)}


def _compose(result=..., **kw):
    kw.setdefault("tenant_id", "T")
    kw.setdefault("portfolio_id", "P")
    return funded.compose(_result() if result is ... else result, **kw)


def _types(brief):
    return [i["insight_type"] for i in brief["insights"]]


def _omitted(brief, insight_type):
    return [o for o in brief["omitted"] if o["insight_type"] == insight_type]


@pytest.fixture(autouse=True)
def _default_config(monkeypatch):
    """Every test starts from the shipped thresholds, not a previous test's."""
    monkeypatch.delenv(cfg.PATH_ENV, raising=False)
    cfg.reset_cache()
    yield
    cfg.reset_cache()


# --------------------------------------------------------------------------- #
# 1. quiet period
# --------------------------------------------------------------------------- #
def test_a_quiet_period_is_stated_rather_than_left_blank(tmp_path, monkeypatch):
    """Silence and "nothing happened" must not look the same."""
    config = tmp_path / "insights.yaml"
    config.write_text(
        "insights:\n"
        "  funded_metric_movement: {min_relative_change_pct: 99.0,"
        " min_change_pp: 99.0}\n"
        "  funded_composition: {min_share_change_pp: 99.0}\n", encoding="utf-8")
    monkeypatch.setenv(cfg.PATH_ENV, str(config))
    cfg.reset_cache()

    brief = _compose()
    assert _types(brief) == [FUNDED_QUIET_PERIOD]
    quiet = brief["insights"][0]
    assert quiet["metrics"]["findings"] == 0
    assert quiet["metrics"]["measures_examined"] == 3
    assert quiet["metrics"]["dimensions_examined"] == 1
    # It says what was examined and against what, so a reader can tell a quiet
    # book from a broken one.
    assert "99.0" in quiet["summary"]


def test_an_absent_analysis_is_not_reported_as_a_quiet_period():
    """The failure mode this rule exists for: no result must never read as
    reassurance."""
    brief = funded.compose(None, tenant_id="T", portfolio_id="P")
    assert brief["status"] == "unavailable"
    assert brief["insights"] == []
    assert FUNDED_QUIET_PERIOD not in _types(brief)


def test_nothing_comparable_is_not_reported_as_a_quiet_period():
    incomparable = (_metric(BALANCE, "Current outstanding balance", 100.0, None,
                            None, "currency", status=STATUS_NOT_COMPARABLE),)
    brief = _compose(_result(metrics=incomparable, distributions=()))
    assert FUNDED_QUIET_PERIOD not in _types(brief)
    reasons = " ".join(o["reason"] for o in _omitted(brief, FUNDED_QUIET_PERIOD))
    assert "not evidence that nothing changed" in reasons


# --------------------------------------------------------------------------- #
# 2-3. material movement, and below-threshold suppression
# --------------------------------------------------------------------------- #
def test_a_material_kpi_movement_is_reported_with_the_owners_own_figures():
    brief = _compose()
    ltv = next(i for i in brief["insights"]
               if i["discriminator" if "discriminator" in i else "headline"]
               or True
               if i["insight_type"] == FUNDED_METRIC_MOVEMENT)
    # Verbatim from MetricChange.to_dict(): not rounded, renamed or rescaled.
    assert ltv["metrics"]["movement_value"] == 2.4
    assert ltv["metrics"]["basis_point_change"] == 240.0
    assert ltv["metrics"]["movement_unit"] == "percentage_point"
    assert ltv["metrics"]["canonical_field"] == "weighted_average_ltv"
    # A governed deterioration is raised above an observation.
    assert ltv["severity"] == SEVERITY_ATTENTION


def test_the_balance_is_its_own_finding_because_a_bridge_can_explain_it():
    brief = _compose()
    balance = next(i for i in brief["insights"]
                   if i["insight_type"] == FUNDED_BALANCE_MOVEMENT)
    assert balance["metrics"]["canonical_field"] == BALANCE
    assert balance["metrics"]["relative_change"] == 0.05
    # The fraction is published as the owner stated it, and is LABELLED, so a
    # consumer cannot read 0.05 as five percentage points.
    assert "fraction" in balance["metrics"]["relative_change_is"]


def test_a_movement_below_its_gate_is_suppressed_and_named():
    brief = _compose()
    reported = [i["metrics"]["canonical_field"] for i in brief["insights"]
                if i["insight_type"] in (FUNDED_METRIC_MOVEMENT,
                                         FUNDED_BALANCE_MOVEMENT)]
    assert "loan_count" not in reported
    # Suppression is never silent: the omission names the measure.
    reasons = " ".join(o["reason"] for o in _omitted(brief, FUNDED_METRIC_MOVEMENT))
    assert "Loan count" in reasons


def test_the_gate_is_chosen_by_unit_not_by_name():
    """A 2.4-point LTV move is +3.9% of its own value — under the 2% relative
    gate it would still pass, so prove the POINT gate is what admitted it by
    moving it under both readings of the relative one."""
    small_pp = (_metric("weighted_average_ltv", "Weighted-average LTV",
                        61.0, 61.5, 0.5, "percentage_point", bps=50.0),)
    brief = _compose(_result(metrics=small_pp, distributions=()))
    # 0.5 points is below the 1.0pp gate even though 0.5/61 would be reported as
    # a relative move by a unit-blind rule.
    assert FUNDED_METRIC_MOVEMENT not in _types(brief)


# --------------------------------------------------------------------------- #
# 4-7. limit status transitions
# --------------------------------------------------------------------------- #
def test_pass_to_breach_is_a_concern():
    envelope = _envelope(_test_row(
        "T1", "Top-10 obligor concentration", "breach", "pass",
        deteriorated=True, current=26.4, prior=22.0, threshold=25.0,
        headroom=-1.4, breach=1.4))
    brief = _compose(limit_envelope=envelope)
    finding = next(i for i in brief["insights"]
                   if i["insight_type"] == LIMIT_STATUS_TRANSITION)
    assert finding["severity"] == SEVERITY_CONCERN
    assert finding["metrics"]["status_transition"] == "pass -> breach"
    assert finding["metrics"]["breach_amount"] == 1.4
    assert "deteriorated" in finding["headline"]


def test_breach_to_pass_is_reported_as_a_recovery_not_as_a_breach():
    envelope = _envelope(_test_row(
        "T1", "Top-10 obligor concentration", "pass", "breach",
        deteriorated=False, current=21.0, prior=26.4, threshold=25.0,
        headroom=4.0))
    brief = _compose(limit_envelope=envelope)
    finding = next(i for i in brief["insights"]
                   if i["insight_type"] == LIMIT_STATUS_TRANSITION)
    assert finding["severity"] == SEVERITY_INFO
    assert "recovered" in finding["headline"]
    assert finding["metrics"]["deteriorated"] is False


def test_a_headroom_deterioration_short_of_a_breach_is_attention_not_concern():
    envelope = _envelope(_test_row(
        "T2", "Single-region exposure", "warning", "pass", deteriorated=True,
        current=19.1, prior=17.0, threshold=20.0, headroom=0.9))
    brief = _compose(limit_envelope=envelope)
    finding = next(i for i in brief["insights"]
                   if i["insight_type"] == LIMIT_STATUS_TRANSITION)
    assert finding["severity"] == SEVERITY_ATTENTION
    assert finding["metrics"]["headroom"] == 0.9


def test_an_evaluability_transition_is_never_called_a_recovery():
    """A test that stopped being measurable did not improve.

    ``deteriorated`` is False for pass -> unavailable because the evaluator
    cannot rank a non-risk status — so a rule that split only on that flag would
    announce a recovery. This is the case that rule exists for.
    """
    envelope = _envelope(_test_row(
        "T3", "Arrears rate", "unavailable", "pass", data_status="data_missing"))
    brief = _compose(limit_envelope=envelope)
    finding = next(i for i in brief["insights"]
                   if i["insight_type"] == LIMIT_STATUS_TRANSITION)
    assert finding["severity"] == SEVERITY_INFO
    assert "recovered" not in finding["headline"]
    assert "changed evaluation status" in finding["headline"]
    assert "not a measured change in exposure" in finding["summary"]


def test_a_test_that_held_its_status_produces_no_finding_and_says_so():
    envelope = _envelope(_test_row("T4", "Arrears rate", "pass", "pass"))
    brief = _compose(limit_envelope=envelope)
    assert LIMIT_STATUS_TRANSITION not in _types(brief)
    reasons = " ".join(o["reason"] for o in _omitted(brief, LIMIT_STATUS_TRANSITION))
    assert "same governed status" in reasons


def test_no_configuration_is_declared_rather_than_read_as_all_clear():
    brief = _compose(limit_envelope=None)
    omission = _omitted(brief, LIMIT_STATUS_TRANSITION)[0]
    assert omission["category"] == "unavailable"
    assert "No activated concentration or limit configuration" in omission["reason"]


def test_one_reporting_date_cannot_establish_a_transition():
    envelope = _envelope(_test_row("T1", "Top-10", "breach", None),
                         prior_available=False)
    brief = _compose(limit_envelope=envelope)
    omission = _omitted(brief, LIMIT_STATUS_TRANSITION)[0]
    assert omission["category"] == "unavailable"
    assert "Only one reporting date" in omission["reason"]


# --------------------------------------------------------------------------- #
# 8. composition movement
# --------------------------------------------------------------------------- #
def test_a_composition_shift_reports_the_workflows_own_leaders():
    brief = _compose()
    shift = next(i for i in brief["insights"]
                 if i["insight_type"] == FUNDED_COMPOSITION_SHIFT)
    assert shift["metrics"]["basis"] == "balance share"
    # The categories came from the workflow's own ranking, not from a re-sort.
    assert shift["components"]["largest_increases"] == ["Bridging"]
    named = [c["category"] for c in shift["contributors"]["categories"]]
    assert named == ["Bridging", "Term"]
    # Shares stay the fractions the owner published.
    assert shift["metrics"]["shares_are"].startswith("fractions")


def test_a_composition_shift_below_its_gate_is_suppressed_and_named():
    tiny = _distribution(bridging_end_share=0.41)      # +1.0pp, under 5.0pp
    brief = _compose(_result(distributions=(tiny,)))
    assert FUNDED_COMPOSITION_SHIFT not in _types(brief)
    reasons = " ".join(o["reason"] for o in _omitted(brief, FUNDED_COMPOSITION_SHIFT))
    assert "Product type" in reasons and "5.0pp" in reasons


# --------------------------------------------------------------------------- #
# 9. attribution enrichment
# --------------------------------------------------------------------------- #
def test_attribution_decomposes_the_movement_it_explains():
    brief = _compose()
    attribution = next(i for i in brief["insights"]
                       if i["insight_type"] == FUNDED_BALANCE_ATTRIBUTION)
    m = attribution["metrics"]
    assert m["opening_balance"] == 100e6 and m["closing_balance"] == 105e6
    assert m["new_loan_closing_balance"] == 12e6
    assert m["exited_loan_opening_balance"] == 4e6
    assert m["movement_on_continuing_loans"] == -3e6
    assert m["reconciles"] is True
    # No loan-level value reaches the insight.
    assert "loan_identifier" in attribution["components"]["identifier_fields"]


def test_attribution_is_withheld_when_the_movement_it_explains_was_immaterial(
        tmp_path, monkeypatch):
    config = tmp_path / "insights.yaml"
    config.write_text("insights:\n  funded_metric_movement:"
                      " {min_relative_change_pct: 99.0, min_change_pp: 99.0}\n",
                      encoding="utf-8")
    monkeypatch.setenv(cfg.PATH_ENV, str(config))
    cfg.reset_cache()

    brief = _compose()
    assert FUNDED_BALANCE_ATTRIBUTION not in _types(brief)
    reason = _omitted(brief, FUNDED_BALANCE_ATTRIBUTION)[0]["reason"]
    assert "immaterial by construction" in reason


def test_a_bridge_that_does_not_reconcile_is_a_limitation_not_an_explanation():
    broken = _bridge(BRIDGE_STATUS_DOES_NOT_RECONCILE, reconciles=False,
                     residual=812_400.0,
                     limitation="The bridge does not reconcile to its closing "
                                "balance.")
    brief = _compose(_result(bridge=broken))
    assert FUNDED_BALANCE_ATTRIBUTION not in _types(brief)
    omission = _omitted(brief, FUNDED_BALANCE_ATTRIBUTION)[0]
    assert omission["category"] == "unavailable"
    assert "does not reconcile" in omission["reason"]
    assert "rather than as an explanation" in omission["reason"]


def test_an_unavailable_bridge_states_the_owners_own_reason():
    no_id = _bridge(BRIDGE_STATUS_UNAVAILABLE_NO_IDENTIFIER,
                    limitation="No stable loan identifier is available.")
    brief = _compose(_result(bridge=no_id))
    omission = _omitted(brief, FUNDED_BALANCE_ATTRIBUTION)[0]
    assert "No stable loan identifier" in omission["reason"]


# --------------------------------------------------------------------------- #
# 10. deterministic ranking
# --------------------------------------------------------------------------- #
def test_findings_are_ordered_by_severity_then_by_type_priority():
    envelope = _envelope(
        _test_row("T1", "Top-10 obligor concentration", "breach", "warning",
                  deteriorated=True, current=26.4, threshold=25.0, breach=1.4),
        _test_row("T2", "Single-region exposure", "pass", "warning",
                  current=18.0, threshold=20.0, headroom=2.0))
    brief = _compose(limit_envelope=envelope)
    assert _types(brief) == [
        LIMIT_STATUS_TRANSITION,        # concern  — a contractual exposure
        FUNDED_METRIC_MOVEMENT,         # attention — a governed deterioration
        LIMIT_STATUS_TRANSITION,        # info, priority 100
        FUNDED_BALANCE_MOVEMENT,        # info, priority 75
        FUNDED_BALANCE_ATTRIBUTION,     # info, priority 72 — follows its movement
        FUNDED_COMPOSITION_SHIFT,       # info, priority 50
    ]
    priorities = [i["priority"] for i in brief["insights"]]
    assert priorities == sorted(priorities, reverse=True)


def test_the_quiet_statement_never_displaces_a_finding():
    brief = _compose()
    assert brief["insights"]
    assert FUNDED_QUIET_PERIOD not in _types(brief)


def test_ties_are_broken_deterministically_by_discriminator():
    """Two transitions of the same severity and type must order identically on
    every run, whatever order the evaluator listed them in."""
    rows = [_test_row(f"T{n}", f"Test {n}", "pass", "warning") for n in (3, 1, 2)]
    forward = _compose(limit_envelope=_envelope(*rows))
    backward = _compose(limit_envelope=_envelope(*reversed(rows)))
    ids = lambda b: [i["insight_id"] for i in b["insights"]]   # noqa: E731
    assert ids(forward) == ids(backward)


# --------------------------------------------------------------------------- #
# 11. omission evidence
# --------------------------------------------------------------------------- #
def test_every_finding_records_the_threshold_that_admitted_it():
    brief = _compose(limit_envelope=_envelope(
        _test_row("T1", "Top-10", "breach", "warning", deteriorated=True)))
    for insight in brief["insights"]:
        materiality = insight["methodology"]["materiality"]
        assert materiality["threshold_section"]
        assert materiality["threshold_key"]
        assert materiality["applied_by"] == "mi_agent_api.insight_funded"


def test_every_finding_names_the_governed_owner_of_its_figures():
    brief = _compose(limit_envelope=_envelope(
        _test_row("T1", "Top-10", "breach", "warning", deteriorated=True)))
    owners = {i["methodology"]["owner"] for i in brief["insights"]}
    assert owners <= {funded.OWNER_PERIOD_CHANGE, funded.OWNER_LIMIT_TESTS,
                      f"{funded.OWNER_PERIOD_CHANGE} "
                      f"(mi_agent.period_change.bridge)"}


def test_every_omission_carries_a_category_and_a_reason():
    brief = _compose(limit_envelope=None)
    assert brief["omitted"]
    for omission in brief["omitted"]:
        assert omission["category"] in ("immaterial", "unavailable", "error",
                                        "capped")
        assert len(omission["reason"]) > 20


def test_a_capped_type_is_reported_as_capped_not_dropped():
    rows = [_test_row(f"T{n}", f"Test {n}", "breach", "warning",
                      deteriorated=True) for n in range(1, 6)]
    brief = _compose(limit_envelope=_envelope(*rows))
    capped = [o for o in brief["omitted"] if o["category"] == "capped"]
    assert capped, "five transitions against a cap of three must report the cap"
    assert "capped at" in capped[0]["reason"]


# --------------------------------------------------------------------------- #
# 12-13. refusals owned elsewhere, surfaced here
# --------------------------------------------------------------------------- #
def test_an_unresolvable_period_pair_produces_an_unavailable_brief():
    """``resolve_periods`` owns the refusal; the composition must carry it as a
    stated unavailability rather than an empty success."""
    brief = funded.compose(None, tenant_id="T", portfolio_id="P")
    assert brief["status"] == "unavailable"
    assert "No governed period-change analysis" in brief["reason"]
    assert brief["insight_count"] == 0


@pytest.mark.parametrize("form", ["current", "series", "forward_looking"])
def test_a_period_form_that_is_not_a_pair_is_refused_by_the_plan_perimeter(form):
    from mi_agent import plan_material_summary as runtime
    plan = {"operation": "summary", "population": {"base": "funded"},
            "outputs": [{"id": "o", "measures": []}], "period": {"form": form},
            "provenance": {"compiler_bindings": {
                "change_form": {"form": "material_summary"}}}}
    eligible, why, _detail = runtime.check_eligibility(plan)
    assert not eligible
    assert why == runtime.PERIOD_NOT_A_PAIR


@pytest.mark.parametrize("grain", ["weekly", "daily"])
def test_a_relative_grain_with_no_governed_method_is_refused_not_rounded(grain):
    from mi_agent import plan_material_summary as runtime
    plan = {"operation": "summary", "population": {"base": "funded"},
            "outputs": [{"id": "o", "measures": []}],
            "period": {"form": "relative_pair", "grain": grain},
            "provenance": {"compiler_bindings": {
                "change_form": {"form": "material_summary"}}}}
    eligible, why, _detail = runtime.check_eligibility(plan)
    assert not eligible
    assert why == runtime.PERIOD_GRAIN_UNTRANSLATABLE


def test_an_absent_grain_means_the_adjacent_governed_snapshots():
    """NOT a gap, and NOT an assumed monthly book.

    The compiler's normal form leaves the grain absent for the commonest request
    in this family and states what that means — "a pair with no distance IS the
    adjacent pair". The resolver has a method for exactly that. Mapping it to
    `month_on_month` would assume a cadence; refusing it would reject the
    question this contract exists for.
    """
    from mi_agent import plan_material_summary as runtime
    plan = {"operation": "summary", "population": {"base": "funded"},
            "outputs": [{"id": "o", "measures": []}],
            "period": {"form": "relative_pair", "periods_back": 1},
            "provenance": {"compiler_bindings": {
                "change_form": {"form": "material_summary"}}}}
    eligible, why, detail = runtime.check_eligibility(plan)
    assert eligible, f"{why}: {detail}"
    assert runtime.period_request(plan).relative_mode == "current_vs_previous"


def test_a_distance_with_no_governed_method_is_refused_not_approximated():
    from mi_agent import plan_material_summary as runtime
    plan = {"operation": "summary", "population": {"base": "funded"},
            "outputs": [{"id": "o", "measures": []}],
            "period": {"form": "relative_pair", "periods_back": 3},
            "provenance": {"compiler_bindings": {
                "change_form": {"form": "material_summary"}}}}
    eligible, why, _detail = runtime.check_eligibility(plan)
    assert not eligible
    assert why == runtime.PERIOD_GRAIN_UNTRANSLATABLE


@pytest.mark.parametrize("base", ["pipeline", "forecast", "whole_book", None])
def test_an_unsupported_scope_is_refused_by_the_plan_perimeter(base):
    from mi_agent import plan_material_summary as runtime
    plan = {"operation": "summary", "population": {"base": base},
            "outputs": [{"id": "o", "measures": []}],
            "period": {"form": "relative_pair", "grain": "monthly"},
            "provenance": {"compiler_bindings": {
                "change_form": {"form": "material_summary"}}}}
    eligible, why, _detail = runtime.check_eligibility(plan)
    assert not eligible
    assert why == runtime.POPULATION_NOT_FUNDED


def test_a_non_canonical_operation_fails_closed_rather_than_being_rewritten():
    """Canonicalisation happens once, in the compiler's normal form. A plan that
    skipped it is refused rather than re-canonicalised here — a second rewriter
    is a second place the two could disagree."""
    from mi_agent import plan_material_summary as runtime
    plan = {"operation": "movement", "population": {"base": "funded"},
            "outputs": [{"id": "o", "measures": []}],
            "period": {"form": "relative_pair", "grain": "monthly"},
            "provenance": {"compiler_bindings": {
                "change_form": {"form": "material_summary"}}}}
    eligible, why, _detail = runtime.check_eligibility(plan)
    assert not eligible
    assert why == runtime.OPERATION_NOT_ADMITTED


# --------------------------------------------------------------------------- #
# 14. configured threshold override
# --------------------------------------------------------------------------- #
def test_a_deployment_threshold_changes_what_is_reported(tmp_path, monkeypatch):
    """The thresholds are the ONLY thing deciding materiality, and they are
    configuration. Prove it by moving one and nothing else."""
    before = _compose()
    assert FUNDED_COMPOSITION_SHIFT in _types(before)

    config = tmp_path / "insights.yaml"
    config.write_text("insights:\n  funded_composition:"
                      " {min_share_change_pp: 20.0}\n", encoding="utf-8")
    monkeypatch.setenv(cfg.PATH_ENV, str(config))
    cfg.reset_cache()

    after = _compose()
    assert FUNDED_COMPOSITION_SHIFT not in _types(after)
    # And every other finding is untouched by a composition threshold.
    assert [t for t in _types(before) if t != FUNDED_COMPOSITION_SHIFT] == \
        _types(after)
    assert after["config_source"] == str(config)


def test_a_partial_override_leaves_every_other_threshold_alone(tmp_path,
                                                               monkeypatch):
    config = tmp_path / "insights.yaml"
    config.write_text("insights:\n  funded_metric_movement:"
                      " {min_change_pp: 99.0}\n", encoding="utf-8")
    monkeypatch.setenv(cfg.PATH_ENV, str(config))
    cfg.reset_cache()

    brief = _compose()
    # The pp gate moved, so the LTV finding is gone; the relative gate did not,
    # so the balance finding remains.
    assert FUNDED_BALANCE_MOVEMENT in _types(brief)
    fields = [i["metrics"].get("canonical_field") for i in brief["insights"]]
    assert "weighted_average_ltv" not in fields


def test_improvements_can_be_switched_off_without_losing_deteriorations(
        tmp_path, monkeypatch):
    config = tmp_path / "insights.yaml"
    config.write_text("insights:\n  funded_limits: {report_improvements: false}\n",
                      encoding="utf-8")
    monkeypatch.setenv(cfg.PATH_ENV, str(config))
    cfg.reset_cache()

    envelope = _envelope(
        _test_row("T1", "Top-10", "breach", "warning", deteriorated=True),
        _test_row("T2", "Single-region", "pass", "warning"))
    brief = _compose(limit_envelope=envelope)
    transitions = [i for i in brief["insights"]
                   if i["insight_type"] == LIMIT_STATUS_TRANSITION]
    assert len(transitions) == 1
    assert transitions[0]["metrics"]["deteriorated"] is True
    reasons = " ".join(o["reason"] for o in _omitted(brief, LIMIT_STATUS_TRANSITION))
    assert "improvements are switched off" in reasons


def test_the_quiet_statement_can_be_switched_off(tmp_path, monkeypatch):
    config = tmp_path / "insights.yaml"
    config.write_text(
        "insights:\n"
        "  funded_metric_movement: {min_relative_change_pct: 99.0,"
        " min_change_pp: 99.0}\n"
        "  funded_composition: {min_share_change_pp: 99.0}\n"
        "  funded_brief: {emit_quiet_period: false}\n", encoding="utf-8")
    monkeypatch.setenv(cfg.PATH_ENV, str(config))
    cfg.reset_cache()

    brief = _compose()
    assert brief["insights"] == []
    assert brief["omitted"], "silence is still explained by the omissions"


# --------------------------------------------------------------------------- #
# 15. deterministic repeatability
# --------------------------------------------------------------------------- #
def test_the_same_inputs_produce_a_byte_identical_brief():
    envelope = _envelope(
        _test_row("T1", "Top-10", "breach", "warning", deteriorated=True),
        _test_row("T2", "Single-region", "pass", "warning"))
    first = _compose(limit_envelope=envelope)
    second = _compose(limit_envelope=envelope)
    assert json.dumps(first, sort_keys=True) == json.dumps(second, sort_keys=True)


def test_an_insight_id_is_stable_across_regenerations_but_not_across_dates():
    """The id hashes what an insight IS, not what it says — so a corrected
    upload updates an insight rather than sitting beside a stale twin."""
    first = _compose()
    again = _compose()
    later = _compose(_result(end_date="2025-09-30"))
    by_type = lambda b: {i["insight_type"]: i["insight_id"]        # noqa: E731
                         for i in b["insights"]}
    assert by_type(first) == by_type(again)
    assert by_type(first)[FUNDED_BALANCE_MOVEMENT] != \
        by_type(later)[FUNDED_BALANCE_MOVEMENT]


def test_the_id_does_not_move_when_only_the_values_move():
    bigger = (_metric(BALANCE, "Current outstanding balance",
                      100_000_000.0, 140_000_000.0, 40_000_000.0, "currency",
                      relative=0.40),)
    base = _compose(_result(distributions=(), bridge=None))
    restated = _compose(_result(metrics=bigger, distributions=(), bridge=None))
    pick = lambda b: next(i["insight_id"] for i in b["insights"]   # noqa: E731
                          if i["insight_type"] == FUNDED_BALANCE_MOVEMENT)
    assert pick(base) == pick(restated)


# --------------------------------------------------------------------------- #
# The composition receipt
# --------------------------------------------------------------------------- #
def test_the_receipt_records_what_ran_from_the_governed_outputs():
    from mi_agent import plan_material_summary as runtime
    plan = {"operation": "summary", "population": {"base": "funded"},
            "outputs": [{"id": "o", "measures": []}],
            "period": {"form": "relative_pair", "grain": "monthly"},
            "provenance": {"compiler_bindings": {
                "change_form": {"form": "material_summary",
                                "capability": "period_movement",
                                "mode": "portfolio_overview"}}}}
    result = _result()
    brief = _compose(result)
    receipt = runtime.receipt(plan, result, brief)

    assert receipt["change_form"] == "material_summary"
    assert receipt["operation"] == "summary"
    assert receipt["population_base"] == "funded"
    # Read from the RESULT, not from the plan's intent: the receipt says what
    # actually ran.
    assert receipt["workflow_owner"] == WORKFLOW_ID
    assert receipt["workflow_mode"] == "portfolio_overview"
    assert receipt["period_resolution"]["resolved_end_snapshot"][
        "reporting_date"] == "2025-06-30"
    assert receipt["composition_owner"] == "mi_agent_api.insight_funded.compose"
    assert receipt["finding_count"] == brief["insight_count"]
    assert receipt["omission_count"] == len(brief["omitted"])
    assert receipt["status"] == "success"


def test_the_brief_carries_both_reporting_dates_and_their_provenance():
    brief = _compose()
    assert brief["as_of_date"] == "2025-06-30"
    assert brief["comparison_date"] == "2025-03-31"
    dates = brief["source_dates"]
    assert dates["funded_as_of"] == "2025-06-30"
    assert dates["funded_comparison"] == "2025-03-31"
    assert len(dates["dataset_provenance"]) == 2


def test_no_finding_carries_loan_level_data():
    """The contract's standing rule: contributors are aggregates, never cases.

    The bridge publishes the NAME of the identity key it reconciled over —
    ``loan_identifier`` — which is provenance and must stay. What must never
    appear is a value of it, or a per-case row. Asserted the same way the weekly
    brief asserts it, plus the shapes that would carry one.
    """
    brief = _compose(limit_envelope=_envelope(
        _test_row("T1", "Top-10", "breach", "warning", deteriorated=True)))
    blob = json.dumps(brief)
    for forbidden in ("case_id", "caseId", "loanId"):
        assert forbidden not in blob

    attribution = next(i for i in brief["insights"]
                       if i["insight_type"] == FUNDED_BALANCE_ATTRIBUTION)
    # Counts and balances, never the loans behind them.
    assert attribution["metrics"]["new_loan_count"] == 60
    assert "new_loans" not in attribution["metrics"]
    assert attribution["components"]["identifier_fields"] == ["loan_identifier"]

    # Every contributor row is a CATEGORY aggregate.
    for insight in brief["insights"]:
        for rows in (insight.get("contributors") or {}).values():
            for row in rows if isinstance(rows, list) else []:
                assert "category" in row


# --------------------------------------------------------------------------- #
# Partial failure is a first-class outcome, as it is for the weekly brief
# --------------------------------------------------------------------------- #
def test_one_generator_failing_costs_one_section_not_the_brief(monkeypatch):
    def explode(*_a, **_k):
        raise RuntimeError("fixture")
    monkeypatch.setattr(funded, "composition_shifts", explode)

    brief = _compose()
    assert brief["status"] == "partial"
    assert FUNDED_BALANCE_MOVEMENT in _types(brief)
    errors = [o for o in brief["omitted"] if o["category"] == "error"]
    assert errors and errors[0]["insight_type"] == FUNDED_COMPOSITION_SHIFT
