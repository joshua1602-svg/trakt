"""Smoke: build a PeriodChangeResult by hand and compose it. No I/O, no pandas frames."""
import json
from mi_agent.period_change.models import (
    AggregateOutcome, BalanceBridge, CategoryShift, DistributionChange,
    FieldSelection, MetricChange, PeriodChangeResult, PeriodResolution,
    PortfolioScopeRef, SnapshotFrame, BRIDGE_STATUS_AVAILABLE,
    STATUS_AVAILABLE, RESULT_SCHEMA_VERSION, WORKFLOW_ID,
)
from mi_agent_api import insight_funded


def outcome(v):
    return AggregateOutcome(value=v, status=STATUS_AVAILABLE, aggregation="sum",
                            valid_population=100)


def metric(field, name, s, e, mv, unit, rel=None, bps=None, interp="neutral"):
    return MetricChange(
        field=field, display_name=name, analytical_concept=field,
        analytical_role="measure", temporality="point_in_time",
        aggregation="sum", weight_field=None, share_basis=None,
        start_value=s, end_value=e, movement_value=mv, movement_unit=unit,
        movement_basis="point_in_time_difference", relative_change=rel,
        basis_point_change=bps, directionality="neutral", interpretation=interp,
        status=STATUS_AVAILABLE, start=outcome(s), end=outcome(e),
        confidence="high", rationale="fixture")


start = SnapshotFrame(snapshot_id="s1", reporting_date="2025-03-31")
end = SnapshotFrame(snapshot_id="s2", reporting_date="2025-06-30")
resolution = PeriodResolution(requested_start=None, requested_end=None,
                              resolution_method="latest_available_pair",
                              start_snapshot=start, end_snapshot=end)

metrics = (
    # balance: +5.0% -> qualifies (>= 2.0%)
    metric("current_outstanding_balance", "Current outstanding balance",
           100_000_000.0, 105_000_000.0, 5_000_000.0, "currency", rel=0.05),
    # loan count: +0.5% -> immaterial
    metric("loan_count", "Loan count", 1000.0, 1005.0, 5.0, "count", rel=0.005),
    # WA LTV: +2.4 points -> qualifies (>= 1.0pp), deterioration
    metric("weighted_average_ltv", "Weighted-average LTV", 61.0, 63.4, 2.4,
           "percentage_point", bps=240.0, interp="deterioration"),
)

dist = DistributionChange(
    field="product_type", display_name="Product type",
    analytical_concept="product_type", status=STATUS_AVAILABLE,
    categories=(
        CategoryShift(category="Bridging", start_count=400, end_count=500,
                      start_count_share=0.40, end_count_share=0.4975,
                      count_share_movement=0.0975,
                      start_balance=40e6, end_balance=52e6,
                      start_balance_share=0.40, end_balance_share=0.495,
                      balance_share_movement=0.095),
        CategoryShift(category="Term", start_count=600, end_count=505,
                      start_count_share=0.60, end_count_share=0.5025,
                      count_share_movement=-0.0975,
                      start_balance=60e6, end_balance=53e6,
                      start_balance_share=0.60, end_balance_share=0.505,
                      balance_share_movement=-0.095),
    ),
    balance_field="current_outstanding_balance",
    start_total_count=1000, end_total_count=1005,
    start_total_balance=100e6, end_total_balance=105e6,
    largest_increases=("Bridging",), largest_decreases=("Term",))

bridge = BalanceBridge(
    status=BRIDGE_STATUS_AVAILABLE, balance_field="current_outstanding_balance",
    identifier_field="loan_identifier", identifier_fields=("loan_identifier",),
    opening_balance=100e6, closing_balance=105e6, new_loan_balance=12e6,
    exited_loan_balance=4e6, continuing_movement=-3e6,
    new_loan_count=60, exited_loan_count=25, continuing_loan_count=945,
    reconciles=True, residual=0.0)

result = PeriodChangeResult(
    workflow_id=WORKFLOW_ID, result_schema_version=RESULT_SCHEMA_VERSION,
    request_interpretation={"mode": "portfolio_overview"},
    portfolio_scope=PortfolioScopeRef(), period_resolution=resolution,
    dataset_provenance=(start.reference(), end.reference()),
    summary={}, metric_changes=metrics, distribution_changes=(dist,),
    balance_bridge=bridge,
    field_selection=FieldSelection(mode="portfolio_overview", measures=(),
                                   dimensions=(), excluded=(),
                                   concepts_covered=(), policy_name="p",
                                   policy_version="1"),
    warnings=(), limitations=(), evidence=(), audit={"calculation_version": "1.0.0"})

limits = {
    "configurationVersion": 3, "configurationHash": "abc", "libraryVersion": "1",
    "reportingDate": "2025-06-30", "priorReportingDate": "2025-03-31",
    "priorAvailable": True,
    "tests": [
        {"testId": "T1", "displayName": "Top-10 obligor concentration",
         "currentValue": 26.4, "priorValue": 24.1, "threshold": 25.0,
         "operator": "<=", "unit": "percent", "utilization": 105.6,
         "headroom": -1.4, "breachAmount": 1.4, "status": "breach",
         "priorStatus": "warning", "statusTransition": "warning -> breach",
         "deteriorated": True, "dataStatus": "ok"},
        {"testId": "T2", "displayName": "Single-region exposure",
         "currentValue": 18.0, "priorValue": 23.0, "threshold": 20.0,
         "operator": "<=", "unit": "percent", "utilization": 90.0,
         "headroom": 2.0, "status": "pass", "priorStatus": "warning",
         "statusTransition": "warning -> pass", "deteriorated": False,
         "dataStatus": "ok"},
        {"testId": "T3", "displayName": "Arrears rate", "status": "pass",
         "priorStatus": "pass", "statusTransition": None, "deteriorated": False},
    ],
}

brief = insight_funded.compose(result, tenant_id="T", portfolio_id="P",
                               limit_envelope=limits, run_id="r1")
print("status:", brief["status"], "| findings:", brief["insight_count"])
for i in brief["insights"]:
    print(f"  [{i['priority']:>3}] {i['severity']:<9} {i['insight_type']:<26} {i['headline']}")
print("omitted:")
for o in brief["omitted"]:
    print(f"  {o['category']:<12} {o['insight_type']:<26} {o['reason'][:90]}")
print()
print("balance insight metrics keys:",
      sorted(next(i for i in brief["insights"]
                  if i["insight_type"] == "FUNDED_BALANCE_MOVEMENT")["metrics"]))
print()
print("--- quiet period: same result, thresholds raised out of reach ---")
import mi_agent_api.insight_config as cfg
cfg._CACHE.clear()
cfg.DEFAULTS["funded_metric_movement"]["min_relative_change_pct"] = 99.0
cfg.DEFAULTS["funded_metric_movement"]["min_change_pp"] = 99.0
cfg.DEFAULTS["funded_composition"]["min_share_change_pp"] = 99.0
import os
os.environ["TRAKT_MI_INSIGHTS_CONFIG"] = "/nonexistent-so-defaults-apply.yaml"
cfg.reset_cache()
quiet = insight_funded.compose(result, tenant_id="T", portfolio_id="P",
                               limit_envelope=None, run_id="r1")
for i in quiet["insights"]:
    print(f"  {i['insight_type']}: {i['headline']}")
    print(f"    {i['summary']}")
for o in quiet["omitted"]:
    print(f"  omitted {o['category']:<12} {o['insight_type']:<26} {o['reason'][:100]}")
