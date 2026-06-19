"""mi_agent.mi_query_spec_v2_examples — canonical MIQuerySpec v2 examples.

A deterministic catalogue of valid MIQuerySpec v2 dicts, one per supported MI
question. Used by the interpretation contract (docs) and by the Phase 7 tests
so "the examples validate" is machine-checked. These are plain dicts (the shape
an LLM parser must emit) — no code, no execution.
"""

from __future__ import annotations

from typing import Any, Dict

CLIENT = "clientA"

EXAMPLES: Dict[str, Dict[str, Any]] = {
    # 1. Existing flat single-CSV query (backward compatible v1 shape).
    "flat_ltv_by_region": {
        "intent": "chart", "chart_type": "bar",
        "metric": "current_loan_to_value", "dimension": "collateral_geography",
        "aggregation": "weighted_avg",
    },
    # 2. Total funded latest.
    "total_funded_latest": {
        "route_id": "mi", "execution_mode": "state", "state": "total_funded",
        "temporal_mode": "latest", "snapshot_client_id": CLIENT,
    },
    # 3. Total pipeline latest.
    "total_pipeline_latest": {
        "route_id": "mi", "state": "total_pipeline", "temporal_mode": "latest",
        "snapshot_client_id": CLIENT,
    },
    # 4. Total forecast-funded latest.
    "total_forecast_funded_latest": {
        "route_id": "mi", "state": "total_forecast_funded",
        "temporal_mode": "latest", "snapshot_client_id": CLIENT,
        "forecast_probability_source": "row", "allow_config_probability": True,
    },
    # 5. Funded trend over last three snapshots.
    "funded_trend": {
        "route_id": "mi", "state": "total_funded", "temporal_mode": "trend",
        "start_date": "2024-01-01", "end_date": "2024-12-31",
        "trend_grain": "monthly", "snapshot_client_id": CLIENT,
        "output_type": "chart", "chart_preference": "line",
    },
    # 6. Funded compare current vs baseline.
    "funded_compare": {
        "route_id": "mi", "state": "total_funded", "temporal_mode": "compare",
        "baseline_date": "2024-01-31", "current_date": "2024-03-31",
        "comparison_basis": "balance", "snapshot_client_id": CLIENT,
    },
    # 7. Pipeline by stage.
    "pipeline_by_stage": {
        "route_id": "mi", "risk_monitor_mode": "concentration",
        "state": "total_pipeline", "concentration_dimension": "pipeline_stage",
        "as_of_date": "2024-02-29", "snapshot_client_id": CLIENT,
    },
    # 8. Forecast-funded by region.
    "forecast_funded_by_region": {
        "route_id": "mi", "risk_monitor_mode": "concentration",
        "state": "total_forecast_funded",
        "concentration_dimension": "geographic_region_obligor",
        "snapshot_client_id": CLIENT,
    },
    # 9. Concentration warning by region.
    "concentration_warning_by_region": {
        "route_id": "mi", "risk_monitor_mode": "concentration",
        "state": "total_funded",
        "concentration_dimension": "geographic_region_obligor",
        "snapshot_client_id": CLIENT,
    },
    # 10. Risk grade migration.
    "risk_grade_migration": {
        "route_id": "mi", "risk_monitor_mode": "migration",
        "migration_dimension": "internal_risk_grade",
        "baseline_date": "2024-01-31", "current_date": "2024-03-31",
        "snapshot_client_id": CLIENT,
    },
    # 11. IFRS9 migration.
    "ifrs9_migration": {
        "route_id": "mi", "risk_monitor_mode": "migration",
        "migration_dimension": "ifrs9_stage",
        "baseline_date": "2024-01-31", "current_date": "2024-03-31",
        "snapshot_client_id": CLIENT,
    },
    # 12. PD bucket migration.
    "pd_bucket_migration": {
        "route_id": "mi", "risk_monitor_mode": "migration",
        "migration_dimension": "pd_bucket",
        "baseline_date": "2024-01-31", "current_date": "2024-03-31",
        "snapshot_client_id": CLIENT,
    },
    # 13. Funded by portfolio using the Trakt portfolio reference (portfolio_id).
    "funded_by_portfolio": {
        "route_id": "mi", "risk_monitor_mode": "concentration",
        "state": "total_funded", "concentration_dimension": "portfolio_id",
        "snapshot_client_id": CLIENT,
    },
    # 14. Quantile balance bands over a funded state (asset-agnostic default).
    "funded_by_balance_quantile": {
        "route_id": "mi", "risk_monitor_mode": "concentration",
        "state": "total_funded", "concentration_dimension": "balance_band",
        "bucket_strategy": "quantile", "bucket_count": 4,
        "bucket_field": "current_outstanding_balance",
        "snapshot_client_id": CLIENT,
    },
}

# Specs that MUST fail validation (used in tests + the interpretation contract).
INVALID_EXAMPLES: Dict[str, Dict[str, Any]] = {
    # Ambiguous bare term not resolved.
    "ambiguous_stage_dimension": {
        "route_id": "mi", "risk_monitor_mode": "concentration",
        "state": "total_pipeline", "concentration_dimension": "stage",
        "snapshot_client_id": CLIENT,
    },
    # Compare without both dates.
    "compare_missing_dates": {
        "route_id": "mi", "state": "total_funded", "temporal_mode": "compare",
        "snapshot_client_id": CLIENT,
    },
    # State without client id.
    "state_missing_client": {
        "route_id": "mi", "state": "total_funded", "temporal_mode": "latest",
    },
    # M&A route asking for a pipeline state.
    "mna_pipeline_state": {
        "route_id": "mna", "state": "total_pipeline", "temporal_mode": "latest",
        "snapshot_client_id": CLIENT,
    },
    # Regulatory route asking for an MI state.
    "regulatory_state": {
        "route_id": "regulatory_annex2", "state": "total_funded",
        "temporal_mode": "latest", "snapshot_client_id": CLIENT,
    },
    # Risk mode without a mode.
    "risk_without_mode": {
        "route_id": "mi", "risk_monitor": True,
        "dimension": "internal_risk_grade", "snapshot_client_id": CLIENT,
    },
    # Invalid enum.
    "invalid_state_enum": {
        "route_id": "mi", "state": "total_unknown", "snapshot_client_id": CLIENT,
    },
}
