"""The Forecast tab's snapshot, as ONE function the tab and the agent both call.

It was assembled inline in the ``/mi/forecast/snapshot`` route, so the only way
to read the tab's figures was an HTTP request, and the agent's forecast runtime
had to reach the same numbers through a different composition. Lifted here, the
route resolves its inputs (the funded frame for the selected run and scope, the
latest governed weekly extract) and hands them to this; the agent resolves the
same inputs through its governed context and hands them to this. One
composition, one set of figures (P0 design §16.2).

It composes; it does not compute. Every figure is its owner's:

    forecastBridge              forecast_bridge.compute_forecast_bridge
    forecastBreakdowns          workspace.forecast_breakdowns
    historicalModelEvidence     pipeline_history.historical_model_evidence
    lineage                     workspace.lineage_for

Scope, timing disclosure and per-portfolio projections stay in the route: they
depend on the request's portfolio context, which the agent's forecast refuses
to narrow (a scoped forecast is refused, §8.6).
"""
from __future__ import annotations

from typing import Any, Dict, Optional

import pandas as pd

from . import forecast_bridge as bridge_mod
from . import pipeline_history
from . import workspace as workspace_mod


def compose_forecast_view(*, client_id: str, run_id: str,
                          funded_reporting_date: Optional[str],
                          funded_df: Optional[pd.DataFrame],
                          pipeline_df: Optional[pd.DataFrame],
                          pipeline_report: Optional[Dict[str, Any]],
                          pipeline_snapshot: Optional[Dict[str, Any]],
                          pipeline_source: Optional[Dict[str, Any]] = None
                          ) -> Dict[str, Any]:
    """The Forecast tab's envelope for these inputs: the bridge, its
    by-dimension breakdowns, the completion-probability basis and evidence,
    and the view's lineage."""
    envelope = bridge_mod.compute_forecast_bridge(
        client_id=client_id, run_id=run_id,
        funded_reporting_date=funded_reporting_date,
        funded_df=funded_df, pipeline_df=pipeline_df,
        pipeline_report=pipeline_report, pipeline_snapshot=pipeline_snapshot,
        pipeline_source=pipeline_source)
    # Forecast-by-dimension breakdowns (funded actual + weighted pipeline),
    # derived by aggregate composition — never a row merge.
    envelope["forecastBreakdowns"] = workspace_mod.forecast_breakdowns(
        funded_df, pipeline_df)
    basis = (pipeline_report or {}).get("completion_probability_basis")
    evidence = pipeline_history.historical_model_evidence(
        (pipeline_report or {}).get("historical_completion_model"), basis)
    envelope["historicalModelEvidence"] = evidence
    envelope["completionProbabilityBasis"] = basis
    source = pipeline_source or {}
    envelope["lineage"] = workspace_mod.lineage_for(
        "forecast", funded_reporting_date=funded_reporting_date,
        pipeline_as_of_date=source.get("pipeline_as_of_date"),
        pipeline_source_folder_date=source.get("pipeline_source_folder_date"),
        current_pipeline_snapshot_date=source.get("current_pipeline_snapshot_date"),
        current_pipeline_source_file=source.get("current_pipeline_source_file"),
        completion_probability_basis=basis, historical_model_evidence=evidence)
    return envelope
