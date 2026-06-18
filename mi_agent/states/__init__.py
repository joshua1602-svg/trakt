"""mi_agent.states — deterministic MI state assembler foundations (Phase 3).

Pure, deterministic functions that turn Phase 2 ``SnapshotStore`` outputs (or
already-loaded DataFrames) into analytical state DataFrames for later MI queries,
reusing the Phase 1 ``analytics_lib`` and the Phase 0B route/state configs.

Deliberate constraints (Phase 3 scope):
  * no orchestration, no MI Agent runtime wiring, no LLM query routing;
  * no M&A agent, no risk monitor, no temporal-trend/migration runtime;
  * no Azure, no Streamlit/chart output;
  * no imports from the legacy ``analytics/`` Streamlit app;
  * no Annex 2 / regulatory changes.
"""

from __future__ import annotations

from .assembler import (
    assemble_state,
    cohort_by_acquired_portfolio,
    cohort_by_date,
    cohort_by_portfolio,
    cohort_by_spv,
    total_forecast_funded,
    total_funded,
    total_pipeline,
)
from .models import StateResult, make_issue
from .route_contracts import (
    allowed_states,
    canonical_state,
    is_state_allowed,
    validate_state_for_route,
)
from .selectors import SnapshotSelector

__all__ = [
    "assemble_state",
    "total_funded",
    "total_pipeline",
    "total_forecast_funded",
    "cohort_by_date",
    "cohort_by_portfolio",
    "cohort_by_spv",
    "cohort_by_acquired_portfolio",
    "StateResult",
    "make_issue",
    "SnapshotSelector",
    "validate_state_for_route",
    "is_state_allowed",
    "allowed_states",
    "canonical_state",
]
