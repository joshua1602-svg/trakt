"""mi_agent.risk_monitor — deterministic risk-monitor foundations (Phase 5).

Early-warning and migration analytics built on the Phase 2 snapshot layer, the
Phase 3 state assembler and the Phase 4 temporal layer, reusing the Phase 1
``analytics_lib`` and the Phase 0/0B risk fields + ``config/mi/risk_monitor.yaml``.

Deliberate constraints (Phase 5 scope):
  * pure, deterministic, frame-in/frame-out;
  * no orchestration, no MI Agent runtime wiring, no LLM query routing;
  * no M&A agent runtime, no Azure, no Streamlit/charts/UI;
  * no imports from the legacy ``analytics/`` Streamlit app;
  * no Annex 2 / regulatory changes.
"""

from __future__ import annotations

from .concentration import (
    concentration_movement,
    funded_concentration,
    top_n_concentration,
)
from .migration import classify_change, migration_matrix, per_loan_movement
from .models import (
    RiskMonitorResult,
    load_risk_monitor_config,
    make_issue,
)
from .monitor import (
    run_concentration,
    run_concentration_movement,
    run_funded_vs_forecast,
    run_migration,
    run_trajectory,
    validate_risk_monitor_route,
)

__all__ = [
    # primitives
    "migration_matrix",
    "per_loan_movement",
    "classify_change",
    "funded_concentration",
    "concentration_movement",
    "top_n_concentration",
    # store-backed entry points
    "run_migration",
    "run_concentration",
    "run_concentration_movement",
    "run_funded_vs_forecast",
    "run_trajectory",
    "validate_risk_monitor_route",
    # models / config
    "RiskMonitorResult",
    "load_risk_monitor_config",
    "make_issue",
]
