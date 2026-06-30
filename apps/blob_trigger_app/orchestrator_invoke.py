"""apps.blob_trigger_app.orchestrator_invoke — the Orchestrator boundary.

The trigger decides WHAT arrived; the Orchestrator Agent decides WHAT TO DO. This
is the single seam the trigger calls. The default implementation wraps
``engine.orchestrator_agent.run_orchestration``; tests inject a recording stub.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Optional

# An orchestrator invoker takes the routing decision and returns a small dict:
#   {"run_id", "status", "central_canonical_path", ...}
OrchestratorInvoker = Callable[..., Dict[str, Any]]


def default_orchestrator_invoker(
    *,
    processing_mode: str,          # "deterministic" | "source_onboarding"
    client_id: str,
    source_portfolio_id: str,
    source_portfolio_type: Optional[str],
    dataset: str,
    frequency: str,
    reporting_period: str,
    input_path: str,
    target: str,                   # "mi" | "all"
    run_regime: bool,
    mapping_config_path: Optional[str],
    out_dir: str,
    acquisition_date: Optional[str] = None,   # acquired portfolios (from _READY.json/registry)
    seller_name: Optional[str] = None,        # acquired portfolios (from _READY.json/registry)
    regime: str = "ESMA_Annex2",
) -> Dict[str, Any]:
    """Invoke the real Orchestrator Agent. Onboarding mode (mi_only vs
    regulatory_mi) follows whether regime output is in scope; processing_mode
    selects discovery vs saved-mapping deterministic processing."""
    from engine.orchestrator_agent import run_orchestration
    from engine.orchestrator_agent.adapters import RealAgentAdapters, PortfolioSpec

    adapters = RealAgentAdapters(
        client_name=client_id,
        onboarding_mode=("regulatory_mi" if run_regime else "mi_only"),
        processing_mode=processing_mode,
        mapping_config_path=mapping_config_path,
    )
    spec = PortfolioSpec(
        source_portfolio_id=source_portfolio_id, input=input_path,
        source_portfolio_type=source_portfolio_type,
        acquisition_date=acquisition_date,
        seller_name=seller_name,
        # If the acquisition_date is supplied (acquired packs), require it; only
        # tolerate "unknown" when none was provided (direct books).
        allow_unknown_acquisition_date=(acquisition_date is None),
    )
    state = run_orchestration(
        client_id, [spec], target=target, regime=(regime if run_regime else None),
        out_root=out_dir, adapters=adapters,
        created_at=datetime.now(timezone.utc).isoformat())
    return {
        "run_id": state.run_id,
        "status": state.status,                       # done | halted | failed
        "central_canonical_path": state.central_canonical_path,
        "blockers": state.blockers,
        "state_path": str(state.state_path()),
    }
