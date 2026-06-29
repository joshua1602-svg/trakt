"""engine.orchestrator_agent — the governed Agentic Orchestration conductor.

A deterministic top-level orchestrator that drives the existing Trakt agents in
a fixed, auditable DAG using their manifest handoffs and ``ready_for_*`` gates:

    Onboarding ─▶ Transformation ─▶ Validation ─▶ (stamp provenance)   [per portfolio]
                                                       │
                                                       ▼
                                                  Assembler ─▶ central canonical
                                                       │
                                                       ▼
                                                  MI (route)

It never re-implements an agent: each stage calls the agent's existing callable
and reads its manifest. It adds the conductor (sequencing, per-portfolio
fan-out), the gate engine (halt on a False readiness flag / blocking exception /
mapping review), per-portfolio provenance assignment, multi-portfolio
consolidation via the Assembler Agent, MI routing, and a resumable run-state for
diligence lineage.

This is the new orchestration spine replacing the legacy ``function_app.py``
Azure-blob trigger. Agent internals, canonical/MI calculations and the
Regime/Annex 2 logic are unchanged.
"""

from .state import (  # noqa: F401
    RunState,
    PortfolioState,
    StepState,
    STEP_PENDING,
    STEP_RUNNING,
    STEP_DONE,
    STEP_HALTED,
    STEP_FAILED,
    STEP_SKIPPED,
)
from .adapters import (  # noqa: F401
    AgentAdapters,
    RealAgentAdapters,
    PortfolioSpec,
    StepResult,
)
from .orchestrator import run_orchestration, new_run_id  # noqa: F401
