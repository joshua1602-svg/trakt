"""mi_workflows.analytical — the analytical capability layer.

A thin layer ABOVE the governed MI Query architecture. It adds one thing the
architecture deliberately did not have: the ability to answer a question whose
answer is more than one deterministic capability.

::

    question
      └─► planner          which governed capabilities, and why      (data, not prose)
            └─► registry   validate the plan against declared inputs
                  └─► executors   the EXISTING deterministic engines
                        └─► findings    structured, evidenced results
                              └─► narrative   composed from findings only

What it is not
--------------
It is not a second MI Agent, a second parser, a second field registry, a second
business-semantics registry or a second chart engine. It owns no financial
calculation: every number in an analytical answer is produced by
``mi_workflows.engine``, ``mi_agent.period_change``, ``mi_agent_api.cohorts``,
``mi_agent_api.pipeline_contract``, ``mi_agent_api.forecast_bridge``,
``mi_agent_api.forecast_extrapolation`` or
``mi_agent_api.concentration_tests_api``.

It is additive by construction. The planner declines every question a single
existing route already answers (``AnalyticalCapability.route_owner``), and the
handler defers when a plan produces nothing computable — so a question the
product answers correctly today keeps exactly the answer it has.
"""

from __future__ import annotations

from .contract import (  # noqa: F401
    COMPOSITION_VERSION,
    AnalyticalCapability,
    AnalyticalPlan,
    CapabilityCall,
    Finding,
    PeriodRef,
    PopulationRef,
)
from .context import AnalyticalContext  # noqa: F401
from .orchestrator import AnalyticalResult, run  # noqa: F401
from .planner import plan_for, plan_from_proposal, validate  # noqa: F401
from .registry import CAPABILITIES, CapabilityRegistry  # noqa: F401
from .route import ROUTE_NAME, recogniser  # noqa: F401

__all__ = [
    "CAPABILITIES", "COMPOSITION_VERSION", "ROUTE_NAME",
    "AnalyticalCapability", "AnalyticalContext", "AnalyticalPlan",
    "AnalyticalResult", "CapabilityCall", "CapabilityRegistry", "Finding",
    "PeriodRef", "PopulationRef",
    "plan_for", "plan_from_proposal", "recogniser", "run", "validate",
]
