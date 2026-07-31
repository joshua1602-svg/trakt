"""operations_control.occ_agent — the OCC Agent's synthetic onboarding capability.

The natural-language operating layer over the existing Operations Control
Centre. It creates and drives **synthetic** onboarding cases from an initial
instruction through to ``READY_FOR_EXECUTION``, reusing the platform's real
onboarding configuration, agents, gates and controls — and never touching live
storage, the live pipeline, production configuration, email or publication.

Layering, innermost first:

    policy      the synthetic execution boundary (no permissions, fail closed)
    states      the case lifecycle and its legal transitions
    case/store  the typed case document and its isolated persistence
    vocabulary  what the agent recognises, read from platform configuration
    pack        the onboarding pack, built from the configuration framework
    artefacts   synthetic client responses and their intended live locations
    configuration  the client configuration candidate, via the real resolver
    execution   the synthetic adapter, driven by the real orchestration conductor
    readiness   the deterministic READY_FOR_EXECUTION decision and its package
    interpretation  natural language in, typed proposals out
    service     the bounded tool surface everything above is reached through
    api         the FastAPI router mounted into the existing OCC API

Nothing in this package is imported by the live OCC path, and the tab it serves
is off unless ``OCC_AGENT_SYNTHETIC_ENABLED`` is set.
"""

from __future__ import annotations

from .policy import (
    FEATURE_FLAG_ENV,
    RUNTIME_MODE_SYNTHETIC,
    SyntheticBoundaryError,
    SyntheticPolicy,
    feature_enabled,
    synthetic_policy,
)
from .service import OccAgentService
from .states import READY_FOR_EXECUTION

__all__ = [
    "FEATURE_FLAG_ENV",
    "OccAgentService",
    "READY_FOR_EXECUTION",
    "RUNTIME_MODE_SYNTHETIC",
    "SyntheticBoundaryError",
    "SyntheticPolicy",
    "feature_enabled",
    "synthetic_policy",
]
