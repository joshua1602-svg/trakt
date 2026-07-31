"""operations_control.occ_agent — practising an onboarding, end to end.

A natural-language operating layer over two things the Operations Control Centre
already has, joining them for the first time:

* **Client Onboarding** (:mod:`operations_control.onboarding`) — the governed
  standing-configuration capability, with its own case model, field catalogue,
  validation, information requests, approval and activation. The OCC Agent
  *drives* it; it does not reimplement any part of it.
* **the delivery pipeline** — the Onboarding Agent, the Orchestration Agent, the
  Assembler Agent and the governed gates, run over a synthetic execution adapter.

Client Onboarding stops at activation and never runs a pipeline. The pipeline
starts from a client that already exists. Between them sits the question an
operator actually has — *if we onboard this client, will their data go through?*
— and this package answers it without creating anything: an onboarding case is
worked through in conversation to **approved**, and a practice execution then
carries it to ``READY_FOR_EXECUTION``.

The one call the feature never makes is ``OnboardingService.activate()``, which
its own docstring calls "the only place active configuration is created". In
synthetic mode the policy refuses it, and a readiness criterion asserts that no
configuration was written.

Layering, innermost first::

    policy          the synthetic execution boundary (no permissions, fail closed)
    states          the practice EXECUTION lifecycle and its legal transitions
    run/store       the execution record beside a case, and its isolated storage
    input_roles     what a delivered file can be, from platform configuration
    derive          execution facts, read off the onboarding case
    artefacts       practice client responses and their intended live locations
    execution       the synthetic adapter, driven by the real conductor
    readiness       the deterministic READY_FOR_EXECUTION decision and its package
    interpretation  natural language in, catalogue-shaped answers out
    service         the bounded tool surface everything above is reached through
    api             the FastAPI router mounted into the existing OCC API

Nothing in this package is imported by the live OCC path, and the tab it serves
is off unless ``OCC_AGENT_SYNTHETIC_ENABLED`` is set.
"""

from __future__ import annotations

from .policy import (
    CAP_ACTIVATE_CONFIGURATION,
    FEATURE_FLAG_ENV,
    RUNTIME_MODE_SYNTHETIC,
    SyntheticBoundaryError,
    SyntheticPolicy,
    feature_enabled,
    synthetic_policy,
)
from .service import AgentCase, OccAgentService
from .states import READY_FOR_EXECUTION

__all__ = [
    "AgentCase",
    "CAP_ACTIVATE_CONFIGURATION",
    "FEATURE_FLAG_ENV",
    "OccAgentService",
    "READY_FOR_EXECUTION",
    "RUNTIME_MODE_SYNTHETIC",
    "SyntheticBoundaryError",
    "SyntheticPolicy",
    "feature_enabled",
    "synthetic_policy",
]
