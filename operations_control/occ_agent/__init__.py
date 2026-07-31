"""operations_control.occ_agent — the onboarding operating process.

The OCC Agent coordinates the whole of an onboarding, from a human's first
instruction to starting the existing Onboarding Agent on the client's first real
delivery. Every part of the work already exists somewhere in Trakt; what did not
exist is the process joining them:

* **Client Onboarding** (:mod:`operations_control.onboarding`) — the governed
  standing-configuration capability, with its own case model, field catalogue,
  validation, information requests, approval and activation. The OCC Agent
  *drives* it; it does not reimplement any part of it.
* **the delivery pipeline** — the Onboarding Agent, the Orchestration Agent, the
  Assembler Agent and the governed gates.
* **the OCC's engine** — the governed input pack: ``create_batch``,
  ``upload_batch_files``, ``start_batch``.

Somebody has to decide what to ask a client, ask it, take the answers in
whatever form they arrive, record them with their provenance, resolve what is
missing, put a complete package in front of a human — and only then activate and
start ingestion. That is this package.

**One workflow, two effects.** Everything up to the human's approval is
identical whether a case is being rehearsed or performed. Only what happens at
the end differs, and that is a single injected
:class:`~operations_control.occ_agent.adapters.ExecutionAdapter`. The synthetic
one refuses activation and is audited for it; the live one sequences the
platform's own governed path. Live execution is off unless
``OCC_AGENT_LIVE_ENABLED`` says otherwise, and every precondition is checked in
one place.

Layering, innermost first::

    policy          the synthetic execution boundary (no permissions, fail closed)
    states          the EXECUTION lifecycle and its legal transitions
    run/store       the execution record beside a case, and its isolated storage
    input_roles     what a delivered file can be, from platform configuration
    derive          execution facts, read off the onboarding case
    extraction      catalogue-derived recognition: what a sentence names
    planning        what an instruction WOULD do, and what it could not read
    artefacts       client responses and their intended live locations
    execution       the rehearsal adapter, driven by the real conductor
    pack            the client-facing pack, projected from the catalogue
    communication   how a pack is issued, and whether anything was actually sent
    readiness       the deterministic READY_FOR_EXECUTION decision
    review          the complete package a human approves
    adapters        the one activation gate, and the two kinds of effect
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
from .adapters import (
    LIVE_FLAG_ENV,
    ActivationRefused,
    ExecutionAdapter,
    LiveExecutionAdapter,
    SyntheticExecutionAdapter,
    live_enabled,
)
from .service import AgentCase, OccAgentService, PartiallyUnderstood
from .states import INGESTION_STARTED, READY_FOR_EXECUTION

__all__ = [
    "ActivationRefused",
    "AgentCase",
    "CAP_ACTIVATE_CONFIGURATION",
    "ExecutionAdapter",
    "FEATURE_FLAG_ENV",
    "INGESTION_STARTED",
    "LIVE_FLAG_ENV",
    "LiveExecutionAdapter",
    "OccAgentService",
    "PartiallyUnderstood",
    "READY_FOR_EXECUTION",
    "RUNTIME_MODE_SYNTHETIC",
    "SyntheticBoundaryError",
    "SyntheticExecutionAdapter",
    "SyntheticPolicy",
    "feature_enabled",
    "live_enabled",
    "synthetic_policy",
]
