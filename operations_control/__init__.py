"""operations_control — the Trakt Operations Control Centre backend.

A governed operational layer that sits ABOVE the existing agents. It owns
workflow orchestration, human approvals, persistent scoped rules, operational
visibility, publication gating, audit, recovery and reruns. It never modifies
agent code, registry files or promotion behaviour — it wraps the existing
``engine.orchestrator_agent`` seam and calls existing persistence/promotion
functions as libraries.

Design pack: docs/operations_control_centre/.
"""

__version__ = "0.1.0"
