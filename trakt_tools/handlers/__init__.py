"""trakt_tools.handlers — the registered tools.

Importing this package registers every tool. Each handler module keeps its heavy
imports (pandas, storage, the analytical engines) *inside* the handler function,
so importing the registry to publish the catalogue or generate the OpenAPI
document costs nothing.

Adding a tool is: write the handler over an EXISTING implementation, declare its
schemas, register it here. If a tool would need new calculation logic, it is not
a tool yet — the calculation belongs in the domain, with the UI and the agent
both calling it.
"""

from __future__ import annotations

from trakt_core.context import SCOPE_RISK_READ

from ..registry import register
from ..spec import ToolSpec
from . import covenants as _covenants

register(ToolSpec(
    name="evaluate_covenants",
    version="1.0.0",
    description=(
        "Evaluate the operator-approved concentration and covenant tests for a "
        "portfolio resource. Returns each test's current value, threshold, "
        "operator, utilisation, headroom, breach amount, status and movement "
        "against the prior governed reporting period, plus the configuration "
        "version and approver that produced them."),
    agent_guidance=(
        "Use this to find out whether a book is within its concentration limits "
        "and covenants, and by how much. Every number is computed by Trakt from "
        "governed data under an operator-approved definition — never estimate or "
        "recompute one yourself. Check 'source': only 'approved_configuration' is "
        "operator-approved."),
    input_schema=_covenants.INPUT_SCHEMA,
    output_schema=_covenants.OUTPUT_SCHEMA,
    required_capability=SCOPE_RISK_READ,
    handler=_covenants.evaluate_covenants,
))

__all__ = ["_covenants"]
