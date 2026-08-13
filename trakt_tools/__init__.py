"""trakt_tools — the governed agent tool layer.

The single surface an external AI agent reaches Trakt through, and the single
place a Trakt business capability is declared as agent-callable.

    external agent  ──►  mi_agent_api.agent_api      (transport + identity)
                    ──►  trakt_tools.execute_governed_tool
                            1 tool exists
                            2 arguments match the published schema
                            3 caller holds the capability
                            4 organisation is entitled to the named resource
                            5 dataset is approved to answer
                            6 EXISTING deterministic implementation runs
                            7 GovernedResult
                            8 audit
                    ──►  mi_agent / analytics_lib / engine   (unchanged)

Three rules hold by construction, and each has a test:

* **No tool computes anything.** Every handler wraps an implementation the UI
  already calls, so an agent and a human cannot be given different numbers.
* **No web framework below the adapter.** This package imports no FastAPI; the
  registry imports nothing heavy at all, so publishing the catalogue or
  generating the OpenAPI document is cheap.
* **Governance runs before data.** Steps 1-5 touch no dataframe.

The registry is also the one source for every agent-facing surface — the REST
tool API, the generated OpenAPI document, a future MCP server and the synthetic
agents all read it, so a tool cannot exist on one surface and not another.
"""

from __future__ import annotations

from .execution import ToolDependencies, build_dependencies, execute_governed_tool
from .registry import all_tools, catalogue, get, register, tool_names
from .schema import SchemaError, object_schema, validate
from .spec import ToolHandler, ToolInvocation, ToolSpec

__all__ = [
    "ToolSpec", "ToolInvocation", "ToolHandler",
    "register", "get", "all_tools", "tool_names", "catalogue",
    "execute_governed_tool", "ToolDependencies", "build_dependencies",
    "validate", "object_schema", "SchemaError",
]
