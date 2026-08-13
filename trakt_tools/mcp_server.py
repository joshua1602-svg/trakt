"""trakt_tools.mcp_server — the MCP server the adapter was waiting for.

``trakt_tools.mcp`` translates the registry into MCP and back, and deliberately
stops short of a server because a server needs an identity integration and a
session lifecycle, and those are decisions rather than translations. This module
makes those decisions for an **in-process** transport, which is what Sprint 4
needs: a real MCP boundary between the Securitisation Readiness Agent and Trakt's
governed tools, without standing up a network listener the sprint does not
require.

What makes this a boundary rather than a passthrough
----------------------------------------------------
The server owns the :class:`~trakt_core.context.ExecutionContext`. A caller
supplies a tool name and arguments and **nothing else** — no tenant, no actor, no
scopes. Identity arrives once, when the session is constructed, from whatever
authenticated the transport, and every call made through that session carries it.
:func:`~trakt_tools.mcp.refuse_identity_in_arguments` is enforced on every call,
so a caller that tries to name its own tenant is rejected rather than quietly
stripped.

That is the whole point of putting MCP here. Sprint 3's agent held an
``ExecutionContext`` and called ``execute_governed_tool`` itself. It behaved
correctly, but the *structure* let it behave otherwise. With the server in
between, the agent no longer holds a context it could alter — it holds a session
that was built with one.

What this is NOT
----------------
Not a second tool interface. ``call_tool`` reaches ``execute_governed_tool``,
the same entry point the HTTP surface and the Sprint 3 session use, and does no
filtering, no aggregation and no arithmetic of its own. If a number differs
depending on whether it arrived over MCP, this module is broken.

Not a network server. There is no socket, no session negotiation and no stdio
framing. Adding those is a transport change and would not alter anything below
this line — which is the property worth having.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Tuple

from trakt_core.context import ExecutionContext
from trakt_core.envelope import STATUS_SUCCESS

from .execution import ToolDependencies, execute_governed_tool
from .mcp import (MCP_ADAPTER_VERSION, MCP_SERVER_NAME,
                  refuse_identity_in_arguments, tool_declarations, tool_result)

#: The MCP protocol revision this server implements. Recorded rather than
#: assumed, so a future upgrade is a visible change and not a silent drift.
MCP_PROTOCOL_VERSION = "2025-06-18"


@dataclass(frozen=True)
class McpServerInfo:
    """What ``initialize`` would return over a network transport."""

    name: str = MCP_SERVER_NAME
    version: str = MCP_ADAPTER_VERSION
    protocol_version: str = MCP_PROTOCOL_VERSION

    def to_dict(self) -> Dict[str, Any]:
        return {
            "protocolVersion": self.protocol_version,
            "serverInfo": {"name": self.name, "version": self.version},
            "capabilities": {"tools": {"listChanged": False}},
        }


@dataclass
class McpSession:
    """One authenticated MCP session over Trakt's governed tools.

    The context is supplied at construction by whatever authenticated the
    transport and is never rebuilt from a call argument. ``calls`` is a count,
    not a log — the audit trail is emitted inside ``execute_governed_tool`` and
    duplicating it here would create a second record that could disagree with
    the first.
    """

    context: ExecutionContext
    dependencies: Optional[ToolDependencies] = None
    info: McpServerInfo = field(default_factory=McpServerInfo)
    calls: int = 0

    # -- discovery ---------------------------------------------------------- #
    def initialize(self) -> Dict[str, Any]:
        return self.info.to_dict()

    def list_tools(self) -> Dict[str, Any]:
        """``tools/list`` — narrowed to what THIS caller may use.

        Narrowing comes from the registry's own catalogue rather than being
        re-derived here, so a tool this session may not call is not advertised
        to it and an autonomous caller does not spend a turn discovering a
        refusal.
        """
        return {"tools": tool_declarations(self.context)}

    # -- execution ---------------------------------------------------------- #
    def call_tool(self, name: str, arguments: Optional[Mapping[str, Any]] = None
                  ) -> Dict[str, Any]:
        """``tools/call`` — one governed execution, as an MCP result.

        A governance refusal comes back as ``isError`` with the envelope intact,
        not as an exception: a refusal is information the caller must be able to
        reason about, and an MCP client that received a bare exception would have
        nothing to reason with.
        """
        args = dict(arguments or {})
        refuse_identity_in_arguments(args)
        self.calls += 1
        result = execute_governed_tool(name, args, self.context,
                                       dependencies=self.dependencies)
        # GovernedResult.to_dict is the single serialisation of an envelope.
        # A second one here could disagree with the HTTP surface's, and the
        # first symptom would be a number that differs by transport.
        return tool_result(result.to_dict())


def governed_payload(mcp_result: Mapping[str, Any]) -> Tuple[Dict[str, Any], str,
                                                             Optional[str]]:
    """Unpack an MCP result into what a governed session records.

    Returns ``(payload, status, error_code)`` in the shape
    :class:`~readiness_agent.session.GovernedSession` already produces for a
    direct call, so the agent above cannot tell which transport it is on. That
    indistinguishability is the Sprint 4 equivalence claim, expressed as code
    rather than as a paragraph.
    """
    envelope = dict(mcp_result.get("structuredContent") or {})
    status = envelope.get("status") or (
        "error" if mcp_result.get("isError") else STATUS_SUCCESS)

    if status == STATUS_SUCCESS:
        return dict(envelope.get("result") or {}), status, None

    error = envelope.get("error") or {}
    code = error.get("code") or "REFUSED"
    return ({"refused": True, "error_code": str(code),
             "message": error.get("message") or ""}, status, str(code))


class McpToolTransport:
    """A governed-session transport that goes through MCP.

    Constructed with a session rather than a context, so the only way to widen
    what it can reach is to be given a different session — which the agent above
    has no way to do.
    """

    def __init__(self, session: McpSession) -> None:
        self._session = session

    @property
    def calls(self) -> int:
        return self._session.calls

    def __call__(self, tool: str, arguments: Mapping[str, Any]
                 ) -> Tuple[Dict[str, Any], str, Optional[str]]:
        return governed_payload(self._session.call_tool(tool, arguments))


def mcp_session(context: ExecutionContext,
                dependencies: Optional[ToolDependencies] = None) -> McpSession:
    return McpSession(context=context, dependencies=dependencies)


#: What a *network* MCP deployment would still owe, over and above this module.
#: Enumerated so nobody reads an in-process server as a shipped one.
NETWORK_TRANSPORT_RESPONSIBILITIES = (
    "terminate TLS and authenticate the session, then build the "
    "ExecutionContext from that authenticated principal",
    "frame stdio or HTTP+SSE per the MCP transport spec",
    "bound concurrency, payload size and session lifetime",
    "surface protocolVersion negotiation rather than asserting one",
)
