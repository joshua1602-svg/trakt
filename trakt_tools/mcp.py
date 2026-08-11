"""trakt_tools.mcp — translating the Trakt tool registry into MCP, and back.

MCP is a **transport and a declaration format**. It is not where business logic
goes, and this module is deliberately shaped to make that structurally true: it
contains four pure functions and no handler, no authorisation, no data access and
no governance decision. An MCP server built on it is a projection of the registry
in exactly the way ``deploy/agent-api/trakt-agent-openapi.yaml`` already is.

The mapping, which is close to one-to-one because both sides speak JSON Schema:

    ToolSpec                     -> MCP tool declaration
      .name                           name
      .description + guidance         description
      .input_schema                   inputSchema      (verbatim)
      .output_schema                  outputSchema     (verbatim)

    ExecutionContext             <- MCP caller context
      the SERVER builds it from the transport's authenticated identity.
      NEVER from a tool argument. This is the load-bearing line: MCP has no
      identity model of its own, so an MCP server in front of Trakt is an
      authentication boundary, and a server that let a caller pass its own
      tenant_id would hand every caller every tenant.

    GovernedResult               -> MCP tool result
      .result                         structuredContent
      a rendered digest               content[0] (text)
      status != success               isError = true

Why the digest exists
---------------------
An MCP client shows ``content`` to its model. Handing over the raw envelope would
put a hundred lines of policy and audit metadata into a context window before any
credit fact appears. The digest states the outcome, the snapshot the numbers came
from, and the warnings — and ``structuredContent`` carries the whole envelope for
anything that wants it. Nothing is dropped; it is ordered.

What is deliberately NOT here
-----------------------------
No server. A server needs a transport, a session lifecycle and an authentication
integration, and each of those is a decision with an operational owner rather
than a translation. What IS here is the part that would otherwise get rewritten
per transport — and it is covered by tests, so the day a server is added, the
mapping is not the risk.
"""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Sequence

from trakt_core.context import ExecutionContext

from .spec import ToolSpec

#: The Trakt-side identifier an MCP deployment is published under.
MCP_SERVER_NAME = "trakt"

#: Bumped when this translation changes shape, independently of any tool version.
MCP_ADAPTER_VERSION = "1.0.0"


# --------------------------------------------------------------------------- #
# ToolSpec -> MCP tool declaration
# --------------------------------------------------------------------------- #
def tool_declaration(spec: ToolSpec) -> Dict[str, Any]:
    """One MCP tool declaration from one :class:`ToolSpec`.

    The schemas are passed through **verbatim**. Rewriting them for MCP would
    create a second contract that could drift from the OpenAPI one, and a tool
    whose shape depends on which door you came through is not one tool.
    """
    description = spec.description
    if spec.agent_guidance:
        # Guidance is part of the description here because MCP has no separate
        # field for it, and a client that cannot see the guidance will use the
        # tool the way the description implies rather than the way it should.
        description = f"{description}\n\nWhen to use: {spec.agent_guidance}"
    return {
        "name": spec.name,
        "title": spec.name.replace("_", " ").title(),
        "description": description,
        "inputSchema": spec.input_schema,
        "outputSchema": spec.output_schema,
        "_meta": {
            "trakt/version": spec.version,
            "trakt/required_capability": spec.required_capability,
            "trakt/resource_argument": spec.resource_argument,
            "trakt/adapter_version": MCP_ADAPTER_VERSION,
        },
    }


def tool_declarations(context: ExecutionContext, *, registry=None
                      ) -> List[Dict[str, Any]]:
    """The tools THIS caller may use, as MCP declarations.

    Narrowed by the same catalogue the HTTP surface publishes, because an agent
    offered a tool it will always be refused wastes a call and then reasons about
    a refusal. MCP's ``tools/list`` is per-session, so the narrowing has a
    natural home — but it must come from the registry's own narrowing rather than
    being re-derived here.
    """
    from . import registry as registry_mod

    registry = registry or registry_mod
    return [tool_declaration(spec) for spec in registry.all_tools()
            if _permitted(spec, context, registry)]


def _permitted(spec: ToolSpec, context: ExecutionContext, registry) -> bool:
    catalogue = registry.catalogue(context)
    return any(entry["name"] == spec.name for entry in catalogue["tools"])


# --------------------------------------------------------------------------- #
# MCP caller context -> ExecutionContext
# --------------------------------------------------------------------------- #
def refuse_identity_in_arguments(arguments: Mapping[str, Any]) -> None:
    """Refuse a call whose ARGUMENTS carry identity claims.

    The single most dangerous thing an MCP adapter could do is accept
    ``tenant_id`` from the tool arguments. MCP has no identity model of its own,
    so the temptation is real and the failure is total: one caller naming another
    tenant reads that tenant's book.

    Trakt's rule is already that :class:`ExecutionContext` is built from
    *validated transport identity only* — never from request-body fields. This
    function makes the rule enforceable at the adapter boundary rather than
    merely documented, and it raises rather than stripping, because a caller that
    sent a tenant_id believed it meant something.
    """
    forbidden = {"tenant_id", "tenantId", "organisation_id", "organisationId",
                 "actor_id", "actorId", "principal", "scopes", "capabilities",
                 "entitlements", "channel"}
    present = sorted(forbidden & set(arguments or {}))
    if present:
        raise ValueError(
            "identity may not be passed as tool arguments: "
            f"{', '.join(present)}. The MCP server builds the ExecutionContext "
            "from the transport's authenticated principal, because an argument "
            "a caller controls cannot establish who that caller is.")


# --------------------------------------------------------------------------- #
# GovernedResult -> MCP tool result
# --------------------------------------------------------------------------- #
def _digest(envelope: Mapping[str, Any]) -> str:
    """The few lines an MCP client should put in front of a model first."""
    status = envelope.get("status")
    capability = envelope.get("capability")
    lines: List[str] = []

    if status == "success":
        lines.append(f"{capability}: success")
    else:
        error = envelope.get("error") or {}
        lines.append(f"{capability}: {status}"
                     + (f" — {error.get('code')}" if error.get("code") else ""))
        if error.get("message"):
            lines.append(error["message"])
        if error.get("retryable") is False:
            # So an autonomous caller stops rather than looping on a decision.
            lines.append("This is a governance decision, not a transient "
                         "failure. Do not retry.")

    snapshot = envelope.get("snapshot") or {}
    if snapshot.get("snapshot_id") or snapshot.get("dataset_label"):
        lines.append(
            "Source: "
            + " ".join(str(v) for v in (snapshot.get("dataset_label"),
                                        snapshot.get("snapshot_id"),
                                        snapshot.get("content_hash")) if v))

    for warning in (envelope.get("warnings") or ())[:6]:
        lines.append(f"Warning: {warning}")

    lines.append("The full governed envelope is in structuredContent. Every "
                 "figure in it was computed by Trakt; do not recompute one.")
    return "\n".join(lines)


def tool_result(envelope: Mapping[str, Any]) -> Dict[str, Any]:
    """An MCP ``CallToolResult`` from a serialised :class:`GovernedResult`.

    ``isError`` is set for a governance refusal as well as a failure, because an
    MCP client that treats a refusal as a successful result will feed a blocked
    envelope to a model as though it were data.
    """
    status = envelope.get("status")
    return {
        "content": [{"type": "text", "text": _digest(envelope)}],
        "structuredContent": dict(envelope),
        "isError": status != "success",
    }


# --------------------------------------------------------------------------- #
# What a server would still have to do
# --------------------------------------------------------------------------- #
#: Enumerated rather than left implicit, so the remaining work is visible and
#: nobody assumes this module is a server.
SERVER_RESPONSIBILITIES = (
    "authenticate the session and build the ExecutionContext from the "
    "authenticated principal — never from tool arguments "
    "(see refuse_identity_in_arguments)",
    "map the transport session to a tenant and organisation the same way "
    "mi_agent_api.agent_auth does for HTTP",
    "call execute_governed_tool and translate the envelope with tool_result",
    "emit the same audit events — the audit trail is per execution and is "
    "already emitted inside execute_governed_tool, so this is a matter of not "
    "bypassing it",
    "bound concurrency and payload size per session",
)
