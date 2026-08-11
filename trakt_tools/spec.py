"""trakt_tools.spec — what a governed agent tool *is*.

A :class:`ToolSpec` is a declaration, not an implementation. It names the tool,
its typed contract, the capability a caller must hold, and the handler that does
the work — and the handler is always an existing Trakt implementation, never
logic written for agents.

That rule is the whole point of the registry. If a tool's handler computed
anything, the agent surface would become a second calculator and could disagree
with the UI. ``tests/test_agent_tool_registry.py`` asserts the covenant tool's
handler reaches the same module the React route does.

The invocation context
----------------------
A handler receives ``(arguments, invocation)``. Everything it is allowed to know
about *who is asking* arrives on the :class:`ToolInvocation` — already
authorised — so a handler never re-derives identity, never re-checks permission,
and has no way to widen its own authority:

    invocation.context      the trusted ExecutionContext
    invocation.authorised   an AuthorisedResource: proof of the permission check
    invocation.snapshot     the dataset identity governance already approved
    invocation.dependencies injectable collaborators (tests substitute here)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Mapping, Optional

from trakt_core.context import KNOWN_CAPABILITIES, ExecutionContext
from trakt_core.entitlement import AuthorisedResource
from trakt_core.envelope import SnapshotRef

from .schema import SchemaError, assert_supported_schema


@dataclass(frozen=True)
class ToolInvocation:
    """One authorised tool call, handed to a handler."""

    context: ExecutionContext
    authorised: AuthorisedResource
    snapshot: Optional[SnapshotRef] = None
    dependencies: Any = None

    @property
    def tenant_id(self) -> str:
        return self.context.tenant_id

    @property
    def request_id(self) -> str:
        return self.context.request_id


#: ``(arguments, invocation) -> result payload``
ToolHandler = Callable[[Dict[str, Any], ToolInvocation], Dict[str, Any]]


@dataclass(frozen=True)
class ToolSpec:
    """One agent-callable Trakt capability.

    ``version`` is part of the published contract from the first tool onward,
    deliberately: a client agent that pins a version is the only way a breaking
    change to a tool can later be made without silently changing what an
    already-deployed agent believes it is calling.
    """

    name: str
    version: str
    description: str
    input_schema: Dict[str, Any]
    output_schema: Dict[str, Any]
    #: The capability a caller must hold, from
    #: :data:`trakt_core.context.KNOWN_CAPABILITIES`. Deliberately the same
    #: vocabulary the entitlement grants use, so there is nothing to translate.
    required_capability: str
    handler: ToolHandler
    #: The input property naming the resource this call is authorised against.
    #: Every tool has one: an agent names a *resource*, never a dataset or a
    #: storage location, and the server produces the predicate.
    resource_argument: str = "resource"
    #: One sentence on what an agent should use this for, in its own terms.
    #: Published in the catalogue alongside ``description``.
    agent_guidance: str = ""

    def __post_init__(self) -> None:
        if not str(self.name or "").strip():
            raise ValueError("ToolSpec requires a name")
        if not str(self.version or "").strip():
            raise ValueError(f"tool {self.name!r} requires a version")
        if self.required_capability not in KNOWN_CAPABILITIES:
            raise ValueError(
                f"tool {self.name!r} requires capability "
                f"{self.required_capability!r}, which is not in "
                "trakt_core.context.KNOWN_CAPABILITIES. Add it there first — "
                "the grant vocabulary and the tool vocabulary are the same set "
                "on purpose.")
        if not callable(self.handler):
            raise ValueError(f"tool {self.name!r} requires a callable handler")

        # A schema keyword the executor cannot enforce must not reach the
        # published contract, so this is a registration-time failure.
        try:
            assert_supported_schema(self.input_schema, path=f"{self.name}.input")
            assert_supported_schema(self.output_schema, path=f"{self.name}.output")
        except SchemaError as exc:
            raise ValueError(str(exc)) from exc

        properties: Mapping[str, Any] = self.input_schema.get("properties") or {}
        if self.resource_argument not in properties:
            raise ValueError(
                f"tool {self.name!r} declares resource_argument "
                f"{self.resource_argument!r} but its input schema has no such "
                "property; a tool with no resource cannot be authorised.")
        if self.resource_argument not in (self.input_schema.get("required") or ()):
            raise ValueError(
                f"tool {self.name!r} must mark {self.resource_argument!r} as "
                "required: an unauthorised-by-omission call must be impossible.")

    # -- publication -------------------------------------------------------- #
    def to_catalogue_entry(self) -> Dict[str, Any]:
        """The published declaration. This is what an agent reads, what the
        OpenAPI document is generated from, and what an MCP adapter would
        translate — one source, three surfaces."""
        return {
            "name": self.name,
            "version": self.version,
            "description": self.description,
            "agent_guidance": self.agent_guidance,
            "required_capability": self.required_capability,
            "resource_argument": self.resource_argument,
            "input_schema": self.input_schema,
            "output_schema": self.output_schema,
        }

    @property
    def capability_id(self) -> str:
        """The identifier this tool's executions are audited under."""
        return f"tool.{self.name}"
