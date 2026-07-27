"""trakt_core.context — the trusted execution context.

An :class:`ExecutionContext` is the *authenticated* answer to "who is asking".
It is built by an interface adapter from a verified identity (an Easy Auth
principal, a validated Entra bearer token, a service identity, or an explicit
internal invocation) and is then passed unchanged down through the governed
capability into policy and authorisation.

The distinction this type exists to enforce:

  * ``ExecutionContext`` carries **trusted** facts — tenant, actor, channel,
    scopes. Nothing here may originate from a request body or query string.
  * A capability *request* model carries **untrusted** caller input — which
    portfolio, which question, which filters.

``tenant_id`` therefore never comes from ``portfolio_id``, and a caller cannot
widen its own access by naming a different portfolio: see
:func:`trakt_core.tenancy.authorise_portfolio_access`.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field, replace
from typing import Any, Dict, FrozenSet, Iterable, Optional

# --------------------------------------------------------------------------- #
# Channels — which interface adapter is calling.
# --------------------------------------------------------------------------- #
CHANNEL_REACT = "react"
CHANNEL_COPILOT = "copilot"
CHANNEL_INTERNAL = "internal"           # in-process Python (deck build, CLI, jobs)
CHANNEL_ENTERPRISE_AGENT = "enterprise_agent"   # reserved for a client-owned agent
CHANNEL_AGENT_TO_AGENT = "agent_to_agent"       # reserved for event/workflow callers

KNOWN_CHANNELS: FrozenSet[str] = frozenset({
    CHANNEL_REACT, CHANNEL_COPILOT, CHANNEL_INTERNAL,
    CHANNEL_ENTERPRISE_AGENT, CHANNEL_AGENT_TO_AGENT,
})

# --------------------------------------------------------------------------- #
# Actor types.
# --------------------------------------------------------------------------- #
ACTOR_USER = "user"          # a signed-in human
ACTOR_SERVICE = "service"    # a machine identity (service principal, agent)
ACTOR_SYSTEM = "system"      # trusted in-process invocation (pipeline, CLI)

KNOWN_ACTOR_TYPES: FrozenSet[str] = frozenset({ACTOR_USER, ACTOR_SERVICE, ACTOR_SYSTEM})

# --------------------------------------------------------------------------- #
# Scopes. Deliberately few: this is a capability gate, not an RBAC framework.
# --------------------------------------------------------------------------- #
SCOPE_PORTFOLIO_READ = "portfolio:read"
SCOPE_MI_QUERY = "mi:query"
SCOPE_ARTEFACT_READ = "artefact:read"

#: The scope set granted to an authenticated MI user or an internal caller. New
#: scopes should be added here only when a capability genuinely gates on them.
DEFAULT_MI_SCOPES: FrozenSet[str] = frozenset({
    SCOPE_PORTFOLIO_READ, SCOPE_MI_QUERY, SCOPE_ARTEFACT_READ,
})


def new_request_id() -> str:
    """A fresh request identifier. Adapters should prefer an inbound correlation
    id where the caller supplied one, but must always have a request id."""
    return f"req_{uuid.uuid4().hex[:16]}"


@dataclass(frozen=True)
class ExecutionContext:
    """Trusted caller identity for one governed execution.

    Immutable by construction so a capability cannot widen its own authority
    mid-execution. Use :meth:`with_correlation` / :func:`dataclasses.replace`
    to derive a variant.
    """

    tenant_id: str
    actor_id: str
    actor_type: str = ACTOR_USER
    channel: str = CHANNEL_INTERNAL
    scopes: FrozenSet[str] = field(default_factory=lambda: DEFAULT_MI_SCOPES)
    request_id: str = ""
    correlation_id: Optional[str] = None
    #: Free-form, non-authoritative labels for logs (e.g. display name). Never
    #: consulted for an access decision.
    actor_label: Optional[str] = None

    def __post_init__(self) -> None:
        if not str(self.tenant_id or "").strip():
            raise ValueError("ExecutionContext requires a tenant_id")
        if not str(self.actor_id or "").strip():
            raise ValueError("ExecutionContext requires an actor_id")
        object.__setattr__(self, "tenant_id", str(self.tenant_id).strip())
        object.__setattr__(self, "actor_id", str(self.actor_id).strip())
        if not self.request_id:
            object.__setattr__(self, "request_id", new_request_id())
        if not isinstance(self.scopes, frozenset):
            object.__setattr__(self, "scopes", frozenset(self.scopes or ()))

    # -- scopes ------------------------------------------------------------ #
    def has_scope(self, scope: str) -> bool:
        return scope in self.scopes

    def require_scope(self, scope: str) -> None:
        """Raise :class:`~trakt_core.errors.TraktError` when the scope is absent."""
        if scope not in self.scopes:
            from .errors import ErrorCode, TraktError
            raise TraktError(
                ErrorCode.SCOPE_MISSING,
                f"This caller does not hold the required scope {scope!r}.",
                request_id=self.request_id,
                details={"required_scope": scope},
            )

    # -- derivation -------------------------------------------------------- #
    def with_correlation(self, correlation_id: Optional[str]) -> "ExecutionContext":
        return replace(self, correlation_id=correlation_id or self.correlation_id)

    # -- serialisation ----------------------------------------------------- #
    def to_audit_dict(self) -> Dict[str, Any]:
        """The non-sensitive projection used in audit events and envelopes.

        Deliberately excludes ``actor_label`` (may carry an email address) —
        ``actor_id`` is the stable, minimal identifier.
        """
        return {
            "tenant_id": self.tenant_id,
            "actor_id": self.actor_id,
            "actor_type": self.actor_type,
            "channel": self.channel,
            "request_id": self.request_id,
            "correlation_id": self.correlation_id,
        }

    # -- constructors ------------------------------------------------------ #
    @classmethod
    def for_internal(
        cls,
        tenant_id: str,
        *,
        actor_id: str = "trakt-internal",
        channel: str = CHANNEL_INTERNAL,
        scopes: Optional[Iterable[str]] = None,
        request_id: Optional[str] = None,
        correlation_id: Optional[str] = None,
    ) -> "ExecutionContext":
        """A trusted in-process context (deck build, CLI, scheduled job, tests).

        Legitimate because the caller is already inside the trust boundary — the
        process is the tenant's own runtime. Never construct this from data that
        crossed an interface boundary.
        """
        return cls(
            tenant_id=tenant_id,
            actor_id=actor_id,
            actor_type=ACTOR_SYSTEM,
            channel=channel,
            scopes=frozenset(scopes) if scopes is not None else DEFAULT_MI_SCOPES,
            request_id=request_id or new_request_id(),
            correlation_id=correlation_id,
        )
