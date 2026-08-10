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
from typing import TYPE_CHECKING, Any, Dict, FrozenSet, Iterable, Optional

if TYPE_CHECKING:  # pragma: no cover - typing only, no runtime import
    # Imported lazily to keep the dependency direction one-way: entitlement
    # builds on context, never the reverse.
    from .entitlement import ResolvedEntitlements
    from .resource import ResourceRef

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
#: Producing an artefact, as distinct from reading one. Separate because
#: generation consumes real compute and republishes what external investors
#: receive, so a deployment may grant reading without granting production.
SCOPE_ARTEFACT_GENERATE = "artefact:generate"
#: Current risk / concentration position. A separate scope because a funder may
#: legitimately be shown where a book stands against its limits without being
#: given the free-form MI query surface, and because ``mi:query`` is a much wider
#: grant than "tell me the limit status".
SCOPE_RISK_READ = "risk:read"
#: Forward-looking risk and exposure. Split from :data:`SCOPE_RISK_READ` for the
#: reason the target design calls out explicitly — forward-looking risk is
#: "where permissioned", so it has to be grantable separately from the current
#: position rather than riding along with it.
SCOPE_FORECAST_READ = "forecast:read"

#: The scope set granted to an authenticated MI user or an internal caller. New
#: scopes should be added here only when a capability genuinely gates on them.
#:
#: ``risk:read`` / ``forecast:read`` are included so that today's users keep
#: doing what they already do: the risk, concentration and forecast routes are
#: not scope-gated at all yet, so adding the scopes here is inert now and is what
#: stops those users losing access at the moment those routes start checking.
DEFAULT_MI_SCOPES: FrozenSet[str] = frozenset({
    SCOPE_PORTFOLIO_READ, SCOPE_MI_QUERY, SCOPE_ARTEFACT_READ,
    SCOPE_ARTEFACT_GENERATE, SCOPE_RISK_READ, SCOPE_FORECAST_READ,
})

#: The capability vocabulary, in full. Deliberately the SAME strings as the
#: scopes above rather than a parallel set of entitlement verbs: a grant says
#: "this organisation may do X to this resource" using exactly the name the
#: capability already gates on, so there is nothing to translate and nothing to
#: drift.
#:
#: The two axes are distinct and both apply. ``ExecutionContext.scopes`` is what
#: this caller may do *at all* (channel- and role-level); an entitlement grant is
#: what an organisation may do *to one resource*. The effective permission is the
#: intersection, and ``trakt_core.entitlement.authorise_resource_access`` checks
#: both.
KNOWN_CAPABILITIES: FrozenSet[str] = frozenset({
    SCOPE_PORTFOLIO_READ, SCOPE_MI_QUERY, SCOPE_ARTEFACT_READ,
    SCOPE_ARTEFACT_GENERATE, SCOPE_RISK_READ, SCOPE_FORECAST_READ,
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
    #: WHO IS ASKING — the external organisation this caller belongs to, resolved
    #: from their validated Microsoft directory by
    #: :mod:`trakt_core.organisation`. Deliberately distinct from ``tenant_id``,
    #: which is WHOSE DATA is served and remains deployment configuration.
    #: ``None`` on a deployment with no organisation config (the historical
    #: single-client shape) and on channels that carry no directory identity.
    #: Carries no authority of its own in Stage 1: nothing branches on it, it is
    #: recorded for audit and is the anchor the entitlement model will use.
    organisation_id: Optional[str] = None
    #: The Microsoft Entra directory (tenant) id the caller's token was issued
    #: by, once that token's signature has been verified. Identifies the caller's
    #: directory ONLY; it never selects which Trakt tenant's data is served.
    microsoft_tenant_id: Optional[str] = None
    #: The resolved, frozen entitlement state for this request — which resources
    #: this organisation may reach and with which capabilities. Resolved ONCE, at
    #: context construction, from server-side configuration; nothing downstream
    #: can widen it, and no request field contributes to it.
    #:
    #: ``None`` means **no entitlements were resolved**, which is the state every
    #: caller is in today: either the deployment has no entitlement config, or
    #: the channel carries no organisation identity. ``None`` is NOT "unlimited"
    #: — :func:`trakt_core.entitlement.authorise_resource_access` refuses a
    #: context that has no entitlements. The legacy path is unaffected because it
    #: uses :func:`trakt_core.tenancy.authorise_portfolio_access`, which is
    #: untouched.
    entitlements: Optional["ResolvedEntitlements"] = None

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
        # Blank -> None, so "no organisation" is one value rather than two, and
        # directory ids compare case-insensitively wherever they are read back.
        object.__setattr__(self, "organisation_id",
                           (str(self.organisation_id).strip().lower() or None)
                           if self.organisation_id is not None else None)
        object.__setattr__(self, "microsoft_tenant_id",
                           (str(self.microsoft_tenant_id).strip().lower() or None)
                           if self.microsoft_tenant_id is not None else None)

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

    # -- entitlements ------------------------------------------------------ #
    @property
    def entitlement_mode(self) -> bool:
        """True when entitlements were resolved for this request.

        False is today's state for every caller and means the entitlement model
        is dormant for this request — not that everything is permitted.
        """
        return self.entitlements is not None

    @property
    def permitted_resources(self) -> FrozenSet["ResourceRef"]:
        """The closed set of resources this caller may reach. Empty when no
        entitlements were resolved — never a wildcard."""
        if self.entitlements is None:
            return frozenset()
        return self.entitlements.permitted_resources

    @property
    def entitlement_version(self) -> Optional[str]:
        """Fingerprint of the grants this context was built from.

        Stable across processes for the same grants, so it can go in an audit
        line and — later — in a cache key, where it is what makes a revoked
        grant invalidate a cached answer.
        """
        return self.entitlements.version if self.entitlements is not None else None

    def capabilities_for(self, resource_ref: "ResourceRef") -> FrozenSet[str]:
        """Capabilities granted against one resource. Empty when none are."""
        if self.entitlements is None:
            return frozenset()
        return self.entitlements.capabilities_for(resource_ref)

    # -- derivation -------------------------------------------------------- #
    def with_correlation(self, correlation_id: Optional[str]) -> "ExecutionContext":
        return replace(self, correlation_id=correlation_id or self.correlation_id)

    # -- serialisation ----------------------------------------------------- #
    def to_audit_dict(self) -> Dict[str, Any]:
        """The non-sensitive projection used in audit events and envelopes.

        Deliberately excludes ``actor_label`` (may carry an email address) —
        ``actor_id`` is the stable, minimal identifier.

        ``organisation_id`` and ``microsoft_tenant_id`` are included: without
        them an audit trail on a multi-organisation deployment cannot say which
        party asked, only which tenant's data answered. Both are non-secret
        identifiers — a directory id appears in every token the caller already
        holds — and no token, claim set or bearer material is projected here.
        """
        return {
            "tenant_id": self.tenant_id,
            "organisation_id": self.organisation_id,
            "microsoft_tenant_id": self.microsoft_tenant_id,
            # Which grants answered, not what they were: a fingerprint, so an
            # audit line can be tied back to a configuration without the log
            # becoming a copy of the entitlement table.
            "entitlement_version": self.entitlement_version,
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
