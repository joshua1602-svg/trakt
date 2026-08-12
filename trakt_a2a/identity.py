"""trakt_a2a.identity — who is calling, and separately, what they may read.

Two questions, deliberately answered by two different mechanisms:

    authenticate()   WHICH AGENT is this?      -> the A2A security scheme
    authorise()      MAY IT READ THIS BOOK?    -> Trakt entitlements

Collapsing them is the failure this module exists to prevent. A valid A2A token
proves an agent's identity; it says nothing about which portfolios that agent's
organisation is entitled to. A deployment that treated "authenticated" as
"authorised" would hand every verified agent every tenant's book — and it would
look correct in every test that only ever calls it with one tenant.

Where enforcement actually lives
--------------------------------
**Not here.** ``execute_governed_tool`` resolves entitlements on every call and
refuses independently; that is the enforcement, and it would still refuse if
this module were deleted. What :func:`authorise` adds is a *fast, honest*
refusal at the boundary: without it the specialist accepts the task, spends
minutes and real money, and discovers on its first governed call that it was
never entitled. The pre-check exists for the caller's benefit, not for safety,
and describing it as the security boundary would be a lie that a future reader
might act on.

Identity never comes from the request body
------------------------------------------
The objective and the portfolio identifier come from the caller. Identity comes
from the transport's validated principal. ``trakt_tools.mcp`` enforces the same
rule one layer down for exactly the same reason.
"""

from __future__ import annotations

from typing import Any, Optional

from trakt_core.context import ExecutionContext

from .server import (ERR_FORBIDDEN, ERR_UNAUTHENTICATED, A2AError,
                     CallerIdentity)


def caller_from_principal(principal: Any, *, tenant_id: str,
                          organisation_id: str,
                          request_id: Optional[str] = None,
                          correlation_id: Optional[str] = None
                          ) -> CallerIdentity:
    """Build a caller from a **validated** enterprise-agent principal.

    Reuses ``mi_agent_api.identity.context_from_agent_principal`` rather than
    minting a context here, so an A2A caller lands in the same identity model as
    every other machine caller: ``actor_type=service``,
    ``channel=enterprise_agent``, and scopes derived from the organisation's
    grants rather than defaulted. A separate identity path for A2A would be a
    second place for a privilege bug to live.
    """
    from mi_agent_api.identity import context_from_agent_principal

    context = context_from_agent_principal(
        principal, tenant_id=tenant_id, request_id=request_id,
        correlation_id=correlation_id)
    return CallerIdentity(
        agent_id=getattr(principal, "subject", "") or "unknown-agent",
        organisation_id=organisation_id,
        context=context)


def authorise(caller: CallerIdentity, resource: str) -> None:
    """Refuse early if this caller is not entitled to this portfolio.

    Raises :class:`A2AError`; returns ``None`` when the caller may proceed.

    The check is deliberately a *membership test against a closed set*.
    ``ExecutionContext.permitted_resources`` is empty when no entitlements were
    resolved and is never a wildcard, so the failure mode of a
    misconfiguration is refusing everything rather than permitting everything.
    """
    permitted = caller.context.permitted_resources
    if not permitted:
        raise A2AError(
            ERR_FORBIDDEN,
            "The calling organisation holds no portfolio entitlements. "
            "Authentication succeeded; authorisation did not. These are "
            "separate grants.")

    if not any(_matches(ref, resource) for ref in permitted):
        # The message names neither the entitled portfolios nor the reason the
        # requested one exists — a refusal that enumerates what the caller
        # cannot see is an enumeration oracle.
        raise A2AError(
            ERR_FORBIDDEN,
            f"Not entitled to {resource!r}. A valid agent token identifies the "
            "caller; it does not grant portfolio access.")


def _matches(ref: Any, resource: str) -> bool:
    for attribute in ("key", "resource_key", "id"):
        value = getattr(ref, attribute, None)
        if value and str(value) == resource:
            return True
    return str(ref) == resource


def unauthenticated() -> A2AError:
    return A2AError(ERR_UNAUTHENTICATED,
                    "Authentication required. Present an enterprise agent "
                    "token carrying the Trakt.Agent role.")
