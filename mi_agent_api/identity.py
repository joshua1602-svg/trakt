"""mi_agent_api/identity.py — authenticated identity → :class:`ExecutionContext`.

The one place a verified platform identity becomes a trusted context. Domain and
application code never see a header, a token or a FastAPI ``Request``; they see
an :class:`~trakt_core.context.ExecutionContext`.

Two identity mechanisms exist today and both land here:

  * **React** — Azure Static Web Apps / App Service Easy Auth performs the login
    and injects ``X-MS-CLIENT-PRINCIPAL``. ``auth.parse_principal`` decodes it.
  * **Copilot** — a Microsoft Entra bearer token validated against the directory
    JWKS by ``copilot_auth``.

Two identities, not one
-----------------------
A governed context now records two separate facts about a request, and keeping
them separate is the whole point:

  * ``tenant_id`` — **whose data is served.** Deployment configuration, exactly
    as before. Never a token claim, never a request field.
  * ``organisation_id`` / ``microsoft_tenant_id`` — **who is asking.** Resolved
    from the caller's signature-verified Entra directory through
    :mod:`trakt_core.organisation`.

A funder signing in from its own directory is ``organisation_id="funder_a"``
asking about ``tenant_id="ERE"``. Stage 1 establishes that identity and records
it; it grants nothing and narrows nothing. Which resources an organisation may
reach is the entitlement stage, and until it lands an organisation is served
exactly what the deployment's tenant configuration already allowed.

With no ``config/organisations.yaml`` the deployment is in compatibility mode:
``resolve_organisation`` returns ``None``, the context's organisation fields stay
unset, and every existing caller is unchanged.

Trusting the injected header
----------------------------
The header is only trustworthy if the platform in front of this process
*overwrites* it. Azure Easy Auth does; a directly reachable App Service with
Easy Auth switched off does not, and then any caller can present a header naming
any role. :func:`platform_auth_status` makes that condition explicit and
:func:`require_trustworthy_platform_auth` fails closed on it in production.

What is enforceable here and what is not:

  * enforceable in code — refuse to trust the header when running in Azure with
    no platform authentication enabled and no explicit operator acknowledgement;
  * **not** enforceable in code — whether the App Service actually restricts
    inbound traffic. See ``docs/governed_capability_architecture.md`` §
    "Deployment requirements".
"""

from __future__ import annotations

import logging
import os
from typing import Any, Iterable, Optional

from trakt_core.context import (
    ACTOR_SERVICE,
    ACTOR_USER,
    CHANNEL_COPILOT,
    CHANNEL_REACT,
    DEFAULT_MI_SCOPES,
    ExecutionContext,
    new_request_id,
)
from trakt_core.errors import ErrorCode, TraktError
from trakt_core.organisation import (
    OrganisationRecord,
    OrganisationRegistry,
    load_organisation_registry,
    normalise_directory_id,
)
from trakt_core.runtime import is_production, running_in_azure

logger = logging.getLogger("mi_agent_api.identity")

#: App setting Azure sets when App Service / SWA platform authentication is on.
AZURE_AUTH_ENABLED_ENV = "WEBSITE_AUTH_ENABLED"

#: Explicit operator acknowledgement that something else in front of this process
#: (a gateway, an access restriction, a service-mesh policy) strips and re-injects
#: the principal header. Only consulted in Azure; never a way to disable auth.
TRUST_PLATFORM_AUTH_ENV = "MI_AGENT_TRUST_PLATFORM_AUTH"

_TRUE = ("1", "true", "yes", "on")


def _flag(name: str) -> bool:
    return (os.environ.get(name) or "").strip().lower() in _TRUE


def platform_auth_status() -> dict:
    """Whether the injected principal header can be trusted, and why.

    Returned as data (rather than just raising) so ``/health`` can surface the
    posture and a deployment check can assert on it.
    """
    in_azure = running_in_azure()
    easy_auth = _flag(AZURE_AUTH_ENABLED_ENV)
    acknowledged = _flag(TRUST_PLATFORM_AUTH_ENV)
    if not in_azure:
        return {"trustworthy": True, "reason": "not running in Azure (local/dev host)",
                "in_azure": False, "platform_auth_enabled": easy_auth}
    if easy_auth:
        return {"trustworthy": True,
                "reason": f"{AZURE_AUTH_ENABLED_ENV} is set: the platform overwrites "
                          "the client-principal header",
                "in_azure": True, "platform_auth_enabled": True}
    if acknowledged:
        return {"trustworthy": True,
                "reason": f"{TRUST_PLATFORM_AUTH_ENV} set: an upstream gateway is "
                          "declared to control the header",
                "in_azure": True, "platform_auth_enabled": False}
    return {
        "trustworthy": False,
        "reason": (f"running in Azure with {AZURE_AUTH_ENABLED_ENV} unset — the "
                   "client-principal header is caller-supplied and must not be "
                   "trusted"),
        "in_azure": True, "platform_auth_enabled": False,
    }


def require_trustworthy_platform_auth() -> None:
    """Fail closed when the principal header cannot be trusted in production.

    A header-authenticated route on a directly reachable App Service with Easy
    Auth off is an authentication bypass, not a degraded mode — so this refuses
    the request rather than serving it.
    """
    if not is_production():
        return
    status = platform_auth_status()
    if status["trustworthy"]:
        return
    logger.error("PLATFORM AUTH NOT TRUSTWORTHY: %s", status["reason"])
    raise TraktError(
        ErrorCode.AUTHENTICATION_NOT_CONFIGURED,
        "Platform authentication is not configured for this deployment, so the "
        "caller's identity cannot be established. Trakt will not answer until it "
        "is.",
    )


def _scopes_for(roles: Iterable[str]) -> frozenset:
    """Scopes for a set of app roles.

    Both MI roles (``client`` and ``operator``) currently carry the same
    read-only MI scopes; the split exists so an operator-only capability can be
    gated later without changing every call site.
    """
    return DEFAULT_MI_SCOPES


def context_from_principal(
    principal: Any,
    *,
    tenant_id: str,
    channel: str = CHANNEL_REACT,
    request_id: Optional[str] = None,
    correlation_id: Optional[str] = None,
) -> ExecutionContext:
    """Build a context from an Easy Auth / SWA principal (``auth.Principal``).

    ``tenant_id`` is deployment configuration, never a principal claim: this is a
    deployment-per-tenant product, and taking the tenant from a token claim would
    make it caller-influenced.

    **No organisation is resolved on this path, deliberately.** Easy Auth injects
    a principal that this process trusts because the platform overwrites the
    header — it does not hand us a signature-verified directory id, and the SWA
    principal shape carries no ``tid`` at all. Resolving an organisation from an
    unverified value would be exactly the mistake ``copilot_auth`` exists to
    avoid, and applying organisation-mode refusal to a channel that cannot
    produce a verified directory would break the signed-in React user for no
    security gain. The React context therefore leaves the organisation fields
    unset. Giving this channel a first-class organisation identity means putting
    a validated token in front of it, which is its own change.
    """
    require_trustworthy_platform_auth()
    if principal is None:
        raise TraktError(ErrorCode.AUTHENTICATION_REQUIRED,
                         "Authentication is required.")
    actor_id = (getattr(principal, "user_id", None)
                or getattr(principal, "user_details", None)
                or "unknown-principal")
    return ExecutionContext(
        tenant_id=tenant_id,
        actor_id=str(actor_id),
        actor_type=ACTOR_USER,
        channel=channel,
        scopes=_scopes_for(getattr(principal, "roles", ()) or ()),
        request_id=request_id or new_request_id(),
        correlation_id=correlation_id,
        actor_label=getattr(principal, "user_details", None),
    )


def resolve_organisation(
    microsoft_tenant_id: Optional[str],
    *,
    registry: Optional[OrganisationRegistry] = None,
    request_id: Optional[str] = None,
) -> Optional[OrganisationRecord]:
    """The organisation behind a **signature-verified** Microsoft directory id.

    ``None`` means compatibility mode — no organisation config is loaded, so this
    deployment makes no claim about who is asking and behaves exactly as it did
    before organisations existed.

    In organisation mode an unregistered or disabled directory raises
    :class:`~trakt_core.errors.TraktError`. There is no permissive branch: a
    deployment that has declared which organisations it serves must not answer
    one it has not.

    Callers must pass a directory id that came out of token *validation*
    (``copilot_auth``). This function cannot distinguish a verified claim from an
    asserted one and does not try to.
    """
    reg = registry if registry is not None else load_organisation_registry()
    return reg.resolve_microsoft_tenant(microsoft_tenant_id, request_id=request_id)


def context_from_copilot_principal(
    principal: Any,
    *,
    tenant_id: str,
    request_id: Optional[str] = None,
    correlation_id: Optional[str] = None,
    organisation_registry: Optional[OrganisationRegistry] = None,
) -> ExecutionContext:
    """Build a context from a validated Entra bearer principal.

    Not subject to :func:`require_trustworthy_platform_auth`: the Copilot routes
    validate the token signature, issuer, audience and expiry themselves
    (``copilot_auth``), so their trust does not depend on an upstream gateway.

    ``tenant_id`` — whose data is served — is the caller-supplied deployment
    configuration and is used verbatim, unchanged by this stage. The principal's
    directory (``tid``) resolves the *organisation* and nothing else: it is
    recorded on the context, and it never reaches ``tenant_id``.
    """
    if principal is None:
        raise TraktError(ErrorCode.AUTHENTICATION_REQUIRED,
                         "A validated bearer token is required.")
    actor_id = (getattr(principal, "subject", None)
                or getattr(principal, "name", None)
                or "unknown-principal")
    # A Copilot call may be delegated (a signed-in user) or app-only. ``scp``
    # implies a user; app-only tokens carry roles instead.
    actor_type = ACTOR_USER if getattr(principal, "scopes", None) else ACTOR_SERVICE

    # WHO IS ASKING. Resolved before the context is built so an unregistered
    # organisation is refused rather than producing a usable context — and so
    # nothing downstream has to remember to check.
    microsoft_tenant_id = normalise_directory_id(getattr(principal, "tenant_id", None))
    organisation = resolve_organisation(
        microsoft_tenant_id, registry=organisation_registry, request_id=request_id)

    return ExecutionContext(
        tenant_id=tenant_id,
        actor_id=str(actor_id),
        actor_type=actor_type,
        channel=CHANNEL_COPILOT,
        scopes=DEFAULT_MI_SCOPES,
        request_id=request_id or new_request_id(),
        correlation_id=correlation_id,
        actor_label=getattr(principal, "name", None),
        organisation_id=organisation.organisation_id if organisation else None,
        microsoft_tenant_id=microsoft_tenant_id,
    )
