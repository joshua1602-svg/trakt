"""auth.py — authentication/authorization for the MI Agent API.

The MI Agent UI is exposed to a client via Azure Static Web Apps with the
``trakt-mi-api`` App Service as a **linked backend**. Azure App Service / Static
Web Apps *platform* authentication (Easy Auth, Entra ID) performs the actual
login and forwards the verified principal to this API as the
``X-MS-CLIENT-PRINCIPAL`` header (base64-encoded JSON). This module does **not**
re-validate tokens — it trusts the platform-injected header, which is the
supported pattern for Easy Auth / SWA linked backends. It:

  * parses the injected principal (both the SWA and App Service Easy Auth
    header shapes are supported);
  * exposes a global FastAPI dependency (:func:`auth_guard`) that requires an
    authenticated principal on every ``/mi/*`` route while leaving
    liveness/probe routes open;
  * resolves that verified identity against the **governed access directory**
    (:mod:`trakt_core.access`), which decides tenant, role, portfolio contexts
    and whether the account is live at all.

Where authorisation comes from
------------------------------
It used to come from the roles Static Web Apps attached to an accepted
invitation (``userRoles`` containing ``client``/``operator``). That is no longer
the authority: invitation acceptance proved unreliable in this deployment, and
more fundamentally, who may read a tenant's book is a Trakt governance decision
that belongs in reviewable configuration rather than in a portal click.

The directory is now the sole authority. A principal that still carries a legacy
SWA ``client``/``operator`` role is accepted and parsed exactly as before — the
role simply no longer decides anything. That keeps existing sessions and the
existing header shapes working while the decision moves; it does NOT grant
access to an identity the directory does not list, because the fail-closed
requirement outranks the compatibility one.

Environment configuration:

  ``MI_AGENT_AUTH_ENABLED``   Enforces auth unless EXPLICITLY disabled. Missing
                              or empty -> auth ON (fail closed). Only an explicit
                              "false"/"0"/"no"/"off" bypasses it, for local dev /
                              the existing test suite.
  ``MI_AGENT_CLIENT_ROLE``    Entra app-role name for client users (default "client")
  ``MI_AGENT_OPERATOR_ROLE``  Entra app-role name for operators (default "operator")
  ``MI_AGENT_CLIENT_ID``      The single tenant this deployment serves (already
                              consumed by app.py); used only for logging/labelling
                              here — data isolation is achieved by loading exactly
                              one client's dataset per deployment.

This module is pure/deterministic and performs no I/O beyond reading the request
header and environment.
"""

from __future__ import annotations

import base64
import binascii
import json
import logging
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set

from fastapi import HTTPException, Request, status

from trakt_core import access
from trakt_core.errors import ErrorCode, TraktError

logger = logging.getLogger("mi_agent_api.auth")

PRINCIPAL_HEADER = "x-ms-client-principal"
# App Service Easy Auth also injects these convenience headers; we fall back to
# them when the full principal blob is absent.
_ID_HEADER = "x-ms-client-principal-id"
_NAME_HEADER = "x-ms-client-principal-name"

# Routes that must stay reachable without a principal (health probes, the friendly
# index, and the OpenAPI docs). Everything else requires auth when enabled.
OPEN_PATHS: Set[str] = {"/", "/health", "/openapi.json", "/docs", "/docs/oauth2-redirect", "/redoc"}

# The Microsoft 365 Copilot action routes authenticate with a validated Entra ID
# bearer token (see copilot_auth.py) instead of the platform-injected Easy Auth
# header, so this guard does not apply to them — their own guard fails closed.
COPILOT_PATH_PREFIX = "/v1/copilot"

#: Query parameters that select a governed portfolio context. ``portfolioContext``
#: is the current name and ``lens`` is the older alias several routes still
#: accept (``_resolve_portfolio_context(portfolioContext or lens, ...)``), so
#: both must be gated — enforcing only the new name would leave the alias as an
#: unchecked way to ask for a context the caller is not entitled to.
#:
#: ``test_governed_access.py`` asserts this set covers every context-selecting
#: parameter the route signatures actually declare, so adding a third alias
#: fails a test rather than silently escaping the check.
CONTEXT_QUERY_PARAMS = ("portfolioContext", "lens")

# The Teams bot messaging endpoint is called by the Bot Framework service, not
# by a signed-in user, so no Easy Auth header exists on those requests either.
# It validates the channel-issued token itself (see teams_bot.py) and fails
# closed when it is unconfigured.
TEAMS_BOT_PATH_PREFIX = "/v1/teams"

# Claim types that carry the role in App Service Easy Auth principals.
_ROLE_CLAIM_TYPES = {
    "roles",
    "http://schemas.microsoft.com/ws/2008/06/identity/claims/role",
}
# Claim types that carry a human-readable name / email.
_NAME_CLAIM_TYPES = {
    "name",
    "preferred_username",
    "emails",
    "http://schemas.xmlsoap.org/ws/2005/05/identity/claims/name",
    "http://schemas.xmlsoap.org/ws/2005/05/identity/claims/emailaddress",
}


def _norm_tenant(value: Optional[str]) -> str:
    """Tenant identifiers compare case-insensitively.

    ``MI_AGENT_CLIENT_ID`` is typed by hand into Azure app settings and the
    directory is typed by hand into YAML; ``ERE`` and ``ere`` naming the same
    tenant must not lock everyone out over capitalisation.
    """
    return str(value or "").strip().lower()


def _auth_enabled() -> bool:
    """Auth is ON unless it is EXPLICITLY disabled. Fail closed: a missing OR
    empty ``MI_AGENT_AUTH_ENABLED`` enforces auth, so a blanked/forgotten env
    var can never silently run the API open with a synthetic operator. Only an
    explicit opt-out token ("false"/"0"/"no"/"off") disables enforcement (for
    local dev / the test suite, which set it explicitly)."""
    return os.environ.get("MI_AGENT_AUTH_ENABLED", "true").strip().lower() not in (
        "false", "0", "no", "off",
    )


def _client_role() -> str:
    return os.environ.get("MI_AGENT_CLIENT_ROLE", "client").strip() or "client"


def _operator_role() -> str:
    return os.environ.get("MI_AGENT_OPERATOR_ROLE", "operator").strip() or "operator"


@dataclass
class Principal:
    """A resolved, authenticated caller."""

    user_id: Optional[str] = None
    user_details: Optional[str] = None          # email / display name
    identity_provider: Optional[str] = None
    roles: Set[str] = field(default_factory=set)
    synthetic: bool = False                      # injected when auth is disabled

    @property
    def is_operator(self) -> bool:
        return _operator_role() in self.roles

    @property
    def is_client(self) -> bool:
        return _client_role() in self.roles

    @property
    def has_mi_role(self) -> bool:
        """Whether the platform attached a legacy SWA MI role.

        Retained for compatibility and for the ``/me`` payload. **No longer an
        access decision** — :mod:`trakt_core.access` is the authority. Kept so
        that a deployment mid-migration can still see what the platform sent.
        """
        return self.is_operator or self.is_client

    def directory_identifiers(self) -> tuple:
        """Verified identifiers to look up in the access directory.

        Both are platform-verified: ``user_details`` is the email/UPN Entra
        returned and ``user_id`` is the object id. Neither is caller-supplied
        once the principal header is trustworthy, which is exactly the condition
        :func:`mi_agent_api.identity.require_trustworthy_platform_auth` enforces.
        """
        return (self.user_details, self.user_id)

    def to_public(self) -> Dict[str, Any]:
        """A minimal, non-sensitive view for logging / echoing to the UI."""
        return {
            "user": self.user_details,
            "roles": sorted(self.roles),
            "isOperator": self.is_operator,
        }


def _decode_roles_and_name(data: Dict[str, Any]) -> Principal:
    """Build a Principal from either header shape.

    SWA shape:  {"identityProvider","userId","userDetails","userRoles",[claims]}
    Easy Auth:  {"auth_typ","name_typ","role_typ","claims":[{"typ","val"}]}
    """
    # --- SWA shape (has userRoles) ---
    if "userRoles" in data or "userId" in data or "userDetails" in data:
        roles = {str(r) for r in (data.get("userRoles") or []) if r}
        return Principal(
            user_id=data.get("userId"),
            user_details=data.get("userDetails"),
            identity_provider=data.get("identityProvider"),
            roles=roles,
        )

    # --- App Service Easy Auth shape (claims list) ---
    claims: List[Dict[str, Any]] = data.get("claims") or []
    role_typ = data.get("role_typ")
    name_typ = data.get("name_typ")
    role_types = set(_ROLE_CLAIM_TYPES) | ({role_typ} if role_typ else set())
    name_types = set(_NAME_CLAIM_TYPES) | ({name_typ} if name_typ else set())

    roles: Set[str] = set()
    name: Optional[str] = None
    user_id: Optional[str] = None
    for c in claims:
        typ, val = c.get("typ"), c.get("val")
        if not val:
            continue
        if typ in role_types:
            roles.add(str(val))
        elif typ in name_types and name is None:
            name = str(val)
        elif typ in ("sub", "http://schemas.microsoft.com/identity/claims/objectidentifier") and user_id is None:
            user_id = str(val)
    return Principal(user_id=user_id, user_details=name, roles=roles)


def parse_principal(header_value: Optional[str]) -> Optional[Principal]:
    """Decode the ``X-MS-CLIENT-PRINCIPAL`` header. Returns None if absent/invalid."""
    if not header_value:
        return None
    try:
        raw = base64.b64decode(header_value)
        data = json.loads(raw.decode("utf-8"))
    except (binascii.Error, ValueError, UnicodeDecodeError) as exc:
        logger.warning("could not decode client principal header: %s", exc)
        return None
    if not isinstance(data, dict):
        return None
    return _decode_roles_and_name(data)


def principal_from_request(request: Request) -> Optional[Principal]:
    """Resolve a Principal from the request, or a synthetic operator when auth
    is disabled (local dev / tests)."""
    if not _auth_enabled():
        return Principal(user_details="local-dev", roles={_operator_role()}, synthetic=True)
    p = parse_principal(request.headers.get(PRINCIPAL_HEADER))
    if p is not None:
        return p
    # Fall back to the convenience headers (id/name) if the full blob is absent
    # but the platform still authenticated the caller.
    uid = request.headers.get(_ID_HEADER)
    if uid:
        return Principal(user_id=uid, user_details=request.headers.get(_NAME_HEADER))
    return None


async def auth_guard(request: Request) -> None:
    """Global dependency: authenticate, then authorise against the directory.

    Open (probe/index/docs) paths pass through. On protected paths:

      * no principal                      -> 401 (the platform did not sign
                                             anyone in)
      * principal not in the directory    -> 403 ACCESS_NOT_PROVISIONED
      * principal present but disabled    -> 403 ACCESS_DISABLED

    The resolved principal is stashed on ``request.state.principal`` and the
    resolved grant on ``request.state.access_grant``, so a handler builds its
    :class:`~trakt_core.context.ExecutionContext` from facts this guard already
    verified rather than resolving identity a second time.

    The two 403s are raised as :class:`~trakt_core.errors.TraktError` rather
    than ``HTTPException`` so they carry a machine-readable ``errorCode`` that
    the UI can distinguish from a generic failure — the app's TraktError handler
    maps them to their status.
    """
    path = request.url.path.rstrip("/") or "/"
    if path in OPEN_PATHS or request.method == "OPTIONS":
        return
    if path.startswith(COPILOT_PATH_PREFIX):
        # Copilot actions are guarded by copilot_auth.copilot_auth_guard (Entra
        # bearer-token validation, fail closed). The Easy-Auth header contract
        # does not exist on those calls.
        return
    if path.startswith(TEAMS_BOT_PATH_PREFIX):
        # Bot Framework activities are guarded by
        # teams_bot.validate_activity_token (channel-issued token validation,
        # fail closed). Same reasoning as the Copilot prefix above.
        return

    if not _auth_enabled():
        # Explicit local-development / test bypass. No principal is verified, so
        # there is nothing to look up: the directory is skipped entirely and the
        # deployment-wide tenant applies. This branch is unreachable in Azure —
        # identity.require_trustworthy_platform_auth refuses to serve a
        # production request whose principal header could be forged.
        request.state.principal = Principal(
            user_details="local-dev", roles={_operator_role()}, synthetic=True)
        request.state.access_grant = None
        return

    principal = principal_from_request(request)
    if principal is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required.",
        )
    # Authenticated. What that identity may see is Trakt's decision, not the
    # platform's: resolve it against the governed directory, which raises the
    # governed 403 for an unprovisioned or disabled account.
    grant = access.resolve_grant(*principal.directory_identifiers())
    # This is a deployment-per-tenant product: the dataset layer resolves data
    # from deployment configuration (``dependencies.default_tenant_id``), not
    # from the context. A grant naming a DIFFERENT tenant therefore cannot be
    # served here — honouring it would hand this deployment's book to someone
    # entitled to another one. Refuse rather than silently serve.
    #
    # Imported lazily: ``dependencies`` pulls in the dataset layer, and auth
    # must stay importable without it.
    from .dependencies import default_tenant_id

    deployment_tenant = default_tenant_id()
    if _norm_tenant(grant.tenant_id) != _norm_tenant(deployment_tenant):
        logger.warning(
            "access grant for %s names tenant %r but this deployment serves %r",
            grant.subject, grant.tenant_id, deployment_tenant)
        raise TraktError(
            ErrorCode.TENANT_MISMATCH,
            "This account is provisioned for a different Trakt tenant than the "
            "one this deployment serves.",
            details={"tenant_id": grant.tenant_id},
        )
    # A grant may be narrowed to particular workspace contexts. Check it here,
    # on the raw request, rather than inside the resolver: the resolver is
    # documented never to raise and returns None on failure, which every route
    # treats as "unscoped" — so refusing there would silently WIDEN a denied
    # request to the full book instead of blocking it.
    for param in CONTEXT_QUERY_PARAMS:
        access.authorise_context(grant, request.query_params.get(param))
    request.state.access_grant = grant
    request.state.principal = principal
