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
    authenticated principal carrying an MI role on every ``/mi/*`` route while
    leaving liveness/probe routes open;
  * distinguishes the ``client`` role (the customer's ~5 users) from the
    ``operator`` role (us), so authorization decisions can branch on it.

Environment configuration:

  ``MI_AGENT_AUTH_ENABLED``   "true" (default) enforces auth; "false" bypasses it
                              for local dev / the existing test suite. **Set it
                              explicitly to true on the App Service.**
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

logger = logging.getLogger("mi_agent_api.auth")

PRINCIPAL_HEADER = "x-ms-client-principal"
# App Service Easy Auth also injects these convenience headers; we fall back to
# them when the full principal blob is absent.
_ID_HEADER = "x-ms-client-principal-id"
_NAME_HEADER = "x-ms-client-principal-name"

# Routes that must stay reachable without a principal (health probes, the friendly
# index, and the OpenAPI docs). Everything else requires auth when enabled.
OPEN_PATHS: Set[str] = {"/", "/health", "/openapi.json", "/docs", "/docs/oauth2-redirect", "/redoc"}

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


def _auth_enabled() -> bool:
    return os.environ.get("MI_AGENT_AUTH_ENABLED", "true").strip().lower() not in (
        "false", "0", "no", "off", "",
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
        return self.is_operator or self.is_client

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
    """Global dependency: enforce authentication + MI role on protected routes.

    Open (probe/index/docs) paths pass through. On protected paths a missing
    principal is 401 and a principal without an MI role is 403. The resolved
    principal is stashed on ``request.state.principal`` for handlers/logging.
    """
    path = request.url.path.rstrip("/") or "/"
    if path in OPEN_PATHS or request.method == "OPTIONS":
        return

    if not _auth_enabled():
        request.state.principal = Principal(
            user_details="local-dev", roles={_operator_role()}, synthetic=True)
        return

    principal = principal_from_request(request)
    if principal is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required.",
        )
    if not principal.has_mi_role:
        # Authenticated but no client/operator app-role assigned → fail closed.
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="No MI access role assigned to this account.",
        )
    request.state.principal = principal
