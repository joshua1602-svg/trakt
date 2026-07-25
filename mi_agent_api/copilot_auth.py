"""copilot_auth.py — bearer-token (Microsoft Entra ID) authentication for the
Microsoft 365 Copilot action routes (``/v1/copilot/*``).

Why a second auth module: the existing ``auth.py`` guard trusts the
``X-MS-CLIENT-PRINCIPAL`` header injected by Azure Easy Auth / Static Web Apps —
the supported pattern for the linked-backend dashboard, but NOT usable for a
Copilot API plugin, which calls the API directly with an OAuth **bearer token**
issued by Entra ID. The Copilot routes therefore validate the token themselves
(signature via the tenant JWKS, issuer, audience, expiry) instead of trusting
any client-supplied header.

Configuration (all read per-request; nothing cached from the environment):

  ``TRAKT_COPILOT_AUTH_MODE``        ``entra`` (default, fail closed) | ``disabled``
                                     (local dev / tests ONLY — logs a warning).
  ``TRAKT_COPILOT_ENTRA_TENANT_ID``  Entra directory (tenant) GUID. Required in
                                     ``entra`` mode; missing → 503 (fail closed).
  ``TRAKT_COPILOT_ENTRA_AUDIENCE``   Accepted audience(s), comma-separated —
                                     typically ``api://<app-id>`` and/or the app
                                     (client) id GUID. Required in ``entra`` mode.
  ``TRAKT_COPILOT_REQUIRED_SCOPE``   Optional scope (``scp``) or app-role
                                     (``roles``) name that the token must carry,
                                     e.g. ``Trakt.Copilot``. Empty → any valid
                                     token from the tenant/audience is accepted.

Fail-safe posture: in the default ``entra`` mode an unconfigured deployment
returns 503 for every Copilot route — the routes are never anonymously callable
in production by omission. Token contents are never logged.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from fastapi import HTTPException, Request, status

logger = logging.getLogger("mi_agent_api.copilot_auth")

try:  # PyJWT is a deploy-time dependency (requirements.txt); tolerate absence
    import jwt as _jwt  # noqa: N816
    from jwt import PyJWKClient as _PyJWKClient
except Exception:  # noqa: BLE001 - missing dependency → entra mode fails closed
    _jwt = None
    _PyJWKClient = None

_MODE_ENV = "TRAKT_COPILOT_AUTH_MODE"
_TENANT_ENV = "TRAKT_COPILOT_ENTRA_TENANT_ID"
_AUDIENCE_ENV = "TRAKT_COPILOT_ENTRA_AUDIENCE"
_SCOPE_ENV = "TRAKT_COPILOT_REQUIRED_SCOPE"

#: JWKS clients keyed by tenant id — PyJWKClient caches the signing keys.
_JWKS_CLIENTS: Dict[str, Any] = {}

_warned_disabled = False


@dataclass
class CopilotPrincipal:
    """The validated caller of a Copilot action."""

    subject: Optional[str] = None          # oid / sub
    name: Optional[str] = None             # preferred_username / name / appid
    tenant_id: Optional[str] = None
    scopes: List[str] = field(default_factory=list)
    roles: List[str] = field(default_factory=list)
    synthetic: bool = False                # injected when auth mode is disabled

    def to_public(self) -> Dict[str, Any]:
        return {"user": self.name, "roles": sorted(self.roles)}


def _mode() -> str:
    return (os.environ.get(_MODE_ENV) or "entra").strip().lower() or "entra"


def _audiences() -> List[str]:
    raw = os.environ.get(_AUDIENCE_ENV) or ""
    return [a.strip() for a in raw.split(",") if a.strip()]


def _jwks_client(tenant_id: str):
    client = _JWKS_CLIENTS.get(tenant_id)
    if client is None:
        client = _PyJWKClient(
            f"https://login.microsoftonline.com/{tenant_id}/discovery/v2.0/keys")
        _JWKS_CLIENTS[tenant_id] = client
    return client


def _allowed_issuers(tenant_id: str) -> List[str]:
    # v2.0 tokens and (legacy) v1.0 tokens from the same tenant.
    return [
        f"https://login.microsoftonline.com/{tenant_id}/v2.0",
        f"https://sts.windows.net/{tenant_id}/",
    ]


def _validate_bearer(token: str, tenant_id: str, audiences: List[str]) -> CopilotPrincipal:
    """Validate signature/audience/expiry/issuer; raise HTTPException(401/403)."""
    try:
        signing_key = _jwks_client(tenant_id).get_signing_key_from_jwt(token)
        claims = _jwt.decode(
            token,
            signing_key.key,
            algorithms=["RS256"],
            audience=audiences,
            options={"require": ["exp", "iss", "aud"]},
        )
    except HTTPException:
        raise
    except Exception:  # noqa: BLE001 - any decode/JWKS failure → 401, no detail leak
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token.")

    if claims.get("iss") not in _allowed_issuers(tenant_id):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token.")

    scopes = [s for s in str(claims.get("scp") or "").split() if s]
    roles = [str(r) for r in (claims.get("roles") or []) if r]

    required = (os.environ.get(_SCOPE_ENV) or "").strip()
    if required and required not in scopes and required not in roles:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="The token does not carry the required Trakt Copilot scope or role.")

    return CopilotPrincipal(
        subject=str(claims.get("oid") or claims.get("sub") or "") or None,
        name=(claims.get("preferred_username") or claims.get("name")
              or claims.get("appid") or claims.get("azp")),
        tenant_id=str(claims.get("tid") or tenant_id),
        scopes=scopes,
        roles=roles,
    )


async def copilot_auth_guard(request: Request) -> None:
    """FastAPI dependency guarding the Copilot action routes.

    ``entra`` mode (default): requires a valid Entra bearer token; unconfigured
    deployments fail closed with 503. ``disabled`` mode: local dev/tests only.
    The validated principal is stashed on ``request.state.copilot_principal``.
    """
    global _warned_disabled
    mode = _mode()

    if mode == "disabled":
        if not _warned_disabled:
            logger.warning(
                "TRAKT_COPILOT_AUTH_MODE=disabled — Copilot routes are UNAUTHENTICATED. "
                "This mode is for local development and tests only.")
            _warned_disabled = True
        request.state.copilot_principal = CopilotPrincipal(
            name="local-dev", synthetic=True)
        return

    if mode != "entra":
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Unknown {_MODE_ENV} value; Copilot actions are unavailable.")

    tenant_id = (os.environ.get(_TENANT_ENV) or "").strip()
    audiences = _audiences()
    if not tenant_id or not audiences:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=("Copilot authentication is not configured "
                    f"({_TENANT_ENV} / {_AUDIENCE_ENV} required)."))
    if _jwt is None or _PyJWKClient is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Copilot authentication dependency (PyJWT) is not installed.")

    header = request.headers.get("authorization") or ""
    if not header.lower().startswith("bearer "):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="A bearer token is required.",
            headers={"WWW-Authenticate": "Bearer"})
    token = header[7:].strip()
    if not token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="A bearer token is required.",
            headers={"WWW-Authenticate": "Bearer"})

    request.state.copilot_principal = _validate_bearer(token, tenant_id, audiences)
