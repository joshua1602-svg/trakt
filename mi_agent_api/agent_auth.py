"""mi_agent_api/agent_auth.py — machine identity for the agent tool API.

The Copilot routes authenticate a Microsoft 365 plugin. These routes
authenticate *a machine* — a client's own agent, a synthetic Buyer or Seller
agent, a scheduled analytical job — calling with an Entra **client-credentials**
token. Same directory allow-list, same JWKS verification, different policy.

What is shared with ``copilot_auth``, deliberately
--------------------------------------------------
The subtle part of accepting more than one directory is the allow-list and the
key-selection step, and it must not exist twice. This module reuses
``copilot_auth.allowed_directories``, ``_select_directory``, ``_jwks_client`` and
``_allowed_issuers`` rather than restating them, so registering an organisation
opens exactly one door in both places.
``tests/test_agent_identity.py::test_agent_auth_reuses_the_copilot_directory_allow_list``
pins those names so a rename breaks loudly instead of silently forking the
allow-list.

What is NOT shared
------------------
The *policy*: audience, required app role, and the fact that an agent token is
expected to be app-only. A Copilot delegated-user scope must not admit a machine
to the tool API, and an agent's app role must not admit it to the Copilot
surface.

Client credentials, and what that means for identity
----------------------------------------------------
An app-only token carries ``roles`` and no ``scp``, and its ``oid`` is the
service principal's object id — a machine, not a person. So:

* ``actor_type`` is ``service`` and ``channel`` is ``enterprise_agent``;
* the **organisation** is resolved from the verified ``tid``, exactly as on the
  Copilot path — an agent belongs to the organisation whose directory issued it;
* the **principal registry is not consulted**. It binds *individuals*, and a
  service principal is not one. An agent acting on behalf of a named user is a
  later, separate mechanism (an ``on_behalf_of`` claim), not this one.

Configuration
-------------
  ``TRAKT_AGENT_AUTH_MODE``        ``entra`` (default, fail closed) | ``disabled``
                                   (local development and tests ONLY; refused
                                   outright when the runtime mode is production).
  ``TRAKT_AGENT_ENTRA_AUDIENCE``   Accepted audience(s), comma-separated.
                                   Required in ``entra`` mode.
  ``TRAKT_AGENT_REQUIRED_ROLE``    App role the token must carry, e.g.
                                   ``Trakt.Agent``. Defaults to ``Trakt.Agent``
                                   rather than empty: an agent surface should
                                   require a deliberate app-role assignment, not
                                   admit any valid token from an accepted
                                   directory.
  ``TRAKT_AGENT_DEV_DIRECTORY``    ``disabled`` mode only. The directory the
                                   synthetic principal presents, so even local
                                   development resolves a real organisation and
                                   real entitlements.

Fail-safe posture: in the default ``entra`` mode an unconfigured deployment
returns 503 for every agent route, so the tool API is never anonymously callable
by omission. Token contents are never logged.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from fastapi import HTTPException, Request, status

from trakt_core.organisation import normalise_directory_id
from trakt_core.runtime import is_production

from . import copilot_auth as _copilot_auth

logger = logging.getLogger("mi_agent_api.agent_auth")

_MODE_ENV = "TRAKT_AGENT_AUTH_MODE"
_AUDIENCE_ENV = "TRAKT_AGENT_ENTRA_AUDIENCE"
_ROLE_ENV = "TRAKT_AGENT_REQUIRED_ROLE"
_DEV_DIRECTORY_ENV = "TRAKT_AGENT_DEV_DIRECTORY"

#: Required unless an operator deliberately clears it. An agent surface should
#: need an app-role assignment, not merely a token from an accepted directory.
DEFAULT_REQUIRED_ROLE = "Trakt.Agent"

_warned_disabled = False


@dataclass
class AgentPrincipal:
    """The validated machine caller of an agent tool route."""

    subject: Optional[str] = None        # service principal object id (oid)
    name: Optional[str] = None           # appid / azp — which application
    directory_id: Optional[str] = None   # the VERIFIED Entra directory (tid)
    roles: List[str] = field(default_factory=list)
    scopes: List[str] = field(default_factory=list)
    synthetic: bool = False              # injected when auth mode is disabled

    def to_public(self) -> Dict[str, Any]:
        """Non-sensitive projection for diagnostics. No token material."""
        return {"agent": self.name, "roles": sorted(self.roles),
                "synthetic": self.synthetic}


def _mode() -> str:
    return (os.environ.get(_MODE_ENV) or "entra").strip().lower() or "entra"


def _audiences() -> List[str]:
    raw = os.environ.get(_AUDIENCE_ENV) or ""
    return [a.strip() for a in raw.split(",") if a.strip()]


def _required_role() -> str:
    """The app role a token must carry. ``TRAKT_AGENT_REQUIRED_ROLE`` set to an
    empty string is an explicit operator decision to drop the requirement; an
    unset variable keeps the default."""
    raw = os.environ.get(_ROLE_ENV)
    return (DEFAULT_REQUIRED_ROLE if raw is None else raw).strip()


def _unauthorised() -> HTTPException:
    """One indistinguishable 401 for every token failure.

    Never says which check failed. Distinguishing "unknown directory" from "bad
    signature" from "wrong audience" would let a caller enumerate which
    organisations this deployment serves.
    """
    return HTTPException(status_code=status.HTTP_401_UNAUTHORIZED,
                         detail="Invalid or expired token.",
                         headers={"WWW-Authenticate": "Bearer"})


def _validate_agent_bearer(token: str, directory_id: str,
                           audiences: List[str]) -> AgentPrincipal:
    """Verify signature, audience, expiry and issuer against ONE accepted
    directory, then apply the agent app-role policy.

    ``directory_id`` is always a value ``copilot_auth._select_directory`` chose
    from the allow-list, never a raw caller-supplied one.
    """
    jwt = _copilot_auth._jwt
    if jwt is None:  # pragma: no cover - guarded by the caller
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Agent authentication dependency (PyJWT) is not installed.")
    try:
        signing_key = _copilot_auth._jwks_client(
            directory_id).get_signing_key_from_jwt(token)
        claims = jwt.decode(
            token,
            signing_key.key,
            algorithms=["RS256"],
            audience=audiences,
            options={"require": ["exp", "iss", "aud"]},
        )
    except HTTPException:
        raise
    except Exception:  # noqa: BLE001 - any decode/JWKS failure → 401, no detail
        raise _unauthorised()

    if claims.get("iss") not in _copilot_auth._allowed_issuers(directory_id):
        raise _unauthorised()

    roles = [str(r) for r in (claims.get("roles") or []) if r]
    scopes = [s for s in str(claims.get("scp") or "").split() if s]

    required = _required_role()
    if required and required not in roles:
        # 403, not 401: the token is valid, the application has simply not been
        # granted the agent role. That distinction is actionable for the caller's
        # administrator and reveals nothing about other organisations.
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=("This application is not assigned the Trakt agent app role."))

    return AgentPrincipal(
        subject=str(claims.get("oid") or claims.get("sub") or "") or None,
        name=(claims.get("appid") or claims.get("azp")
              or claims.get("preferred_username") or claims.get("name")),
        directory_id=normalise_directory_id(claims.get("tid") or directory_id),
        roles=roles,
        scopes=scopes,
    )


async def agent_auth_guard(request: Request) -> None:
    """FastAPI dependency guarding ``/v1/agent/*``.

    Stashes the validated principal on ``request.state.agent_principal``.
    """
    global _warned_disabled
    mode = _mode()

    if mode == "disabled":
        if is_production():
            # A development bypass must not be reachable in production, and this
            # refuses rather than silently upgrading to `entra` — an operator who
            # set this variable should see that it did not take effect.
            logger.error("%s=disabled is not permitted in production", _MODE_ENV)
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Agent authentication is misconfigured for this deployment.")
        directory = normalise_directory_id(os.environ.get(_DEV_DIRECTORY_ENV))
        if not directory:
            # Never anonymous, even locally. Without a directory no organisation
            # resolves, so no entitlements resolve, so every tool call would be
            # refused anyway — failing here says why.
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=(f"{_MODE_ENV}=disabled requires {_DEV_DIRECTORY_ENV} to "
                        "name the development organisation's directory."))
        if not _warned_disabled:
            logger.warning(
                "%s=disabled — agent routes are UNAUTHENTICATED and act as "
                "directory %s. Local development and tests only.",
                _MODE_ENV, directory)
            _warned_disabled = True
        request.state.agent_principal = AgentPrincipal(
            subject="local-dev-agent", name="local-dev-agent",
            directory_id=directory, roles=[_required_role() or "Trakt.Agent"],
            synthetic=True)
        return

    if mode != "entra":
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Unknown {_MODE_ENV} value; the agent API is unavailable.")

    directories = _copilot_auth.allowed_directories()
    audiences = _audiences()
    if not directories or not audiences:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=("Agent authentication is not configured (a registered "
                    f"organisation directory, plus {_AUDIENCE_ENV}, are required)."))
    if _copilot_auth._jwt is None or _copilot_auth._PyJWKClient is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Agent authentication dependency (PyJWT) is not installed.")

    header = request.headers.get("authorization") or ""
    if not header.lower().startswith("bearer "):
        raise _unauthorised()
    token = header[7:].strip()
    if not token:
        raise _unauthorised()

    directory = _copilot_auth._select_directory(token, directories)
    if directory is None:
        raise _unauthorised()

    request.state.agent_principal = _validate_agent_bearer(
        token, directory, audiences)
