"""copilot_auth.py — bearer-token (Microsoft Entra ID) authentication for the
Microsoft 365 Copilot action routes (``/v1/copilot/*``).

Why a second auth module: the existing ``auth.py`` guard trusts the
``X-MS-CLIENT-PRINCIPAL`` header injected by Azure Easy Auth / Static Web Apps —
the supported pattern for the linked-backend dashboard, but NOT usable for a
Copilot API plugin, which calls the API directly with an OAuth **bearer token**
issued by Entra ID. The Copilot routes therefore validate the token themselves
(signature via the tenant JWKS, issuer, audience, expiry) instead of trusting
any client-supplied header.

Accepting more than one directory
---------------------------------
This module used to accept exactly one Entra directory GUID. It now validates
against a **set** of accepted directories, drawn from two additive sources:

  * ``TRAKT_COPILOT_ENTRA_TENANT_ID`` — now a comma-separated list. A single
    value behaves exactly as it did before, so the existing ERE deployment needs
    no configuration change.
  * the organisation registry (``config/organisations.yaml``, see
    :mod:`trakt_core.organisation`) — every enabled organisation's directories,
    so registering an organisation is enough and its GUID does not have to be
    duplicated into an app setting.

Which directory's keys a token is verified against is chosen by reading the
``tid`` claim **without verifying the signature**, and that is safe for one
specific reason: the claim only *selects a key set from an allow-list*, and
nothing in the token is trusted until the signature verifies against that
directory's published JWKS. A token naming a directory that is not on the
allow-list is refused before any key is fetched, and a token naming a directory
whose signing keys the caller does not hold cannot verify. A token with no
``tid`` is resolvable only when the deployment accepts exactly one directory —
the historical single-tenant shape — and is otherwise refused as ambiguous.

**A validated ``tid`` says which directory is asking. It never says which Trakt
tenant's data is served** — that stays deployment configuration, resolved in
``mi_agent_api.identity`` and ``mi_agent_api.dependencies``. This module returns
the directory on the principal; it does not choose a tenant.

Configuration (all read per-request; nothing cached from the environment):

  ``TRAKT_COPILOT_AUTH_MODE``        ``entra`` (default, fail closed) | ``disabled``
                                     (local dev / tests ONLY — logs a warning).
  ``TRAKT_COPILOT_ENTRA_TENANT_ID``  Entra directory (tenant) GUID, or a
                                     comma-separated list of them. Optional when
                                     an organisation registry supplies at least
                                     one directory; with neither source
                                     populated, ``entra`` mode → 503.
  ``TRAKT_COPILOT_ENTRA_AUDIENCE``   Accepted audience(s), comma-separated —
                                     typically ``api://<app-id>`` and/or the app
                                     (client) id GUID. Required in ``entra`` mode.
  ``TRAKT_COPILOT_REQUIRED_SCOPE``   Optional scope (``scp``) or app-role
                                     (``roles``) name that the token must carry,
                                     e.g. ``Trakt.Copilot``. Empty → any valid
                                     token from an accepted directory/audience
                                     is accepted.

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

from trakt_core.organisation import load_organisation_registry, normalise_directory_id

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


def _configured_directories() -> List[str]:
    """Directories named in the app setting. One value is the historical shape."""
    raw = os.environ.get(_TENANT_ENV) or ""
    out: List[str] = []
    for part in raw.split(","):
        directory = normalise_directory_id(part)
        if directory and directory not in out:
            out.append(directory)
    return out


def allowed_directories(registry: Optional[Any] = None) -> List[str]:
    """Every Entra directory this deployment will accept a token from.

    The union of the app setting and the enabled organisations, sorted so the
    "exactly one directory" fallback below is deterministic. Empty means the
    deployment is unconfigured, which ``entra`` mode turns into a 503.

    A directory present only in the app setting authenticates here and is then
    refused during organisation resolution when the registry is configured
    (``trakt_core.organisation`` fails closed on an unregistered directory).
    Two independent gates, both closed by default.
    """
    directories = set(_configured_directories())
    try:
        reg = registry if registry is not None else load_organisation_registry()
        directories |= set(reg.microsoft_tenant_ids(enabled_only=True))
    except Exception:  # noqa: BLE001 - registry faults must not widen the set
        logger.exception("organisation registry unreadable; using the app setting only")
    return sorted(directories)


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


def _select_directory(token: str, allowed: List[str]) -> Optional[str]:
    """Which accepted directory's signing keys this token should be verified against.

    Reads ``tid`` from the **unverified** payload. That is a key-selection step,
    not a trust decision: the returned directory is only ever one already on
    ``allowed``, and the caller immediately verifies the signature against that
    directory's published keys, so a forged ``tid`` selects a key set the forger
    cannot sign for. Returns ``None`` when the token names no accepted directory,
    which the caller turns into a 401.
    """
    try:
        unverified = _jwt.decode(
            token, options={"verify_signature": False, "verify_aud": False,
                            "verify_exp": False, "verify_iss": False})
    except Exception:  # noqa: BLE001 - an undecodable token is simply invalid
        return None

    directory = normalise_directory_id(
        (unverified or {}).get("tid") if isinstance(unverified, dict) else None)
    if directory:
        return directory if directory in allowed else None
    # No tid claim. Resolvable only against a single-directory deployment — the
    # shape this module had before it accepted a set. With several accepted
    # directories the caller's organisation is genuinely ambiguous, and guessing
    # is exactly the behaviour that must not exist on a multi-tenant path.
    return allowed[0] if len(allowed) == 1 else None


def _validate_bearer(token: str, tenant_id: str, audiences: List[str]) -> CopilotPrincipal:
    """Validate signature/audience/expiry/issuer against ONE accepted directory.

    ``tenant_id`` is the directory chosen by :func:`_select_directory` from the
    accepted allow-list — never a raw caller-supplied value.
    """
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

    In ``disabled`` mode the synthetic principal carries no directory, so a
    deployment that has *also* registered organisations will refuse it during
    organisation resolution. That combination is not supported and the refusal
    is the correct outcome: an unauthenticated local-development principal has
    no organisation, and inventing one would be the permissive fallback that
    organisation mode exists to remove.
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

    directories = allowed_directories()
    audiences = _audiences()
    if not directories or not audiences:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=("Copilot authentication is not configured "
                    f"({_TENANT_ENV} or a registered organisation, plus "
                    f"{_AUDIENCE_ENV}, are required)."))
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

    directory = _select_directory(token, directories)
    if directory is None:
        # An unaccepted (or ambiguous) directory is reported exactly like any
        # other invalid token: the response must not confirm which directories
        # this deployment serves.
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token.")

    request.state.copilot_principal = _validate_bearer(token, directory, audiences)
