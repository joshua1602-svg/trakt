"""mi_agent_api/react_auth.py — Entra bearer authentication for the React dashboard.

The dashboard authenticates today through Azure Static Web Apps: the platform
performs the sign-in and injects ``X-MS-CLIENT-PRINCIPAL``, which ``auth.py``
trusts. That works only for users the Static Web App can sign in — which, with a
tenant-pinned ``openIdIssuer``, means one directory plus B2B guests invited into
it. A SaaS product cannot ask a customer's staff to become guests in the
supplier's directory, so the dashboard moves to the same shape the Copilot and
agent surfaces already use: the SPA obtains an Entra **access token** for the MI
API and sends it as ``Authorization: Bearer``, and this module validates it.

    external work account -> /organizations -> MSAL (auth code + PKCE)
        -> access token for api://<mi-api-app-id>
        -> THIS MODULE validates it
        -> trakt_core.organisation / .principal decide whether they may in

What is reused rather than rebuilt
----------------------------------
The subtle part of accepting tokens from more than one directory is the
allow-list and the key-selection step, and it must not exist three times. This
module reuses ``copilot_auth.allowed_directories``, ``_select_directory``,
``_jwks_client`` and ``_allowed_issuers`` — exactly as ``agent_auth`` does, and
for the same reason: registering an organisation opens one door, in one place.
``tests/test_react_bearer_auth.py`` pins those names so a rename breaks loudly
instead of silently forking the allow-list.

What is NOT shared: the *policy*. The audience is the MI API's own, and this
surface serves a **signed-in person**, so a delegated token is required:

  * ``scp`` must be present and must carry the required scope. A Copilot scope
    or an agent app role must not admit a caller here, and vice versa;
  * an **app-only** token (``roles`` and no ``scp``) is refused outright. A
    machine identity belongs on ``/v1/agent/*``, which applies its own policy;
  * ``tid`` and ``oid`` must both be present. They are the key the Trakt
    principal registry authorises on, and a token without them cannot be
    authorised at all — so it is refused rather than partially trusted.

Nothing here decides what the caller may *see*. A validated token says which
directory and which individual is asking; ``mi_agent_api.identity`` then resolves
the organisation, the principal binding and the entitlements, and the Trakt
tenant whose data is served remains deployment configuration throughout.

Migration mode
--------------
``TRAKT_MI_REACT_AUTH_MODE`` exists so the cutover is reversible rather than a
flag day, and it is read per request:

  ``swa``     (default) — only the Static Web Apps principal header is accepted.
              A bearer token is ignored. This is the pre-migration shape, and
              deploying this module with the default changes nothing.
  ``both``    — a request carrying ``Authorization: Bearer`` is authenticated
              here; anything else falls through to the header path. This is the
              cutover mode: the new path can be exercised while the existing
              admin login keeps working.
  ``bearer``  — only bearer tokens are accepted. ``X-MS-CLIENT-PRINCIPAL`` stops
              being a credential, which is the point of the final stage: while
              the header is still honoured, anything that can reach the API
              directly can present one.

Configuration
-------------
  ``TRAKT_MI_REACT_AUTH_MODE``       ``swa`` (default) | ``both`` | ``bearer``.
                                     An unrecognised value fails closed (503)
                                     rather than falling back to a mode the
                                     operator did not choose.
  ``TRAKT_MI_ENTRA_AUDIENCE``        Accepted audience(s), comma-separated —
                                     typically ``api://<mi-api-app-id>`` and/or
                                     the bare app id GUID. Required in ``both``
                                     and ``bearer`` modes.
  ``TRAKT_MI_REQUIRED_SCOPE``        The delegated scope the token must carry.
                                     Defaults to ``MI.Access`` rather than empty:
                                     a user surface should require a scope the
                                     SPA was deliberately granted, not admit any
                                     token that happens to name this audience.
                                     Set explicitly to empty to drop the check.

The directories whose tokens are accepted come from
``copilot_auth.allowed_directories()`` — the union of the (historically named)
``TRAKT_COPILOT_ENTRA_TENANT_ID`` app setting and every enabled organisation in
``config/organisations.yaml``. Before that file exists, the allow-list is exactly
the app setting, so enabling ``both`` does not widen who may sign in; once it
exists, registering an organisation is all that is needed.

Token contents are never logged.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from fastapi import HTTPException, Request, status

from trakt_core.organisation import normalise_directory_id
from trakt_core.principal import normalise_object_id

from . import copilot_auth as _copilot_auth

logger = logging.getLogger("mi_agent_api.react_auth")

MODE_ENV = "TRAKT_MI_REACT_AUTH_MODE"
AUDIENCE_ENV = "TRAKT_MI_ENTRA_AUDIENCE"
SCOPE_ENV = "TRAKT_MI_REQUIRED_SCOPE"
#: Directories the DASHBOARD accepts tokens from, comma-separated.
#:
#: Additive to the two sources that already existed. It exists because the only
#: way to name a directory for this surface was ``TRAKT_COPILOT_ENTRA_TENANT_ID``
#: — a setting named after a different product — or a ``config/organisations.yaml``
#: that a deployment may not have yet. An operator turning the dashboard on had
#: to configure Copilot to do it, and if they did not, every request answered 503
#: with the audience set correctly.
#:
#: Deliberately NOT read by ``copilot_auth``: a directory admitted to the
#: dashboard should not silently become admissible on the Copilot surface. The
#: union flows one way.
TENANT_ENV = "TRAKT_MI_ENTRA_TENANT_ID"

#: Only the Static Web Apps principal header authenticates the dashboard.
MODE_SWA = "swa"
#: Either credential is accepted; a bearer token takes precedence when present.
MODE_BOTH = "both"
#: Only a validated bearer token authenticates the dashboard.
MODE_BEARER = "bearer"

KNOWN_MODES = (MODE_SWA, MODE_BOTH, MODE_BEARER)

#: Required unless an operator deliberately clears it. Mirrors the reasoning in
#: ``agent_auth.DEFAULT_REQUIRED_ROLE``: a surface should need a deliberate
#: grant, not merely a token from an accepted directory.
#: The SPA requests it as ``api://<mi-api-app-id>/MI.Access``; only the scope
#: name reaches the token's ``scp`` claim, which is what is compared here.
DEFAULT_REQUIRED_SCOPE = "MI.Access"


@dataclass
class ReactPrincipal:
    """The validated, signed-in human behind a dashboard request."""

    #: Entra object id (``oid``) — the stable per-directory identifier the Trakt
    #: principal registry authorises on. Never an email address.
    subject: Optional[str] = None
    #: ``preferred_username`` / ``name``. Operator-facing only: display, audit
    #: labelling and diagnostics. Never an authorisation input.
    name: Optional[str] = None
    #: The VERIFIED Entra directory (``tid``) the token was issued by.
    directory_id: Optional[str] = None
    scopes: List[str] = field(default_factory=list)
    roles: List[str] = field(default_factory=list)

    def to_public(self) -> Dict[str, Any]:
        """A minimal, non-sensitive view for ``/me`` and logging.

        ``isOperator`` is always False here, and that is a security decision
        rather than an omission. The dashboard's operator features are gated on
        this flag, and the obvious way to set it — an ``operator`` app role in
        the token's ``roles`` claim — does not survive multitenancy: once this
        API is provisioned in a customer's directory, THEIR administrator can
        assign THEIR users any app role the application defines. A privilege an
        external tenant can grant itself is not a privilege. Operator status must
        therefore come from a Trakt-side fact — an ``organisation_type:
        operator`` registration in ``config/organisations.yaml``, resolved onto
        the context in :mod:`mi_agent_api.identity` — and until that is wired to
        this response a bearer-authenticated operator sees the client view.
        Degraded, never elevated.

        The directory and object id are the caller's own identifiers — they are
        in every token they already hold — so echoing them back to that same
        caller reveals nothing, and it is how an administrator obtains the
        ``(tid, oid)`` pair a principal registration is written against without
        going digging in logs.
        """
        return {
            "user": self.name,
            "roles": sorted(self.roles),
            "isOperator": False,
            "scopes": sorted(self.scopes),
            "microsoftTenantId": self.directory_id,
            "microsoftObjectId": self.subject,
        }


def auth_mode() -> str:
    """The configured migration mode, lower-cased. Unrecognised values are
    returned as-is so the caller can fail closed and say what it read."""
    return (os.environ.get(MODE_ENV) or MODE_SWA).strip().lower() or MODE_SWA


def bearer_enabled(mode: Optional[str] = None) -> bool:
    """Whether this deployment accepts bearer tokens on the dashboard path."""
    return (mode if mode is not None else auth_mode()) in (MODE_BOTH, MODE_BEARER)


def bearer_offered(request: Request) -> bool:
    """Whether the request presents an ``Authorization: Bearer`` credential."""
    return (request.headers.get("authorization") or "").lower().startswith("bearer ")


def _audiences() -> List[str]:
    raw = os.environ.get(AUDIENCE_ENV) or ""
    return [a.strip() for a in raw.split(",") if a.strip()]


def _dashboard_directories() -> List[str]:
    """Directories named by this surface's own app setting."""
    out: List[str] = []
    for part in (os.environ.get(TENANT_ENV) or "").split(","):
        directory = normalise_directory_id(part)
        if directory and directory not in out:
            out.append(directory)
    return out


def allowed_directories() -> List[str]:
    """Every Entra directory the DASHBOARD accepts a token from.

    The union of three sources, all additive and all fail-closed when empty:
    this surface's own :data:`TENANT_ENV`, the historical
    ``TRAKT_COPILOT_ENTRA_TENANT_ID``, and every enabled organisation in
    ``config/organisations.yaml``. Sorted so the single-directory fallback in
    ``copilot_auth._select_directory`` stays deterministic.
    """
    combined = set(_dashboard_directories())
    try:
        combined |= set(_copilot_auth.allowed_directories())
    except Exception:  # noqa: BLE001 - a registry fault must not widen the set
        logger.exception("organisation registry unreadable; using app settings only")
    return sorted(combined)


def missing_configuration() -> List[str]:
    """Which halves of the bearer contract are absent, named as app settings.

    Returned rather than merely counted so the 503 can say WHICH one is missing.
    "not configured (a directory, plus an audience, are required)" sends an
    operator to check both when exactly one is empty, and on this deployment the
    directory list was the empty one while the audience was set correctly.
    """
    missing: List[str] = []
    if not allowed_directories():
        missing.append(f"{TENANT_ENV} (or a registered organisation directory)")
    if not _audiences():
        missing.append(AUDIENCE_ENV)
    return missing


def bearer_configured() -> bool:
    """Whether a bearer token could validate at all on this deployment.

    Both halves are required — at least one accepted directory and at least one
    accepted audience — and without them :func:`validate_request` answers 503 to
    every token. Surfaced on ``/health`` so that state is diagnosable without
    attempting a login, because a 503 for "unconfigured" and a 401 for "bad
    token" look identical from the browser.
    """
    try:
        return not missing_configuration()
    except Exception:  # noqa: BLE001 - a probe must never take the route down
        logger.exception("could not determine bearer configuration state")
        return False


def _required_scope() -> str:
    """The delegated scope a token must carry. ``TRAKT_MI_REQUIRED_SCOPE`` set to
    an empty string is an explicit operator decision to drop the requirement; an
    unset variable keeps the default."""
    raw = os.environ.get(SCOPE_ENV)
    return (DEFAULT_REQUIRED_SCOPE if raw is None else raw).strip()


def _unauthorised() -> HTTPException:
    """One indistinguishable 401 for every token failure.

    Never says which check failed. Distinguishing "unknown directory" from "bad
    signature" from "wrong audience" would let a caller enumerate which
    organisations this deployment serves.
    """
    return HTTPException(status_code=status.HTTP_401_UNAUTHORIZED,
                         detail="Invalid or expired token.",
                         headers={"WWW-Authenticate": "Bearer"})


def _unconfigured(missing: Optional[List[str]] = None) -> HTTPException:
    absent = missing if missing is not None else missing_configuration()
    return HTTPException(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        detail=("Dashboard bearer authentication is not configured on this "
                f"deployment. Missing: {', '.join(absent)}."))


def _validate_react_bearer(token: str, directory_id: str,
                           audiences: List[str]) -> ReactPrincipal:
    """Verify signature, audience, expiry and issuer against ONE accepted
    directory, then apply the delegated-user policy.

    ``directory_id`` is always a value ``copilot_auth._select_directory`` chose
    from the allow-list, never a raw caller-supplied one.
    """
    jwt = _copilot_auth._jwt
    if jwt is None:  # pragma: no cover - guarded by the caller
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Dashboard authentication dependency (PyJWT) is not installed.")
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

    # The directory must be stated by the token itself, and must be the one whose
    # keys just verified it. ``_select_directory`` tolerates a missing ``tid`` on
    # a single-directory deployment for the Copilot path's benefit; this surface
    # will grow past one directory by design, and a principal is keyed on
    # ``(tid, oid)``, so an unstated directory is refused here rather than
    # inferred.
    directory = normalise_directory_id(claims.get("tid"))
    if directory is None or directory != normalise_directory_id(directory_id):
        raise _unauthorised()

    # The individual must be identifiable by the stable claim the registry keys
    # on. ``sub`` is deliberately NOT accepted as a fallback: it is pairwise per
    # application, so the same person carries a different ``sub`` for a different
    # client and a registration written against one would silently stop matching.
    subject = normalise_object_id(claims.get("oid"))
    if subject is None:
        raise _unauthorised()

    scopes = [s for s in str(claims.get("scp") or "").split() if s]
    roles = [str(r) for r in (claims.get("roles") or []) if r]

    if not scopes:
        # No delegated scope means this is not a signed-in user: an app-only
        # (client-credentials) token carries ``roles`` and no ``scp``. Machines
        # have their own surface and their own policy; admitting one here would
        # let a service principal act as a dashboard user, with no individual
        # behind it for the principal registry to check.
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=("This endpoint requires a signed-in user. An application-only "
                    "token cannot be used for dashboard access."))

    required = _required_scope()
    if required and required not in scopes:
        # 403, not 401: the token is valid, the user has simply not been granted
        # (or not consented to) this API's scope. That distinction is actionable
        # for their administrator and reveals nothing about other organisations.
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=(f"The token does not carry the required {required} scope for "
                    "the Trakt MI API."))

    return ReactPrincipal(
        subject=subject,
        name=(claims.get("preferred_username") or claims.get("name")
              or claims.get("upn")),
        directory_id=directory,
        scopes=scopes,
        roles=roles,
    )


def validate_request(request: Request) -> ReactPrincipal:
    """Authenticate one dashboard request from its bearer token.

    Raises :class:`fastapi.HTTPException` — 401 for any token that does not
    validate, 403 for a valid token that lacks the delegated scope, 503 when the
    deployment is not configured to accept bearer tokens at all. Never returns
    ``None``: a caller that reaches here has already decided a bearer credential
    was offered.
    """
    absent = missing_configuration()
    if absent:
        raise _unconfigured(absent)
    directories = allowed_directories()
    audiences = _audiences()
    if _copilot_auth._jwt is None or _copilot_auth._PyJWKClient is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Dashboard authentication dependency (PyJWT) is not installed.")

    header = request.headers.get("authorization") or ""
    if not header.lower().startswith("bearer "):
        raise _unauthorised()
    token = header[7:].strip()
    if not token:
        raise _unauthorised()

    directory = _copilot_auth._select_directory(token, directories)
    if directory is None:
        # An unaccepted (or ambiguous) directory is reported exactly like any
        # other invalid token: the response must not confirm which directories
        # this deployment serves.
        raise _unauthorised()

    return _validate_react_bearer(token, directory, audiences)
