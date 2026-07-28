"""mi_agent_api/gateway.py — the API's route surface, decoupled from its host.

Why this module exists
----------------------
The MI Agent API is reachable through more than one front door, and the door it
sits behind is a *deployment* decision that the application had no way to
express:

* **Azure Static Web Apps linked backend** (the topology
  ``mi_agent_api/auth.py`` and ``docs/auth_setup_runbook.md`` describe). The
  browser calls the API **same-origin under ``/api``** so Easy Auth can inject
  ``X-MS-CLIENT-PRINCIPAL``; SWA reverse-proxies the request to the App Service
  **with the path intact**. The API therefore receives ``/api/mi/query``.
* **Direct/absolute base URL** (what the SWA build workflow currently compiles
  into the bundle). The browser calls the App Service host directly and the API
  receives ``/mi/query``.

Every route in this app is registered bare (``/mi/query``), so under the first
topology the App Service answered **404** for every question the chatbot could
ask — the reported production symptom. That is not a missing route; it is a
missing *contract*: nothing stated which path forms the API serves, and nothing
tested it.

This module supplies that contract. :func:`install_gateway_prefix` normalises an
inbound path by stripping the configured gateway prefix **only when the remainder
resolves to a route the application actually serves**, so:

* both path forms work, under either topology and under a future gateway that
  mounts the API somewhere else;
* a genuinely unknown path still 404s with its original path intact, so a real
  routing mistake is not masked;
* nothing about routing, authentication or the analytical path changes — the
  guard in ``auth.py`` and every handler see the normalised path exactly as they
  would have under the direct topology.

Configuration
-------------
``MI_AGENT_API_PREFIX``
    The gateway prefix to accept, default ``/api``. Set to an empty value to
    disable normalisation entirely. Leading/trailing slashes are tolerated.

``MI_AGENT_ALLOWED_ORIGIN``
    A single browser origin (typically the Static Web App) to add to the CORS
    allow-list, for the absolute-base-URL topology where the UI is genuinely
    cross-origin. Additive to ``MI_AGENT_CORS_ORIGINS``; there is still no ``*``.
"""

from __future__ import annotations

import logging
import os
from typing import Iterable, List, Optional, Set

logger = logging.getLogger("mi_agent_api.gateway")

#: Environment variable naming the gateway prefix this deployment sits behind.
API_PREFIX_ENV = "MI_AGENT_API_PREFIX"

#: The default. Azure Static Web Apps forwards linked-backend traffic under
#: ``/api``; accepting it out of the box means the documented topology works
#: without a further app setting.
DEFAULT_API_PREFIX = "/api"

#: Additional browser origin (the deployed front end) for the CORS allow-list.
ALLOWED_ORIGIN_ENV = "MI_AGENT_ALLOWED_ORIGIN"


def _normalise_prefix(raw: Optional[str]) -> str:
    """``"api"`` / ``"/api/"`` / ``"  /api "`` -> ``"/api"``; empty -> ``""``."""
    text = (raw or "").strip()
    if not text:
        return ""
    text = "/" + text.strip("/")
    return "" if text == "/" else text


def api_prefix() -> str:
    """The gateway prefix this deployment accepts, or ``""`` when disabled.

    An unset variable yields the default ``/api`` (fail *open* on reachability,
    which is safe: normalisation never widens what the app serves, it only lets
    the documented deployment topology reach the routes that already exist).
    """
    raw = os.environ.get(API_PREFIX_ENV)
    if raw is None:
        return DEFAULT_API_PREFIX
    return _normalise_prefix(raw)


def extra_cors_origins() -> List[str]:
    """The deployed front-end origin, when one is configured."""
    origin = (os.environ.get(ALLOWED_ORIGIN_ENV) or "").strip().rstrip("/")
    return [origin] if origin else []


def _servable_paths(app) -> Set[str]:
    """Every literal path the application serves.

    Parameterised routes (none today) are deliberately excluded: normalisation
    only ever strips a prefix when the remainder is an *exact* known path, which
    keeps the rule trivially auditable.
    """
    paths: Set[str] = set()
    for route in getattr(app, "routes", []):
        path = getattr(route, "path", None)
        if isinstance(path, str) and path:
            paths.add(path)
    return paths


def strip_prefix(path: str, prefix: str, servable: Iterable[str]) -> str:
    """Return ``path`` with ``prefix`` removed, when that yields a known route.

    Pure and total — this is the whole normalisation rule, isolated so it can be
    tested without an ASGI stack:

    * no prefix configured, or the path does not start with it -> unchanged;
    * stripping yields a path the app serves -> stripped;
    * stripping yields something unknown -> unchanged, so the caller still gets a
      404 naming the path they actually requested.
    """
    if not prefix or not path.startswith(prefix):
        return path
    remainder = path[len(prefix):] or "/"
    if not remainder.startswith("/"):
        return path
    known = set(servable)
    if remainder in known:
        return remainder
    # Tolerate a trailing slash the gateway may add ("/api/mi/query/").
    if remainder.endswith("/") and remainder[:-1] in known:
        return remainder[:-1]
    return path


def install_gateway_prefix(app) -> str:
    """Teach ``app`` to answer under its gateway prefix as well as bare.

    Returns the prefix installed (``""`` when disabled), so the caller can report
    it on ``/health`` — a deployment is then diagnosable with one request instead
    of by reading a build log.
    """
    prefix = api_prefix()
    if not prefix:
        logger.info("gateway prefix normalisation disabled (%s is empty)", API_PREFIX_ENV)
        return ""

    @app.middleware("http")
    async def _normalise_gateway_path(request, call_next):  # noqa: ANN001
        scope = request.scope
        original = scope.get("path", "")
        rewritten = strip_prefix(original, prefix, _servable_paths(app))
        if rewritten != original:
            scope["path"] = rewritten
            # Keep raw_path consistent so anything reading it (logging,
            # downstream middleware) sees the same path the router matched.
            raw = scope.get("raw_path")
            if isinstance(raw, (bytes, bytearray)):
                scope["raw_path"] = rewritten.encode("latin-1")
        return await call_next(request)

    logger.info("gateway prefix normalisation active: %s/* -> /*", prefix)
    return prefix
