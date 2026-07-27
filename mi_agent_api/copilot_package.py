"""copilot_package.py — Copilot package version telemetry (lightweight).

The sideloaded Copilot package is domain-scoped per client and re-uploaded by
each tenant's IT admin, so tenants can drift onto different package versions.
The shipped package stamps its version onto every request (an
``X-Trakt-Package-Version`` header, defaulted in the OpenAPI to the version that
package was built from); the backend logs it against the version it currently
expects, so operators can spot a tenant running a stale package.

Single source of truth: :data:`COPILOT_PACKAGE_VERSION` is asserted equal to the
Teams manifest ``version`` and the OpenAPI ``info.version`` by
``tests/.../test_copilot_package.py`` — bump all three together on a release
that changes the package.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # avoid a hard FastAPI import at module load
    from fastapi import Request

logger = logging.getLogger("mi_agent_api.copilot")

#: The package version THIS backend build ships / expects. Keep in lock-step with
#: deploy/copilot-agent/manifest.json ("version") and trakt-copilot-openapi.yaml
#: ("info.version").
COPILOT_PACKAGE_VERSION = "2.0.0"

#: Header the shipped package sends on every request (see the OpenAPI spec).
PACKAGE_VERSION_HEADER = "X-Trakt-Package-Version"

#: Logged when a request carries no package version (older package, or a caller
#: that does not forward the header).
_UNSPECIFIED = "unspecified"


def read_package_version(request: "Request") -> str:
    """The package version the calling tenant's package reported, or
    ``"unspecified"`` when the header is absent."""
    return request.headers.get(PACKAGE_VERSION_HEADER) or _UNSPECIFIED


def log_request(request: "Request", operation: str) -> str:
    """Emit one telemetry line identifying the tenant's package version and flag
    drift from the backend's expected version. Returns the reported version."""
    reported = read_package_version(request)
    drift = reported not in (COPILOT_PACKAGE_VERSION, _UNSPECIFIED)
    logger.info(
        "copilot_request op=%s package_version=%s backend_package_version=%s%s",
        operation, reported, COPILOT_PACKAGE_VERSION,
        " STALE_TENANT_PACKAGE" if drift else "")
    return reported
