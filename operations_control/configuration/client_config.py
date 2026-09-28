"""operations_control.configuration.client_config — the one client-config authority.

WHAT THIS SETTLES
-----------------
After onboarding, a client's standing configuration is the artefact OCC
activated for it::

    operations-control/<client_id>/onboarding/artefacts/current/
        config/client/config_client_<client_id>.yaml

Both consumers must read THAT, and the same one:

* **MI** — reporting currency, asset class, branding, engine conventions;
* **Regime / Annex 2** — the originator identity a regulatory return carries.

Two readers of the same fact that can disagree is a defect waiting for a
deadline, so this module is the single place either asks.

WHY A CLIENT WITH NO ACTIVATED CONFIG GETS NOTHING
--------------------------------------------------
``get_active_client_config`` returns ``None`` — NOT_CONFIGURED — and the caller
fails closed. It never substitutes:

* another client's configuration. The incumbent's file standing in for a client
  that has not been onboarded is how a regulatory return ends up carrying
  someone else's LEI, originator name and establishment country. That is not a
  degraded answer; it is a wrong one that looks right.
* the repository file, in production. ``config/client/config_client_<id>.yaml``
  is where a configuration was authored BEFORE onboarding owned it. Reading it
  as a production fallback means an activated configuration and a stale repo
  file disagree silently, with no version, no attribution and no audit.

An unonboarded client is a VALID platform state — precisely the state
production is in between a client wipe and its fresh onboarding — and the
platform must say "not configured" rather than guess.

THE DEVELOPMENT ESCAPE HATCH, AND WHY IT CANNOT LEAK
-----------------------------------------------------
``TRAKT_CLIENT_CONFIG_DEV_OVERRIDE`` names a file to read instead. It is
honoured ONLY outside production runtime mode (``trakt_core.runtime``), which
in Azure is unconditionally production — so the override is inert in a
deployment even if the variable is set by accident. Development and migration
keep the path they need; production cannot reach it.
"""

from __future__ import annotations

import hashlib
import logging
import os
from dataclasses import dataclass
from typing import Any, Dict, Optional

logger = logging.getLogger("operations_control.configuration.client_config")

#: A file to read INSTEAD of the activated artefact. Development and migration
#: only: ignored whenever the runtime mode is production.
DEV_OVERRIDE_ENV = "TRAKT_CLIENT_CONFIG_DEV_OVERRIDE"

#: Reported as the provenance of a dev override, so a caller that logs the
#: source can never present one as a governed activation.
SOURCE_OCC_ACTIVATED = "occ_activated"
SOURCE_DEV_OVERRIDE = "dev_override"


@dataclass(frozen=True)
class ActiveClientConfig:
    """One client's activated configuration, with where it came from.

    The provenance travels with the document deliberately: §6 of the cutover
    asks MI and regime to prove they resolved the SAME artefact, and they can
    only do that if each can say which one it read.
    """

    client_id: str
    document: Dict[str, Any]
    uri: str
    content_hash: str
    version: Optional[int] = None
    source: str = SOURCE_OCC_ACTIVATED

    @property
    def is_governed(self) -> bool:
        """True only for a genuinely activated OCC configuration."""
        return self.source == SOURCE_OCC_ACTIVATED


def _parse(text: str, where: str) -> Optional[Dict[str, Any]]:
    try:
        import yaml
        doc = yaml.safe_load(text) or {}
    except Exception:  # noqa: BLE001 — a malformed config is not a crash
        logger.warning("client configuration could not be parsed (%s)", where,
                       exc_info=True)
        return None
    return dict(doc) if isinstance(doc, dict) else None


def _dev_override() -> Optional[str]:
    """The override path, when this runtime is allowed to honour one."""
    location = (os.environ.get(DEV_OVERRIDE_ENV) or "").strip()
    if not location:
        return None
    try:
        from trakt_core.runtime import is_production
    except Exception:  # noqa: BLE001 — unknown runtime is treated as production
        return None
    if is_production():
        logger.warning("%s is set but ignored: this runtime is production",
                       DEV_OVERRIDE_ENV)
        return None
    return location


def get_active_client_config(client_id: Optional[str], *, store: Any = None
                             ) -> Optional[ActiveClientConfig]:
    """The activated configuration for ``client_id``, or ``None``.

    ``None`` means NOT CONFIGURED and the caller must fail closed. It never
    means "use someone else's".

    ``store`` is an :class:`~operations_control.stores.OpsStore`; omitted, one
    is built from the environment. Passing it lets a caller that already holds
    one avoid a second client, and lets tests point at a temporary container.
    """
    override = _dev_override()
    if override:
        try:
            with open(override, "r", encoding="utf-8") as handle:
                text = handle.read()
        except Exception:  # noqa: BLE001
            logger.warning("%s names a file that cannot be read: %s",
                           DEV_OVERRIDE_ENV, override, exc_info=True)
            return None
        doc = _parse(text, override)
        if doc is None:
            return None
        return ActiveClientConfig(
            client_id=str(client_id or ""), document=doc, uri=override,
            content_hash=hashlib.sha256(text.encode("utf-8")).hexdigest(),
            version=None, source=SOURCE_DEV_OVERRIDE)

    if not client_id:
        # Resolving without a client and then reading "the" configuration is
        # how one tenant's governed decision reaches another.
        return None

    try:
        from ..onboarding.artefacts import client_config_rel
        from ..onboarding.store import OnboardingStore
        from ..stores import OpsStore
        ops = store if store is not None else OpsStore.from_env()
        onboarding = OnboardingStore(ops)
        rel = client_config_rel(client_id)
        text = onboarding.read_artefact(client_id, rel)
    except Exception:  # noqa: BLE001 — unreachable store is NOT a fallback
        logger.warning("the activated configuration for %s could not be read",
                       client_id, exc_info=True)
        return None
    if not text:
        return None

    doc = _parse(text, f"{client_id} activated artefact")
    if doc is None:
        return None

    version: Optional[int] = None
    try:
        current = onboarding.current(client_id)
        version = int(current.version) if current else None
    except Exception:  # noqa: BLE001 — the document is what matters
        version = None

    return ActiveClientConfig(
        client_id=client_id, document=doc,
        uri=f"blob://{ops.layout.container}/{client_id}/onboarding/"
            f"artefacts/current/{rel}",
        content_hash=hashlib.sha256(text.encode("utf-8")).hexdigest(),
        version=version, source=SOURCE_OCC_ACTIVATED)


def is_configured(client_id: Optional[str], *, store: Any = None) -> bool:
    """Has this client been onboarded and activated?"""
    return get_active_client_config(client_id, store=store) is not None
