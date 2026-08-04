#!/usr/bin/env python3
"""Delivery configuration for the Teams notification capability.

The distinction this module exists to hold: **this configuration controls
delivery, not analytical truth.** Nothing here is a materiality threshold, a
concentration limit or a risk rule. Those live where they already live —
``config/mi/insights.yaml`` via :mod:`mi_agent_api.insight_config`, the governed
concentration-test configuration, and the risk rules. Duplicating any of them
here would create a second, quietly divergent definition of "material", which is
the specific failure this separation prevents.

What IS configurable here: whether to send at all, to whom, in which scope,
whether each of the two messages is enabled, how many risk items a card may
carry, and how far the action language may go.

Loaded through the same thin-YAML convention as ``insight_config`` — one file,
documented defaults, partial overrides safe, a configuration problem degrading
to the defaults rather than to a failed send.
"""

from __future__ import annotations

import copy
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger("trakt_notifications.config")

_REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PATH = _REPO_ROOT / "config" / "mi" / "teams_notifications.yaml"

#: Override for tests and for a deployment shipping its own delivery settings.
PATH_ENV = "TRAKT_TEAMS_NOTIFICATIONS_CONFIG"

#: Master kill switch, independent of the config file, so an operator can stop
#: delivery without a redeploy. Only an explicit opt-in token enables it.
ENABLED_ENV = "TRAKT_TEAMS_NOTIFICATIONS"
_ON = ("1", "true", "on", "yes", "enabled")

# --------------------------------------------------------------------------- #
# Recommendation levels — the action-language policy
# --------------------------------------------------------------------------- #
LEVEL_OBSERVATION = "observation"
LEVEL_CONSIDERATION = "consideration"
LEVEL_APPROVED_ACTION = "approved_action"

#: Ordering, low to high. A message may never exceed the configured level.
LEVEL_RANK = {LEVEL_OBSERVATION: 0, LEVEL_CONSIDERATION: 1,
              LEVEL_APPROVED_ACTION: 2}

#: Delivery scopes. Only ``personal`` is implemented in v1; the others are
#: named so that a configuration asking for them fails loudly rather than
#: silently delivering to a personal chat instead.
SCOPE_PERSONAL = "personal"
KNOWN_SCOPES = (SCOPE_PERSONAL,)

DEFAULTS: Dict[str, Any] = {
    "enabled": False,
    "delivery_scope": SCOPE_PERSONAL,
    "update_message_enabled": True,
    "risk_message_enabled": True,
    # Level 2 is reachable ONLY where an explicit client action rule defines
    # the wording. Defaulting here to level 2 would let a deployment escalate
    # by configuration alone, which is precisely what must not be possible.
    "recommendation_level": LEVEL_CONSIDERATION,
    "maximum_risk_items": 3,
    "maximum_update_items": 5,
    # Named pilot recipients. An empty list means nothing is delivered — v1
    # has no self-service subscription and no directory synchronisation, so an
    # unconfigured deployment sends to nobody rather than to everybody.
    "recipients": [],
    # Delivery mechanics.
    "max_attempts": 5,
    "retry_backoff_seconds": [30, 120, 600, 1800],
}


def _yaml_load(path: Path) -> Dict[str, Any]:
    try:
        import yaml  # deferred: config is optional, the dependency is not free
    except Exception:  # noqa: BLE001 - a missing parser degrades to defaults
        logger.warning("teams notifications: PyYAML unavailable, using defaults")
        return {}
    try:
        with path.open("r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh) or {}
    except FileNotFoundError:
        return {}
    except Exception as exc:  # noqa: BLE001
        logger.warning("teams notifications: %s unreadable (%s), using defaults",
                       path, exc)
        return {}
    if not isinstance(data, dict):
        return {}
    # Accept both a bare mapping and one nested under the documented key.
    return data.get("teams_notifications") or data


def config_path() -> Path:
    override = (os.environ.get(PATH_ENV) or "").strip()
    return Path(override) if override else DEFAULT_PATH


def load() -> Dict[str, Any]:
    """The resolved delivery settings. Read per call so an operator toggle
    takes effect without a process restart."""
    conf = copy.deepcopy(DEFAULTS)
    path = config_path()
    overrides = _yaml_load(path) if path.exists() else {}
    for key, value in overrides.items():
        if key in conf:
            conf[key] = value
        else:
            logger.warning("teams notifications: ignoring unknown setting %r", key)
    conf["source"] = str(path) if path.exists() else "defaults"

    # The environment kill switch can only ever DISABLE relative to the file,
    # or enable a deployment whose file says nothing. It is checked explicitly
    # so "off" is reachable without editing configuration under pressure.
    env = (os.environ.get(ENABLED_ENV) or "").strip().lower()
    if env:
        conf["enabled"] = env in _ON
        conf["enabled_source"] = ENABLED_ENV
    else:
        conf["enabled_source"] = conf["source"]

    level = str(conf.get("recommendation_level") or LEVEL_CONSIDERATION)
    if level not in LEVEL_RANK:
        logger.warning("teams notifications: unknown recommendation_level %r, "
                       "falling back to %s", level, LEVEL_CONSIDERATION)
        conf["recommendation_level"] = LEVEL_CONSIDERATION

    scope = str(conf.get("delivery_scope") or SCOPE_PERSONAL)
    if scope not in KNOWN_SCOPES:
        logger.warning("teams notifications: delivery_scope %r is not "
                       "implemented; delivery is disabled", scope)
        conf["enabled"] = False
        conf["disabled_reason"] = f"delivery_scope {scope!r} is not implemented"

    conf["maximum_risk_items"] = max(0, int(conf.get("maximum_risk_items") or 0))
    conf["maximum_update_items"] = max(0, int(conf.get("maximum_update_items") or 0))
    return conf


def enabled() -> bool:
    return bool(load().get("enabled"))


def recommendation_level() -> str:
    return str(load().get("recommendation_level") or LEVEL_CONSIDERATION)


def configured_recipients() -> List[Dict[str, Any]]:
    """The named pilot recipients declared in configuration.

    These are *selectors*, not destinations: a configured entry only becomes
    deliverable once a real Teams conversation reference has been captured for
    it and an operator has mapped it to portfolio contexts. Configuration alone
    can never authorise a delivery.
    """
    raw = load().get("recipients") or []
    out: List[Dict[str, Any]] = []
    for entry in raw:
        if isinstance(entry, str):
            out.append({"entra_object_id": entry})
        elif isinstance(entry, dict):
            out.append(dict(entry))
    return out


def retry_delay_seconds(attempt: int) -> int:
    """Backoff for the ``attempt``-th retry (1-based), flat after the last step."""
    steps = load().get("retry_backoff_seconds") or DEFAULTS["retry_backoff_seconds"]
    if not steps:
        return 60
    idx = min(max(attempt, 1) - 1, len(steps) - 1)
    try:
        return int(steps[idx])
    except (TypeError, ValueError):
        return 60


def max_attempts() -> int:
    try:
        return max(1, int(load().get("max_attempts") or 1))
    except (TypeError, ValueError):
        return int(DEFAULTS["max_attempts"])


def bot_credentials() -> Dict[str, Optional[str]]:
    """Bot identity from the environment / Key Vault-backed app settings.

    Never from configuration files and never logged. A missing value is
    returned as ``None`` so the caller fails closed with an explicit reason
    rather than attempting an unauthenticated send.
    """
    return {
        "app_id": (os.environ.get("TRAKT_TEAMS_BOT_APP_ID") or "").strip() or None,
        "app_password": (os.environ.get("TRAKT_TEAMS_BOT_APP_PASSWORD") or "").strip() or None,
        "tenant_id": (os.environ.get("TRAKT_TEAMS_BOT_TENANT_ID") or "").strip() or None,
    }
