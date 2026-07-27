"""trakt_core.runtime — the runtime mode, and why it cannot be flipped in Azure.

Tests and local development legitimately need to answer from fixture data.
Production must never do that. A single explicit mode expresses the difference:

    TRAKT_RUNTIME_MODE = production | development | test        (default: production)

Two properties make this safe rather than "just another env flag":

1. **The default is production.** An unset, empty or unrecognised value is
   production, so forgetting the variable fails closed.
2. **Azure refuses a non-production mode.** :func:`validate_runtime_mode` raises
   at import/startup when a non-production mode is requested while the Azure
   markers are present, so a stray app setting cannot quietly turn a live
   deployment into one that answers from synthetic data.

The mode is process-level and is read once per call from the environment; it is
never taken from a request. There is deliberately no per-request override.
"""

from __future__ import annotations

import logging
import os

logger = logging.getLogger("trakt_core.runtime")

MODE_PRODUCTION = "production"
MODE_DEVELOPMENT = "development"
MODE_TEST = "test"

_NON_PRODUCTION_MODES = frozenset({MODE_DEVELOPMENT, MODE_TEST})
_ALL_MODES = frozenset({MODE_PRODUCTION}) | _NON_PRODUCTION_MODES

RUNTIME_MODE_ENV = "TRAKT_RUNTIME_MODE"

#: Environment markers that mean "this process is a live Azure App Service /
#: Functions host". Mirrors apps.blob_trigger_app.storage; duplicated as literals
#: so trakt_core stays dependency-free, with a drift test in
#: tests/test_governance_runtime_mode.py.
_AZURE_MARKERS = ("WEBSITE_INSTANCE_ID", "WEBSITE_SITE_NAME")


def _raw_mode() -> str:
    return (os.environ.get(RUNTIME_MODE_ENV) or "").strip().lower()


def running_in_azure() -> bool:
    return any(os.environ.get(m) for m in _AZURE_MARKERS)


def runtime_mode() -> str:
    """The effective runtime mode. Unknown / unset values resolve to production.

    In Azure the mode is always production: a non-production request is both
    refused by :func:`validate_runtime_mode` at startup and ignored here, so the
    two cannot disagree even if startup validation was bypassed.
    """
    raw = _raw_mode()
    if raw not in _ALL_MODES:
        return MODE_PRODUCTION
    if raw in _NON_PRODUCTION_MODES and running_in_azure():
        return MODE_PRODUCTION
    return raw


def is_production() -> bool:
    return runtime_mode() == MODE_PRODUCTION


def validate_runtime_mode() -> None:
    """Fail closed at startup on an unsafe or unrecognised mode.

    Raises ``RuntimeError`` when a non-production mode is requested inside Azure.
    An unrecognised value is logged and treated as production rather than raising,
    so a typo degrades safely instead of taking the service down.
    """
    raw = _raw_mode()
    if raw and raw not in _ALL_MODES:
        logger.error(
            "%s=%r is not one of %s — running in %s.",
            RUNTIME_MODE_ENV, raw, sorted(_ALL_MODES), MODE_PRODUCTION)
        return
    if raw in _NON_PRODUCTION_MODES and running_in_azure():
        raise RuntimeError(
            f"{RUNTIME_MODE_ENV}={raw!r} is not permitted in Azure: a deployed "
            "Trakt runtime must not be able to answer from fixture or synthetic "
            f"data. Remove the {RUNTIME_MODE_ENV} app setting."
        )
    if raw in _NON_PRODUCTION_MODES:
        logger.warning(
            "%s=%s — fixture and synthetic data sources are permitted. This mode "
            "is for local development and tests only.", RUNTIME_MODE_ENV, raw)
