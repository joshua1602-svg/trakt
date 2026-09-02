"""Who the dashboard says it is reporting on.

The dashboard header showed ``str(client_id).upper()`` — ``CLIENT_001``, or the
literal placeholder ``PLATFORM`` when nothing identified the tenant at all. That
is an internal identifier on the one surface a client looks at first.

The client's NAME is a governed fact, captured at onboarding and held in exactly
the places this estate already keeps per-client governed configuration. This
module reads it from those places and nowhere else. It never derives a name from
the tape, and never invents one: with no governed name the caller keeps the
identifier, which is today's behaviour.

Tenant safety is the whole design constraint. Labelling one client's book with
another client's name is worse than showing an identifier, so every source here
is either KEYED BY the client id (so it can only ever return that client's own
name) or explicitly addressed to it. The one inference — the repository client
layer naming the deployment's single tenant — is made only when the tenancy
registry says this deployment is single-tenant, which is the same assumption
``trakt_core.tenancy`` already documents and applies.
"""
from __future__ import annotations

import logging
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Optional

from . import currency as _currency

logger = logging.getLogger(__name__)

#: The per-client governed directory, resolved against the REPOSITORY rather
#: than the working directory — the same root, and the same reasoning, as
#: ``risk_limits._CONFIG_ROOT``: whether a client is named must not depend on
#: where the process happened to be started.
_CLIENTS_ROOT = Path(__file__).resolve().parents[1] / "config" / "clients"
_CLIENT_FILE = "client.yaml"

#: Keys that may carry the client's name, most-governed first. ``display_name``
#: is what the client configuration and the tenant registry both use;
#: ``client_name`` is what OCC's adoption answers call the same field.
_NAME_KEYS = ("display_name", "client_name", "business_name", "name")


def _text(value: Any) -> Optional[str]:
    if isinstance(value, str) and value.strip():
        return value.strip()
    return None


def _name_from_block(block: Any) -> Optional[str]:
    if not isinstance(block, dict):
        return None
    for key in _NAME_KEYS:
        found = _text(block.get(key))
        if found:
            return found
    return None


def _name_from_document(doc: Any) -> Optional[str]:
    """A client name declared anywhere a governed document conventionally puts
    it: under ``client:``, or at the top level."""
    if not isinstance(doc, dict):
        return None
    return _name_from_block(doc.get("client")) or _name_from_block(doc)


# --------------------------------------------------------------------------- #
# 1. The per-client governed directory — keyed by id
# --------------------------------------------------------------------------- #
@lru_cache(maxsize=16)
def _load_client_file(client_id: str) -> Dict[str, Any]:
    """``config/clients/<client_id>/client.yaml``, or ``{}``.

    Never raises: an unreadable or malformed file degrades to "no governed
    name", so the header falls back to the identifier rather than 500-ing.
    """
    path = _CLIENTS_ROOT / client_id / _CLIENT_FILE
    try:
        if not path.exists():
            return {}
        import yaml
        doc = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        if not isinstance(doc, dict):
            return {}
        # The directory already keys this file to a client, so the declared id
        # is redundant — which makes a MISMATCH a copy-paste error, not a
        # governed statement. Refuse it rather than serve one client's name from
        # another's directory.
        declared = _text((doc.get("client") or {}).get("client_id")
                         if isinstance(doc.get("client"), dict) else None)
        if declared and declared.casefold() != str(client_id).casefold():
            logger.warning("client identity at %s declares client_id %r; "
                           "ignoring it for %r", path, declared, client_id)
            return {}
        return doc
    except Exception as exc:  # noqa: BLE001 - identity must never break MI
        logger.info("client identity unavailable at %s: %s", path, exc)
        return {}


# --------------------------------------------------------------------------- #
# 2. The tenant registry — keyed by id
# --------------------------------------------------------------------------- #
def _registry():
    """The loaded tenant registry, or ``None`` when it cannot be read."""
    try:
        from trakt_core.tenancy import load_tenant_registry
        return load_tenant_registry()
    except Exception as exc:  # noqa: BLE001 - identity must never break MI
        logger.info("tenant registry unavailable: %s", exc)
        return None


def _name_from_registry(client_id: str) -> Optional[str]:
    registry = _registry()
    record = registry.get(client_id) if registry is not None else None
    return _text(getattr(record, "display_name", None)) if record is not None else None


def _multi_tenant_deployment() -> bool:
    """True when an explicit tenancy config declares the tenants served.

    With one, a client configuration that does not name its client cannot be
    assumed to be about the client being rendered.
    """
    registry = _registry()
    return bool(getattr(registry, "configured", False))


def _served_tenant() -> Optional[str]:
    """The single tenant this deployment serves, per ``dependencies``."""
    try:
        from .dependencies import default_tenant_id
        return _text(default_tenant_id())
    except Exception as exc:  # noqa: BLE001 - identity must never break MI
        logger.info("served tenant unresolved: %s", exc)
        return None


# --------------------------------------------------------------------------- #
# 3. The governed client configuration — addressed to this client
# --------------------------------------------------------------------------- #
def _config_is_addressed_to(client_id: str, doc: Dict[str, Any]) -> bool:
    """Whether this configuration is about ``client_id``.

    True when the configuration says so itself — ``client.client_id`` matches.
    That is the only positive answer when the configuration declares a client at
    all: a config that names ``ere_funding_uk`` is ERE's, and reading its name
    onto a different tenant is precisely the misattribution this module exists
    to prevent, however few tenants the deployment serves.

    A configuration that declares NO client is taken to describe the single
    tenant a single-tenant deployment serves — the case ``trakt_core.tenancy``
    already treats as "the context's tenant owns this deployment". With an
    explicit tenancy registry, or for any client that is not the served tenant,
    even that inference is refused.
    """
    declared = _text((doc.get("client") or {}).get("client_id")
                     if isinstance(doc.get("client"), dict) else None)
    if declared:
        return declared.casefold() == str(client_id).casefold()
    if _multi_tenant_deployment():
        return False
    served = _served_tenant()
    return bool(served and served.casefold() == str(client_id).casefold())


def _name_from_client_config(client_id: str) -> Optional[str]:
    location = _currency.client_config_path(client_id)
    if not location:
        return None
    doc = _currency._load_client_config(location)
    if not doc or not _config_is_addressed_to(client_id, doc):
        return None
    return _name_from_document(doc)


# --------------------------------------------------------------------------- #
# Public
# --------------------------------------------------------------------------- #
def governed_client_name(client_id: Optional[str]) -> Optional[str]:
    """The client's approved name, or ``None`` when none is governed.

    Precedence, most-specific first — every source keyed by, or addressed to,
    this client:

      1. ``config/clients/<client_id>/client.yaml``;
      2. the tenant registry's ``display_name`` for this tenant;
      3. the governed client configuration, when it is this client's.

    Never raises. An unidentified caller gets ``None``: a client must be
    identified before anything can name it.
    """
    cid = _text(client_id)
    if not cid:
        return None
    return (_name_from_document(_load_client_file(cid))
            or _name_from_registry(cid)
            or _name_from_client_config(cid))


def portfolio_label(client_id: Optional[str]) -> str:
    """The client label for the dashboard header.

    The governed client name where one exists, else the identifier upper-cased —
    exactly the previous behaviour, so a deployment that has declared no name
    renders precisely what it rendered before.
    """
    return governed_client_name(client_id) or str(client_id or "").upper()


def cache_clear() -> None:
    """Drop memoised client identity (tests, and config reload)."""
    _load_client_file.cache_clear()
