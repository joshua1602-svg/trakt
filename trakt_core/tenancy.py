"""trakt_core.tenancy — tenant identity, portfolio selection, and the gate between them.

The defect this module closes: the MI service previously computed

    effective_portfolio_id = portfolio_id or client_id

so a caller-supplied ``portfolio_id`` *replaced* the authenticated client for the
rest of the execution. Tenant identity was therefore request-controlled.

The model here separates three things that were conflated:

  * **tenant** — who the caller is, from :class:`~trakt_core.context.ExecutionContext`.
    Never from the request body.
  * **portfolio selector** — which book within that tenant the caller wants
    (``ERE``, ``direct_001``, ``acquired_002``). Request-supplied, and authorised
    against the tenant before anything is read.
  * **run id** — which dated cut of that book. Request-supplied, validated for
    shape only.

``portfolio_id`` keeps its existing wire format — ``"{selector}"`` or
``"{selector}/{run_id}"`` — so React and Copilot are unchanged.

Configuration
-------------
``config/tenancy.yaml`` (optional) declares tenants::

    tenants:
      ERE:
        display_name: Equity Release Europe
        default_portfolio: ERE
        portfolios: [ERE, direct_001, acquired_001, acquired_002]
        portfolio_patterns: ['^(direct|acquired)_\\d+$']

When no config file is present the deployment is treated as single-tenant: the
tenant from the trusted context owns its own namespace, any well-formed selector
within it is allowed, and the *dataset resolver* still scopes every read to that
tenant's storage root. That preserves today's ERE behaviour while removing the
request's ability to redefine the tenant.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, FrozenSet, Optional, Tuple

from .context import SCOPE_PORTFOLIO_READ, ExecutionContext
from . import config_cache
from .errors import ErrorCode, TraktError

#: A portfolio selector / run id is a plain slug. Anchored, no separators, so a
#: value can never traverse a storage path or inject a container segment.
_SAFE_TOKEN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")

DEFAULT_TENANCY_CONFIG = "config/tenancy.yaml"
TENANCY_CONFIG_ENV = "TRAKT_TENANCY_CONFIG"


def _valid_token(value: str) -> bool:
    return bool(_SAFE_TOKEN.match(value or "")) and ".." not in value


@dataclass(frozen=True)
class TenantRecord:
    """One tenant's portfolio namespace."""

    tenant_id: str
    display_name: Optional[str] = None
    default_portfolio_id: Optional[str] = None
    #: Explicit allow-list. Empty means "not restricted by an explicit list".
    portfolio_ids: FrozenSet[str] = field(default_factory=frozenset)
    #: Additional regex patterns a selector may match.
    portfolio_patterns: Tuple[str, ...] = ()
    #: True when this record was synthesised for a single-tenant deployment with
    #: no tenancy config. Such a tenant owns any well-formed selector.
    open_namespace: bool = False

    def default_selector(self) -> str:
        return self.default_portfolio_id or self.tenant_id

    def allows_selector(self, selector: str) -> bool:
        if selector == self.tenant_id:
            return True
        if self.open_namespace and not self.portfolio_ids and not self.portfolio_patterns:
            return True
        if selector in self.portfolio_ids:
            return True
        return any(re.match(p, selector) for p in self.portfolio_patterns)


@dataclass(frozen=True)
class AuthorisedPortfolio:
    """Proof that a context may read a portfolio.

    The dataset resolver requires one of these rather than raw strings, so it is
    not possible to load data that has not been through
    :func:`authorise_portfolio_access`.
    """

    tenant_id: str
    #: The book within the tenant (``ERE``, ``direct_001``, …). Always populated:
    #: falls back to the tenant's default when the caller named no portfolio.
    selector: str
    #: The dated cut, when one was requested.
    run_id: Optional[str] = None
    #: Exactly what the caller asked for, or ``None`` when it named nothing.
    #: Kept distinct from ``selector`` because downstream dataset resolution has
    #: a documented "no portfolio selected" behaviour (serve the active dataset)
    #: that differs from "the default portfolio was selected"; substituting the
    #: default here would silently change which frame an existing caller gets.
    requested_portfolio_id: Optional[str] = None

    @property
    def portfolio_id(self) -> str:
        """The wire-format identifier, as React and Copilot use it."""
        return f"{self.selector}/{self.run_id}" if self.run_id else self.selector

    @property
    def explicit(self) -> bool:
        """True when the caller named a portfolio rather than taking the default."""
        return self.requested_portfolio_id is not None

    def to_dict(self) -> Dict[str, Any]:
        return {"tenant_id": self.tenant_id, "portfolio_id": self.portfolio_id,
                "selector": self.selector, "run_id": self.run_id}


class TenantRegistry:
    """The set of tenants this deployment serves."""

    def __init__(self, records: Dict[str, TenantRecord], *, configured: bool) -> None:
        self._records = dict(records)
        #: True when an explicit tenancy config was loaded. When False the
        #: registry synthesises an open namespace for the context's tenant.
        self.configured = configured

    # -- lookup ------------------------------------------------------------- #
    def get(self, tenant_id: str) -> Optional[TenantRecord]:
        return self._records.get(tenant_id)

    def resolve(self, tenant_id: str) -> TenantRecord:
        """The record for ``tenant_id``.

        With an explicit config an unknown tenant is denied. Without one, a
        single-tenant deployment gets an open namespace for its own tenant.
        """
        record = self._records.get(tenant_id)
        if record is not None:
            return record
        if self.configured:
            raise TraktError(
                ErrorCode.PERMISSION_DENIED,
                "This deployment does not serve the authenticated tenant.",
                details={"tenant_id": tenant_id})
        return TenantRecord(tenant_id=tenant_id, open_namespace=True)

    def tenant_ids(self) -> FrozenSet[str]:
        return frozenset(self._records)

    def __len__(self) -> int:  # pragma: no cover - convenience
        return len(self._records)


def load_tenant_registry(
    config_path: Optional[str | Path] = None,
    *,
    default_tenant_id: Optional[str] = None,
) -> TenantRegistry:
    """Load ``config/tenancy.yaml`` if present, else build a single-tenant registry.

    Never raises on a malformed or unreadable config: a broken tenancy file falls
    back to the single-tenant registry rather than taking the API down, and the
    dataset resolver still scopes reads to the context tenant.

    Parsed once per content version — see :mod:`trakt_core.config_cache`. This is
    the registry the MI Query path loads on **every** request through
    ``mi_agent_api.dependencies.build_dependencies``, so it is the one the human
    channels feel. ``default_tenant_id`` is part of the cache key because it
    changes the result for identical file content.
    """
    path = Path(config_path or os.environ.get(TENANCY_CONFIG_ENV) or DEFAULT_TENANCY_CONFIG)
    return config_cache.get_or_load(
        "tenancy", path,
        lambda: _load_tenant_registry(path, default_tenant_id=default_tenant_id),
        extra_key=(default_tenant_id,))


def _load_tenant_registry(
    path: Path,
    *,
    default_tenant_id: Optional[str] = None,
) -> TenantRegistry:
    """The uncached load. Behaviour is unchanged from before caching existed."""
    records: Dict[str, TenantRecord] = {}
    configured = False

    if path.exists():
        try:
            import yaml
            raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
            for tenant_id, spec in (raw.get("tenants") or {}).items():
                spec = spec or {}
                records[str(tenant_id)] = TenantRecord(
                    tenant_id=str(tenant_id),
                    display_name=spec.get("display_name"),
                    default_portfolio_id=spec.get("default_portfolio"),
                    portfolio_ids=frozenset(str(p) for p in (spec.get("portfolios") or [])),
                    portfolio_patterns=tuple(str(p) for p in (spec.get("portfolio_patterns") or [])),
                )
            configured = bool(records)
        except Exception:  # noqa: BLE001 - a bad config must not break the API
            import logging
            logging.getLogger("trakt_core.tenancy").exception(
                "tenancy config %s could not be parsed; falling back to "
                "single-tenant resolution", path)
            records, configured = {}, False

    if not configured and default_tenant_id:
        records[default_tenant_id] = TenantRecord(
            tenant_id=default_tenant_id, open_namespace=True)

    return TenantRegistry(records, configured=configured)


def split_portfolio_id(portfolio_id: Optional[str]) -> Tuple[Optional[str], Optional[str]]:
    """``"{selector}"`` / ``"{selector}/{run_id}"`` -> ``(selector, run_id)``."""
    if not portfolio_id:
        return None, None
    text = str(portfolio_id).strip()
    if not text:
        return None, None
    if "/" in text:
        selector, run_id = text.split("/", 1)
        return (selector.strip() or None), (run_id.strip() or None)
    return text, None


def authorise_portfolio_access(
    context: ExecutionContext,
    portfolio_id: Optional[str],
    *,
    registry: TenantRegistry,
) -> AuthorisedPortfolio:
    """Authorise ``portfolio_id`` for ``context``, or raise :class:`TraktError`.

    Called before any dataset is resolved or loaded. The tenant always comes from
    ``context``; ``portfolio_id`` can only ever *narrow* within it.
    """
    context.require_scope(SCOPE_PORTFOLIO_READ)
    record = registry.resolve(context.tenant_id)

    requested = str(portfolio_id).strip() if portfolio_id else None
    selector, run_id = split_portfolio_id(requested)
    if selector is None:
        selector = record.default_selector()

    if not _valid_token(selector):
        raise TraktError(
            ErrorCode.INVALID_INPUT,
            "The portfolio identifier is not a valid portfolio selector.",
            request_id=context.request_id)
    if run_id is not None and not _valid_token(run_id):
        raise TraktError(
            ErrorCode.INVALID_INPUT,
            "The reporting-run identifier is not valid.",
            request_id=context.request_id)

    # Naming another tenant is a tenant mismatch, not a missing portfolio — the
    # caller is reaching outside its own namespace and should be told so.
    if selector != context.tenant_id and selector in registry.tenant_ids():
        raise TraktError(
            ErrorCode.TENANT_MISMATCH,
            "The requested portfolio belongs to a different tenant.",
            request_id=context.request_id,
            details={"portfolio_id": selector})

    if not record.allows_selector(selector):
        raise TraktError(
            ErrorCode.PORTFOLIO_NOT_AUTHORISED,
            "The authenticated tenant is not authorised for this portfolio.",
            request_id=context.request_id,
            details={"portfolio_id": selector})

    return AuthorisedPortfolio(
        tenant_id=context.tenant_id, selector=selector, run_id=run_id,
        requested_portfolio_id=requested)
