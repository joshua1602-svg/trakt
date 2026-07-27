"""mi_agent_api/portfolio_context.py — THE governed portfolio-context service.

    capability id:  mi.portfolio.context

One resolution of "which portfolios does this workspace selection mean, and what
is analytically valid for them", shared by every channel:

    React MI Agent  ──┐
    M365 Copilot    ──┤
    PPTX / decks    ──┼──►  resolve_context(context_id)  ──►  ResolvedContext
    drill-through   ──┤                                        · PortfolioRegistry
    future agents   ──┘                                        · PortfolioScope
                                                               · capabilities

The service owns three things and no more:

  * building the governed :class:`~trakt_core.portfolio.PortfolioRegistry` from
    the active canonical dataset plus the governed metadata overlay;
  * resolving a requested context onto portfolios (dynamically — a group is
    whatever is currently typed that way);
  * resolving capabilities, intersecting declared origination metadata with the
    pipeline data actually discovered for the deployment.

It imports no web framework, so the HTTP routes, the Copilot action layer and
the deck generator all call exactly the same code.
"""

from __future__ import annotations

import logging
import os
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional

from trakt_core.portfolio import (  # noqa: F401  (capabilities_to_dict re-exported)
    CapabilityState,
    PortfolioRegistry,
    PortfolioScope,
    capabilities_to_dict,
    resolve_capabilities,
    resolve_scope,
)

logger = logging.getLogger("mi_agent_api.portfolio_context")

#: Stable capability identifier for the context contract.
CAPABILITY = "mi.portfolio.context"


@dataclass(frozen=True)
class ResolvedContext:
    """Everything a channel needs to render one portfolio selection."""

    registry: PortfolioRegistry
    scope: PortfolioScope
    capabilities: Dict[str, CapabilityState]
    #: Portfolios for which governed pipeline data was actually discovered.
    #: ``None`` means "not discoverable in this deployment" (declared metadata
    #: is then authoritative) — deliberately distinct from "none found".
    pipeline_portfolios: Optional[List[str]] = None

    def capability(self, name: str) -> Optional[CapabilityState]:
        return self.capabilities.get(name)

    def enabled(self, name: str) -> bool:
        state = self.capabilities.get(name)
        return bool(state and state.enabled)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "scope": self.scope.to_dict(),
            "capabilities": capabilities_to_dict(self.capabilities),
            "portfolios": [p.to_dict() for p in self.registry],
            "contexts": self.registry.contexts(),
        }


def _client_id(explicit: Optional[str] = None) -> Optional[str]:
    if explicit:
        return explicit
    from .datasets import _platform_client_id
    try:
        return _platform_client_id() or os.environ.get("MI_AGENT_CLIENT_ID") or None
    except Exception:  # noqa: BLE001 - discovery must never break the context
        return os.environ.get("MI_AGENT_CLIENT_ID") or None


def active_frame():
    """The active governed dataset, or ``None`` when none is loaded."""
    from .data_source import get_dataframe
    try:
        return get_dataframe()
    except Exception as exc:  # noqa: BLE001 - an unavailable dataset is not a 500
        logger.info("portfolio context: active dataset unavailable: %s", exc)
        return None


#: Small registry cache, ``key -> (frame, registry)``. Building the registry
#: groups over the whole tape and reads the metadata overlay, and every routed
#: chat question resolves a context, so without this the same registry is rebuilt
#: per request. The frame is kept in the VALUE deliberately: holding a reference
#: is what guarantees ``id(frame)`` cannot be recycled onto a different frame
#: while its entry is live. Bounded, so a long-running process cannot accumulate
#: frames.
_REGISTRY_CACHE: "OrderedDict[tuple, tuple]" = OrderedDict()
_REGISTRY_CACHE_MAX = 3


def reset_registry_cache() -> None:
    """Drop the cached registries (used after a dataset switch, and by tests)."""
    _REGISTRY_CACHE.clear()


def build_registry(df=None, *, client_id: Optional[str] = None,
                   metadata: Optional[Mapping[str, Mapping[str, Any]]] = None
                   ) -> PortfolioRegistry:
    """The governed registry for a frame (defaults to the active dataset)."""
    from mi_agent.portfolio_scope import registry_for_frame
    cid = _client_id(client_id)
    frame = df if df is not None else active_frame()
    if metadata is not None or frame is None:
        # An explicit overlay is a caller-specific view — never cached.
        return registry_for_frame(frame, client_id=cid, metadata=metadata)

    key = (id(frame), len(frame), cid)
    cached = _REGISTRY_CACHE.get(key)
    if cached is not None:
        _REGISTRY_CACHE.move_to_end(key)
        return cached[1]
    registry = registry_for_frame(frame, client_id=cid)
    _REGISTRY_CACHE[key] = (frame, registry)
    while len(_REGISTRY_CACHE) > _REGISTRY_CACHE_MAX:
        _REGISTRY_CACHE.popitem(last=False)
    return registry


def discovered_pipeline_portfolios(client_id: Optional[str] = None,
                                   registry: Optional[PortfolioRegistry] = None
                                   ) -> Optional[List[str]]:
    """Portfolios with governed pipeline data actually available, or ``None``.

    Resolution order, most specific first:

      1. provenance carried by the discovered pipeline extracts themselves;
      2. an explicit ``dataset: pipeline`` registration in the source registry;
      3. pipeline sources discovered for the client but with no per-portfolio
         attribution — the deployment shape ERE runs today, where the originating
         portfolios in the registry are the ones the pipeline belongs to;
      4. no pipeline root or no sources at all → ``[]`` (nothing has pipeline
         data), which correctly disables the Pipeline tab everywhere.

    ``None`` is returned only when discovery could not run, so declared metadata
    stays authoritative rather than a transient storage fault disabling a tab.
    """
    cid = _client_id(client_id)

    from mi_agent.portfolio_metadata import pipeline_registered_portfolios
    registered = pipeline_registered_portfolios(cid)

    try:
        from .datasets import _pipeline_discovery_root
        from . import pipeline_contract as pipeline_mod
        root = _pipeline_discovery_root()
    except Exception as exc:  # noqa: BLE001
        logger.info("pipeline discovery unavailable: %s", exc)
        return registered

    if not root:
        return [] if registered is None else registered
    try:
        sources = pipeline_mod.discover_pipeline_sources(root, client_id=cid)
    except Exception as exc:  # noqa: BLE001 - discovery must never break the context
        logger.info("pipeline source discovery failed: %s", exc)
        return registered
    if not sources:
        return [] if registered is None else registered

    # (1) per-source provenance, when the governed extract carries it.
    attributed: List[str] = []
    for src in sources:
        pid = src.get("source_portfolio_id") or src.get("sourcePortfolioId")
        pid = str(pid).strip() if pid else ""
        if pid and pid not in attributed:
            attributed.append(pid)
    if attributed:
        return attributed

    # (2) explicit source-registry registration.
    if registered:
        return registered

    # (3) sources exist for the client but are not attributed per portfolio: the
    # originating portfolios in the registry own them. This preserves ERE's
    # current single-originator behaviour without hard-coding direct_001.
    reg = registry if registry is not None else build_registry(client_id=cid)
    return [p.portfolio_id for p in reg if p.originates and p.pipeline_data_available]


def resolve_context(context_id: Optional[Any] = None, *,
                    df=None,
                    client_id: Optional[str] = None,
                    registry: Optional[PortfolioRegistry] = None,
                    pipeline_portfolios: Optional[List[str]] = None,
                    discover_pipeline: bool = True) -> ResolvedContext:
    """Resolve one workspace context end to end.

    This is the single entry point every channel uses. Passing ``df`` scopes the
    resolution to a specific run's frame; omitting it uses the active governed
    dataset.
    """
    reg = registry if registry is not None else build_registry(df, client_id=client_id)
    scope = resolve_scope(reg, context_id)
    pipes = pipeline_portfolios
    if pipes is None and discover_pipeline:
        pipes = discovered_pipeline_portfolios(client_id=client_id, registry=reg)
    caps = resolve_capabilities(reg, scope, pipeline_portfolios=pipes)
    return ResolvedContext(registry=reg, scope=scope, capabilities=caps,
                           pipeline_portfolios=pipes)


def context_index(client_id: Optional[str] = None, df=None) -> Dict[str, Any]:
    """The full portfolio-context contract for a client.

    The hierarchy (Total → type groups → portfolios), each portfolio's governed
    metadata, and the resolved capability set for EVERY selectable context — so a
    channel can gate its navigation without a round trip per selection, and can
    never compute applicability itself.
    """
    reg = build_registry(df, client_id=client_id)
    pipes = discovered_pipeline_portfolios(client_id=client_id, registry=reg)
    contexts = reg.contexts()
    for ctx in contexts:
        scope = resolve_scope(reg, ctx["context_id"])
        caps = resolve_capabilities(reg, scope, pipeline_portfolios=pipes)
        ctx["capabilities"] = capabilities_to_dict(caps)
    return {
        "available": bool(reg),
        "client_id": reg.client_id,
        "default_context_id": contexts[0]["context_id"] if contexts else None,
        "contexts": contexts,
        "portfolios": [p.to_dict() for p in reg],
        "portfolio_types": reg.types(),
        "pipeline_portfolios": pipes,
    }


def scope_frame(df, context_id: Optional[Any] = None, *,
                registry: Optional[PortfolioRegistry] = None,
                client_id: Optional[str] = None):
    """``(scoped_frame, scope)`` for a frame and a requested context.

    The ONE filter every portfolio-scoped surface applies. No route, chart,
    forecast or export re-implements it.
    """
    from mi_agent.portfolio_scope import apply_scope
    reg = registry if registry is not None else build_registry(df, client_id=client_id)
    scope = resolve_scope(reg, context_id)
    return apply_scope(df, scope), scope


def scope_metadata(df, scope: PortfolioScope, *,
                   capabilities: Optional[Mapping[str, CapabilityState]] = None,
                   fields: Optional[List[str]] = None) -> Dict[str, Any]:
    """The governed scope block every portfolio-scoped response carries.

    Channels render this; they never derive it. Field coverage is included when
    the caller names the fields its answer depends on.
    """
    from mi_agent.portfolio_scope import coverage_for_frame
    coverage = coverage_for_frame(df, scope, fields=fields or ())
    block: Dict[str, Any] = coverage.to_dict()
    block["scope"] = scope.to_dict()
    if capabilities is not None:
        block["capabilities"] = capabilities_to_dict(capabilities)
    return block
