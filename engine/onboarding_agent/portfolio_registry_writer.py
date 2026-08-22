#!/usr/bin/env python3
"""engine.onboarding_agent.portfolio_registry_writer — publish governed
per-portfolio metadata that onboarding established.

Onboarding decides a portfolio's ASSET CLASS (see
:mod:`engine.onboarding_agent.onboarding_context`). Until now that decision
lived only in the onboarding artefacts: ``27_onboarding_context.json`` is read
by the onboarding agent and its workbench and by nothing downstream, so MI never
learned what kind of book it was reporting on and could only ever admit
``cross_asset`` semantics.

This module closes that handoff. It writes the decision into the EXISTING
governed portfolio registry — the same optional file that already carries
origination capability, forecast treatment and any supplied runoff curve
(``config/client/portfolio_registry.yaml``, or wherever
``TRAKT_PORTFOLIO_REGISTRY`` points). One writer, one file, one source of truth:
:mod:`mi_agent.portfolio_metadata` reads it, ``trakt_core.portfolio`` resolves it
onto ``PortfolioRecord.asset_class``, and every MI consumer reads it from there.

What this module deliberately does NOT do:

* it does not stamp an ``asset_class`` column onto the canonical tape. The asset
  class is a fact about the BOOK, not about an individual loan, and duplicating
  it per row would create a second source of truth that could disagree;
* it does not invent an asset class. A portfolio onboarding could not classify
  is written without one, and MI then reads the absence conservatively rather
  than assuming a class;
* it does not overwrite entries it did not author. Merging is by portfolio id,
  and only the keys this writer owns are replaced.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional

import yaml

logger = logging.getLogger("engine.onboarding_agent.portfolio_registry_writer")

#: The keys this writer owns. Anything else already in the file is preserved
#: untouched, so a hand-authored runoff curve or forecast treatment survives a
#: re-run of onboarding.
_OWNED_KEYS = ("asset_class",)

_ID_KEY = "source_portfolio_id"


def normalise_asset_class(value: Any) -> Optional[str]:
    """The governed asset-class vocabulary, resolved once at this boundary.

    Delegates to :mod:`mi_agent.portfolio_metadata`, which is where the synonym
    table lives, so the writer and the reader can never disagree about what
    "equity_release_mortgage" means.
    """
    from mi_agent.portfolio_metadata import normalise_asset_class as _n

    return _n(value)


def build_entries(portfolios: Iterable[Mapping[str, Any]], *,
                  asset_class: Any = None) -> List[Dict[str, Any]]:
    """Registry entries for the portfolios onboarding established.

    ``portfolios`` are mappings carrying at least ``source_portfolio_id``. A
    per-portfolio ``asset_class`` wins over the client-level ``asset_class``
    fallback, which is how a client with two books of different asset classes is
    represented without a second mechanism.
    """
    default = normalise_asset_class(asset_class)
    entries: List[Dict[str, Any]] = []
    for portfolio in portfolios or ():
        pid = str(portfolio.get(_ID_KEY) or portfolio.get("portfolio_id") or "").strip()
        if not pid:
            continue
        resolved = normalise_asset_class(portfolio.get("asset_class")) or default
        entry: Dict[str, Any] = {_ID_KEY: pid}
        if resolved:
            entry["asset_class"] = resolved
        else:
            logger.info("no governed asset class for portfolio %r; MI will read "
                        "it as unknown rather than assume one", pid)
        entries.append(entry)
    return entries


def merge_entries(existing: Iterable[Mapping[str, Any]],
                  updates: Iterable[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    """Existing registry entries with this writer's keys applied.

    Entries the writer did not author are preserved in place and in order, so
    publishing an asset class never disturbs a supplied runoff curve.
    """
    merged: List[Dict[str, Any]] = [dict(e) for e in (existing or ())]
    index = {str(e.get(_ID_KEY) or e.get("portfolio_id") or "").lower(): e
             for e in merged}
    for update in updates or ():
        pid = str(update.get(_ID_KEY) or "").lower()
        target = index.get(pid)
        if target is None:
            merged.append(dict(update))
            index[pid] = merged[-1]
            continue
        for key in _OWNED_KEYS:
            if key in update:
                target[key] = update[key]
    return merged


def publish(path: str | Path, portfolios: Iterable[Mapping[str, Any]], *,
            asset_class: Any = None) -> Dict[str, Any]:
    """Write the governed portfolio registry, merging with anything already there.

    Returns the document written, so a caller can record it as evidence.
    """
    target = Path(path)
    document: Dict[str, Any] = {}
    if target.exists():
        try:
            document = yaml.safe_load(target.read_text(encoding="utf-8")) or {}
        except Exception:  # noqa: BLE001 - a malformed file is not silently lost
            logger.exception("existing portfolio registry at %s could not be "
                             "parsed; refusing to overwrite it", target)
            raise
    if not isinstance(document, dict):
        raise ValueError(f"portfolio registry at {target} is not a mapping")

    document["portfolios"] = merge_entries(
        document.get("portfolios") or [],
        build_entries(portfolios, asset_class=asset_class))
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        "# Governed portfolio registry — per-portfolio business metadata.\n"
        "#\n"
        "# Written by engine.onboarding_agent.portfolio_registry_writer at\n"
        "# onboarding. `asset_class` is the decision onboarding made about what\n"
        "# kind of book this is; MI reads it from here and from nowhere else.\n"
        "# Hand-authored keys (runoff curves, forecast treatment) are preserved.\n"
        + yaml.safe_dump(document, sort_keys=False),
        encoding="utf-8")
    logger.info("published governed portfolio metadata for %d portfolio(s) to %s",
                len(document["portfolios"]), target)
    return document
