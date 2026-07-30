"""operations_control.catalogue — which portfolios a client actually has.

There is no new registry here, and there must not be one. This module READS the
two that already exist and joins them:

* :mod:`apps.blob_trigger_app.source_registry` — the authoritative record of
  which ``source_portfolio_id`` belongs to which client, which datasets (funded /
  pipeline) it delivers, at what frequency, and whether the funded book is
  ``regime_required``. This is what the automated intake route consults, so it is
  what the manual route must consult too.
* ``config/client/portfolio_registry.yaml`` via
  :mod:`mi_agent.portfolio_metadata` — an OPTIONAL overlay carrying the
  human-readable ``source_portfolio_label`` ("ERE Direct Originations"). Absent
  by default; when absent, the identifier is the display name.

Asset class is reported as it genuinely is today: the platform is configured for
exactly one asset (``ASSET_MODEL``) and the effective-configuration resolver
takes it as a default rather than looking it up per portfolio. It is shown as
platform-level context, never as something derived from the portfolio — see the
current-state note in ``docs/operations_control_centre``.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from .contracts import (
    BATCH_DATASET_DEFAULT,
    BATCH_FREQUENCY_DEFAULT,
    OUTCOME_MI,
    OUTCOME_MI_ANNEX2,
)

logger = logging.getLogger("trakt.operations_control.catalogue")

#: Book types, in operator language.
BOOK_TYPE_LABELS = {"direct": "Direct originations",
                    "acquired": "Acquired book"}

DATASET_LABELS = {"funded": "Funded book", "pipeline": "Pipeline",
                  "forecast": "Forecast"}


def _labels(client_id: str) -> Dict[str, Dict[str, Any]]:
    """The optional display-name overlay. Never fatal — it is an overlay."""
    try:
        from mi_agent.portfolio_metadata import load_portfolio_metadata
        return load_portfolio_metadata(client_id)
    except Exception:  # noqa: BLE001 — a missing overlay is the normal case
        logger.debug("no portfolio label overlay available", exc_info=True)
        return {}


def _book_type(portfolio_id: str, records: List[Any]) -> str:
    for r in records:
        if getattr(r, "source_portfolio_type", None):
            return str(r.source_portfolio_type)
    pid = (portfolio_id or "").lower()
    return "acquired" if pid.startswith("acquired") else "direct"


def describe_portfolio(portfolio_id: str, records: List[Any],
                       overlay: Optional[Dict[str, Any]] = None
                       ) -> Dict[str, Any]:
    """One governed portfolio, as an operator needs to see it."""
    from .configuration.packages import ASSET_MODEL

    overlay = overlay or {}
    datasets = sorted({str(r.dataset) for r in records})
    # Regime scope is a question only the funded book answers — the same rule
    # the automated intake applies before it consults the registry at all.
    regime_required = any(bool(getattr(r, "regime_required", False))
                          for r in records
                          if str(r.dataset) == BATCH_DATASET_DEFAULT)
    frequencies = {str(r.dataset): str(r.frequency) for r in records}
    asset_id, asset = next(iter(ASSET_MODEL.items()), ("", {"label": ""}))
    label = (overlay.get("source_portfolio_label") or overlay.get("label")
             or portfolio_id)
    book_type = _book_type(portfolio_id, records)
    return {
        "portfolio_id": portfolio_id,
        "display_name": str(label),
        "book_type": book_type,
        "book_type_label": BOOK_TYPE_LABELS.get(book_type, book_type),
        "datasets": datasets,
        "dataset_labels": [DATASET_LABELS.get(d, d) for d in datasets],
        "frequencies": frequencies,
        "regime_required": regime_required,
        # What Trakt will prepare for this book. Derived here from the same
        # source the automated route uses, so the operator is never asked to
        # guess it and the two doors cannot disagree.
        "outcome": OUTCOME_MI_ANNEX2 if regime_required else OUTCOME_MI,
        "outcome_label": ("MI Reporting + ESMA Annex 2 delivery"
                          if regime_required else "MI Reporting"),
        "reporting_requirement": ("Management information and the ESMA Annex 2 "
                                  "delivery" if regime_required
                                  else "Management information only"),
        "asset_id": asset_id,
        "asset_label": asset.get("label", ""),
        "statuses": sorted({str(getattr(r, "status", "")) for r in records
                            if getattr(r, "status", "")}),
        "registered": True,
    }


def list_portfolios(registry: Any, client_id: str) -> List[Dict[str, Any]]:
    """Every portfolio registered to one client, in display order.

    Only this client's records are ever read, so a portfolio belonging to
    somebody else can never appear in — or be chosen from — this list.
    """
    try:
        records = [r for r in registry.records() if r.client_id == client_id]
    except Exception:  # noqa: BLE001 — an unreadable registry is "none known"
        logger.exception("could not read the source registry for %s", client_id)
        return []
    overlay = _labels(client_id)
    by_portfolio: Dict[str, List[Any]] = {}
    for r in records:
        by_portfolio.setdefault(str(r.source_portfolio_id), []).append(r)
    out = [describe_portfolio(pid, recs, overlay.get(pid.lower()))
           for pid, recs in by_portfolio.items()]
    return sorted(out, key=lambda p: p["display_name"].lower())


def find_portfolio(registry: Any, client_id: str,
                   portfolio_id: str) -> Optional[Dict[str, Any]]:
    """One portfolio of this client, or ``None`` when the client does not have it.

    ``None`` covers both "no such portfolio" and "that portfolio belongs to a
    different client" — the caller must treat them the same, because telling
    them apart would reveal another client's portfolio names.
    """
    for portfolio in list_portfolios(registry, client_id):
        if portfolio["portfolio_id"] == portfolio_id:
            return portfolio
    return None


def unregistered_portfolio(portfolio_id: str) -> Dict[str, Any]:
    """A portfolio Trakt has never seen: onboarding, not an error.

    Regime scope is unknown until the book is registered, so this mirrors the
    automated route's own default and prepares management information only.
    """
    from .configuration.packages import ASSET_MODEL

    asset_id, asset = next(iter(ASSET_MODEL.items()), ("", {"label": ""}))
    book_type = "acquired" if (portfolio_id or "").lower().startswith(
        "acquired") else "direct"
    return {
        "portfolio_id": portfolio_id,
        "display_name": portfolio_id,
        "book_type": book_type,
        "book_type_label": BOOK_TYPE_LABELS.get(book_type, book_type),
        "datasets": [BATCH_DATASET_DEFAULT],
        "dataset_labels": [DATASET_LABELS[BATCH_DATASET_DEFAULT]],
        "frequencies": {BATCH_DATASET_DEFAULT: BATCH_FREQUENCY_DEFAULT},
        "regime_required": False,
        "outcome": OUTCOME_MI,
        "outcome_label": "MI Reporting",
        "reporting_requirement": (
            "Management information only until this book is registered for "
            "regulatory reporting"),
        "asset_id": asset_id,
        "asset_label": asset.get("label", ""),
        "statuses": [],
        "registered": False,
    }
