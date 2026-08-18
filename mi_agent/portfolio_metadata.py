"""mi_agent.portfolio_metadata — the governed per-portfolio metadata overlay.

:mod:`trakt_core.portfolio` defines *what* a portfolio record contains. This
module resolves *where the values come from* for a running deployment, and it is
the only place that knows:

  1. the optional governed portfolio registry file
     (``TRAKT_PORTFOLIO_REGISTRY`` → ``config/client/portfolio_registry.yaml``),
     which carries per-portfolio origination capability, forecast treatment and
     any SUPPLIED runoff curve; and
  2. the source registry (``apps.blob_trigger_app.source_registry``), whose
     ``dataset: pipeline`` records are the operational evidence that a portfolio
     originates and supplies pipeline data.

Both are optional. With neither present the registry still resolves entirely from
canonical provenance plus the single governed default table in
``trakt_core.portfolio.DEFAULT_ORIGINATION_BY_TYPE`` — which is why ERE works
today with no new configuration, while a future originating acquired vehicle (or
an acquired book with a supplied runoff curve) needs only a metadata entry.

Runoff curves are *consumed*, never generated: Trakt holds an acquired balance
constant and discloses it unless the client supplies an approved curve.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

logger = logging.getLogger("mi_agent.portfolio_metadata")

_REPO_ROOT = Path(__file__).resolve().parents[1]

#: Default location of the governed portfolio registry file. Optional.
DEFAULT_REGISTRY_PATH = _REPO_ROOT / "config" / "client" / "portfolio_registry.yaml"

#: Environment override for the registry file (path or ``blob://`` URI).
ENV_REGISTRY_PATH = "TRAKT_PORTFOLIO_REGISTRY"

#: Keys a portfolio entry may set. Anything else is ignored rather than trusted.
_ALLOWED_KEYS = frozenset({
    "portfolio_id", "source_portfolio_id", "source_portfolio_type", "portfolio_type",
    "source_portfolio_label", "label", "originates", "pipeline_data_available",
    "forecast_treatment", "runoff_profile_id", "runoff_profile", "runoff_curve",
    "monthly_retention", "reporting_dates",
    # The governed ASSET CLASS, established once by onboarding. It is what lets
    # MI compose the common semantic core plus this asset's own metrics and
    # dimensions (mi_agent.business_semantics.applies_to_asset). Carried here
    # rather than on the tape because it is a fact about the BOOK, not about any
    # individual loan, and because onboarding — not the extract — decides it.
    "asset_class",
})

#: Asset-class synonyms, mapped onto the Business Semantics Registry vocabulary.
#:
#: Onboarding's own inference (``engine.onboarding_agent.onboarding_context``)
#: speaks a longer dialect — "equity_release_mortgage" — while the BSR, the
#: stratification catalogue and ``config/asset`` speak the short form. Rather
#: than teach every consumer both, the boundary normalises ONCE, here, on the
#: way in. An unrecognised value passes through lowercased: this table resolves
#: known synonyms, it does not restrict what a client may declare.
_ASSET_CLASS_SYNONYMS = {
    "equity_release_mortgage": "equity_release",
    "lifetime_mortgage": "equity_release",
    "erm": "equity_release",
    "residential_mortgage_loan": "residential_mortgage",
    "rmbs": "residential_mortgage",
    "sme_loan": "sme",
    "bridging": "bridge",
    "bridging_finance": "bridge",
    "bridge_loan": "bridge",
    "equipment_finance": "equipment_leasing",
    "asset_finance": "equipment_leasing",
    "cre": "commercial_real_estate",
}


def normalise_asset_class(value: Any) -> Optional[str]:
    """One governed asset-class vocabulary, resolved at the boundary."""
    text = str(value or "").strip().lower().replace("-", "_").replace(" ", "_")
    if not text:
        return None
    return _ASSET_CLASS_SYNONYMS.get(text, text)


def registry_path() -> Optional[str]:
    """The configured registry location, or ``None`` when none exists."""
    configured = os.environ.get(ENV_REGISTRY_PATH)
    if configured:
        return configured
    return str(DEFAULT_REGISTRY_PATH) if DEFAULT_REGISTRY_PATH.exists() else None


def _read_text(location: str) -> Optional[str]:
    """Read the registry file from disk or, for a ``blob://`` URI, via storage."""
    if "://" in location:
        try:
            from apps.blob_trigger_app.storage import open_storage
            storage = open_storage()
            if not storage.exists(location):
                return None
            return storage.read_text(location)
        except Exception as exc:  # noqa: BLE001 - metadata must never break MI
            logger.info("portfolio registry unavailable at %s: %s", location, exc)
            return None
    path = Path(location)
    if not path.exists():
        return None
    try:
        return path.read_text(encoding="utf-8")
    except OSError as exc:
        logger.info("portfolio registry unreadable at %s: %s", location, exc)
        return None


def _entries_for_client(doc: Mapping[str, Any],
                        client_id: Optional[str]) -> List[Mapping[str, Any]]:
    """Portfolio entries for ``client_id``, supporting both file shapes.

    Flat (single-tenant deployments)::

        portfolios:
          - source_portfolio_id: acquired_001
            runoff_profile_id: seller_a_2026

    Per-client (a shared registry)::

        clients:
          ERE:
            portfolios: [...]
    """
    clients = doc.get("clients")
    if isinstance(clients, Mapping):
        entries: List[Mapping[str, Any]] = []
        for cid, block in clients.items():
            if client_id and str(cid).strip().lower() != str(client_id).strip().lower():
                continue
            if isinstance(block, Mapping):
                entries.extend(block.get("portfolios") or [])
        if entries or client_id:
            return entries
    return list(doc.get("portfolios") or [])


def load_portfolio_metadata(client_id: Optional[str] = None,
                            *, location: Optional[str] = None
                            ) -> Dict[str, Dict[str, Any]]:
    """The governed metadata overlay, keyed by ``source_portfolio_id``.

    Returns ``{}`` when no registry is configured — the supported and current
    state for ERE. Never raises: a malformed registry degrades to "no overlay"
    with a log line rather than taking MI down.
    """
    loc = location or registry_path()
    if not loc:
        return {}
    raw = _read_text(loc)
    if not raw or not raw.strip():
        return {}
    try:
        import yaml
        doc = yaml.safe_load(raw) or {}
    except Exception as exc:  # noqa: BLE001
        logger.warning("portfolio registry parse failed (%s): %s", loc, exc)
        return {}
    if not isinstance(doc, Mapping):
        return {}

    out: Dict[str, Dict[str, Any]] = {}
    for entry in _entries_for_client(doc, client_id):
        if not isinstance(entry, Mapping):
            continue
        pid = entry.get("source_portfolio_id") or entry.get("portfolio_id")
        pid = str(pid).strip() if pid is not None else ""
        if not pid:
            continue
        allowed = {k: v for k, v in entry.items() if k in _ALLOWED_KEYS}
        if "asset_class" in allowed:
            normalised = normalise_asset_class(allowed["asset_class"])
            if normalised:
                allowed["asset_class"] = normalised
            else:
                allowed.pop("asset_class")
        out[pid.lower()] = allowed
    return out


# --------------------------------------------------------------------------- #
# Operational evidence: which portfolios actually register pipeline data
# --------------------------------------------------------------------------- #
def pipeline_registered_portfolios(client_id: Optional[str] = None) -> Optional[List[str]]:
    """Portfolios with an active ``dataset: pipeline`` source registration.

    This is the operational counterpart to the declared metadata: a portfolio
    only originates *in practice* once a pipeline source is registered for it.
    Returns ``None`` when no source registry is reachable, which the capability
    resolver reads as "fall back to declared metadata" rather than "nothing
    originates" — so an unreachable registry can never silently disable Pipeline
    for a direct book that has always had it.
    """
    uri = os.environ.get("TRAKT_SOURCE_REGISTRY_URI")
    if not uri:
        return None
    try:
        from apps.blob_trigger_app.source_registry import SourceRegistry, STATUS_RETIRED
        from apps.blob_trigger_app.storage import open_storage
        registry = SourceRegistry(uri, storage=open_storage())
        out: List[str] = []
        for record in registry.records():
            if str(getattr(record, "dataset", "")).strip().lower() != "pipeline":
                continue
            if str(getattr(record, "status", "")).strip().lower() == STATUS_RETIRED:
                continue
            if client_id and str(getattr(record, "client_id", "")).strip().lower() \
                    != str(client_id).strip().lower():
                continue
            pid = str(getattr(record, "source_portfolio_id", "") or "").strip()
            if pid and pid not in out:
                out.append(pid)
        return out
    except Exception as exc:  # noqa: BLE001 - discovery must never break MI
        logger.info("source-registry pipeline discovery unavailable: %s", exc)
        return None
