"""mi_agent_api/decks.py — investor PPTX deck discovery + resolution.

The blob-trigger orchestration publishes the generated investor deck to a durable
location (``apps.blob_trigger_app.pptx_stage.persist_investor_deck`` →
``processed/decks/{client}/latest|{period}/investor_pack.pptx``). This module lets
the MI API DISCOVER what decks exist (latest + dated periods) and RESOLVE one to a
local file to serve — without ever exposing a raw blob path to the UI.

Two source modes, resolved in order:
  1. ``MI_AGENT_DECK_ROOT`` — a local directory laid out as
     ``{client}/latest/investor_pack.pptx`` and ``{client}/{period}/…`` (dev/tests);
  2. otherwise the durable blob store (``open_storage`` + ``Layout`` deck prefix),
     used in Azure where the orchestration publishes decks.

Everything degrades gracefully: no configured source, no decks, or an unreadable
file all yield an empty listing / ``None`` — never an exception to the caller.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger("trakt.mi_agent.decks")

DECK_NAME = "investor_pack.pptx"
POINTER_NAME = "latest_investor_pack.json"
_LATEST = "latest"


def _scratch() -> Path:
    return Path(os.environ.get("MI_AGENT_SCRATCH", "/tmp/trakt/mi_platform")) / "decks"


# --------------------------------------------------------------------------- #
# Local-root mode (dev / tests)
# --------------------------------------------------------------------------- #
def _local_root() -> Optional[Path]:
    root = os.environ.get("MI_AGENT_DECK_ROOT")
    if not root:
        return None
    p = Path(root)
    return p if p.exists() else None


def _local_periods(root: Path, client_id: str) -> List[str]:
    base = root / client_id
    if not base.exists():
        return []
    periods = []
    for child in sorted(base.iterdir()):
        if child.is_dir() and child.name != _LATEST and (child / DECK_NAME).exists():
            periods.append(child.name)
    return periods


def _local_pointer(root: Path, client_id: str) -> Dict[str, Any]:
    ptr = root / client_id / _LATEST / POINTER_NAME
    if ptr.exists():
        try:
            return json.loads(ptr.read_text(encoding="utf-8")) or {}
        except Exception:  # noqa: BLE001
            return {}
    return {}


# --------------------------------------------------------------------------- #
# Blob mode (Azure durable store)
# --------------------------------------------------------------------------- #
def _blob_ctx():
    """(storage, layout) when a durable blob deck store is usable, else None."""
    try:
        from apps.blob_trigger_app.storage import open_storage
        from apps.blob_trigger_app.layout import Layout
        return open_storage(), Layout.from_env()
    except Exception as exc:  # noqa: BLE001
        logger.warning("deck blob context unavailable: %s", exc)
        return None


def _blob_periods(storage, layout, client_id: str) -> List[str]:
    prefix = layout.deck_prefix(client_id)
    try:
        uris = storage.list(prefix)
    except Exception as exc:  # noqa: BLE001
        logger.warning("deck list failed for %s: %s", prefix, exc)
        return []
    periods = []
    for uri in uris:
        if not uri.endswith(DECK_NAME):
            continue
        # …/decks/{client}/{period}/investor_pack.pptx → take {period}
        parts = uri.rstrip("/").split("/")
        if len(parts) >= 2 and parts[-1] == DECK_NAME:
            period = parts[-2]
            if period != _LATEST:
                periods.append(period)
    return sorted(set(periods))


def _blob_pointer(storage, layout, client_id: str) -> Dict[str, Any]:
    uri = layout.deck_latest_pointer_uri(client_id)
    try:
        if storage.exists(uri):
            return json.loads(storage.read_text(uri)) or {}
    except Exception:  # noqa: BLE001
        return {}
    return {}


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #
def list_decks(client_id: str) -> Dict[str, Any]:
    """Discover the decks available for a client, UI-safe (no raw paths).

    Returns ``{available, latest: {period, generatedAt} | null, decks: [{period}],
    client_id}``. ``decks`` are dated reporting-period decks (newest first); the
    ``latest`` pointer is the current deck the download action defaults to.
    """
    client_id = client_id or ""
    latest: Optional[Dict[str, Any]] = None
    periods: List[str] = []

    root = _local_root()
    if root is not None:
        periods = _local_periods(root, client_id)
        has_latest = (root / client_id / _LATEST / DECK_NAME).exists()
        ptr = _local_pointer(root, client_id)
        if has_latest:
            latest = {"period": ptr.get("reporting_period"),
                      "generatedAt": ptr.get("generated_at")}
    else:
        ctx = _blob_ctx()
        if ctx is not None:
            storage, layout = ctx
            periods = _blob_periods(storage, layout, client_id)
            try:
                has_latest = storage.exists(layout.deck_latest_uri(client_id))
            except Exception:  # noqa: BLE001
                has_latest = False
            ptr = _blob_pointer(storage, layout, client_id)
            if has_latest:
                latest = {"period": ptr.get("reporting_period"),
                          "generatedAt": ptr.get("generated_at")}

    decks = [{"period": p} for p in sorted(periods, reverse=True)]
    available = latest is not None or bool(decks)
    return {"available": available, "latest": latest, "decks": decks,
            "client_id": client_id}


def resolve_deck_local(client_id: str, period: Optional[str] = None
                       ) -> Optional[Tuple[Path, str]]:
    """Resolve a deck to a LOCAL file to serve, plus a friendly download filename.

    ``period`` selects a dated deck; ``None`` (or ``"latest"``) resolves the
    latest pointer. Returns ``(local_path, download_name)`` or ``None`` when the
    requested deck does not exist / cannot be fetched. Never raises.
    """
    client_id = client_id or ""
    want_latest = period in (None, "", _LATEST)
    friendly_period = (period if not want_latest else "latest")
    download_name = f"{client_id or 'client'}_investor_deck_{friendly_period}.pptx"

    root = _local_root()
    if root is not None:
        rel = _LATEST if want_latest else str(period)
        candidate = root / client_id / rel / DECK_NAME
        return (candidate, download_name) if candidate.exists() else None

    ctx = _blob_ctx()
    if ctx is None:
        return None
    storage, layout = ctx
    uri = (layout.deck_latest_uri(client_id) if want_latest
           else layout.deck_period_uri(client_id, str(period)))
    try:
        if not storage.exists(uri):
            return None
        # Prefer a local mirror path (filesystem storage); else download to scratch.
        local = storage._local_path(uri) if hasattr(storage, "_local_path") else None
        if local is not None and Path(str(local)).exists():
            return (Path(str(local)), download_name)
        dest = _scratch() / client_id / f"{friendly_period}_{DECK_NAME}"
        dest.parent.mkdir(parents=True, exist_ok=True)
        path = storage.download_file(uri, dest)
        return (Path(str(path)), download_name)
    except Exception as exc:  # noqa: BLE001
        logger.warning("deck resolve failed for %s: %s", uri, exc)
        return None
