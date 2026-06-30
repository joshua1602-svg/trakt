"""apps.blob_trigger_app.azure_io — Azure Blob Storage I/O helpers.

Thin wrappers around ``azure-storage-blob`` used by the root Event Grid handler
to fetch a blob, read the ``_READY.json`` marker, and list/download the files of
a now-complete reporting pack. Kept separate from the router so the routing core
stays Azure-free and unit-testable. ``azure`` is imported lazily so importing
this module never requires the SDK until a function is actually called.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional


def _container_client(container: str):
    from azure.storage.blob import BlobServiceClient  # type: ignore
    conn = os.environ["TRAKT_BLOB_CONNECTION"]
    return BlobServiceClient.from_connection_string(conn).get_container_client(container)


def folder_prefix(blob_path: str) -> str:
    """Folder (pack) prefix for a blob path: everything up to the last '/'."""
    return blob_path.rsplit("/", 1)[0] + "/" if "/" in blob_path else ""


def list_pack_files(container: str, prefix: str, *, marker: str) -> List[str]:
    """Return the data-file blob names (basename) under ``prefix`` (marker excluded)."""
    cc = _container_client(container)
    names: List[str] = []
    for b in cc.list_blobs(name_starts_with=prefix):
        base = b.name.rsplit("/", 1)[-1]
        if base == marker:
            continue
        names.append(base)
    return names


def download_pack(container: str, prefix: str, dest: Path, *, marker: str) -> List[Path]:
    """Download every non-marker blob under ``prefix`` into ``dest``."""
    cc = _container_client(container)
    dest.mkdir(parents=True, exist_ok=True)
    out: List[Path] = []
    for b in cc.list_blobs(name_starts_with=prefix):
        base = b.name.rsplit("/", 1)[-1]
        if base == marker:
            continue
        target = dest / base
        target.write_bytes(cc.download_blob(b.name).readall())
        out.append(target)
    return out


def read_marker_metadata(container: str, blob_path: str) -> Optional[Dict[str, Any]]:
    """Download and parse a ``_READY.json`` marker; ``None`` if absent/unparseable.

    A non-JSON marker (e.g. a legacy empty ``_READY`` sentinel) returns ``None``
    rather than failing — processing then proceeds with no metadata overrides.
    """
    try:
        cc = _container_client(container)
        raw = cc.download_blob(blob_path).readall()
    except Exception:  # noqa: BLE001 — absent/unreadable marker → no metadata
        return None
    if not raw:
        return None
    try:
        data = json.loads(raw)
    except Exception:  # noqa: BLE001
        return None
    return data if isinstance(data, dict) else None


def download_single(container: str, blob_path: str, dest: Path) -> Path:
    """Download one blob to ``dest`` (used for single-file deterministic packs)."""
    cc = _container_client(container)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(cc.download_blob(blob_path).readall())
    return dest
