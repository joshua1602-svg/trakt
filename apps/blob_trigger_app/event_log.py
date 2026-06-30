"""apps.blob_trigger_app.event_log — per-trigger event manifest."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict


def make_event_id(blob_path: str, created_at: str) -> str:
    h = hashlib.sha1(f"{blob_path}|{created_at}".encode("utf-8")).hexdigest()
    return "evt_" + h[:16]


def write_event_manifest(manifest: Dict[str, Any], out_dir: str | Path) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{manifest['event_id']}.json"
    path.write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
    return path
