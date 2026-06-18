"""analytics_lib.config_loader — small, pure YAML config helpers.

Phase 1 shared analytics library. This module only *reads* YAML configuration
(e.g. ``config/mi/buckets.yaml`` from Phase 0B). It performs no file-system
writes, no network access, and imports nothing from the legacy ``analytics/``
Streamlit app, Azure, or any LLM client.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import yaml

# Repo root = parent of the ``analytics_lib`` package directory.
REPO_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_BUCKETS_PATH = REPO_ROOT / "config" / "mi" / "buckets.yaml"


def load_yaml(path: Path | str) -> Dict[str, Any]:
    """Load a YAML file into a plain dict (``{}`` for an empty document)."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"config file not found: {path}")
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    return data or {}


def load_bucket_config(path: Optional[Path | str] = None) -> Dict[str, Any]:
    """Load and lightly validate ``config/mi/buckets.yaml``.

    Returns the ``buckets`` mapping ``{bucket_key: spec}``. The structure is
    validated only enough to fail fast on a malformed file; per-bucket edge /
    label problems are reported as structured issues at *materialisation* time
    (see :mod:`analytics_lib.buckets`), not here.
    """
    cfg = load_yaml(path or DEFAULT_BUCKETS_PATH)
    buckets = cfg.get("buckets")
    if not isinstance(buckets, dict) or not buckets:
        raise ValueError(
            "bucket config must contain a non-empty 'buckets' mapping"
        )
    return buckets
