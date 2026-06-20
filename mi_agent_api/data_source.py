"""Resolve the canonical dataframe and semantics registry for the API.

For v1 this serves a synthetic demo portfolio shipped in the repo. The CSV and
semantics paths can be overridden via environment variables so the same API can
later point at a real canonical snapshot without code changes:

    MI_AGENT_DATA_CSV   - path to a canonical_typed.csv
    MI_AGENT_SEMANTICS  - path to mi_semantics_field_registry.yaml
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Optional

import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_SEMANTICS = _REPO_ROOT / "mi_agent" / "mi_semantics_field_registry.yaml"


def _find_demo_csv() -> Optional[Path]:
    override = os.environ.get("MI_AGENT_DATA_CSV")
    if override:
        p = Path(override)
        return p if p.exists() else None
    candidates = sorted(_REPO_ROOT.glob("synthetic_demo/**/*canonical_typed.csv"))
    return candidates[0] if candidates else None


def semantics_path() -> Path:
    override = os.environ.get("MI_AGENT_SEMANTICS")
    return Path(override) if override else DEFAULT_SEMANTICS


@lru_cache(maxsize=1)
def get_dataframe() -> pd.DataFrame:
    """Load and cache the active canonical dataframe."""
    csv = _find_demo_csv()
    if csv is None:
        raise FileNotFoundError(
            "No canonical CSV found. Set MI_AGENT_DATA_CSV or add a "
            "synthetic_demo/**/*canonical_typed.csv file."
        )
    return pd.read_csv(csv)


def data_source_label() -> str:
    csv = _find_demo_csv()
    return csv.name if csv else "unavailable"
