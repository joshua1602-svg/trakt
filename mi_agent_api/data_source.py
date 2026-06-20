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


def _materialise_mi_buckets(df: pd.DataFrame) -> pd.DataFrame:
    """Add registry-named bucket dimensions using the EXISTING bucketing engine.

    The MI executor reuses pre-materialised bucket columns (age_bucket,
    ltv_bucket, ticket_bucket, ...) and deliberately does not create them. The
    Streamlit dashboard runs this step via analytics/mi_prep; the API must do the
    equivalent so the same registry/catalogue dimensions resolve here too.

    We reuse the canonical source of truth — ``analytics_lib.buckets`` with
    ``config/mi/buckets.yaml`` — and write columns under their registry semantic
    names (``target="semantic_field"``: borrower_age_bucket→age_bucket,
    balance_band→ticket_bucket, ...). No bucket logic is duplicated here.
    """
    try:
        from analytics_lib.buckets import load_bucket_config, materialise_buckets

        out, _issues, _applied = materialise_buckets(
            df, load_bucket_config(), target="semantic_field"
        )
        return out
    except Exception:  # noqa: BLE001 - bucketing is additive; never block a query
        return df


@lru_cache(maxsize=1)
def get_dataframe() -> pd.DataFrame:
    """Load and cache the active canonical dataframe (with MI bucket dimensions)."""
    csv = _find_demo_csv()
    if csv is None:
        raise FileNotFoundError(
            "No canonical CSV found. Set MI_AGENT_DATA_CSV or add a "
            "synthetic_demo/**/*canonical_typed.csv file."
        )
    return _materialise_mi_buckets(pd.read_csv(csv))


def data_source_label() -> str:
    csv = _find_demo_csv()
    return csv.name if csv else "unavailable"
