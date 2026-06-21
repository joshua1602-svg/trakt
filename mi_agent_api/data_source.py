"""Resolve the canonical dataframe and semantics registry for the API.

The same API can serve either a synthetic demo portfolio (default, shipped in the
repo) or a **promoted funded central lender tape** from an onboarding run — so the
React dashboard reflects real promoted output rather than demo data. Selection is
environment-driven (no code change), resolved in priority order:

    1. MI_AGENT_CENTRAL_TAPE          - explicit path to an onboarding run's
                                        ``18_central_lender_tape.csv``.
    2. MI_AGENT_ONBOARDING_OUTPUT_ROOT + MI_AGENT_CLIENT_ID + MI_AGENT_RUN_ID
                                      - resolve the promoted central lender tape for
                                        a run generically by client_id / run_id.
    3. MI_AGENT_DATA_CSV              - explicit canonical_typed.csv.
    4. synthetic_demo/**/*canonical_typed.csv  (default demo).

    MI_AGENT_SEMANTICS               - path to mi_semantics_field_registry.yaml.

The promoted funded central lender tape already uses canonical field names
(current_outstanding_balance, current_valuation_amount, exposure_currency_
denomination, reporting_date, ...) that align with the MI semantic layer, so it is
served as-is; bucket dimensions are materialised additively (same as the demo).
Because the tape is period-scoped, the dashboard inherently shows the funded
universe (e.g. 33 / 73 loans), never the old 2,196-row universe and never
pipeline rows.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Dict, Optional, Tuple

import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_SEMANTICS = _REPO_ROOT / "mi_agent" / "mi_semantics_field_registry.yaml"

_CENTRAL_TAPE_NAME = "18_central_lender_tape.csv"

# Source kinds (surfaced on /health).
KIND_FUNDED_TAPE = "funded_central_lender_tape"
KIND_EXPLICIT_CSV = "explicit_csv"
KIND_SYNTHETIC_DEMO = "synthetic_demo"
KIND_UNAVAILABLE = "unavailable"


def _resolve_central_tape() -> Optional[Path]:
    """Resolve a promoted run's central lender tape generically by client/run."""
    explicit = os.environ.get("MI_AGENT_CENTRAL_TAPE")
    if explicit:
        p = Path(explicit)
        return p if p.exists() else None

    root = os.environ.get("MI_AGENT_ONBOARDING_OUTPUT_ROOT")
    if not root:
        return None
    root_path = Path(root)
    client_id = os.environ.get("MI_AGENT_CLIENT_ID", "")
    run_id = os.environ.get("MI_AGENT_RUN_ID", "")

    # Common run-folder layouts (most specific first); generic by client/run.
    candidates = [
        root_path / "central" / _CENTRAL_TAPE_NAME,
        root_path / client_id / run_id / "output" / "central" / _CENTRAL_TAPE_NAME,
        root_path / "runs" / client_id / "onboarding" / run_id / "central" / _CENTRAL_TAPE_NAME,
        root_path / run_id / "output" / "central" / _CENTRAL_TAPE_NAME,
    ]
    for c in candidates:
        if c.exists():
            return c
    # Fallback: glob for the run's tape anywhere under the root.
    if run_id:
        hits = sorted(root_path.glob(f"**/{run_id}/**/{_CENTRAL_TAPE_NAME}"))
        if hits:
            return hits[0]
    hits = sorted(root_path.glob(f"**/{_CENTRAL_TAPE_NAME}"))
    return hits[0] if hits else None


def _find_demo_csv() -> Optional[Path]:
    """Back-compat: the explicit/demo CSV (MI_AGENT_DATA_CSV or synthetic demo)."""
    override = os.environ.get("MI_AGENT_DATA_CSV")
    if override:
        p = Path(override)
        return p if p.exists() else None
    candidates = sorted(_REPO_ROOT.glob("synthetic_demo/**/*canonical_typed.csv"))
    return candidates[0] if candidates else None


def resolve_data_source() -> Tuple[Optional[Path], str]:
    """``(path, kind)`` for the active data source, by priority."""
    tape = _resolve_central_tape()
    if tape is not None:
        return tape, KIND_FUNDED_TAPE
    if os.environ.get("MI_AGENT_DATA_CSV"):
        p = _find_demo_csv()
        return p, (KIND_EXPLICIT_CSV if p else KIND_UNAVAILABLE)
    demo = _find_demo_csv()
    return demo, (KIND_SYNTHETIC_DEMO if demo else KIND_UNAVAILABLE)


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
    csv, _kind = resolve_data_source()
    if csv is None:
        raise FileNotFoundError(
            "No data source found. Set MI_AGENT_CENTRAL_TAPE (a promoted "
            "18_central_lender_tape.csv), MI_AGENT_ONBOARDING_OUTPUT_ROOT + "
            "MI_AGENT_CLIENT_ID + MI_AGENT_RUN_ID, MI_AGENT_DATA_CSV, or add a "
            "synthetic_demo/**/*canonical_typed.csv file."
        )
    return _materialise_mi_buckets(pd.read_csv(csv, low_memory=False))


def data_source_label() -> str:
    csv, _kind = resolve_data_source()
    return csv.name if csv else KIND_UNAVAILABLE


def data_source_kind() -> str:
    _csv, kind = resolve_data_source()
    return kind


def data_source_info() -> Dict[str, str]:
    """Full data-source descriptor for /health (kind, label, path, client/run)."""
    csv, kind = resolve_data_source()
    info = {
        "kind": kind,
        "label": csv.name if csv else KIND_UNAVAILABLE,
        "path": str(csv) if csv else "",
    }
    if kind == KIND_FUNDED_TAPE:
        info["client_id"] = os.environ.get("MI_AGENT_CLIENT_ID", "")
        info["run_id"] = os.environ.get("MI_AGENT_RUN_ID", "")
    return info
