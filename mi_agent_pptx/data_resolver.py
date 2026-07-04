"""mi_agent_pptx.data_resolver — normalise the typed tape into an analytical frame.

Turns a raw canonical typed tape (as loaded by :mod:`artifact_loader`) into a
deterministic analytical frame the deck can aggregate against, while delegating
*all* bucketing to the registry-authorised engine
(:mod:`analytics_lib.buckets`, which reads ``config/mi/buckets.yaml``). The only
transformations performed here are **field resolution** (which physical column
carries the balance / loan-id / cut-off date) and **scale normalisation**
(the registry declares LTV as a ``decimal_fraction``; some source tapes carry it
as whole-number points). These are canonicalisation steps, not economic
derivations — no LTV, forecast, or weighted metric is *computed* here.

The resolver never raises on missing columns: it records coverage so the deck's
metric/chart resolvers can fall back to branded placeholders and the appendix
can report exactly which fields were and were not available.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import numpy as np
import pandas as pd

from .registry_loader import REPO_ROOT, RegistryLoader

# ITL/NUTS code -> readable region name lookup (repo reference table). Raw ITL
# codes (e.g. "TLI3") read as amateur on an investor deck; the dashboard shows
# readable regions, so we enrich a display column when the tape carries codes.
_ITL_LOOKUP_PATH = REPO_ROOT / "uk_itl_master_lookup_v2.csv"
REGION_DISPLAY_COL = "region"

# Physical column candidates for the reserved analytical columns. Ordered by
# preference; the state/API layer standardises on ``current_outstanding_balance``
# while the ESMA typed tape carries ``current_principal_balance``.
BALANCE_CANDIDATES = [
    "current_outstanding_balance",
    "current_principal_balance",
    "outstanding_balance",
    "principal_balance",
]
LOAN_ID_CANDIDATES = [
    "unique_identifier",
    "underlying_exposure_identifier",
    "loan_identifier",
    "loan_id",
    "original_underlying_exposure_identifier",
]
CUTOFF_CANDIDATES = [
    "data_cut_off_date",
    "reporting_date",
    "cut_off_date",
]

# Canonical analytical column names written onto the resolved frame.
BALANCE_COL = "current_outstanding_balance"
LOAN_ID_COL = "loan_id"

# Fields the registry declares on a 0..1 fraction scale; normalise points->fraction.
_FRACTION_FIELDS = {
    "current_loan_to_value",
    "original_loan_to_value",
    "indexed_loan_to_value",
    "stressed_LTV",
}


@dataclass
class ResolvedData:
    """A normalised analytical frame plus its resolved reserved columns."""

    df: pd.DataFrame
    balance_col: Optional[str]
    loan_id_col: Optional[str]
    as_of_date: Optional[str]
    applied_buckets: Dict[str, Optional[str]] = field(default_factory=dict)
    present_fields: Set[str] = field(default_factory=set)
    issues: List[Dict[str, Any]] = field(default_factory=list)

    @property
    def loan_count(self) -> int:
        return int(len(self.df)) if self.df is not None else 0

    def has_field(self, key: str) -> bool:
        return key in self.present_fields

    def bucket_column(self, bucket_key: str) -> Optional[str]:
        """Resolved output column for a registry bucket, or ``None``."""
        return self.applied_buckets.get(bucket_key)


@lru_cache(maxsize=1)
def _itl_region_map() -> Dict[str, str]:
    """Map ITL3 codes -> readable ITL1 region name (cached; ``{}`` if absent)."""
    try:
        if not _ITL_LOOKUP_PATH.exists():
            return {}
        lk = pd.read_csv(_ITL_LOOKUP_PATH, low_memory=False)
        out: Dict[str, str] = {}
        for _, r in lk.iterrows():
            code = str(r.get("itl3_code", "")).strip()
            name = str(r.get("itl1_name", "")).strip()
            # Drop the redundant "(England)" qualifier for compact deck labels.
            name = name.replace(" (England)", "")
            if code and name:
                out[code] = name
        return out
    except Exception:  # pragma: no cover - defensive
        return {}


def _enrich_region(df: pd.DataFrame) -> None:
    """Add a readable ``region`` column from ITL codes, in place (best effort)."""
    src = None
    for c in ("geographic_region_obligor", "geographic_region_collateral",
              "collateral_geography"):
        if c in df.columns and df[c].notna().any():
            src = c
            break
    if src is None:
        return
    vals = df[src].astype("string")
    looks_itl = vals.dropna().str.match(r"^TL", na=False).mean() if vals.notna().any() else 0
    if looks_itl and looks_itl > 0.5:
        rmap = _itl_region_map()
        if rmap:
            df[REGION_DISPLAY_COL] = vals.map(lambda v: rmap.get(str(v), v)
                                              if pd.notna(v) else v)
            return
    # Already readable (or no map) — mirror the source into the display column.
    df[REGION_DISPLAY_COL] = df[src]


def _first_present(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def _normalise_fraction(series: pd.Series) -> pd.Series:
    """Normalise a ratio column to a 0..1 fraction (points -> fraction).

    Uses the same heuristic as the MI Agent funded-prep layer: if the non-null
    median looks like whole-number points (> 1.5), divide by 100. This is a
    scale canonicalisation the registry declares (``scale: decimal_fraction``),
    not an economic derivation.
    """
    num = pd.to_numeric(series, errors="coerce")
    valid = num.dropna()
    if valid.empty:
        return num
    if float(valid.median()) > 1.5:
        return num / 100.0
    return num


def resolve_data(
    tape: pd.DataFrame,
    registries: RegistryLoader,
    *,
    as_of_date: Optional[str] = None,
    materialise_buckets: Optional[List[str]] = None,
) -> ResolvedData:
    """Normalise *tape* into a :class:`ResolvedData` analytical frame.

    Parameters
    ----------
    tape:
        The raw canonical typed tape.
    registries:
        Registry accessor (for bucket config + field scale hints).
    as_of_date:
        Explicit as-of/cut-off date; when absent it is inferred from a
        ``data_cut_off_date``-style column.
    materialise_buckets:
        Optional subset of registry bucket keys to materialise (default: all
        buckets declared in ``config/mi/buckets.yaml``).
    """
    df = tape.copy()
    issues: List[Dict[str, Any]] = []

    # -- reserved column resolution --------------------------------------
    balance_src = _first_present(df, BALANCE_CANDIDATES)
    if balance_src is not None and balance_src != BALANCE_COL:
        # Alias to the canonical name without dropping the original.
        df[BALANCE_COL] = pd.to_numeric(df[balance_src], errors="coerce")
    elif balance_src == BALANCE_COL:
        df[BALANCE_COL] = pd.to_numeric(df[BALANCE_COL], errors="coerce")
    else:
        issues.append({
            "code": "missing_balance_field",
            "severity": "error",
            "message": "No balance column found (current_outstanding_balance / "
                       "current_principal_balance).",
        })

    loan_id_src = _first_present(df, LOAN_ID_CANDIDATES)
    if loan_id_src is not None:
        df[LOAN_ID_COL] = df[loan_id_src]

    # -- as-of date ------------------------------------------------------
    resolved_as_of = as_of_date
    if not resolved_as_of:
        cutoff_col = _first_present(df, CUTOFF_CANDIDATES)
        if cutoff_col is not None:
            vals = df[cutoff_col].dropna().astype(str)
            if not vals.empty:
                resolved_as_of = str(vals.mode().iloc[0])

    # -- scale normalisation of fraction fields --------------------------
    for fkey in _FRACTION_FIELDS:
        if fkey in df.columns:
            df[fkey] = _normalise_fraction(df[fkey])

    # -- readable region label (ITL/NUTS code -> region name) ------------
    _enrich_region(df)

    # -- present-field coverage (canonical semantic keys) ----------------
    present: Set[str] = set()
    for key in registries.fields.keys():
        spec = registries.field_spec(key)
        if not spec:
            continue
        col = spec.canonical_field
        if col in df.columns and df[col].notna().any():
            present.add(key)
    # Always record the reserved balance as present when resolved.
    if balance_src is not None:
        present.add("current_outstanding_balance")

    # -- registry-authorised bucket materialisation ----------------------
    applied: Dict[str, Optional[str]] = {}
    try:
        from analytics_lib.buckets import materialise_buckets as _mb

        bucket_cfg = registries.buckets or None
        df, bucket_issues, applied = _mb(
            df, config=bucket_cfg, buckets=materialise_buckets
        )
        issues.extend(bucket_issues)
    except Exception as exc:  # pragma: no cover - analytics_lib always present
        issues.append({
            "code": "bucket_engine_unavailable",
            "severity": "warning",
            "message": f"Registry bucket engine unavailable: {exc}",
        })

    return ResolvedData(
        df=df,
        balance_col=BALANCE_COL if balance_src is not None else None,
        loan_id_col=LOAN_ID_COL if loan_id_src is not None else None,
        as_of_date=resolved_as_of,
        applied_buckets=applied,
        present_fields=present,
        issues=issues,
    )
