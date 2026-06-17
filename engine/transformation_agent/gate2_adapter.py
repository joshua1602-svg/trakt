"""
gate2_adapter.py
================

Clean adapter that lets the existing, frozen ``engine.gate_2_transform``
canonical-transform logic consume the **Onboarding Agent central tape** and
the **handoff field contract**, instead of raw Gate 1 outputs.

The adapter ONLY reuses deterministic Gate 2 primitives:

  * :func:`load_registry`                       — field registry
  * :func:`apply_types`                         — date/number/bool/currency typing
  * :func:`resolve_canonical_enum_normalization`/:func:`apply_canonical_enum_normalization`
                                                — internal enum standardisation
                                                  (NOT regime / ESMA projection)
  * :func:`apply_config_defaults`               — config-driven default fill

It deliberately does NOT call the Gate 2 ``derive_fields`` geography / LTV /
reporting-date derivations, because those stray into projection-stage concerns.
Deterministic, contract-scoped defaulting and derivation are handled by the
Transformation Agent itself.

The adapter never re-runs raw Gate 1 canonicalisation, never discovers sources
and never fuzzy-matches columns — the central tape is already canonical.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd

from engine.gate_2_transform import canonical_transform as ct

ND_PATTERN = re.compile(r"^ND\d+$", re.IGNORECASE)


def load_registry_fields(registry_path: str) -> Dict[str, Any]:
    """Return the full ``fields`` mapping from the canonical field registry."""
    registry = ct.load_registry(Path(registry_path))
    return registry.get("fields", {}) or {}


def is_nd(value: Any) -> bool:
    if value is None:
        return False
    try:
        if pd.isna(value):
            return False
    except (TypeError, ValueError):
        pass
    return bool(ND_PATTERN.match(str(value).strip()))


def normalize_types(
    df: pd.DataFrame,
    fields_meta: Dict[str, Any],
    *,
    currency_synonyms: Optional[dict] = None,
    dayfirst: bool = True,
) -> Dict[str, Any]:
    """Deterministically normalise dates/numbers/booleans/currency via Gate 2.

    Returns the Gate 2 ``apply_types`` report (per-field parse failure metrics),
    so the Transformation Agent can surface uncontrolled parse failures as
    transformation issues rather than letting them pass silently.
    """
    return ct.apply_types(df, fields_meta, currency_synonyms, dayfirst=dayfirst)


def normalize_enums(df: pd.DataFrame, config: Optional[dict] = None) -> Dict[str, Any]:
    """Apply Gate 2 canonical enum normalisation (internal standardisation).

    This is NOT ESMA/regime code projection — it standardises known enum
    synonyms/casing to their canonical internal value. Unmapped values are left
    untouched (never guessed) and reported in the returned per-field summary.
    """
    norm_map = ct.resolve_canonical_enum_normalization(config or {})
    report = ct.apply_canonical_enum_normalization(df, norm_map)
    report["_normalization_map_fields"] = sorted(norm_map.keys())
    return report


def apply_defaults(df: pd.DataFrame, defaults: Dict[str, Any]) -> Dict[str, Any]:
    """Fill blanks with config-driven defaults using the Gate 2 primitive.

    ``defaults`` is a flat ``{canonical_field: value}`` map. Reuses Gate 2's
    :func:`apply_config_defaults` by wrapping the map under the ``defaults`` key
    it expects. Existing non-blank values are never overwritten.
    """
    report = ct.apply_config_defaults(df, {"defaults": defaults or {}})
    report["_defaulted_fields"] = sorted((defaults or {}).keys())
    return report
