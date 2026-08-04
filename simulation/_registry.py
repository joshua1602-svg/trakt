"""simulation._registry — read-only access to the PRODUCTION field registry.

The dialects format each value according to the ``format`` the production
registry declares for that canonical field, so a generated source exercises the
real typing rules in ``canonical_transform.apply_types`` rather than a
simulator-local idea of what a date or a decimal looks like.

Read-only by construction: nothing in this package writes to the registry.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Dict, List

import yaml

from .models import PORTFOLIO_TYPE_BY_ASSET

REPO_ROOT = Path(__file__).resolve().parents[1]
REGISTRY_PATH = REPO_ROOT / "config" / "system" / "fields_registry.yaml"


@lru_cache(maxsize=1)
def _registry() -> dict:
    return yaml.safe_load(REGISTRY_PATH.read_text(encoding="utf-8")) or {}


@lru_cache(maxsize=8)
def selectable_fields(asset_class: str) -> frozenset:
    """The canonical fields production selects for this asset class.

    Mirrors ``semantic_alignment.select_registry_fields``: everything typed
    ``common`` plus everything typed with the asset class's own
    ``--portfolio-type``.
    """
    portfolio_type = PORTFOLIO_TYPE_BY_ASSET[asset_class]
    out = set()
    for name, meta in (_registry().get("fields") or {}).items():
        ptype = str((meta or {}).get("portfolio_type", "")).strip().lower()
        if ptype in ("common", portfolio_type):
            out.add(name)
    return frozenset(out)


@lru_cache(maxsize=8)
def field_formats(asset_class: str) -> Dict[str, str]:
    """``{canonical_field: declared format}`` for the selectable fields."""
    fields = selectable_fields(asset_class)
    return {name: str((meta or {}).get("format") or "")
            for name, meta in (_registry().get("fields") or {}).items()
            if name in fields}


@lru_cache(maxsize=1)
def core_fields() -> frozenset:
    return frozenset(
        name for name, meta in (_registry().get("fields") or {}).items()
        if (meta or {}).get("core_canonical") is True)


def unsupported_fields(asset_class: str, fields) -> List[str]:
    """Fields a profile wants to emit that production would not select.

    Used by ``tests/test_simulation_asset_profiles.py`` to prove no asset class
    is quietly relying on a canonical field the pipeline will discard.
    """
    selectable = selectable_fields(asset_class)
    return [f for f in fields if f not in selectable]


__all__ = ["REGISTRY_PATH", "core_fields", "field_formats", "selectable_fields",
           "unsupported_fields"]
