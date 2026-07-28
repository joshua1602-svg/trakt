"""mi_agent.period_change.units — what kind of number a governed field is.

§11 requires movements to be expressed in the right unit: a currency amount
moves by an amount and a percentage, a rate moves by percentage points (and
basis points), a share moves by percentage points, and a date or a category does
not move by a percentage at all.

The Business Semantics Registry deliberately does not carry a unit — it carries
*meaning*. The unit is already governed elsewhere, in two registries this module
reads in a fixed precedence:

1. ``mi_agent/mi_semantics_field_registry.yaml`` — the curated MI layer, whose
   ``format`` (currency | percent | integer | decimal | boolean | date | string)
   is the same fact the React adapters and the dataset contract already use;
2. ``config/system/fields_registry.yaml`` — the canonical source registry, whose
   ``format`` (decimal | Y/N | list | date | integer | percentage | …) covers
   every BSR field, including the ones outside the curated MI layer;
3. the BSR's own ``default_aggregation`` — a last resort that distinguishes a
   summed amount from a share, never a guess from the field's NAME.

Nothing here parses a field name. A field absent from both registries and
carrying no decisive aggregation resolves to ``ratio``, which suppresses the
percentage-of-start expression rather than inventing one.

Percent storage scale
---------------------
A percent column may be stored as a fraction (0.51) or as points (51). That is
decided once in the platform, by
``mi_agent.mi_dataset_profile.percent_storage_scale``; this module reuses it so a
period-change percentage-point movement matches every other percentage in the
product.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping, Optional

import yaml

from ..business_semantics import (
    AGG_AVERAGE,
    AGG_DISTRIBUTION,
    AGG_SHARE,
    AGG_SUM,
    AGG_WEIGHTED_AVERAGE,
    BusinessSemanticsEntry,
)
from .models import (
    UNIT_COUNT,
    UNIT_CURRENCY,
    UNIT_NOT_APPLICABLE,
    UNIT_PERCENTAGE_POINT,
    UNIT_RATIO,
)

_REPO_ROOT = Path(__file__).resolve().parents[2]
MI_SEMANTICS_PATH = _REPO_ROOT / "mi_agent" / "mi_semantics_field_registry.yaml"
FIELDS_REGISTRY_PATH = _REPO_ROOT / "config" / "system" / "fields_registry.yaml"

SOURCE_MI_SEMANTICS = "mi_semantics_field_registry"
SOURCE_FIELDS_REGISTRY = "fields_registry"
SOURCE_AGGREGATION_DEFAULT = "bsr_default_aggregation"

#: MI semantic registry ``format`` → movement unit.
_MI_FORMAT_UNIT: Mapping[str, str] = {
    "currency": UNIT_CURRENCY,
    "percent": UNIT_PERCENTAGE_POINT,
    "integer": UNIT_COUNT,
    "decimal": UNIT_RATIO,
    "boolean": UNIT_COUNT,
    "date": UNIT_NOT_APPLICABLE,
    "string": UNIT_NOT_APPLICABLE,
}

#: Canonical registry ``format`` → movement unit. ``decimal`` is genuinely
#: ambiguous in the canonical registry (it covers both balances and ratios), so
#: it defers to the BSR aggregation below rather than asserting a unit.
_CANONICAL_FORMAT_UNIT: Mapping[str, Optional[str]] = {
    "percentage": UNIT_PERCENTAGE_POINT,
    "integer": UNIT_COUNT,
    "year": UNIT_COUNT,
    "date": UNIT_NOT_APPLICABLE,
    "list": UNIT_NOT_APPLICABLE,
    "string": UNIT_NOT_APPLICABLE,
    "currency_code": UNIT_NOT_APPLICABLE,
    "Y/N": None,
    "decimal": None,
}

#: Aggregation → unit, the final fallback.
_AGGREGATION_UNIT: Mapping[str, str] = {
    AGG_SUM: UNIT_CURRENCY,
    AGG_SHARE: UNIT_PERCENTAGE_POINT,
    AGG_WEIGHTED_AVERAGE: UNIT_RATIO,
    AGG_AVERAGE: UNIT_RATIO,
    AGG_DISTRIBUTION: UNIT_NOT_APPLICABLE,
}


@dataclass(frozen=True)
class FieldUnit:
    """The resolved movement unit for one governed field, and where it came from."""

    unit: str
    source: str
    registry_format: Optional[str] = None

    @property
    def is_point_unit(self) -> bool:
        return self.unit == UNIT_PERCENTAGE_POINT

    @property
    def supports_relative_change(self) -> bool:
        """True when "x% higher than the start value" is a sound statement.

        False for percentage points (a rate that moves 4% → 5% has risen by one
        point, and calling that "25% higher" invites misreading) and for
        non-numeric units.
        """
        return self.unit in (UNIT_CURRENCY, UNIT_COUNT, UNIT_RATIO)

    def to_dict(self) -> dict:
        return {"unit": self.unit, "source": self.source,
                "registry_format": self.registry_format}


def _read_yaml(path: Path) -> Mapping[str, Any]:
    if not path.exists():
        return {}
    with path.open(encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


@lru_cache(maxsize=2)
def _mi_formats() -> Mapping[str, str]:
    fields = _read_yaml(MI_SEMANTICS_PATH).get("fields") or {}
    return {name: str(spec.get("format") or "")
            for name, spec in fields.items() if isinstance(spec, Mapping)}


@lru_cache(maxsize=2)
def _canonical_formats() -> Mapping[str, str]:
    fields = _read_yaml(FIELDS_REGISTRY_PATH).get("fields") or {}
    return {name: str(spec.get("format") or "")
            for name, spec in fields.items() if isinstance(spec, Mapping)}


@lru_cache(maxsize=2)
def _mi_tiers() -> Mapping[str, str]:
    """``{canonical_field: core|extended}`` from the curated MI layer.

    Used by the field-selection policy (§10) as the "existing MI core/extended
    status where available" input, so overview selection prefers fields the
    product already treats as core MI.
    """
    fields = _read_yaml(MI_SEMANTICS_PATH).get("fields") or {}
    return {name: str(spec.get("mi_tier") or "")
            for name, spec in fields.items() if isinstance(spec, Mapping)}


def mi_tier(field: str) -> Optional[str]:
    """``core`` | ``extended`` | ``None`` when the field is outside the MI layer."""
    return _mi_tiers().get(field) or None


def reset_cache() -> None:
    """Drop the registry caches. For tests that point at temporary registries."""
    _mi_formats.cache_clear()
    _canonical_formats.cache_clear()
    _mi_tiers.cache_clear()


def resolve_unit(entry: BusinessSemanticsEntry) -> FieldUnit:
    """The movement unit for one BSR entry, by governed registry precedence."""
    mi_format = _mi_formats().get(entry.field)
    if mi_format and mi_format in _MI_FORMAT_UNIT:
        unit = _MI_FORMAT_UNIT[mi_format]
        # A flag summarised as a share is a percentage-point movement, whatever
        # its storage format says.
        if entry.default_aggregation == AGG_SHARE:
            unit = UNIT_PERCENTAGE_POINT
        elif entry.default_aggregation == AGG_DISTRIBUTION:
            unit = UNIT_NOT_APPLICABLE
        return FieldUnit(unit, SOURCE_MI_SEMANTICS, mi_format)

    if entry.default_aggregation == AGG_SHARE:
        return FieldUnit(UNIT_PERCENTAGE_POINT, SOURCE_AGGREGATION_DEFAULT,
                         _canonical_formats().get(entry.field) or None)
    if entry.default_aggregation == AGG_DISTRIBUTION:
        return FieldUnit(UNIT_NOT_APPLICABLE, SOURCE_AGGREGATION_DEFAULT,
                         _canonical_formats().get(entry.field) or None)

    canonical_format = _canonical_formats().get(entry.field)
    if canonical_format in _CANONICAL_FORMAT_UNIT:
        unit = _CANONICAL_FORMAT_UNIT[canonical_format]
        if unit is not None:
            return FieldUnit(unit, SOURCE_FIELDS_REGISTRY, canonical_format)

    unit = _AGGREGATION_UNIT.get(entry.default_aggregation, UNIT_RATIO)
    return FieldUnit(unit, SOURCE_AGGREGATION_DEFAULT, canonical_format)
