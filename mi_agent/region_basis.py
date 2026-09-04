"""Which region a figure was measured on, and how much of the book resolved.

THE PROBLEM. Three field families wear the word "Region", and they are not
interchangeable:

    reporting   canonical_region_reporting · canonical_region_detail ·
                collateral_geography          — what a reader calls "Region"
    NUTS3       geographic_region_obligor · geographic_region_collateral
    ITL3        geographic_region_*_itl3      — resolves only where a postcode does

`engine.region_taxonomy` already records, per row, how each raw value was
mapped (`region_mapping_method`: exact / synonym / unresolved / absent) so that
"a consolidated answer can always explain where each category came from". Until
this module existed, nothing read that column. An answer grouped by region was
computed from the rows that carried a governed value and said nothing about the
ones that did not, and said nothing about which of the three families produced
it.

This module answers two questions about an executed query and nothing else:

    which region field did it measure, and at which level?
    what share of the frame carries a governed value at that level?

It decides nothing, changes no figure, and resolves no value. It is a reader.

`tests/test_region_topology.py` pins these three tuples against the five
surfaces that each keep their own list, so a field added to one and not the
others fails loudly rather than quietly joining an alias pool.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence

from engine.region_taxonomy import (
    FIELD_METHOD, METHOD_ABSENT, METHOD_EXACT, METHOD_SYNONYM,
    METHOD_UNRESOLVED)

#: The three granularities. See the module docstring — and the topology test,
#: which is where a new region field has to be classified before it can be used.
LEVEL_REPORTING = "reporting"
LEVEL_NUTS3 = "nuts3"
LEVEL_ITL3 = "itl3"

REPORTING_FIELDS = ("canonical_region_reporting", "canonical_region_detail",
                    "collateral_geography")
NUTS3_FIELDS = ("geographic_region_obligor", "geographic_region_collateral")
ITL3_FIELDS = ("geographic_region_obligor_itl3",
               "geographic_region_collateral_itl3")

#: Human words for the level, used in the disclosure sentence. A reader who has
#: never heard of NUTS3 still has to be able to tell two answers apart.
LEVEL_LABELS = {
    LEVEL_REPORTING: "harmonised reporting region",
    LEVEL_NUTS3: "NUTS3 region",
    LEVEL_ITL3: "ITL3 sub-region",
}

#: The methods the provenance column can record, in report order.
METHODS: Sequence[str] = (METHOD_EXACT, METHOD_SYNONYM, METHOD_UNRESOLVED,
                          METHOD_ABSENT)

#: Only these two columns are the ones `region_mapping_method` describes; the
#: raw source columns are its INPUT, so a per-method breakdown alongside them
#: would be describing a different thing from the field that was measured.
_METHOD_DESCRIBES = ("canonical_region_reporting", "canonical_region_detail")

_LEVEL_BY_FIELD: Dict[str, str] = {}
for _f in REPORTING_FIELDS:
    _LEVEL_BY_FIELD[_f] = LEVEL_REPORTING
for _f in NUTS3_FIELDS:
    _LEVEL_BY_FIELD[_f] = LEVEL_NUTS3
for _f in ITL3_FIELDS:
    _LEVEL_BY_FIELD[_f] = LEVEL_ITL3
del _f


def level_of(field: Optional[str]) -> Optional[str]:
    """Which family ``field`` belongs to, or None if it is not a region field."""
    return _LEVEL_BY_FIELD.get(field or "")


def region_fields(keys: Iterable[Optional[str]]) -> List[str]:
    """The region fields among ``keys``, in the order given, deduplicated."""
    seen: List[str] = []
    for key in keys or ():
        if key and level_of(key) and key not in seen:
            seen.append(key)
    return seen


@dataclass(frozen=True)
class RegionBasis:
    """The region a figure was measured on, and the coverage behind it.

    ``rows``/``resolved``/``share`` are None when the frame was not available:
    an unknown coverage is stated as unknown, never as complete.
    """

    field: str
    level: str
    rows: Optional[int] = None
    resolved: Optional[int] = None
    methods: Optional[Dict[str, int]] = None

    @property
    def share(self) -> Optional[float]:
        if not self.rows or self.resolved is None:
            return None
        return round(self.resolved / self.rows, 4)

    @property
    def partial(self) -> bool:
        share = self.share
        return share is not None and share < 1.0

    def label(self) -> str:
        return LEVEL_LABELS.get(self.level, self.level)

    def disclosure(self) -> Optional[str]:
        """The sentence for a PARTIALLY covered basis. Full coverage is silent:
        a caveat printed on every answer is a caveat nobody reads."""
        if not self.partial:
            return None
        return (f"Region basis: {self.label()} — {self.resolved:,} of "
                f"{self.rows:,} rows carry a governed region "
                f"({self.share * 100:.1f}%); the rest are excluded from the "
                "regional breakdown")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "field": self.field,
            "level": self.level,
            "levelLabel": self.label(),
            "rows": self.rows,
            "resolved": self.resolved,
            "share": self.share,
            "methods": dict(self.methods or {}),
        }


def basis_for(fields: Iterable[Optional[str]], frame=None) -> Optional[RegionBasis]:
    """The basis for a query that measured ``fields``, or None if none is a
    region field. The FIRST region field wins — a query grouped by region and
    filtered by region is measured at one level, and the group is the axis the
    reader is looking at."""
    candidates = region_fields(fields)
    if not candidates:
        return None
    field = candidates[0]
    rows, resolved, methods = _coverage(field, frame)
    return RegionBasis(field=field, level=level_of(field) or "",
                       rows=rows, resolved=resolved, methods=methods)


def _coverage(field: str, frame):
    """``(rows, resolved, methods)`` for ``field`` over ``frame``.

    Resolution is read from the field itself — a row with no governed value at
    that level cannot appear in a breakdown of it, whatever the reason. The
    per-method split is added only for the two columns the provenance column
    actually describes.
    """
    if frame is None:
        return None, None, None
    try:
        rows = int(len(frame.index))
        if not rows or field not in frame.columns:
            return None, None, None
        column = frame[field]
        text = column.astype(str).str.strip().str.lower()
        present = column.notna() & ~text.isin(["", "nan", "none", "nat",
                                               "<na>", "null"])
        resolved = int(present.sum())
        methods = None
        if field in _METHOD_DESCRIBES and FIELD_METHOD in frame.columns:
            counts = frame[FIELD_METHOD].astype(str)
            methods = {m: int((counts == m).sum()) for m in METHODS}
            methods = {m: n for m, n in methods.items() if n}
        return rows, resolved, methods
    except Exception:                                        # noqa: BLE001
        # Coverage is a disclosure. It must never be the reason an answer that
        # would otherwise stand does not ship.
        return None, None, None
