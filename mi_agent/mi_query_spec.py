#!/usr/bin/env python3
"""
mi_query_spec.py

v1 MIQuerySpec — a small, serialisable description of an MI query / chart
request.  Implemented with the standard library `dataclasses` (no pydantic
dependency) so it is safe to import anywhere in the repo.

A spec is purely declarative: it names *semantic field keys* (keys in
mi_semantics_field_registry.yaml) and how they should be combined.  It does
NOT contain data and never executes anything.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

# Allowed enumerations (kept as module constants so the validator and parser
# can share them).
INTENTS = {"chart", "table", "summary"}
CHART_TYPES = {"bar", "line", "scatter", "bubble", "heatmap", "treemap", "none"}
AGGREGATIONS = {
    "sum", "avg", "weighted_avg", "count", "count_distinct",
    "median", "distribution", "loan_level", "balance_sum",
}
OUTPUT_FORMATS = {"chart", "table", "text", "chart_and_table"}

# Semantic-field-bearing fields on the spec (used by referenced_fields and the
# validator).  `dimensions`/`hierarchy` are list-valued and handled separately.
_SCALAR_FIELD_SLOTS = ("metric", "dimension", "x", "y", "size", "color", "weight_field")
_LIST_FIELD_SLOTS = ("dimensions", "hierarchy")


@dataclass
class MIQuerySpec:
    """v1 specification for an MI query/chart request."""

    intent: str = "chart"                       # chart | table | summary
    chart_type: str = "none"                    # bar|line|scatter|bubble|heatmap|treemap|none

    # Semantic field keys (all optional; which are required depends on chart_type)
    metric: Optional[str] = None
    dimension: Optional[str] = None
    x: Optional[str] = None
    y: Optional[str] = None
    size: Optional[str] = None
    color: Optional[str] = None

    aggregation: str = "count"                  # see AGGREGATIONS
    weight_field: Optional[str] = None

    filters: Dict[str, Any] = field(default_factory=dict)
    top_n: Optional[int] = None

    # Multi-dimension support (heatmap / treemap)
    dimensions: List[str] = field(default_factory=list)
    hierarchy: List[str] = field(default_factory=list)

    title: Optional[str] = None
    explanation: Optional[str] = None
    output_format: str = "chart"                # chart | table | text | chart_and_table

    # ------------------------------------------------------------------ #
    # Serialisation
    # ------------------------------------------------------------------ #
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=False)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MIQuerySpec":
        if not isinstance(data, dict):
            raise TypeError("MIQuerySpec.from_dict expects a dict")
        known = {f for f in cls.__dataclass_fields__}  # type: ignore[attr-defined]
        kwargs = {k: v for k, v in data.items() if k in known}
        # Defensive normalisation of list/dict slots.
        for slot in _LIST_FIELD_SLOTS:
            if slot in kwargs and kwargs[slot] is None:
                kwargs[slot] = []
        if "filters" in kwargs and kwargs["filters"] is None:
            kwargs["filters"] = {}
        return cls(**kwargs)

    @classmethod
    def from_json(cls, text: str) -> "MIQuerySpec":
        return cls.from_dict(json.loads(text))

    # ------------------------------------------------------------------ #
    # Introspection
    # ------------------------------------------------------------------ #
    def referenced_fields(self) -> List[str]:
        """Return every semantic field key referenced by this spec (unique, ordered)."""
        out: List[str] = []
        for slot in _SCALAR_FIELD_SLOTS:
            value = getattr(self, slot)
            if value:
                out.append(value)
        for slot in _LIST_FIELD_SLOTS:
            for value in getattr(self, slot) or []:
                if value:
                    out.append(value)
        # filter keys are field references too
        for key in self.filters or {}:
            if key:
                out.append(key)
        # de-duplicate, preserve order
        seen = set()
        unique = []
        for f in out:
            if f not in seen:
                seen.add(f)
                unique.append(f)
        return unique
