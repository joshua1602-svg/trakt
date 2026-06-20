"""Project the real MI semantic layer into a catalogue payload for the UI.

Everything here is derived from the authoritative sources:
  - mi_agent.mi_query_spec  (enums)
  - mi_semantics_field_registry.yaml  (fields, via mi_agent_workflow.load_mi_semantics)
No values are invented.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any, Dict, List

from mi_agent.mi_query_spec import (
    AGGREGATIONS,
    CHART_TYPES,
    OUTPUT_FORMATS,
    RISK_MONITOR_MODES,
    STATES,
    TEMPORAL_MODES,
)
from mi_agent.mi_agent_workflow import load_mi_semantics

from .data_source import semantics_path


def _field_entry(key: str, meta: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "key": key,
        "label": meta.get("business_name") or meta.get("display_name") or key,
        "description": meta.get("business_description"),
        "format": meta.get("format"),
        "default_aggregation": meta.get("default_aggregation"),
        "allowed_aggregations": meta.get("allowed_aggregations", []),
        "allowed_chart_roles": meta.get("allowed_chart_roles", []),
        "bucketed": bool(meta.get("bucket_field")) or key.endswith("_bucket"),
        "mi_tier": meta.get("mi_tier"),
        "synonyms": meta.get("synonyms", []),
    }


@lru_cache(maxsize=1)
def build_catalogue() -> Dict[str, Any]:
    semantics = load_mi_semantics(str(semantics_path()))
    fields: Dict[str, Any] = semantics.get("fields", {})

    dimensions: List[Dict[str, Any]] = []
    measures: List[Dict[str, Any]] = []
    dates: List[Dict[str, Any]] = []
    for key, meta in sorted(fields.items()):
        role = meta.get("role")
        if role == "dimension":
            dimensions.append(_field_entry(key, meta))
        elif role == "metric":
            measures.append(_field_entry(key, meta))
        elif role == "date":
            dates.append(_field_entry(key, meta))

    # Chart types the agent can actually emit (drop the "none" sentinel).
    chart_types = sorted(ct for ct in CHART_TYPES if ct != "none")

    return {
        "states": sorted(STATES),
        "dimensions": dimensions,
        "measures": measures,
        "dates": dates,
        "aggregations": sorted(AGGREGATIONS),
        "chart_types": chart_types,
        "temporal_modes": sorted(TEMPORAL_MODES),
        "risk_monitor_modes": sorted(RISK_MONITOR_MODES),
        "output_formats": sorted(OUTPUT_FORMATS),
        # Filters: dimensions are the filterable axes in v1.
        "filters": [{"key": d["key"], "label": d["label"]} for d in dimensions],
        "counts": {
            "dimensions": len(dimensions),
            "measures": len(measures),
            "dates": len(dates),
        },
    }
