"""Reading a compiled GovernedQueryPlan — the helpers every specialist runtime
shares, so no runtime reads a slot differently from another.

Pure: it reads the plan dict and nothing else. No question, no parse, no data.
"""
from __future__ import annotations

from typing import Any, List, Mapping, Optional, Tuple

#: The region "by region" governs to: the client's reporting taxonomy (the
#: compiler's `reporting` level). The only geography a specialist runtime groups
#: by, because it is the one every book is harmonised to (D12).
REPORTING_REGION = "canonical_region_reporting"


def as_mapping(plan: Any) -> Mapping[str, Any]:
    if isinstance(plan, Mapping):
        return plan
    to_dict = getattr(plan, "to_dict", None)
    return to_dict() if callable(to_dict) else {}


def single_output(plan: Any) -> Optional[Mapping[str, Any]]:
    outputs = as_mapping(plan).get("outputs") or ()
    return outputs[0] if len(outputs) == 1 else None


def requested_dimensions(plan: Any) -> List[str]:
    """The governed fields the plan's output groups by (binding field first)."""
    output = single_output(plan) or {}
    return [str(d.get("canonical_field") or d.get("concept") or "")
            for d in (output.get("dimensions") or ())]


def region_axis(plan: Any) -> Optional[str]:
    """The reporting-region field a plan groups by, or None.

    A plan asks for geography on its own slot, not as a dimension. Only a
    grouping at the reporting level is served; a region FILTER ("in London")
    or another level is a different question.
    """
    body = as_mapping(plan)
    output = single_output(body) or {}
    geography = output.get("geography") or body.get("geography") or {}
    if not isinstance(geography, Mapping) or not geography.get("group_by"):
        return None
    if geography.get("values"):
        return None
    field = str(geography.get("canonical_field") or "")
    return field if field == REPORTING_REGION else None


def asks_geography(plan: Any) -> bool:
    body = as_mapping(plan)
    output = single_output(body) or {}
    return bool(output.get("geography")
                or (body.get("geography") or {}).get("requested"))


def grouping_axes(plan: Any) -> List[str]:
    """Every field the plan groups by: its dimensions, then a region grouping."""
    region = region_axis(plan)
    return requested_dimensions(plan) + ([region] if region else [])


def plan_filters(plan: Any) -> List[Mapping[str, Any]]:
    """The plan's predicates, top-level and on its output."""
    body = as_mapping(plan)
    output = single_output(body) or {}
    return [f for f in list(body.get("filters") or ()) + list(output.get("filters") or ())
            if isinstance(f, Mapping)]


def member_filter(predicate: Mapping[str, Any]) -> Optional[Tuple[str, str]]:
    """`(field, value)` when a predicate is ONE equality on ONE value, else None."""
    field = str(predicate.get("canonical_field") or predicate.get("concept") or "")
    if not field or str(predicate.get("comparator") or "eq") != "eq":
        return None
    value = predicate.get("value")
    if isinstance(value, (list, tuple)):
        value = value[0] if len(value) == 1 else None
    if value is None or isinstance(value, (dict, bool)):
        return None
    return field, str(value).strip()
