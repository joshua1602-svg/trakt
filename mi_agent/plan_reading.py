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


#: THE LATEST PAIR, as a relative pair: the latest snapshot and the one before
#: it. What a change of the pipeline means when no pair is named (owner
#: decision D15: "strictly between the two most recent pipeline snapshots").
LATEST_PAIR: Mapping[str, Any] = {"form": "relative_pair", "periods_back": 1}


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


def change_form_of(plan: Any) -> Optional[str]:
    """The analytical form the COMPILER read, from the plan's own bindings."""
    provenance = as_mapping(plan).get("provenance") or {}
    bindings = provenance.get("compiler_bindings") or {}
    form = (bindings.get("change_form") or {}).get("form")
    return str(form) if form else None


def names_latest_pair(plan: Any) -> bool:
    """Does a CHANGE plan name the latest pair without naming a pair?

    One reading for every owner of a change over the pipeline's snapshots
    (owner decision D15). A change needs two states; when the plan names
    none, which two is decided here, not by each runtime:

        "the previous period" ("since the last snapshot")   the latest pair
        the current state, or nothing stated, where the     the latest pair
          change form owns its window (a "what changed"
          summary compares the current state with the
          previous one — the compiler's own default)

    A plan that states a pair (a relative pair, named periods) is read by its
    owner as stated; a plan with no change form is not a change.
    """
    body = as_mapping(plan)
    change = change_form_of(body)
    if not change:
        return False
    form = str((body.get("period") or {}).get("form") or "")
    if form == "previous_reporting_period":
        return True
    if form != "current":
        return False
    from mi_agent.interpretation_v2.vocabulary import (
        CHANGE_FORM_ABSENT_PERIOD_DEFAULT)
    return CHANGE_FORM_ABSENT_PERIOD_DEFAULT.get(change) is not None


def pair_period(plan: Any) -> Mapping[str, Any]:
    """The pair a change plan is read over: its own period, or the latest
    pair when it names the latest pair implicitly (`names_latest_pair`)."""
    body = as_mapping(plan)
    return (dict(LATEST_PAIR) if names_latest_pair(body)
            else dict(body.get("period") or {}))
