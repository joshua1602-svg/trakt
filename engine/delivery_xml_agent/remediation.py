"""
remediation.py
==============

Turns Delivery/XML Agent ``63_delivery_issues`` rows into a practical
remediation grouping for the XML-readiness roadmap.

The grouping is the single source of truth shared by:

  * ``docs/xml_readiness_remediation_roadmap.md`` (human roadmap), and
  * ``scripts/inspect_delivery_xml_readiness.py`` (diagnostic helper).

It is pure (no I/O, no raising) and keyed off the delivery blocker-type
vocabulary defined in :mod:`engine.delivery_xml_agent.delivery_xml_agent`.
"""

from __future__ import annotations

from collections import OrderedDict
from typing import Any, Dict, List

from engine.delivery_xml_agent.delivery_xml_agent import (
    BT_CLIENT, BT_OPERATOR_OR_CONFIG, BT_CONFIG, BT_SOURCE_MAPPING,
    BT_ND_DEFAULT_MISSING, BT_FORMAT, BT_STRUCTURE_DEFERRED, BT_TEMPLATE_ORDER,
    OWN_CLIENT, OWN_OPERATOR, OWN_CONFIG, OWN_PROJECTION, OWN_DELIVERY,
)

__all__ = ["REMEDIATION_GROUPS", "group_delivery_issues"]

# Ordered remediation groups (mirrors the roadmap Task-4 grouping 1..7).
# Each entry: (key, title, {blocker_types}, owner, needed_before_preview,
#              needed_before_production)
REMEDIATION_GROUPS = [
    {
        "key": "client_onboarding",
        "title": "Client onboarding decisions",
        "blocker_types": {BT_CLIENT},
        "owner": OWN_CLIENT,
        "needed_before_preview": True,
        "needed_before_production": True,
    },
    {
        "key": "operator_review",
        "title": "Operator decisions",
        "blocker_types": {BT_OPERATOR_OR_CONFIG},
        "owner": OWN_OPERATOR,
        "needed_before_preview": True,
        "needed_before_production": True,
    },
    {
        "key": "config_mapping",
        "title": "Config mapping decisions",
        "blocker_types": {BT_CONFIG},
        "owner": OWN_CONFIG,
        "needed_before_preview": True,
        "needed_before_production": True,
    },
    {
        "key": "source_projection",
        "title": "Source / projection mapping gaps",
        "blocker_types": {BT_SOURCE_MAPPING, BT_FORMAT},
        "owner": OWN_PROJECTION,
        "needed_before_preview": True,
        "needed_before_production": True,
    },
    {
        "key": "nd_default",
        "title": "ND / default policy gaps",
        "blocker_types": {BT_ND_DEFAULT_MISSING},
        "owner": OWN_CONFIG,
        "needed_before_preview": True,
        "needed_before_production": True,
    },
    {
        "key": "delivery_structure",
        "title": "Delivery structure gaps",
        "blocker_types": {BT_STRUCTURE_DEFERRED},
        "owner": OWN_DELIVERY,
        # Not gated in v1 (record-group is tagged, nesting deferred to v2), but
        # required before production XML.
        "needed_before_preview": False,
        "needed_before_production": True,
    },
    {
        "key": "template_order",
        "title": "Template / order gaps",
        "blocker_types": {BT_TEMPLATE_ORDER},
        "owner": OWN_DELIVERY,
        "needed_before_preview": True,
        "needed_before_production": True,
    },
]

_TYPE_TO_GROUP = {
    bt: g["key"] for g in REMEDIATION_GROUPS for bt in g["blocker_types"]
}


def group_delivery_issues(issues: List[Dict[str, Any]]) -> "OrderedDict[str, Dict[str, Any]]":
    """Group ``63_delivery_issues`` rows into the ordered remediation buckets.

    Returns an ``OrderedDict`` keyed by group key. Each value carries the group
    metadata plus ``codes`` (sorted unique ESMA codes), ``issue_ids`` and
    ``issue_count``. Groups with no matching issues are still present (empty), so
    the roadmap can show "none for this run" explicitly.
    """
    out: "OrderedDict[str, Dict[str, Any]]" = OrderedDict()
    for g in REMEDIATION_GROUPS:
        out[g["key"]] = {
            **{k: g[k] for k in (
                "key", "title", "owner", "needed_before_preview",
                "needed_before_production")},
            "blocker_types": sorted(g["blocker_types"]),
            "codes": set(),
            "issue_ids": [],
            "issue_count": 0,
        }

    for issue in issues or []:
        bt = str(issue.get("delivery_blocker_type", "")).strip()
        key = _TYPE_TO_GROUP.get(bt)
        if key is None:
            continue
        bucket = out[key]
        bucket["issue_count"] += 1
        iid = str(issue.get("delivery_issue_id", "")).strip()
        if iid:
            bucket["issue_ids"].append(iid)
        # an issue may carry one code or a comma-joined list (template order).
        raw = str(issue.get("esma_code", "")).strip()
        for code in (c.strip() for c in raw.split(",")):
            if code:
                bucket["codes"].add(code)

    # finalise: sort codes for stable output.
    for bucket in out.values():
        bucket["codes"] = sorted(bucket["codes"])
    return out
