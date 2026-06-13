"""
field_scope.py
=============

PART 3 — resolve the in/out-of-scope field universe for an onboarding mode,
driven entirely by the EXISTING fields_registry.yaml metadata:

  * ``category``       : regulatory | analytics
  * ``core_canonical`` : true | false   (the registry's core-field flag)

No new taxonomy is introduced. The key precedence rule is that
``core_canonical`` fields are ALWAYS included even when their ``category`` is in
the mode's excluded categories — a core balance/date/identifier must never be
dropped just because it is tagged ``regulatory``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set

from engine.gate_1_alignment.semantic_alignment import (
    load_field_registry,
    select_registry_fields,
)
from .mode_policy import ModePolicy, field_group_patterns


@dataclass
class FieldScopeResult:
    mode_name: str = ""
    included_fields: Set[str] = field(default_factory=set)
    excluded_fields: Set[str] = field(default_factory=set)
    blocking_fields: Set[str] = field(default_factory=set)
    analytics_fields: Set[str] = field(default_factory=set)
    regulatory_fields: Set[str] = field(default_factory=set)
    core_canonical_fields: Set[str] = field(default_factory=set)
    out_of_scope_reason_by_field: Dict[str, str] = field(default_factory=dict)

    def is_excluded(self, field_name: str) -> bool:
        return field_name in self.excluded_fields

    def category_of(self, field_name: str) -> str:
        if field_name in self.regulatory_fields:
            return "regulatory"
        if field_name in self.analytics_fields:
            return "analytics"
        return ""

    def counts(self) -> Dict[str, int]:
        return {
            "included_fields_count": len(self.included_fields),
            "excluded_fields_count": len(self.excluded_fields),
            "excluded_regulatory_fields_count": len(self.excluded_fields & self.regulatory_fields),
            "core_canonical_fields_count": len(self.core_canonical_fields),
            "analytics_fields_count": len(self.analytics_fields),
            "regulatory_fields_count": len(self.regulatory_fields),
            "blocking_fields_count": len(self.blocking_fields),
        }


def _field_group_members(universe: List[str], groups: List[str]) -> Set[str]:
    """Name-pattern match registry fields to conceptual groups (warehouse…)."""
    if not groups:
        return set()
    patterns = field_group_patterns()
    members: Set[str] = set()
    for grp in groups:
        for pat in patterns.get(grp, [grp]):
            members.update(f for f in universe if pat in f.lower())
    return members


def resolve_field_scope(
    registry: dict | str | Path,
    policy: ModePolicy,
    portfolio_type: str = "equity_release",
    regulatory_reporting_enabled: bool = False,
) -> FieldScopeResult:
    """Resolve included / excluded / blocking field sets for ``policy``'s mode."""
    if not isinstance(registry, dict):
        registry = load_field_registry(Path(registry))
    fields = registry.get("fields", {}) or {}

    universe = select_registry_fields(registry, portfolio_type)
    uni_set = set(universe)

    core_canonical = {f for f in universe if (fields.get(f, {}) or {}).get("core_canonical") is True}
    regulatory = {f for f in universe if (fields.get(f, {}) or {}).get("category") == "regulatory"}
    analytics = {f for f in universe if (fields.get(f, {}) or {}).get("category") == "analytics"}

    # Are regulatory fields active for this mode?
    regulatory_active = "regulatory" in policy.include_categories
    if not regulatory_active and policy.regulatory_fields_active_if:
        regulatory_active = regulatory_reporting_enabled

    # ---- Inclusion ----
    included: Set[str] = set()
    if policy.include_core_canonical:
        included |= core_canonical  # core always wins over category exclusion
    for cat in policy.include_categories:
        included |= {f for f in universe if (fields.get(f, {}) or {}).get("category") == cat}
    if regulatory_active:
        included |= regulatory
    included |= _field_group_members(universe, policy.include_field_groups)
    included &= uni_set

    # ---- Exclusion ----
    excluded = uni_set - included
    reasons: Dict[str, str] = {}
    for f in excluded:
        cat = (fields.get(f, {}) or {}).get("category", "")
        if cat in policy.exclude_categories:
            reasons[f] = f"category '{cat}' excluded by mode '{policy.name}'"
        else:
            reasons[f] = f"not in scope for mode '{policy.name}'"

    # ---- Blocking ----
    rules = policy.blocking_rules or {}
    blocking: Set[str] = set()
    if rules.get("structural_viability_only"):
        struct = set(policy.structural_viability_fields) & uni_set
        blocking = struct or (core_canonical & {"loan_identifier", "current_principal_balance"})
    elif rules.get("core_canonical_missing", True):
        blocking = set(core_canonical)
    # Only ever block on included fields.
    blocking &= included

    return FieldScopeResult(
        mode_name=policy.name,
        included_fields=included,
        excluded_fields=excluded,
        blocking_fields=blocking,
        analytics_fields=analytics,
        regulatory_fields=regulatory,
        core_canonical_fields=core_canonical,
        out_of_scope_reason_by_field=reasons,
    )
