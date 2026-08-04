"""simulation.dialects._plan — the declared column plan for one period.

Schema evolution is DECLARED on the manifest, never random. This module turns
those declarations into a concrete column plan (which canonical fields appear,
under which headers, in which order, plus any irrelevant extras) for one
dialect at one reporting period.

Keeping the evolution logic here rather than inside each renderer means the
three dialects cannot drift into three different readings of the same declared
variation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Sequence, Tuple

from ..models import (
    EVOLVE_ADD_OPTIONAL,
    EVOLVE_AMBIGUOUS,
    EVOLVE_CHANGE_ALIAS,
    EVOLVE_CHANGE_ENUMS,
    EVOLVE_DROP_MANDATORY,
    EVOLVE_DROP_OPTIONAL,
    EVOLVE_REORDER,
    CaseManifest,
    SchemaEvolution,
)
from .vocabulary import IRRELEVANT_COLUMNS, headers_for

#: Canonical fields whose absence must block the run. Dropping one is what the
#: ``missing mandatory data`` guard case does.
MANDATORY_BALANCE_FIELDS = ("current_outstanding_balance",
                            "current_principal_balance")


@dataclass
class ColumnPlan:
    """The concrete columns one dialect writes for one period."""

    #: ``(source_header, canonical_field_or_None)`` in file order.
    columns: List[Tuple[str, str]] = field(default_factory=list)
    #: Extra headers carrying no canonical meaning.
    extras: List[str] = field(default_factory=list)
    #: Canonical fields deliberately omitted this period, with the reason.
    omitted: Dict[str, str] = field(default_factory=dict)
    #: True when the plan deliberately makes the schema unresolvable.
    ambiguous: bool = False
    #: Which enum vocabulary to render statuses in.
    enum_style: str = "primary"

    @property
    def headers(self) -> List[str]:
        return [h for h, _ in self.columns] + list(self.extras)

    def canonical_for(self, header: str) -> str:
        for h, canon in self.columns:
            if h == header:
                return canon
        return ""


def _active(evolution: Sequence[SchemaEvolution], period_index: int,
            kind: str) -> List[SchemaEvolution]:
    return [e for e in evolution
            if e.kind == kind and period_index >= e.from_period]


def build_plan(case: CaseManifest, *, flavour: str, period_index: int,
               fields: Sequence[str], evolution: Sequence[SchemaEvolution],
               with_extras: bool = False) -> ColumnPlan:
    """The column plan for one dialect flavour at one reporting period."""
    header_map = headers_for(case.asset_class, flavour)
    ordered = [f for f in fields if f in header_map]

    plan = ColumnPlan()

    # ---- an ambiguous schema resolves to nothing, by design ---------------- #
    if _active(evolution, period_index, EVOLVE_AMBIGUOUS):
        plan.ambiguous = True
        plan.columns = [(f"col_{i:03d}", f) for i, f in enumerate(ordered)]
        return plan

    omitted: Dict[str, str] = {}

    # ---- a withheld mandatory column -------------------------------------- #
    for e in _active(evolution, period_index, EVOLVE_DROP_MANDATORY):
        for name in MANDATORY_BALANCE_FIELDS:
            if name in ordered:
                ordered.remove(name)
                omitted[name] = "withheld mandatory column (declared guard case)"

    # ---- a withheld optional column --------------------------------------- #
    for e in _active(evolution, period_index, EVOLVE_DROP_OPTIONAL):
        name = str(e.detail.get("column") or "")
        if name in ordered:
            ordered.remove(name)
            omitted[name] = "lender stopped supplying this optional column"

    # ---- a renamed header (the contract must still resolve it) ------------- #
    renamed: Dict[str, str] = {}
    for e in _active(evolution, period_index, EVOLVE_CHANGE_ALIAS):
        canonical = str(e.detail.get("canonical_field") or "")
        new_header = str(e.detail.get("new_header") or "")
        if canonical and new_header:
            renamed[canonical] = new_header

    # ---- column order ------------------------------------------------------ #
    if _active(evolution, period_index, EVOLVE_REORDER):
        # A deterministic rotation, not a shuffle: the reorder must be
        # reproducible from the manifest alone.
        offset = (period_index * 7) % max(1, len(ordered))
        ordered = ordered[offset:] + ordered[:offset]

    plan.columns = [(renamed.get(f, header_map[f]), f) for f in ordered]

    # ---- an optional column the lender starts supplying --------------------- #
    for e in _active(evolution, period_index, EVOLVE_ADD_OPTIONAL):
        name = str(e.detail.get("column") or "")
        if name and name not in ordered:
            # Emitted as an EXTRA header: the framework does not invent a
            # canonical value for it, so it must be reported as unmapped or
            # resolved by the global mapper — either way it is additive.
            plan.extras.append(name)

    if with_extras:
        plan.extras.extend(IRRELEVANT_COLUMNS)

    if _active(evolution, period_index, EVOLVE_CHANGE_ENUMS):
        plan.enum_style = "alternate"

    plan.omitted = omitted
    return plan


def format_switched(case: CaseManifest, period_index: int,
                    evolution: Sequence[SchemaEvolution]) -> str:
    """The dialect a declared ``switch_format`` moves to at this period, if any."""
    from ..models import EVOLVE_SWITCH_FORMAT
    for e in _active(evolution, period_index, EVOLVE_SWITCH_FORMAT):
        target = str(e.detail.get("to") or "")
        if target:
            return target
    return ""


__all__ = ["ColumnPlan", "MANDATORY_BALANCE_FIELDS", "build_plan",
           "format_switched"]
