"""mi_agent_api/movement_receipt — governed evidence for a ranked movement.

THE GAP THIS CLOSES. Every receipt in the estate was built by a route, and the
only element-level evidence channel — `metadata.rankedMovement` — is published
by exactly one of them. A composed answer therefore carried numbers and no audit
trail, and whether a declared element had been honoured was unverifiable off
that route. The C7 matrix recorded Receipt as RED on all four gates for that
reason.

WHAT THIS IS, AND WHAT IT DELIBERATELY IS NOT
---------------------------------------------
It is built from the SAME EXECUTION FACTS that produced the answer — the frames
that were read, the values that were summed, the deltas that were computed — and
it is built BEFORE any prose exists. Narration is not the evidence owner here;
prose is derived from the receipt, never the receipt from prose. Nothing in this
module reads a chart column name, an artifact title or an answer string.

It knows nothing about `period_change`, and imports nothing from it.

A `MovementReceipt` is sufficient to re-derive the answer: for every ranked
element it carries the group, the measure, both periods, both values, the
absolute and percentage movement, the basis, the direction, the limit and the
rank position — plus the population the whole thing was computed over.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

#: Schema version. A consumer that stores a receipt can tell which shape it has.
RECEIPT_SCHEMA = "movement_receipt/1"


@dataclass(frozen=True)
class PopulationEvidence:
    """What the movement was computed over, and how it was narrowed.

    `row_counts` is per period and is what makes a filter's effect checkable:
    a predicate that changes nothing leaves the counts equal to the unfiltered
    ones, and a receipt that cannot show that cannot prove a filter ran.
    """

    dataset: Optional[str] = None
    portfolio_ids: Tuple[str, ...] = ()
    predicates: Tuple[Tuple[str, str, Any], ...] = ()
    row_counts: Tuple[int, ...] = ()
    unfiltered_row_counts: Tuple[int, ...] = ()

    @property
    def narrowed(self) -> bool:
        """True when the predicates actually removed rows."""
        return bool(self.predicates) and (
            tuple(self.row_counts) != tuple(self.unfiltered_row_counts))

    def to_dict(self) -> Dict[str, Any]:
        return {"dataset": self.dataset,
                "portfolioIds": list(self.portfolio_ids),
                "predicates": [list(p) for p in self.predicates],
                "rowCounts": list(self.row_counts),
                "unfilteredRowCounts": list(self.unfiltered_row_counts),
                "narrowed": self.narrowed}


@dataclass(frozen=True)
class RankedElement:
    """One ranked group, with everything needed to re-derive its position."""

    rank: int
    group_value: str
    start_value: float
    end_value: float
    absolute_movement: float
    percentage_movement: Optional[float]

    def to_dict(self) -> Dict[str, Any]:
        return {"rank": self.rank, "groupValue": self.group_value,
                "startValue": self.start_value, "endValue": self.end_value,
                "absoluteMovement": self.absolute_movement,
                "percentageMovement": self.percentage_movement}


@dataclass(frozen=True)
class MovementReceipt:
    """Governed evidence for a ranked (or unranked) historical movement."""

    schema: str = RECEIPT_SCHEMA
    measure: Optional[str] = None
    aggregation: Optional[str] = None
    grouping_dimension: Optional[str] = None
    #: Every governed field the named term could have bound to, best first, and
    #: the one that was bound. An availability difference must never look like a
    #: substitution, which is the whole reason the contract carries alternates.
    grouping_candidates: Tuple[str, ...] = ()
    start_period: Optional[str] = None
    end_period: Optional[str] = None
    ranking_basis: Optional[str] = None
    ranking_direction: Optional[str] = None
    ordering_limit: Optional[int] = None
    elements: Tuple[RankedElement, ...] = ()
    #: Groups that were analysed but did not move in the requested direction.
    excluded_groups: Tuple[str, ...] = ()
    population: PopulationEvidence = field(default_factory=PopulationEvidence)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema": self.schema, "measure": self.measure,
            "aggregation": self.aggregation,
            "groupingDimension": self.grouping_dimension,
            "groupingCandidates": list(self.grouping_candidates),
            "startPeriod": self.start_period, "endPeriod": self.end_period,
            "rankingBasis": self.ranking_basis,
            "rankingDirection": self.ranking_direction,
            "orderingLimit": self.ordering_limit,
            "elements": [e.to_dict() for e in self.elements],
            "excludedGroups": list(self.excluded_groups),
            "population": self.population.to_dict(),
        }

    # ----------------------------------------------------------------- #
    # Completeness, checkable rather than asserted
    # ----------------------------------------------------------------- #
    #: The facts a ranked-movement answer cannot be audited without.
    REQUIRED = ("measure", "grouping_dimension", "start_period", "end_period",
                "ranking_basis", "ranking_direction")

    def missing_facts(self) -> List[str]:
        """Which required facts this receipt does not carry."""
        gaps = [name for name in self.REQUIRED if getattr(self, name) in (None, "")]
        if not self.elements:
            gaps.append("elements")
        for element in self.elements:
            if element.start_value is None or element.end_value is None:
                gaps.append(f"values[{element.group_value}]")
            # The movement must RECONCILE, not merely be present: a receipt that
            # carries a delta contradicting its own endpoints proves nothing.
            if round(element.end_value - element.start_value, 2) != round(
                    element.absolute_movement, 2):
                gaps.append(f"movement_does_not_reconcile[{element.group_value}]")
        return gaps

    @property
    def complete(self) -> bool:
        return not self.missing_facts()

    @property
    def chronological(self) -> bool:
        """The pair must run earlier -> later. D4, made structural."""
        if not (self.start_period and self.end_period):
            return False
        return str(self.start_period) <= str(self.end_period)


def build_movement_receipt(*, measure: Optional[str], aggregation: Optional[str],
                           grouping_dimension: Optional[str],
                           grouping_candidates: Sequence[str],
                           periods: Sequence[str],
                           levels: Sequence[Dict[str, float]],
                           ranked: Sequence[Tuple[str, float]],
                           basis: Optional[str], direction: Optional[str],
                           limit: Optional[int],
                           population: PopulationEvidence) -> MovementReceipt:
    """A receipt from execution facts. No prose is read and none is produced.

    `levels` are the per-group values AT each period, in period order, exactly
    as the executor computed them; `ranked` is the ordering it produced. Both
    come from the run, so the receipt cannot drift from the answer — it is the
    same numbers, not a re-derivation of them.
    """
    start_levels = dict(levels[0]) if levels else {}
    end_levels = dict(levels[-1]) if levels else {}
    elements = []
    for position, (group, movement) in enumerate(ranked, start=1):
        start = float(start_levels.get(group, 0.0))
        end = float(end_levels.get(group, 0.0))
        elements.append(RankedElement(
            rank=position, group_value=str(group),
            start_value=round(start, 2), end_value=round(end, 2),
            absolute_movement=round(movement, 2),
            percentage_movement=(round(movement / start * 100, 2)
                                 if start else None)))
    ranked_names = {str(g) for g, _ in ranked}
    excluded = tuple(sorted(str(g) for g in (set(start_levels) | set(end_levels))
                            if str(g) not in ranked_names))
    return MovementReceipt(
        measure=measure, aggregation=aggregation,
        grouping_dimension=grouping_dimension,
        grouping_candidates=tuple(str(c) for c in grouping_candidates),
        start_period=(str(periods[0]) if periods else None),
        end_period=(str(periods[-1]) if periods else None),
        ranking_basis=basis, ranking_direction=direction, ordering_limit=limit,
        elements=tuple(elements), excluded_groups=excluded,
        population=population)
