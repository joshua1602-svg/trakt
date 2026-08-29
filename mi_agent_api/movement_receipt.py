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
    """One ranked group, with everything needed to re-derive its position.

    THE LAST FOUR ARE CARRIED, NOT COMPUTED. When an engine has already
    produced the ranked rows, the receipt takes its figures verbatim. Re-deriving
    a percentage from the endpoints here would be a second calculation of the
    same fact, and two calculations of one fact eventually disagree — which is
    the duplication this receipt exists to remove, not to add.
    """

    rank: int
    group_value: str
    start_value: Optional[float]
    end_value: Optional[float]
    absolute_movement: Optional[float]
    percentage_movement: Optional[float]
    #: The figure the ranking actually sorted on, in the basis's own units.
    rank_value: Optional[float] = None
    #: Whether the group was present in one period or both.
    presence: Optional[str] = None
    #: The engine's own note about this row, if it made one.
    note: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {"rank": self.rank, "groupValue": self.group_value,
                "startValue": self.start_value, "endValue": self.end_value,
                "absoluteMovement": self.absolute_movement,
                "percentageMovement": self.percentage_movement,
                "rankValue": self.rank_value, "presence": self.presence,
                "note": self.note}


@dataclass(frozen=True)
class MovementReceipt:
    """Governed evidence for a ranked (or unranked) historical movement."""

    schema: str = RECEIPT_SCHEMA
    measure: Optional[str] = None
    aggregation: Optional[str] = None
    grouping_dimension: Optional[str] = None
    #: The reader-facing name of that dimension. Narration needs a label and
    #: must not manufacture one from a chart column or a field name.
    grouping_display_name: Optional[str] = None
    #: Every governed field the named term could have bound to, best first, and
    #: the one that was bound. An availability difference must never look like a
    #: substitution, which is the whole reason the contract carries alternates.
    grouping_candidates: Tuple[str, ...] = ()
    start_period: Optional[str] = None
    end_period: Optional[str] = None
    ranking_basis: Optional[str] = None
    #: The basis in the reader's words, as the ranking engine stated it.
    basis_label: Optional[str] = None
    ranking_direction: Optional[str] = None
    ordering_limit: Optional[int] = None
    #: How many groups the analysis considered, so "top 2 of 3" is evidence
    #: rather than a count narration takes elsewhere.
    groups_analysed: Optional[int] = None
    #: How many analysed groups did not move in the requested direction.
    direction_excluded: int = 0
    #: Groups the ranking REFUSED to place, each with the reason it gave.
    #: Distinct from `excluded_groups`, which is every analysed group absent
    #: from `elements` — a limit truncation included.
    exclusions: Tuple[Tuple[str, str], ...] = ()
    elements: Tuple[RankedElement, ...] = ()
    #: Groups that were analysed but did not move in the requested direction.
    excluded_groups: Tuple[str, ...] = ()
    population: PopulationEvidence = field(default_factory=PopulationEvidence)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema": self.schema, "measure": self.measure,
            "aggregation": self.aggregation,
            "groupingDimension": self.grouping_dimension,
            "groupingDisplayName": self.grouping_display_name,
            "groupingCandidates": list(self.grouping_candidates),
            "startPeriod": self.start_period, "endPeriod": self.end_period,
            "rankingBasis": self.ranking_basis,
            "basisLabel": self.basis_label,
            "rankingDirection": self.ranking_direction,
            "orderingLimit": self.ordering_limit,
            "groupsAnalysed": self.groups_analysed,
            "directionExcluded": self.direction_excluded,
            "exclusions": [list(e) for e in self.exclusions],
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
            if (element.start_value is None or element.end_value is None
                    or element.absolute_movement is None):
                # RECORDED, THEN SKIPPED. The gap is already reported; running
                # the reconciliation on a missing endpoint raises TypeError, so
                # a receipt with an unmeasurable element would crash its own
                # audit instead of failing it. Nothing is removed from `gaps`.
                gaps.append(f"values[{element.group_value}]")
                continue
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
                           population: PopulationEvidence,
                           elements: Optional[Sequence[RankedElement]] = None,
                           grouping_display_name: Optional[str] = None,
                           basis_label: Optional[str] = None,
                           analysed_groups: Sequence[str] = (),
                           direction_excluded: int = 0,
                           exclusions: Sequence[Tuple[str, str]] = (),
                           ) -> MovementReceipt:
    """A receipt from execution facts. No prose is read and none is produced.

    `levels` are the per-group values AT each period, in period order, exactly
    as the executor computed them; `ranked` is the ordering it produced. Both
    come from the run, so the receipt cannot drift from the answer — it is the
    same numbers, not a re-derivation of them.

    ONE BUILDER, TWO KINDS OF CALLER, AND NO SECOND CALCULATION. An executor
    that produced per-period levels passes `levels`; a ranking engine that
    already produced the ranked rows passes them as `elements` and they are
    taken VERBATIM. The alternative — re-deriving a percentage the engine had
    already stated — would put two calculations of one fact in the estate, and
    they would eventually disagree in the answer's favour or the receipt's.
    `levels` is still read for `excluded_groups` in both cases, because which
    analysed groups are absent from the ranking is a fact about the run and not
    about any one row.
    """
    start_levels = dict(levels[0]) if levels else {}
    end_levels = dict(levels[-1]) if levels else {}
    #: Which groups the analysis considered. An executor that supplies `levels`
    #: has already said so by naming them; a ranking engine that supplies
    #: `elements` has not, because its rows are only the ones it placed.
    analysed = ({str(g) for g in analysed_groups} if analysed_groups
                else (set(start_levels) | set(end_levels)))
    if elements is None:
        built: List[RankedElement] = []
        for position, (group, movement) in enumerate(ranked, start=1):
            start = float(start_levels.get(group, 0.0))
            end = float(end_levels.get(group, 0.0))
            built.append(RankedElement(
                rank=position, group_value=str(group),
                start_value=round(start, 2), end_value=round(end, 2),
                absolute_movement=round(movement, 2),
                percentage_movement=(round(movement / start * 100, 2)
                                     if start else None)))
        elements = built
        ranked_names = {str(g) for g, _ in ranked}
    else:
        elements = list(elements)
        ranked_names = {str(e.group_value) for e in elements}
    excluded = tuple(sorted(g for g in analysed if g not in ranked_names))
    return MovementReceipt(
        measure=measure, aggregation=aggregation,
        grouping_dimension=grouping_dimension,
        grouping_display_name=grouping_display_name,
        grouping_candidates=tuple(str(c) for c in grouping_candidates),
        start_period=(str(periods[0]) if periods else None),
        end_period=(str(periods[-1]) if periods else None),
        ranking_basis=basis, basis_label=basis_label,
        ranking_direction=direction, ordering_limit=limit,
        groups_analysed=(len(analysed) if analysed else None),
        direction_excluded=int(direction_excluded),
        exclusions=tuple((str(c), str(r)) for c, r in (exclusions or ())),
        elements=tuple(elements), excluded_groups=excluded,
        population=population)
