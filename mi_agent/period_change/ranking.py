#!/usr/bin/env python3
"""mi_agent/period_change/ranking.py — P1C ranked period-over-period movement.

Ranking ONLY. Every number ranked here was produced by the governed
period-change engine (:func:`mi_agent.period_change.distribution.distribution_change`);
nothing is recalculated and no methodology is introduced.

    governed period-change engine
            ↓
    CategoryShift per category
            ↓
    rank_movement()            ← this module: pure, deterministic
            ↓
    RankedMovement

What the governed engine already gives per category (its contract, verified
against :class:`~mi_agent.period_change.models.CategoryShift`):

    start_count / end_count                      counts at each date
    start_count_share / end_count_share          composition, each date
    count_share_movement                         composition shift
    start_balance / end_balance                  balance at each date
    start_balance_share / end_balance_share      balance composition
    balance_share_movement                       balance composition shift
    presence                                     both | start_only | end_only

It does NOT give absolute or percentage movement. Those are differences of two
figures the engine already reconciled, computed here so the ranking basis is
explicit and auditable rather than implied.

Tie-breaking follows the convention ``distribution_change`` already uses for its
own top-shift lists: the ranking value first, then the category name ascending.
Two equal movements therefore always order the same way, and pandas row order
can never decide business semantics.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from analytics_lib.stratify import UNKNOWN_LABEL

# --------------------------------------------------------------------------- #
# Ranking bases — what "grew the most" is measured ON
# --------------------------------------------------------------------------- #
#: Absolute movement in the governed balance measure (end - start).
BASIS_BALANCE_ABSOLUTE = "balance_absolute"
#: Relative movement in the governed balance measure ((end - start) / start).
BASIS_BALANCE_PERCENT = "balance_percent"
#: Absolute movement in loan count (end - start).
BASIS_COUNT_ABSOLUTE = "count_absolute"
#: Movement in the category's SHARE of the book, in percentage points.
BASIS_BALANCE_SHARE = "balance_share"
#: Movement in the category's share of loan COUNT, in percentage points.
BASIS_COUNT_SHARE = "count_share"

#: Human labels for the execution receipt.
BASIS_LABELS = {
    BASIS_BALANCE_ABSOLUTE: "absolute balance movement",
    BASIS_BALANCE_PERCENT: "percentage balance movement",
    BASIS_COUNT_ABSOLUTE: "loan-count movement",
    BASIS_BALANCE_SHARE: "share-of-balance movement",
    BASIS_COUNT_SHARE: "share-of-count movement",
}

DIRECTION_INCREASE = "increase"
DIRECTION_DECREASE = "decrease"
DIRECTION_ANY = "movement"

#: A percentage movement needs a non-zero opening figure. A category that did
#: not exist at the opening date has no percentage growth — reporting one would
#: be dividing by zero and calling the result infinite growth.
_NO_OPENING_BASE = "no opening balance, so percentage movement is undefined"

#: The missing-value bucket every governed stratification uses. It is a real row
#: in the distribution, but it is not a REGION or a PRODUCT — answering "which
#: region grew the most?" with "Unknown" names the absence of a value as though
#: it were a value. Excluded by default and always DISCLOSED, never dropped
#: silently: a caller who wants it ranked passes ``exclude_categories=()``.
_NOT_A_CATEGORY = "the unknown bucket is missing data, not a reported category"


@dataclass
class RankedCategory:
    """One category's movement, with every figure the answer may quote."""

    category: str
    start_value: Optional[float]
    end_value: Optional[float]
    absolute_movement: Optional[float]
    percent_movement: Optional[float]
    #: The value this row was actually ranked on.
    rank_value: Optional[float]
    presence: str
    #: Set when a figure could not be computed, and why.
    note: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "category": self.category,
            "start_value": self.start_value,
            "end_value": self.end_value,
            "absolute_movement": self.absolute_movement,
            "percent_movement": self.percent_movement,
            "rank_value": self.rank_value,
            "presence": self.presence,
            "note": self.note,
        }


@dataclass
class RankedMovement:
    """The ranked result. ``ok=False`` means the ranking could not be produced."""

    ok: bool
    basis: str
    direction: str
    rows: List[RankedCategory] = field(default_factory=list)
    top_n: Optional[int] = None
    #: Categories excluded from the ranking, with the reason.
    excluded: List[Tuple[str, str]] = field(default_factory=list)
    #: How many ranked categories moved the OTHER way and so are not listed.
    #: Reported rather than silently dropped: "3 regions grew" reads very
    #: differently when the reader knows the other 9 did not.
    direction_excluded: int = 0
    reason: Optional[str] = None

    @property
    def basis_label(self) -> str:
        return BASIS_LABELS.get(self.basis, self.basis)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ok": self.ok,
            "basis": self.basis,
            "basisLabel": self.basis_label,
            "direction": self.direction,
            "topN": self.top_n,
            "rows": [r.to_dict() for r in self.rows],
            "excluded": [{"category": c, "reason": r} for c, r in self.excluded],
            "directionExcluded": self.direction_excluded,
            "reason": self.reason,
        }


def _values_for(shift: Any, basis: str) -> Tuple[Optional[float], Optional[float]]:
    """``(start, end)`` on the requested basis, or ``(None, None)``."""
    if basis in (BASIS_BALANCE_ABSOLUTE, BASIS_BALANCE_PERCENT):
        return getattr(shift, "start_balance", None), getattr(shift, "end_balance", None)
    if basis == BASIS_COUNT_ABSOLUTE:
        return float(getattr(shift, "start_count", 0)), float(getattr(shift, "end_count", 0))
    if basis == BASIS_BALANCE_SHARE:
        return (getattr(shift, "start_balance_share", None),
                getattr(shift, "end_balance_share", None))
    if basis == BASIS_COUNT_SHARE:
        return (getattr(shift, "start_count_share", None),
                getattr(shift, "end_count_share", None))
    return None, None


def rank_movement(shifts: Sequence[Any], *, basis: str,
                  direction: str = DIRECTION_INCREASE,
                  top_n: Optional[int] = None,
                  exclude_categories: Sequence[str] = (UNKNOWN_LABEL,)
                  ) -> RankedMovement:
    """Rank governed category movements. Pure: no I/O, no recalculation.

    ``direction`` selects which end of the ranking is returned:
    ``increase`` (largest positive first), ``decrease`` (largest negative
    first), or ``movement`` (largest absolute magnitude first, either sign).

    A category the requested basis cannot express — a percentage movement with
    no opening balance, a balance basis on a book with no governed balance
    field — is EXCLUDED WITH A REASON rather than ranked as zero. Ranking it as
    zero would bury it mid-table and present an unknown as a fact.
    """
    if basis not in BASIS_LABELS:
        return RankedMovement(ok=False, basis=basis, direction=direction,
                              reason=f"unsupported ranking basis {basis!r}")
    if top_n is not None and top_n <= 0:
        return RankedMovement(ok=False, basis=basis, direction=direction,
                              reason=f"invalid top-N ({top_n})")
    if direction not in (DIRECTION_INCREASE, DIRECTION_DECREASE, DIRECTION_ANY):
        return RankedMovement(ok=False, basis=basis, direction=direction,
                              reason=f"unsupported direction {direction!r}")

    rows: List[RankedCategory] = []
    excluded: List[Tuple[str, str]] = []
    skip = {str(c).strip().lower() for c in (exclude_categories or ())}

    for shift in shifts or ():
        category = str(getattr(shift, "category", "") or "")
        start, end = _values_for(shift, basis)
        presence = str(getattr(shift, "presence", "both"))

        if category.strip().lower() in skip:
            excluded.append((category, _NOT_A_CATEGORY))
            continue
        if start is None or end is None:
            excluded.append((category, "the basis is not reported in both snapshots"))
            continue

        absolute = float(end) - float(start)
        percent: Optional[float] = None
        if float(start) != 0.0:
            percent = absolute / abs(float(start)) * 100.0

        if basis == BASIS_BALANCE_PERCENT:
            if percent is None:
                excluded.append((category, _NO_OPENING_BASE))
                continue
            rank_value: Optional[float] = percent
        else:
            rank_value = absolute

        rows.append(RankedCategory(
            category=category, start_value=float(start), end_value=float(end),
            absolute_movement=absolute, percent_movement=percent,
            rank_value=rank_value, presence=presence,
            note=None if percent is not None else _NO_OPENING_BASE))

    if not rows:
        return RankedMovement(
            ok=False, basis=basis, direction=direction, excluded=excluded,
            reason="no category could be ranked on the requested basis")

    ranked_total = len(rows)
    if direction == DIRECTION_INCREASE:
        rows = [r for r in rows if (r.rank_value or 0.0) > 0]
        rows.sort(key=lambda r: (-(r.rank_value or 0.0), r.category))
    elif direction == DIRECTION_DECREASE:
        rows = [r for r in rows if (r.rank_value or 0.0) < 0]
        rows.sort(key=lambda r: ((r.rank_value or 0.0), r.category))
    else:
        rows.sort(key=lambda r: (-abs(r.rank_value or 0.0), r.category))

    if not rows:
        moved = "increased" if direction == DIRECTION_INCREASE else "decreased"
        return RankedMovement(
            ok=False, basis=basis, direction=direction, excluded=excluded,
            reason=f"no category {moved} on this basis between the two periods")

    direction_excluded = ranked_total - len(rows)
    if top_n is not None:
        rows = rows[:top_n]

    return RankedMovement(ok=True, basis=basis, direction=direction,
                          rows=rows, top_n=top_n, excluded=excluded,
                          direction_excluded=direction_excluded)
