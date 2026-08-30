"""mi_agent_api.materiality — is this a driver, or is it just the largest number?

Deterministic commentary has a failure mode that is easy to miss because every
sentence it writes is arithmetically true. Ranking contributors and naming the
top one produces:

    "Movement was concentrated by region. South East contributed the largest
     increase (+£4.4m)."

on a book where seven regions each grew between £3.7m and £4.4m. Nothing there
is false. It is still wrong, because "concentrated" and "largest" invite the
reader to act on a distinction the data does not support — and the reader cannot
see the other six numbers to know that.

This module decides, once, whether a contribution set has a driver at all. Two
independent tests, both of which must pass:

  DOMINANCE   the leader's share of total absolute movement clears a floor;
  SEPARATION  the leader is far enough ahead of the runner-up to be
              distinguishable from it.

Either alone is insufficient. Dominance alone calls a leader in a two-way tie at
50 % each. Separation alone calls a leader that moved the book by nothing.

Where no driver is found the honest reading — that the movement was broadly
distributed — is itself the finding, and is returned as one.

Nothing here computes an economic value. It reads contributions a governed
decomposition already produced and classifies their SHAPE.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

#: The leader must carry at least this share of total absolute movement. Below
#: it, the movement is spread across the book however the ranking happens to
#: order it.
DOMINANCE_SHARE = 0.35

#: The leader must exceed the runner-up by at least this share of the leader's
#: own magnitude. At 0.20 a £4.4m leader needs a runner-up below £3.52m; the
#: seven-way £3.7-4.4m spread that motivated this module fails it, correctly.
SEPARATION_MARGIN = 0.20

#: Movements below this share of the opening balance are not worth naming a
#: driver for at all, whatever their internal shape.
MATERIAL_MOVEMENT_SHARE = 0.005

#: Classifications.
SHAPE_DRIVEN = "driven"                    # one contributor leads materially
SHAPE_CONCENTRATED = "concentrated"        # a few lead together
SHAPE_DISTRIBUTED = "broadly_distributed"  # no distinguishable leader
SHAPE_IMMATERIAL = "immaterial"            # the total movement is too small
SHAPE_EMPTY = "no_contributions"


@dataclass(frozen=True)
class Contribution:
    """One category's signed contribution to a movement."""

    label: str
    value: float

    @property
    def magnitude(self) -> float:
        return abs(float(self.value or 0.0))


@dataclass(frozen=True)
class Shape:
    """What a contribution set looks like, and what may honestly be said of it."""

    shape: str
    leader: Optional[Contribution] = None
    runner_up: Optional[Contribution] = None
    leader_share: Optional[float] = None
    separation: Optional[float] = None
    total_magnitude: float = 0.0
    contributor_count: int = 0
    #: Contributors that together clear ``DOMINANCE_SHARE`` — the honest answer
    #: to "which of these mattered" when no single one leads.
    leading_group: Tuple[Contribution, ...] = field(default_factory=tuple)

    @property
    def has_driver(self) -> bool:
        """Whether a single contributor may be named as THE driver."""
        return self.shape == SHAPE_DRIVEN

    def to_dict(self) -> Dict[str, Any]:
        return {
            "shape": self.shape,
            "hasDriver": self.has_driver,
            "leader": self.leader.label if self.leader else None,
            "leaderValue": self.leader.value if self.leader else None,
            "leaderShare": self.leader_share,
            "separation": self.separation,
            "contributorCount": self.contributor_count,
            "totalMagnitude": self.total_magnitude,
            "leadingGroup": [c.label for c in self.leading_group],
        }


def _as_contributions(rows: Sequence[Any], *, label_key: str = "label",
                      value_key: str = "value") -> List[Contribution]:
    out: List[Contribution] = []
    for row in rows or ():
        if isinstance(row, Contribution):
            out.append(row)
            continue
        if isinstance(row, Mapping):
            label, value = row.get(label_key), row.get(value_key)
        else:
            label, value = getattr(row, label_key, None), getattr(row, value_key, None)
        if label is None or value is None:
            continue
        try:
            out.append(Contribution(str(label), float(value)))
        except (TypeError, ValueError):
            continue
    return out


def classify(rows: Sequence[Any], *, label_key: str = "label",
             value_key: str = "value", base: Optional[float] = None,
             dominance_share: float = DOMINANCE_SHARE,
             separation_margin: float = SEPARATION_MARGIN,
             material_share: float = MATERIAL_MOVEMENT_SHARE) -> Shape:
    """Classify a contribution set.

    ``base`` is the opening balance the movement is measured against, where the
    caller has one. Supplying it enables the immateriality test: a movement worth
    a twentieth of a percent of the book has no driver worth naming, however
    lopsided its internal split.
    """
    contributions = _as_contributions(rows, label_key=label_key, value_key=value_key)
    contributions = [c for c in contributions if c.magnitude > 0]
    if not contributions:
        return Shape(shape=SHAPE_EMPTY)

    ordered = sorted(contributions, key=lambda c: c.magnitude, reverse=True)
    total = sum(c.magnitude for c in ordered)
    leader = ordered[0]
    runner_up = ordered[1] if len(ordered) > 1 else None
    leader_share = leader.magnitude / total if total else 0.0
    separation = ((leader.magnitude - runner_up.magnitude) / leader.magnitude
                  if runner_up and leader.magnitude else 1.0)

    # The group that together clears the dominance floor — what to name when no
    # single contributor does.
    group: List[Contribution] = []
    running = 0.0
    for c in ordered:
        group.append(c)
        running += c.magnitude
        if total and running / total >= dominance_share:
            break

    common = dict(leader=leader, runner_up=runner_up, leader_share=leader_share,
                  separation=separation, total_magnitude=round(total, 2),
                  contributor_count=len(ordered), leading_group=tuple(group))

    if base:
        try:
            if abs(float(base)) and total / abs(float(base)) < material_share:
                return Shape(shape=SHAPE_IMMATERIAL, **common)
        except (TypeError, ValueError):
            pass

    if len(ordered) == 1:
        return Shape(shape=SHAPE_DRIVEN, **common)
    if leader_share >= dominance_share and separation >= separation_margin:
        return Shape(shape=SHAPE_DRIVEN, **common)
    if len(group) <= max(2, len(ordered) // 3) and leader_share >= dominance_share / 2:
        return Shape(shape=SHAPE_CONCENTRATED, **common)
    return Shape(shape=SHAPE_DISTRIBUTED, **common)


def describe(shape: Shape, *, dimension: str, money=str,
             direction: Optional[str] = None) -> Optional[str]:
    """One governed sentence about the shape, or ``None`` when there is nothing
    honest to say.

    ``money`` formats a value in the governed reporting currency (pass
    ``insight_generators.money``); it is a parameter so this module never owns a
    currency symbol.
    """
    noun = str(dimension or "category").strip().lower()
    if shape.shape in (SHAPE_EMPTY,):
        return None
    if shape.shape == SHAPE_IMMATERIAL:
        return f"Movement across {noun} was immaterial for the period."
    if shape.shape == SHAPE_DRIVEN and shape.leader:
        verb = direction or ("increase" if shape.leader.value >= 0 else "reduction")
        return (f"{shape.leader.label} drove the movement by {noun} "
                f"({money(abs(shape.leader.value))} {verb}, "
                f"{shape.leader_share * 100:.0f}% of total movement).")
    if shape.shape == SHAPE_CONCENTRATED and shape.leading_group:
        names = ", ".join(c.label for c in shape.leading_group[:3])
        return (f"Movement by {noun} was concentrated in {names} "
                f"({sum(c.magnitude for c in shape.leading_group) / shape.total_magnitude * 100:.0f}% "
                f"of total movement).")
    return (f"Movement was broadly distributed across "
            f"{shape.contributor_count} {noun} categories — no single one drove it.")
