"""CAPABILITY OWNERSHIP BEATS A GENERIC MEASURE COLLISION.

Before any measure binds, the parser asks every governed capability whether it
CLAIMS a span of the sentence. A claimed span is masked from measure binding
(so "collateral value" inside "how much collateral value are we able to borrow
against" cannot bind the valuation measure), the claim is recorded on the parse
metadata, and the capability's route honours it. Two capabilities claiming one
span is UNRESOLVABLE and the parser refuses rather than picking one.

The owners are asked; nothing here knows any vocabulary. To add a capability's
claim, give it a `claims(question)` reader and register it in `_OWNERS`.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple


@dataclass(frozen=True)
class CapabilityClaim:
    capability: str
    concept: str
    start: int
    end: int
    evidence: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {"capability": self.capability, "concept": self.concept,
                "start": self.start, "end": self.end, "evidence": self.evidence}

    def overlaps(self, other: "CapabilityClaim") -> bool:
        return self.start < other.end and other.start < self.end


@dataclass(frozen=True)
class Arbitration:
    winner: Optional[CapabilityClaim]
    conflicts: Tuple[Tuple[CapabilityClaim, CapabilityClaim], ...] = ()
    claims: Tuple[CapabilityClaim, ...] = ()

    @property
    def unresolvable(self) -> bool:
        return bool(self.conflicts)


def _owners() -> Tuple[Callable[[Optional[str]], Iterable[Tuple[str, str, int, int, str]]], ...]:
    from .borrowing_base import capacity

    return (capacity.claims,)


def claims(question: Optional[str], semantics: Any = None) -> Tuple[CapabilityClaim, ...]:
    """Every capability claim on ``question``, from the owners themselves."""
    out: List[CapabilityClaim] = []
    for owner in _owners():
        try:
            for cap, concept, start, end, evidence in owner(question):
                out.append(CapabilityClaim(str(cap), str(concept), int(start),
                                           int(end), str(evidence)))
        except Exception:  # noqa: BLE001 - an owner that cannot answer claims nothing
            continue
    return tuple(sorted(out, key=lambda c: (c.start, -(c.end - c.start))))


def arbitrate(candidates: Sequence[CapabilityClaim]) -> Arbitration:
    """ONE owner per span. Overlapping claims from DIFFERENT capabilities are a
    conflict — the sentence is unresolvable and the caller refuses. Overlapping
    claims from one capability are one claim (the widest). The winner is the
    widest claim standing."""
    items = list(candidates or ())
    conflicts: List[Tuple[CapabilityClaim, CapabilityClaim]] = []
    for i, a in enumerate(items):
        for b in items[i + 1:]:
            if a.overlaps(b) and a.capability != b.capability:
                conflicts.append((a, b))
    if conflicts:
        return Arbitration(None, tuple(conflicts), tuple(items))
    if not items:
        return Arbitration(None, (), ())
    winner = max(items, key=lambda c: (c.end - c.start, -c.start))
    return Arbitration(winner, (), tuple(items))


def mask(text: str, candidates: Sequence[CapabilityClaim]) -> str:
    """``text`` with every claimed span blanked, offsets preserved."""
    out = list(text or "")
    for c in candidates or ():
        for i in range(max(0, c.start), min(len(out), c.end)):
            out[i] = " "
    return "".join(out)


__all__ = ["CapabilityClaim", "Arbitration", "claims", "arbitrate", "mask"]
