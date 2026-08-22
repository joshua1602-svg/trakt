#!/usr/bin/env python3
"""clause_splitting_phase1/span_model.py

The six-span vocabulary the Phase 1 benefit instruments are declared in.

This is a MEASUREMENT artefact, not product code. Nothing here is imported by
the MI agent, and nothing here knows how a splitter would work. It exists so a
probe can declare *what the correct clause split is* independently of which
tree is being scored — the tagged release candidate, which has no spans at all,
or a later branch that does.

Span states
-----------
``filled``          the span has a resolved value
``empty``           the span is absent, and that is normal
``unresolvable``    wording was recognised as belonging to this span but could
                    not be resolved. Never folds into the subject; triggers a
                    clarification.

Concept vocabulary
------------------
Spans are declared in *concept* terms, not surface text, so the same probe can
be scored against trees that use different internal names:

  subject     registry metric keys (``current_outstanding_balance``), or the
              pseudo-concept ``loan_count`` for a counted population.
  grouping    registry dimension keys (``collateral_geography``), or a time
              axis written ``time:month`` / ``time:quarter`` / ``time:week`` /
              ``time:unspecified``.
  filter      registry field keys the filter clause narrows on.
  period      period-kind tokens: ``window`` (last N units), ``current``,
              ``as_at``, ``since``, ``explicit_range``, ``comparison``.
  target      ``value:<number>`` for a threshold stated in the question, or
              ``configured`` for a plan/budget reference (rule 40).
  operation   one of OPERATIONS below.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

#: Operation classes. Rules A1-A9 of the draft rule set decide exactly one.
OPERATIONS = {
    "count",                # how many / number of / count of
    "amount",               # how much / total / value of
    "average_or_rate",      # average / mean / weighted average / typical
    "movement",             # how has X changed / moved / grown / fallen
    "ranking",              # which has the largest / top / highest / lowest
    "forward",              # when will / forecast / projected / estimate
    "forward_probability",  # are we likely to / is there a risk of
    "neutral",              # show / list / give me — type comes from subject
}

#: Span states.
FILLED = "filled"
EMPTY = "empty"
UNRESOLVABLE = "unresolvable"

#: The five content spans plus operation. Order is the reading order of the
#: rule set (§1 table), not a precedence order.
SPAN_NAMES = ("operation", "subject", "grouping", "filter", "period", "target")


@dataclass
class Span:
    """One span: its state, its resolved concepts, and the text it consumed."""

    state: str = EMPTY
    concepts: List[str] = field(default_factory=list)
    text: Optional[str] = None
    #: Set when state == UNRESOLVABLE: the wording that could not be resolved.
    unmatched: Optional[str] = None

    def as_dict(self) -> Dict[str, Any]:
        return {
            "state": self.state,
            "concepts": sorted(self.concepts),
            "text": self.text,
            "unmatched": self.unmatched,
        }


@dataclass
class Spans:
    """The six spans a question is cut into."""

    operation: Span = field(default_factory=Span)
    subject: Span = field(default_factory=Span)
    grouping: Span = field(default_factory=Span)
    filter: Span = field(default_factory=Span)
    period: Span = field(default_factory=Span)
    target: Span = field(default_factory=Span)
    #: Free-standing wording that matched no opener at all (draft rule 35).
    #: Distinct from a span whose state is UNRESOLVABLE, which at least knows
    #: which span the wording belongs to.
    unresolved_residue: List[str] = field(default_factory=list)
    #: Diagnostics, never scored.
    notes: List[str] = field(default_factory=list)

    def as_dict(self) -> Dict[str, Any]:
        out = {name: getattr(self, name).as_dict() for name in SPAN_NAMES}
        out["unresolved_residue"] = sorted(self.unresolved_residue)
        out["notes"] = list(self.notes)
        return out

    def concepts_of(self, name: str) -> List[str]:
        return sorted(getattr(self, name).concepts)

    def state_of(self, name: str) -> str:
        return getattr(self, name).state


def filled(concepts, text=None) -> Span:
    concepts = [concepts] if isinstance(concepts, str) else list(concepts)
    return Span(state=FILLED if concepts else EMPTY, concepts=concepts, text=text)


def unresolvable(unmatched: str, text: Optional[str] = None) -> Span:
    return Span(state=UNRESOLVABLE, concepts=[], text=text, unmatched=unmatched)
