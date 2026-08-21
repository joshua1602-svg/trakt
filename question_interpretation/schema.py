#!/usr/bin/env python3
"""question_interpretation/schema.py — Stage 1.

The transport object for lexical interpretation of a question. **Data only.**
Nothing here parses, resolves, or decides anything: Stage 1 defines the shape
and populates it from the EXISTING interpreters, read-only.

The central distinction
-----------------------
This object carries **linguistic claims, not execution claims**.

    dimension:
      raw_text: "borrower type"
      role: grouping

It does NOT say ``field_key = borrower_type`` and it does NOT say
``status = applied``. Those belong to layers downstream:

    QuestionInterpretation   what did the user ask for, in words
                             -> spans, raw text, roles
    ResolvedConcept          what does that mean in this portfolio
                             -> field keys, alt_keys, resolution status
    ExecutionFacet           what did the system manage to apply
                             -> applied / unavailable / lost, reason text

``applied`` is strictly stronger than "present in the question": it carries
execution evidence, and it exists because twelve of thirteen routes ignored
``spec.filters`` and a back-book question was answered whole-book with
``ok=True``. A pre-execution object cannot reproduce it and must not try. The
same holds for ``lost``.

Slot states
-----------
Every slot is ``filled``, ``empty``, or ``unresolvable``. Empty is normal.
Unresolvable is explicit and never a silent omission — it is what stops an
unresolved fragment being quietly folded into the subject.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

# --------------------------------------------------------------------------- #
# Slot states
# --------------------------------------------------------------------------- #
FILLED = "filled"
EMPTY = "empty"
UNRESOLVABLE = "unresolvable"
STATES = (FILLED, EMPTY, UNRESOLVABLE)

# --------------------------------------------------------------------------- #
# Controlled vocabularies. Deliberately small; corrected by the corpus, never
# widened to make one question fit.
# --------------------------------------------------------------------------- #
#: What kind of answer the question asks for.
COUNT = "count"
AMOUNT = "amount"
AVERAGE = "average"
MOVEMENT = "movement"
RANKING = "ranking"
FORWARD = "forward"
COVERAGE = "coverage"
NEUTRAL = "neutral"
OPERATION_TYPES = (COUNT, AMOUNT, AVERAGE, MOVEMENT, RANKING, FORWARD,
                   COVERAGE, NEUTRAL)

#: The SYNTACTIC role a named dimension plays in this sentence. Not a semantic
#: judgement about the registry: both `region` and `borrower type` are
#: dimensions; in "balance by region for joint borrowers" their roles differ.
#: This is the slot that resolves the KIND_GROUPING conflation.
GROUPING = "grouping"
FILTER = "filter"
UNRESOLVED_ROLE = "unresolved"
DIMENSION_ROLES = (GROUPING, FILTER, UNRESOLVED_ROLE)

#: Where a target threshold came from. `stated` is in the question; `configured`
#: references a plan or budget the question does not state.
STATED = "stated"
CONFIGURED = "configured"
TARGET_SOURCES = (STATED, CONFIGURED)

#: Time grains a question can name.
GRAINS = ("day", "week", "month", "quarter", "year")


@dataclass(frozen=True)
class Span:
    """Where in the question a claim came from.

    Character offsets into the ORIGINAL question string, so a consumer can show
    the user which words produced a decision, and so precedence between two
    overlapping claims is decidable. A facet has a rendered label and no
    offsets, which is why precedence is not decidable from facets.
    """

    start: int
    end: int

    def __post_init__(self) -> None:
        if self.start < 0 or self.end < self.start:
            raise ValueError("invalid span (%r, %r)" % (self.start, self.end))

    def text_of(self, question: str) -> str:
        return question[self.start:self.end]

    def overlaps(self, other: "Span") -> bool:
        return not (self.end <= other.start or self.start >= other.end)


@dataclass
class Slot:
    """One claim about the question. State plus the words that produced it."""

    state: str = EMPTY
    raw_text: Optional[str] = None
    span: Optional[Span] = None
    #: Set only when state == UNRESOLVABLE: why it could not be resolved.
    reason: Optional[str] = None
    #: Which existing interpreter produced this claim. Stage 1 diagnostic; it is
    #: how a disagreement is attributed to a source.
    source: Optional[str] = None

    def __post_init__(self) -> None:
        if self.state not in STATES:
            raise ValueError("unknown slot state %r" % (self.state,))

    def as_dict(self) -> Dict[str, Any]:
        return {"state": self.state, "raw_text": self.raw_text,
                "span": [self.span.start, self.span.end] if self.span else None,
                "reason": self.reason, "source": self.source}


@dataclass
class OperationClaim(Slot):
    type: Optional[str] = None
    modifiers: Tuple[str, ...] = ()

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.type is not None and self.type not in OPERATION_TYPES:
            raise ValueError("unknown operation type %r" % (self.type,))

    def as_dict(self) -> Dict[str, Any]:
        d = super().as_dict()
        d.update({"type": self.type, "modifiers": list(self.modifiers)})
        return d


@dataclass
class SubjectClaim(Slot):
    #: The measure the words appear to name, as a CANDIDATE only. Naming it
    #: `candidate_concept` rather than `field_key` is deliberate: resolution
    #: belongs to ResolvedConcept, and a candidate may turn out to resolve to
    #: nothing in this portfolio.
    candidate_concept: Optional[str] = None

    def as_dict(self) -> Dict[str, Any]:
        d = super().as_dict()
        d["candidate_concept"] = self.candidate_concept
        return d


@dataclass
class DimensionClaim(Slot):
    role: str = UNRESOLVED_ROLE
    candidate_concept: Optional[str] = None

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.role not in DIMENSION_ROLES:
            raise ValueError("unknown dimension role %r" % (self.role,))

    def as_dict(self) -> Dict[str, Any]:
        d = super().as_dict()
        d.update({"role": self.role, "candidate_concept": self.candidate_concept})
        return d


@dataclass
class FilterClaim(Slot):
    """A narrowing condition, as the QUESTION states it.

    ``operator`` and ``value`` are the words' own content. The FIELD the
    condition bears on is not here: every `threshold` facet in the release
    candidate carries ``field_key=None`` for the same reason — identifying that
    a clause exists is a different job from resolving what it binds.
    """

    operator: Optional[str] = None
    value: Optional[str] = None
    #: Set when the condition names a dimension VALUE ("in London", "status is
    #: offer") rather than a numeric bound.
    categorical_value: Optional[str] = None

    def as_dict(self) -> Dict[str, Any]:
        d = super().as_dict()
        d.update({"operator": self.operator, "value": self.value,
                  "categorical_value": self.categorical_value})
        return d


@dataclass
class TimeClaim:
    """Everything temporal the question states.

    `requested_grain` is the axis a series is broken down BY; `trend_window` is
    the period it is narrowed TO. "balance by month over the last 6 months"
    states both, and they are different claims about the same dimension.
    """

    comparison_period: Slot = field(default_factory=Slot)
    requested_grain: Slot = field(default_factory=Slot)
    trend_window: Slot = field(default_factory=Slot)
    #: The grain value itself, when requested_grain is filled.
    grain: Optional[str] = None

    def __post_init__(self) -> None:
        if self.grain is not None and self.grain not in GRAINS:
            raise ValueError("unknown grain %r" % (self.grain,))

    def as_dict(self) -> Dict[str, Any]:
        return {"comparison_period": self.comparison_period.as_dict(),
                "requested_grain": self.requested_grain.as_dict(),
                "trend_window": self.trend_window.as_dict(),
                "grain": self.grain}


@dataclass
class TargetClaim(Slot):
    """A threshold the answer solves FOR, not one that narrows the population.

    Without this slot the value in "when do we reach £100m" lands in `filters`
    and narrows the book to loans of £100m or more. `source=configured` covers
    "are we on target" / "versus plan", where the threshold is not in the
    question at all and the slot is filled-but-unresolvable if no plan exists.
    """

    value: Optional[str] = None
    #: Named `target_source`, not `source`: `Slot.source` already records which
    #: interpreter produced the claim, and one field cannot mean both.
    target_source: Optional[str] = None

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.target_source is not None and self.target_source not in TARGET_SOURCES:
            raise ValueError("unknown target source %r" % (self.target_source,))

    def as_dict(self) -> Dict[str, Any]:
        d = super().as_dict()
        d.update({"value": self.value, "target_source": self.target_source})
        return d


@dataclass
class PopulationClaim(Slot):
    """A governed population the question names — "the back book", "new lending".

    Separate from `dimensions` because the release candidate resolves these by
    INTENT rather than by an applied filter, which is the recorded hypothesis
    for why 32c263a's classification over-assigned POPULATION and blocked 160
    runs. Keeping the linguistic claim separate from the resolution is what
    lets Stage 4 test that hypothesis without re-deciding it here.
    """

    concept: Optional[str] = None

    def as_dict(self) -> Dict[str, Any]:
        d = super().as_dict()
        d["concept"] = self.concept
        return d


@dataclass
class QuestionInterpretation:
    """One question, interpreted lexically. Carried, not acted on."""

    question: str = ""
    operation: OperationClaim = field(default_factory=OperationClaim)
    subject: SubjectClaim = field(default_factory=SubjectClaim)
    dimensions: List[DimensionClaim] = field(default_factory=list)
    filters: List[FilterClaim] = field(default_factory=list)
    time: TimeClaim = field(default_factory=TimeClaim)
    target: TargetClaim = field(default_factory=TargetClaim)
    population: List[PopulationClaim] = field(default_factory=list)
    #: Wording no interpreter claimed. Never folded into the subject.
    residue: List[str] = field(default_factory=list)
    #: Stage 1 diagnostics: which interpreters were consulted, and what they
    #: could not supply. Never read by a consumer.
    notes: List[str] = field(default_factory=list)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "question": self.question,
            "operation": self.operation.as_dict(),
            "subject": self.subject.as_dict(),
            "dimensions": [d.as_dict() for d in self.dimensions],
            "filters": [f.as_dict() for f in self.filters],
            "time": self.time.as_dict(),
            "target": self.target.as_dict(),
            "population": [p.as_dict() for p in self.population],
            "residue": list(self.residue),
            "notes": list(self.notes),
        }

    # -- read-only views, no decisions ------------------------------------ #
    def dimensions_with_role(self, role: str) -> List[DimensionClaim]:
        if role not in DIMENSION_ROLES:
            raise ValueError("unknown dimension role %r" % (role,))
        return [d for d in self.dimensions if d.role == role]

    def unresolvable_slots(self) -> List[Tuple[str, Slot]]:
        out: List[Tuple[str, Slot]] = []
        for name in ("operation", "subject", "target"):
            slot = getattr(self, name)
            if slot.state == UNRESOLVABLE:
                out.append((name, slot))
        for name in ("comparison_period", "requested_grain", "trend_window"):
            slot = getattr(self.time, name)
            if slot.state == UNRESOLVABLE:
                out.append(("time.%s" % name, slot))
        for d in self.dimensions:
            if d.state == UNRESOLVABLE:
                out.append(("dimension", d))
        for f in self.filters:
            if f.state == UNRESOLVABLE:
                out.append(("filter", f))
        for p in self.population:
            if p.state == UNRESOLVABLE:
                out.append(("population", p))
        return out
