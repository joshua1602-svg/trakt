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
NEUTRAL = "neutral"
#: CORRECTION 3, earned by the Stage 1 corpus. `coverage` was in the first
#: draft and is REMOVED: it was produced 0 times in 690 real-surface questions,
#: and no existing interpreter supplies it. An unsupplied member invites someone
#: to populate it by intuition, which is the failure the corpus-first rule
#: exists to prevent. Re-add it when a question demands it, with the question.
OPERATION_TYPES = (COUNT, AMOUNT, AVERAGE, MOVEMENT, RANKING, FORWARD, NEUTRAL)

#: The SYNTACTIC role a named dimension plays in this sentence. Not a semantic
#: judgement about the registry: both `region` and `borrower type` are
#: dimensions; in "balance by region for joint borrowers" their roles differ.
#: This is the slot that resolves the KIND_GROUPING conflation.
GROUPING = "grouping"
FILTER = "filter"
UNRESOLVED_ROLE = "unresolved"
DIMENSION_ROLES = (GROUPING, FILTER, UNRESOLVED_ROLE)

#: CORRECTION 5, earned by the Stage 1 corpus — and NARROWED by it.
#:
#: Stage 1 proposed splitting `unresolved` into "no source has an opinion" and
#: "the sources disagree". The corpus supports only the first: 55 of 690
#: real-surface questions name a dimension no source assigns a role to, and
#: ZERO name a dimension two sources put in different roles. A `conflicted`
#: value would therefore have been invented from intuition, so it is NOT added.
#:
#: What IS required is that an unresolved role says WHY, in `Slot.reason`, so
#: the distinction can be made from evidence if a conflicting case ever appears.
ROLE_UNATTRIBUTED = "no source supplies a role"

#: Where a target threshold came from. `stated` is in the question; `configured`
#: references a plan or budget the question does not state.
STATED = "stated"
CONFIGURED = "configured"
TARGET_SOURCES = (STATED, CONFIGURED)

#: CORRECTION 4, earned by the Stage 1 corpus. `configured` is a STATED
#: REQUIREMENT of the contract, so it stays in the vocabulary — but it has ZERO
#: corpus evidence: the wording ("on target", "versus plan") appears in 0 of 690
#: real-surface questions, and no interpreter supplies it. The projection's own
#: regex, which was the only thing producing it, has been removed: a regex owned
#: by the projection is a reading the projection invented, not one it observed.
#: Populating this slot requires an interpreter that supplies it.
UNSUPPLIED_TARGET_SOURCES = (CONFIGURED,)

#: Time grains a question can name.
GRAINS = ("day", "week", "month", "quarter", "year")

#: Which SOURCE PORTFOLIO(s) a question scopes to.
#:
#: PHASE 1A. Added because the compositional plan layer could not be built
#: without it: `mi_agent.portfolio_lens.resolve_lens` resolved
#: `source_portfolio_type=acquired` for "Summarise the acquired book" while this
#: object emitted nothing, so a downstream plan could not tell Total from
#: Acquired and an empty `population` list had to be read as "we do not know".
#:
#: `mi_agent.portfolio_lens` REMAINS THE SINGLE OWNER of this reading. The claim
#: below carries that owner's answer; it never re-derives one, and there is no
#: vocabulary here for a planner to match against.
SCOPE_TOTAL = "total"
SCOPE_DIRECT = "direct"
SCOPE_ACQUIRED = "acquired"
#: One or more named books (an SPV, a cohort id) chosen explicitly.
SCOPE_COHORT = "cohort"
SOURCE_SCOPES = (SCOPE_TOTAL, SCOPE_DIRECT, SCOPE_ACQUIRED, SCOPE_COHORT)

#: PHASE 1G — the BROAD BUSINESS POPULATION, kept separate from the specific
#: portfolio selection.
#:
#: The two are different concepts and conflating them is what produced the
#: hard-coded hierarchy this phase removes: Funded -> Acquired -> Acquired Book
#: 1 -> SPV1 -> ... . A request is instead a base population, optionally
#: narrowed to specific governed portfolios:
#:
#:     "the funded book"      base=funded    portfolios=()          (unrestricted)
#:     "the acquired book"    base=acquired  portfolios=(every acquired id)
#:     "SPV2"                 base=funded    portfolios=('spv2',)
#:
#: `SPV2` is NOT `base=acquired`: which category a named portfolio belongs to is
#: a property of the PORTFOLIO, held by the registry, not a property of the
#: request. Reading it as its category is the widening §8 forbids.
BASE_FUNDED = "funded"
BASE_DIRECT = "direct"
BASE_ACQUIRED = "acquired"
BASE_POPULATIONS = (BASE_FUNDED, BASE_DIRECT, BASE_ACQUIRED)

#: PHASE 1G — WHERE the resolved scope came from.
#:
#: Phase 1F stopped because the contract could not answer this. It carried WHICH
#: scope was resolved and not WHETHER THE USER ASKED FOR IT, and those are
#: different facts: "portfolio summary" and "portfolio summary across all
#: portfolios" both resolve to `total`, and production answers them differently
#: when the workspace carries a selection — the first defers to it, the second
#: overrides it. Measured across the owned surface, a plan that could not tell
#: them apart widened 14 of 54 (question, caller) combinations.
#:
#:     explicit_user   the QUESTION named this scope. It wins over any caller
#:                     context — that is the whole reason this value exists.
#:     caller_context  the question said nothing; this is the workspace
#:                     selection the caller supplied.
#:     default         the question said nothing and no caller context was
#:                     supplied, so the complete funded population applies.
#:     unresolved      the question NAMED a scope and it could not be resolved.
#:                     Never Funded — see `UNRESOLVABLE`.
PROV_EXPLICIT_USER = "explicit_user"
PROV_CALLER_CONTEXT = "caller_context"
PROV_DEFAULT = "default"
PROV_UNRESOLVED = "unresolved"
SCOPE_PROVENANCES = (PROV_EXPLICIT_USER, PROV_CALLER_CONTEXT, PROV_DEFAULT,
                     PROV_UNRESOLVED)


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
    """One claim about the question. State plus the words that produced it.

    CORRECTION 2, earned by the Stage 1 corpus: ``span`` is absent far more
    often than it is present. 170 of the dimension and filter claims raised
    across 690 real-surface questions have no recoverable span, because the
    interpreter that made the claim emitted a field key or a rendered label
    rather than the words. A consumer must therefore treat ``span`` as
    genuinely optional and must never require it.

    Position capture is deliberately NOT added here. Making an interpreter
    emit offsets is a change to that interpreter, which belongs to Stage 3, one
    consumer at a time. Until then the honest record is that the span is
    absent, and ``has_span`` says so.
    """

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

    @property
    def has_span(self) -> bool:
        """Whether this claim can be located in the question at all."""
        return self.span is not None

    def as_dict(self) -> Dict[str, Any]:
        return {"state": self.state, "raw_text": self.raw_text,
                "span": [self.span.start, self.span.end] if self.span else None,
                "has_span": self.has_span,
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


#: What a filter claim actually carries. CORRECTION 1, earned by the Stage 1
#: corpus: on 76 of 690 real-surface questions ONE filter clause is read twice,
#: by two interpreters, each supplying a different half and neither supplying
#: the other's.
#:
#:    WORDING  the words of the clause      — the facet layer supplies this
#:    BOUND    operator and value           — the parser supplies this
#:    FIELD    which field it bears on      — the parser supplies this
#:
#: Before this correction a half-claim was inferred from which attributes
#: happened to be None, which is indistinguishable from "the interpreter looked
#: and found nothing". A claim must say what it knows.
WORDING = "wording"
BOUND = "bound"
FIELD = "field"
CLAIM_CONTENTS = (WORDING, BOUND, FIELD)


@dataclass
class FilterClaim(Slot):
    """A narrowing condition, as the QUESTION states it.

    ``operator`` and ``value`` are the words' own content. The FIELD the
    condition bears on is not here: every `threshold` facet in the release
    candidate carries ``field_key=None`` for the same reason — identifying that
    a clause exists is a different job from resolving what it binds.

    A claim may be a HALF of one clause. ``provides`` says which half, and
    ``clause_id`` is how two halves of the same clause are linked. Stage 1
    established that no existing interpreter emits anything that makes the link
    sound, so ``clause_id`` stays None until an interpreter supplies a basis
    for it — an unjoined pair is reported as unjoined, never guessed at.
    """

    operator: Optional[str] = None
    value: Optional[str] = None
    #: Set when the condition names a dimension VALUE ("in London", "status is
    #: offer") rather than a numeric bound.
    categorical_value: Optional[str] = None
    #: Which halves of the clause this claim carries.
    provides: Tuple[str, ...] = ()
    #: Identity of the clause. Two claims sharing one are two halves of the
    #: same clause. None means "not joined", which is a reportable state and
    #: not an absence.
    clause_id: Optional[str] = None

    def __post_init__(self) -> None:
        super().__post_init__()
        unknown = set(self.provides) - set(CLAIM_CONTENTS)
        if unknown:
            raise ValueError("unknown claim contents %s" % sorted(unknown))

    @property
    def is_half_claim(self) -> bool:
        """True when this claim knows the wording or the binding, not both."""
        has_wording = WORDING in self.provides
        has_binding = bool({BOUND, FIELD} & set(self.provides))
        return has_wording != has_binding

    def as_dict(self) -> Dict[str, Any]:
        d = super().as_dict()
        d.update({"operator": self.operator, "value": self.value,
                  "categorical_value": self.categorical_value,
                  "provides": list(self.provides), "clause_id": self.clause_id,
                  "is_half_claim": self.is_half_claim})
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
class SourceScopeClaim(Slot):
    """Which source portfolio(s) the question scopes to, as its OWNER read it.

    Separate from `population` deliberately, and the separation is the point:

    * a source-portfolio lens and a seasoning segment are DIFFERENT AXES. "the
      front book" is a seasoning population; "the acquired book" is a source
      lens; a question can name both, and neither implies the other.
    * `population` is a LIST of narrowings, and `total` is not a narrowing.
      Putting "no source narrowing" in a list of narrowings is how absence and
      Total become indistinguishable, which is exactly the ambiguity this claim
      exists to remove.

    THE FIVE STATES A CONSUMER MUST BE ABLE TO TELL APART:

        state=FILLED, scope=total      the owner READ the question and found no
                                       source narrowing. Explicitly unrestricted.
        state=FILLED, scope=direct     the direct book
        state=FILLED, scope=acquired   the acquired book
        state=FILLED, scope=cohort     named book(s); `portfolio_ids` carries them
        state=EMPTY                    the owner was NOT consulted. NOT Total.
        state=UNRESOLVABLE             consulted and could not resolve; `reason`
                                       says why. NOT Total.

    PHASE 1E adds the case that made the last state load-bearing: a question
    that NAMES a book the governed registry does not hold ("the Highgate
    Mortgages Book", "acquired_001"). That is UNRESOLVABLE — the owner was
    consulted, read a scope, and could not resolve it — and it is emphatically
    not `scope=total`. Before the owner could tell the two apart, such a
    question was answered for every book under the name of one.

    A consumer that treats EMPTY as Total has widened a population the question
    may have narrowed — the P1L defect. `state` is what distinguishes them, and
    `scope` is meaningful only when `state` is FILLED.
    """

    scope: Optional[str] = None
    #: The explicitly named book ids, when `scope` is `cohort`. Empty otherwise.
    #:
    #: PHASE 1E — these are GOVERNED PORTFOLIO IDS, the identity the registry
    #: keys on, and they are the only identity a consumer may filter or join on.
    #: Phase 1D established that MI's text side used to recognise the STORAGE
    #: convention (`acquired_001`, a blob folder name) and nothing else, so a
    #: consumer that treated whatever wording arrived here as an id would have
    #: been filtering on a name the governed model does not hold.
    portfolio_ids: Tuple[str, ...] = ()
    #: PHASE 1E. The governed DISPLAY LABEL for `portfolio_ids`, when the owner
    #: could supply one — the name React renders in its selector.
    #:
    #: Separate from `raw_text` on purpose, and the separation is the point:
    #: `raw_text` is THE WORDING THAT ASKED ("the alp_acquired book"), and this
    #: is WHAT IT RESOLVED TO ("ALP Acquired Back Book"). They are frequently
    #: different, and a reader owed an explanation of what was answered needs
    #: the second while an audit of what was asked needs the first. Collapsing
    #: them loses one of the two.
    portfolio_label: Optional[str] = None
    #: PHASE 1G. The broad business population this request is about, kept
    #: separate from `portfolio_ids`. See `BASE_POPULATIONS`.
    base_population: Optional[str] = None
    #: PHASE 1G. Where the resolved scope came from. See `SCOPE_PROVENANCES`.
    #: This is the fact Phase 1F stopped for.
    provenance: Optional[str] = None

    @property
    def stated_by_user(self) -> bool:
        """Whether the QUESTION named this scope.

        The precedence fact, exposed as one boolean so a consumer never has to
        re-derive it by comparing strings. True also for an unresolved scope: a
        name MI cannot find is still a name the user said, and forgetting that
        is how it becomes Total.
        """
        return self.provenance in (PROV_EXPLICIT_USER, PROV_UNRESOLVED)

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.scope is not None and self.scope not in SOURCE_SCOPES:
            raise ValueError("unknown source scope %r" % (self.scope,))
        if (self.base_population is not None
                and self.base_population not in BASE_POPULATIONS):
            raise ValueError("unknown base population %r" % (self.base_population,))
        if self.provenance is not None and self.provenance not in SCOPE_PROVENANCES:
            raise ValueError("unknown scope provenance %r" % (self.provenance,))
        # A resolved scope with no provenance is the Phase 1F blocker in object
        # form: a consumer reading it cannot tell a stated scope from a defaulted
        # one, and the two require opposite precedence. Refused rather than
        # allowed to default, because defaulting it here would pick a side.
        if self.state == FILLED and self.provenance is None:
            raise ValueError("a filled source-scope claim must carry provenance")
        if self.state == FILLED and self.scope is None:
            raise ValueError("a filled source-scope claim must name a scope")
        # PHASE 1E. A cohort claim with no id names nothing resolvable. Allowing
        # it would let `narrows` be True with nothing to narrow BY, and the
        # honest reading of that state is UNRESOLVABLE, which the owner is
        # required to state rather than leave to be inferred here.
        if self.state == FILLED and self.scope == SCOPE_COHORT and not self.portfolio_ids:
            raise ValueError("a filled cohort claim must carry at least one "
                             "governed portfolio id")

    @property
    def narrows(self) -> bool:
        """Whether this claim narrows the population at all.

        `total` is a resolved reading AND not a narrowing; both are true and a
        consumer usually needs the second.
        """
        return self.state == FILLED and self.scope != SCOPE_TOTAL

    def as_dict(self) -> Dict[str, Any]:
        d = super().as_dict()
        d.update({"scope": self.scope, "portfolio_ids": list(self.portfolio_ids),
                  "portfolio_label": self.portfolio_label,
                  "base_population": self.base_population,
                  "provenance": self.provenance,
                  "stated_by_user": self.stated_by_user,
                  "narrows": self.narrows})
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
    #: PHASE 1A. Which source portfolio(s) the question scopes to, carried from
    #: `mi_agent.portfolio_lens`. Single-valued: a question has at most one.
    source_scope: SourceScopeClaim = field(default_factory=SourceScopeClaim)
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
            "source_scope": self.source_scope.as_dict(),
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
        for name in ("operation", "subject", "target", "source_scope"):
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
