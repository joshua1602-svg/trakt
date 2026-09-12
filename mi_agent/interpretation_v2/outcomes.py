"""Typed compilation outcomes and the governed reason codes that explain them.

Compilation is NOT binary. A question that cannot become an executable plan is
not an exception — it is a governed answer of its own, and the reason has to be
machine-readable so a caller can act on it rather than parse prose.

Three outcomes only::

    PLAN      a GovernedQueryPlan was produced
    CLARIFY   the intent is well-formed but under-determined; a human can settle it
    REFUSE    the intent cannot be honoured on this book, at all

The distinction between CLARIFY and REFUSE is deliberate and is the compiler's
to make, never the model's. CLARIFY means *the same question, answered more
precisely, would compile*. REFUSE means it would not: the concept is not
registered, the book does not carry it, or the composition is not supported.
Neither ever degrades into "the nearest thing we could run".

No user-facing prose is invented here. A reason carries a code, the subject it
is about, and the source spans (if the model supplied any) that led to it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

# --------------------------------------------------------------------------- #
# Outcomes
# --------------------------------------------------------------------------- #

OUTCOME_PLAN = "PLAN"
OUTCOME_CLARIFY = "CLARIFY"
OUTCOME_REFUSE = "REFUSE"

OUTCOMES = (OUTCOME_PLAN, OUTCOME_CLARIFY, OUTCOME_REFUSE)


# --------------------------------------------------------------------------- #
# Governed reason codes
# --------------------------------------------------------------------------- #
# Grouped by the stage that raises them. A code is added here before it is
# raised anywhere, so the vocabulary of refusals is enumerable.

# -- model boundary --------------------------------------------------------- #
#: The interpreter could not be invoked at all (no key, no client, transport
#: failure). A plan is never produced from an absent model.
MODEL_UNAVAILABLE = "MODEL_UNAVAILABLE"
#: The model returned something that is not a CandidateIntent.
MODEL_OUTPUT_MALFORMED = "MODEL_OUTPUT_MALFORMED"
#: The model returned executable code, SQL or a dataframe expression somewhere
#: in its output. Fail closed — never strip it and continue.
MODEL_OUTPUT_CONTAINS_CODE = "MODEL_OUTPUT_CONTAINS_CODE"
#: The model tried to supply a physical binding (a snapshot id, an explicit
#: date, a canonical column) in a slot that is semantic only.
MODEL_OUTPUT_CONTAINS_PHYSICAL_BINDING = "MODEL_OUTPUT_CONTAINS_PHYSICAL_BINDING"

# -- intent validation ------------------------------------------------------ #
INTENT_SCHEMA_INVALID = "INTENT_SCHEMA_INVALID"
INTENT_SCHEMA_VERSION_UNSUPPORTED = "INTENT_SCHEMA_VERSION_UNSUPPORTED"
MISSING_REQUIRED_SLOT = "MISSING_REQUIRED_SLOT"
CONFLICTING_CLAIMS = "CONFLICTING_CLAIMS"

# -- binding ---------------------------------------------------------------- #
#: The term is not in the governed semantic vocabulary at all.
UNREGISTERED_CONCEPT = "UNREGISTERED_CONCEPT"
#: The term is governed, but this book does not carry the data behind it.
CONCEPT_UNAVAILABLE = "CONCEPT_UNAVAILABLE"
#: The term is governed but resolves to more than one governed binding, and
#: nothing in the intent chooses between them.
AMBIGUOUS_MEASURE = "AMBIGUOUS_MEASURE"
AMBIGUOUS_DIMENSION = "AMBIGUOUS_DIMENSION"
AMBIGUOUS_POPULATION = "AMBIGUOUS_POPULATION"
AMBIGUOUS_PERIOD = "AMBIGUOUS_PERIOD"
AMBIGUOUS_GEOGRAPHY = "AMBIGUOUS_GEOGRAPHY"
#: The model itself flagged the question as needing clarification.
MODEL_FLAGGED_AMBIGUITY = "MODEL_FLAGGED_AMBIGUITY"

# -- composition authorisation ---------------------------------------------- #
UNSUPPORTED_OPERATION = "UNSUPPORTED_OPERATION"
UNSUPPORTED_STATISTIC = "UNSUPPORTED_STATISTIC"
UNSUPPORTED_COMPOSITION = "UNSUPPORTED_COMPOSITION"
UNSUPPORTED_FILTER = "UNSUPPORTED_FILTER"
WEIGHT_NOT_PERMITTED = "WEIGHT_NOT_PERMITTED"
CAPABILITY_UNAVAILABLE = "CAPABILITY_UNAVAILABLE"
#: The reader's question was understood, and the owner that would answer it is
#: not built yet. Distinct from CAPABILITY_UNAVAILABLE, which says a capability
#: is not registered on this book: this says the ANALYTICAL FORM has no
#: implementation anywhere. Kept separate because the two call for different
#: things — one is an onboarding fact, the other is a roadmap fact — and because
#: answering either with the nearest available analysis is the substitution the
#: change_form slot exists to end.
CHANGE_FORM_NOT_CONNECTED = "CHANGE_FORM_NOT_CONNECTED"
INVALID_GEOGRAPHY_BASIS = "INVALID_GEOGRAPHY_BASIS"
PERIOD_UNRESOLVED = "PERIOD_UNRESOLVED"

#: Every code this package may emit. A code outside this set is a bug, and the
#: CompileResult constructor says so rather than passing it through.
REASON_CODES = frozenset({
    MODEL_UNAVAILABLE,
    MODEL_OUTPUT_MALFORMED,
    MODEL_OUTPUT_CONTAINS_CODE,
    MODEL_OUTPUT_CONTAINS_PHYSICAL_BINDING,
    INTENT_SCHEMA_INVALID,
    INTENT_SCHEMA_VERSION_UNSUPPORTED,
    MISSING_REQUIRED_SLOT,
    CONFLICTING_CLAIMS,
    UNREGISTERED_CONCEPT,
    CONCEPT_UNAVAILABLE,
    AMBIGUOUS_MEASURE,
    AMBIGUOUS_DIMENSION,
    AMBIGUOUS_POPULATION,
    AMBIGUOUS_PERIOD,
    AMBIGUOUS_GEOGRAPHY,
    MODEL_FLAGGED_AMBIGUITY,
    UNSUPPORTED_OPERATION,
    UNSUPPORTED_STATISTIC,
    UNSUPPORTED_COMPOSITION,
    UNSUPPORTED_FILTER,
    WEIGHT_NOT_PERMITTED,
    CAPABILITY_UNAVAILABLE,
    CHANGE_FORM_NOT_CONNECTED,
    INVALID_GEOGRAPHY_BASIS,
    PERIOD_UNRESOLVED,
})

#: Codes whose honest outcome is "ask, don't guess". Everything else refuses.
#: A code appears in exactly one of these two sets — the split is the whole
#: point of having both outcomes, so it is data, not a judgement made per call.
CLARIFIABLE_CODES = frozenset({
    AMBIGUOUS_MEASURE,
    AMBIGUOUS_DIMENSION,
    AMBIGUOUS_POPULATION,
    AMBIGUOUS_PERIOD,
    AMBIGUOUS_GEOGRAPHY,
    MODEL_FLAGGED_AMBIGUITY,
    MISSING_REQUIRED_SLOT,
    INVALID_GEOGRAPHY_BASIS,
})


@dataclass(frozen=True)
class CompileReason:
    """One machine-readable reason a compilation did not produce a plan.

    ``subject`` names the thing the reason is about (a semantic term, a slot
    name, a capability id) so a caller can address it without string-matching
    the code. ``spans`` are the model's own source spans, carried only as
    provenance — nothing in the compiler reads them to decide anything.
    """

    code: str
    subject: str = ""
    detail: str = ""
    spans: Tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.code not in REASON_CODES:
            raise ValueError(f"ungoverned reason code: {self.code!r}")

    @property
    def clarifiable(self) -> bool:
        return self.code in CLARIFIABLE_CODES

    def to_dict(self) -> Dict[str, Any]:
        return {"code": self.code, "subject": self.subject,
                "detail": self.detail, "spans": list(self.spans)}


def outcome_for(reasons: Sequence[CompileReason]) -> str:
    """The governed outcome implied by a set of reasons.

    A single non-clarifiable reason refuses the whole compilation: a question
    that is partly unanswerable is not answered in part. Only when EVERY reason
    is clarifiable is the honest outcome "ask".
    """
    if not reasons:
        return OUTCOME_PLAN
    if all(r.clarifiable for r in reasons):
        return OUTCOME_CLARIFY
    return OUTCOME_REFUSE


@dataclass(frozen=True)
class CompileResult:
    """What the deterministic compiler produced, and why.

    ``plan`` is present if and only if ``outcome == PLAN``. That invariant is
    enforced here rather than trusted, because "a refusal that still carries a
    runnable plan" is exactly the failure this architecture exists to prevent.
    """

    outcome: str
    plan: Optional[Any] = None                 # GovernedQueryPlan (avoid a cycle)
    reasons: Tuple[CompileReason, ...] = ()
    intent: Optional[Any] = None               # CandidateIntent, for provenance
    compiler_version: str = ""

    def __post_init__(self) -> None:
        if self.outcome not in OUTCOMES:
            raise ValueError(f"unknown outcome: {self.outcome!r}")
        if self.outcome == OUTCOME_PLAN and self.plan is None:
            raise ValueError("outcome PLAN with no plan")
        if self.outcome != OUTCOME_PLAN and self.plan is not None:
            raise ValueError(f"outcome {self.outcome} must not carry a plan")

    @property
    def is_plan(self) -> bool:
        return self.outcome == OUTCOME_PLAN

    def codes(self) -> List[str]:
        return [r.code for r in self.reasons]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "outcome": self.outcome,
            "reasons": [r.to_dict() for r in self.reasons],
            "compiler_version": self.compiler_version,
            "plan": self.plan.to_dict() if self.plan is not None else None,
        }


def refuse(*reasons: CompileReason, intent: Any = None,
           compiler_version: str = "") -> CompileResult:
    """A non-plan outcome, with the outcome chosen by the reasons themselves."""
    reasons = tuple(reasons)
    if not reasons:
        raise ValueError("a non-plan outcome needs at least one reason")
    return CompileResult(outcome=outcome_for(reasons), plan=None, reasons=reasons,
                         intent=intent, compiler_version=compiler_version)
