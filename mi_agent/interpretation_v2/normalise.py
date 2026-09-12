"""One economic meaning, one canonical contract representation.

Three redundancies survived policy recalibration, and all three are the same
kind of fault: the contract lets two different CandidateIntents describe one
economic request, so two readings of one question diverge over a choice that
changes no authorised work. Run 6's adjudication named the first two; the
recalibration probe re-evidenced all three on Q18/Q19/Q20.

    1. PERIOD LABELS IN IDENTITY.   Q19A/B carry ``labels=["last month"]`` and
       Q19C carries ``["month-on-month"]``. Nothing else differs. Those are the
       question's own WORDS, and for a form whose window is fully determined by
       ``form``/``grain``/``periods_back`` they state nothing the contract does
       not already say.

    2. TWO SPELLINGS OF ONE RELATIVE PERIOD.   For an operation that needs two
       periods, ``previous_reporting_period`` can only mean "this period against
       the previous one" — which is what ``relative_pair`` + ``periods_back=1``
       says explicitly. Q20A took the first spelling and Q20B/C the second.

    3. TWO IMPLEMENTATION OWNERS FOR ONE MEASURE.   A specialist measure is
       owned by exactly one capability, yet the model also states ``capability``
       and could name a different one. Choosing between internal owners is not
       the interpreter's job.

WHAT THIS MODULE MAY NOT DO
---------------------------
Normalisation REMOVES representational freedom. It never adds meaning. It does
not re-read the question — there is no question here to read, which is the same
guarantee :mod:`~mi_agent.interpretation_v2.compiler` makes and for the same
reason. It does not widen a population, invent a measure, select a snapshot,
calculate anything, or change which arithmetic runs.

Normalisation 3 is deliberately BOUNDED. It removes the model's authority to
pick an owner, by deriving the owner from measure ownership instead. It does not
collapse ``period_movement``/``current_outstanding_balance`` against
``funded_bridge``/``funded_balance_movement``: those name two different governed
measures, and deciding which one answers "how did the book change?" would be
changing economic meaning, not normalising a representation. See
:data:`BOUNDED` for what is left and why it stops here.

Every rewrite is RECORDED. ``NormalisationResult.applied`` names each one, the
compiler copies them into plan provenance, and the model's original claim still
travels in ``intent_claims``. An audit can always see what the model said and
what the contract then did to it.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import List, Mapping, Optional, Sequence, Tuple

from .intent import CandidateIntent, SemanticTime
from .plan import LABELS_ARE_WORDING_ONLY, identity_labels
from .vocabulary import CHANGE_FORM_CAPABILITY, GovernedVocabulary

__all__ = ["NORMAL_FORM_VERSION", "PAIR_IMPLYING_OPERATIONS",
           "CANONICAL_PAIR_FORM", "CANONICAL_PAIR_PERIODS_BACK", "BOUNDED",
           "LABELS_ARE_WORDING_ONLY", "identity_labels",
           "NormalisationResult", "canonical_intent"]

#: The normal form's own version, separate from the intent schema version: the
#: schema did not change, the canonicalisation of it is new.
NORMAL_FORM_VERSION = "candidate_intent_normal_form/1.0"

# --------------------------------------------------------------------------- #
# 1. period labels
# --------------------------------------------------------------------------- #
#
# Normalisation 1 lives in :mod:`~mi_agent.interpretation_v2.plan`, because it is
# a property of plan IDENTITY rather than of the intent: the plan still carries
# every label it was given, and only the content hash stops depending on them.
# It is re-exported here so all three normalisations are discoverable from one
# place, and so there is exactly one definition of which forms make a label
# load-bearing (see ``plan.LABELS_ARE_WORDING_ONLY`` for that reasoning).

# --------------------------------------------------------------------------- #
# 2. relative period
# --------------------------------------------------------------------------- #

#: Operations that report a change ACROSS periods and do not own their own
#: window. For exactly these, ``previous_reporting_period`` cannot mean one
#: period — a movement over a single snapshot is not a movement — so it means
#: the pair, and the pair has a canonical spelling.
#:
#: This is the compiler's ``_MULTI_PERIOD_OPERATIONS``, minus
#: ``_CAPABILITY_OWNED_PERIOD_OPERATIONS`` (an operation whose window belongs to
#: the capability is not ours to rewrite), minus ``series`` (a series is not a
#: pair). What survives is ``movement``, and a test asserts that derivation
#: against the compiler's own sets so the two cannot drift apart.
#:
#: ``compare`` is deliberately ABSENT. Per the interpreter's contract a
#: comparison is between two POPULATIONS or two DIMENSION VALUES, never between
#: two periods, so "compare, as at the previous period" is a single-period
#: request and rewriting it to a pair would invent a second period nobody asked
#: for.
PAIR_IMPLYING_OPERATIONS: frozenset = frozenset({"movement"})

#: The canonical spelling of "this period against the previous one".
CANONICAL_PAIR_FORM = "relative_pair"
CANONICAL_PAIR_PERIODS_BACK = 1

#: There is a THIRD spelling, and leaving it out made the first attempt at this
#: normalisation converge nothing. ``relative_pair`` with ``periods_back``
#: ABSENT is already treated as complete by the compiler — unlike ``range`` and
#: ``series``, a pair with no distance raises no AMBIGUOUS_PERIOD — which can only
#: mean the adjacent pair. So absent and ``1`` are one relationship written two
#: ways, and Q19C/Q20C differed from their siblings by nothing else.
#:
#: Scoped to ``relative_pair`` deliberately. ``periods_back=0`` is NOT folded in
#: (a pair zero periods apart is not the adjacent pair), and no other form is
#: touched — a forward-looking horizon of absent-versus-zero is a different
#: question, and it belongs to the representational backlog this sprint is told
#: not to broaden into.
PAIR_DISTANCE_IS_IMPLIED_WHEN_ABSENT = True

# --------------------------------------------------------------------------- #
# 3. what stays un-normalised, and why
# --------------------------------------------------------------------------- #

#: Recorded so the boundary is in the source rather than only in a report.
#:
#: ``period_movement`` owns no measures: it reports the period-on-period change
#: of whatever generic measure it is given. ``funded_bridge`` OWNS
#: ``funded_balance_movement`` and decides its own arithmetic. A reading that
#: asks for the first and a reading that asks for the second are asking for two
#: different governed numbers, and they may or may not be the same number —
#: which is the bridge's arithmetic to answer, not this module's.
#:
#: So the owner CHOICE is normalised away and the measure choice is not. What
#: remains is an interpretation difference, and it belongs to interpretation.
BOUNDED: Mapping[str, str] = {
    "movement_measure_choice": (
        "period_movement/current_outstanding_balance against "
        "funded_bridge/funded_balance_movement is a choice between two governed "
        "MEASURES, not between two owners of one measure. Collapsing it would "
        "decide whether a period change is a net movement or a bridge figure, "
        "which is deterministic analytical behaviour this sprint may not change."),
}


@dataclass(frozen=True)
class NormalisationResult:
    """A canonical intent, and the record of how it got that way."""

    intent: CandidateIntent
    applied: Tuple[str, ...] = ()
    normal_form_version: str = NORMAL_FORM_VERSION

    @property
    def changed(self) -> bool:
        return bool(self.applied)


def _owning_capability(intent: CandidateIntent,
                       vocabulary: GovernedVocabulary) -> Optional[str]:
    """The one capability that owns every specialist measure in this intent.

    ``None`` when there is no specialist measure, or when more than one
    capability owns one. Both are cases where ownership does not determine the
    implementation, so nothing may be derived from it. (The compiler separately
    refuses an output that MIXES owners; this only reads them.)
    """
    owners = set()
    saw_measure = False
    for output in intent.effective_outputs():
        for measure in output.measures:
            saw_measure = True
            concept = vocabulary.resolve(measure.concept)
            if concept is not None and concept.owning_capability:
                owners.add(concept.owning_capability)
    if not saw_measure or len(owners) != 1:
        return None
    return next(iter(owners))


def canonical_intent(intent: CandidateIntent,
                     vocabulary: GovernedVocabulary,
                     *,
                     capability_operations: Optional[Mapping[str, frozenset]] = None,
                     ) -> NormalisationResult:
    """Rewrite an intent into the one representation of its economic meaning.

    Pure and total: the same intent always yields the same canonical intent, and
    a canonical intent is its own canonical form (see the idempotence test).
    Nothing here consults the question, and nothing invents a slot value that was
    not already implied by the slots present.

    ``capability_operations`` is the compiler's own capability -> operations map.
    Normalisation 3 consults it to stay outcome-neutral: an owner is only bound
    when it supports the operation already stated, so a rewrite can never turn a
    plan into an unsupported composition.
    """
    applied: List[str] = []

    # -- 2. relative period: one spelling for one relationship -------------- #
    # Done before 1, because it can change `form`, and which labels count as
    # wording depends on the form.
    time = intent.time
    spans_two_periods = intent.operation in PAIR_IMPLYING_OPERATIONS
    if spans_two_periods and time.form == "previous_reporting_period":
        periods_back = (time.periods_back if time.periods_back is not None
                        else CANONICAL_PAIR_PERIODS_BACK)
        intent = replace(intent, time=SemanticTime(
            form=CANONICAL_PAIR_FORM, labels=time.labels, grain=time.grain,
            periods_back=periods_back))
        applied.append(
            f"relative_period: previous_reporting_period -> "
            f"{CANONICAL_PAIR_FORM}+periods_back={periods_back} "
            f"(operation {intent.operation!r} spans two periods)")
        time = intent.time

    # The third spelling: a pair with no distance IS the adjacent pair, because
    # the compiler accepts it as complete and no other reading is available.
    if (PAIR_DISTANCE_IS_IMPLIED_WHEN_ABSENT
            and time.form == CANONICAL_PAIR_FORM
            and time.periods_back is None):
        intent = replace(intent, time=SemanticTime(
            form=time.form, labels=time.labels, grain=time.grain,
            periods_back=CANONICAL_PAIR_PERIODS_BACK))
        applied.append(
            f"relative_period: {CANONICAL_PAIR_FORM} with no distance -> "
            f"periods_back={CANONICAL_PAIR_PERIODS_BACK} (the adjacent pair)")

    # -- 3. implementation owner: derived, not claimed ---------------------- #
    owner = _owning_capability(intent, vocabulary)
    if owner is not None and owner != intent.capability:
        supported = (capability_operations or {}).get(owner)
        if supported is None or intent.operation in supported:
            applied.append(
                f"implementation_owner: capability {intent.capability!r} -> "
                f"{owner!r} (derived from measure ownership)")
            intent = replace(intent, capability=owner)

    # -- 4. the form of a change names its owner ---------------------------- #
    #
    # WHY THIS IS A NORMALISATION AND NOT A DECISION. `change_form` states which
    # analytical question was asked; `CHANGE_FORM_CAPABILITY` states which owner
    # implements that question. Neither is the model's to choose, and putting
    # them together removes freedom rather than adding meaning — the same thing
    # normalisation 3 does for a specialist measure.
    #
    # PRECEDENCE. An explicitly named specialist measure still wins: rule 3 has
    # already bound its owner above, and that owner is an explicit semantic
    # requirement rather than a structural implication. Where the two DISAGREE
    # the intent is left exactly as it is, so the compiler sees the conflict and
    # can refuse or clarify. Silently preferring either one would decide, on the
    # reader's behalf, whether they asked for a decomposition or a delta — which
    # is the substitution this slot exists to end.
    form = getattr(intent, "change_form", None)
    if form:
        implied = CHANGE_FORM_CAPABILITY.get(form)
        conflict = (owner is not None and implied is not None and owner != implied)
        if conflict:
            applied.append(
                f"change_form: {form!r} implies {implied!r} but the measure is "
                f"owned by {owner!r} — left unresolved for the compiler")
        elif implied is not None and implied != intent.capability:
            supported = (capability_operations or {}).get(implied)
            if supported is None or intent.operation in supported:
                applied.append(
                    f"change_form: capability {intent.capability!r} -> "
                    f"{implied!r} (derived from change_form {form!r})")
                intent = replace(intent, capability=implied)

    return NormalisationResult(intent=intent, applied=tuple(applied))
