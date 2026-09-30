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
from .vocabulary import (CAPABILITY_CHANGE_FORMS,
                         CHANGE_FORM_CANONICAL_OPERATION,
                         CHANGE_FORM_CAPABILITY,
                         CHANGE_FORM_OPERATION_VARIANTS,
                         POPULATION_CHANGE_OWNER,
                         GovernedVocabulary)

__all__ = ["NORMAL_FORM_VERSION", "PAIR_IMPLYING_OPERATIONS",
           "CANONICAL_PAIR_FORM", "CANONICAL_PAIR_PERIODS_BACK", "BOUNDED",
           "DERIVED_POPULATION_OF", "DERIVED_POPULATION_SPELLINGS",
           "SINGLE_FIGURE_OPERATION", "GROUPED_FIGURE_OPERATION",
           "LABELS_ARE_WORDING_ONLY", "identity_labels",
           "NormalisationResult", "canonical_intent"]

#: The normal form's own version, separate from the intent schema version: the
#: schema did not change, the canonicalisation of it is new.
NORMAL_FORM_VERSION = "candidate_intent_normal_form/1.1"

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
    "forecast_of_the_pipeline": (
        "A forecast intent on the PIPELINE base asks what the pipeline alone "
        "converts into, which is a different figure from the funded book "
        "projected forward. Rule 6 leaves it as stated, so the forecast runtime "
        "refuses it by population rather than answering it as the whole "
        "forecast."),
}

# --------------------------------------------------------------------------- #
# 6. a derived population has one spelling
# --------------------------------------------------------------------------- #

#: THE POPULATION A CAPABILITY'S OUTPUT IS. `forecast` is not a dataset anyone
#: loads: it is the funded book projected forward from funded and pipeline
#: inputs (P0 design §4.1), and it is the only population the forecast runtime
#: executes. So a forecast intent names its population twice — once in
#: `capability`, once in `population.base` — and the production bank shows the
#: model spelling the second one two ways for one question: "when does the book
#: reach £100m" arrived five times as `forecast` and four times as `funded`,
#: and the four were refused for a population nobody asked to change.
DERIVED_POPULATION_OF: Mapping[str, str] = {"forecast": "forecast"}

#: WHICH STATED BASES ARE SPELLINGS OF THE DERIVED ONE, and which are not.
#: `funded` is: under the forecast capability it means "the funded book, going
#: forward", which is what `forecast` means. It is also the schema's DEFAULT,
#: so an intent that stated no base arrives as `funded` and means the same.
#:
#: `pipeline` is NOT, and neither is `whole_book`. "Of the current offer
#: pipeline, how much should convert?" asks for the pipeline's contribution on
#: its own; rewriting it to `forecast` would answer with funded plus pipeline,
#: which is a wider population than the one named. Those stay as stated and are
#: refused by the runtime that cannot execute them — see
#: `BOUNDED["forecast_of_the_pipeline"]`.
DERIVED_POPULATION_SPELLINGS: frozenset = frozenset({"funded"})


# --------------------------------------------------------------------------- #
# 7. a grouped figure is a breakdown
# --------------------------------------------------------------------------- #

#: "When are pipeline cases expected to complete?" (must-answer [135], the
#: 2026-09-30 10:26 check) arrived as `point_in_time` over
#: `expected_completion_date` GROUPED BY `origin_stage`, and was refused: the
#: compiler forbids a grouping on a single figure because "a grouping makes the
#: output a breakdown, not a point_in_time" (compiler `_GROUPING_FORBIDDEN`).
#: The compiler had already named the meaning. The operation label and the
#: grouping state ONE thing twice — how many figures the answer is — and the
#: grouping is the slot that says it, so the label has one canonical value.
#: This is the same redundancy as rule 5's linguistic variants: one request,
#: two spellings, one of them refused.
#:
#: WHAT IT MAY NOT DO. It reads the operation and whether every output groups;
#: it adds no dimension, drops no filter and picks no member. It fires only when
#: EVERY output groups (a mixed intent keeps its refusal, because rewriting it
#: would make the ungrouped output the unsupported one) and only when the
#: capability produces a breakdown (`limit_assessment` does not, and keeps its
#: refusal). A question about one member keeps its filter, so the breakdown is
#: that member's row. A ranking is not inferred: "which stage completes first"
#: needs an order the reading did not state, and inventing one would be adding
#: meaning.
SINGLE_FIGURE_OPERATION = "point_in_time"
GROUPED_FIGURE_OPERATION = "breakdown"


def _every_output_groups(intent: CandidateIntent) -> bool:
    """The compiler's own test of "grouped", read from the intent's slots.

    An output groups when it names a dimension or a geography grouping, its own
    or the top-level one it inherits (compiler `_authorise_composition`).
    """
    top = intent.geography.requested and intent.geography.group_by
    outputs = intent.effective_outputs()
    return bool(outputs) and all(
        bool(output.dimensions) or top
        or (output.geography.requested and output.geography.group_by)
        for output in outputs)


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
            periods_back=periods_back,
            # A rewrite changes the SPELLING of a temporal form the reading
            # stated; it cannot unstate it. Dropping this would make a
            # normalised explicit pair look like a reading that said nothing,
            # and the deterministic layer would then apply a default over the
            # top of the reader's own words.
            stated=time.stated))
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
            periods_back=CANONICAL_PAIR_PERIODS_BACK, stated=time.stated))
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
    form_conflict = False
    # The measure's owner may implement this form over its own measures (the
    # pipeline's change between two dated extracts): then the owner stands and
    # there is nothing to reconcile — the form still decides the operation.
    owner_implements = bool(form and owner is not None
                            and form in CAPABILITY_CHANGE_FORMS.get(owner, ()))
    # With NO measure named at all, the POPULATION names the owner: "what
    # moved in the pipeline" is the pipeline's to implement, where it
    # implements the form (D15) — never the funded book's owner of the same
    # form. A named measure keeps rule 3's ownership: a funded measure asked of
    # the pipeline is refused for its population, not re-homed.
    names_a_measure = any(output.measures
                          for output in intent.effective_outputs())
    population_owner = (None if names_a_measure else POPULATION_CHANGE_OWNER.get(
        getattr(intent.population, "base", None) or ""))
    if (form and owner is None and population_owner
            and form in CAPABILITY_CHANGE_FORMS.get(population_owner, ())):
        owner_implements = population_owner == intent.capability
    if form and not owner_implements:
        implied = CHANGE_FORM_CAPABILITY.get(form)
        if (owner is None and population_owner
                and form in CAPABILITY_CHANGE_FORMS.get(population_owner, ())):
            implied = population_owner
        form_conflict = (owner is not None and implied is not None
                         and owner != implied)
        if form_conflict:
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

    # -- 5. one analytical form, one canonical operation -------------------- #
    #
    # WHY THIS IS A NORMALISATION AND NOT A DECISION, AGAIN. `change_form`
    # defines the analytical form; `operation` expresses the requested action
    # WITHIN that form. Where the two overlap, the form is authoritative over
    # ownership and over who decides the candidate set, and the operation is
    # authoritative over the result shape. A form whose action has several
    # equally correct spellings therefore has ONE canonical operation, and the
    # other spellings collapse to it. Like rules 3 and 4 this removes freedom
    # rather than adding meaning: after it, one analytical form has exactly one
    # executable shape, and a runtime cannot be handed two.
    #
    # STRUCTURED SLOTS ONLY. The form and the operation are read; the question,
    # the provenance text and the evidence are not — this module imports no
    # parser and no `re`, and could not inspect wording if it wanted to.
    #
    # THIS IS NOT A DEFAULT FROM `movement` TO `material_summary`. It fires only
    # where the form was ALREADY STATED by the interpreter. An intent with no
    # `change_form` is untouched here and refused or clarified by the ordinary
    # contract: whether a bare "what changed" IS a material summary is an
    # interpretation question, and the compiler does not guess it.
    #
    # AND NOT WHERE RULE 4 LEFT A CONFLICT. If a named specialist measure
    # disagrees with the form about the owner, rule 4 deliberately left the
    # intent as it stands so the compiler can see the disagreement. Rewriting
    # the operation on top of that would half-resolve it, which is worse than
    # either resolution.
    canonical = CHANGE_FORM_CANONICAL_OPERATION.get(form or "")
    if canonical and not form_conflict and intent.operation != canonical:
        if intent.operation in CHANGE_FORM_OPERATION_VARIANTS.get(form, frozenset()):
            applied.append(
                f"change_form: operation {intent.operation!r} -> {canonical!r} "
                f"(a linguistic variant of the action {form!r} names)")
            intent = replace(intent, operation=canonical)
        # An operation OUTSIDE the variant set states a shape this form's owner
        # does not produce. It is left exactly as it is, so the compiler refuses
        # it against the capability's own operation set rather than this module
        # silently flattening it into a summary.

    # -- 6. a derived population has one spelling --------------------------- #
    #
    # AFTER RULES 3-5, because they can set the capability: an intent naming
    # `forecast_milestone_date` under `generic_analysis` becomes a forecast
    # intent in rule 3, and its base is then this rule's to canonicalise.
    #
    # STRUCTURED SLOTS ONLY, and NO WIDENING. The capability and the stated base
    # are read; nothing else is. Only a base listed in
    # `DERIVED_POPULATION_SPELLINGS` is rewritten, lens, seasoning and a named
    # source travel unchanged, and a pipeline base is left for the runtime to
    # refuse. The model's original base still travels in `intent_claims`.
    derived = DERIVED_POPULATION_OF.get(intent.capability)
    population = intent.population
    if (derived is not None and population.base != derived
            and population.base in DERIVED_POPULATION_SPELLINGS):
        applied.append(
            f"derived_population: base {population.base!r} -> {derived!r} "
            f"(capability {intent.capability!r} outputs the {derived!r} "
            f"population)")
        intent = replace(intent, population=replace(population, base=derived))

    # -- 7. a grouped figure is a breakdown ---------------------------------- #
    #
    # LAST, because rules 3-4 can set the capability, and whether the owner
    # produces a breakdown is this rule's condition. STRUCTURED SLOTS ONLY: the
    # operation, the groupings and the capability's own operation set.
    if (intent.operation == SINGLE_FIGURE_OPERATION
            and _every_output_groups(intent)):
        supported = (capability_operations or {}).get(intent.capability)
        if supported is None or GROUPED_FIGURE_OPERATION in supported:
            applied.append(
                f"grouped_figure: operation {SINGLE_FIGURE_OPERATION!r} -> "
                f"{GROUPED_FIGURE_OPERATION!r} (every output is grouped, so "
                f"the answer is one figure per member)")
            intent = replace(intent, operation=GROUPED_FIGURE_OPERATION)

    return NormalisationResult(intent=intent, applied=tuple(applied))
