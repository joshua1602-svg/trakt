"""The deterministic compiler: CandidateIntent -> GovernedQueryPlan | Refuse | Clarify.

THE ONE RULE THIS MODULE OBEYS
------------------------------
**The compiler never re-reads the question.**

Once a CandidateIntent exists, the English is finished with. There is no regex
over the original sentence here, no recogniser cascade, no second opinion drawn
from wording. This module does not import ``re``, and
``tests/interpretation_v2/test_compiler_does_not_reread_language.py`` asserts
both that (structurally) and the behaviour it protects: the same intent carrying
two completely different question strings compiles to the same plan, byte for
byte, apart from the provenance that records which question it was.

Source-span evidence may be COPIED into provenance. It is never read to decide
anything.

WHAT IT OWNS
------------
    validate     is the intent well-formed, complete, self-consistent?
    bind         semantic term -> governed concept -> canonical field
    authorise    is this capability/operation/statistic/composition supported?
    produce      a GovernedQueryPlan, or a typed refusal/clarification

WHAT IT REFUSES TO DO
---------------------
Choose the nearest available semantics to avoid a refusal. Every degradation
path a parser would normally take — the almost-matching field, the neighbouring
statistic, the whole book when the population could not be honoured — is a
governed reason code here instead. Where the compiler DOES supply a value the
question did not state, it is a registry default, it is recorded on the binding
as defaulted, and it is listed in the plan's provenance notes.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any, Dict, FrozenSet, Iterable, List, Mapping, Optional, Sequence, Tuple

from .intent import (
    CandidateIntent,
    RequestedOutput,
    SemanticFilter,
    SemanticGeography,
    SemanticMeasure,
)
from .normalise import NORMAL_FORM_VERSION, canonical_intent
from .outcomes import (
    AMBIGUOUS_DIMENSION,
    AMBIGUOUS_GEOGRAPHY,
    AMBIGUOUS_MEASURE,
    AMBIGUOUS_PERIOD,
    AMBIGUOUS_POPULATION,
    CAPABILITY_UNAVAILABLE,
    CHANGE_FORM_NOT_CONNECTED,
    CONCEPT_UNAVAILABLE,
    CONFLICTING_CLAIMS,
    INVALID_GEOGRAPHY_BASIS,
    MISSING_REQUIRED_SLOT,
    MODEL_FLAGGED_AMBIGUITY,
    PERIOD_UNRESOLVED,
    UNREGISTERED_CONCEPT,
    UNSUPPORTED_COMPOSITION,
    UNSUPPORTED_FILTER,
    UNSUPPORTED_OPERATION,
    UNSUPPORTED_STATISTIC,
    WEIGHT_NOT_PERMITTED,
    CompileReason,
    CompileResult,
    OUTCOME_PLAN,
    outcome_for,
)
from .plan import (
    PLAN_SCHEMA_VERSION,
    DimensionBinding,
    FilterBinding,
    GeographyBinding,
    GovernedQueryPlan,
    MeasureBinding,
    OutputPlan,
    PeriodBinding,
    PlanProvenance,
    PopulationBinding,
    TargetBinding,
)
from .vocabulary import (
    CHANGE_FORM_CAPABILITY,
    CHANGE_FORM_MODE,
    CAPABILITY_OPERATIONS,
    STATISTICS_FORBIDDING_WEIGHT,
    STATISTICS_REQUIRING_WEIGHT,
    GovernedVocabulary,
    SemanticConcept,
    load_governed_vocabulary,
)

COMPILER_VERSION = "interpretation_v2.compiler/1.0.0"


# --------------------------------------------------------------------------- #
# Governed contracts the compiler applies
# --------------------------------------------------------------------------- #

#: (level, basis) -> canonical region field. Mirrors the three field families
#: governed by ``mi_agent.region_basis``; a pair absent from this table is not a
#: geography Trakt can measure, and says so rather than picking a neighbour.
_GEOGRAPHY_CONTRACT: Mapping[Tuple[str, str], str] = {
    ("reporting", "reporting_taxonomy"): "canonical_region_reporting",
    ("reporting", "collateral"): "canonical_region_reporting",
    ("nuts3", "obligor"): "geographic_region_obligor",
    ("nuts3", "collateral"): "geographic_region_collateral",
    ("itl3", "obligor"): "geographic_region_obligor_itl3",
    ("itl3", "collateral"): "geographic_region_collateral_itl3",
    ("postcode", "collateral"): "postcode",
    ("postcode", "reporting_taxonomy"): "postcode",
}

#: The level a bare "by region" resolves to. It is the client's approved
#: harmonised reporting taxonomy — the governed answer to the word "region",
#: recorded as a default on the binding so an answer can disclose it.
_DEFAULT_GEOGRAPHY_LEVEL = "reporting"
_DEFAULT_GEOGRAPHY_BASIS = "reporting_taxonomy"

#: Levels at which a basis MUST be stated: the borrower's region and the
#: property's region are different answers at NUTS3 and ITL3, and the book
#: carries both. Guessing between them is exactly the substitution this
#: architecture exists to stop.
_BASIS_REQUIRED_LEVELS = frozenset({"nuts3", "itl3"})

#: Semantic time form -> the governed period contract that resolves it, and
#: whether this package can consider it settled without touching a book.
_PERIOD_CONTRACT: Mapping[str, Tuple[str, bool]] = {
    "current": ("latest_governed_reporting_period", True),
    "previous_reporting_period": ("previous_governed_reporting_period", True),
    "relative_pair": ("governed_reporting_period_pair", True),
    "explicit_period": ("named_period_span", False),
    "range": ("governed_period_span", False),
    "series": ("governed_period_series", False),
    "forward_looking": ("forecast_horizon", False),
}

#: A forward-looking period is only meaningful where a capability owns a
#: forward methodology. Asking a point-in-time capability about the future is an
#: unsupported composition, not a period that happens to be empty.
_FORWARD_CAPABILITIES = frozenset({"forecast", "limit_assessment", "pipeline"})

#: Operations that require at least one grouping (a dimension or a geography).
_GROUPING_REQUIRED = frozenset({"breakdown", "rank", "distribution"})

#: Capabilities that OWN their own grouping. "Where are our largest
#: concentrations?" is a rank over the governed concentration tests, and which
#: tests those are is the capability's to know — demanding the question name a
#: dimension would be asking the reader to know the methodology, which is the
#: thing specialist ownership exists to prevent.
_CAPABILITIES_OWNING_GROUPING = frozenset({
    "concentration", "limit_assessment", "portfolio_summary",
    "pipeline_stage_movement", "borrowing_base",
})

#: Operations whose PERIOD WINDOW belongs to the capability, not the question.
#: "How many cases moved from KFI to Application?" is a complete question: the
#: movement window is part of what a stage transition IS. Requiring the intent
#: to state two periods made the first live run refuse eleven well-formed
#: pipeline questions — the compiler demanding that the model specify a
#: methodology detail it was deliberately never shown.
_CAPABILITY_OWNED_PERIOD_OPERATIONS = frozenset({
    "transition", "arrivals", "departures", "stayers", "reconciliation",
    "bridge",
})

#: Operations for which a grouping changes the question into a different one.
_GROUPING_FORBIDDEN = frozenset({"point_in_time"})

#: Operations that need no measure — the capability decides what it reports.
_MEASURE_OPTIONAL = frozenset({
    "summary", "reconciliation", "eligibility", "bridge", "departures",
    "forecast_milestone", "utilisation", "headroom",
})

#: Operations that need a period richer than a single point.
_MULTI_PERIOD_OPERATIONS = frozenset({
    "series", "movement", "bridge", "transition", "arrivals", "departures",
    "stayers", "reconciliation",
})
_MULTI_PERIOD_FORMS = frozenset({
    "previous_reporting_period", "relative_pair", "explicit_period", "range",
    "series",
})

#: Comparator -> the shape its value must have. A comparator whose value is the
#: wrong shape is an unsupported filter, never a coerced one.
_COMPARATOR_ARITY: Mapping[str, str] = {
    "gt": "scalar", "gte": "scalar", "lt": "scalar", "lte": "scalar",
    "eq": "scalar", "ne": "scalar",
    "between": "pair", "in": "list", "not_in": "list",
}

#: Comparators that only make sense against something ordered.
_ORDERED_COMPARATORS = frozenset({"gt", "gte", "lt", "lte", "between"})

#: Registry formats that ARE ordered. A dimension can still be numeric —
#: "number of borrowers >= 2" is an ordinary predicate on a governed dimension —
#: so the ordered check reads the format, not just the role. Only a categorical
#: or boolean value has no order at all.
_ORDERED_FORMATS = frozenset({"integer", "number", "float", "decimal",
                              "currency", "percentage", "percent", "date",
                              "datetime", "year"})

#: The governed field carrying each population axis. The compiler picks these;
#: the model names only the semantic axis value.
_POPULATION_LENS_FIELD = "source_portfolio_type"
#: The governed IDENTITY column a named source portfolio binds to. Distinct from
#: the role column above: one says which book, the other what kind.
_SOURCE_ID_FIELD = "source_portfolio_id"
_POPULATION_SEASONING_FIELD = "seasoning_segment"
_POPULATION_BASE_FIELD = "funded_status"


class CompilerContext:
    """The governed context one compilation runs against.

    The vocabulary here is the FULL governed one, never the book-narrowed view.
    That is deliberate: the compiler has to be able to tell "Trakt does not
    govern this concept" from "Trakt governs it and this book does not carry
    it", and a vocabulary with the absent concepts removed can only say the
    first. Telling a client that Trakt has never heard of an indexed LTV, when
    the truth is that their tape does not carry one, is a different and worse
    answer.

    Book narrowing belongs to what the INTERPRETER is shown
    (``GovernedVocabulary.for_book``), so the model is not offered concepts this
    book cannot answer.

    Immutable in practice and deterministic by construction.
    """

    __slots__ = ("vocabulary", "capabilities", "available_fields", "book_id",
                 "interpreter_vocabulary", "source_registry")

    def __init__(self, vocabulary: Optional[GovernedVocabulary] = None, *,
                 capabilities: Optional[Iterable[str]] = None,
                 available_fields: Optional[Iterable[str]] = None,
                 book_id: Optional[str] = None,
                 source_registry: Any = None) -> None:
        self.vocabulary = vocabulary or load_governed_vocabulary()
        self.capabilities: FrozenSet[str] = (
            frozenset(capabilities) & self.vocabulary.capabilities
            if capabilities is not None else self.vocabulary.capabilities)
        self.available_fields: FrozenSet[str] = frozenset(
            str(f).strip() for f in (available_fields or ()) if f)
        #: THE CLIENT'S GOVERNED SOURCE PORTFOLIOS, for binding a named source.
        #: A `trakt_core.portfolio.PortfolioRegistry`, or None where the caller
        #: supplied none — in which case a question naming a source is REFUSED,
        #: because a name that cannot be checked against anything is a name
        #: nobody governed. It is the same kind of per-book knowledge as
        #: `available_fields` and `book_id`, and it is client-scoped by
        #: construction: a registry is built for one client.
        self.source_registry = source_registry
        self.book_id = book_id
        #: The narrowed view to hand an interpreter for this book.
        self.interpreter_vocabulary = (
            self.vocabulary.for_book(self.available_fields, book_id=book_id,
                                     capabilities=self.capabilities)
            if self.available_fields or capabilities is not None
            else self.vocabulary)

    def field_present(self, canonical_field: Optional[str]) -> bool:
        if canonical_field is None:
            return True
        if not self.available_fields:
            return True
        return canonical_field in self.available_fields


class DeterministicCompiler:
    """Compiles a CandidateIntent against one governed context.

    Stateless between calls. Holding no state is what makes rule I — the same
    intent always compiles the same way — a property of the design rather than a
    thing that happens to be true today.
    """

    version = COMPILER_VERSION

    def __init__(self, context: Optional[CompilerContext] = None) -> None:
        self.context = context or CompilerContext()

    # -- entry point -------------------------------------------------------- #

    def compile(self, intent: CandidateIntent) -> CompileResult:
        reasons: List[CompileReason] = []
        notes: List[str] = []

        # 0. NORMALISE ------------------------------------------------------ #
        # One economic meaning, one contract representation. This only REMOVES
        # representational freedom: it rewrites a redundant spelling into the
        # canonical one and derives an implementation owner the model should
        # never have been choosing. The model's own claim is still recorded
        # verbatim in provenance, so an audit can see both.
        claimed = intent
        normalisation = canonical_intent(
            intent, self.context.vocabulary,
            capability_operations=CAPABILITY_OPERATIONS)
        intent = normalisation.intent
        notes.extend(f"normalised [{NORMAL_FORM_VERSION}]: {applied}"
                     for applied in normalisation.applied)

        # A. VALIDATE ------------------------------------------------------- #
        reasons.extend(self._validate(intent))

        # B/C. BIND + AUTHORISE --------------------------------------------- #
        capability = intent.capability
        if capability not in self.context.capabilities:
            reasons.append(CompileReason(CAPABILITY_UNAVAILABLE, capability,
                                         "not registered on this book"))
        allowed_ops = CAPABILITY_OPERATIONS.get(capability, frozenset())
        if allowed_ops and intent.operation not in allowed_ops:
            reasons.append(CompileReason(
                UNSUPPORTED_OPERATION, intent.operation,
                f"not an operation of capability {capability!r}"))

        # AN ANALYTICAL FORM WITH NO OWNER IS REFUSED, NOT SUBSTITUTED.
        # `CHANGE_FORM_CAPABILITY` maps every form to the capability that
        # implements it, and one of them maps to nothing: the funded
        # material-change composition is not built. The reader asked what
        # changed; answering with a balance delta, a portfolio overview or a
        # bridge would be answering a different question, which is precisely
        # what the change_form slot exists to stop. So the question is
        # understood, recorded, and declined.
        form = getattr(intent, "change_form", None)
        if form and form in CHANGE_FORM_CAPABILITY \
                and CHANGE_FORM_CAPABILITY[form] is None:
            reasons.append(CompileReason(
                CHANGE_FORM_NOT_CONNECTED, form,
                f"the question was read as {form!r}, and no governed owner "
                f"implements that analytical form yet"))

        for disclosed in intent.ambiguity:
            if not disclosed.blocking:
                notes.append(f"disclosed reading [{disclosed.slot}]: {disclosed.note}")

        population, pop_reasons, pop_notes = self._bind_population(intent)
        reasons.extend(pop_reasons)
        notes.extend(pop_notes)

        period, period_reasons = self._bind_period(intent)
        reasons.extend(period_reasons)

        top_geography, geo_reasons, geo_notes = self._bind_geography(
            intent.geography, slot="geography")
        reasons.extend(geo_reasons)
        notes.extend(geo_notes)

        top_filters, filter_reasons = self._bind_filters(intent.filters,
                                                         slot="filters")
        reasons.extend(filter_reasons)

        target, target_reasons = self._bind_target(intent)
        reasons.extend(target_reasons)

        outputs: List[OutputPlan] = []
        for output in intent.effective_outputs():
            plan_output, out_reasons, out_notes = self._bind_output(
                output, intent=intent, inherited_geography=top_geography)
            reasons.extend(out_reasons)
            notes.extend(out_notes)
            if plan_output is not None:
                outputs.append(plan_output)

        # A geography that was ASKED FOR but could not be bound already has its
        # own reason. Letting the composition check add "this breakdown has
        # nothing to group by" on top would bury the root cause under a harsher
        # code and turn a clarification into a refusal.
        grouping_intended = intent.geography.requested and intent.geography.group_by
        reasons.extend(self._authorise_composition(
            intent, outputs, top_geography, period,
            grouping_intended=grouping_intended))

        # D. PRODUCE -------------------------------------------------------- #
        if reasons:
            return CompileResult(outcome=outcome_for(reasons), plan=None,
                                 reasons=tuple(reasons), intent=claimed,
                                 compiler_version=self.version)

        plan = GovernedQueryPlan(
            schema_version=PLAN_SCHEMA_VERSION,
            capability=capability,
            operation=intent.operation,
            population=population,
            outputs=tuple(outputs),
            period=period,
            target=target,
            comparison_kind=intent.comparison.kind,
            comparison_left=intent.comparison.left,
            comparison_right=intent.comparison.right,
            filters=tuple(top_filters),
            geography=top_geography,
            provenance=self._provenance(claimed, outputs, top_geography,
                                        population, notes,
                                        normalised=normalisation),
        )
        return CompileResult(outcome=OUTCOME_PLAN, plan=plan, reasons=(),
                             intent=claimed, compiler_version=self.version)

    # -- A. validate -------------------------------------------------------- #

    def _validate(self, intent: CandidateIntent) -> List[CompileReason]:
        reasons: List[CompileReason] = []

        # A BLOCKING ambiguity is honoured, not overruled: "I could not tell
        # which of these you meant" is the cheapest correct signal in the
        # system. A non-blocking one is a disclosed assumption — the model chose
        # and said so — and travels to the plan's provenance instead, because
        # refusing to answer a question because somebody was careful about it
        # would punish exactly the behaviour this layer wants.
        for flagged in intent.ambiguity:
            if flagged.blocking:
                reasons.append(CompileReason(
                    MODEL_FLAGGED_AMBIGUITY, flagged.slot or "intent",
                    flagged.note, tuple(flagged.options)))

        # Conflicting claims: a comparison that names no sides, or sides on a
        # comparison that is 'none'.
        # Every comparison kind names two sides, now that the redundant
        # `period_pair` member is gone: a period-on-period movement is stated by
        # the operation and the time form, not by a third flag.
        comparison = intent.comparison
        if comparison.kind != "none" \
                and not (comparison.left and comparison.right):
            reasons.append(CompileReason(
                CONFLICTING_CLAIMS, "comparison",
                f"kind {comparison.kind!r} with no pair to compare"))
        if comparison.kind == "none" and (comparison.left or comparison.right):
            reasons.append(CompileReason(
                CONFLICTING_CLAIMS, "comparison",
                "a pair was named but no comparison was requested"))

        # A measure needs to exist somewhere unless the operation owns its own.
        if intent.operation not in _MEASURE_OPTIONAL:
            has_measure = bool(intent.measures) or any(
                o.measures for o in intent.effective_outputs())
            if not has_measure:
                reasons.append(CompileReason(
                    MISSING_REQUIRED_SLOT, "measures",
                    f"operation {intent.operation!r} needs at least one measure"))

        # Output ids must be distinct — two outputs that answer to the same name
        # cannot both be reported.
        ids = [o.id for o in intent.outputs]
        if len(ids) != len(set(ids)):
            reasons.append(CompileReason(CONFLICTING_CLAIMS, "outputs",
                                         "duplicate output ids"))
        return reasons

    # -- B. bind ------------------------------------------------------------ #

    def _resolve(self, term: str, *, slot: str,
                 expect_role: Optional[str] = None,
                 ambiguity_code: str = UNREGISTERED_CONCEPT
                 ) -> Tuple[Optional[SemanticConcept], Optional[CompileReason]]:
        """One proposed term -> one governed concept, or a governed refusal.

        The model may have READ this identifier out of the registry. That does
        not shorten the check: existence, ambiguity, asset applicability and
        portfolio availability are all re-derived here, against the same
        authoritative index the metadata tools serve from. A concept the model
        found is a PROPOSAL; this method is where it becomes a binding or a
        refusal, and there is no path between the two.

        Exact-match only, with no fuzzy fallback. A word claimed by more than
        one governed concept resolves to none and names the candidates.
        """
        vocabulary = self.context.vocabulary
        concept = vocabulary.resolve(term)
        if concept is None:
            candidates = vocabulary.candidates(term)
            if len(candidates) > 1:
                return None, CompileReason(
                    ambiguity_code, term,
                    f"{term!r} names more than one governed concept ({slot})",
                    tuple(c.concept_id for c in candidates))
            return None, CompileReason(UNREGISTERED_CONCEPT, term,
                                       f"not a governed concept ({slot})")
        if not vocabulary.applies_here(concept):
            return None, CompileReason(
                CONCEPT_UNAVAILABLE, term,
                f"governed, but does not apply to asset class "
                f"{vocabulary.asset_class!r}")
        if not self.context.field_present(concept.canonical_field):
            return None, CompileReason(CONCEPT_UNAVAILABLE, term,
                                       "governed, but absent from this book")
        if expect_role is not None and concept.role != expect_role:
            return None, CompileReason(
                UNSUPPORTED_COMPOSITION, term,
                f"{term!r} is a {concept.role}, not a {expect_role} ({slot})")
        return concept, None

    def _bind_population(self, intent: CandidateIntent
                         ) -> Tuple[PopulationBinding, List[CompileReason], List[str]]:
        population = intent.population
        reasons: List[CompileReason] = []
        notes: List[str] = []
        predicates: List[FilterBinding] = []
        bound_source_id: Optional[str] = None

        # Only a NON-default axis produces a predicate. The default book state
        # is a dataset selection, not a row filter, and conflating the two is
        # what mi_agent.population exists to prevent.
        if population.lens != "all":
            if self.context.field_present(_POPULATION_LENS_FIELD):
                predicates.append(FilterBinding(
                    concept="portfolio_lens", comparator="eq",
                    canonical_field=_POPULATION_LENS_FIELD, value=population.lens))
                notes.append(f"population lens {population.lens!r} bound to "
                             f"{_POPULATION_LENS_FIELD}")
            else:
                reasons.append(CompileReason(
                    CONCEPT_UNAVAILABLE, "population.lens",
                    f"this book carries no {_POPULATION_LENS_FIELD}"))
        # A NAMED SOURCE IS AN IDENTITY, bound to the identity column. It is a
        # separate axis from the lens above: a role says which KIND of book, a
        # name says WHICH book, and a client may hold two acquired ones. Both may
        # be stated, and both then apply — the registry decides whether they
        # agree, not this module.
        if population.source_reference:
            predicate, reason = self._bind_source(population.source_reference)
            if reason is not None:
                reasons.append(reason)
            else:
                predicates.append(predicate)
                notes.append(f"source {population.source_reference!r} bound to "
                             f"{_SOURCE_ID_FIELD}={predicate.value!r}")
                bound_source_id = predicate.value

        if population.seasoning != "any":
            if self.context.field_present(_POPULATION_SEASONING_FIELD):
                predicates.append(FilterBinding(
                    concept="seasoning_segment", comparator="eq",
                    canonical_field=_POPULATION_SEASONING_FIELD,
                    value=population.seasoning))
                notes.append(f"population seasoning {population.seasoning!r} bound "
                             f"to {_POPULATION_SEASONING_FIELD}")
            else:
                reasons.append(CompileReason(
                    CONCEPT_UNAVAILABLE, "population.seasoning",
                    f"this book carries no {_POPULATION_SEASONING_FIELD}"))
        if population.base in ("pipeline", "forecast"):
            if not self.context.field_present(_POPULATION_BASE_FIELD):
                reasons.append(CompileReason(
                    CONCEPT_UNAVAILABLE, "population.base",
                    f"this book carries no {_POPULATION_BASE_FIELD}"))
            else:
                notes.append(f"population base {population.base!r} selects the "
                             f"{population.base} dataset")

        return (PopulationBinding(base=population.base, lens=population.lens,
                                  seasoning=population.seasoning,
                                  scope_predicates=tuple(predicates),
                                  source_reference=population.source_reference,
                                  source_portfolio_id=bound_source_id),
                reasons, notes)

    def _bind_source(self, reference: str):
        """A governed source NAME -> a predicate on the identity column.

        Every way this can fail refuses. There is no nearest match, no widening
        to the whole book, and no falling back to a role: answering about a book
        the reader did not name is the failure this axis exists to prevent, and
        it is indistinguishable from a correct answer once rendered.
        """
        from trakt_core.portfolio import SOURCE_AMBIGUOUS, SOURCE_UNKNOWN

        registry = self.context.source_registry
        if registry is None:
            return None, CompileReason(
                CONCEPT_UNAVAILABLE, "population.source_reference",
                f"no governed source registry is available for this book, so "
                f"{reference!r} cannot be resolved to a portfolio")
        if not self.context.field_present(_SOURCE_ID_FIELD):
            return None, CompileReason(
                CONCEPT_UNAVAILABLE, "population.source_reference",
                f"this book carries no {_SOURCE_ID_FIELD}")

        record, why = registry.resolve_reference(reference)
        if record is None:
            declared = sorted({n for n in registry.declared_names()})
            if why == SOURCE_AMBIGUOUS:
                return None, CompileReason(
                    AMBIGUOUS_POPULATION, "population.source_reference",
                    f"{reference!r} names more than one governed portfolio; "
                    f"name one of {declared}")
            assert why == SOURCE_UNKNOWN
            return None, CompileReason(
                CONCEPT_UNAVAILABLE, "population.source_reference",
                f"{reference!r} is not a governed source portfolio for this "
                f"book; it governs {declared}")
        return FilterBinding(concept="source_portfolio", comparator="eq",
                             canonical_field=_SOURCE_ID_FIELD,
                             value=record.portfolio_id), None

    def _bind_period(self, intent: CandidateIntent
                     ) -> Tuple[PeriodBinding, List[CompileReason]]:
        time = intent.time
        reasons: List[CompileReason] = []
        contract, settled = _PERIOD_CONTRACT.get(time.form, ("", False))
        if not contract:
            reasons.append(CompileReason(PERIOD_UNRESOLVED, time.form,
                                         "no governed period contract"))
            return PeriodBinding(form=time.form), reasons

        if time.form == "explicit_period" and not time.labels:
            reasons.append(CompileReason(
                PERIOD_UNRESOLVED, "time.labels",
                "an explicit period was claimed but no period was named"))
        if time.form in ("range", "series") and not (
                time.labels or time.periods_back or time.grain):
            reasons.append(CompileReason(
                AMBIGUOUS_PERIOD, "time",
                f"a {time.form} with no span, grain or period count"))
        if time.form == "forward_looking" and intent.capability not in _FORWARD_CAPABILITIES:
            reasons.append(CompileReason(
                UNSUPPORTED_COMPOSITION, "time.form",
                f"capability {intent.capability!r} owns no forward methodology"))

        owned = intent.operation in _CAPABILITY_OWNED_PERIOD_OPERATIONS
        return (PeriodBinding(form=time.form, labels=time.labels, grain=time.grain,
                              periods_back=time.periods_back, contract=contract,
                              resolved=settled, owned_by_capability=owned),
                reasons)

    def _bind_target(self, intent: CandidateIntent
                     ) -> Tuple[Optional[TargetBinding], List[CompileReason]]:
        """The threshold a milestone or limit question is asking about.

        A target is not a filter: "when will we reach £100m" does not narrow the
        population to loans over £100m, it asks when an aggregate crosses a
        line. Run 4 had the interpreter refusing three milestone questions
        because it had correctly declined to express the figure as a filter and
        had nowhere else to put it.

        A milestone with no target is MISSING_REQUIRED_SLOT: "when will we reach
        it?" is not a question until the question says what "it" is.
        """
        reasons: List[CompileReason] = []
        if intent.target is None:
            if intent.operation == "forecast_milestone":
                reasons.append(CompileReason(
                    MISSING_REQUIRED_SLOT, "target",
                    "a milestone needs the threshold it is a milestone of"))
            return None, reasons

        concept, reason = self._resolve(intent.target.concept, slot="target")
        if reason is not None:
            return None, [reason]
        if not isinstance(intent.target.value, (int, float)) \
                or isinstance(intent.target.value, bool):
            return None, [CompileReason(
                UNSUPPORTED_FILTER, "target",
                "a threshold must be a number")]
        return (TargetBinding(concept=concept.concept_id,
                              comparator=intent.target.comparator,
                              value=intent.target.value,
                              canonical_field=concept.canonical_field,
                              capability_owner=concept.owning_capability),
                reasons)

    def _bind_geography(self, geography: SemanticGeography, *, slot: str
                        ) -> Tuple[Optional[GeographyBinding], List[CompileReason],
                                   List[str]]:
        if not geography.requested:
            return None, [], []
        reasons: List[CompileReason] = []
        notes: List[str] = []

        level = geography.level
        defaulted = False
        default_reason = ""
        if level is None:
            level = _DEFAULT_GEOGRAPHY_LEVEL
            defaulted = True
            default_reason = ("no level was stated; the governed default is the "
                              "client's harmonised reporting taxonomy")

        basis = geography.basis
        if basis is None:
            if level in _BASIS_REQUIRED_LEVELS:
                reasons.append(CompileReason(
                    AMBIGUOUS_GEOGRAPHY, f"{slot}.basis",
                    f"at level {level!r} the borrower's region and the "
                    f"property's region are different answers",
                    ("obligor", "collateral")))
                return None, reasons, notes
            basis = _DEFAULT_GEOGRAPHY_BASIS
            defaulted = True
            default_reason = (default_reason or
                              f"no basis was stated; level {level!r} is governed "
                              f"by the reporting taxonomy")

        canonical_field = _GEOGRAPHY_CONTRACT.get((level, basis))
        if canonical_field is None:
            reasons.append(CompileReason(
                INVALID_GEOGRAPHY_BASIS, f"{slot}",
                f"basis {basis!r} is not measurable at level {level!r}"))
            return None, reasons, notes
        if not self.context.field_present(canonical_field):
            reasons.append(CompileReason(
                CONCEPT_UNAVAILABLE, f"{slot}",
                f"this book carries no {level} geography on a {basis} basis"))
            return None, reasons, notes

        if defaulted:
            notes.append(f"geography defaulted: {default_reason}")
        if geography.values:
            notes.append(f"geography restriction bound to {canonical_field}")
        return (GeographyBinding(requested_basis=geography.basis,
                                 requested_level=geography.level,
                                 resolved_level=level,
                                 canonical_field=canonical_field,
                                 group_by=geography.group_by,
                                 values=tuple(geography.values),
                                 defaulted=defaulted,
                                 default_reason=default_reason),
                reasons, notes)

    def _bind_filters(self, filters: Sequence[SemanticFilter], *, slot: str
                      ) -> Tuple[List[FilterBinding], List[CompileReason]]:
        bound: List[FilterBinding] = []
        reasons: List[CompileReason] = []
        for index, predicate in enumerate(filters):
            where = f"{slot}[{index}]"
            concept, reason = self._resolve(predicate.concept, slot=where)
            if reason is not None:
                reasons.append(reason)
                continue
            arity = _COMPARATOR_ARITY.get(predicate.comparator)
            value = predicate.value
            if arity == "scalar" and isinstance(value, (list, tuple)):
                reasons.append(CompileReason(
                    UNSUPPORTED_FILTER, predicate.concept,
                    f"comparator {predicate.comparator!r} takes one value"))
                continue
            if arity == "pair" and not (isinstance(value, (list, tuple))
                                        and len(value) == 2):
                reasons.append(CompileReason(
                    UNSUPPORTED_FILTER, predicate.concept,
                    "'between' takes exactly two bounds"))
                continue
            if arity == "list" and not isinstance(value, (list, tuple)):
                reasons.append(CompileReason(
                    UNSUPPORTED_FILTER, predicate.concept,
                    f"comparator {predicate.comparator!r} takes a list"))
                continue
            if value is None and predicate.comparator not in ("eq", "ne"):
                reasons.append(CompileReason(
                    UNSUPPORTED_FILTER, predicate.concept,
                    f"comparator {predicate.comparator!r} needs a value"))
                continue
            ordered = (concept.role in ("measure", "date")
                       or str(concept.unit or "").strip().lower() in _ORDERED_FORMATS)
            if predicate.comparator in _ORDERED_COMPARATORS and not ordered:
                reasons.append(CompileReason(
                    UNSUPPORTED_FILTER, predicate.concept,
                    f"{concept.role} {predicate.concept!r} is not ordered, so "
                    f"{predicate.comparator!r} has no meaning on it"))
                continue
            bound.append(FilterBinding(
                concept=concept.concept_id, comparator=predicate.comparator,
                canonical_field=concept.canonical_field,
                value=list(value) if isinstance(value, tuple) else value,
                capability_owner=concept.owning_capability))
        return bound, reasons

    def _bind_measure(self, measure: SemanticMeasure, *, slot: str
                      ) -> Tuple[Optional[MeasureBinding], List[CompileReason],
                                 List[str]]:
        reasons: List[CompileReason] = []
        notes: List[str] = []
        concept, reason = self._resolve(measure.concept, slot=slot,
                                        ambiguity_code=AMBIGUOUS_MEASURE)
        if reason is not None:
            return None, [reason], notes

        # A specialist measure carries its own methodology. The statistic is the
        # capability's, not a free choice, so a statistic named against one is a
        # claim about arithmetic the interpreter has no business making.
        if concept.is_specialist:
            if measure.statistic is not None:
                reasons.append(CompileReason(
                    UNSUPPORTED_STATISTIC, measure.concept,
                    f"{concept.owning_capability} owns how {measure.concept!r} "
                    f"is calculated; a statistic cannot be imposed on it"))
            if measure.weight is not None:
                reasons.append(CompileReason(
                    WEIGHT_NOT_PERMITTED, measure.concept,
                    "a specialist measure carries its own weighting"))
            if reasons:
                return None, reasons, notes
            return (MeasureBinding(concept=concept.concept_id, statistic="capability",
                                   canonical_field=None,
                                   capability_owner=concept.owning_capability),
                    reasons, notes)

        if concept.role == "dimension" and measure.statistic in (None, "count",
                                                                 "count_distinct"):
            statistic = measure.statistic or "count"
        elif concept.role != "measure":
            return None, [CompileReason(
                UNSUPPORTED_COMPOSITION, measure.concept,
                f"{measure.concept!r} is a {concept.role} and cannot carry "
                f"statistic {measure.statistic!r}")], notes
        else:
            statistic = measure.statistic

        defaulted = False
        if statistic is None:
            statistic = concept.default_statistic
            defaulted = True
            if statistic is None:
                return None, [CompileReason(
                    MISSING_REQUIRED_SLOT, measure.concept,
                    "no statistic was named and the registry sets no default")], notes
            notes.append(f"statistic for {concept.concept_id!r} defaulted to "
                         f"{statistic!r} by the governed registry")
        elif concept.allowed_statistics and statistic not in concept.allowed_statistics:
            return None, [CompileReason(
                UNSUPPORTED_STATISTIC, measure.concept,
                f"{statistic!r} is not permitted on {measure.concept!r} "
                f"(allowed: {list(concept.allowed_statistics)})")], notes

        weight_concept: Optional[str] = None
        weight_field: Optional[str] = None
        if statistic in STATISTICS_REQUIRING_WEIGHT:
            term = measure.weight or concept.default_weight_concept
            if term is None:
                return None, [CompileReason(
                    MISSING_REQUIRED_SLOT, measure.concept,
                    f"{statistic!r} needs a weight and none is governed for "
                    f"{measure.concept!r}")], notes
            weight, weight_reason = self._resolve(term, slot=f"{slot}.weight",
                                                  expect_role="measure")
            if weight_reason is not None:
                return None, [weight_reason], notes
            if measure.weight is None:
                notes.append(f"weight for {concept.concept_id!r} defaulted to "
                             f"{weight.concept_id!r} by the governed registry")
            weight_concept, weight_field = weight.concept_id, weight.canonical_field
        elif measure.weight is not None and statistic in STATISTICS_FORBIDDING_WEIGHT:
            return None, [CompileReason(
                WEIGHT_NOT_PERMITTED, measure.concept,
                f"statistic {statistic!r} takes no weight")], notes

        return (MeasureBinding(concept=concept.concept_id, statistic=statistic,
                               canonical_field=concept.canonical_field,
                               weight_concept=weight_concept,
                               weight_field=weight_field,
                               statistic_defaulted=defaulted),
                reasons, notes)

    def _bind_output(self, output: RequestedOutput, *, intent: CandidateIntent,
                     inherited_geography: Optional[GeographyBinding]
                     ) -> Tuple[Optional[OutputPlan], List[CompileReason], List[str]]:
        reasons: List[CompileReason] = []
        notes: List[str] = []
        slot = f"outputs[{output.id}]"

        measures: List[MeasureBinding] = []
        for index, measure in enumerate(output.measures):
            bound, measure_reasons, measure_notes = self._bind_measure(
                measure, slot=f"{slot}.measures[{index}]")
            reasons.extend(measure_reasons)
            notes.extend(measure_notes)
            if bound is not None:
                measures.append(bound)

        dimensions: List[DimensionBinding] = []
        geography_from_dimension: Optional[SemanticGeography] = None
        for index, term in enumerate(output.dimensions):
            concept, reason = self._resolve(term, slot=f"{slot}.dimensions[{index}]",
                                            expect_role="dimension",
                                            ambiguity_code=AMBIGUOUS_DIMENSION)
            if reason is not None:
                reasons.append(reason)
                continue
            # A geography concept named as a dimension is NOT grouped as a bare
            # column: it travels into the geography contract, which is where
            # basis and level are governed. The model naming
            # `geographic_region_obligor` has already answered "whose region",
            # so the compiler reads the basis off the concept rather than
            # asking again.
            if concept.is_geography:
                geography_from_dimension = SemanticGeography(
                    requested=True, basis=concept.geography_basis,
                    level=concept.geography_level, group_by=True)
                notes.append(f"{concept.concept_id!r} routed into the geography "
                             f"contract as basis={concept.geography_basis}, "
                             f"level={concept.geography_level}")
                continue
            dimensions.append(DimensionBinding(
                concept=concept.concept_id, canonical_field=concept.canonical_field,
                capability_owner=concept.owning_capability))

        filters, filter_reasons = self._bind_filters(output.filters,
                                                     slot=f"{slot}.filters")
        reasons.extend(filter_reasons)

        requested_geography = output.geography
        if geography_from_dimension is not None and not requested_geography.requested:
            requested_geography = geography_from_dimension
        geography, geo_reasons, geo_notes = self._bind_geography(
            requested_geography, slot=f"{slot}.geography")
        reasons.extend(geo_reasons)
        notes.extend(geo_notes)
        if geography is None and not requested_geography.requested:
            geography = inherited_geography

        if reasons:
            return None, reasons, notes
        return (OutputPlan(id=output.id, measures=tuple(measures),
                           dimensions=tuple(dimensions), filters=tuple(filters),
                           geography=geography),
                reasons, notes)

    # -- C. authorise composition ------------------------------------------- #

    def _authorise_composition(self, intent: CandidateIntent,
                               outputs: Sequence[OutputPlan],
                               geography: Optional[GeographyBinding],
                               period: PeriodBinding, *,
                               grouping_intended: bool = False
                               ) -> List[CompileReason]:
        reasons: List[CompileReason] = []
        operation = intent.operation

        capability_groups = intent.capability in _CAPABILITIES_OWNING_GROUPING
        for output in outputs:
            grouped = bool(output.dimensions) or grouping_intended or (
                output.geography is not None and output.geography.group_by)
            if operation in _GROUPING_REQUIRED and not grouped \
                    and not capability_groups:
                reasons.append(CompileReason(
                    UNSUPPORTED_COMPOSITION, operation,
                    f"a {operation} needs something to group by "
                    f"(output {output.id!r})"))
            if operation in _GROUPING_FORBIDDEN and grouped:
                reasons.append(CompileReason(
                    UNSUPPORTED_COMPOSITION, operation,
                    f"a grouping makes output {output.id!r} a breakdown, not a "
                    f"{operation}"))
            if operation not in _MEASURE_OPTIONAL and not output.measures:
                reasons.append(CompileReason(
                    MISSING_REQUIRED_SLOT, f"outputs[{output.id}].measures",
                    f"operation {operation!r} needs a measure in every output"))
            # A capability's own measures may not be mixed with generic ones in
            # one output: the specialist owns the whole figure, and a plan that
            # asked it to sit beside a column sum would be asking two different
            # owners for one number.
            owners = {m.capability_owner for m in output.measures}
            if len(owners) > 1:
                reasons.append(CompileReason(
                    UNSUPPORTED_COMPOSITION, f"outputs[{output.id}]",
                    "a specialist measure cannot share an output with a "
                    "generically-composed one"))

        if operation in _MULTI_PERIOD_OPERATIONS \
                and operation not in _CAPABILITY_OWNED_PERIOD_OPERATIONS \
                and period.form not in _MULTI_PERIOD_FORMS:
            reasons.append(CompileReason(
                UNSUPPORTED_COMPOSITION, operation,
                f"a {operation} needs two or more periods; the intent states "
                f"{period.form!r}"))

        return reasons

    # -- D. provenance ------------------------------------------------------ #

    def _provenance(self, intent: CandidateIntent, outputs: Sequence[OutputPlan],
                    geography: Optional[GeographyBinding],
                    population: PopulationBinding,
                    notes: Sequence[str],
                    *, normalised: Optional[Any] = None) -> PlanProvenance:
        """Record who decided what.

        ``intent_claims`` is the model's reading, copied verbatim and read by
        nothing. ``compiler_bindings`` is every physical choice this module
        made. Keeping them apart is what lets an auditor answer the question the
        architecture is judged on.

        ``intent`` here is the CLAIMED intent, before normalisation. That matters:
        a provenance recording the canonical form would quietly lose what the
        model actually said, and "did the model pick this?" is the question the
        split exists to answer. What normalisation changed is recorded beside it,
        under ``compiler_bindings["normalisation"]``, because the rewrite is the
        compiler's decision and belongs on the compiler's side of the line.
        """
        claims: Dict[str, Any] = {
            "capability": intent.capability,
            "operation": intent.operation,
            "population": list(intent.population.key()),
            "measures": [list(m.key()) for m in intent.measures],
            "dimensions": list(intent.dimensions),
            "filters": [list(f.key()) for f in intent.filters],
            "geography": list(intent.geography.key()),
            "time": list(intent.time.key()),
            "comparison": list(intent.comparison.key()),
            "target": list(intent.target.key()) if intent.target else None,
            "change_form": intent.change_form,
            "evidence": [{"claim": e.claim, "text": e.text} for e in intent.evidence],
        }
        bindings: Dict[str, Any] = {
            "geography_field": geography.canonical_field if geography else None,
            "population_predicates": [
                {"field": p.canonical_field, "comparator": p.comparator,
                 "value": p.value} for p in population.scope_predicates],
            "outputs": {
                output.id: {
                    "measures": [
                        {"concept": m.concept, "field": m.canonical_field,
                         "statistic": m.statistic, "weight_field": m.weight_field,
                         "capability_owner": m.capability_owner}
                        for m in output.measures],
                    "dimensions": [
                        {"concept": d.concept, "field": d.canonical_field}
                        for d in output.dimensions],
                    "filters": [
                        {"concept": f.concept, "field": f.canonical_field,
                         "comparator": f.comparator} for f in output.filters],
                } for output in outputs},
            "normalisation": {
                "normal_form_version": NORMAL_FORM_VERSION,
                "applied": list(getattr(normalised, "applied", ()) or ()),
            },
            # THE COMPILER'S READING OF THE ANALYTICAL FORM, not the model's.
            # The form itself is a claim and sits above in ``intent_claims``;
            # the owner and the mode it implies are this module's decision, from
            # its own mapping, and belong on the compiler's side of the line.
            # A runtime that dispatches on the form reads THIS — never the
            # claim — so no serving decision is ever taken from the model's raw
            # reading.
            "change_form": {
                "form": intent.change_form,
                "capability": CHANGE_FORM_CAPABILITY.get(intent.change_form),
                "mode": CHANGE_FORM_MODE.get(intent.change_form or ""),
            } if intent.change_form else None,
        }
        return PlanProvenance(
            question=intent.provenance.question,
            model_id=intent.provenance.model_id,
            interpreter_version=intent.provenance.interpreter_version,
            vocabulary_version=self.context.vocabulary.version,
            compiler_version=self.version,
            intent_schema_version=intent.schema_version,
            intent_claims=claims,
            compiler_bindings=bindings,
            notes=tuple(notes),
            usage=dict(intent.provenance.usage or {}),
        )


def compile_intent(intent: CandidateIntent,
                   context: Optional[CompilerContext] = None) -> CompileResult:
    """Convenience entry point for one-off compilation."""
    return DeterministicCompiler(context).compile(intent)
