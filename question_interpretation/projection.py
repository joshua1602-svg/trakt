#!/usr/bin/env python3
"""question_interpretation/projection.py — Stage 1, read-only.

Populates a `QuestionInterpretation` from the EXISTING interpreters. Nothing
here decides anything of its own: every slot records which interpreter supplied
it, in `Slot.source`, so a gap is attributable to a source rather than to this
file. Where no existing interpreter supplies a slot, the slot stays empty and
the reason is recorded in `notes` — that absence is the Stage 1 finding.

**No production code is modified or imported for effect.** Every call below is a
read of an existing function.

Sources consulted, and what each supplies:

    llm_query_parser._deterministic_parse   operation (via aggregation),
                                            subject, dimensions, filters,
                                            target (forecast_target_value)
    execution_receipt.detect_requested_facets
                                            dimensions (named), filters
                                            (threshold/geographic clauses),
                                            operation cues (ranking, projection,
                                            statistic)
    execution_receipt.requested_dimension_terms
                                            dimension raw text and offsets
    answer_type.asked                       operation type, independently
    period_request.requested_unit/span      time grain and window
    population / seasoning                  population concepts
    portfolio_lens.resolve_lens             source-portfolio scope (Phase 1A)
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

from . import lexical as _LEX
from .schema import (
    AMOUNT, AVERAGE, BASE_ACQUIRED, BASE_DIRECT, BASE_FUNDED, BOUND, COUNT,
    EMPTY, FIELD, FILLED, FILTER, FORWARD,
    GRAINS, GROUPING, MOVEMENT, NEUTRAL,
    PROV_CALLER_CONTEXT, PROV_DEFAULT, PROV_EXPLICIT_USER, PROV_UNRESOLVED,
    RANKING, ROLE_UNATTRIBUTED,
    SCOPE_ACQUIRED, SCOPE_COHORT, SCOPE_DIRECT, SCOPE_TOTAL, SOURCE_SCOPES,
    STATED,
    UNRESOLVABLE, UNRESOLVED_ROLE, WORDING, DimensionClaim, FilterClaim,
    OperationClaim, PopulationClaim, QuestionInterpretation, RowPredicateClaim, Slot,
    SourceScopeClaim, Span, SubjectClaim, TargetClaim, TimeClaim,
    DatasetClaim, DATASET_FUNDED,
)

#: How the parser's aggregation reads as an operation type. `coverage` has no
#: parser equivalent — recorded as a Stage 1 gap rather than invented.
_AGG_TO_OPERATION = {
    "count": COUNT, "count_distinct": COUNT,
    "sum": AMOUNT, "balance_sum": AMOUNT,
    "avg": AVERAGE, "weighted_avg": AVERAGE, "median": AVERAGE,
    "distribution": NEUTRAL, "loan_level": NEUTRAL,
}

#: `answer_type.asked` speaks a different vocabulary. Mapped only where the two
#: genuinely mean the same thing; `rate`/`age`/`date` have no operation
#: equivalent and are recorded, not forced.
_ANSWER_TYPE_TO_OPERATION = {"count": COUNT, "currency": AMOUNT, "rate": AVERAGE}

_FORWARD_FACETS = {"projection"}
_RANKING_FACETS = {"ranking"}
_FILTER_FACET_KINDS = {"geographic_scope", "threshold", "row_population",
                       "stress_scenario"}
_GROUPING_FACET_KINDS = {"grouping_dimension"}


def _span_of(question: str, needle: Optional[str]) -> Optional[Span]:
    """Offsets of `needle` in `question`, or None if it cannot be located.

    A facet carries a rendered label rather than offsets, so this recovers the
    span where the label happens to be a literal substring and gives up
    otherwise. How often it gives up is a Stage 1 finding.
    """
    if not needle:
        return None
    m = re.search(re.escape(needle.strip()), question, re.I)
    return Span(m.start(), m.end()) if m else None


def from_parts(question: str, *, spec, facets, dim_terms,
               semantics: dict, registry=None,
               caller_scope=None, caller_dataset=None) -> QuestionInterpretation:
    """Assemble the object from interpreter output that ALREADY EXISTS.

    This is the Stage 2 entry point. It re-interprets nothing: the spec and the
    facets are handed in, having been produced by the pipeline for its own
    reasons, and this only records what they say. `answer_type.asked` and
    `period_request` are consulted because they are existing interpreters whose
    readings belong on the object and which nothing else carries — not to form
    a new opinion.

    Faithful population includes faithfully recording what today's interpreters
    get WRONG. No role is corrected here; a disagreement is noted, not resolved.
    """
    from mi_agent import answer_type as AT
    from mi_agent import period_request as PR

    qi = QuestionInterpretation(question=question)
    facet_kinds = {f.kind for f in facets}

    _operation(qi, spec, facet_kinds, AT)
    _subject(qi, spec)
    _dimensions(qi, spec, dim_terms, facets)
    _filters(qi, spec, facets)
    _time(qi, spec, PR)
    _target(qi, spec, facets)
    _population(qi, spec, facets)
    _row_predicates(qi, spec, semantics)
    _source_scope(qi, registry, caller_scope)
    _dataset(qi, caller_dataset)
    _note_join_state(qi)
    return qi


def _note_join_state(qi: QuestionInterpretation) -> None:
    """Say whether the filter clause is whole, and name the half that is not.

    The facet layer now emits offsets; the parser does not, because emitting
    them means changing how `_parse_filters` rewrites the question, which would
    forfeit Stage 2's byte-identical guarantee. So the join is HALF-BUILT, and
    the object says so rather than implying a link it cannot make.
    """
    wording = [f for f in qi.filters if WORDING in f.provides]
    binding = [f for f in qi.filters if {BOUND, FIELD} & set(f.provides)]
    if not wording or not binding:
        return
    located = sum(1 for f in wording if f.has_span)
    qi.notes.append(
        "filter join HALF-BUILT: %d wording claim(s), %d of them located by "
        "span; %d binding claim(s), none located — the parser half supplies no "
        "offsets, so clause_id stays None"
        % (len(wording), located, len(binding)))


def project(question: str, *, semantics: dict, frame=None,
            registry=None, caller_scope=None,
            caller_dataset=None) -> QuestionInterpretation:
    """Build a QuestionInterpretation by asking the existing interpreters.

    The read-only Stage 1 path: runs the interpreters itself, then assembles.
    Production uses `from_parts`, which assembles from output already computed.
    """
    from mi_agent import execution_receipt as R
    from mi_agent.llm_query_parser import _deterministic_parse

    spec, _meta = _deterministic_parse(question, semantics)
    cols = list(frame.columns) if frame is not None and hasattr(frame, "columns") else None
    dim_terms = R.requested_dimension_terms(question, semantics, cols)
    facets = R.detect_requested_facets(question, semantics, frame=frame,
                                       requested_dimensions=dim_terms)
    return from_parts(question, spec=spec, facets=facets, dim_terms=dim_terms,
                      semantics=semantics, registry=registry,
                      caller_scope=caller_scope, caller_dataset=caller_dataset)


# --------------------------------------------------------------------------- #
def _operation(qi, spec, facet_kinds, AT) -> None:
    asked = AT.asked(qi.question)
    qi.notes.append("answer_type.asked=%s" % asked)

    if facet_kinds & _FORWARD_FACETS or getattr(spec, "forecast_question", None):
        qi.operation = OperationClaim(state=FILLED, type=FORWARD,
                                      source="parser.forecast/facet.projection")
        return
    if facet_kinds & _RANKING_FACETS or getattr(spec, "ranking_mode", None):
        qi.operation = OperationClaim(state=FILLED, type=RANKING,
                                      source="parser.ranking_mode/facet.ranking")
        return
    if getattr(spec, "compare_periods", None) or \
            getattr(spec, "temporal_mode", None) == "compare" or \
            getattr(spec, "bridge_query", False):
        qi.operation = OperationClaim(state=FILLED, type=MOVEMENT,
                                      source="parser.compare/bridge")
        return
    mapped = _AGG_TO_OPERATION.get(getattr(spec, "aggregation", None))
    if mapped is None:
        qi.notes.append("operation: no source supplies it")
        return
    # The two independent readings, recorded when they differ. Not reconciled:
    # reconciling here would hide the disagreement Stage 1 exists to report.
    from_answer_type = _ANSWER_TYPE_TO_OPERATION.get(asked)
    if from_answer_type and from_answer_type != mapped:
        qi.notes.append("operation DISAGREEMENT parser=%s answer_type=%s"
                        % (mapped, from_answer_type))
    qi.operation = OperationClaim(state=FILLED, type=mapped,
                                  source="parser.aggregation")


def _subject(qi, spec) -> None:
    """The subject slot, plus the SPAN the subject may be named in.

    The span comes from the single owner of the subject-side split, so the
    object now carries the same decision `answer_type.subject_side` returns
    rather than each consumer re-deriving it.
    """
    from .lexical import subject_side_span
    start, end = subject_side_span(qi.question)
    span = Span(start, end) if end > start else None
    raw = qi.question[start:end].strip() or None

    metric = getattr(spec, "metric", None)
    if metric:
        qi.subject = SubjectClaim(state=FILLED, candidate_concept=metric,
                                  raw_text=raw, span=span,
                                  source="parser.metric")
        return
    if getattr(spec, "aggregation", None) in ("count", "count_distinct"):
        qi.subject = SubjectClaim(state=FILLED, candidate_concept="loan_count",
                                  raw_text=raw, span=span,
                                  source="parser.aggregation=count")
        return
    qi.subject = SubjectClaim(state=EMPTY, raw_text=raw, span=span,
                              source="lexical.subject_side_span")
    qi.notes.append("subject: no source supplies a concept; the subject-side "
                    "span is carried regardless")


def _dimensions(qi, spec, dim_terms, facets) -> None:
    """The role split. This is the one place Stage 1 must NOT invent an answer.

    The parser puts a dimension in `spec.dimension(s)` or in `spec.filters`, so
    its slot assignment is readable. The facet layer raises KIND_GROUPING for
    both, so it supplies the term and NOT the role. Where the parser is silent,
    the role is recorded as `unresolved` rather than guessed.
    """
    parser_groups = [d for d in ([getattr(spec, "dimension", None)]
                                 + list(getattr(spec, "dimensions", None) or [])) if d]
    parser_filters = set((getattr(spec, "filters", None) or {}).keys())
    # The parser's OWN bridge attribution axis. `spec.bridge_dimension` is a
    # governed field key, populated only on bridge questions, and it IS the
    # grouping the waterfall attributes movement by — so the projection must
    # carry that role rather than emit `unresolved` for a fact the parser already
    # settled. This is projection, not reinterpretation: no raw text is read, the
    # match below is governed-key to governed-key, and it fires for exactly the
    # one claim whose key equals `spec.bridge_dimension`.
    bridge_dim = getattr(spec, "bridge_dimension", None)

    seen = set()
    for key, term, _alt in dim_terms:
        if key in seen:
            continue
        seen.add(key)
        if key in parser_groups:
            role, src = GROUPING, "parser.dimension"
        elif bridge_dim is not None and key == bridge_dim:
            role, src = GROUPING, "parser.bridge_dimension"
        elif key in parser_filters:
            role, src = FILTER, "parser.filters"
        else:
            role, src = UNRESOLVED_ROLE, "facet.grouping_dimension(role not supplied)"
            qi.notes.append("dimension %s: named by the facet layer, no role "
                            "from any source" % key)
        qi.dimensions.append(DimensionClaim(
            state=FILLED, raw_text=term, span=_span_of(qi.question, term),
            role=role, candidate_concept=key, source=src,
            # CORRECTION 5: an unresolved role must say WHY.
            reason=ROLE_UNATTRIBUTED if role == UNRESOLVED_ROLE else None))

    # A dimension the parser assigned that the facet layer never named.
    for key in parser_groups:
        if key not in seen:
            seen.add(key)
            qi.dimensions.append(DimensionClaim(
                state=FILLED, raw_text=None, role=GROUPING,
                candidate_concept=key, source="parser.dimension(no facet term)"))
            qi.notes.append("dimension %s: parser only, no raw text available" % key)

    # THE PIPELINE STAGE AXIS.
    #
    # `pipeline_stage` is a governed dimension in the pipeline field contract and
    # a categorical stratification over `total_pipeline`, and the facet layer
    # still never named it — so nothing above can raise it, and the one consumer
    # that needs it re-read the raw sentence instead. This raises it from the
    # single governed reader, into the SAME claim every other dimension uses.
    #
    # The role is the distinction the reader deliberately does not make: a
    # question naming one stage is NARROWING to it, and one naming only the axis
    # is SPLITTING by it. Naming both narrows — "offer-stage cases" asks about
    # offers, not for a split across all five.
    stage, names_axis = _LEX.pipeline_stage_request(qi.question)
    if (stage or names_axis) and _LEX.PIPELINE_STAGE_FIELD not in seen:
        seen.add(_LEX.PIPELINE_STAGE_FIELD)
        qi.dimensions.append(DimensionClaim(
            state=FILLED, raw_text=None,
            role=FILTER if stage else GROUPING,
            candidate_concept=_LEX.PIPELINE_STAGE_FIELD,
            source="lexical.pipeline_stage_request"))


def _filters(qi, spec, facets) -> None:
    """Filter clauses, from both readings, kept separate.

    Every `threshold` facet carries field_key=None — it identifies the clause
    and not the field — so the facet supplies raw_text and the parser supplies
    the operator and value. Where only one of them speaks, that is recorded.
    """
    parser_filters: Dict[str, Any] = dict(getattr(spec, "filters", None) or {})
    claimed_fields = {d.candidate_concept for d in qi.dimensions
                      if d.role == FILTER}

    for f in facets:
        if f.kind not in _FILTER_FACET_KINDS:
            continue
        # The detector's own offsets where it has them — the label is a
        # RE-RENDERING of the words ("£250k" becomes 250), so locating it by
        # substring search fails on exactly the cases the join needs.
        span = Span(*f.span) if getattr(f, "span", None) else _span_of(qi.question, f.label)
        qi.filters.append(FilterClaim(
            state=FILLED, raw_text=f.label, span=span,
            # The facet supplies the WORDING of the clause. Every threshold
            # facet carries field_key=None, so it supplies neither field nor
            # bound — recorded, not inferred from which attributes are None.
            provides=(WORDING,) if f.field_key is None else (WORDING, FIELD),
            source="facet.%s" % f.kind))
        if f.field_key is None:
            qi.notes.append("filter %r: facet identifies the clause, no field "
                            "from the facet layer" % f.label)

    for key, condition in parser_filters.items():
        if key in claimed_fields:
            continue
        op = val = None
        cat = None
        if isinstance(condition, dict):
            op, val = condition.get("op"), condition.get("value")
        else:
            cat = str(condition)
        provides = [FIELD]
        if op is not None or val is not None or cat is not None:
            provides.append(BOUND)
        qi.filters.append(FilterClaim(
            state=FILLED, raw_text=None, operator=op,
            value=None if val is None else str(val), categorical_value=cat,
            # The parser supplies the FIELD and the BOUND, never the wording.
            provides=tuple(provides),
            source="parser.filters[%s]" % key))
        qi.notes.append("filter on %s: parser supplies the field and bound, no "
                        "raw text" % key)

    # The stage a question narrows to, as a governed categorical value. Canonical
    # (`OFFER`, not "offer issued") because the reader normalises through the one
    # authoritative stage map, so a consumer never has to spell-match.
    stage, _axis = _LEX.pipeline_stage_request(qi.question)
    if stage and _LEX.PIPELINE_STAGE_FIELD not in parser_filters:
        qi.filters.append(FilterClaim(
            state=FILLED, raw_text=None, categorical_value=stage,
            provides=(FIELD, BOUND),
            source="lexical.pipeline_stage_request"))


def _time(qi, spec, PR) -> None:
    unit = PR.requested_unit(qi.question)
    span = PR.requested_span(qi.question)

    if unit in GRAINS:
        qi.time.requested_grain = Slot(state=FILLED, raw_text=unit,
                                       source="period_request.requested_unit")
        qi.time.grain = unit
    elif unit:
        qi.notes.append("time grain %r outside the controlled vocabulary" % unit)

    if span is not None:
        qi.time.trend_window = Slot(state=FILLED, raw_text=getattr(span, "label", None),
                                    source="period_request.requested_span")
        # TARGET-STATE CLOSURE. The MAGNITUDE, not only the wording. The slot
        # above says a window was named; these say which one, and without them
        # `chat_routing._route_period_movement` had to ask the owner again.
        qi.time.window_periods = getattr(span, "periods", None)
        qi.time.window_governed = bool(getattr(span, "governed", False))
    if getattr(spec, "compare_periods", None):
        periods = tuple(str(p) for p in spec.compare_periods)
        qi.time.comparison_period = Slot(
            state=FILLED, raw_text=", ".join(periods),
            source="parser.compare_periods")
        # THE VALUES, not only the wording. The slot above says a comparison was
        # named; this says which periods, so a consumer never has to split the
        # display join back into structure. Same owner, same call, one read.
        qi.time.comparison_periods = periods

    grain_on_spec = getattr(spec, "trend_grain", None)
    if qi.time.grain and not grain_on_spec:
        qi.notes.append("time grain %r read by period_request, NOT carried on "
                        "the spec (trend_grain=None)" % qi.time.grain)


def _target(qi, spec, facets) -> None:
    value = getattr(spec, "forecast_target_value", None)
    if value is not None:
        qi.target = TargetClaim(state=FILLED, value=str(value),
                                target_source=STATED,
                                source="parser.forecast_target_value")
        return
    # CORRECTION 4: the configured sense is NOT read here. A regex owned by the
    # projection is a reading the projection invented, not one it observed, and
    # this one never fired on the real corpus anyway. The slot stays empty until
    # an interpreter supplies it.


def _governed_ids(scope_name, lens, registry):
    """The governed portfolio IDs a resolved scope selects.

    PHASE 1G §10. A CATEGORY is resolved through the registry to the ids it
    currently contains, so "the acquired book" carries every acquired portfolio
    and cannot collapse onto whichever one happens to exist first. A NAMED
    portfolio carries exactly itself.

    `total` carries no ids on purpose: the complete funded population is
    UNRESTRICTED, not an enumeration, and listing today's members would make a
    newly onboarded portfolio silently absent from a question that asked for the
    whole book. `base_population` is what says which population it is.

    Never `resolve_scope`'s fallback list: that returns EVERY id with
    `fell_back_to_total=True` for a scope it could not resolve, and taking it
    here is precisely the widening the claim exists to prevent.
    """
    if registry is None or scope_name == SCOPE_TOTAL:
        return ()
    if scope_name == SCOPE_COHORT:
        ids = tuple(getattr(lens, "cohort_ids", ()) or ())
        if not ids and getattr(lens, "cohort_id", None):
            ids = (lens.cohort_id,)
        return tuple(i for i in ids if registry.get(i) is not None)
    try:
        return tuple(p.portfolio_id for p in registry.of_type(scope_name))
    except Exception:  # noqa: BLE001 - a registry that cannot answer carries none
        return ()


#: Which BROAD POPULATION a resolved scope is about. A named portfolio sits
#: inside the funded population; its category belongs to the portfolio, not to
#: the request (§8).
_BASE_FOR_SCOPE = {SCOPE_TOTAL: BASE_FUNDED, SCOPE_DIRECT: BASE_DIRECT,
                   SCOPE_ACQUIRED: BASE_ACQUIRED, SCOPE_COHORT: BASE_FUNDED}


def _source_scope(qi, registry=None, caller_scope=None) -> None:
    """Carry `mi_agent.portfolio_lens`'s reading. It stays the single owner.

    One call, to the resolver that already decides this for every route today.
    Nothing here matches a phrase, and no vocabulary lives in this module or
    downstream of it: if the owner widens what it recognises, this widens with
    it, and if the owner is unavailable the claim is UNRESOLVABLE rather than
    silently Total.

    `resolve_lens` returns a resolved `total` lens when it reads the question and
    finds no source narrowing, so Total arrives as a POSITIVE reading. That is
    what lets a consumer tell "explicitly the whole book" from "nobody looked",
    which the empty `population` list could not.

    PHASE 1E — ``registry``. Handed a governed :class:`PortfolioRegistry`, the
    owner resolves a book NAMED in the question to its governed id, and says
    UNRESOLVED for a name it does not hold. Both readings are what make the
    claim's identity canonical: without a registry the owner can only recognise
    the storage convention, so `portfolio_ids` would carry a storage folder name
    that the governed model does not key on (Phase 1D). The registry is PASSED
    IN rather than discovered here — this module reaches into no application
    state, and that is what keeps it a transport object.
    """
    try:
        from mi_agent import portfolio_lens as _lens_owner
    except Exception as exc:  # noqa: BLE001 - the claim records the gap
        qi.source_scope = SourceScopeClaim(
            state=UNRESOLVABLE, source="mi_agent.portfolio_lens",
            reason="the source-portfolio lens owner is unavailable: %s" % exc)
        return
    try:
        # PHASE 1G. The owner is asked TWO things, both of them its own:
        # what the question resolves to, and whether the question named a scope
        # at all. The second is the fact Phase 1F stopped for, and asking the
        # owner for it is what keeps this module free of a second reader of the
        # question — no phrase list is introduced here and none may be.
        stated = bool(_lens_owner.mentions_portfolio(qi.question)
                      or _lens_owner.names_governed_portfolio(qi.question, registry))
        lens = (_lens_owner.resolve_lens(qi.question, registry=registry)
                if registry is not None else _lens_owner.resolve_lens(qi.question))
    except Exception as exc:  # noqa: BLE001
        qi.source_scope = SourceScopeClaim(
            state=UNRESOLVABLE, source="mi_agent.portfolio_lens",
            reason="the source-portfolio lens could not be resolved: %s" % exc)
        return

    # PRECEDENCE, applied once, here, from the owner's own reading:
    #   the question named a scope   -> it wins, whatever the caller supplied
    #   it did not, a caller did     -> the caller's selection
    #   neither                      -> the complete funded population
    # The rule is not re-decided downstream; the claim records the outcome AND
    # which of the three happened, so no consumer has to re-derive it.
    provenance = PROV_EXPLICIT_USER if stated else PROV_DEFAULT
    if not stated and caller_scope is not None:
        try:
            supplied = _lens_owner.lens_from_selection(caller_scope,
                                                       registry=registry)
        except TypeError:                       # pre-1G signature
            supplied = _lens_owner.lens_from_selection(caller_scope)
        if supplied is not None and supplied.name != _lens_owner.LENS_TOTAL:
            lens, provenance = supplied, PROV_CALLER_CONTEXT

    name = getattr(lens, "name", None)
    # PHASE 1E. The owner NAMED a scope and could not resolve it. That is the
    # contract's UNRESOLVABLE, stated as such and carrying the wording that
    # asked — never `total`, which is the widening this whole phase closes.
    if name == getattr(_lens_owner, "LENS_UNRESOLVED", "unresolved"):
        requested = getattr(lens, "label", None)
        qi.source_scope = SourceScopeClaim(
            state=UNRESOLVABLE, raw_text=requested,
            span=_span_of(qi.question, requested),
            source="mi_agent.portfolio_lens",
            provenance=PROV_UNRESOLVED,
            reason="the question names %r, which is not a governed portfolio "
                   "for this book" % (requested,))
        return
    if name not in SOURCE_SCOPES:
        # A lens kind this contract has no member for. Recorded as unresolvable
        # rather than mapped onto the nearest one — a substitution here would be
        # invisible downstream.
        qi.source_scope = SourceScopeClaim(
            state=UNRESOLVABLE, source="mi_agent.portfolio_lens",
            reason="the owner resolved a lens this contract cannot carry: %r"
                   % (name,))
        return

    ids = _governed_ids(name, lens, registry)
    if not ids and name == SCOPE_COHORT:
        ids = tuple(getattr(lens, "cohort_ids", ()) or ())
        if not ids and getattr(lens, "cohort_id", None):
            ids = (lens.cohort_id,)
    label = getattr(lens, "label", None)
    # Only a NARROWING has wording in the question to point at; `total` is the
    # absence of a scope phrase, so it carries no raw_text and no span.
    raw = None if name == SCOPE_TOTAL else label
    span = _span_of(qi.question, raw)
    # PHASE 1E. With a registry, `label` is the GOVERNED display label, which is
    # frequently not the wording that asked — "the alp_acquired book" resolves
    # to "ALP Acquired Back Book". Carry both, and do not pretend the governed
    # label was the wording: when it is not a substring of the question there is
    # no span, and `raw_text` falls back to the id that WAS named.
    portfolio_label = label if name == SCOPE_COHORT else None
    if name == SCOPE_COHORT and span is None:
        named = next((pid for pid in ids
                      if _span_of(qi.question, pid) is not None), None)
        if named is not None:
            raw, span = named, _span_of(qi.question, named)
    if name == SCOPE_COHORT and not ids:
        # A cohort reading with no governed id is not something this contract
        # can carry as FILLED (the schema refuses it), and inventing one would
        # be the substitution the claim exists to prevent.
        qi.source_scope = SourceScopeClaim(
            state=UNRESOLVABLE, raw_text=label,
            span=_span_of(qi.question, label),
            source="mi_agent.portfolio_lens",
            reason="the owner read a named book (%r) but resolved no governed "
                   "portfolio id for it" % (label,))
        return
    qi.source_scope = SourceScopeClaim(
        state=FILLED, raw_text=raw, span=span,
        scope=name, portfolio_ids=ids, portfolio_label=portfolio_label,
        base_population=_BASE_FOR_SCOPE.get(name),
        provenance=provenance,
        source="mi_agent.portfolio_lens")


def _dataset(qi, caller_dataset=None) -> None:
    """Carry the ONE governed dataset decision onto the contract.

    `mi_agent_api.workspace.resolve_dataset` is the single semantic owner and
    this is the handoff: everything downstream reads `qi.dataset` rather than
    re-deciding from the sentence.

    ``caller_dataset`` is ACCEPTED AND IGNORED, and its retirement is the point
    of this function's current shape. It used to be the fallback when the
    question named no view — which meant the active workspace tab decided what
    a question MEANT, so "the balance by seasoning segment excluding pipeline
    cases" was served from the pipeline on the pipeline tab: the sentence ruled
    the pipeline out and the tab put it back. Natural-language MI is
    self-contained. The question decides; the tab displays.

    Provenance follows from that. There are now two cases and not three:
    the QUESTION named the dataset, or the governed DEFAULT applied.
    `PROV_CALLER_CONTEXT` is no longer reachable for this axis, which is a
    property worth being able to assert rather than merely believe.
    """
    try:
        from mi_agent_api.workspace import resolve_dataset, view_named_by_question
    except Exception as exc:  # noqa: BLE001 - the claim records the gap
        qi.dataset = DatasetClaim(
            state=UNRESOLVABLE, source="mi_agent_api.workspace",
            reason="the dataset owner is unavailable: %s" % exc)
        return
    try:
        dataset = resolve_dataset(qi.question)
        named = view_named_by_question(qi.question)
    except Exception as exc:  # noqa: BLE001
        qi.dataset = DatasetClaim(
            state=UNRESOLVABLE, source="mi_agent_api.workspace",
            reason="the dataset could not be resolved: %s" % exc)
        return

    # A question that named a view outright, and one that named a pre-funding
    # artefact, are both the USER stating the dataset. Only the fall-through to
    # `funded` is the governed default. `raw_text` carries the view name when
    # there was one, because that is the span the reader can point at.
    if named is not None:
        provenance, raw = PROV_EXPLICIT_USER, named
    elif dataset != DATASET_FUNDED:
        provenance, raw = PROV_EXPLICIT_USER, None
    else:
        provenance, raw = PROV_DEFAULT, None
    qi.dataset = DatasetClaim(
        state=FILLED, dataset=dataset, provenance=provenance, raw_text=raw,
        span=_span_of(qi.question, raw),
        source="mi_agent_api.workspace.resolve_dataset")


def _row_predicates(qi, spec, semantics) -> None:
    """Carry the predicates the parser already RESOLVED into the contract.

    `llm_query_parser._filter_field_of` binds a clause to its governed field
    once, upstream of every route — measured, `chat_routing` calls neither it nor
    `_resolve_subject` nor the subject vocabulary. `spec.filters` therefore
    arrives ALREADY keyed by governed field, and the only reason a compositional
    plan cannot read it is that the projection wrote that key into a provenance
    STRING — `source="parser.filters[current_loan_to_value]"` — rather than a
    structure.

    This reads it structurally, through the SAME call the population ledger
    already makes. `material_predicates` is the one normaliser: it turns a
    numeric `{"op": "gt", "value": 50.0}`, a bare categorical value and a list
    into one `Predicate(field, op, value)` shape, and it excludes
    `source_portfolio_id` by name because that phrase family is SCOPE and travels
    on `source_scope`.

    Nothing is re-derived and no wording is read. A question with no governed
    predicate contributes nothing, which is a legitimate empty and not a failure.
    """
    from mi_agent.population import material_predicates

    filters = dict(getattr(spec, "filters", None) or {})
    if not filters:
        return
    for predicate in material_predicates(filters, semantics):
        qi.row_predicates.append(RowPredicateClaim(
            state=FILLED, raw_text=None,
            field_key=predicate.field, operator=predicate.op,
            value=predicate.value,
            source="parser.filters via population.material_predicates"))


def _population(qi, spec, facets) -> None:
    for f in facets:
        if f.kind not in ("row_population", "cohort_comparison"):
            continue
        qi.population.append(PopulationClaim(
            state=FILLED, raw_text=f.label, span=_span_of(qi.question, f.label),
            concept=f.field_key, source="facet.%s" % f.kind))

    # PHASE 1G §13 — the ORIGINATION VINTAGE, carried from the owner that
    # already reads it. `cohort_vintage` is set by the deterministic parser; it
    # was simply never projected, so a question naming both a portfolio and a
    # vintage arrived carrying only the portfolio.
    #
    # A SEPARATE CLAIM, not a source scope. Vintage is WHEN a loan was written
    # and portfolio identity is WHERE IT CAME FROM: "the 2025 vintage for SPV2"
    # is one of each, and neither implies the other. Collapsing them would
    # recreate the hierarchy this phase exists to remove — a scope value per
    # vintage, per portfolio.
    #
    # NOT NEW CAPABILITY: nothing here reads the question, and the point-in-time
    # drop Phase 1D recorded (`cohort_vintage` is only set when the question
    # also carries a progression marker) is unchanged and still open. This
    # carries what the owner supplies, no more.
    vintage = getattr(spec, "cohort_vintage", None)
    if vintage:
        qi.population.append(PopulationClaim(
            state=FILLED, raw_text=str(vintage),
            span=_span_of(qi.question, str(vintage)),
            concept="cohort_vintage", source="parser.cohort_vintage"))
