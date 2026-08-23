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

from .schema import (
    AMOUNT, AVERAGE, BOUND, COUNT, EMPTY, FIELD, FILLED, FILTER, FORWARD,
    GRAINS, GROUPING, MOVEMENT, NEUTRAL, RANKING, ROLE_UNATTRIBUTED,
    SCOPE_COHORT, SCOPE_TOTAL, SOURCE_SCOPES, STATED,
    UNRESOLVABLE, UNRESOLVED_ROLE, WORDING, DimensionClaim, FilterClaim,
    OperationClaim, PopulationClaim, QuestionInterpretation, Slot,
    SourceScopeClaim, Span, SubjectClaim, TargetClaim, TimeClaim,
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
               semantics: dict) -> QuestionInterpretation:
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
    _source_scope(qi)
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


def project(question: str, *, semantics: dict, frame=None) -> QuestionInterpretation:
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
                      semantics=semantics)


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

    seen = set()
    for key, term, _alt in dim_terms:
        if key in seen:
            continue
        seen.add(key)
        if key in parser_groups:
            role, src = GROUPING, "parser.dimension"
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
    if getattr(spec, "compare_periods", None):
        qi.time.comparison_period = Slot(
            state=FILLED, raw_text=", ".join(str(p) for p in spec.compare_periods),
            source="parser.compare_periods")

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


def _source_scope(qi) -> None:
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
    """
    try:
        from mi_agent import portfolio_lens as _lens_owner
    except Exception as exc:  # noqa: BLE001 - the claim records the gap
        qi.source_scope = SourceScopeClaim(
            state=UNRESOLVABLE, source="mi_agent.portfolio_lens",
            reason="the source-portfolio lens owner is unavailable: %s" % exc)
        return
    try:
        lens = _lens_owner.resolve_lens(qi.question)
    except Exception as exc:  # noqa: BLE001
        qi.source_scope = SourceScopeClaim(
            state=UNRESOLVABLE, source="mi_agent.portfolio_lens",
            reason="the source-portfolio lens could not be resolved: %s" % exc)
        return

    name = getattr(lens, "name", None)
    if name not in SOURCE_SCOPES:
        # A lens kind this contract has no member for. Recorded as unresolvable
        # rather than mapped onto the nearest one — a substitution here would be
        # invisible downstream.
        qi.source_scope = SourceScopeClaim(
            state=UNRESOLVABLE, source="mi_agent.portfolio_lens",
            reason="the owner resolved a lens this contract cannot carry: %r"
                   % (name,))
        return

    ids = tuple(getattr(lens, "cohort_ids", ()) or ())
    if not ids and name == SCOPE_COHORT and getattr(lens, "cohort_id", None):
        ids = (lens.cohort_id,)
    label = getattr(lens, "label", None)
    # Only a NARROWING has wording in the question to point at; `total` is the
    # absence of a scope phrase, so it carries no raw_text and no span.
    raw = None if name == SCOPE_TOTAL else label
    qi.source_scope = SourceScopeClaim(
        state=FILLED, raw_text=raw, span=_span_of(qi.question, raw),
        scope=name, portfolio_ids=ids, source="mi_agent.portfolio_lens")


def _population(qi, spec, facets) -> None:
    for f in facets:
        if f.kind not in ("row_population", "cohort_comparison"):
            continue
        qi.population.append(PopulationClaim(
            state=FILLED, raw_text=f.label, span=_span_of(qi.question, f.label),
            concept=f.field_key, source="facet.%s" % f.kind))
