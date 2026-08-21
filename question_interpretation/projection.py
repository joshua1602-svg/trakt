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
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

from .schema import (
    AMOUNT, AVERAGE, CONFIGURED, COUNT, EMPTY, FILLED, FILTER, FORWARD,
    GRAINS, GROUPING, MOVEMENT, NEUTRAL, RANKING, STATED, UNRESOLVABLE,
    UNRESOLVED_ROLE, DimensionClaim, FilterClaim, OperationClaim,
    PopulationClaim, QuestionInterpretation, Slot, Span, SubjectClaim,
    TargetClaim, TimeClaim,
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


def project(question: str, *, semantics: dict, frame=None) -> QuestionInterpretation:
    """Build a QuestionInterpretation by asking the existing interpreters."""
    from mi_agent import answer_type as AT
    from mi_agent import execution_receipt as R
    from mi_agent import period_request as PR
    from mi_agent.llm_query_parser import _deterministic_parse

    qi = QuestionInterpretation(question=question)
    spec, _meta = _deterministic_parse(question, semantics)

    cols = list(frame.columns) if frame is not None and hasattr(frame, "columns") else None
    dim_terms = R.requested_dimension_terms(question, semantics, cols)
    facets = R.detect_requested_facets(question, semantics, frame=frame,
                                       requested_dimensions=dim_terms)
    facet_kinds = {f.kind for f in facets}

    _operation(qi, spec, facet_kinds, AT)
    _subject(qi, spec)
    _dimensions(qi, spec, dim_terms, facets)
    _filters(qi, spec, facets)
    _time(qi, spec, PR)
    _target(qi, spec, facets)
    _population(qi, spec, facets)
    return qi


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
    metric = getattr(spec, "metric", None)
    if metric:
        qi.subject = SubjectClaim(state=FILLED, candidate_concept=metric,
                                  source="parser.metric")
        return
    if getattr(spec, "aggregation", None) in ("count", "count_distinct"):
        qi.subject = SubjectClaim(state=FILLED, candidate_concept="loan_count",
                                  source="parser.aggregation=count")
        return
    qi.notes.append("subject: no source supplies it")


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
            role=role, candidate_concept=key, source=src))

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
        qi.filters.append(FilterClaim(
            state=FILLED, raw_text=f.label, span=_span_of(qi.question, f.label),
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
        qi.filters.append(FilterClaim(
            state=FILLED, raw_text=None, operator=op,
            value=None if val is None else str(val), categorical_value=cat,
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
    if re.search(r"\b(on target|against target|versus plan|vs plan|"
                 r"versus budget|vs budget)\b", qi.question, re.I):
        qi.target = TargetClaim(
            state=UNRESOLVABLE, raw_text="target",
            target_source=CONFIGURED,
            reason="threshold referenced from configuration; no source supplies it",
            source="projection(no interpreter supplies a configured target)")
        qi.notes.append("target: configured sense present, no source supplies it")


def _population(qi, spec, facets) -> None:
    for f in facets:
        if f.kind not in ("row_population", "cohort_comparison"):
            continue
        qi.population.append(PopulationClaim(
            state=FILLED, raw_text=f.label, span=_span_of(qi.question, f.label),
            concept=f.field_key, source="facet.%s" % f.kind))
