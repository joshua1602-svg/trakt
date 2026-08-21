#!/usr/bin/env python3
"""clause_splitting_phase1/rc_projection.py

Projects the release candidate's ``MIQuerySpec`` into the six-span model, so
the RC can be scored by the same probes as a tree that has real spans.

The RC has no spans. It has a flat spec whose slots nevertheless carry the same
information under different names, and the projection reads them out. This is
the ONLY honest way to score the RC on a classification instrument: without it
the RC scores zero on everything by definition, which would prove nothing.

The projection is deliberately GENEROUS to the RC. Wherever the RC carries a
signal that is equivalent to a span, the projection credits it:

  * ``chart_type == "line"`` is read as a time-axis grouping, even though the
    RC has no grain and no grouping slot holding it. Without this the RC would
    fail every time-series probe on a technicality rather than on capability.
  * ``aggregation == "count"`` with no metric is read as a subject of
    ``loan_count`` — the counted population — rather than an empty subject.
  * ``ranking_mode`` / ``top_n`` are read as a ranking operation even though
    the RC also sets an aggregation underneath.

Nothing here is imported by product code, and no product behaviour depends on
it. It reads a spec that already exists.
"""
from __future__ import annotations

from typing import Any, Optional

from .span_model import FILLED, UNRESOLVABLE, Span, Spans, filled

#: Aggregations the RC uses, mapped to operation classes.
_AGG_TO_OPERATION = {
    "count": "count",
    "count_distinct": "count",
    "sum": "amount",
    "balance_sum": "amount",
    "avg": "average_or_rate",
    "weighted_avg": "average_or_rate",
    "median": "average_or_rate",
    "distribution": "neutral",
    "loan_level": "neutral",
}

#: RC forecast-question kinds that are forward-looking operations.
_FORWARD_KINDS = {
    "reach_threshold", "run_rate", "run_rate_annualised", "scenario",
    "conversion", "pipeline_needed", "compare_models", "extrapolation_curve",
}


def _operation(spec: Any) -> Span:
    """Operation class, in the RC's own precedence order.

    Forward beats ranking beats movement beats the aggregation, because that is
    the order the RC's own recognisers run in (``_deterministic_parse`` checks
    the forecast recogniser first, then bridge/cohort/compare, then risk).
    """
    if getattr(spec, "forecast_question", None) in _FORWARD_KINDS or \
       getattr(spec, "forecast_mode", None):
        return filled(["forward"], text="<rc: forecast recogniser>")
    if getattr(spec, "ranking_mode", None) or getattr(spec, "top_n", None):
        return filled(["ranking"], text="<rc: ranking_mode/top_n>")
    if getattr(spec, "compare_periods", None) or \
       getattr(spec, "temporal_mode", None) == "compare" or \
       getattr(spec, "bridge_query", False):
        return filled(["movement"], text="<rc: compare/bridge recogniser>")
    agg = getattr(spec, "aggregation", None)
    mapped = _AGG_TO_OPERATION.get(agg)
    if mapped is None:
        return Span()
    return filled([mapped], text="<rc: aggregation=%s>" % agg)


def _subject(spec: Any) -> Span:
    metric = getattr(spec, "metric", None)
    if metric:
        return filled([metric], text="<rc: metric>")
    agg = getattr(spec, "aggregation", None)
    if agg in ("count", "count_distinct"):
        # A counted population with no measure: the subject IS the population.
        return filled(["loan_count"], text="<rc: aggregation=count, no metric>")
    return Span()


def _time_axis(spec: Any) -> Optional[str]:
    """The RC's only time-axis signal, read as generously as it can be read."""
    grain = getattr(spec, "trend_grain", None)
    if grain:
        return "time:%s" % {"daily": "day", "weekly": "week",
                            "monthly": "month", "quarterly": "quarter"}.get(grain, grain)
    if getattr(spec, "temporal_mode", None) == "trend":
        return "time:unspecified"
    if getattr(spec, "chart_type", None) == "line":
        return "time:unspecified"
    if getattr(spec, "cohort_progression", False):
        return "time:unspecified"
    return None


def _grouping(spec: Any) -> Span:
    concepts = []
    for key in ("dimension", "bridge_dimension", "migration_dimension",
                "concentration_dimension", "risk_dimension"):
        v = getattr(spec, key, None)
        if v:
            concepts.append(v)
    for key in ("dimensions", "hierarchy"):
        for v in getattr(spec, key, None) or []:
            if v:
                concepts.append(v)
    axis = _time_axis(spec)
    if axis:
        concepts.append(axis)
    # De-duplicate, preserving nothing but the set — spans are unordered here.
    return filled(sorted(set(concepts)), text="<rc: dimension slots + chart_type>")


def _filter(spec: Any) -> Span:
    keys = list((getattr(spec, "filters", None) or {}).keys())
    keys += list((getattr(spec, "state_filters", None) or {}).keys())
    return filled(sorted(set(keys)), text="<rc: filters + state_filters>")


def _period(spec: Any) -> Span:
    """Period-kind tokens the RC's date slots imply.

    The RC has no period span. It has absolute date slots that a recogniser
    fills, plus ``compare_periods``. A relative window (*the last three
    months*) has nowhere to live unless a recogniser resolved it to dates, so
    this projection reports what the RC actually carries, not what the question
    said.
    """
    kinds = []
    if getattr(spec, "compare_periods", None):
        kinds.append("comparison")
    if getattr(spec, "as_of_date", None) or getattr(spec, "reporting_date", None) \
       or getattr(spec, "cut_off_date", None):
        kinds.append("as_at")
    if getattr(spec, "start_date", None) or getattr(spec, "end_date", None):
        kinds.append("explicit_range")
    if getattr(spec, "baseline_date", None) or getattr(spec, "current_date", None):
        kinds.append("comparison")
    return filled(sorted(set(kinds)), text="<rc: date slots>")


def _target(spec: Any) -> Span:
    v = getattr(spec, "forecast_target_value", None)
    if v is None:
        return Span()
    # Normalise to a stable string so probes can declare an exact threshold.
    if float(v) == int(float(v)):
        return filled(["value:%d" % int(float(v))], text="<rc: forecast_target_value>")
    return filled(["value:%s" % v], text="<rc: forecast_target_value>")


def project(spec: Any) -> Spans:
    """Project one RC ``MIQuerySpec`` into the six-span model."""
    s = Spans(
        operation=_operation(spec),
        subject=_subject(spec),
        grouping=_grouping(spec),
        filter=_filter(spec),
        period=_period(spec),
        target=_target(spec),
    )
    # The RC has no channel for an unresolvable span and no channel for
    # unrecognised residue. Both stay empty by construction, and the scorer
    # records that as a structural miss rather than a wrong answer.
    s.notes.append("rc-projection: no unresolvable channel exists in the RC")
    return s


# --------------------------------------------------------------------------- #
# Workflow-aware projection.
#
# The parse-only projection above reads `_deterministic_parse` and nothing else.
# That understates the release candidate badly, and the first re-baseline run on
# the correct tree proved it: four probes it scored as failures produce correct,
# specific governed clarifications end to end — including
# "'large loans' does not state a threshold I can apply", which is exactly the
# filled-but-unresolvable filter the probe asked for.
#
# The release candidate expresses a large part of its classification AFTER
# parsing, in `execution_receipt`: a facet layer that detects material facets
# from the RAW QUESTION before any parsing decision, carries a field key per
# facet, and reconciles each against what execution actually did. Its statuses
# map onto this model's span states directly:
#
#   applied                              -> filled
#   unavailable / unsupported / rejected -> filled-but-unresolvable (disclosed)
#   lost                                 -> filled-but-unresolvable (fails closed)
#
# So the release candidate DOES have an unresolvable channel. It is not in the
# spec, which is why a parse-only reading cannot see it.
# --------------------------------------------------------------------------- #

#: Facet kind -> the span it bears on.
_FACET_SPAN = {
    "geographic_scope": "filter",
    "row_population": "filter",
    "threshold": "filter",
    "grouping_dimension": "grouping",
    "ranking": "operation",
    "requested_statistic": "operation",
    "comparison_period": "period",
    "projection": "operation",
    "multi_measure": "subject",
    "unresolved_measure": "subject",
    "relationship": "subject",
    "aggregate_contribution": "operation",
}


def project_with_receipt(spec: Any, receipt: Any) -> Spans:
    """Project a release-candidate result using its spec AND its receipt.

    ``receipt`` is the ``execution_receipt`` dict the workflow returns. A facet
    that did not apply marks its span unresolvable, carrying the facet's own
    label as the unmatched wording — which is what the release candidate would
    say to the user.
    """
    spans = project(spec)
    spans.notes = [n for n in spans.notes
                   if "no unresolvable channel" not in n]

    facets = (receipt or {}).get("facets") or []
    for f in facets:
        if not isinstance(f, dict):
            continue
        status = f.get("status")
        if status == "applied":
            continue
        span_name = _FACET_SPAN.get(f.get("kind"))
        if span_name is None:
            spans.notes.append("facet %s (%s) has no span mapping"
                               % (f.get("kind"), status))
            continue
        span = getattr(spans, span_name)
        span.state = UNRESOLVABLE
        span.unmatched = f.get("label") or f.get("kind")
        # A facet that did not apply must NOT leave its concept sitting in the
        # span as though it had: that is the silent-narrowing shape.
        field_key = f.get("field")
        if field_key and field_key in span.concepts:
            span.concepts = [c for c in span.concepts if c != field_key]
    return spans
