#!/usr/bin/env python3
"""mi_agent/mi_query_contract.py

The formal contract for a parsed MI query, plus the fail-closed dimension
invariant and the end-to-end query trace.

Contract (the fields a parsed MI query carries — see :class:`MIQuerySpec`):
    intent               chart | table | summary
    metric(s)            spec.metric (+ derived-metric definition)
    dimensions/group_bys spec.dimensions (ordered) and/or spec.dimension
    filters              spec.filters (applied) / spec.unavailable_filters (rejected)
    date/reporting scope  temporal_mode, compare_periods, as_of_date, run_id
    portfolio scope       spec.portfolio_lens (total | direct | acquired | cohort)
    chart/table pref      spec.chart_type / spec.output_format
    sort/rank/top-N       spec.ranking_mode, sort_by, sort_direction, top_n, limit
    aggregation           spec.aggregation
    weighted-average      spec.weight_field (required when aggregation == weighted_avg)
    output shape          result_type (summary | table | loan_level) + artifacts
    reconciliation        input/included/excluded records + balance + coverage %

THE INVARIANT (fail closed):
    Every grouping dimension the parser attached to a spec MUST be either
      (a) applied in execution (present in the executor's group columns / result
          columns), OR
      (b) explicitly rejected with a reason (recorded in the executor's
          ``rejected_dimensions`` metadata).
    A parsed dimension that is neither applied nor rejected is a SILENT DROP and
    the query is refused rather than answered with a misleading result.

Pure and deterministic: no I/O, no LLM. Trivially unit-testable.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


def _fields(semantics: dict) -> Dict[str, dict]:
    return semantics.get("fields", {}) if isinstance(semantics, dict) else {}


def canonical_of(key: str, semantics: dict) -> str:
    """The canonical dataframe column for a semantic field key."""
    return (_fields(semantics).get(key, {}) or {}).get("canonical_field", key)


def business_name_of(key: str, semantics: dict) -> str:
    e = _fields(semantics).get(key, {}) or {}
    return e.get("business_name") or e.get("display_name") or key


def _spec_get(spec, attr: str):
    """Read a spec field whether ``spec`` is a MIQuerySpec object or a serialised
    dict (the workflow stores the object; the API response / harness see a dict)."""
    if isinstance(spec, dict):
        return spec.get(attr)
    return getattr(spec, attr, None)


def all_group_dims(spec) -> List[str]:
    """Every grouping dimension the parser attached to the spec, in order and
    de-duplicated (``dimensions`` first, then ``dimension``, then a hierarchy).
    This mirrors the executor's own ``_all_group_dims`` — the authoritative set
    that must be grouped by or rejected. Accepts an object or a dict spec."""
    out: List[str] = []
    dimension = _spec_get(spec, "dimension")
    seq = (list(_spec_get(spec, "dimensions") or [])
           + ([dimension] if dimension else [])
           + list(_spec_get(spec, "hierarchy") or []))
    for k in seq:
        if k and k not in out:
            out.append(k)
    return out


@dataclass
class InvariantResult:
    """Outcome of the parsed-vs-executed dimension check."""

    ok: bool
    applied: List[str] = field(default_factory=list)
    rejected: List[Dict[str, Any]] = field(default_factory=list)
    dropped: List[Dict[str, Any]] = field(default_factory=list)

    def message(self) -> str:
        if self.ok:
            return "all parsed dimensions were applied or explicitly rejected"
        names = ", ".join(d["dimension"] for d in self.dropped)
        return (f"parsed dimension(s) neither applied nor rejected: {names}. "
                "Refusing to answer with a silently dropped dimension.")


def check_dimension_invariant(spec, query_result, semantics: dict) -> InvariantResult:
    """The fail-closed dimension invariant.

    A parsed grouping dimension is *applied* when its semantic key is in the
    executor's ``group_field_keys`` OR its canonical column is in the result
    columns. It is *rejected* when recorded in ``rejected_dimensions``. Anything
    else is a silent drop → ``ok=False``.
    """
    parsed = all_group_dims(spec)
    if not parsed:
        return InvariantResult(ok=True)

    meta = getattr(query_result, "metadata", None) or {}
    group_keys = set(meta.get("group_field_keys") or [])
    rejected_meta = list(meta.get("rejected_dimensions") or [])
    rejected_keys = {r.get("dimension") for r in rejected_meta}

    result_cols: set = set()
    data = getattr(query_result, "data", None)
    if data is not None and hasattr(data, "columns"):
        result_cols = {str(c) for c in data.columns}

    applied: List[str] = []
    dropped: List[Dict[str, Any]] = []
    for key in parsed:
        canonical = canonical_of(key, semantics)
        is_applied = key in group_keys or canonical in result_cols
        if is_applied:
            applied.append(key)
        elif key in rejected_keys:
            continue  # explicitly rejected with a reason — allowed
        else:
            dropped.append({
                "dimension": key,
                "canonical": canonical,
                "business_name": business_name_of(key, semantics),
                "reason": "parsed but not present in the executed group columns "
                          "or result — silent drop",
            })
    return InvariantResult(ok=not dropped, applied=applied,
                           rejected=rejected_meta, dropped=dropped)


def _chart_axes(artifacts: Optional[List[Dict[str, Any]]]) -> Optional[Dict[str, Any]]:
    """The axis mapping of the first chart artifact, for the trace."""
    for art in artifacts or []:
        if art.get("type") == "chart":
            return {
                "chartType": art.get("chartType"),
                "xKey": art.get("xKey"),
                "yKey": art.get("yKey"),
                "valueKey": art.get("valueKey"),
                "seriesKeys": [s.get("key") for s in (art.get("series") or [])],
            }
    return None


def build_query_trace(*, question: str, spec, parse_meta: Optional[dict],
                      query_result, semantics: dict,
                      invariant: Optional[InvariantResult] = None,
                      artifacts: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
    """Assemble the end-to-end query trace so it is immediately obvious whether a
    problem is parser-, executor- or renderer-side."""
    meta = getattr(query_result, "metadata", None) or {}
    parsed = all_group_dims(spec)
    inv = invariant or (check_dimension_invariant(spec, query_result, semantics)
                        if query_result is not None else InvariantResult(ok=True))

    result_cols: List[str] = []
    data = getattr(query_result, "data", None)
    if data is not None and hasattr(data, "columns"):
        result_cols = [str(c) for c in data.columns]

    return {
        "rawQuery": question,
        "normalisedQuery": (question or "").strip().lower(),
        "intent": getattr(spec, "intent", None),
        "parserMode": (parse_meta or {}).get("parser_mode"),
        "parserConfidence": (parse_meta or {}).get("parser_confidence"),
        "metric": getattr(spec, "metric", None),
        "aggregation": getattr(spec, "aggregation", None),
        "weightField": getattr(spec, "weight_field", None),
        "dimensionsParsed": [
            {"key": k, "canonical": canonical_of(k, semantics),
             "businessName": business_name_of(k, semantics)} for k in parsed],
        "filtersParsed": dict(getattr(spec, "filters", None) or {}),
        "rejectedDimensions": inv.rejected,
        "rejectedFilters": [
            {"filter": f, "reason": "field unavailable in this dataset"}
            for f in (getattr(spec, "unavailable_filters", None) or [])],
        "executedGroupFieldKeys": list(meta.get("group_field_keys") or []),
        "executedGroupCols": [canonical_of(k, semantics)
                              for k in (meta.get("group_field_keys") or [])],
        "resultType": getattr(query_result, "result_type", None),
        "resultColumns": result_cols,
        "chartAxes": _chart_axes(artifacts),
        "topN": getattr(spec, "top_n", None),
        "sortBy": getattr(spec, "sort_by", None),
        "sortDirection": getattr(spec, "sort_direction", None),
        "portfolioLens": getattr(spec, "portfolio_lens", None),
        "reconciliation": meta.get("reconciliation"),
        "invariant": {
            "ok": inv.ok,
            "appliedDimensions": inv.applied,
            "droppedDimensions": inv.dropped,
            "message": inv.message(),
        },
    }
