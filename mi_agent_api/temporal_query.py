#!/usr/bin/env python3
"""ONE calculation owner for the MI Query Agent's temporal answers.

WHAT THIS REPLACES, and why it is not a third temporal shape.

`evolution.py` grew its own arithmetic — `_bal_sum`, `_weighted_avg`,
`_breakdown`, `_group_balance` — because the dashboard's series surfaces needed
period metrics before there was a way to run the governed executor against an
arbitrary frame. The MI QUERY ROUTE then borrowed it: a "how has balance changed
over time" question was answered by that arithmetic, while the same question
without "over time" was answered by `mi_query_executor`. Two owners of SUM, and
nothing but discipline keeping them equal.

Adding TIME x DIMENSION to that arrangement would have made three. So the MI
query route's temporal answers now go through here, and here does no arithmetic
at all:

    funded_frames                       period discovery + prepared frames
      -> ONE ordinary MIQuerySpec       compiled once, from the interpretation
      -> execute_mi_query(spec, df)     THE calculation owner, once per period
      -> composition                    ordering and assembly, nothing more

THE INVARIANT THIS BUYS. For the latest period the temporal answer and the
ordinary current-period answer are the same call on the same frame, so they
cannot disagree. That is a property of the shape, not of a test — the test just
proves the shape holds.

WHAT TIME OWNS: which periods, in what order, and how the per-period results are
assembled. Nothing else. Measure, operation, filters, dimension, buckets,
geography basis, portfolio scope and availability are the executor's, exactly as
they are for a question with no time axis.

WHAT IS DELIBERATELY NOT HERE. No region normalisation, no label folding, no
taxonomy. The canonical values the executor returns are passed through
unchanged: if the upstream canonical frame carries imperfect region labels, that
is an onboarding contract to fix upstream, and repairing it here would put a
second taxonomy in the temporal layer and hide the real defect.

THE DASHBOARD IS NOT TOUCHED. `/mi/evolution/funded`, movement summaries,
cohorts, the bridge and forecast still use `evolution.py` as before. This module
is the MI query route's seam only.
"""
from __future__ import annotations

import dataclasses
from typing import Any, Callable, Dict, List, Optional, Sequence

#: Fields that express "and do this across time". They are what makes a spec a
#: TEMPORAL request, and they are exactly what a PER-PERIOD execution must not
#: carry: `chart_type="line"` would send the executor to `_execute_line`, which
#: plots a series within one frame — a different question entirely.
_TEMPORAL_WRAPPER = {
    "temporal_mode": None,
    "as_of_date": None,
    "trend_grain": None,
    "compare_periods": (),
    "baseline_date": None,
    "current_date": None,
    "x": None,
}


def period_spec(spec) -> Any:
    """The ordinary, non-temporal spec this question is asking per period.

    COMPILED ONCE, BY THE CALLER, AND REUSED. The semantic meaning of the
    question is settled before any period is looked at; each period varies the
    DATA and nothing else. Rebuilding it per period would let two periods
    resolve the same sentence differently, which is precisely the failure mode
    that makes a temporal answer untrustworthy.

    Only the temporal wrapper is removed. Measure, operation, filters,
    dimensions, portfolio lens and every other governed element survive
    untouched, because they are what the reader asked for.
    """
    changes: Dict[str, Any] = {}
    for field_name, empty in _TEMPORAL_WRAPPER.items():
        if not hasattr(spec, field_name):
            continue
        current = getattr(spec, field_name)
        if isinstance(empty, tuple):
            if current:
                changes[field_name] = list(empty)
        elif current is not None:
            changes[field_name] = empty

    # THE SHAPE THE EXECUTOR DISPATCHES ON. A grouping makes this an ordinary
    # grouped question ("bar"); without one it is an ordinary scalar
    # ("summary"). Both are shapes the executor already answers for a question
    # with no time axis — this chooses between them, it does not invent a third.
    grouped = bool(_grouping_keys(spec))
    if getattr(spec, "chart_type", None) == "line" or grouped:
        changes["chart_type"] = "bar" if grouped else "none"
    if not grouped and getattr(spec, "intent", None) == "chart":
        changes["intent"] = "summary"
    if grouped and getattr(spec, "intent", None) == "summary":
        changes["intent"] = "chart"

    if not changes:
        return spec
    return dataclasses.replace(spec, **changes)


def _grouping_keys(spec) -> List[str]:
    """The governed grouping this spec carries, in the contract's order."""
    keys: List[str] = []
    for key in (list(getattr(spec, "dimensions", None) or [])
                + [getattr(spec, "dimension", None)]):
        if key and key not in keys:
            keys.append(str(key))
    return keys


def is_grouped(spec) -> bool:
    """Whether this question asks for a breakdown as well as a time axis."""
    return bool(_grouping_keys(spec))


def execute_temporal(frames: Sequence[Dict[str, Any]], spec, semantics, *,
                     executor: Optional[Callable[..., Any]] = None,
                     missing_dimension_policy: str = "exclude",
                     ) -> Dict[str, Any]:
    """Run ONE governed spec across the prepared period frames.

    ``frames`` are what `evolution.funded_frames` returns — already prepared by
    the same `prepare_funded_mi_dataset` contract the current-period path uses,
    and already narrowed to the governed portfolio scope. This function does not
    read tapes, resolve configuration, or prepare anything: it would be a second
    owner of those too.

    RAISES on a period the spec cannot be executed against, and that is the
    governed behaviour rather than an oversight. A dimension that is genuinely
    unavailable on one required historical frame makes the SERIES unanswerable;
    dropping the period would silently change the window the reader asked about,
    and substituting a field would answer a different question. The caller
    defers to the point-in-time path, which refuses with the executor's own
    reason — the same fail-closed route `_filtered_funded_evo` already takes for
    a population it cannot apply.
    """
    from mi_agent.mi_query_executor import execute_mi_query

    run = executor or execute_mi_query
    per_period_spec = period_spec(spec)     # ONCE, outside the loop.
    grouped = is_grouped(per_period_spec)

    periods: List[Dict[str, Any]] = []
    sources: List[str] = []
    for frame in frames:
        df = frame.get("df")
        if df is None:
            continue
        result = run(per_period_spec, df, semantics, validate=False,
                     missing_dimension_policy=missing_dimension_policy)
        metadata = getattr(result, "metadata", {}) or {}
        reporting_date = frame.get("reporting_date") or frame.get("run_id")
        periods.append({
            "period": (str(reporting_date)[:7] if reporting_date
                       else frame.get("run_id")),
            "reporting_date": reporting_date,
            "run_id": frame.get("run_id"),
            # The executor's own rows, passed through. Composition orders and
            # labels periods; it does not touch a value or a category name.
            "rows": _records(result),
            "rowCount": int(getattr(result, "row_count", 0) or 0),
            "resultType": getattr(result, "result_type", None),
            # The executor's own statement of which columns it grouped on, so a
            # composer never has to guess which key in a row is the category.
            "groupKeys": tuple(metadata.get("group_field_keys") or ()),
            "reconciliation": metadata.get("reconciliation"),
        })
        if frame.get("source"):
            sources.append(str(frame["source"]))

    return {"periods": periods, "sourceFiles": sources, "grouped": grouped,
            "spec": per_period_spec}


def _records(result) -> List[Dict[str, Any]]:
    """The executor's rows as plain records, with values unchanged."""
    data = getattr(result, "data", None)
    if data is None or getattr(data, "empty", True):
        return []
    return data.to_dict(orient="records")


#: Keys the executor adds to EVERY grouped row alongside the measure. They are
#: composition, not the measure the reader asked for, so a measure search skips
#: them; `loan_count` is the measure only when the question counts rows, which
#: the caller states rather than this module inferring.
_COMPOSED_KEYS = ("loan_count", "concentration_pct")


def measure_key(rows: Sequence[Dict[str, Any]], group_keys: Sequence[str] = (),
                *, count_metric: bool) -> Optional[str]:
    """Which key in the executor's rows carries the measure. NAMING, not maths.

    The executor names its own output column from the metric and the
    aggregation (``current_outstanding_balance_sum``), so the alternative to
    reading it back is rebuilding that name here — a second owner of the
    executor's column naming, wrong the day an aggregation is added.
    """
    if count_metric:
        return "loan_count"
    skip = set(group_keys or ()) | set(_COMPOSED_KEYS)
    for row in rows or ():
        for key in row:
            if key not in skip:
                return key
    return None


def series_by_category(periods: Sequence[Dict[str, Any]], group_key: str,
                       value_key: str) -> Dict[str, List[Optional[float]]]:
    """``{category: [value per period]}`` — composition, not calculation.

    A category absent from one period's rows is ``None`` for that period, which
    is "no observations here", NOT zero and NOT a missing dimension. Those are
    different states and the presentation contract keeps them different: a
    fabricated 0 would read as "the book had nothing there", which is a claim
    this layer has no evidence for.
    """
    categories: List[str] = []
    for period in periods:
        for row in period.get("rows") or []:
            label = row.get(group_key)
            if label is not None and str(label) not in categories:
                categories.append(str(label))
    out: Dict[str, List[Optional[float]]] = {}
    for category in categories:
        values: List[Optional[float]] = []
        for period in periods:
            match = next((r for r in (period.get("rows") or [])
                          if str(r.get(group_key)) == category), None)
            values.append(None if match is None else match.get(value_key))
        out[category] = values
    return out
