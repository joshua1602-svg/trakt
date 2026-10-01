"""THE SEMANTIC ENGINE: one reader for every capability's semantic model
(P0 design §16, §20).

A capability declares its figures in `config/mi/semantic_model/<capability>.yaml`
(`mi_agent.semantic_model`); this module is the ONE engine that serves them. The
Cortex shape: the model declares what a figure is and where its owner publishes
it; the engine resolves a plan against that declaration and reads the figure; a
new figure is a new entry in a file, not a new branch in a runtime. The forecast
and the pipeline stage-movement capabilities both serve their declared figures
through it, and any capability that gains a model does the same.

What it does, all of it by lookup in the owner's published output:

    check(...)          can the owner's output answer THIS plan — one figure,
                        one governed member, one breakdown it publishes, or its
                        own curve to a stated horizon?
    serve_figure(...)   read that figure: `(shape, value, cells, paths, extra)`
    period_change(...)  the change between two governed figures of one measure
                        — the ONE place such a change is computed, for every
                        capability that states one (a pinned test holds the
                        engine to these two operators and nowhere else)

What it never does: add up members, re-group rows, project, interpolate, or
read a question. A capability's own perimeter — a held reading, a milestone
rule, which inputs a view needs and how old they may be — stays with that
capability; the engine is what they share.
"""
from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Tuple

from mi_agent import plan_reading as _plan
from mi_agent import semantic_model as _model

__all__ = ["Refusal", "check", "binding_for", "member_of", "requires_met",
           "serve_figure", "inputs_available", "inputs_used", "receipt",
           "period_change",
           "MEASURE_NOT_SUPPORTED", "OPERATION_NOT_SUPPORTED",
           "PERIOD_NOT_SUPPORTED", "TARGET_NOT_SUPPORTED",
           "DIMENSION_NOT_SUPPORTED", "FILTERS_NOT_SUPPORTED",
           "GEOGRAPHY_NOT_SUPPORTED", "AMBIGUOUS_READING", "FIELD_UNAVAILABLE",
           "FIGURE_UNAVAILABLE", "FIGURE_WITHHELD"]

# Reasons. Stable strings: the evidence ledger groups by them, and each
# capability's runtime re-exports the ones it states.
MEASURE_NOT_SUPPORTED = "MEASURE_NOT_SUPPORTED"
OPERATION_NOT_SUPPORTED = "OPERATION_NOT_SUPPORTED"
PERIOD_NOT_SUPPORTED = "PERIOD_NOT_SUPPORTED"
TARGET_NOT_SUPPORTED = "TARGET_NOT_SUPPORTED"
DIMENSION_NOT_SUPPORTED = "DIMENSION_NOT_SUPPORTED"
FILTERS_NOT_SUPPORTED = "FILTERS_NOT_SUPPORTED"
GEOGRAPHY_NOT_SUPPORTED = "GEOGRAPHY_NOT_SUPPORTED"
#: The owner computes this shape, but a live run showed the model using it for
#: questions that want different figures. Held, not served.
AMBIGUOUS_READING = "AMBIGUOUS_READING"
#: The owner's output carries no figure for what the plan asked (a member it did
#: not publish, a breakdown on another basis). Refused, never substituted.
FIELD_UNAVAILABLE = "FIELD_UNAVAILABLE"
#: The owner published no value for the figure itself.
FIGURE_UNAVAILABLE = "FIGURE_UNAVAILABLE"
#: The owner does not state the figure, and says why (D21: it depends on a
#: stage rate the client's history cannot yet measure).
FIGURE_WITHHELD = "FIGURE_WITHHELD"


class Refusal(Exception):
    """A plan the owner's output cannot answer, with the reason and why."""

    def __init__(self, reason: str, detail: str):
        super().__init__(f"{reason}: {detail}")
        self.reason = reason
        self.detail = detail


# --------------------------------------------------------------------------- #
# the plan, as the model reads it
# --------------------------------------------------------------------------- #

def member_of(plan: Any) -> Optional[Tuple[str, str]]:
    """The one governed member a plan filters on, e.g. `(pipeline_stage,
    OFFER)`; None when it filters on nothing (or on more than one thing)."""
    filters = _plan.plan_filters(plan)
    return _plan.member_filter(filters[0]) if len(filters) == 1 else None


def binding_for(measure: _model.Measure, plan: Any
                ) -> Tuple[Optional[str], Optional[Tuple[str, str]], Mapping[str, Any]]:
    """`(axis, member, binding)`: the breakdown axis or the filtered member the
    plan names, and the semantic model's binding of it for this measure."""
    axes = _plan.grouping_axes(plan)
    axis = axes[0] if axes else None
    member = member_of(plan)
    key = axis or (member[0] if member else "")
    return axis, member, measure.by.get(key, {})


# --------------------------------------------------------------------------- #
# the perimeter every declared figure shares
# --------------------------------------------------------------------------- #

def names_window(m: _model.Measure, period: Mapping[str, Any]) -> bool:
    """Does the plan's period name one of the figure's published windows — a
    `range` of a whole number of periods at the window's grain (D22)?"""
    back = period.get("periods_back")
    return bool(m.window) and str(period.get("form") or "") == "range" \
        and str(period.get("grain") or "") == m.window.get("grain") \
        and isinstance(back, int) and not isinstance(back, bool) and back >= 1


def period_as_stated(m: _model.Measure, period: Mapping[str, Any]
                     ) -> Mapping[str, Any]:
    """The plan's period as THIS figure reads it.

    A figure stated only looking forward — a projection — has no "now": the
    current state, or no period at all, is the figure as its owner publishes
    it, over its own horizon ("the base forecast" is the base line of the
    forecast, not a figure for today). The same rule as a change form owning
    its window (`plan_reading.names_latest_pair`): where the question states
    no horizon, the owner's is read, and the answer states it. A stated
    horizon, or any other period, is read as stated.
    """
    form = str(period.get("form") or "")
    if (form in ("", "current") and "forward_looking" in m.periods
            and "current" not in m.periods):
        return {**dict(period), "form": "forward_looking"}
    return period


def check(model: _model.SemanticModel, plan: Any, measure: str, operation: str,
          *, held: Optional[Mapping[Tuple[str, str], str]] = None
          ) -> Tuple[bool, str, str]:
    """Can the owner's output answer THIS plan by lookup? `(ok, reason, detail)`.

    `held` is the capability's own list of `(operation, measure)` readings a
    live run showed going to the wrong questions; they are refused
    `AMBIGUOUS_READING` until a run shows the readings have moved.
    """
    body = _plan.as_mapping(plan)
    m = model.measure(measure)
    if m is None:
        return (False, MEASURE_NOT_SUPPORTED,
                f"measure {measure!r} is not in the {model.capability} semantic "
                f"model (served: {sorted(model.measures)})")
    if operation not in m.operations:
        return (False, OPERATION_NOT_SUPPORTED,
                f"operation={operation!r} is not how {measure} is served "
                f"({sorted(m.operations)})")
    period = body.get("period") or {}
    # THE GRAIN FIRST: it is the durable refusal. "By month" asks for a figure
    # per period whatever the measure turns out to mean, so a point figure
    # refuses it before a hold is even consulted. A WINDOW's grain is the unit
    # its length is counted in, not a figure per period.
    grain = period.get("grain")
    if grain and grain not in m.grains and not names_window(m, period):
        return (False, PERIOD_NOT_SUPPORTED,
                f"a {grain} grain asks for a figure per period; {measure} is "
                f"not published per {grain}")
    why_held = (held or {}).get((operation, measure))
    if why_held:
        return (False, AMBIGUOUS_READING,
                f"{operation}/{measure} is held until a live run shows the "
                f"readings have moved to their own concepts: {why_held}")
    if isinstance(body.get("target"), Mapping):
        return (False, TARGET_NOT_SUPPORTED,
                f"{measure} takes no threshold")
    if _plan.asks_geography(body) and _plan.region_axis(body) is None:
        return (False, GEOGRAPHY_NOT_SUPPORTED,
                f"{model.views[m.view].label} publishes figures by the client's "
                f"reporting region only; a region filter or another level is "
                f"not a figure it publishes")
    axes = _plan.grouping_axes(body)
    if len(axes) > 1:
        return (False, DIMENSION_NOT_SUPPORTED,
                f"the owner publishes one breakdown at a time; this plan "
                f"groups by {axes}")
    if axes and axes[0] not in m.by:
        return (False, DIMENSION_NOT_SUPPORTED,
                f"the owner publishes {measure} by {sorted(m.by) or 'nothing'}, "
                f"not by {axes[0]!r}")
    filters = _plan.plan_filters(body)
    member = member_of(body)
    if filters:
        if member is None or axes:
            return (False, FILTERS_NOT_SUPPORTED,
                    "one equality on one governed value is served, and not "
                    "together with a breakdown")
        dim, value = member
        if dim not in m.by:
            return (False, FILTERS_NOT_SUPPORTED,
                    f"the owner publishes {measure} per {sorted(m.by) or 'nothing'}, "
                    f"not per {dim!r}")
        governed = model.dimensions[dim].values if dim in model.dimensions else ()
        if governed and value not in governed:
            return (False, FILTERS_NOT_SUPPORTED,
                    f"{dim}={value!r} is not a governed value "
                    f"({', '.join(governed)})")
    if not (axes or filters or m.value or m.series):
        return (False, DIMENSION_NOT_SUPPORTED,
                f"{measure} is published per {sorted(m.by)}; break it down")
    period = period_as_stated(m, period)
    form = str(period.get("form") or "")
    if form not in m.periods:
        return (False, PERIOD_NOT_SUPPORTED,
                f"period.form={form!r} is not how {measure} is stated "
                f"({sorted(m.periods)})")
    if form == "range" and m.window:
        if not names_window(m, period):
            return (False, PERIOD_NOT_SUPPORTED,
                    f"{measure} is published over a window of whole "
                    f"{m.window['grain'].replace('ly', '')}s, stated as a "
                    f"number of them (periods_back) at grain "
                    f"{m.window['grain']!r}; this plan states "
                    f"{ {k: period.get(k) for k in ('grain', 'periods_back', 'labels')} }")
        if filters:
            return (False, FILTERS_NOT_SUPPORTED,
                    f"{measure} over a named window is one figure; its "
                    f"members are published for the owner's own window only")
    ahead = period.get("periods_ahead")
    if ahead is not None and not m.series:
        return (False, PERIOD_NOT_SUPPORTED,
                f"{measure} is one figure, not a projection over a horizon")
    if m.series and period.get("labels") and ahead is None:
        return (False, PERIOD_NOT_SUPPORTED,
                "the horizon was stated only in words; a curve is selected "
                "to a stated number of periods ahead, never to a label")
    return True, "", ""


# --------------------------------------------------------------------------- #
# reading the owner's output — lookups only
# --------------------------------------------------------------------------- #

def requires_met(binding: Mapping[str, Any], payload: Mapping[str, Any],
                 what: Any) -> None:
    """A breakdown served only on a governed basis (a region breakdown in the
    reporting taxonomy) is refused on any other, never substituted."""
    for path, expected in (binding.get("requires") or {}).items():
        got = _model.read(payload, path)
        if got != expected:
            raise Refusal(FIELD_UNAVAILABLE,
                          f"the owner published {what} on {got!r}, not "
                          f"{expected!r}; it is not served on another basis")


def serve_figure(m: _model.Measure, payload: Mapping[str, Any], *,
                 axis: Optional[str], member: Optional[Tuple[str, str]],
                 binding: Mapping[str, Any], period: Mapping[str, Any],
                 unavailable: str = FIGURE_UNAVAILABLE
                 ) -> Tuple[str, Any, Optional[List[Dict[str, Any]]], Any, Dict[str, Any]]:
    """`(shape, value, cells, paths read, extra receipt facts)` for the figure
    the plan resolved to — the owner's value, member, breakdown or curve."""
    if names_window(m, period):
        return _windowed(m, payload, int(period["periods_back"]),
                         unavailable=unavailable)
    if m.series:
        return _series(m, payload, member=member, period=period,
                       unavailable=unavailable)
    if axis:
        cells = _breakdown(axis, payload, binding)
        if not cells:
            raise Refusal(FIELD_UNAVAILABLE,
                          f"the owner published no {m.name} by {axis}")
        extra = ({"axis_basis": _model.read(payload, binding["basis"])}
                 if binding.get("basis") else {})
        withheld = _withheld(m, payload)
        if withheld and any(c.get("value") is None for c in cells):
            extra["withheld"] = withheld
        flag = binding.get("provisional_unless")
        if flag:
            extra["provisional_members"] = [
                c[axis] for c in cells if not c.get(flag)]
        notes = [str(c["note"]) for c in cells
                 if binding.get("note") and c.get("note")]
        if notes:
            extra["member_notes"] = list(dict.fromkeys(notes))
        path = binding.get("rows") or binding.get("map") or {
            k: spec.get("value") for k, spec in (binding.get("members") or {}).items()}
        return "grouped", None, cells, path, extra
    if member:
        value, path = _member(member, payload, binding)
        if value is None and _withheld(m, payload):
            raise Refusal(FIGURE_WITHHELD, _withheld(m, payload))
        if value is None:
            raise Refusal(FIELD_UNAVAILABLE,
                          f"the owner published no {m.name} for "
                          f"{member[0]}={member[1]!r}")
        return "scalar", value, None, path, _evidence(member, payload, binding)
    value = _model.read(payload, m.value)
    if value is None and _withheld(m, payload):
        raise Refusal(FIGURE_WITHHELD, _withheld(m, payload))
    if value is None:
        raise Refusal(unavailable, f"the owner published no {m.name} ({m.value})")
    return "scalar", value, None, m.value, {}


def _withheld(m: _model.Measure, payload: Mapping[str, Any]) -> str:
    """The owner's own reason for not stating the figure, if it gives one."""
    return str(_model.read(payload, m.withheld) or "") if m.withheld else ""


def _windowed(m: _model.Measure, payload: Mapping[str, Any], length: int, *,
              unavailable: str
              ) -> Tuple[str, Any, None, str, Dict[str, Any]]:
    """The figure over the window of `length` periods, as its owner published
    it. A window the owner's history does not cover is refused, never
    shortened, stretched or scaled from another."""
    w = m.window
    rows = _model.read(payload, w["rows"]) or []
    lengths = sorted(int(r[w["key"]]) for r in rows
                     if isinstance(r.get(w["key"]), int))
    row = next((r for r in rows if r.get(w["key"]) == length), None)
    unit = str(w["grain"]).replace("ly", "")
    if row is None:
        covered = (f"{lengths[0]} to {lengths[-1]} {unit}s" if lengths
                   else f"no whole window of {unit}s")
        raise Refusal(PERIOD_NOT_SUPPORTED,
                      f"the owner's history covers {m.label.lower()} over "
                      f"{covered}; a {length}-{unit} window is not one it can "
                      f"state from the history it holds")
    value = row.get(w["value"])
    path = f"{w['rows']}[{w['key']}={length}].{w['value']}"
    if value is None:
        raise Refusal(unavailable,
                      f"the owner published no {m.label.lower()} over "
                      f"{length} {unit}s: {row}")
    return "scalar", value, None, path, {
        "window": {"length": length, "unit": unit,
                   **{k: row.get(k) for k in (w.get("also") or ())}}}


def _member(member: Tuple[str, str], payload: Mapping[str, Any],
            binding: Mapping[str, Any]) -> Tuple[Any, str]:
    value = member[1]
    if "members" in binding:
        path = binding["members"][value]["value"]
        return _model.read(payload, path), path
    if "map" in binding:
        row = (_model.read(payload, binding["map"]) or {}).get(value)
        path = f"{binding['map']}.{value}.{binding['value']}"
        return (row or {}).get(binding["value"]), path
    rows = _model.read(payload, binding.get("rows", "")) or []
    found = next((r for r in rows if str(r.get(binding["key"])) == value), None)
    path = f"{binding.get('rows')}[{binding.get('key')}={value}].{binding.get('value')}"
    return (found or {}).get(binding.get("value")), path


def _axis_label(m: _model.Measure, axis: Optional[str]) -> Optional[str]:
    return (m.by.get(axis) or {}).get("label") if axis else None


def _evidence(member: Tuple[str, str], payload: Mapping[str, Any],
              binding: Mapping[str, Any]) -> Dict[str, Any]:
    """The owner's own evidence for ONE member of a map — the counts it
    declares beside the figure, and whether it flagged the figure as measured
    on too few cases. Read, never judged here."""
    if "map" not in binding:
        return {}
    row = (_model.read(payload, binding["map"]) or {}).get(member[1]) or {}
    out: Dict[str, Any] = {}
    also = tuple(binding.get("also") or ())
    if also:
        out["member_evidence"] = {name: row.get(name) for name in also}
    flag = binding.get("provisional_unless")
    if flag:
        out["provisional"] = not row.get(flag)
    # The owner's own caveat for this member, stated with the figure (D27:
    # "the forecast weights no KFI case").
    note = row.get(binding["note"]) if binding.get("note") else None
    if note:
        out["member_notes"] = [str(note)]
    return out


def _breakdown(axis: str, payload: Mapping[str, Any],
               binding: Mapping[str, Any]) -> List[Dict[str, Any]]:
    if "members" in binding:
        return [{axis: key, "value": _model.read(payload, spec["value"])}
                for key, spec in binding["members"].items()]
    also = tuple(binding.get("also") or ())
    flag = binding.get("provisional_unless")
    carried = (*also, *((flag,) if flag else ()))
    if "map" in binding:
        note = binding.get("note")
        return [{axis: str(key), "value": (row or {}).get(binding["value"]),
                 **{name: (row or {}).get(name) for name in carried},
                 **({"note": (row or {}).get(note)} if note else {})}
                for key, row in (_model.read(payload, binding["map"]) or {}).items()]
    return [{axis: str(row.get(binding["key"])), "value": row.get(binding["value"]),
             **{name: row.get(name) for name in also}}
            for row in (_model.read(payload, binding["rows"]) or [])]


def _series(m: _model.Measure, payload: Mapping[str, Any], *,
            member: Optional[Tuple[str, str]], period: Mapping[str, Any],
            unavailable: str
            ) -> Tuple[str, Any, List[Dict[str, Any]], Any, Dict[str, Any]]:
    """The owner's curve: its rows, SELECTED to the stated horizon — never
    re-projected, extended or interpolated. A horizon beyond the owner's
    published one is refused, because nothing here can see past it."""
    series = m.series
    rows = _model.read(payload, series["rows"]) or []
    published = _model.read(payload, series["horizon"])
    ahead = period.get("periods_ahead")
    if ahead is not None:
        if published is None or ahead > published:
            raise Refusal(PERIOD_NOT_SUPPORTED,
                          f"the owner projects {published} month(s) ahead; "
                          f"{ahead} were asked for, and a curve is not "
                          f"extended past its owner's horizon")
        rows = [r for r in rows if 1 <= (r.get(series["step"]) or 0) <= ahead]
    # The curve's bands are the one breakdown the model declares as COLUMNS of
    # the owner's rows (the forecast's downside / base / upside).
    banded = next((b for b in m.by.values() if "columns" in b), {})
    columns = tuple(banded.get("columns") or ())
    if member:
        columns = (member[1],)
    cells = [{"period": r.get(series["period"]), "offset": r.get(series["step"]),
              **{c: r.get(c) for c in columns}} for r in rows]
    if not cells:
        raise Refusal(unavailable, f"the owner published no {m.name} rows")
    extra = {"series_columns": list(columns),
             "horizon": {"published_months": published, "periods_ahead": ahead}}
    return "series", None, cells, series["rows"], extra


# --------------------------------------------------------------------------- #
# the view's inputs and the receipt every declared figure carries
# --------------------------------------------------------------------------- #

def inputs_available(model: _model.SemanticModel, m: _model.Measure,
                     payload: Mapping[str, Any]) -> None:
    """Every input the figure uses is one the owner says it has; otherwise the
    figure cannot be stated, and is refused rather than stated from less."""
    view = model.views[m.view]
    for name in m.inputs:
        path = view.available_when.get(name)
        if path and not _model.read(payload, path):
            raise Refusal(FIGURE_UNAVAILABLE,
                          f"{view.label} has no governed {name} input, so "
                          f"{m.label.lower()} cannot be stated")


def inputs_used(model: _model.SemanticModel, m: _model.Measure,
                payload: Mapping[str, Any]) -> Dict[str, Dict[str, Any]]:
    """The dated inputs the figure used, each with the date its owner published
    (D4: an answer says what it is as at)."""
    view = model.views[m.view]
    return {name: {"as_of": _model.read(payload, view.inputs[name]["as_of"]),
                   "label": view.inputs[name].get("label") or name,
                   "owner": view.owner}
            for name in m.inputs if name in view.inputs}


def receipt(model: _model.SemanticModel, m: _model.Measure, plan: Any, *,
            payload: Mapping[str, Any], shape: str, paths: Any,
            axis: Optional[str], member: Optional[Tuple[str, str]],
            extra: Mapping[str, Any]) -> Dict[str, Any]:
    """What ran, as the coverage owner and the answer read it: the capability,
    the measure and its label and unit, the member or axis it was read for, the
    owner and the path of the figure in its output, and the inputs it used."""
    body = _plan.as_mapping(plan)
    view = model.views[m.view]
    context = {key: _model.read(payload, path) for key, path in m.context.items()}
    return {
        "capability": model.capability,
        "population_base": m.population,
        "operation": str(body.get("operation") or ""),
        "measure_concept": m.name,
        "measure_kind": "semantic_model",
        "measure_label": m.label,
        "unit": m.unit,
        "result_shape": shape,
        "execution_owner": view.owner,
        "semantic_model": f"config/mi/semantic_model/{model.capability}.yaml",
        "read_path": paths,
        "definition_decision": m.decision or None,
        "applied_predicates": ([{"field": member[0], "op": "eq",
                                 "values": [member[1]]}] if member else []),
        "group_field_keys": [axis] if axis else [],
        "member": ({"dimension": member[0], "value": member[1]}
                   if member else None),
        "inputs": inputs_used(model, m, payload),
        "context": context,
        # The explain sentence states the owner's own figure's companions; a
        # figure over another window than the owner's has none of them.
        "explain": (m.explain if (shape == "scalar" and not member
                                  and "window" not in extra) else ""),
        "caveat": m.caveat,
        # The axis as THIS figure names it, where the file says: the same
        # `origin_stage` is the stage a rate is measured from and the stage a
        # live case is at now.
        "axis_label": _axis_label(m, axis or (member[0] if member else None)),
        **context,
        **dict(extra),
    }


# --------------------------------------------------------------------------- #
# the change between two governed figures
# --------------------------------------------------------------------------- #

def period_change(earlier: Any, later: Any) -> Dict[str, Optional[float]]:
    """The change from `earlier` to `later`, two governed figures of ONE
    measure that their owner published for two stated dates.

    THE ONE PLACE. A period-on-period change is a derived figure every
    capability can be asked for; computing it in each runtime would give each
    its own rounding and its own rule for a zero base. So it is computed here,
    once: the absolute change, and the relative change where the earlier figure
    is not zero (a change from nothing has no percentage, and none is
    invented). Both figures and both dates travel with it on the receipt.
    """
    a = None if earlier is None else float(earlier)
    b = None if later is None else float(later)
    if a is None or b is None:
        return {"from": a, "to": b, "change": None, "change_pct": None}
    change = b - a
    pct = (change / abs(a) * 100.0) if a else None
    return {"from": a, "to": b, "change": round(change, 2),
            "change_pct": None if pct is None else round(pct, 2)}
