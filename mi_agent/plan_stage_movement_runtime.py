#!/usr/bin/env python3
"""The Pipeline Stage Movement runtime: a GovernedQueryPlan, executed by its owner.

WHY THIS EXISTS. Every stage-movement question in the certification bank — 27
of them, "how many cases moved from KFI to Application", "where did cases
leaving Offer go", "reconcile the Application stage" — compiles to a clean plan

    capability=pipeline_stage_movement  operation=transition
    population.base=pipeline  measures=[cases_moved]
    filters=[origin_stage = KFI, destination_stage = APPLICATION]

and was refused `POPULATION_NOT_EXECUTABLE` by the funded gate, while the legacy
route answered all but one of them correctly from an owner that already exists.
The owner was never missing. The arrow to it was.

WHAT THIS IS. A TRANSLATOR, like the pipeline and forecast runtimes. It turns
the plan's structured slots into the owner's own reading — the `StageMovement`
the legacy route builds by reading the sentence — asks the owner for the
governed payload, and lets the owner's own `compose` select the figures by key:

    figures   movement_detail.resolve_stage_transition_detail — the latest
                governed pair of weekly extracts, the same resolver the React
                movement-detail endpoint and the deck consume
    wording   stage_movement_query.compose — pure: a reading and a payload in,
                an answer out, every figure a key lookup

It computes nothing: no snapshot is loaded or joined here, no case matched, no
stage compared. And it never reads the question — the legacy module's sentence
reader is exactly what this replaces with the plan.

DELIBERATELY NARROW, and refused rather than approximated:

    a movement over any period but the latest governed pair   PERIOD_NOT_SUPPORTED
    "arrivals" that also name an origin stage                 FILTERS_NOT_SUPPORTED
      (a new arrival has no origin; the plan asks for two different things)
    a transition without both stages, a stayer or departure
      without its stage                                       FILTERS_NOT_SUPPORTED
    a scoped population, geography, a comparison              refused by name
"""
from __future__ import annotations

from typing import Any, Dict, FrozenSet, List, Mapping, Optional, Sequence, Tuple

from mi_agent import plan_reading as _plan_reading
from mi_agent import semantic_engine as _engine
from mi_agent import semantic_model as _semantic_model

CAPABILITY = "pipeline_stage_movement"

#: THE CAPABILITY'S MEASURED RATES (D2a, P0 design §20): cohort conversion,
#: stage pull-through and the historical completion rate, declared once in its
#: semantic model and served by the one engine every declared figure uses.
#: The movement between the latest pair of extracts stays with the owner's own
#: wording below.
MODEL = _semantic_model.load(CAPABILITY)
KIND_MOVEMENT = "movement"
KIND_CATALOGUE = "semantic_model"

#: WHICH POPULATION THIS RUNTIME EXECUTES: the weekly pipeline extracts, the
#: same population the pipeline runtime declares, under its own declaration.
EXECUTABLE_POPULATIONS: FrozenSet[str] = frozenset({"pipeline"})
EXECUTION_POPULATION = "pipeline"

# Ineligibility reasons. Stable strings: the ledger groups by them.
NOT_A_PLAN = "NOT_A_PLAN"
CAPABILITY_NOT_STAGE_MOVEMENT = "CAPABILITY_NOT_STAGE_MOVEMENT"
POPULATION_NOT_PIPELINE = "POPULATION_NOT_PIPELINE"
NOT_SINGLE_OUTPUT = "NOT_SINGLE_OUTPUT"
SCOPE_NOT_SUPPORTED = "SCOPE_NOT_SUPPORTED"
GEOGRAPHY_NOT_SUPPORTED = "GEOGRAPHY_NOT_SUPPORTED"
COMPARISON_NOT_SUPPORTED = "COMPARISON_NOT_SUPPORTED"
OPERATION_NOT_SUPPORTED = "OPERATION_NOT_SUPPORTED"
MEASURE_NOT_SUPPORTED = "MEASURE_NOT_SUPPORTED"
FILTERS_NOT_SUPPORTED = "FILTERS_NOT_SUPPORTED"
DIMENSION_NOT_SUPPORTED = "DIMENSION_NOT_SUPPORTED"
PERIOD_NOT_SUPPORTED = "PERIOD_NOT_SUPPORTED"
SOURCE_UNAVAILABLE = "SOURCE_UNAVAILABLE"
MOVEMENT_UNAVAILABLE = "MOVEMENT_UNAVAILABLE"
EXECUTION_FAILED = "EXECUTION_FAILED"

#: The stage axes a plan filters by. `pipeline_stage` is how a reconciliation
#: names its one stage; the other two are the capability's own dimensions.
ORIGIN = "origin_stage"
DESTINATION = "destination_stage"
STAGE = "pipeline_stage"

#: operation -> (the owner's subtype, the measures it answers -> the owner's
#: measure, the filters it requires, the filters it allows, the axes it allows).
#: The whole of the translation, as data.
_COUNT, _AMOUNT, _AMOUNT_CHANGE = "count", "amount", "amount_change"
OPERATIONS: Mapping[str, Mapping[str, Any]] = {
    "transition": {"subtype": "transition",
                   "measures": {"cases_moved": _COUNT, "amount_moved": _AMOUNT},
                   "require": (ORIGIN, DESTINATION), "allow": (),
                   "axes": ()},
    "arrivals": {"subtype": "new_arrival",
                 "measures": {"cases_arrived": _COUNT},
                 "require": (DESTINATION,), "allow": (), "axes": ()},
    "stayers": {"subtype": "stayer",
                "measures": {"cases_stayed": _COUNT,
                             "stayer_amount_change": _AMOUNT_CHANGE},
                "require": (), "allow": (), "axes": (),
                "one_of": (ORIGIN, STAGE)},
    "departures": {"subtype": "departure",
                   "measures": {"cases_departed": _COUNT},
                   "require": (), "allow": (), "axes": (DESTINATION,),
                   "one_of": (ORIGIN, STAGE)},
    "reconciliation": {"subtype": "reconciliation",
                       "measures": {"stage_opening": _COUNT,
                                    "stage_closing": _COUNT,
                                    "cases_arrived": _COUNT,
                                    "cases_departed": _COUNT},
                       "require": (), "allow": (), "axes": (),
                       "one_of": (STAGE, ORIGIN), "measures_optional": True},
}

#: The only period this owner answers for without a date: its own latest
#: governed pair of weekly extracts, which every answer states.
PERIOD_FORMS: FrozenSet[str] = frozenset({"current"})

#: WHAT MOVED IN THE WHOLE PIPELINE (vocabulary 2.17.0): the `material_summary`
#: change form over this owner — every case between the latest pair of extracts
#: classified once as arrived, moved stage, left or stayed, from the payload's
#: own event totals and per-stage reconciliation. The pair is the latest
#: extract and the one before it, which is what "the previous" means for the
#: pipeline (owner decision D15), so a period stated as "since the previous
#: extract" names the owner's own pair.
SUMMARY_FORM = "material_summary"
#: The figures the summary states, any of which a plan may name.
SUMMARY_MEASURES: FrozenSet[str] = frozenset({
    "cases_moved", "amount_moved", "cases_arrived", "cases_departed",
    "cases_stayed", "stayer_amount_change"})

OWNER = "movement_detail.resolve_stage_transition_detail"
WORDING_OWNER = "stage_movement_query.compose"


# --------------------------------------------------------------------------- #
# reading the plan
# --------------------------------------------------------------------------- #

def _as_mapping(plan: Any) -> Mapping[str, Any]:
    if isinstance(plan, Mapping):
        return plan
    to_dict = getattr(plan, "to_dict", None)
    return to_dict() if callable(to_dict) else {}


def claims(plan: Any) -> bool:
    """Is this plan THIS runtime's? The capability, and nothing else."""
    return str(_as_mapping(plan).get("capability") or "") == CAPABILITY


def _measures(body: Mapping[str, Any]) -> List[str]:
    output = _single_output(body) or {}
    return [str(m.get("concept") or "") for m in (output.get("measures") or ())]


change_form_of = _plan_reading.change_form_of


def is_summary(plan: Any) -> bool:
    return change_form_of(plan) == SUMMARY_FORM


def serves_figures_together(plan: Any) -> bool:
    """Does this runtime answer the plan's figures as ONE owner answer? The
    summary and a stage's reconciliation state several figures from one
    payload; every other shape is one figure, and a plan naming several is
    composed from one-figure plans (`plan_composition`)."""
    body = _as_mapping(plan)
    if is_summary(body):
        return True
    spec = OPERATIONS.get(str(body.get("operation") or "")) or {}
    return bool(spec.get("measures_optional"))


def kind_of(plan: Any) -> str:
    """A declared rate (the semantic model's) or a movement (the owner's)."""
    measures = _measures(_as_mapping(plan))
    return (KIND_CATALOGUE if len(measures) == 1 and MODEL.measure(measures[0])
            else KIND_MOVEMENT)


def _single_output(plan: Mapping[str, Any]) -> Optional[Mapping[str, Any]]:
    outputs = plan.get("outputs") or ()
    return outputs[0] if len(outputs) == 1 else None


def _axis(binding: Mapping[str, Any]) -> str:
    """A binding's governed axis: its field, or — for a capability-owned
    dimension that has no canonical field — its concept."""
    return str(binding.get("canonical_field") or binding.get("concept") or "")


def stage_filters(plan: Any) -> Tuple[Dict[str, str], List[str]]:
    """`({axis: stage}, problems)` from the plan's own filters. Equality only."""
    body = _as_mapping(plan)
    output = _single_output(body) or {}
    found: Dict[str, str] = {}
    problems: List[str] = []
    for binding in tuple(body.get("filters") or ()) + tuple(output.get("filters") or ()):
        axis = _axis(binding)
        if axis not in (ORIGIN, DESTINATION, STAGE):
            problems.append(f"{axis!r} is not a stage axis")
            continue
        if str(binding.get("comparator") or "eq") != "eq":
            problems.append(f"{axis} is compared {binding.get('comparator')!r}, "
                            f"not named")
            continue
        value = binding.get("value")
        if not isinstance(value, str) or not value.strip():
            problems.append(f"{axis} names no single stage ({value!r})")
            continue
        if axis in found:
            problems.append(f"{axis} is named twice")
            continue
        found[axis] = value.strip().upper()
    return found, problems


# --------------------------------------------------------------------------- #
# the perimeter
# --------------------------------------------------------------------------- #

def check_eligibility(plan: Any) -> Tuple[bool, str, str]:
    """`(eligible, reason, detail)`. Reads the plan only, and refuses by default."""
    body = _as_mapping(plan)
    if not body:
        return False, NOT_A_PLAN, "no plan was supplied"
    if not claims(body):
        return (False, CAPABILITY_NOT_STAGE_MOVEMENT,
                f"capability={body.get('capability')!r} is not {CAPABILITY!r}")
    population = body.get("population") or {}
    base = str(population.get("base") or "")
    if base not in EXECUTABLE_POPULATIONS:
        return (False, POPULATION_NOT_PIPELINE,
                f"population.base={base!r} is not executed by the stage movement "
                f"runtime (it executes {sorted(EXECUTABLE_POPULATIONS)})")
    if (str(population.get("lens") or "all") != "all"
            or str(population.get("seasoning") or "any") != "any"
            or population.get("source_reference")
            or population.get("scope_predicates")):
        return (False, SCOPE_NOT_SUPPORTED,
                "the stage movement owner reports the whole pipeline; a scoped "
                "movement would be answered over a population it did not name")
    output = _single_output(body)
    if output is None:
        return False, NOT_SINGLE_OUTPUT, "exactly one output is served"
    if output.get("geography") or (body.get("geography") or {}).get("requested"):
        return (False, GEOGRAPHY_NOT_SUPPORTED,
                "the stage movement owner has no geography breakdown")
    if str(body.get("comparison_kind") or "none") != "none":
        return (False, COMPARISON_NOT_SUPPORTED,
                "the stage movement runtime serves no population comparison")

    operation = str(body.get("operation") or "")
    if is_summary(body):
        return _check_summary(body, output)
    if kind_of(body) == KIND_CATALOGUE:
        return _engine.check(MODEL, body, _measures(body)[0], operation)
    spec = OPERATIONS.get(operation)
    if spec is None:
        return (False, OPERATION_NOT_SUPPORTED,
                f"operation={operation!r} has no stage movement owner here "
                f"(served: {sorted(OPERATIONS)})")
    form = str((body.get("period") or {}).get("form") or "")
    if form not in PERIOD_FORMS:
        return (False, PERIOD_NOT_SUPPORTED,
                f"period.form={form!r}: the owner answers for its latest "
                f"governed pair of weekly extracts, which the answer states")

    measures = [str(m.get("concept") or "") for m in (output.get("measures") or ())]
    unknown = [m for m in measures if m not in spec["measures"]]
    if unknown or (not measures and not spec.get("measures_optional")):
        return (False, MEASURE_NOT_SUPPORTED,
                f"{operation} answers {sorted(spec['measures'])}; this plan names "
                f"{measures}")
    if len(measures) > 1 and not spec.get("measures_optional"):
        return (False, MEASURE_NOT_SUPPORTED,
                f"{operation} answers one measure; this plan names {measures}")

    axes = [_axis(d) for d in (output.get("dimensions") or ())]
    if any(axis not in spec["axes"] for axis in axes):
        return (False, DIMENSION_NOT_SUPPORTED,
                f"{operation} groups by {list(spec['axes']) or 'nothing'}; this "
                f"plan names {axes}")

    stages, problems = stage_filters(body)
    if problems:
        return False, FILTERS_NOT_SUPPORTED, "; ".join(problems)
    missing = [axis for axis in spec["require"] if axis not in stages]
    one_of = spec.get("one_of") or ()
    named = [axis for axis in one_of if axis in stages]
    allowed = set(spec["require"]) | set(spec["allow"]) | set(named[:1])
    extra = sorted(set(stages).difference(allowed))
    if missing or (one_of and len(named) != 1) or extra:
        return (False, FILTERS_NOT_SUPPORTED,
                f"{operation} needs {list(spec['require']) or list(one_of)[:1]} "
                f"and no other stage; this plan names {stages}"
                + ("; a new arrival has no origin stage" if operation == "arrivals"
                   and ORIGIN in stages else ""))
    return True, "", ""


def _check_summary(body: Mapping[str, Any], output: Mapping[str, Any]
                   ) -> Tuple[bool, str, str]:
    """What moved in the whole pipeline: no stage named (one stage's movement
    is that stage's reconciliation), the owner's own figures only, broken down
    by stage at most, over the owner's latest pair of extracts."""
    measures = _measures(body)
    unknown = [m for m in measures if m not in SUMMARY_MEASURES]
    if unknown:
        return (False, MEASURE_NOT_SUPPORTED,
                f"what moved in the pipeline states {sorted(SUMMARY_MEASURES)}; "
                f"this plan names {unknown}")
    axes = [_axis(d) for d in (output.get("dimensions") or ())]
    if any(axis != STAGE for axis in axes):
        return (False, DIMENSION_NOT_SUPPORTED,
                f"what moved in the pipeline is broken down by stage only; "
                f"this plan names {axes}")
    stages, problems = stage_filters(body)
    if problems or stages:
        return (False, FILTERS_NOT_SUPPORTED,
                "what moved at one stage is that stage's reconciliation; the "
                "summary is of the whole pipeline"
                + (f" ({'; '.join(problems)})" if problems else ""))
    # The owner's pair is the latest two extracts: a plan naming it
    # implicitly (D15, `plan_reading.names_latest_pair`) or as the relative
    # pair one extract back.
    period = _plan_reading.pair_period(body)
    form = str(period.get("form") or "")
    back = period.get("periods_back")
    grain = period.get("grain")
    if form != "relative_pair" or back not in (None, 1) \
            or grain not in (None, "weekly"):
        return (False, PERIOD_NOT_SUPPORTED,
                f"the owner answers for the latest extract and the one before "
                f"it (D15); period.form={form!r}, periods_back={back!r}, "
                f"grain={grain!r} names another pair")
    return True, "", ""


def reading_for(plan: Any) -> Dict[str, Any]:
    """The owner's own reading, from the plan's structured slots. An ELIGIBLE
    plan only. Returned as the `StageMovement` fields, so the owner's type is
    constructed where it is used rather than imported into the perimeter."""
    body = _as_mapping(plan)
    if is_summary(body):
        return {"subtype": "summary", "measure": _COUNT, "source": None,
                "destination": None, "stage": None}
    output = _single_output(body) or {}
    spec = OPERATIONS[str(body.get("operation") or "")]
    measures = [str(m.get("concept") or "") for m in (output.get("measures") or ())]
    stages, _ = stage_filters(body)
    one_of = [stages[a] for a in (spec.get("one_of") or ()) if a in stages]
    return {
        "subtype": spec["subtype"],
        "measure": spec["measures"][measures[0]] if measures else _COUNT,
        "source": stages.get(ORIGIN) if spec["subtype"] == "transition" else None,
        "destination": (stages.get(DESTINATION)
                        if spec["subtype"] in ("transition", "new_arrival")
                        else None),
        "stage": one_of[0] if one_of else None,
    }


# --------------------------------------------------------------------------- #
# execution — the owner, asked with what the plan says
# --------------------------------------------------------------------------- #

class StageMovementOutcome:
    """What ran, what it produced, and the evidence that proves both."""

    __slots__ = ("ok", "reason", "detail", "answer", "rows", "columns", "receipt",
                 "value", "cells")

    def __init__(self, *, ok: bool, reason: str = "", detail: str = "",
                 answer: Optional[str] = None,
                 rows: Optional[List[Dict[str, Any]]] = None,
                 columns: Optional[List[Dict[str, Any]]] = None,
                 receipt: Optional[Dict[str, Any]] = None,
                 value: Any = None,
                 cells: Optional[List[Dict[str, Any]]] = None) -> None:
        self.ok = ok
        self.reason = reason
        self.detail = detail
        self.answer = answer
        self.rows = rows or []
        self.columns = columns or []
        self.receipt = receipt or {}
        self.value = value
        self.cells = cells


def _execute_catalogue(body: Mapping[str, Any],
                       history_model: Optional[Mapping[str, Any]]
                       ) -> StageMovementOutcome:
    """A declared rate, read by the engine from the owner's published model —
    the case history the production request already built. Nothing computed."""
    m = MODEL.measure(_measures(body)[0])
    if not isinstance(history_model, Mapping) or not history_model:
        return StageMovementOutcome(
            ok=False, reason=SOURCE_UNAVAILABLE,
            detail=f"no pipeline case history was supplied, so "
                   f"{m.label.lower()} cannot be stated")
    try:
        _engine.inputs_available(MODEL, m, history_model)
        axis, member, binding = _engine.binding_for(m, body)
        _engine.requires_met(binding, history_model, axis or member)
        shape, value, cells, paths, extra = _engine.serve_figure(
            m, history_model, axis=axis, member=member, binding=binding,
            period=body.get("period") or {})
    except _engine.Refusal as refusal:
        return StageMovementOutcome(ok=False, reason=refusal.reason,
                                    detail=refusal.detail[:300])
    receipt = _engine.receipt(MODEL, m, body, payload=history_model,
                              shape=shape, paths=paths, axis=axis,
                              member=member, extra=extra)
    return StageMovementOutcome(ok=True, receipt=receipt, value=value,
                                cells=cells, rows=cells or [])


def execute(plan: Any, *, root: Any, client_id: str,
            history_model: Optional[Mapping[str, Any]] = None
            ) -> StageMovementOutcome:
    """An ELIGIBLE plan, through `resolve_stage_transition_detail` and `compose`.

    `root`, `client_id` and `history_model` are the pipeline inputs the
    production request already resolved — the ones the legacy route hands this
    same owner. Nothing is discovered here.
    """
    if kind_of(plan) == KIND_CATALOGUE:
        return _execute_catalogue(_as_mapping(plan), history_model)
    from mi_agent_api import currency as currency_mod
    from mi_agent_api import movement_detail as detail_mod
    from mi_agent_api import stage_movement_query as owner_words

    body = _as_mapping(plan)
    if not root or not client_id:
        return StageMovementOutcome(
            ok=False, reason=SOURCE_UNAVAILABLE,
            detail="no governed weekly pipeline root or client was supplied")
    fields = reading_for(body)
    reading = owner_words.StageMovement(**fields)
    try:
        payload = detail_mod.resolve_stage_transition_detail(
            root, client_id, historical_model=history_model)
    except Exception as exc:                                         # noqa: BLE001
        return StageMovementOutcome(ok=False, reason=EXECUTION_FAILED,
                                    detail=f"{type(exc).__name__}: {exc}"[:300])

    def money(value: Any) -> str:
        return currency_mod.format_money(value, suffixes=("bn", "m", "k"))

    answer, rows, refusal = owner_words.compose(reading, payload, money=money)
    if refusal is not None or answer is None:
        return StageMovementOutcome(
            ok=False, reason=MOVEMENT_UNAVAILABLE,
            detail=str(refusal or "the owner produced no answer")[:300])

    output = _single_output(body) or {}
    stages, _ = stage_filters(body)
    # THE STAGES THE PAYLOAD WAS SELECTED ON, as execution evidence: each is a
    # field of the reading `compose` answered, keyed by the plan's own axis.
    used = {"source": ORIGIN, "destination": DESTINATION, "stage": None}
    applied = []
    for slot, value in fields.items():
        if slot not in used or not value:
            continue
        axis = used[slot] or next(a for a in (STAGE, ORIGIN) if stages.get(a) == value)
        applied.append({"field": axis, "canonical_field": axis, "op": "eq",
                        "values": [value]})
    measures = [str(m.get("concept") or "") for m in (output.get("measures") or ())]
    axes = [_axis(d) for d in (output.get("dimensions") or ())]
    receipt: Dict[str, Any] = {
        "capability": CAPABILITY,
        "population_base": EXECUTION_POPULATION,
        "operation": str(body.get("operation") or ""),
        "measure_concept": measures[0] if measures else None,
        "measure_concepts": measures,
        "reading": fields,
        "applied_predicates": applied,
        # Departures are always reported by destination, so an axis asked for
        # is an axis the owner's rows carry.
        "group_field_keys": axes,
        "execution_owner": OWNER,
        "wording_owner": WORDING_OWNER,
        "dataset": {"identity": "governed_weekly_pipeline_extracts",
                    "as_of_date": payload.get("as_of_date"),
                    "comparison_date": payload.get("comparison_date"),
                    "identifier": payload.get("identifier")},
        "methodology_version": (payload.get("methodology") or {}).get("version"),
        "result_shape": "rows",
    }
    rows = [dict(r) for r in rows]
    return StageMovementOutcome(ok=True, answer=answer, rows=rows,
                                columns=owner_words._columns(rows),
                                receipt=receipt)
