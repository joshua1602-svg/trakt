#!/usr/bin/env python3
"""The Forecast specialist runtime: a GovernedQueryPlan, executed by Forecast.

WHY THIS EXISTS. The production bank asked 48 forecast questions and the
governed path served none of them. The compiler reads them correctly —

    capability=forecast  operation=forecast_milestone  population.base=forecast
    measures=[forecast_milestone_date]  target=forecast_funded_balance >= £100m

— and the canary then refused every one, `POPULATION_NOT_EXECUTABLE`, because
no governed runtime declared the forecast population. The forecast owner was
never the gap; the arrow to it was. This is that arrow (P0 design §6).

WHAT THIS IS, AND WHAT IT IS NOT. A TRANSLATOR, exactly as the pipeline runtime
is. It turns an already-compiled plan into the arguments the existing forecast
owner already takes, calls it, and writes down what ran. Two paths:

    every figure the     READ from its owner's published output at the path
    semantic model       `config/mi/semantic_model/forecast.yaml` declares (P0
    declares            design §16) — one figure, one governed member, the
                         owner's own breakdown, or its own curve selected to a
                         stated horizon. The owners are the ones the Forecast
                         tab renders:
                           forecast_view.compose_forecast_view   the tab's bridge,
                             breakdowns and weighting disclosure (D6: the
                             forecast funded balance is its point figure)
                           forecast_extrapolation.build_extrapolation   the
                             scale-up forecast: run-rate, scenario bands, curve
                             and milestone ladder
    milestone date       the same scale-up owner with the reader's threshold
    for a stated          added to its ladder, then
    threshold            forecast_extrapolation.milestone_answer — the ONE rule
                         deciding which of four answers a milestone question
                         has, lifted out of the legacy router so this module
                         reads the rule instead of copying it

It computes no figure. There is no projection, no division by a run-rate and no
milestone date worked out here — every number in the receipt is read out of
the owner's own output, and a test asserts it. The one subtraction in this
module is between two DATES, and it is the D1 perimeter check below, not a
forecast.

A DERIVED POPULATION DECLARES ITS INPUTS, AND ITS SKEW (P0 D1). `forecast` is
not a dataset anyone loads: it is the funded book projected forward from a
funded snapshot and the pipeline's observed completion flow, which are cut at
different dates. So this runtime says which inputs fed the figure it serves,
states each one's vintage, and measures how far apart they are against the
ceiling the funded book already uses —
`period_change.selection.SelectionPolicy.max_snapshot_gap_days`, the same
method with the same whole-book context, so there is no second 45:

    inputs within the ceiling   ANSWER, both vintages on the receipt
    inputs beyond the ceiling   REFUSE  POPULATION_VINTAGE_SKEW
    a used input with no date   REFUSE  POPULATION_INPUT_UNRESOLVED

Only a figure that actually COMPOSES the two datasets has a skew. The owner's
run-rate falls back to funded balance growth when the pipeline has no observed
completion flow, and then the pipeline is not an input at all; "the book has
already reached £25m" reads the funded balance and nothing else. The receipt
says which inputs each answer used rather than claiming both every time.

WHAT IS REFUSED, AND WHY, rather than approximated:

    a forecast funded        D6 made the tab's point figure the definition. The
    balance SERIES           only forecast-balance series pairs each funded
                             month with that month's last extract — a different
                             definition — so a series of it is refused; the
                             month-by-month curve is projected_funded_balance.
    a "what if"              D2b — its own operation with a typed `assumption`
                             slot (§6.2). The scenario BANDS the owner publishes
                             are served; an assumed change is not.
    a scoped forecast        the run-rate is book-wide; applying it to one book
                             or one role is the silent widening §8.6 forbids.
    a breakdown, filter or   anything the semantic model does not declare the
    geography the owner      owner publishing: refused, never recomputed here.
    does not publish
    a horizon beyond the     the curve is selected, never extended.
    owner's

IT NEVER READS THE QUESTION. No parse, no recogniser, no router, and nothing
from `chat_routing` — whose milestone closure is the reason the milestone rule
had to move to the owner before this module could exist.
"""
from __future__ import annotations

from datetime import date
from typing import Any, Dict, FrozenSet, List, Mapping, Optional, Tuple

from mi_agent import plan_reading as _plan
from mi_agent import semantic_engine as _engine
from mi_agent import semantic_model as _semantic_model
from mi_agent.interpretation_v2.vocabulary import NAMED_THRESHOLDS

CAPABILITY = "forecast"

# WHICH POPULATIONS THIS RUNTIME EXECUTES — `EXECUTABLE_POPULATIONS` and
# `EXECUTION_POPULATION`, below `MODEL`: they are read from the semantic model,
# where each figure declares the population it is measured over.

#: THE INPUTS A FORECAST IS DERIVED FROM (P0 design §4.1, §6.1). Declared so a
#: reader — and a test — can see the lineage without reading the owner.
#:
#: `pipeline` is required WHEN it is the completion signal. The owner prefers the
#: pipeline's observed COMPLETED flow and falls back to funded balance growth
#: when there is none; in the fallback the pipeline fed nothing and there is no
#: second vintage to state.
#:
#: `conversion` is D2a: stage movement OWNS the calculation and forecast only
#: consumes it. No figure this runtime serves cites a conversion rate yet, so it
#: is declared, attributed, and never computed here.
POPULATION_INPUTS: Mapping[str, Mapping[str, Any]] = {
    "funded": {"required": True,
               "as_of": "governed monthly funded snapshot",
               "owner": "evolution.funded_evolution"},
    "pipeline": {"required": False,
                 "required_when": "the completion signal is the observed "
                                  "completion flow, or the figure is the "
                                  "forecast funded balance",
                 "as_of": "governed weekly pipeline extract",
                 "owner": "evolution.pipeline_funnel_evolution"},
    "conversion": {"required": False,
                   "owner": "pipeline_stage_movement",
                   "consumed_not_computed": True},
}

#: THE CONTEXT THE D1 CEILING IS READ FOR. A forecast this runtime serves is
#: whole-book (a scoped one is refused), and the whole-book context id the funded
#: route hands the same policy method is the lens layer's `total`. Stated rather
#: than imported: that module recognises natural language, and a test pins the
#: two equal instead.
WHOLE_BOOK_CONTEXT_ID = "total"

#: The method that owns the ceiling, named on every receipt that used it.
CEILING_OWNER = "period_change.selection.SelectionPolicy.max_snapshot_gap_days"

# Ineligibility and refusal reasons. Stable strings: the ledger groups by them.
NOT_A_PLAN = "NOT_A_PLAN"
CAPABILITY_NOT_FORECAST = "CAPABILITY_NOT_FORECAST"
POPULATION_NOT_FORECAST = "POPULATION_NOT_FORECAST"
#: The plan's base is one this runtime executes, but not the one its figure is
#: measured over (the semantic model's `population`).
POPULATION_NOT_MEASURED = "POPULATION_NOT_MEASURED"
NOT_SINGLE_OUTPUT = "NOT_SINGLE_OUTPUT"
SCOPE_NOT_SUPPORTED = "SCOPE_NOT_SUPPORTED"
FILTERS_NOT_SUPPORTED = _engine.FILTERS_NOT_SUPPORTED
DIMENSION_NOT_SUPPORTED = _engine.DIMENSION_NOT_SUPPORTED
GEOGRAPHY_NOT_SUPPORTED = _engine.GEOGRAPHY_NOT_SUPPORTED
COMPARISON_NOT_SUPPORTED = "COMPARISON_NOT_SUPPORTED"
MEASURE_NOT_SUPPORTED = _engine.MEASURE_NOT_SUPPORTED
OPERATION_NOT_SUPPORTED = _engine.OPERATION_NOT_SUPPORTED
PERIOD_NOT_SUPPORTED = _engine.PERIOD_NOT_SUPPORTED
TARGET_NOT_SUPPORTED = _engine.TARGET_NOT_SUPPORTED
#: The owner computes this shape, but the production bank shows the model
#: using it for questions that want different figures. Held, not served.
AMBIGUOUS_READING = _engine.AMBIGUOUS_READING
INPUTS_UNAVAILABLE = "INPUTS_UNAVAILABLE"
FORECAST_UNAVAILABLE = "FORECAST_UNAVAILABLE"
EXECUTION_FAILED = "EXECUTION_FAILED"
#: The owner's output carries no figure for what the plan asked (a member it
#: did not publish, a breakdown on another basis). Refused, never substituted.
FIELD_UNAVAILABLE = _engine.FIELD_UNAVAILABLE
#: P0 §9 — the two reasons a derived population adds.
POPULATION_INPUT_UNRESOLVED = "POPULATION_INPUT_UNRESOLVED"
POPULATION_VINTAGE_SKEW = "POPULATION_VINTAGE_SKEW"
#: A milestone named "scale" for a portfolio whose stage is not recorded (D9):
#: there is no threshold to reach, and none is guessed.
SCALE_NOT_CONFIGURED = "SCALE_NOT_CONFIGURED"

#: THE CAPABILITY'S SEMANTIC MODEL (P0 design §16). Every forecast figure but
#: one is declared there — what it is, which owner publishes it, and the path of
#: the figure in that owner's output — and served by ONE generic reader below.
#: A new forecast figure is an entry in that file and a test pinning it to the
#: tab, not a branch in this module.
MODEL = _semantic_model.load(CAPABILITY)
SEMANTIC_MODEL_FILE = "config/mi/semantic_model/forecast.yaml"

#: The capability's own population: what a forecast figure is measured over
#: unless its entry says otherwise, and the view the analytical context composes.
EXECUTION_POPULATION = MODEL.population

#: WHICH POPULATIONS THIS RUNTIME EXECUTES: `forecast`, and `pipeline` for the
#: figures that describe the pipeline input alone (its exclusions from
#: weighting). Read from the model, so a figure and the population it may be
#: asked about are declared in one place. Never `funded`: a runtime dispatched
#: above the funded gate that also claimed funded would take funded plans past
#: the gate that guards them. Normalisation rule 6 is what lets a forecast plan
#: spelled with a `funded` base reach this set.
EXECUTABLE_POPULATIONS: FrozenSet[str] = frozenset(MODEL.populations())
if "funded" in EXECUTABLE_POPULATIONS:
    raise _semantic_model.SemanticModelError(
        f"{SEMANTIC_MODEL_FILE} declares a figure measured over the funded book; "
        f"the forecast runtime is dispatched above the funded gate and must not "
        f"execute it")

#: THE ONE FIGURE THAT IS NOT A LOOKUP: the milestone for a threshold the
#: question names. The owner answers it only when asked about that amount (the
#: reader's target is added to its ladder), and its rule
#: (`milestone_answer`) decides which of four answers it has. The LADDER — the
#: dates the owner projects for its governed thresholds — is a lookup, and is
#: in the semantic model.
MILESTONE = ("forecast_milestone", "forecast_milestone_date")
KIND_MILESTONE = "milestone"
KIND_CATALOGUE = "semantic_model"
MILESTONE_PERIOD_FORMS: FrozenSet[str] = frozenset({"forward_looking"})

#: The owner decision that defines the forecast funded balance, named on every
#: refusal and receipt that depends on it.
BALANCE_DECISION = "D6"

#: SHAPES THE OWNER COMPUTES BUT A PLAN CANNOT YET BE TRUSTED TO MEAN.
#:
#: Measured, not supposed: the plans the canary recorded for the owner's
#: production bank (2026-09-28, read back by `qb_plan_readback`) put several
#: different questions into each of these shapes, and the runtime can only
#: answer the shape. Serving it would answer some of those questions with a
#: different figure than they asked for, which the design's hard gate (WRONG
#: answers must stay at zero) forbids. Vocabulary 2.7.0 gives each of those
#: questions a concept of its own (§16.3); the hold is released only when a
#: live run shows the readings have moved there. Legacy serves them meanwhile.
HELD_READINGS: Mapping[Tuple[str, str], str] = {
    ("forecast_projection", "forecast_funded_balance"): (
        "this shape arrived for the expected funded balance [87], but also for "
        "the extrapolation curve [114], the base scenario [117] and the funded "
        "share of the forecast [94] — four different figures"),
    # ("point_in_time", "forecast_completion_rate") — RELEASED on the 19:59
    # full bank (2026-09-30, vocabulary 2.18.0; P0 design §31), by the hold's
    # own condition: the readings moved to their own concepts. The KFI-to-
    # completion rate [121] reads as the stage completion rate, the annualised
    # run-rate [113] as its own measure, the forecast's method [98] asks back,
    # and the 8- and 12-week run-rates [125, 126] keep their window, which
    # the measure (stated only for the current window) refuses. What remains in
    # the shape is the current run-rate [112], which it answers.
    # `qb_recorded_intents_20260930_1959.json` is the evidence.
}

#: A milestone's threshold: what the owner's rule answers, and nothing wider.
#: `milestone_answer` decides "already reached" as `current >= threshold`, so a
#: strictly-greater-than threshold is a different question at the boundary and
#: is refused rather than answered as though it were the same one.
TARGET_CONCEPT = "forecast_funded_balance"
TARGET_COMPARATORS: FrozenSet[str] = frozenset({"gte"})

#: WHICH OWNER PRODUCED THE FIGURES, named on the receipt.
OWNER_EXTRAPOLATION = "forecast_extrapolation.build_extrapolation"
OWNER_MILESTONE_RULE = "forecast_extrapolation.milestone_answer"
OWNER_VIEW = MODEL.views["forecast_view"].owner
OWNER_BRIDGE = "forecast_bridge.compute_forecast_bridge"


# --------------------------------------------------------------------------- #
# reading the plan
# --------------------------------------------------------------------------- #

def _as_mapping(plan: Any) -> Mapping[str, Any]:
    return _plan.as_mapping(plan)


def claims(plan: Any) -> bool:
    """Is this plan THIS runtime's? The capability, and nothing else."""
    return str(_as_mapping(plan).get("capability") or "") == CAPABILITY


def _single_output(plan: Mapping[str, Any]) -> Optional[Mapping[str, Any]]:
    return _plan.single_output(plan)


def _measure(plan: Mapping[str, Any]) -> str:
    output = _single_output(plan) or {}
    measures = output.get("measures") or ()
    return str(measures[0].get("concept") or "") if len(measures) == 1 else ""


def execution_population(plan: Any) -> str:
    """The population THIS plan's figure is measured over: its measure's entry
    in the semantic model. What the population gate proves against the plan's
    base and what the receipt states — never inferred from the answer."""
    measure = MODEL.measure(_measure(_as_mapping(plan)))
    return measure.population if measure is not None else EXECUTION_POPULATION


def kind_of(plan: Any) -> str:
    """Which path an ELIGIBLE plan takes. Empty for anything else."""
    body = _as_mapping(plan)
    if (str(body.get("operation") or ""), _measure(body)) == MILESTONE:
        return KIND_MILESTONE
    return KIND_CATALOGUE if MODEL.measure(_measure(body)) else ""


def target_of(plan: Any) -> Optional[Mapping[str, Any]]:
    target = _as_mapping(plan).get("target")
    return target if isinstance(target, Mapping) else None


def _member(plan: Any) -> Optional[Tuple[str, str]]:
    """The one governed member a plan filters on (the engine's reading)."""
    return _engine.member_of(plan)


# --------------------------------------------------------------------------- #
# the perimeter
# --------------------------------------------------------------------------- #

def check_eligibility(plan: Any) -> Tuple[bool, str, str]:
    """`(eligible, reason, detail)`. Reads the plan only, and refuses by default."""
    body = _as_mapping(plan)
    if not body:
        return False, NOT_A_PLAN, "no plan was supplied"
    if not claims(body):
        return (False, CAPABILITY_NOT_FORECAST,
                f"capability={body.get('capability')!r} is not {CAPABILITY!r}; "
                f"this runtime executes one capability and refuses the rest")

    population = body.get("population") or {}
    base = str(population.get("base") or "")
    if base not in EXECUTABLE_POPULATIONS:
        return (False, POPULATION_NOT_FORECAST,
                f"population.base={base!r} is not executed by the forecast "
                f"runtime (it executes {sorted(EXECUTABLE_POPULATIONS)})")
    scoped = [slot for slot, whole in (("lens", "all"), ("seasoning", "any"))
              if str(population.get(slot) or whole) != whole]
    if population.get("source_reference") or population.get("scope_predicates"):
        scoped.append("source")
    if scoped:
        return (False, SCOPE_NOT_SUPPORTED,
                f"the forecast owner's completion run-rate is book-wide, so a "
                f"forecast scoped by {scoped} would apply it to a population it "
                f"was not measured on")

    output = _single_output(body)
    if output is None:
        return (False, NOT_SINGLE_OUTPUT,
                "the forecast runtime serves exactly one output")
    if str(body.get("comparison_kind") or "none") != "none":
        return (False, COMPARISON_NOT_SUPPORTED,
                "the forecast runtime serves no population comparison")
    measures = [str(m.get("concept") or "") for m in (output.get("measures") or ())]
    if len(measures) != 1:
        return (False, MEASURE_NOT_SUPPORTED,
                f"exactly one measure is served; this plan names {measures}")
    measured_over = execution_population(body)
    if base != measured_over:
        return (False, POPULATION_NOT_MEASURED,
                f"{measures[0]!r} is measured over the {measured_over!r} "
                f"population, and this plan asks about {base!r}: a forecast of "
                f"the pipeline alone is a different figure from the funded book "
                f"projected forward, and neither is answered for the other")
    operation = str(body.get("operation") or "")
    if (operation, measures[0]) == MILESTONE:
        return _check_milestone(body, output)
    return _check_catalogue(body, measures[0], operation)


def _check_milestone(body: Mapping[str, Any],
                     output: Mapping[str, Any]) -> Tuple[bool, str, str]:
    """The milestone for ONE stated threshold: the owner's rule answers it."""
    if _plan.plan_filters(body):
        return (False, FILTERS_NOT_SUPPORTED,
                "no forecast owner narrows a milestone by a predicate")
    if output.get("dimensions"):
        return (False, DIMENSION_NOT_SUPPORTED,
                "a milestone for a stated threshold is one date; for the "
                "governed ladder, break forecast_milestone_date down by "
                "funding_threshold with no target")
    if _plan.asks_geography(body):
        return (False, GEOGRAPHY_NOT_SUPPORTED,
                "no forecast owner projects a milestone by geography")
    period = body.get("period") or {}
    if period.get("grain"):
        return (False, PERIOD_NOT_SUPPORTED,
                f"a {period.get('grain')} grain asks for a figure per period; "
                f"a milestone is a single date")
    form = str(period.get("form") or "")
    if form not in MILESTONE_PERIOD_FORMS:
        return (False, PERIOD_NOT_SUPPORTED,
                f"period.form={form!r} is not how a milestone is stated "
                f"({sorted(MILESTONE_PERIOD_FORMS)})")
    target = target_of(body)
    value = (target or {}).get("value")
    # A positive amount, or a governed NAMED threshold ("scale", D9) whose
    # figure the portfolio supplies at execution.
    named = isinstance(value, str) and value in NAMED_THRESHOLDS
    if (not target or str(target.get("concept") or "") != TARGET_CONCEPT
            or str(target.get("comparator") or "") not in TARGET_COMPARATORS
            or (not named and (isinstance(value, bool)
                               or not isinstance(value, (int, float))
                               or value <= 0))):
        return (False, TARGET_NOT_SUPPORTED,
                f"a milestone is served for a positive {TARGET_CONCEPT} "
                f"threshold compared {sorted(TARGET_COMPARATORS)}; this plan "
                f"states {target!r}")
    return True, "", ""


def _check_catalogue(body: Mapping[str, Any], measure: str,
                     operation: str) -> Tuple[bool, str, str]:
    """A figure the semantic model declares: the engine's perimeter, with this
    capability's held readings — and one refusal worded for the forecast."""
    m = MODEL.measure(measure)
    if (m is not None and operation not in m.operations
            and measure == "forecast_funded_balance" and operation == "series"):
        return (False, OPERATION_NOT_SUPPORTED,
                f"the forecast funded balance is the tab's point figure "
                f"over the latest extract ({BALANCE_DECISION}); a series "
                f"of it would pair each funded month with that month's last "
                f"extract — a different definition. The month-by-month "
                f"curve is projected_funded_balance")
    return _engine.check(MODEL, body, measure, operation, held=HELD_READINGS)


# --------------------------------------------------------------------------- #
# execution — the existing owner, called with what the plan says
# --------------------------------------------------------------------------- #

class ForecastOutcome:
    """What ran, what it produced, and the evidence that proves both."""

    __slots__ = ("ok", "reason", "detail", "value", "cells", "receipt")

    def __init__(self, *, ok: bool, reason: str = "", detail: str = "",
                 value: Any = None, cells: Optional[List[Dict[str, Any]]] = None,
                 receipt: Optional[Dict[str, Any]] = None) -> None:
        self.ok = ok
        self.reason = reason
        self.detail = detail
        self.value = value
        self.cells = cells
        self.receipt = receipt or {}


#: An owner, or the plan against the owner's output, cannot answer — the
#: engine's refusal, so one `except` catches both.
_Refusal = _engine.Refusal


def _refuse(reason: str, detail: str) -> ForecastOutcome:
    return ForecastOutcome(ok=False, reason=reason, detail=detail[:300])


def _as_date(text: Any) -> Optional[date]:
    try:
        return date.fromisoformat(str(text)[:10]) if text else None
    except ValueError:
        return None


def vintage_skew(inputs: Mapping[str, Mapping[str, Any]], *,
                 ceiling_days: Optional[int]) -> Tuple[bool, str, str, Optional[int]]:
    """D1 over the inputs a figure USED: `(ok, reason, detail, skew_days)`.

    One input has no skew. Two or more are as far apart as their earliest and
    latest cut-off, and that distance may not exceed the ceiling the funded book
    already uses. A used input with no parseable date is unresolved: a vintage
    that cannot be stated cannot be caveated either.
    """
    dated: Dict[str, date] = {}
    for name, row in inputs.items():
        when = _as_date(row.get("as_of"))
        if when is None:
            return (False, POPULATION_INPUT_UNRESOLVED,
                    f"the {name} input has no governed as-at date "
                    f"({row.get('as_of')!r}); a forecast that cannot state its "
                    f"{name} vintage is refused rather than footnoted", None)
        dated[name] = when
    if len(dated) < 2:
        return True, "", "", None
    earliest, latest = min(dated.values()), max(dated.values())
    skew = (latest - earliest).days
    if ceiling_days is None:
        return (False, POPULATION_VINTAGE_SKEW,
                f"no governed vintage ceiling is configured, so a {skew}-day "
                f"gap between {sorted((k, v.isoformat()) for k, v in dated.items())} "
                f"cannot be judged", skew)
    if skew > int(ceiling_days):
        return (False, POPULATION_VINTAGE_SKEW,
                f"the inputs are {skew} days apart "
                f"({', '.join(f'{k} {v.isoformat()}' for k, v in sorted(dated.items()))}) "
                f"and the configured ceiling is {ceiling_days} days; refresh the "
                f"older input or widen the ceiling in "
                f"config/period_change_selection.yaml", skew)
    return True, "", "", skew


def _ceiling(policy: Any) -> Optional[int]:
    if policy is None:
        from mi_agent.period_change.selection import load_policy
        policy = load_policy()
    return policy.max_snapshot_gap_days(WHOLE_BOOK_CONTEXT_ID)




def execute(plan: Any, *, output_root: Any, pipeline_root: Any, client_id: str,
            run_id: Optional[str] = None,
            history_model: Optional[Mapping[str, Any]] = None,
            funded_frame_resolver: Any = None,
            semantics: Optional[Mapping[str, Any]] = None,
            policy: Any = None) -> ForecastOutcome:
    """An ELIGIBLE forecast plan, through the owner that already computes it.

    `output_root`, `pipeline_root`, `client_id` and `history_model` are the
    inputs the production request already resolved — the same ones the legacy
    forecast route hands the scale-up owner. `funded_frame_resolver` and
    `semantics` are what the governed analytical context reads the funded book
    and the latest weekly extract through. This module discovers none of them.
    `policy` is injectable for tests; production reads the governed file.
    """
    body = _as_mapping(plan)
    kind = kind_of(body)
    if not kind:
        return _refuse(OPERATION_NOT_SUPPORTED,
                       "execute() was called with a plan check_eligibility "
                       "does not admit")
    if not output_root or not client_id:
        return _refuse(INPUTS_UNAVAILABLE,
                       "no governed funded source root or client was supplied "
                       "for this request")
    request = {"output_root": output_root, "pipeline_root": pipeline_root,
               "client_id": client_id, "run_id": run_id,
               "history_model": history_model,
               "funded_frame_resolver": funded_frame_resolver,
               "semantics": semantics}
    try:
        if kind == KIND_MILESTONE:
            return _execute_milestone(body, request=request, policy=policy)
        return _execute_catalogue(body, request=request, policy=policy)
    except _Refusal as refusal:
        return _refuse(refusal.reason, refusal.detail)


# --------------------------------------------------------------------------- #
# the views: each owner, opened with the inputs the request resolved
# --------------------------------------------------------------------------- #

def _analytical_context(request: Mapping[str, Any]) -> Any:
    """The governed context that resolves the funded book and the latest weekly
    extract — the one the composed answer uses. `question` is empty on purpose:
    nothing downstream reads it, and this module never has one to give."""
    from mi_workflows.analytical.context import AnalyticalContext

    resolver = request.get("funded_frame_resolver")
    return AnalyticalContext(
        question="", spec=None, spec_dict={},
        semantics=dict(request.get("semantics") or {}),
        client_id=request["client_id"], run_id=request.get("run_id"),
        output_root=request["output_root"],
        pipeline_root=request.get("pipeline_root"),
        view=EXECUTION_POPULATION, lens=None,
        frame_resolver=resolver, base_frame_resolver=resolver)


def _open_forecast_view(request: Mapping[str, Any], **_: Any
                        ) -> Tuple[Mapping[str, Any], Dict[str, Tuple[str, Dict[str, Any]]],
                                   Dict[str, Any]]:
    """The Forecast tab's envelope (`forecast_view.compose_forecast_view`), for
    the funded book and latest extract the governed context resolves."""
    from mi_workflows.analytical import executors as composer

    if request.get("funded_frame_resolver") is None:
        raise _Refusal(INPUTS_UNAVAILABLE,
                       "no governed funded frame resolver was supplied, and the "
                       "Forecast tab's view reads the funded book through one")
    ctx = _analytical_context(request)
    try:
        payload = composer.forecast_view(ctx)
    except Exception as exc:                                         # noqa: BLE001
        raise _Refusal(EXECUTION_FAILED, f"{type(exc).__name__}: {exc}")
    view = MODEL.views["forecast_view"]
    inputs = {name: (name, {"as_of": _semantic_model.read(payload, spec["as_of"]),
                            "label": spec.get("label") or name,
                            "owner": OWNER_BRIDGE})
              for name, spec in view.inputs.items()}
    return payload, inputs, {"caveats": [str(w) for w in ctx.warnings]}


def _open_scale_up(request: Mapping[str, Any], threshold: Optional[float] = None
                   ) -> Tuple[Mapping[str, Any], Dict[str, Tuple[str, Dict[str, Any]]],
                              Dict[str, Any]]:
    """The scale-up owner's output (`build_extrapolation`), with the reader's
    threshold added to its ladder when a milestone names one. The run-rate's
    own input is the SIGNAL it used: the pipeline's observed completion flow,
    or the funded book where it fell back to funded growth."""
    from mi_agent_api import forecast_extrapolation as fx_mod

    try:
        fx = fx_mod.build_extrapolation(
            request["output_root"], request.get("pipeline_root"),
            request["client_id"], request.get("run_id"),
            history_model=request.get("history_model"),
            extra_thresholds=([threshold] if threshold is not None else ()))
    except Exception as exc:                                         # noqa: BLE001
        raise _Refusal(EXECUTION_FAILED, f"{type(exc).__name__}: {exc}")
    rr = fx.get("completionRunRateForecast") or {}
    if not rr.get("available"):
        raise _Refusal(FORECAST_UNAVAILABLE,
                       f"the forecast owner published no completion run-rate: "
                       f"{rr.get('caveat') or rr.get('status') or 'unavailable'}")
    assumptions = rr.get("assumptions") or {}
    signal_kind = assumptions.get("completionSignalKind")
    if signal_kind not in (fx_mod.SIGNAL_OBSERVED_COMPLETION_FLOW,
                           fx_mod.SIGNAL_FUNDED_GROWTH_PROXY):
        raise _Refusal(POPULATION_INPUT_UNRESOLVED,
                       f"the owner did not say which completion signal its "
                       f"run-rate used ({signal_kind!r}), so its inputs cannot "
                       f"be stated")
    # The inputs' words are the semantic model's (the view's declared inputs),
    # not this module's.
    declared = MODEL.views["scale_up"].inputs
    funded = {"as_of": fx.get("fundedReportingDate"),
              "reporting_period": fx.get("reportingPeriod"),
              "label": declared["funded"]["label"],
              "owner": POPULATION_INPUTS["funded"]["owner"]}
    flow = {"as_of": fx.get("completionFlowExtractDate"),
            "label": declared["signal"]["label"],
            "owner": POPULATION_INPUTS["pipeline"]["owner"]}
    pipeline_fed = signal_kind == fx_mod.SIGNAL_OBSERVED_COMPLETION_FLOW
    inputs = {"funded": ("funded", funded),
              "signal": ("pipeline", flow) if pipeline_fed else ("funded", funded)}
    notes = {
        "model": rr.get("model"),
        "completion_signal": {"kind": signal_kind,
                              "description": assumptions.get("completionSignal")},
        "scenario_basis": rr.get("scenarioBasis"),
        "observed_months": rr.get("observedMonths"),
        "horizon_months": assumptions.get("horizonMonths"),
        "data_sufficiency": fx.get("dataSufficiency"),
        "caveats": [str(c) for c in (rr.get("caveats") or ())],
    }
    return fx, inputs, notes


_OPEN_VIEW = {"forecast_view": _open_forecast_view, "scale_up": _open_scale_up}


def _used_inputs(names: Tuple[str, ...],
                 inputs: Mapping[str, Tuple[str, Dict[str, Any]]]
                 ) -> Dict[str, Dict[str, Any]]:
    """The inputs a figure USED, under the name each one actually is."""
    return {inputs[name][0]: dict(inputs[name][1]) for name in names}


def _vintage(used: Mapping[str, Mapping[str, Any]], policy: Any
             ) -> Tuple[Optional[int], Optional[int]]:
    ceiling = _ceiling(policy)
    ok, why, detail, skew = vintage_skew(used, ceiling_days=ceiling)
    if not ok:
        raise _Refusal(why, detail)
    return skew, ceiling


# --------------------------------------------------------------------------- #
# the semantic model's figures — every one a lookup in its owner's output
# --------------------------------------------------------------------------- #

def _execute_catalogue(body: Mapping[str, Any], *, request: Mapping[str, Any],
                       policy: Any) -> ForecastOutcome:
    """A figure the semantic model declares, read from its owner's output.

    One figure (`value`), one governed member of a breakdown, the owner's own
    breakdown, or the owner's own curve selected to a stated horizon. The
    semantic model says which path; this reads it. Nothing is added, divided,
    projected or re-grouped here.
    """
    m = MODEL.measure(_measure(body))
    view = MODEL.views[m.view]
    payload, inputs, notes = _OPEN_VIEW[m.view](request)

    axis, member, binding = _engine.binding_for(m, body)
    # A breakdown served only on a governed basis (the forecast by region is
    # the reporting taxonomy on both books) is refused, never substituted.
    _engine.requires_met(binding, payload, axis or member)

    names = m.inputs
    if member and "members" in binding:
        names = tuple(binding["members"][member[1]].get("inputs") or m.inputs)
    for name in names:
        path = view.available_when.get(name)
        if path and not _semantic_model.read(payload, path):
            raise _Refusal(POPULATION_INPUT_UNRESOLVED,
                           f"{view.label} has no governed {name} input, so "
                           f"{m.label.lower()} cannot be stated — a forecast "
                           f"with no pipeline is not a forecast with a footnote")
    used = _used_inputs(names, inputs)
    skew, ceiling = _vintage(used, policy)

    shape, value, cells, paths, extra = _engine.serve_figure(
        m, payload, axis=axis, member=member, binding=binding,
        period=_engine.period_as_stated(m, body.get("period") or {}),
        unavailable=FORECAST_UNAVAILABLE)

    context = {key: _semantic_model.read(payload, path)
               for key, path in m.context.items()}
    receipt: Dict[str, Any] = {
        "capability": CAPABILITY,
        "population_base": execution_population(body),
        "operation": str(body.get("operation") or ""),
        "measure_concept": m.name,
        "measure_kind": KIND_CATALOGUE,
        "measure_label": m.label,
        "unit": m.unit,
        "result_shape": shape,
        # WHICH OWNER, AND WHICH OF ITS FIGURES: the view's function and the
        # path of the figure in its output, as the semantic model declares it.
        "execution_owner": view.owner,
        "semantic_model": SEMANTIC_MODEL_FILE,
        "read_path": paths,
        "definition_decision": m.decision or None,
        "applied_predicates": ([{"field": member[0], "op": "eq",
                                 "values": [member[1]]}] if member else []),
        "group_field_keys": [axis] if axis else [],
        "member": ({"dimension": member[0], "value": member[1]}
                   if member else None),
        "inputs": used,
        "inputs_declared": sorted(POPULATION_INPUTS),
        "input_vintage_skew_days": skew,
        "input_vintage_ceiling_days": ceiling,
        "input_vintage_ceiling_owner": CEILING_OWNER,
        # The owner's companion figures the answer states beside this one,
        # named as the semantic model names them.
        "context": context,
        # The explain sentence states the owner's own figure's companions; a
        # figure over another window than the owner's (D22) has none of them.
        "explain": (m.explain if (shape == "scalar" and not member
                                  and "window" not in extra) else ""),
        **context,
        **extra,
        **notes,
    }
    return ForecastOutcome(ok=True, value=value, cells=cells, receipt=receipt)


# --------------------------------------------------------------------------- #
# the milestone for a stated threshold — the owner's rule decides
# --------------------------------------------------------------------------- #

def _execute_milestone(body: Mapping[str, Any], *, request: Mapping[str, Any],
                       policy: Any) -> ForecastOutcome:
    """A milestone for the threshold the question named, from the scale-up
    owner, decided by its own rule (`milestone_answer`)."""
    from mi_agent_api import forecast_extrapolation as fx_mod

    target = target_of(body) or {}
    scale = None
    if isinstance(target.get("value"), str):
        # D9: "scale" is the PORTFOLIO's threshold for its recorded stage —
        # resolved here, after interpretation, and never guessed.
        from mi_agent_api import scale_policy
        scale, why, detail = scale_policy.resolve(request["client_id"])
        if scale is None:
            raise _Refusal(SCALE_NOT_CONFIGURED, detail)
        threshold = scale.threshold
    else:
        threshold = float(target["value"])
    fx, inputs, notes = _open_scale_up(request, threshold=threshold)
    rr = fx.get("completionRunRateForecast") or {}

    current = fx.get("currentFundedBalance")
    decided = fx_mod.milestone_answer(rr.get("milestones") or (), threshold, current)
    if decided["state"] == fx_mod.MILESTONE_ALREADY_REACHED:
        used = _used_inputs(("funded",), inputs)
    else:
        used = _used_inputs(("funded", "signal"), inputs)
    if (decided["state"] == fx_mod.MILESTONE_PROJECTED
            and not (decided["milestone"] or {}).get("baseDate")):
        why_not = "; ".join(rr.get("caveats") or ()) or "no positive run-rate"
        raise _Refusal(FORECAST_UNAVAILABLE,
                       f"the owner projected no base date for this threshold: "
                       f"{why_not}")
    skew, ceiling = _vintage(used, policy)

    milestone = dict(decided.get("milestone") or {})
    receipt: Dict[str, Any] = {
        "capability": CAPABILITY,
        "population_base": execution_population(body),
        "operation": str(body.get("operation") or ""),
        "measure_concept": _measure(body),
        "measure_kind": KIND_MILESTONE,
        "result_shape": "scalar",
        "execution_owner": OWNER_EXTRAPOLATION,
        "decision_owner": OWNER_MILESTONE_RULE,
        # No predicate was requested (a filtered milestone is refused) and none
        # was applied; no axis likewise. Statements, not omissions.
        "applied_predicates": [],
        "group_field_keys": [],
        # WHICH INPUTS FED THIS FIGURE, each with the date it is as at, and
        # which ones were declared but did not.
        "inputs": used,
        "inputs_declared": sorted(POPULATION_INPUTS),
        "input_vintage_skew_days": skew,
        "input_vintage_ceiling_days": ceiling,
        "input_vintage_ceiling_owner": CEILING_OWNER,
        "current_funded_balance": current,
        "base_monthly_run_rate": rr.get("baseMonthlyRunRate"),
        "annualised_run_rate": rr.get("annualisedRunRate"),
        "scenario_monthly_run_rate": dict(rr.get("scenarioMonthlyRunRate") or {}),
        "target": {"concept": str(target.get("concept")),
                   "comparator": str(target.get("comparator")),
                   "value": target.get("value")},
        # The threshold the milestone was measured against — the amount the
        # question named, or the portfolio's scale (D9).
        "threshold_applied": threshold,
        "gap_to_threshold": decided.get("gap"),
        "scale": scale.to_dict() if scale is not None else None,
        "milestone_state": decided["state"],
        "milestone": {k: milestone.get(k) for k in (
            "threshold", "thresholdLabel", "reached", "baseDate",
            "downsideDate", "upsideDate", "baseMonths", "downsideMonths",
            "upsideMonths") if k in milestone},
        **notes,
    }
    value = (milestone.get("baseDate")
             if decided["state"] == fx_mod.MILESTONE_PROJECTED else None)
    return ForecastOutcome(ok=True, value=value, receipt=receipt)
