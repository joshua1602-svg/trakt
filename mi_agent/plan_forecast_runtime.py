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
owner already takes, calls it, and writes down what ran:

    milestone date     forecast_extrapolation.build_extrapolation, with the
                         reader's threshold added to the ladder, then
                       forecast_extrapolation.milestone_answer — the ONE rule
                         deciding which of four answers a milestone question
                         has, lifted out of the legacy router so this module
                         reads the rule instead of copying it
    completion
    run-rate           the same owner's Model A `baseMonthlyRunRate`
    forecast funded    the analytical composer's `funded_balance_forecast` —
    balance              `forecast_bridge.compute_forecast_bridge` over the
                         LATEST weekly extract, the figure the React Forecast
                         tab shows (D6, settled by the owner)

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

    a forecast funded        D6 made the composer's point figure the definition.
    balance SERIES           The only forecast series in the estate pairs each
                             funded month with that month's last extract — a
                             different definition (£96.2m against £94.1m on the
                             production book) — so a series is refused rather
                             than answered from it.
    scenario                 D2b — its own operation with a typed `assumption`
                             slot, built separately (Change 2b).
    a scoped forecast        the run-rate is book-wide; applying it to one book
                             or one role is the silent widening §8.6 forbids.
    filters, axes,           no forecast owner narrows or groups these figures.
    geography, comparison

IT NEVER READS THE QUESTION. No parse, no recogniser, no router, and nothing
from `chat_routing` — whose milestone closure is the reason the milestone rule
had to move to the owner before this module could exist.
"""
from __future__ import annotations

from datetime import date
from typing import Any, Dict, FrozenSet, Mapping, Optional, Tuple

CAPABILITY = "forecast"

#: WHICH POPULATION THIS RUNTIME EXECUTES. Its own declaration, and never
#: `funded`: a runtime dispatched above the funded gate that also claimed funded
#: would take funded plans past the gate that guards them. Normalisation rule 6
#: is what lets a forecast plan spelled with a `funded` base reach this set.
EXECUTABLE_POPULATIONS: FrozenSet[str] = frozenset({"forecast"})

#: The population this runtime declares to the population gate.
EXECUTION_POPULATION = "forecast"

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
NOT_SINGLE_OUTPUT = "NOT_SINGLE_OUTPUT"
SCOPE_NOT_SUPPORTED = "SCOPE_NOT_SUPPORTED"
FILTERS_NOT_SUPPORTED = "FILTERS_NOT_SUPPORTED"
DIMENSION_NOT_SUPPORTED = "DIMENSION_NOT_SUPPORTED"
GEOGRAPHY_NOT_SUPPORTED = "GEOGRAPHY_NOT_SUPPORTED"
COMPARISON_NOT_SUPPORTED = "COMPARISON_NOT_SUPPORTED"
MEASURE_NOT_SUPPORTED = "MEASURE_NOT_SUPPORTED"
OPERATION_NOT_SUPPORTED = "OPERATION_NOT_SUPPORTED"
PERIOD_NOT_SUPPORTED = "PERIOD_NOT_SUPPORTED"
TARGET_NOT_SUPPORTED = "TARGET_NOT_SUPPORTED"
#: The owner computes this shape, but the production bank shows the model
#: using it for questions that want different figures. Held, not served.
AMBIGUOUS_READING = "AMBIGUOUS_READING"
INPUTS_UNAVAILABLE = "INPUTS_UNAVAILABLE"
FORECAST_UNAVAILABLE = "FORECAST_UNAVAILABLE"
EXECUTION_FAILED = "EXECUTION_FAILED"
#: P0 §9 — the two reasons a derived population adds.
POPULATION_INPUT_UNRESOLVED = "POPULATION_INPUT_UNRESOLVED"
POPULATION_VINTAGE_SKEW = "POPULATION_VINTAGE_SKEW"

#: WHAT THIS RUNTIME SERVES, as (operation, measure) -> the figure it reads.
KIND_MILESTONE = "milestone"
KIND_RUN_RATE = "completion_run_rate"
KIND_BALANCE = "forecast_funded_balance"
SERVED: Mapping[Tuple[str, str], str] = {
    ("forecast_milestone", "forecast_milestone_date"): KIND_MILESTONE,
    ("point_in_time", "forecast_completion_rate"): KIND_RUN_RATE,
    # D6: "what is the forecast funded balance" and "what will the book grow to
    # on the current pipeline" are one figure, the composer's.
    ("point_in_time", "forecast_funded_balance"): KIND_BALANCE,
    ("forecast_projection", "forecast_funded_balance"): KIND_BALANCE,
}

#: The period form each served figure is stated in. A milestone looks forward;
#: a run-rate is the current one. Anything else — "the 8-week run-rate", a
#: weekly series — has no owner here and is refused rather than re-windowed.
PERIOD_FORMS: Mapping[str, FrozenSet[str]] = {
    KIND_MILESTONE: frozenset({"forward_looking"}),
    KIND_RUN_RATE: frozenset({"current"}),
    # "What is the forecast funded balance?" arrives as current, "what will the
    # book grow to?" as forward-looking; both ask for the one composer figure.
    KIND_BALANCE: frozenset({"current", "forward_looking"}),
}

#: The owner decision that defines the forecast funded balance, named on every
#: refusal and receipt that depends on it.
BALANCE_DECISION = "D6"

#: SHAPES THE OWNER COMPUTES BUT A PLAN CANNOT YET BE TRUSTED TO MEAN.
#:
#: Measured, not supposed: the plans the canary recorded for the owner's
#: production bank (2026-09-28, read back by `qb_plan_readback`) put several
#: different questions into each of these shapes, and the runtime can only
#: answer the shape. The vocabulary gives the model a name and no definition for
#: these measures — "Owned by the forecast capability" — so it has nothing to
#: separate them by. Serving the shape would answer some of those questions with
#: a different figure than they asked for, which the design's hard gate (WRONG
#: answers must stay at zero) forbids. They are held until the vocabulary
#: defines the measures (P2) and a live run shows the readings separate; legacy
#: serves them meanwhile, as it did.
HELD_READINGS: Mapping[Tuple[str, str], str] = {
    ("forecast_projection", "forecast_funded_balance"): (
        "this shape arrived for the expected funded balance [87], but also for "
        "the extrapolation curve [114], the base scenario [117] and the funded "
        "share of the forecast [94] — four different figures"),
    ("point_in_time", "forecast_completion_rate"): (
        "this shape arrived for the completion run-rate [112, 113], but also "
        "for a KFI-to-completion conversion rate [121] and for the forecast's "
        "method [98]"),
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
OWNER_COMPOSER = "mi_workflows.analytical.executors.funded_balance_forecast"
OWNER_BRIDGE = "forecast_bridge.compute_forecast_bridge"


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


def _single_output(plan: Mapping[str, Any]) -> Optional[Mapping[str, Any]]:
    outputs = plan.get("outputs") or ()
    return outputs[0] if len(outputs) == 1 else None


def _measure(plan: Mapping[str, Any]) -> str:
    output = _single_output(plan) or {}
    measures = output.get("measures") or ()
    return str(measures[0].get("concept") or "") if len(measures) == 1 else ""


def kind_of(plan: Any) -> str:
    """Which served figure an ELIGIBLE plan asks for. Empty for anything else."""
    body = _as_mapping(plan)
    return SERVED.get((str(body.get("operation") or ""), _measure(body)), "")


def target_of(plan: Any) -> Optional[Mapping[str, Any]]:
    target = _as_mapping(plan).get("target")
    return target if isinstance(target, Mapping) else None


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
                f"runtime (it executes {sorted(EXECUTABLE_POPULATIONS)}); a "
                f"forecast of the pipeline alone is a different figure from the "
                f"funded book projected forward")
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
    if body.get("filters") or output.get("filters"):
        return (False, FILTERS_NOT_SUPPORTED,
                "no forecast owner narrows its figures by a predicate")
    if output.get("dimensions"):
        return (False, DIMENSION_NOT_SUPPORTED,
                "no forecast owner groups its figures by an axis")
    if output.get("geography") or (body.get("geography") or {}).get("requested"):
        return (False, GEOGRAPHY_NOT_SUPPORTED,
                "no forecast owner breaks its figures down by geography")
    if str(body.get("comparison_kind") or "none") != "none":
        return (False, COMPARISON_NOT_SUPPORTED,
                "the forecast runtime serves no population comparison")

    measures = [str(m.get("concept") or "") for m in (output.get("measures") or ())]
    if len(measures) != 1:
        return (False, MEASURE_NOT_SUPPORTED,
                f"exactly one measure is served; this plan names {measures}")
    operation = str(body.get("operation") or "")
    kind = SERVED.get((operation, measures[0]))
    if not kind and measures[0] == KIND_BALANCE:
        return (False, OPERATION_NOT_SUPPORTED,
                f"the forecast funded balance is the composer's point figure "
                f"over the latest extract ({BALANCE_DECISION}); "
                f"operation={operation!r} would need a series of it, and the "
                f"only forecast series pairs each funded month with that "
                f"month's last extract — a different definition")
    if not kind:
        served = sorted(f"{op}/{m}" for op, m in SERVED)
        return (False, OPERATION_NOT_SUPPORTED,
                f"operation={operation!r} with measure {measures[0]!r} has no "
                f"forecast owner here (served: {served})")

    # THE GRAIN FIRST: it is the durable refusal. "By month" asks for a figure
    # per period whatever the measure turns out to mean, so it must still
    # refuse after the hold below is lifted.
    period = body.get("period") or {}
    if period.get("grain"):
        return (False, PERIOD_NOT_SUPPORTED,
                f"a {period.get('grain')} grain asks for a figure per period; "
                f"the {kind} figure is a single point")
    held = HELD_READINGS.get((operation, measures[0]))
    if held:
        return (False, AMBIGUOUS_READING,
                f"{operation}/{measures[0]} is held until the vocabulary defines "
                f"the measure: {held}")
    form = str(period.get("form") or "")
    if form not in PERIOD_FORMS[kind]:
        return (False, PERIOD_NOT_SUPPORTED,
                f"period.form={form!r} is not how the {kind} figure is stated "
                f"({sorted(PERIOD_FORMS[kind])})")

    target = target_of(body)
    if kind == KIND_MILESTONE:
        value = (target or {}).get("value")
        if (not target or str(target.get("concept") or "") != TARGET_CONCEPT
                or str(target.get("comparator") or "") not in TARGET_COMPARATORS
                or isinstance(value, bool)
                or not isinstance(value, (int, float)) or value <= 0):
            return (False, TARGET_NOT_SUPPORTED,
                    f"a milestone is served for a positive "
                    f"{TARGET_CONCEPT} threshold compared "
                    f"{sorted(TARGET_COMPARATORS)}; this plan states {target!r}")
    elif target:
        return (False, TARGET_NOT_SUPPORTED,
                f"the {kind} figure takes no threshold; this plan states "
                f"{target!r}")
    return True, "", ""


# --------------------------------------------------------------------------- #
# execution — the existing owner, called with what the plan says
# --------------------------------------------------------------------------- #

class ForecastOutcome:
    """What ran, what it produced, and the evidence that proves both."""

    __slots__ = ("ok", "reason", "detail", "value", "receipt")

    def __init__(self, *, ok: bool, reason: str = "", detail: str = "",
                 value: Any = None,
                 receipt: Optional[Dict[str, Any]] = None) -> None:
        self.ok = ok
        self.reason = reason
        self.detail = detail
        self.value = value
        self.receipt = receipt or {}


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
    forecast route hands this same owner. `funded_frame_resolver` and
    `semantics` are what the legacy analytical route hands the composer. This
    module discovers none of them. `policy` is injectable for tests;
    production reads the governed file.
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
    if kind == KIND_BALANCE:
        return _execute_balance(
            body, output_root=output_root, pipeline_root=pipeline_root,
            client_id=client_id, run_id=run_id,
            funded_frame_resolver=funded_frame_resolver, semantics=semantics,
            policy=policy)
    return _execute_run_rate(body, kind=kind, output_root=output_root,
                             pipeline_root=pipeline_root, client_id=client_id,
                             run_id=run_id, history_model=history_model,
                             policy=policy)


def _execute_balance(body: Mapping[str, Any], *, output_root: Any,
                     pipeline_root: Any, client_id: str, run_id: Optional[str],
                     funded_frame_resolver: Any,
                     semantics: Optional[Mapping[str, Any]],
                     policy: Any) -> ForecastOutcome:
    """The forecast funded balance, from the composer the Forecast tab agrees with.

    THE OWNER IS `funded_balance_forecast`, called with the context the legacy
    analytical route builds — the governed funded frame from the resolver
    `mi_service` supplies, the latest governed weekly extract the context
    resolves itself, no lens (a scoped forecast never reaches here) — and it
    delegates to `forecast_bridge.compute_forecast_bridge`, the function
    `/mi/forecast/snapshot` calls for the React Forecast tab. So the governed
    figure, the legacy composed answer and the tab are one definition (D6).

    `question` is empty on purpose: the executor does not read it, and this
    module never has one to give.
    """
    from mi_workflows.analytical import contract as contract_mod
    from mi_workflows.analytical import executors as composer
    from mi_workflows.analytical.context import AnalyticalContext

    if funded_frame_resolver is None:
        return _refuse(INPUTS_UNAVAILABLE,
                       "no governed funded frame resolver was supplied, and the "
                       "composer reads the funded book through one")
    ctx = AnalyticalContext(
        question="", spec=None, spec_dict={}, semantics=dict(semantics or {}),
        client_id=client_id, run_id=run_id, output_root=output_root,
        pipeline_root=pipeline_root, view=EXECUTION_POPULATION, lens=None,
        frame_resolver=funded_frame_resolver,
        base_frame_resolver=funded_frame_resolver)
    try:
        findings = composer.funded_balance_forecast(ctx)
    except Exception as exc:                                         # noqa: BLE001
        return _refuse(EXECUTION_FAILED, f"{type(exc).__name__}: {exc}")

    def _finding(metric: str) -> Any:
        return next((f for f in findings if f.metric == metric
                     and f.kind == contract_mod.KIND_FORECAST), None)

    forecast = _finding("forecast_funded_balance")
    expected = _finding("weighted_expected_funded_amount")
    if forecast is None or not forecast.ok or forecast.value is None:
        # The composer says no governed pipeline exists. A forecast with no
        # pipeline is not a forecast with a footnote (P0 §9).
        return _refuse(POPULATION_INPUT_UNRESOLVED,
                       (getattr(forecast, "note", None)
                        or "the composer produced no forecast funded balance"))
    evidence = dict(forecast.evidence or {})
    expected_evidence = dict(getattr(expected, "evidence", None) or {})
    used = {
        "funded": {"as_of": evidence.get("fundedReportingDate"),
                   "label": "funded book", "owner": OWNER_BRIDGE},
        "pipeline": {"as_of": evidence.get("pipelineAsOfDate"),
                     "label": "pipeline extract", "owner": OWNER_BRIDGE},
    }
    ceiling = _ceiling(policy)
    ok, why, detail, skew = vintage_skew(used, ceiling_days=ceiling)
    if not ok:
        return _refuse(why, detail)

    receipt: Dict[str, Any] = {
        "capability": CAPABILITY,
        "population_base": EXECUTION_POPULATION,
        "operation": str(body.get("operation") or ""),
        "measure_concept": _measure(body),
        "measure_kind": KIND_BALANCE,
        "result_shape": "scalar",
        "execution_owner": OWNER_COMPOSER,
        "calculation_owner": evidence.get("engine") or OWNER_BRIDGE,
        "definition_decision": BALANCE_DECISION,
        "applied_predicates": [],
        "group_field_keys": [],
        "inputs": used,
        "inputs_declared": sorted(POPULATION_INPUTS),
        "input_vintage_skew_days": skew,
        "input_vintage_ceiling_days": ceiling,
        "input_vintage_ceiling_owner": CEILING_OWNER,
        "formula": evidence.get("formula"),
        "forecast_funded_balance": forecast.value,
        "current_funded_balance": evidence.get("fundedBalance"),
        "weighted_expected_funded_amount":
            evidence.get("weightedExpectedFundedAmount"),
        "forecast_loan_count": evidence.get("forecastLoanCount"),
        "eligible_case_count": expected_evidence.get("eligibleCaseCount"),
        "excluded_case_count": expected_evidence.get("excludedCaseCount"),
        "excluded_amount": expected_evidence.get("excludedFromWeightingAmount"),
        "probability_basis": forecast.probability_basis,
        "caveats": [str(w) for w in ctx.warnings],
    }
    return ForecastOutcome(ok=True, value=forecast.value, receipt=receipt)


def _execute_run_rate(body: Mapping[str, Any], *, kind: str, output_root: Any,
                      pipeline_root: Any, client_id: str, run_id: Optional[str],
                      history_model: Optional[Mapping[str, Any]],
                      policy: Any) -> ForecastOutcome:
    """A milestone or the completion run-rate, from the scale-up owner."""
    from mi_agent_api import forecast_extrapolation as fx_mod

    target = target_of(body) if kind == KIND_MILESTONE else None
    threshold = float(target["value"]) if target else None
    try:
        fx = fx_mod.build_extrapolation(
            output_root, pipeline_root, client_id, run_id,
            history_model=history_model,
            extra_thresholds=([threshold] if threshold is not None else ()))
    except Exception as exc:                                         # noqa: BLE001
        return _refuse(EXECUTION_FAILED, f"{type(exc).__name__}: {exc}")

    rr = fx.get("completionRunRateForecast") or {}
    if not rr.get("available"):
        return _refuse(FORECAST_UNAVAILABLE,
                       f"the forecast owner published no completion run-rate: "
                       f"{rr.get('caveat') or rr.get('status') or 'unavailable'}")
    assumptions = rr.get("assumptions") or {}
    signal_kind = assumptions.get("completionSignalKind")
    pipeline_fed = signal_kind == fx_mod.SIGNAL_OBSERVED_COMPLETION_FLOW
    if signal_kind not in (fx_mod.SIGNAL_OBSERVED_COMPLETION_FLOW,
                           fx_mod.SIGNAL_FUNDED_GROWTH_PROXY):
        return _refuse(POPULATION_INPUT_UNRESOLVED,
                       f"the owner did not say which completion signal its "
                       f"run-rate used ({signal_kind!r}), so its inputs cannot "
                       f"be stated")

    funded_input = {"as_of": fx.get("fundedReportingDate"),
                    "reporting_period": fx.get("reportingPeriod"),
                    "label": "funded book",
                    "owner": POPULATION_INPUTS["funded"]["owner"]}
    pipeline_input = {"as_of": fx.get("completionFlowExtractDate"),
                      "label": "pipeline completion flow",
                      "owner": POPULATION_INPUTS["pipeline"]["owner"]}
    signal_input = pipeline_input if pipeline_fed else funded_input

    current = fx.get("currentFundedBalance")
    decided: Optional[Mapping[str, Any]] = None
    if kind == KIND_MILESTONE:
        decided = fx_mod.milestone_answer(rr.get("milestones") or (),
                                          threshold, current)
        if decided["state"] == fx_mod.MILESTONE_ALREADY_REACHED:
            used = {"funded": funded_input}
        else:
            used = {"funded": funded_input,
                    ("pipeline" if pipeline_fed else "funded"): signal_input}
        if (decided["state"] == fx_mod.MILESTONE_PROJECTED
                and not (decided["milestone"] or {}).get("baseDate")):
            why_not = "; ".join(rr.get("caveats") or ()) or "no positive run-rate"
            return _refuse(FORECAST_UNAVAILABLE,
                           f"the owner projected no base date for this "
                           f"threshold: {why_not}")
    else:
        used = {("pipeline" if pipeline_fed else "funded"): signal_input}

    ceiling = _ceiling(policy)
    ok, why, detail, skew = vintage_skew(used, ceiling_days=ceiling)
    if not ok:
        return _refuse(why, detail)

    milestone = dict((decided or {}).get("milestone") or {})
    receipt: Dict[str, Any] = {
        "capability": CAPABILITY,
        "population_base": EXECUTION_POPULATION,
        "operation": str(body.get("operation") or ""),
        "measure_concept": _measure(body),
        "measure_kind": kind,
        "result_shape": "scalar",
        "execution_owner": OWNER_EXTRAPOLATION,
        "model": rr.get("model"),
        # No predicate was requested (a filtered forecast is refused) and none
        # was applied; no axis likewise. Statements, not omissions.
        "applied_predicates": [],
        "group_field_keys": [],
        # WHICH INPUTS FED THIS FIGURE, each with the date it is as at, and
        # which ones were declared but did not.
        "inputs": {name: dict(row) for name, row in used.items()},
        "inputs_declared": sorted(POPULATION_INPUTS),
        "input_vintage_skew_days": skew,
        "input_vintage_ceiling_days": ceiling,
        "input_vintage_ceiling_owner": CEILING_OWNER,
        "completion_signal": {"kind": signal_kind,
                              "description": assumptions.get("completionSignal")},
        "current_funded_balance": current,
        "base_monthly_run_rate": rr.get("baseMonthlyRunRate"),
        "annualised_run_rate": rr.get("annualisedRunRate"),
        "scenario_monthly_run_rate": dict(rr.get("scenarioMonthlyRunRate") or {}),
        "scenario_basis": rr.get("scenarioBasis"),
        "observed_months": rr.get("observedMonths"),
        "horizon_months": assumptions.get("horizonMonths"),
        "data_sufficiency": fx.get("dataSufficiency"),
        "caveats": [str(c) for c in (rr.get("caveats") or ())],
    }
    if kind == KIND_MILESTONE:
        receipt.update({
            "decision_owner": OWNER_MILESTONE_RULE,
            "target": {"concept": str(target.get("concept")),
                       "comparator": str(target.get("comparator")),
                       "value": target.get("value")},
            "milestone_state": decided["state"],
            "milestone": {k: milestone.get(k) for k in (
                "threshold", "thresholdLabel", "reached", "baseDate",
                "downsideDate", "upsideDate", "baseMonths", "downsideMonths",
                "upsideMonths") if k in milestone},
        })
        value = milestone.get("baseDate") if decided["state"] == \
            fx_mod.MILESTONE_PROJECTED else None
    else:
        value = rr.get("baseMonthlyRunRate")
    return ForecastOutcome(ok=True, value=value, receipt=receipt)
