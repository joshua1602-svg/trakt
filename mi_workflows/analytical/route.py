"""mi_workflows.analytical.route — the layer's single entry point.

Recognition, execution and presentation for the analytical capability layer,
shaped into the SAME governed envelope every other route returns. There is no
new renderer, no new artifact type and no second execution path.

Precedence
----------
The recogniser is registered at a priority AHEAD of the single-capability
routes, because a composite question would otherwise be caught by whichever of
them matched first and answered in part. That is safe only because recognition
is strict: :func:`recognise` matches only when the deterministic planner
produces a plan, and the planner produces one only for composite work no
existing route owns (see ``planner.plan_for``). Everything else falls through
untouched.

The handler additionally returns ``None`` — deferring to the next candidate,
exactly as the registry's contract allows — whenever a plan executes but
produces nothing computable. A book with no governed pipeline therefore gets
the answer it got before this layer existed, not a page of blanks.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

from .contract import (
    Finding,
    KIND_COHORT,
    KIND_COMPARISON,
    KIND_COMPOSITION,
    KIND_COMPOSITION_SHIFT,
    KIND_FORECAST,
    KIND_LIMIT,
    KIND_MEASURE,
    KIND_MOVEMENT,
    KIND_THRESHOLD,
    KIND_TIMING,
)
from .context import AnalyticalContext
from . import narrative as narrative_mod
from . import orchestrator as orchestrator_mod
from . import planner as planner_mod
from . import populations as pops
from .registry import CAPABILITIES

logger = logging.getLogger("mi_workflows.analytical.route")

#: Stable route id. Appears as ``metadata.route`` on the answer.
ROUTE_NAME = "analytical_composition"

#: Priority ahead of the migrated single-capability routes (the earliest of
#: which is ``scenario`` at 10), because a composite question must not be
#: answered in part by whichever narrower recogniser happens to match first.
ROUTE_PRIORITY = 5

#: Recognition confidence. Higher than ``DEFAULT_CONFIDENCE`` so a genuine
#: composite question outranks a single-capability recogniser that also matched
#: — the arbitration the recogniser registry was built to support.
ROUTE_CONFIDENCE = 0.8


# --------------------------------------------------------------------------- #
# Recognition
# --------------------------------------------------------------------------- #
def plan_for_request(request: Any):
    """The deterministic plan for a routed request, or ``None``."""
    try:
        return planner_mod.plan_for(
            request.question, spec=request.spec,
            frame=lambda: _recognition_frame(request),
            semantics=request.semantics)
    except pops.FabricatedPopulation as exc:
        logger.info("analytical plan refused: %s", exc)
        return None
    except Exception as exc:  # noqa: BLE001 - planning must never break routing
        logger.warning("analytical planning failed: %s", exc)
        return None


def _recognition_frame(request: Any):
    """A frame for category matching, only when a resolver was supplied.

    Recognition runs on every routed question, so this is deliberately behind a
    callable: only the last population-pair strategy — two values of one
    governed dimension — needs data, and it is reached only when a question is
    comparative and names neither a provenance nor a seasoning pair. Everything
    else recognises without touching storage.
    """
    resolver = (getattr(request, "base_frame_resolver", None)
                or getattr(request, "frame_resolver", None))
    if resolver is None:
        return None
    try:
        client_id = getattr(request, "client_id", None)
        return resolver(client_id, getattr(request, "run_id", None))
    except Exception:  # noqa: BLE001
        return None


def recognise(request: Any):
    """Match only a question the deterministic planner will compose."""
    from mi_agent_api.recogniser_registry import Recognition

    plan = plan_for_request(request)
    if plan is None or not plan.is_composite:
        return Recognition.no("no composite analytical plan")
    return Recognition.yes(ROUTE_CONFIDENCE, f"analytical plan: {plan.intent}")


# --------------------------------------------------------------------------- #
# Handling
# --------------------------------------------------------------------------- #
def handle(request: Any) -> Optional[Dict[str, Any]]:
    """Execute the plan and shape the governed answer, or defer."""
    from mi_agent_api import chat_routing as _routing

    plan = plan_for_request(request)
    if plan is None or not plan.is_composite:
        return None

    ctx = AnalyticalContext(
        question=request.question, spec=request.spec,
        spec_dict=dict(request.spec_dict or {}), semantics=request.semantics,
        client_id=request.client_id, run_id=request.run_id,
        output_root=request.output_root, pipeline_root=request.pipeline_root,
        portfolio_id=request.portfolio_id, as_of=request.as_of,
        view=request.view,
        lens=_routing._resolve_lens(request.question, request.source_lens),
        frame_resolver=request.frame_resolver,
        base_frame_resolver=getattr(request, "base_frame_resolver", None))

    result = orchestrator_mod.run(plan, ctx)
    if not result.usable:
        # Nothing computed. Hand the question back rather than presenting a
        # composite answer made entirely of gaps — the book then gets exactly
        # the answer it had before this layer existed.
        logger.info("analytical plan %s produced no computable finding; deferring",
                    plan.intent)
        return None
    if not result.satisfied:
        # Something computed, but not the thing the question needed. Refusing
        # here, in the words of the capability that could not run, is the only
        # honest option: the figures that DID compute answer a different
        # question, and offering them would be a substitution.
        return _unmet_envelope(result, request)
    return build_envelope(result, request)


def _unmet_envelope(result: orchestrator_mod.AnalyticalResult, request: Any
                    ) -> Dict[str, Any]:
    """A controlled refusal naming exactly what could not be produced."""
    from mi_agent_api import chat_routing as _routing

    reasons = list(result.unmet_reasons()) or [
        "the governed capability this question needs produced no result for "
        "this book."]
    answer = (" ".join(reasons)
              + " I have not substituted the figures that did compute for the "
                "ones you asked for.")
    envelope = _routing._envelope(
        ok=False, question=request.question, answer=answer,
        spec=dict(request.spec_dict or {}), artifacts=[],
        source_notes=[{
            "label": "Analytical plan",
            "note": (f"{result.plan.intent.replace('_', ' ')}: "
                     + " → ".join(c.capability for c in result.plan.calls)
                     + ". Required: "
                     + ", ".join(result.plan.required_kinds) + "."),
        }],
        warnings=list(result.warnings) + reasons, route=ROUTE_NAME,
        error=reasons[0], lens_applied=True)
    envelope["controlledRefusal"] = True
    envelope["metadata"]["analyticalComposition"] = composition_evidence(result)
    envelope["metadata"]["analyticalPlan"] = result.plan.to_dict()
    envelope["metadata"]["analyticalExecution"] = [
        e.to_dict() for e in result.execution]
    return envelope


# --------------------------------------------------------------------------- #
# Presentation — the existing artifact union only
# --------------------------------------------------------------------------- #
_MEASURE_COLUMNS = [
    {"key": "measure", "label": "Measure", "align": "left"},
    {"key": "population", "label": "Population", "align": "left"},
    {"key": "period", "label": "Period", "align": "left"},
    {"key": "prior", "label": "Prior", "align": "right"},
    {"key": "current", "label": "Current", "align": "right"},
    {"key": "change", "label": "Change", "align": "right"},
]

_COMPARISON_COLUMNS = [
    {"key": "measure", "label": "Measure", "align": "left"},
    {"key": "left", "label": "A", "align": "right"},
    {"key": "right", "label": "B", "align": "right"},
    {"key": "difference", "label": "Difference", "align": "right"},
]

_FORECAST_COLUMNS = [
    {"key": "item", "label": "Item", "align": "left"},
    {"key": "value", "label": "Value", "align": "right"},
    {"key": "when", "label": "Expected", "align": "left"},
    {"key": "basis", "label": "Basis", "align": "left"},
]

_COHORT_COLUMNS = [
    {"key": "vintage", "label": "Vintage", "align": "left"},
    {"key": "balance", "label": "Balance", "align": "right"},
    {"key": "share", "label": "Share", "align": "right"},
    {"key": "loans", "label": "Loans", "align": "right"},
    {"key": "waLtv", "label": "WA LTV", "align": "right"},
    {"key": "waRate", "label": "WA rate", "align": "right"},
    {"key": "waMonthsOnBook", "label": "WA months on book", "align": "right"},
]


def _fraction_pct(value: Optional[float]) -> str:
    """A cohort percent metric, which ``cohort_analysis`` emits as a fraction."""
    return "—" if value is None else f"{float(value) * 100:.2f}%"


#: Governed cohort concepts, as the execution receipt names them. A population
#: kind maps to exactly one, so a plan cannot claim to have compared a cohort it
#: did not split the book by.
_COHORT_CONCEPT_BY_POPULATION_KIND = {
    pops.KIND_SEASONING: "vintage",
    pops.KIND_PROVENANCE: "sourcing",
}


def composition_evidence(result: orchestrator_mod.AnalyticalResult
                         ) -> Dict[str, Any]:
    """What this plan ACTUALLY computed, for the execution receipt.

    Execution evidence, not intent: every field below is read off the findings
    that were produced, so a plan that intended to compare two periods but
    resolved only one cannot claim the comparison. The receipt reconciles the
    question's facets against this block
    (``mi_agent.execution_receipt.analytical_evidence``).
    """
    periods: List[str] = []
    for finding in result.findings:
        period = finding.period
        if period is None or not period.start or not period.end:
            continue
        if period.start == period.end:
            continue
        for date in (period.start, period.end):
            if date not in periods:
                periods.append(str(date))

    comparisons = [f for f in result.of_kind(KIND_COMPARISON) if f.ok]
    compared_populations: List[str] = []
    measures: List[str] = []
    for finding in comparisons:
        for ref in (finding.population, finding.comparand):
            if ref is not None and ref.key not in compared_populations:
                compared_populations.append(ref.key)
        if finding.metric and finding.metric not in measures:
            measures.append(finding.metric)

    # The row predicates the plan ACTUALLY narrowed to, from the findings that
    # were produced. A facet asking whether the answer was scoped to a place or
    # a category is reconciled against this, not against route identity.
    narrowed: List[Dict[str, Any]] = []
    for finding in result.findings:
        ref = finding.population
        if ref is None or ref.is_total or not ref.predicate or not ref.rows:
            continue
        for part in str(ref.predicate).split(";"):
            piece = part.strip()
            if " = " not in piece or piece.startswith("portfolio lens"):
                continue
            field, _, value = piece.partition(" = ")
            entry = {"field": field.strip(), "value": value.strip(),
                     "rows": ref.rows, "dataset": ref.dataset}
            if entry not in narrowed:
                narrowed.append(entry)

    cohort_concept: Optional[str] = None
    for call in result.plan.calls:
        population = call.inputs.get("population")
        if isinstance(population, pops.PopulationSpec):
            concept = _COHORT_CONCEPT_BY_POPULATION_KIND.get(population.kind)
            if concept:
                cohort_concept = concept
                break
    if cohort_concept is None and result.of_kind(KIND_COHORT):
        cohort_concept = "vintage"

    projected = any(f.ok and (f.forecast_value is not None or f.forecast_date)
                    for f in result.of_kind(KIND_FORECAST, KIND_TIMING,
                                            KIND_THRESHOLD))
    return {
        "route": ROUTE_NAME,
        "intent": result.plan.intent,
        "capabilities": [c.capability for c in result.plan.calls],
        "compositionVersion": result.plan.composition_version,
        "planOrigin": result.plan.origin,
        "periodsCompared": periods,
        "projected": bool(projected),
        "cohortConcept": cohort_concept,
        "populationsCompared": compared_populations,
        "narrowedTo": narrowed,
        "measuresCompared": measures,
        "findings": len(result.findings),
    }


def _reconciliation(result: orchestrator_mod.AnalyticalResult) -> Dict[str, Any]:
    """The datasets this plan actually read.

    Named from the capabilities that ran, not assumed: an offer-stage question
    reads the pipeline and nothing else, and reporting it as a full-coverage
    funded answer would misdescribe what was measured.
    """
    datasets: List[str] = []
    for call in result.plan.calls:
        capability = CAPABILITIES.get(call.capability)
        for name in (capability.datasets if capability else ()):
            if name not in datasets:
                datasets.append(name)
    funded = any(d.startswith("funded") for d in datasets)
    return {"dataset": "+".join(datasets) or "funded",
            # Full coverage is claimed only for the funded book, which every
            # governed population here is measured over in full.
            "coverage_by_balance_pct": 100.0 if funded else None}


def build_envelope(result: orchestrator_mod.AnalyticalResult, request: Any
                   ) -> Dict[str, Any]:
    """The governed answer envelope for one analytical result."""
    from mi_agent_api import chat_routing as _routing

    spec_dict = dict(request.spec_dict or {})
    portfolio_id = request.portfolio_id
    as_of = request.as_of
    artifacts: List[Dict[str, Any]] = []

    movements = [f for f in result.of_kind(KIND_MOVEMENT) if f.ok]
    measures = [f for f in result.of_kind(KIND_MEASURE) if f.ok and f.value is not None]
    if movements or measures:
        rows = [{
            "measure": f.label,
            "population": (f.population.label if f.population else "—"),
            "period": narrative_mod._period_text(f),
            "prior": narrative_mod.value_text(f, f.prior_value),
            "current": narrative_mod.value_text(f),
            "change": narrative_mod._delta_text(f) or "—",
        } for f in (movements + measures)]
        artifacts.append(_routing._table_artifact(
            "Governed measures", columns=_MEASURE_COLUMNS, rows=rows,
            spec=spec_dict, portfolio_id=portfolio_id, as_of=as_of,
            description=("Every figure produced by a governed deterministic "
                         "capability; the narrative above states these and "
                         "nothing else.")))

    comparisons = [f for f in result.of_kind(KIND_COMPARISON) if f.ok]
    if comparisons:
        rows = [{
            "measure": f.label,
            "left": narrative_mod.value_text(f),
            "right": narrative_mod.value_text(f, f.comparand_value),
            "difference": narrative_mod._delta_text(f) or "—",
        } for f in comparisons]
        left = comparisons[0].population
        right = comparisons[0].comparand
        artifacts.append(_routing._table_artifact(
            f"{left.label if left else 'A'} vs {right.label if right else 'B'}",
            columns=_COMPARISON_COLUMNS, rows=rows, spec=spec_dict,
            portfolio_id=portfolio_id, as_of=as_of,
            description="Differences computed with the platform's single "
                        "comparison definition."))

    forecasts = [f for f in result.of_kind(KIND_FORECAST, KIND_TIMING, KIND_THRESHOLD)
                 if f.ok]
    if forecasts:
        rows = [{
            "item": f.label,
            "value": narrative_mod.value_text(
                f, f.forecast_value if f.forecast_value is not None else f.value),
            "when": f.forecast_date or narrative_mod._period_text(f) or "—",
            "basis": f.probability_basis or "—",
        } for f in forecasts]
        artifacts.append(_routing._table_artifact(
            "Forecast", columns=_FORECAST_COLUMNS, rows=rows, spec=spec_dict,
            portfolio_id=portfolio_id, as_of=as_of,
            description="Probability-weighted expectations from the governed "
                        "forecast engines. Not a commitment."))
        timing = [f for f in result.of_kind(KIND_TIMING)
                  if f.ok and f.forecast_value is not None and f.forecast_date]
        if len(timing) >= 2:
            artifacts.append(_routing._chart_artifact(
                "Expected completions by month", chart_type="bar",
                x_key="month",
                rows=[{"month": f.forecast_date, "amount": f.forecast_value}
                      for f in timing],
                series=[{"key": "amount", "label": "Expected completions",
                         "color": _routing._PALETTE[0]}],
                value_format="gbp", spec=spec_dict, portfolio_id=portfolio_id,
                as_of=as_of))

    cohorts = [f for f in result.of_kind(KIND_COHORT) if f.ok]
    if cohorts:
        rows = [{
            "vintage": f.label,
            "balance": narrative_mod.money(f.value),
            "share": ("—" if f.share is None else f"{f.share * 100:.1f}%"),
            "loans": f"{int((f.evidence or {}).get('loanCount') or 0):,}",
            "waLtv": _fraction_pct((f.evidence or {}).get("waLtv")),
            "waRate": _fraction_pct((f.evidence or {}).get("waRate")),
            "waMonthsOnBook": ("—" if (f.evidence or {}).get("waMonthsOnBook") is None
                               else f"{(f.evidence or {})['waMonthsOnBook']:.1f}"),
        } for f in cohorts]
        artifacts.append(_routing._table_artifact(
            "Origination vintages", columns=_COHORT_COLUMNS, rows=rows,
            spec=spec_dict, portfolio_id=portfolio_id, as_of=as_of,
            description="Static-pool aggregates per governed origination "
                        "vintage."))

    composition = [f for f in result.of_kind(KIND_COMPOSITION, KIND_COMPOSITION_SHIFT)
                   if f.ok]
    if composition:
        rows = [{
            "measure": f.label,
            "population": (f.population.label if f.population else "—"),
            "period": narrative_mod._period_text(f),
            "prior": ("—" if f.prior_share is None
                      else f"{f.prior_share * 100:.1f}%"),
            "current": ("—" if f.share is None else f"{f.share * 100:.1f}%"),
            "change": narrative_mod._delta_text(f) or "—",
        } for f in composition]
        artifacts.append(_routing._table_artifact(
            "Composition", columns=_MEASURE_COLUMNS, rows=rows, spec=spec_dict,
            portfolio_id=portfolio_id, as_of=as_of,
            description="Shares over the full in-scope denominator; unknown "
                        "values are disclosed, never dropped."))

    limits = [f for f in result.of_kind(KIND_LIMIT) if f.ok]
    if limits:
        rows = [{
            "measure": f.label,
            "population": (f.population.label if f.population else "—"),
            "period": narrative_mod._period_text(f),
            "prior": narrative_mod.value_text(f, f.limit),
            "current": narrative_mod.value_text(f),
            "change": ("—" if f.headroom is None else f"{f.headroom:,.2f}"),
        } for f in limits]
        artifacts.append(_routing._table_artifact(
            "Limits and headroom", columns=_MEASURE_COLUMNS, rows=rows,
            spec=spec_dict, portfolio_id=portfolio_id, as_of=as_of,
            description="From the governed limit evaluation service."))

    source_notes = [dict(n) for n in result.source_notes]
    source_notes.append({
        "label": "Analytical plan",
        "note": (f"{result.plan.intent.replace('_', ' ')}: "
                 + " → ".join(c.capability for c in result.plan.calls)
                 + f". {result.plan.rationale}"),
    })
    engines = sorted({r.engine for r in result.execution if r.engine})
    if engines:
        source_notes.append({
            "label": "Deterministic engines",
            "note": "; ".join(engines) + ". Every figure is produced by these; "
                    "the narrative is composed from their structured findings.",
        })

    envelope = _routing._envelope(
        ok=True, question=request.question,
        answer=narrative_mod.compose(result), spec=spec_dict,
        artifacts=artifacts,
        reconciliation=_reconciliation(result),
        source_notes=source_notes,
        warnings=list(result.warnings), route=ROUTE_NAME, lens_applied=True)
    metadata = envelope["metadata"]
    metadata["analyticalComposition"] = composition_evidence(result)
    metadata["analyticalPlan"] = result.plan.to_dict()
    metadata["analyticalExecution"] = [e.to_dict() for e in result.execution]
    metadata["analyticalFindings"] = [f.to_dict() for f in result.findings]
    metadata["analyticalCapabilities"] = list(CAPABILITIES.ids())
    if result.population_evidence:
        metadata["populationApplied"] = dict(result.population_evidence)
    return envelope


# --------------------------------------------------------------------------- #
# Registration
# --------------------------------------------------------------------------- #
def recogniser():
    """The :class:`Recogniser` this layer registers."""
    from mi_agent_api.recogniser_registry import Recogniser

    return Recogniser(
        name=ROUTE_NAME, priority=ROUTE_PRIORITY, lens_aware=True,
        description=("Composes two or more governed deterministic analytical "
                     "capabilities into one answer."),
        metadata={
            "layer": "analytical_capability",
            "capabilities": list(CAPABILITIES.ids()),
            "intents": list(planner_mod.ALL_INTENTS),
            "defers_to": list(CAPABILITIES.owned_routes()),
        },
        recognise=recognise, handle=handle)
