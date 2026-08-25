"""mi_workflows.analytical.executors — thin adapters, no arithmetic.

Every function here marshals declared inputs into an engine that already exists
and shapes its output into :class:`~mi_workflows.analytical.contract.Finding`
objects. The rule this module is written to, and which
``tests/test_analytical_capability_layer.py`` enforces by parsing the source:

    **No adapter computes a financial result.**

Sums, weighted averages, shares, distributions, comparisons, movements,
probabilities, run-rates, projections and headroom are all produced by
``mi_workflows.engine``, ``mi_agent.period_change``, ``mi_agent_api.cohorts``,
``mi_agent_api.pipeline_contract``, ``mi_agent_api.forecast_bridge``,
``mi_agent_api.forecast_extrapolation`` and
``mi_agent_api.concentration_tests_api``. An adapter reads those results,
attaches the population, period and evidence they were computed under, and
stops.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from mi_workflows import engine
from mi_workflows.semantics import SemanticEntry

from .contract import (
    DATASET_PIPELINE,
    Finding,
    KIND_COHORT,
    KIND_COMPOSITION,
    KIND_COMPOSITION_SHIFT,
    KIND_FORECAST,
    KIND_LIMIT,
    KIND_MEASURE,
    KIND_MOVEMENT,
    KIND_THRESHOLD,
    KIND_TIMING,
    PeriodRef,
    PopulationRef,
    STATUS_OK,
    STATUS_QUALIFIED,
    STATUS_UNAVAILABLE,
)
from .context import AnalyticalContext
from . import populations as pops

logger = logging.getLogger("mi_workflows.analytical.executors")

BALANCE = "current_outstanding_balance"

#: Ordered profile dimensions. Each is a governed MI-semantics dimension; the
#: ones a book does not carry are skipped and the omission is disclosed. The
#: order is the order a lending CFO reads a book in — where it lends, at what
#: leverage, to whom, at what ticket and at what price.
PROFILE_DIMENSIONS: Tuple[Tuple[str, str], ...] = (
    ("collateral_geography", "Region"),
    ("ltv_bucket", "LTV band"),
    ("age_bucket", "Borrower age band"),
    ("ticket_bucket", "Ticket size"),
    ("interest_rate_bucket", "Interest rate band"),
)

#: The governed measures that describe an origination profile, in reading order.
PROFILE_MEASURES: Tuple[str, ...] = (
    "loan_count", BALANCE, "current_loan_to_value", "current_interest_rate",
    "youngest_borrower_age",
)

#: The subset of :data:`PROFILE_MEASURES` the Business Semantics Registry
#: governs for PERIOD CHANGE. ``loan_count`` is deliberately absent: the
#: registry tags it for portfolio comparison only, and asking the period-change
#: workflow for a field it does not govern for that workflow would be this layer
#: overriding the registry rather than reading it. The population's own row
#: counts carry the loan count instead, from execution evidence.
PROFILE_MOVEMENT_MEASURES: Tuple[str, ...] = tuple(
    m for m in PROFILE_MEASURES if m != "loan_count")


# --------------------------------------------------------------------------- #
# Shared helpers
# --------------------------------------------------------------------------- #
def _entry(ctx: AnalyticalContext, field: str) -> Optional[SemanticEntry]:
    bsr = ctx.business_semantics()
    return bsr.get(field) if bsr is not None else None


def _unit(ctx: AnalyticalContext, field: str) -> str:
    return engine.unit_for_field(field, ctx.semantics)


def percent_scale(frame, column: str) -> float:
    """The multiplier that puts a percent column into PERCENTAGE POINTS.

    The funded tape's convention is not uniform — the same book can store LTV as
    a fraction (0.3967) and the interest rate in points (6.575), and a prepared
    frame can differ from a dated snapshot of the same book. The platform
    already owns the decision
    (:func:`mi_agent.mi_dataset_profile.percent_storage_scale`); this reads it
    and returns a scale, so an answer states 39.68% whichever convention the
    frame it read happened to use. It changes no calculation — the aggregate is
    computed on the stored values and only its presentation is normalised, and
    the scale that was applied is recorded in the finding's evidence.
    """
    from mi_agent.mi_dataset_profile import PERCENT_POINTS, percent_storage_scale

    if frame is None or column not in getattr(frame, "columns", []):
        return 1.0
    try:
        return (1.0 if percent_storage_scale(frame[column]) == PERCENT_POINTS
                else 100.0)
    except Exception:  # noqa: BLE001 - an unreadable column is left as stored
        return 1.0


def _aggregate(ctx: AnalyticalContext, frame, field: str
               ) -> Tuple[Optional[engine.MetricAggregate], Optional[SemanticEntry]]:
    """One governed aggregate, with the registry choosing the aggregation.

    The Business Semantics Registry decides HOW a measure aggregates and what it
    weights by; ``mi_workflows.engine`` implements that decision. This function
    chooses neither.
    """
    entry = _entry(ctx, field)
    if entry is None or frame is None:
        return None, entry
    agg = engine.aggregate(
        frame, field, entry.default_aggregation,
        weight_field=entry.weight_field, share_basis=entry.share_basis,
        unit=_unit(ctx, field))
    return agg, entry


def _measure_finding(capability: str, field: str, agg: engine.MetricAggregate,
                     entry: SemanticEntry, population: PopulationRef,
                     period: Optional[PeriodRef], *, engine_name: str,
                     scale: float = 1.0) -> Finding:
    value = (agg.value if agg.value is None or agg.unit != engine.UNIT_PERCENT
             else agg.value * scale)
    return Finding(
        capability=capability, kind=KIND_MEASURE,
        label=entry.display_name, metric=field,
        population=population, period=period,
        unit=agg.unit, aggregation=agg.aggregation, value=value,
        status=STATUS_OK if agg.value is not None else STATUS_UNAVAILABLE,
        note=(None if agg.value is not None else
              f"{entry.display_name} has no valid values in this population."),
        evidence={
            "engine": engine_name,
            "calculationVersion": engine.CALCULATION_VERSION,
            "aggregation": agg.aggregation,
            "weightBasis": agg.weight_basis,
            "population": agg.population,
            "validPopulation": agg.valid_population,
            "excludedPopulation": agg.excluded_population,
            "denominator": agg.denominator,
            "storedValue": agg.value,
            "percentScale": (scale if agg.unit == engine.UNIT_PERCENT else None),
        })


#: The Period Change workflow reports movement in its OWN unit vocabulary
#: (``mi_agent.period_change.models``). Mapping it to the engine's display units
#: here — rather than teaching the narrative a second vocabulary — keeps one
#: formatter for every finding this layer produces.
_MOVEMENT_UNIT_TO_DISPLAY = {
    "currency": engine.UNIT_CURRENCY,
    "count": engine.UNIT_INTEGER,
    "percentage_point": engine.UNIT_PERCENT,
    "ratio": engine.UNIT_SHARE,
    "years": engine.UNIT_DECIMAL,
    "not_applicable": engine.UNIT_DECIMAL,
}


def _population_unavailable(capability: str, kind: str,
                            population: pops.PopulationSpec,
                            evidence: Mapping[str, Any]) -> Optional[Finding]:
    """Refuse when a requested population could not actually be applied.

    ``apply_population`` leaves the frame ALONE for a predicate whose column the
    book does not carry. Measuring the untouched frame and labelling it with the
    population would present a whole-book figure as a narrowed one — the P1L
    failure mode — so the capability reports unavailable instead.
    """
    missing = list((evidence or {}).get("unavailable") or ())
    if not missing:
        return None
    return Finding(
        capability=capability, kind=kind, label=population.label,
        population=PopulationRef(key=population.key, label=population.label),
        status=STATUS_UNAVAILABLE,
        note=(f"{population.label} could not be applied to this book "
              f"({'; '.join(sorted(set(missing)))}), so no figure was produced "
              "for it and the whole book was not measured in its place."))


def _period_ref(ctx: AnalyticalContext, end: Optional[str] = None) -> PeriodRef:
    snaps = ctx.snapshots()
    available = tuple(str(s.reporting_date) for s in snaps if s.reporting_date)
    return PeriodRef(end=end or (available[-1] if available else ctx.as_of),
                     available=available)


# --------------------------------------------------------------------------- #
# 1. portfolio_snapshot
# --------------------------------------------------------------------------- #
def portfolio_snapshot(ctx: AnalyticalContext, *, measures: Sequence[str],
                       population: pops.PopulationSpec = pops.TOTAL,
                       ) -> List[Finding]:
    """Governed measures for one population at the current reporting date."""
    from mi_agent_api import chat_routing as _routing

    frame = ctx.base_frame()
    if frame is None:
        return [Finding(capability="portfolio_snapshot", kind=KIND_MEASURE,
                        label="Funded position", status=STATUS_UNAVAILABLE,
                        note="The governed funded book could not be resolved.")]
    narrowed, ref, evidence = pops.apply(
        frame, population, ctx.semantics,
        lens_filter=_routing._apply_lens_filter)
    refusal = _population_unavailable("portfolio_snapshot", KIND_MEASURE,
                                     population, evidence)
    if refusal is not None:
        return [refusal]
    period = _period_ref(ctx)
    out: List[Finding] = []
    for field in measures:
        agg, entry = _aggregate(ctx, narrowed, field)
        if agg is None or entry is None:
            out.append(Finding(
                capability="portfolio_snapshot", kind=KIND_MEASURE,
                label=field.replace("_", " "), metric=field, population=ref,
                period=period, status=STATUS_UNAVAILABLE,
                note=(f"{field} is not a governed measure in the business "
                      "semantics registry, so no value was produced.")))
            continue
        out.append(_measure_finding(
            "portfolio_snapshot", field, agg, entry, ref, period,
            engine_name="mi_workflows.engine",
            scale=percent_scale(narrowed, field)))
    return out


# --------------------------------------------------------------------------- #
# 2. population_profile
# --------------------------------------------------------------------------- #
def population_profile(ctx: AnalyticalContext, *,
                       population: pops.PopulationSpec = pops.TOTAL,
                       dimensions: Sequence[str] = (),
                       basis: str = engine.BASIS_EXPOSURE,
                       top_n: int = 5) -> List[Finding]:
    """How one population is composed across governed dimensions."""
    from mi_agent_api import chat_routing as _routing

    frame = ctx.base_frame()
    if frame is None:
        return [Finding(capability="population_profile", kind=KIND_COMPOSITION,
                        label="Composition", status=STATUS_UNAVAILABLE,
                        note="The governed funded book could not be resolved.")]
    narrowed, ref, evidence = pops.apply(
        frame, population, ctx.semantics,
        lens_filter=_routing._apply_lens_filter)
    refusal = _population_unavailable("population_profile", KIND_COMPOSITION,
                                     population, evidence)
    if refusal is not None:
        return [refusal]
    wanted = list(dimensions) or [d for d, _ in PROFILE_DIMENSIONS]
    labels = dict(PROFILE_DIMENSIONS)
    period = _period_ref(ctx)
    out: List[Finding] = []
    skipped: List[str] = []
    for field in wanted:
        ranked = engine.ranked_distribution(narrowed, field,
                                            exposure_field=BALANCE, basis=basis)
        if ranked is None:
            skipped.append(labels.get(field, field))
            continue
        for bucket in ranked.buckets[:top_n]:
            out.append(Finding(
                capability="population_profile", kind=KIND_COMPOSITION,
                label=f"{labels.get(field, field)}: {bucket.category}",
                metric=field, population=ref, period=period,
                unit=engine.UNIT_SHARE, aggregation=engine.AGG_DISTRIBUTION,
                value=bucket.exposure, share=bucket.exposure_share,
                rank=bucket.rank, rank_of=len(ranked.buckets),
                evidence={
                    "engine": "mi_workflows.engine.ranked_distribution",
                    "calculationVersion": engine.CALCULATION_VERSION,
                    "basis": ranked.basis, "dimension": field,
                    "population": ranked.population,
                    "unknownCount": ranked.unknown_count,
                    "unknownShare": ranked.unknown_count_share,
                    "categories": len(ranked.buckets),
                    "cumulativeShare": bucket.cumulative_share,
                    "count": bucket.count,
                }))
    if skipped:
        ctx.warn("Not profiled — this book does not carry: "
                 + ", ".join(sorted(skipped)) + ".")
    return out


# --------------------------------------------------------------------------- #
# 3. period_movement
# --------------------------------------------------------------------------- #
#: How a plan may frame the comparison window.
WINDOW_RETAINED = "retained_window"     # earliest → latest governed snapshot
WINDOW_LATEST_PAIR = "latest_pair"      # the two most recent snapshots


def period_movement(ctx: AnalyticalContext, *,
                    population: pops.PopulationSpec = pops.TOTAL,
                    measures: Sequence[str] = (),
                    dimensions: Sequence[str] = (),
                    window: str = WINDOW_RETAINED) -> List[Finding]:
    """Governed movement of measures (and composition) for one population.

    Delegates entirely to :mod:`mi_agent.period_change` — the governed Period
    Change Analysis workflow — after narrowing each governed snapshot to the
    population with the same predicate machinery the point-in-time path uses.
    Every movement figure, unit, directionality and interpretation below is the
    workflow's; nothing is recomputed here.
    """
    from mi_agent.period_change.models import (
        MODE_PORTFOLIO_OVERVIEW, MODE_REQUESTED_METRIC, PeriodChangeFailure,
        PeriodResolution, SnapshotFrame,
    )
    from mi_agent.period_change.periods import PeriodRequest
    from mi_agent.period_change.workflow import (
        PeriodChangeRequest, run_period_change_analysis,
    )
    from mi_agent import period_request as _period_request
    from mi_agent import population as _population
    from mi_agent_api import chat_routing as _routing

    snaps = ctx.snapshots()
    if len(snaps) < 2:
        return [Finding(
            capability="period_movement", kind=KIND_MOVEMENT,
            label="Movement", population=PopulationRef(population.key,
                                                       population.label),
            status=STATUS_UNAVAILABLE,
            note=("Fewer than two governed reporting snapshots are retained for "
                  "this book, so no movement can be measured."))]

    predicates = _population.material_predicates(population.filters, ctx.semantics)
    narrowed: List[SnapshotFrame] = []
    unavailable: List[str] = []
    for snap in snaps:
        frame = snap.frame
        if population.lens_term:
            frame = _routing._apply_lens_filter(frame, pops.resolve_lens(population))
        if predicates:
            frame, ev = _population.apply_population(frame, predicates, ctx.semantics)
            if not ev.is_usable:
                # FAIL CLOSED: `apply_population` hands back no frame for a
                # predicate it could not execute, so there is nothing to
                # measure. The refusal below is the governed outcome; building
                # a snapshot out of the un-narrowed book first is exactly the
                # substitution this route exists not to make.
                unavailable.extend(ev.unavailable
                                   or [ev.blocked_reason or "predicate could not execute"])
                break
        narrowed.append(SnapshotFrame(
            snapshot_id=snap.snapshot_id, reporting_date=snap.reporting_date,
            frame=frame, dataset_label=snap.dataset_label,
            dataset_reference=snap.dataset_reference, row_count=int(len(frame)),
            portfolio_ids=snap.portfolio_ids))
    if unavailable:
        return [Finding(
            capability="period_movement", kind=KIND_MOVEMENT, label="Movement",
            population=PopulationRef(population.key, population.label),
            status=STATUS_UNAVAILABLE,
            note=("This book does not carry the field the requested population "
                  f"needs ({'; '.join(sorted(set(unavailable)))}), so no "
                  "narrowed movement was calculated."))]
    rows_before = int(len(snaps[-1].frame))
    rows_after = int(len(narrowed[-1].frame))

    # THE SPAN THE QUESTION ASKED FOR, ahead of the plan's default window.
    #
    # WINDOW_RETAINED opens at the EARLIEST retained snapshot. That is the right
    # default for a question that names no period, and silent widening for one
    # that does: "in the last few months" was answered over twelve months on a
    # thirteen-snapshot book. It read as correct on three snapshots only because
    # the widest window there happened to be two months.
    #
    # Two cases, and they are not the same:
    #   * A span the question COUNTED ("this year", "the last 6 months") is
    #     honoured where the history reaches, and clarified where it does not —
    #     a shorter window is a different number, and the D rule already governs
    #     that.
    #   * A span a GOVERNED CONVENTION settled ("recent", "lately", "a few
    #     months") is honoured where the history reaches, and where it does not
    #     the whole retained window is used and BOTH are disclosed. "Recent" is
    #     satisfied by any window inside the recent window, so a book with less
    #     history than the convention still answers the question that was asked;
    #     "this year" answered over one month does not.
    span = _period_request.requested_span(ctx.question)
    span_start = None
    if window == WINDOW_RETAINED:
        if span is not None and len(narrowed) > span.periods:
            span_start = str(narrowed[-1 - span.periods].reporting_date)
        else:
            span_start = str(narrowed[0].reporting_date)
            if span is not None and span.governed:
                ctx.warn(
                    f"The governed recent window is {span.periods} months; this "
                    f"book retains {len(narrowed) - 1}, so the comparison opens "
                    f"at the earliest retained snapshot "
                    f"({narrowed[0].reporting_date}).")
    if span is not None and span.governed and span_start is not None:
        ctx.warn(f"'{ctx.question.strip()}' names no countable period, so it is "
                 f"answered {span.disclosure()}.")

    request = PeriodChangeRequest(
        question=ctx.question,
        mode=MODE_REQUESTED_METRIC if measures else MODE_PORTFOLIO_OVERVIEW,
        period_request=(PeriodRequest(
            requested_start=span_start,
            requested_end=str(narrowed[-1].reporting_date))
            if window == WINDOW_RETAINED else PeriodRequest()),
        requested_fields=tuple(measures),
        include_bridge=False,
        composition_focus=bool(dimensions) or not measures)
    try:
        result = run_period_change_analysis(request, tuple(narrowed))
    except PeriodChangeFailure as failure:
        return [Finding(
            capability="period_movement", kind=KIND_MOVEMENT, label="Movement",
            population=PopulationRef(population.key, population.label),
            status=STATUS_UNAVAILABLE,
            note=("The governed period-change analysis could not run: "
                  f"{failure.reason.replace('_', ' ')}."))]

    resolution: PeriodResolution = result.period_resolution
    period = PeriodRef(start=str(resolution.start_snapshot.reporting_date),
                       end=str(resolution.end_snapshot.reporting_date),
                       method=resolution.resolution_method,
                       available=tuple(str(s.reporting_date) for s in snaps))
    predicate_text = "; ".join(p.describe() for p in predicates)
    if not predicate_text and population.lens_term:
        # The RESOLVED lens, with the registry's explicit portfolio ids — the
        # same text ``populations.apply`` records, so a population reads
        # identically whichever capability produced it.
        lens = pops.resolve_lens(population)
        ids = (lens.filters or {}).get("source_portfolio_id") or []
        predicate_text = (f"portfolio lens = {lens.label}"
                          + (f" ({', '.join(sorted(str(i) for i in ids))})"
                             if ids else ""))
    # A3 — the membership of this population at the START of the compared
    # period. Taken from the snapshot the governed resolution actually chose,
    # matched by reporting date, so it describes the same two dates the movement
    # figures do. Every one of these counts was already computed above when the
    # snapshots were narrowed; none is recalculated here.
    start_date = str(resolution.start_snapshot.reporting_date)
    rows_prior = next((int(s.row_count) for s in narrowed
                       if str(s.reporting_date) == start_date), None)
    ref = PopulationRef(key=population.key, label=population.label,
                        predicate=predicate_text or None,
                        rows=rows_after, rows_before=rows_before,
                        rows_prior=rows_prior,
                        time_relative=(population.kind == pops.KIND_SEASONING),
                        is_total=population.is_total)

    out: List[Finding] = []
    wanted = set(measures) if measures else None
    for change in result.metric_changes:
        if wanted is not None and change.field not in wanted:
            continue
        # The Period Change workflow already normalises a percent measure to
        # percentage points (``period_change.calculations``, which reads the same
        # storage-scale decision this module does). Re-scaling here would apply
        # it twice, so the value is taken exactly as the workflow reports it and
        # only its unit NAME is mapped to the display vocabulary.
        unit = _MOVEMENT_UNIT_TO_DISPLAY.get(change.movement_unit,
                                             engine.UNIT_DECIMAL)
        out.append(Finding(
            capability="period_movement", kind=KIND_MOVEMENT,
            label=change.display_name, metric=change.field,
            population=ref, period=period,
            unit=unit, aggregation=change.aggregation,
            value=change.end_value, prior_value=change.start_value,
            change=change.movement_value,
            relative_change=change.relative_change,
            rank=change.movement_rank, rank_of=change.rank_population,
            status=(STATUS_OK if change.comparable else STATUS_QUALIFIED),
            note=(None if change.comparable else
                  f"{change.display_name}: {change.status.replace('_', ' ')}."),
            evidence={
                "engine": "mi_agent.period_change",
                "calculationVersion": result.audit.get("calculation_version"),
                "resolutionMethod": resolution.resolution_method,
                "movementBasis": change.movement_basis,
                "directionality": change.directionality,
                "interpretation": change.interpretation,
                "significance": change.significance,
                "startPopulation": change.start.valid_population,
                "endPopulation": change.end.valid_population,
                "startDenominator": change.start.denominator,
                "endDenominator": change.end.denominator,
                "movementUnit": change.movement_unit,
                "percentScale": "normalised by mi_agent.period_change",
            }))

    focus = set(dimensions)
    labels = dict(PROFILE_DIMENSIONS)
    for dist in result.distribution_changes:
        if focus and dist.field not in focus:
            continue
        moved = sorted(
            (c for c in dist.categories if c.balance_share_movement is not None),
            key=lambda c: -abs(c.balance_share_movement or 0.0))[:3]
        for cat in moved:
            out.append(Finding(
                capability="period_movement", kind=KIND_COMPOSITION_SHIFT,
                label=f"{labels.get(dist.field, dist.display_name)}: {cat.category}",
                metric=dist.field, population=ref, period=period,
                unit=engine.UNIT_SHARE, aggregation=engine.AGG_DISTRIBUTION,
                value=cat.end_balance, prior_value=cat.start_balance,
                share=cat.end_balance_share, prior_share=cat.start_balance_share,
                change=cat.balance_share_movement,
                evidence={
                    "engine": "mi_agent.period_change.distribution",
                    "calculationVersion": result.audit.get("calculation_version"),
                    "dimension": dist.field, "presence": cat.presence,
                    "startCount": cat.start_count, "endCount": cat.end_count,
                    "startTotalBalance": dist.start_total_balance,
                    "endTotalBalance": dist.end_total_balance,
                }))
    if population.kind == pops.KIND_SEASONING:
        # A seasoning segment is a ROLLING population: the front book at the
        # closing date is not the front book at the opening date, because loans
        # age out of it. The movement is therefore a movement in the segment,
        # not in a fixed set of loans, and a reader who is not told that will
        # read a shrinking front book as falling origination.
        ctx.warn(
            f"{population.label} is a rolling population — loans move out of it "
            "as they season, so this is the movement in the segment between the "
            "two reporting dates, not the movement of one fixed set of loans.")
    for warning in result.warnings:
        ctx.warn(warning)
    for limitation in result.limitations:
        ctx.warn(limitation)
    return out


# --------------------------------------------------------------------------- #
# 4. vintage_analysis
# --------------------------------------------------------------------------- #
def vintage_analysis(ctx: AnalyticalContext, *,
                     population: pops.PopulationSpec = pops.TOTAL,
                     dimension: str = "vintage") -> List[Finding]:
    """Static-pool metrics per origination vintage.

    Delegates to :func:`mi_agent_api.cohorts.cohort_analysis`, which owns the
    per-cohort balance, loan count, book share and balance-weighted LTV, rate
    and months-on-book. Nothing is recomputed here.
    """
    from mi_agent_api import chat_routing as _routing
    from mi_agent_api import cohorts as cohorts_mod

    frame = ctx.base_frame()
    if frame is None:
        return [Finding(capability="vintage_analysis", kind=KIND_COHORT,
                        label="Vintages", status=STATUS_UNAVAILABLE,
                        note="The governed funded book could not be resolved.")]
    narrowed, ref, evidence = pops.apply(
        frame, population, ctx.semantics,
        lens_filter=_routing._apply_lens_filter)
    refusal = _population_unavailable("vintage_analysis", KIND_COHORT,
                                     population, evidence)
    if refusal is not None:
        return [refusal]
    period = _period_ref(ctx)
    result = cohorts_mod.cohort_analysis(
        narrowed, client_id=ctx.client_id, portfolio_id=ctx.portfolio_id or "",
        reporting_date=period.end, dimension=dimension)
    if not result.get("available"):
        return [Finding(capability="vintage_analysis", kind=KIND_COHORT,
                        label="Vintages", population=ref, period=period,
                        status=STATUS_UNAVAILABLE,
                        note=("Vintage analysis is not available for this book: "
                              f"{result.get('reason')}."))]
    metrics = result.get("metricsAvailable") or []
    out: List[Finding] = []
    cohorts = result.get("cohorts") or []
    for index, row in enumerate(cohorts, start=1):
        out.append(Finding(
            capability="vintage_analysis", kind=KIND_COHORT,
            label=str(row.get("cohort")), metric="vintage_year",
            population=ref, period=period,
            unit=engine.UNIT_CURRENCY, aggregation=engine.AGG_SUM,
            value=row.get("balance"),
            share=((row.get("sharePct") or 0.0) / 100.0
                   if row.get("sharePct") is not None else None),
            rank=index, rank_of=len(cohorts),
            evidence={
                "engine": "mi_agent_api.cohorts.cohort_analysis",
                "cohortBasis": result.get("cohortBasis"),
                "metricsAvailable": list(metrics),
                "loanCount": row.get("loanCount"),
                "waLtv": row.get("waLtv"),
                "waRate": row.get("waRate"),
                "waMonthsOnBook": row.get("waMonthsOnBook"),
                "totalBalance": result.get("totalBalance"),
                "totalLoanCount": result.get("totalLoanCount"),
            }))
    return out


# --------------------------------------------------------------------------- #
# 5. pipeline_stock
# --------------------------------------------------------------------------- #
STAGE_OFFER = "OFFER"


def _no_pipeline(capability: str, kind: str, meta: Mapping[str, Any]) -> Finding:
    return Finding(
        capability=capability, kind=kind, label="Pipeline",
        status=STATUS_UNAVAILABLE,
        note=(str(meta.get("reason"))
              or "No governed pipeline source is available for this book."))


def pipeline_stock(ctx: AnalyticalContext, *, stages: Sequence[str] = ()
                   ) -> List[Finding]:
    """Case count and gross amount per open pipeline stage."""
    snapshot = ctx.pipeline_snapshot()
    if snapshot is None:
        _frame, _report, meta = ctx.pipeline()
        return [_no_pipeline("pipeline_stock", KIND_MEASURE, meta)]
    wanted = {str(s).upper() for s in stages} or None
    as_of = snapshot.get("pipelineAsOfDate")
    period = PeriodRef(end=as_of)
    rows = snapshot.get("stageBreakdown") or []
    out: List[Finding] = []
    for row in rows:
        stage = str(row.get("stage") or "").upper()
        if wanted is not None and stage not in wanted:
            continue
        ref = PopulationRef(key=f"pipeline:{stage.lower()}",
                            label=f"pipeline cases at {stage.title()} stage",
                            predicate=f"pipeline_stage = {stage}",
                            rows=int(row.get("caseCount") or 0),
                            rows_before=int(snapshot.get("pipelineRowCount") or 0),
                            exposure=row.get("pipelineAmount"),
                            dataset=DATASET_PIPELINE)
        out.append(Finding(
            capability="pipeline_stock", kind=KIND_MEASURE,
            label=f"{stage.title()} stage pipeline", metric="pipeline_amount",
            population=ref, period=period,
            unit=engine.UNIT_CURRENCY, aggregation=engine.AGG_SUM,
            value=row.get("pipelineAmount"),
            evidence={
                "engine": "mi_agent_api.pipeline_contract.compute_pipeline_snapshot",
                "caseCount": row.get("caseCount"),
                "weightedExpectedFundedAmount": row.get("weightedExpectedFundedAmount"),
                "pipelineAsOfDate": as_of,
                "sourceFile": snapshot.get("sourceFile"),
                "totalPipelineAmount": snapshot.get("pipelineAmount"),
                "totalPipelineCases": snapshot.get("pipelineRowCount"),
            }))
    if not out:
        return [Finding(
            capability="pipeline_stock", kind=KIND_MEASURE,
            label="Pipeline", period=period, status=STATUS_UNAVAILABLE,
            note=("The governed pipeline carries no open cases at "
                  f"{', '.join(sorted(wanted)) if wanted else 'any'} stage."))]
    ctx.note("Pipeline", f"Weekly pipeline extract as at {as_of} "
                         f"({snapshot.get('sourceFile') or 'governed source'}).")
    _note_probability_basis(ctx)
    return out


# --------------------------------------------------------------------------- #
# 6. pipeline_completion_forecast
# --------------------------------------------------------------------------- #
def money(value: Optional[float]) -> str:
    """Currency for a disclosure line, in the request's resolved currency."""
    from mi_agent_api import currency as currency_mod
    return currency_mod.format_money(value, suffixes=("bn", "m", "k"))


#: Plain-English readings of the governed completion-probability bases. A CFO
#: reading an expected completion figure is entitled to know whether it rests on
#: what the book has actually done or on a configured assumption.
_BASIS_NOTES = {
    "historical": ("Completion probabilities are the EMPIRICAL stage rates "
                   "observed across the retained weekly pipeline extracts."),
    "stage_config": ("Completion probabilities are the CONFIGURED stage "
                     "assumptions: the observed sample is below the governed "
                     "sufficiency floor."),
    "mixed_historical_and_config": (
        "Completion probabilities are empirical stage rates where the observed "
        "sample clears the governed sufficiency floor, and the configured stage "
        "assumption where it does not."),
}


def _note_probability_basis(ctx: AnalyticalContext) -> None:
    """State what the completion probabilities rest on, and their known bias."""
    _frame, _report, meta = ctx.pipeline()
    basis = meta.get("basis")
    if not basis:
        return
    note = _BASIS_NOTES.get(str(basis), f"Completion probability basis: {basis}.")
    weeks = meta.get("weeklyExtractsUsed")
    if weeks:
        note += (f" Observed over {weeks} retained weekly extract(s); a case that "
                 "has not yet had time to complete within that window counts "
                 "against the rate, so an empirical rate reads as a floor.")
    ctx.note("Completion probability", note)


def _within_horizon(breakdown: Sequence[Mapping[str, Any]], as_of: Optional[str],
                    horizon_months: Optional[int]) -> List[Mapping[str, Any]]:
    """The expected-completion months a plan asked to see.

    With no horizon, every month the governed breakdown carries. With one, the
    months from the pipeline as-of month forward, capped at that many — so "the
    next three months" shows three months and does not quietly include a month
    that has already passed.
    """
    rows = list(breakdown)
    if horizon_months is None:
        return rows
    start = str(as_of or "")[:7]
    future = [r for r in rows if not start or str(r.get("month")) >= start]
    return future[:max(0, int(horizon_months))]


def pipeline_completion_forecast(ctx: AnalyticalContext, *,
                                 stages: Sequence[str] = (),
                                 horizon_months: Optional[int] = None
                                 ) -> List[Finding]:
    """Expected completion amount and timing for open pipeline cases.

    The completion probability on each case, and the expected completion month
    derived from it, are the governed pipeline preparation layer's — empirical
    stage rates over the retained weekly window where the observed sample clears
    the sufficiency floor, the configured stage assumption below it. This
    adapter reads them; it never assigns one.
    """
    frame, report, meta = ctx.pipeline()
    if frame is None or report is None:
        return [_no_pipeline("pipeline_completion_forecast", KIND_FORECAST, meta)]
    from mi_agent_api import pipeline_contract as pipeline_mod
    from mi_agent_api import pipeline_prep as prep_mod

    work = frame
    stage_label = "the open pipeline"
    wanted = {str(s).upper() for s in stages}
    if wanted and "pipeline_stage" in work.columns:
        work = work[work["pipeline_stage"].astype(str).str.upper().isin(wanted)]
        stage_label = f"pipeline cases at {', '.join(sorted(wanted)).title()} stage"
    elif not wanted and "pipeline_stage" in work.columns:
        work = work[work["pipeline_stage"].astype(str).str.upper()
                    .isin(prep_mod.ACTIVE_STAGES)]

    as_of = meta.get("asOf")
    period = PeriodRef(end=as_of)
    ref = PopulationRef(key="pipeline:" + ("+".join(sorted(wanted)).lower() or "active"),
                        label=stage_label,
                        predicate=("pipeline_stage in "
                                   f"{sorted(wanted) or list(prep_mod.ACTIVE_STAGES)}"),
                        rows=int(len(work)),
                        rows_before=int(len(frame)),
                        dataset=DATASET_PIPELINE)
    if work is None or work.empty:
        return [Finding(
            capability="pipeline_completion_forecast", kind=KIND_FORECAST,
            label="Expected completions", population=ref, period=period,
            status=STATUS_UNAVAILABLE,
            note=f"The governed pipeline carries no open cases in {stage_label}.")]

    # The EXPECTED AMOUNT comes from the stage breakdown, which covers every
    # case in scope. The completion-month breakdown covers only cases carrying
    # completion-timing evidence, so using it for the amount would understate
    # the expectation and disagree with the forecast bridge. Both subtotals are
    # the governed pipeline contract's own; nothing is re-derived here.
    stages_out = pipeline_mod._dimension_breakdown(work, "pipeline_stage",
                                                   key_name="stage")
    total_weighted = sum(float(r.get("weightedExpectedFundedAmount") or 0.0)
                         for r in stages_out)
    gross_amount = sum(float(r.get("pipelineAmount") or 0.0) for r in stages_out)
    breakdown = pipeline_mod._expected_completion_breakdown(work)
    timed_weighted = sum(float(r.get("weightedExpectedFundedAmount") or 0.0)
                         for r in breakdown)
    basis = meta.get("basis")
    _note_probability_basis(ctx)
    untimed = round(total_weighted - timed_weighted, 2)
    if untimed > 0.005:
        ctx.warn(
            f"{money(untimed)} of the expected completion amount carries no "
            "completion-timing evidence in the governed pipeline, so it is "
            "included in the expected amount but not in the monthly profile.")
    out: List[Finding] = [Finding(
        capability="pipeline_completion_forecast", kind=KIND_FORECAST,
        label=f"Expected completion amount from {stage_label}",
        metric="weighted_expected_funded_amount",
        population=ref, period=period,
        unit=engine.UNIT_CURRENCY, aggregation=engine.AGG_SUM,
        forecast_value=round(total_weighted, 2), value=round(total_weighted, 2),
        probability_basis=basis,
        evidence={
            "engine": "mi_agent_api.pipeline_contract._dimension_breakdown",
            "grossPipelineAmount": round(gross_amount, 2),
            "completionProbabilityBasis": basis,
            "weeklyExtractsUsed": meta.get("weeklyExtractsUsed"),
            "pipelineAsOfDate": as_of,
            "timedExpectedAmount": round(timed_weighted, 2),
            "untimedExpectedAmount": untimed,
            "months": len(breakdown),
            "stages": stages_out,
        })]
    for row in _within_horizon(breakdown, as_of, horizon_months):
        month = str(row.get("month"))
        out.append(Finding(
            capability="pipeline_completion_forecast", kind=KIND_TIMING,
            label=f"Expected to complete in {month}",
            metric="weighted_expected_funded_amount",
            population=ref, period=period, unit=engine.UNIT_CURRENCY,
            aggregation=engine.AGG_SUM,
            forecast_value=row.get("weightedExpectedFundedAmount"),
            value=row.get("weightedExpectedFundedAmount"),
            forecast_date=month, probability_basis=basis,
            evidence={
                "engine": "mi_agent_api.pipeline_contract._expected_completion_breakdown",
                "caseCount": row.get("caseCount"),
                "grossExpectedFundedAmount": row.get("expectedFundedAmount"),
            }))
    return out


# --------------------------------------------------------------------------- #
# 7. funded_balance_forecast
# --------------------------------------------------------------------------- #
def _open_stage_amount(frame) -> float:
    """Gross amount at the OPEN funnel stages, from the governed breakdown."""
    if frame is None or "pipeline_stage" not in getattr(frame, "columns", []):
        return 0.0
    from mi_agent_api import pipeline_contract as pipeline_mod
    from mi_agent_api import pipeline_prep as prep_mod

    rows = pipeline_mod._dimension_breakdown(frame, "pipeline_stage",
                                             key_name="stage")
    return round(sum(float(r.get("pipelineAmount") or 0.0) for r in rows
                     if str(r.get("stage") or "").upper()
                     in prep_mod.ACTIVE_STAGES), 2)


def _settled_stage_component(frame) -> float:
    """Expected completions the extract already shows at a settled stage.

    Read from the governed stage breakdown, so the figure is the pipeline
    contract's own subtotal rather than a second aggregation of the column.
    """
    if frame is None or "pipeline_stage" not in getattr(frame, "columns", []):
        return 0.0
    from mi_agent_api import pipeline_contract as pipeline_mod
    from mi_agent_api import pipeline_prep as prep_mod

    rows = pipeline_mod._dimension_breakdown(frame, "pipeline_stage",
                                             key_name="stage")
    return round(sum(float(r.get("weightedExpectedFundedAmount") or 0.0)
                     for r in rows
                     if str(r.get("stage") or "").upper()
                     not in prep_mod.ACTIVE_STAGES), 2)


def funded_balance_forecast(ctx: AnalyticalContext) -> List[Finding]:
    """Current funded balance plus expected completions, from the governed bridge."""
    from mi_agent_api import forecast_bridge as bridge_mod

    frame, report, meta = ctx.pipeline()
    funded = ctx.base_frame()
    snapshot = ctx.pipeline_snapshot()
    bridge = bridge_mod.compute_forecast_bridge(
        client_id=ctx.client_id, run_id=str(ctx.run_id or ""),
        funded_reporting_date=_period_ref(ctx).end,
        funded_df=funded, pipeline_df=frame, pipeline_report=report,
        pipeline_snapshot=snapshot,
        pipeline_source=(meta.get("source")
                         if isinstance(meta.get("source"), dict) else None),
    ).get("forecastBridge") or {}

    period = _period_ref(ctx)
    total = PopulationRef(key="total", label="the whole funded book",
                          rows=bridge.get("fundedLoanCount"),
                          exposure=bridge.get("fundedBalance"), is_total=True)
    findings = [Finding(
        capability="funded_balance_forecast", kind=KIND_MEASURE,
        label="Current funded balance", metric=BALANCE,
        population=total, period=period, unit=engine.UNIT_CURRENCY,
        aggregation=engine.AGG_SUM, value=bridge.get("fundedBalance"),
        evidence={"engine": "mi_agent_api.forecast_bridge.compute_forecast_bridge",
                  "fundedLoanCount": bridge.get("fundedLoanCount"),
                  "fundedReportingDate": bridge.get("fundedReportingDate")})]

    if not bridge.get("pipelineAvailable"):
        findings.append(Finding(
            capability="funded_balance_forecast", kind=KIND_FORECAST,
            label="Forecast funded balance", metric="forecast_funded_balance",
            population=total, period=period, status=STATUS_UNAVAILABLE,
            note=("No governed pipeline source is available for this book, so a "
                  "forecast funded balance cannot be produced. The current "
                  "funded balance above is not a forecast.")))
        return findings

    # The bridge weights EVERY case the weekly extract carries, so its gross
    # amount and case count are the EXTRACT's, not the open pipeline's. Labelling
    # them "open pipeline" would overstate what is still to come by the value of
    # the cases the extract already shows as completed.
    pipeline_ref = PopulationRef(
        key="pipeline:extract", label="the governed pipeline extract",
        predicate="every case in the weekly extract",
        rows=bridge.get("pipelineCaseCount"),
        exposure=bridge.get("pipelineAmount"),
        dataset=DATASET_PIPELINE)
    findings.append(Finding(
        capability="funded_balance_forecast", kind=KIND_MEASURE,
        label="Gross pipeline in the governed extract", metric="pipeline_amount",
        population=pipeline_ref, period=PeriodRef(end=bridge.get("pipelineAsOfDate")),
        unit=engine.UNIT_CURRENCY, aggregation=engine.AGG_SUM,
        value=bridge.get("pipelineAmount"),
        evidence={"engine": "mi_agent_api.forecast_bridge.compute_forecast_bridge",
                  "pipelineCaseCount": bridge.get("pipelineCaseCount"),
                  "openStageAmount": _open_stage_amount(frame),
                  "excludedFromWeightingAmount": bridge.get("excludedFromWeightingAmount"),
                  "excludedCaseCount": bridge.get("excludedCaseCount")}))
    # B5 — the governed exclusion, stated as a fact about the forecast rather
    # than as a diagnostic. This used to warn that the expected completions
    # INCLUDED settled cases, which was true and was the defect; now they are
    # excluded, and what a reader needs to know is which population the forward
    # figure covers and what it leaves out.
    excluded_amount = bridge.get("excludedFromWeightingAmount") or 0.0
    excluded_cases = bridge.get("excludedCaseCount") or 0
    if excluded_cases:
        ctx.warn(
            "The forward forecast covers the open pipeline only. "
            f"{excluded_cases} case(s) worth {money(excluded_amount)} are "
            "excluded because the extract already shows them as completed or "
            "withdrawn, so they are not future completions.")
    # The forward expectation covers the ELIGIBLE population, not every case the
    # extract carries, so it must not be labelled with the whole-extract
    # population reference used for the gross figure above.
    eligible_ref = PopulationRef(
        key="pipeline:eligible", label="the open pipeline",
        predicate="pipeline cases the governed config includes in forward funding",
        rows=bridge.get("eligibleCaseCount"),
        rows_before=bridge.get("pipelineCaseCount"),
        dataset=DATASET_PIPELINE)
    findings.append(Finding(
        capability="funded_balance_forecast", kind=KIND_FORECAST,
        label="Expected completions from the open pipeline",
        metric="weighted_expected_funded_amount",
        population=eligible_ref, period=PeriodRef(end=bridge.get("pipelineAsOfDate")),
        unit=engine.UNIT_CURRENCY, aggregation=engine.AGG_SUM,
        forecast_value=bridge.get("weightedExpectedFundedAmount"),
        value=bridge.get("weightedExpectedFundedAmount"),
        probability_basis=bridge.get("completionProbabilityBasis"),
        evidence={"engine": "mi_agent_api.forecast_bridge.compute_forecast_bridge",
                  "blendedWeightedConversion": bridge.get("blendedWeightedConversion"),
                  "excludedFromWeightingAmount": excluded_amount,
                  "excludedCaseCount": excluded_cases,
                  "eligibleCaseCount": bridge.get("eligibleCaseCount"),
                  "forecastReadiness": (bridge.get("forecastReadiness") or {}).get("status")}))
    findings.append(Finding(
        capability="funded_balance_forecast", kind=KIND_FORECAST,
        label="Forecast funded balance", metric="forecast_funded_balance",
        population=total, period=period, unit=engine.UNIT_CURRENCY,
        aggregation=engine.AGG_SUM,
        forecast_value=bridge.get("forecastFundedBalance"),
        value=bridge.get("forecastFundedBalance"),
        probability_basis=bridge.get("completionProbabilityBasis"),
        evidence={
            "engine": "mi_agent_api.forecast_bridge.compute_forecast_bridge",
            "formula": ("forecast_funded_balance = current_funded_balance + "
                        "sum(expected_funded_amount * completion_probability)"),
            "fundedBalance": bridge.get("fundedBalance"),
            "weightedExpectedFundedAmount": bridge.get("weightedExpectedFundedAmount"),
            "forecastLoanCount": bridge.get("forecastLoanCount"),
            "fundedReportingDate": bridge.get("fundedReportingDate"),
            "pipelineAsOfDate": bridge.get("pipelineAsOfDate"),
        }))
    _note_probability_basis(ctx)
    readiness = (bridge.get("forecastReadiness") or {})
    for warning in readiness.get("warnings") or []:
        ctx.warn(str(warning))
    return findings


# --------------------------------------------------------------------------- #
# 8-9. completion_run_rate / threshold_projection
# --------------------------------------------------------------------------- #
def _extrapolation(ctx: AnalyticalContext) -> Dict[str, Any]:
    """The governed scale-up extrapolation, built at most once per request.

    Building it replays the retained weekly extracts, so it is memoised on the
    request and never built for a plan that does not need it.
    """
    def _build() -> Dict[str, Any]:
        from mi_agent_api import forecast_extrapolation as fx_mod
        from mi_agent_api import datasets as datasets_mod
        try:
            model = datasets_mod._pipeline_history(ctx.client_id)
        except Exception:  # noqa: BLE001 - history is additive, never fatal
            model = None
        try:
            return fx_mod.build_extrapolation(
                ctx.output_root, ctx.pipeline_root, ctx.client_id,
                ctx.run_id, history_model=model) or {}
        except Exception as exc:  # noqa: BLE001
            logger.warning("extrapolation unavailable: %s", exc)
            return {}

    return ctx.memoised("extrapolation", _build)


def completion_run_rate(ctx: AnalyticalContext) -> List[Finding]:
    """The governed completion run-rate (month-on-month funded growth)."""
    fx = _extrapolation(ctx)
    rr = fx.get("completionRunRateForecast") or {}
    period = _period_ref(ctx)
    total = PopulationRef(key="total", label="the whole funded book", is_total=True)
    if not rr.get("available"):
        return [Finding(
            capability="completion_run_rate", kind=KIND_FORECAST,
            label="Completion run-rate", population=total, period=period,
            status=STATUS_UNAVAILABLE,
            note=("A completion run-rate could not be derived: "
                  f"{rr.get('caveat') or 'insufficient completion history'}."))]
    return [Finding(
        capability="completion_run_rate", kind=KIND_FORECAST,
        label="Base monthly completion run-rate", metric=BALANCE,
        population=total, period=period, unit=engine.UNIT_CURRENCY,
        aggregation=engine.AGG_SUM, value=rr.get("baseMonthlyRunRate"),
        forecast_value=rr.get("baseMonthlyRunRate"),
        probability_basis="completion_run_rate_extrapolation",
        evidence={
            "engine": "mi_agent_api.forecast_extrapolation.run_rate_model",
            "annualisedRunRate": rr.get("annualisedRunRate"),
            "observedMonths": rr.get("observedMonths"),
            "scenarioMonthlyRunRate": rr.get("scenarioMonthlyRunRate"),
            "signal": "month-on-month funded growth",
            "currentFundedBalance": fx.get("currentFundedBalance"),
        })]


def threshold_projection(ctx: AnalyticalContext, *, threshold: float
                         ) -> List[Finding]:
    """The date a projected funded balance crosses ``threshold``."""
    fx = _extrapolation(ctx)
    rr = fx.get("completionRunRateForecast") or {}
    current = fx.get("currentFundedBalance")
    period = _period_ref(ctx)
    total = PopulationRef(key="total", label="the whole funded book", is_total=True)
    milestones = rr.get("milestones") or []
    match = next((m for m in milestones if m.get("threshold") == threshold), None)
    if match is None:
        above = [m for m in milestones if (m.get("threshold") or 0) >= threshold]
        match = above[0] if above else None
    if not rr.get("available") or match is None:
        return [Finding(
            capability="threshold_projection", kind=KIND_THRESHOLD,
            label=f"Date reaching {threshold:,.0f}", population=total,
            period=period, status=STATUS_UNAVAILABLE,
            note=("No governed projection reaches that threshold within the "
                  "forecast horizon."))]
    reached = bool(match.get("reached"))
    return [Finding(
        capability="threshold_projection", kind=KIND_THRESHOLD,
        label=f"Reaching {match.get('thresholdLabel') or threshold}",
        metric=BALANCE, population=total, period=period,
        unit=engine.UNIT_CURRENCY, value=current,
        forecast_value=float(match.get("threshold")),
        forecast_date=(None if reached else match.get("baseDate")),
        probability_basis="completion_run_rate_extrapolation",
        status=STATUS_OK if (reached or match.get("baseDate")) else STATUS_UNAVAILABLE,
        note=(None if (reached or match.get("baseDate")) else
              "The projection does not reach this threshold in the horizon."),
        evidence={
            "engine": "mi_agent_api.forecast_extrapolation",
            "reached": reached,
            "downsideDate": match.get("downsideDate"),
            "upsideDate": match.get("upsideDate"),
            "baseMonthlyRunRate": rr.get("baseMonthlyRunRate"),
            "observedMonths": rr.get("observedMonths"),
            "currentFundedBalance": current,
        })]


# --------------------------------------------------------------------------- #
# 10. concentration_limits
# --------------------------------------------------------------------------- #
def concentration_limits(ctx: AnalyticalContext, *, category: Optional[str] = None,
                         top_n: int = 5) -> List[Finding]:
    """Governed limits, actuals and headroom, ranked by proximity to the limit.

    Every figure comes from the single governed evaluation service
    (``compute_concentration_tests`` — the approved concentration configuration
    where one exists, the extracted Schedule 8 monitor where it does not).
    Ranking is an ordering of that service's own headroom; no limit, actual or
    headroom is calculated here.
    """
    from mi_agent_api import concentration_tests_api as conc_mod

    envelope = conc_mod.compute_concentration_tests(ctx.output_root, ctx.client_id,
                                                    ctx.run_id)
    tests = [t for t in (envelope.get("tests") or [])
             if t.get("headroom") is not None]
    if category:
        tests = [t for t in tests if t.get("category") == category]
    period = PeriodRef(end=envelope.get("reportingDate") or _period_ref(ctx).end)
    total = PopulationRef(key="total", label="the whole funded book", is_total=True)
    if not tests:
        return [Finding(
            capability="concentration_limits", kind=KIND_LIMIT,
            label="Concentration limits", population=total, period=period,
            status=STATUS_UNAVAILABLE,
            note=("No governed concentration limit with a computable headroom is "
                  "configured for this book."))]
    tests.sort(key=lambda t: float(t.get("headroom")))
    out: List[Finding] = []
    for index, test in enumerate(tests[:top_n], start=1):
        out.append(Finding(
            capability="concentration_limits", kind=KIND_LIMIT,
            label=str(test.get("displayName") or test.get("label")),
            metric=test.get("testId") or test.get("metricId"),
            population=total, period=period,
            unit=test.get("unit"), value=test.get("currentValue"),
            limit=test.get("threshold"), headroom=test.get("headroom"),
            limit_status=test.get("status"),
            rank=index, rank_of=len(tests),
            evidence={
                "engine": "mi_agent_api.concentration_tests_api.compute_concentration_tests",
                "source": envelope.get("source"),
                "configurationVersion": envelope.get("configurationVersion"),
                "operator": test.get("operator"),
                "category": test.get("category"),
                "priorValue": test.get("priorValue"),
                "reportingDate": envelope.get("reportingDate"),
            }))
    return out
