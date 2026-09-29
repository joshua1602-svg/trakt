#!/usr/bin/env python3
"""The Pipeline specialist runtime: a GovernedQueryPlan, executed by Pipeline.

WHY THIS EXISTS. Slices 1 and 2 built exactly one governed execution owner —
`generic_analysis` over canonical loan-tape fields, and its temporal variant.
Every SPECIALIST capability was refused at the perimeter by design, and the
connectivity trace measured what that costs: a real Pipeline intent compiles to

    capability=pipeline  operation=summary  population.base=pipeline
    measures=[pipeline_amount]  canonical_field=null  statistic="capability"

and stops three times over — `CAPABILITY_NOT_GENERIC`, then `spec_for_plan`
raising because there is no field to bind, then the population gate. Pipeline is
not missing. The tier that executes a specialist plan is.

WHAT THIS IS, AND WHAT IT IS NOT. It is a TRANSLATOR. It turns an
already-compiled plan into the arguments the existing Pipeline owners already
take, calls them, and writes down what ran. It computes nothing:

    balance          pipeline_contract.load_prepared_pipeline -> report
                       ["total_pipeline_amount"]
    case count       the same report ["row_count"]
    count by stage   the same report ["stage_counts"]
    amount by stage  mi_query_executor.execute_mi_query over the SAME prepared
                       frame — the owner the legacy /mi/query path already uses
    weekly history   evolution.pipeline_evolution -> ["periods"], ["byStage"]

THE GROUPED AMOUNT WAS REFUSED HERE FIRST, and that refusal was a defect of
tracing rather than of the estate. The first trace asked which of the PIPELINE
CAPABILITY's own owners produces an amount per stage, found that the preparation
report carries `stage_counts` and no per-stage amount, and concluded that only
the weekly owner breaks amount down by stage. It never asked the question that
decides it: what does the accepted legacy path do with "Show pipeline amount by
stage"? It binds Balance summed by Pipeline Stage and hands it to
`execute_mi_query` over this very frame, which carries
`current_outstanding_balance` and `pipeline_stage` as ordinary columns. The
figure existed all along, under a general owner rather than a pipeline-specific
one, and this runtime now fetches it from there.

WHICH COLUMN IS "THE PIPELINE AMOUNT" IS THE CAPABILITY'S TO SAY.
`pipeline_amount` is capability-owned in the governed vocabulary and carries no
canonical field by design, so the plan cannot name one and this runtime must not
guess. It reads `pipeline_prep.PIPELINE_AMOUNT_FIELD` — the column the Pipeline
owner's own `total_pipeline_amount` sums — which makes the stage figures and the
total the same definition by construction instead of by coincidence.

Every one of those takes DATA and never the question — which is the property
that makes plan-native execution possible at all, and the one the trace had to
establish before a line of this was written.

IT NEVER READS THE QUESTION. No parse, no recogniser, no
`workspace.resolve_dataset`, no metric inferred from text, no stage inferred
from text. The plan is authoritative, and a test asserts this module names none
of those seams.

PIPELINE TIME IS NOT FUNDED TIME. The funded temporal runtime resolves MONTHLY
governed snapshots through `SnapshotStore`; Pipeline history is WEEKLY extracts
under a different owner. This dispatches to the weekly owner and never to the
funded store — asserted, because reusing the funded store for symmetry would
silently answer a pipeline question from funded snapshots.

DELIBERATELY NARROW. One capability, explicitly. No plug-in framework, no
registry, no empty adapters for the six capabilities not migrated. When a second
specialist arrives and shows a genuinely common interface, extract it then.
"""
from __future__ import annotations

from typing import Any, Dict, FrozenSet, List, Mapping, Optional, Sequence, Tuple

CAPABILITY = "pipeline"

#: WHICH POPULATION THIS RUNTIME EXECUTES. Its own declaration, deliberately not
#: merged with the funded runtime's: one global list saying both runtimes execute
#: both populations would prove nothing, and proving nothing is how a pipeline
#: question came to be answerable from funded rows.
EXECUTABLE_POPULATIONS: FrozenSet[str] = frozenset({"pipeline"})

#: The dataset identity this runtime declares to the population gate.
EXECUTION_POPULATION = "pipeline"

# Ineligibility reasons. Stable strings: the ledger groups by them.
NOT_A_PLAN = "NOT_A_PLAN"
CAPABILITY_NOT_PIPELINE = "CAPABILITY_NOT_PIPELINE"
POPULATION_NOT_PIPELINE = "POPULATION_NOT_PIPELINE"
MEASURE_NOT_SUPPORTED = "MEASURE_NOT_SUPPORTED"
DIMENSION_NOT_SUPPORTED = "DIMENSION_NOT_SUPPORTED"
OPERATION_NOT_SUPPORTED = "OPERATION_NOT_SUPPORTED"
PERIOD_NOT_SUPPORTED = "PERIOD_NOT_SUPPORTED"
FILTERS_NOT_SUPPORTED = "FILTERS_NOT_SUPPORTED"
GEOGRAPHY_NOT_SUPPORTED = "GEOGRAPHY_NOT_SUPPORTED"
COMPARISON_NOT_SUPPORTED = "COMPARISON_NOT_SUPPORTED"
NOT_SINGLE_OUTPUT = "NOT_SINGLE_OUTPUT"
SOURCE_UNAVAILABLE = "SOURCE_UNAVAILABLE"
HISTORY_UNAVAILABLE = "HISTORY_UNAVAILABLE"
EXECUTION_FAILED = "EXECUTION_FAILED"
#: A named month the plan's label does not read as one, one that the weekly
#: history holds in more than one year, and one it does not hold at all.
PERIOD_LABEL_UNRESOLVED = "PERIOD_LABEL_UNRESOLVED"
PERIOD_LABEL_AMBIGUOUS = "PERIOD_LABEL_AMBIGUOUS"
PERIOD_NOT_AVAILABLE = "PERIOD_NOT_AVAILABLE"

#: GOVERNED MEASURE -> WHICH EXISTING PIPELINE FIGURE. The map is the whole of
#: the measure translation, and it is a map rather than a computation on purpose:
#: every value on the right is read out of a report the Pipeline owner already
#: produced. `pipeline_amount` is capability-owned and has no canonical field,
#: which is valid specialist semantics and is exactly why the generic adapter
#: could not bind it.
_AMOUNT = "amount"
_COUNT = "count"
SUPPORTED_MEASURES: Mapping[str, str] = {
    "pipeline_amount": _AMOUNT,
    "pipeline_case_count": _COUNT,
    # A bare row count. `loan` is the governed concept the compiler emits for
    # "how many cases", and on the pipeline tape a row IS a case.
    "loan": _COUNT,
    "loan_count": _COUNT,
}

#: The one dimension the existing Pipeline owners group by.
SUPPORTED_DIMENSIONS: FrozenSet[str] = frozenset({"pipeline_stage"})

#: Current-frame operations. `summary` is what a specialist capability emits for
#: "what is in the pipeline"; the other two are the ordinary scalar and grouped
#: shapes.
CURRENT_OPERATIONS: FrozenSet[str] = frozenset({"summary", "point_in_time",
                                                "breakdown"})
CURRENT_PERIOD_FORMS: FrozenSet[str] = frozenset({"current"})

#: Temporal operations, served by the WEEKLY evolution owner.
TEMPORAL_OPERATIONS: FrozenSet[str] = frozenset({"series", "breakdown"})
TEMPORAL_PERIOD_FORMS: FrozenSet[str] = frozenset({"series"})

#: DATED operations: the pipeline AT named points in time rather than now or
#: across the whole history — "October and November", "latest against the week
#: before". Served by the same weekly owner; this runtime only chooses WHICH
#: extracts, by the rule below, and never computes the change between them.
DATED_PERIOD_FORMS: FrozenSet[str] = frozenset({"explicit_period",
                                                "relative_pair"})
DATED_OPERATIONS: FrozenSet[str] = frozenset({"point_in_time", "summary",
                                              "breakdown", "series"})
DATED_GRAINS: FrozenSet[str] = frozenset({"weekly", "monthly"})

#: D7, the owner's decision (2026-09-28): a named month means the LAST weekly
#: extract dated within it. Year-aware — a bare month the history holds in two
#: years is ambiguous and clarifies, exactly as it does for funded snapshots —
#: and the chosen extract's date is stated on the receipt and the answer.
MONTH_RULE = "D7: the last weekly extract dated within the named month"

#: The grain the weekly extracts are on. Stated, not inferred, and carried into
#: the receipt so a reader can see the answer is weekly rather than monthly.
TEMPORAL_GRAIN = "weekly"
TEMPORAL_BASIS = "governed_weekly_pipeline_extracts"


# --------------------------------------------------------------------------- #
# reading the plan
# --------------------------------------------------------------------------- #

def _as_mapping(plan: Any) -> Mapping[str, Any]:
    if isinstance(plan, Mapping):
        return plan
    to_dict = getattr(plan, "to_dict", None)
    return to_dict() if callable(to_dict) else {}


def claims(plan: Any) -> bool:
    """Is this plan THIS runtime's? The capability, and nothing else.

    Read off `GovernedQueryPlan.capability`. Not the question, not the dataset,
    not a word in a sentence — which is what makes the dispatch a governed
    decision rather than a second router.
    """
    return str(_as_mapping(plan).get("capability") or "") == CAPABILITY


def _single_output(plan: Mapping[str, Any]) -> Optional[Mapping[str, Any]]:
    outputs = plan.get("outputs") or ()
    return outputs[0] if len(outputs) == 1 else None


def requested_measures(plan: Any) -> List[str]:
    output = _single_output(_as_mapping(plan)) or {}
    return [str(m.get("concept") or "") for m in (output.get("measures") or ())]


def requested_dimensions(plan: Any) -> List[str]:
    """The governed FIELDS this plan groups by.

    A plan's dimensions are BINDINGS, not strings — `{"concept": …,
    "canonical_field": …}` — and `canonical_field` is the half the executor and
    the receipt both speak. Read the same way `plan_runtime_adapter.spec_for_plan`
    reads them, because a second reading of the same slot is how a grouping comes
    to be requested under one name and proved under another.
    """
    output = _single_output(_as_mapping(plan)) or {}
    return [str(d.get("canonical_field") or d.get("concept") or "")
            for d in (output.get("dimensions") or ())]


def is_dated(plan: Any) -> bool:
    """Does this plan ask for the pipeline AT named points in time?"""
    period = _as_mapping(plan).get("period") or {}
    return str(period.get("form") or "") in DATED_PERIOD_FORMS


def is_temporal(plan: Any) -> bool:
    """Does this plan ask for the WEEKLY history rather than the current extract?

    From the plan's own period form. `plan_temporal_runtime.claims` is
    deliberately not consulted: that owner speaks for the FUNDED monthly
    snapshots, and asking it whether a pipeline plan is temporal would make the
    funded store's contract decide a pipeline question.
    """
    period = _as_mapping(plan).get("period") or {}
    return str(period.get("form") or "") in TEMPORAL_PERIOD_FORMS


# --------------------------------------------------------------------------- #
# the perimeter
# --------------------------------------------------------------------------- #

def check_eligibility(plan: Any) -> Tuple[bool, str, str]:
    """`(eligible, reason, detail)`. Reads the plan only, and refuses by default.

    Every shape this runtime does not already have an owner for is refused
    rather than approximated. In particular FILTERS are refused outright: the
    balance is read from the Pipeline owner's own report and there is no way to
    narrow it without recomputing the figure here, which would be a second
    implementation of a calculation somebody else owns.
    """
    body = _as_mapping(plan)
    if not body:
        return False, NOT_A_PLAN, "no plan was supplied"
    if not claims(body):
        return (False, CAPABILITY_NOT_PIPELINE,
                f"capability={body.get('capability')!r} is not {CAPABILITY!r}; "
                f"this runtime executes one capability and refuses the rest")

    population = body.get("population") or {}
    base = str(population.get("base") or "")
    if base not in EXECUTABLE_POPULATIONS:
        return (False, POPULATION_NOT_PIPELINE,
                f"population.base={base!r} is not executed by the pipeline "
                f"runtime (it executes {sorted(EXECUTABLE_POPULATIONS)})")

    output = _single_output(body)
    if output is None:
        return (False, NOT_SINGLE_OUTPUT,
                "the pipeline runtime serves exactly one output")

    if body.get("filters") or output.get("filters"):
        return (False, FILTERS_NOT_SUPPORTED,
                "the pipeline figures are read from the Pipeline owner's own "
                "report, which cannot be narrowed without recomputing them "
                "here; a filtered pipeline question is refused rather than "
                "answered by a second implementation")
    if output.get("geography") or (body.get("geography") or {}).get("requested"):
        return (False, GEOGRAPHY_NOT_SUPPORTED,
                "the pipeline owners expose no geography breakdown")
    if str(body.get("comparison_kind") or "none") != "none":
        return (False, COMPARISON_NOT_SUPPORTED,
                "the pipeline runtime serves no population comparison")

    measures = [str(m.get("concept") or "") for m in (output.get("measures") or ())]
    if len(measures) != 1:
        return (False, MEASURE_NOT_SUPPORTED,
                f"exactly one measure is served; this plan names {measures}")
    if measures[0] not in SUPPORTED_MEASURES:
        return (False, MEASURE_NOT_SUPPORTED,
                f"measure {measures[0]!r} has no existing pipeline owner "
                f"(served: {sorted(SUPPORTED_MEASURES)})")

    dimensions = requested_dimensions(body)
    if len(dimensions) > 1:
        return (False, DIMENSION_NOT_SUPPORTED,
                f"the pipeline owners group by one dimension; this plan names "
                f"{dimensions}")
    if dimensions and dimensions[0] not in SUPPORTED_DIMENSIONS:
        return (False, DIMENSION_NOT_SUPPORTED,
                f"dimension {dimensions[0]!r} is not a governed pipeline "
                f"grouping (served: {sorted(SUPPORTED_DIMENSIONS)})")

    operation = str(body.get("operation") or "")
    period = body.get("period") or {}
    form = str(period.get("form") or "")

    if is_temporal(body):
        if operation not in TEMPORAL_OPERATIONS:
            return (False, OPERATION_NOT_SUPPORTED,
                    f"operation={operation!r} has no weekly pipeline owner")
        return True, "", ""

    if is_dated(body):
        if operation not in DATED_OPERATIONS:
            return (False, OPERATION_NOT_SUPPORTED,
                    f"operation={operation!r} has no dated pipeline owner")
        if form == "explicit_period" and not (period.get("labels") or ()):
            return (False, PERIOD_NOT_SUPPORTED,
                    "an explicit period with no label names no point in time")
        if form == "relative_pair":
            grain = str(period.get("grain") or "")
            back = period.get("periods_back")
            if grain not in DATED_GRAINS:
                return (False, PERIOD_NOT_SUPPORTED,
                        f"a relative pair needs a weekly or monthly grain to "
                        f"say what 'previous' means; this plan states "
                        f"{grain or 'none'!r}")
            if isinstance(back, bool) or not isinstance(back, int) or back < 1:
                return (False, PERIOD_NOT_SUPPORTED,
                        f"a relative pair needs a positive distance; this plan "
                        f"states periods_back={back!r}")
        return True, "", ""

    if form not in CURRENT_PERIOD_FORMS:
        return (False, PERIOD_NOT_SUPPORTED,
                f"period.form={form!r} is neither the current extract nor the "
                f"weekly series; the pipeline estate offers no other history")
    if operation not in CURRENT_OPERATIONS:
        return (False, OPERATION_NOT_SUPPORTED,
                f"operation={operation!r} has no current pipeline owner")
    return True, "", ""


# --------------------------------------------------------------------------- #
# execution — the existing owners, called with what the plan says
# --------------------------------------------------------------------------- #

class PipelineOutcome:
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


#: WHICH LOWER-LEVEL OWNER PRODUCED THE FIGURES. Named in the receipt because
#: this runtime dispatches to more than one, and "the pipeline runtime answered"
#: does not distinguish a total read off a preparation report from a grouped sum
#: the generic executor computed over the same frame. A reader reconciling an
#: answer needs to know which.
OWNER_REPORT = "pipeline_contract.load_prepared_pipeline"
OWNER_EVOLUTION = "evolution.pipeline_evolution"
OWNER_GENERIC_EXECUTOR = "mi_query_executor.execute_mi_query"
#: The function that fills the Pipeline tab's amount and case-count tiles.
OWNER_OPEN_TOTALS = "pipeline_contract.open_totals"


def _receipt(plan: Mapping[str, Any], *, measure: str, kind: str,
             dimensions: Sequence[str], dataset: Mapping[str, Any],
             result_shape: str, owner: str,
             periods: Optional[Sequence[str]] = None
             ) -> Dict[str, Any]:
    """The governed execution receipt for one pipeline execution.

    EVERY FIELD IS AN EXECUTION FACT. The dataset identity is the file the
    Pipeline owner actually read; the periods are the extract dates it actually
    selected; the measure and dimensions are what it was asked for and what it
    grouped by. Nothing here is reconstructed from the question, and nothing is
    asserted that the owner did not report.
    """
    row: Dict[str, Any] = {
        "capability": CAPABILITY,
        "population_base": EXECUTION_POPULATION,
        "dataset": dict(dataset),
        "measure_concept": measure,
        "measure_kind": kind,
        "group_field_keys": [str(d) for d in dimensions],
        # The pipeline runtime refuses filtered plans, so an empty list here is
        # a statement rather than an omission: no predicate was requested and
        # none was applied.
        "applied_predicates": [],
        "result_shape": result_shape,
        "execution_owner": owner,
    }
    if periods is not None:
        row["temporal_basis"] = TEMPORAL_BASIS
        row["grain"] = TEMPORAL_GRAIN
        row["selected_periods"] = [str(p) for p in periods]
    return row


def _execute_current_grouped_amount(plan: Any, *, body: Mapping[str, Any],
                                    frame: Any, semantics: Any, measure: str,
                                    kind: str, dimensions: Sequence[str],
                                    dataset: Mapping[str, Any]) -> PipelineOutcome:
    """A CURRENT pipeline amount, grouped, through the estate's existing engine.

    THE OWNER IS `execute_mi_query`, and it is the owner the legacy /mi/query
    path already uses for exactly this question: the prepared pipeline frame
    carries `current_outstanding_balance` and `pipeline_stage` as ordinary
    columns, and "Show pipeline amount by stage" has been answered as Balance
    summed by Pipeline Stage since before this runtime existed. Nothing new is
    computed; a figure that used to be refused here is now fetched from where it
    already lived.

    THE SPEC IS BUILT FROM THE PLAN, by `adapter.spec_for_plan` — the same
    translator the funded runtime uses, so the axes, the predicates and the
    rendering rules have one owner and not two. The only thing passed in is the
    MEASURE BINDING, because `pipeline_amount` is capability-owned: the governed
    vocabulary gives it no canonical field on purpose, and the capability is the
    thing entitled to say which column its arithmetic runs over. That column is
    `pipeline_prep.PIPELINE_AMOUNT_FIELD` — the one the Pipeline owner's own
    `total_pipeline_amount` sums — so the grouped figures and the total come
    from the same definition by construction rather than by coincidence.

    NO QUESTION IS READ, no groupby is written here, and no measure owner is
    created.
    """
    from mi_agent import plan_runtime_adapter as adapter
    from mi_agent.mi_query_executor import execute_mi_query
    from mi_agent.query_plan import SUM
    from mi_agent_api.pipeline_prep import PIPELINE_AMOUNT_FIELD

    if semantics is None:
        return PipelineOutcome(
            ok=False, reason=SOURCE_UNAVAILABLE,
            detail="no governed field registry was supplied; a grouped pipeline "
                   "amount is executed by the generic engine, which requires one")
    if frame is None or PIPELINE_AMOUNT_FIELD not in getattr(frame, "columns", ()):
        return PipelineOutcome(
            ok=False, reason=EXECUTION_FAILED,
            detail=f"the prepared pipeline frame carries no "
                   f"{PIPELINE_AMOUNT_FIELD!r} column")

    try:
        spec = adapter.spec_for_plan(
            plan, measure_binding=(PIPELINE_AMOUNT_FIELD, SUM))
        result = execute_mi_query(spec, frame, semantics)
    except Exception as exc:                                         # noqa: BLE001
        return PipelineOutcome(ok=False, reason=EXECUTION_FAILED,
                               detail=f"{type(exc).__name__}: {exc}"[:200])

    # THE EXECUTOR IS ASKED WHETHER IT DID WHAT THE SPEC SAID, rather than
    # assumed to have. `reconcile_receipt` is the funded runtime's own structural
    # check and it reads the executor's metadata, so a grouping silently dropped
    # or a predicate silently skipped fails here instead of being rendered.
    ok, why_not = adapter.reconcile_receipt(spec, result)
    if not ok:
        return PipelineOutcome(ok=False, reason=EXECUTION_FAILED,
                               detail=f"executor receipt did not reconcile: "
                                      f"{why_not}"[:200])

    column = adapter.value_column(spec)
    data = getattr(result, "data", None)
    axis = dimensions[0]
    if data is None or column not in getattr(data, "columns", ()) \
            or axis not in getattr(data, "columns", ()):
        return PipelineOutcome(
            ok=False, reason=EXECUTION_FAILED,
            detail=f"the executor returned no {column!r} by {axis!r}")

    cells = [{axis: str(row[axis]),
              "value": (float(row[column]) if row[column] is not None else None)}
             for _, row in data.iterrows()]

    metadata = dict(getattr(result, "metadata", None) or {})
    receipt = _receipt(body, measure=measure, kind=kind, dimensions=dimensions,
                       dataset=dataset, result_shape="grouped",
                       owner=OWNER_GENERIC_EXECUTOR)
    # WHAT THE ENGINE REPORTS IT DID, beside what it was asked for. The keys are
    # the generic receipt's own, so the coverage ledger reads a specialist
    # grouped answer with the vocabulary it already has.
    receipt.update({
        "measure_field": spec.metric,
        "aggregation": metadata.get("aggregation") or spec.aggregation,
        "group_field_keys": list(metadata.get("group_field_keys") or dimensions),
        "applied_predicates": list(metadata.get("applied_predicates") or ()),
        "input_row_count": metadata.get("input_row_count"),
        "filtered_row_count": metadata.get("filtered_row_count"),
    })
    return PipelineOutcome(ok=True, cells=cells, value=None, receipt=receipt)


def execute_current(plan: Any, *, source: Any,
                    semantics: Any = None,
                    history_model: Optional[Mapping[str, Any]] = None
                    ) -> PipelineOutcome:
    """The CURRENT pipeline extract, through the owner that already prepares it.

    `source` is the governed discovery scope the caller resolved — this runtime
    does not discover, does not read a root, and does not choose a client.

    `semantics` is the governed field registry the production request already
    loaded. It is needed only for a GROUPED AMOUNT, which the generic executor
    owns; a scalar or a count is read off the preparation report and needs none.
    """
    from mi_agent_api import pipeline_contract as pipeline_mod

    body = _as_mapping(plan)
    output = _single_output(body) or {}
    measure = str((output.get("measures") or [{}])[0].get("concept") or "")
    kind = SUPPORTED_MEASURES[measure]
    dimensions = requested_dimensions(body)

    if not source:
        return PipelineOutcome(ok=False, reason=SOURCE_UNAVAILABLE,
                               detail="no governed pipeline source was supplied "
                                      "for this request")
    try:
        frame, report = pipeline_mod.load_prepared_pipeline(
            source, historical_model=history_model)
    except Exception as exc:                                         # noqa: BLE001
        return PipelineOutcome(ok=False, reason=EXECUTION_FAILED,
                               detail=f"{type(exc).__name__}: {exc}"[:200])

    # THE LIVE PIPELINE, by the dashboard's own rule. Completed and withdrawn
    # cases stay in ERE's weekly extract with their balance; the Pipeline tab
    # counts KFI / Application / Offer only and discloses the rest, and so does
    # every figure below. Governed pipeline plans carry no stage filter (they
    # are refused above), so no plan here names a stage itself.
    live, live_scope = pipeline_mod.live_pipeline_scope(frame)
    scope = source if isinstance(source, Mapping) else {}
    dataset = {
        "identity": "governed_pipeline_extract",
        "source_file": str(scope.get("source_file") or ""),
        "as_of_date": str(scope.get("pipeline_as_of_date")
                          or report.get("pipeline_as_of_date") or ""),
        "row_count": int(len(live)),
        "extract_row_count": int(report.get("row_count", len(frame))),
    }

    if dimensions and kind == _AMOUNT:
        # A CURRENT AMOUNT BY STAGE, through the GENERIC DETERMINISTIC ENGINE.
        #
        # This runtime first refused it, on the reading that the preparation
        # report carries `stage_counts` and no per-stage amount and that the
        # only per-stage amounts in the estate were the weekly ones. That trace
        # was INCOMPLETE and the refusal was wrong: it looked only at the
        # pipeline CAPABILITY's owners and never at the owner the legacy
        # /mi/query path actually uses, which is `execute_mi_query` running over
        # this same prepared pipeline frame. "Show pipeline amount by stage" has
        # been answered as Balance summed by Pipeline Stage all along.
        #
        # So the figures come from the engine that already produces them, not
        # from a groupby written here — there is no `sum`, no `groupby` and no
        # second arithmetic in this module. The measure binding is the one thing
        # the plan cannot carry, because `pipeline_amount` is capability-owned
        # and has no canonical field by design; it is supplied from
        # `pipeline_prep.PIPELINE_AMOUNT_FIELD`, which is the column the Pipeline
        # owner's own `total_pipeline_amount` sums. The capability states the
        # field; the engine does the arithmetic; neither is invented here.
        outcome = _execute_current_grouped_amount(
            plan, body=body, frame=live, semantics=semantics,
            measure=measure, kind=kind, dimensions=dimensions, dataset=dataset)
        if outcome.ok:
            outcome.receipt["pipeline_scope"] = _noted(live_scope)
        return outcome

    if dimensions:
        # COUNTS BY STAGE, read off the preparation report. `stage_counts` is
        # the Pipeline owner's own grouping; nothing is grouped here.
        counts = report.get("stage_counts") or {}
        if not isinstance(counts, Mapping) or not counts:
            return PipelineOutcome(
                ok=False, reason=EXECUTION_FAILED,
                detail="the pipeline report carried no stage counts")
        # The live stages only, as the Pipeline tab's stage breakdown shows
        # them; the closed and unmapped stages are disclosed, not charted.
        cells = [{dimensions[0]: str(stage), "value": float(count)}
                 for stage, count in sorted(counts.items())
                 if str(stage).strip().upper() in live_scope["open_stages"]]
        receipt = _receipt(body, measure=measure, kind=kind,
                           dimensions=dimensions, dataset=dataset,
                           result_shape="grouped", owner=OWNER_REPORT)
        receipt["pipeline_scope"] = _noted(live_scope)
        return PipelineOutcome(ok=True, cells=cells, value=None,
                               receipt=receipt)

    # THE PIPELINE TAB'S OWN TOTALS. `open_totals` is the function that fills
    # the tab's `pipelineAmount` and `pipelineRowCount` tiles, called on the
    # same prepared frame, so the answer and the tile are one computation.
    totals = pipeline_mod.open_totals(frame)
    value = totals["amount"] if kind == _AMOUNT else totals["cases"]
    if value is None:
        return PipelineOutcome(ok=False, reason=EXECUTION_FAILED,
                               detail=f"the pipeline report carried no "
                                      f"{measure!r} figure")
    receipt = _receipt(body, measure=measure, kind=kind, dimensions=[],
                       dataset=dataset, result_shape="scalar",
                       owner=OWNER_OPEN_TOTALS)
    receipt["pipeline_scope"] = _noted(live_scope)
    return PipelineOutcome(ok=True, value=float(value), cells=None,
                           receipt=receipt)


#: The weekly series' totals are already the live pipeline: `pipeline_evolution`
#: sums the open stages of each extract (the Pipeline tab's weekly chart reads
#: the same figures). Stated on the receipt so the answer can say so.
def _noted(scope: Mapping[str, Any]) -> Dict[str, Any]:
    """The scope with the sentence the answer prints about it — written by the
    Pipeline owner, so both answer paths word the exclusion alike."""
    from mi_agent_api.pipeline_contract import live_pipeline_note

    row = dict(scope)
    row["note"] = live_pipeline_note(row)
    return row


def _series_scope() -> Dict[str, Any]:
    from mi_agent_api.pipeline_prep import OPEN_STAGES

    return {"population": "open", "basis": OWNER_EVOLUTION,
            "open_stages": list(OPEN_STAGES)}


def _live_stage_rows(rows: Sequence[Mapping[str, Any]]
                     ) -> Tuple[List[Mapping[str, Any]], Dict[str, Any]]:
    """The live stages of a weekly stage breakdown, and which were left out.

    `pipeline_evolution` publishes EVERY stage per extract, because the funnel
    and conversion views need completions and withdrawals. A pipeline answer is
    the live pipeline, so the closed and unmapped stages are dropped here — by
    the Pipeline owner's own `OPEN_STAGES` — and named, never silently lost.
    """
    from mi_agent_api.pipeline_prep import OPEN_STAGES

    kept = [r for r in rows
            if str(r.get("stage") or "").strip().upper() in OPEN_STAGES]
    left = sorted({str(r.get("stage") or "") for r in rows} -
                  {str(r.get("stage") or "") for r in kept})
    return kept, {"population": "open", "basis": OWNER_EVOLUTION,
                  "open_stages": list(OPEN_STAGES), "excluded_stages": left}


def execute_temporal(plan: Any, *, root: Any, client_id: str,
                     to_run_id: Optional[str] = None,
                     history_model: Optional[Mapping[str, Any]] = None
                     ) -> PipelineOutcome:
    """The WEEKLY pipeline history, through the evolution owner that builds it.

    Never the funded `SnapshotStore`. Pipeline history is weekly extracts and
    funded history is monthly snapshots; routing one through the other would
    answer a pipeline question from a funded catalogue, which is the single
    worst outcome available here.
    """
    from mi_agent_api import evolution as evolution_mod

    body = _as_mapping(plan)
    output = _single_output(body) or {}
    measure = str((output.get("measures") or [{}])[0].get("concept") or "")
    kind = SUPPORTED_MEASURES[measure]
    dimensions = requested_dimensions(body)

    if not root or not client_id:
        return PipelineOutcome(
            ok=False, reason=HISTORY_UNAVAILABLE,
            detail="no governed weekly pipeline history was supplied for this "
                   "request")
    try:
        series = evolution_mod.pipeline_evolution(
            root, client_id, to_run_id, historical_model=history_model)
    except Exception as exc:                                         # noqa: BLE001
        return PipelineOutcome(ok=False, reason=EXECUTION_FAILED,
                               detail=f"{type(exc).__name__}: {exc}"[:200])

    periods = list(series.get("periods") or ())
    by_stage = list(series.get("byStage") or ())
    if not periods:
        return PipelineOutcome(
            ok=False, reason=HISTORY_UNAVAILABLE,
            detail="the governed weekly pipeline history is empty for this "
                   "client")
    weeks = [str(p.get("week") or p.get("extract_date") or "") for p in periods]
    dataset = {
        "identity": "governed_weekly_pipeline_extracts",
        "extracts_used": int(series.get("uniqueWeeklyExtractsUsed") or len(periods)),
        "source_files": [str(f) for f in (series.get("sourceFiles") or ())],
    }

    if dimensions:
        if not by_stage:
            return PipelineOutcome(
                ok=False, reason=HISTORY_UNAVAILABLE,
                detail="no weekly pipeline extracts carry a stage breakdown")
        key = "value" if kind == _AMOUNT else "count"
        live_rows, live_scope = _live_stage_rows(by_stage)
        cells = [{"period": str(r.get("period") or ""),
                  dimensions[0]: str(r.get("stage") or ""),
                  "value": (float(r.get(key)) if r.get(key) is not None else None)}
                 for r in live_rows]
        receipt = _receipt(body, measure=measure, kind=kind,
                           dimensions=dimensions, dataset=dataset,
                           result_shape="grouped_series", owner=OWNER_EVOLUTION,
                           periods=sorted({c["period"] for c in cells}))
        receipt["pipeline_scope"] = _noted(live_scope)
        return PipelineOutcome(ok=True, cells=cells, value=None, receipt=receipt)

    metric = "pipeline_amount" if kind == _AMOUNT else "pipeline_case_count"
    cells = [{"period": str(p.get("week") or p.get("extract_date") or ""),
              "value": ((float((p.get("metrics") or {}).get(metric))
                         if (p.get("metrics") or {}).get(metric) is not None
                         else None))}
             for p in periods]
    receipt = _receipt(body, measure=measure, kind=kind, dimensions=[],
                       dataset=dataset, result_shape="series",
                       owner=OWNER_EVOLUTION, periods=weeks)
    receipt["pipeline_scope"] = _noted(_series_scope())
    return PipelineOutcome(ok=True, cells=cells, value=None, receipt=receipt)


# --------------------------------------------------------------------------- #
# dated — the pipeline AT named points in time
# --------------------------------------------------------------------------- #

def _month_of(extract_date: Any) -> Optional[Tuple[int, int]]:
    text = str(extract_date or "")
    try:
        return int(text[0:4]), int(text[5:7])
    except ValueError:
        return None


def last_extract_in_month(dates: Sequence[str], *, month: int,
                          year: Optional[int]) -> Tuple[Optional[str], str, str]:
    """D7 for one named month: `(extract_date, reason, detail)`.

    The last weekly extract dated within the month. A month with no year that
    the history holds in more than one year is AMBIGUOUS — choosing the latest
    would answer for a year the reader did not name, which is the defect the
    legacy owner (`temporal_compare._match_period`) carries.
    """
    in_month = [d for d in dates
                if (_month_of(d) or (0, 0))[1] == month
                and (year is None or (_month_of(d) or (0, 0))[0] == year)]
    if not in_month:
        return (None, PERIOD_NOT_AVAILABLE,
                f"no weekly extract falls in month {month}"
                + (f" of {year}" if year else "")
                + f"; the history runs {min(dates) if dates else '-'} to "
                  f"{max(dates) if dates else '-'}")
    years = sorted({(_month_of(d) or (0, 0))[0] for d in in_month})
    if year is None and len(years) > 1:
        return (None, PERIOD_LABEL_AMBIGUOUS,
                f"month {month} occurs in {years}; name the year")
    return max(in_month), "", ""


def _dated_selection(plan: Mapping[str, Any], dates: Sequence[str]
                     ) -> Tuple[List[Dict[str, str]], str, str]:
    """Which extracts a dated plan names: `(resolution rows, reason, detail)`.

    A SELECTION of dates, by the owner's rule, and nothing else — every figure
    still comes from the weekly owner for exactly those extracts.
    """
    from mi_agent.period_labels import parse_anchor

    period = plan.get("period") or {}
    ordered = sorted(d for d in dates if d)
    rows: List[Dict[str, str]] = []
    if str(period.get("form") or "") == "explicit_period":
        seen = set()
        for label in period.get("labels") or ():
            anchor = parse_anchor(label)
            if anchor is None:
                return ([], PERIOD_LABEL_UNRESOLVED,
                        f"{label!r} does not name a month this contract reads")
            key = (anchor.month, anchor.year)
            if key in seen:
                continue
            seen.add(key)
            chosen, why, detail = last_extract_in_month(
                ordered, month=anchor.month, year=anchor.year)
            if chosen is None:
                return [], why, f"{label!r}: {detail}"
            rows.append({"requested": str(label), "extract_date": chosen,
                         "rule": MONTH_RULE})
        return rows, "", ""

    grain = str(period.get("grain") or "")
    back = int(period.get("periods_back"))
    if grain == "weekly":
        if len(ordered) <= back:
            return ([], PERIOD_NOT_AVAILABLE,
                    f"{back} week(s) back needs {back + 1} weekly extracts; the "
                    f"history holds {len(ordered)}")
        rule = (f"the latest weekly extract, and the one {back} extract(s) "
                f"before it")
        return ([{"requested": f"{back} week(s) before the latest",
                  "extract_date": ordered[-1 - back], "rule": rule},
                 {"requested": "latest", "extract_date": ordered[-1],
                  "rule": rule}], "", "")
    latest = _month_of(ordered[-1]) if ordered else None
    if latest is None:
        return [], PERIOD_NOT_AVAILABLE, "the weekly history is empty"
    index = latest[0] * 12 + (latest[1] - 1) - back
    earlier = (index // 12, index % 12 + 1)
    out = []
    for requested, (year, month) in ((f"{back} month(s) before the latest",
                                      earlier), ("latest month", latest)):
        chosen, why, detail = last_extract_in_month(ordered, month=month,
                                                    year=year)
        if chosen is None:
            return [], why, detail
        out.append({"requested": requested, "extract_date": chosen,
                    "rule": MONTH_RULE})
    return out, "", ""


def execute_dated(plan: Any, *, root: Any, client_id: str,
                  to_run_id: Optional[str] = None,
                  history_model: Optional[Mapping[str, Any]] = None
                  ) -> PipelineOutcome:
    """The pipeline at named points in time, through the weekly owner.

    `evolution.pipeline_evolution` — the owner the weekly series already comes
    from — supplies every figure; this chooses which of its extracts the plan
    named (D7 for a month, the extract order for "latest against previous") and
    returns their figures as they are. The CHANGE between them is not computed
    here: the plan asked for the pipeline at those dates, and a movement is a
    different operation with a different owner.
    """
    from mi_agent_api import evolution as evolution_mod

    body = _as_mapping(plan)
    output = _single_output(body) or {}
    measure = str((output.get("measures") or [{}])[0].get("concept") or "")
    kind = SUPPORTED_MEASURES[measure]
    dimensions = requested_dimensions(body)

    if not root or not client_id:
        return PipelineOutcome(
            ok=False, reason=HISTORY_UNAVAILABLE,
            detail="no governed weekly pipeline history was supplied for this "
                   "request")
    try:
        series = evolution_mod.pipeline_evolution(
            root, client_id, to_run_id, historical_model=history_model)
    except Exception as exc:                                         # noqa: BLE001
        return PipelineOutcome(ok=False, reason=EXECUTION_FAILED,
                               detail=f"{type(exc).__name__}: {exc}"[:200])

    periods = list(series.get("periods") or ())
    by_date = {str(p.get("extract_date") or ""): p for p in periods}
    resolution, why, detail = _dated_selection(body, list(by_date))
    if why:
        return PipelineOutcome(ok=False, reason=why, detail=detail[:300])
    chosen = [row["extract_date"] for row in resolution]
    dataset = {
        "identity": "governed_weekly_pipeline_extracts",
        "extracts_used": len(chosen),
        "source_files": [str(by_date[d].get("source_file") or "") for d in chosen],
    }

    if dimensions:
        key = "value" if kind == _AMOUNT else "count"
        rows, live_scope = _live_stage_rows(
            [r for r in (series.get("byStage") or ())
             if str(r.get("period") or "") in set(chosen)])
        if not rows:
            return PipelineOutcome(
                ok=False, reason=HISTORY_UNAVAILABLE,
                detail="the chosen weekly extracts carry no stage breakdown")
        cells = [{"period": str(r.get("period") or ""),
                  dimensions[0]: str(r.get("stage") or ""),
                  "value": (float(r.get(key)) if r.get(key) is not None else None)}
                 for r in rows]
        shape = "grouped_dated"
    else:
        live_scope = _series_scope()
        metric = "pipeline_amount" if kind == _AMOUNT else "pipeline_case_count"
        cells = [{"period": d,
                  "value": ((float((by_date[d].get("metrics") or {}).get(metric))
                             if (by_date[d].get("metrics") or {}).get(metric)
                             is not None else None))}
                 for d in chosen]
        shape = "dated"
    receipt = _receipt(body, measure=measure, kind=kind, dimensions=dimensions,
                       dataset=dataset, result_shape=shape,
                       owner=OWNER_EVOLUTION, periods=chosen)
    receipt["period_resolution"] = resolution
    receipt["pipeline_scope"] = _noted(live_scope)
    return PipelineOutcome(ok=True, cells=cells, value=None, receipt=receipt)

