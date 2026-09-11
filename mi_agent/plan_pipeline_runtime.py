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

    balance        pipeline_contract.load_prepared_pipeline -> report
                     ["total_pipeline_amount"]
    case count       the same report ["row_count"]
    by stage         the same report ["stage_counts"]
    weekly history   evolution.pipeline_evolution -> ["periods"], ["byStage"]

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

    if form not in CURRENT_PERIOD_FORMS:
        return (False, PERIOD_NOT_SUPPORTED,
                f"period.form={form!r} is neither the current extract nor the "
                f"weekly series; the pipeline estate offers no other history")
    if operation not in CURRENT_OPERATIONS:
        return (False, OPERATION_NOT_SUPPORTED,
                f"operation={operation!r} has no current pipeline owner")
    # A CURRENT grouped AMOUNT has no existing owner: the preparation report
    # carries `stage_counts` and no per-stage amount, and the only per-stage
    # amounts in the estate are the WEEKLY ones the evolution owner builds.
    # Refused rather than summed here.
    if dimensions and SUPPORTED_MEASURES[measures[0]] == _AMOUNT:
        return (False, MEASURE_NOT_SUPPORTED,
                "the current pipeline report carries counts by stage and no "
                "amount by stage; only the weekly owner breaks amount down by "
                "stage, so this is refused rather than recomputed")
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


def _receipt(plan: Mapping[str, Any], *, measure: str, kind: str,
             dimensions: Sequence[str], dataset: Mapping[str, Any],
             result_shape: str, periods: Optional[Sequence[str]] = None
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
    }
    if periods is not None:
        row["temporal_basis"] = TEMPORAL_BASIS
        row["grain"] = TEMPORAL_GRAIN
        row["selected_periods"] = [str(p) for p in periods]
    return row


def execute_current(plan: Any, *, source: Any,
                    history_model: Optional[Mapping[str, Any]] = None
                    ) -> PipelineOutcome:
    """The CURRENT pipeline extract, through the owner that already prepares it.

    `source` is the governed discovery scope the caller resolved — this runtime
    does not discover, does not read a root, and does not choose a client.
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

    scope = source if isinstance(source, Mapping) else {}
    dataset = {
        "identity": "governed_pipeline_extract",
        "source_file": str(scope.get("source_file") or ""),
        "as_of_date": str(scope.get("pipeline_as_of_date")
                          or report.get("pipeline_as_of_date") or ""),
        "row_count": int(report.get("row_count", len(frame))),
    }

    if dimensions:
        # COUNTS BY STAGE, read off the preparation report. `stage_counts` is
        # the Pipeline owner's own grouping; nothing is grouped here.
        counts = report.get("stage_counts") or {}
        if not isinstance(counts, Mapping) or not counts:
            return PipelineOutcome(
                ok=False, reason=EXECUTION_FAILED,
                detail="the pipeline report carried no stage counts")
        cells = [{dimensions[0]: str(stage), "value": float(count)}
                 for stage, count in sorted(counts.items())]
        return PipelineOutcome(
            ok=True, cells=cells, value=None,
            receipt=_receipt(body, measure=measure, kind=kind,
                             dimensions=dimensions, dataset=dataset,
                             result_shape="grouped"))

    if kind == _AMOUNT:
        value = report.get("total_pipeline_amount")
    else:
        value = report.get("row_count", len(frame))
    if value is None:
        return PipelineOutcome(ok=False, reason=EXECUTION_FAILED,
                               detail=f"the pipeline report carried no "
                                      f"{measure!r} figure")
    return PipelineOutcome(
        ok=True, value=float(value), cells=None,
        receipt=_receipt(body, measure=measure, kind=kind, dimensions=[],
                         dataset=dataset, result_shape="scalar"))


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
        cells = [{"period": str(r.get("period") or ""),
                  dimensions[0]: str(r.get("stage") or ""),
                  "value": (float(r.get(key)) if r.get(key) is not None else None)}
                 for r in by_stage]
        return PipelineOutcome(
            ok=True, cells=cells, value=None,
            receipt=_receipt(body, measure=measure, kind=kind,
                             dimensions=dimensions, dataset=dataset,
                             result_shape="grouped_series",
                             periods=sorted({c["period"] for c in cells})))

    metric = "pipeline_amount" if kind == _AMOUNT else "pipeline_case_count"
    cells = [{"period": str(p.get("week") or p.get("extract_date") or ""),
              "value": ((float((p.get("metrics") or {}).get(metric))
                         if (p.get("metrics") or {}).get(metric) is not None
                         else None))}
             for p in periods]
    return PipelineOutcome(
        ok=True, cells=cells, value=None,
        receipt=_receipt(body, measure=measure, kind=kind, dimensions=[],
                         dataset=dataset, result_shape="series",
                         periods=weeks))
