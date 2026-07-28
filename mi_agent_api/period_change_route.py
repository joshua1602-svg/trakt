"""mi_agent_api/period_change_route — the Period Change Analysis adapter.

The seam between the governed workflow (``mi_agent.period_change``, which knows
nothing about datasets, HTTP or presentation) and the platform (which knows
nothing about business semantics). It does three things and nothing else:

1. **supplies snapshots.** ``evolution.funded_frames`` is the EXISTING governed
   per-period frame service — the same one the Evolution tab, the funded bridge,
   the temporal-compare route and the movement summary already read. Nothing new
   walks storage, and the resolved portfolio lens is applied with the same
   ``chat_routing._apply_lens_filter`` those routes use, so a period-change
   answer covers exactly the portfolios every other routed answer would;
2. **runs the workflow** and maps a controlled ``PeriodChangeFailure`` onto the
   repository's existing error taxonomy (``trakt_core.errors.ErrorCode``);
3. **renders** the structured result through the existing chat envelope and
   artifact builders. The full governed result travels intact under the
   additive ``periodChange`` key, so a channel that wants the structure gets the
   structure and one that wants a table gets a table — from the same numbers.

The adapter never calculates. Every figure it renders was computed by the
workflow; if a number is not in the result, it is not in the answer.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Sequence, Tuple

from trakt_core.errors import ErrorCode

from mi_agent.period_change import (
    PeriodChangeRequest,
    recognise,
    run_period_change_analysis,
)
from mi_agent.period_change.models import (
    BRIDGE_STATUS_AVAILABLE,
    FAIL_AMBIGUOUS_PERIOD_RANGE,
    FAIL_CROSS_TENANT_ACCESS,
    FAIL_IDENTICAL_SNAPSHOTS,
    FAIL_INSUFFICIENT_SNAPSHOTS,
    FAIL_NO_ELIGIBLE_FIELDS,
    FAIL_PORTFOLIO_ABSENT_AT_PERIOD,
    FAIL_REGISTRY_UNAVAILABLE,
    FAIL_REVERSED_PERIOD_RANGE,
    UNIT_COUNT,
    UNIT_CURRENCY,
    UNIT_PERCENTAGE_POINT,
    WORKFLOW_ID,
    PeriodChangeFailure,
    PeriodChangeResult,
    PortfolioScopeRef,
    SnapshotFrame,
)

logger = logging.getLogger("mi_agent_api.period_change_route")

#: The route id. Matches the workflow identifier so a routed answer, an audit
#: line and the workflow all name the same capability.
ROUTE_NAME = WORKFLOW_ID

#: Controlled failure reason → the repository's existing error taxonomy. The
#: workflow keeps its own vocabulary (it must not import the API layer); the
#: mapping lives here, once.
FAILURE_ERROR_CODES: Dict[str, str] = {
    FAIL_INSUFFICIENT_SNAPSHOTS: ErrorCode.NO_MATCHING_RECORDS,
    FAIL_PORTFOLIO_ABSENT_AT_PERIOD: ErrorCode.NO_MATCHING_RECORDS,
    FAIL_NO_ELIGIBLE_FIELDS: ErrorCode.NO_MATCHING_RECORDS,
    FAIL_AMBIGUOUS_PERIOD_RANGE: ErrorCode.AMBIGUOUS_QUESTION,
    FAIL_REVERSED_PERIOD_RANGE: ErrorCode.INVALID_INPUT,
    FAIL_IDENTICAL_SNAPSHOTS: ErrorCode.UNSUPPORTED_QUESTION,
    FAIL_CROSS_TENANT_ACCESS: ErrorCode.PORTFOLIO_NOT_AUTHORISED,
    FAIL_REGISTRY_UNAVAILABLE: ErrorCode.INTERNAL_ERROR,
}


# --------------------------------------------------------------------------- #
# Snapshot supply — the EXISTING governed per-period frame service
# --------------------------------------------------------------------------- #
def build_snapshots(output_root: Optional[str], client_id: str, *,
                    to_run_id: Optional[str] = None,
                    lens: Any = None) -> Tuple[SnapshotFrame, ...]:
    """Governed portfolio snapshots, oldest → newest, for a client and lens.

    ``lens`` is a RESOLVED portfolio lens (``chat_routing._resolve_lens``), whose
    filters carry the registry's explicit portfolio-id list. Narrowing uses the
    same ``_apply_lens_filter`` the funded-bridge and movement routes use, so a
    period-change answer covers exactly the portfolios every other routed answer
    would cover for the same lens.
    """
    from . import chat_routing
    from . import evolution as evolution_mod
    from mi_agent import portfolio_scope as scope_mod

    frames = evolution_mod.funded_frames(output_root, client_id, to_run_id)
    snapshots: List[SnapshotFrame] = []
    for frame in frames:
        source_df = frame.get("df")
        if source_df is None:
            continue
        df = (chat_routing._apply_lens_filter(source_df, lens)
              if lens is not None else source_df)
        source = frame.get("source")
        snapshots.append(SnapshotFrame(
            snapshot_id=str(frame.get("run_id")),
            reporting_date=frame.get("reporting_date"),
            frame=df,
            # A label, never a path: a result and a log must not carry storage
            # layout (the discipline trakt_core.envelope.SnapshotRef applies).
            dataset_label=_dataset_label(source),
            dataset_reference=str(frame.get("run_id")),
            row_count=int(len(df)),
            # Presence is read from the UNNARROWED snapshot: a portfolio that
            # the lens selected but that this reporting date never contained
            # must be reported as absent, not as a snapshot full of nulls.
            portfolio_ids=_portfolio_ids(source_df, scope_mod)))
    return tuple(snapshots)


def _dataset_label(source: Any) -> Optional[str]:
    if not source:
        return None
    text = str(source)
    return text.rsplit("/", 1)[-1].rsplit("\\", 1)[-1]


def _portfolio_ids(df: Any, scope_mod: Any) -> Tuple[str, ...]:
    """Source portfolios present in a frame, via the governed scope helper."""
    from trakt_core.portfolio import FIELD_PORTFOLIO_ID

    try:
        records = scope_mod.portfolio_records(df)
    except Exception:  # noqa: BLE001 - absent provenance is not an error
        return ()
    return tuple(sorted({str(r.get(FIELD_PORTFOLIO_ID)) for r in records
                         if r.get(FIELD_PORTFOLIO_ID)}))


def scope_ref_from_lens(lens: Any, *, tenant_id: Optional[str] = None,
                        asset_classes: Sequence[str] = ()) -> PortfolioScopeRef:
    """The workflow's scope reference for a RESOLVED portfolio lens.

    A resolved lens carries the registry's explicit portfolio-id list in its
    filters (``chat_routing._resolve_lens``), never a type string, so the scope
    the workflow records is exactly the scope the frames were narrowed to.
    """
    from mi_agent import portfolio_lens as lens_mod

    filters = dict(getattr(lens, "filters", None) or {})
    ids = filters.get(lens_mod.SOURCE_ID_FIELD) or ()
    if isinstance(ids, str):
        ids = (ids,)
    context = None
    try:
        context = lens_mod.context_id(lens) if lens is not None else None
    except Exception:  # noqa: BLE001 - a lens fault must not fail the analysis
        context = None
    return PortfolioScopeRef(
        tenant_id=tenant_id, context_id=context,
        label=getattr(lens, "label", None),
        portfolio_ids=tuple(str(i) for i in ids),
        asset_classes=resolve_asset_classes(asset_classes))


#: BSR asset-applicability values, so a caller-supplied class is validated
#: against the registry's controlled vocabulary rather than passed through.
KNOWN_ASSET_CLASSES: Tuple[str, ...] = (
    "commercial_real_estate", "corporate", "equipment_leasing", "equity_release",
    "residential_mortgage", "sme",
)


def resolve_asset_classes(explicit: Sequence[str] = ()) -> Tuple[str, ...]:
    """The BSR asset classes a period-change analysis may use.

    Deliberately conservative: an asset class is used ONLY when a caller states
    it. The MI layer has no governed asset-class signal at query time —
    ``snapshots.portfolio_risk_type`` exists but *defaults* to ``erm`` with no
    evidence, so treating its output as an asset classification would admit
    equity-release fields onto every book, which is precisely what §9 forbids.

    With no stated class the result is ``()``, and the registry then admits only
    ``cross_asset`` entries (92 of the 106 period-change entries). The extension
    point is this function's argument: a governed portfolio-metadata asset class,
    or an API caller that knows its book, supplies it and the asset-specific
    entries become eligible with no other change.
    """
    return tuple(a for a in explicit if a in KNOWN_ASSET_CLASSES)


# --------------------------------------------------------------------------- #
# Direct invocation — the governed capability, callable without the chat path
# --------------------------------------------------------------------------- #
def analyse_period_change(*, client_id: str, output_root: Optional[str],
                          question: str = "",
                          mode: Optional[str] = None,
                          requested_fields: Sequence[str] = (),
                          requested_concepts: Sequence[str] = (),
                          period_request: Any = None,
                          to_run_id: Optional[str] = None,
                          scope: Any = None,
                          tenant_id: Optional[str] = None,
                          authorised_portfolio_ids: Sequence[str] = (),
                          include_bridge: bool = True,
                          asset_classes: Sequence[str] = (),
                          source_id: Optional[str] = None,
                          ) -> PeriodChangeResult:
    """Run the workflow for a client and scope. Raises ``PeriodChangeFailure``.

    The entry point for a direct Python caller, an API route or a scheduled job.
    The chat route below is one caller among several, not the only way in.
    """
    from mi_agent.period_change.periods import PeriodRequest
    from mi_agent.period_change.models import MODE_PORTFOLIO_OVERVIEW

    snapshots = build_snapshots(output_root, client_id, to_run_id=to_run_id,
                                lens=scope)
    scope_ref = scope_ref_from_lens(scope, tenant_id=tenant_id,
                                    asset_classes=asset_classes)
    request = PeriodChangeRequest(
        question=question, mode=mode or MODE_PORTFOLIO_OVERVIEW,
        period_request=period_request or PeriodRequest(),
        requested_fields=tuple(requested_fields),
        requested_concepts=tuple(requested_concepts),
        scope=scope_ref, include_bridge=include_bridge,
        authorised_portfolio_ids=tuple(authorised_portfolio_ids),
        source_id=source_id)
    return run_period_change_analysis(request, snapshots)


# --------------------------------------------------------------------------- #
# Recognition — reads the SINGLE parse on the route request
# --------------------------------------------------------------------------- #
def recognise_request(req: Any) -> Any:
    """``Recognition`` for the governed recogniser registry."""
    from .recogniser_registry import Recognition

    intent = recognise(req.question, spec=req.spec, view=req.view,
                       semantics_context=req.semantics_context)
    if not intent.matched:
        return Recognition.no(intent.reason)
    return Recognition.yes(reason=f"{intent.reason}:{intent.mode}")


# --------------------------------------------------------------------------- #
# Chat route
# --------------------------------------------------------------------------- #
def route_period_change(question: str, spec: Any, spec_dict: Dict[str, Any], *,
                        client_id: str, run_id: Optional[str],
                        output_root: Optional[str],
                        portfolio_id: Optional[str], as_of: Optional[str],
                        source_lens: Any = None,
                        semantics_context: Optional[Dict[str, Any]] = None,
                        view: str = "funded",
                        lens: Any = None) -> Optional[Dict[str, Any]]:
    """The governed period-change answer, or ``None`` to defer to the next route."""
    from . import chat_routing

    intent = recognise(question, spec=spec, view=view,
                       semantics_context=semantics_context)
    if not intent.matched:
        return None

    resolved_lens = lens if lens is not None else chat_routing._resolve_lens(
        question, source_lens)
    snapshots = build_snapshots(output_root, client_id, to_run_id=run_id,
                                lens=resolved_lens)
    scope_ref = scope_ref_from_lens(resolved_lens)

    # The bridge is computed for every broad question, and for a narrow one only
    # when the question actually asked what drove the movement. Reconciling the
    # book to answer "how did LTV change?" is work nobody asked for.
    from mi_agent.period_change.models import MODE_REQUESTED_METRIC

    request = PeriodChangeRequest(
        question=question, mode=intent.mode,
        period_request=intent.period_request,
        requested_fields=intent.requested_fields,
        requested_concepts=intent.requested_concepts,
        scope=scope_ref,
        include_bridge=(intent.include_bridge
                        or intent.mode != MODE_REQUESTED_METRIC),
        composition_focus=intent.composition_focus)

    try:
        result = run_period_change_analysis(request, snapshots)
    except PeriodChangeFailure as failure:
        return _failure_envelope(question, spec_dict, failure)

    return _render(result, question, spec_dict, portfolio_id, as_of)


def _failure_envelope(question: str, spec_dict: Dict[str, Any],
                      failure: PeriodChangeFailure) -> Dict[str, Any]:
    """A controlled refusal. Explicit, never a fabricated or substituted answer."""
    from . import chat_routing

    envelope = chat_routing._envelope(
        ok=False, question=question, answer=failure.message, spec=spec_dict,
        artifacts=[], route=ROUTE_NAME, error=failure.message,
        lens_applied=True,
        warnings=[f"period-change unavailable: {failure.reason}"])
    meta = envelope["metadata"]
    # ``controlledUnsupported`` is the existing contract for "I will not answer
    # that": HTTP 200 with ok:false, rather than a data error.
    meta["controlledUnsupported"] = True
    meta["workflowId"] = WORKFLOW_ID
    meta["periodChangeFailure"] = failure.to_dict()
    meta["errorCode"] = FAILURE_ERROR_CODES.get(failure.reason,
                                                ErrorCode.UNSUPPORTED_QUESTION)
    return envelope


# --------------------------------------------------------------------------- #
# Presentation — deterministic, from the result only
# --------------------------------------------------------------------------- #
def _format_value(value: Optional[float], unit: str) -> str:
    from . import chat_routing

    if value is None:
        return "—"
    if unit == UNIT_CURRENCY:
        return chat_routing._gbp(value)
    if unit == UNIT_PERCENTAGE_POINT:
        return f"{float(value):.2f}%"
    if unit == UNIT_COUNT:
        return f"{int(round(float(value))):,}"
    return f"{float(value):,.4f}"


def _format_movement(value: Optional[float], unit: str) -> str:
    if value is None:
        return "—"
    sign = "+" if value >= 0 else "−"
    if unit == UNIT_PERCENTAGE_POINT:
        return f"{sign}{abs(float(value)):.2f} pp"
    return f"{sign}{_format_value(abs(float(value)), unit)}"


def _metric_rows(result: PeriodChangeResult) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for metric in result.metric_changes:
        rows.append({
            "metric": metric.display_name,
            "canonical_field": metric.field,
            "concept": metric.analytical_concept,
            "temporality": metric.temporality,
            "aggregation": metric.aggregation,
            "start_value": _format_value(metric.start_value, metric.movement_unit),
            "end_value": _format_value(metric.end_value, metric.movement_unit),
            "movement": _format_movement(metric.movement_value, metric.movement_unit),
            "relative_change": ("—" if metric.relative_change is None
                                else f"{metric.relative_change * 100:+.1f}%"),
            "interpretation": metric.interpretation,
            "significance": metric.significance,
            "status": metric.status,
            "confidence": metric.confidence,
        })
    return rows


_METRIC_COLUMNS = [
    {"key": "metric", "label": "Metric"},
    {"key": "concept", "label": "Concept"},
    {"key": "temporality", "label": "Temporality"},
    {"key": "start_value", "label": "Opening"},
    {"key": "end_value", "label": "Closing"},
    {"key": "movement", "label": "Movement"},
    {"key": "relative_change", "label": "Relative"},
    {"key": "interpretation", "label": "Interpretation"},
    {"key": "significance", "label": "Observed significance"},
    {"key": "status", "label": "Status"},
]

_DISTRIBUTION_COLUMNS = [
    {"key": "dimension", "label": "Dimension"},
    {"key": "category", "label": "Category"},
    {"key": "start_count", "label": "Opening count"},
    {"key": "end_count", "label": "Closing count"},
    {"key": "count_share_movement", "label": "Count-share movement"},
    {"key": "balance_share_movement", "label": "Balance-share movement"},
    {"key": "presence", "label": "Presence"},
]

_BRIDGE_COLUMNS = [
    {"key": "component", "label": "Component"},
    {"key": "amount", "label": "Amount"},
    {"key": "loans", "label": "Loans"},
]


def _distribution_rows(result: PeriodChangeResult) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for dist in result.distribution_changes:
        for category in dist.categories:
            rows.append({
                "dimension": dist.display_name,
                "canonical_field": dist.field,
                "category": category.category,
                "start_count": category.start_count,
                "end_count": category.end_count,
                "count_share_movement": (
                    "—" if category.count_share_movement is None
                    else f"{category.count_share_movement * 100:+.2f} pp"),
                "balance_share_movement": (
                    "—" if category.balance_share_movement is None
                    else f"{category.balance_share_movement * 100:+.2f} pp"),
                "presence": category.presence,
            })
    return rows


def _bridge_rows(result: PeriodChangeResult) -> List[Dict[str, Any]]:
    bridge = result.balance_bridge
    if bridge is None or bridge.status != BRIDGE_STATUS_AVAILABLE:
        return []
    fmt = lambda v: _format_value(v, UNIT_CURRENCY)  # noqa: E731
    return [
        {"component": "Opening balance", "amount": fmt(bridge.opening_balance),
         "loans": bridge.continuing_loan_count + bridge.exited_loan_count},
        {"component": "New loans (closing balance)",
         "amount": fmt(bridge.new_loan_balance), "loans": bridge.new_loan_count},
        {"component": "Exited loans (opening balance)",
         "amount": f"−{fmt(bridge.exited_loan_balance)}",
         "loans": bridge.exited_loan_count},
        {"component": "Movement on continuing loans",
         "amount": _format_movement(bridge.continuing_movement, UNIT_CURRENCY),
         "loans": bridge.continuing_loan_count},
        {"component": "Closing balance", "amount": fmt(bridge.closing_balance),
         "loans": bridge.continuing_loan_count + bridge.new_loan_count},
    ]


def build_answer(result: PeriodChangeResult) -> str:
    """Plain-language rendering of the deterministic summary.

    Every clause restates a value from ``result.summary``; no fact is added, no
    cause is asserted, and no movement is described as material.
    """
    from . import chat_routing

    summary = dict(result.summary)
    period = summary.get("period") or {}
    start = chat_routing._date_label(period.get("start"))
    end = chat_routing._date_label(period.get("end"))
    parts = [f"Between {start} and {end}, "
             f"{summary.get('metrics_comparable', 0)} of "
             f"{summary.get('metrics_analysed', 0)} governed metrics could be "
             f"compared across both snapshots."]

    top = summary.get("top_movements") or []
    if top:
        described = "; ".join(
            f"{row['display_name']} {_format_movement(row['movement_value'], row['movement_unit'])} "
            f"({_format_value(row['start_value'], row['movement_unit'])} → "
            f"{_format_value(row['end_value'], row['movement_unit'])})"
            for row in top)
        parts.append(f"The largest observed movements by rank were {described}.")

    improvements = summary.get("improvements") or []
    deteriorations = summary.get("deteriorations") or []
    if improvements or deteriorations:
        parts.append(
            f"Against the registry's directionality, {len(improvements)} metric(s) "
            f"moved in the improving direction and {len(deteriorations)} in the "
            f"deteriorating direction.")

    shifts = summary.get("largest_composition_shifts") or []
    if shifts:
        described = "; ".join(
            f"{row['display_name']} — {row['category']} "
            f"{row['count_share_movement'] * 100:+.2f} pp of count share"
            for row in shifts)
        parts.append(f"The largest observed composition shifts were {described}.")

    bridge = summary.get("balance_bridge") or {}
    if bridge.get("status") == BRIDGE_STATUS_AVAILABLE:
        parts.append(
            f"The balance bridge reconciles: opening "
            f"{_format_value(bridge['opening_balance'], UNIT_CURRENCY)} "
            f"{_format_movement(bridge['new_loan_closing_balance'], UNIT_CURRENCY)} "
            f"new lending, "
            f"−{_format_value(bridge['exited_loan_opening_balance'], UNIT_CURRENCY)} "
            f"exits, "
            f"{_format_movement(bridge['movement_on_continuing_loans'], UNIT_CURRENCY)} "
            f"on continuing loans, closing "
            f"{_format_value(bridge['closing_balance'], UNIT_CURRENCY)}.")

    parts.append(summary.get("materiality") or "")
    return " ".join(p for p in parts if p)


def _render(result: PeriodChangeResult, question: str, spec_dict: Dict[str, Any],
            portfolio_id: Optional[str], as_of: Optional[str]) -> Dict[str, Any]:
    from . import chat_routing

    payload = result.to_dict()
    artifacts: List[Dict[str, Any]] = []

    summary = payload["summary"]
    kpis = [
        {"label": "Opening period",
         "value": chat_routing._date_label(summary["period"]["start"])},
        {"label": "Closing period",
         "value": chat_routing._date_label(summary["period"]["end"])},
        {"label": "Metrics compared",
         "value": f"{summary['metrics_comparable']}/{summary['metrics_analysed']}"},
        {"label": "Improved / deteriorated",
         "value": f"{len(summary['improvements'])} / {len(summary['deteriorations'])}"},
    ]
    artifacts.append(chat_routing._summary_kpi_artifact(
        "Period change — governed overview", kpis, spec=spec_dict,
        portfolio_id=portfolio_id, as_of=as_of,
        description=(f"Resolved by {summary['period']['resolution_method']} "
                     f"from the governed Business Semantics Registry.")))

    metric_rows = _metric_rows(result)
    if metric_rows:
        artifacts.append(chat_routing._table_artifact(
            "Metric movements", columns=_METRIC_COLUMNS, rows=metric_rows,
            spec=spec_dict, portfolio_id=portfolio_id, as_of=as_of,
            description=(f"{len(metric_rows)} governed metrics, selected by "
                         f"policy {payload['field_selection']['policy_name']} "
                         f"v{payload['field_selection']['policy_version']}.")))

    distribution_rows = _distribution_rows(result)
    if distribution_rows:
        artifacts.append(chat_routing._table_artifact(
            "Composition shifts", columns=_DISTRIBUTION_COLUMNS,
            rows=distribution_rows, spec=spec_dict, portfolio_id=portfolio_id,
            as_of=as_of,
            description="Count and balance share movement by category."))

    bridge_rows = _bridge_rows(result)
    if bridge_rows:
        artifacts.append(chat_routing._table_artifact(
            "Balance bridge", columns=_BRIDGE_COLUMNS, rows=bridge_rows,
            spec=spec_dict, portfolio_id=portfolio_id, as_of=as_of,
            description=("Opening to closing balance over a stable loan "
                         "identifier.")))

    envelope = chat_routing._envelope(
        ok=True, question=question, answer=build_answer(result), spec=spec_dict,
        artifacts=artifacts, route=ROUTE_NAME, lens_applied=True,
        warnings=list(payload["warnings"]) + list(payload["limitations"]),
        source_notes=[{
            "label": "Business Semantics Registry",
            "detail": (f"v{payload['audit']['business_semantics']['registry_version']} "
                       f"(schema {payload['audit']['business_semantics']['schema_version']})"),
        }, {
            "label": "Governed snapshots",
            "detail": " → ".join(
                str(ref.get("reporting_date") or ref.get("snapshot_id"))
                for ref in payload["dataset_provenance"]),
        }])

    # Additive: the complete governed result travels intact alongside the
    # existing envelope keys. No existing field is renamed or removed.
    envelope["periodChange"] = payload
    meta = envelope["metadata"]
    meta["workflowId"] = WORKFLOW_ID
    meta["periodResolution"] = payload["period_resolution"]["resolution_method"]
    meta["periodChangeMode"] = payload["request_interpretation"]["mode"]
    return envelope
