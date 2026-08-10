"""mi_agent_api/mi_service.py — THE governed MI capability.

    capability id:  mi.question.answer

One analytical implementation; every interface is an adapter over it:

    React MI Agent  ──┐
    M365 Copilot    ──┤
    Python / job    ──┼──►  execute_governed_mi_query(request, context, deps)
    future agent    ──┘            │
                                   ├─ scope check             (ExecutionContext)
                                   ├─ portfolio authorisation (trakt_core.tenancy)
                                   ├─ source-approval policy  (trakt_core.policy)
                                   ├─ dataset resolution      (mi_agent_api.datasets)
                                   ├─ parsing · intent routing · deterministic
                                   │  calculation · validation · reconciliation
                                   │  · provenance · artifacts    (unchanged)
                                   └─►  GovernedResult[dict]

Three properties hold by construction:

* **No web framework below the adapter.** This module imports no FastAPI and no
  ``mi_agent_api.app``; dataset resolution lives in ``mi_agent_api.datasets``.
  ``tests/test_governance_dependency_direction.py`` asserts it.
* **Tenant is never request data.** ``context.tenant_id`` comes from
  authentication or a trusted internal invocation. A caller-supplied
  ``portfolio_id`` can only narrow within the tenant, never redefine it.
* **Governance runs before data.** Scope, portfolio authorisation and source
  approval all execute before a dataframe is touched.

Compatibility: the analytical payload is unchanged. It is carried verbatim in
``GovernedResult.result``; ``mi_agent_api.presenters`` returns that dictionary to
React with an additive ``governance`` block and reshapes the same object for
Copilot. No pre-existing field was renamed or removed.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from trakt_core import perf as _perf
from trakt_core.audit import emit_audit_event
from trakt_core.context import SCOPE_MI_QUERY, ExecutionContext
from trakt_core.envelope import (
    STATUS_BLOCKED,
    STATUS_ERROR,
    STATUS_SUCCESS,
    AuditMetadata,
    GovernedResult,
    PolicyState,
    ProvenanceRef,
    ScopeRef,
    SnapshotRef,
)
from trakt_core.errors import ErrorCategory, ErrorCode, TraktError
from trakt_core.policy import evaluate_source_approval
from trakt_core.tenancy import AuthorisedPortfolio, authorise_portfolio_access

from . import chat_routing as chat_routing_mod
from . import currency as currency_mod
from . import workspace as workspace_mod
from .adapters import adapt_workflow_result
from .dependencies import CapabilityDependencies, build_dependencies

logger = logging.getLogger("mi_agent_api.mi_service")

#: Stable capability identifier. Part of the external contract.
CAPABILITY = "mi.question.answer"

#: Retained for callers that still import it. The tenant now comes from the
#: ExecutionContext; this is only the historical portfolio-selector fallback.
DEFAULT_CLIENT_ID = "client_001"


# --------------------------------------------------------------------------- #
# Channel-neutral request contract
# --------------------------------------------------------------------------- #
@dataclass
class MiQueryRequest:
    """One governed MI question, expressed without any channel/HTTP concepts.

    This is *untrusted* caller input. Trusted identity lives in
    :class:`~trakt_core.context.ExecutionContext`.
    """

    question: str
    #: ``"{selector}"`` or ``"{selector}/{run_id}"``. Narrows within the tenant.
    portfolio_id: Optional[str] = None
    as_of_date: Optional[str] = None
    filters: Optional[Dict[str, Any]] = None
    dataset_context: Optional[str] = None
    context: Optional[Any] = None
    source_portfolio_lens: Optional[str] = None
    #: DEPRECATED as an identity input. Retained because adapters used it as the
    #: fallback portfolio selector when the caller named none, and dropping it
    #: would change which frame those callers resolve. Never authoritative for
    #: tenancy: a value disagreeing with the trusted tenant is rejected with
    #: TENANT_MISMATCH. New callers should leave it unset.
    client_id: Optional[str] = None
    #: Channel-neutral execution options (reserved; no analytical effect).
    options: Dict[str, Any] = field(default_factory=dict)

    def effective_portfolio_id(self) -> Optional[str]:
        """The portfolio selector the analysis runs against.

        Unchanged from the original implementation so dataset resolution behaves
        exactly as before; ``None`` still means "the active governed dataset".
        """
        return self.portfolio_id or self.client_id or None


def split_portfolio(portfolio_id: Optional[str],
                    default_client: str = DEFAULT_CLIENT_ID) -> tuple[str, Optional[str]]:
    """``("{client}/{run}" | "{client}" | None) -> (client_id, run_id | None)``."""
    if not portfolio_id:
        return default_client, None
    if "/" in portfolio_id:
        client_id, run_id = portfolio_id.split("/", 1)
        return (client_id or default_client), (run_id or None)
    return portfolio_id, None


# --------------------------------------------------------------------------- #
# Analytical envelope helpers (the payload React and Copilot already consume)
# --------------------------------------------------------------------------- #
def _governed_context(envelope: Dict[str, Any], *, req: MiQueryRequest,
                      client_id: str, run_id: Optional[str], view: str,
                      run_required: bool) -> Dict[str, Any]:
    """Stamp the channel-neutral analytical metadata onto the envelope.

    Additive only — every pre-existing React key is left exactly as the adapter
    produced it, so the React contract is preserved byte-for-byte.
    """
    from .data_source import data_source_kind, data_source_label

    meta = envelope.setdefault("metadata", {})
    if not isinstance(meta, dict):
        return envelope
    meta["datasetContext"] = view
    meta["selectedClient"] = client_id
    meta["selectedPortfolio"] = req.effective_portfolio_id()
    # A run is reported ONLY where the analysis genuinely used one; a
    # point-in-time question against the active governed dataset has none.
    meta["selectedRun"] = run_id if run_required else None
    meta["runRequired"] = bool(run_required)
    try:
        meta["dataSourceKind"] = data_source_kind()
        meta["dataSourceLabel"] = data_source_label()
    except Exception as exc:  # noqa: BLE001 - labelling must never fail a query
        logger.warning("data-source labelling failed: %s", exc)
        meta.setdefault("dataSourceKind", "unknown")
        meta.setdefault("dataSourceLabel", "unknown")
    envelope.setdefault("assumptions", [])
    envelope.setdefault("warnings", [])
    envelope.setdefault("diagnostics", [])
    envelope.setdefault("sourceNotes", [])
    return envelope


def _error_envelope(msg: str, *, req: MiQueryRequest, view: str) -> Dict[str, Any]:
    """The analytical failure payload. Shape unchanged from the original."""
    return {
        "ok": False, "error": msg, "question": req.question,
        "answer": msg, "interpreted": "", "spec": {},
        "validation": {"ok": False, "errors": [msg], "warnings": [],
                       "resolved_fields": {}},
        "artifacts": [], "warnings": [], "assumptions": [], "diagnostics": [],
        "sourceNotes": [],
        "metadata": {"engine": "mi_agent", "source": "python", "mock": False,
                     "datasetContext": view},
    }


# --------------------------------------------------------------------------- #
# Envelope assembly
# --------------------------------------------------------------------------- #
def _snapshot_ref(descriptor: Any, approval_state: Optional[str]) -> SnapshotRef:
    return SnapshotRef(
        source_kind=getattr(descriptor, "source_kind", None),
        source_base=getattr(descriptor, "source_base", None),
        dataset_label=getattr(descriptor, "label", None),
        reporting_date=getattr(descriptor, "reporting_date", None),
        snapshot_id=getattr(descriptor, "snapshot_id", None),
        content_hash=getattr(descriptor, "content_hash", None),
        row_count=getattr(descriptor, "row_count", None),
        approval_state=approval_state,
        source_portfolios=tuple(getattr(descriptor, "source_portfolios", ()) or ()),
    )


def _audit(context: ExecutionContext, *, outcome: str, started_at: str,
           t0: float, portfolio_id: Optional[str], snapshot_id: Optional[str],
           error_code: Optional[str]) -> AuditMetadata:
    return AuditMetadata(
        capability=CAPABILITY, request_id=context.request_id,
        correlation_id=context.correlation_id, tenant_id=context.tenant_id,
        organisation_id=context.organisation_id,
        microsoft_tenant_id=context.microsoft_tenant_id,
        actor_id=context.actor_id, actor_type=context.actor_type,
        channel=context.channel, portfolio_id=portfolio_id,
        snapshot_id=snapshot_id, outcome=outcome, started_at=started_at,
        duration_ms=int((time.monotonic() - t0) * 1000), error_code=error_code)


#: Error categories that mean "governance refused", as opposed to "the engine
#: could not answer". They map to STATUS_BLOCKED so a machine caller can tell a
#: policy decision from a capability failure without parsing prose.
_BLOCKING_CATEGORIES = frozenset({
    ErrorCategory.AUTHENTICATION, ErrorCategory.AUTHORISATION, ErrorCategory.POLICY,
})


def _failure(context: ExecutionContext, error: TraktError, *, started_at: str,
             t0: float, portfolio_id: Optional[str], view: str,
             req: MiQueryRequest, policy: PolicyState,
             snapshot: Optional[SnapshotRef] = None) -> GovernedResult[Dict[str, Any]]:
    """A governed refusal or failure, carrying the analytical payload shape the
    existing channels already render."""
    status = STATUS_BLOCKED if error.category in _BLOCKING_CATEGORIES else STATUS_ERROR
    return GovernedResult(
        capability=CAPABILITY, status=status, request_id=context.request_id,
        correlation_id=context.correlation_id, tenant_id=context.tenant_id,
        portfolio_id=portfolio_id, snapshot=snapshot,
        result=_error_envelope(error.message, req=req, view=view),
        warnings=(), policy=policy, provenance=None, error=error,
        audit=_audit(context, outcome=status, started_at=started_at, t0=t0,
                     portfolio_id=portfolio_id,
                     snapshot_id=snapshot.snapshot_id if snapshot else None,
                     error_code=error.code))


# --------------------------------------------------------------------------- #
# The capability
# --------------------------------------------------------------------------- #
def execute_governed_mi_query(
    request: MiQueryRequest,
    context: ExecutionContext,
    dependencies: Optional[CapabilityDependencies] = None,
) -> GovernedResult[Dict[str, Any]]:
    """Answer one governed MI question.

    Governance runs first and in a fixed order — scope, portfolio authorisation,
    source approval — so no dataframe is touched for a caller that is not
    entitled to it, or for a dataset that is not approved.

    Never raises for an analytical fault: a controlled ``GovernedResult`` with a
    typed error is returned instead, so a channel can never turn a governed
    failure into a plausible narrative.
    """
    deps = dependencies or build_dependencies()
    started_at = datetime.now(timezone.utc).isoformat()
    t0 = time.monotonic()

    view = workspace_mod.resolve_active_view(
        request.question,
        request.dataset_context
        or (request.context.get("activeView") or request.context.get("datasetContext")
            if isinstance(request.context, dict) else None))
    policy = PolicyState(runtime_mode=deps.runtime_mode)
    requested_portfolio = request.effective_portfolio_id()

    # ---- 1. governance: scope, tenancy, portfolio authorisation ---------- #
    try:
        context.require_scope(SCOPE_MI_QUERY)

        # A deprecated adapter-supplied client_id may act as a portfolio fallback
        # but may never disagree with the authenticated tenant.
        if request.client_id and request.client_id != context.tenant_id:
            raise TraktError(
                ErrorCode.TENANT_MISMATCH,
                "The supplied client identifier does not match the authenticated "
                "tenant.",
                request_id=context.request_id)

        authorised: AuthorisedPortfolio = authorise_portfolio_access(
            context, requested_portfolio, registry=deps.tenant_registry)
        policy = PolicyState(runtime_mode=deps.runtime_mode, tenant_authorised=True,
                             capability_allowed=True)
    except TraktError as err:
        result = _failure(context, err, started_at=started_at, t0=t0,
                          portfolio_id=requested_portfolio, view=view, req=request,
                          policy=policy)
        emit_audit_event(result)
        return result

    # ---- 2. governance: is this dataset allowed to answer? --------------- #
    try:
        descriptor = deps.datasets.describe_active_dataset()
        approval = evaluate_source_approval(
            descriptor.source_base, mode=deps.runtime_mode,
            dataset_available=descriptor.available)
    except Exception as exc:  # noqa: BLE001 - a storage fault is not a raw 500
        logger.exception("dataset description failed for tenant=%s", context.tenant_id)
        err = TraktError(ErrorCode.STORAGE_UNAVAILABLE,
                         "The governed data store is currently unavailable.",
                         request_id=context.request_id, cause=exc)
        result = _failure(context, err, started_at=started_at, t0=t0,
                          portfolio_id=authorised.portfolio_id, view=view,
                          req=request, policy=policy)
        emit_audit_event(result)
        return result

    snapshot = _snapshot_ref(descriptor, approval.state)
    if not approval.approved:
        err = TraktError(approval.error_code or ErrorCode.DATA_SOURCE_NOT_APPROVED,
                         approval.reason, request_id=context.request_id)
        result = _failure(
            context, err, started_at=started_at, t0=t0,
            portfolio_id=authorised.portfolio_id, view=view, req=request,
            policy=PolicyState(runtime_mode=deps.runtime_mode, tenant_authorised=True,
                               capability_allowed=True, data_approved=False,
                               notes=(approval.reason,)),
            snapshot=snapshot)
        emit_audit_event(result)
        return result

    policy = PolicyState(
        runtime_mode=deps.runtime_mode, tenant_authorised=True,
        capability_allowed=True, data_approved=True, fixture_source=approval.fixture,
        notes=(approval.reason,) if approval.fixture else ())

    # ---- 3. the analytical execution (unchanged) ------------------------- #
    payload = _run_analysis(request, authorised, view, deps)

    ok = bool(payload.get("ok"))
    status = STATUS_SUCCESS if ok else STATUS_ERROR
    error = None
    if not ok:
        error = TraktError(
            _classify_analytical_failure(payload),
            str(payload.get("error")
                or "The MI Agent could not answer this question."),
            request_id=context.request_id)

    result: GovernedResult[Dict[str, Any]] = GovernedResult(
        capability=CAPABILITY, status=status, request_id=context.request_id,
        correlation_id=context.correlation_id, tenant_id=context.tenant_id,
        portfolio_id=authorised.portfolio_id, snapshot=snapshot, result=payload,
        warnings=tuple(str(w) for w in (payload.get("warnings") or [])),
        policy=policy, scope=_scope_ref(payload),
        provenance=ProvenanceRef(
            source_notes=tuple(payload.get("sourceNotes") or ()),
            reconciliation=payload.get("reconciliation"), snapshot=snapshot),
        error=error,
        audit=_audit(context, outcome=status, started_at=started_at, t0=t0,
                     portfolio_id=authorised.portfolio_id,
                     snapshot_id=snapshot.snapshot_id,
                     error_code=error.code if error else None))
    emit_audit_event(result)
    return result


def _scope_ref(payload: Dict[str, Any]) -> Optional[ScopeRef]:
    """The governed portfolio scope + coverage for this answer, if it had one.

    Read from the analytical payload the engine produced — never recomputed
    here, so the envelope and the answer can never disagree about who
    contributed.
    """
    if not isinstance(payload, dict):
        return None
    coverage = payload.get("portfolioCoverage")
    if coverage:
        return ScopeRef.from_dict(coverage)
    scope = payload.get("portfolioScope")
    if scope:
        ids = tuple(scope.get("portfolio_ids") or ())
        return ScopeRef(context_id=scope.get("context_id"),
                        context_kind=scope.get("context_kind"),
                        label=scope.get("label"),
                        portfolios_in_scope=ids, portfolios_used=ids,
                        is_fully_consolidated=True)
    return None


def _stamp_routed_scope(routed: Dict[str, Any], req: MiQueryRequest) -> None:
    """Stamp the resolved portfolio scope onto a ROUTED analytical answer.

    Routed intents never reach the executor, so they carry no field-level
    coverage. Recording the resolved scope keeps the governed envelope complete
    for every route: a caller always learns which portfolios an answer covers.

    Critically, the scope stamped is the scope the ANSWER HAS, not the scope the
    caller asked for. A route that could not narrow its figures reports
    ``metadata.lensApplied is False`` (see ``chat_routing._disclose_lens_scope``)
    and is stamped with the FULL platform scope, because that is what its numbers
    actually cover. Stamping the requested lens on an un-narrowed answer turned
    the governed envelope — the one control that exists to prevent
    misattribution — into the thing performing it.
    """
    if not isinstance(routed, dict) or routed.get("portfolioScope"):
        return
    try:
        from mi_agent import portfolio_lens as plens

        from . import portfolio_context as ctx_mod

        meta = routed.get("metadata") or {}
        lens_applied = meta.get("lensApplied")
        if lens_applied is False:
            context_id = plens.LENS_TOTAL
        else:
            lens = plens.resolve_lens_with_default(
                req.question,
                plens.lens_from_selection(req.source_portfolio_lens)
                if req.source_portfolio_lens is not None else None)
            context_id = plens.context_id(lens)
        resolved = ctx_mod.resolve_context(context_id, discover_pipeline=False)
        routed["portfolioScope"] = resolved.scope.to_dict()
    except Exception as exc:  # noqa: BLE001 - disclosure must never break a route
        logger.info("routed scope stamping skipped: %s", exc)


def _classify_analytical_failure(payload: Dict[str, Any]) -> str:
    """Map an engine-reported failure onto a stable code.

    Lets a machine caller distinguish "I will not answer that" (unsupported /
    ambiguous) from "no rows matched" from "the calculation broke", which the
    previous free-text ``error`` string could not express.
    """
    meta = payload.get("metadata") or {}
    if meta.get("controlledUnsupported"):
        return ErrorCode.UNSUPPORTED_QUESTION
    if meta.get("unmappedQuestion"):
        return ErrorCode.AMBIGUOUS_QUESTION
    errors = ((payload.get("validation") or {}).get("errors") or [])
    joined = " ".join(str(e) for e in errors).lower()
    if "unmapped_question" in joined:
        return ErrorCode.AMBIGUOUS_QUESTION
    if "no rows" in joined or "no matching" in joined:
        return ErrorCode.NO_MATCHING_RECORDS
    return ErrorCode.CALCULATION_FAILED


def _resolve_frame(ds, view: str, portfolio_id: Optional[str]):
    """``(frame, error)`` for this request, resolved defensively.

    Called ONCE, before parsing, so the single parse can be resolved against the
    dataset's real columns. Never raises: a frame that cannot be resolved yields
    ``(None, message)``, and routing still runs — several governed capabilities
    (forecast, risk limits, conversion) answer from run artefacts and do not need
    a frame at all.
    """
    try:
        return ds._resolve_query_frame(view, portfolio_id)
    except FileNotFoundError as exc:
        return None, str(exc)
    except Exception:  # noqa: BLE001 - data load/prep must not raw-500
        # The exception type/message is logged, never returned: an internal class
        # name is not something a client should see.
        logger.exception("MI query frame resolution failed for portfolio=%r view=%r",
                         portfolio_id, view)
        return None, "Could not load the governed data for this query."


def _run_analysis(req: MiQueryRequest, authorised: AuthorisedPortfolio, view: str,
                  deps: CapabilityDependencies) -> Dict[str, Any]:
    """The analytical pipeline.

    The question is parsed **once**, above routing, and the resulting
    :class:`~mi_agent.parsed_question.ParsedQuestion` is threaded through both
    the recogniser registry and the executor — so the spec that was routed on is
    the spec that runs. Previously each stage parsed independently, which cost
    double the parse time and let routing and execution disagree.

    Routed governed capabilities first (compare / evolution / forecast / risk /
    geo / cohort / bridge / scenario); anything unmatched falls through to the
    deterministic point-in-time executor.
    """
    from mi_agent.mi_agent_workflow import run_mi_agent_query
    from mi_agent.mi_query_validator import load_mi_semantics
    from mi_agent.parsed_question import ParsedQuestion

    from .data_source import semantics_path

    ds = deps.datasets
    # The analytical layer keeps receiving exactly what the caller asked for.
    # ``None`` means "the active governed dataset" — substituting the tenant's
    # default selector here would change which frame existing callers resolve.
    portfolio_id = authorised.requested_portfolio_id
    client_id, run_id = split_portfolio(portfolio_id)
    semantics = load_mi_semantics(semantics_path())
    llm_cfg = ds._mi_llm_config()

    # ---- resolve the frame, then parse: ONCE, before anything branches ---- #
    # Request-scoped display currency first, so routed answers format in the
    # book's currency too (tape -> config -> GBP; cached per client).
    try:
        ds._apply_request_currency(client_id, portfolio_id)
    except Exception as exc:  # noqa: BLE001 - currency must never fail a query
        logger.info("request currency resolution skipped: %s", exc)
    with _perf.stage("mi_query.resolve_frame"):
        df, frame_error = _resolve_frame(ds, view, portfolio_id)

    try:
        with _perf.stage("mi_query.parse"):
            parsed = ParsedQuestion.parse(
                req.question, semantics,
                available_columns=set(df.columns) if df is not None else None,
                llm_enabled=llm_cfg.enabled, model=llm_cfg.model,
                # Extension point: supply a Business Semantics Registry resolver
                # here to attach governed business-term metadata to every parse.
                # It flows to every recogniser via
                # ``RouteRequest.semantics_context`` with no further plumbing.
                semantics_resolver=deps.semantics_resolver)
    except Exception:  # noqa: BLE001 - a parser fault is a controlled failure
        logger.exception("MI query parse failed for question=%r", req.question)
        return _error_envelope("The MI Agent could not interpret this question.",
                               req=req, view=view)

    routed = None
    try:
        def _routed_frame(cli: str, rid: Optional[str]):
            """The funded frame for a routed intent, resolved by exactly the same
            governed resolver the point-in-time path uses. With no run selected
            this is the ACTIVE governed dataset — a point-in-time question (e.g.
            regional concentration) must never require a run id."""
            pid = f"{cli}/{rid}" if rid else (cli or None)
            frame, err = ds._resolve_query_frame("funded", pid)
            return None if err else frame

        routed = chat_routing_mod.try_route(
            req.question, portfolio_id=portfolio_id, view=view,
            output_root=ds._onboarding_output_root(),
            pipeline_root=ds._pipeline_discovery_root(),
            semantics=semantics,
            # DEFERRED, not built here: the historical completion model replays
            # every retained weekly extract, and only the scenario / cohort
            # conversion / run-rate forecast routes use it. Passing the builder
            # means an ordinary question ("balance by region") never pays for it.
            history_model_provider=lambda: ds._pipeline_history(client_id),
            as_of=req.as_of_date,
            source_lens=req.source_portfolio_lens or None,
            frame_resolver=_routed_frame,
            extra_filters=req.filters or None,
            parsed=parsed)
    except Exception as exc:  # noqa: BLE001 - routing must never break the chat
        logger.warning("chat routing failed; using point-in-time path: %s", exc)
        routed = None
    if routed is not None:
        route = (routed.get("metadata") or {}).get("route") if isinstance(routed, dict) else None
        _stamp_routed_scope(routed, req)
        return _governed_context(routed, req=req, client_id=client_id, run_id=run_id,
                                 view=view, run_required=_route_requires_run(route))

    # ---- point-in-time: active governed dataset (or the selected run) ----- #
    if frame_error:
        return _error_envelope(frame_error, req=req, view=view)
    if df is None:
        return _error_envelope("Could not load the governed data for this query.",
                               req=req, view=view)

    currency_mod.resolve_and_set(df)

    llm_enabled, llm_model = llm_cfg.enabled, llm_cfg.model
    runner = deps.query_runner or run_mi_agent_query
    try:
        workflow = runner(
            req.question, df, str(semantics_path()),
            parser_mode="llm" if llm_enabled else "deterministic",
            llm_enabled=llm_enabled, model=llm_model,
            extra_filters=req.filters or None,
            source_portfolio_lens=req.source_portfolio_lens or None,
            parsed=parsed)
        result = adapt_workflow_result(
            workflow, portfolio_id=portfolio_id, as_of=req.as_of_date)
    except Exception:  # noqa: BLE001 - surface, don't 500
        logger.exception("MI query failed for question=%r portfolio=%r",
                         req.question, portfolio_id)
        return _error_envelope(
            "The MI Agent could not complete this query.", req=req, view=view)

    meta = result.setdefault("metadata", {}) if isinstance(result, dict) else {}
    if isinstance(meta, dict):
        meta["llm"] = {"enabled": llm_cfg.enabled, "available": llm_cfg.available,
                       "model": llm_cfg.model if llm_cfg.available else None,
                       "status": llm_cfg.status}
        if workflow.get("portfolio_lens"):
            meta["portfolioLens"] = workflow["portfolio_lens"]
    # Governed portfolio scope + coverage. The BACKEND states which portfolios
    # were in scope, which answered and which could not; every channel renders
    # these facts rather than deriving its own.
    if isinstance(result, dict):
        if workflow.get("portfolio_scope"):
            result["portfolioScope"] = workflow["portfolio_scope"]
        if workflow.get("portfolio_coverage"):
            result["portfolioCoverage"] = workflow["portfolio_coverage"]
    # An LLM that was requested but is unusable is a configuration fault the
    # operator must see, not a silent downgrade.
    if llm_cfg.enabled and not llm_cfg.available and isinstance(result, dict):
        result.setdefault("warnings", []).extend(llm_cfg.warnings)
    # A point-in-time answer is run-scoped only when a run was explicitly selected.
    return _governed_context(result, req=req, client_id=client_id, run_id=run_id,
                             view=view, run_required=bool(run_id))


#: Routes whose analytical intent GENUINELY needs a specific run / history:
#: dated comparison, evolution, cohort progression, forecast, run-scoped risk.
#: Everything else (notably geographic exposure) is point-in-time and answers
#: from the active governed dataset.
_RUN_SCOPED_ROUTES = {
    "temporal_compare", "evolution", "evolution_funnel",
    "evolution_pipeline_stage", "forecast_extrapolation", "scenario",
    "cohort_progression", "cohort_conversion", "risk_limits", "funded_bridge",
    # Period Change Analysis resolves and compares two governed snapshots, so
    # the run it closed on is genuinely part of the answer.
    "period_change_analysis",
}


def _route_requires_run(route: Optional[str]) -> bool:
    return bool(route) and route in _RUN_SCOPED_ROUTES
