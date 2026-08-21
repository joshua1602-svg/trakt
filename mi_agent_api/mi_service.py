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
from typing import Any, Dict, List, Optional, Union

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
    #: The portfolio scope the caller has selected. A LIST selects several books
    #: explicitly and resolves to exactly those — never to their provenance
    #: type, which would widen the answer to books the caller did not choose.
    source_portfolio_lens: Optional[Union[str, List[str]]] = None
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


#: Parser provenance for the evaluation harness. The distinction that matters:
#: a deterministic FALLBACK after an LLM failure is not an LLM result, and a
#: harness that cannot tell them apart reports a fiction.
_LLM_FAILURE_CATEGORIES = (
    ("authentication", ("authenticationerror", "401", "api key")),
    ("rate_limit", ("ratelimit", "429", "overloaded")),
    ("timeout", ("timeout", "timed out", "deadline")),
    ("parse_failure", ("failed validation", "did not return a usable")),
)


def _parser_provenance(workflow: Dict[str, Any]) -> Dict[str, Any]:
    """``{parser_used, llm_failure}`` — non-secret, for evaluation only."""
    meta = (workflow.get("metadata") or {}) if isinstance(workflow, dict) else {}
    parse_meta = meta.get("parse_metadata") or {}
    detail = str(parse_meta.get("parser_mode_detail")
                 or meta.get("parser_mode_detail") or "")
    mode = str(parse_meta.get("parser_mode") or workflow.get("parser_mode") or "")
    status = str(parse_meta.get("status") or "").lower()

    if detail == "deterministic_fallback":
        used = "deterministic_fallback_after_llm_failure"
    elif mode == "llm" or detail.startswith("llm"):
        used = "llm"
    else:
        used = "deterministic"

    failure = None
    if used == "deterministic_fallback_after_llm_failure" or detail == "validation_failed":
        # "validation_failed" already names the failure: the model returned a
        # spec the governed validator rejected. Reporting that as "unknown" hid
        # the single most common LLM failure mode behind a placeholder.
        failure = ("parse_failure" if detail == "validation_failed" else "unknown")
        for category, needles in _LLM_FAILURE_CATEGORIES:
            if any(n in status for n in needles):
                failure = category
                break
    return {"parser_used": used, "llm_failure": failure,
            "parser_mode_detail": detail or None,
            "specialist_intent_carried": parse_meta.get("specialist_intent_carried") or []}


def _guard_routed_answer(routed: Dict[str, Any], *, question: str,
                         route: Optional[str], semantics: Dict[str, Any],
                         frame, parsed=None) -> Dict[str, Any]:
    """P0 semantic guard for an answer produced by a routed governed capability.

    A routed answer never reaches the point-in-time executor, so the workflow's
    guard cannot see it. The check here is deliberately narrower: a route
    declares its identity and its own scope, which is enough to tell whether a
    period comparison, a stress condition, a value threshold or a ranking the
    user asked for was actually part of what ran.

    The bar for refusing is a facet that would change the NUMBER or that IS the
    subject of the question. A facet a listing route simply could not narrow to
    (asking about London and receiving every region's limit) is disclosed, not
    refused: the requested category is present and no single figure is being
    passed off as the narrow one. Never refuses on an unprovable facet, so a
    working governed route cannot be disabled by this check.
    """
    if not isinstance(routed, dict) or not routed.get("ok"):
        return routed
    try:
        from mi_agent import execution_receipt as receipt_mod

        spec = routed.get("spec") if isinstance(routed.get("spec"), dict) else {}
        # The measure SET the route declares it compared, not just the spec's
        # singular metric: a P1E spec can carry a set with metric=None, and the
        # substitution check then had nothing to compare against.
        _compared = receipt_mod.comparison_evidence(routed)
        substitution = receipt_mod.detect_measure_substitution(
            question, route=route, metric_key=(spec or {}).get("metric"),
            executed_concepts=receipt_mod.comparison_measure_concepts(_compared))
        facets = receipt_mod.detect_requested_facets(
            question, semantics, frame=frame,
            requested_dimensions=receipt_mod.requested_dimension_terms(
                question, semantics,
                available_columns=receipt_mod.book_columns(frame)),
            # The parser's resolved filters, so the narrowing owner consumes that
            # answer instead of claiming the same field.
            resolved_filters=set(getattr(getattr(parsed, "spec", None),
                                         "filters", None) or ()))
        # D2 — THE ROLE DECISION, CONSUMED RATHER THAN DEFAULTED.
        #
        # `requested_dimension_terms` raises every named dimension as a
        # grouping. On the point-in-time path a later reader gave each one the
        # role its sources actually assigned; on this path nothing did, so a
        # dimension the parser positively slotted as a FILTER was asserted to be
        # a breakdown and the routed reconciler stamped that assertion. Measured
        # across 693 corpus questions, the two paths asserted a different role on
        # 37 — every one of them the same divergence.
        #
        # The parse is the one already threaded through routing (it supplies the
        # governed population predicates a few lines above), so this reads a
        # decision that was taken once rather than re-deriving it.
        #
        # `settle_unresolved=False`: the ROLE is settled here, but what a routed
        # answer owes an UNRESOLVED role is not, because that turns on evidence
        # D7 (B12) has yet to repair. See the branch in the split for the
        # measurement behind that choice.
        facets = receipt_mod._split_named_dimension_roles(
            facets, getattr(parsed, "spec", None) or {}, semantics,
            receipt_mod.book_columns(frame),
            question=question, settle_unresolved=False)
        granularity = receipt_mod.granularity_facets(question, route)
        # P1L: the material row population the spec carries. Raised from the
        # governed spec, proven from execution evidence the route reports — a
        # route that reports nothing leaves these LOST and the answer refuses,
        # instead of presenting a whole-book figure for a narrowed question.
        population = receipt_mod.population_facets(spec, semantics)
        import os as _od
        if _od.environ.get("P1L_DEBUG"):
            import sys as _sd
            print(f"P1LG route={route} specfilters={(spec or {}).get('filters')} "
                  f"evidence={(routed.get('metadata') or {}).get('populationApplied')}", file=_sd.stderr)
        receipt_mod.reconcile_population(
            population, (routed.get("metadata") or {}).get("populationApplied"),
            dataset_columns=receipt_mod.book_columns(frame))
        # A filter naming the population the ANALYTICAL PLAN already resolved
        # from the intent is a no-op, not a loss. "Of the current offer pipeline,
        # how much should convert?" sometimes parsed with an explicit
        # pipeline_stage = Offer filter and sometimes without; the plan resolves
        # OFFER either way and declares it on its findings, but the funded-frame
        # narrowing ledger has nothing to report for a pipeline predicate, so
        # the facet was stamped LOST and the same question answered on one run
        # and refused on the next.
        #
        # Accepted only on the plan's own declaration, and only when that
        # declaration names BOTH the same field and the value asked for — a plan
        # that narrowed to KFI cannot satisfy a request for OFFER, and
        # account_status = offer (right value, wrong field) stays lost.
        for _facet in population:
            if (_facet.status != receipt_mod.APPLIED
                    and receipt_mod._analytical_population_satisfies(routed, _facet)):
                _facet.status, _facet.reason = receipt_mod.APPLIED, ""
        # EVERY population on this receipt is stamped from the same evidence,
        # whoever raised it. Three raisers now: the ledger from `spec.filters`,
        # the detector from the seasoning owner's decision, and the role owner
        # reclassifying a named dimension the parser slotted as a filter.
        #
        # Two failure modes are closed here, both of them recurrences:
        #
        #   DUPLICATE  the ledger and the detector raise the same governed
        #              population — the same decision seen from two places — and
        #              one is stamped applied while the other is left lost, so
        #              the answer refuses itself. Live for ten minutes in
        #              7c46f81. Deduped on (kind, field, label): the label
        #              carries the predicate, so two facets agreeing on all
        #              three ARE the same claim.
        #   UNSTAMPED  a population that is NOT a duplicate is pulled out of the
        #              list below and never reaches `reconcile_routed_facets`,
        #              so it keeps its LOST default whatever execution did — and
        #              a lost population blocks. That is e35a01b's shape, a
        #              reclassification into a kind with no receiver, arriving on
        #              this path.
        #
        # Both are closed by MERGING into the ledger's list: a population from
        # any raiser is stamped from the same evidence by the same two calls as
        # it joins, and the merged list is the only one that reaches the receipt.
        # `test_every_routed_population_is_stamped` is what says so.
        _seen = {(f.kind, f.field_key, f.label) for f in population}
        for _extra in facets:
            if _extra.kind != receipt_mod.KIND_POPULATION:
                continue
            _key = (_extra.kind, _extra.field_key, _extra.label)
            if _key in _seen:
                continue
            _seen.add(_key)
            population.append(_extra)
            receipt_mod.reconcile_population(
                [_extra], (routed.get("metadata") or {}).get("populationApplied"),
                dataset_columns=receipt_mod.book_columns(frame))
            if (_extra.status != receipt_mod.APPLIED
                    and receipt_mod._analytical_population_satisfies(routed, _extra)):
                _extra.status, _extra.reason = receipt_mod.APPLIED, ""
        facets = [f for f in facets
                  if f.kind != receipt_mod.KIND_POPULATION] + population
        if not facets and not substitution and not granularity:
            # Nothing to adjudicate, but the answer still states what governed
            # capability produced it and as at when — the receipt is required on
            # every successful substantive answer, not only contested ones.
            receipt = receipt_mod.build_routed_receipt(
                route=route, envelope=routed, facets=[])
            routed["executionSummary"] = receipt.to_dict()
            line = receipt.render()
            if line:
                routed["answer"] = f"{(routed.get('answer') or '').rstrip()}\n\n{line}"
            return routed
        # The granularity facet joins the list BEFORE reconciliation, not after.
        # It used to be appended below, which is why its status had to be
        # written at detection: nothing ever adjudicated it. A facet whose
        # outcome is decided before execution cannot record a request that
        # SUCCEEDED, and a rule can only be enforced on a request that is
        # represented.
        if granularity:
            facets = list(facets) + list(granularity)
        _population = [f for f in facets if f.kind == receipt_mod.KIND_POPULATION]
        facets = receipt_mod.reconcile_routed_facets(
            [f for f in facets if f.kind != receipt_mod.KIND_POPULATION],
            route=route, semantics=semantics,
            available_columns=receipt_mod.book_columns(frame),
            envelope=routed)
        facets = list(facets) + _population
        # A temporal route may have compared a shorter span than the question
        # named ("since inception" answered as one month). Verified against the
        # periods the route itself declares.
        facets = receipt_mod.check_period_grain(facets, routed)
        # And the WINDOW, separately from the grain. A series at the right level
        # over fewer periods than the question named is a different defect from
        # a series at the wrong level, owed a different sentence.
        facets = receipt_mod.check_window_coverage(facets, routed, question, route)
        receipt = receipt_mod.build_routed_receipt(
            route=route, envelope=routed, facets=facets)
        verdict, message = receipt_mod.assess(receipt, substitution=substitution)
        routed["executionSummary"] = receipt.to_dict()
        routed["semanticGuard"] = {"verdict": verdict, "message": message,
                                   "route": route,
                                   "facets": [f.to_dict() for f in facets]}
        if verdict in (receipt_mod.VERDICT_REFUSE,
                       receipt_mod.VERDICT_CLARIFY):
            routed["ok"] = False
            routed["error"] = message
            routed["answer"] = message
            routed["artifacts"] = []
            routed["controlledRefusal"] = True
            routed["clarificationRequested"] = (
                verdict == receipt_mod.VERDICT_CLARIFY)
            routed.setdefault("warnings", []).append(message)
        elif verdict == receipt_mod.VERDICT_PARTIAL and message:
            routed["answer"] = f"{(routed.get('answer') or '').rstrip()}\n\n{message}"
            routed.setdefault("warnings", []).append(message)
        else:
            line = receipt.render()
            if line:
                routed["answer"] = f"{(routed.get('answer') or '').rstrip()}\n\n{line}"
    except Exception:  # noqa: BLE001 - the guard must never break a governed route
        logger.exception("routed semantic guard failed for question=%r", question)
    return routed


def _fail_closed_analytical(result: Dict[str, Any], *, question: str,
                            view: str) -> Dict[str, Any]:
    """§7 THE FAIL-CLOSED SAFETY RULE.

    A materially analytical question that no governed route claimed has reached
    the generic point-in-time executor. That executor answers from ONE snapshot
    of the funded tape with whatever measure and dimension the parse produced. It
    has no concept of a pipeline, a limit, a run rate or a forecast — so for a
    question in one of those families it cannot be right, only plausible.

    These are the four measured cases this exists to stop, all of them ``ok=True``
    with a green guard before it:

        "How many loans are we completing at the moment?"  -> 11,035 loans
        "What completion rate are we running at?"          -> £1.96bn
        "Where are we closest to our limits?"              -> WA LTV by region
        "Which of our limits are most at risk?"            -> balance by status

    The check is STRUCTURAL and runs AFTER execution, not before it, for the same
    reason the P0 receipt does: the question is not what the answer was meant to
    be, it is what the answer demonstrably carries. An answer that DOES carry the
    structure the question needs is left completely alone — "how does the front
    book compare with the back book?", answered by grouping on the seasoning
    segment with both sides present, is a real comparison reached by another
    mechanism, and refusing it would lose a capability the product has.

    Never refuses a question the boundary did not recognise as materially
    analytical. "Balance by region" is not analytical and keeps the answer it has
    always had.
    """
    if not isinstance(result, dict) or not result.get("ok"):
        return result
    try:
        from mi_workflows.analytical import intent as intent_mod

        reading = intent_mod.classify(question)
        if not reading.materially_analytical:
            return result
        spec = result.get("spec") if isinstance(result.get("spec"), dict) else {}
        evidence = {
            # The point-in-time path reads the funded/arrears view of the loan
            # tape. It never reads the governed pipeline extract.
            "dataset": view,
            # And it reads ONE governed snapshot. A cross-period answer comes
            # from a route, and a route would have claimed the question.
            "periods": 1,
            "forecast": False,
            "limits": False,
            "grouping": spec.get("dimension") or "",
            "populations": 0,
        }
        unmet = intent_mod.unmet_requirements(reading, evidence=evidence)
        if not unmet:
            return result

        message = intent_mod.refusal_message(reading, unmet)
        result["ok"] = False
        result["error"] = message
        result["answer"] = message
        result["artifacts"] = []
        result["controlledRefusal"] = True
        result.setdefault("warnings", []).append(message)
        # The receipt and the guard must tell the SAME story as the answer.
        # A green guard beside a refusal reads as a spurious refusal, and an
        # execution summary still carrying "11,035 loans" leaves on the envelope
        # the very figure the refusal says it will not substitute — a reader (or
        # a channel rendering the receipt) would find the number anyway.
        from mi_agent import execution_receipt as receipt_mod

        result["executionSummary"] = None
        result["semanticGuard"] = {
            "verdict": receipt_mod.VERDICT_REFUSE, "message": message,
            "route": None,
            "facets": [{"kind": r, "status": receipt_mod.UNAVAILABLE,
                        "label": intent_mod.REQUIREMENT_REASONS.get(r, r)}
                       for r in unmet]}
        meta = result.setdefault("metadata", {})
        if isinstance(meta, dict):
            block = reading.to_dict()
            block["unmet"] = list(unmet)
            block["failClosed"] = True
            meta["analyticalIntent"] = block
    except Exception:  # noqa: BLE001 - the boundary must never break an answer
        logger.exception("analytical fail-closed check failed for question=%r",
                         question)
    return result


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

        # P1L — GOVERNED POPULATION PROPAGATION.
        #
        # The population is resolved ONCE, here, and the specialist routes
        # receive an ALREADY-CORRECT frame rather than each re-interpreting
        # spec.filters for itself. Thirteen routes reading the same dict would
        # be thirteen chances to disagree about what "the back book" means.
        #
        # A route that resolves its frame through this seam therefore honours
        # the population automatically and reports evidence. A route that builds
        # its own frame (or reads a run artefact) reports nothing, and the P0
        # population ledger then refuses rather than letting a whole-book figure
        # answer a narrowed question.
        from mi_agent import population as _population_mod

        _predicates = _population_mod.material_predicates(
            (parsed.spec.filters if parsed is not None else None), semantics)
        _population_evidence: Dict[str, Any] = {}

        def _population_frame(cid, rid):
            frame_in = _routed_frame(cid, rid)
            if frame_in is None or not _predicates:
                return frame_in
            narrowed, ev = _population_mod.apply_population(
                frame_in, _predicates, semantics)
            _population_evidence.update(ev.to_dict())
            return narrowed

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
            frame_resolver=_population_frame,
            extra_filters=req.filters or None,
            parsed=parsed,
            base_frame_resolver=_routed_frame)
        if isinstance(routed, dict) and _population_evidence:
            routed.setdefault("metadata", {})["populationApplied"] = dict(
                _population_evidence)
    except Exception as exc:  # noqa: BLE001 - routing must never break the chat
        logger.warning("chat routing failed; using point-in-time path: %s", exc)
        routed = None
    if routed is not None:
        route = (routed.get("metadata") or {}).get("route") if isinstance(routed, dict) else None
        if isinstance(routed, dict):
            rmeta = routed.setdefault("metadata", {})
            if isinstance(rmeta, dict):
                rmeta.setdefault("parserProvenance", _parser_provenance(
                    {"metadata": {"parse_metadata": dict(getattr(parsed, "meta", {}) or {})}}))
        _stamp_routed_scope(routed, req)
        routed = _guard_routed_answer(routed, question=req.question, route=route,
                                      semantics=semantics, frame=df,
                                      parsed=parsed)
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
        # NOTE the key: ``llmConfig``, not ``llm``. The adapter puts the
        # parser's own LLM block (call count, tokens, cost) on ``metadata.llm``;
        # overwriting it here previously destroyed the only evidence of whether
        # a model was actually consulted, which let a deterministic fallback be
        # reported as an LLM result.
        meta["llmConfig"] = {"enabled": llm_cfg.enabled,
                             "available": llm_cfg.available,
                             "model": llm_cfg.model if llm_cfg.available else None,
                             "status": llm_cfg.status}
        meta.setdefault("parserProvenance", _parser_provenance(workflow))
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
    # §7 — a materially analytical question must never leave here with a
    # confident current-position figure that answers something else.
    result = _fail_closed_analytical(result, question=req.question, view=view)
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
    # The analytical capability layer composes dated snapshots and the weekly
    # pipeline extract, so the run it closed on is part of the answer for the
    # same reason.
    "analytical_composition",
}


def _route_requires_run(route: Optional[str]) -> bool:
    return bool(route) and route in _RUN_SCOPED_ROUTES
