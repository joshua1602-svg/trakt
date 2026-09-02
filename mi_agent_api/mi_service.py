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
    #: The workspace tab the caller is on. Carried for the UI and for callers
    #: that echo it back; it is NOT an input to dataset semantics. A question
    #: means the same thing on every tab — see
    #: `mi_agent_api.workspace.resolve_dataset`.
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
                      run_required: bool, semantics: Optional[Dict[str, Any]] = None,
                      frame: Any = None) -> Dict[str, Any]:
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
    _stamp_semantic_coverage(envelope, question=req.question,
                             semantics=semantics, frame=frame)
    return _enforce_model_availability(_enforce_semantic_coverage(envelope))


def _stamp_semantic_coverage(envelope: Dict[str, Any], *, question: str,
                             semantics: Optional[Dict[str, Any]],
                             frame: Any) -> None:
    """Record which governed concepts the question stated, and their disposition.

    THE ONE SEAM. Both return paths — routed and point-in-time — pass through
    `_governed_context`, so this sees every answer the service emits without a
    per-route call and without a route deciding anything about coverage.

    DISCLOSE ONLY. It publishes `metadata.semanticCoverage` and changes nothing:
    no route, no answer, no refusal. The enforcement that reads it is a separate
    change, deliberately, because the ledger's first measurement is what tells
    us whether enforcement is safe to switch on.

    Never raises into a request. A ledger that cannot be built is absent, which
    reads as "not measured" rather than "clean" — the standing F3 rule.
    """
    if semantics is None:
        return
    try:
        from question_interpretation import completeness as _coverage

        envelope["metadata"]["semanticCoverage"] = _coverage.coverage_report(
            question, envelope, semantics,
            available_values=_book_values(frame, semantics) if frame is not None else None,
            available_columns=set(frame.columns) if frame is not None else None,
            frame=frame)
    except Exception as exc:  # noqa: BLE001 - coverage must never cost an answer
        logger.info("semantic coverage unavailable: %s: %s", type(exc).__name__, exc)


#: The words a refusal uses for a concept the answer did not account for. No
#: implementation vocabulary reaches the reader: "coverage", "ledger" and
#: "unaccounted" are ours, not theirs.
_COVERAGE_REFUSAL = (
    "I understood that you asked about %s, but I could not confirm it was "
    "applied to this calculation. I have not answered over a wider population "
    "instead.")


def _enforce_semantic_coverage(envelope: Dict[str, Any]) -> Dict[str, Any]:
    """Refuse an answer that lost a governed concept the question stated.

    THE INVARIANT this whole build exists for. Opus may recover more or less of
    a sentence from run to run; what it may never do is change the analytical
    meaning silently. A concept it drops now costs ANSWERABILITY — a refusal
    naming the concept — and never a widened population.

    It fires only on `UNACCOUNTED`. A concept the estate declined and said so is
    `UNSUPPORTED` and keeps its existing governed behaviour; a concept nothing
    named was never in the ledger. There is no confidence score here, nothing is
    guessed, and no field is chosen: the rule is that a stated concept with no
    disposition may not be answered over.

    Applied to SUCCESSFUL answers only. A refusal is already a refusal, and
    re-refusing it would replace a specific governed reason with a general one.
    """
    if not envelope.get("ok"):
        return envelope
    from question_interpretation import completeness as _coverage

    missing = _coverage.unaccounted_concepts(
        (envelope.get("metadata") or {}).get("semanticCoverage"))
    if not missing:
        return envelope
    named = sorted({str(m.get("term") or m.get("value") or m.get("field"))
                    for m in missing})
    message = _COVERAGE_REFUSAL % _join_terms(named)
    envelope["ok"] = False
    envelope["error"] = message
    envelope["answer"] = message
    envelope["artifacts"] = []
    envelope.setdefault("warnings", []).append(message)
    return envelope


def _join_terms(terms: List[str]) -> str:
    if len(terms) == 1:
        return terms[0]
    return "%s and %s" % (", ".join(terms[:-1]), terms[-1])


#: The words an availability refusal uses. It says what happened and what was
#: NOT done, in the reader's vocabulary: no model name, no arm, no proposal.
_AVAILABILITY_REFUSAL = (
    "I could not complete the language-understanding step for this question, so "
    "I have not answered it. Answering from the partial reading alone risks "
    "answering a narrower question than the one you asked. Please try again.")


def _enforce_model_availability(envelope: Dict[str, Any]) -> Dict[str, Any]:
    """An unavailable augmentation call refuses; it does not quietly narrow.

    THE SECOND HALF OF THE INVARIANT. Coverage catches a concept the estate can
    NAME and execution did not carry. This catches the case coverage cannot see:
    the augmentation arm was switched on, its call did not happen or could not
    be read, and the deterministic reading — which may be narrower than the
    sentence — would otherwise be executed and published as though the whole
    question had been understood.

    Measured, on this build: with the arm on and the credit exhausted, twenty of
    twenty runs of one product-scoped question returned a whole-book answer.
    Availability changed the meaning of the answer. It may change whether Trakt
    answers; it may not change what it answers.

    The rule fires on the arm's OWN status and nothing else. A successful call
    that validly proposes no concepts reports `no_change` and is untouched — it
    is a different event from a call that did not happen, and the arm never
    infers one from the other. An arm that is switched off publishes no evidence
    and is likewise untouched.

    Successful answers only, and after coverage: where both would refuse, the
    coverage refusal names the concept, which is the more useful sentence.

    NO EXCEPTION IS ADMITTED. The estate has no completeness proof independent
    of the deterministic parse — the coverage ledger and the execution receipt
    are both built from the same owners, so neither can certify a reading whose
    gap is a term no owner names. "Size" is exactly such a term. Until an
    independent proof exists, unavailability refuses.
    """
    if not envelope.get("ok"):
        return envelope
    from . import concept_merge_arm as _arm

    evidence = (envelope.get("metadata") or {}).get("conceptMerge")
    if not isinstance(evidence, dict):
        return envelope
    if evidence.get("status") != _arm.PROPOSAL_UNAVAILABLE:
        return envelope
    envelope["ok"] = False
    envelope["error"] = _AVAILABILITY_REFUSAL
    envelope["answer"] = _AVAILABILITY_REFUSAL
    envelope["artifacts"] = []
    envelope["controlledRefusal"] = True
    envelope.setdefault("warnings", []).append(_AVAILABILITY_REFUSAL)
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


def _finish(result: GovernedResult[Dict[str, Any]],
            request: MiQueryRequest) -> GovernedResult[Dict[str, Any]]:
    """The single exit for every governed MI query.

    Emits the existing metadata-only audit line (unchanged), and records the
    governed telemetry document an operator reviews in OCC. Both are
    non-raising by construction: neither may turn an answered query into a
    failed one, and neither changes the result that is returned.

    The OCC imports are deliberately local to this function. The record is an
    OCC document written into the OCC store, so this module is the writer and
    ``operations_control`` is the owner — a one-way dependency, declared in
    deploy/trakt-mi-api/package_contents.txt so the App Service actually ships
    it. Kept function-local so importing the MI service does not pull the
    control plane in at module scope.
    """
    emit_audit_event(result)
    try:
        from operations_control import mi_query_telemetry
        from operations_control.stores import OpsStore
        mi_query_telemetry.record(
            OpsStore.from_env(), result, question=request.question,
            requested_portfolio=request.effective_portfolio_id())
    except Exception:  # noqa: BLE001 — telemetry must never fail a query
        logger.warning("mi query telemetry unavailable for request_id=%s",
                       result.request_id, exc_info=True)
    return result


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

    # THE DATASET IS THE QUESTION'S, NOT THE TAB'S.
    #
    # This used to fold `request.dataset_context` (the active React tab) in as
    # the fallback, so the same sentence was served from a different dataset
    # depending on which tab it was typed on — including
    # "the balance by seasoning segment excluding pipeline cases", served from
    # the pipeline on the pipeline tab. The tab still selects what the UI
    # DISPLAYS; it no longer decides what a question MEANS.
    view = workspace_mod.resolve_dataset(request.question)
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
        return _finish(result, request)

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
        return _finish(result, request)

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
        return _finish(result, request)

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
    return _finish(result, request)


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


def _book_values(frame, semantics):
    """The book's governed category values, or ``None`` if they cannot be read.

    Never fatal: a catalogue that cannot be built leaves the parser exactly as
    it was before it existed.
    """
    try:
        from mi_agent import execution_receipt as receipt_mod
        return receipt_mod.book_values(frame, semantics)
    except Exception:  # noqa: BLE001 - a missing catalogue is not an error
        logger.info("book value catalogue unavailable", exc_info=True)
        return None


def _owned_question(question: Optional[str], available_values) -> str:
    """``question`` with spans a governed categorical VALUE has claimed blanked.

    Delegation only — `mi_agent.categorical_spans` owns the rule. With no
    catalogue the sentence comes back unchanged.
    """
    if not question or not available_values:
        return question or ""
    try:
        from mi_agent.categorical_spans import mask_value_spans

        return mask_value_spans(question, available_values)
    except Exception:  # noqa: BLE001 - the owner missing must not change a reading
        return question


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

    It runs on a CONTROLLED NON-DELIVERY too, and for the same reason
    `_guard_unresolved_scope` does: the route said it could not produce the
    analysis, and a facet owner may have a more specific reason than the one the
    route reached for. Measured — "Show funded balance evolution by month for
    London" came back as "No reporting periods are available to build a funded
    balance trend" once that envelope stopped being success-shaped, in place of
    the geographic-scope refusal that names what the reader actually asked for.
    An execution FAILURE is still excluded: a route that broke has adjudicated
    nothing.
    """
    if not isinstance(routed, dict):
        return routed
    if not routed.get("ok") and not _is_controlled_non_delivery(routed):
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
        # The ROUTE'S OWN grain declaration, so the receipt adjudicates against
        # what was published rather than an assertion made about the route.
        granularity = receipt_mod.granularity_facets(question, route, routed)
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
                            view: str, available_values=None) -> Dict[str, Any]:
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

        # GOVERNED SPAN OWNERSHIP. The intent vocabulary owns no book field, so
        # a family word found inside a span the book has already claimed as a
        # categorical VALUE belongs to the value. Measured: brokers called
        # "Growth Partners" and "Completion Network" made every question about
        # them refuse — one as a movement question, one as a pipeline question.
        reading = intent_mod.classify(_owned_question(question, available_values))
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


def _guard_temporal_honouring(envelope: Dict[str, Any], *, question: str,
                              semantics: Dict[str, Any], frame
                              ) -> Dict[str, Any]:
    """P0 — TEMPORAL HONOURING, enforced on the object that is about to ship.

    Structural and post-execution, for the same reason `_fail_closed_analytical`
    above is: the question is not what the answer was MEANT to be, it is what
    the answer demonstrably carries.

    WHY IT IS HERE AND NOT IN EITHER GUARD
    --------------------------------------
    The artifacts do not exist yet where the two semantic guards run. The
    point-in-time guard runs inside `mi_agent_workflow` before the adapter
    renders anything, and the routed guard runs before a route's envelope is
    contextualised. A rule whose whole premise is "read the rendered rows"
    cannot live where there are no rendered rows, and moving it earlier would
    have forced it back onto the receipt — the one thing that cannot prove this.

    So it runs LAST, on both paths, and the two call sites are enumerated in
    `_run_analysis` immediately below. `test_both_paths_reach_the_temporal_guard`
    is what keeps them at two.

    Runs only on an answer that is about to STAND. An answer that already
    refuses or clarifies has nothing to discard silently, and re-adjudicating it
    would rewrite refusals this contract already gets right.
    """
    if not isinstance(envelope, dict) or not envelope.get("ok"):
        return envelope
    try:
        from mi_agent import execution_receipt as receipt_mod

        # The book's own value map, so limb 2 costs no new vocabulary: the
        # segment signal is two or more governed values named in one sentence.
        # `dimension_values` is the existing owner of what values this book
        # carries, and it is asked here rather than a word list being written.
        facets = receipt_mod.temporal_honouring_facets(
            question, envelope.get("artifacts"),
            receipt_mod.dimension_values(frame, semantics)
            if frame is not None else None)
        if not facets:
            return envelope
        # THE REFUSAL SENTENCE IS NOT WRITTEN HERE. It is produced by `assess`
        # from a receipt carrying the lost facet, so it is the same sentence the
        # eighteen refusals this surface already gets right are written in. A
        # second author of that wording would be the defect this programme
        # spent seven consolidations removing.
        receipt = receipt_mod.ExecutionReceipt(facets=list(facets))
        verdict, message = receipt_mod.assess(receipt)
        if verdict != receipt_mod.VERDICT_REFUSE or not message:
            return envelope
        envelope["ok"] = False
        envelope["error"] = message
        envelope["answer"] = message
        envelope["artifacts"] = []
        envelope["controlledRefusal"] = True
        # The receipt and the guard must tell the SAME story as the answer, and
        # the execution summary must not leave the very figure the refusal says
        # it will not substitute sitting on the envelope for a channel to render.
        envelope["executionSummary"] = None
        envelope["semanticGuard"] = {
            "verdict": verdict, "message": message,
            "route": (envelope.get("metadata") or {}).get("route"),
            "facets": [f.to_dict() for f in facets]}
        envelope.setdefault("warnings", []).append(message)
    except Exception:  # noqa: BLE001 - the guard must never break a governed route
        logger.exception("temporal honouring guard failed for question=%r",
                         question)
    return envelope


def _is_controlled_non_delivery(envelope: Dict[str, Any]) -> bool:
    """A route said "I could not produce this" — not "here it is", not "it broke".

    THE SCOPE DISCLOSURE OUTRANKS A DATA-AVAILABILITY MESSAGE. These guards used
    to test `ok` alone, which was sound while a route that could not deliver
    still returned `ok=True`: the guard ran, saw the unheld portfolio, and
    replaced the generic message with the specific one. Once such an envelope
    became the `ok:false` controlled refusal it always should have been, the
    guard stopped running and the reader was told

        "I can't build a funded balance bridge yet: at least two funded
         reporting periods are needed for a bridge."

    about a book this platform has never onboarded — true, and not the thing
    they needed to know. So the precondition is now "was this DELIVERED, or was
    it a controlled non-delivery whose reason a more specific owner can improve".

    An execution FAILURE is deliberately excluded: `_execution_failure_envelope`
    carries no `controlledUnsupported`, and a route that broke has not reasoned
    about scope at all.
    """
    meta = envelope.get("metadata") or {}
    return bool(meta.get("controlledUnsupported")) and not meta.get("executionFailure")


def _guard_unknown_category(envelope: Dict[str, Any]) -> Dict[str, Any]:
    """A category the question NAMED and no governed field carries.

    The point-in-time workflow has refused this since it was written. A ROUTED
    answer did not: the comparison recogniser collected the unresolved note and
    published nothing, so "which region added the most balance last month for
    Atlantis loans?" returned the UNFILTERED whole-book ranking with ok=true and
    the dropped qualifier unmentioned. Which route happens to claim a question
    is not a fact about whether its qualifier resolved — the same reasoning
    `_guard_unresolved_scope` records for portfolio scope.

    THE SENTENCE IS NOT WRITTEN HERE. `llm_query_parser.unknown_category_refusal`
    owns it and the workflow calls the same function, so one obstacle cannot be
    described two ways.
    """
    if not isinstance(envelope, dict) or not envelope.get("ok"):
        return envelope
    from mi_agent import llm_query_parser as _parser

    notes = ((envelope.get("spec") or {}).get("unavailable_filters")) or []
    message = _parser.unknown_category_refusal(notes)
    if not message:
        return envelope
    envelope["ok"] = False
    envelope["error"] = message
    envelope["answer"] = message
    envelope["artifacts"] = []
    envelope["controlledRefusal"] = True
    envelope.setdefault("warnings", []).append(message)
    meta = envelope.setdefault("metadata", {})
    if isinstance(meta, dict):
        meta["controlledRefusal"] = True
        meta["controlledUnsupported"] = True
    return envelope


def _guard_unresolved_scope(envelope: Dict[str, Any], *, question: str,
                            semantics: Dict[str, Any], frame) -> Dict[str, Any]:
    """PHASE 1E. A portfolio the question NAMED and the registry does not hold.

    Applied at BOTH answer sites, for the reason Phase 0 recorded as a
    governance prerequisite: a receipt proof that holds only on the routed path
    is not a proof. Measured with the routed guard alone in place:

        "What is the funded balance by region for the Highgate Mortgages Book?"
        -> ok=True, "Total Balance, grouped by Region, 12 groups, 11,035 loans"

    — the whole book, under the name of a book this platform has never
    onboarded, because the question fell through to the point-in-time path and
    the routed guard never saw it. Which route happens to claim a question is
    not a fact about whether its scope resolved.

    THE REFUSAL SENTENCE IS NOT WRITTEN HERE. `unresolved_scope_facets` raises
    the request as a LOST narrowing and `assess` produces the wording, so this
    refusal reads exactly like every other dropped-narrowing refusal on this
    surface. A second author of that sentence would be the defect this
    programme spent seven consolidations removing.

    The registry is built from THE FRAME THIS ANSWER WAS COMPUTED FROM, never
    from the process-wide active dataset: `build_registry()` with no frame
    calls `active_frame()`, which populates a TTL cache, and a disclosure step
    has no business changing what the next request reads.
    """
    if not isinstance(envelope, dict):
        return envelope
    if not envelope.get("ok") and not _is_controlled_non_delivery(envelope):
        return envelope
    try:
        from mi_agent import execution_receipt as receipt_mod

        from . import portfolio_context as _ctx_registry

        # The book's own value map, so a scope this tape carries as a VALUE
        # ("the London book", "the South East book") is read as the population
        # it is rather than as a portfolio nobody onboarded. The lens layer has
        # no vocabulary for that; this guard does, and it is where the refusal
        # is decided.
        facets = receipt_mod.unresolved_scope_facets(
            question, registry=_ctx_registry.build_registry(frame),
            known_values=(receipt_mod.dimension_values(frame, semantics)
                          if frame is not None else None))
        if not facets:
            return envelope
        receipt = receipt_mod.ExecutionReceipt(facets=list(facets))
        verdict, message = receipt_mod.assess(receipt)
        if verdict != receipt_mod.VERDICT_REFUSE or not message:
            return envelope
        envelope["ok"] = False
        envelope["error"] = message
        envelope["answer"] = message
        envelope["artifacts"] = []
        envelope["controlledRefusal"] = True
        # The receipt and the guard must tell the SAME story as the answer, and
        # the execution summary must not leave the very figure the refusal says
        # it will not substitute sitting on the envelope for a channel to render.
        envelope["executionSummary"] = None
        envelope["semanticGuard"] = {
            "verdict": verdict, "message": message,
            "route": (envelope.get("metadata") or {}).get("route"),
            "facets": [f.to_dict() for f in facets]}
        envelope.setdefault("warnings", []).append(message)
        meta = envelope.setdefault("metadata", {})
        if isinstance(meta, dict):
            meta["controlledRefusal"] = True
            meta["lensApplied"] = False
    except Exception as exc:  # noqa: BLE001 - never break a governed answer
        logger.info("unresolved-scope disclosure skipped: %s", exc)
    return envelope


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
        # GOVERNED SPAN OWNERSHIP for the DATASET, once the book can be read.
        #
        # The dataset is decided at the top of this request, before any
        # authorisation and therefore before any tape can be opened, so the owner
        # answers there without the one piece of evidence this rule needs: the
        # book's own categorical values. Measured, a broker called "Pipeline
        # Mortgage Club" served every question about it from the pipeline
        # extract — 8 cases in place of its 63 funded loans.
        #
        # This is the SAME owner asked once more, with the evidence it lacked;
        # it is not a second decision. It runs only where the first answer was
        # NOT the default dataset, and it can only ever return to the default —
        # a question that names a dataset outside every claimed span keeps its
        # answer. The funded frame is loaded to read the catalogue and then used
        # as the request's frame, so nothing is loaded twice.
        if view != workspace_mod.DEFAULT_VIEW:
            base_df, base_error = _resolve_frame(
                ds, workspace_mod.DEFAULT_VIEW, portfolio_id)
            # The masking is applied to the QUESTION, not handed to the owner:
            # `resolve_dataset` takes one argument and a guard exists to keep it
            # that way. Same owner, same signature, a sentence it is entitled to
            # read.
            owned_view = workspace_mod.resolve_dataset(_owned_question(
                req.question,
                _book_values(base_df, semantics) if base_df is not None else None))
            if owned_view != view:
                logger.info("dataset %r re-read as %r under span ownership",
                            view, owned_view)
                view, df, frame_error = owned_view, base_df, base_error

    try:
        with _perf.stage("mi_query.parse"):
            parsed = ParsedQuestion.parse(
                req.question, semantics,
                available_columns=set(df.columns) if df is not None else None,
                # THE BOOK'S OWN CATEGORY VALUES. Without them the parser has
                # no way to tell which governed field a named category belongs
                # to, and bound every one to geography.
                available_values=(_book_values(df, semantics)
                                  if df is not None else None),
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

        # B17: the caller-supplied drill-through filters are merged BEFORE the
        # predicates are computed, not after.
        #
        # `try_route` calls `parsed.merge_filters(extra_filters)` itself, so the
        # drill was on the spec by the time `population_facets(spec)` read it —
        # and absent from the predicates the frame resolver narrows on. Raised,
        # never applied, refused: every drill-through on a routed question came
        # back "the population collateral_geography = South East … could not be
        # applied", fail-closed and correct in outcome, wrong in cause.
        #
        # Merging here is idempotent, so the call inside `try_route` still stands
        # and this does not become a second owner of the merge.
        if parsed is not None and req.filters:
            parsed.merge_filters(req.filters)
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
    # WHAT THE CONCEPT-MERGE ARM DID, on whichever envelope this request
    # produces. The arm runs inside routing (it needs the interpretation, which
    # routing builds) and carries its evidence out on the parse metadata,
    # because the point-in-time path returns no routed envelope to stamp and
    # executes the very spec the arm changed.
    _concept_merge = (getattr(parsed, "meta", None) or {}).get("conceptMerge")

    if routed is not None:
        route = (routed.get("metadata") or {}).get("route") if isinstance(routed, dict) else None
        if isinstance(routed, dict):
            rmeta = routed.setdefault("metadata", {})
            if isinstance(rmeta, dict):
                rmeta.setdefault("parserProvenance", _parser_provenance(
                    {"metadata": {"parse_metadata": dict(getattr(parsed, "meta", {}) or {})}}))
                if _concept_merge is not None:
                    rmeta["conceptMerge"] = _concept_merge
        _stamp_routed_scope(routed, req)
        routed = _guard_routed_answer(routed, question=req.question, route=route,
                                      semantics=semantics, frame=df,
                                      parsed=parsed)
        # P0 SITE 1 OF 2 — temporal honouring, on the rendered routed envelope.
        routed = _guard_temporal_honouring(routed, question=req.question,
                                          semantics=semantics, frame=df)
        # PHASE 1E SITE 1 OF 2 — an unresolved portfolio scope, on the same terms.
        routed = _guard_unresolved_scope(routed, question=req.question,
                                         semantics=semantics, frame=df)
        # SITE 1 OF 2 — a named category this book does not carry.
        routed = _guard_unknown_category(routed)
        return _governed_context(routed, req=req, client_id=client_id, run_id=run_id,
                                 view=view, run_required=_route_requires_run(route),
                                 semantics=semantics, frame=df)

    # ---- point-in-time: active governed dataset (or the selected run) ----- #
    if frame_error:
        return _error_envelope(frame_error, req=req, view=view)
    if df is None:
        return _error_envelope("Could not load the governed data for this query.",
                               req=req, view=view)

    currency_mod.resolve_and_set(df, client_id=client_id)

    llm_enabled, llm_model = llm_cfg.enabled, llm_cfg.model
    runner = deps.query_runner or run_mi_agent_query
    try:
        workflow = runner(
            req.question, df, str(semantics_path()),
            parser_mode="llm" if llm_enabled else "deterministic",
            llm_enabled=llm_enabled, model=llm_model,
            extra_filters=req.filters or None,
            source_portfolio_lens=req.source_portfolio_lens or None,
            # THE DATASET THIS FRAME CAME FROM. The receipt describes an
            # unfiltered population, and with nothing to describe it WITH it
            # said "entire funded portfolio" whatever had been read — so a
            # pipeline figure was published under the funded book's name.
            dataset=view,
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
        if _concept_merge is not None:
            meta["conceptMerge"] = _concept_merge
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
    result = _fail_closed_analytical(result, question=req.question, view=view,
                                     available_values=_book_values(df, semantics)
                                     if df is not None else None)
    # P0 SITE 2 OF 2 — temporal honouring, on the rendered point-in-time result.
    result = _guard_temporal_honouring(result, question=req.question,
                                       semantics=semantics, frame=df)
    # PHASE 1E SITE 2 OF 2 — an unresolved portfolio scope, on the same terms.
    result = _guard_unknown_category(result)
    result = _guard_unresolved_scope(result, question=req.question,
                                     semantics=semantics, frame=df)
    # A point-in-time answer is run-scoped only when a run was explicitly selected.
    return _governed_context(result, req=req, client_id=client_id, run_id=run_id,
                             semantics=semantics, frame=df,
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
