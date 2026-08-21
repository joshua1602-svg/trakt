"""operations_control.occ_agent.api — the OCC Agent routes.

A router mounted into the **existing** Operations Control API, behind the
existing operator authentication and the existing tenancy binding. There is no
second application and no second auth mechanism.

Three controls sit in front of every route:

* the feature flag (``OCC_AGENT_SYNTHETIC_ENABLED``) — the router is only
  mounted when it is set, and :func:`_require_feature` refuses even then if it
  is turned off at runtime;
* the operator principal from :func:`operations_control.api.auth.authenticate`;
* the tenant binding — the tenant is taken from the principal's client binding,
  never from the request body, and a case belonging to another tenant answers
  404 exactly as the live routes do.

The onboarding half of a practice case is deliberately *not* re-exposed here in
full: it already has routes, screens and a wizard under ``/ops/onboarding``, and
this router links to them rather than shadowing them. What it adds is the
conversation, the practice execution, and the readiness verdict.

Errors are :class:`operations_control.engine.OpsError` subclasses, so the app's
existing exception handler returns them in the existing operator-safe envelope
with no new error shape.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from pydantic import BaseModel

from ..api.auth import Principal, authenticate
from ..engine import OpsError
from . import fixtures as _fixtures
from . import states as _states
from .policy import FEATURE_FLAG_ENV, feature_enabled
from .service import AgentCase, OccAgentService

router = APIRouter(prefix="/ops/agent", tags=["occ-agent"])

#: Set by :func:`configure` at mount time. Injected rather than constructed per
#: request so tests can supply a tmp-dir-backed service.
_service: Optional[OccAgentService] = None


def configure(service: OccAgentService) -> None:
    global _service
    _service = service


def get_service() -> OccAgentService:
    if _service is None:  # pragma: no cover — configure() runs at mount
        raise OpsError("OCC_AGENT_UNAVAILABLE",
                       "The OCC Agent is not available.",
                       http_status=503)
    return _service


def _require_feature() -> None:
    if not feature_enabled():
        raise OpsError(
            "OCC_AGENT_DISABLED",
            "The OCC Agent is not switched on for this environment.",
            http_status=404)


def _tenant_for(principal: Principal, requested: Optional[str] = None) -> str:
    """The tenant this principal operates practice cases under.

    Taken from the principal's own binding. A principal bound to several clients
    may name which one; a name outside the binding is refused as not-found, so
    another tenant's namespace is never revealed.
    """
    bound = [c for c in principal.clients if c != "*"]
    if requested:
        if not principal.allows(requested):
            raise HTTPException(status_code=404, detail={
                "errorCode": "OPS_NOT_FOUND",
                "message": "That could not be found."})
        return requested
    if bound:
        return bound[0]
    if "*" in principal.clients:
        # An unrestricted operator still works inside a named tenant namespace,
        # so practice cases are never filed against a wildcard.
        return "synthetic"
    raise HTTPException(status_code=404, detail={
        "errorCode": "OPS_NOT_FOUND", "message": "That could not be found."})


def _load(service: OccAgentService, tenant: str, case_ref: str) -> AgentCase:
    return service.load(tenant, case_ref)


# --------------------------------------------------------------------------- #
# Request models
# --------------------------------------------------------------------------- #

class CreateCase(BaseModel):
    instruction: str = ""
    tenant: Optional[str] = None
    fixture_id: str = ""


class Instruct(BaseModel):
    text: str
    confirm: bool = False
    tenant: Optional[str] = None


class SaveStep(BaseModel):
    step: str
    payload: Dict[str, Any] = {}
    tenant: Optional[str] = None


class RequestInformation(BaseModel):
    items: Optional[List[Dict[str, Any]]] = None
    due_date: str = ""
    note: str = ""
    tenant: Optional[str] = None


class RecordResponse(BaseModel):
    request_id: str
    answers: Dict[str, Any] = {}
    note: str = ""
    accept: bool = True
    tenant: Optional[str] = None


class DecisionAnswer(BaseModel):
    decision_id: str
    action: str                            # approve | amend | reject
    value: str = ""
    reason: str = ""
    tenant: Optional[str] = None


class RunTarget(BaseModel):
    """Which delivery a practice run is for.

    Standing configuration has no concept of a period, so it is named here.
    """

    portfolio_id: str = ""
    dataset: str = ""
    reporting_period: str = ""
    tenant: Optional[str] = None


class TenantBody(BaseModel):
    tenant: Optional[str] = None
    reason: str = ""


class LoadFixture(BaseModel):
    fixture_id: str
    tenant: Optional[str] = None
    #: Drive the whole scenario, rather than only seeding its artefacts.
    run: bool = True


class ClientResponse(BaseModel):
    """A structured client submission.

    ``answers`` is keyed by authoritative catalogue keys — ``section.field`` or
    ``section[index].field``. A key the catalogue does not declare, or one the
    client was not asked, is refused rather than stored: there is deliberately
    no free-text lane into a governed case.
    """

    answers: Dict[str, Any] = {}
    request_id: str = ""
    #: When false, answers to questions the client was not asked are reported
    #: and skipped rather than refusing the whole submission.
    strict: bool = True
    tenant: Optional[str] = None


class SendPack(BaseModel):
    """Issue the approved pack. ``to`` overrides the recorded contacts."""

    to: Optional[List[str]] = None
    tenant: Optional[str] = None


class IngestMail(BaseModel):
    """Take named replies into this case.

    ``message_ids`` are the mailbox identifiers reported by ``GET
    /cases/{case_ref}/mail``. There is deliberately no "ingest everything
    waiting": an operator chooses which replies belong to the case they are
    looking at, and the correlator refuses any that it cannot tie there by
    evidence.
    """

    message_ids: List[str] = []
    tenant: Optional[str] = None


class ConfirmActivation(BaseModel):
    """The last, separate act before anything reaches production.

    ``confirmation`` is the operator's own words, recorded on the audit event.
    Approving the configuration is a different call, and does not start this.
    """

    confirmation: str = ""
    tenant: Optional[str] = None


# --------------------------------------------------------------------------- #
# Meta
# --------------------------------------------------------------------------- #

@router.get("/meta")
def meta(principal: Principal = Depends(authenticate)) -> Dict[str, Any]:
    """What the tab needs to render itself: the lifecycle, and the scenarios."""
    _require_feature()
    service = get_service()
    return {"ok": True,
            "enabled": True,
            "flag": FEATURE_FLAG_ENV,
            "runtime_mode": service.policy.runtime_mode,
            "policy": service.policy.to_dict(),
            "lifecycle": _states.lifecycle(),
            "onboarding_reference": service.onboarding.reference(),
            "scenarios": _fixtures.catalogue(),
            "tenant": _tenant_for(principal)}


# --------------------------------------------------------------------------- #
# Cases
# --------------------------------------------------------------------------- #

@router.get("/cases")
def list_cases(state: Optional[str] = None,
               tenant: Optional[str] = None,
               principal: Principal = Depends(authenticate)) -> Dict[str, Any]:
    _require_feature()
    service = get_service()
    return {"ok": True,
            "cases": service.list_cases(_tenant_for(principal, tenant),
                                        state=state)}


@router.post("/cases")
def create_case(body: CreateCase,
                principal: Principal = Depends(authenticate)) -> Dict[str, Any]:
    _require_feature()
    service = get_service()
    case = service.create_case(tenant=_tenant_for(principal, body.tenant),
                               initiating_user=principal.name,
                               instruction=body.instruction,
                               fixture_id=body.fixture_id)
    return {"ok": True, **service.status(case)}


@router.get("/cases/{case_ref}")
def get_case(case_ref: str, tenant: Optional[str] = None,
             principal: Principal = Depends(authenticate)) -> Dict[str, Any]:
    _require_feature()
    service = get_service()
    return {"ok": True,
            **service.status(_load(service, _tenant_for(principal, tenant),
                                   case_ref))}


@router.get("/cases/{case_ref}/audit")
def get_audit(case_ref: str, tenant: Optional[str] = None,
              principal: Principal = Depends(authenticate)) -> Dict[str, Any]:
    _require_feature()
    service = get_service()
    t = _tenant_for(principal, tenant)
    agent_case = _load(service, t, case_ref)   # tenancy check before any read
    return {"ok": True, "events": service.store.list_audit(t, case_ref),
            "chain_intact": service.store.verify_audit_chain(t, case_ref),
            "onboarding_events": agent_case.case.events}


@router.get("/cases/{case_ref}/preview")
def get_preview(case_ref: str, tenant: Optional[str] = None,
                principal: Principal = Depends(authenticate)) -> Dict[str, Any]:
    """What activation WOULD create. It creates nothing."""
    _require_feature()
    service = get_service()
    agent_case = _load(service, _tenant_for(principal, tenant), case_ref)
    return {"ok": True, "preview": service.preview(agent_case),
            "written": False, "execution_status": "not_activated"}


@router.get("/cases/{case_ref}/checklist")
def get_checklist(case_ref: str, tenant: Optional[str] = None,
                  principal: Principal = Depends(authenticate)) -> Dict[str, Any]:
    """What the client still has to tell us. Client Onboarding's own list."""
    _require_feature()
    service = get_service()
    agent_case = _load(service, _tenant_for(principal, tenant), case_ref)
    return {"ok": True,
            "checklist": service.onboarding.client_checklist(agent_case.case),
            "requests": agent_case.case.information_requests}


@router.get("/cases/{case_ref}/readiness")
def get_readiness(case_ref: str, tenant: Optional[str] = None,
                  principal: Principal = Depends(authenticate)) -> Dict[str, Any]:
    _require_feature()
    service = get_service()
    agent_case = _load(service, _tenant_for(principal, tenant), case_ref)
    # Readiness is a waypoint the run passes THROUGH, so the package is offered
    # from the status it recorded rather than from where the run is now — a
    # case that has gone on to review still has a readiness package.
    ready = agent_case.run.readiness_status == _states.READY_FOR_EXECUTION
    return {"ok": True, "readiness": service.evaluate_readiness(agent_case),
            "package": service.readiness_package(agent_case) if ready else None}


# --------------------------------------------------------------------------- #
# The conversation
# --------------------------------------------------------------------------- #

@router.post("/cases/{case_ref}/instruct")
def instruct(case_ref: str, body: Instruct,
             principal: Principal = Depends(authenticate)) -> Dict[str, Any]:
    _require_feature()
    service = get_service()
    agent_case = _load(service, _tenant_for(principal, body.tenant), case_ref)
    result = service.instruct(agent_case, text=body.text, actor=principal.name,
                              confirm=body.confirm)
    return {"ok": True, "reply": result.reply, "proposal": result.proposal,
            "applied": result.applied, **service.status(result.case)}


# --------------------------------------------------------------------------- #
# The onboarding half
# --------------------------------------------------------------------------- #

@router.post("/cases/{case_ref}/steps")
def save_step(case_ref: str, body: SaveStep,
              principal: Principal = Depends(authenticate)) -> Dict[str, Any]:
    """Answer one wizard step directly, through Client Onboarding itself."""
    _require_feature()
    service = get_service()
    agent_case = _load(service, _tenant_for(principal, body.tenant), case_ref)
    agent_case.case = service.onboarding.save_step(
        case_id=case_ref, step=body.step, payload=body.payload,
        by=principal.name)
    return {"ok": True, **service.status(agent_case)}


@router.post("/cases/{case_ref}/information-requests")
def request_information(case_ref: str, body: RequestInformation,
                       principal: Principal = Depends(authenticate)
                       ) -> Dict[str, Any]:
    _require_feature()
    service = get_service()
    agent_case = _load(service, _tenant_for(principal, body.tenant), case_ref)
    return {"ok": True,
            **service.status(service.request_client_information(
                agent_case, actor=principal.name, items=body.items,
                due_date=body.due_date, note=body.note))}


@router.post("/cases/{case_ref}/information-requests/respond")
def record_response(case_ref: str, body: RecordResponse,
                    principal: Principal = Depends(authenticate)
                    ) -> Dict[str, Any]:
    _require_feature()
    service = get_service()
    agent_case = _load(service, _tenant_for(principal, body.tenant), case_ref)
    return {"ok": True,
            **service.status(service.record_client_response(
                agent_case, request_id=body.request_id, actor=principal.name,
                answers=body.answers, note=body.note, accept=body.accept))}


@router.post("/cases/{case_ref}/submit")
def submit_for_approval(case_ref: str, body: TenantBody,
                        principal: Principal = Depends(authenticate)
                        ) -> Dict[str, Any]:
    _require_feature()
    service = get_service()
    agent_case = _load(service, _tenant_for(principal, body.tenant), case_ref)
    return {"ok": True,
            **service.status(service.submit_for_approval(
                agent_case, actor=principal.name))}


@router.post("/cases/{case_ref}/approve")
def approve_onboarding(case_ref: str, body: TenantBody,
                       principal: Principal = Depends(authenticate)
                       ) -> Dict[str, Any]:
    """Approve the onboarding. Records the decision; creates nothing."""
    _require_feature()
    service = get_service()
    agent_case = _load(service, _tenant_for(principal, body.tenant), case_ref)
    return {"ok": True,
            **service.status(service.approve_onboarding(
                agent_case, actor=principal.name, reason=body.reason))}


@router.post("/cases/{case_ref}/request-changes")
def request_changes(case_ref: str, body: TenantBody,
                    principal: Principal = Depends(authenticate)
                    ) -> Dict[str, Any]:
    _require_feature()
    service = get_service()
    agent_case = _load(service, _tenant_for(principal, body.tenant), case_ref)
    return {"ok": True,
            **service.status(service.request_changes(
                agent_case, actor=principal.name,
                reason=body.reason or "Changes requested."))}


# --------------------------------------------------------------------------- #
# The execution half
# --------------------------------------------------------------------------- #

@router.post("/cases/{case_ref}/target")
def set_run_target(case_ref: str, body: RunTarget,
                   principal: Principal = Depends(authenticate)) -> Dict[str, Any]:
    """Name which delivery this practice run is for."""
    _require_feature()
    service = get_service()
    agent_case = _load(service, _tenant_for(principal, body.tenant), case_ref)
    run = agent_case.run
    if body.portfolio_id:
        run.portfolio_id = body.portfolio_id
    if body.dataset:
        run.dataset = body.dataset
    if body.reporting_period:
        run.reporting_period = body.reporting_period
    run.facts = service.facts(agent_case).to_dict()
    service.store.save(run)
    return {"ok": True, **service.status(agent_case)}


@router.post("/cases/{case_ref}/artefacts")
async def upload_artefacts(case_ref: str,
                           files: List[UploadFile] = File(...),
                           tenant: Optional[str] = Form(default=None),
                           principal: Principal = Depends(authenticate)
                           ) -> Dict[str, Any]:
    """Upload practice artefacts.

    There is deliberately no way to name a storage location: the intended live
    URI is derived server-side from the onboarding case's own identity, and
    nothing is written to it.
    """
    _require_feature()
    service = get_service()
    agent_case = _load(service, _tenant_for(principal, tenant), case_ref)
    for upload in files:
        data = await upload.read()
        agent_case = service.register_synthetic_artefact(
            agent_case, filename=upload.filename or "", data=data,
            actor=principal.name)
    return {"ok": True,
            **service.status(service.classify_artefacts(
                agent_case, actor=principal.name))}


@router.post("/cases/{case_ref}/artefacts/fixture")
def load_fixture_artefacts(case_ref: str, body: LoadFixture,
                           principal: Principal = Depends(authenticate)
                           ) -> Dict[str, Any]:
    """Attach a repository fixture's files as this case's client response."""
    _require_feature()
    service = get_service()
    agent_case = _load(service, _tenant_for(principal, body.tenant), case_ref)
    scenario = _fixtures.by_id(body.fixture_id)
    for spec in scenario.files:
        agent_case = service.register_synthetic_artefact(
            agent_case, filename=spec.filename,
            data=spec.content.encode("utf-8"), actor=principal.name,
            fixture_id=scenario.fixture_id, declared_type=spec.declared_type)
    return {"ok": True,
            **service.status(service.classify_artefacts(
                agent_case, actor=principal.name))}


@router.post("/cases/{case_ref}/responses/generate")
def generate_answers(case_ref: str, body: TenantBody,
                     principal: Principal = Depends(authenticate)
                     ) -> Dict[str, Any]:
    """Answer this case's outstanding client questions, synthetically.

    Distinct from ``/artefacts/generate``, which makes up FILES. Both exist
    because "receive client responses" and "receive required artefacts" are
    two different steps, and a generated loan tape does nothing for a checklist
    of unanswered questions.
    """
    _require_feature()
    service = get_service()
    agent_case = _load(service, _tenant_for(principal, body.tenant), case_ref)
    return {"ok": True,
            **service.status(service.generate_synthetic_answers(
                agent_case, actor=principal.name))}


@router.post("/cases/{case_ref}/artefacts/generate")
def generate_response(case_ref: str, body: TenantBody,
                      principal: Principal = Depends(authenticate)
                      ) -> Dict[str, Any]:
    """Generate a client response for this case's own requirements."""
    _require_feature()
    service = get_service()
    agent_case = _load(service, _tenant_for(principal, body.tenant), case_ref)
    return {"ok": True,
            **service.status(service.generate_synthetic_response(
                agent_case, actor=principal.name))}


@router.post("/cases/{case_ref}/run")
def run_onboarding(case_ref: str, body: TenantBody,
                   principal: Principal = Depends(authenticate)) -> Dict[str, Any]:
    _require_feature()
    service = get_service()
    agent_case = _load(service, _tenant_for(principal, body.tenant), case_ref)
    return {"ok": True,
            **service.status(service.run_synthetic_onboarding(
                agent_case, actor=principal.name))}


@router.post("/cases/{case_ref}/decisions")
def answer_decision(case_ref: str, body: DecisionAnswer,
                    principal: Principal = Depends(authenticate)) -> Dict[str, Any]:
    _require_feature()
    service = get_service()
    agent_case = _load(service, _tenant_for(principal, body.tenant), case_ref)
    return {"ok": True,
            **service.status(service.resolve_decision(
                agent_case, decision_id=body.decision_id, action=body.action,
                value=body.value, reason=body.reason, actor=principal.name))}


@router.post("/cases/{case_ref}/plan")
def generate_plan(case_ref: str, body: TenantBody,
                  principal: Principal = Depends(authenticate)) -> Dict[str, Any]:
    _require_feature()
    service = get_service()
    agent_case = _load(service, _tenant_for(principal, body.tenant), case_ref)
    return {"ok": True,
            **service.status(service.generate_orchestration_plan(
                agent_case, actor=principal.name))}


@router.post("/cases/{case_ref}/readiness/approve")
def approve_readiness(case_ref: str, body: TenantBody,
                      principal: Principal = Depends(authenticate)
                      ) -> Dict[str, Any]:
    _require_feature()
    service = get_service()
    agent_case = _load(service, _tenant_for(principal, body.tenant), case_ref)
    return {"ok": True,
            **service.status(service.approve_execution_readiness(
                agent_case, actor=principal.name))}


# --------------------------------------------------------------------------- #
# The client pack
# --------------------------------------------------------------------------- #

@router.post("/cases/{case_ref}/pack/draft")
def draft_pack(case_ref: str, body: TenantBody,
               principal: Principal = Depends(authenticate)) -> Dict[str, Any]:
    _require_feature()
    service = get_service()
    agent_case = _load(service, _tenant_for(principal, body.tenant), case_ref)
    return {"ok": True,
            **service.status(service.draft_pack(agent_case,
                                                actor=principal.name))}


@router.get("/cases/{case_ref}/pack")
def get_pack(case_ref: str, tenant: Optional[str] = None,
             principal: Principal = Depends(authenticate)) -> Dict[str, Any]:
    """The pack as it stands, plus the document a human would actually read."""
    _require_feature()
    service = get_service()
    t = _tenant_for(principal, tenant)
    agent_case = _load(service, t, case_ref)
    built = service.build_pack(agent_case)
    return {"ok": True, "pack": built.to_dict(), "document": built.document(),
            "status": agent_case.run.pack_status,
            "history": agent_case.run.pack_history,
            "receipt": agent_case.run.pack_receipt}


@router.post("/cases/{case_ref}/pack/approve")
def approve_pack(case_ref: str, body: TenantBody,
                 principal: Principal = Depends(authenticate)) -> Dict[str, Any]:
    _require_feature()
    service = get_service()
    agent_case = _load(service, _tenant_for(principal, body.tenant), case_ref)
    return {"ok": True,
            **service.status(service.approve_pack(
                agent_case, actor=principal.name, reason=body.reason))}


@router.post("/cases/{case_ref}/pack/send")
def send_pack(case_ref: str, body: SendPack,
              principal: Principal = Depends(authenticate)) -> Dict[str, Any]:
    _require_feature()
    service = get_service()
    agent_case = _load(service, _tenant_for(principal, body.tenant), case_ref)
    return {"ok": True,
            **service.status(service.send_pack(agent_case,
                                               actor=principal.name,
                                               to=body.to))}


# --------------------------------------------------------------------------- #
# The client's reply
# --------------------------------------------------------------------------- #

def _mail_reader() -> Any:
    """The reply reader, or ``None`` when this deployment does not carry it.

    Imported lazily and behind a guard for the same reason the outbound adapter
    is built that way in :mod:`operations_control.api.app`: ``trakt_mail``
    depends on this package and not the other way round, so importing it at
    module level would be a cycle — and a deployment that ships without it, or
    without ``reportlab``, must still serve every other route rather than
    failing to start.
    """
    try:
        from trakt_mail import ingest as _ingest
    except Exception:  # noqa: BLE001 — an absent integration is not an error
        return None
    return _ingest


#: What the routes say when the package is absent. Distinct from "reading is
#: not enabled", which is a setting, and from "nothing has arrived", which is
#: an answer.
_NO_READER = "This deployment does not carry the mailbox reader."


@router.get("/cases/{case_ref}/mail")
def waiting_mail(case_ref: str, tenant: Optional[str] = None,
                 principal: Principal = Depends(authenticate)
                 ) -> Dict[str, Any]:
    """What has arrived in the OCC mailbox, and what it belongs to.

    Read-only in both systems: nothing is written to a case and nothing is
    marked read, so an operator may look as often as they like. Messages the
    correlator could not tie to a case are reported too — that is a question
    for a person, and hiding it would leave a client's reply sitting unread.
    """
    _require_feature()
    service = get_service()
    resolved = _tenant_for(principal, tenant)
    agent_case = _load(service, resolved, case_ref)
    reader = _mail_reader()
    if reader is None:
        return {"ok": True, "case_ref": agent_case.case_ref,
                "mail": {"messages": [], "matched": 0, "unmatched": 0,
                         "for_this_case": [], "note": _NO_READER}}
    try:
        found = reader.waiting(service, tenant=resolved)
    except reader.MailIngestError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from None
    return {"ok": True,
            "case_ref": agent_case.case_ref,
            "mail": {**found.to_dict(),
                     "for_this_case": [w.to_dict()
                                       for w in found.for_case(case_ref)]}}


@router.post("/cases/{case_ref}/mail/ingest")
def ingest_mail(case_ref: str, body: IngestMail,
                principal: Principal = Depends(authenticate)
                ) -> Dict[str, Any]:
    """Take the named replies into this case.

    Attachments are registered through the same governed method an operator's
    own upload uses. The client's WORDS are recorded and left alone: applying
    them to the case is an instruction a human gives, with the interpreter
    showing its reading first, exactly as if they had typed it.
    """
    _require_feature()
    service = get_service()
    resolved = _tenant_for(principal, body.tenant)
    agent_case = _load(service, resolved, case_ref)
    if not body.message_ids:
        raise HTTPException(status_code=400,
                            detail="Name at least one message to take in.")
    reader = _mail_reader()
    if reader is None:
        raise HTTPException(status_code=501, detail=_NO_READER)
    try:
        found = reader.waiting(service, tenant=resolved)
    except reader.MailIngestError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from None

    by_id = {w.message.graph_id: w for w in found.messages}
    results: List[Dict[str, Any]] = []
    for message_id in body.message_ids:
        chosen = by_id.get(message_id)
        if chosen is None:
            raise HTTPException(
                status_code=404,
                detail=(f"Message {message_id!r} is no longer in the mailbox "
                        "folder this deployment reads."))
        try:
            agent_case, outcome = reader.ingest(
                service, agent_case, chosen, actor=principal.name)
        except reader.MailIngestError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from None
        results.append(outcome.to_dict())
    return {"ok": True, "ingested": results,
            **service.status(agent_case)}


@router.get("/cases/{case_ref}/form")
def get_client_form(case_ref: str, tenant: Optional[str] = None,
                    principal: Principal = Depends(authenticate)
                    ) -> Dict[str, Any]:
    """The structured form the client should see now.

    Progressive and conditional: a step whose trigger has not happened is
    reported as locked rather than served empty.
    """
    _require_feature()
    service = get_service()
    agent_case = _load(service, _tenant_for(principal, tenant), case_ref)
    return {"ok": True, "form": service.client_form(agent_case).to_dict()}


@router.post("/cases/{case_ref}/form")
def submit_client_form(case_ref: str, body: ClientResponse,
                       principal: Principal = Depends(authenticate)
                       ) -> Dict[str, Any]:
    """Persist a structured client response, verbatim.

    This is the deterministic lane: every value reaches the case through
    ``OnboardingService.save_step`` exactly as submitted.
    """
    _require_feature()
    service = get_service()
    agent_case = _load(service, _tenant_for(principal, body.tenant), case_ref)
    return {"ok": True,
            **service.status(service.submit_client_response(
                agent_case, actor=principal.name, response=body.answers,
                request_id=body.request_id, strict=body.strict))}


@router.get("/cases/{case_ref}/classification")
def get_classification(case_ref: str, tenant: Optional[str] = None,
                       principal: Principal = Depends(authenticate)
                       ) -> Dict[str, Any]:
    """Every catalogue field, in one of the five categories, and why."""
    _require_feature()
    service = get_service()
    agent_case = _load(service, _tenant_for(principal, tenant), case_ref)
    return {"ok": True, **service.classify_case(agent_case)}


# --------------------------------------------------------------------------- #
# Review, approval and the confirmation gate
# --------------------------------------------------------------------------- #

@router.post("/cases/{case_ref}/review")
def request_review(case_ref: str, body: TenantBody,
                   principal: Principal = Depends(authenticate)) -> Dict[str, Any]:
    _require_feature()
    service = get_service()
    agent_case = _load(service, _tenant_for(principal, body.tenant), case_ref)
    return {"ok": True,
            **service.status(service.request_activation(
                agent_case, actor=principal.name))}


@router.get("/cases/{case_ref}/review")
def get_review(case_ref: str, tenant: Optional[str] = None,
               principal: Principal = Depends(authenticate)) -> Dict[str, Any]:
    _require_feature()
    service = get_service()
    agent_case = _load(service, _tenant_for(principal, tenant), case_ref)
    package = service.build_review_package(agent_case)
    return {"ok": True, "package": package.to_dict(),
            "document": package.document()}


@router.post("/cases/{case_ref}/activation/approve")
def approve_activation(case_ref: str, body: TenantBody,
                       principal: Principal = Depends(authenticate)
                       ) -> Dict[str, Any]:
    """Approve the configuration. This starts nothing — see the confirm route."""
    _require_feature()
    service = get_service()
    agent_case = _load(service, _tenant_for(principal, body.tenant), case_ref)
    return {"ok": True,
            **service.status(service.approve_activation(
                agent_case, actor=principal.name, reason=body.reason))}


@router.get("/cases/{case_ref}/activation")
def get_activation(case_ref: str, tenant: Optional[str] = None,
                   principal: Principal = Depends(authenticate)
                   ) -> Dict[str, Any]:
    """What confirming would do, and every reason it currently may not."""
    _require_feature()
    service = get_service()
    agent_case = _load(service, _tenant_for(principal, tenant), case_ref)
    return {"ok": True, **service.activation_confirmation(agent_case)}


@router.post("/cases/{case_ref}/activation/confirm")
def confirm_activation(case_ref: str, body: ConfirmActivation,
                       principal: Principal = Depends(authenticate)
                       ) -> Dict[str, Any]:
    """The one route that can reach production, through the one gate.

    In a synthetic environment it is always refused, and the refusal is audited.
    """
    _require_feature()
    service = get_service()
    agent_case = _load(service, _tenant_for(principal, body.tenant), case_ref)
    return {"ok": True,
            **service.status(service.confirm_activation(
                agent_case, actor=principal.name,
                confirmation=body.confirmation))}


@router.post("/cases/{case_ref}/cancel")
def cancel_case(case_ref: str, body: TenantBody,
                principal: Principal = Depends(authenticate)) -> Dict[str, Any]:
    _require_feature()
    service = get_service()
    agent_case = _load(service, _tenant_for(principal, body.tenant), case_ref)
    return {"ok": True,
            **service.status(service.cancel(agent_case, actor=principal.name,
                                            reason=body.reason))}


@router.post("/scenarios/run")
def run_scenario_route(body: LoadFixture,
                       principal: Principal = Depends(authenticate)
                       ) -> Dict[str, Any]:
    """Create a case from a fixture and drive it as far as its controls allow."""
    _require_feature()
    from .scenarios import run_scenario
    service = get_service()
    tenant = _tenant_for(principal, body.tenant)
    if not body.run:
        agent_case = service.create_case(
            tenant=tenant, initiating_user=principal.name,
            instruction=_fixtures.by_id(body.fixture_id).instruction,
            fixture_id=body.fixture_id)
        return {"ok": True, **service.status(agent_case)}
    outcome = run_scenario(service, body.fixture_id, tenant=tenant,
                           actor=principal.name)
    # Keyed as ``scenario`` rather than ``run``: the status projection already
    # carries the run document, and one of them silently overwriting the other
    # is exactly the kind of bug a spread makes easy.
    return {"ok": True, "scenario": outcome.to_dict(),
            **service.status(outcome.case)}
