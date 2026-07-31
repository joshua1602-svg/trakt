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
                       "The onboarding agent is not available.",
                       http_status=503)
    return _service


def _require_feature() -> None:
    if not feature_enabled():
        raise OpsError(
            "OCC_AGENT_DISABLED",
            "The onboarding agent is not switched on for this environment.",
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
    ready = agent_case.run.state == _states.READY_FOR_EXECUTION
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
