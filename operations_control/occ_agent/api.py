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
from .case import SyntheticCase
from .policy import FEATURE_FLAG_ENV, feature_enabled
from .service import OccAgentService

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
    """The tenant this principal operates synthetic cases under.

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
        # so synthetic cases are never filed against a wildcard.
        return "synthetic"
    raise HTTPException(status_code=404, detail={
        "errorCode": "OPS_NOT_FOUND", "message": "That could not be found."})


def _load(service: OccAgentService, tenant: str, case_id: str) -> SyntheticCase:
    return service.load_case(tenant, case_id)


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


class ApplyChange(BaseModel):
    updates: Dict[str, Any] = {}
    tenant: Optional[str] = None


class DecisionAnswer(BaseModel):
    decision_id: str
    action: str                            # approve | amend | reject
    value: str = ""
    reason: str = ""
    tenant: Optional[str] = None


class ReturnToStage(BaseModel):
    target_state: str
    reason: str = ""
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
    return {"ok": True,
            "enabled": True,
            "flag": FEATURE_FLAG_ENV,
            "runtime_mode": get_service().policy.runtime_mode,
            "policy": get_service().policy.to_dict(),
            "lifecycle": _states.lifecycle(),
            "returnable_states": list(_states.RETURNABLE_STATES),
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


@router.get("/cases/{case_id}")
def get_case(case_id: str, tenant: Optional[str] = None,
             principal: Principal = Depends(authenticate)) -> Dict[str, Any]:
    _require_feature()
    service = get_service()
    case = _load(service, _tenant_for(principal, tenant), case_id)
    return {"ok": True, **service.status(case)}


@router.get("/cases/{case_id}/audit")
def get_audit(case_id: str, tenant: Optional[str] = None,
              principal: Principal = Depends(authenticate)) -> Dict[str, Any]:
    _require_feature()
    service = get_service()
    t = _tenant_for(principal, tenant)
    _load(service, t, case_id)             # tenancy check before any read
    return {"ok": True, "events": service.store.list_audit(t, case_id),
            "chain_intact": service.store.verify_audit_chain(t, case_id)}


@router.get("/cases/{case_id}/pack")
def get_pack(case_id: str, tenant: Optional[str] = None,
             principal: Principal = Depends(authenticate)) -> Dict[str, Any]:
    _require_feature()
    service = get_service()
    case = _load(service, _tenant_for(principal, tenant), case_id)
    return {"ok": True, "pack": case.onboarding_pack,
            "required_artefacts": case.required_artefacts,
            "issued_at": case.pack_issued_synthetically_at}


@router.get("/cases/{case_id}/configuration")
def get_configuration(case_id: str, tenant: Optional[str] = None,
                      principal: Principal = Depends(authenticate)
                      ) -> Dict[str, Any]:
    _require_feature()
    service = get_service()
    case = _load(service, _tenant_for(principal, tenant), case_id)
    return {"ok": True, "proposed": case.proposed_configuration,
            "confirmed": case.confirmed_configuration,
            "provenance": case.configuration_provenance}


@router.get("/cases/{case_id}/readiness")
def get_readiness(case_id: str, tenant: Optional[str] = None,
                  principal: Principal = Depends(authenticate)) -> Dict[str, Any]:
    _require_feature()
    service = get_service()
    case = _load(service, _tenant_for(principal, tenant), case_id)
    return {"ok": True, "readiness": service.evaluate_readiness(case),
            "package": (service.readiness_package(case)
                        if case.state == _states.READY_FOR_EXECUTION else None)}


# --------------------------------------------------------------------------- #
# The conversation
# --------------------------------------------------------------------------- #

@router.post("/cases/{case_id}/instruct")
def instruct(case_id: str, body: Instruct,
             principal: Principal = Depends(authenticate)) -> Dict[str, Any]:
    _require_feature()
    service = get_service()
    case = _load(service, _tenant_for(principal, body.tenant), case_id)
    result = service.instruct(case, text=body.text, actor=principal.name,
                              confirm=body.confirm)
    return {"ok": True, "reply": result.reply, "proposal": result.proposal,
            "applied": result.applied, **service.status(result.case)}


@router.post("/cases/{case_id}/requirements/confirm")
def confirm_requirements(case_id: str, body: TenantBody,
                         principal: Principal = Depends(authenticate)
                         ) -> Dict[str, Any]:
    _require_feature()
    service = get_service()
    case = _load(service, _tenant_for(principal, body.tenant), case_id)
    return {"ok": True,
            **service.status(service.confirm_requirements(
                case, actor=principal.name))}


@router.post("/cases/{case_id}/requirements/correct")
def correct_requirements(case_id: str, body: ApplyChange,
                         principal: Principal = Depends(authenticate)
                         ) -> Dict[str, Any]:
    _require_feature()
    service = get_service()
    case = _load(service, _tenant_for(principal, body.tenant), case_id)
    return {"ok": True,
            **service.status(service.apply_requirement_change(
                case, updates=body.updates, actor=principal.name))}


@router.post("/cases/{case_id}/pack/generate")
def generate_pack(case_id: str, body: TenantBody,
                  principal: Principal = Depends(authenticate)) -> Dict[str, Any]:
    _require_feature()
    service = get_service()
    case = _load(service, _tenant_for(principal, body.tenant), case_id)
    return {"ok": True,
            **service.status(service.generate_onboarding_pack(
                case, actor=principal.name))}


@router.post("/cases/{case_id}/pack/approve")
def approve_pack(case_id: str, body: TenantBody,
                 principal: Principal = Depends(authenticate)) -> Dict[str, Any]:
    _require_feature()
    service = get_service()
    case = _load(service, _tenant_for(principal, body.tenant), case_id)
    return {"ok": True,
            **service.status(service.approve_onboarding_pack(
                case, actor=principal.name))}


@router.post("/cases/{case_id}/artefacts")
async def upload_artefacts(case_id: str,
                           files: List[UploadFile] = File(...),
                           tenant: Optional[str] = Form(default=None),
                           principal: Principal = Depends(authenticate)
                           ) -> Dict[str, Any]:
    """Upload synthetic artefacts.

    There is deliberately no way to name a storage location: the intended live
    URI is derived server-side from the case's own confirmed identity, and
    nothing is written to it.
    """
    _require_feature()
    service = get_service()
    case = _load(service, _tenant_for(principal, tenant), case_id)
    for upload in files:
        data = await upload.read()
        case = service.register_synthetic_artefact(
            case, filename=upload.filename or "", data=data,
            actor=principal.name)
    return {"ok": True, **service.status(case)}


@router.post("/cases/{case_id}/artefacts/fixture")
def load_fixture_artefacts(case_id: str, body: LoadFixture,
                           principal: Principal = Depends(authenticate)
                           ) -> Dict[str, Any]:
    """Attach a repository fixture's files as this case's client response."""
    _require_feature()
    service = get_service()
    case = _load(service, _tenant_for(principal, body.tenant), case_id)
    scenario = _fixtures.by_id(body.fixture_id)
    for spec in scenario.files:
        case = service.register_synthetic_artefact(
            case, filename=spec.filename, data=spec.content.encode("utf-8"),
            actor=principal.name, fixture_id=scenario.fixture_id,
            declared_type=spec.declared_type)
    return {"ok": True, **service.status(case)}


@router.post("/cases/{case_id}/artefacts/classify")
def classify(case_id: str, body: TenantBody,
             principal: Principal = Depends(authenticate)) -> Dict[str, Any]:
    _require_feature()
    service = get_service()
    case = _load(service, _tenant_for(principal, body.tenant), case_id)
    return {"ok": True,
            **service.status(service.classify_artefacts(
                case, actor=principal.name))}


@router.post("/cases/{case_id}/configuration/draft")
def draft_config(case_id: str, body: TenantBody,
                 principal: Principal = Depends(authenticate)) -> Dict[str, Any]:
    _require_feature()
    service = get_service()
    case = _load(service, _tenant_for(principal, body.tenant), case_id)
    return {"ok": True,
            **service.status(service.draft_client_config(
                case, actor=principal.name))}


@router.post("/cases/{case_id}/configuration/correct")
def correct_config(case_id: str, body: ApplyChange,
                   principal: Principal = Depends(authenticate)) -> Dict[str, Any]:
    """Supply or correct a configuration value, then resolve again."""
    _require_feature()
    service = get_service()
    case = _load(service, _tenant_for(principal, body.tenant), case_id)
    return {"ok": True,
            **service.status(service.correct_client_config(
                case, updates=body.updates, actor=principal.name))}


@router.post("/cases/{case_id}/configuration/approve")
def approve_config(case_id: str, body: TenantBody,
                   principal: Principal = Depends(authenticate)) -> Dict[str, Any]:
    _require_feature()
    service = get_service()
    case = _load(service, _tenant_for(principal, body.tenant), case_id)
    return {"ok": True,
            **service.status(service.approve_client_config(
                case, actor=principal.name))}


@router.post("/cases/{case_id}/run")
def run_onboarding(case_id: str, body: TenantBody,
                   principal: Principal = Depends(authenticate)) -> Dict[str, Any]:
    _require_feature()
    service = get_service()
    case = _load(service, _tenant_for(principal, body.tenant), case_id)
    return {"ok": True,
            **service.status(service.run_synthetic_onboarding(
                case, actor=principal.name))}


@router.post("/cases/{case_id}/decisions")
def answer_decision(case_id: str, body: DecisionAnswer,
                    principal: Principal = Depends(authenticate)) -> Dict[str, Any]:
    _require_feature()
    service = get_service()
    case = _load(service, _tenant_for(principal, body.tenant), case_id)
    return {"ok": True,
            **service.status(service.resolve_decision(
                case, decision_id=body.decision_id, action=body.action,
                value=body.value, reason=body.reason, actor=principal.name))}


@router.post("/cases/{case_id}/plan")
def generate_plan(case_id: str, body: TenantBody,
                  principal: Principal = Depends(authenticate)) -> Dict[str, Any]:
    _require_feature()
    service = get_service()
    case = _load(service, _tenant_for(principal, body.tenant), case_id)
    return {"ok": True,
            **service.status(service.generate_orchestration_plan(
                case, actor=principal.name))}


@router.post("/cases/{case_id}/readiness/approve")
def approve_readiness(case_id: str, body: TenantBody,
                      principal: Principal = Depends(authenticate)
                      ) -> Dict[str, Any]:
    _require_feature()
    service = get_service()
    case = _load(service, _tenant_for(principal, body.tenant), case_id)
    return {"ok": True,
            **service.status(service.approve_execution_readiness(
                case, actor=principal.name))}


@router.post("/cases/{case_id}/return")
def return_to_stage(case_id: str, body: ReturnToStage,
                    principal: Principal = Depends(authenticate)) -> Dict[str, Any]:
    _require_feature()
    service = get_service()
    case = _load(service, _tenant_for(principal, body.tenant), case_id)
    return {"ok": True,
            **service.status(service.return_to_stage(
                case, target_state=body.target_state, reason=body.reason,
                actor=principal.name))}


@router.post("/cases/{case_id}/cancel")
def cancel_case(case_id: str, body: TenantBody,
                principal: Principal = Depends(authenticate)) -> Dict[str, Any]:
    _require_feature()
    service = get_service()
    case = _load(service, _tenant_for(principal, body.tenant), case_id)
    return {"ok": True,
            **service.status(service.cancel(case, actor=principal.name,
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
        case = service.create_case(tenant=tenant,
                                   initiating_user=principal.name,
                                   instruction=_fixtures.by_id(
                                       body.fixture_id).instruction,
                                   fixture_id=body.fixture_id)
        return {"ok": True, **service.status(case)}
    run = run_scenario(service, body.fixture_id, tenant=tenant,
                       actor=principal.name)
    return {"ok": True, "run": run.to_dict(), **service.status(run.case)}
