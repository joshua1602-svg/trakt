"""operations_control.api.concentration_routes — the review surface API.

Concentration-test governance for operators: extract proposals from a client
response, review them with full provenance, answer confirmation questions,
edit permitted parameters, decide, and activate.

Tenant-bound exactly like every other Operations Control route: the client
must be inside the authenticated operator's binding (outside it → 404, never
403). ``approve`` and ``activate`` require an administrator — the same split
Client Onboarding uses, because both decide what the funded book is monitored
against from now on.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from ..concentration import ConcentrationError, ConcentrationGovernanceService
from .auth import Principal, authenticate, require_admin, require_client

router = APIRouter(prefix="/ops/concentration", tags=["concentration-tests"])


def _service() -> ConcentrationGovernanceService:
    from .app import get_engine
    return ConcentrationGovernanceService(get_engine().store)


def _guard(principal: Principal, client: str) -> None:
    require_client(principal, client)


def _run(action):
    try:
        return action()
    except ConcentrationError as exc:
        raise HTTPException(status_code=409, detail={
            "errorCode": "CONCENTRATION_REFUSED", "message": str(exc)})


# --------------------------------------------------------------------------- #
# Request models
# --------------------------------------------------------------------------- #
class ExtractRequest(BaseModel):
    source_reference: str
    text: str = ""
    rows: Optional[List[Dict[str, Any]]] = None
    onboarding_run_id: str = ""
    portfolio_id: str = ""


class ConfirmationAnswer(BaseModel):
    question: str
    answer: str


class ProposalUpdate(BaseModel):
    metric_id: Optional[str] = None
    parameters: Optional[Dict[str, Any]] = None
    threshold: Optional[float] = None
    operator: Optional[str] = None
    effective_date: Optional[str] = None
    expiry_date: Optional[str] = None
    resolved_concerns: Optional[List[str]] = None
    note: str = ""


class Approval(BaseModel):
    effective_date: str = ""
    comments: str = ""
    severity: str = "high"
    supersedes_test_id: str = ""


class Reasoned(BaseModel):
    reason: str


class Clarification(BaseModel):
    question: str


class Activation(BaseModel):
    reason: str = ""


# --------------------------------------------------------------------------- #
# Routes
# --------------------------------------------------------------------------- #
@router.get("/{client}/review")
def review(client: str, principal: Principal = Depends(authenticate)
           ) -> Dict[str, Any]:
    _guard(principal, client)
    return _service().review_package(client)


@router.get("/{client}/library")
def library(client: str, principal: Principal = Depends(authenticate)
            ) -> Dict[str, Any]:
    _guard(principal, client)
    svc = _service()
    return {"libraryVersion": svc.lib.library_version,
            "metrics": [m.to_dict() for m in svc.lib.all()]}


@router.post("/{client}/extract")
def extract(client: str, body: ExtractRequest,
            principal: Principal = Depends(authenticate)) -> Dict[str, Any]:
    _guard(principal, client)
    svc = _service()
    proposals = _run(lambda: svc.extract(
        client, actor=principal.name, source_reference=body.source_reference,
        text=body.text, rows=body.rows,
        onboarding_run_id=body.onboarding_run_id,
        portfolio_id=body.portfolio_id))
    return {"proposals": [p.to_dict() for p in proposals],
            "count": len(proposals)}


@router.get("/{client}/proposals/{proposal_id}")
def proposal(client: str, proposal_id: str,
             principal: Principal = Depends(authenticate)) -> Dict[str, Any]:
    _guard(principal, client)
    svc = _service()
    try:
        return svc.get_proposal(client, proposal_id).to_dict()
    except ConcentrationError:
        raise HTTPException(status_code=404, detail={
            "errorCode": "OPS_NOT_FOUND", "message": "That could not be found."})


@router.post("/{client}/proposals/{proposal_id}/answer")
def answer(client: str, proposal_id: str, body: ConfirmationAnswer,
           principal: Principal = Depends(authenticate)) -> Dict[str, Any]:
    _guard(principal, client)
    svc = _service()
    return _run(lambda: svc.answer_confirmation(
        client, proposal_id, actor=principal.name,
        question=body.question, answer=body.answer)).to_dict()


@router.post("/{client}/proposals/{proposal_id}/update")
def update(client: str, proposal_id: str, body: ProposalUpdate,
           principal: Principal = Depends(authenticate)) -> Dict[str, Any]:
    _guard(principal, client)
    svc = _service()
    return _run(lambda: svc.update_proposal(
        client, proposal_id, actor=principal.name,
        metric_id=body.metric_id, parameters=body.parameters,
        threshold=body.threshold, operator=body.operator,
        effective_date=body.effective_date, expiry_date=body.expiry_date,
        resolved_concerns=body.resolved_concerns, note=body.note)).to_dict()


@router.post("/{client}/proposals/{proposal_id}/approve")
def approve(client: str, proposal_id: str, body: Approval,
            principal: Principal = Depends(authenticate)) -> Dict[str, Any]:
    _guard(principal, client)
    require_admin(principal)
    svc = _service()
    return _run(lambda: svc.approve(
        client, proposal_id, actor=principal.name,
        effective_date=body.effective_date, comments=body.comments,
        severity=body.severity,
        supersedes_test_id=body.supersedes_test_id)).to_dict()


@router.post("/{client}/proposals/{proposal_id}/reject")
def reject(client: str, proposal_id: str, body: Reasoned,
           principal: Principal = Depends(authenticate)) -> Dict[str, Any]:
    _guard(principal, client)
    svc = _service()
    return _run(lambda: svc.reject(client, proposal_id,
                                   actor=principal.name,
                                   reason=body.reason)).to_dict()


@router.post("/{client}/proposals/{proposal_id}/unsupported")
def unsupported(client: str, proposal_id: str, body: Reasoned,
                principal: Principal = Depends(authenticate)) -> Dict[str, Any]:
    _guard(principal, client)
    svc = _service()
    return _run(lambda: svc.mark_unsupported(client, proposal_id,
                                             actor=principal.name,
                                             reason=body.reason)).to_dict()


@router.post("/{client}/proposals/{proposal_id}/not-applicable")
def not_applicable(client: str, proposal_id: str, body: Reasoned,
                   principal: Principal = Depends(authenticate)
                   ) -> Dict[str, Any]:
    _guard(principal, client)
    svc = _service()
    return _run(lambda: svc.mark_not_applicable(client, proposal_id,
                                                actor=principal.name,
                                                reason=body.reason)).to_dict()


@router.post("/{client}/proposals/{proposal_id}/clarify")
def clarify(client: str, proposal_id: str, body: Clarification,
            principal: Principal = Depends(authenticate)) -> Dict[str, Any]:
    _guard(principal, client)
    svc = _service()
    return _run(lambda: svc.request_clarification(
        client, proposal_id, actor=principal.name,
        question=body.question)).to_dict()


@router.post("/{client}/proposals/{proposal_id}/supersede")
def supersede(client: str, proposal_id: str, body: Reasoned,
              principal: Principal = Depends(authenticate)) -> Dict[str, Any]:
    _guard(principal, client)
    require_admin(principal)
    svc = _service()
    return _run(lambda: svc.supersede(client, proposal_id,
                                      actor=principal.name,
                                      reason=body.reason)).to_dict()


@router.post("/{client}/activate")
def activate(client: str, body: Activation,
             principal: Principal = Depends(authenticate)) -> Dict[str, Any]:
    _guard(principal, client)
    require_admin(principal)
    svc = _service()
    config = _run(lambda: svc.activate(client, actor=principal.name,
                                       reason=body.reason))
    return config.to_dict()


@router.get("/{client}/versions")
def versions(client: str, principal: Principal = Depends(authenticate)
             ) -> Dict[str, Any]:
    _guard(principal, client)
    svc = _service()
    return {"versions": svc.store.list_versions(client)}


@router.get("/{client}/versions/{version}")
def version(client: str, version: int,
            principal: Principal = Depends(authenticate)) -> Dict[str, Any]:
    _guard(principal, client)
    svc = _service()
    config = svc.store.get_version(client, version)
    if config is None:
        raise HTTPException(status_code=404, detail={
            "errorCode": "OPS_NOT_FOUND", "message": "That could not be found."})
    return config.to_dict()
