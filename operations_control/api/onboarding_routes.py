"""operations_control.api.onboarding_routes — the Client Onboarding API.

Tenant-bound throughout, via the same principal the rest of the Operations
Control API uses: a client outside the operator's binding answers 404, never
403.

Two acts require an administrator, because they decide what every future
delivery for a client resolves against:

  * **approve** — records the decision;
  * **activate** — writes the configuration.

Everything else, including opening a case and gathering answers, is ordinary
operator work. A case that has not named a client yet is not tenant-scoped; the
moment it names one, the binding applies.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from ..onboarding.case import CaseError, OnboardingCase
from ..onboarding.service import OnboardingService
from .auth import Principal, authenticate, require_admin, require_client

router = APIRouter(prefix="/ops/onboarding", tags=["client-onboarding"])


def _service() -> OnboardingService:
    from .app import get_engine
    engine = get_engine()
    return OnboardingService(engine.store,
                             registry_factory=engine._source_registry)


def _visible(principal: Principal) -> Optional[List[str]]:
    if "*" in principal.clients:
        return None
    return [c for c in principal.clients]


def _guard(principal: Principal, case: OnboardingCase) -> None:
    """A case belonging to a client outside the binding does not exist."""
    if case.client_id:
        require_client(principal, case.client_id)


def _load(service: OnboardingService, principal: Principal,
          case_id: str) -> OnboardingCase:
    case = service.load_case(case_id)
    _guard(principal, case)
    return case


# --------------------------------------------------------------------------- #
# Request models
# --------------------------------------------------------------------------- #

class StartMigration(BaseModel):
    client_id: str


class StartAmendment(BaseModel):
    client_id: str


class SaveStep(BaseModel):
    step: str
    payload: Dict[str, Any] = {}


class ReasonBody(BaseModel):
    reason: str = ""


class CreateRequest(BaseModel):
    items: List[Dict[str, Any]] = []
    responsible_party: str = "client"
    due_date: str = ""
    note: str = ""


class RecordResponse(BaseModel):
    note: str = ""
    answers: Dict[str, Any] = {}
    evidence: List[Dict[str, str]] = []


class ReviewResponse(BaseModel):
    accept: bool = True
    note: str = ""


class QuestionBody(BaseModel):
    question: str = ""


class ResolutionBody(BaseModel):
    resolution: str = ""


class PipelineBody(BaseModel):
    portfolio_id: str


class SampleBody(BaseModel):
    #: [{"name": "LoanExtract.csv", "headers": ["loan_id", ...]}]
    files: List[Dict[str, Any]] = []


# --------------------------------------------------------------------------- #
# Reference data
# --------------------------------------------------------------------------- #

@router.get("/reference")
def reference(principal: Principal = Depends(authenticate)) -> Dict[str, Any]:
    """The information model: sections, fields, vocabularies, steps, statuses.

    The browser renders whatever this returns. Adding a field to the governed
    catalogue reaches the wizard without a frontend change.
    """
    return {"ok": True, **_service().reference()}


# --------------------------------------------------------------------------- #
# Home
# --------------------------------------------------------------------------- #

@router.get("/home")
def home(principal: Principal = Depends(authenticate)) -> Dict[str, Any]:
    return {"ok": True, "home": _service().home(visible=_visible(principal))}


# --------------------------------------------------------------------------- #
# Opening a case
# --------------------------------------------------------------------------- #

@router.post("/cases", status_code=201)
def start_new_client(principal: Principal = Depends(authenticate)
                     ) -> Dict[str, Any]:
    """Start a new client onboarding. Blank: no client is selected, and nothing
    is read from any existing configuration."""
    service = _service()
    case = service.start_new_client(by=principal.name)
    return {"ok": True, "case": service.present_case(case)}


@router.post("/cases/migration", status_code=201)
def start_migration(body: StartMigration,
                    principal: Principal = Depends(authenticate)
                    ) -> Dict[str, Any]:
    require_client(principal, body.client_id)
    service = _service()
    case = service.start_migration(client_id=body.client_id,
                                   by=principal.name)
    return {"ok": True, "case": service.present_case(case)}


@router.post("/cases/amendment", status_code=201)
def start_amendment(body: StartAmendment,
                    principal: Principal = Depends(authenticate)
                    ) -> Dict[str, Any]:
    require_client(principal, body.client_id)
    service = _service()
    case = service.start_amendment(client_id=body.client_id,
                                   by=principal.name)
    return {"ok": True, "case": service.present_case(case)}


# --------------------------------------------------------------------------- #
# Working a case
# --------------------------------------------------------------------------- #

@router.get("/cases/{case_id}")
def get_case(case_id: str,
             principal: Principal = Depends(authenticate)) -> Dict[str, Any]:
    service = _service()
    case = _load(service, principal, case_id)
    return {"ok": True, "case": service.present_case(case)}


@router.put("/cases/{case_id}")
def save_step(case_id: str, body: SaveStep,
              principal: Principal = Depends(authenticate)) -> Dict[str, Any]:
    service = _service()
    _load(service, principal, case_id)
    # Step 1 is where a case first names its client. Check the binding BEFORE
    # saving, so a refused case never lands under that tenant.
    if body.step == "client":
        proposed = str((body.payload or {}).get("client_id") or "")
        if proposed:
            require_client(principal, proposed)
    case = service.save_step(case_id=case_id, step=body.step,
                             payload=body.payload, by=principal.name)
    return {"ok": True, "case": service.present_case(case)}


@router.post("/cases/{case_id}/pipeline-book")
def add_pipeline_book(case_id: str, body: PipelineBody,
                      principal: Principal = Depends(authenticate)
                      ) -> Dict[str, Any]:
    service = _service()
    _load(service, principal, case_id)
    case = service.add_pipeline_source(case_id=case_id,
                                       portfolio_id=body.portfolio_id,
                                       by=principal.name)
    return {"ok": True, "case": service.present_case(case)}


@router.post("/cases/{case_id}/sample")
def register_sample(case_id: str, body: SampleBody,
                    principal: Principal = Depends(authenticate)
                    ) -> Dict[str, Any]:
    """Record a sample pack the client has supplied.

    Turns the file format, the expected file names and often the asset class
    from questions into things Trakt reads off the pack.
    """
    service = _service()
    _load(service, principal, case_id)
    case = service.register_sample(case_id=case_id, files=body.files,
                                   by=principal.name)
    return {"ok": True, "case": service.present_case(case)}


@router.delete("/cases/{case_id}/sources/{portfolio_id}/{dataset}")
def remove_source(case_id: str, portfolio_id: str, dataset: str,
                  principal: Principal = Depends(authenticate)
                  ) -> Dict[str, Any]:
    service = _service()
    _load(service, principal, case_id)
    case = service.remove_source(case_id=case_id,
                                 key=f"{portfolio_id}/{dataset}",
                                 by=principal.name)
    return {"ok": True, "case": service.present_case(case)}


# --------------------------------------------------------------------------- #
# Information requests
# --------------------------------------------------------------------------- #

@router.get("/cases/{case_id}/checklist")
def checklist(case_id: str,
              principal: Principal = Depends(authenticate)) -> Dict[str, Any]:
    """What the client still has to supply, derived from the catalogue's
    conditional requirements. Suitable for sending out as-is."""
    service = _service()
    case = _load(service, principal, case_id)
    return {"ok": True, "checklist": service.client_checklist(case),
            "case_id": case.case_id, "client_name": case.client_name}


@router.post("/cases/{case_id}/requests", status_code=201)
def create_request(case_id: str, body: CreateRequest,
                   principal: Principal = Depends(authenticate)
                   ) -> Dict[str, Any]:
    service = _service()
    _load(service, principal, case_id)
    case = service.create_request(case_id=case_id, items=body.items,
                                  by=principal.name,
                                  responsible_party=body.responsible_party,
                                  due_date=body.due_date, note=body.note)
    return {"ok": True, "case": service.present_case(case)}


@router.post("/cases/{case_id}/requests/{request_id}/sent")
def mark_sent(case_id: str, request_id: str,
              principal: Principal = Depends(authenticate)) -> Dict[str, Any]:
    service = _service()
    _load(service, principal, case_id)
    case = service.mark_request_sent(case_id=case_id, request_id=request_id,
                                     by=principal.name)
    return {"ok": True, "case": service.present_case(case)}


@router.post("/cases/{case_id}/requests/{request_id}/response")
def record_response(case_id: str, request_id: str, body: RecordResponse,
                    principal: Principal = Depends(authenticate)
                    ) -> Dict[str, Any]:
    service = _service()
    _load(service, principal, case_id)
    case = service.record_response(case_id=case_id, request_id=request_id,
                                   by=principal.name, note=body.note,
                                   answers=body.answers,
                                   evidence=body.evidence)
    return {"ok": True, "case": service.present_case(case)}


@router.post("/cases/{case_id}/requests/{request_id}/review")
def review_response(case_id: str, request_id: str, body: ReviewResponse,
                    principal: Principal = Depends(authenticate)
                    ) -> Dict[str, Any]:
    service = _service()
    _load(service, principal, case_id)
    case = service.review_response(case_id=case_id, request_id=request_id,
                                   accept=body.accept, by=principal.name,
                                   note=body.note)
    return {"ok": True, "case": service.present_case(case)}


@router.post("/cases/{case_id}/questions", status_code=201)
def add_question(case_id: str, body: QuestionBody,
                 principal: Principal = Depends(authenticate)) -> Dict[str, Any]:
    service = _service()
    _load(service, principal, case_id)
    case = service.add_question(case_id=case_id, question=body.question,
                                by=principal.name)
    return {"ok": True, "case": service.present_case(case)}


@router.post("/cases/{case_id}/questions/{question_id}/resolve")
def resolve_question(case_id: str, question_id: str, body: ResolutionBody,
                     principal: Principal = Depends(authenticate)
                     ) -> Dict[str, Any]:
    service = _service()
    _load(service, principal, case_id)
    case = service.resolve_question(case_id=case_id, question_id=question_id,
                                    resolution=body.resolution,
                                    by=principal.name)
    return {"ok": True, "case": service.present_case(case)}


# --------------------------------------------------------------------------- #
# Readiness, preview, approval, activation
# --------------------------------------------------------------------------- #

@router.get("/cases/{case_id}/preview")
def preview(case_id: str,
            principal: Principal = Depends(authenticate)) -> Dict[str, Any]:
    """Exactly what activation would create or change. Writes nothing."""
    service = _service()
    case = _load(service, principal, case_id)
    return {"ok": True, "preview": service.preview(case)}


@router.post("/cases/{case_id}/submit")
def submit(case_id: str,
           principal: Principal = Depends(authenticate)) -> Dict[str, Any]:
    service = _service()
    _load(service, principal, case_id)
    case = service.submit_for_approval(case_id=case_id, by=principal.name)
    return {"ok": True, "case": service.present_case(case)}


@router.post("/cases/{case_id}/changes-required")
def request_changes(case_id: str, body: ReasonBody,
                    principal: Principal = Depends(authenticate)
                    ) -> Dict[str, Any]:
    service = _service()
    _load(service, principal, case_id)
    case = service.request_changes(case_id=case_id, by=principal.name,
                                   reason=body.reason)
    return {"ok": True, "case": service.present_case(case)}


@router.post("/cases/{case_id}/approve")
def approve(case_id: str, body: ReasonBody,
            principal: Principal = Depends(authenticate)) -> Dict[str, Any]:
    """Record the decision. No configuration is written here."""
    service = _service()
    case = _load(service, principal, case_id)
    require_admin(principal)
    if case.client_id:
        require_client(principal, case.client_id)
    case = service.approve(case_id=case_id, by=principal.name,
                           reason=body.reason)
    return {"ok": True, "case": service.present_case(case)}


@router.post("/cases/{case_id}/activate")
def activate(case_id: str,
             principal: Principal = Depends(authenticate)) -> Dict[str, Any]:
    """Create the client's configuration. The only place it is written."""
    service = _service()
    case = _load(service, principal, case_id)
    require_admin(principal)
    require_client(principal, case.client_id or "")
    result = service.activate(case_id=case_id, by=principal.name)
    return {"ok": True, **result}


@router.post("/cases/{case_id}/withdraw")
def withdraw(case_id: str, body: ReasonBody,
             principal: Principal = Depends(authenticate)) -> Dict[str, Any]:
    service = _service()
    _load(service, principal, case_id)
    case = service.withdraw(case_id=case_id, by=principal.name,
                            reason=body.reason)
    return {"ok": True, "case": service.present_case(case)}


# --------------------------------------------------------------------------- #
# Active clients
# --------------------------------------------------------------------------- #

@router.get("/clients/{client_id}")
def client_detail(client_id: str,
                  principal: Principal = Depends(authenticate)
                  ) -> Dict[str, Any]:
    require_client(principal, client_id)
    return {"ok": True, "client": _service().client_view(client_id)}


@router.get("/clients/{client_id}/versions/{version}")
def client_version(client_id: str, version: int,
                   principal: Principal = Depends(authenticate)
                   ) -> Dict[str, Any]:
    require_client(principal, client_id)
    record = _service().cases.version(client_id, version)
    if record is None:
        raise HTTPException(status_code=404, detail={
            "errorCode": "OPS_NOT_FOUND", "message": "That could not be found."})
    return {"ok": True, "version": record.to_dict()}
