"""operations_control.api.onboarding_routes — the Client Onboarding API.

Every route is tenant-bound through the same :mod:`~operations_control.api.auth`
principal the rest of the Operations Control API uses: a client outside the
operator's binding answers 404, never 403, so the existence of another tenant's
configuration is not revealed.

Approving configuration is an administrative act — it changes what every future
delivery for that client resolves with — so the write routes additionally
require an administrator. Reading and drafting do not.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from ..onboarding.service import OnboardingService
from .auth import Principal, authenticate, require_admin, require_client

router = APIRouter(prefix="/ops/onboarding", tags=["client-onboarding"])


def _service() -> OnboardingService:
    from .app import get_engine
    engine = get_engine()
    return OnboardingService(engine.store,
                             registry_factory=engine._source_registry)


class StartDraftBody(BaseModel):
    client_id: str = ""
    #: Populate the draft from the client's existing configuration.
    adopt: bool = False


class SaveStepBody(BaseModel):
    step: str
    payload: Dict[str, Any] = {}


class ApproveBody(BaseModel):
    reason: str = ""


class DiscardBody(BaseModel):
    reason: str = ""


def _require_draft_client(principal: Principal, client_id: str) -> None:
    """A draft that has not named a client yet is not yet tenant-scoped; once
    it names one, the ordinary binding applies."""
    if client_id:
        require_client(principal, client_id)


# --------------------------------------------------------------------------- #
# Reference data
# --------------------------------------------------------------------------- #

@router.get("/vocabularies")
def vocabularies(principal: Principal = Depends(authenticate)) -> Dict[str, Any]:
    """Every governed option list the wizard may offer, with its owner."""
    service = _service()
    return {"ok": True, "vocabularies": service.vocabularies(),
            "standing_fields": service.spec()}


# --------------------------------------------------------------------------- #
# Home
# --------------------------------------------------------------------------- #

@router.get("/clients")
def clients(principal: Principal = Depends(authenticate)) -> Dict[str, Any]:
    service = _service()
    visible = (None if "*" in principal.clients
               else [c for c in principal.clients])
    return {"ok": True, "clients": service.list_clients(visible=visible)}


@router.get("/clients/{client_id}")
def client_detail(client_id: str,
                  principal: Principal = Depends(authenticate)) -> Dict[str, Any]:
    require_client(principal, client_id)
    return {"ok": True, "client": _service().client_view(client_id)}


@router.get("/clients/{client_id}/versions/{version}")
def client_version(client_id: str, version: int,
                   principal: Principal = Depends(authenticate)) -> Dict[str, Any]:
    require_client(principal, client_id)
    return {"ok": True, "version": _service().version_view(client_id, version)}


# --------------------------------------------------------------------------- #
# Drafts
# --------------------------------------------------------------------------- #

@router.post("/drafts", status_code=201)
def start_draft(body: StartDraftBody,
                principal: Principal = Depends(authenticate)) -> Dict[str, Any]:
    _require_draft_client(principal, body.client_id)
    draft = _service().start_draft(client_id=body.client_id,
                                   by=principal.name, adopt=body.adopt)
    return {"ok": True, "draft": draft}


@router.get("/drafts/{draft_id}")
def get_draft(draft_id: str, client: str = "",
              principal: Principal = Depends(authenticate)) -> Dict[str, Any]:
    _require_draft_client(principal, client)
    service = _service()
    doc = service.load_draft(client, draft_id)
    _require_draft_client(principal, doc.get("client_id", ""))
    return {"ok": True, "draft": service.present_draft(doc)}


@router.put("/drafts/{draft_id}")
def save_step(draft_id: str, body: SaveStepBody, client: str = "",
              principal: Principal = Depends(authenticate)) -> Dict[str, Any]:
    _require_draft_client(principal, client)
    service = _service()
    doc = service.load_draft(client, draft_id)
    _require_draft_client(principal, doc.get("client_id", ""))
    # Step 1 is where a draft first names its client. Check the binding BEFORE
    # saving: a draft written under a client id and refused afterwards would
    # still have landed in that tenant's prefix.
    if body.step == "client":
        _require_draft_client(principal,
                              str((body.payload or {}).get("client_id") or ""))
    draft = service.save_step(client_id=client, draft_id=draft_id,
                              step=body.step, payload=body.payload,
                              by=principal.name)
    return {"ok": True, "draft": draft}


@router.get("/drafts/{draft_id}/review")
def review_draft(draft_id: str, client: str = "",
                 principal: Principal = Depends(authenticate)) -> Dict[str, Any]:
    _require_draft_client(principal, client)
    service = _service()
    doc = service.load_draft(client, draft_id)
    _require_draft_client(principal, doc.get("client_id", ""))
    return {"ok": True, "review": service.review(client_id=client,
                                                 draft_id=draft_id)}


@router.post("/drafts/{draft_id}/approve")
def approve_draft(draft_id: str, body: ApproveBody, client: str = "",
                  principal: Principal = Depends(authenticate)) -> Dict[str, Any]:
    """Write the configuration. Administrator only: this changes what every
    future delivery for the client resolves with."""
    _require_draft_client(principal, client)
    require_admin(principal)
    service = _service()
    doc = service.load_draft(client, draft_id)
    _require_draft_client(principal, doc.get("client_id", ""))
    result = service.approve(client_id=client, draft_id=draft_id,
                             by=principal.name, reason=body.reason)
    require_client(principal, result["client_id"])
    return {"ok": True, **result}


@router.post("/drafts/{draft_id}/discard")
def discard_draft(draft_id: str, body: DiscardBody, client: str = "",
                  principal: Principal = Depends(authenticate)) -> Dict[str, Any]:
    _require_draft_client(principal, client)
    service = _service()
    doc = service.load_draft(client, draft_id)
    _require_draft_client(principal, doc.get("client_id", ""))
    service.discard_draft(client_id=client, draft_id=draft_id,
                          by=principal.name, reason=body.reason)
    return {"ok": True}
