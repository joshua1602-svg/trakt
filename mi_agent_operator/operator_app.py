"""mi_agent_operator.operator_app — the Operator console API + UI (standalone).

A SEPARATE FastAPI app from the client MI dashboard. Server-side auth is
MANDATORY and fail-closed: every ``/api`` route requires the operator token
(``TRAKT_OPERATOR_TOKEN``) in the ``X-Operator-Token`` header (or
``Authorization: Bearer``). If the token is not configured the API refuses all
requests (503) — hiding the UI is never the control; the server is.

Run locally:
    TRAKT_OPERATOR_TOKEN=dev-secret \
    TRAKT_STORAGE_BACKEND=file TRAKT_LOCAL_BLOB_ROOT=/path/to/blobroot \
    uvicorn mi_agent_operator.operator_app:app --port 8099

In production run it as its OWN App Service (not the client app), fronted by
Entra ID / an IP allowlist, with the token in Key Vault. CORS is locked to
``TRAKT_OPERATOR_CORS_ORIGINS`` (default: same-origin only).
"""

from __future__ import annotations

import hmac
import os
from pathlib import Path
from typing import Any, Dict, Optional

from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel

from .service import OperatorService

_STATIC = Path(__file__).resolve().parent / "static"

app = FastAPI(title="Trakt Operator Console", docs_url=None, redoc_url=None)

# CORS: locked down by default (this is an internal operator surface). Only add
# middleware when origins are explicitly configured — never a wildcard here.
_cors = [o.strip() for o in os.environ.get("TRAKT_OPERATOR_CORS_ORIGINS", "").split(",") if o.strip()]
if _cors:
    from fastapi.middleware.cors import CORSMiddleware
    app.add_middleware(CORSMiddleware, allow_origins=_cors,
                       allow_methods=["*"], allow_headers=["*"])


# --------------------------------------------------------------------------- #
# Auth (fail closed)
# --------------------------------------------------------------------------- #

def _configured_token() -> Optional[str]:
    return os.environ.get("TRAKT_OPERATOR_TOKEN") or None


def require_operator(
    x_operator_token: Optional[str] = Header(default=None),
    authorization: Optional[str] = Header(default=None),
) -> str:
    """Fail-closed operator gate. Returns the operator identity (the token holder)."""
    configured = _configured_token()
    if not configured:
        raise HTTPException(status_code=503,
                            detail="operator console disabled: TRAKT_OPERATOR_TOKEN not set")
    presented = x_operator_token
    if not presented and authorization and authorization.lower().startswith("bearer "):
        presented = authorization[7:].strip()
    if not presented or not hmac.compare_digest(presented, configured):
        raise HTTPException(status_code=401, detail="invalid or missing operator token")
    # A named operator (for the audit trail) may be supplied; else a stable label.
    return os.environ.get("TRAKT_OPERATOR_NAME", "operator")


def _svc() -> OperatorService:
    return OperatorService.from_env()


# --------------------------------------------------------------------------- #
# Request models
# --------------------------------------------------------------------------- #

class ApproveBody(BaseModel):
    mapping_id: Optional[str] = None
    mapping_config_path: Optional[str] = None


class RejectBody(BaseModel):
    reason: str


class EditBody(BaseModel):
    updates: Dict[str, Any]


# --------------------------------------------------------------------------- #
# UI + health
# --------------------------------------------------------------------------- #

@app.get("/", response_class=HTMLResponse)
def index() -> HTMLResponse:
    html = (_STATIC / "operator_ui.html").read_text(encoding="utf-8")
    return HTMLResponse(html)


@app.get("/healthz")
def healthz() -> Dict[str, Any]:
    return {"ok": True, "auth_configured": bool(_configured_token())}


# --------------------------------------------------------------------------- #
# API (all operator-gated)
# --------------------------------------------------------------------------- #

@app.get("/api/queue")
def api_queue(_: str = Depends(require_operator)) -> JSONResponse:
    return JSONResponse({"items": _svc().queue()})


@app.get("/api/audit")
def api_audit(_: str = Depends(require_operator)) -> JSONResponse:
    return JSONResponse({"items": _svc().audit()})


@app.get("/api/item/{approval_id}")
def api_item(approval_id: str, _: str = Depends(require_operator)) -> JSONResponse:
    item = _svc().item(approval_id)
    if item is None:
        raise HTTPException(status_code=404, detail=f"no such approval: {approval_id}")
    return JSONResponse(item)


@app.post("/api/item/{approval_id}/approve")
def api_approve(approval_id: str, body: ApproveBody,
                operator: str = Depends(require_operator)) -> JSONResponse:
    try:
        return JSONResponse(_svc().approve(
            approval_id, decided_by=operator, mapping_id=body.mapping_id,
            mapping_config_path=body.mapping_config_path))
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@app.post("/api/item/{approval_id}/reject")
def api_reject(approval_id: str, body: RejectBody,
               operator: str = Depends(require_operator)) -> JSONResponse:
    if not (body.reason or "").strip():
        raise HTTPException(status_code=400, detail="a rejection reason is required")
    try:
        return JSONResponse(_svc().reject(approval_id, reason=body.reason, decided_by=operator))
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc))


@app.post("/api/item/{approval_id}/edit")
def api_edit(approval_id: str, body: EditBody,
             _: str = Depends(require_operator)) -> JSONResponse:
    try:
        return JSONResponse({"status": "edited",
                             "artifact": _svc().edit(approval_id, body.updates)})
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
