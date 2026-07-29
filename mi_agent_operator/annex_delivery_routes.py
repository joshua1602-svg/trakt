"""mi_agent_operator.annex_delivery_routes — Annex delivery in the Operator console.

An **additive** router. It adds the stages the Operations Control Centre did not
previously cover — everything after projection — and changes nothing about the
existing source-approval queue, the run records or the blob-trigger semantics.

    projection → delivery normalisation → XML generation → XSD validation
    → operator review → publication

Two controls, both server-side:

* **Operator gate.** Every route depends on the console's existing fail-closed
  ``require_operator`` check. Without ``TRAKT_OPERATOR_TOKEN`` the console
  refuses everything, and that is unchanged here.

* **Tenant resolution.** The console is a managed-service surface, so the
  operator legitimately acts on behalf of a named client — but the client
  identifier arrives from a browser, so it is *resolved and authorised* rather
  than trusted. A client not present in the deployment's tenancy configuration
  is refused, as is one outside ``TRAKT_OPERATOR_TENANTS`` when that allowlist
  is set. The resulting :class:`~trakt_core.context.ExecutionContext` is what the
  agent authorises against; portfolio selectors are then narrowed inside the
  tenant by :func:`trakt_core.tenancy.authorise_portfolio_access` exactly as they
  are on the client-facing API.

Mount it with one line in ``operator_app.py``::

    from .annex_delivery_routes import router as annex_delivery_router
    app.include_router(annex_delivery_router)
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, Header, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from trakt_core.context import (
    ACTOR_SERVICE, CHANNEL_INTERNAL, DEFAULT_MI_SCOPES, ExecutionContext,
)
from trakt_core.errors import TraktError
from trakt_core.tenancy import load_tenant_registry

from engine.annex_delivery_agent.agent import DEFAULT_STORE_ROOT, AnnexDeliveryAgent
from engine.annex_delivery_agent.contracts import DeliveryRequest
from engine.annex_delivery_agent.occ_adapter import AnnexDeliveryOperatorService
from engine.annex_delivery_agent.persistence import DeliveryRunStore
from engine.annex_delivery_agent.publication import FilesystemPublicationSink

router = APIRouter(prefix="/api/annex-delivery", tags=["annex-delivery"])


def require_operator(
    x_operator_token: Optional[str] = Header(default=None),
    authorization: Optional[str] = Header(default=None),
) -> str:
    """The console's existing fail-closed operator gate, resolved at request time.

    Imported lazily rather than at module scope: ``operator_app`` mounts this
    router at the end of its own import, so a module-level import back into it
    would be circular and would silently leave the routes unmounted whenever
    this module happened to be imported first.
    """
    from .operator_app import require_operator as _gate

    return _gate(x_operator_token=x_operator_token, authorization=authorization)

#: Where delivery runs are stored operationally. Separate from production/latest.
STORE_ROOT_ENV = "TRAKT_ANNEX_DELIVERY_ROOT"
#: Where approved packages are published.
PUBLISH_ROOT_ENV = "TRAKT_ANNEX_DELIVERY_PUBLISH_ROOT"
#: Optional comma-separated allowlist of clients this console may act for.
TENANT_ALLOWLIST_ENV = "TRAKT_OPERATOR_TENANTS"


# --------------------------------------------------------------------------- #
# Request models — caller input, never trusted as identity
# --------------------------------------------------------------------------- #


class PrepareBody(BaseModel):
    client_id: str
    annex_id: str = "ESMA_Annex2"
    annex_version: Optional[str] = None
    reporting_date: str
    projected_input: str
    portfolio_id: Optional[str] = None
    client_config_reference: Optional[str] = None
    currency: Optional[str] = None
    performance_mode: Optional[str] = None
    force_rerun: bool = False
    force_reason: Optional[str] = None


class DecisionBody(BaseModel):
    client_id: str
    annex_id: str = "ESMA_Annex2"
    annex_version: Optional[str] = None
    reporting_date: str
    portfolio_id: Optional[str] = None
    note: str = ""
    reason: str = ""


# --------------------------------------------------------------------------- #
# Context construction
# --------------------------------------------------------------------------- #


def _allowlist() -> List[str]:
    return [t.strip() for t in os.environ.get(TENANT_ALLOWLIST_ENV, "").split(",") if t.strip()]


def operator_context(client_id: str, operator: str) -> ExecutionContext:
    """Build a trusted context for ``client_id``, or refuse.

    The browser names a client; this function decides whether that client exists
    and whether this console may act for it. Nothing downstream re-checks the
    string, so the check has to be here and has to fail closed.
    """
    tenant = str(client_id or "").strip()
    if not tenant:
        raise HTTPException(status_code=400, detail="client_id is required")

    allowed = _allowlist()
    if allowed and tenant not in allowed:
        raise HTTPException(status_code=403,
                            detail="this console is not authorised for that client")

    registry = load_tenant_registry()
    if registry.configured and registry.get(tenant) is None:
        raise HTTPException(status_code=403, detail="unknown client")

    return ExecutionContext(
        tenant_id=tenant,
        actor_id=operator,
        actor_type=ACTOR_SERVICE,
        channel=CHANNEL_INTERNAL,
        scopes=DEFAULT_MI_SCOPES,
        actor_label=f"operator-console:{operator}",
    )


def _service() -> AnnexDeliveryOperatorService:
    store = DeliveryRunStore(os.environ.get(STORE_ROOT_ENV) or DEFAULT_STORE_ROOT)
    publish_root = os.environ.get(PUBLISH_ROOT_ENV)
    sink = FilesystemPublicationSink(publish_root) if publish_root else None
    return AnnexDeliveryOperatorService(
        AnnexDeliveryAgent(store=store, publication_sink=sink))


def _request(body: Any, *, projected_input: str = "") -> DeliveryRequest:
    options: Dict[str, Any] = {}
    for key in ("currency", "performance_mode"):
        value = getattr(body, key, None)
        if value:
            options[key] = value
    return DeliveryRequest(
        annex_id=body.annex_id,
        annex_version=body.annex_version,
        reporting_date=body.reporting_date,
        projected_input=projected_input or getattr(body, "projected_input", "") or "",
        output_root=os.environ.get(STORE_ROOT_ENV) or DEFAULT_STORE_ROOT,
        portfolio_id=body.portfolio_id,
        client_config_reference=getattr(body, "client_config_reference", None),
        force_rerun=bool(getattr(body, "force_rerun", False)),
        force_reason=getattr(body, "force_reason", None),
        options=options,
    )


def _handle(exc: TraktError) -> HTTPException:
    return HTTPException(status_code=exc.http_status, detail=exc.message)


# --------------------------------------------------------------------------- #
# Routes
# --------------------------------------------------------------------------- #


@router.get("/reports")
def api_reports(_: str = Depends(require_operator)) -> JSONResponse:
    """The annex reports this deployment can prepare."""
    return JSONResponse({"reports": _service().supported_reports()})


@router.post("/prepare")
def api_prepare(body: PrepareBody,
                operator: str = Depends(require_operator)) -> JSONResponse:
    """Run the delivery lifecycle. Never publishes, whatever the result."""
    context = operator_context(body.client_id, operator)
    try:
        return JSONResponse(_service().prepare(context, _request(body)))
    except TraktError as exc:
        raise _handle(exc)


@router.post("/runs/{run_id}/request-approval")
def api_request_approval(run_id: str, body: DecisionBody,
                         operator: str = Depends(require_operator)) -> JSONResponse:
    context = operator_context(body.client_id, operator)
    try:
        return JSONResponse(_service().request_approval(context, _request(body), run_id=run_id))
    except TraktError as exc:
        raise _handle(exc)


@router.post("/runs/{run_id}/approve")
def api_approve(run_id: str, body: DecisionBody,
                operator: str = Depends(require_operator)) -> JSONResponse:
    """Approve a prepared report. This is the only route that makes it publishable."""
    context = operator_context(body.client_id, operator)
    try:
        return JSONResponse(_service().approve(context, _request(body), run_id=run_id,
                                               note=body.note))
    except TraktError as exc:
        raise _handle(exc)


@router.post("/runs/{run_id}/reject")
def api_reject(run_id: str, body: DecisionBody,
               operator: str = Depends(require_operator)) -> JSONResponse:
    if not (body.reason or "").strip():
        raise HTTPException(status_code=400, detail="a rejection reason is required")
    context = operator_context(body.client_id, operator)
    try:
        return JSONResponse(_service().reject(context, _request(body), run_id=run_id,
                                              reason=body.reason))
    except TraktError as exc:
        raise _handle(exc)


@router.post("/runs/{run_id}/publish")
def api_publish(run_id: str, body: DecisionBody,
                operator: str = Depends(require_operator)) -> JSONResponse:
    """Publish an approved report. Refuses anything not explicitly approved."""
    context = operator_context(body.client_id, operator)
    try:
        return JSONResponse(_service().publish(context, _request(body), run_id=run_id))
    except TraktError as exc:
        raise _handle(exc)


__all__ = ["router", "operator_context", "STORE_ROOT_ENV", "PUBLISH_ROOT_ENV",
           "TENANT_ALLOWLIST_ENV"]
