"""operations_control.api.app — the Operations Control API.

Run locally:

    TRAKT_STORAGE_BACKEND=file \
    TRAKT_LOCAL_BLOB_ROOT=.localblob \
    TRAKT_OPS_OPERATORS='{"dev-token": {"name": "Operator", "clients": ["*"]}}' \
    uvicorn operations_control.api.app:app --port 8100

All endpoints require an operator token and are tenant-bound. Errors are
returned as operator-safe envelopes — no traces, paths or internals ever leave
the API.
"""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional

from fastapi import Depends, FastAPI, File, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from ..contracts import (
    DEC_OPEN,
    PUBLICATION_SCOPE_DEFAULT,
    RUN_AWAITING_PUBLICATION,
    RUN_BLOCKED,
    RUN_CANCELLED,
    RUN_NEEDS_REVIEW,
    RUN_PUBLISHED,
    RUN_RECEIVED,
)
from ..engine import OpsEngine, OpsError
from ..onboarding.case import CaseError as _CaseError
from ..stores import OpsStore
from . import presenters, workflow_view
from .auth import Principal, authenticate, require_admin, require_client

logger = logging.getLogger("trakt.operations_control.api")

_engine: Optional[OpsEngine] = None


def get_engine() -> OpsEngine:
    global _engine
    if _engine is None:
        _engine = OpsEngine(OpsStore.from_env())
    return _engine


def set_engine(engine: OpsEngine) -> None:
    """Test / composition seam."""
    global _engine
    _engine = engine


@asynccontextmanager
async def _lifespan(app: FastAPI):
    try:
        recovered = get_engine().recover_on_startup()
        if recovered:
            logger.info("recovered %d interrupted workflow(s)", len(recovered))
    except Exception:  # noqa: BLE001 — startup recovery must not kill the API
        logger.exception("startup recovery failed")
    yield


app = FastAPI(title="Trakt Operations Control API", docs_url=None,
              redoc_url=None, lifespan=_lifespan)

_cors = [o.strip() for o in os.environ.get(
    "TRAKT_OPS_CORS_ORIGINS", "http://localhost:5173").split(",") if o.strip()]
app.add_middleware(CORSMiddleware, allow_origins=_cors,
                   allow_methods=["*"],
                   allow_headers=["X-Operator-Token", "Authorization",
                                  "Content-Type"])


#: Client Onboarding — the governed standing-configuration capability. It sits
#: alongside Operations: Operations processes deliveries, Onboarding creates the
#: configuration those deliveries resolve with.
from . import onboarding_routes  # noqa: E402  (router needs `app` above)

app.include_router(onboarding_routes.router)

#: Concentration-test governance — extraction, operator review, approval and
#: versioned activation of the funded-book concentration tests. Same tenancy
#: and audit model as Client Onboarding; the activated configuration is what
#: the MI evaluation service reads.
from . import concentration_routes  # noqa: E402  (router needs `app` above)

app.include_router(concentration_routes.router)


def mount_occ_agent(application: FastAPI) -> bool:
    """Mount the OCC Agent tab's routes, when the feature is switched on.

    Additive and flag-gated: with ``OCC_AGENT_SYNTHETIC_ENABLED`` unset nothing
    is imported and no route exists, so the live API is byte-for-byte what it
    was. A failure to mount is logged and swallowed deliberately — a
    pre-scale capability must never be able to stop the Operations Control API
    from starting.
    """
    from ..occ_agent.policy import feature_enabled
    if not feature_enabled():
        return False
    try:
        from apps.blob_trigger_app.storage import open_storage
        from ..occ_agent import api as occ_agent_api
        from ..occ_agent.service import OccAgentService
        occ_agent_api.configure(OccAgentService(open_storage()))
        application.include_router(occ_agent_api.router)
        logger.info("OCC Agent (synthetic) routes mounted")
        return True
    except Exception:  # noqa: BLE001 — never block API startup
        logger.exception("OCC Agent routes could not be mounted")
        return False


mount_occ_agent(app)


@app.exception_handler(OpsError)
async def _ops_error(request: Request, exc: OpsError):
    return JSONResponse(status_code=exc.http_status,
                        content={"ok": False, "errorCode": exc.code,
                                 "message": exc.message})


@app.exception_handler(_CaseError)
async def _case_error(request: Request, exc: _CaseError):
    """An illegal onboarding transition is an operator mistake, not a fault."""
    return JSONResponse(status_code=exc.http_status,
                        content={"ok": False, "errorCode": exc.code,
                                 "message": exc.message})


@app.exception_handler(Exception)
async def _unhandled(request: Request, exc: Exception):
    logger.exception("unhandled error on %s", request.url.path)
    return JSONResponse(status_code=500, content={
        "ok": False, "errorCode": "OPS_INTERNAL",
        "message": "Something went wrong on our side. Nothing has been lost — "
                   "try again in a moment."})


@app.get("/health")
def health() -> Dict[str, Any]:
    eng = get_engine()
    storage_ok = True
    try:
        eng.store.known_clients()
    except Exception:  # noqa: BLE001
        storage_ok = False
    return {"ok": storage_ok, "service": "operations-control",
            "auth_configured": bool(os.environ.get("TRAKT_OPS_OPERATORS")),
            "storage_ok": storage_ok}


# --------------------------------------------------------------------------- #
# Request models
# --------------------------------------------------------------------------- #

class RegisterDelivery(BaseModel):
    client_id: str
    portfolio_id: str
    input_path: str
    dataset: str = "funded"
    frequency: str = "monthly"
    reporting_period: str = ""


class CreateWorkflow(BaseModel):
    client_id: str
    delivery_id: str
    outcome: str                       # mi | mi_annex2
    workflow_type: Optional[str] = None   # override; None = accept suggestion
    override_reason: str = ""
    start: bool = True


class DecisionBody(BaseModel):
    action: str                        # approve | reject | amend
    value: str = ""
    scope: str = "portfolio"
    reason: str = ""


class ReasonBody(BaseModel):
    reason: str = ""


# --------------------------------------------------------------------------- #
# Dashboard
# --------------------------------------------------------------------------- #

@app.get("/ops/me")
def me(principal: Principal = Depends(authenticate)) -> Dict[str, Any]:
    """Who is signed in, and what may they see.

    The browser uses this only to decide what to OFFER — every administrator
    route re-checks the role server-side, so hiding a link is never the
    security boundary.
    """
    return {"ok": True, "principal": {
        "name": principal.name,
        "role": principal.role,
        "is_admin": principal.is_admin,
        "all_clients": "*" in principal.clients,
        "clients": [c for c in principal.clients if c != "*"],
    }}


@app.get("/ops/dashboard")
def dashboard(principal: Principal = Depends(authenticate)) -> Dict[str, Any]:
    eng = get_engine()
    clients = principal.visible_clients(eng.store.known_clients())
    tiles = {"new_deliveries": 0, "needs_attention": 0, "blocked": 0,
             "ready_to_publish": 0, "recently_published": 0}
    attention: List[Dict[str, Any]] = []
    recent_pubs: List[Dict[str, Any]] = []
    for c in clients:
        rows = eng.store.list_workflows(c)
        open_by_wf = _open_counts(eng, c)
        for row in rows:
            s = row.get("status")
            n_open = open_by_wf.get(row["workflow_id"], 0)
            if s == RUN_RECEIVED:
                tiles["new_deliveries"] += 1
            elif s == RUN_NEEDS_REVIEW:
                tiles["needs_attention"] += 1
                attention.append(presenters.present_workflow_row(row, n_open))
            elif s == RUN_BLOCKED:
                tiles["blocked"] += 1
                attention.append(presenters.present_workflow_row(row, n_open))
            elif s == RUN_AWAITING_PUBLICATION:
                tiles["ready_to_publish"] += 1
                attention.append(presenters.present_workflow_row(row, n_open))
        for pub in eng.store.list_publications(c)[:5]:
            if pub.get("status") == "published":
                tiles["recently_published"] += 1
                recent_pubs.append(presenters.present_publication(pub))
    return {"ok": True, "tiles": tiles,
            "needs_attention": attention[:20],
            "recently_published": recent_pubs[:10]}


def _open_counts(eng: OpsEngine, client_id: str) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for d in eng.store.open_decisions(client_id):
        wf = d.get("workflow_id", "")
        out[wf] = out.get(wf, 0) + 1
    return out


# --------------------------------------------------------------------------- #
# Clients & deliveries
# --------------------------------------------------------------------------- #

@app.get("/ops/clients")
def clients(principal: Principal = Depends(authenticate)) -> Dict[str, Any]:
    eng = get_engine()
    known = principal.visible_clients(eng.store.known_clients())
    # Clients already known to the production source registry too.
    try:
        reg = eng._source_registry()
        for r in reg.records():
            if principal.allows(r.client_id) and r.client_id not in known:
                known.append(r.client_id)
    except Exception:  # noqa: BLE001
        pass
    return {"ok": True, "clients": sorted(known)}


@app.post("/ops/deliveries")
def register_delivery(body: RegisterDelivery,
                      principal: Principal = Depends(authenticate)) -> Dict[str, Any]:
    require_client(principal, body.client_id)
    eng = get_engine()
    doc = eng.register_delivery(
        client_id=body.client_id, portfolio_id=body.portfolio_id,
        input_path=body.input_path, dataset=body.dataset,
        frequency=body.frequency, reporting_period=body.reporting_period,
        registered_by=principal.name)
    return {"ok": True, "delivery": presenters.present_delivery(doc)}


@app.get("/ops/deliveries")
def list_deliveries(client: str,
                    principal: Principal = Depends(authenticate)) -> Dict[str, Any]:
    require_client(principal, client)
    eng = get_engine()
    return {"ok": True, "deliveries": [presenters.present_delivery(d)
                                       for d in eng.store.list_deliveries(client)]}


# --------------------------------------------------------------------------- #
# Input batches (OCC-owned readiness — replaces the _READY.json sentinel)
# --------------------------------------------------------------------------- #

class CreateBatch(BaseModel):
    client_id: str
    portfolio_id: str
    reporting_date: str
    workflow_type: str                 # mi | mi_annex2
    auto_start_when_ready: bool = False
    # Which book these files describe: funded | pipeline. Blank keeps the funded
    # default, so a client that does not send the field behaves exactly as before.
    # The engine refuses an unknown dataset, and refuses pipeline + mi_annex2 —
    # a pipeline view never carries a regime delivery.
    dataset: str = ""


class RegisterBatchFile(BaseModel):
    path: str                          # file or directory to register


def _role_labels() -> Dict[str, str]:
    from ..intake import load_requirements
    return load_requirements().get("role_labels") or {}


@app.post("/ops/batches", status_code=201)
def create_batch(body: CreateBatch,
                 principal: Principal = Depends(authenticate)) -> Dict[str, Any]:
    require_client(principal, body.client_id)
    eng = get_engine()
    batch = eng.create_batch(
        client_id=body.client_id, portfolio_id=body.portfolio_id,
        reporting_date=body.reporting_date, workflow_type=body.workflow_type,
        created_by=principal.name,
        auto_start_when_ready=body.auto_start_when_ready,
        dataset=body.dataset)
    return {"ok": True, "batch": presenters.present_batch(batch,
                                                          _role_labels())}


@app.get("/ops/batches")
def list_batches(client: str,
                 principal: Principal = Depends(authenticate)) -> Dict[str, Any]:
    require_client(principal, client)
    eng = get_engine()
    labels = _role_labels()
    return {"ok": True,
            "batches": [presenters.present_batch(b, labels)
                        for b in eng.intake.list_batches(client)]}


@app.get("/ops/batches/{batch_id}")
def get_batch(batch_id: str, client: str,
              principal: Principal = Depends(authenticate)) -> Dict[str, Any]:
    require_client(principal, client)
    eng = get_engine()
    batch = eng.intake.load_batch(client, batch_id)
    if batch is None:
        raise HTTPException(status_code=404, detail={
            "errorCode": "OPS_NOT_FOUND", "message": "That could not be found."})
    return {"ok": True, "batch": presenters.present_batch(batch,
                                                          _role_labels())}


#: Server-side directories the deprecated path-registration route may read from.
#: EMPTY BY DEFAULT — a browser cannot name a location on the server unless an
#: administrator has explicitly opted a directory in. Manual deliveries go
#: through ``/ops/batches/{id}/upload`` instead, where the destination is derived
#: server-side from controlled fields.
SERVER_PATH_ROOTS_ENV = "TRAKT_OPS_SERVER_PATH_ROOTS"

#: Total bytes one upload request may carry. Deliberately modest: the request
#: body is held in memory by the single worker that also runs governed workflows
#: on background threads, and the platform in front of this service enforces its
#: own request timeout. Raise it only with a real delivery to measure against.
MAX_UPLOAD_MB_ENV = "TRAKT_OPS_MAX_UPLOAD_MB"
DEFAULT_MAX_UPLOAD_MB = 64


def _allowed_server_path(path: str) -> bool:
    """True when ``path`` resolves inside an explicitly allow-listed root."""
    roots = [r.strip() for r in
             os.environ.get(SERVER_PATH_ROOTS_ENV, "").split(os.pathsep)
             if r.strip()]
    if not roots:
        return False
    try:
        resolved = os.path.realpath(path)
    except (OSError, ValueError):
        return False
    for root in roots:
        real_root = os.path.realpath(root)
        if resolved == real_root or resolved.startswith(real_root + os.sep):
            return True
    return False


@app.post("/ops/batches/{batch_id}/files")
def register_batch_file(batch_id: str, body: RegisterBatchFile, client: str,
                        principal: Principal = Depends(authenticate)
                        ) -> Dict[str, Any]:
    """Deprecated: register files already sitting on the server's own disk.

    Kept for server-side operational tooling only, and fail-closed — the caller
    must be an administrator AND the location must be inside a directory an
    administrator has allow-listed. The Operations Control Centre itself uploads
    through ``/ops/batches/{batch_id}/upload``, where the browser never names a
    location at all.
    """
    require_client(principal, client)
    require_admin(principal)
    if not _allowed_server_path(body.path):
        raise HTTPException(status_code=403, detail={
            "errorCode": "OPS_PATH_NOT_ALLOWED",
            "message": "Trakt does not take files from a location you type in. "
                       "Send the files themselves."})
    eng = get_engine()
    batch = eng.register_batch_file(client_id=client, batch_id=batch_id,
                                    source_path=body.path,
                                    received_by=principal.name)
    return {"ok": True, "batch": presenters.present_batch(batch,
                                                          _role_labels())}


@app.post("/ops/batches/{batch_id}/upload")
async def upload_batch_files(batch_id: str, client: str,
                             files: List[UploadFile] = File(...),
                             principal: Principal = Depends(authenticate)
                             ) -> Dict[str, Any]:
    """Send the files for a manually created delivery.

    The request carries file CONTENT only. The governed destination is derived
    on the server from the input pack's own client, portfolio, book, frequency
    and reporting period, and every source file is placed and registered before
    readiness is assessed — so the run manifest that authorises processing is
    always written last.
    """
    require_client(principal, client)
    limit_mb = int(os.environ.get(MAX_UPLOAD_MB_ENV, DEFAULT_MAX_UPLOAD_MB))
    limit = max(1, limit_mb) * 1024 * 1024
    uploads: List[tuple] = []
    total = 0
    for upload in files:
        data = await upload.read()
        total += len(data)
        if total > limit:
            raise HTTPException(status_code=413, detail={
                "errorCode": "OPS_UPLOAD_TOO_LARGE",
                "message": f"That is more than Trakt accepts in one go "
                           f"({limit_mb} MB). Send fewer files at a time."})
        uploads.append((upload.filename or "", data))
    eng = get_engine()
    batch = eng.upload_batch_files(client_id=client, batch_id=batch_id,
                                   uploads=uploads,
                                   received_by=principal.name)
    return {"ok": True, "batch": presenters.present_batch(batch,
                                                          _role_labels())}


@app.post("/ops/batches/{batch_id}/start")
def start_batch(batch_id: str, client: str,
                principal: Principal = Depends(authenticate)) -> Dict[str, Any]:
    require_client(principal, client)
    eng = get_engine()
    batch = eng.start_batch(client_id=client, batch_id=batch_id,
                            actor=principal.name)
    return {"ok": True, "batch": presenters.present_batch(batch,
                                                          _role_labels())}


# --------------------------------------------------------------------------- #
# Workflows
# --------------------------------------------------------------------------- #

@app.get("/ops/workflows")
def list_workflows(client: Optional[str] = None, status: Optional[str] = None,
                   principal: Principal = Depends(authenticate)) -> Dict[str, Any]:
    eng = get_engine()
    clients = ([client] if client
               else principal.visible_clients(eng.store.known_clients()))
    rows: List[Dict[str, Any]] = []
    for c in clients:
        require_client(principal, c)
        open_by_wf = _open_counts(eng, c)
        for row in eng.store.list_workflows(c):
            if status and row.get("status") != status:
                continue
            # A cancelled delivery is kept, never deleted, but it is off the
            # working list: it needs nothing from anybody. Asking for it by
            # status still finds it.
            if not status and row.get("status") == RUN_CANCELLED:
                continue
            rows.append(presenters.present_workflow_row(
                row, open_by_wf.get(row["workflow_id"], 0)))
    rows.sort(key=lambda r: r.get("created_at") or "", reverse=True)
    return {"ok": True, "workflows": rows}


@app.post("/ops/workflows", status_code=201)
def create_workflow(body: CreateWorkflow,
                    principal: Principal = Depends(authenticate)) -> Dict[str, Any]:
    require_client(principal, body.client_id)
    eng = get_engine()
    run = eng.create_workflow(
        client_id=body.client_id, delivery_id=body.delivery_id,
        outcome=body.outcome, workflow_type=body.workflow_type,
        created_by=principal.name, override_reason=body.override_reason)
    if body.start and run.status == RUN_RECEIVED:
        run = eng.start(run, actor=principal.name)
    return {"ok": True, "workflow": _full_workflow(eng, run.client_id,
                                                   run.workflow_id)}


def _load_owned_workflow(eng: OpsEngine, principal: Principal,
                         workflow_id: str, client: Optional[str]):
    """Resolve a workflow server-side and enforce tenancy. The client hint is
    only a lookup aid — authorisation always re-checks the stored document."""
    candidates = ([client] if client
                  else principal.visible_clients(eng.store.known_clients()))
    for c in candidates:
        if not principal.allows(c):
            continue
        run = eng.store.load_workflow(c, workflow_id)
        if run is not None:
            require_client(principal, run.client_id)
            return run
    raise HTTPException(status_code=404, detail={
        "errorCode": "OPS_NOT_FOUND", "message": "That could not be found."})


def _full_workflow(eng: OpsEngine, client_id: str,
                   workflow_id: str) -> Dict[str, Any]:
    run = eng.store.load_workflow(client_id, workflow_id)
    results = {}
    for stage in run.applicable_stages:
        gar = eng.store.load_result(client_id, workflow_id, stage)
        if gar is not None:
            results[stage] = gar.to_dict()
    decisions = eng.store.list_decisions(client_id, workflow_id=workflow_id)
    n_open = len([d for d in decisions if d.get("status") == DEC_OPEN])
    out = presenters.present_workflow(run, results, n_open)

    # The durable case file: the stages already completed, the stage the
    # delivery is at, and what happens next. Built from the same persisted
    # documents; it decides nothing.
    batch = (eng.intake.load_batch(client_id, run.batch_id)
             if run.batch_id else None)
    publication = eng.store.load_publication(
        client_id, run.reporting_period or "unspecified")
    if publication is not None and publication.get("workflow_id") != workflow_id:
        publication = None
    steps = workflow_view.build_steps(
        run, results, batch=batch,
        decisions=[presenters.present_decision(d) for d in decisions],
        publication=publication, role_labels=_role_labels())
    out["steps"] = steps
    out["current_step"] = workflow_view.current_step(steps)
    return out


@app.get("/ops/workflows/{workflow_id}")
def get_workflow(workflow_id: str, client: Optional[str] = None,
                 principal: Principal = Depends(authenticate)) -> Dict[str, Any]:
    eng = get_engine()
    run = _load_owned_workflow(eng, principal, workflow_id, client)
    return {"ok": True, "workflow": _full_workflow(eng, run.client_id,
                                                   run.workflow_id)}


@app.post("/ops/workflows/{workflow_id}/rerun")
def rerun_workflow(workflow_id: str, client: Optional[str] = None,
                   principal: Principal = Depends(authenticate)) -> Dict[str, Any]:
    eng = get_engine()
    run = _load_owned_workflow(eng, principal, workflow_id, client)
    eng.rerun(run, actor=principal.name)
    return {"ok": True, "workflow": _full_workflow(eng, run.client_id,
                                                   run.workflow_id)}


@app.post("/ops/workflows/{workflow_id}/cancel")
def cancel_workflow(workflow_id: str, body: ReasonBody,
                    client: Optional[str] = None,
                    principal: Principal = Depends(authenticate)) -> Dict[str, Any]:
    eng = get_engine()
    run = _load_owned_workflow(eng, principal, workflow_id, client)
    eng.cancel(run, actor=principal.name, reason=body.reason)
    return {"ok": True, "workflow": _full_workflow(eng, run.client_id,
                                                   run.workflow_id)}


# --------------------------------------------------------------------------- #
# Review Centre
# --------------------------------------------------------------------------- #

@app.get("/ops/reviews")
def list_reviews(client: Optional[str] = None,
                 workflow_id: Optional[str] = None,
                 principal: Principal = Depends(authenticate)) -> Dict[str, Any]:
    eng = get_engine()
    clients = ([client] if client
               else principal.visible_clients(eng.store.known_clients()))
    items: List[Dict[str, Any]] = []
    for c in clients:
        require_client(principal, c)
        for d in eng.store.open_decisions(c, workflow_id):
            items.append(presenters.present_decision(d))
    items.sort(key=lambda d: (not d["blocking"], d.get("created_at") or ""))
    return {"ok": True, "reviews": items}


@app.get("/ops/reviews/{decision_id}")
def get_review(decision_id: str, client: Optional[str] = None,
               principal: Principal = Depends(authenticate)) -> Dict[str, Any]:
    eng = get_engine()
    doc = _load_owned_decision(eng, principal, decision_id, client)
    return {"ok": True, "review": presenters.present_decision(doc)}


def _load_owned_decision(eng: OpsEngine, principal: Principal,
                         decision_id: str, client: Optional[str]):
    candidates = ([client] if client
                  else principal.visible_clients(eng.store.known_clients()))
    for c in candidates:
        if not principal.allows(c):
            continue
        doc = eng.store.load_decision(c, decision_id)
        if doc is not None:
            require_client(principal, doc.get("client_id", c))
            return doc
    raise HTTPException(status_code=404, detail={
        "errorCode": "OPS_NOT_FOUND", "message": "That could not be found."})


@app.post("/ops/reviews/{decision_id}/decision")
def decide(decision_id: str, body: DecisionBody, client: Optional[str] = None,
           principal: Principal = Depends(authenticate)) -> Dict[str, Any]:
    eng = get_engine()
    doc = _load_owned_decision(eng, principal, decision_id, client)
    result = eng.resolve_decision(
        client_id=doc["client_id"], decision_id=decision_id,
        action=body.action, actor=principal.name, value=body.value,
        scope=body.scope, reason=body.reason,
        actor_is_admin=principal.is_admin)
    return {"ok": True,
            "review": presenters.present_decision(result["decision"]),
            "rule": (presenters.present_rule(result["rule"])
                     if result.get("rule") else None),
            "rerun_scheduled": result.get("rerun_scheduled", False)}


# --------------------------------------------------------------------------- #
# Rules Library
# --------------------------------------------------------------------------- #

@app.get("/ops/rules")
def list_rules(client: Optional[str] = None, q: str = "",
               kind: Optional[str] = None, scope: Optional[str] = None,
               principal: Principal = Depends(authenticate)) -> Dict[str, Any]:
    eng = get_engine()
    clients = ([client] if client
               else principal.visible_clients(eng.store.known_clients()))
    rules: List[Dict[str, Any]] = []
    for c in clients:
        require_client(principal, c)
        rules.extend(r.to_dict() for r in eng.rules.list_current(c))
    rules.extend(r.to_dict() for r in eng.rules.list_current(None))
    out = []
    for r in rules:
        if kind and r.get("kind") != kind:
            continue
        if scope and r.get("scope") != scope:
            continue
        pr = presenters.present_rule(r)
        if q and q.lower() not in (pr["source_term"] + " "
                                   + pr["approved_meaning"] + " "
                                   + pr["description"]).lower():
            continue
        out.append(pr)
    return {"ok": True, "rules": out}


@app.get("/ops/rules/{rule_id}/history")
def rule_history(rule_id: str, client: Optional[str] = None,
                 principal: Principal = Depends(authenticate)) -> Dict[str, Any]:
    eng = get_engine()
    candidates = ([client] if client
                  else principal.visible_clients(eng.store.known_clients()))
    for c in [*candidates, None]:
        if c is not None and not principal.allows(c):
            continue
        history = eng.rules.history(c, rule_id)
        if history:
            return {"ok": True,
                    "history": [presenters.present_rule(r.to_dict())
                                for r in history]}
    raise HTTPException(status_code=404, detail={
        "errorCode": "OPS_NOT_FOUND", "message": "That could not be found."})


# --------------------------------------------------------------------------- #
# Publication + history
# --------------------------------------------------------------------------- #

class PublishBody(BaseModel):
    #: How far the operator intended this approval to reach. Recorded and
    #: audited; the platform-wide scope is not reachable from a delivery.
    remember_scope: str = PUBLICATION_SCOPE_DEFAULT


@app.post("/ops/workflows/{workflow_id}/publish")
def approve_publication(workflow_id: str, body: Optional[PublishBody] = None,
                        client: Optional[str] = None,
                        principal: Principal = Depends(authenticate)) -> Dict[str, Any]:
    eng = get_engine()
    run = _load_owned_workflow(eng, principal, workflow_id, client)
    pub = eng.approve_publication(client_id=run.client_id,
                                  workflow_id=workflow_id,
                                  actor=principal.name,
                                  remember_scope=(body.remember_scope if body
                                                  else PUBLICATION_SCOPE_DEFAULT))
    return {"ok": True, "publication": presenters.present_publication(pub)}


@app.post("/ops/workflows/{workflow_id}/hold")
def reject_publication(workflow_id: str, body: ReasonBody,
                       client: Optional[str] = None,
                       principal: Principal = Depends(authenticate)) -> Dict[str, Any]:
    eng = get_engine()
    run = _load_owned_workflow(eng, principal, workflow_id, client)
    pub = eng.reject_publication(client_id=run.client_id,
                                 workflow_id=workflow_id,
                                 actor=principal.name, reason=body.reason)
    return {"ok": True, "publication": (presenters.present_publication(pub)
                                        if pub else None)}


@app.get("/ops/history")
def history(client: Optional[str] = None,
            principal: Principal = Depends(authenticate)) -> Dict[str, Any]:
    eng = get_engine()
    clients = ([client] if client
               else principal.visible_clients(eng.store.known_clients()))
    pubs: List[Dict[str, Any]] = []
    for c in clients:
        require_client(principal, c)
        pubs.extend(presenters.present_publication(p)
                    for p in eng.store.list_publications(c))
    return {"ok": True, "history": pubs}


# --------------------------------------------------------------------------- #
# Administrator configuration area (system / regime / asset packages)
# --------------------------------------------------------------------------- #

class DraftBody(BaseModel):
    from_version: Optional[int] = None
    edits: Dict[str, str] = {}
    notes: str = ""


class RollbackBody(BaseModel):
    to_version: int


def _packages(eng: OpsEngine):
    from ..configuration.packages import ConfigPackageStore
    return ConfigPackageStore(eng.store)


def _admin_audit(eng: OpsEngine, event: str, actor: str, detail: Dict[str, Any]):
    eng.store.append_audit("_admin", event, actor=actor, detail=detail)


@app.get("/ops/admin/config")
def admin_config_overview(principal: Principal = Depends(authenticate)
                          ) -> Dict[str, Any]:
    require_admin(principal)
    from ..configuration import admin_views
    from ..configuration.packages import ASSET_MODEL
    eng = get_engine()
    view = admin_views.overview(eng.store, _packages(eng), by=principal.name)
    return {"ok": True, "asset_model": ASSET_MODEL, **view}


@app.get("/ops/admin/config/catalogue")
def admin_config_catalogue(principal: Principal = Depends(authenticate)
                           ) -> Dict[str, Any]:
    """Asset and regime entities plus the compatibility matrix between them."""
    require_admin(principal)
    from ..configuration import admin_views
    pkgs = _packages(get_engine())
    regimes = admin_views.describe_regimes(pkgs, pkgs.repo)
    assets = admin_views.with_readiness(admin_views.describe_assets(pkgs),
                                        regimes)
    return {"ok": True, "assets": assets, "regimes": regimes,
            "compatibility": admin_views.compatibility_matrix(assets, regimes),
            "issues": admin_views.compatibility_issues(assets, regimes)}


@app.get("/ops/admin/config/audit")
def admin_config_audit(limit: int = 200,
                       principal: Principal = Depends(authenticate)
                       ) -> Dict[str, Any]:
    """The administrator configuration audit trail, in plain English."""
    require_admin(principal)
    from ..configuration import admin_views
    trail = admin_views.audit_trail(get_engine().store,
                                    limit=max(1, min(limit, 500)))
    return {"ok": True, **trail}


@app.get("/ops/admin/config/{layer}/compare")
def admin_config_compare(layer: str, from_version: int, to_version: int,
                         principal: Principal = Depends(authenticate)
                         ) -> Dict[str, Any]:
    """Structured comparison between two versions of one package layer."""
    require_admin(principal)
    from ..configuration import admin_views
    pkgs = _packages(get_engine())
    pkgs.ensure_seeded(layer, by=principal.name)
    try:
        comparison = admin_views.compare(pkgs, layer, from_version, to_version)
    except ValueError:
        raise HTTPException(status_code=404, detail={
            "errorCode": "OPS_NOT_FOUND", "message": "That could not be found."})
    return {"ok": True, "comparison": comparison}


@app.get("/ops/admin/config/{layer}/impact")
def admin_config_impact(layer: str, version: Optional[int] = None,
                        principal: Principal = Depends(authenticate)
                        ) -> Dict[str, Any]:
    """Clients, portfolios and workflows pinned to a package version."""
    require_admin(principal)
    from ..configuration import admin_views
    eng = get_engine()
    pkgs = _packages(eng)
    if version is None:
        version = pkgs.ensure_seeded(layer, by=principal.name)["version"]
    return {"ok": True,
            "impact": admin_views.impact(eng.store, layer, int(version))}


@app.get("/ops/admin/config/{layer}/{version}")
def admin_config_version(layer: str, version: int, file: Optional[str] = None,
                         principal: Principal = Depends(authenticate)
                         ) -> Dict[str, Any]:
    require_admin(principal)
    pkgs = _packages(get_engine())
    pkgs.ensure_seeded(layer, by=principal.name)
    doc = pkgs.get_version(layer, version)
    if doc is None:
        raise HTTPException(status_code=404, detail={
            "errorCode": "OPS_NOT_FOUND", "message": "That could not be found."})
    from ..configuration import admin_views
    files = doc.get("files") or {}
    out = {k: v for k, v in doc.items() if k != "files"}
    out["files"] = {rel: {"sha256": f["sha256"],
                          "size": len(f.get("content", ""))}
                    for rel, f in files.items()}
    out["file_list"] = admin_views.describe_files(doc)
    out["package_hash"] = admin_views.package_hash(doc)
    out["dependencies"] = admin_views.LAYER_DEPENDENCIES.get(layer, [])
    out["validation_summary"] = admin_views.explain_validation(
        layer, doc.get("validation"))
    out["activation_blockers"] = admin_views.activation_blockers(
        pkgs, layer, version)
    if doc.get("based_on_version"):
        base = pkgs.get_version(layer, doc["based_on_version"])
        out["changed_files"] = [
            {"path": rel, "label": admin_views.friendly_file_name(rel)}
            for rel in admin_views.changed_files(base, doc)]
    if file and file in files:
        out["file_content"] = files[file]["content"]
        out["file_label"] = admin_views.friendly_file_name(file)
    return {"ok": True, "version": out}


@app.post("/ops/admin/config/{layer}/draft")
def admin_config_draft(layer: str, body: DraftBody,
                       principal: Principal = Depends(authenticate)
                       ) -> Dict[str, Any]:
    require_admin(principal)
    eng = get_engine()
    doc = _packages(eng).create_draft(layer, by=principal.name,
                                      from_version=body.from_version,
                                      edits=body.edits, notes=body.notes)
    _admin_audit(eng, "config_draft_created", principal.name,
                 {"layer": layer, "version": doc["version"],
                  "based_on": doc.get("based_on_version"),
                  "edited_files": sorted(body.edits)})
    return {"ok": True, "version": doc["version"], "status": doc["status"]}


@app.post("/ops/admin/config/{layer}/{version}/validate")
def admin_config_validate(layer: str, version: int,
                          principal: Principal = Depends(authenticate)
                          ) -> Dict[str, Any]:
    require_admin(principal)
    eng = get_engine()
    from ..configuration import admin_views
    doc = _packages(eng).validate_version(layer, version)
    _admin_audit(eng, "config_validated", principal.name,
                 {"layer": layer, "version": version,
                  "ok": doc["validation"]["ok"],
                  "problems": doc["validation"]["problems"]})
    return {"ok": True, "validation": doc["validation"],
            "validation_summary": admin_views.explain_validation(
                layer, doc["validation"])}


@app.post("/ops/admin/config/{layer}/{version}/activate")
def admin_config_activate(layer: str, version: int,
                          principal: Principal = Depends(authenticate)
                          ) -> Dict[str, Any]:
    require_admin(principal)
    from ..configuration import admin_views
    eng = get_engine()
    pkgs = _packages(eng)
    blockers = admin_views.activation_blockers(pkgs, layer, version)
    if blockers:
        raise HTTPException(status_code=409, detail={
            "errorCode": "OPS_CONFIG_INCOMPATIBLE",
            "message": blockers[0]["message"],
            "blockers": blockers})
    try:
        doc = pkgs.activate_version(layer, version, by=principal.name)
    except ValueError:
        raise HTTPException(status_code=409, detail={
            "errorCode": "OPS_CONFIG_NOT_READY",
            "message": "This version has not passed its checks, so it cannot "
                       "be activated. Check it first."})
    _admin_audit(eng, "config_activated", principal.name,
                 {"layer": layer, "version": version})
    return {"ok": True, "active_version": doc["version"]}


@app.post("/ops/admin/config/{layer}/rollback")
def admin_config_rollback(layer: str, body: RollbackBody,
                          principal: Principal = Depends(authenticate)
                          ) -> Dict[str, Any]:
    require_admin(principal)
    from ..configuration import admin_views
    eng = get_engine()
    pkgs = _packages(eng)
    blockers = admin_views.activation_blockers(pkgs, layer, body.to_version)
    if blockers:
        raise HTTPException(status_code=409, detail={
            "errorCode": "OPS_CONFIG_INCOMPATIBLE",
            "message": blockers[0]["message"],
            "blockers": blockers})
    doc = pkgs.rollback(layer, body.to_version, by=principal.name)
    _admin_audit(eng, "config_rolled_back", principal.name,
                 {"layer": layer, "to_version": body.to_version})
    return {"ok": True, "active_version": doc["version"]}


@app.get("/ops/audit")
def audit(client: str, workflow_id: Optional[str] = None,
          principal: Principal = Depends(authenticate)) -> Dict[str, Any]:
    require_client(principal, client)
    eng = get_engine()
    rows = eng.store.list_audit(client)
    if workflow_id:
        rows = [r for r in rows if r.get("workflow_id") == workflow_id]
    return {"ok": True, "audit": rows,
            "chain_intact": eng.store.verify_audit_chain(client)}
