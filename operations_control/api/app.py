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

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from ..contracts import (
    DEC_OPEN,
    RUN_AWAITING_PUBLICATION,
    RUN_BLOCKED,
    RUN_NEEDS_REVIEW,
    RUN_PUBLISHED,
    RUN_RECEIVED,
)
from ..engine import OpsEngine, OpsError
from ..stores import OpsStore
from . import presenters
from .auth import Principal, authenticate, require_client

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


@app.exception_handler(OpsError)
async def _ops_error(request: Request, exc: OpsError):
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
    n_open = len(eng.store.open_decisions(client_id, workflow_id))
    return presenters.present_workflow(run, results, n_open)


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
        scope=body.scope, reason=body.reason)
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

@app.post("/ops/workflows/{workflow_id}/publish")
def approve_publication(workflow_id: str, client: Optional[str] = None,
                        principal: Principal = Depends(authenticate)) -> Dict[str, Any]:
    eng = get_engine()
    run = _load_owned_workflow(eng, principal, workflow_id, client)
    pub = eng.approve_publication(client_id=run.client_id,
                                  workflow_id=workflow_id,
                                  actor=principal.name)
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
