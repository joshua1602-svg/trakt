"""mi_agent_api/agent_api.py — the external agent tool API.

    GET  /v1/agent/tools              the tools THIS caller may use, with schemas
    POST /v1/agent/tools/{tool_name}  invoke one, with typed JSON arguments

A transport adapter and nothing else. It does exactly three things — turn a
validated machine identity into an :class:`~trakt_core.context.ExecutionContext`,
hand the JSON body to :func:`trakt_tools.execute_governed_tool`, and serialise
the :class:`~trakt_core.envelope.GovernedResult`. No business logic, no schema
decisions, no permission decisions and no dataset choice live here.

Deliberately NOT exposed
------------------------
* **The internal FastAPI surface.** The ~35 ``/mi/*`` routes are shaped for the
  React workspace, are not governed capabilities, and several are not scope-gated
  at all. This router is a separate, closed surface: an agent can reach exactly
  what the tool registry declares.
* **The natural-language MI interpreter.** ``POST /mi/query`` takes prose and
  runs Trakt's own LLM over it. Routing an external agent through that would put
  two probabilistic hops in one call — the agent choosing words and Trakt
  guessing what they meant. An agent calls a typed tool with typed arguments;
  choosing *which* tool is the only inference in the loop.
* **Anything about the deployment's other organisations.** The catalogue is
  narrowed to the caller's own grants, and an unauthorised resource is refused
  identically to a nonexistent one.

The response body is the full ``GovernedResult.to_dict()`` — capability, schema
version, status, request id, correlation id, tenant, resource, snapshot, policy
state, provenance, audit metadata and a typed error. An agent reads ``status``
and ``error.code``, never prose.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, Optional

from fastapi import APIRouter, Depends, Request
from fastapi.responses import JSONResponse

import trakt_tools
from trakt_core.errors import ErrorCode, TraktError, http_status_for

from . import agent_auth
from . import identity as identity_mod
from .dependencies import default_tenant_id

logger = logging.getLogger("mi_agent_api.agent_api")

router = APIRouter(prefix="/v1/agent", tags=["agent"],
                   dependencies=[Depends(agent_auth.agent_auth_guard)])


def _context(request: Request):
    """The trusted context for this agent call.

    The tenant is deployment configuration. The organisation and the entitlements
    come from the verified directory. The request id honours an inbound
    ``X-Request-Id`` / ``X-Correlation-Id`` so an agent's own trace joins up with
    Trakt's audit trail — which is what makes "show me every tool call behind
    this decision" answerable later.
    """
    principal = getattr(request.state, "agent_principal", None)
    return identity_mod.context_from_agent_principal(
        principal,
        tenant_id=default_tenant_id(),
        request_id=request.headers.get("x-request-id") or None,
        correlation_id=request.headers.get("x-correlation-id") or None,
    )


def _error_response(err: TraktError, *, request_id: Optional[str] = None
                    ) -> JSONResponse:
    """A governed refusal in the same envelope shape as a governed answer.

    An agent should not need two parsers — one for results and one for failures
    that happened before a result could exist.
    """
    payload = {
        "capability": None,
        "schema_version": None,
        "status": "blocked",
        "request_id": err.request_id or request_id,
        "result": None,
        "error": err.to_dict(),
    }
    return JSONResponse(status_code=http_status_for(err.code), content=payload)


@router.get("/tools")
def list_tools(request: Request) -> Any:
    """The tool catalogue for this caller, with JSON Schemas.

    Narrowed to the capabilities this agent's grants actually confer, and
    accompanied by the closed set of resource references it may name. An agent
    that reads this needs to guess nothing: it knows which tools it can call and
    which resources it can call them against.
    """
    try:
        context = _context(request)
    except TraktError as err:
        return _error_response(err)
    payload = trakt_tools.catalogue(context)
    payload["request_id"] = context.request_id
    return payload


@router.post("/tools/{tool_name}")
def invoke_tool(tool_name: str, arguments: Optional[Dict[str, Any]] = None,
                request: Request = None) -> Any:
    """Invoke one governed tool.

    The body IS the arguments object, validated against the tool's published
    input schema before anything else happens. Governance — capability,
    entitlement, data approval — then runs before any data is touched, and the
    whole execution is audited whatever the outcome.
    """
    try:
        context = _context(request)
    except TraktError as err:
        return _error_response(err)

    result = trakt_tools.execute_governed_tool(
        tool_name, arguments or {}, context)
    payload = result.to_dict()
    status = result.http_status
    if status == 200:
        return payload
    return JSONResponse(status_code=status, content=payload)


def enabled() -> bool:
    """Whether this deployment mounts the agent API.

    Off by default. Mounting a machine-callable surface is a deployment decision
    an operator makes explicitly, not something that appears because the code
    shipped — the same posture ``teams_bot.enabled()`` takes.
    """
    return (os.environ.get("TRAKT_AGENT_API_ENABLED") or "").strip().lower() in (
        "1", "true", "yes", "on")
