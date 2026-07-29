"""operations_control.api.auth — fail-closed operator auth + tenancy binding.

Every operator principal is TENANT-BOUND: it carries the explicit list of
client ids it may operate on (or ``*`` for all). Client, portfolio and blob
identifiers supplied by the browser are never trusted — every route resolves
the resource server-side and checks the principal's client binding; a resource
outside the binding answers 404 (existence is not revealed).

Configuration (fail-closed — no config means NO access, HTTP 503):

    TRAKT_OPS_OPERATORS='{"<token>": {"name": "Alice", "clients": ["ERE"]}}'

The token arrives as ``X-Operator-Token`` or ``Authorization: Bearer <token>``
(the same mechanism as the existing operator console). Swapping this dependency
for Entra/Easy-Auth principals later changes only this module.
"""

from __future__ import annotations

import hmac
import json
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from fastapi import Header, HTTPException

OPERATORS_ENV = "TRAKT_OPS_OPERATORS"


@dataclass(frozen=True)
class Principal:
    name: str
    clients: tuple = field(default_factory=tuple)   # ("*",) = all clients

    def allows(self, client_id: str) -> bool:
        return "*" in self.clients or client_id in self.clients

    def visible_clients(self, known: List[str]) -> List[str]:
        if "*" in self.clients:
            return list(known)
        return [c for c in known if c in self.clients]


def _load_operators() -> Dict[str, Principal]:
    raw = os.environ.get(OPERATORS_ENV, "")
    if not raw.strip():
        return {}
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return {}
    out: Dict[str, Principal] = {}
    for token, spec in data.items():
        if not isinstance(spec, dict):
            continue
        out[token] = Principal(name=str(spec.get("name") or "operator"),
                               clients=tuple(spec.get("clients") or ()))
    return out


def authenticate(x_operator_token: Optional[str] = Header(default=None),
                 authorization: Optional[str] = Header(default=None)) -> Principal:
    """FastAPI dependency. Fail-closed: unconfigured -> 503; bad token -> 401."""
    operators = _load_operators()
    if not operators:
        raise HTTPException(status_code=503, detail={
            "errorCode": "OPS_AUTH_UNCONFIGURED",
            "message": "The Operations Control Centre is not set up yet. "
                       "Ask your administrator."})
    supplied = x_operator_token or ""
    if not supplied and authorization and authorization.lower().startswith("bearer "):
        supplied = authorization[7:].strip()
    for token, principal in operators.items():
        if supplied and hmac.compare_digest(supplied, token):
            return principal
    raise HTTPException(status_code=401, detail={
        "errorCode": "OPS_UNAUTHENTICATED",
        "message": "Please sign in to continue."})


def require_client(principal: Principal, client_id: str) -> None:
    """Server-side tenancy check. Outside the binding -> 404 (not 403), so the
    existence of other clients' resources is never revealed."""
    if not client_id or not principal.allows(client_id):
        raise HTTPException(status_code=404, detail={
            "errorCode": "OPS_NOT_FOUND",
            "message": "That could not be found."})
