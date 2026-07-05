"""Auth enforcement tests for the MI Agent API.

These opt back into auth (the suite disables it by default via conftest) and
verify: probe routes stay open; protected routes require a platform principal;
an authenticated caller without an MI role is refused; client and operator roles
are admitted and distinguished; and /health does not leak the dataset path.
"""

from __future__ import annotations

import base64
import json
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from mi_agent_api.app import app
from mi_agent_api import auth as auth_mod

client = TestClient(app)


def _swa_principal(roles, user="user@client-domain.com", idp="aad") -> str:
    payload = {"identityProvider": idp, "userId": "oid-123",
               "userDetails": user, "userRoles": list(roles)}
    return base64.b64encode(json.dumps(payload).encode("utf-8")).decode("ascii")


def _easyauth_principal(roles, name="ops@trakt.io") -> str:
    claims = [{"typ": "roles", "val": r} for r in roles]
    claims.append({"typ": "name", "val": name})
    payload = {"auth_typ": "aad", "name_typ": "name", "role_typ": "roles", "claims": claims}
    return base64.b64encode(json.dumps(payload).encode("utf-8")).decode("ascii")


@pytest.fixture()
def auth_on(monkeypatch):
    monkeypatch.setenv("MI_AGENT_AUTH_ENABLED", "true")
    monkeypatch.setenv("MI_AGENT_CLIENT_ROLE", "client")
    monkeypatch.setenv("MI_AGENT_OPERATOR_ROLE", "operator")
    yield


# ---- guard behaviour --------------------------------------------------------

def test_health_is_open_without_auth(auth_on):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["ok"] is True


def test_protected_route_requires_principal(auth_on):
    r = client.get("/me")
    assert r.status_code == 401


def test_authenticated_without_mi_role_is_forbidden(auth_on):
    hdr = _swa_principal(["authenticated"])
    r = client.get("/me", headers={"x-ms-client-principal": hdr})
    assert r.status_code == 403


def test_client_role_is_admitted(auth_on):
    hdr = _swa_principal(["authenticated", "client"])
    r = client.get("/me", headers={"x-ms-client-principal": hdr})
    assert r.status_code == 200
    body = r.json()
    assert body["authenticated"] is True
    assert body["isOperator"] is False
    assert "client" in body["roles"]


def test_operator_role_is_admitted_and_flagged(auth_on):
    hdr = _easyauth_principal(["operator"])
    r = client.get("/me", headers={"x-ms-client-principal": hdr})
    assert r.status_code == 200
    assert r.json()["isOperator"] is True


def test_health_does_not_leak_dataset_path(auth_on):
    r = client.get("/health")
    body = r.json()
    assert "dataSourceInfo" not in body
    # no value anywhere in the health body should be an absolute server path
    flat = json.dumps(body)
    assert "/home/" not in flat and "\\Users\\" not in flat


# ---- principal parsing (unit) ----------------------------------------------

def test_parse_swa_shape():
    p = auth_mod.parse_principal(_swa_principal(["authenticated", "client"], user="a@b.com"))
    assert p is not None
    assert p.user_details == "a@b.com"
    assert p.is_client and not p.is_operator


def test_parse_easyauth_shape():
    p = auth_mod.parse_principal(_easyauth_principal(["operator"], name="ops@trakt.io"))
    assert p is not None
    assert p.is_operator
    assert p.user_details == "ops@trakt.io"


def test_parse_invalid_header_returns_none():
    assert auth_mod.parse_principal("not-base64!!") is None
    assert auth_mod.parse_principal("") is None
    assert auth_mod.parse_principal(None) is None


def test_auth_disabled_injects_synthetic_operator(monkeypatch):
    monkeypatch.setenv("MI_AGENT_AUTH_ENABLED", "false")
    r = client.get("/me")
    assert r.status_code == 200
    assert r.json()["isOperator"] is True


def test_empty_auth_flag_fails_closed(monkeypatch):
    # An empty MI_AGENT_AUTH_ENABLED (blanked / forgotten env var) must NOT run
    # the API open — it enforces auth just like the unset default.
    monkeypatch.setenv("MI_AGENT_AUTH_ENABLED", "")
    assert auth_mod._auth_enabled() is True
    r = client.get("/me")
    assert r.status_code == 401


def test_whitespace_auth_flag_fails_closed(monkeypatch):
    monkeypatch.setenv("MI_AGENT_AUTH_ENABLED", "   ")
    assert auth_mod._auth_enabled() is True
    r = client.get("/me")
    assert r.status_code == 401


@pytest.mark.parametrize("val", ["false", "0", "no", "off", "FALSE", "Off"])
def test_explicit_opt_out_disables_auth(monkeypatch, val):
    monkeypatch.setenv("MI_AGENT_AUTH_ENABLED", val)
    assert auth_mod._auth_enabled() is False
