"""Auth enforcement tests for the MI Agent API.

These opt back into auth (the suite disables it by default via conftest) and
verify: probe routes stay open; protected routes require a platform principal;
an authenticated caller the governed access directory does not list is refused;
provisioned client and operator identities are admitted and distinguished; and
/health does not leak the dataset path.

Authorisation moved out of the platform's roles and into the directory
(``trakt_core.access``), so these tests provision identities rather than
attaching roles to the principal.
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
from trakt_core import access

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


#: The identities the helpers above present, provisioned in the governed access
#: directory. Authorisation no longer comes from the roles in the principal —
#: it comes from here — so a test that expects admission has to provision the
#: identity, exactly as a real deployment does.
_DIRECTORY = """
version: 1
principals:
  - identities: [user@client-domain.com]
    tenant_id: {tenant}
    role: client
  - identities: [ops@trakt.io]
    tenant_id: {tenant}
    role: operator
"""


@pytest.fixture()
def auth_on(monkeypatch, tmp_path):
    monkeypatch.setenv("MI_AGENT_AUTH_ENABLED", "true")
    monkeypatch.setenv("MI_AGENT_CLIENT_ROLE", "client")
    monkeypatch.setenv("MI_AGENT_OPERATOR_ROLE", "operator")
    monkeypatch.setenv("MI_AGENT_CLIENT_ID", "test-tenant")
    path = tmp_path / "access.yaml"
    path.write_text(_DIRECTORY.format(tenant="test-tenant"), encoding="utf-8")
    monkeypatch.setenv("TRAKT_ACCESS_CONFIG", str(path))
    access.reset_cache()
    yield
    access.reset_cache()


# ---- guard behaviour --------------------------------------------------------

def test_health_is_open_without_auth(auth_on):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["ok"] is True


def test_protected_route_requires_principal(auth_on):
    r = client.get("/me")
    assert r.status_code == 401


def test_authenticated_but_unprovisioned_is_forbidden(auth_on):
    """Signed in, but nobody Trakt knows. Refused, and told why."""
    hdr = _swa_principal(["authenticated"], user="stranger@elsewhere.com")
    r = client.get("/me", headers={"x-ms-client-principal": hdr})
    assert r.status_code == 403
    assert r.json()["errorCode"] == "ACCESS_NOT_PROVISIONED"


def test_a_provisioned_client_is_admitted(auth_on):
    """Admission comes from the directory. The principal here carries NO MI
    role — only "authenticated" — which is exactly what SWA sends once
    invitation roles are abandoned."""
    hdr = _swa_principal(["authenticated"])
    r = client.get("/me", headers={"x-ms-client-principal": hdr})
    assert r.status_code == 200
    body = r.json()
    assert body["authenticated"] is True
    assert body["isOperator"] is False
    assert body["role"] == "client"


def test_a_provisioned_operator_is_admitted_and_flagged(auth_on):
    hdr = _easyauth_principal([])
    r = client.get("/me", headers={"x-ms-client-principal": hdr})
    assert r.status_code == 200
    assert r.json()["isOperator"] is True


def test_a_platform_role_cannot_substitute_for_provisioning(auth_on):
    """The old authority is now inert: claiming ``operator`` in the principal
    does not admit an identity the directory does not list. This is the
    property that makes the directory authoritative rather than additive."""
    hdr = _swa_principal(["authenticated", "operator"], user="stranger@elsewhere.com")
    r = client.get("/me", headers={"x-ms-client-principal": hdr})
    assert r.status_code == 403
    assert r.json()["errorCode"] == "ACCESS_NOT_PROVISIONED"


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
