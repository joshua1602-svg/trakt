#!/usr/bin/env python3
"""The browser-to-API authentication contract, end to end.

Three things were true before this file existed, and none of them was tested:

  * the SPA was built with an absolute cross-origin API URL, so the browser sent
    neither a cookie nor a token and the API received anonymous requests;
  * the Static Web App config declared no identity provider and no role
    restriction, so nothing performed a sign-in;
  * ``test_auth.py`` injected ``X-MS-CLIENT-PRINCIPAL`` by hand, proving the
    PARSER worked while proving nothing about whether that header could ever
    arrive.

Every existing auth test passes on the broken deployment. That is the gap these
tests close: they assert the relationship between what the front end is built to
call, what the Static Web App enforces, and what the API requires — the three
sides that have to agree for a request to be authenticated at all.
"""

from __future__ import annotations

import base64
import json
import sys
import unittest
from pathlib import Path
from typing import Any, Dict, List

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import pytest
from fastapi.testclient import TestClient

from mi_agent_api.app import app
from mi_agent_api import gateway
from trakt_core import access

SWA_CONFIG = _REPO_ROOT / "frontend" / "mi-agent-ui" / "staticwebapp.config.json"
SWA_WORKFLOW = (_REPO_ROOT / ".github" / "workflows"
                / "azure-static-web-apps-nice-smoke-067ac7603.yml")

client = TestClient(app)


def _principal(roles: List[str], user: str = "user@client-domain.com") -> str:
    """The SWA-shaped principal a linked backend forwards."""
    return base64.b64encode(json.dumps({
        "identityProvider": "aad", "userId": "oid-123",
        "userDetails": user, "userRoles": roles,
    }).encode("utf-8")).decode("ascii")


#: Provisioned identities for these tests. Note what is NOT here: any role in
#: the principal itself. Since authorisation moved to the governed directory,
#: the principal only has to say who signed in — which is all SWA sends once
#: invitation roles are out of the picture.
_DIRECTORY = """
version: 1
principals:
  - identities: [user@client-domain.com]
    tenant_id: test-tenant
    role: client
  - identities: [ops@trakt.io]
    tenant_id: test-tenant
    role: operator
"""


@pytest.fixture()
def auth_on(monkeypatch, tmp_path):
    monkeypatch.setenv("MI_AGENT_AUTH_ENABLED", "true")
    monkeypatch.setenv("MI_AGENT_CLIENT_ROLE", "client")
    monkeypatch.setenv("MI_AGENT_OPERATOR_ROLE", "operator")
    monkeypatch.setenv("MI_AGENT_CLIENT_ID", "test-tenant")
    path = tmp_path / "access.yaml"
    path.write_text(_DIRECTORY, encoding="utf-8")
    monkeypatch.setenv("TRAKT_ACCESS_CONFIG", str(path))
    access.reset_cache()
    yield
    access.reset_cache()


def _mi_routes() -> List[tuple]:
    """Every literal MI route the app serves, with a method it accepts.

    Discovered rather than listed: a hand-written list goes stale the moment a
    route is added, which is precisely how an unprotected endpoint ships.

    The method matters. FastAPI resolves the auth dependency AFTER matching the
    method, so a GET to a POST-only route returns 405 without the guard ever
    running — no data, but not the 401 this file is asserting either. Exercising
    each route with a method it accepts is what makes the assertion meaningful.
    """
    found = {}
    for route in app.routes:
        path = getattr(route, "path", None)
        methods = getattr(route, "methods", None) or set()
        if not isinstance(path, str) or not path.startswith("/mi/") or "{" in path:
            continue
        verb = "GET" if "GET" in methods else ("POST" if "POST" in methods else None)
        if verb:
            found[path] = verb
    return sorted(found.items())


def _call(path: str, verb: str, headers=None):
    if verb == "POST":
        return client.post(path, json={}, headers=headers)
    return client.get(path, headers=headers)


# =========================================================================== #
# 1. No direct anonymous MI access
# =========================================================================== #
class TestNoAnonymousMiAccess:
    """With auth on, an unauthenticated caller reaches no MI data at all."""

    def test_there_are_mi_routes_to_protect(self):
        # Guards the two tests below from silently passing on an empty list.
        assert len(_mi_routes()) >= 15

    def test_every_mi_route_refuses_an_anonymous_caller(self, auth_on):
        checked = []
        for path, verb in _mi_routes():
            r = _call(path, verb)
            assert r.status_code == 401, (
                f"{verb} {path} answered {r.status_code} to an anonymous "
                "caller; every /mi/* route must require a principal")
            checked.append(path)
        assert checked, "no route was exercised"

    def test_the_same_holds_under_the_linked_backend_prefix(self, auth_on):
        """The /api form is the DEPLOYED one — it must not be a way in."""
        prefix = gateway.api_prefix() or "/api"
        for path, verb in _mi_routes():
            r = _call(f"{prefix}{path}", verb)
            assert r.status_code == 401, (
                f"{verb} {prefix}{path} answered {r.status_code} anonymously")

    def test_the_identity_route_refuses_an_anonymous_caller(self, auth_on):
        assert client.get("/me").status_code == 401
        assert client.get("/api/me").status_code == 401

    def test_an_authenticated_but_unprovisioned_caller_gets_nothing(self, auth_on):
        """Signed in is not the same as entitled.

        A real Entra tenant will happily authenticate anyone the app
        registration admits, so "the platform signed them in" cannot be the
        whole check. This is the route-by-route proof that it is not.
        """
        hdr = {"x-ms-client-principal": _principal(
            ["anonymous", "authenticated"], user="stranger@elsewhere.com")}
        for path, verb in _mi_routes():
            r = _call(path, verb, headers=hdr)
            assert r.status_code == 403, (
                f"{verb} {path} admitted an unprovisioned caller")
            assert r.json().get("errorCode") == "ACCESS_NOT_PROVISIONED", (
                f"{verb} {path} refused without saying the account needs "
                "provisioning — the UI cannot explain a bare 403")

    def test_a_malformed_principal_is_not_a_way_in(self, auth_on):
        for bad in ("", "not-base64", base64.b64encode(b"[]").decode(),
                    base64.b64encode(b"{").decode()):
            r = client.get("/me", headers={"x-ms-client-principal": bad})
            assert r.status_code == 401, f"malformed principal {bad!r} was admitted"

    def test_health_stays_open_so_the_platform_can_probe_it(self, auth_on):
        assert client.get("/health").status_code == 200


# =========================================================================== #
# 2. Linked-backend principal handling
# =========================================================================== #
class TestLinkedBackendPrincipal:
    """The path a real request takes: SWA authenticates, forwards the principal
    under /api, the gateway strips the prefix, the guard resolves the identity
    against the governed directory.

    Every principal below carries ``["authenticated"]`` and nothing more. That
    is deliberate and is the central claim of this file after the migration: it
    is exactly what Static Web Apps sends for a signed-in user with no
    invitation role, which is the state every real user is in.
    """

    def test_a_provisioned_client_under_the_api_prefix_is_admitted(self, auth_on):
        r = client.get("/api/me", headers={
            "x-ms-client-principal": _principal(["authenticated"])})
        assert r.status_code == 200
        body = r.json()
        assert body["authenticated"] is True
        assert body["role"] == "client"
        assert body["isOperator"] is False

    def test_a_provisioned_operator_is_admitted_and_flagged(self, auth_on):
        r = client.get("/api/me", headers={
            "x-ms-client-principal": _principal(["authenticated"], user="ops@trakt.io")})
        assert r.status_code == 200
        assert r.json()["isOperator"] is True

    def test_the_prefixed_and_bare_forms_resolve_to_the_same_route(self, auth_on):
        hdr = {"x-ms-client-principal": _principal(["authenticated"])}
        assert client.get("/me", headers=hdr).json() == \
            client.get("/api/me", headers=hdr).json()

    def test_an_mi_route_is_served_under_the_api_prefix(self, auth_on):
        """The whole point of the switch: /api/mi/... must not 404."""
        hdr = {"x-ms-client-principal": _principal(["authenticated"])}
        r = client.get("/api/mi/snapshots", headers=hdr)
        assert r.status_code == 200, (
            "the linked-backend path form must resolve — this exact mismatch "
            "produced the production 404s that gateway.py was written to fix")

    def test_an_unknown_prefixed_path_still_404s_with_its_own_path(self, auth_on):
        hdr = {"x-ms-client-principal": _principal(["authenticated"])}
        assert client.get("/api/mi/not-a-real-route", headers=hdr).status_code == 404

    def test_the_easy_auth_principal_shape_is_also_accepted(self, auth_on):
        """App Service Easy Auth sends a claims list, not userRoles — both are
        the platform's, and both must work. The directory looks up the name
        claim, so no role claim is needed."""
        payload = {"auth_typ": "aad", "name_typ": "name", "role_typ": "roles",
                   "claims": [{"typ": "name", "val": "ops@trakt.io"}]}
        hdr = base64.b64encode(json.dumps(payload).encode()).decode()
        r = client.get("/api/me", headers={"x-ms-client-principal": hdr})
        assert r.status_code == 200
        assert r.json()["isOperator"] is True

    def test_the_health_route_reports_the_path_forms_it_answers_on(self):
        """One request should diagnose a topology mismatch."""
        routing = client.get("/health").json().get("routing") or {}
        assert routing, "/health must disclose its routing so a deployment is diagnosable"


# =========================================================================== #
# 3. The three sides have to agree
# =========================================================================== #
class TestDeploymentContract:
    """The front end, the Static Web App and the API must describe the same
    topology. Nothing enforced that before, which is how the deployment came to
    send anonymous cross-origin requests to an API expecting a platform header.
    """

    @staticmethod
    def _swa() -> Dict[str, Any]:
        return json.loads(SWA_CONFIG.read_text(encoding="utf-8"))

    def test_the_front_end_is_built_to_call_the_api_same_origin(self):
        text = SWA_WORKFLOW.read_text(encoding="utf-8")
        assert "VITE_AGENT_API_URL: /api" in text, (
            "the bundle must call the linked backend same-origin; an absolute "
            "URL sends no cookie and no token, so the API sees an anonymous "
            "request and Easy Auth has nothing to validate")
        assert "VITE_AGENT_API_URL: https://" not in text

    def test_the_static_web_app_declares_an_identity_provider(self):
        providers = (self._swa().get("auth") or {}).get("identityProviders") or {}
        aad = providers.get("azureActiveDirectory") or {}
        registration = aad.get("registration") or {}
        assert registration.get("openIdIssuer"), "no Entra issuer configured"
        assert registration.get("clientIdSettingName")
        assert registration.get("clientSecretSettingName")

    def test_the_whole_site_requires_a_signed_in_user(self):
        routes = self._swa().get("routes") or []
        catch_all = [r for r in routes if r.get("route") == "/*"]
        assert catch_all, "no catch-all route rule — the site would be public"
        assert catch_all[0].get("allowedRoles") == ["authenticated"]

    def test_the_api_route_requires_authentication_only(self):
        """SWA is the AUTHENTICATION boundary, not the authorisation one.

        It used to require ``client``/``operator`` here, which meant every user
        needed an accepted SWA invitation. Invitation acceptance never created a
        role record in this deployment, so every signed-in user was refused at
        this rule with a 403 the UI could not explain.

        Requiring only ``authenticated`` is not a weakening: an anonymous caller
        still cannot pass, and what a signed-in caller may actually SEE is now
        decided by the governed access directory behind this rule — which fails
        closed on an identity it does not list.
        """
        routes = self._swa().get("routes") or []
        api = [r for r in routes if r.get("route") == "/api/*"]
        assert api, "no rule guarding the linked backend"
        assert set(api[0].get("allowedRoles") or []) == {"authenticated"}, (
            "the linked backend must still require a signed-in user; "
            "authorisation is the access directory's job, not SWA's")

    def test_no_invitation_only_route_remains(self):
        """The invitation flow is abandoned, so the rule that existed only to
        expose it should not linger and imply the mechanism is still in use."""
        routes = self._swa().get("routes") or []
        invitation = [r for r in routes
                      if "invitation" in str(r.get("route", "")).lower()]
        assert not invitation, (
            f"invitation-specific route(s) still declared: {invitation}")

    def test_the_api_path_is_not_swallowed_by_the_spa_fallback(self):
        excluded = ((self._swa().get("navigationFallback") or {})
                    .get("exclude") or [])
        assert "/api/*" in excluded, (
            "without this the SPA's index.html is returned for API calls, which "
            "surfaces as an unparseable response rather than an auth error")

    def test_unused_identity_providers_are_closed_off(self):
        routes = self._swa().get("routes") or []
        blocked = {r.get("route"): r.get("statusCode") for r in routes
                   if str(r.get("route", "")).startswith("/.auth/login/")}
        for provider in ("/.auth/login/github", "/.auth/login/twitter"):
            assert blocked.get(provider) == 404, f"{provider} is still reachable"

    def test_an_unauthenticated_visitor_is_sent_to_sign_in(self):
        override = (self._swa().get("responseOverrides") or {}).get("401") or {}
        assert override.get("statusCode") == 302
        assert "/.auth/login/aad" in (override.get("redirect") or "")


# =========================================================================== #
# 4. No MSAL / no client-side token acquisition
# =========================================================================== #
class TestNoClientSideTokenAcquisition:
    """Option A deliberately does NOT add MSAL: the platform performs the
    sign-in and forwards the principal. A token library appearing in the SPA
    would mean a second, divergent authentication path."""

    FRONTEND = _REPO_ROOT / "frontend" / "mi-agent-ui"

    def test_no_msal_or_identity_dependency(self):
        pkg = json.loads((self.FRONTEND / "package.json").read_text())
        deps = {**(pkg.get("dependencies") or {}), **(pkg.get("devDependencies") or {})}
        for name in deps:
            assert "msal" not in name.lower(), f"{name} adds a second auth path"
            assert not name.startswith("@azure/identity")

    def test_the_client_sends_no_hand_rolled_credential(self):
        src = self.FRONTEND / "src"
        offenders = []
        for path in src.rglob("*.ts*"):
            if path.name.endswith(".test.ts") or path.name.endswith(".test.tsx"):
                continue
            text = path.read_text(encoding="utf-8")
            for needle in ("acquireTokenSilent", "acquireTokenPopup",
                           "Authorization: `Bearer", 'Authorization": "Bearer'):
                if needle in text:
                    offenders.append(f"{path.relative_to(_REPO_ROOT)}: {needle}")
        assert not offenders, (
            "the SPA must not attach its own credential — the Static Web App "
            f"forwards the verified principal instead: {offenders}")


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
