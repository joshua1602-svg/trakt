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
import re
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


@pytest.fixture()
def auth_on(monkeypatch):
    monkeypatch.setenv("MI_AGENT_AUTH_ENABLED", "true")
    monkeypatch.setenv("MI_AGENT_CLIENT_ROLE", "client")
    monkeypatch.setenv("MI_AGENT_OPERATOR_ROLE", "operator")
    yield


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

    def test_an_authenticated_caller_without_an_mi_role_gets_nothing(self, auth_on):
        """Signed in is not the same as entitled."""
        hdr = {"x-ms-client-principal": _principal(["anonymous", "authenticated"])}
        for path, verb in _mi_routes():
            r = _call(path, verb, headers=hdr)
            assert r.status_code == 403, (
                f"{verb} {path} admitted a caller with no MI role")

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
    under /api, the gateway strips the prefix, the guard admits the role."""

    def test_a_client_principal_under_the_api_prefix_is_admitted(self, auth_on):
        r = client.get("/api/me", headers={
            "x-ms-client-principal": _principal(["authenticated", "client"])})
        assert r.status_code == 200
        body = r.json()
        assert body["authenticated"] is True
        assert "client" in body["roles"]
        assert body["isOperator"] is False

    def test_an_operator_principal_is_admitted_and_flagged(self, auth_on):
        r = client.get("/api/me", headers={
            "x-ms-client-principal": _principal(["authenticated", "operator"])})
        assert r.status_code == 200
        assert r.json()["isOperator"] is True

    def test_the_prefixed_and_bare_forms_resolve_to_the_same_route(self, auth_on):
        hdr = {"x-ms-client-principal": _principal(["authenticated", "operator"])}
        assert client.get("/me", headers=hdr).json() == \
            client.get("/api/me", headers=hdr).json()

    def test_an_mi_route_is_served_under_the_api_prefix(self, auth_on):
        """The whole point of the switch: /api/mi/... must not 404."""
        hdr = {"x-ms-client-principal": _principal(["authenticated", "operator"])}
        r = client.get("/api/mi/snapshots", headers=hdr)
        assert r.status_code == 200, (
            "the linked-backend path form must resolve — this exact mismatch "
            "produced the production 404s that gateway.py was written to fix")

    def test_an_unknown_prefixed_path_still_404s_with_its_own_path(self, auth_on):
        hdr = {"x-ms-client-principal": _principal(["authenticated", "operator"])}
        assert client.get("/api/mi/not-a-real-route", headers=hdr).status_code == 404

    def test_the_easy_auth_principal_shape_is_also_accepted(self, auth_on):
        """App Service Easy Auth sends a claims list, not userRoles — both are
        the platform's, and both must work."""
        payload = {"auth_typ": "aad", "name_typ": "name", "role_typ": "roles",
                   "claims": [{"typ": "roles", "val": "operator"},
                              {"typ": "name", "val": "ops@trakt.io"}]}
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

    def test_the_api_route_requires_an_mi_role(self):
        routes = self._swa().get("routes") or []
        api = [r for r in routes if r.get("route") == "/api/*"]
        assert api, "no rule guarding the linked backend"
        assert set(api[0].get("allowedRoles") or []) == {"client", "operator"}, (
            "the linked backend must be restricted to the MI roles, so an "
            "authenticated user without one cannot reach portfolio data")

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
class TestExactlyOneCredentialPath:
    """One place acquires a token, one place attaches it.

    This class used to assert the opposite — that the SPA contained no MSAL and
    attached no credential of its own — because under Option A the platform did
    the sign-in and forwarded the principal. Phase 3 of the multitenant
    migration deliberately reverses that: the SPA signs the user in against
    Entra `/organizations` and calls the API with an access token, because a
    Static Web Apps session cannot be issued to a user from a directory the site
    has no invitation into.

    What has NOT changed is the reason the original rule existed: two
    independent ways to authenticate is how one of them silently stops being
    enforced. So the rule is now about SHAPE rather than absence — token
    acquisition lives in src/auth, attachment lives in the API client, and no
    screen assembles an Authorization header of its own.
    """

    FRONTEND = _REPO_ROOT / "frontend" / "mi-agent-ui"

    @staticmethod
    def _code(text: str) -> str:
        """The source with comments removed.

        These rules are about what the code DOES, and this file's own subject
        matter means the auth modules discuss `Authorization: Bearer` and
        `acquireTokenSilent` in prose constantly. Scanning raw text makes the
        documentation the violation, which teaches the next person to write less
        of it.
        """
        no_block = re.sub(r"/\*.*?\*/", "", text, flags=re.DOTALL)
        return re.sub(r"^\s*//.*$", "", no_block, flags=re.MULTILINE)

    def _sources(self):
        for path in (self.FRONTEND / "src").rglob("*.ts*"):
            if ".test." in path.name:
                continue
            yield path, self._code(path.read_text(encoding="utf-8"))

    def test_token_acquisition_lives_only_in_the_auth_layer(self):
        offenders = [
            str(path.relative_to(_REPO_ROOT))
            for path, text in self._sources()
            if ("acquireToken" in text or "PublicClientApplication" in text)
            and path.parent.name != "auth"
        ]
        assert not offenders, (
            "MSAL token acquisition outside src/auth — a second acquisition site "
            f"is a second policy to keep in step: {offenders}")

    def test_only_the_shared_helper_builds_an_authorization_header(self):
        offenders = [
            f"{path.relative_to(_REPO_ROOT)}"
            for path, text in self._sources()
            if "Bearer " in text and path.name != "tokenProvider.ts"
        ]
        assert not offenders, (
            "a hand-rolled Authorization header — every request must go through "
            f"authorizationHeaders() so there is one place to audit: {offenders}")

    def test_no_confidential_client_library_reaches_the_browser(self):
        pkg = json.loads((self.FRONTEND / "package.json").read_text())
        deps = {**(pkg.get("dependencies") or {}), **(pkg.get("devDependencies") or {})}
        # msal-node performs the confidential-client flows (client secret,
        # certificate). In a SPA bundle those credentials would be public.
        assert "@azure/msal-node" not in deps
        assert "@azure/identity" not in deps

    def test_the_spa_holds_no_client_secret(self):
        for path, text in self._sources():
            assert "clientSecret" not in text and "client_secret" not in text, (
                f"{path.relative_to(_REPO_ROOT)} names a client secret; the SPA "
                "is a public client using Authorization Code + PKCE")


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
