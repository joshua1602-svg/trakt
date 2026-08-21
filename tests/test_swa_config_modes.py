#!/usr/bin/env python3
"""The two hosting-layer configurations, and the rule that picks between them.

Static Web Apps enforces its routes BEFORE any JavaScript runs. That is fine
when Easy Auth IS the authentication — and fatal when it is not: the default
config answers an anonymous ``GET /`` with 401, the response override turns that
into a 302 to ``/.auth/login/aad``, and index.html is never served. MSAL cannot
start a sign-in from a page the browser never received, which is why a
bearer-mode build was still landing on

    login.microsoftonline.com/organizations/oauth2/v2.0/authorize
      ?redirect_uri=…/.auth/login/aad/callback&scope=openid profile email

— Easy Auth's request, not the SPA's, and with no ``api://…/MI.Access`` in it.

So there are two configs and the deploy picks one. What these tests hold in
place:

  * the DEFAULT file keeps today's Easy Auth behaviour exactly, because it is
    what a rollback (unset the flag) returns to. "Make the shell anonymous for
    every build" would also fix the redirect and would silently drop the gate
    from the deployment we have not migrated yet;
  * the BEARER file serves the shell anonymously and intercepts nothing;
  * the workflow ships the one that matches the flag.

Anonymous SHELL is not anonymous DATA. In bearer mode the only thing served
without a credential is the static bundle; every MI route is answered by
trakt-mi-api, which fails closed (mi_agent_api/react_auth.py) and never sees a
Static Web Apps role either way.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
_UI = _REPO_ROOT / "frontend" / "mi-agent-ui"
DEFAULT_CONFIG = _UI / "staticwebapp.config.json"
BEARER_CONFIG = _UI / "staticwebapp.bearer.json"
WORKFLOW = (_REPO_ROOT / ".github" / "workflows"
            / "azure-static-web-apps-nice-smoke-067ac7603.yml")


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _routes(cfg: dict) -> dict:
    return {r.get("route"): r for r in cfg.get("routes", [])}


def _roles(cfg: dict, route: str) -> set:
    return set((_routes(cfg).get(route) or {}).get("allowedRoles") or [])


def _login_redirect(cfg: dict) -> str:
    return ((cfg.get("responseOverrides") or {}).get("401") or {}).get("redirect") or ""


# =========================================================================== #
# The default config — what a rollback returns to
# =========================================================================== #
class TestEasyAuthModeIsPreserved:
    def test_the_shell_still_requires_a_signed_in_user(self):
        assert _roles(_load(DEFAULT_CONFIG), "/*") == {"authenticated"}

    def test_an_anonymous_visitor_is_still_sent_to_easy_auth(self):
        assert "/.auth/login/aad" in _login_redirect(_load(DEFAULT_CONFIG))

    def test_the_linked_backend_still_requires_an_mi_role(self):
        assert _roles(_load(DEFAULT_CONFIG), "/api/*") == {"client", "operator"}

    def test_the_default_has_not_been_quietly_opened_up(self):
        # The one change that would "fix" the redirect for every deployment at
        # once, including the one still relying on Easy Auth.
        assert "anonymous" not in _roles(_load(DEFAULT_CONFIG), "/*")


# =========================================================================== #
# The bearer config — what the pilot ships
# =========================================================================== #
class TestBearerModeServesTheShell:
    def test_the_bearer_config_exists(self):
        assert BEARER_CONFIG.is_file(), (
            "the workflow copies this file over staticwebapp.config.json when "
            "VITE_MI_BEARER_AUTH=true")

    def test_visiting_the_root_anonymously_reaches_the_app(self):
        """The property the whole change exists for.

        Three things together decide it: the catch-all admits anonymous, so `/`
        is not 401'd; nothing rewrites 401 to Easy Auth, so it cannot be turned
        into a redirect; and the navigation fallback serves index.html. Any one
        of them missing puts the browser back on the Easy Auth authorize URL.
        """
        cfg = _load(BEARER_CONFIG)
        assert "anonymous" in _roles(cfg, "/*")
        assert "/.auth/login" not in _login_redirect(cfg)
        assert (cfg.get("navigationFallback") or {}).get("rewrite") == "/index.html"

    def test_nothing_redirects_a_401_into_easy_auth(self):
        cfg = _load(BEARER_CONFIG)
        overrides = cfg.get("responseOverrides") or {}
        assert "/.auth/login" not in json.dumps(overrides)

    def test_the_api_path_carries_no_static_web_apps_role_gate(self):
        # Authorization is the token's job now. A `client`/`operator` role here
        # would refuse a validly-signed-in external user before the API ever saw
        # their token — they hold no Static Web Apps invitation and never will.
        assert not _roles(_load(BEARER_CONFIG), "/api/*") & {"client", "operator"}

    def test_react_is_left_to_start_the_sign_in(self):
        # No route may hand the browser to Easy Auth on its own. `/login` in the
        # default config rewrites to /.auth/login/aad; under MSAL that would be
        # a second, competing sign-in path.
        cfg = _load(BEARER_CONFIG)
        for route in cfg.get("routes", []):
            target = f"{route.get('rewrite', '')} {route.get('redirect', '')}"
            assert "/.auth/login/aad" not in target, (
                f"route {route.get('route')} starts an Easy Auth sign-in")

    def test_other_identity_providers_stay_closed(self):
        cfg = _load(BEARER_CONFIG)
        for provider in ("/.auth/login/github", "/.auth/login/twitter",
                         "/.auth/login/facebook"):
            assert (_routes(cfg).get(provider) or {}).get("statusCode") == 404

    @pytest.mark.parametrize("header,value", [
        ("X-Content-Type-Options", "nosniff"),
        ("X-Frame-Options", "DENY"),
        ("Referrer-Policy", "no-referrer"),
    ])
    def test_security_headers_survive_the_switch(self, header, value):
        assert _load(BEARER_CONFIG).get("globalHeaders", {}).get(header) == value

    def test_the_navigation_fallback_is_identical_to_the_default(self):
        # A deep link must not 404, and an API path must not be answered with
        # index.html. Both modes ship the same rules.
        assert _load(BEARER_CONFIG)["navigationFallback"] == \
            _load(DEFAULT_CONFIG)["navigationFallback"]

    def test_the_easy_auth_provider_registration_is_kept(self):
        """Kept deliberately: it makes rollback a flag rather than a redeploy of
        different content, and during the dual-mode pilot the operator can still
        reach /.auth/login/aad. Nothing ROUTES there any more, which is the part
        that mattered."""
        reg = ((_load(BEARER_CONFIG).get("auth") or {})
               .get("identityProviders") or {}).get("azureActiveDirectory")
        assert reg, "the provider registration was dropped, so rollback is no longer a flag"


# =========================================================================== #
# The workflow picks one
# =========================================================================== #
class TestTheDeploymentSelectsAConfig:
    def test_the_bearer_config_is_shipped_when_the_flag_is_on(self):
        text = WORKFLOW.read_text(encoding="utf-8")
        assert 'if [ "${VITE_MI_BEARER_AUTH}" = "true" ]' in text
        assert "cp staticwebapp.bearer.json staticwebapp.config.json" in text

    def test_the_flag_reaches_the_build(self):
        text = WORKFLOW.read_text(encoding="utf-8")
        assert "VITE_MI_BEARER_AUTH:" in text
        assert "VITE_MI_API_SCOPE:" in text, (
            "the SPA cannot request api://…/MI.Access from a bundle compiled "
            "without the scope")

    def test_the_workflow_verifies_the_mode_it_shipped(self):
        # A build that says bearer and ships the gate fails the same way this
        # change exists to fix, and does it silently.
        text = WORKFLOW.read_text(encoding="utf-8")
        assert "catch-all is not anonymous" in text
        assert "a 401 still redirects to Easy Auth" in text
