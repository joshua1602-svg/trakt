"""The MI Query Agent's HTTP route surface, as a contract rather than an
assumption.

This suite exists because of a production 404. The React client calls
``/mi/query``; the API serves ``/mi/query``; both were correct — and every
question still returned 404, because the Static Web Apps **linked-backend**
topology (the one ``mi_agent_api/auth.py`` and ``docs/auth_setup_runbook.md``
describe) forwards the request as ``/api/mi/query``, and nothing in the
application served that path. The failure was invisible to every existing test
because no test asserted the relationship between *the paths the client calls*
and *the paths the app serves*.

These tests assert exactly that relationship, in both directions and under both
deployment topologies, so the same class of failure cannot recur silently.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from mi_agent_api import gateway
from mi_agent_api.app import app

client = TestClient(app)

_REPO_ROOT = Path(__file__).resolve().parents[2]
_HTTP_CLIENT_TS = (_REPO_ROOT / "frontend" / "mi-agent-ui" / "src" / "api"
                   / "HttpAgentClient.ts")

#: Literal API paths appearing in the React HTTP client, e.g. `/mi/snapshots` or
#: a template literal `` `/mi/cohorts?${...}` ``. The query string is dropped.
_CLIENT_PATH_RE = re.compile(r"""["'`](/(?:mi|me|health|v1)[A-Za-z0-9/_\-]*)""")


def _client_paths() -> set[str]:
    source = _HTTP_CLIENT_TS.read_text(encoding="utf-8")
    return {m.group(1).rstrip("/") or "/" for m in _CLIENT_PATH_RE.finditer(source)}


def _servable() -> set[str]:
    return {r.path for r in app.routes if getattr(r, "path", None)}


# --------------------------------------------------------------------------- #
# The contract: every path the UI calls is a path the API serves
# --------------------------------------------------------------------------- #
def test_the_react_client_only_calls_paths_the_api_serves():
    """The regression guard for the 404.

    Read the literal paths out of the shipped TypeScript client and require the
    FastAPI app to serve every one. A route renamed on either side now fails
    here instead of in production.
    """
    called = _client_paths()
    assert called, "no API paths found in HttpAgentClient.ts — has the file moved?"
    missing = sorted(p for p in called if p not in _servable())
    assert not missing, (
        f"The React client calls {missing}, which the MI Agent API does not "
        "serve. This is the exact shape of the production 404.")


def test_the_query_endpoint_is_reachable_under_both_deployment_topologies():
    """Bare (direct base URL) and prefixed (SWA linked backend) both answer."""
    payload = {"question": "what is the total current balance?"}
    bare = client.post("/mi/query", json=payload)
    prefixed = client.post(f"{gateway.api_prefix()}/mi/query", json=payload)
    assert bare.status_code == 200, bare.text
    assert prefixed.status_code == 200, prefixed.text
    # Not merely both-200: the SAME governed answer, so a gateway cannot change
    # what the agent says.
    assert prefixed.json()["ok"] == bare.json()["ok"]
    assert prefixed.json()["spec"]["metric"] == bare.json()["spec"]["metric"]


@pytest.mark.parametrize("path", ["/mi/snapshots", "/mi/portfolio-context",
                                  "/mi/catalogue", "/health", "/me"])
def test_every_ui_facing_get_is_reachable_under_the_gateway_prefix(path):
    prefixed = client.get(f"{gateway.api_prefix()}{path}")
    assert prefixed.status_code == client.get(path).status_code


def test_health_states_the_paths_this_deployment_answers_on():
    """A 404 must be diagnosable with one request, not by reading a build log."""
    routing = client.get("/health").json()["routing"]
    assert routing["apiPrefix"] == "/api"
    assert routing["queryPaths"] == ["/mi/query", "/api/mi/query"]


# --------------------------------------------------------------------------- #
# Normalisation must not WIDEN what the API serves
# --------------------------------------------------------------------------- #
def test_an_unknown_path_still_404s_with_its_own_path():
    """Prefix stripping must never mask a genuine routing mistake."""
    assert client.post("/api/mi/does-not-exist", json={"question": "x"}).status_code == 404
    assert client.post("/nope/mi/query", json={"question": "x"}).status_code == 404
    assert client.get("/api/api/health").status_code == 404


def test_strip_prefix_is_a_total_function_over_its_rule():
    servable = {"/mi/query", "/health"}
    assert gateway.strip_prefix("/api/mi/query", "/api", servable) == "/mi/query"
    assert gateway.strip_prefix("/api/mi/query/", "/api", servable) == "/mi/query"
    assert gateway.strip_prefix("/mi/query", "/api", servable) == "/mi/query"
    # Unknown remainder -> unchanged, so the caller sees the path they asked for.
    assert gateway.strip_prefix("/api/unknown", "/api", servable) == "/api/unknown"
    # A path that merely starts with the prefix STRING is not a prefixed path.
    assert gateway.strip_prefix("/apifoo/health", "/api", servable) == "/apifoo/health"
    # Disabled -> identity.
    assert gateway.strip_prefix("/api/mi/query", "", servable) == "/api/mi/query"


@pytest.mark.parametrize("raw,expected", [
    (None, "/api"), ("", ""), ("   ", ""), ("api", "/api"), ("/api/", "/api"),
    ("  /gateway  ", "/gateway"), ("/", ""),
])
def test_prefix_configuration_is_forgiving_about_slashes(monkeypatch, raw, expected):
    if raw is None:
        monkeypatch.delenv(gateway.API_PREFIX_ENV, raising=False)
    else:
        monkeypatch.setenv(gateway.API_PREFIX_ENV, raw)
    assert gateway.api_prefix() == expected


def test_the_deployed_front_end_origin_can_be_allowed_without_a_wildcard():
    """The cross-origin topology needs the SWA host on the allow-list; a
    wildcard is still never produced."""
    import os

    os.environ[gateway.ALLOWED_ORIGIN_ENV] = "https://nice-smoke-067ac7603.azurestaticapps.net/"
    try:
        assert gateway.extra_cors_origins() == [
            "https://nice-smoke-067ac7603.azurestaticapps.net"]
    finally:
        del os.environ[gateway.ALLOWED_ORIGIN_ENV]
    assert gateway.extra_cors_origins() == []


def test_the_spa_fallback_cannot_swallow_an_api_call():
    """Static Web Apps rewrites unmatched paths to index.html. If /api and /mi
    are not excluded, an API request is answered with the SPA — which for a POST
    surfaces as the 404 this suite exists to prevent."""
    import json

    config = json.loads(
        (_REPO_ROOT / "frontend" / "mi-agent-ui" / "staticwebapp.config.json")
        .read_text(encoding="utf-8"))
    exclude = config["navigationFallback"]["exclude"]
    for required in ("/api/*", "/mi/*", "/health", "/me"):
        assert required in exclude, (
            f"{required} must be excluded from the SPA navigation fallback")
