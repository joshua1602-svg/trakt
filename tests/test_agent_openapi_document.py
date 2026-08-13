#!/usr/bin/env python3
"""The published OpenAPI document stays in lock-step with the tool registry.

``deploy/copilot-agent/trakt-copilot-openapi.yaml`` is hand-authored and a test
keeps it honest. That works for three fixed actions; the agent surface grows a
route per tool, so the document is *generated* and this test is what makes the
guarantee automatic — a tool added without regenerating fails here rather than
shipping a contract that omits it.

It also holds the line that matters more: the document must describe the closed
tool surface and NOT the internal API.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest
import yaml

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import trakt_tools

DOCUMENT_PATH = _REPO_ROOT / "deploy" / "agent-api" / "trakt-agent-openapi.yaml"


@pytest.fixture(scope="module")
def document():
    assert DOCUMENT_PATH.exists(), (
        "run scripts/build_agent_openapi.py to generate the document")
    return yaml.safe_load(DOCUMENT_PATH.read_text(encoding="utf-8"))


def test_the_document_is_not_stale():
    """The one assertion that keeps every other one true."""
    result = subprocess.run(
        [sys.executable, "scripts/build_agent_openapi.py", "--check"],
        cwd=str(_REPO_ROOT), capture_output=True, text=True, timeout=180)
    assert result.returncode == 0, result.stderr or result.stdout


def test_every_registered_tool_has_a_route(document):
    expected = {f"/v1/agent/tools/{spec.name}" for spec in trakt_tools.all_tools()}
    documented = {p for p in document["paths"] if p != "/v1/agent/tools"}
    assert documented == expected


def test_each_route_publishes_the_registrys_own_schema(document):
    """The schema in the document IS the registry's, not a restatement."""
    for spec in trakt_tools.all_tools():
        route = document["paths"][f"/v1/agent/tools/{spec.name}"]["post"]
        body = route["requestBody"]["content"]["application/json"]["schema"]
        assert body == spec.input_schema
        assert spec.version in route["description"]
        assert spec.required_capability in route["description"]


def test_the_document_describes_only_the_tool_surface(document):
    """FastAPI's own /openapi.json would describe whichever routes the process
    happens to have mounted — the whole internal API. This document must not."""
    for path in document["paths"]:
        assert path.startswith("/v1/agent/tools"), path
    blob = DOCUMENT_PATH.read_text(encoding="utf-8")
    for internal in ("/mi/query", "/mi/snapshots", "/mi/decks", "/v1/copilot",
                     "/ops/"):
        assert internal not in blob, f"the agent document mentions {internal}"


def test_the_document_is_openapi_3_0_for_tool_loaders(document):
    assert document["openapi"].startswith("3.0"), (
        "agent tool loaders consume 3.0.x; FastAPI's generated 3.1 is why this "
        "document is built separately")


def test_bearer_authentication_is_required_globally(document):
    assert document["security"] == [{"entraBearer": []}]
    scheme = document["components"]["securitySchemes"]["entraBearer"]
    assert scheme["type"] == "http" and scheme["scheme"] == "bearer"


def test_the_server_url_is_still_a_placeholder(document):
    """A committed document must not carry a real deployment hostname; the
    operator sets it at packaging time."""
    url = document["servers"][0]["url"]
    assert "example" in url
    assert "description" in document["servers"][0]


def test_the_error_contract_tells_an_agent_what_not_to_retry(document):
    """`blocked` vs `error` and `retryable` are the two facts an autonomous
    caller needs; a document that omitted them would leave it guessing."""
    result = document["components"]["schemas"]["GovernedResult"]
    assert set(result["properties"]["status"]["enum"]) == {
        "success", "partial_success", "blocked", "error"}
    assert "do not retry" in result["properties"]["status"]["description"].lower()

    error = document["components"]["schemas"]["TraktError"]
    assert "retryable" in error["required"]
    assert "code" in error["required"]

    forbidden = document["components"]["responses"]["Forbidden"]["description"]
    assert "RESOURCE_NOT_AUTHORISED" in forbidden
    assert "do not retry" in forbidden.lower()
