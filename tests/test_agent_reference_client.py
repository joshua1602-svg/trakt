#!/usr/bin/env python3
"""Sprint 1 exit criterion: an external agent completes the loop.

    external process ──HTTP──► Trakt Agent API ──► governed tool registry
                                              ──► existing covenant engine
                                              ──► GovernedResult + audit

This is the only test in the Sprint 1 set that uses a **real socket**. The point
is the boundary: the client is a separate program that has no Trakt imports and
no access to a dataframe, a file or a configuration object. Everything it knows,
it learned from an HTTP response.

``test_the_reference_client_imports_nothing_from_trakt`` is the load-bearing
assertion. Without it, the demonstration would prove nothing about an external
agent — an in-process caller can always be given more than it should have.
"""

from __future__ import annotations

import ast
import importlib.util
import json
import logging
import socket
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict, List

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from trakt_core.errors import ErrorCode

from tests.test_agent_governed_execution import (
    EVALUATED_ENVELOPE,
    PORTFOLIO_A,
    PORTFOLIO_B,
    _deps as _tool_deps,
)
from tests.test_agent_identity_and_api import (
    AGENT_DIR,
    AGENT_ORG,
    TENANT,
    _registry,
    _store,
)

_CLIENT_PATH = _REPO_ROOT / "scripts" / "agent_reference_client.py"


def _load_client_module():
    spec = importlib.util.spec_from_file_location("agent_reference_client",
                                                  _CLIENT_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# --------------------------------------------------------------------------- #
# The boundary claim
# --------------------------------------------------------------------------- #
def test_the_reference_client_imports_nothing_from_trakt():
    """The claim the whole proof rests on.

    Parsed from the AST rather than grepped, so a lazy import inside a function
    is caught too. An external agent that could `import trakt_tools` would be
    able to skip every control this sprint exists to demonstrate.
    """
    tree = ast.parse(_CLIENT_PATH.read_text(encoding="utf-8"))
    imported: List[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.append(node.module)

    forbidden_roots = {"trakt_core", "trakt_tools", "mi_agent", "mi_agent_api",
                       "engine", "analytics", "analytics_lib",
                       "operations_control", "snapshot", "config", "pandas"}
    offenders = [name for name in imported
                 if name.split(".")[0] in forbidden_roots]
    assert offenders == [], (
        f"the reference client imports {offenders} — it must reach Trakt over "
        "HTTP only")


def test_the_reference_client_reaches_trakt_only_over_http():
    """Every Trakt interaction goes through one small HTTP class, and the client
    never reads portfolio data from anywhere else."""
    source = _CLIENT_PATH.read_text(encoding="utf-8")
    assert "urllib.request" in source
    assert "/v1/agent/tools" in source
    for forbidden in ("read_csv", "sqlite3", "blob://"):
        assert forbidden not in source, f"reference client uses {forbidden}"

    # No filesystem reads. Checked on the AST so `urlopen` is not mistaken for
    # `open`, and so a call built by name is still caught.
    tree = ast.parse(source)
    file_opens = [
        node for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
        and node.func.id in {"open", "exec", "eval", "__import__"}
    ]
    assert file_opens == [], "the reference client opens files or evaluates code"


# --------------------------------------------------------------------------- #
# A real server on a real port
# --------------------------------------------------------------------------- #
def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


@pytest.fixture()
def live_server(monkeypatch):
    """The agent router, served over HTTP, wired to the fixture governance."""
    import uvicorn
    from fastapi import FastAPI

    from mi_agent_api import agent_api, agent_auth, identity as identity_mod

    monkeypatch.setenv("TRAKT_AGENT_AUTH_MODE", "disabled")
    monkeypatch.setenv("TRAKT_AGENT_DEV_DIRECTORY", AGENT_DIR)
    monkeypatch.setattr(agent_auth, "is_production", lambda: False)
    monkeypatch.setattr(agent_api, "default_tenant_id", lambda: TENANT)

    real_context = identity_mod.context_from_agent_principal

    def _patched_context(principal, **kwargs):
        kwargs.setdefault("organisation_registry", _registry())
        kwargs.setdefault("entitlement_store", _store())
        return real_context(principal, **kwargs)

    monkeypatch.setattr(agent_api.identity_mod, "context_from_agent_principal",
                        _patched_context)

    real_execute = agent_api.trakt_tools.execute_governed_tool

    def _execute(tool_name, arguments, context, **kwargs):
        return real_execute(tool_name, arguments, context,
                            dependencies=_tool_deps())

    monkeypatch.setattr(agent_api.trakt_tools, "execute_governed_tool", _execute)

    app = FastAPI()
    app.include_router(agent_api.router)

    port = _free_port()
    config = uvicorn.Config(app, host="127.0.0.1", port=port, log_level="error")
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()

    deadline = time.monotonic() + 20
    while not server.started and time.monotonic() < deadline:
        time.sleep(0.05)
    if not server.started:  # pragma: no cover - CI resource starvation
        server.should_exit = True
        pytest.skip("the test server did not start in time")

    yield f"http://127.0.0.1:{port}"

    server.should_exit = True
    thread.join(timeout=10)


def test_an_external_agent_completes_the_whole_loop(live_server, caplog):
    """The Sprint 1 exit criterion, end to end.

    A separate program authenticates, discovers the tool, invokes it, receives a
    governed response carrying provenance and configuration metadata, and Trakt
    records an audit event for the call.
    """
    client = _load_client_module()
    api = client.TraktAgentAPI(live_server)

    # --- discover -------------------------------------------------------- #
    catalogue = api.list_tools()
    names = [t["name"] for t in catalogue["tools"]]
    assert "evaluate_covenants" in names
    assert catalogue["organisation_id"] == AGENT_ORG
    assert catalogue["resources"]["risk:read"] == [PORTFOLIO_A.key]

    tool = next(t for t in catalogue["tools"] if t["name"] == "evaluate_covenants")
    assert tool["input_schema"]["required"] == ["resource"]
    assert tool["version"]

    # --- invoke, with Trakt's audit logger captured ---------------------- #
    with caplog.at_level(logging.INFO, logger="trakt.audit"):
        result = api.invoke("evaluate_covenants", {"resource": PORTFOLIO_A.key})

    assert result["status"] == "success"
    assert result["capability"] == "tool.evaluate_covenants"
    assert result["tenant_id"] == TENANT

    # --- provenance and configuration metadata are present --------------- #
    assert result["snapshot"]["snapshot_id"] == "snap_test_0001"
    assert result["snapshot"]["content_hash"] == "sha256:abc123"
    assert result["policy"]["data_approved"] is True

    lineage = result["result"]["lineage"]
    assert lineage["configuration_version"] == 7
    assert lineage["activated_by"] == "A. Operator"
    assert lineage["library_version"] == "1.0.0"

    # --- the numbers are Trakt's, unchanged ------------------------------ #
    breach = next(t for t in result["result"]["tests"] if t["test_id"] == "GEO_SE")
    source = EVALUATED_ENVELOPE["tests"][0]
    assert breach["current_value"] == source["currentValue"]
    assert breach["threshold"] == source["threshold"]
    assert breach["headroom"] == source["headroom"]
    assert breach["status"] == "breach"

    # --- an audit event was generated ------------------------------------ #
    events = [r.getMessage() for r in caplog.records if r.name == "trakt.audit"]
    assert len(events) == 1
    event = json.loads(events[0].split("trakt.audit ", 1)[1])
    assert event["capability"] == "tool.evaluate_covenants"
    assert event["organisation_id"] == AGENT_ORG
    assert event["actor_type"] == "service"
    assert event["channel"] == "enterprise_agent"
    assert event["outcome"] == "success"
    assert event["request_id"] == result["request_id"]


def test_the_external_agent_is_refused_the_portfolio_it_was_not_granted(live_server):
    """The permission boundary holds across the wire, not just in process."""
    client = _load_client_module()
    api = client.TraktAgentAPI(live_server)

    refused = api.invoke("evaluate_covenants", {"resource": PORTFOLIO_B.key})
    assert refused["status"] == "blocked"
    assert refused["error"]["code"] == ErrorCode.RESOURCE_NOT_AUTHORISED
    assert refused["result"] is None

    # And the client recorded it as a normal outcome, not a transport failure.
    assert api.calls[-1]["http_status"] == 403
    assert api.calls[-1]["error_code"] == ErrorCode.RESOURCE_NOT_AUTHORISED


def test_the_scripted_provider_completes_the_workflow_with_no_model(live_server):
    """The claim that makes the demonstration meaningful.

    If the same workflow completes with nothing reasoning in the middle, then the
    permissions, the calculation, the evidence and the audit trail are properties
    of Trakt — not of whichever model happened to be driving.
    """
    client = _load_client_module()
    api = client.TraktAgentAPI(live_server)
    provider = client.ScriptedProvider(PORTFOLIO_A.key,
                                       tool="evaluate_covenants")

    record = client.run(api, provider, PORTFOLIO_A.key)

    assert len(record["calls"]) == 1
    assert record["calls"][0]["status"] == "success"
    assert "breach" in record["narrative"]
    # Every figure in the narrative came from a tool result.
    assert str(EVALUATED_ENVELOPE["tests"][0]["currentValue"]) in record["narrative"]
    assert record["correlation_id"].startswith("agent-run-")


def test_the_correlation_id_joins_the_agents_calls_to_trakts_audit_trail(live_server, caplog):
    """What later makes 'show me every tool call behind this decision' a query."""
    client = _load_client_module()
    api = client.TraktAgentAPI(live_server, correlation_id="buyer-run-99")

    with caplog.at_level(logging.INFO, logger="trakt.audit"):
        api.invoke("evaluate_covenants", {"resource": PORTFOLIO_A.key})
        api.invoke("evaluate_covenants", {"resource": PORTFOLIO_B.key})

    events = [json.loads(r.getMessage().split("trakt.audit ", 1)[1])
              for r in caplog.records if r.name == "trakt.audit"]
    assert len(events) == 2
    assert {e["correlation_id"] for e in events} == {"buyer-run-99"}
    assert {e["outcome"] for e in events} == {"success", "blocked"}


# --------------------------------------------------------------------------- #
# Provider translation stays at the client boundary
# --------------------------------------------------------------------------- #
def test_each_provider_translates_the_same_published_schema():
    """Trakt publishes JSON Schema and nothing provider-specific. Each adapter
    translates it here, at the client boundary — which is what keeps the platform
    model-agnostic."""
    client = _load_client_module()
    import trakt_tools

    published = trakt_tools.get("evaluate_covenants").to_catalogue_entry()

    anthropic_tool = client.AnthropicProvider._translate(published)
    assert anthropic_tool["name"] == "evaluate_covenants"
    assert anthropic_tool["input_schema"] is published["input_schema"]

    openai_tool = client.OpenAIProvider._translate(published)
    assert openai_tool["type"] == "function"
    assert openai_tool["function"]["name"] == "evaluate_covenants"
    assert openai_tool["function"]["parameters"] is published["input_schema"]

    # Neither translation reached back into Trakt to reshape anything.
    assert published["input_schema"]["type"] == "object"


def test_the_system_instruction_forbids_the_model_computing_anything():
    client = _load_client_module()
    instruction = client.SYSTEM_INSTRUCTION.lower()
    assert "never state a figure trakt did not return" in instruction
    assert "do not compute" in instruction
    assert "call a tool" in instruction
