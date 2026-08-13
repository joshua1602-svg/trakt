#!/usr/bin/env python3
"""MCP is a transport, and this suite exists to keep it one.

The risk with adopting a protocol like MCP is not technical difficulty — the
mapping is nearly one-to-one because both sides speak JSON Schema. The risk is
that the adapter gradually acquires opinions: a reshaped schema here, a
convenience default there, an identity read from an argument because MCP has no
identity model of its own. Each is small; together they are a second contract.

So the assertions here are mostly *negative*:

* the schemas are the registry's own objects, not copies (a copy can drift);
* nothing is added to the tool surface that the HTTP surface does not have;
* identity in tool arguments is refused, loudly;
* a governance refusal is marked as an error rather than handed to a model as
  data.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import trakt_tools
from trakt_core.context import SCOPE_LOAN_READ, SCOPE_RISK_READ
from trakt_tools import mcp
from trakt_tools.registry import all_tools, get

from tests.test_agent_loan_retrieval import PORTFOLIO_A, _context


# =========================================================================== #
# The declaration is the registry's, not a copy of it
# =========================================================================== #
def test_every_registered_tool_translates_to_an_mcp_declaration():
    """One source, three surfaces: HTTP, OpenAPI and MCP. A tool that exists on
    one and not another is precisely the drift the registry exists to stop."""
    for spec in all_tools():
        declaration = mcp.tool_declaration(spec)
        assert declaration["name"] == spec.name
        assert declaration["inputSchema"]["type"] == "object"
        assert declaration["outputSchema"]["type"] == "object"
        assert declaration["_meta"]["trakt/version"] == spec.version
        assert (declaration["_meta"]["trakt/required_capability"]
                == spec.required_capability)


def test_the_schemas_are_passed_through_verbatim():
    """Rewriting a schema for MCP would create a second contract that can drift
    from the OpenAPI one, and a tool whose shape depends on which door you came
    through is not one tool."""
    spec = get("get_loans")
    declaration = mcp.tool_declaration(spec)
    assert declaration["inputSchema"] is spec.input_schema
    assert declaration["outputSchema"] is spec.output_schema


def test_the_agent_guidance_survives_into_the_description():
    """MCP has no separate guidance field, and a client that cannot see the
    guidance will use the tool the way the description implies rather than the
    way it should — calling get_loan in a loop, for instance."""
    declaration = mcp.tool_declaration(get("get_loans"))
    assert "When to use:" in declaration["description"]
    assert "call this ONCE with a list" in declaration["description"]


def test_the_mcp_tool_list_is_narrowed_the_same_way_the_http_one_is():
    context = _context(PORTFOLIO_A, capabilities=(SCOPE_RISK_READ,))
    declared = {d["name"] for d in mcp.tool_declarations(context)}
    published = {t["name"] for t in trakt_tools.catalogue(context)["tools"]}
    assert declared == published
    assert "evaluate_covenants" in declared
    # A loan:read tool is absent, because this caller would only be refused.
    assert "get_loans" not in declared


def test_mcp_adds_no_tool_the_http_surface_does_not_have():
    """The adapter is a projection. If it could add a tool, it would be a second
    surface with its own capabilities rather than a transport."""
    context = _context(PORTFOLIO_A, capabilities=(SCOPE_RISK_READ, SCOPE_LOAN_READ))
    declared = {d["name"] for d in mcp.tool_declarations(context)}
    assert declared <= {spec.name for spec in all_tools()}


# =========================================================================== #
# Identity: the load-bearing refusal
# =========================================================================== #
@pytest.mark.parametrize("field", ["tenant_id", "tenantId", "organisation_id",
                                   "actor_id", "scopes", "entitlements",
                                   "channel"])
def test_identity_in_tool_arguments_is_refused(field):
    """The single most dangerous thing an MCP adapter could do. MCP has no
    identity model, so the temptation is real and the failure is total: one
    caller naming another tenant reads that tenant's book."""
    with pytest.raises(ValueError) as excinfo:
        mcp.refuse_identity_in_arguments({"resource": PORTFOLIO_A.key, field: "x"})
    assert field in str(excinfo.value)
    assert "authenticated principal" in str(excinfo.value)


def test_identity_is_refused_rather_than_stripped():
    """Silently dropping the claim would let the call proceed under a DIFFERENT
    identity from the one the caller believed it was using."""
    with pytest.raises(ValueError):
        mcp.refuse_identity_in_arguments({"tenant_id": "OTHER_TENANT"})


def test_ordinary_arguments_pass():
    mcp.refuse_identity_in_arguments({"resource": PORTFOLIO_A.key,
                                      "loan_ids": ["LN000001"],
                                      "shape": "structured"})
    mcp.refuse_identity_in_arguments({})


# =========================================================================== #
# The result: a refusal must not read as data
# =========================================================================== #
def test_a_successful_envelope_carries_the_whole_thing_plus_a_digest():
    envelope = {
        "capability": "tool.evaluate_covenants", "status": "success",
        "request_id": "req_1", "tenant_id": "ERE",
        "snapshot": {"snapshot_id": "snap_1", "dataset_label": "july tape",
                     "content_hash": "sha256:abc"},
        "result": {"tests": [{"test_id": "T1"}]},
        "warnings": ["these limits are indicative"],
    }
    out = mcp.tool_result(envelope)
    assert out["isError"] is False
    assert out["structuredContent"] == envelope
    text = out["content"][0]["text"]
    assert "success" in text
    assert "snap_1" in text and "sha256:abc" in text
    assert "these limits are indicative" in text
    assert "do not recompute one" in text


def test_a_governance_refusal_is_marked_as_an_error():
    """An MCP client that treated a refusal as a successful result would feed a
    blocked envelope to a model as though it were data."""
    envelope = {
        "capability": "tool.get_loans", "status": "blocked",
        "request_id": "req_2", "tenant_id": "ERE", "result": None,
        "error": {"code": "RESOURCE_NOT_AUTHORISED",
                  "message": "You are not entitled to that resource.",
                  "retryable": False},
    }
    out = mcp.tool_result(envelope)
    assert out["isError"] is True
    text = out["content"][0]["text"]
    assert "RESOURCE_NOT_AUTHORISED" in text
    assert "Do not retry" in text


def test_a_failure_is_also_an_error_but_is_not_told_to_stop_retrying():
    envelope = {"capability": "tool.get_loans", "status": "error",
                "request_id": "req_3", "tenant_id": "ERE", "result": None,
                "error": {"code": "STORAGE_UNAVAILABLE",
                          "message": "The store is unavailable.",
                          "retryable": True}}
    out = mcp.tool_result(envelope)
    assert out["isError"] is True
    assert "Do not retry" not in out["content"][0]["text"]


def test_the_digest_is_a_summary_and_the_envelope_is_never_truncated():
    """Handing a model a hundred lines of policy and audit metadata before any
    credit fact appears is a cost, not a safeguard. Nothing is dropped — it is
    ordered."""
    envelope = {"capability": "tool.stratify", "status": "success",
                "request_id": "r", "tenant_id": "ERE",
                "warnings": [f"warning {i}" for i in range(20)],
                "result": {"groups": [{"g": i} for i in range(500)]}}
    out = mcp.tool_result(envelope)
    assert out["structuredContent"]["result"]["groups"] == envelope["result"]["groups"]
    assert len(out["content"][0]["text"].splitlines()) < 15


# =========================================================================== #
# The boundary is documented, not assumed
# =========================================================================== #
def test_the_adapter_holds_no_business_logic():
    """Structural. An MCP module that could compute, authorise or read data
    would be a second implementation of the governed path."""
    import inspect

    source = inspect.getsource(mcp)
    for forbidden in ("pandas", "read_csv", "authorise_", "execute_governed_tool(",
                      "DataFrame", "require_scope"):
        assert forbidden not in source, (
            f"trakt_tools.mcp references {forbidden!r} — the adapter is a "
            "translation, not a second governed path")


def test_what_a_server_still_has_to_do_is_written_down():
    """So nobody mistakes this module for a server."""
    assert len(mcp.SERVER_RESPONSIBILITIES) >= 4
    joined = " ".join(mcp.SERVER_RESPONSIBILITIES)
    assert "never from tool arguments" in joined
    assert "execute_governed_tool" in joined
    assert "audit" in joined
