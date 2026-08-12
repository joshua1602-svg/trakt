#!/usr/bin/env python3
"""Can a general agent delegate a specialist objective across a standards A2A
boundary, and does the answer survive the crossing unchanged?

Sprint 4's claim has two halves and this file tests both, because either alone
would be misleading. The protocol half — discovery, task lifecycle, artifacts,
authorisation, errors — is cheap to exercise and is done here with a stub
specialist, so a protocol bug is not hidden behind a model's variance. The
*equivalence* half is the one that matters: the numbers a caller receives over
A2A must be the numbers Trakt computed, and the MCP boundary underneath must
change nothing. That is tested against real governed tools over the real
hidden-truth portfolio, with no model involved and no money spent.

What a passing run here does NOT prove: that the specialist reaches good
conclusions. That is Sprint 3's experiment and it needs a model. This file
proves the pipe is honest.
"""

from __future__ import annotations

import ast
import json
import sys
from pathlib import Path
from typing import Any, Dict

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from trakt_core.context import SCOPE_LOAN_READ, SCOPE_RISK_READ
from trakt_tools.execution import ToolDependencies
from trakt_tools.mcp_server import McpSession, McpToolTransport

from enterprise_agent.client import EnterpriseAgent
from readiness_agent.session import GovernedSession
from tests.planted_portfolio import planted_lineage_index
from tests.readiness_agent_portfolio import PERIODS, snapshots
from tests.test_agent_governed_execution import _Datasets, _Descriptor
from tests.test_agent_loan_retrieval import (PORTFOLIO_A, PORTFOLIO_B, TENANT,
                                             _catalogue, _context,
                                             _CountingResolver)
from trakt_a2a.card import SKILL_ID, A2A_PROTOCOL_VERSION, agent_card
from trakt_a2a.server import (ERR_FORBIDDEN, ERR_TASK_NOT_FOUND,
                              ERR_UNAUTHENTICATED, A2AError, CallerIdentity,
                              SecuritisationReadinessA2AServer)

ENDPOINT = "https://trakt.example/a2a"
BOTH = (SCOPE_LOAN_READ, SCOPE_RISK_READ)


# --------------------------------------------------------------------------- #
# harness
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def snaps():
    return snapshots()


def _deps(snaps):
    return ToolDependencies(
        datasets=_Datasets(_Descriptor(snapshot_id="secr-demo-final")),
        runtime_mode="test", catalogue=_catalogue(), output_root="/unused",
        loan_frame_resolver=_CountingResolver(snaps[PERIODS[-1]]),
        loan_history_resolver=lambda: snaps,
        lineage_index=planted_lineage_index(TENANT, "secr-demo-final"))


def _caller(org: str = "ere") -> CallerIdentity:
    return CallerIdentity(agent_id="a2a_test_agent", organisation_id=org,
                          context=_context(capabilities=BOTH))


#: A specialist whose "assessment" is fixed. Using a model here would make a
#: protocol test depend on model variance, and a flaky protocol test teaches
#: people to re-run rather than to read.
STUB_ASSESSMENT = {
    "overall_readiness": "MATERIAL_REMEDIATION_REQUIRED",
    "summary": "Concentration and collateral concerns.",
    "strengths": ["Weighted-average LTV is comfortable at the headline level."],
    "material_findings": [{
        "title": "Geographic concentration",
        "fact": "London is 31.0% of balance, measured by concentration.",
        "rule": "Securitisation criteria cap a single region at 27% — BREACH. "
                "The warehouse agreement permits 35% — PASS.",
        "judgement": "Structural rather than curable by loan removal.",
        "severity": "high",
        "evidence_tools": ["concentration", "stratify"],
    }],
    "could_not_assess": [{"metric": "Borrower concentration",
                          "status": "UNAVAILABLE",
                          "implication": "No borrower identifier on the tape."}],
    "further_diligence": ["Obtain refreshed valuations for the stale cohort."],
}


class _StubRun:
    assessment = STUB_ASSESSMENT
    stopped_reason = "submitted"
    efficiency = {"total_calls": 30, "distinct_calls": 30,
                  "by_tool": {"concentration": 1, "stratify": 6}}


def _server(snaps, *, assessor=None, authoriser=None, streaming=False):
    def session_factory(caller: CallerIdentity, resource: str):
        mcp = McpSession(context=caller.context, dependencies=_deps(snaps))
        return GovernedSession(caller.context, resource,
                               dependencies=_deps(snaps),
                               transport=McpToolTransport(mcp))

    def default_authoriser(caller: CallerIdentity, resource: str) -> None:
        if caller.organisation_id != "ere":
            raise A2AError(ERR_FORBIDDEN,
                           "Not entitled: a valid agent token identifies the "
                           "caller; it does not grant portfolio access.")
        if resource != PORTFOLIO_A.key:
            raise A2AError(ERR_FORBIDDEN, f"Not entitled to {resource!r}.")

    return SecuritisationReadinessA2AServer(
        endpoint_url=ENDPOINT, session_factory=session_factory,
        assessor=assessor or (lambda session: _StubRun()),
        authoriser=authoriser or default_authoriser, streaming=streaming)


def _client(server, caller=None) -> EnterpriseAgent:
    identity = caller if caller is not None else _caller()
    return EnterpriseAgent(
        organisation=identity.organisation_id,
        transport=lambda request: server.handle(request, identity),
        card_fetcher=server.agent_card)


# --------------------------------------------------------------------------- #
# 1. the Agent Card is a business contract, not a metric catalogue
# --------------------------------------------------------------------------- #
def test_the_card_carries_every_field_the_a2a_specification_requires():
    card = agent_card(endpoint_url=ENDPOINT)
    required = {"name", "description", "version", "capabilities",
                "supportedInterfaces", "defaultInputModes",
                "defaultOutputModes", "skills"}
    assert required <= set(card), sorted(required - set(card))
    interface = card["supportedInterfaces"][0]
    assert interface["protocolBinding"] == "JSONRPC"
    assert interface["protocolVersion"] == A2A_PROTOCOL_VERSION


def test_the_card_never_leaks_a_trakt_metric_or_tool_name():
    """The caller must learn WHAT can be delegated, not HOW Trakt does it.

    A card naming ``composition_vintage_share`` would couple every caller to a
    Trakt internal that is free to change, and the coupling would only surface
    when we renamed it.
    """
    from trakt_tools.registry import all_tools

    published = json.dumps(agent_card(endpoint_url=ENDPOINT)).lower()
    for spec in all_tools():
        assert spec.name.lower() not in published, (
            f"the Agent Card leaks the governed tool name {spec.name!r}")
    for methodology in ("observed_cpr", "observed_smm", "contractual_wal@",
                        "current_ltv@", "observed_default_rate",
                        "composition_vintage_share", "ltv_weighted_average"):
        assert methodology not in published, (
            f"the Agent Card leaks the internal identifier {methodology!r}")


def test_the_card_states_what_the_specialist_will_not_do():
    """Advertised limits let a caller predict a refusal instead of discovering
    it after delegating."""
    from trakt_a2a.card import SKILL_LIMITS

    joined = " ".join(SKILL_LIMITS).lower()
    assert "rating" in joined
    assert "scenario" in joined or "stress" in joined


# --------------------------------------------------------------------------- #
# 2. discovery through the protocol, not through a Python import
# --------------------------------------------------------------------------- #
def test_the_enterprise_agent_discovers_the_specialist_from_the_card(snaps):
    client = _client(_server(snaps))
    skill = client.find_skill_for(
        "I need a securitisation readiness assessment for a loan portfolio")
    assert skill is not None
    assert skill["id"] == SKILL_ID


def test_discovery_also_yields_the_endpoint_and_the_auth_requirement(snaps):
    client = _client(_server(snaps))
    assert client.endpoint() == ENDPOINT
    assert client.authentication_required() == ["enterprise_agent_oauth"]


def test_a_need_the_specialist_does_not_serve_matches_nothing_useful(snaps):
    """Skill matching must not enthusiastically claim unrelated work."""
    client = _client(_server(snaps))
    skill = client.find_skill_for("book a meeting room for Thursday")
    assert skill is None, (
        "an unrelated need matched the securitisation skill; a caller that "
        "delegates everything to the first agent it finds is not discovering")


# --------------------------------------------------------------------------- #
# 3. delegation of an outcome
# --------------------------------------------------------------------------- #
def test_an_objective_is_delegated_and_completes_as_a_task(snaps):
    client = _client(_server(snaps))
    outcome = client.delegate(
        "Assess this portfolio for securitisation readiness. Identify material "
        "strengths, weaknesses, risks and areas requiring further diligence.",
        portfolio=PORTFOLIO_A.key)
    assert outcome.state == "completed", outcome.error
    assert outcome.task_id and outcome.context_id
    assert outcome.headline == "MATERIAL_REMEDIATION_REQUIRED"


def test_the_task_records_its_lifecycle_transitions(snaps):
    server = _server(snaps)
    client = _client(server)
    outcome = client.delegate("Assess for securitisation readiness.",
                              portfolio=PORTFOLIO_A.key)
    task = server.store.get(outcome.task_id)
    assert [entry["state"] for entry in task.history] == [
        "submitted", "working", "completed"]


def test_tasks_get_returns_the_same_task(snaps):
    server = _server(snaps)
    client = _client(server)
    delegated = client.delegate("Assess for securitisation readiness.",
                                portfolio=PORTFOLIO_A.key)
    polled = client.poll(delegated.task_id)
    assert polled.task_id == delegated.task_id
    assert polled.state == "completed"


# --------------------------------------------------------------------------- #
# 4. the artifact keeps the evidence model
# --------------------------------------------------------------------------- #
def test_the_result_is_structured_not_prose(snaps):
    outcome = _client(_server(snaps)).delegate(
        "Assess for securitisation readiness.", portfolio=PORTFOLIO_A.key)
    result = outcome.result
    for key in ("overallReadiness", "materialFindings", "couldNotAssess",
                "furtherDiligence", "strengths"):
        assert key in result, f"the A2A artifact dropped {key!r}"


def test_fact_rule_and_judgement_survive_the_boundary_separately(snaps):
    """The distinction between what was measured, whose threshold applies and
    what the specialist thinks is the product. A wrapper that merged them into
    a sentence would be lossy in the one way that matters."""
    outcome = _client(_server(snaps)).delegate(
        "Assess for securitisation readiness.", portfolio=PORTFOLIO_A.key)
    finding = outcome.result["materialFindings"][0]
    assert finding["fact"] and finding["rule"] and finding["judgement"]
    assert finding["fact"] != finding["judgement"]
    # The rule must still name its authority, not merely a number.
    assert "warehouse" in finding["rule"].lower()
    assert "securitisation" in finding["rule"].lower()
    assert finding["severity"] == "high"
    assert finding["evidenceTools"]


def test_a_capability_state_survives_the_boundary(snaps):
    outcome = _client(_server(snaps)).delegate(
        "Assess for securitisation readiness.", portfolio=PORTFOLIO_A.key)
    gap = outcome.result["couldNotAssess"][0]
    assert gap["status"] == "UNAVAILABLE"
    assert gap["implication"]


def test_the_artifact_carries_an_evidence_trail(snaps):
    outcome = _client(_server(snaps)).delegate(
        "Assess for securitisation readiness.", portfolio=PORTFOLIO_A.key)
    trail = outcome.result["evidenceTrail"]
    assert trail["governedCalls"] == 30
    assert "concentration" in trail["toolsUsed"]


# --------------------------------------------------------------------------- #
# 5. authorisation — authenticated is not authorised
# --------------------------------------------------------------------------- #
def test_an_unauthenticated_caller_is_refused(snaps):
    server = _server(snaps)
    response = server.handle({"jsonrpc": "2.0", "id": "1",
                              "method": "message/send", "params": {}}, None)
    assert response["error"]["code"] == ERR_UNAUTHENTICATED


def test_a_caller_entitled_to_one_portfolio_is_refused_another(snaps):
    server = _server(snaps)
    with pytest.raises(A2AError) as raised:
        server.message_send(
            {"message": {"parts": [{"kind": "text",
                                    "text": "Assess for readiness."}],
                         "metadata": {"resource": PORTFOLIO_B.key}}},
            _caller())
    assert raised.value.code == ERR_FORBIDDEN
    assert "not entitled" in raised.value.message.lower()


def test_a_caller_from_another_organisation_is_refused(snaps):
    server = _server(snaps)
    with pytest.raises(A2AError) as raised:
        server.message_send(
            {"message": {"parts": [{"kind": "text",
                                    "text": "Assess for readiness."}],
                         "metadata": {"resource": PORTFOLIO_A.key}}},
            _caller(org="other-org"))
    assert raised.value.code == ERR_FORBIDDEN


def test_a_refused_delegation_never_reaches_the_specialist(snaps):
    """Authorisation must gate BEFORE the expensive part. A boundary that
    refuses only after running the assessment has spent the money it was
    protecting."""
    ran = []
    server = _server(snaps, assessor=lambda session: ran.append(1) or _StubRun())
    with pytest.raises(A2AError):
        server.message_send(
            {"message": {"parts": [{"kind": "text", "text": "Assess."}],
                         "metadata": {"resource": PORTFOLIO_B.key}}},
            _caller())
    assert ran == [], "the specialist ran despite an authorisation failure"


def test_one_organisation_cannot_read_anothers_task(snaps):
    server = _server(snaps)
    outcome = _client(server).delegate("Assess for securitisation readiness.",
                                       portfolio=PORTFOLIO_A.key)
    with pytest.raises(A2AError) as raised:
        server.tasks_get({"id": outcome.task_id}, _caller(org="other-org"))
    assert raised.value.code == ERR_FORBIDDEN


def test_an_unknown_task_is_not_found(snaps):
    with pytest.raises(A2AError) as raised:
        _server(snaps).tasks_get({"id": "task_missing"}, _caller())
    assert raised.value.code == ERR_TASK_NOT_FOUND


# --------------------------------------------------------------------------- #
# 6. scope — the specialist declines what it does not advertise
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("objective", [
    "Perform a formal external credit rating of this portfolio.",
    "Give me a probability of default for each loan.",
    "Run a stress test under a severe downturn scenario.",
    "What is the expected WAL of this portfolio?",
])
def test_an_unsupported_objective_is_rejected_not_approximated(snaps, objective):
    server = _server(snaps)
    result = server.message_send(
        {"message": {"parts": [{"kind": "text", "text": objective}],
                     "metadata": {"resource": PORTFOLIO_A.key}}}, _caller())
    assert result["status"]["state"] == "rejected", result
    assert result["artifacts"] == []
    assert "outside the advertised skill" in \
        result["status"]["message"]["reason"].lower()


def test_a_rejected_objective_does_not_run_the_specialist(snaps):
    ran = []
    server = _server(snaps, assessor=lambda s: ran.append(1) or _StubRun())
    server.message_send(
        {"message": {"parts": [{"kind": "text",
                                "text": "Perform a formal credit rating."}],
                     "metadata": {"resource": PORTFOLIO_A.key}}}, _caller())
    assert ran == []


# --------------------------------------------------------------------------- #
# 7. follow-up on the same context
# --------------------------------------------------------------------------- #
def test_a_follow_up_answers_from_evidence_without_re_running(snaps):
    runs = []
    server = _server(snaps, assessor=lambda s: runs.append(1) or _StubRun())
    client = _client(server)
    first = client.delegate("Assess for securitisation readiness.",
                            portfolio=PORTFOLIO_A.key)
    assert len(runs) == 1

    follow = client.ask_follow_up(
        "What is the main reason this portfolio is not ready?",
        context_id=first.context_id)

    assert follow.state == "completed"
    assert len(runs) == 1, "the follow-up re-ran the whole portfolio"
    assert follow.result["portfolioReRun"] is False
    assert follow.result["mostMaterialFinding"]["title"] == \
        "Geographic concentration"


def test_a_follow_up_keeps_the_evidence_model(snaps):
    server = _server(snaps)
    client = _client(server)
    first = client.delegate("Assess for securitisation readiness.",
                            portfolio=PORTFOLIO_A.key)
    follow = client.ask_follow_up(
        "Give me the evidence supporting the most material finding.",
        context_id=first.context_id)
    finding = follow.result["mostMaterialFinding"]
    assert finding["fact"] and finding["rule"] and finding["evidenceTools"]


def test_an_unmatched_follow_up_says_so_rather_than_inventing_an_answer(snaps):
    server = _server(snaps)
    client = _client(server)
    first = client.delegate("Assess for securitisation readiness.",
                            portfolio=PORTFOLIO_A.key)
    follow = client.ask_follow_up("What is the weather in Edinburgh?",
                                  context_id=first.context_id)
    assert "did not match" in follow.result["note"]
    assert follow.result["materialFindings"]


# --------------------------------------------------------------------------- #
# 8. idempotency and error propagation
# --------------------------------------------------------------------------- #
def test_retrying_the_same_task_id_does_not_run_a_second_assessment(snaps):
    runs = []
    server = _server(snaps, assessor=lambda s: runs.append(1) or _StubRun())
    params = {"message": {"taskId": "task_fixed",
                          "parts": [{"kind": "text", "text": "Assess."}],
                          "metadata": {"resource": PORTFOLIO_A.key}}}
    first = server.message_send(params, _caller())
    second = server.message_send(params, _caller())
    assert len(runs) == 1, "a retry started a second paid assessment"
    assert first["id"] == second["id"] == "task_fixed"
    assert second["status"]["state"] == "completed"


def test_a_specialist_failure_reaches_a_failed_task_without_leaking_internals(
        snaps):
    def exploding(session):
        raise RuntimeError("/srv/trakt/secret/path.py exploded on line 42")

    server = _server(snaps, assessor=exploding)
    result = server.message_send(
        {"message": {"parts": [{"kind": "text", "text": "Assess."}],
                     "metadata": {"resource": PORTFOLIO_A.key}}}, _caller())
    assert result["status"]["state"] == "failed"
    reason = result["status"]["message"]["reason"]
    assert "secret" not in reason and "line 42" not in reason
    assert reason == "RuntimeError while executing the assessment"


def test_an_unexpected_error_becomes_a_bounded_rpc_error(snaps):
    def exploding(session):
        raise RuntimeError("boom")

    server = _server(snaps, assessor=exploding,
                     authoriser=lambda c, r: (_ for _ in ()).throw(
                         ValueError("/etc/passwd")))
    response = server.handle(
        {"jsonrpc": "2.0", "id": "1", "method": "message/send",
         "params": {"message": {"parts": [{"kind": "text", "text": "Assess."}],
                                "metadata": {"resource": PORTFOLIO_A.key}}}},
        _caller())
    assert "passwd" not in json.dumps(response)


def test_a_missing_portfolio_identifier_is_an_invalid_request(snaps):
    server = _server(snaps)
    response = server.handle(
        {"jsonrpc": "2.0", "id": "1", "method": "message/send",
         "params": {"message": {"parts": [{"kind": "text", "text": "Assess."}]}}},
        _caller())
    assert "error" in response


def test_an_unknown_method_is_refused(snaps):
    response = _server(snaps).handle(
        {"jsonrpc": "2.0", "id": "1", "method": "tasks/dance", "params": {}},
        _caller())
    assert "error" in response


# --------------------------------------------------------------------------- #
# 9. MCP equivalence — the claim the whole architecture rests on
# --------------------------------------------------------------------------- #
def _strip_timing(payload: Any) -> Any:
    """Remove measured durations. They differ between any two executions of the
    same computation and are not part of the answer."""
    if isinstance(payload, dict):
        return {k: _strip_timing(v) for k, v in payload.items()
                if k not in ("duration_ms", "elapsed_ms")}
    if isinstance(payload, list):
        return [_strip_timing(v) for v in payload]
    return payload


@pytest.mark.parametrize("tool,args", [
    ("portfolio_capabilities", {"include_definition": True}),
    ("portfolio_summary", {}),
    ("concentration", {"dimension": "geographic_region_collateral", "top_n": 5}),
    ("stratify", {"dimension": "current_loan_to_value", "limit": 10}),
    ("default_analysis", {"months": 6}),
    ("loss_analysis", {"months": 6}),
    ("contractual_analytics", {"metrics": ["contractual_wal"]}),
])
def test_a_governed_result_is_identical_over_mcp_and_direct(snaps, tool, args):
    """The MCP boundary must be transport, not a second answer.

    If any figure differed by transport, every equivalence claim in the Sprint 4
    report would be false — so this compares whole payloads rather than spot
    values, with only measured durations removed.
    """
    direct = GovernedSession(_context(capabilities=BOTH), PORTFOLIO_A.key,
                             dependencies=_deps(snaps))
    mcp = McpSession(context=_context(capabilities=BOTH),
                     dependencies=_deps(snaps))
    via_mcp = GovernedSession(_context(capabilities=BOTH), PORTFOLIO_A.key,
                              dependencies=_deps(snaps),
                              transport=McpToolTransport(mcp))

    assert (_strip_timing(direct.call(tool, dict(args)))
            == _strip_timing(via_mcp.call(tool, dict(args))))
    assert direct._calls[-1].status == via_mcp._calls[-1].status


def test_a_refusal_is_identical_over_mcp_and_direct(snaps):
    """A refusal must cross MCP as a refusal, not as an empty success an agent
    could read as 'no findings'."""
    direct = GovernedSession(_context(capabilities=BOTH), PORTFOLIO_B.key,
                             dependencies=_deps(snaps))
    mcp = McpSession(context=_context(capabilities=BOTH),
                     dependencies=_deps(snaps))
    via_mcp = GovernedSession(_context(capabilities=BOTH), PORTFOLIO_B.key,
                              dependencies=_deps(snaps),
                              transport=McpToolTransport(mcp))
    args = {"requests": [{"metric_id": "ltv_weighted_average"}]}
    a = direct.call("readiness_metrics", dict(args))
    b = via_mcp.call("readiness_metrics", dict(args))
    assert a.get("refused") is True and b.get("refused") is True
    assert a.get("error_code") == b.get("error_code")


def test_mcp_refuses_identity_supplied_as_a_tool_argument(snaps):
    """MCP has no identity model of its own, so an adapter that accepted a
    tenant from the arguments would hand every caller every tenant."""
    mcp = McpSession(context=_context(capabilities=BOTH),
                     dependencies=_deps(snaps))
    with pytest.raises(ValueError, match="identity may not be passed"):
        mcp.call_tool("portfolio_summary",
                      {"resource": PORTFOLIO_A.key, "tenant_id": "other"})


def test_mcp_advertises_only_the_tools_this_caller_may_use(snaps):
    narrow = McpSession(context=_context(capabilities=(SCOPE_LOAN_READ,)),
                        dependencies=_deps(snaps))
    wide = McpSession(context=_context(capabilities=BOTH),
                      dependencies=_deps(snaps))
    assert len(narrow.list_tools()["tools"]) < len(wide.list_tools()["tools"])


def test_the_delegated_run_reaches_trakt_through_mcp(snaps):
    """Proves the execution path rather than asserting it: the MCP session's
    own counter must have advanced by the assessment's governed calls."""
    seen: Dict[str, Any] = {}

    def session_factory(caller: CallerIdentity, resource: str):
        mcp = McpSession(context=caller.context, dependencies=_deps(snaps))
        seen["mcp"] = mcp
        return GovernedSession(caller.context, resource,
                               dependencies=_deps(snaps),
                               transport=McpToolTransport(mcp))

    def assessor(session):
        session.call("portfolio_summary", {})
        session.call("concentration",
                     {"dimension": "geographic_region_collateral", "top_n": 5})
        return _StubRun()

    server = SecuritisationReadinessA2AServer(
        endpoint_url=ENDPOINT, session_factory=session_factory,
        assessor=assessor, authoriser=lambda c, r: None)
    server.message_send(
        {"message": {"parts": [{"kind": "text", "text": "Assess."}],
                     "metadata": {"resource": PORTFOLIO_A.key}}}, _caller())
    assert seen["mcp"].calls == 2, "the specialist bypassed the MCP boundary"


# --------------------------------------------------------------------------- #
# 10. the caller is actually general, and the server actually has no analytics
# --------------------------------------------------------------------------- #
def _source(module_path: str) -> str:
    return (_REPO_ROOT / module_path).read_text(encoding="utf-8")


def test_the_enterprise_caller_contains_no_securitisation_knowledge():
    """Parsed, not eyeballed. A reviewer misses one added constant; the AST
    does not."""
    from trakt_tools.registry import all_tools

    source = _source("enterprise_agent/client.py").lower()
    for spec in all_tools():
        assert spec.name.lower() not in source, (
            f"the general caller names the governed tool {spec.name!r}")
    for term in ("ltv", "arrears", "prepayment", "cpr", "vintage", "concentration",
                 "default_rate", "cure", "severity", "wal", "covenant",
                 "loan_to_value", "dpd"):
        assert term not in source, (
            f"the general caller carries securitisation knowledge: {term!r}")


def test_the_enterprise_caller_holds_no_numeric_thresholds():
    """A threshold in the caller would mean it was judging, not delegating."""
    tree = ast.parse(_source("enterprise_agent/client.py"))
    numbers = [n.value for n in ast.walk(tree)
               if isinstance(n, ast.Constant) and isinstance(n.value, (int, float))
               and not isinstance(n.value, bool)]
    assert all(n in (0, 1, 3, 12) for n in numbers), (
        f"unexpected numeric constants in the general caller: {numbers}")


def test_the_enterprise_caller_cannot_import_trakt_analytics():
    tree = ast.parse(_source("enterprise_agent/client.py"))
    imported = {node.module for node in ast.walk(tree)
                if isinstance(node, ast.ImportFrom) and node.module}
    imported |= {alias.name for node in ast.walk(tree)
                 if isinstance(node, ast.Import) for alias in node.names}
    forbidden = [m for m in imported
                 if m and m.split(".")[0] in {"analytics_lib", "trakt_tools",
                                              "trakt_core", "readiness_agent",
                                              "mi_agent_api", "pandas"}]
    assert not forbidden, f"the general caller imports Trakt internals: {forbidden}"


def test_the_a2a_server_performs_no_arithmetic():
    """The boundary must not become a place where numbers are made."""
    tree = ast.parse(_source("trakt_a2a/server.py"))
    arithmetic = [n for n in ast.walk(tree)
                  if isinstance(n, ast.BinOp)
                  and isinstance(n.op, (ast.Sub, ast.Mult, ast.Div, ast.Mod,
                                        ast.Pow, ast.FloorDiv))]
    assert not arithmetic, (
        "the A2A server contains arithmetic; it must transport results, "
        "not compute them")


def test_the_a2a_package_cannot_reach_trakts_analytics():
    for module in ("trakt_a2a/server.py", "trakt_a2a/card.py",
                   "trakt_a2a/tasks.py"):
        tree = ast.parse(_source(module))
        imported = {node.module for node in ast.walk(tree)
                    if isinstance(node, ast.ImportFrom) and node.module}
        forbidden = [m for m in imported
                     if m and m.split(".")[0] in {"analytics_lib", "pandas"}]
        assert not forbidden, f"{module} imports analytics: {forbidden}"
