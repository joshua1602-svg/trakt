#!/usr/bin/env python3
"""Governance, audit correlation and protocol overhead across the A2A boundary.

Sprint 4's model-independent half. Everything here runs without a model, which
is the point: infrastructure defects should be found by tests that cost nothing,
not discovered halfway through a paid evaluation run.

Three questions, in order of how much they matter:

**Is the A2A pre-check load-bearing?** It must not be. ``trakt_a2a.identity``
refuses an unentitled caller before the specialist spends minutes and money, and
that is a cost and latency protection — *not* the security boundary. The
security boundary is ``execute_governed_tool``, which resolves entitlements on
every call and would still refuse if the whole A2A layer were deleted. The tests
below prove that by deleting it: they install an authoriser that permits
everything, and check Trakt refuses anyway.

**Can one delegation be reconstructed end to end?** Caller, organisation,
portfolio, task, every governed call, and the final artifact — from stored
records, without the specialist's process still being alive, and without a
tracing system.

**What does the protocol itself cost?** Measured separately from the specialist's
work, because "A2A is slow" and "credit analysis takes minutes" are different
claims and only one of them would be A2A's fault.
"""

from __future__ import annotations

import json
import sys
import time
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
from trakt_a2a.card import AGENT_CARD_PATH, agent_card
from trakt_a2a.server import (A2AError, CallerIdentity,
                              SecuritisationReadinessA2AServer)

ENDPOINT = "https://trakt.example/a2a"
BOTH = (SCOPE_LOAN_READ, SCOPE_RISK_READ)


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


class _Run:
    """A specialist that records what its session did, without a model."""

    def __init__(self, session, calls):
        self.session = session
        self.assessment = {
            "overall_readiness": "READY_WITH_ISSUES",
            "summary": "Recorded by a stub specialist.",
            "material_findings": [{
                "title": "Geographic concentration",
                "fact": "London is 31.0% of balance.",
                "rule": "Securitisation criteria cap a region at 27% — BREACH.",
                "judgement": "Structural.", "severity": "high",
                "evidence_tools": ["concentration"]}],
            "could_not_assess": [], "strengths": [], "further_diligence": [],
        }
        self.stopped_reason = "submitted"
        self.efficiency = session.efficiency()
        self.transcript = session.transcript()
        self.calls = calls


def _server(snaps, *, authoriser=None, tool_plan=(("portfolio_summary", {}),)):
    """A server whose specialist makes a fixed set of real governed calls."""
    made: Dict[str, Any] = {}

    def session_factory(caller, resource, correlation_id=None):
        context = caller.context
        if correlation_id:
            context = context.with_correlation(correlation_id)
        mcp = McpSession(context=context, dependencies=_deps(snaps))
        made["mcp"] = mcp
        made["context"] = context
        return GovernedSession(context, resource, dependencies=_deps(snaps),
                               transport=McpToolTransport(mcp))

    def assessor(session):
        for tool, args in tool_plan:
            session.call(tool, dict(args))
        return _Run(session, made)

    server = SecuritisationReadinessA2AServer(
        endpoint_url=ENDPOINT, session_factory=session_factory,
        assessor=assessor,
        authoriser=authoriser if authoriser is not None else _strict)
    server._made = made          # test-only handle on what the factory built
    return server


def _strict(caller: CallerIdentity, resource: str) -> None:
    from trakt_a2a.server import ERR_FORBIDDEN
    if caller.organisation_id != "ere" or resource != PORTFOLIO_A.key:
        raise A2AError(ERR_FORBIDDEN, f"Not entitled to {resource!r}.")


def _delegate(server, caller=None, portfolio=PORTFOLIO_A.key,
              objective="Assess this portfolio for securitisation readiness."):
    identity = caller if caller is not None else _caller()
    client = EnterpriseAgent(organisation=identity.organisation_id,
                             transport=lambda r: server.handle(r, identity),
                             card_fetcher=server.agent_card)
    return client, client.delegate(objective, portfolio=portfolio)


# --------------------------------------------------------------------------- #
# 1. the A2A pre-check is NOT the security boundary
# --------------------------------------------------------------------------- #
def test_trakt_refuses_an_unentitled_portfolio_even_with_the_precheck_disabled(
        snaps):
    """The load-bearing security test.

    The authoriser permits everything — as it would if the pre-check were
    removed, misconfigured, or bypassed by a caller reaching the specialist
    another way. Trakt's governed execution must refuse anyway, on its own
    entitlement resolution, with the delegation still reaching a terminal state
    rather than hanging or half-succeeding.
    """
    server = _server(snaps, authoriser=lambda caller, resource: None,
                     tool_plan=(("portfolio_summary", {}),))
    _, outcome = _delegate(server, portfolio=PORTFOLIO_B.key)

    transcript = server.transcript_for(outcome.task_id)
    assert transcript, "the specialist made no governed call at all"
    refused = [c for c in transcript if c["status"] != "success"]
    assert refused, (
        "the A2A pre-check was disabled and Trakt served the portfolio anyway; "
        "entitlement enforcement is not independent of this layer")
    assert all(c["error_code"] for c in refused)


def test_trakt_refuses_another_tenants_portfolio_with_the_precheck_disabled(
        snaps):
    server = _server(snaps, authoriser=lambda caller, resource: None)
    _, outcome = _delegate(server, portfolio="other-tenant/portfolio/SECRET")
    transcript = server.transcript_for(outcome.task_id)
    assert transcript and all(c["status"] != "success" for c in transcript)


def test_a_capability_the_caller_lacks_is_refused_with_the_precheck_disabled(
        snaps):
    """Being delegated to over A2A confers no authority.

    A session holding only ``loan:read`` must not reach a ``risk:read`` tool,
    however it was asked.
    """
    narrow = CallerIdentity(agent_id="a2a_test_agent", organisation_id="ere",
                            context=_context(capabilities=(SCOPE_LOAN_READ,)))
    server = _server(snaps, authoriser=lambda caller, resource: None,
                     tool_plan=(("default_analysis", {}),))
    _, outcome = _delegate(server, caller=narrow)
    transcript = server.transcript_for(outcome.task_id)
    assert transcript and transcript[0]["status"] != "success"


def test_the_precheck_still_saves_the_expensive_work(snaps):
    """Its actual job: refuse before the specialist runs, not instead of Trakt."""
    ran = []

    def counting_assessor(session):
        ran.append(1)
        return _Run(session, {})

    server = SecuritisationReadinessA2AServer(
        endpoint_url=ENDPOINT,
        session_factory=lambda c, r, correlation_id=None: None,
        assessor=counting_assessor, authoriser=_strict)
    with pytest.raises(A2AError):
        server.message_send(
            {"message": {"parts": [{"kind": "text", "text": "Assess."}],
                         "metadata": {"resource": PORTFOLIO_B.key}}}, _caller())
    assert ran == []


# --------------------------------------------------------------------------- #
# 2. end-to-end audit correlation
# --------------------------------------------------------------------------- #
def test_one_delegation_is_reconstructable_end_to_end(snaps):
    server = _server(snaps, tool_plan=(
        ("portfolio_summary", {}),
        ("concentration", {"dimension": "geographic_region_collateral",
                           "top_n": 5})))
    _, outcome = _delegate(server)

    task = server.store.get(outcome.task_id)
    transcript = server.transcript_for(outcome.task_id)

    # Who asked, on whose behalf, for what.
    assert task.metadata["calling_agent"] == "a2a_test_agent"
    assert task.metadata["organisation"] == "ere"
    assert task.metadata["resource"] == PORTFOLIO_A.key
    assert task.metadata["actor_type"] == "service"
    assert task.metadata["channel"]

    # What happened, in order, with a terminal state.
    assert [h["state"] for h in task.history] == ["submitted", "working",
                                                  "completed"]
    # Which governed capabilities answered.
    assert [c["tool"] for c in transcript] == ["portfolio_summary",
                                               "concentration"]
    assert all(c["status"] == "success" for c in transcript)
    assert task.metadata["governedCalls"] == len(transcript)
    # And the deliverable.
    assert outcome.result["materialFindings"][0]["fact"]


def test_the_task_id_is_the_correlation_id_carried_into_trakt(snaps):
    """One identifier links the A2A task to every governed execution beneath it.

    Without this, reconstructing a delegation means correlating on timestamps,
    which stops working the moment two callers delegate at once.
    """
    server = _server(snaps)
    _, outcome = _delegate(server)
    assert server._made["context"].correlation_id == outcome.task_id


def test_the_audit_trail_holds_actions_not_reasoning(snaps):
    """A stored chain of thought reads as evidence and is not."""
    server = _server(snaps)
    _, outcome = _delegate(server)
    blob = json.dumps(server.transcript_for(outcome.task_id)).lower()
    for leak in ("thinking", "reasoning", "chain_of_thought", "rationale"):
        assert leak not in blob


def test_the_transcript_survives_the_run_object(snaps):
    """An auditor arrives later than the run. The record must outlive it."""
    server = _server(snaps)
    _, outcome = _delegate(server)
    server._made.clear()
    assert server.transcript_for(outcome.task_id)


# --------------------------------------------------------------------------- #
# 3. protocol overhead, measured apart from the specialist's work
# --------------------------------------------------------------------------- #
def test_protocol_overhead_is_small_beside_the_governed_work(snaps):
    """A2A must not be materially wasteful.

    Deliberately a loose bound rather than a precise budget: this asserts the
    protocol is not a bottleneck, which is the real question. A tight threshold
    here would fail on a loaded machine and teach people to re-run tests.
    """
    server = _server(snaps, tool_plan=(("portfolio_summary", {}),))

    started = time.perf_counter()
    _, outcome = _delegate(server)
    total_ms = (time.perf_counter() - started) * 1000.0

    transcript = server.transcript_for(outcome.task_id)
    governed_ms = sum(c["elapsed_ms"] for c in transcript)
    overhead_ms = total_ms - governed_ms

    assert outcome.state == "completed"
    assert overhead_ms < 1000.0, (
        f"A2A overhead {overhead_ms:.1f}ms on {governed_ms:.1f}ms of governed "
        "work — the protocol layer is doing something expensive")


def test_card_discovery_costs_nothing_measurable(snaps):
    server = _server(snaps)
    started = time.perf_counter()
    for _ in range(50):
        server.agent_card()
    per_call_ms = (time.perf_counter() - started) * 1000.0 / 50
    assert per_call_ms < 20.0


# --------------------------------------------------------------------------- #
# 4. Agent Card security posture (assessment, not cryptography)
# --------------------------------------------------------------------------- #
def test_the_card_declares_how_a_caller_must_authenticate():
    """A card that named no security scheme would invite anonymous access to
    lender portfolio intelligence."""
    card = agent_card(endpoint_url=ENDPOINT)
    assert card["securityRequirements"]
    scheme = card["securitySchemes"]["enterprise_agent_oauth"]
    assert scheme["type"] == "oauth2"
    assert "clientCredentials" in scheme["flows"]


def test_the_card_separates_agent_identity_from_portfolio_entitlement():
    """The distinction, stated where a caller will actually read it."""
    described = agent_card(endpoint_url=ENDPOINT)[
        "securitySchemes"]["enterprise_agent_oauth"]["description"].lower()
    assert "which agent" in described
    assert "does not" in described and "portfolio" in described


def test_the_card_is_unsigned_and_that_is_recorded_not_hidden():
    """A2A v1.0 supports signed Agent Cards. This deployment does not sign, and
    the production gap is stated in the report rather than papered over with
    bespoke cryptography. The card must therefore carry no signature claim it
    cannot honour."""
    assert "signatures" not in agent_card(endpoint_url=ENDPOINT)


def test_the_well_known_path_is_the_one_the_specification_fixes():
    assert AGENT_CARD_PATH == "/.well-known/agent-card.json"
