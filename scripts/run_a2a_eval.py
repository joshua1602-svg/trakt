#!/usr/bin/env python3
"""Run the Securitisation Readiness Agent through the A2A boundary.

The Sprint 4 counterpart to ``scripts/run_readiness_agent_eval.py``. The
specialist, its prompt, the governed tools, the hidden-truth portfolio and the
scorer are all identical — the only difference is that the objective arrives
over A2A from a general enterprise agent instead of being handed to
``run_assessment`` directly. That is the whole experiment, so nothing else may
vary.

    python scripts/run_a2a_eval.py --runs 1 --out <path>

Requires ``ANTHROPIC_API_KEY``. Output goes wherever ``--out`` says; keep it out
of the repository, because a run record contains a full assessment and is
evaluation evidence rather than source.

Persistence
-----------
Each completed run is written before the next begins, atomically. Sprint 3 lost
three paid-for assessments to a runner that accumulated in memory and wrote once
at the end; work that costs money is persisted the moment it exists.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

ENDPOINT = "https://trakt.example/a2a"

#: What the enterprise agent's user asked for. Business language only — no
#: metric, no threshold, no tool, no investigation plan. Identical in substance
#: to the Sprint 3 objective so the two are comparable.
BUSINESS_NEED = ("I need a specialist securitisation readiness assessment for "
                 "a loan portfolio.")

OBJECTIVE = (
    "Assess this portfolio for securitisation readiness. Identify material "
    "strengths, weaknesses, risks and areas requiring further diligence. "
    "Support material conclusions with governed evidence."
)

FOLLOW_UP = "What is the most material issue, and what evidence supports it?"

UNSUPPORTED = ("Assign a formal Moody's credit rating to this transaction.")


def _build(snaps):
    """The A2A server, wired to the real specialist over a real MCP session."""
    from trakt_core.context import SCOPE_LOAN_READ, SCOPE_RISK_READ
    from trakt_tools.execution import ToolDependencies
    from trakt_tools.mcp_server import McpSession, McpToolTransport

    from readiness_agent.agent import DEFAULT_MODEL, OBJECTIVE as AGENT_OBJECTIVE
    from readiness_agent.agent import run_assessment
    from readiness_agent.session import GovernedSession
    from tests.planted_portfolio import planted_lineage_index
    from tests.readiness_agent_portfolio import PERIODS
    from tests.test_agent_governed_execution import _Datasets, _Descriptor
    from tests.test_agent_loan_retrieval import (PORTFOLIO_A, TENANT,
                                                 _catalogue, _context,
                                                 _CountingResolver)
    from trakt_a2a.server import (A2AError, CallerIdentity,
                                  ERR_FORBIDDEN,
                                  SecuritisationReadinessA2AServer)

    def deps():
        return ToolDependencies(
            datasets=_Datasets(_Descriptor(snapshot_id="secr-demo-final")),
            runtime_mode="test", catalogue=_catalogue(), output_root="/unused",
            loan_frame_resolver=_CountingResolver(snaps[PERIODS[-1]]),
            loan_history_resolver=lambda: snaps,
            lineage_index=planted_lineage_index(TENANT, "secr-demo-final"))

    made: Dict[str, Any] = {}

    def session_factory(caller, resource, correlation_id=None):
        context = caller.context
        if correlation_id:
            context = context.with_correlation(correlation_id)
        mcp = McpSession(context=context, dependencies=deps())
        made["mcp"] = mcp
        return GovernedSession(context, resource, dependencies=deps(),
                               transport=McpToolTransport(mcp))

    def assessor(session):
        # The Sprint 3 agent, unchanged, with its own objective and prompt.
        return run_assessment(
            session, objective=AGENT_OBJECTIVE,
            context_note=("This portfolio is being considered for a term "
                          "securitisation. Supplied rule packs, where any "
                          "exist, are available through the governed rule-pack "
                          "tooling."),
            model=os.environ.get("TRAKT_A2A_MODEL", DEFAULT_MODEL))

    def authoriser(caller, resource):
        if caller.organisation_id != "ere" or resource != PORTFOLIO_A.key:
            raise A2AError(ERR_FORBIDDEN, f"Not entitled to {resource!r}.")

    caller = CallerIdentity(
        agent_id="a2a_test_agent", organisation_id="ere",
        context=_context(capabilities=(SCOPE_LOAN_READ, SCOPE_RISK_READ)))
    server = SecuritisationReadinessA2AServer(
        endpoint_url=ENDPOINT, session_factory=session_factory,
        assessor=assessor, authoriser=authoriser)
    return server, caller, made, PORTFOLIO_A.key


def _client(server, caller):
    from enterprise_agent.client import EnterpriseAgent

    return EnterpriseAgent(
        organisation=caller.organisation_id,
        transport=lambda request: server.handle(request, caller),
        card_fetcher=server.agent_card)


def one_run(snaps, index: int) -> Dict[str, Any]:
    """One complete delegation: discover, delegate, follow up, probe the limit."""
    server, caller, made, portfolio = _build(snaps)
    client = _client(server, caller)
    record: Dict[str, Any] = {"run_index": index}

    # -- discovery, through the card rather than a Python import ------------ #
    t0 = time.perf_counter()
    card = client.discover()
    skill = client.find_skill_for(BUSINESS_NEED)
    record["discovery"] = {
        "ms": round((time.perf_counter() - t0) * 1000, 2),
        "skill_id": (skill or {}).get("id"),
        "endpoint": client.endpoint(),
        "auth_required": client.authentication_required(),
        "card": card,
    }

    # -- delegation --------------------------------------------------------- #
    started = time.perf_counter()
    outcome = client.delegate(OBJECTIVE, portfolio=portfolio)
    record["elapsed_s"] = round(time.perf_counter() - started, 2)

    transcript = server.transcript_for(outcome.task_id)
    task = server.store.get(outcome.task_id)
    governed_ms = sum(c["elapsed_ms"] for c in transcript)

    record["delegation"] = {
        "task_id": outcome.task_id,
        "context_id": outcome.context_id,
        "state": outcome.state,
        "states": [h["state"] for h in (task.history if task else [])],
        "result": outcome.result,
        "error": outcome.error,
    }
    record["transcript"] = transcript
    record["performance"] = {
        "total_s": record["elapsed_s"],
        "governed_ms": round(governed_ms, 2),
        "protocol_overhead_ms": round(
            record["elapsed_s"] * 1000 - governed_ms, 2),
        "governed_calls": len(transcript),
    }
    record["audit"] = dict(task.metadata) if task else {}

    # -- follow-up on the same context -------------------------------------- #
    if outcome.state == "completed":
        t1 = time.perf_counter()
        follow = client.ask_follow_up(FOLLOW_UP, context_id=outcome.context_id)
        record["follow_up"] = {
            "ms": round((time.perf_counter() - t1) * 1000, 2),
            "state": follow.state,
            "result": follow.result,
            "governed_calls_after": len(server.transcript_for(outcome.task_id)),
        }

    # -- an objective outside the advertised skill -------------------------- #
    refused = client.delegate(UNSUPPORTED, portfolio=portfolio)
    record["unsupported"] = {"state": refused.state, "error": refused.error}
    return record


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("ANTHROPIC_API_KEY is not set; refusing to run.", file=sys.stderr)
        return 2

    import logging
    import warnings
    warnings.filterwarnings("ignore")
    logging.disable(logging.WARNING)

    from tests.readiness_agent_portfolio import snapshots

    snaps = snapshots()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    records: List[Dict[str, Any]] = []
    if out.exists():
        try:
            records = json.loads(out.read_text(encoding="utf-8"))
            print(f"resuming: {len(records)} run(s) already in {out}")
        except (ValueError, OSError):
            records = []

    for index in range(len(records) + 1, len(records) + args.runs + 1):
        print(f"\n=== A2A run {index} ===", flush=True)
        record = one_run(snaps, index)
        records.append(record)
        _persist(out, records)
        perf = record["performance"]
        print(f"  -> {record['delegation']['state']}; "
              f"{perf['governed_calls']} governed calls; "
              f"{perf['governed_ms']:.0f}ms Trakt / {perf['total_s']:.1f}s total; "
              f"overhead {perf['protocol_overhead_ms']:.0f}ms  "
              f"[saved {len(records)}]", flush=True)

    print(f"\nwrote {len(records)} run(s) to {out}")
    return 0


def _persist(out: Path, records) -> None:
    """Write-and-rename, so an interruption never leaves a half-written file."""
    temporary = out.with_suffix(out.suffix + ".partial")
    temporary.write_text(json.dumps(records, indent=2, default=str),
                         encoding="utf-8")
    os.replace(temporary, out)


if __name__ == "__main__":
    raise SystemExit(main())
