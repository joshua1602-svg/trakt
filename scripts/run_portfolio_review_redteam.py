#!/usr/bin/env python3
"""Run the Portfolio Review agent against the adversarial period scenarios.

Not collected by pytest and requires ``ANTHROPIC_API_KEY``. Follows the same
split as ``run_readiness_agent_eval.py``: this script spends money and cannot be
re-run for free, the scorer next to it is free and will be re-run every time the
scoring rule is corrected. Keeping them apart means a scoring change never
implies a re-run, and a re-run never silently re-scores under a changed rule.

    python scripts/run_portfolio_review_redteam.py --runs 2 --out <path>

The agent is given the objective, the resource and the period dates. It is never
given ``Scenario.traps`` — the scenario module is imported for its BUILDER, and
the facts it will be scored against travel to the scorer inside the run record
rather than through the model's context.

Output goes wherever ``--out`` says. Keep it out of the repository: a run record
contains full governed tool payloads and is evaluation evidence, not source.
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

#: Deliberately not "ERE". The production path must carry no assumption about
#: which lender it is running for, and a red-team that only ever ran as the
#: original tenant would never notice one.
CLIENT_ID = "client2"


class _Recorder:
    """Wraps a governed session and keeps the FULL payload of every call.

    ``GovernedSession.transcript`` keeps a digest, which is the right thing for
    an audit record and useless for grounding: to decide whether a number in the
    narrative came from Trakt, the scorer needs the numbers Trakt actually
    returned. This adds no capability — it forwards ``call`` unchanged and
    cannot reach anything the session would not have returned anyway.
    """

    def __init__(self, session):
        self._session = session
        self.payloads: List[Dict[str, Any]] = []

    def call(self, tool: str, arguments=None):
        payload = self._session.call(tool, arguments)
        self.payloads.append({"tool": tool, "arguments": dict(arguments or {}),
                              "result": payload})
        return payload

    def __getattr__(self, name):
        return getattr(self._session, name)


def build_session(output_root: Path):
    """A governed session over one scenario's snapshot root."""
    import pandas as pd

    from readiness_agent.session import GovernedSession
    from trakt_core.context import (
        ACTOR_SERVICE, CHANNEL_ENTERPRISE_AGENT, ExecutionContext,
        SCOPE_LOAN_READ, SCOPE_RISK_READ,
    )
    from trakt_core.entitlement import EntitlementStore, Grant
    from trakt_core.resource import (
        KIND_PORTFOLIO, ResourceCatalogue, ResourceRecord, ResourceRef,
    )
    from trakt_tools.execution import ToolDependencies

    from tests.portfolio_review_redteam import CENTRAL_TAPE, CURRENT_RUN
    from tests.test_agent_governed_execution import _Datasets, _Descriptor

    book = ResourceRef(CLIENT_ID, KIND_PORTFOLIO, "total")
    catalogue = ResourceCatalogue(
        [ResourceRecord(ref=book, display_label="Total book",
                        whole_tenant_book=True)], configured=True)

    current = pd.read_csv(
        output_root / CLIENT_ID / CURRENT_RUN / "central" / CENTRAL_TAPE,
        low_memory=False)

    caps = (SCOPE_LOAN_READ, SCOPE_RISK_READ)
    grants = EntitlementStore(
        [Grant(organisation_id="redteam", resource_ref=book,
               capabilities=frozenset(caps))], configured=True, validate=False)
    context = ExecutionContext(
        tenant_id=CLIENT_ID, actor_id="sp-redteam", actor_type=ACTOR_SERVICE,
        channel=CHANNEL_ENTERPRISE_AGENT, scopes=frozenset(caps),
        organisation_id="redteam", entitlements=grants.resolve("redteam"))

    deps = ToolDependencies(
        datasets=_Datasets(_Descriptor(snapshot_id="redteam-canonical")),
        runtime_mode="test", catalogue=catalogue, output_root=str(output_root),
        loan_frame_resolver=lambda: current,
        pipeline_root=None)
    return GovernedSession(context, book.key, dependencies=deps)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=int, default=1,
                        help="repeats per scenario, for variance")
    parser.add_argument("--out", required=True)
    parser.add_argument("--model", default=None)
    parser.add_argument("--max-steps", type=int, default=24)
    parser.add_argument("--only", default=None, help="one scenario key")
    parser.add_argument("--data-root", default=None)
    args = parser.parse_args()

    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("ANTHROPIC_API_KEY is not set; refusing to run.", file=sys.stderr)
        return 2

    import logging
    import warnings
    warnings.filterwarnings("ignore")
    logging.getLogger().setLevel(logging.ERROR)

    from portfolio_review.controller import run_review
    from readiness_agent.agent import DEFAULT_MODEL
    from trakt_core import config_cache

    import tests.portfolio_review_redteam as rt

    data_root = Path(args.data_root or (Path(args.out).parent / "redteam_data"))
    rt.clear(data_root)
    scenarios = rt.build(data_root, CLIENT_ID)
    if args.only:
        scenarios = [s for s in scenarios if s.key == args.only]
        if not scenarios:
            print(f"no scenario named {args.only!r}", file=sys.stderr)
            return 2

    model = args.model or DEFAULT_MODEL
    records: List[Dict[str, Any]] = []

    for scenario in scenarios:
        for attempt in range(1, args.runs + 1):
            config_cache.reset()
            session = _Recorder(build_session(scenario.root))
            started = time.perf_counter()
            print(f"  {scenario.key} run {attempt}/{args.runs} ...",
                  file=sys.stderr, flush=True)
            try:
                outcome = run_review(
                    session, period=scenario.period, client_id=CLIENT_ID,
                    output_root=str(scenario.root), model=model,
                    max_steps=args.max_steps)
                error = None
            except Exception as exc:  # noqa: BLE001 - a failed run is a result
                outcome, error = None, f"{type(exc).__name__}: {exc}"

            records.append({
                "scenario": scenario.key,
                "title": scenario.title,
                "trap": scenario.trap,
                "evidence_class": scenario.evidence_class,
                "derivation": scenario.derivation,
                "traps": scenario.traps,
                "run": attempt,
                "model": model,
                "elapsed_s": round(time.perf_counter() - started, 2),
                "error": error,
                "outcome": outcome.to_dict() if outcome else None,
                "payloads": session.payloads,
            })

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(records, indent=2, default=str),
                   encoding="utf-8")
    print(f"wrote {len(records)} run(s) to {out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
