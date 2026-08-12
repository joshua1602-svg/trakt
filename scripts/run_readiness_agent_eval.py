#!/usr/bin/env python3
"""Run the Securitisation Readiness Agent against the hidden-truth portfolio.

Not collected by pytest and requires ``ANTHROPIC_API_KEY``. The agent receives
the sparse objective, the portfolio resource and nothing else — in particular
it never sees ``ANSWER_KEY``, which this script deliberately does not import.

    python scripts/run_readiness_agent_eval.py --runs 1 --out <path>

Output goes wherever ``--out`` says. Keep it out of the repository: a run
record contains the full assessment and is evaluation evidence, not source.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def build_session():
    """A governed session over the hidden-truth portfolio.

    Imports the portfolio BUILDER only. ``ANSWER_KEY`` is in the same module
    and is never touched here — the agent's runtime must not be able to reach
    the thing it is being scored against, and the cheapest guarantee is not
    importing it.
    """
    from trakt_tools.execution import ToolDependencies

    from readiness_agent.session import GovernedSession
    from tests.planted_portfolio import planted_lineage_index
    from tests.readiness_agent_portfolio import PERIODS, snapshots
    from tests.test_agent_governed_execution import _Datasets, _Descriptor
    from tests.test_agent_loan_retrieval import (
        PORTFOLIO_A, TENANT, _catalogue, _context, _CountingResolver)
    from trakt_core.context import SCOPE_LOAN_READ, SCOPE_RISK_READ

    snaps = snapshots()
    deps = ToolDependencies(
        datasets=_Datasets(_Descriptor(snapshot_id="secr-demo-final")),
        runtime_mode="test", catalogue=_catalogue(), output_root="/unused",
        loan_frame_resolver=_CountingResolver(snaps[PERIODS[-1]]),
        loan_history_resolver=lambda: snaps,
        lineage_index=planted_lineage_index(TENANT, "secr-demo-final"))
    context = _context(capabilities=(SCOPE_LOAN_READ, SCOPE_RISK_READ))
    return GovernedSession(context, PORTFOLIO_A.key, dependencies=deps)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--out", required=True)
    parser.add_argument("--model", default=None)
    parser.add_argument("--max-steps", type=int, default=40)
    args = parser.parse_args()

    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("ANTHROPIC_API_KEY is not set; refusing to run.", file=sys.stderr)
        return 2

    import logging
    import warnings
    warnings.filterwarnings("ignore")
    logging.disable(logging.WARNING)

    from readiness_agent.agent import DEFAULT_MODEL, OBJECTIVE, run_assessment

    model = args.model or DEFAULT_MODEL
    records = []
    for index in range(1, args.runs + 1):
        session = build_session()
        print(f"\n=== run {index}/{args.runs} ({model}) ===", flush=True)

        def on_step(step: int, tool: str, _i=index) -> None:
            print(f"  step {step:2d}  {tool}", flush=True)

        run = run_assessment(
            session,
            objective=OBJECTIVE,
            context_note=(
                "This portfolio is being considered for a term securitisation. "
                "Supplied rule packs, where any exist, are available through "
                "the governed rule-pack tooling."),
            model=model, max_steps=args.max_steps, on_step=on_step)

        record = run.to_dict()
        record["run_index"] = index
        records.append(record)
        print(f"  -> {run.stopped_reason}; {run.efficiency.get('total_calls')} "
              f"tool calls; {run.usage.get('input_tokens')} in / "
              f"{run.usage.get('output_tokens')} out tokens; "
              f"{run.elapsed_s:.1f}s", flush=True)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(records, indent=2, default=str),
                   encoding="utf-8")
    print(f"\nwrote {len(records)} run(s) to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
