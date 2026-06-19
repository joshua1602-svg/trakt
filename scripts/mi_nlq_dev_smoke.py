#!/usr/bin/env python3
"""DEV-ONLY live Anthropic NLQ → MI runtime smoke (Phase 8C).

THIS IS NOT PART OF THE AUTOMATED TEST SUITE. It is a manual developer aid for
sanity-checking a *real* Anthropic client end-to-end against a small synthetic
local snapshot. It still routes everything through the governed pipeline:

    question -> AnthropicClient (real) -> MIQuerySpec JSON
             -> MIQuerySpec.normalized() -> validate_query_spec()
             -> run_mi_query()   (the ONLY execution engine)

The LLM never computes analytics — it only proposes a spec. Invalid / ambiguous
interpretations are NOT executed.

Requirements / safety:
  * Requires the ANTHROPIC_API_KEY environment variable to be set; refuses to run
    otherwise. No key is ever read from anywhere else.
  * Requires the optional ``anthropic`` SDK to be installed (imported lazily by
    AnthropicClient). The normal test suite does NOT depend on the SDK.
  * Output is clearly labelled DEV SMOKE ONLY.

Usage:
    ANTHROPIC_API_KEY=sk-... python scripts/mi_nlq_dev_smoke.py "show total funded"
    ANTHROPIC_API_KEY=sk-... python scripts/mi_nlq_dev_smoke.py   # runs a few defaults
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

DEFAULT_QUESTIONS = [
    "show total funded",
    "show funded balance by region",
    "show risk grade migration",
    "show me something vague",   # expected: clarification, not executed
]

CLIENT = "devsmoke"
REPORTING_DATES = ["2024-01-31", "2024-02-29", "2024-03-31"]


def _build_store(tmp_root):
    import pandas as pd

    from snapshot.adapters import LocalFsSnapshotStore
    from snapshot.model import SnapshotHeader

    def loan(lid, status, stage, bal, region, grade, ifrs, pd_b, prob=None):
        return {
            "loan_identifier": lid, "funded_status": status,
            "pipeline_stage": stage, "current_outstanding_balance": float(bal),
            "geographic_region_obligor": region, "broker_channel": "Broker A",
            "portfolio_id": "PF_001", "internal_risk_grade": grade,
            "ifrs9_stage": ifrs, "pd_bucket": pd_b,
            "forecast_funding_probability": prob, "origination_date": "2020-01-15",
        }

    frames = [
        pd.DataFrame([
            loan("F1", "funded", "completed", 100, "North", "A", "Stage 1", "<0.25%"),
            loan("F2", "funded", "completed", 200, "South", "B", "Stage 1", "0.25-0.5%"),
            loan("P1", "pipeline", "OFFER", 50, "North", None, None, None, 0.5),
        ]),
        pd.DataFrame([
            loan("F1", "funded", "completed", 100, "North", "A", "Stage 1", "<0.25%"),
            loan("F2", "funded", "completed", 220, "South", "B", "Stage 1", "0.25-0.5%"),
            loan("F3", "funded", "completed", 300, "North", "C", "Stage 2", "1-2.5%"),
        ]),
        pd.DataFrame([
            loan("F1", "funded", "completed", 100, "North", "A", "Stage 1", "<0.25%"),
            loan("F2", "funded", "completed", 220, "South", "C", "Stage 2", "0.5-1%"),
            loan("F3", "funded", "completed", 300, "North", "C", "Stage 2", "1-2.5%"),
        ]),
    ]
    store = LocalFsSnapshotStore(root=Path(tmp_root) / "snaps")
    for i, (rd, frame) in enumerate(zip(REPORTING_DATES, frames)):
        store.register_snapshot(
            SnapshotHeader(client_id=CLIENT, route="mi", reporting_date=rd,
                           source_file_id=f"sha256:dev{i}", cadence="monthly",
                           upload_timestamp=f"{rd}T09:00:00"),
            frame)
    return store


def main(argv):
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("ERROR: ANTHROPIC_API_KEY is not set. This dev smoke calls the real "
              "Anthropic API and refuses to run without it.", file=sys.stderr)
        return 2

    from mi_agent.interpreter import InterpreterContext, interpret_and_run_mi_query
    from mi_agent.interpreter.anthropic import AnthropicClient
    from mi_agent.mi_query_validator import load_mi_semantics
    from mi_agent.risk_monitor import load_risk_monitor_config

    questions = argv[1:] or DEFAULT_QUESTIONS
    semantics = load_mi_semantics(
        REPO_ROOT / "mi_agent" / "mi_semantics_field_registry.yaml")
    risk_cfg = load_risk_monitor_config()
    client = AnthropicClient()          # lazily imports the anthropic SDK
    ctx = InterpreterContext(snapshot_client_id=CLIENT, route_id="mi",
                             as_of="2024-03-31", prev_period="2024-02-29",
                             range_start="2024-01-01")

    print("=" * 70)
    print("DEV SMOKE ONLY — live Anthropic interpretation, synthetic local data")
    print("The LLM only proposes a MIQuerySpec; run_mi_query computes everything.")
    print("=" * 70)

    with tempfile.TemporaryDirectory() as tmp:
        store = _build_store(tmp)
        for q in questions:
            print(f"\nQ: {q}")
            result = interpret_and_run_mi_query(
                q, ctx, client, store, semantics=semantics, risk_config=risk_cfg)
            spec = result.normalized_spec
            print(f"  interpreted spec: "
                  f"{spec.to_json(indent=0) if spec else None}")
            if not result.executed:
                if result.clarification_required:
                    print("  NOT EXECUTED — clarification required: "
                          f"{result.interpretation.clarification_question}")
                else:
                    print(f"  NOT EXECUTED — issues: {result.issue_codes()}")
                continue
            rr = result.runtime_result
            print(f"  EXECUTED via run_mi_query: mode={rr.mode} "
                  f"rows={rr.row_count} ok={rr.ok}")
            if rr.row_count:
                print(rr.data.head(10).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
