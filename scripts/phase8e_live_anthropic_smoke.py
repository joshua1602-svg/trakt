#!/usr/bin/env python3
"""DEV-ONLY Phase 8E — live Anthropic MI interpretation smoke + evaluation.

THIS IS NOT PART OF THE AUTOMATED TEST SUITE and never runs in CI. It is a
controlled, manual evaluation of whether Claude can interpret a fixed set of
natural-language MI questions into MIQuerySpec v2 accurately enough to run against
the existing synthetic/local MI runtime.

Every question is routed through the SAME governed pipeline used everywhere else:

    question
      -> AnthropicClient (real, Phase 8B adapter)   # proposes a spec only
      -> MIQuerySpec.normalized()
      -> validate_query_spec()
      -> interpret_and_run_mi_query / run_mi_query   # the ONLY execution engine

The LLM never computes analytics, never bypasses validation, and invalid or
ambiguous interpretations are NEVER executed.

Safety / constraints:
  * Requires ANTHROPIC_API_KEY; refuses to run (exit 2) if it is missing.
  * Uses synthetic, in-memory, local snapshot data only — NEVER real client data.
  * Writes observed results to a gitignored artefact
    (artifacts/phase8e_live_anthropic_smoke_results.json) because the output may
    contain raw provider responses.
  * The optional ``anthropic`` SDK is imported lazily by AnthropicClient; the
    normal test suite does not depend on it.

Usage:
    ANTHROPIC_API_KEY=sk-... python scripts/phase8e_live_anthropic_smoke.py
    ANTHROPIC_API_KEY=sk-... python scripts/phase8e_live_anthropic_smoke.py --out path.json
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

DEFAULT_OUT = REPO_ROOT / "artifacts" / "phase8e_live_anthropic_smoke_results.json"

CLIENT = "phase8e_devsmoke"
REPORTING_DATES = ["2024-01-31", "2024-02-29", "2024-03-31"]

# Controlled smoke set (Phase 8A/8D). Each carries the expected SAFE behaviour:
#   "execute"  -> should interpret to a valid spec and run successfully;
#   "clarify"  -> should NOT execute (clarification or validation block).
SMOKE_QUESTIONS = [
    ("show total funded", "execute"),
    ("show total pipeline", "execute"),
    ("show forecast funded", "execute"),
    ("trend funded balance over the last three months", "execute"),
    ("compare funded balance to last month", "execute"),
    ("show funded balance by portfolio", "execute"),
    ("show funded balance by region", "execute"),
    ("show pipeline by stage", "execute"),
    ("show concentration by region", "execute"),
    ("show risk grade migration", "execute"),
    ("show IFRS stage migration", "execute"),
    ("show PD bucket migration", "execute"),
    ("show risk", "clarify"),
    ("show changes", "clarify"),
    ("show stage", "clarify"),
    ("show portfolio", "clarify"),
    ("show rate", "clarify"),
]


def _build_store(tmp_root):
    """Three synthetic canonical MI snapshots in a local FS store."""
    import pandas as pd

    from snapshot.adapters import LocalFsSnapshotStore
    from snapshot.model import SnapshotHeader

    def loan(lid, status, stage, bal, region, broker, portfolio,
             grade=None, ifrs=None, pd_b=None, prob=None):
        return {
            "loan_identifier": lid, "funded_status": status,
            "pipeline_stage": stage, "current_outstanding_balance": float(bal),
            "geographic_region_obligor": region, "broker_channel": broker,
            "portfolio_id": portfolio, "internal_risk_grade": grade,
            "ifrs9_stage": ifrs, "pd_bucket": pd_b,
            "forecast_funding_probability": prob, "origination_date": "2020-01-15",
        }

    frames = [
        pd.DataFrame([
            loan("F1", "funded", "completed", 100, "North", "Broker A", "PF_001",
                 "A", "Stage 1", "<0.25%"),
            loan("F2", "funded", "completed", 200, "South", "Broker B", "PF_002",
                 "B", "Stage 1", "0.25-0.5%"),
            loan("P1", "pipeline", "OFFER", 50, "North", "Broker A", "PF_001",
                 prob=0.5),
            loan("P2", "pipeline", "KFI", 40, "South", "Broker B", "PF_002",
                 prob=0.25),
        ]),
        pd.DataFrame([
            loan("F1", "funded", "completed", 100, "North", "Broker A", "PF_001",
                 "A", "Stage 1", "<0.25%"),
            loan("F2", "funded", "completed", 220, "South", "Broker B", "PF_002",
                 "B", "Stage 1", "0.25-0.5%"),
            loan("F3", "funded", "completed", 300, "North", "Broker A", "PF_001",
                 "C", "Stage 2", "1-2.5%"),
            loan("P1", "pipeline", "OFFER", 50, "North", "Broker A", "PF_001",
                 prob=0.5),
            loan("P2", "pipeline", "APPLICATION", 40, "South", "Broker B",
                 "PF_002", prob=0.45),
        ]),
        pd.DataFrame([
            loan("F1", "funded", "completed", 100, "North", "Broker A", "PF_001",
                 "A", "Stage 1", "<0.25%"),
            loan("F2", "funded", "completed", 220, "South", "Broker B", "PF_002",
                 "C", "Stage 2", "0.5-1%"),
            loan("F3", "funded", "completed", 300, "North", "Broker A", "PF_001",
                 "C", "Stage 2", "1-2.5%"),
            loan("P1", "pipeline", "OFFER", 50, "North", "Broker A", "PF_001",
                 prob=0.5),
        ]),
    ]
    store = LocalFsSnapshotStore(root=Path(tmp_root) / "snaps")
    for i, (rd, frame) in enumerate(zip(REPORTING_DATES, frames)):
        store.register_snapshot(
            SnapshotHeader(client_id=CLIENT, route="mi", reporting_date=rd,
                           source_file_id=f"sha256:p8e{i}", cadence="monthly",
                           upload_timestamp=f"{rd}T09:00:00"),
            frame)
    return store


class _RecordingClient:
    """Wraps an Anthropic-style client to capture the raw model output per call,
    so the evaluation artefact can record exactly what Claude returned."""

    def __init__(self, inner):
        self._inner = inner
        self.last_raw = None

    def complete_mi_spec_json(self, prompt):
        raw = self._inner.complete_mi_spec_json(prompt)
        self.last_raw = raw
        return raw


def _grade(expected: str, result) -> bool:
    """SAFE-behaviour grading.

    execute -> must have executed with an ok runtime result.
    clarify -> must NOT have executed (clarification or validation block both
               count as safe).
    """
    if expected == "execute":
        return bool(result.executed and result.runtime_result is not None
                    and result.runtime_result.ok)
    return not result.executed


def _evaluate(question, expected, ctx, client, store, semantics, risk_cfg):
    from mi_agent.interpreter import interpret_and_run_mi_query

    rec = _RecordingClient(client)
    result = interpret_and_run_mi_query(
        question, ctx, rec, store, semantics=semantics, risk_config=risk_cfg)
    interp = result.interpretation
    vr = interp.validation_result
    spec = result.normalized_spec

    runtime_summary = None
    if result.executed and result.runtime_result is not None:
        rr = result.runtime_result
        runtime_summary = {
            "mode": rr.mode, "result_type": rr.result_type,
            "row_count": rr.row_count, "ok": rr.ok,
            "chart_instruction": rr.chart_instruction,
            "metadata_keys": sorted(rr.metadata.keys()),
        }

    return {
        "raw_question": question,
        "expected_behaviour": expected,
        "raw_claude_output": rec.last_raw,
        "parsed_candidate_spec": interp.candidate_spec,
        "normalized_spec": spec.to_dict() if spec is not None else None,
        "validation_result": (
            {"ok": vr.ok, "codes": vr.codes()} if vr is not None else None),
        "executed": result.executed,
        "runtime_result_summary": runtime_summary,
        "issue_codes": result.issue_codes(),
        "clarification_question": interp.clarification_question,
        "passed": _grade(expected, result),
    }


def run_smoke(out_path: Path = DEFAULT_OUT) -> int:
    """Run the live smoke. Requires ANTHROPIC_API_KEY. Returns a process code."""
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("ERROR: ANTHROPIC_API_KEY is not set. This dev smoke calls the real "
              "Anthropic API and refuses to run without it.", file=sys.stderr)
        return 2

    from mi_agent.interpreter import InterpreterContext
    from mi_agent.interpreter.anthropic import AnthropicClient
    from mi_agent.mi_query_validator import load_mi_semantics
    from mi_agent.risk_monitor import load_risk_monitor_config

    semantics = load_mi_semantics(
        REPO_ROOT / "mi_agent" / "mi_semantics_field_registry.yaml")
    risk_cfg = load_risk_monitor_config()
    client = AnthropicClient()       # lazily imports the anthropic SDK
    ctx = InterpreterContext(snapshot_client_id=CLIENT, route_id="mi",
                             as_of="2024-03-31", prev_period="2024-02-29",
                             range_start="2024-01-01")

    print("=" * 72)
    print("DEV SMOKE ONLY — Phase 8E live Anthropic interpretation evaluation")
    print("Synthetic/local data. The LLM proposes a MIQuerySpec; run_mi_query")
    print("computes everything. Invalid/ambiguous specs are never executed.")
    print("=" * 72)

    records = []
    with tempfile.TemporaryDirectory() as tmp:
        store = _build_store(tmp)
        for question, expected in SMOKE_QUESTIONS:
            rec = _evaluate(question, expected, ctx, client, store, semantics,
                            risk_cfg)
            records.append(rec)
            status = "PASS" if rec["passed"] else "FAIL"
            outcome = ("executed" if rec["executed"]
                       else ("clarified" if rec["clarification_question"]
                             else "blocked"))
            print(f"[{status}] {question!r} -> {outcome} "
                  f"(codes={rec['issue_codes']})")

    passed = sum(1 for r in records if r["passed"])
    summary = {
        "phase": "8E",
        "note": "DEV SMOKE ONLY — synthetic/local data; may contain provider "
                "responses; do not commit.",
        "total": len(records), "passed": passed,
        "failed": len(records) - passed, "results": records,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2, default=str),
                        encoding="utf-8")
    print(f"\n{passed}/{len(records)} passed. Wrote {out_path}")
    return 0 if passed == len(records) else 1


def main(argv) -> int:
    out = DEFAULT_OUT
    if "--out" in argv:
        out = Path(argv[argv.index("--out") + 1])
    return run_smoke(out)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
