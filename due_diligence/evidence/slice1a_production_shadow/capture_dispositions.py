#!/usr/bin/env python3
"""Phase 4 evidence: one real record per disposition, with no live model call.

The tests prove the isolation; this proves the EVIDENCE, by running the
production orchestration for every disposition it can reach and showing what each
record actually contains. Run against `ReplayClient` over the frozen run-8
payloads, so the plans here are the plans the sign-off adjudicated.

Records are written to a temporary sink and a summary is written beside this
script. Full records for two dispositions are included verbatim so a reader can
see the shape without running anything — chosen because they are the two that
matter most: an executed grouped plan (the widest record) and an ineligible one
(where the gate's reason is the whole point).

Run: `python due_diligence/evidence/slice1a_production_shadow/capture_dispositions.py`
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from mi_agent import plan_runtime_adapter as adapter                  # noqa: E402
from mi_agent import plan_shadow_evidence as evidence                 # noqa: E402
from mi_agent import plan_shadow_wiring as wiring                    # noqa: E402
from mi_agent.interpretation_v2.opus_interpreter import (              # noqa: E402
    OpusInterpreter, ReplayClient, UnavailableClient)
from mi_agent.mi_query_validator import load_mi_semantics            # noqa: E402
from mi_agent.tests import portfolio_truth_oracle as truth           # noqa: E402

FROZEN = (_REPO_ROOT / "mi_agent" / "interpretation_v2" / "evidence"
          / "run8_135_signoff_2b00172.json")
HERE = Path(__file__).resolve().parent
CLIENT = "acme"

SCENARIOS = (
    ("EXECUTED_SCALAR", "How many loans are to borrowers over 55 with LTV above 50%?",
     CLIENT, "replay"),
    ("EXECUTED_GROUPED", "Chart the balance by LTV bucket and borrower-age bucket.",
     CLIENT, "replay"),
    ("INELIGIBLE", "What is the weighted-average LTV of lump sum loans in the "
                   "Direct book?", CLIENT, "replay"),
    ("CLARIFY", "How has the profile of our new lending changed over the last few "
                "months?", CLIENT, "replay"),
    ("INTERPRETER_FAILURE", "How many loans are to borrowers over 55 with LTV "
                            "above 50%?", CLIENT, "unavailable"),
    ("EXECUTION_ERROR", "How many loans are to borrowers over 55 with LTV above "
                        "50%?", CLIENT, "replay"),
    ("OUTSIDE_CANARY", "How many loans are to borrowers over 55 with LTV above "
                       "50%?", "beta", "replay"),
)


def payloads() -> Dict[str, Any]:
    body = json.loads(FROZEN.read_text())
    return {r["question"]: r["raw_payload"] for r in body["results"]
            if r.get("raw_payload")}


def main() -> int:
    semantics = load_mi_semantics(
        str(_REPO_ROOT / "mi_agent" / "mi_semantics_field_registry.yaml"))
    book = truth.canonical_book()
    recorded = payloads()

    summary: List[Dict[str, Any]] = []
    full: Dict[str, Any] = {}

    with tempfile.TemporaryDirectory() as tmp:
        os.environ[adapter.SHADOW_ENV_VAR] = adapter.SHADOW_ON
        os.environ[adapter.LEDGER_ENV_VAR] = os.path.join(tmp, "ledger.jsonl")
        os.environ[evidence.SINK_ENV_VAR] = os.path.join(tmp, "evidence.jsonl")
        os.environ[wiring.CANARY_ENV_VAR] = CLIENT
        wiring.set_dispatch(wiring._dispatch_inline)
        try:
            for label, question, client, kind in SCENARIOS:
                if kind == "unavailable":
                    wiring.set_interpreter_factory(
                        lambda: OpusInterpreter(UnavailableClient("no key here")))
                else:
                    wiring.set_interpreter_factory(
                        lambda: OpusInterpreter(
                            ReplayClient(recorded, model_id="claude-opus-5")))

                served = {"ok": True, "value": 1.0, "route": None,
                          "answer": "the legacy answer"}
                before = json.dumps(served, sort_keys=True)
                frame = object() if label == "EXECUTION_ERROR" else book
                record = wiring.observe_request(
                    question=question, client_id=client, run_id="2026-03",
                    result=served, frame=frame, semantics=semantics,
                    view="funded", portfolio_id=f"{client}/2026-03")
                unchanged = json.dumps(served, sort_keys=True) == before

                summary.append({
                    "scenario": label,
                    "disposition": (record or {}).get("disposition"),
                    "served_envelope_unchanged": unchanged,
                    "evidence_persisted": (record or {}).get("evidence_persisted"),
                    "model_id": ((record or {}).get("model") or {}).get("model_id"),
                    "plan_id": ((record or {}).get("compiler") or {}).get("plan_id"),
                    "eligible": ((record or {}).get("eligibility") or {}
                                 ).get("eligible"),
                    "eligibility_reason": ((record or {}).get("eligibility") or {}
                                           ).get("reason"),
                    "value": ((record or {}).get("execution") or {}).get("value"),
                    "grouped_cells": len((((record or {}).get("execution") or {}
                                           ).get("grouped_cells")) or ()),
                    "execution_error": ((record or {}).get("execution") or {}
                                        ).get("error"),
                })
                if label in ("EXECUTED_GROUPED", "INELIGIBLE"):
                    clean, _ = evidence.redact(record)
                    full[label] = clean
        finally:
            wiring.set_interpreter_factory(None)
            wiring.set_dispatch(None)
            for key in (adapter.SHADOW_ENV_VAR, adapter.LEDGER_ENV_VAR,
                        evidence.SINK_ENV_VAR, wiring.CANARY_ENV_VAR):
                os.environ.pop(key, None)

    out = {
        "what_this_is": "one record per reachable disposition, produced by the "
                        "production orchestration with a replayed frozen payload",
        "live_model_calls": 0,
        "canary_client": CLIENT,
        "summary": summary,
        "full_records": full,
    }
    (HERE / "disposition_matrix.json").write_text(
        json.dumps(out, indent=2, default=str) + "\n")

    print(f"{'scenario':<22}{'disposition':<22}{'served?':<9}"
          f"{'persisted':<11}{'cells':<7}reason/value")
    for row in summary:
        extra = (row["eligibility_reason"] or row["execution_error"]
                 or row["value"] or "")
        print(f"{row['scenario']:<22}{str(row['disposition']):<22}"
              f"{'SAME' if row['served_envelope_unchanged'] else 'CHANGED':<9}"
              f"{str(row['evidence_persisted']):<11}{row['grouped_cells']:<7}"
              f"{str(extra)[:48]}")
    served_safe = all(row["served_envelope_unchanged"] for row in summary)
    print(f"\nUSER_VISIBLE_RESPONSE_DIFFS  {0 if served_safe else 'NON-ZERO'}")
    print(f"WRITTEN  "
          f"{(HERE / 'disposition_matrix.json').relative_to(_REPO_ROOT)}")
    return 0 if served_safe else 1


if __name__ == "__main__":
    raise SystemExit(main())
