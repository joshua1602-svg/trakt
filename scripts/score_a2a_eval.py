#!/usr/bin/env python3
"""Score A2A-delegated assessments with the unmodified Sprint 3 scorer.

    python scripts/score_a2a_eval.py --runs-file <a2a> [--direct-file <sprint3>]

The point of this script is that it contains no scoring. It converts an A2A
artifact back into the shape ``tests.readiness_agent_scoring`` already reads,
then calls that module untouched. Writing a second scorer for the delegated runs
would have made the comparison meaningless — any difference could then have been
the scorer rather than the specialist.

The conversion is itself evidence
---------------------------------
If a field cannot be reconstructed from the artifact, the protocol lost it. So
this adapter is deliberately a plain renaming — ``overallReadiness`` to
``overall_readiness``, ``evidenceTools`` to ``evidence_tools`` — with no
defaults, no inference and no repair. A round-trip that still scores is a
round-trip that preserved the evidence model.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

PRICE_PER_MTOK = {"input": 3.00, "output": 15.00}


def sprint3_shape(record: Dict[str, Any]) -> Dict[str, Any]:
    """One A2A run record, in the shape the Sprint 3 scorer reads.

    Pure renaming. Anything absent stays absent, so a field the protocol
    dropped scores as missing rather than being quietly filled in.
    """
    artifact = (record.get("delegation") or {}).get("result") or {}
    findings = []
    for finding in artifact.get("materialFindings") or []:
        findings.append({
            "title": finding.get("title"),
            "fact": finding.get("fact"),
            "rule": finding.get("rule"),
            "judgement": finding.get("judgement"),
            "severity": finding.get("severity"),
            "evidence_tools": finding.get("evidenceTools") or [],
        })
    assessment = {
        "overall_readiness": artifact.get("overallReadiness"),
        "summary": artifact.get("summary"),
        "strengths": artifact.get("strengths") or [],
        "material_findings": findings,
        "could_not_assess": artifact.get("couldNotAssess") or [],
        "further_diligence": artifact.get("furtherDiligence") or [],
    }
    performance = record.get("performance") or {}
    return {
        "run_index": record.get("run_index"),
        "stopped_reason": (record.get("delegation") or {}).get("state", ""),
        "assessment": assessment if artifact else None,
        "transcript": record.get("transcript") or [],
        "efficiency": {
            "total_calls": performance.get("governed_calls"),
            "repeated_calls": performance.get("repeated_calls"),
        },
        "usage": performance.get("usage") or {},
    }


def _table(rows, headers) -> str:
    widths = [max(len(str(h)), *(len(str(r[i])) for r in rows)) if rows
              else len(str(h)) for i, h in enumerate(headers)]
    out = ["  ".join(str(h).ljust(w) for h, w in zip(headers, widths)),
           "  ".join("-" * w for w in widths)]
    out.extend("  ".join(str(c).ljust(w) for c, w in zip(row, widths))
               for row in rows)
    return "\n".join(out)


def _summarise(scores, label: str) -> Dict[str, Any]:
    from tests.readiness_agent_scoring import FOUND, MISSED

    rows = [[s.summary()["run"], s.overall_readiness or "-",
             f"{s.discovery_rate}%",
             sum(1 for c in s.cases if c.status == FOUND),
             sum(1 for c in s.cases if c.status not in (FOUND, MISSED)),
             sum(1 for c in s.cases if c.status == MISSED),
             len(s.numerical_errors), len(s.methodology_errors),
             len(s.false_positives), len(s.epistemic_failures),
             s.efficiency.get("total_calls")] for s in scores]
    print(f"\n{label}")
    print(_table(rows, ["run", "verdict", "discovery", "found", "part", "miss",
                        "num_err", "meth_err", "false_pos", "epist", "calls"]))
    rates = [s.discovery_rate for s in scores]
    return {"min": min(rates), "max": max(rates),
            "mean": round(sum(rates) / len(rates), 1)}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-file", required=True)
    parser.add_argument("--direct-file", default=None,
                        help="Sprint 3 direct runs, for the control comparison")
    args = parser.parse_args()

    from tests.readiness_agent_portfolio import ANSWER_KEY
    from tests.readiness_agent_scoring import (FOUND, MISSED, consistency,
                                               score_runs)

    records = json.loads(Path(args.runs_file).read_text(encoding="utf-8"))
    delegated = [sprint3_shape(r) for r in records]
    scores = score_runs(delegated)

    print(f"{len(scores)} A2A-delegated run(s) from {args.runs_file}")
    envelope = _summarise(scores, "PER-RUN SCORE — A2A DELEGATED")

    print("\nPER-CASE MATRIX")
    case_ids = [c["id"] for c in ANSWER_KEY if c.get("agent_should_discover")]
    short = {FOUND: "F", MISSED: "-"}
    rows = []
    for case_id in case_ids:
        row = [case_id]
        for score in scores:
            hit = next((c for c in score.cases if c.case_id == case_id), None)
            row.append(short.get(hit.status, "p") if hit else "?")
        rows.append(row)
    print(_table(rows, ["case"] + [f"r{s.run_index}" for s in scores]))
    print("F = found, p = partial, - = missed")

    print("\nPROTOCOL RELIABILITY (deterministic — must be 5/5)")
    protocol = []
    for record in records:
        delegation = record.get("delegation") or {}
        artifact = delegation.get("result") or {}
        follow = record.get("follow_up") or {}
        findings = artifact.get("materialFindings") or []
        protocol.append([
            record["run_index"],
            delegation.get("state"),
            "->".join(delegation.get("states") or []),
            "yes" if artifact else "NO",
            "yes" if all(f.get("fact") and f.get("rule") is not None
                         and f.get("judgement") for f in findings) else "NO",
            len(artifact.get("couldNotAssess") or []),
            follow.get("state", "-"),
            "no" if follow.get("result", {}).get("portfolioReRun") is False
            else "?",
            (record.get("unsupported") or {}).get("state"),
        ])
    print(_table(protocol, ["run", "state", "lifecycle", "artifact",
                            "fact/rule/judgement", "gaps", "follow-up",
                            "re-ran?", "unsupported"]))

    print("\nPERFORMANCE")
    perf_rows = [[r["run_index"], f"{r['performance']['total_s']:.1f}s",
                  f"{r['performance']['llm_ms']/1000:.1f}s",
                  f"{r['performance']['governed_ms']:.0f}ms",
                  f"{r['performance']['protocol_overhead_ms']:.1f}ms",
                  r["performance"]["governed_calls"],
                  f"{r['discovery']['ms']:.2f}ms"] for r in records]
    print(_table(perf_rows, ["run", "total", "LLM", "Trakt", "A2A overhead",
                             "calls", "discovery"]))

    print("\nCONSISTENCY — A2A")
    for label, value in consistency(scores).items():
        print(f"  {label:28s} {value}")

    if args.direct_file:
        direct_records = json.loads(
            Path(args.direct_file).read_text(encoding="utf-8"))
        direct_scores = score_runs(direct_records)
        direct_envelope = _summarise(direct_scores,
                                     "PER-RUN SCORE — DIRECT (Sprint 3)")
        print("\nDIRECT vs A2A — discovery envelope")
        print(f"  direct   min {direct_envelope['min']}%  "
              f"mean {direct_envelope['mean']}%  max {direct_envelope['max']}%")
        print(f"  A2A      min {envelope['min']}%  "
              f"mean {envelope['mean']}%  max {envelope['max']}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
