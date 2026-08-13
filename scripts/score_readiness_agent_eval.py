#!/usr/bin/env python3
"""Score a set of Readiness Agent runs and report their consistency.

Separate from the runner on purpose. Running costs money and cannot be
repeated; scoring is free and will be repeated every time the scorer or the
answer key is corrected. Keeping them apart means a scoring change never
implies re-running, and a re-run never silently re-scores under a different
key.

    python scripts/score_readiness_agent_eval.py --runs-file <path>

Reads the run records the runner wrote. Imports ``ANSWER_KEY`` freely — this
process never talks to a model, so there is nothing here for the truth to
leak into.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

#: Published list price for the evaluation model, US dollars per million
#: tokens. Stated as data rather than folded into a formula so that a price
#: change is a one-line edit and every cost in the report is traceable to it.
PRICE_PER_MTOK = {"input": 3.00, "output": 15.00}


def _cost(usage: Dict[str, int]) -> float:
    return (usage.get("input_tokens", 0) / 1e6 * PRICE_PER_MTOK["input"]
            + usage.get("output_tokens", 0) / 1e6 * PRICE_PER_MTOK["output"])


def _table(rows: Sequence[Sequence[Any]], headers: Sequence[str]) -> str:
    widths = [max(len(str(h)), *(len(str(r[i])) for r in rows)) if rows
              else len(str(h)) for i, h in enumerate(headers)]
    line = "  ".join(str(h).ljust(w) for h, w in zip(headers, widths))
    out = [line, "  ".join("-" * w for w in widths)]
    out.extend("  ".join(str(c).ljust(w) for c, w in zip(row, widths))
               for row in rows)
    return "\n".join(out)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-file", required=True)
    parser.add_argument("--json", default=None,
                        help="also write the scored summary as JSON")
    args = parser.parse_args()

    from tests.readiness_agent_portfolio import ANSWER_KEY
    from tests.readiness_agent_scoring import FOUND, MISSED, consistency, score_runs

    records = json.loads(Path(args.runs_file).read_text(encoding="utf-8"))
    scores = score_runs(records)

    print(f"{len(scores)} run(s) from {args.runs_file}\n")

    print("PER-RUN SCORE")
    print(_table(
        [[s.summary()["run"], s.overall_readiness or "-",
          f"{s.discovery_rate}%",
          sum(1 for c in s.cases if c.status == FOUND),
          sum(1 for c in s.cases if c.status not in (FOUND, MISSED)),
          sum(1 for c in s.cases if c.status == MISSED),
          len(s.numerical_errors), len(s.methodology_errors),
          len(s.false_positives), len(s.epistemic_failures),
          s.efficiency.get("total_calls"), s.efficiency.get("repeated_calls")]
         for s in scores],
        ["run", "verdict", "discovery", "found", "part", "miss",
         "num_err", "meth_err", "false_pos", "epist", "calls", "repeat"]))

    print("\nPER-CASE MATRIX")
    case_ids = [c["id"] for c in ANSWER_KEY if c.get("agent_should_discover")]
    short = {FOUND: "F", MISSED: "-"}
    rows: List[List[Any]] = []
    for case_id in case_ids:
        row: List[Any] = [case_id]
        for score in scores:
            hit = next((c for c in score.cases if c.case_id == case_id), None)
            row.append(short.get(hit.status, "p") if hit else "?")
        rows.append(row)
    print(_table(rows, ["case"] + [f"r{s.run_index}" for s in scores]))
    print("F = found, p = partial, - = missed")

    print("\nTOKENS AND COST")
    token_rows = [[s.run_index, f"{s.usage.get('input_tokens', 0):,}",
                   f"{s.usage.get('output_tokens', 0):,}",
                   s.efficiency.get("total_calls"),
                   f"${_cost(s.usage):.2f}"] for s in scores]
    total_in = sum(s.usage.get("input_tokens", 0) for s in scores)
    total_out = sum(s.usage.get("output_tokens", 0) for s in scores)
    token_rows.append(["all", f"{total_in:,}", f"{total_out:,}", "",
                       f"${sum(_cost(s.usage) for s in scores):.2f}"])
    print(_table(token_rows, ["run", "input", "output", "calls", "cost"]))

    summary = consistency(scores)
    print("\nCONSISTENCY")
    for label, value in summary.items():
        print(f"  {label:28s} {value}")

    if args.json:
        Path(args.json).write_text(json.dumps({
            "per_run": [s.summary() for s in scores],
            "per_case": {case_id: [
                next((c.status for c in s.cases if c.case_id == case_id), None)
                for s in scores] for case_id in case_ids},
            "consistency": summary,
            "tokens": {"input": total_in, "output": total_out},
        }, indent=2), encoding="utf-8")
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
