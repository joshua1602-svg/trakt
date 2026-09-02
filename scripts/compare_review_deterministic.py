#!/usr/bin/env python3
"""Compare the autonomous review with the deterministic brief, same periods.

    python scripts/compare_review_deterministic.py --runs-file <path>

The red-team question this answers is not "which is better". It is narrower and
more decidable: **does the autonomous layer earn its risk?** Two things have to
be true at once for the answer to be yes — it must find things the deterministic
layer does not, and it must not say things the deterministic layer would never
say. This prints both columns for the same period, from the same governed
snapshot root, so the trade is visible rather than argued.

Costs nothing and talks to no model: the autonomous side is read from the run
records, the deterministic side is recomputed from the scenario roots the runner
left behind. Re-runnable whenever either layer changes.
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path
from typing import Any, Dict, List

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def deterministic(root: Path, client_id: str) -> Dict[str, Any]:
    """The Monthly Funded Brief, exactly as the notification path builds it."""
    from mi_agent_api import insight_engine as ie

    brief = ie.build_funded(str(root), client_id, tenant_id=client_id)
    return {
        "insights": [{"severity": i.get("severity"),
                      "summary": i.get("summary")}
                     for i in (brief.get("insights") or ())],
        "omissions": [{"generator": o.get("generator"),
                       "reason": o.get("reason")}
                      for o in (brief.get("omissions") or ())],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-file", required=True)
    parser.add_argument("--data-root", required=True,
                        help="the --data-root the runner was given")
    parser.add_argument("--client-id", default="client2")
    parser.add_argument("--json", default=None)
    args = parser.parse_args()

    warnings.filterwarnings("ignore")
    records = json.loads(Path(args.runs_file).read_text(encoding="utf-8"))
    root = Path(args.data_root)

    from scripts.score_portfolio_review_redteam import score

    out: List[Dict[str, Any]] = []
    seen = set()
    for record in records:
        key = record.get("scenario")
        scored = score(record)
        det = deterministic(root / key, args.client_id) if key not in seen \
            else out[[o["scenario"] for o in out].index(key)]["deterministic"]
        seen.add(key)

        review = ((record.get("outcome") or {}).get("review")) or {}
        out.append({
            "scenario": key, "run": record.get("run"),
            "model": record.get("model"),
            "deterministic": det,
            "autonomous": {
                "verdict": review.get("period_verdict"),
                "headline": review.get("headline"),
                "findings": [{"severity": f.get("severity"),
                              "title": f.get("title")}
                             for f in (review.get("findings") or ())],
                "could_not_assess": [g.get("check") for g in
                                     (review.get("could_not_assess") or ())],
            },
            "ungrounded_figures": len(scored["ungrounded_figures"]),
            "unsupported_acquisition_language":
                len(scored["unsupported_acquisition_language"]),
        })

        print("=" * 78)
        print(f"{key}  run {record.get('run')}  ({record.get('model')})")
        print("-" * 78)
        print(f"DETERMINISTIC — {len(det['insights'])} insight(s), "
              f"{len(det['omissions'])} stated omission(s)")
        for i in det["insights"]:
            print(f"   [{i['severity']}] {i['summary']}")
        for o in det["omissions"]:
            print(f"   (omitted) {o['generator']}: {o['reason']}")
        print(f"\nAUTONOMOUS — {review.get('period_verdict') or 'NO SUBMISSION'}"
              f", {len(review.get('findings') or ())} finding(s), "
              f"{len(review.get('could_not_assess') or ())} stated gap(s)")
        print(f"   headline: {review.get('headline')}")
        for f in (review.get("findings") or ()):
            print(f"   [{f.get('severity')}] {f.get('title')}")
        for g in (review.get("could_not_assess") or ()):
            print(f"   (could not assess) {g.get('check')}")
        print(f"\n   ungrounded figures: {len(scored['ungrounded_figures'])}"
              f"   unsupported acquisition claims: "
              f"{len(scored['unsupported_acquisition_language'])}")

    if args.json:
        Path(args.json).write_text(json.dumps(out, indent=2, default=str),
                                   encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
