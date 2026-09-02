#!/usr/bin/env python3
"""Run the deterministic Teams brief over the §12 period set and report it.

    python scripts/run_deterministic_period_suite.py

Talks to no model and costs nothing, so it is re-run whenever either the
generator set or the mandate changes. Every period is built from the committed
multibook canonical by deletion, column scaling or re-keying — no row is
authored — so the distributions under test are the ones the pipeline produced.

What this checks is the half of the estate that was already CONDITIONAL GO: the
deterministic brief. The autonomous agent's own re-run is
``run_portfolio_review_redteam.py`` and needs credentials.
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

CLIENT_ID = "client2"


def _brief(root: Path, client_id: str) -> Dict[str, Any]:
    from mi_agent_api import insight_engine as ie

    return ie.build_funded(str(root), client_id, tenant_id=client_id)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--json", default=None)
    args = parser.parse_args()

    warnings.filterwarnings("ignore")
    import logging
    logging.getLogger().setLevel(logging.ERROR)

    import tests.portfolio_review_redteam as rt

    root = Path(args.data_root or (_REPO_ROOT / ".deterministic_periods"))
    rt.clear(root)
    periods = rt.build_periods(root, CLIENT_ID)

    records: List[Dict[str, Any]] = []
    for scenario in periods:
        client = scenario.traps.get("client_id", CLIENT_ID)
        try:
            brief = _brief(scenario.root, client)
            error = None
        except Exception as exc:  # noqa: BLE001 - a failure is a result
            brief, error = {}, f"{type(exc).__name__}: {exc}"

        insights = brief.get("insights") or []
        omissions = brief.get("omissions") or []
        text = " ".join(str(i.get("summary") or "") for i in insights)
        records.append({
            "scenario": scenario.key, "title": scenario.title,
            "client_id": client, "evidence_class": scenario.evidence_class,
            "derivation": scenario.derivation, "error": error,
            "insight_count": len(insights),
            "omission_count": len(omissions),
            "severities": sorted({str(i.get("severity")) for i in insights}),
            "summaries": [str(i.get("summary") or "") for i in insights],
            "omissions": [{"generator": o.get("generator"),
                           "reason": o.get("reason")} for o in omissions],
            "mentions_acquisition": "acquisition" in text.lower()
            or "acquir" in text.lower(),
            "words": len(text.split()),
        })

        print("=" * 78)
        print(f"{scenario.key}  [{client}]  — {scenario.title}")
        print(f"  built: {scenario.derivation}")
        if error:
            print(f"  ERROR: {error}")
        print(f"  {len(insights)} insight(s), {len(omissions)} omission(s), "
              f"{len(text.split())} words")
        for insight in insights:
            print(f"    [{insight.get('severity')}] {insight.get('summary')}")
        for omission in omissions:
            print(f"    (omitted) {omission.get('generator')}: "
                  f"{omission.get('reason')}")

    print("\n" + "=" * 78)
    failed = [r for r in records if r["error"]]
    print(f"{len(records) - len(failed)}/{len(records)} periods produced a brief")
    total = sum(r["words"] for r in records)
    print(f"total narrative across all periods: {total} words "
          f"(mean {total // max(1, len(records))})")

    if args.json:
        Path(args.json).write_text(json.dumps(records, indent=2, default=str),
                                   encoding="utf-8")
    rt.clear(root)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
