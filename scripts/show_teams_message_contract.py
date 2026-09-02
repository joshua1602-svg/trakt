#!/usr/bin/env python3
"""Render the production Teams message for each period type, end to end.

    python scripts/show_teams_message_contract.py [--enrich <runs.json>]

Not a mock. This runs the real `trigger.on_publication_approved` against real
pipeline canonical, through the real resolver, generator, enrichment and card
renderer, and prints what a recipient would actually see for each period type.
It is the evidence behind the message contract in the sprint report.

Nothing is delivered. Recipients are never populated, so the trigger stores the
batch and stops — which is the shadow-mode behaviour §19 asks about, exercised
rather than described.

``--enrich`` replays autonomous findings recorded by
``run_portfolio_review_redteam.py`` so the enriched contract can be shown
without spending model credit; the enrichment path taken is the production one
either way.
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

CLIENT_ID = "client2"

#: Which recorded agent scenario stands in for which deterministic period.
_ENRICH_FOR = {
    "quiet": "D_quiet",
    "organic_growth": "B_organic",
    "acquisition": "A_acquisition",
    "acquisition_masking_decline": "E_mixed",
    "concentration_warning": "C_risk_warning",
}


class _Outcome:
    """A recorded ReviewOutcome, replayed. Same duck type `enrich` reads."""

    def __init__(self, record: Dict[str, Any]):
        outcome = record.get("outcome") or {}
        self.card = outcome.get("card")
        self.gate_status = outcome.get("gate_status") or ""
        self.steps = outcome.get("steps")
        self.efficiency = outcome.get("efficiency") or {}
        self.dropped_findings = outcome.get("dropped_findings") or []
        self.unsupported_claims = outcome.get("unsupported_claims") or []
        self.out_of_mandate_calls = outcome.get("out_of_mandate_calls") or []


def _render(batch, enrichment_record: Dict[str, Any]) -> str:
    from trakt_notifications.contract import MESSAGE_PORTFOLIO_UPDATE

    lines: List[str] = []
    for message in batch.messages:
        kind = ("PORTFOLIO UPDATE" if message.message_type ==
                MESSAGE_PORTFOLIO_UPDATE else "RISK REVIEW")
        lines.append(f"  --- {kind} [{message.severity}] ---")
        lines.append(f"  {message.headline}")
        if message.summary:
            lines.append(f"  {message.summary}")
        deterministic = [i for i in message.items
                         if i.metric != "autonomous_observation"]
        autonomous = [i for i in message.items
                      if i.metric == "autonomous_observation"]
        for item in deterministic:
            mark = "!" if item.unavailable else "-"
            lines.append(f"    {mark} {item.text}")
        if autonomous:
            lines.append("")
            lines.append("    Management observations")
            for item in autonomous:
                lines.append(f"      • {item.text}")
        lines.append("")
    words = sum(len((m.headline or "").split()) + len((m.summary or "").split())
                + sum(len((i.text or "").split()) for i in m.items)
                for m in batch.messages)
    lines.append(f"  [{words} words · enrichment {enrichment_record.get('status')}"
                 f" · gate {enrichment_record.get('gate_status') or '—'}"
                 f" · added {enrichment_record.get('added', 0)}"
                 f" · dropped {len(enrichment_record.get('dropped') or [])}]")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--enrich", default=None,
                        help="a runs.json from run_portfolio_review_redteam")
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--only", default=None)
    args = parser.parse_args()

    warnings.filterwarnings("ignore")
    import logging
    logging.getLogger().setLevel(logging.ERROR)

    from trakt_notifications import enrichment, generate, sources
    from trakt_core import config_cache

    import tests.portfolio_review_redteam as rt

    recorded: Dict[str, Dict[str, Any]] = {}
    if args.enrich:
        for path in str(args.enrich).split(","):
            for record in json.loads(Path(path).read_text(encoding="utf-8")):
                recorded.setdefault(str(record.get("scenario")), record)

    root = Path(args.data_root or (_REPO_ROOT / ".message_contract"))
    rt.clear(root)
    periods = rt.build_periods(root, CLIENT_ID)
    if args.only:
        periods = [p for p in periods if p.key == args.only]

    for scenario in periods:
        client = scenario.traps.get("client_id", CLIENT_ID)
        config_cache.reset()
        print("=" * 78)
        print(f"{scenario.key}  [{client}]  — {scenario.title}")
        print("=" * 78)

        inputs = sources.resolve(
            tenant_id=client, portfolio_id=client,
            portfolio_context="total", funded_run_id=None,
            want_pipeline=False, want_funded=True,
            output_root=str(scenario.root))
        batch = generate.build(inputs, update_type="FUNDED",
                               approved_run_ids=[rt.CURRENT_RUN])

        key = _ENRICH_FOR.get(scenario.key)
        record = recorded.get(key or "")
        reviewer = (lambda r=record: _Outcome(r)) if record else None
        batch = enrichment.enrich(batch, reviewer=reviewer)
        print(_render(batch, enrichment.record_of(batch)))

    rt.clear(root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
