#!/usr/bin/env python3
"""Capture the governed stage-transition payload for the React component test.

The fixture is the ENGINE'S OWN OUTPUT, not a hand-written approximation of it.
That is the point: the React test asserts against exactly the object the deck
renders and the API returns, so a component that quietly recomputed a count
could not agree with it, and a payload change that the panel does not handle
shows up as a failing React test rather than as a wrong number in a browser.

`mi_agent_api/tests/test_stage_transition_exposure.py` re-runs the engine and
compares it to the committed file, so the fixture cannot silently drift out of
date. Regenerate with:

    python frontend/mi-agent-ui/scripts/generate_stage_transition_fixture.py
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

warnings.simplefilter("ignore")
REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

#: The deterministic two-snapshot pack built for the engine sprint. Small enough
#: that every expected number in the React test is arithmetic on the fixture.
PIPELINE = REPO / "tests" / "fixtures" / "pipeline_transition_2w"
TARGET = (REPO / "frontend" / "mi-agent-ui" / "src" / "test" / "fixtures"
          / "stageTransitionDetail.json")

#: Resolution provenance, not analytical content — it varies with the run
#: layout, so pinning it would make the fixture environment-dependent.
_VOLATILE = ("run_id",)


def build() -> dict:
    from mi_agent_api import movement_detail as md
    detail = md.resolve_stage_transition_detail(str(PIPELINE), "client_001")
    for key in _VOLATILE:
        detail.pop(key, None)
    return detail


def main() -> int:
    detail = build()
    if not detail.get("available"):
        print(f"refusing to write an unavailable fixture: {detail.get('reason')}")
        return 1
    TARGET.parent.mkdir(parents=True, exist_ok=True)
    TARGET.write_text(json.dumps(detail, indent=2, sort_keys=True) + "\n",
                      encoding="utf-8")
    print(f"wrote {TARGET.relative_to(REPO)} "
          f"({len(detail['transitions'])} transitions, "
          f"{len(detail['departures'])} departure groups)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
