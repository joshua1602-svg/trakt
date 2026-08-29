#!/usr/bin/env python3
"""migration_phase0/classify_estate.py — attribute every estate failure.

READ-ONLY. Reads a pytest short-summary capture (``-rf --tb=no``) and partitions
the failures by subsystem, so the Phase 0 baseline can answer the only question
that matters for attribution:

    Is any failing test in the MI QUERY AGENT path — the surface this migration
    touches — or are they all in subsystems it does not?

A migration is measured against A5 ("any existing delivered/refused behaviour
moves and cannot be explained"). That comparison is only meaningful if the
failures present BEFORE the migration are recorded by name.

The raw pytest capture it reads is a transient intermediate and is gitignored;
the durable artefact is ``migration_phase0/estate.json``, which carries every
failing node id BY NAME. Regenerate the capture with::

    TRAKT_RUNTIME_MODE=development python -m pytest \
        tests/ mi_agent/tests/ mi_agent_api/tests/ \
        -q -p no:randomly --tb=no -rf > migration_phase0/estate_full.txt 2>&1

    python -m migration_phase0.classify_estate migration_phase0/estate_full.txt
"""
from __future__ import annotations

import json
import re
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

_REPO = Path(__file__).resolve().parent.parent

#: Path fragments that put a test IN the MI query agent path — the surface this
#: migration touches. Ordered most specific first.
MI_PATH = (
    "mi_agent/tests/", "mi_agent_api/tests/",
    "tests/test_mi_", "tests/test_analytical", "tests/test_p0_", "tests/test_p1",
    "tests/test_migration_preregistered", "tests/test_time_axis",
    "tests/test_recogniser", "tests/analytical/", "tests/test_fabricated_population",
    "tests/test_chat_", "tests/test_period_change", "tests/test_routed",
)

#: Subsystems this migration does not touch.
OTHER = {
    "regulatory_watch": ("regulatory_watch",),
    "annex2 / xml delivery": ("annex2", "xml_builder", "delivery"),
    "simulation": ("simulation",),
    "pptx rendering": ("pptx",),
    "operations control": ("operations_control", "occ_"),
    "onboarding / engine": ("onboarding", "engine", "gate_"),
}


def classify(nodeid: str) -> str:
    path = nodeid.split("::", 1)[0]
    for fragment in MI_PATH:
        if fragment in path:
            return "MI QUERY AGENT PATH"
    for label, fragments in OTHER.items():
        if any(f in path for f in fragments):
            return label
    return "other / unclassified"


def introduced(head: Path, base: Optional[Path]) -> Dict[str, List[str]]:
    """The by-name diff A5 needs: what failing names HEAD has that base does not.

    Counts are useless for attribution — two runs can total the same and share
    nothing. Only the NAME SETS decide whether this work introduced a failure.
    """
    if base is None or not base.exists():
        return {}
    pattern = re.compile(r"^(?:FAILED|ERROR)\s+(\S+)", flags=re.M)
    head_names = set(pattern.findall(head.read_text(errors="ignore")))
    base_names = set(pattern.findall(base.read_text(errors="ignore")))
    return {"introduced": sorted(head_names - base_names),
            "resolved": sorted(base_names - head_names),
            "headCount": len(head_names), "baseCount": len(base_names)}


def main(argv: List[str]) -> int:
    capture = Path(argv[1]).resolve() if len(argv) > 1 else \
        (_REPO / "migration_phase0/estate_full.txt")
    text = capture.read_text(encoding="utf-8", errors="ignore")

    failures = re.findall(r"^(?:FAILED|ERROR)\s+(\S+)", text, flags=re.M)
    tally = re.search(
        r"^(\d+) failed,\s*(\d+) passed(?:,\s*(\d+) skipped)?"
        r"(?:,\s*(\d+) xfailed)?(?:.*?(\d+) errors?)?", text, flags=re.M)

    buckets: Dict[str, List[str]] = defaultdict(list)
    for nodeid in failures:
        buckets[classify(nodeid)].append(nodeid)

    print("=" * 78)
    print("ESTATE FAILURE ATTRIBUTION")
    print("=" * 78)
    if tally:
        print(f"\n  {tally.group(1)} failed, {tally.group(2)} passed, "
              f"{tally.group(3) or 0} skipped, {tally.group(4) or 0} xfailed, "
              f"{tally.group(5) or 0} errors")
    print(f"  {len(failures)} failing/erroring node ids captured\n")

    mi = buckets.get("MI QUERY AGENT PATH", [])
    for label in sorted(buckets, key=lambda k: -len(buckets[k])):
        marker = "  <-- THE MIGRATION SURFACE" if label == "MI QUERY AGENT PATH" else ""
        print(f"  {label:26s} {len(buckets[label]):5d}{marker}")

    print("\n" + "-" * 78)
    if mi:
        print(f"IN THE MI QUERY AGENT PATH — {len(mi)}. Every one, by name:\n")
        for nodeid in sorted(mi):
            print(f"  {nodeid}")
    else:
        print("IN THE MI QUERY AGENT PATH — NONE.")
    print("-" * 78)

    base_path = Path(argv[2]).resolve() if len(argv) > 2 else None
    diff = introduced(capture, base_path)
    if diff:
        print("\n" + "-" * 78)
        print(f"BY-NAME DIFF vs the clean base capture ({base_path})")
        print(f"  HEAD failing names : {diff['headCount']}")
        print(f"  BASE failing names : {diff['baseCount']}")
        print(f"  INTRODUCED         : {len(diff['introduced'])}"
              f"  <- required EMPTY")
        for name in diff["introduced"]:
            print(f"      {name}")
        print(f"  RESOLVED           : {len(diff['resolved'])}")
        for name in diff["resolved"]:
            print(f"      {name}")
        print("-" * 78)

    out = _REPO / "migration_phase0" / "estate.json"
    out.write_text(json.dumps({
        "capturedAtSha": subprocess.run(
            ["git", "-C", str(_REPO), "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True).stdout.strip(),
        "capture": (str(capture.relative_to(_REPO))
                    if capture.is_relative_to(_REPO) else str(capture)),
        "runner": ("pytest tests/ mi_agent/tests/ mi_agent_api/tests/ "
                   "-q -p no:randomly --tb=no -rf"),
        "tally": {
            "failed": int(tally.group(1)) if tally else None,
            "passed": int(tally.group(2)) if tally else None,
            "skipped": int(tally.group(3) or 0) if tally else None,
            "xfailed": int(tally.group(4) or 0) if tally else None,
            "errors": int(tally.group(5) or 0) if tally else None,
        },
        "by_subsystem": {k: len(v) for k, v in sorted(buckets.items())},
        "mi_query_agent_path_failures_by_name": sorted(mi),
        "all_failures_by_name": sorted(failures),
        "byNameDiffVsBase": diff,
        "note": ("Recorded at Phase 0 so A5 can compare BY NAME. A failure "
                 "present here is not the migration's; a failure absent here "
                 "that appears later is."),
    }, indent=2) + "\n", encoding="utf-8")
    print(f"\nwrote {out.relative_to(_REPO)}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
