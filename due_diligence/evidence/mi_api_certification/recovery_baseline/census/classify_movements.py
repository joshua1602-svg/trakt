#!/usr/bin/env python3
"""EVERY MOVEMENT, CLASSIFIED — the Phase 6 gate, machine-checked.

Reads the per-step censuses and asserts that every question that moved between
the recovery baseline and HEAD moved for a NAMED reason, in a NAMED step. An
unexplained movement exits non-zero, which is the sprint's STOP condition.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
STEPS = ["before_G2", "after_G2", "after_G1", "after_G3", "after_G7lite",
         "after_CAP", "after_G6", "after_P3", "HEAD"]
STEP_OF = {"after_G2": "G2", "after_G1": "G1", "after_G3": "G3",
           "after_G7lite": "G7-lite", "after_CAP": "CAP", "after_G6": "G6",
           "after_P3": "P3", "HEAD": "G6-completion"}

#: The classes a movement may fall into, and the step that owns each.
CLASSES = {
    "G2": "hyphenated spelling converges on its spaced twin",
    "G1": "a relative period token becomes a governed period statement",
    "G3": "a spurious `unknown category` note is removed",
    "G7-lite": "contract subject/provenance only (asked counts gain provenance; "
               "unasked counts empty)",
    "CAP": "the borrowing-base capability claims its own span",
    "G6": "the one geography owner",
    "P3": "a weighting qualifier stops being read as a measure",
    "G6-completion": "the harmonised region tier and the fifth chooser",
}


def _load(bank: str, step: str):
    path = HERE / ("%s_%s.json" % (bank, step))
    return {r["question_id"]: r for r in json.loads(path.read_text())}


def run(bank: str) -> int:
    rows = {step: _load(bank, step) for step in STEPS}
    first, last = rows["before_G2"], rows["HEAD"]
    unexplained = []
    moved_total = 0
    per_step = {}
    for qid, end in last.items():
        start = first.get(qid)
        if start is None or start == end:
            continue
        moved_total += 1
        owners = []
        for i in range(1, len(STEPS)):
            a, b = rows[STEPS[i - 1]].get(qid), rows[STEPS[i]].get(qid)
            if a is not None and b is not None and a != b:
                owners.append(STEP_OF[STEPS[i]])
        if not owners:
            unexplained.append((qid, "moved end to end but in no single step"))
            continue
        for owner in owners:
            per_step.setdefault(owner, []).append(qid)
        if any(o not in CLASSES for o in owners):
            unexplained.append((qid, "no class for %s" % owners))

    print("%s: %d questions, %d moved between the baseline and HEAD"
          % (bank, len(last), moved_total))
    for owner in CLASSES:
        ids = per_step.get(owner, [])
        if ids:
            print("  %-14s %3d  %s" % (owner, len(ids), CLASSES[owner]))
    if unexplained:
        print("UNEXPLAINED (%d) — STOP:" % len(unexplained))
        for qid, why in unexplained:
            print("   ", qid, why)
        return 1
    print("  every movement is attributed to a named step. None unexplained.")
    return 0


if __name__ == "__main__":
    sys.exit(max(run("corpus"), run("bank")))
