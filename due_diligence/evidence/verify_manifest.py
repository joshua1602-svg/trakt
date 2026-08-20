#!/usr/bin/env python3
"""Verify every evidence input against MANIFEST.json. Exit 1 on any mismatch.

    python3 due_diligence/evidence/verify_manifest.py

Checks that each artefact exists and hashes to the recorded sha256, and that the
standing provenance rule holds: every run file names the harness revision that
produced it, and that revision is itself pinned in the manifest.
"""
from __future__ import annotations
import gzip, hashlib, json, sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent.parent
EV = REPO / "due_diligence/evidence"
S = Path("/tmp/claude-0/-home-user-trakt/8fa44461-4d82-5a00-b9a3-1df72087ab19/scratchpad")
M = json.loads((EV / "MANIFEST.json").read_text())

#: Artefacts live either in the repo or in the measurement scratch. Both roots
#: are tried; a file found in neither is a failure, not a skip.
ROOTS = [EV, REPO, S]


def locate(rel: str) -> Path | None:
    for root in ROOTS:
        p = root / rel
        if p.exists():
            return p
    return None


failures, checked, absent = [], 0, []
for a in M["artefacts"]:
    if a.get("status") == "MISSING":
        absent.append(a["path"]); continue
    p = locate(a["path"])
    if p is None:
        failures.append(f"NOT FOUND        {a['path']}"); continue
    raw = p.read_bytes()
    if a.get("gzSha256") is not None:
        # Run files are stored gzipped. The manifest's sha256 is of the
        # CONTENT, because gzip output is not byte-stable across
        # implementations and a content hash is what a reader wants to check.
        # The compressed bytes are pinned separately so the stored file is
        # covered too.
        got_gz = hashlib.sha256(raw).hexdigest()
        if got_gz != a["gzSha256"]:
            failures.append(f"GZ HASH MISMATCH {a['path']}\n"
                            f"                 manifest {a['gzSha256'][:24]}…\n"
                            f"                 on disk  {got_gz[:24]}…")
        raw = gzip.decompress(raw)
    got = hashlib.sha256(raw).hexdigest()
    checked += 1
    if got != a["sha256"]:
        failures.append(f"HASH MISMATCH    {a['path']}\n"
                        f"                 manifest {a['sha256'][:24]}…\n"
                        f"                 on disk  {got[:24]}…")

# The standing rule: a run file must name its producing harness, and that
# harness must itself be pinned somewhere in the manifest.
pinned = {a["sha256"] for a in M["artefacts"] if a.get("sha256")}
for a in M["artefacts"]:
    if not a["group"].startswith("run-file"):
        continue
    producer = a.get("producedByHarnessSha256")
    if not producer:
        failures.append(f"NO PRODUCER      {a['path']} does not name its harness")
    elif producer not in pinned:
        failures.append(f"UNPINNED HARNESS {a['path']} names harness "
                        f"{producer[:16]}… which is not pinned in the manifest")

print(f"manifest      : {M['reportHead']}  ({len(M['artefacts'])} artefacts)")
print(f"hash-verified : {checked}")
if absent:
    print(f"recorded missing: {len(absent)}")
if failures:
    print(f"\nFAILURES ({len(failures)}):")
    for f in failures:
        print("  " + f)
    sys.exit(1)
print("\nOK — every evidence input matches the manifest, and every run file "
      "names a pinned harness.")
