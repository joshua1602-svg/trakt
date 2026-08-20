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
import os
#: Overridable so the control can be TESTED — a verifier nothing can prove
#: wrong is not a control. tests/test_evidence_manifest.py points this at a
#: deliberately corrupted manifest and requires a non-zero exit.
_MANIFEST_PATH = Path(os.environ.get("TRAKT_EVIDENCE_MANIFEST")
                      or (EV / "MANIFEST.json"))
M = json.loads(_MANIFEST_PATH.read_text())

#: Artefacts live either in the repo or in the measurement scratch. Both roots
#: are tried; a file found in neither is a failure, not a skip.
ROOTS = [EV, REPO, S]


def locate(rel: str) -> Path | None:
    for root in ROOTS:
        p = root / rel
        if p.exists():
            return p
    return None


#: EVIDENCE vs CODE PROVENANCE — two things this used to conflate.
#:
#: Most artefacts here are EVIDENCE: fixtures, run files, frozen expectations,
#: measurement instruments. They must never change after a measurement was taken
#: against them, and drift is a hard failure.
#:
#: The ``code+config`` group is different. It records WHICH PRODUCTION CODE the
#: manifest was generated against — all fifteen entries hash to tree
#: ``generatedFromTree``. Production code is what this programme exists to
#: change, so every legitimate code change made this verifier fail. A control
#: that cries wolf on ordinary work stops being run, and that is not a
#: hypothetical: it is why the real breakage of Tranche D — an EVIDENCE artefact
#: edited in place — sat unnoticed for a whole tranche.
#:
#: So code drift is now classified, not failed: reported as INFO against the
#: recorded tree, with the manifest left untouched. Evidence drift stays fatal.
CODE_GROUP = "code+config"


def git_blob(ref: str, rel: str) -> bytes | None:
    import subprocess
    try:
        return subprocess.run(["git", "show", f"{ref}:{rel}"], cwd=REPO,
                              capture_output=True, check=True).stdout
    except Exception:
        return None


failures, checked, absent = [], 0, []
code_drift: list[str] = []
for a in M["artefacts"]:
    if a.get("status") == "MISSING":
        absent.append(a["path"]); continue
    p = locate(a["path"])
    if p is None:
        failures.append(f"NOT FOUND        {a['path']}"); continue
    if a["group"] == CODE_GROUP:
        # Provenance, not evidence. Confirm the RECORDED TREE still carries the
        # bytes the manifest names — that is the claim the manifest actually
        # makes — and report any working-tree divergence separately.
        ref = M.get("generatedFromTree") or M.get("reportHead") or "HEAD"
        blob = git_blob(ref, a["path"])
        if blob is None:
            failures.append(f"NOT IN TREE      {a['path']} at {ref[:7]}")
            continue
        if hashlib.sha256(blob).hexdigest() != a["sha256"]:
            failures.append(f"PROVENANCE BROKEN {a['path']}: the recorded tree "
                            f"{ref[:7]} no longer carries the hashed bytes")
            continue
        checked += 1
        if hashlib.sha256(p.read_bytes()).hexdigest() != a["sha256"]:
            code_drift.append(a["path"])
        continue
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
if code_drift:
    ref = (M.get("generatedFromTree") or "")[:7]
    print(f"\ncode moved on since {ref} ({len(code_drift)}) — expected, not a "
          f"failure. The manifest records which code the V1 measurement ran "
          f"against; it does not freeze the product:")
    for path in code_drift:
        print(f"  {path}")
if failures:
    print(f"\nFAILURES ({len(failures)}):")
    for f in failures:
        print("  " + f)
    sys.exit(1)
print("\nOK — every evidence input matches the manifest, every code+config entry "
      "still verifies against the tree it was recorded at, and every run file "
      "names a pinned harness.")
