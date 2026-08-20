"""Rebuild the client-readiness fixture pack and hash every artefact.

The pack is NOT committed. That follows the repository's own convention rather
than departing from it: the V1 funded tapes under ``demo_platform/workspace/``
are gitignored and reproduced from a committed generator, with only their hashes
carried in the evidence manifest. This pack is 211 MB raw (about 30 MB gzipped),
so committing the bytes would add a large derived artefact to the tree while
proving strictly less than a digest does — a digest fails if the generator drifts,
whereas a committed copy silently diverges from it.

Run this to reproduce and verify::

    python3 due_diligence/evidence/client_readiness/hash_fixture_pack.py --check

``--check`` rebuilds into a temporary root and compares against the committed
MANIFEST.json without writing anything; the default rewrites the manifest.

Determinism is the property that makes this work: two independent builds of the
pack produce the same pack digest, so a reviewer who runs this gets the same
bytes the measurements were taken on, or a failure that says which artefact moved.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Dict, List

_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parents[2]
sys.path.insert(0, str(_REPO_ROOT))

MANIFEST_PATH = _HERE / "MANIFEST.json"

#: The committed sources the pack is reproduced from. Hashed alongside the
#: artefacts: a manifest that hashes outputs without the code that produced them
#: cannot tell a fixture change from a generator change.
GENERATOR_SOURCES = (
    "tests/analytical/client_pack.py",
    "tests/analytical/client_fixture.py",
    "tests/analytical/client_funded.py",
    "tests/analytical/second_book.py",
    "config/client/pipeline_expected_funding.yaml",
)


#: Measurement instruments written or relocated during this sprint. Hashed here
#: rather than in the V1 manifest, and hashed at all because an instrument that
#: changes between two runs makes their comparison meaningless — the failure this
#: pack's own repair record describes.
INSTRUMENT_SOURCES = (
    "due_diligence/evidence/forecast_composition_hardening/tranche_d/three_axis_typed.py",
    "due_diligence/evidence/client_readiness/hash_fixture_pack.py",
)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _group(rel: str) -> str:
    parts = rel.split("/")
    if parts[0] == "processed":
        return f"fixture:funded:{parts[2]}"
    if len(parts) > 1 and parts[1] == "pipeline":
        return f"fixture:pipeline:{parts[0]}"
    return "fixture:other"


def _role(rel: str) -> str:
    parts = rel.split("/")
    if parts[0] == "processed":
        return f"funded {parts[-1].split('.')[0]} @ {parts[3]}"
    if len(parts) > 1 and parts[1] == "pipeline":
        return f"weekly pipeline extract @ {parts[2]}"
    return rel


def build_manifest(root: Path) -> Dict[str, object]:
    from tests.analytical import client_pack

    truth = client_pack.build(root)
    files = sorted(p for p in root.rglob("*") if p.is_file())

    pack = hashlib.sha256()
    artefacts: List[Dict[str, object]] = []
    for path in files:
        rel = str(path.relative_to(root))
        digest = _sha256(path)
        pack.update(rel.encode())
        pack.update(path.read_bytes())
        artefacts.append({"group": _group(rel), "path": rel,
                          "role": _role(rel), "bytes": path.stat().st_size,
                          "sha256": digest})

    sources = [{"group": "generator", "path": rel,
                "bytes": (_REPO_ROOT / rel).stat().st_size,
                "sha256": _sha256(_REPO_ROOT / rel)}
               for rel in GENERATOR_SOURCES]

    instruments = [{"group": "instrument", "path": rel,
                    "bytes": (_REPO_ROOT / rel).stat().st_size,
                    "sha256": _sha256(_REPO_ROOT / rel)}
                   for rel in INSTRUMENT_SOURCES
                   if (_REPO_ROOT / rel).exists()]

    try:
        head = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                              cwd=_REPO_ROOT, capture_output=True, text=True,
                              check=True).stdout.strip()
    except Exception:                       # pragma: no cover - not a git tree
        head = ""

    return {
        "pack": "client_readiness",
        "separateFrom": "due_diligence/evidence/MANIFEST.json",
        "generatedAtTree": head,
        "manifestNotes": {
            "immutableV1": "This pack is SEPARATE from the V1 manifest and shares no artefact with it. The V1 three-snapshot funded tapes, the twelve-week pipeline extracts and everything hashed against them are untouched; pipeline_fixture.py and the default path of second_book.build() still produce them byte-for-byte.",
            "notCommitted": "The artefacts below are reproduced from the generator, not stored. 211 MB raw / ~30 MB gzipped, and the repository already gitignores the V1 funded tapes on the same basis. A digest fails when the generator drifts; a committed copy silently diverges from it.",
            "supersetProperty": "The thirteen-month kestrelmoor funded history reproduces the V1 three cuts byte-for-byte. Any measurement that moves between the V1 fixture and this one is therefore attributable to the added history, not to a regenerated book.",
            "stressBook": "kestrelmoor carries a declining-KFI profile over the final thirteen weeks (intake 61 -> 21/week); alderbridge is stationary as the control.",
            "manifestRepair": "three_axis.py, hashed in the V1 manifest, was extended IN PLACE during Tranche D to add an answer-type axis. The manifest was not updated, so V1 verification failed from commit c9ac20b onward and was not noticed because nothing ran the verifier automatically. Repaired additively: three_axis.py is restored to its hashed content and verifies, the extended instrument is tranche_d/three_axis_typed.py, and tests/test_evidence_manifest.py now runs both verifications in the suite. The relocated instrument reproduces the recorded det_post figures exactly — 176 answer-type matches, 0 diverge, 740/752 semantic.",
            "groundTruth": "The pipeline generator's per-transition probabilities are chosen so the implied stage-to-funding rates equal the governed 0.20/0.45/0.75 exactly. They are recoverable from matured observations to within 0.001-0.027, which is what lets an empirical estimator be judged against truth rather than against itself.",
        },
        "packDigest": pack.hexdigest(),
        "totalBytes": sum(int(a["bytes"]) for a in artefacts),
        "artefactCount": len(artefacts),
        "truth": {
            "stressBook": truth["stress_book"],
            "controlBook": truth["control_book"],
            "reportingDates": truth["reporting_dates"],
            "lastPipelineWeek": truth["last_pipeline_week"],
            "funded": {c: t["snapshots"] for c, t in truth["funded"].items()},
        },
        "generatorSources": sources,
        "instrumentSources": instruments,
        "artefacts": artefacts,
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true",
                        help="verify against the committed manifest; write nothing")
    parser.add_argument("--root", help="build into this root instead of a temp dir")
    args = parser.parse_args(argv)

    root = Path(args.root) if args.root else Path(tempfile.mkdtemp(prefix="client_pack_"))
    try:
        built = build_manifest(root)
        if not args.check:
            MANIFEST_PATH.write_text(json.dumps(built, indent=2) + "\n",
                                     encoding="utf-8")
            print(f"wrote {MANIFEST_PATH.relative_to(_REPO_ROOT)}: "
                  f"{built['artefactCount']} artefacts, "
                  f"{built['totalBytes']:,} bytes, digest {built['packDigest']}")
            return 0

        committed = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
        problems = []
        if committed.get("packDigest") != built["packDigest"]:
            problems.append(f"pack digest {committed.get('packDigest')} "
                            f"!= rebuilt {built['packDigest']}")
        was = {a["path"]: a["sha256"] for a in committed.get("artefacts", [])}
        now = {a["path"]: a["sha256"] for a in built["artefacts"]}
        for path in sorted(set(was) | set(now)):
            if was.get(path) != now.get(path):
                problems.append(f"artefact differs: {path}")
        for src in built["generatorSources"]:
            prior = next((s for s in committed.get("generatorSources", [])
                          if s["path"] == src["path"]), None)
            if prior and prior["sha256"] != src["sha256"]:
                problems.append(f"generator changed since the manifest: {src['path']}")
        if problems:
            for p in problems[:20]:
                print("FAIL:", p)
            if len(problems) > 20:
                print(f"... and {len(problems) - 20} more")
            return 1
        print(f"OK: {built['artefactCount']} artefacts reproduce exactly "
              f"(digest {built['packDigest']})")
        return 0
    finally:
        if not args.root:
            shutil.rmtree(root, ignore_errors=True)


if __name__ == "__main__":                  # pragma: no cover - CLI
    raise SystemExit(main())
