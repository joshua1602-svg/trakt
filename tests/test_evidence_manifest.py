"""The evidence manifest is verified by the test suite, not only by hand.

Why this exists
---------------
MANIFEST.json hashes every evidence input, so an artefact that changes after it
was measured is detectable. Nothing ran that check automatically, and the gap
showed: three_axis.py — a hashed instrument — was extended in place during
Tranche D to add a fourth axis, and manifest verification failed silently for
the rest of the tranche because nobody re-ran the verifier. The repair was to
restore the hashed file and put the extended instrument in a new one; this test
is the part that stops it recurring.

It SKIPS rather than fails when the fixture inputs are absent. Those artefacts
are gitignored and reproduced from their generators, so a fresh clone legitimately
does not have them — but a tree that does have them must verify.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[1]
_VERIFIER = _REPO / "due_diligence" / "evidence" / "verify_manifest.py"

#: One artefact per generated fixture family. If these are absent the tree has
#: not materialised its fixtures and there is nothing to verify.
_FIXTURE_PROBES = (
    _REPO / "demo_platform" / "workspace" / "store" / "processed" / "platform"
    / "alderbridge" / "2026-06-30" / "platform_canonical_typed.csv",
)


@pytest.mark.skipif(not _VERIFIER.exists(), reason="evidence pack not present")
def test_every_evidence_input_matches_the_manifest():
    if not all(p.exists() for p in _FIXTURE_PROBES):
        pytest.skip("generated fixtures not materialised in this tree")
    proc = subprocess.run([sys.executable, str(_VERIFIER)],
                          capture_output=True, text=True, cwd=str(_REPO))
    assert proc.returncode == 0, (
        "evidence manifest verification failed — an artefact changed after it "
        "was measured, or a run file names an unpinned harness:\n"
        + proc.stdout + proc.stderr)


def _run_verifier(manifest=None):
    env = dict(**__import__("os").environ)
    if manifest is not None:
        env["TRAKT_EVIDENCE_MANIFEST"] = str(manifest)
    return subprocess.run([sys.executable, str(_VERIFIER)],
                          capture_output=True, text=True, cwd=str(_REPO), env=env)


@pytest.mark.skipif(not _VERIFIER.exists(), reason="evidence pack not present")
def test_the_verifier_fails_when_an_evidence_artefact_drifts(tmp_path):
    """The control must be provably able to fail.

    Tranche D edited a hashed instrument in place and manifest verification
    failed silently for a whole tranche, because nothing ran it. Adding it to
    the suite closes that. This closes the other half: a verifier nobody has
    ever seen fail is not evidence that nothing has drifted.
    """
    import json
    if not all(p.exists() for p in _FIXTURE_PROBES):
        pytest.skip("generated fixtures not materialised in this tree")
    manifest = json.loads(
        (_REPO / "due_diligence" / "evidence" / "MANIFEST.json").read_text())
    victim = next(a for a in manifest["artefacts"] if a["group"] != "code+config")
    victim["sha256"] = "0" * 64
    corrupted = tmp_path / "MANIFEST.json"
    corrupted.write_text(json.dumps(manifest))
    proc = _run_verifier(corrupted)
    assert proc.returncode != 0, "a drifted EVIDENCE artefact must be fatal"
    assert victim["path"] in proc.stdout


@pytest.mark.skipif(not _VERIFIER.exists(), reason="evidence pack not present")
def test_production_code_moving_on_is_reported_but_not_fatal():
    """Code drift is provenance, not corruption.

    The code+config group records which production code the V1 measurement ran
    against. Production code is what this programme exists to change, so failing
    on it made the verifier cry wolf on ordinary work — which is precisely how
    its one real alarm was lost.
    """
    if not all(p.exists() for p in _FIXTURE_PROBES):
        pytest.skip("generated fixtures not materialised in this tree")
    proc = _run_verifier()
    assert proc.returncode == 0
    if "code moved on since" in proc.stdout:
        # Drift is reported, and reported as RE-PINNED — never as an open-ended
        # "expected" state, which would soften the property being verified.
        assert "RE-PINNED" in proc.stdout
        assert "not re-baselined" in proc.stdout


@pytest.mark.skipif(not _VERIFIER.exists(), reason="evidence pack not present")
def test_production_code_that_moved_and_is_not_re_pinned_is_fatal(tmp_path):
    """Drift is allowed; becoming UNPINNED is not.

    A V1 code+config file must be byte-identical in the working tree, or pinned
    at its current state in the successor manifest. A permanent "expected drift"
    state would soften the property that makes a manifest evidence at all — the
    same shape as the stale manifest that failed silently for a fortnight.

    Simulated by pointing the verifier at an EMPTY successor manifest, so the
    files that legitimately moved this sprint have nowhere to be pinned.
    """
    import json
    import os
    if not all(p.exists() for p in _FIXTURE_PROBES):
        pytest.skip("generated fixtures not materialised in this tree")
    empty = tmp_path / "successor.json"
    empty.write_text(json.dumps({"productionSources": []}))
    env = dict(os.environ, TRAKT_SUCCESSOR_MANIFEST=str(empty))
    proc = subprocess.run([sys.executable, str(_VERIFIER)],
                          capture_output=True, text=True, cwd=str(_REPO), env=env)
    if "code moved on since" not in _run_verifier().stdout:
        pytest.skip("no production code has moved on in this tree")
    assert proc.returncode != 0, "unpinned production code must be fatal"
    assert "UNPINNED CODE" in proc.stdout


@pytest.mark.skipif(not _VERIFIER.exists(), reason="evidence pack not present")
def test_the_client_readiness_pack_reproduces_from_its_generator():
    """The client-readiness fixture is not committed; its digest is.

    Rebuilding it must give the same bytes, or every Tranche E/F/G measurement
    was taken on a fixture nobody else can reproduce.
    """
    hasher = (_REPO / "due_diligence" / "evidence" / "client_readiness"
              / "hash_fixture_pack.py")
    if not hasher.exists():
        pytest.skip("client-readiness pack not present")
    proc = subprocess.run([sys.executable, str(hasher), "--check"],
                          capture_output=True, text=True, cwd=str(_REPO))
    assert proc.returncode == 0, (
        "the client-readiness fixture no longer reproduces from its "
        "generator:\n" + proc.stdout + proc.stderr)
