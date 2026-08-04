"""Tests for the deployment-package verifier.

Runnable with pytest, or standalone (`python3 test_verify_package.py`), matching
the convention in ``test_verify_flex_workflow.py`` so CI can call either.

The cases are written from the real omissions rather than from the checklist.
``operations_control/`` shipped missing once and only surfaced on the first blob
arrival; ``trakt_notifications/`` and ``trakt_core/`` would have shipped missing
the same way and been quieter still, because the timer trigger swallows the
ImportError and logs it. A verifier that passed a package without them would
have been worth nothing, so each is asserted to FAIL rather than merely
asserting the good package passes.
"""

from __future__ import annotations

import io
import json
import pathlib
import sys
import zipfile
from contextlib import redirect_stdout

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import verify_package as vp  # noqa: E402

REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]

#: The top-level paths the deployment workflow's zip step ships. Kept here so a
#: divergence between this test and the workflow is visible in one diff.
PACKAGED_TREES = (
    "apps", "engine", "operations_control", "mi_agent", "mi_agent_api",
    "mi_agent_pptx", "analytics_lib", "trakt_notifications", "trakt_core",
    "config",
)
PACKAGED_FILES = ("function_app.py", "host.json", "requirements.txt")


def build_package(dest: pathlib.Path, *,
                  omit: tuple[str, ...] = ()) -> pathlib.Path:
    """Build an archive shaped like the workflow's, optionally omitting trees.

    Only the files the manifest check looks at are written, so a case runs in
    milliseconds instead of zipping the repository.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(dest, "w") as zf:
        for name in PACKAGED_FILES:
            source = REPO_ROOT / name
            zf.writestr(name, source.read_text(encoding="utf-8")
                        if source.exists() else "")
        for tree in PACKAGED_TREES:
            if tree in omit:
                continue
            for required in vp.REQUIRED_PATHS:
                if required.startswith(f"{tree}/"):
                    zf.writestr(required, "")
    return dest


def run_manifest(zip_path: pathlib.Path) -> tuple[int, str]:
    """Run the manifest check, capturing its exit code and output."""
    buffer = io.StringIO()
    code = 0
    try:
        with redirect_stdout(buffer):
            vp.check_manifest(zip_path)
    except SystemExit as exit_code:
        code = int(exit_code.code or 0)
    return code, buffer.getvalue()


# --------------------------------------------------------------------------- #
# The good package
# --------------------------------------------------------------------------- #
def test_a_complete_package_passes(tmp_path):
    code, output = run_manifest(build_package(tmp_path / "deploy.zip"))
    assert code == 0, output


# --------------------------------------------------------------------------- #
# The notification worker — the omission this verifier was extended for
# --------------------------------------------------------------------------- #
def test_omitting_trakt_notifications_fails_verification(tmp_path):
    """Without it the timer registers, fires, logs and delivers nothing."""
    code, output = run_manifest(
        build_package(tmp_path / "deploy.zip", omit=("trakt_notifications",)))
    assert code != 0, "a package with no delivery worker must not verify"
    assert "trakt_notifications/delivery.py" in output


def test_omitting_trakt_core_fails_verification(tmp_path):
    """mi_agent_api.insight_engine imports trakt_core.perf at module scope, so
    the worker cannot resolve governed MI without it."""
    code, output = run_manifest(
        build_package(tmp_path / "deploy.zip", omit=("trakt_core",)))
    assert code != 0, "a package with no trakt_core must not verify"
    assert "trakt_core/perf.py" in output


def test_omitting_both_names_both(tmp_path):
    code, output = run_manifest(build_package(
        tmp_path / "deploy.zip", omit=("trakt_notifications", "trakt_core")))
    assert code != 0
    assert "trakt_notifications/__init__.py" in output
    assert "trakt_core/__init__.py" in output


# --------------------------------------------------------------------------- #
# The existing OCC verification must survive unchanged
# --------------------------------------------------------------------------- #
def test_omitting_operations_control_still_fails(tmp_path):
    code, output = run_manifest(
        build_package(tmp_path / "deploy.zip", omit=("operations_control",)))
    assert code != 0
    assert "operations_control/engine.py" in output


def test_omitting_the_blob_trigger_app_still_fails(tmp_path):
    code, output = run_manifest(
        build_package(tmp_path / "deploy.zip", omit=("apps",)))
    assert code != 0
    assert "apps/blob_trigger_app/occ_intake.py" in output


def test_the_occ_required_paths_are_all_still_declared():
    """A regression guard on the guard: extending REQUIRED_PATHS must not have
    displaced anything the OCC chain depends on."""
    for path in ("function_app.py", "host.json", "requirements.txt",
                 "apps/blob_trigger_app/__init__.py",
                 "apps/blob_trigger_app/occ_intake.py",
                 "engine/__init__.py",
                 "operations_control/__init__.py",
                 "operations_control/engine.py",
                 "operations_control/stores.py",
                 "mi_agent_api/__init__.py"):
        assert path in vp.REQUIRED_PATHS


# --------------------------------------------------------------------------- #
# The import probe
# --------------------------------------------------------------------------- #
def test_the_probe_exercises_both_trigger_chains():
    """`import function_app` succeeds with either package missing, which is
    exactly why the probe has to reach past it."""
    probe = vp.IMPORT_PROBE
    # OCC chain (Event Grid trigger).
    assert "from apps.blob_trigger_app import occ_intake" in probe
    assert "from operations_control.engine import OpsEngine" in probe
    # Notification chain (timer trigger).
    assert "from trakt_notifications.delivery import run_once" in probe
    assert "from trakt_notifications.trigger import on_publication_approved" in probe
    assert "import trakt_core.perf" in probe
    assert "from mi_agent_api import insight_engine" in probe


def test_the_workflow_ships_every_tree_the_verifier_requires():
    """The zip step and REQUIRED_PATHS must not drift apart — that divergence IS
    the defect class this file exists to prevent."""
    workflow = (REPO_ROOT / ".github" / "workflows"
                / "main_trakt-blob-trigger-v2.yml").read_text(encoding="utf-8")
    trees = {path.split("/")[0] for path in vp.REQUIRED_PATHS if "/" in path}
    for tree in sorted(trees):
        assert f"{tree}/ \\" in workflow, (
            f"{tree}/ is required by verify_package.py but the deployment "
            f"workflow's zip step does not ship it")


if __name__ == "__main__":
    import tempfile

    failures = 0
    names = sorted(n for n in list(globals()) if n.startswith("test_"))
    for name in names:
        fn = globals()[name]
        try:
            if "tmp_path" in fn.__code__.co_varnames[:fn.__code__.co_argcount]:
                with tempfile.TemporaryDirectory() as td:
                    fn(pathlib.Path(td))
            else:
                fn()
        except AssertionError as exc:
            failures += 1
            print(f"FAIL: {name}\n      {exc}")
        else:
            print(f"pass: {name}")
    print(f"\npassed: {len(names) - failures}   failed: {failures}")
    sys.exit(1 if failures else 0)
