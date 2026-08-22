#!/usr/bin/env python3
"""Everything `npm run build` needs must actually be in the repository.

This exists because of a real failure. `package.json` gained

    "build": "tsc -b && vite build && node scripts/check-bundle-leaks.mjs dist"

while the script it names was matched by a defensive `.gitignore` rule
(`*secret*`, alongside `*key*` and `*cred*`) under its original filename. It ran
locally, `git add -A` silently skipped it, and the build broke on the first clean
checkout — which is Oryx, inside the Static Web Apps deployment, on main:

    Error: Cannot find module
      '/github/workspace/frontend/mi-agent-ui/scripts/check-bundle-leaks.mjs'

An ignored file is invisible in exactly the places you look: it is present in the
working tree, absent from `git status`, and absent from the clone CI makes. So
the check is not "does this file exist" — it is "does git have it".

The `.gitignore` rules are good and stay. Files the build needs are named so they
do not trip them; `!snapshot/keys.py` is the precedent for the alternative when a
name cannot be changed.
"""

from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
_UI = _REPO_ROOT / "frontend" / "mi-agent-ui"
PACKAGE_JSON = _UI / "package.json"


def _scripts() -> dict:
    return json.loads(PACKAGE_JSON.read_text(encoding="utf-8")).get("scripts", {})


def _referenced_files() -> list[tuple[str, str]]:
    """(script name, repo-relative path) for every local file an npm script runs.

    Only `node <path>` forms: `tsc`, `vite` and `vitest` are resolved from
    node_modules by npm, not from the working tree.
    """
    found = []
    for name, body in _scripts().items():
        for match in re.finditer(r"node\s+([\w./-]+\.(?:mjs|cjs|js))", body or ""):
            found.append((name, f"frontend/mi-agent-ui/{match.group(1)}"))
    return found


def _tracked(path: str) -> bool:
    result = subprocess.run(
        ["git", "ls-files", "--error-unmatch", path],
        cwd=_REPO_ROOT, capture_output=True, text=True)
    return result.returncode == 0


def test_the_build_script_still_references_a_local_script():
    # Guards the tests below from passing vacuously if the reference is removed.
    assert _referenced_files(), (
        "no `node <file>` reference found in package.json scripts — if the "
        "bundle scan was removed deliberately, remove this file too")


@pytest.mark.parametrize("script_name,path", _referenced_files())
def test_every_script_the_build_runs_exists(script_name, path):
    assert (_REPO_ROOT / path).is_file(), (
        f"npm script {script_name!r} runs {path}, which is not on disk")


@pytest.mark.parametrize("script_name,path", _referenced_files())
def test_every_script_the_build_runs_is_committed(script_name, path):
    assert _tracked(path), (
        f"npm script {script_name!r} runs {path}, which git does NOT track. It "
        "works here and fails on every clean checkout — including the Oryx "
        "build inside the Static Web Apps deployment. Check `git check-ignore "
        f"-v {path}`: the defensive *secret*/*key*/*cred* rules in .gitignore "
        "catch innocent filenames, and `git add -A` skips them without a word")


def test_no_file_the_build_needs_is_gitignored(script_name="build"):
    """The same fault stated the other way round, so the message names the rule.

    A file can be tracked AND ignored (added before the rule existed), which
    keeps CI working today and breaks it the moment someone re-adds the file.
    """
    for name, path in _referenced_files():
        result = subprocess.run(["git", "check-ignore", "-v", path],
                                cwd=_REPO_ROOT, capture_output=True, text=True)
        assert result.returncode != 0, (
            f"npm script {name!r} runs {path}, which .gitignore matches: "
            f"{result.stdout.strip()}. Rename the file so it does not trip the "
            "rule, or add a negation next to `!snapshot/keys.py`")
