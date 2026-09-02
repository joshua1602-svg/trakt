#!/usr/bin/env python3
"""tests/test_mi_api_appservice_packaging — guard the App Service artefact.

The container image gets its packaging guarantee from a build-time smoke import
(``deploy/trakt-mi-api/Dockerfile``), which is what once caught a missing
``snapshot``. The App Service deployment had no equivalent: it uploaded the whole
checkout, so nothing could ever be *missing* — and the price was that Oryx read
the repo-root requirements.txt as the runtime contract and installed scikit-learn,
scipy, streamlit and the rest on every deploy.

Now that the deployment stages a narrow artefact, both halves of that trade need
a guard:

  * **completeness** — every repo package the API imports must be in
    ``package_contents.txt``, or the App Service starts and 500s on the first
    request. The list is checked against the *recomputed* import closure of
    ``mi_agent_api.app``, so adding an import to the API without staging its
    package fails here.
  * **sufficiency** — every third-party distribution reachable from that closure
    must be in ``deploy/trakt-mi-api/requirements.txt``, or the App Service build
    succeeds and the app cannot import.
  * **economy** — nothing ships that the runtime never opens. The two checks
    above both ask "is everything NEEDED present?"; neither asks "is everything
    PRESENT needed?", and that blind spot let 14.5 MB of frozen measurement
    baselines into a 28.0 MB artefact, unnoticed, because the package list names
    directories and evidence is committed beside the code it measures. The
    economy checks read ``package_excludes.txt`` and hold BOTH directions: the
    declared files must not ship, and — the safety half — nothing declared may be
    something the staged code actually reads.

Run: python -m pytest tests/test_mi_api_appservice_packaging.py
"""

from __future__ import annotations

import ast
import os
import re
import subprocess
import sys
import unittest
from pathlib import Path
from typing import Dict, Set

_REPO = Path(__file__).resolve().parents[1]
_MANIFEST = _REPO / "deploy" / "trakt-mi-api" / "package_contents.txt"
_REQUIREMENTS = _REPO / "deploy" / "trakt-mi-api" / "requirements.txt"
_EXCLUDES = _REPO / "deploy" / "trakt-mi-api" / "package_excludes.txt"
_WORKFLOW = _REPO / ".github" / "workflows" / "deploy-mi-api.yml"

#: Third-party import name -> the distribution that provides it. Only the names
#: the closure can actually produce need an entry.
_IMPORT_TO_DISTRIBUTION = {
    "fastapi": "fastapi",
    "uvicorn": "uvicorn",
    "uvicorn_worker": "uvicorn-worker",
    "gunicorn": "gunicorn",
    "jwt": "PyJWT",
    "pandas": "pandas",
    "numpy": "numpy",
    "yaml": "PyYAML",
    "openpyxl": "openpyxl",
    "azure": "azure-storage-blob",
    "plotly": "plotly",
    "pptx": "python-pptx",
    "matplotlib": "matplotlib",
    "anthropic": "anthropic",
    "pydantic": "pydantic",      # arrives with fastapi
    "starlette": "fastapi",
    "kaleido": None,             # guarded optional import (mi_chart_factory)
}


def _manifest_paths() -> list:
    """The staged paths, parsed exactly as the workflow parses them."""
    out = []
    for raw in _MANIFEST.read_text(encoding="utf-8").splitlines():
        path = raw.split("#", 1)[0].strip()
        if path:
            out.append(path)
    return out


def _import_closure() -> Dict[str, Set[str]]:
    """Repo modules and third-party imports reachable from ``mi_agent_api.app``.

    Seeded from a real import (so conditional module-level wiring is included),
    then walked statically so imports inside functions — the deck-generation
    route reaching ``apps.blob_trigger_app.pptx_stage``, the blob storage client,
    the LLM parser — are followed too. Returns {"repo": {...}, "third": {...}}.
    """
    env = dict(os.environ, TRAKT_RUNTIME_MODE="test", PYTHONPATH=str(_REPO))
    script = r"""
import ast, json, os, sys
os.environ.setdefault("TRAKT_RUNTIME_MODE", "test")
import mi_agent_api.app  # noqa: F401  - seed the closure with a real import

root = os.getcwd()
stdlib = set(sys.stdlib_module_names)
repo_tops = {d for d in os.listdir(root)
             if os.path.isdir(d) and os.path.exists(os.path.join(d, "__init__.py"))}

seed = set()
for mod in list(sys.modules.values()):
    f = getattr(mod, "__file__", None)
    if f and os.path.abspath(f).startswith(root + os.sep):
        seed.add(os.path.relpath(f, root))

def resolve(name):
    parts = name.split(".")
    if parts[0] not in repo_tops:
        return []
    out = []
    for candidate in (parts, parts[:-1]):
        if not candidate:
            continue
        p = os.path.join(*candidate)
        if os.path.isfile(p + ".py"):
            out.append(p + ".py")
        if os.path.isfile(os.path.join(p, "__init__.py")):
            out.append(os.path.join(p, "__init__.py"))
        if out:
            break
    return out

frontier, seen, third = list(seed), set(seed), set()
while frontier:
    path = frontier.pop()
    try:
        tree = ast.parse(open(path, encoding="utf-8").read())
    except Exception:
        continue
    pkg = os.path.dirname(path).replace(os.sep, ".")
    for node in ast.walk(tree):
        names = []
        if isinstance(node, ast.Import):
            names = [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                names = [f"{pkg}.{node.module}" if node.module else pkg]
            else:
                names = [node.module or ""]
        for name in names:
            top = name.split(".")[0]
            if not top:
                continue
            if top in repo_tops:
                for f in resolve(name):
                    if f not in seen:
                        seen.add(f)
                        frontier.append(f)
            elif top not in stdlib:
                third.add(top)

print(json.dumps({
    "repo": sorted({p.split(os.sep)[0] for p in seen}),
    "third": sorted(third),
}))
"""
    proc = subprocess.run([sys.executable, "-c", script], cwd=str(_REPO),
                          capture_output=True, text=True, env=env)
    if proc.returncode != 0:
        raise AssertionError(f"import closure failed:\n{proc.stderr[-3000:]}")
    import json
    payload = json.loads(proc.stdout.strip().splitlines()[-1])
    return {"repo": set(payload["repo"]), "third": set(payload["third"])}


def _requirement_names() -> Set[str]:
    """Distribution names in the artefact requirements, lower-cased.

    ``PyJWT[crypto]>=2.8`` -> ``pyjwt``; comment lines (including the
    deliberately-excluded list) are ignored.
    """
    names = set()
    for raw in _REQUIREMENTS.read_text(encoding="utf-8").splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        name = re.split(r"[<>=!\[;]", line, 1)[0].strip()
        if name:
            names.add(name.lower())
    return names


class TestArtefactCompleteness(unittest.TestCase):
    """Everything the API imports must be staged."""

    def test_every_reachable_repo_package_is_staged(self):
        staged = {p.split("/")[0] for p in _manifest_paths()}
        missing = sorted(_import_closure()["repo"] - staged)
        self.assertEqual(missing, [], (
            "package_contents.txt does not stage repo package(s) the MI API "
            f"imports: {missing}. The App Service would start and fail on the "
            "first request that reaches them."))

    def test_startup_script_is_staged_at_the_artefact_root(self):
        """`bash startup.sh` resolves against /home/site/wwwroot, so the file has
        to be at the artefact root — not nested under a directory."""
        self.assertIn("startup.sh", _manifest_paths())

    def test_every_staged_path_exists(self):
        missing = [p for p in _manifest_paths() if not (_REPO / p).exists()]
        self.assertEqual(missing, [], f"staged path(s) do not exist: {missing}")

    def test_configuration_data_is_staged(self):
        """config/ is read at runtime (semantics, tenancy, organisations,
        principals, entitlements); configs/pptx is the deck definition."""
        staged = set(_manifest_paths())
        self.assertIn("config", staged)
        self.assertIn("configs", staged)


class TestRequirementsSufficiency(unittest.TestCase):
    """Everything the API imports must be installable from the contract."""

    def test_the_startup_command_can_run(self):
        """startup.sh execs gunicorn with a uvicorn worker class. Missing either
        is `command not found` — exit code 127 — at container start."""
        names = _requirement_names()
        for required in ("gunicorn", "uvicorn", "uvicorn-worker", "fastapi"):
            self.assertIn(required, names,
                          f"{required} missing from the App Service contract")

    def test_token_validation_has_its_crypto_backend(self):
        """PyJWT without [crypto] cannot verify RS256, so every Entra token check
        fails closed. The extra must survive edits to this file."""
        text = _REQUIREMENTS.read_text(encoding="utf-8")
        self.assertRegex(text, r"(?im)^\s*PyJWT\[crypto\]")

    def test_every_reachable_distribution_is_declared(self):
        names = _requirement_names()
        missing = []
        for imported in sorted(_import_closure()["third"]):
            dist = _IMPORT_TO_DISTRIBUTION.get(imported, imported)
            if dist is None:                      # guarded optional import
                continue
            if dist.lower() not in names:
                missing.append(f"{imported} (provides: {dist})")
        self.assertEqual(missing, [], (
            "the MI API imports package(s) absent from "
            f"deploy/trakt-mi-api/requirements.txt: {missing}"))

    def test_the_repository_wide_heavyweights_are_not_installed(self):
        """The point of the change: these are unreachable from mi_agent_api.app
        and were costing 20+ minutes of Oryx build time on every deploy. If one
        genuinely becomes a runtime import, the closure test above fails first
        and tells you to add it deliberately."""
        names = _requirement_names()
        for excluded in ("streamlit", "scikit-learn", "rapidfuzz", "reportlab",
                         "azure-functions"):
            self.assertNotIn(excluded, names,
                             f"{excluded} is back in the App Service contract")


class TestWorkflowWiring(unittest.TestCase):
    """The workflow must actually deploy the artefact, not the checkout."""

    def test_the_deploy_step_uploads_the_staged_artefact(self):
        text = _WORKFLOW.read_text(encoding="utf-8")
        self.assertIn("package: ${{ env.ARTIFACT_ZIP }}", text, (
            "azure/webapps-deploy has no package: input — it would default to "
            "the whole checkout, and Oryx would read the repo-root "
            "requirements.txt as the runtime contract again"))
        self.assertNotIn("package: ${{ env.STAGE_DIR }}", text, (
            "package: must name the zip this workflow builds, not the staging "
            "FOLDER. Handing the action a folder leaves the archive layout to "
            "the action; a tree that lands one level deep inside the archive "
            "gives Oryx no root requirements.txt to detect (no antenv, no "
            "oryx-manifest.toml) and leaves /home/site/wwwroot/startup.sh "
            "absent — the exact four messages the App Service reported"))

    def test_the_artefact_is_zipped_from_inside_the_staging_directory(self):
        """`cd` into the staging dir before zipping is what puts its CONTENTS at
        the archive root. Zipping the directory by name from outside would wrap
        everything one level deeper, which is the failure this guards."""
        text = _WORKFLOW.read_text(encoding="utf-8")
        self.assertRegex(
            text, r'\(cd "\$\{STAGE_DIR\}" && zip -qr ',
            "the artefact must be zipped from INSIDE ${STAGE_DIR}")

    def test_the_zip_layout_is_asserted_before_deploying(self):
        """A layout assertion that only exists in this test file cannot fail the
        deploy. The workflow has to check it too, on the artefact it is about to
        upload."""
        text = _WORKFLOW.read_text(encoding="utf-8")
        for needle in ("grep -qx 'startup.sh'", "grep -qx 'requirements.txt'",
                       "^mi_agent_api/app.py$"):
            self.assertIn(needle, text, (
                f"the workflow does not assert {needle!r} on the built zip, so "
                "a nested artefact would deploy 'successfully' and fail at "
                "container start"))

    def test_the_workflow_installs_the_api_contract_not_the_root_one(self):
        text = _WORKFLOW.read_text(encoding="utf-8")
        self.assertIn("deploy/trakt-mi-api/requirements.txt", text)
        self.assertNotRegex(
            text, r"pip install -r requirements\.txt",
            "the workflow installs the repo-root requirements again")

    def test_the_workflow_reads_the_manifest(self):
        text = _WORKFLOW.read_text(encoding="utf-8")
        self.assertIn("deploy/trakt-mi-api/package_contents.txt", text)


class TestStagedArtefactImports(unittest.TestCase):
    """The App Service equivalent of the container's build-time smoke import.

    ``TestArtefactCompleteness`` checks that the manifest LISTS every package the
    import closure names. This checks the artefact the workflow actually builds
    can be IMPORTED — which is a different question, and the one that was open.

    The gap this closes, concretely: ``question_interpretation`` was reachable
    from ``mi_agent_api.app`` and absent from ``package_contents.txt``, and
    because ``mi_agent.llm_query_parser`` and ``mi_agent.execution_receipt``
    import it at MODULE level, the App Service failed at STARTUP rather than on
    the first request that reached it. The completeness test detects the missing
    name; only an import proves the artefact runs.

    Staging is the workflow's own loop (``cp -R --parents`` per manifest line,
    then drop ``tests``/``__pycache__``) over ~13 MB, and the import runs in a
    subprocess whose ``sys.path`` contains the staging directory ALONE — the
    repo is not importable, or the test would pass on a manifest that stages
    nothing.
    """

    def test_the_staged_artefact_imports_the_asgi_app(self):
        import shutil
        import tempfile

        with tempfile.TemporaryDirectory(prefix="mi-api-stage-") as stage:
            stage_dir = Path(stage)
            for path in _manifest_paths():
                source = _REPO / path
                self.assertTrue(source.exists(), (
                    f"package_contents.txt lists {path!r}, which does not exist "
                    "— the deploy workflow fails on this before it uploads"))
                destination = stage_dir / path
                destination.parent.mkdir(parents=True, exist_ok=True)
                if source.is_dir():
                    shutil.copytree(
                        source, destination, dirs_exist_ok=True,
                        ignore=shutil.ignore_patterns(
                            "__pycache__", "tests", ".pytest_cache"))
                else:
                    shutil.copy2(source, destination)

            # The REPO must not be importable, or this would pass on a manifest
            # that stages nothing. Third-party site-packages stay reachable:
            # whether the App Service can INSTALL them is
            # TestRequirementsSufficiency's question, not this one.
            probe = (
                "import sys, os\n"
                "stage, repo = %r, %r\n"
                "sys.path = [p for p in sys.path\n"
                "            if p and os.path.abspath(p) != repo]\n"
                "sys.path.insert(0, stage)\n"
                "assert repo not in [os.path.abspath(p) for p in sys.path], sys.path\n"
                "import mi_agent_api.app\n"
                "print('ok')\n" % (str(stage_dir), str(_REPO)))
            completed = subprocess.run(
                [sys.executable, "-c", probe],
                cwd=str(stage_dir), capture_output=True, text=True,
                env=dict(os.environ, TRAKT_RUNTIME_MODE="test", PYTHONPATH=""))

        self.assertEqual(completed.returncode, 0, (
            "the staged App Service artefact cannot import "
            "mi_agent_api.app, so `bash startup.sh` -> gunicorn would fail at "
            "STARTUP:\n"
            f"{completed.stderr.strip()[-2000:]}"))
        self.assertIn("ok", completed.stdout)


if __name__ == "__main__":
    unittest.main()


def _exclude_patterns() -> "list[str]":
    """Glob patterns from ``package_excludes.txt``; comments and blanks ignored."""
    out = []
    for raw in _EXCLUDES.read_text(encoding="utf-8").splitlines():
        line = raw.split("#", 1)[0].strip()
        if line:
            out.append(line)
    return out


class TestArtefactEconomy(unittest.TestCase):
    """Nothing ships that the runtime never opens — and nothing needed is pruned."""

    def test_every_declared_exclusion_matches_something(self):
        """A pattern that matches nothing is a stale rule, not a protection."""
        stale = [pat for pat in _exclude_patterns()
                 if not list(_REPO.glob(pat))]
        self.assertEqual(stale, [], (
            f"package_excludes.txt patterns match no file: {stale}. Either the "
            "file was renamed and the rule is now silently protecting nothing, "
            "or the rule was never right."))

    def test_no_excluded_file_is_read_by_staged_code(self):
        """THE SAFETY HALF. Excluding a file the runtime opens recreates exactly
        the failure `package_contents.txt` exists to prevent: an App Service that
        starts and 500s on the first request that reaches it.

        Checked by NAME across every staged package's Python — the same evidence
        that justified each rule. A file loaded by glob or directory walk would
        not be caught here, so the staged packages are also asserted to contain
        no such load below."""
        staged_py = []
        for top in sorted({p.split("/")[0] for p in _manifest_paths()}):
            root = _REPO / top
            if root.is_dir():
                staged_py.extend(q for q in root.rglob("*.py")
                                 if "tests" not in q.parts)
        blobs = {q: q.read_text(encoding="utf-8", errors="ignore")
                 for q in staged_py}
        offenders = []
        for pat in _exclude_patterns():
            for match in _REPO.glob(pat):
                name = match.name
                for path, text in blobs.items():
                    if name in text:
                        offenders.append(f"{name} named by {path.relative_to(_REPO)}")
        self.assertEqual(offenders, [], (
            "package_excludes.txt would prune file(s) the staged code reads: "
            f"{offenders}"))

    def test_the_workflow_applies_the_exclusions(self):
        """A declared rule that the staging step does not read is decoration."""
        workflow = _WORKFLOW.read_text(encoding="utf-8")
        self.assertIn("package_excludes.txt", workflow, (
            "deploy-mi-api.yml does not read package_excludes.txt, so the "
            "declared exclusions never reach the artefact."))

    def test_the_artefact_stays_within_its_weight_budget(self):
        """A backstop the exclude list cannot give: total staged weight.

        Named patterns only catch the bulk someone thought to name. This catches
        the next 15 MB whatever it is called. Raise the budget deliberately, with
        a reason — never to make a red test go green.
        """
        budget_mb = 20
        # TRACKED FILES ONLY. The workflow stages from a fresh checkout, so a
        # local scratch store or a gitignored fixture is not in the artefact and
        # must not be in the measurement — otherwise the budget reads whatever
        # happens to be in a developer's working tree and the number means
        # nothing. (Measured while writing this: an ignored 27 MB test store made
        # the same artefact read 38 MB here and 11 MB in CI.)
        tracked = subprocess.run(
            ["git", "ls-files", "-z", "--"] + _manifest_paths(),
            cwd=_REPO, capture_output=True, text=True, check=True)
        excluded = {q.resolve() for pat in _exclude_patterns()
                    for q in _REPO.glob(pat)}
        total = 0
        for rel in tracked.stdout.split("\0"):
            if not rel:
                continue
            q = _REPO / rel
            if "tests" in q.parts or q.name.endswith(".pyc"):
                continue
            if not q.is_file() or q.resolve() in excluded:
                continue
            total += q.stat().st_size
        size_mb = total / (1024 * 1024)
        self.assertLess(size_mb, budget_mb, (
            f"the App Service artefact is {size_mb:.1f} MB, over its {budget_mb} MB "
            "budget. Every megabyte is uploaded on each deploy and unpacked on "
            "each cold start. Either it is runtime code that has genuinely grown, "
            "or it is evidence that belongs in package_excludes.txt."))
