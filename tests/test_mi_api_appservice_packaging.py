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
        self.assertIn("package: ${{ env.STAGE_DIR }}", text, (
            "azure/webapps-deploy has no package: input — it would default to "
            "the whole checkout, and Oryx would read the repo-root "
            "requirements.txt as the runtime contract again"))

    def test_the_workflow_installs_the_api_contract_not_the_root_one(self):
        text = _WORKFLOW.read_text(encoding="utf-8")
        self.assertIn("deploy/trakt-mi-api/requirements.txt", text)
        self.assertNotRegex(
            text, r"pip install -r requirements\.txt",
            "the workflow installs the repo-root requirements again")

    def test_the_workflow_reads_the_manifest(self):
        text = _WORKFLOW.read_text(encoding="utf-8")
        self.assertIn("deploy/trakt-mi-api/package_contents.txt", text)


if __name__ == "__main__":
    unittest.main()
