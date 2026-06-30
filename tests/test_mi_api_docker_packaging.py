#!/usr/bin/env python3
"""tests/test_mi_api_docker_packaging — guard the MI API container packaging.

Catches the class of bug where a repo-level runtime package the MI API needs
(e.g. ``snapshot``) is missing from the image — either by failing to import or by
being excluded in ``.dockerignore``.

Run: python -m unittest tests.test_mi_api_docker_packaging
"""

from __future__ import annotations

import subprocess
import sys
import unittest
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]

# Repo-level packages the MI API imports transitively (mi_agent → snapshot /
# analytics_lib / engine; the API itself → apps.blob_trigger_app; config = data).
_REQUIRED_RUNTIME_PACKAGES = (
    "mi_agent_api", "mi_agent", "snapshot", "analytics_lib", "engine", "apps", "config",
)


class TestMiApiDockerPackaging(unittest.TestCase):

    def test_import_app_and_snapshot(self):
        """The exact container smoke import must succeed."""
        proc = subprocess.run(
            [sys.executable, "-c", "import mi_agent_api.app; import snapshot.model"],
            cwd=str(_REPO), capture_output=True, text=True)
        self.assertEqual(proc.returncode, 0,
                         f"import failed:\nstdout={proc.stdout}\nstderr={proc.stderr}")

    def test_dockerignore_does_not_exclude_runtime_packages(self):
        """A required runtime package must not be excluded in .dockerignore."""
        dockerignore = _REPO / ".dockerignore"
        self.assertTrue(dockerignore.exists(), ".dockerignore missing")
        patterns = set()
        for raw in dockerignore.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            patterns.add(line.strip("/"))          # normalise pkg, /pkg, pkg/
        offenders = [p for p in _REQUIRED_RUNTIME_PACKAGES if p in patterns]
        self.assertEqual(offenders, [],
                         f".dockerignore excludes runtime package(s): {offenders}")

    def test_required_packages_are_real_packages(self):
        """Each required package exists with an __init__.py (or is the config dir)."""
        for pkg in _REQUIRED_RUNTIME_PACKAGES:
            p = _REPO / pkg
            self.assertTrue(p.is_dir(), f"missing dir: {pkg}")
            if pkg != "config":
                self.assertTrue((p / "__init__.py").exists(), f"not a package: {pkg}")


if __name__ == "__main__":
    unittest.main()
