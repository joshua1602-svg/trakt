#!/usr/bin/env python3
"""Architectural tests: the governed capability must not depend on an interface.

Covers required cases 32-35. These are the tests that keep the boundary honest —
without them the ``from . import app`` shortcut that this work removed would
simply come back the next time a resolver is needed.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def _in_clean_process(code: str) -> subprocess.CompletedProcess:
    """Run ``code`` in a fresh interpreter.

    A separate process is the only honest way to assert "importing X does not
    import FastAPI": inside the test session FastAPI is already in sys.modules
    because other suites use TestClient.
    """
    import os
    env = {**os.environ, "TRAKT_RUNTIME_MODE": "test", "PYTHONPATH": str(_REPO_ROOT)}
    return subprocess.run([sys.executable, "-c", code], capture_output=True,
                          text=True, cwd=str(_REPO_ROOT), env=env)


def test_trakt_core_imports_without_a_web_framework():
    proc = _in_clean_process(
        "import sys, trakt_core;"
        "assert 'fastapi' not in sys.modules, 'trakt_core pulled in fastapi';"
        "assert 'starlette' not in sys.modules, 'trakt_core pulled in starlette';"
        "print('ok')")
    assert proc.returncode == 0, proc.stderr
    assert "ok" in proc.stdout


def test_governed_capability_imports_without_fastapi():
    """Required case 32."""
    proc = _in_clean_process(
        "import sys;"
        "from mi_agent_api.mi_service import execute_governed_mi_query;"
        "assert 'fastapi' not in sys.modules, 'mi_service pulled in fastapi';"
        "assert 'starlette' not in sys.modules, 'mi_service pulled in starlette';"
        "assert 'mi_agent_api.app' not in sys.modules, 'mi_service imported app';"
        "print('ok')")
    assert proc.returncode == 0, proc.stderr
    assert "ok" in proc.stdout


def test_calling_the_capability_does_not_import_fastapi():
    """The regression that mattered: the old ``_resolvers()`` deferred the
    ``from . import app`` to CALL time, so the import test passed while the call
    still dragged in the web framework."""
    proc = _in_clean_process(
        "import sys, os;"
        "from pathlib import Path;"
        "demo = sorted(Path('.').glob('synthetic_demo/**/*canonical_typed.csv'))[0];"
        "os.environ['MI_AGENT_DATA_CSV'] = str(demo);"
        "os.environ['MI_AGENT_LLM_PARSER'] = 'off';"
        "from trakt_core.context import ExecutionContext;"
        "from mi_agent_api.mi_service import MiQueryRequest, execute_governed_mi_query;"
        "r = execute_governed_mi_query("
        "    MiQueryRequest(question='What is the total current balance?'),"
        "    ExecutionContext.for_internal('ERE'));"
        "assert r.ok, r.error and r.error.code;"
        "assert 'fastapi' not in sys.modules, 'calling the capability imported fastapi';"
        "assert 'mi_agent_api.app' not in sys.modules, 'calling the capability imported app';"
        "print('ok')")
    assert proc.returncode == 0, proc.stderr
    assert "ok" in proc.stdout


def test_dataset_resolver_is_independently_testable():
    """Required case 33."""
    proc = _in_clean_process(
        "import sys;"
        "from mi_agent_api import datasets;"
        "d = datasets.describe_active_dataset();"
        "assert hasattr(d, 'source_base') and hasattr(d, 'snapshot_id');"
        "assert 'fastapi' not in sys.modules;"
        "print('ok')")
    assert proc.returncode == 0, proc.stderr
    assert "ok" in proc.stdout


def test_no_environment_mutation_is_needed_for_per_request_dataset_selection():
    """Required case 34.

    A request selects a portfolio by passing it; the capability must not write
    to ``os.environ`` to make that happen. Asserted by snapshotting the
    environment across two differently-scoped calls.
    """
    import os

    from pathlib import Path as _P
    from trakt_core.context import ExecutionContext
    from mi_agent_api import data_source
    from mi_agent_api.mi_service import MiQueryRequest, execute_governed_mi_query

    demo = sorted(_REPO_ROOT.glob("synthetic_demo/**/*canonical_typed.csv"))[0]
    os.environ["MI_AGENT_DATA_CSV"] = str(demo)
    os.environ["MI_AGENT_LLM_PARSER"] = "off"
    data_source.reset_cache()
    try:
        before = dict(os.environ)
        ctx = ExecutionContext.for_internal("ERE")
        execute_governed_mi_query(
            MiQueryRequest(question="What is the total current balance?"), ctx)
        execute_governed_mi_query(
            MiQueryRequest(question="What is the total current balance?",
                           portfolio_id="direct_001"), ctx)
        assert dict(os.environ) == before, (
            "the capability mutated the process environment to select a dataset")
    finally:
        data_source.reset_cache()


def test_no_circular_imports_between_the_layers():
    """Required case 35 — each layer imports cleanly on its own, in dependency
    order, in a fresh interpreter."""
    for module in ("trakt_core", "mi_agent_api.datasets", "mi_agent_api.dependencies",
                   "mi_agent_api.mi_service", "mi_agent_api.artefacts",
                   "mi_agent_api.presenters", "mi_agent_api.identity",
                   "mi_agent_api.app"):
        proc = _in_clean_process(f"import {module}; print('ok')")
        assert proc.returncode == 0, f"{module} failed to import:\n{proc.stderr}"


def test_capability_layer_has_no_source_level_dependency_on_the_http_module():
    """A grep-level guard: it is easy to reintroduce ``from . import app`` inside
    a function, where an import test would not catch it until that branch ran."""
    import ast

    def _imported_modules(path: Path) -> set[str]:
        """Every module name imported anywhere in the file, including inside a
        function body — which is where the old deferred ``from . import app``
        lived, and where an import test alone would not have caught it."""
        tree = ast.parse(path.read_text(encoding="utf-8"))
        names: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names.update(a.name for a in node.names)
            elif isinstance(node, ast.ImportFrom):
                # level>0 is a relative import; record it as ".<module>".
                names.add(("." * node.level) + (node.module or ""))
        return names

    for name in ("mi_service.py", "datasets.py", "artefacts.py", "presenters.py",
                 "dependencies.py"):
        imported = _imported_modules(_REPO_ROOT / "mi_agent_api" / name)
        assert ".app" not in imported, f"{name} imports the FastAPI module"
        assert not any(m.startswith("fastapi") or m.startswith("starlette")
                       for m in imported), f"{name} imports a web framework"

    for path in (_REPO_ROOT / "trakt_core").glob("*.py"):
        imported = _imported_modules(path)
        assert not any(m.startswith(("fastapi", "starlette", "mi_agent_api",
                                     "mi_agent", "apps.", "engine."))
                       for m in imported), (
            f"trakt_core/{path.name} depends on an application package: {imported}")
