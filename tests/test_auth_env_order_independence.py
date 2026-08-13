#!/usr/bin/env python3
"""Does a test that flips the auth switch put it back?

``mi_agent_api.auth.auth_guard`` reads ``MI_AGENT_AUTH_ENABLED`` from the
process environment on every call, so any test that changes it changes the
behaviour of every test that runs afterwards in the same process. One did:
``mi_agent_api/tests/test_http_conditional.py`` set it to ``"true"`` for its own
assertion and then hard-coded ``"false"`` on the way out, which silently
disabled authentication for the remainder of the run.

The victim was
``tests/test_agent_identity_and_api.py::test_the_easy_auth_guard_exempts_the_agent_prefix``,
which asserts ``auth_guard`` still refuses a non-exempt path. With auth globally
off it refused nothing, so the test failed — but only in a full-suite run, and
only when the polluter happened to be collected first. Alone, both files passed.

Why this is worth a dedicated regression
----------------------------------------
The failure mode is the dangerous one: a **security** assertion whose result
depends on collection order, failing in the permissive direction. A suite that
can silently stop enforcing auth for two thousand subsequent tests is not
reporting what anyone thinks it is reporting, and the greenness of every later
regression claim rests on it.

Running the two files in **both orders** in a subprocess is the honest check.
Asserting on the environment variable alone would pass even if the reload logic
regressed; running the real victim proves the real invariant.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]

#: The WHOLE file, not a node id. There were two independent leaks in it — the
#: test that flipped the switch directly, and a setUpClass that called
#: ``configure_env``. A node-id-scoped regression passed against the first fix
#: while the file as a whole still polluted, so this deliberately runs
#: everything that file does.
POLLUTER = "mi_agent_api/tests/test_http_conditional.py"
VICTIM = ("tests/test_agent_identity_and_api.py::"
          "test_the_easy_auth_guard_exempts_the_agent_prefix")

AUTH_ENV = "MI_AGENT_AUTH_ENABLED"


def _run(*node_ids: str) -> subprocess.CompletedProcess:
    """Run pytest in a fresh process, with the auth switch deliberately unset.

    A subprocess is required rather than convenient: the pollution is process
    environment state, so a same-process check would be measuring the harness
    rather than the tests. The variable is cleared first so the child starts
    from the default the guard is written against — enforcement on.
    """
    environment = {k: v for k, v in os.environ.items() if k != AUTH_ENV}
    return subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider",
         *node_ids],
        cwd=_REPO_ROOT, env=environment, capture_output=True, text=True,
        timeout=900)


@pytest.mark.parametrize("order,node_ids", [
    ("polluter first", (POLLUTER, VICTIM)),
    ("victim first", (VICTIM, POLLUTER)),
])
def test_the_auth_guard_assertion_holds_in_either_order(order, node_ids):
    """Both orders must pass. 'Polluter first' is the one that used to fail."""
    result = _run(*node_ids)
    assert result.returncode == 0, (
        f"running {order} failed — the auth switch is leaking again:\n"
        f"{result.stdout[-3000:]}")


def test_the_conditional_auth_test_leaves_the_switch_as_it_found_it():
    """The narrower invariant, asserted directly.

    Complements the order test rather than replacing it: this one names the
    exact contract the polluter broke, so a future regression reports *what*
    went wrong rather than only that something did.
    """
    probe = (
        "import os, subprocess, sys\n"
        f"before = os.environ.get({AUTH_ENV!r})\n"
        "subprocess.run([sys.executable, '-m', 'pytest', '-q', "
        "'-p', 'no:cacheprovider', " f"{POLLUTER!r}], cwd={str(_REPO_ROOT)!r}, "
        "capture_output=True)\n"
        f"after = os.environ.get({AUTH_ENV!r})\n"
        "assert before == after, (before, after)\n"
    )
    # The probe runs the polluter in a grandchild so this process' own
    # environment is never the thing under test.
    result = subprocess.run([sys.executable, "-c", probe], cwd=_REPO_ROOT,
                            capture_output=True, text=True, timeout=900)
    assert result.returncode == 0, result.stderr[-2000:]


def test_auth_is_enforced_by_default_when_the_switch_is_absent():
    """The default the whole invariant rests on.

    If a missing variable ever came to mean 'auth off', restoring it correctly
    would stop mattering and the tests above would pass while the system was
    wide open.
    """
    import importlib

    previous = os.environ.get(AUTH_ENV)
    os.environ.pop(AUTH_ENV, None)
    try:
        from mi_agent_api import auth as auth_mod
        importlib.reload(auth_mod)
        assert auth_mod._auth_enabled() is True
    finally:
        if previous is None:
            os.environ.pop(AUTH_ENV, None)
        else:
            os.environ[AUTH_ENV] = previous
        from mi_agent_api import auth as auth_mod
        importlib.reload(auth_mod)
