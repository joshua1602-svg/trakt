"""Repository-wide pytest configuration.

Sets the governed runtime mode to ``test`` for the whole suite. This is the
*sanctioned* way for automated tests to use fixture and synthetic datasets: the
production source-approval policy (``trakt_core.policy``) refuses anything but a
platform canonical or a promoted central tape when the mode is ``production``,
and ``production`` is the default whenever ``TRAKT_RUNTIME_MODE`` is unset.

This does not weaken the control:

* the default outside the test suite is ``production`` (fail closed);
* ``trakt_core.runtime.validate_runtime_mode`` refuses a non-production mode
  when the Azure markers are present, so this setting cannot take effect in a
  deployed environment;
* ``tests/test_governance_source_policy.py`` flips the mode back to
  ``production`` and asserts that every channel refuses fixture data.

Tests that need to exercise production behaviour set the mode explicitly with
``monkeypatch.setenv`` and are unaffected by the session default below.
"""

from __future__ import annotations

import os


def pytest_configure(config):
    os.environ.setdefault("TRAKT_RUNTIME_MODE", "test")
    # "slow" marks the end-to-end rehearsals that drive the real agents over a
    # real tape (tests/test_occ_go_live_e2e.py). They are part of the normal
    # suite; the marker exists so they can be deselected during quick loops
    # with -m "not slow".
    config.addinivalue_line(
        "markers", "slow: end-to-end run with the real agents (minutes, not "
                   "seconds)")
