"""Shared test configuration for the MI Agent API suite.

The API now enforces authentication by default (see mi_agent_api/auth.py). The
existing functional tests exercise the endpoints directly without a platform
principal header, so we disable auth enforcement for the suite by default. Tests
that specifically verify auth behaviour (test_auth.py) opt back in by setting
MI_AGENT_AUTH_ENABLED=true within the test.
"""

from __future__ import annotations

import os

import pytest


@pytest.fixture(autouse=True)
def _disable_auth_by_default(monkeypatch):
    # Only set it when a test hasn't explicitly configured auth itself.
    if "MI_AGENT_AUTH_ENABLED" not in os.environ:
        monkeypatch.setenv("MI_AGENT_AUTH_ENABLED", "false")
    yield
