#!/usr/bin/env python3
"""Production source-approval policy and the runtime mode that scopes it.

Covers required cases 17-21. The rule under test: a production caller — through
ANY channel — is refused a synthetic, fixture, unavailable or otherwise
unapproved dataset, and the refusal happens inside the governed capability
rather than in one adapter.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from trakt_core import policy, runtime
from trakt_core.errors import ErrorCode


# --------------------------------------------------------------------------- #
# The policy matrix
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("base", sorted(policy.PRODUCTION_APPROVED_SOURCE_BASES))
def test_governed_bases_are_approved_in_production(base):
    decision = policy.evaluate_source_approval(base, mode=runtime.MODE_PRODUCTION)
    assert decision.approved is True
    assert decision.fixture is False


@pytest.mark.parametrize("base", sorted(policy.FIXTURE_SOURCE_BASES))
def test_fixture_bases_are_refused_in_production(base):
    decision = policy.evaluate_source_approval(base, mode=runtime.MODE_PRODUCTION)
    assert decision.approved is False
    assert decision.error_code == ErrorCode.DATA_SOURCE_NOT_APPROVED


@pytest.mark.parametrize("base", sorted(policy.FIXTURE_SOURCE_BASES))
def test_fixture_bases_are_permitted_in_test_mode(base):
    """Required case 20 — the sanctioned path for automated tests."""
    decision = policy.evaluate_source_approval(base, mode=runtime.MODE_TEST)
    assert decision.approved is True
    assert decision.fixture is True


def test_unavailable_is_distinguished_from_unapproved():
    """An agent must be able to tell "come back later" from "never"."""
    unavailable = policy.evaluate_source_approval("unavailable",
                                                  mode=runtime.MODE_PRODUCTION)
    assert unavailable.error_code == ErrorCode.DATA_SOURCE_UNAVAILABLE

    from trakt_core.errors import is_retryable
    assert is_retryable(ErrorCode.DATA_SOURCE_UNAVAILABLE) is True
    assert is_retryable(ErrorCode.DATA_SOURCE_NOT_APPROVED) is False


def test_a_loaded_but_unreadable_dataset_is_unavailable():
    decision = policy.evaluate_source_approval("platform_canonical",
                                               mode=runtime.MODE_PRODUCTION,
                                               dataset_available=False)
    assert decision.approved is False
    assert decision.error_code == ErrorCode.DATA_SOURCE_UNAVAILABLE


def test_synthetic_refusal_says_why():
    decision = policy.evaluate_source_approval("synthetic_demo",
                                               mode=runtime.MODE_PRODUCTION)
    assert "synthetic" in decision.reason.lower()


def test_policy_bases_match_the_resolver_vocabulary():
    """Drift guard: the policy names resolution bases as literals so trakt_core
    stays dependency-free. If ``resolve_data_source`` grows a new base, this
    fails rather than the new base silently defaulting to 'fixture'."""
    from mi_agent_api import data_source
    import inspect

    source = inspect.getsource(data_source.resolve_data_source)
    known = (policy.PRODUCTION_APPROVED_SOURCE_BASES
             | policy.FIXTURE_SOURCE_BASES
             | {policy.BASE_UNAVAILABLE})
    for base in known:
        assert f'"{base}"' in source, (
            f"policy knows base {base!r} but resolve_data_source no longer emits it")

    import re
    emitted = set(re.findall(r'return\s+\w+,\s*"(\w+)"', source))
    emitted |= set(re.findall(r'"(\w+)"\s+if\s+\w+\s+else\s+"(\w+)"', source)[0]
                   if re.findall(r'"(\w+)"\s+if\s+\w+\s+else\s+"(\w+)"', source) else [])
    unknown = {b for b in emitted if b} - known
    assert not unknown, f"resolve_data_source emits base(s) the policy does not know: {unknown}"


# --------------------------------------------------------------------------- #
# Runtime mode — the reason a fixture cannot silently become live
# --------------------------------------------------------------------------- #
def test_unset_mode_is_production(monkeypatch):
    monkeypatch.delenv("TRAKT_RUNTIME_MODE", raising=False)
    monkeypatch.delenv("WEBSITE_SITE_NAME", raising=False)
    monkeypatch.delenv("WEBSITE_INSTANCE_ID", raising=False)
    assert runtime.runtime_mode() == runtime.MODE_PRODUCTION
    assert runtime.is_production() is True


def test_unrecognised_mode_degrades_to_production(monkeypatch):
    monkeypatch.setenv("TRAKT_RUNTIME_MODE", "staging-ish")
    monkeypatch.delenv("WEBSITE_SITE_NAME", raising=False)
    monkeypatch.delenv("WEBSITE_INSTANCE_ID", raising=False)
    assert runtime.runtime_mode() == runtime.MODE_PRODUCTION
    runtime.validate_runtime_mode()  # logs, does not raise


def test_non_production_mode_is_refused_inside_azure(monkeypatch):
    """The property that makes TRAKT_RUNTIME_MODE safe rather than 'just another
    env flag': it cannot take effect in a deployed environment."""
    monkeypatch.setenv("TRAKT_RUNTIME_MODE", "test")
    monkeypatch.setenv("WEBSITE_SITE_NAME", "trakt-mi-api")
    with pytest.raises(RuntimeError, match="not permitted in Azure"):
        runtime.validate_runtime_mode()
    # Even if startup validation were bypassed, the effective mode is production.
    assert runtime.runtime_mode() == runtime.MODE_PRODUCTION


def test_azure_markers_match_the_storage_layer():
    """trakt_core duplicates the Azure markers as literals to stay dependency
    free; this keeps the two definitions aligned."""
    from apps.blob_trigger_app import storage
    assert set(runtime._AZURE_MARKERS) == set(storage._AZURE_MARKERS)


# --------------------------------------------------------------------------- #
# Landing-page demo isolation (required case 21)
#
# How the isolation actually works, which is worth stating precisely because it
# is NOT the source-approval policy:
#
#   * ``demo_platform`` is a CAPTURE-TIME generator. It deliberately runs the real
#     API against a synthetic pack shaped exactly like a production platform
#     canonical (that is the point — the fixtures have to be real API output), and
#     writes committed fixtures. It is never deployed.
#   * The landing page / demo film then serve those COMMITTED fixtures. Nothing a
#     visitor touches calls the MI API.
#   * Because the demo pack presents as ``platform_canonical``, the base-level
#     source-approval check does not distinguish it from real client data. Its
#     isolation rests on the STORAGE BACKEND instead: the pack is only reachable
#     with ``TRAKT_STORAGE_BACKEND=file`` + ``TRAKT_LOCAL_BLOB_ROOT``, and in
#     Azure ``open_storage`` forces blob storage and refuses to fall back to the
#     filesystem. The tests below pin both halves.
# --------------------------------------------------------------------------- #
def test_production_capability_does_not_depend_on_the_demo_generator():
    """The dependency runs one way: the generator may drive the API, never the
    reverse. If production code imported ``demo_platform`` the demo pack would
    become part of the deployed runtime."""
    offenders = []
    for pkg in ("trakt_core", "mi_agent_api", "mi_agent"):
        root = _REPO_ROOT / pkg
        if not root.exists():  # pragma: no cover - repo layout guard
            continue
        for path in root.rglob("*.py"):
            if "tests" in path.parts:
                continue
            if "demo_platform" in path.read_text(encoding="utf-8", errors="ignore"):
                offenders.append(str(path.relative_to(_REPO_ROOT)))
    assert not offenders, (
        "production modules must not reference the demo generator: "
        + ", ".join(offenders))


def test_landing_page_serves_committed_fixtures_not_a_live_api():
    """The film/landing assets are pre-generated JSON committed to the repo."""
    fixtures = _REPO_ROOT / "demo-video" / "public" / "fixtures"
    if not fixtures.exists():  # pragma: no cover - repo layout guard
        pytest.skip("demo-video fixtures not present")
    committed = sorted(fixtures.glob("*.json"))
    assert committed, "expected committed demo fixtures"
    # No fixture may carry an API base URL that would make the page call out.
    for path in committed:
        text = path.read_text(encoding="utf-8", errors="ignore")
        assert "azurewebsites.net" not in text, f"{path.name} points at a live API"


def test_demo_pack_is_unreachable_from_an_azure_runtime(monkeypatch):
    """The storage-backend half of the isolation.

    In Azure the filesystem backend is refused outright, so a deployed process
    cannot be pointed at the demo pack's local blob root even if the app settings
    named one.
    """
    from apps.blob_trigger_app import storage

    monkeypatch.setenv("WEBSITE_SITE_NAME", "trakt-mi-api")
    monkeypatch.delenv("TRAKT_BLOB_CONNECTION", raising=False)
    monkeypatch.delenv("AzureWebJobsStorage", raising=False)
    monkeypatch.setenv("TRAKT_LOCAL_BLOB_ROOT", "/tmp/demo-pack")
    monkeypatch.delenv("TRAKT_STORAGE_BACKEND", raising=False)

    decision = storage.decide_backend()
    assert decision["backend"] == "azure_blob"
    with pytest.raises(ValueError, match="Refusing to fall back"):
        storage.open_storage()


def test_no_production_override_flag_exists():
    """There must be no caller-controlled way to re-admit synthetic data."""
    import inspect
    for module in (policy, runtime):
        source = inspect.getsource(module)
        for banned in ("allow_synthetic", "allow_demo", "force_approve",
                       "skip_approval"):
            assert banned not in source, f"{module.__name__} exposes {banned!r}"
