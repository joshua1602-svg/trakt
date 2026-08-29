"""Commercial go-live sprint — route-result cache correctness.

The cache is keyed on the SAME validator the response already carries, so its
correctness argument is the ETag's: a hit is by construction a response the
server would otherwise have re-sent as a 304. These tests pin that the key
actually separates every axis the sprint names, and that an unidentifiable or
failed response is never stored.

Correctness before latency: every test here is about what the cache must NOT
serve.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from mi_agent_api import http_cache  # noqa: E402


def _etag(**kw):
    base = dict(route="mi.snapshot", tenant="client_001", scope="total",
                identity=["ERE", "2026-07-31", ("config", "abc"), "root"])
    base.update(kw)
    return http_cache.build_etag(**base)


# --------------------------------------------------------------------------- #
# The key separates every axis the cache must never cross
# --------------------------------------------------------------------------- #
def test_a_different_tenant_is_a_different_key():
    assert _etag() != _etag(tenant="client_002")


def test_a_different_portfolio_scope_is_a_different_key():
    assert _etag() != _etag(scope="spv_a")


def test_a_different_client_or_run_is_a_different_key():
    assert _etag() != _etag(identity=["OTHER", "2026-07-31", ("config", "abc"), "root"])
    assert _etag() != _etag(identity=["ERE", "2026-06-30", ("config", "abc"), "root"])


def test_a_different_data_version_is_a_different_key():
    """The dataset identity carries the published bytes' own validator."""
    assert _etag(identity=["ERE", "2026-07-31", ("config", "abc"), ("funded_etag", "v1")]) \
        != _etag(identity=["ERE", "2026-07-31", ("config", "abc"), ("funded_etag", "v2")])


def test_a_different_configuration_version_is_a_different_key():
    """Re-approving a client's governed configuration must invalidate results
    derived from it — the reporting currency is one such derivation."""
    assert _etag() != _etag(identity=["ERE", "2026-07-31", ("config", "xyz"), "root"])


def test_a_different_route_is_a_different_key():
    assert _etag() != _etag(route="mi.cohorts")


def test_configuration_is_part_of_the_run_identity():
    identity = http_cache.dataset_identity("ERE", "2026-07-31")
    if identity is None:
        pytest.skip("no dataset root configured in this environment")
    assert any(isinstance(p, tuple) and p and p[0] == "config" for p in identity)


def test_the_configuration_fingerprint_tracks_content(tmp_path, monkeypatch):
    from mi_agent_api import currency as currency_mod
    currency_mod._load_client_config.cache_clear()
    cfg = tmp_path / "c.yaml"
    cfg.write_text("portfolio:\n  base_currency: GBP\n", encoding="utf-8")
    monkeypatch.setenv(currency_mod.ENV_CLIENT_CONFIG, str(cfg))
    first = http_cache.config_fingerprint("ERE")
    cfg.write_text("portfolio:\n  base_currency: EUR\n", encoding="utf-8")
    currency_mod._load_client_config.cache_clear()
    assert http_cache.config_fingerprint("ERE") != first
    currency_mod._load_client_config.cache_clear()


# --------------------------------------------------------------------------- #
# What must never be stored
# --------------------------------------------------------------------------- #
def test_an_unidentifiable_response_is_computed_every_time():
    """No validator means no caching — the same rule that governs whether a
    conditional response is offered at all."""
    calls = []
    for _ in range(3):
        http_cache.cached(None, lambda: calls.append(1) or {"ok": True})
    assert len(calls) == 3


def test_a_failed_payload_is_never_stored():
    """A transient failure must not be replayed to later callers."""
    calls = []

    def build():
        calls.append(1)
        return {"ok": False, "error": "transient"}

    tag = _etag(route="mi.test.failure")
    assert http_cache.cached(tag, build)["error"] == "transient"
    assert http_cache.cached(tag, build)["error"] == "transient"
    assert len(calls) == 2, "a failed payload was cached"


def test_an_unavailable_payload_is_never_stored():
    calls = []

    def build():
        calls.append(1)
        return {"available": False, "reason": "no ITL3 field on the tape"}

    tag = _etag(route="mi.test.unavailable")
    http_cache.cached(tag, build)
    http_cache.cached(tag, build)
    assert len(calls) == 2


def test_a_successful_payload_is_computed_once_per_validator():
    calls = []

    def build():
        calls.append(1)
        return {"ok": True, "value": 42}

    tag = _etag(route="mi.test.success")
    assert http_cache.cached(tag, build)["value"] == 42
    assert http_cache.cached(tag, build)["value"] == 42
    assert len(calls) == 1


def test_disabling_the_http_cache_disables_result_caching(monkeypatch):
    """``TRAKT_HTTP_CACHE=off`` must restore the previous behaviour exactly."""
    monkeypatch.setenv("TRAKT_HTTP_CACHE", "off")
    calls = []
    tag = 'W/"deadbeef"'
    for _ in range(2):
        http_cache.cached(tag, lambda: calls.append(1) or {"ok": True})
    assert len(calls) == 2


def test_a_builder_that_raises_leaves_the_cache_untouched():
    tag = _etag(route="mi.test.raises")

    def boom():
        raise RuntimeError("compute failed")

    with pytest.raises(RuntimeError):
        http_cache.cached(tag, boom)
    assert http_cache.cached(tag, lambda: {"ok": True})["ok"] is True
