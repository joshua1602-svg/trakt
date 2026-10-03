#!/usr/bin/env python3
"""The start-up warm prepares what a first weekly question reads, and again
whenever the data changes.

On 2026-10-02 "pipeline amount evolution by week" took 135 s well after start:
the warm prepared every extract's full frame twice (with and without the
history model) into a 64-entry memo, kept almost none, and built none of the
per-extract summaries the weekly series read. These pin that after the warm
the series prepares nothing, and that the warm runs again when — and only
when — a new extract or funded run changes the data.
"""
from __future__ import annotations

import tempfile

import pytest

from mi_agent_api import app as app_mod
from mi_agent_api import evolution as evolution_mod
from mi_agent_api import serving_cache
from tests.test_pipeline_blob_root_discovery import _BlobRootFixture
from trakt_core import perf


def test_after_the_warm_the_weekly_series_prepares_nothing():
    from mi_agent_api import datasets
    with tempfile.TemporaryDirectory() as td:
        fx = _BlobRootFixture(td)
        with fx.env():
            serving_cache.clear_all()
            datasets._PIPELINE_MIRROR_CACHE.update(root=None, sig=None, local=None)
            model = datasets._pipeline_history("ERE")
            app_mod._warm_pipeline_extracts("ERE", model)
            root = datasets._pipeline_discovery_root()
            with perf.collect(route="t") as c:
                series = evolution_mod.pipeline_evolution(
                    root, "ERE", None, historical_model=model)
                caches = c.snapshot()["caches"]
            assert len(series.get("periods") or ()) == len(fx.DATES)
            assert caches.get("pipeline_extract_summary", {}).get("miss", 0) == 0
            assert caches.get("pipeline_prep", {}).get("miss", 0) == 0


class _Stop(Exception):
    pass


def _run_loop(monkeypatch, signatures):
    """Run the warm loop over a sequence of data signatures; count the warms."""
    warmed = []
    seq = iter(signatures)
    monkeypatch.setattr(app_mod, "_warm_signature", lambda: next(seq))
    monkeypatch.setattr(app_mod, "_warm_caches", lambda: warmed.append(1))
    monkeypatch.setattr(serving_cache, "prune_other_builds", lambda: 0)
    monkeypatch.setenv("TRAKT_MI_WARM_INTERVAL_S", "1")
    calls = {"n": 0}

    def fake_sleep(_s):
        calls["n"] += 1
        if calls["n"] >= len(signatures):
            raise _Stop()

    monkeypatch.setattr(app_mod.time, "sleep", fake_sleep)
    with pytest.raises(_Stop):
        app_mod._warm_loop()
    return len(warmed)


def test_the_warm_runs_again_only_when_the_data_changes(monkeypatch):
    # start; unchanged; unchanged; a new weekly extract; unchanged
    assert _run_loop(monkeypatch, ["a", "a", "a", "b", "b"]) == 2


def test_a_failed_data_check_never_re_warms_in_a_loop(monkeypatch):
    def failing():
        raise RuntimeError("storage unreachable")
    warmed = []
    monkeypatch.setattr(app_mod, "_warm_signature", failing)
    monkeypatch.setattr(app_mod, "_warm_caches", lambda: warmed.append(1))
    monkeypatch.setattr(serving_cache, "prune_other_builds", lambda: 0)
    monkeypatch.setenv("TRAKT_MI_WARM_INTERVAL_S", "1")
    calls = {"n": 0}

    def fake_sleep(_s):
        calls["n"] += 1
        if calls["n"] >= 3:
            raise _Stop()

    monkeypatch.setattr(app_mod.time, "sleep", fake_sleep)
    with pytest.raises(_Stop):
        app_mod._warm_loop()
    assert len(warmed) == 1          # the start-up warm, and no more


def test_an_interval_of_zero_warms_once_and_stops(monkeypatch):
    warmed = []
    monkeypatch.setattr(app_mod, "_warm_signature", lambda: "a")
    monkeypatch.setattr(app_mod, "_warm_caches", lambda: warmed.append(1))
    monkeypatch.setattr(serving_cache, "prune_other_builds", lambda: 0)
    monkeypatch.setenv("TRAKT_MI_WARM_INTERVAL_S", "0")
    app_mod._warm_loop()
    assert warmed == [1]


def test_health_reports_the_warm(monkeypatch):
    from fastapi.testclient import TestClient
    monkeypatch.setattr(app_mod, "_warm_loop", lambda: None)
    monkeypatch.setitem(app_mod._WARM_STATUS, "state", "warm")
    with TestClient(app_mod.app) as client:
        root = client.get("/").json()
    assert root["warmup"] == "warm"
