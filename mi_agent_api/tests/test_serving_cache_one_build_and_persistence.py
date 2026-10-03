#!/usr/bin/env python3
"""One build per key, and small results that outlive the process.

The 2026-10-02 sign-off took 135 s on "pipeline amount evolution by week": the
first weekly question prepared every one of ~90 extracts, and so did the next
worker, the next restart and the next weekly upload. These pin the two cache
guarantees that keep that work out of a question:

  * concurrent requests for the same key wait for ONE build — a question that
    arrives while the warm is building an extract's summary takes that result;
  * a ``persist=True`` cache keeps each value on persistent storage under the
    deployed commit, so a restart reads it back, and another build never does.
"""
from __future__ import annotations

import json
import threading
import time

import pytest

from mi_agent_api import build_info as build_info_mod
from mi_agent_api import serving_cache as sc


@pytest.fixture
def persistent(monkeypatch, tmp_path):
    """The persistent tier switched on, under a stamped build, in a temp dir."""
    monkeypatch.setenv("TRAKT_SERVING_CACHE_DIR", str(tmp_path))
    monkeypatch.setenv("TRAKT_BUILD_COMMIT", "a" * 40)
    monkeypatch.delenv("TRAKT_SERVING_CACHE_PERSIST", raising=False)
    monkeypatch.delenv("TRAKT_SERVING_CACHE", raising=False)
    build_info_mod.build_info.cache_clear()
    yield tmp_path
    build_info_mod.build_info.cache_clear()


def _restarted(name: str) -> sc.BoundedCache:
    """A fresh cache of the same name: what a new process starts with."""
    return sc.BoundedCache(name, max_entries=8, persist=True)


# -- one build per key ------------------------------------------------------- #
def test_concurrent_requests_for_one_key_build_it_once():
    cache = sc.BoundedCache("single_flight_once", max_entries=8)
    calls, release = [], threading.Event()

    def build():
        calls.append(1)
        release.wait(5)
        return {"built": True}

    results = []
    threads = [threading.Thread(target=lambda: results.append(
        cache.get_or_build("k", build))) for _ in range(4)]
    for t in threads:
        t.start()
    time.sleep(0.2)
    release.set()
    for t in threads:
        t.join(5)
    assert len(calls) == 1
    assert results == [{"built": True}] * 4


def test_different_keys_still_build_in_parallel():
    cache = sc.BoundedCache("single_flight_parallel", max_entries=8)
    both_inside = threading.Barrier(2, timeout=5)

    def build():
        both_inside.wait()          # deadlocks unless the two builds overlap
        return 1

    threads = [threading.Thread(target=cache.get_or_build, args=(k, build))
               for k in ("a", "b")]
    for t in threads:
        t.start()
    for t in threads:
        t.join(5)
    assert not both_inside.broken


def test_a_failed_build_is_not_shared_each_waiter_builds_for_itself():
    cache = sc.BoundedCache("single_flight_failure", max_entries=8)
    first_inside, release = threading.Event(), threading.Event()
    outcomes = []

    def failing():
        first_inside.set()
        release.wait(5)
        raise RuntimeError("unreadable extract")

    def owner():
        try:
            cache.get_or_build("k", failing)
        except RuntimeError as exc:
            outcomes.append(("owner", str(exc)))

    def waiter():
        outcomes.append(("waiter", cache.get_or_build("k", lambda: "rebuilt")))

    a = threading.Thread(target=owner)
    a.start()
    first_inside.wait(5)
    b = threading.Thread(target=waiter)
    b.start()
    time.sleep(0.2)
    release.set()
    a.join(5)
    b.join(5)
    assert ("owner", "unreadable extract") in outcomes
    assert ("waiter", "rebuilt") in outcomes


def test_a_builder_that_asks_for_its_own_key_does_not_wait_on_itself():
    cache = sc.BoundedCache("single_flight_reentrant", max_entries=8)

    def build():
        return cache.get_or_build("k", lambda: "inner")

    assert cache.get_or_build("k", build) == "inner"


# -- the persistent tier ----------------------------------------------------- #
def test_a_restart_reads_the_value_back_instead_of_building(persistent):
    calls = []
    first = _restarted("persist_restart")
    assert first.get_or_build("k", lambda: calls.append(1) or {"rows": 3}) == {"rows": 3}
    again = _restarted("persist_restart")
    assert again.get_or_build("k", lambda: calls.append(1) or {"rows": 99}) == {"rows": 3}
    assert calls == [1]
    assert again.disk_hits == 1


def test_another_build_never_reads_this_builds_entries(persistent, monkeypatch):
    _restarted("persist_build").get_or_build("k", lambda: {"v": "old code"})
    monkeypatch.setenv("TRAKT_BUILD_COMMIT", "b" * 40)
    build_info_mod.build_info.cache_clear()
    assert _restarted("persist_build").get_or_build(
        "k", lambda: {"v": "new code"}) == {"v": "new code"}


def test_only_a_value_json_returns_unchanged_is_written(persistent):
    _restarted("persist_tuple").get_or_build("k", lambda: {"window": (8, 12)})
    assert not list(persistent.rglob("*.json"))
    _restarted("persist_dict").get_or_build("k", lambda: {"window": [8, 12]})
    written = list(persistent.rglob("*.json"))
    assert len(written) == 1
    assert json.loads(written[0].read_text())["value"] == {"window": [8, 12]}


def test_a_memory_only_cache_writes_nothing(persistent):
    sc.BoundedCache("memory_only", max_entries=8).get_or_build("k", lambda: {"v": 1})
    assert not list(persistent.rglob("*.json"))


def test_the_tier_is_off_without_a_build_stamp_or_when_switched_off(
        monkeypatch, tmp_path):
    monkeypatch.setenv("TRAKT_SERVING_CACHE_DIR", str(tmp_path))
    monkeypatch.delenv("TRAKT_BUILD_COMMIT", raising=False)
    monkeypatch.setattr(build_info_mod, "_STAMP", tmp_path / "missing.json")
    build_info_mod.build_info.cache_clear()
    try:
        assert sc.persist_dir() is None              # a local, unstamped run
        monkeypatch.setenv("TRAKT_BUILD_COMMIT", "c" * 40)
        build_info_mod.build_info.cache_clear()
        assert sc.persist_dir() == tmp_path / ("c" * 40)
        monkeypatch.setenv("TRAKT_SERVING_CACHE_PERSIST", "off")
        assert sc.persist_dir() is None
    finally:
        build_info_mod.build_info.cache_clear()


def test_other_builds_entries_are_pruned(persistent):
    stale = persistent / ("z" * 40) / "pipeline_extract_summary"
    stale.mkdir(parents=True)
    (stale / "x.json").write_text("{}")
    _restarted("persist_prune").get_or_build("k", lambda: {"v": 1})
    assert sc.prune_other_builds() == 1
    assert not (persistent / ("z" * 40)).exists()
    assert (persistent / ("a" * 40)).exists()


# -- the identity of a mirrored file ------------------------------------------ #
def test_a_mirrored_file_is_identified_by_its_etag_not_its_copy_time(tmp_path):
    data = tmp_path / "pipeline_snapshot.csv"
    sidecar = tmp_path / ("pipeline_snapshot.csv" + sc.ETAG_SIDECAR_SUFFIX)
    data.write_text("a,b\n1,2\n")
    sidecar.write_text('"0x1"')
    first = sc.file_identity(data)
    time.sleep(0.01)
    data.write_text("a,b\n1,2\n")                    # downloaded again, same bytes
    assert sc.file_identity(data) == first == 'etag:"0x1":8'
    sidecar.write_text('"0x2"')                      # republished
    assert sc.file_identity(data) != first


def test_a_file_without_a_sidecar_keeps_the_filesystem_identity(tmp_path):
    data = tmp_path / "extract.csv"
    data.write_text("x")
    st = data.stat()
    assert sc.file_identity(data) == f"{st.st_mtime_ns}:{st.st_size}"
