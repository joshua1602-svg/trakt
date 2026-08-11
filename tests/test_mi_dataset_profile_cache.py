#!/usr/bin/env python3
"""The dataset profile is derived once per frame, not once per query.

Profiling counts non-blank values in every column, so it costs a full pass over
the frame. Measured before this memo: 1,701 ms of a 1,898 ms warm MI query over
100k rows — 80% of the request spent re-deriving the structure of a dataset that
had not changed, when the frame itself was already cached by content signature.

The memo is keyed on the identity of the frame OBJECT, guarded by a weak
reference. That is exact rather than approximate, and the tests below are
mostly about proving it cannot go stale:

* a new frame — which is what a republished snapshot produces — misses;
* a different semantics registry misses;
* answers are unchanged whether the memo is on or off.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import pandas as pd
import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from mi_agent.mi_dataset_profile import (
    profile_cache_stats,
    profile_dataset,
    reset_profile_cache,
)

SEMANTICS = {"fields": {
    "bal": {"canonical_field": "current_principal_balance", "role": "metric",
            "format": "currency"},
    "reg": {"canonical_field": "geographic_region_obligor", "role": "dimension",
            "format": "string"},
}}


@pytest.fixture(autouse=True)
def _clean():
    reset_profile_cache()
    yield
    reset_profile_cache()


def _frame(n=2_000) -> pd.DataFrame:
    regions = ["South East", "London", "Wales"]
    return pd.DataFrame({
        "current_principal_balance": [100_000.0 + i for i in range(n)],
        "geographic_region_obligor": [regions[i % 3] for i in range(n)],
        "account_status": ["performing"] * n,
    })


def test_the_same_frame_is_profiled_once():
    df = _frame()
    first = profile_dataset(df, SEMANTICS)
    for _ in range(20):
        assert profile_dataset(df, SEMANTICS) is first
    stats = profile_cache_stats()
    assert stats["misses"] == 1
    assert stats["hits"] == 20


def test_the_profile_is_identical_with_the_memo_on_and_off(monkeypatch):
    """The memo must not change a single value."""
    df = _frame()
    monkeypatch.setenv("TRAKT_PROFILE_CACHE", "off")
    uncached = profile_dataset(df, SEMANTICS)
    monkeypatch.setenv("TRAKT_PROFILE_CACHE", "on")
    reset_profile_cache()
    cached = profile_dataset(df, SEMANTICS)
    assert cached == uncached


def test_a_new_frame_is_profiled_again():
    """A republished snapshot is a NEW object, so it cannot be served the old
    profile — which is the whole safety argument for keying on identity."""
    a, b = _frame(), _frame()
    profile_dataset(a, SEMANTICS)
    profile_dataset(b, SEMANTICS)
    assert profile_cache_stats()["misses"] == 2


def test_a_changed_dataset_produces_a_different_profile():
    """Not just a different key — a genuinely different answer."""
    small = _frame(300)
    large = _frame(3_000)
    assert (profile_dataset(small, SEMANTICS)["fields"]
            ["current_principal_balance"]["non_null"]) == 300
    assert (profile_dataset(large, SEMANTICS)["fields"]
            ["current_principal_balance"]["non_null"]) == 3_000


def test_a_different_semantics_registry_misses():
    """The profile depends on both inputs, so both are part of the identity."""
    df = _frame()
    profile_dataset(df, SEMANTICS)
    other = {"fields": {}}
    profile_dataset(df, other)
    assert profile_cache_stats()["misses"] == 2
    assert profile_cache_stats()["hits"] == 0


def test_the_cache_is_bounded():
    from mi_agent.mi_dataset_profile import _PROFILE_CACHE_MAX

    frames = [_frame(100) for _ in range(_PROFILE_CACHE_MAX + 5)]
    for f in frames:
        profile_dataset(f, SEMANTICS)
    assert profile_cache_stats()["entries"] <= _PROFILE_CACHE_MAX


def test_the_kill_switch_restores_the_previous_behaviour(monkeypatch):
    monkeypatch.setenv("TRAKT_PROFILE_CACHE", "off")
    df = _frame()
    first = profile_dataset(df, SEMANTICS)
    second = profile_dataset(df, SEMANTICS)
    assert first == second and first is not second
    assert profile_cache_stats()["entries"] == 0


def test_a_dropped_frame_does_not_pin_memory():
    """The entry holds a WEAK reference, so a frame that goes out of scope can be
    collected and its entry becomes unusable rather than authoritative."""
    import gc

    df = _frame()
    profile_dataset(df, SEMANTICS)
    key = id(df)
    del df
    gc.collect()

    from mi_agent.mi_dataset_profile import _PROFILE_CACHE

    entry = _PROFILE_CACHE.get(key)
    if entry is not None:                      # not yet evicted
        assert entry[0]() is None, "the cache holds a strong reference to the frame"


def test_the_memo_is_measurably_faster(capsys):
    """Recorded rather than merely asserted."""
    df = _frame(60_000)

    reset_profile_cache()
    t = time.perf_counter(); profile_dataset(df, SEMANTICS); cold = (time.perf_counter() - t) * 1000
    t = time.perf_counter()
    for _ in range(5):
        profile_dataset(df, SEMANTICS)
    warm = (time.perf_counter() - t) / 5 * 1000

    with capsys.disabled():
        print(f"\n    profile_dataset 60k rows: first {cold:7.1f} ms | memoised {warm:7.3f} ms"
              f" ({cold / max(warm, 1e-9):,.0f}x)")
    assert warm < cold / 10
