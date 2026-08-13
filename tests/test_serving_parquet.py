#!/usr/bin/env python3
"""The columnar serving copy: faster, and never a second dataset.

``pd.read_csv`` re-parses and re-infers every column on every cold load — 8.0
seconds at a million rows, on the first request after any restart, any scale-out
event and any dataset change. The serving copy makes that 0.26 seconds.

The whole risk of a cache like this is that it stops agreeing with its source, so
the first test in this file is the one that matters:
``test_the_serving_copy_is_the_same_frame_as_the_csv``. Everything else here is
about degradation — an unwritable directory, a corrupt file, no pyarrow, a
changed CSV — because a serving optimisation must never be able to make the
platform unavailable or, worse, quietly wrong.

Measured on this machine, pyarrow import warmed (a long-lived server has it):

        rows      csv MB   pq MB   read_csv    parquet   speedup
      10,000         2.6     0.8      55 ms       9 ms      6.5x
     100,000        26.0     7.7     719 ms      28 ms     26.1x
   1,000,000       259.6    60.5   7,979 ms     255 ms     31.3x
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from mi_agent_api import serving_parquet as sp

pytest.importorskip("pyarrow")


@pytest.fixture(autouse=True)
def _clean(monkeypatch):
    monkeypatch.delenv(sp.ENABLED_ENV, raising=False)
    monkeypatch.delenv(sp.CACHE_DIR_ENV, raising=False)
    # Cache everything, so the behavioural tests can use small frames. The
    # production threshold has its own test below.
    monkeypatch.setenv(sp.MIN_BYTES_ENV, "0")
    sp.reset_stats()
    yield
    sp.reset_stats()


def _frame(n: int = 40_000) -> pd.DataFrame:
    """Canonical-shaped and deliberately mixed: floats, ints, strings, nulls.

    The null column matters — a column that is entirely empty in the CSV is
    where a round trip most easily changes dtype.
    """
    rng = np.random.default_rng(7)
    return pd.DataFrame({
        "loan_identifier": [f"LN{i:08d}" for i in range(n)],
        "borrower_identifier": [f"BR{i // 3:08d}" for i in range(n)],
        "source_portfolio_id": ["direct_001" if i % 2 else "acquired_001"
                                for i in range(n)],
        "account_status": ["performing" if i % 5 else "arrears" for i in range(n)],
        "current_principal_balance": rng.uniform(50_000, 900_000, n).round(2),
        "current_loan_to_value": rng.uniform(20, 95, n).round(2),
        "number_of_days_in_arrears": rng.integers(0, 200, n),
        "geographic_region_collateral": [["South East", "London", "Wales"][i % 3]
                                         for i in range(n)],
        "current_valuation_date": "2026-06-30",
        "never_populated": [None] * n,
        "reporting_date": "2026-07-31",
    })


def _write_csv(tmp_path: Path, n: int = 40_000) -> Path:
    path = tmp_path / "canonical_typed.csv"
    _frame(n).to_csv(path, index=False)
    return path


# =========================================================================== #
# The claim that matters: the copy is not a second dataset
# =========================================================================== #
def test_the_serving_copy_is_the_same_frame_as_the_csv(tmp_path):
    """A serving copy that differs from its source is not a cache, it is a
    second dataset — and a second dataset is exactly what this architecture
    refuses to have."""
    csv = _write_csv(tmp_path)

    from_csv, first = sp.read_canonical(csv)
    assert first == "miss"
    from_parquet, second = sp.read_canonical(csv)
    assert second == "hit"

    pd.testing.assert_frame_equal(from_csv, from_parquet)
    assert list(from_csv.columns) == list(from_parquet.columns)
    assert from_csv.dtypes.to_dict() == from_parquet.dtypes.to_dict()


def test_dtypes_survive_the_round_trip_rather_than_being_re_inferred(tmp_path):
    """The copy is written FROM the csv-derived frame, so reading it back
    reproduces that frame. This is why it is MORE faithful than a second CSV
    parse, not less: read_csv re-infers, and re-inference is where '1.0', '1'
    and '01' stop agreeing."""
    csv = _write_csv(tmp_path, 5_000)
    baseline = pd.read_csv(csv, low_memory=False)
    sp.read_canonical(csv)
    served, outcome = sp.read_canonical(csv)
    assert outcome == "hit"
    assert served.dtypes.to_dict() == baseline.dtypes.to_dict()
    assert served["never_populated"].isna().all()


def test_the_csv_remains_authoritative_and_the_copy_is_disposable(tmp_path):
    """Deleting every copy costs one slow load and nothing else."""
    csv = _write_csv(tmp_path)
    expected, _ = sp.read_canonical(csv)
    assert sp.read_canonical(csv)[1] == "hit"

    removed = sp.purge(csv, all_revisions=True)
    assert removed == 1
    rebuilt, outcome = sp.read_canonical(csv)
    assert outcome == "miss"
    pd.testing.assert_frame_equal(expected, rebuilt)


# =========================================================================== #
# Invalidation: by construction, not by TTL
# =========================================================================== #
def test_a_changed_csv_is_never_served_from_the_old_copy(tmp_path):
    """Content identity is in the FILENAME, so a stale copy is not invalidated —
    it is simply never asked for. There is no invalidation to get wrong."""
    csv = _write_csv(tmp_path, 5_000)
    sp.read_canonical(csv)
    assert sp.read_canonical(csv)[1] == "hit"
    stale_path = sp.serving_path(csv)

    time.sleep(0.01)
    _frame(6_000).to_csv(csv, index=False)      # the source changed

    frame, outcome = sp.read_canonical(csv)
    assert outcome == "miss"
    assert len(frame) == 6_000
    assert sp.serving_path(csv) != stale_path
    # The old copy still exists on disk and is simply orphaned — never read.
    assert stale_path.exists()


def test_the_format_version_is_part_of_the_key(tmp_path):
    """Changing what the copy MEANS must not read old copies back."""
    csv = _write_csv(tmp_path, 5_000)
    assert f"__v{sp.SERVING_FORMAT_VERSION}__" in sp.serving_path(csv).name


# =========================================================================== #
# Degradation: never an outage, never a wrong answer
# =========================================================================== #
def test_an_unwritable_cache_directory_falls_back_to_the_csv(tmp_path,
                                                             monkeypatch):
    csv = _write_csv(tmp_path, 5_000)
    monkeypatch.setenv(sp.CACHE_DIR_ENV, "/proc/nonexistent/cannot-write-here")

    frame, outcome = sp.read_canonical(csv)
    assert outcome == "miss"
    assert len(frame) == 5_000              # served, from the CSV
    assert sp.stats()["write_failed"] == 1


def test_a_corrupt_copy_falls_back_and_rebuilds_itself(tmp_path):
    csv = _write_csv(tmp_path, 5_000)
    expected, _ = sp.read_canonical(csv)
    target = sp.serving_path(csv)
    target.write_bytes(b"this is not a parquet file")

    frame, outcome = sp.read_canonical(csv)
    assert outcome == "miss"
    assert sp.stats()["read_failed"] == 1
    pd.testing.assert_frame_equal(expected, frame)
    # Rebuilt, so the next load is fast again rather than permanently degraded.
    assert sp.read_canonical(csv)[1] == "hit"


def test_a_column_parquet_cannot_represent_falls_back_without_failing(tmp_path):
    """A serving optimisation must never be able to make the platform
    unavailable, so an unserialisable frame is a warning and a CSV read."""
    csv = tmp_path / "awkward.csv"
    _frame(5_000).to_csv(csv, index=False)

    def _read_csv_with_a_bad_column(path):
        frame = pd.read_csv(path, low_memory=False)
        # Mixed scalar/mapping in one object column. Arrow refuses this, and it
        # is the realistic shape: a tape column that is a number on most rows and
        # a structure on a few.
        frame["unserialisable"] = [1 if i % 2 else {"k": "v"}
                                   for i in range(len(frame))]
        return frame

    frame, outcome = sp.read_canonical(csv, read_csv=_read_csv_with_a_bad_column)
    assert outcome == "miss"
    assert len(frame) == 5_000
    assert sp.stats()["write_failed"] == 1
    assert not sp.serving_path(csv).exists()


def test_no_partial_file_is_left_behind_when_a_write_fails(tmp_path):
    """Two workers starting together both miss and both write. A third reading a
    half-written Parquet is the one failure worse than having no cache, so the
    write is atomic and a failure leaves nothing."""
    csv = tmp_path / "awkward.csv"
    _frame(5_000).to_csv(csv, index=False)

    def _bad(path):
        frame = pd.read_csv(path, low_memory=False)
        frame["unserialisable"] = [1 if i % 2 else {"k": "v"}
                                   for i in range(len(frame))]
        return frame

    sp.read_canonical(csv, read_csv=_bad)
    cache_dir = sp.serving_path(csv).parent
    leftovers = list(cache_dir.glob(".partial-*")) if cache_dir.exists() else []
    assert leftovers == []


def test_the_kill_switch_restores_exactly_the_previous_behaviour(tmp_path,
                                                                 monkeypatch):
    csv = _write_csv(tmp_path, 5_000)
    monkeypatch.setenv(sp.ENABLED_ENV, "0")

    frame, outcome = sp.read_canonical(csv)
    assert outcome == "disabled"
    assert sp.serving_path(csv) is None or not sp.serving_path(csv).exists()
    pd.testing.assert_frame_equal(frame, pd.read_csv(csv, low_memory=False))


def test_a_small_tape_is_not_cached_because_the_copy_would_be_overhead(
        tmp_path, monkeypatch):
    monkeypatch.setenv(sp.MIN_BYTES_ENV, str(sp.MIN_BYTES_TO_CACHE))
    csv = tmp_path / "small.csv"
    _frame(50).to_csv(csv, index=False)
    frame, outcome = sp.read_canonical(csv)
    assert outcome == "skipped"
    assert len(frame) == 50
    assert not sp.serving_path(csv).exists()


def test_a_missing_source_does_not_raise_from_the_cache_layer(tmp_path):
    missing = tmp_path / "not_here.csv"
    assert sp.source_identity(missing) is None
    assert sp.serving_path(missing) is None
    with pytest.raises(FileNotFoundError):
        sp.read_canonical(missing)          # the CSV read's own error, unchanged


# =========================================================================== #
# The measurement that justifies it
# =========================================================================== #
def test_the_serving_copy_is_materially_faster_and_smaller(tmp_path, capsys):
    """Recorded rather than merely asserted. The threshold is deliberately loose
    (2x) — this runs on shared CI hardware and the point is the ORDER of the
    difference, not a precise figure."""
    import pyarrow  # noqa: F401 - warm the import, as a long-lived server has

    csv = _write_csv(tmp_path, 120_000)
    sp.read_canonical(csv)                                   # build the copy

    def _best(fn, rounds=3):
        return min((lambda t0: (fn(), (time.perf_counter() - t0) * 1000)[1])(
            time.perf_counter()) for _ in range(rounds))

    csv_ms = _best(lambda: pd.read_csv(csv, low_memory=False))
    parquet_ms = _best(lambda: sp.read_canonical(csv)[0])
    copy = sp.serving_path(csv)

    with capsys.disabled():
        print(f"\n    read_csv      : {csv_ms:8.0f} ms  "
              f"({csv.stat().st_size / 1e6:.1f} MB)"
              f"\n    serving copy  : {parquet_ms:8.0f} ms  "
              f"({copy.stat().st_size / 1e6:.1f} MB)"
              f"\n    improvement   : {csv_ms / max(parquet_ms, 1e-6):8.1f}x")

    assert parquet_ms < csv_ms / 2, (
        f"the serving copy gained only {csv_ms / max(parquet_ms, 1e-6):.1f}x")
    assert copy.stat().st_size < csv.stat().st_size


def test_the_outcome_is_reported_so_a_slow_request_is_explainable(tmp_path):
    """'Why was this request slow' should be answerable from the response, not
    only from a log the caller cannot read."""
    csv = _write_csv(tmp_path, 5_000)
    assert sp.read_canonical(csv)[1] == "miss"
    assert sp.read_canonical(csv)[1] == "hit"
    counters = sp.stats()
    assert counters["miss"] == 1 and counters["hit"] == 1 and counters["write"] == 1
