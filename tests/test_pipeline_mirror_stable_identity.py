"""A mirrored weekly extract keeps its identity until its bytes change.

Every prepared frame, extract summary and history model is cached on its source
file's identity, and a mirrored file's identity was the time it was copied. So
each new weekly extract re-downloaded the whole history, each worker's first
mirror overwrote the other's files, and each restart copied them again — and
every one emptied every pipeline cache (a cold weekly series prepares ~90
extracts: 135 s on 2026-10-02). The mirror now downloads only what changed and
records the store's ETag beside each file, and that ETag is the identity.
"""
from __future__ import annotations

import tempfile

from tests.test_pipeline_blob_root_discovery import _BlobRootFixture
from trakt_core import perf


def _fresh_worker():
    """What a second worker or a new container starts with: no mirror memo."""
    from mi_agent_api import datasets
    datasets._PIPELINE_MIRROR_CACHE.update(root=None, sig=None, local=None)
    return datasets


def _identities(local):
    from pathlib import Path
    from mi_agent_api import serving_cache
    return {str(p.relative_to(local)): serving_cache.file_identity(p)
            for p in Path(local).rglob("pipeline_snapshot.csv")}


def test_a_second_worker_downloads_nothing_and_keeps_every_identity():
    with tempfile.TemporaryDirectory() as td:
        fx = _BlobRootFixture(td)
        with fx.env():
            local = _fresh_worker()._materialise_pipeline_root_uncached(fx.ROOT)
            before = _identities(local)
            assert len(before) == len(fx.DATES)
            assert all(i.startswith("etag:") for i in before.values())
            with perf.collect(route="t") as c:
                assert _fresh_worker()._materialise_pipeline_root_uncached(fx.ROOT) == local
                counters = c.snapshot()["counters"]
            assert "storage.filesystem.download_file" not in counters
            assert _identities(local) == before


def test_a_republished_extract_is_the_only_one_downloaded_and_reidentified():
    with tempfile.TemporaryDirectory() as td:
        fx = _BlobRootFixture(td)
        with fx.env():
            local = _fresh_worker()._materialise_pipeline_root_uncached(fx.ROOT)
            before = _identities(local)
            target = (fx.local_blob_root / "processed-v2" / "pipeline" / "ERE"
                      / fx.DATES[-1] / "pipeline_snapshot.csv")
            target.write_text(target.read_text() + "\n")
            with perf.collect(route="t") as c:
                _fresh_worker()._materialise_pipeline_root_uncached(fx.ROOT)
                counters = c.snapshot()["counters"]
            assert counters.get("storage.filesystem.download_file") == 1
            after = _identities(local)
            changed = {k for k in before if before[k] != after[k]}
            assert changed == {f"{fx.DATES[-1]}/pipeline_snapshot.csv"}


def test_a_snapshot_withdrawn_from_storage_stops_being_history():
    with tempfile.TemporaryDirectory() as td:
        fx = _BlobRootFixture(td)
        with fx.env():
            local = _fresh_worker()._materialise_pipeline_root_uncached(fx.ROOT)
            gone = (fx.local_blob_root / "processed-v2" / "pipeline" / "ERE"
                    / fx.DATES[0] / "pipeline_snapshot.csv")
            gone.unlink()
            _fresh_worker()._materialise_pipeline_root_uncached(fx.ROOT)
            assert f"{fx.DATES[0]}/pipeline_snapshot.csv" not in _identities(local)
            assert len(_identities(local)) == len(fx.DATES) - 1
