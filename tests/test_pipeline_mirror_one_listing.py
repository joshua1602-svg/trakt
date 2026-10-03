"""The pipeline mirror revalidates with ONE storage listing (P0 design §26).

The 13:49 check measured `pipeline_root_mirror` at ~12s on every question,
funded ones included: its freshness signature asked storage for each dated
weekly snapshot's ETag, one HEAD per snapshot. The List Blobs response already
carries every blob's ETag, so the signature is built from the listing.
"""
from __future__ import annotations

import tempfile
from datetime import datetime, timezone

from apps.blob_trigger_app import storage as storage_mod
from tests.test_pipeline_blob_root_discovery import _BlobRootFixture
from trakt_core import perf


def test_the_mirror_asks_storage_once_whatever_the_number_of_extracts():
    from mi_agent_api import datasets as app
    with tempfile.TemporaryDirectory() as td:
        fx = _BlobRootFixture(td)
        with fx.env():
            app._PIPELINE_MIRROR_CACHE.update(root=None, sig=None, local=None)
            with perf.collect(route="t") as c:
                local = app._materialise_pipeline_root_uncached(fx.ROOT)
                counters = c.snapshot()["counters"]
            assert local and local != fx.ROOT
            assert counters.get("storage.filesystem.list_etags") == 1
            assert "storage.filesystem.etag" not in counters
            # Unchanged storage: the second request reuses the mirror, still
            # with one listing and no download.
            with perf.collect(route="t") as c:
                assert app._materialise_pipeline_root_uncached(fx.ROOT) == local
                counters = c.snapshot()["counters"]
            assert counters.get("storage.filesystem.list_etags") == 1
            assert "storage.filesystem.read_bytes" not in counters


def test_a_republished_extract_changes_the_signature():
    from mi_agent_api import datasets as app
    from apps.blob_trigger_app.storage import open_storage
    with tempfile.TemporaryDirectory() as td:
        fx = _BlobRootFixture(td)
        with fx.env():
            app._PIPELINE_MIRROR_CACHE.update(root=None, sig=None, local=None)
            app._materialise_pipeline_root_uncached(fx.ROOT)
            before = app._PIPELINE_MIRROR_CACHE["sig"]
            target = (fx.local_blob_root / "processed-v2" / "pipeline" / "ERE"
                      / fx.DATES[-1] / "pipeline_snapshot.csv")
            target.write_text(target.read_text() + "\n")
            app._materialise_pipeline_root_uncached(fx.ROOT)
            assert app._PIPELINE_MIRROR_CACHE["sig"] != before
            listed = open_storage().list_etags(fx.ROOT)
            assert listed[f"{fx.ROOT}{fx.DATES[-1]}/pipeline_snapshot.csv"] == \
                open_storage().etag(f"{fx.ROOT}{fx.DATES[-1]}/pipeline_snapshot.csv")


def test_the_blob_listing_carries_each_blobs_etag(monkeypatch):
    class Item:
        def __init__(self, name, etag):
            self.name, self.etag = name, etag
            self.last_modified = datetime(2026, 9, 24, tzinfo=timezone.utc)

    class Container:
        def list_blobs(self, name_starts_with=""):
            return [Item("pipeline/ERE/2026-09-24/pipeline_snapshot.csv", '"0x1"'),
                    Item("pipeline/ERE/latest/pipeline_snapshot.csv", '"0x2"')]

    class Service:
        def get_container_client(self, _):
            return Container()

    blob = storage_mod.BlobStorage("UseDevelopmentStorage=true")
    monkeypatch.setattr(blob, "_svc", lambda: Service())
    assert blob.list_etags("blob://processed-v2/pipeline/ERE/") == {
        "blob://processed-v2/pipeline/ERE/2026-09-24/pipeline_snapshot.csv": '"0x1"',
        "blob://processed-v2/pipeline/ERE/latest/pipeline_snapshot.csv": '"0x2"'}


def test_the_bank_runner_opens_the_request_scope_the_http_layer_opens(monkeypatch):
    from mi_agent_api import question_bank as qb
    from mi_agent_api import request_scope

    seen = []

    class Result:
        result = {"ok": True, "answer": "x", "metadata": {}}

    def ask(question, **_):
        seen.append(request_scope.active() is not None)
        return Result()

    monkeypatch.setattr(qb, "_ask", ask)
    record = qb.run_one({"id": "q", "category": "c", "question": "q"},
                        portfolio=None, lens=None)
    assert seen == [True]
    assert "storage_calls" in record["timing"]
