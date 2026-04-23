from __future__ import annotations

from datetime import datetime, timezone
import json

from analytics import blob_storage


class _FakeDownload:
    def __init__(self, payload: bytes):
        self._payload = payload

    def readall(self) -> bytes:
        return self._payload


class _FakeBlobClient:
    def __init__(self, container, name: str):
        self._container = container
        self._name = name

    def download_blob(self):
        if self._name not in self._container.blob_payloads:
            raise FileNotFoundError(self._name)
        return _FakeDownload(self._container.blob_payloads[self._name])

    def upload_blob(self, data, overwrite: bool = False):
        if hasattr(data, "read"):
            payload = data.read()
        else:
            payload = data
        self._container.blob_payloads[self._name] = payload


class _BlobItem:
    def __init__(self, name: str, last_modified: datetime, etag: str = "", size: int = 1):
        self.name = name
        self.last_modified = last_modified
        self.etag = etag
        self.size = size


class _FakeContainer:
    def __init__(self, blobs: list[_BlobItem] | None = None):
        self._blobs = blobs or []
        self.blob_payloads: dict[str, bytes] = {}

    def list_blobs(self, name_starts_with: str = ""):
        return [b for b in self._blobs if b.name.startswith(name_starts_with)]

    def get_blob_client(self, name: str):
        return _FakeBlobClient(self, name)


def test_list_pipeline_snapshots_sorted_newest_first(monkeypatch):
    c = _FakeContainer(
        blobs=[
            _BlobItem("mi/pipeline_snapshots/a.csv", datetime(2026, 1, 1, tzinfo=timezone.utc)),
            _BlobItem("mi/pipeline_snapshots/b.csv", datetime(2026, 1, 2, tzinfo=timezone.utc)),
            _BlobItem("mi/pipeline_snapshots/readme.txt", datetime(2026, 1, 3, tzinfo=timezone.utc)),
        ]
    )
    monkeypatch.setattr(blob_storage, "_get_container_client", lambda container=None: c)

    snapshots = blob_storage.list_pipeline_snapshots()

    assert [s.blob_name for s in snapshots] == [
        "mi/pipeline_snapshots/b.csv",
        "mi/pipeline_snapshots/a.csv",
    ]
    assert blob_storage.get_latest_pipeline_snapshot().blob_name == "mi/pipeline_snapshots/b.csv"


def test_register_latest_pipeline_snapshot_is_idempotent(monkeypatch):
    c = _FakeContainer()
    pointer = "mi/pipeline_snapshots/latest_pipeline_snapshot.json"
    c.blob_payloads[pointer] = json.dumps(
        {
            "blob_name": "mi/pipeline_snapshots/pipeline_abc.csv",
            "source_etag": "etag-1",
        }
    ).encode("utf-8")

    monkeypatch.setattr(blob_storage, "_get_container_client", lambda container=None: c)

    changed = blob_storage.register_latest_pipeline_snapshot(
        blob_name="mi/pipeline_snapshots/pipeline_abc.csv",
        source_etag="etag-1",
    )
    assert changed is False

    changed = blob_storage.register_latest_pipeline_snapshot(
        blob_name="mi/pipeline_snapshots/pipeline_xyz.csv",
        source_etag="etag-2",
        source_blob="inbound/pipeline/weekly.csv",
    )
    assert changed is True

    payload = json.loads(c.blob_payloads[pointer].decode("utf-8"))
    assert payload["blob_name"] == "mi/pipeline_snapshots/pipeline_xyz.csv"
    assert payload["source_etag"] == "etag-2"
