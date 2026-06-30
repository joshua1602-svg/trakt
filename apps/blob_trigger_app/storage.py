"""apps.blob_trigger_app.storage — filesystem / Blob storage abstraction.

Production persists registry, approvals, manifests and canonicals to **Azure
Blob Storage**; local dev and tests use the **filesystem**. Both are addressed
by ``blob://{container}/{key}`` URIs so call sites are backend-agnostic:

  * **local / tests** — a :class:`Storage` maps ``blob://c/k`` to
    ``{local_root}/c/k`` on disk (no Azure needed);
  * **Azure** — :class:`BlobStorage` maps the same URI to a real container/blob.

Plain paths (no ``blob://`` scheme) are treated as filesystem paths as-is, so
existing local behaviour is unchanged.
"""

from __future__ import annotations

import logging
import os
import shutil
import traceback
from contextlib import contextmanager
from pathlib import Path
from typing import List, Optional

BLOB_SCHEME = "blob://"

logger = logging.getLogger("trakt.blob_trigger.storage")


@contextmanager
def _write_guard(op: str, uri: str, backend: str):
    """Log the full traceback + the URI of any failing storage write, re-raise.

    This is the seam that turns a silent Azure 'Executed (Failed)' into an
    identifiable first-failing persistence operation in the logs.
    """
    try:
        yield
    except Exception:  # noqa: BLE001 — log then re-raise (never swallow)
        logger.error("STORAGE WRITE FAILED backend=%s op=%s uri=%s\n%s",
                     backend, op, uri, traceback.format_exc())
        raise


def is_blob_uri(uri: str) -> bool:
    return str(uri).startswith(BLOB_SCHEME)


def split_blob_uri(uri: str) -> "tuple[str, str]":
    """``blob://container/key/parts`` → ``(container, "key/parts")``."""
    if not is_blob_uri(uri):
        raise ValueError(f"not a blob uri: {uri!r}")
    rest = uri[len(BLOB_SCHEME):]
    container, _, key = rest.partition("/")
    if not container:
        raise ValueError(f"blob uri missing container: {uri!r}")
    return container, key


def join_uri(base: str, *parts: str) -> str:
    """Join URI/path parts with '/', tolerating a trailing slash on ``base``."""
    out = base.rstrip("/")
    for p in parts:
        out = f"{out}/{str(p).strip('/')}"
    return out


class Storage:
    """Filesystem-backed storage. ``blob://`` URIs map under ``local_root``.

    This is the local/test backend and the base class for :class:`BlobStorage`.
    """

    def __init__(self, local_root: str | os.PathLike | None = None):
        self.local_root = Path(local_root) if local_root else Path.cwd()

    # -- URI → local path -------------------------------------------------- #
    def _local_path(self, uri: str) -> Path:
        if is_blob_uri(uri):
            container, key = split_blob_uri(uri)
            return self.local_root / container / key
        return Path(uri)

    # -- primitives -------------------------------------------------------- #
    def exists(self, uri: str) -> bool:
        return self._local_path(uri).exists()

    def read_text(self, uri: str) -> str:
        return self._local_path(uri).read_text(encoding="utf-8")

    def read_bytes(self, uri: str) -> bytes:
        return self._local_path(uri).read_bytes()

    _backend_name = "filesystem"

    def write_text(self, uri: str, text: str) -> str:
        with _write_guard("write_text", uri, self._backend_name):
            p = self._local_path(uri)
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(text, encoding="utf-8")
        return uri

    def write_bytes(self, uri: str, data: bytes) -> str:
        with _write_guard("write_bytes", uri, self._backend_name):
            p = self._local_path(uri)
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(data)
        return uri

    def upload_file(self, local_path: str | os.PathLike, uri: str) -> str:
        with _write_guard(f"upload_file<-{local_path}", uri, self._backend_name):
            p = self._local_path(uri)
            p.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(str(local_path), str(p))
        return uri

    def download_file(self, uri: str, local_path: str | os.PathLike) -> Path:
        with _write_guard(f"download_file->{local_path}", uri, self._backend_name):
            dest = Path(local_path)
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(str(self._local_path(uri)), str(dest))
        return dest

    def list(self, prefix_uri: str) -> List[str]:
        """Return the URIs of files under ``prefix_uri`` (recursive)."""
        base = self._local_path(prefix_uri)
        if not base.exists():
            return []
        out: List[str] = []
        for f in sorted(base.rglob("*")):
            if f.is_file():
                rel = f.relative_to(base).as_posix()
                out.append(join_uri(prefix_uri, rel))
        return out


class BlobStorage(Storage):
    """Azure Blob-backed storage. ``blob://{container}/{key}`` → real blob.

    ``azure-storage-blob`` is imported lazily so importing this module never
    requires the SDK until a method is called against Azure.
    """

    def __init__(self, connection_string: str):
        super().__init__(local_root=None)
        self._conn = connection_string
        self.__svc = None

    def _svc(self):
        if self.__svc is None:
            from azure.storage.blob import BlobServiceClient  # type: ignore
            self.__svc = BlobServiceClient.from_connection_string(self._conn)
        return self.__svc

    def _client(self, uri: str):
        container, key = split_blob_uri(uri)
        return self._svc().get_blob_client(container, key)

    def exists(self, uri: str) -> bool:
        return self._client(uri).exists()

    def read_bytes(self, uri: str) -> bytes:
        return self._client(uri).download_blob().readall()

    def read_text(self, uri: str) -> str:
        return self.read_bytes(uri).decode("utf-8")

    _backend_name = "azure_blob"

    def write_bytes(self, uri: str, data: bytes) -> str:
        with _write_guard("write_bytes", uri, self._backend_name):
            self._client(uri).upload_blob(data, overwrite=True)
        return uri

    def write_text(self, uri: str, text: str) -> str:
        return self.write_bytes(uri, text.encode("utf-8"))

    def upload_file(self, local_path: str | os.PathLike, uri: str) -> str:
        with _write_guard(f"upload_file<-{local_path}", uri, self._backend_name):
            with open(local_path, "rb") as fh:
                self._client(uri).upload_blob(fh, overwrite=True)
        return uri

    def download_file(self, uri: str, local_path: str | os.PathLike) -> Path:
        with _write_guard(f"download_file->{local_path}", uri, self._backend_name):
            dest = Path(local_path)
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(self.read_bytes(uri))
        return dest

    def list(self, prefix_uri: str) -> List[str]:
        container, key = split_blob_uri(prefix_uri)
        cc = self._svc().get_container_client(container)
        return [f"{BLOB_SCHEME}{container}/{b.name}"
                for b in cc.list_blobs(name_starts_with=key)]


def open_storage(*, connection_string: Optional[str] = None,
                 local_root: str | os.PathLike | None = None,
                 backend: Optional[str] = None) -> Storage:
    """Factory. ``backend='blob'`` (or a connection string + ``backend!='file'``)
    selects Azure Blob; otherwise a filesystem-backed :class:`Storage`.

    Reads ``TRAKT_STORAGE_BACKEND`` / ``TRAKT_BLOB_CONNECTION`` /
    ``TRAKT_LOCAL_BLOB_ROOT`` from the environment when args are omitted.
    """
    backend = (backend or os.environ.get("TRAKT_STORAGE_BACKEND") or "").strip().lower()
    conn = connection_string or os.environ.get("TRAKT_BLOB_CONNECTION")
    if backend == "blob" or (backend != "file" and conn and os.environ.get("WEBSITE_INSTANCE_ID")):
        if not conn:
            raise ValueError("blob storage backend requires TRAKT_BLOB_CONNECTION")
        return BlobStorage(conn)
    root = local_root or os.environ.get("TRAKT_LOCAL_BLOB_ROOT") or Path.cwd()
    return Storage(root)
