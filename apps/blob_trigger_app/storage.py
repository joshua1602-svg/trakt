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

import functools
import logging
import os
import shutil
import traceback
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

BLOB_SCHEME = "blob://"

logger = logging.getLogger("trakt.blob_trigger.storage")


def _observed(op: str) -> Callable:
    """Count and time a storage READ primitive on the active perf collector.

    A pure no-op unless a caller has opened a :func:`trakt_core.perf.collect`
    block — which only the MI API middleware does. Ingestion therefore pays a
    single ``ContextVar.get()`` per call and behaves identically to before.
    Never alters the return value and never raises on its own account.

    Only ``list`` / ``etag`` / ``download_file`` / ``read_bytes`` / ``exists``
    are instrumented: those are the round-trips the serving layer is being tuned
    against. ``read_text`` is not counted directly — on the Blob backend it
    delegates to ``read_bytes``, so counting both would double-count one
    round-trip.
    """
    def decorate(fn: Callable) -> Callable:
        @functools.wraps(fn)
        def wrapper(self, *args, **kwargs):
            try:
                from trakt_core import perf
            except Exception:  # noqa: BLE001 - storage must not depend on perf
                return fn(self, *args, **kwargs)
            collector = perf.active()
            if collector is None:
                return fn(self, *args, **kwargs)
            import time as _time
            t0 = _time.perf_counter()
            try:
                return fn(self, *args, **kwargs)
            finally:
                try:
                    collector.add_count(
                        f"storage.{self._backend_name}.{op}", 1,
                        (_time.perf_counter() - t0) * 1000.0)
                except Exception:  # noqa: BLE001
                    pass
        return wrapper
    return decorate


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
    @_observed("exists")
    def exists(self, uri: str) -> bool:
        return self._local_path(uri).exists()

    def read_text(self, uri: str) -> str:
        return self._local_path(uri).read_text(encoding="utf-8")

    @_observed("read_bytes")
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

    @_observed("create_exclusive")
    def create_exclusive(self, uri: str, text: str) -> bool:
        """Create ``uri`` only if it does not exist. ``True`` if this call created it.

        The concurrency primitive the append-only stores are built on. It is
        deliberately *not* ``exists()`` followed by ``write_text()``: that pair
        has a window between the check and the write in which another writer can
        create the same URI, and both writers then believe they won.

        Filesystem: ``O_CREAT | O_EXCL``, which the kernel guarantees is atomic
        even across processes. Blob: ``overwrite=False``, which is a conditional
        PUT server-side. Both backends have a native answer, so nothing here
        needs a lock, a lease or a coordination service.
        """
        with _write_guard("create_exclusive", uri, self._backend_name):
            p = self._local_path(uri)
            p.parent.mkdir(parents=True, exist_ok=True)
            try:
                fd = os.open(p, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
            except FileExistsError:
                return False
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as handle:
                    handle.write(text)
            except Exception:
                # A partially written exclusive file is worse than none: the
                # loser of the next race would chain onto a truncated record.
                try:
                    p.unlink()
                except OSError:
                    pass
                raise
        return True

    def upload_file(self, local_path: str | os.PathLike, uri: str) -> str:
        with _write_guard(f"upload_file<-{local_path}", uri, self._backend_name):
            p = self._local_path(uri)
            p.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(str(local_path), str(p))
        return uri

    @_observed("download_file")
    def download_file(self, uri: str, local_path: str | os.PathLike) -> Path:
        with _write_guard(f"download_file->{local_path}", uri, self._backend_name):
            dest = Path(local_path)
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(str(self._local_path(uri)), str(dest))
        return dest

    @_observed("list")
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

    @_observed("etag")
    def etag(self, uri: str) -> Optional[str]:
        """A cheap change-token for ``uri`` (no full read): filesystem uses
        ``mtime_ns:size``. ``None`` when the target does not exist. Callers use it
        to detect that a re-published artifact changed and re-read only then."""
        p = self._local_path(uri)
        try:
            st = p.stat()
        except OSError:
            return None
        return f"{st.st_mtime_ns}:{st.st_size}"


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

    @_observed("exists")
    def exists(self, uri: str) -> bool:
        return self._client(uri).exists()

    @_observed("read_bytes")
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

    @_observed("create_exclusive")
    def create_exclusive(self, uri: str, text: str) -> bool:
        """Conditional PUT: ``overwrite=False`` makes the service reject a blob
        that already exists, which is the same guarantee ``O_CREAT | O_EXCL``
        gives on the filesystem — evaluated at the storage account, so it holds
        across processes, hosts and scale-out instances."""
        with _write_guard("create_exclusive", uri, self._backend_name):
            try:
                self._client(uri).upload_blob(text.encode("utf-8"),
                                              overwrite=False)
            except Exception as exc:  # noqa: BLE001 - narrowed below
                if type(exc).__name__ == "ResourceExistsError":
                    return False
                raise
        return True

    def upload_file(self, local_path: str | os.PathLike, uri: str) -> str:
        with _write_guard(f"upload_file<-{local_path}", uri, self._backend_name):
            with open(local_path, "rb") as fh:
                self._client(uri).upload_blob(fh, overwrite=True)
        return uri

    def download_file(self, uri: str, local_path: str | os.PathLike) -> Path:
        # NOT decorated: this delegates to ``read_bytes``, which is. Counting
        # both would report two round-trips for one download.
        with _write_guard(f"download_file->{local_path}", uri, self._backend_name):
            dest = Path(local_path)
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(self.read_bytes(uri))
        return dest

    @_observed("list")
    def list(self, prefix_uri: str) -> List[str]:
        container, key = split_blob_uri(prefix_uri)
        cc = self._svc().get_container_client(container)
        return [f"{BLOB_SCHEME}{container}/{b.name}"
                for b in cc.list_blobs(name_starts_with=key)]

    @_observed("etag")
    def etag(self, uri: str) -> Optional[str]:
        """The blob's ETag via a cheap properties HEAD (no download). ``None`` when
        absent — so the MI API re-reads a re-published canonical only when it
        actually changed."""
        try:
            props = self._client(uri).get_blob_properties()
        except Exception:  # noqa: BLE001 — missing blob / transient → treat as no etag
            return None
        return getattr(props, "etag", None) or str(getattr(props, "last_modified", ""))


#: Env vars set by Azure App Service / Functions in the cloud (NOT locally).
#: WEBSITE_SITE_NAME is the reliable marker across plans (incl. Linux / Flex /
#: Elastic Premium) where WEBSITE_INSTANCE_ID may be absent.
_AZURE_MARKERS = ("WEBSITE_INSTANCE_ID", "WEBSITE_SITE_NAME")
#: Connection-string app settings, in priority order. AzureWebJobsStorage is the
#: Functions built-in and is used only as an in-Azure fallback.
_PRIMARY_CONN_VAR = "TRAKT_BLOB_CONNECTION"
_FALLBACK_CONN_VAR = "AzureWebJobsStorage"


def running_in_azure() -> bool:
    return any(os.environ.get(m) for m in _AZURE_MARKERS)


def _azure_marker_seen() -> Optional[str]:
    return next((m for m in _AZURE_MARKERS if os.environ.get(m)), None)


def _resolve_connection(explicit: Optional[str], *, in_azure: bool) -> "tuple[Optional[str], Optional[str]]":
    """Return (connection_string, source_name). Primary is TRAKT_BLOB_CONNECTION;
    AzureWebJobsStorage is used only as an in-Azure fallback."""
    if explicit:
        return explicit, "argument"
    primary = os.environ.get(_PRIMARY_CONN_VAR)
    if primary:
        return primary, _PRIMARY_CONN_VAR
    if in_azure:
        fb = os.environ.get(_FALLBACK_CONN_VAR)
        if fb:
            return fb, _FALLBACK_CONN_VAR
    return None, None


def decide_backend(connection_string: Optional[str] = None,
                   backend: Optional[str] = None) -> Dict[str, Any]:
    """Pure decision (no construction, never raises) — also used for startup logs."""
    requested = (backend or os.environ.get("TRAKT_STORAGE_BACKEND") or "").strip().lower()
    in_azure = running_in_azure()
    conn, conn_source = _resolve_connection(connection_string, in_azure=in_azure)
    primary_conn_present = bool(connection_string or os.environ.get(_PRIMARY_CONN_VAR))

    if requested == "file":
        chosen, reason = "filesystem", "TRAKT_STORAGE_BACKEND=file (explicit override)"
    elif requested == "blob":
        chosen, reason = "azure_blob", "TRAKT_STORAGE_BACKEND=blob (explicit)"
    elif in_azure:
        chosen, reason = "azure_blob", f"running in Azure ({_azure_marker_seen()} set)"
    elif primary_conn_present:
        chosen, reason = "azure_blob", f"{_PRIMARY_CONN_VAR} present (blob connection configured)"
    else:
        chosen, reason = "filesystem", "no Azure markers and no blob connection string"
    return {
        "backend": chosen, "reason": reason, "in_azure": in_azure,
        "azure_marker": _azure_marker_seen(),
        "connection_detected": bool(conn), "connection_source": conn_source,
        "connection_string": conn, "requested_backend": requested or None,
    }


def open_storage(*, connection_string: Optional[str] = None,
                 local_root: str | os.PathLike | None = None,
                 backend: Optional[str] = None) -> Storage:
    """Factory. Selects Azure Blob vs filesystem and logs the decision.

    Selection (``TRAKT_STORAGE_BACKEND`` overrides; else auto):
      * ``file``  → filesystem (local/dev override);
      * ``blob``  → Azure Blob;
      * **running in Azure** (``WEBSITE_INSTANCE_ID``/``WEBSITE_SITE_NAME``) →
        Azure Blob, ALWAYS — never the read-only wwwroot filesystem;
      * a ``TRAKT_BLOB_CONNECTION`` present → Azure Blob;
      * otherwise → filesystem.

    In Azure a missing connection string is a hard error (we refuse to silently
    write ``blob://`` URIs onto the read-only ``/home/site/wwwroot`` filesystem).
    """
    d = decide_backend(connection_string, backend)
    logger.info(
        "STORAGE BACKEND SELECTED backend=%s reason=%s in_azure=%s azure_marker=%s "
        "connection_detected=%s connection_source=%s requested=%s",
        d["backend"], d["reason"], d["in_azure"], d["azure_marker"],
        d["connection_detected"], d["connection_source"], d["requested_backend"])

    if d["backend"] == "azure_blob":
        if not d["connection_string"]:
            raise ValueError(
                "Azure Blob storage selected but no connection string found. Set "
                f"{_PRIMARY_CONN_VAR} (or {_FALLBACK_CONN_VAR}). Refusing to fall "
                "back to the read-only filesystem in Azure.")
        return BlobStorage(d["connection_string"])

    root = local_root or os.environ.get("TRAKT_LOCAL_BLOB_ROOT") or Path.cwd()
    return Storage(root)
