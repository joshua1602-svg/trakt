"""
Azure Blob Storage integration for the Streamlit dashboard.

Provides helpers to list and download canonical CSV files from the
"outbound" container on traktstorage so the dashboard can read
pipeline outputs directly from Azure.

Authentication priority:
    1. DATA_STORAGE_CONNECTION  – connection string (same env var as function_app.py)
    2. AZURE_STORAGE_ACCOUNT    – account name → uses DefaultAzureCredential
       (works with Managed Identity on App Service, or az login locally)
"""

from __future__ import annotations

import io
import logging
import os
import tempfile
from datetime import datetime, timezone
import json
from dataclasses import dataclass

# Sentinel used as a sort-key fallback when a blob's last_modified is None.
# Must be a datetime (not 0 / int) to stay in the same type-space and avoid
# TypeError when Python compares keys during list.sort().
_EPOCH = datetime(1970, 1, 1, tzinfo=timezone.utc)

import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Client factories
# ---------------------------------------------------------------------------

_OUTBOUND_CONTAINER = os.environ.get("TRAKT_OUTBOUND_CONTAINER", "outbound")
_PIPELINE_SNAPSHOTS_PREFIX = os.environ.get("TRAKT_PIPELINE_SNAPSHOT_PREFIX", "mi/pipeline_snapshots/")
_PIPELINE_LATEST_POINTER = os.environ.get(
    "TRAKT_PIPELINE_SNAPSHOT_POINTER_BLOB",
    f"{_PIPELINE_SNAPSHOTS_PREFIX.rstrip('/')}/latest_pipeline_snapshot.json",
)


def _get_blob_service_client():
    """Return a BlobServiceClient using connection string or DefaultAzureCredential."""
    conn_str = os.environ.get("DATA_STORAGE_CONNECTION")
    if conn_str:
        from azure.storage.blob import BlobServiceClient
        return BlobServiceClient.from_connection_string(conn_str)

    account_name = os.environ.get("AZURE_STORAGE_ACCOUNT")
    if account_name:
        from azure.identity import DefaultAzureCredential
        from azure.storage.blob import BlobServiceClient
        credential = DefaultAzureCredential()
        account_url = f"https://{account_name}.blob.core.windows.net"
        return BlobServiceClient(account_url, credential=credential)

    raise EnvironmentError(
        "Set DATA_STORAGE_CONNECTION (connection string) or "
        "AZURE_STORAGE_ACCOUNT (account name for Managed Identity) "
        "to connect to Azure Blob Storage."
    )


def _get_container_client(container: str | None = None):
    """Return a ContainerClient for the given (or default outbound) container."""
    client = _get_blob_service_client()
    return client.get_container_client(container or _OUTBOUND_CONTAINER)


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PipelineSnapshot:
    blob_name: str
    last_modified: datetime | None
    size: int | None = None
    etag: str | None = None

def list_canonical_csvs(
    container: str | None = None,
    prefix: str = "",
    dashboard_only: bool = True,
) -> list[str]:
    """
    List CSV blobs in the outbound container.

    When *dashboard_only* is True (the default) only post-transform
    ``_canonical_typed.csv`` files are returned — these use the **active**
    schema produced by MI-mode pipeline runs and are the only files the
    Streamlit dashboard should consume.

    Set *dashboard_only=False* to see every CSV (intermediate artefacts,
    validation reports, full-schema regulatory outputs, etc.).
    """
    cc = _get_container_client(container)
    blobs = [
        b for b in cc.list_blobs(name_starts_with=prefix)
        if b.name.lower().endswith(".csv")
    ]
    if dashboard_only:
        # Use endswith (not `in`) so validation artefacts whose paths contain
        # "canonical_typed" mid-name are never mistakenly included — e.g.
        # out_validation/<stem>_canonical_typed_canonical_violations.csv
        blobs = [b for b in blobs if b.name.lower().endswith("_canonical_typed.csv")]

    # Sort ascending by last_modified so the newest pipeline output is last.
    # The Streamlit selectbox uses index=len-1 to pre-select the final entry.
    # Use _EPOCH (a datetime) as the None-fallback instead of 0 (an int) —
    # mixing types would raise TypeError in Python 3 during comparison.
    blobs.sort(key=lambda b: (b.last_modified or _EPOCH, b.name))
    return [b.name for b in blobs]


def get_most_recent_canonical_csv(
    container: str | None = None,
    prefix: str = "",
) -> str | None:
    """
    Return the blob name of the most recently uploaded ``_canonical_typed.csv``
    in the outbound container, or *None* if no matching file exists.

    Selection rules (per product spec):
      1. File name must end with ``_canonical_typed.csv``.
      2. Most recently uploaded blob wins (highest ``last_modified`` timestamp).

    Unlike :func:`list_canonical_csvs` this function performs a single
    ``max`` pass instead of a full sort, making it slightly cheaper when only
    the latest file is needed.
    """
    cc = _get_container_client(container)
    blobs = [
        b for b in cc.list_blobs(name_starts_with=prefix)
        if b.name.lower().endswith("_canonical_typed.csv")
    ]
    if not blobs:
        return None
    most_recent = max(blobs, key=lambda b: b.last_modified or _EPOCH)
    return most_recent.name


def download_blob_to_dataframe(
    blob_name: str,
    container: str | None = None,
    **read_csv_kwargs,
) -> pd.DataFrame:
    """
    Download a CSV blob and return it as a DataFrame.

    Uses the same read_csv parameters the dashboard already relies on
    (low_memory=False by default).
    """
    cc = _get_container_client(container)
    blob_client = cc.get_blob_client(blob_name)
    data = blob_client.download_blob().readall()
    read_csv_kwargs.setdefault("low_memory", False)
    return pd.read_csv(io.BytesIO(data), **read_csv_kwargs)


def download_blob_bytes(blob_name: str, container: str | None = None) -> bytes:
    """Download a blob and return raw bytes (for non-CSV files)."""
    cc = _get_container_client(container)
    blob_client = cc.get_blob_client(blob_name)
    return blob_client.download_blob().readall()


def list_pipeline_snapshots(
    container: str | None = None,
    prefix: str | None = None,
) -> list[PipelineSnapshot]:
    """List pipeline snapshot CSV blobs, newest first."""
    cc = _get_container_client(container)
    start_prefix = prefix if prefix is not None else _PIPELINE_SNAPSHOTS_PREFIX
    blobs = [
        b for b in cc.list_blobs(name_starts_with=start_prefix)
        if b.name.lower().endswith(".csv")
    ]
    blobs.sort(key=lambda b: (b.last_modified or _EPOCH, b.name), reverse=True)
    return [
        PipelineSnapshot(
            blob_name=b.name,
            last_modified=b.last_modified,
            size=getattr(b, "size", None),
            etag=getattr(b, "etag", None),
        )
        for b in blobs
    ]


def get_latest_pipeline_snapshot(
    container: str | None = None,
    prefix: str | None = None,
) -> PipelineSnapshot | None:
    """Return the latest pipeline snapshot or None when no snapshots exist."""
    snapshots = list_pipeline_snapshots(container=container, prefix=prefix)
    return snapshots[0] if snapshots else None


def download_pipeline_snapshot_to_tempfile(
    blob_name: str,
    container: str | None = None,
) -> str:
    """Download a pipeline snapshot blob into a local tempfile path and return it."""
    data = download_blob_bytes(blob_name=blob_name, container=container)
    tmp = tempfile.NamedTemporaryFile(prefix="pipeline_snapshot_", suffix=".csv", delete=False)
    with tmp:
        tmp.write(data)
    return tmp.name


def register_latest_pipeline_snapshot(
    blob_name: str,
    container: str | None = None,
    *,
    source_blob: str | None = None,
    source_etag: str | None = None,
    last_modified: str | None = None,
    pointer_blob_name: str | None = None,
) -> bool:
    """Write/update pointer metadata for latest pipeline snapshot.

    Returns True when pointer changed, False when existing pointer already
    references the same blob + source etag (idempotent duplicate event).
    """
    cc = _get_container_client(container)
    pointer_blob = pointer_blob_name or _PIPELINE_LATEST_POINTER
    pointer_client = cc.get_blob_client(pointer_blob)

    existing: dict[str, str] = {}
    try:
        existing_raw = pointer_client.download_blob().readall()
        existing = json.loads(existing_raw.decode("utf-8"))
    except Exception:
        existing = {}

    if (
        existing.get("blob_name") == blob_name
        and existing.get("source_etag") == (source_etag or "")
    ):
        logger.info("Pipeline snapshot pointer unchanged for blob=%s etag=%s", blob_name, source_etag)
        return False

    payload = {
        "blob_name": blob_name,
        "source_blob": source_blob or "",
        "source_etag": source_etag or "",
        "registered_at_utc": datetime.now(timezone.utc).isoformat(),
        "last_modified": last_modified or "",
    }
    pointer_client.upload_blob(
        json.dumps(payload, indent=2).encode("utf-8"),
        overwrite=True,
    )
    logger.info("Pipeline snapshot pointer updated: %s -> %s", pointer_blob, blob_name)
    return True


# ---------------------------------------------------------------------------
# Portfolio snapshot helpers (Balance Evolution multi-month support)
# ---------------------------------------------------------------------------

_SNAPSHOTS_PREFIX = "snapshots/"


def write_portfolio_snapshot(
    df: pd.DataFrame,
    as_of_date: str,
    balance_col: str = "total_balance",
    orig_date_col: str = "origination_date",
    ltv_col: str | None = "current_ltv",
    rate_col: str | None = "interest_rate",
    container: str | None = None,
) -> str:
    """Write a compact per-origination-year summary to
    outbound/snapshots/<as_of_date>-summary.csv.

    Called by the dashboard after each data load so the Balance Evolution
    chart accumulates historical months without requiring multi-CSV uploads.
    Returns the blob name written.
    """
    _df = df.copy()

    if orig_date_col in _df.columns:
        _df["_orig_year"] = (
            pd.to_datetime(_df[orig_date_col], errors="coerce").dt.year.astype("Int64")
        )
    elif "origination_year" in _df.columns:
        _df["_orig_year"] = pd.to_numeric(_df["origination_year"], errors="coerce").astype("Int64")
    else:
        _df["_orig_year"] = pd.NA

    _df["_bal"] = pd.to_numeric(_df.get(balance_col, 0), errors="coerce").fillna(0.0)

    agg_spec: dict = {
        "account_count": ("_orig_year", "count"),
        "total_balance": ("_bal", "sum"),
    }
    if ltv_col and ltv_col in _df.columns:
        _df["_ltv"] = pd.to_numeric(_df[ltv_col], errors="coerce")
        agg_spec["avg_ltv"] = ("_ltv", "mean")
    if rate_col and rate_col in _df.columns:
        _df["_rate"] = pd.to_numeric(_df[rate_col], errors="coerce")
        agg_spec["avg_rate"] = ("_rate", "mean")

    snap = _df.groupby("_orig_year", dropna=False).agg(**agg_spec).reset_index()
    snap.rename(columns={"_orig_year": "origination_year"}, inplace=True)
    snap.insert(0, "as_of_date", as_of_date)

    buf = io.BytesIO()
    snap.to_csv(buf, index=False)
    buf.seek(0)

    blob_name = f"{_SNAPSHOTS_PREFIX}{as_of_date}-summary.csv"
    _get_container_client(container).get_blob_client(blob_name).upload_blob(
        buf, overwrite=True
    )
    logger.info("Snapshot written: %s", blob_name)
    return blob_name


def load_all_portfolio_snapshots(container: str | None = None) -> pd.DataFrame:
    """Read every *-summary.csv from the snapshots/ prefix and return a
    concatenated DataFrame ready for the Balance Evolution chart.

    Columns: as_of_date, origination_year, account_count, total_balance,
             avg_ltv (optional), avg_rate (optional).

    Returns an empty DataFrame if no snapshots exist yet.
    """
    cc = _get_container_client(container)
    blob_names = sorted(
        b.name
        for b in cc.list_blobs(name_starts_with=_SNAPSHOTS_PREFIX)
        if b.name.endswith("-summary.csv")
    )
    if not blob_names:
        return pd.DataFrame()

    frames = []
    for name in blob_names:
        raw = cc.get_blob_client(name).download_blob().readall()
        frames.append(pd.read_csv(io.BytesIO(raw)))

    result = pd.concat(frames, ignore_index=True)
    result["as_of_date"] = pd.to_datetime(result["as_of_date"], errors="coerce")
    return result


def upload_file_to_blob(
    local_path: str,
    blob_name: str,
    container: str | None = None,
) -> str:
    """Upload a local file to blob storage and return the destination blob name."""
    cc = _get_container_client(container)
    with open(local_path, "rb") as f:
        cc.get_blob_client(blob_name).upload_blob(f, overwrite=True)
    logger.info("Uploaded blob: %s", blob_name)
    return blob_name


def upload_bytes_to_blob(
    data: bytes,
    blob_name: str,
    container: str | None = None,
) -> str:
    """Upload raw bytes to blob storage and return the destination blob name."""
    cc = _get_container_client(container)
    cc.get_blob_client(blob_name).upload_blob(data, overwrite=True)
    logger.info("Uploaded blob: %s", blob_name)
    return blob_name


def is_azure_configured() -> bool:
    """Return True if Azure blob credentials are available in the environment."""
    return bool(
        os.environ.get("DATA_STORAGE_CONNECTION")
        or os.environ.get("AZURE_STORAGE_ACCOUNT")
    )
