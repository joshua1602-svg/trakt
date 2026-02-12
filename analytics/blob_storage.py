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
from functools import lru_cache

import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Client factories
# ---------------------------------------------------------------------------

_OUTBOUND_CONTAINER = os.environ.get("TRAKT_OUTBOUND_CONTAINER", "outbound")


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

def list_canonical_csvs(container: str | None = None, prefix: str = "") -> list[str]:
    """
    List CSV blobs in the outbound container.

    Returns blob names like:
        tape/out/canonical_typed.csv
        tape/out/validation_report.csv
    """
    cc = _get_container_client(container)
    blobs = cc.list_blobs(name_starts_with=prefix)
    return sorted(
        b.name for b in blobs
        if b.name.lower().endswith(".csv")
    )


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


def is_azure_configured() -> bool:
    """Return True if Azure blob credentials are available in the environment."""
    return bool(
        os.environ.get("DATA_STORAGE_CONNECTION")
        or os.environ.get("AZURE_STORAGE_ACCOUNT")
    )
