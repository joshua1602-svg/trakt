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
    blobs = cc.list_blobs(name_starts_with=prefix)
    names = sorted(
        b.name for b in blobs
        if b.name.lower().endswith(".csv")
    )
    if dashboard_only:
        names = [n for n in names if "canonical_typed" in n.lower()]
    return names


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


def is_azure_configured() -> bool:
    """Return True if Azure blob credentials are available in the environment."""
    return bool(
        os.environ.get("DATA_STORAGE_CONNECTION")
        or os.environ.get("AZURE_STORAGE_ACCOUNT")
    )
