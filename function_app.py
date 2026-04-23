"""
Azure Event Grid Trigger for the trakt pipeline.

Upload a CSV to the "inbound" container on traktstorage and the pipeline
runs automatically. Outputs are written back to the "outbound" container.

Mode routing (checked in order):
    1. Folder prefix:   inbound/regulatory/tape.csv     → --mode regulatory
    2. Filename hint:   inbound/tape_regulatory.csv     → --mode regulatory
    3. Default:         inbound/tape.csv                → --mode mi

Examples:
    inbound/tape.csv                → MI (active schema, dashboard-ready)
    inbound/mi/tape.csv             → MI (explicit)
    inbound/tape_regulatory.csv     → Regulatory (full schema, ESMA Annex 2-9)
    inbound/regulatory/tape.csv     → Regulatory (folder-based)
    inbound/tape_annex12.csv        → Annex 12

For annex12 mode, set app setting:
    TRAKT_ANNEX12_CONFIG = <path to annex12 config yaml>

For regulatory mode, set app setting:
    TRAKT_REGIME = ESMA_Annex2   (or Annex3, Annex4, Annex8, Annex9)
"""

from __future__ import annotations

import logging
import io
import json
import os
import subprocess
import sys
import tempfile
from datetime import timezone
from pathlib import Path

import azure.functions as func
from azure.storage.blob import BlobServiceClient
import pandas as pd

app = func.FunctionApp()

PROJECT_ROOT = Path(__file__).resolve().parent
ORCHESTRATOR = PROJECT_ROOT / "engine" / "orchestrator" / "trakt_run.py"
PIPELINE_INBOUND_PREFIX = os.environ.get("TRAKT_PIPELINE_INBOUND_PREFIX", "pipeline/")
PIPELINE_SNAPSHOT_OUTBOUND_PREFIX = os.environ.get(
    "TRAKT_PIPELINE_SNAPSHOT_PREFIX",
    "mi/pipeline_snapshots/",
)
PIPELINE_LATEST_POINTER_BLOB = os.environ.get(
    "TRAKT_PIPELINE_SNAPSHOT_POINTER_BLOB",
    f"{PIPELINE_SNAPSHOT_OUTBOUND_PREFIX.rstrip('/')}/latest_pipeline_snapshot.json",
)


def _parse_mode_from_path(blob_path: str) -> str:
    """Derive pipeline mode from folder structure or filename.

    Resolution order:
        1. Folder prefix:  inbound/regulatory/tape.csv  → regulatory
        2. Filename hint:  inbound/tape_regulatory.csv   → regulatory
        3. Default:        inbound/tape.csv              → mi (active schema)

    This lets users distinguish runs without creating folders.
    """
    parts = Path(blob_path).parts  # e.g. ("inbound", "annex12", "tape.csv")
    # 1. Folder-based routing
    if len(parts) >= 2:
        folder = parts[-2].lower()
        if folder in ("mi", "annex12", "regulatory"):
            return folder
    # 2. Filename-based fallback
    stem = Path(blob_path).stem.lower()
    for mode in ("regulatory", "annex12"):
        if mode in stem:
            return mode
    return "mi"


def _download_blob(container: str, blob_name: str, dest: Path) -> None:
    """Download a blob from traktstorage to a local path."""
    conn_str = os.environ["DATA_STORAGE_CONNECTION"]
    client = BlobServiceClient.from_connection_string(conn_str)
    blob_client = client.get_blob_client(container, blob_name)
    dest.parent.mkdir(parents=True, exist_ok=True)
    with dest.open("wb") as f:
        f.write(blob_client.download_blob().readall())


def _upload_outputs(out_dir: Path, container: str, prefix: str) -> list[str]:
    """Upload all files in out_dir to the outbound container."""
    conn_str = os.environ["DATA_STORAGE_CONNECTION"]
    client = BlobServiceClient.from_connection_string(conn_str)
    container_client = client.get_container_client(container)

    # Create container if it doesn't exist
    try:
        container_client.create_container()
    except Exception:
        pass  # already exists

    uploaded = []
    for f in out_dir.rglob("*"):
        if f.is_file():
            blob_name = f"{prefix}/{f.relative_to(out_dir)}"
            with f.open("rb") as data:
                container_client.upload_blob(blob_name, data, overwrite=True)
            uploaded.append(blob_name)
    return uploaded


def _copy_blob_between_containers(
    src_container: str,
    src_blob_name: str,
    dst_container: str,
    dst_blob_name: str,
) -> None:
    conn_str = os.environ["DATA_STORAGE_CONNECTION"]
    client = BlobServiceClient.from_connection_string(conn_str)
    src_client = client.get_blob_client(src_container, src_blob_name)
    dst_client = client.get_blob_client(dst_container, dst_blob_name)
    dst_client.upload_blob(src_client.download_blob().readall(), overwrite=True)


def _register_latest_pipeline_snapshot(
    *,
    blob_name: str,
    source_blob: str,
    source_etag: str,
    last_modified: str,
    container: str = "outbound",
) -> bool:
    """Update latest snapshot pointer blob with idempotency on blob+etag."""
    conn_str = os.environ["DATA_STORAGE_CONNECTION"]
    client = BlobServiceClient.from_connection_string(conn_str)
    pointer_client = client.get_blob_client(container, PIPELINE_LATEST_POINTER_BLOB)

    existing: dict[str, str] = {}
    try:
        existing = json.loads(pointer_client.download_blob().readall().decode("utf-8"))
    except Exception:
        existing = {}

    if existing.get("blob_name") == blob_name and existing.get("source_etag") == source_etag:
        logging.info("Pipeline snapshot pointer unchanged for blob=%s etag=%s", blob_name, source_etag)
        return False

    payload = {
        "blob_name": blob_name,
        "source_blob": source_blob,
        "source_etag": source_etag,
        "last_modified": last_modified,
    }
    pointer_client.upload_blob(json.dumps(payload, indent=2).encode("utf-8"), overwrite=True)
    logging.info("Pipeline snapshot pointer updated: %s -> %s", PIPELINE_LATEST_POINTER_BLOB, blob_name)
    return True


def _validate_pipeline_snapshot_csv(container: str, blob_path: str) -> tuple[str, str]:
    conn_str = os.environ["DATA_STORAGE_CONNECTION"]
    client = BlobServiceClient.from_connection_string(conn_str)
    blob_client = client.get_blob_client(container, blob_path)
    props = blob_client.get_blob_properties()
    etag = (props.etag or "").replace('"', "")
    last_modified = (
        props.last_modified.astimezone(timezone.utc).isoformat()
        if props.last_modified
        else ""
    )

    payload = blob_client.download_blob().readall()
    if not payload:
        raise ValueError(f"Pipeline snapshot is empty: {container}/{blob_path}")

    # Basic readability validation: parse header + first rows.
    pd.read_csv(io.BytesIO(payload), nrows=5, low_memory=False)
    return etag, last_modified


def _ingest_pipeline_snapshot(container: str, blob_path: str) -> None:
    if not blob_path.lower().endswith(".csv"):
        logging.info("Skipping pipeline snapshot non-CSV: %s/%s", container, blob_path)
        return

    etag, last_modified = _validate_pipeline_snapshot_csv(container, blob_path)
    stem = Path(blob_path).stem
    safe_etag = etag[:12] if etag else "noetag"
    dst_blob = f"{PIPELINE_SNAPSHOT_OUTBOUND_PREFIX.rstrip('/')}/{stem}_{safe_etag}.csv"

    _copy_blob_between_containers(
        src_container=container,
        src_blob_name=blob_path,
        dst_container="outbound",
        dst_blob_name=dst_blob,
    )
    updated = _register_latest_pipeline_snapshot(
        blob_name=dst_blob,
        source_blob=f"{container}/{blob_path}",
        source_etag=etag,
        last_modified=last_modified,
    )
    if updated:
        logging.info("Registered new pipeline snapshot: outbound/%s", dst_blob)
    else:
        logging.info("Duplicate pipeline snapshot event ignored for etag=%s blob=%s", etag, blob_path)


@app.event_grid_trigger(arg_name="event")
def trakt_blob_trigger(event: func.EventGridEvent):
    data = event.get_json()
    subject = event.subject  # e.g. /blobServices/default/containers/inbound/blobs/mi/tape.csv

    # Extract container and blob path from subject
    # subject format: /blobServices/default/containers/{container}/blobs/{blob_path}
    parts = subject.split("/blobs/", 1)
    if len(parts) != 2:
        logging.warning(f"Unexpected subject format: {subject}")
        return

    container = parts[0].rsplit("/", 1)[-1]  # "inbound"
    blob_path = parts[1]                      # "mi/tape.csv" or "tape.csv"
    filename = Path(blob_path).name           # "tape.csv"

    # Only process blobs from the inbound container
    if container != "inbound":
        logging.info(f"Skipping blob from container '{container}': {blob_path}")
        return

    # Skip non-CSV files
    if not filename.lower().endswith(".csv"):
        logging.info(f"Skipping non-CSV blob: {blob_path}")
        return

    if blob_path.lower().startswith(PIPELINE_INBOUND_PREFIX.lower()):
        logging.info("Pipeline snapshot ingest event detected: %s/%s", container, blob_path)
        _ingest_pipeline_snapshot(container, blob_path)
        return

    logging.info(
        f"Event Grid trigger fired: {container}/{blob_path} "
        f"(url: {data.get('url', 'n/a')})"
    )

    # -- Resolve mode from folder path ------------------------------------
    full_path = f"{container}/{blob_path}"
    mode = _parse_mode_from_path(full_path)

    # -- Download blob to temp file ---------------------------------------
    tmp_dir = tempfile.mkdtemp(prefix="trakt_")
    input_path = Path(tmp_dir) / filename
    _download_blob(container, blob_path, input_path)

    logging.info(f"Downloaded {input_path.stat().st_size} bytes to {input_path}")

    out_dir = Path(tmp_dir) / "out"
    val_dir = Path(tmp_dir) / "out_validation"
    pipeline_out_dir = Path(tmp_dir) / "out_pipeline"

    # -- Build command ----------------------------------------------------
    cmd = [
        sys.executable, str(ORCHESTRATOR),
        "--mode", mode,
        "--input", str(input_path),
        "--out-dir", str(out_dir),
        "--validation-out-dir", str(val_dir),
        "--pipeline-output-dir", str(pipeline_out_dir),
    ]

    # Annex12 requires --config (pass via blob metadata or env var)
    if mode == "annex12":
        config = os.environ.get("TRAKT_ANNEX12_CONFIG")
        if not config:
            raise ValueError(
                "annex12 mode requires TRAKT_ANNEX12_CONFIG app setting"
            )
        cmd.extend(["--config", config])

    # Regulatory requires --regime
    if mode == "regulatory":
        regime = os.environ.get("TRAKT_REGIME")
        if not regime:
            raise ValueError(
                "regulatory mode requires TRAKT_REGIME app setting "
                "(e.g. ESMA_Annex2)"
            )
        cmd.extend(["--regime", regime])

    # -- Run pipeline -----------------------------------------------------
    # Pass the worker's sys.path so the subprocess can find installed packages
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(sys.path)

    logging.info(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True, env=env)

    logging.info(result.stdout)
    if result.returncode != 0:
        logging.error(result.stderr)
        raise RuntimeError(
            f"Pipeline failed (exit {result.returncode}):\n{result.stderr}"
        )

    # -- Upload outputs to outbound container -----------------------------
    # Prefix with mode so MI (active schema) and regulatory (full schema)
    # outputs are kept separate.  The dashboard only browses mi/ files.
    stem = input_path.stem
    uploaded = _upload_outputs(out_dir, "outbound", f"{mode}/{stem}/out")
    uploaded += _upload_outputs(val_dir, "outbound", f"{mode}/{stem}/out_validation")
    if pipeline_out_dir.exists():
        uploaded += _upload_outputs(pipeline_out_dir, "outbound", f"{mode}/{stem}/out_pipeline")

    logging.info(
        f"Pipeline complete: {len(uploaded)} artifacts uploaded to "
        f"outbound/{mode}/{stem}/"
    )
