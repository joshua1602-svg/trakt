"""
Azure Blob Trigger for the trakt pipeline.

Upload a CSV to the "inbound" container and the pipeline runs automatically.
Outputs are written back to the "outbound" container.

Folder convention for mode routing:
    inbound/mi/<file>.csv           → --mode mi
    inbound/annex12/<file>.csv      → --mode annex12
    inbound/regulatory/<file>.csv   → --mode regulatory   (requires metadata)
    inbound/<file>.csv              → --mode mi  (default)

For annex12 mode, set blob metadata:
    config = <path to annex12 config yaml>

For regulatory mode, set blob metadata:
    regime = ESMA_Annex2   (or Annex3, Annex4, Annex8, Annex9)
"""

from __future__ import annotations

import logging
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import azure.functions as func
from azure.storage.blob import BlobServiceClient

app = func.FunctionApp()

PROJECT_ROOT = Path(__file__).resolve().parent
ORCHESTRATOR = PROJECT_ROOT / "engine" / "orchestrator" / "trakt_run.py"


def _parse_mode_from_path(blob_path: str) -> str:
    """Derive pipeline mode from the blob's folder structure."""
    parts = Path(blob_path).parts  # e.g. ("inbound", "annex12", "tape.csv")
    if len(parts) >= 2:
        folder = parts[-2].lower()
        if folder in ("mi", "annex12", "regulatory"):
            return folder
    return "mi"


def _upload_outputs(out_dir: Path, container: str, prefix: str) -> list[str]:
    """Upload all files in out_dir to the outbound container."""
    conn_str = os.environ["AzureWebJobsStorage"]
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


@app.blob_trigger(
    arg_name="blob",
    path="inbound/{name}",
    connection="AzureWebJobsStorage",
    source="EventGrid",
)
def trakt_blob_trigger(blob: func.InputStream):
    filename = Path(blob.name).name
    blob_path = blob.name  # e.g. "inbound/mi/tape_2026Q1.csv"

    logging.info(f"Blob trigger fired: {blob_path} ({blob.length} bytes)")

    # -- Resolve mode from folder path ------------------------------------
    mode = _parse_mode_from_path(blob_path)

    # -- Write blob to temp file ------------------------------------------
    tmp_dir = tempfile.mkdtemp(prefix="trakt_")
    input_path = Path(tmp_dir) / filename
    input_path.write_bytes(blob.read())

    out_dir = Path(tmp_dir) / "out"
    val_dir = Path(tmp_dir) / "out_validation"

    # -- Build command ----------------------------------------------------
    cmd = [
        sys.executable, str(ORCHESTRATOR),
        "--mode", mode,
        "--input", str(input_path),
        "--out-dir", str(out_dir),
        "--validation-out-dir", str(val_dir),
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
    logging.info(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)

    logging.info(result.stdout)
    if result.returncode != 0:
        logging.error(result.stderr)
        raise RuntimeError(
            f"Pipeline failed (exit {result.returncode}):\n{result.stderr}"
        )

    # -- Upload outputs to outbound container -----------------------------
    stem = input_path.stem
    uploaded = _upload_outputs(out_dir, "outbound", f"{stem}/out")
    uploaded += _upload_outputs(val_dir, "outbound", f"{stem}/out_validation")

    logging.info(
        f"Pipeline complete: {len(uploaded)} artifacts uploaded to "
        f"outbound/{stem}/"
    )
