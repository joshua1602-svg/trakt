#!/usr/bin/env python3
"""
blob_trigger.py - Cloud entry point for the trakt pipeline.

Watches a blob storage container for new data-tape uploads, determines the
pipeline mode from the upload path convention, runs the orchestrator, and
uploads results back to blob storage.

Folder convention (inside the container):
    uploads/{client_id}/mi/          → mode=mi
    uploads/{client_id}/annex12/     → mode=annex12
    uploads/{client_id}/regulatory/  → mode=regulatory, regime inferred from filename or metadata

The trigger can run as:
  1. Azure Function       (azure_handler)
  2. AWS Lambda           (lambda_handler)
  3. CLI for local testing (python blob_trigger.py --provider local --path tape.csv --mode mi)

Environment variables (cloud):
    TRAKT_STORAGE_PROVIDER   azure | aws        (default: azure)
    TRAKT_STORAGE_CONN_STR   connection string  (Azure) or bucket name (AWS)
    TRAKT_CONTAINER          container / bucket  (default: "trakt-data")
    TRAKT_DEFAULT_REGIME     fallback regime     (default: "ESMA_Annex2")
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# Path constants
# ---------------------------------------------------------------------------
ORCHESTRATOR = Path(__file__).resolve().parent / "trakt_run.py"
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
CONFIG_ROOT  = PROJECT_ROOT / "config"


# ---------------------------------------------------------------------------
# Mode inference from blob path
# ---------------------------------------------------------------------------

def infer_mode_and_regime(blob_path: str) -> tuple[str, Optional[str]]:
    """
    Infer pipeline mode and regime from the blob upload path.

    Convention:
        uploads/<client>/mi/<file>          → ("mi", None)
        uploads/<client>/annex12/<file>      → ("annex12", None)
        uploads/<client>/regulatory/<file>   → ("regulatory", <regime>)

    For regulatory mode the regime is read from:
      1. Filename suffix:  tape_ESMA_Annex2.csv  → ESMA_Annex2
      2. Env fallback:     TRAKT_DEFAULT_REGIME   → ESMA_Annex2
    """
    parts = blob_path.replace("\\", "/").lower().split("/")

    if "annex12" in parts:
        return "annex12", None
    if "regulatory" in parts:
        regime = _extract_regime_from_filename(blob_path)
        return "regulatory", regime
    # Default to MI if path doesn't match a regulatory pattern
    return "mi", None


def _extract_regime_from_filename(blob_path: str) -> str:
    """Try to pull ESMA_AnnexN from the filename, else use env default."""
    import re
    match = re.search(r"(ESMA_Annex\d)", blob_path, re.IGNORECASE)
    if match:
        return match.group(1)
    return os.environ.get("TRAKT_DEFAULT_REGIME", "ESMA_Annex2")


# ---------------------------------------------------------------------------
# Storage adapters
# ---------------------------------------------------------------------------

class StorageAdapter:
    """Base class for blob storage providers."""

    def download(self, blob_path: str, local_path: Path) -> None:
        raise NotImplementedError

    def upload(self, local_path: Path, blob_path: str) -> None:
        raise NotImplementedError

    def list_blobs(self, prefix: str) -> list[str]:
        raise NotImplementedError


class AzureBlobAdapter(StorageAdapter):
    """Azure Blob Storage via azure-storage-blob SDK."""

    def __init__(self) -> None:
        from azure.storage.blob import BlobServiceClient
        conn_str = os.environ["TRAKT_STORAGE_CONN_STR"]
        container = os.environ.get("TRAKT_CONTAINER", "trakt-data")
        self.client = BlobServiceClient.from_connection_string(conn_str)
        self.container = self.client.get_container_client(container)

    def download(self, blob_path: str, local_path: Path) -> None:
        blob = self.container.get_blob_client(blob_path)
        with open(local_path, "wb") as f:
            f.write(blob.download_blob().readall())

    def upload(self, local_path: Path, blob_path: str) -> None:
        blob = self.container.get_blob_client(blob_path)
        with open(local_path, "rb") as f:
            blob.upload_blob(f, overwrite=True)

    def list_blobs(self, prefix: str) -> list[str]:
        return [b.name for b in self.container.list_blobs(name_starts_with=prefix)]


class AwsS3Adapter(StorageAdapter):
    """AWS S3 via boto3 SDK."""

    def __init__(self) -> None:
        import boto3
        bucket_name = os.environ.get("TRAKT_CONTAINER", "trakt-data")
        self.s3 = boto3.client("s3")
        self.bucket = bucket_name

    def download(self, blob_path: str, local_path: Path) -> None:
        self.s3.download_file(self.bucket, blob_path, str(local_path))

    def upload(self, local_path: Path, blob_path: str) -> None:
        self.s3.upload_file(str(local_path), self.bucket, blob_path)

    def list_blobs(self, prefix: str) -> list[str]:
        resp = self.s3.list_objects_v2(Bucket=self.bucket, Prefix=prefix)
        return [o["Key"] for o in resp.get("Contents", [])]


class LocalAdapter(StorageAdapter):
    """Local filesystem adapter for testing."""

    def download(self, blob_path: str, local_path: Path) -> None:
        shutil.copy2(blob_path, local_path)

    def upload(self, local_path: Path, blob_path: str) -> None:
        Path(blob_path).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(local_path, blob_path)

    def list_blobs(self, prefix: str) -> list[str]:
        p = Path(prefix)
        if p.is_dir():
            return [str(f) for f in p.rglob("*") if f.is_file()]
        return []


def get_adapter(provider: Optional[str] = None) -> StorageAdapter:
    provider = provider or os.environ.get("TRAKT_STORAGE_PROVIDER", "azure")
    if provider == "azure":
        return AzureBlobAdapter()
    if provider == "aws":
        return AwsS3Adapter()
    if provider == "local":
        return LocalAdapter()
    raise ValueError(f"Unknown storage provider: {provider}")


# ---------------------------------------------------------------------------
# Pipeline runner
# ---------------------------------------------------------------------------

def run_pipeline(
    input_path: Path,
    mode: str,
    regime: Optional[str] = None,
    config: Optional[str] = None,
    out_dir: Optional[Path] = None,
) -> dict:
    """
    Invoke the orchestrator as a subprocess and return the run manifest.
    """
    if out_dir is None:
        out_dir = input_path.parent / "out"

    val_dir = input_path.parent / "out_validation"

    cmd = [
        sys.executable, str(ORCHESTRATOR),
        "--mode", mode,
        "--input", str(input_path),
        "--out-dir", str(out_dir),
        "--validation-out-dir", str(val_dir),
    ]

    if mode == "annex12":
        cfg = config or str(CONFIG_ROOT / "client" / "config_client_annex12.yaml")
        cmd.extend(["--config", cfg])

    if mode == "regulatory" and regime:
        cmd.extend(["--regime", regime])

    env = dict(os.environ)
    env["PYTHONUTF8"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"

    result = subprocess.run(cmd, env=env)
    if result.returncode != 0:
        raise RuntimeError(f"Pipeline failed with exit code {result.returncode}")

    manifest_path = out_dir / "run_manifest.json"
    if manifest_path.exists():
        return json.loads(manifest_path.read_text(encoding="utf-8"))
    return {"status": "completed", "exit_code": result.returncode}


def upload_results(adapter: StorageAdapter, out_dir: Path, dest_prefix: str) -> list[str]:
    """Upload all output artefacts to blob storage."""
    uploaded = []
    for f in out_dir.rglob("*"):
        if f.is_file():
            blob_path = f"{dest_prefix}/{f.relative_to(out_dir)}"
            adapter.upload(f, blob_path)
            uploaded.append(blob_path)
    return uploaded


# ---------------------------------------------------------------------------
# Cloud handlers
# ---------------------------------------------------------------------------

def handle_blob_event(blob_path: str, provider: Optional[str] = None) -> dict:
    """
    Core handler: download tape, run pipeline, upload results.
    Called by cloud-specific handlers below.
    """
    adapter = get_adapter(provider)
    mode, regime = infer_mode_and_regime(blob_path)

    with tempfile.TemporaryDirectory(prefix="trakt_") as tmpdir:
        tmpdir = Path(tmpdir)
        local_input = tmpdir / Path(blob_path).name
        out_dir = tmpdir / "out"
        out_dir.mkdir()

        # Download input tape
        adapter.download(blob_path, local_input)

        # Run pipeline
        manifest = run_pipeline(local_input, mode, regime=regime, out_dir=out_dir)

        # Upload results alongside the input
        dest_prefix = str(Path(blob_path).parent / "results" / manifest.get("run_id", "run"))
        uploaded = upload_results(adapter, out_dir, dest_prefix)

        # Also upload validation outputs
        val_dir = tmpdir / "out_validation"
        if val_dir.exists():
            uploaded += upload_results(adapter, val_dir, f"{dest_prefix}/validation")

    return {
        "mode": mode,
        "regime": regime,
        "manifest": manifest,
        "uploaded": uploaded,
    }


def azure_handler(msg) -> None:
    """Azure Function blob trigger entry point."""
    blob_path = msg.get_body().decode("utf-8") if hasattr(msg, "get_body") else str(msg)
    result = handle_blob_event(blob_path, provider="azure")
    print(json.dumps(result, indent=2))


def lambda_handler(event, context):
    """AWS Lambda S3 trigger entry point."""
    for record in event.get("Records", []):
        blob_path = record["s3"]["object"]["key"]
        result = handle_blob_event(blob_path, provider="aws")
        return {"statusCode": 200, "body": json.dumps(result)}


# ---------------------------------------------------------------------------
# CLI for local testing
# ---------------------------------------------------------------------------

def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(description="trakt blob trigger (local testing)")
    ap.add_argument("--path", required=True, help="Path to input tape (local file or blob path)")
    ap.add_argument("--mode", choices=["mi", "annex12", "regulatory"], default=None,
                     help="Override mode (default: infer from path)")
    ap.add_argument("--regime", default=None, help="Regime for regulatory mode")
    ap.add_argument("--config", default=None, help="Annex 12 config YAML override")
    ap.add_argument("--provider", choices=["local", "azure", "aws"], default="local")
    ap.add_argument("--out-dir", default="out", help="Output directory")
    args = ap.parse_args()

    if args.provider == "local":
        # Skip blob download/upload, just run the pipeline directly
        input_path = Path(args.path)
        if not input_path.exists():
            raise FileNotFoundError(f"Input not found: {input_path}")

        mode = args.mode
        regime = args.regime
        if not mode:
            mode, regime = infer_mode_and_regime(args.path)

        out_dir = Path(args.out_dir)
        manifest = run_pipeline(input_path, mode, regime=regime, config=args.config, out_dir=out_dir)
        print(json.dumps(manifest, indent=2))
    else:
        result = handle_blob_event(args.path, provider=args.provider)
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
