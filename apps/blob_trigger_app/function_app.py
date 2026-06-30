"""apps.blob_trigger_app.function_app — thin Azure Functions blob trigger.

Routing/inference only — NO business logic. On a blob upload it downloads the
file, then delegates to :func:`router.handle_blob_event`, which parses the path,
fingerprints the schema, checks the source registry, decides source-onboarding
vs deterministic processing, invokes the Orchestrator Agent, and writes an event
manifest.

Path convention:
    raw/{client_id}/{dataset}/{frequency}/{source_portfolio_id}/{reporting_period}/{filename}

App settings (see local.settings.example.json):
    TRAKT_BLOB_CONTAINER    blob container watched (default raw; e.g. raw-v2).
                            Referenced as %TRAKT_BLOB_CONTAINER% in the binding
                            path AND read in code to anchor the path parser.
    TRAKT_BLOB_CONNECTION   storage connection (app-setting name, not the string)
    TRAKT_SOURCE_REGISTRY   path to the source registry (default config/source_registry.yaml)
    TRAKT_TRIGGER_OUT       event-log + run output dir (default out/blob_trigger)
"""

from __future__ import annotations

import logging
import os
import tempfile
from pathlib import Path

import azure.functions as func  # type: ignore

from .router import handle_blob_event
from .schema_fingerprint import fingerprint_pack
from .source_registry import SourceRegistry

app = func.FunctionApp()

_REGISTRY_PATH = os.environ.get("TRAKT_SOURCE_REGISTRY", "config/source_registry.yaml")
_OUT_DIR = os.environ.get("TRAKT_TRIGGER_OUT", "out/blob_trigger")
# Read in code to anchor the path parser; the host resolves %TRAKT_BLOB_CONTAINER%
# in the binding path below from the SAME app setting.
_CONTAINER = os.environ.get("TRAKT_BLOB_CONTAINER", "raw")
# Completion sentinel (Option A): the uploader writes this file LAST; only it
# starts processing, against the now-complete reporting folder.
_PACK_MARKER = os.environ.get("TRAKT_PACK_MARKER", "_READY")


def _name_in_container(blob_name: str) -> str:
    """Strip a leading ``{container}/`` from blob.name if present."""
    prefix = _CONTAINER + "/"
    return blob_name[len(prefix):] if blob_name.startswith(prefix) else blob_name


def _download_pack(folder_prefix: str, dest: Path) -> None:
    """Download every non-marker blob under ``folder_prefix`` into ``dest``."""
    from azure.storage.blob import BlobServiceClient  # type: ignore
    conn = os.environ["TRAKT_BLOB_CONNECTION"]
    svc = BlobServiceClient.from_connection_string(conn)
    container = svc.get_container_client(_CONTAINER)
    for b in container.list_blobs(name_starts_with=folder_prefix):
        fname = b.name.rsplit("/", 1)[-1]
        if fname == _PACK_MARKER:
            continue
        (dest / fname).write_bytes(container.download_blob(b.name).readall())


@app.blob_trigger(arg_name="blob", path="%TRAKT_BLOB_CONTAINER%/{name}",
                  connection="TRAKT_BLOB_CONNECTION")
def on_raw_blob_upload(blob: func.InputStream) -> None:
    blob_path = blob.name  # e.g. ".../direct_001/2026-09-30/loan_tape.xlsx" or .../_READY
    filename = Path(blob_path).name
    logging.info("blob trigger: %s (%s bytes)", blob_path, blob.length)
    registry = SourceRegistry(_REGISTRY_PATH)

    # Data-file event → acknowledge as a pack member; do NOT start the pipeline.
    if filename != _PACK_MARKER:
        manifest = handle_blob_event(
            blob_path, registry=registry, out_dir=_OUT_DIR,
            container=_CONTAINER, pack_marker=_PACK_MARKER)
        logging.info("pack member received: %s (status=%s)", blob_path, manifest.get("status"))
        return

    # Marker event → the folder is complete. Download the pack, fingerprint it,
    # and route once against the assembled pack directory.
    name_in_container = _name_in_container(blob_path)
    folder_prefix = name_in_container.rsplit("/", 1)[0] + "/"
    pack_dir = Path(tempfile.mkdtemp(prefix="trakt_pack_"))
    _download_pack(folder_prefix, pack_dir)
    data_files = [p for p in pack_dir.iterdir() if p.is_file()]
    schema = fingerprint_pack(data_files) if data_files else None
    primary = str(sorted(data_files)[0]) if data_files else None

    manifest = handle_blob_event(
        blob_path, registry=registry, out_dir=_OUT_DIR, container=_CONTAINER,
        pack_marker=_PACK_MARKER, input_dir_override=str(pack_dir),
        local_input_path=primary, schema_info=schema)
    logging.info("pack complete → trigger decision: status=%s decision=%s target=%s",
                 manifest.get("status"), manifest.get("decision"),
                 (manifest.get("selected_target") or {}).get("target"))
