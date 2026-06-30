"""Azure Functions deployment entrypoint (root) — Event Grid handler.

Azure Functions loads the Function App from the **root** ``function_app.py``.
The Azure event source stays **Event Grid** (an Event Grid subscription on the
storage account delivers blob-created events here). This handler does
**routing/inference only**: it parses the event subject, accepts the configured
container, and delegates to the Azure-free decision core in
``apps.blob_trigger_app.router``.

Flow:
    Blob uploaded to %TRAKT_BLOB_CONTAINER%  (e.g. raw-v2)
        → Event Grid → this handler
        → apps.blob_trigger_app.router.handle_blob_event
        → pack completeness (_READY.json) → source registry → Orchestrator
        → Assembler Agent (central platform canonical refresh)

The legacy handler hardcoded the accepted container to a constant and silently
skipped ``raw-v2``. The accepted container is now configuration
(``TRAKT_BLOB_CONTAINER``), not a constant.

App settings (see local.settings.example.json):
    TRAKT_BLOB_CONTAINER    accepted container (default ``raw``; production ``raw-v2``)
    TRAKT_BLOB_CONNECTION   storage connection (app-setting name, not the string)
    TRAKT_PACK_MARKER       completion marker filename (default ``_READY.json``)
    TRAKT_SOURCE_REGISTRY   source registry path (default config/source_registry.yaml)
    TRAKT_TRIGGER_OUT       writable runtime output root (event log, orchestrator
                            state, accepted/platform canonicals). Azure-safe
                            default: /tmp/trakt/blob_trigger in Azure,
                            out/blob_trigger locally.
"""

from __future__ import annotations

import logging
import os
import tempfile
from pathlib import Path

import azure.functions as func  # type: ignore

from apps.blob_trigger_app import azure_io
from apps.blob_trigger_app.eventgrid import classify_blob_event
from apps.blob_trigger_app.router import handle_blob_event
from apps.blob_trigger_app.runtime_paths import resolve_output_root
from apps.blob_trigger_app.schema_fingerprint import fingerprint_pack
from apps.blob_trigger_app.source_registry import SourceRegistry

app = func.FunctionApp()

_CONTAINER = os.environ.get("TRAKT_BLOB_CONTAINER", "raw")
_PACK_MARKER = os.environ.get("TRAKT_PACK_MARKER", "_READY.json")
_REGISTRY_PATH = os.environ.get("TRAKT_SOURCE_REGISTRY", "config/source_registry.yaml")
# Writable runtime output root — Azure-safe (defaults to /tmp/trakt/blob_trigger
# in Azure, out/blob_trigger locally; TRAKT_TRIGGER_OUT overrides). Everything
# downstream (event manifests, orchestrator state, accepted/platform canonicals)
# derives from this single root, so nothing writes under read-only wwwroot.
_OUT_DIR = resolve_output_root()


@app.event_grid_trigger(arg_name="event")
def on_raw_blob_event(event: func.EventGridEvent) -> None:
    subject = event.subject or ""
    ref = classify_blob_event(subject, _CONTAINER)

    # Reject only blobs outside the configured container (the raw-v2 fix).
    if not ref.accepted:
        logging.info("Skipping blob: %s", ref.reason)
        return

    blob_path = ref.blob_path                       # path within the container
    filename = blob_path.rsplit("/", 1)[-1]
    registry = SourceRegistry(_REGISTRY_PATH)
    logging.info("event grid: container=%s blob=%s", ref.container, blob_path)

    # Data-file event → acknowledge as a pack member; do NOT start the pipeline.
    if filename != _PACK_MARKER:
        manifest = handle_blob_event(
            blob_path, registry=registry, out_dir=_OUT_DIR,
            container=ref.container, pack_marker=_PACK_MARKER)
        logging.info("pack member received: %s (status=%s)", blob_path, manifest.get("status"))
        return

    # Marker event → the folder is complete. Read marker metadata, list + download
    # the pack, fingerprint all DATA files (never the marker), and route once.
    marker_meta = azure_io.read_marker_metadata(ref.container, blob_path)
    prefix = azure_io.folder_prefix(blob_path)
    pack_names = azure_io.list_pack_files(ref.container, prefix, marker=_PACK_MARKER)
    pack_dir = Path(tempfile.mkdtemp(prefix="trakt_pack_"))
    data_files = azure_io.download_pack(ref.container, prefix, pack_dir, marker=_PACK_MARKER)
    schema = fingerprint_pack(data_files) if data_files else None
    primary = str(sorted(data_files)[0]) if data_files else None

    manifest = handle_blob_event(
        blob_path, registry=registry, out_dir=_OUT_DIR, container=ref.container,
        pack_marker=_PACK_MARKER, input_dir_override=str(pack_dir),
        local_input_path=primary, schema_info=schema,
        marker_metadata=marker_meta, pack_files=pack_names)
    logging.info(
        "pack complete → decision=%s status=%s target=%s central=%s",
        manifest.get("event_decision"), manifest.get("status"),
        (manifest.get("selected_target") or {}).get("target"),
        manifest.get("central_canonical_path"))
