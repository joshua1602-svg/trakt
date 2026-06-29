"""apps.blob_trigger_app.function_app — thin Azure Functions blob trigger.

Routing/inference only — NO business logic. On a blob upload it downloads the
file, then delegates to :func:`router.handle_blob_event`, which parses the path,
fingerprints the schema, checks the source registry, decides source-onboarding
vs deterministic processing, invokes the Orchestrator Agent, and writes an event
manifest.

Path convention:
    raw/{client_id}/{dataset}/{frequency}/{source_portfolio_id}/{reporting_period}/{filename}

App settings (see local.settings.example.json):
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
from .source_registry import SourceRegistry

app = func.FunctionApp()

_REGISTRY_PATH = os.environ.get("TRAKT_SOURCE_REGISTRY", "config/source_registry.yaml")
_OUT_DIR = os.environ.get("TRAKT_TRIGGER_OUT", "out/blob_trigger")


@app.blob_trigger(arg_name="blob", path="raw/{name}",
                  connection="TRAKT_BLOB_CONNECTION")
def on_raw_blob_upload(blob: func.InputStream) -> None:
    blob_path = blob.name  # e.g. "raw/ERE/funded/monthly/direct_001/2026-09-30/loan_tape.xlsx"
    logging.info("blob trigger: %s (%s bytes)", blob_path, blob.length)

    # Download the blob to a temp dir whose layout the Orchestrator can consume.
    tmp = Path(tempfile.mkdtemp(prefix="trakt_blob_"))
    filename = Path(blob_path).name
    local_path = tmp / filename
    local_path.write_bytes(blob.read())

    registry = SourceRegistry(_REGISTRY_PATH)
    manifest = handle_blob_event(
        blob_path, registry=registry, out_dir=_OUT_DIR,
        local_input_path=str(local_path))
    logging.info("trigger decision: status=%s decision=%s target=%s",
                 manifest.get("status"), manifest.get("decision"),
                 (manifest.get("selected_target") or {}).get("target"))
