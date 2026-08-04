"""Azure Functions deployment entrypoint (root) — Event Grid handler.

Azure Functions loads the Function App from the **root** ``function_app.py``.
The Azure event source stays **Event Grid** (an Event Grid subscription on the
storage account delivers blob-created events here). This handler does
**routing/inference only**: it parses the event subject, accepts the configured
container, and delegates to the Azure-free decision core in
``apps.blob_trigger_app.router``.

Flow (OCC-owned readiness — the _READY.json sentinel is NOT supported):
    Blob uploaded to %TRAKT_BLOB_CONTAINER%  (e.g. raw-v2)
        → Event Grid → this handler
        → apps.blob_trigger_app.occ_intake.handle_arrival
        → OCC input batch: file registered + roles recognised
        → OCC readiness (roles + configuration + decisions)
        → immutable internal run manifest → governed onboarding run

The legacy handler hardcoded the accepted container to a constant and silently
skipped ``raw-v2``. The accepted container is now configuration
(``TRAKT_BLOB_CONTAINER``), not a constant.

App settings (see local.settings.example.json):
    TRAKT_BLOB_CONTAINER    accepted container (default ``raw``; production ``raw-v2``)
    TRAKT_BLOB_CONNECTION   storage connection (app-setting name, not the string)
    TRAKT_OPS_CONTAINER     operations-control governance container
    TRAKT_SOURCE_REGISTRY   source registry path (default config/source_registry.yaml)
    TRAKT_TRIGGER_OUT       writable runtime output root (event log, orchestrator
                            state, accepted/platform canonicals). Azure-safe
                            default: /tmp/trakt/blob_trigger in Azure,
                            out/blob_trigger locally.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

import azure.functions as func  # type: ignore

from apps.blob_trigger_app.eventgrid import classify_blob_event
from apps.blob_trigger_app.layout import Layout
from apps.blob_trigger_app.runtime_paths import resolve_output_root
from apps.blob_trigger_app.storage import decide_backend, open_storage

app = func.FunctionApp()

_CONTAINER = os.environ.get("TRAKT_BLOB_CONTAINER", "raw")
# Writable runtime SCRATCH root — Azure-safe (defaults to /tmp/trakt/blob_trigger
# in Azure, out/blob_trigger locally; TRAKT_TRIGGER_OUT overrides). Final
# artifacts are uploaded to durable Blob Storage via ProductionPersistence.
_OUT_DIR = resolve_output_root()


def _log_startup() -> None:
    """One-time startup diagnostics: storage backend decision + resolved URIs."""
    try:
        layout = Layout.from_env()
        d = decide_backend()                      # pure decision, never raises
        logging.info(
            "TRAKT STARTUP: selected_backend=%s reason=%s azure_connection_detected=%s "
            "(source=%s) registry_uri=%s processed_container=%s platform_latest(ERE)=%s "
            "regime_prefix(ERE)=%s scratch_out=%s",
            d["backend"], d["reason"], d["connection_detected"], d["connection_source"],
            layout.registry_uri, f"blob://{layout.processed_container}/",
            layout.platform_latest_uri("ERE"), layout.regime_prefix("ERE", "{period}"),
            _OUT_DIR)
    except Exception:  # noqa: BLE001 — never let diagnostics break startup
        logging.exception("TRAKT STARTUP diagnostics failed")


_log_startup()


#: How often the proactive-notification delivery worker runs. Every five
#: minutes: an approved update is not urgent to the minute, and a longer
#: interval would make a transient Teams failure look like a lost message to
#: whoever is watching. Overridable per deployment.
_NOTIFY_SCHEDULE = os.environ.get("TRAKT_TEAMS_DELIVERY_SCHEDULE",
                                  "0 */5 * * * *")


@app.timer_trigger(schedule=_NOTIFY_SCHEDULE, arg_name="timer",
                   run_on_startup=False, use_monitor=True)
def deliver_teams_notifications(timer: func.TimerRequest) -> None:
    """Drain the notification outbox for the tenant this deployment serves.

    Deliberately separate from ingestion. Approval writes durable intent and
    returns; this turns intent into Teams messages, retries transient failures
    on its own schedule, and cannot delay or fail a publication.

    Runs as a SINGLE worker: blob writes are last-writer-wins, so two workers
    racing for one outbox item would both believe they hold it. The send-time
    idempotency check makes that harmless rather than duplicating a message,
    but single-instance is the supported configuration.
    """
    tenant = (os.environ.get("TRAKT_TEAMS_TRAKT_TENANT") or "").strip()
    if not tenant:
        return                       # nothing configured; nothing to deliver
    try:
        from trakt_notifications.config import enabled
        if not enabled():
            return
        from trakt_notifications.delivery import run_once
        report = run_once(tenant)
    except Exception:  # noqa: BLE001 - a delivery pass must not kill the app
        logging.exception("TEAMS NOTIFICATION DELIVERY FAILED tenant=%s", tenant)
        return
    if report.attempted or report.errors:
        logging.info("teams delivery: %s", report.to_dict())


@app.event_grid_trigger(arg_name="event")
def on_raw_blob_event(event: func.EventGridEvent) -> None:
    """Event Grid entrypoint. Wraps dispatch so any uncaught exception is logged
    with its full traceback (and the subject) before propagating — otherwise
    Azure reports only 'Executed (Failed)' with no diagnostics."""
    try:
        _dispatch(event)
    except Exception:
        logging.exception("BLOB-TRIGGER HANDLER FAILED subject=%s", getattr(event, "subject", None))
        raise


def _dispatch(event: func.EventGridEvent) -> None:
    """OCC-owned readiness (no sentinel): every accepted blob is REGISTERED
    with the Operations Control Centre intake service, which recognises file
    roles, evaluates pack completeness + configuration readiness, creates the
    immutable internal run manifest and starts the governed run itself.

    ``_READY.json`` is NO LONGER SUPPORTED: legacy sentinel arrivals are
    logged + audited as unsupported artefacts and never trigger execution.
    """
    subject = event.subject or ""
    ref = classify_blob_event(subject, _CONTAINER)

    # Reject only blobs outside the configured container (the raw-v2 fix).
    if not ref.accepted:
        logging.info("Skipping blob: %s", ref.reason)
        return

    blob_path = ref.blob_path                       # path within the container
    logging.info("event grid: container=%s blob=%s", ref.container, blob_path)

    from apps.blob_trigger_app import occ_intake

    def _download(container: str, path: str, dest_dir: Path) -> Path:
        dest = dest_dir / path.rsplit("/", 1)[-1]
        open_storage().download_file(f"blob://{container}/{path}", dest)
        return dest

    result = occ_intake.handle_arrival(ref.container, blob_path,
                                       download=_download)
    logging.info(
        "occ intake: registered=%s reason=%s batch=%s status=%s workflow=%s",
        result.get("registered"), result.get("reason", ""),
        result.get("batch_id", ""), result.get("status", ""),
        result.get("workflow_id", ""))
