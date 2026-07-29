"""apps.blob_trigger_app.occ_intake — blob arrivals into OCC-owned readiness.

The governed replacement for the ``_READY.json`` sentinel on the production
Event Grid route. The trigger's responsibility is reduced to: validate the
blob path and tenant context, register the file into the correct OCC input
batch, and ask the OCC readiness service to reassess. The OCC decides
completeness from recognised semantic input roles, effective-configuration
status and open decisions — and starts the run itself (auto-start for
automated recurring sources).

Legacy ``_READY.json`` arrivals are registered as unsupported legacy artefacts
and NEVER trigger execution.
"""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional

from .path_parser import PathParseError, parse_blob_path

logger = logging.getLogger("trakt.blob_trigger.occ_intake")

LEGACY_SENTINEL = "_READY.json"


def _engine():
    from operations_control.engine import OpsEngine
    from operations_control.stores import OpsStore
    return OpsEngine(OpsStore.from_env())


def _outcome_for(engine, client_id: str, portfolio_id: str) -> str:
    """Workflow selection for automated arrivals: regime-required sources get
    the full Annex 2 delivery workflow, everything else MI."""
    try:
        rec = engine._source_registry().lookup(client_id, portfolio_id,
                                               "funded", "monthly")
        if rec is not None and rec.regime_required:
            return "mi_annex2"
    except Exception:  # noqa: BLE001
        pass
    return "mi"


def handle_arrival(container: str, blob_path: str, *,
                   download) -> Dict[str, Any]:
    """Register one arrived blob with the OCC readiness service.

    ``download(container, blob_path, dest_dir) -> Path`` is injected by the
    caller (Azure download in production, local copy in tests) so this module
    stays I/O-agnostic.
    """
    filename = blob_path.rsplit("/", 1)[-1]
    try:
        parsed = parse_blob_path(blob_path, container)
    except PathParseError as exc:
        logger.info("intake: unparseable path skipped: %s (%s)",
                    blob_path, exc)
        return {"registered": False, "reason": "unparseable_path"}

    engine = _engine()
    client_id = parsed.client_id
    outcome = _outcome_for(engine, client_id, parsed.source_portfolio_id)
    batch = engine.create_batch(
        client_id=client_id, portfolio_id=parsed.source_portfolio_id,
        reporting_date=parsed.reporting_period, workflow_type=outcome,
        created_by="blob-trigger", auto_start_when_ready=True)

    if filename == LEGACY_SENTINEL:
        # Sentinel is unsupported: recorded + audited, never triggers a run.
        batch.setdefault("legacy_sentinels_ignored", [])
        if blob_path not in batch["legacy_sentinels_ignored"]:
            batch["legacy_sentinels_ignored"].append(blob_path)
            engine.intake.save_batch(batch)
        engine.store.append_audit(
            client_id, "legacy_sentinel_ignored", actor="blob-trigger",
            detail={"batch_id": batch["batch_id"], "blob_path": blob_path,
                    "note": "_READY.json is no longer supported and does not "
                            "trigger processing"})
        logger.info("intake: legacy _READY.json ignored: %s", blob_path)
        return {"registered": False, "reason": "legacy_sentinel_ignored",
                "batch_id": batch["batch_id"]}

    with tempfile.TemporaryDirectory(prefix="occ_intake_") as td:
        local = download(container, blob_path, Path(td))
        batch = engine.register_batch_file(
            client_id=client_id, batch_id=batch["batch_id"],
            source_path=str(local), received_by="blob-trigger")
    logger.info("intake: registered %s -> batch %s status=%s",
                blob_path, batch["batch_id"], batch["status"])
    return {"registered": True, "batch_id": batch["batch_id"],
            "status": batch["status"],
            "workflow_id": batch.get("workflow_id", "")}
