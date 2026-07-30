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

#: The only dataset in scope for regime (ESMA Annex 2) reporting. Everything else
#: — pipeline, forecast — is an MI view and is separated from the funded delivery.
DATASET_FUNDED = "funded"

OUTCOME_MI = "mi"
OUTCOME_MI_ANNEX2 = "mi_annex2"


def _engine():
    from operations_control.engine import OpsEngine
    from operations_control.stores import OpsStore
    return OpsEngine(OpsStore.from_env())


def _outcome_for(engine, client_id: str, portfolio_id: str,
                 dataset: str) -> str:
    """Workflow selection for automated arrivals.

    Two rules, in this order:

    1. Only the FUNDED book is ever in scope for regime reporting. Pipeline is a
       forward-looking MI view and must never produce an ESMA Annex 2 delivery,
       whatever the registry says — so it short-circuits before any lookup and no
       flag can pull it in.
    2. A funded book — direct OR acquired — gets the full Annex 2 delivery
       workflow when its source is flagged ``regime_required``. Acquired is not a
       second-class book; whether it is in scope is the registry's decision, not
       the book type's.

    The lookup matches on dataset only, never frequency: an acquired book is
    registered at the frequency of its first delivery (commonly ad hoc) and is
    reported monthly thereafter, so a frequency-keyed lookup would find it once
    and miss it forever after.
    """
    if dataset != DATASET_FUNDED:
        logger.info("intake: dataset %r is MI-only by rule (regime reporting "
                    "applies to the funded book)", dataset)
        return OUTCOME_MI
    try:
        records = engine._source_registry().records_for_dataset(
            client_id, portfolio_id, DATASET_FUNDED)
    except Exception:  # noqa: BLE001
        # Never let a registry problem silently decide a regime question.
        logger.exception(
            "intake: could not read the source registry for %s/%s; defaulting "
            "to %s. If this book is regime-required its Annex 2 delivery has "
            "NOT been produced.", client_id, portfolio_id, OUTCOME_MI)
        return OUTCOME_MI

    if not records:
        logger.warning(
            "intake: no funded source record for %s/%s, so regime scope is "
            "unknown; defaulting to %s", client_id, portfolio_id, OUTCOME_MI)
        return OUTCOME_MI
    if any(r.regime_required for r in records):
        logger.info("intake: %s/%s is regime-required -> %s",
                    client_id, portfolio_id, OUTCOME_MI_ANNEX2)
        return OUTCOME_MI_ANNEX2
    return OUTCOME_MI


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
    outcome = _outcome_for(engine, client_id, parsed.source_portfolio_id,
                           parsed.dataset)
    # `dataset` keeps a pipeline arrival in its own input pack. Without it the
    # pack is keyed on client + portfolio + period + workflow only, so a pipeline
    # file and a funded file for the same portfolio and period that both resolve
    # to MI would be assessed as one delivery.
    batch = engine.create_batch(
        client_id=client_id, portfolio_id=parsed.source_portfolio_id,
        reporting_date=parsed.reporting_period, workflow_type=outcome,
        created_by="blob-trigger", auto_start_when_ready=True,
        dataset=parsed.dataset)

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
