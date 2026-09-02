#!/usr/bin/env python3
"""The approval hook — the one place ingestion touches notifications.

Called at the end of :meth:`operations_control.engine.OpsEngine.approve_publication`,
which is the single governed point at which an update is approved, artefacts
are promoted to ``latest``, and governed MI outputs are safe to consume.

Two properties matter more here than anywhere else in the package:

* **It cannot fail the approval.** An operator approving a publication is
  performing a governed action; a notification defect must never turn that into
  an error, a rollback or a retry. Every path returns an outcome object, and
  the caller wraps it besides.

* **It does not send.** It resolves, generates, and writes durable intent. The
  actual Teams call happens later, in a worker, so an approval never waits on
  Microsoft's availability and an API restart mid-approval loses nothing.

Deduplication, correction and supersession are all decided here, from the two
deterministic identities on the batch:

  * the same approved run regenerated → same ``notification_batch_id`` → the
    batch already exists and nothing is re-enqueued;
  * a corrected upload for the same reporting dates → same ``reporting_key``,
    different batch id → the new batch is labelled a correction, undelivered
    originals are superseded, delivered originals are marked corrected.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from apps.blob_trigger_app.storage import Storage, open_storage

from . import enrichment, generate, sources, telemetry
from .config import load as load_config
from .contract import NotificationBatch, validate
from .layout import NotificationLayout
from .outbox import Outbox
from .recipients import RecipientStore, select
from .store import BatchStore

logger = logging.getLogger("trakt_notifications.trigger")

# Suppression reasons — every one is recorded, so "nothing was sent" is always
# an answerable question rather than an absence.
SUPPRESS_DISABLED = "notifications are disabled for this deployment"
SUPPRESS_NOT_APPROVED = "the run is not in an approved, published state"
SUPPRESS_UNKNOWN_DATASET = "the approved dataset is not one that notifies"
SUPPRESS_DUPLICATE = "this exact approved update has already been notified"
SUPPRESS_INVALID = "the generated batch failed contract validation"
SUPPRESS_NO_RECIPIENTS = "no authorised recipient is registered"


@dataclass
class TriggerOutcome:
    """What the hook did. Returned rather than raised, always."""

    sent_to_outbox: bool = False
    suppressed_reason: Optional[str] = None
    notification_batch_id: Optional[str] = None
    reporting_key: Optional[str] = None
    is_correction: bool = False
    outbox_item_ids: List[str] = field(default_factory=list)
    recipients: int = 0
    refusals: List[Dict[str, str]] = field(default_factory=list)
    superseded: List[str] = field(default_factory=list)
    corrected: List[str] = field(default_factory=list)
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "sent_to_outbox": self.sent_to_outbox,
            "suppressed_reason": self.suppressed_reason,
            "notification_batch_id": self.notification_batch_id,
            "reporting_key": self.reporting_key,
            "is_correction": self.is_correction,
            "outbox_item_ids": list(self.outbox_item_ids),
            "recipients": self.recipients,
            "refusals": list(self.refusals),
            "superseded": list(self.superseded),
            "corrected": list(self.corrected),
            "error": self.error,
        }


#: Datasets that produce a notification. Anything else — a regulatory delivery,
#: an onboarding artefact — is approved through the same path and must not.
NOTIFYING_DATASETS = ("pipeline", "funded")


def on_publication_approved(*, tenant_id: str, portfolio_id: str,
                            datasets: List[str],
                            approved_run_ids: Optional[List[str]] = None,
                            portfolio_context: str = "total",
                            run_status: Optional[str] = None,
                            funded_run_id: Optional[str] = None,
                            storage: Optional[Storage] = None,
                            layout: Optional[NotificationLayout] = None,
                            expected_microsoft_tenant: Optional[str] = None,
                            reviewer: Optional[Callable[[], Any]] = None,
                            ) -> TriggerOutcome:
    """Generate and enqueue the two messages for one approved update.

    ``tenant_id`` comes from the governed run, never from a request. There is
    no parameter by which a caller can name a different tenant, which is what
    makes cross-tenant delivery unreachable rather than merely guarded.
    """
    outcome = TriggerOutcome()
    try:
        return _run(outcome, tenant_id=tenant_id, portfolio_id=portfolio_id,
                    datasets=datasets, approved_run_ids=approved_run_ids or [],
                    portfolio_context=portfolio_context, run_status=run_status,
                    funded_run_id=funded_run_id, storage=storage, layout=layout,
                    expected_microsoft_tenant=expected_microsoft_tenant,
                    reviewer=reviewer)
    except Exception as exc:  # noqa: BLE001 - a publication must never fail here
        logger.exception("notifications: trigger failed for %s/%s",
                         tenant_id, portfolio_id)
        outcome.error = str(exc)
        outcome.suppressed_reason = "the notification could not be generated"
        telemetry.failure(telemetry.EVT_SUPPRESSED, tenant_id=tenant_id,
                          error=str(exc))
        return outcome


def _run(outcome: TriggerOutcome, *, tenant_id: str, portfolio_id: str,
         datasets: List[str], approved_run_ids: List[str],
         portfolio_context: str, run_status: Optional[str],
         funded_run_id: Optional[str], storage: Optional[Storage],
         layout: Optional[NotificationLayout],
         expected_microsoft_tenant: Optional[str],
         reviewer: Optional[Callable[[], Any]] = None) -> TriggerOutcome:
    conf = load_config()

    telemetry.emit(telemetry.EVT_UPDATE_RECEIVED, tenant_id=tenant_id,
                   portfolio_context=portfolio_context,
                   datasets=",".join(sorted(str(d) for d in datasets)),
                   run_status=run_status)

    # -- eligibility ------------------------------------------------------- #
    if not conf.get("enabled"):
        return _suppress(outcome, SUPPRESS_DISABLED, tenant_id)

    # The run must be in the published state. This hook is only called from
    # approve_publication, but a defensive check costs nothing and makes the
    # precondition explicit rather than implied by the call site.
    if run_status is not None and str(run_status) != "published":
        return _suppress(outcome, SUPPRESS_NOT_APPROVED, tenant_id)

    notifying = [d for d in datasets
                 if str(d).strip().lower() in NOTIFYING_DATASETS]
    if not notifying:
        return _suppress(outcome, SUPPRESS_UNKNOWN_DATASET, tenant_id)

    update_type = generate.update_type_for(notifying)
    normalised = {str(d).strip().lower() for d in notifying}

    storage = storage or open_storage()
    layout = layout or NotificationLayout.from_env()
    batches = BatchStore(storage, layout)
    outbox = Outbox(storage, layout)

    # -- governed MI resolution -------------------------------------------- #
    inputs = sources.resolve(
        tenant_id=tenant_id, portfolio_id=portfolio_id,
        portfolio_context=portfolio_context,
        funded_run_id=funded_run_id,
        want_pipeline="pipeline" in normalised,
        want_funded="funded" in normalised)

    # -- generation -------------------------------------------------------- #
    batch = generate.build(inputs, update_type=update_type,
                           approved_run_ids=approved_run_ids,
                           eligibility={"datasets": sorted(normalised),
                                        "run_status": run_status})

    # -- autonomous enrichment, strictly additive --------------------------- #
    # Deliberately AFTER the guaranteed briefing exists and BEFORE validation,
    # so an enriched batch is held to exactly the same contract as a plain one.
    # `enrich` cannot raise and cannot remove a deterministic item; if the
    # autonomous layer is blocked, errors or is not configured, `batch` comes
    # back as `generate.build` made it.
    batch = enrichment.enrich(batch, reviewer=reviewer)
    outcome.notification_batch_id = batch.notification_batch_id
    outcome.reporting_key = batch.reporting_key

    problems = validate(batch)
    if problems:
        logger.warning("notifications: batch failed validation: %s",
                       "; ".join(problems))
        return _suppress(outcome, SUPPRESS_INVALID, tenant_id)

    # -- duplicate: the same approved run, already notified ----------------- #
    if batches.exists(tenant_id, batch.notification_batch_id):
        telemetry.emit(telemetry.EVT_DEDUPLICATED, tenant_id=tenant_id,
                       batch_id=batch.notification_batch_id,
                       update_type=update_type)
        return _suppress(outcome, SUPPRESS_DUPLICATE, tenant_id)

    # -- correction: the same reporting position, a different run ---------- #
    prior = batches.prior_batch_id(tenant_id, batch.reporting_key)
    if prior and prior != batch.notification_batch_id:
        batch.corrects_batch_id = prior
        generate.label_corrections(batch)
        outcome.is_correction = True
        telemetry.emit(telemetry.EVT_CORRECTION, tenant_id=tenant_id,
                       batch_id=batch.notification_batch_id,
                       corrects=prior, update_type=update_type)

    telemetry.emit(telemetry.EVT_BATCH_GENERATED, tenant_id=tenant_id,
                   batch_id=batch.notification_batch_id,
                   update_type=update_type,
                   is_correction=batch.is_correction,
                   unavailable=len(inputs.unavailable))
    telemetry.emit(telemetry.EVT_INSIGHTS_SELECTED, tenant_id=tenant_id,
                   batch_id=batch.notification_batch_id,
                   insight_count=sum(len(m.insight_ids) for m in batch.messages))

    # -- recipients --------------------------------------------------------- #
    recipient_store = RecipientStore(storage, layout)
    eligible, refused = select(
        recipient_store, tenant_id=tenant_id,
        portfolio_context=portfolio_context,
        expected_microsoft_tenant=expected_microsoft_tenant)
    outcome.recipients = len(eligible)
    outcome.refusals = [{"recipient_id": r.recipient_id, "code": r.code,
                         "reason": r.reason} for r in refused]
    telemetry.emit(telemetry.EVT_RECIPIENTS_RESOLVED, tenant_id=tenant_id,
                   batch_id=batch.notification_batch_id,
                   eligible=len(eligible), refused=len(refused))

    # The batch is stored and the reporting position recorded even with no
    # recipient: the generated content is the audit record of what WOULD have
    # been said, and a recipient authorised tomorrow must not cause yesterday's
    # update to be re-notified as though it were new.
    batches.save(batch)
    batches.record_reporting(batch)

    if not eligible:
        return _suppress(outcome, SUPPRESS_NO_RECIPIENTS, tenant_id)

    # -- supersede / correct the prior batch's items ------------------------ #
    if batch.is_correction:
        superseded = outbox.supersede_undelivered(
            tenant_id, batch.reporting_key,
            superseded_by=batch.notification_batch_id,
            keep_batch_id=batch.notification_batch_id)
        corrected = outbox.mark_corrected(
            tenant_id, batch.reporting_key,
            corrected_by=batch.notification_batch_id,
            keep_batch_id=batch.notification_batch_id)
        outcome.superseded = [i.item_id for i in superseded]
        outcome.corrected = [i.item_id for i in corrected]
        if superseded:
            telemetry.emit(telemetry.EVT_SUPERSEDED, tenant_id=tenant_id,
                           batch_id=batch.notification_batch_id,
                           superseded=len(superseded))

    # -- durable intent ----------------------------------------------------- #
    items = outbox.enqueue(batch, eligible)
    outcome.outbox_item_ids = [i.item_id for i in items]
    outcome.sent_to_outbox = bool(items)
    telemetry.emit(telemetry.EVT_OUTBOX_CREATED, tenant_id=tenant_id,
                   batch_id=batch.notification_batch_id,
                   items=len(items), recipients=len(eligible))
    return outcome


def _suppress(outcome: TriggerOutcome, reason: str,
              tenant_id: str) -> TriggerOutcome:
    outcome.suppressed_reason = reason
    telemetry.emit(telemetry.EVT_SUPPRESSED, tenant_id=tenant_id,
                   reason=reason, batch_id=outcome.notification_batch_id)
    return outcome
