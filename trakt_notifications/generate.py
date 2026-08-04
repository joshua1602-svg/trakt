#!/usr/bin/env python3
"""Assembling the two-message batch from governed inputs.

Everything about the batch is deterministic: the same approved update resolved
twice produces the same batch id, the same message ids, the same ordering and
the same wording. ``generated_at`` is the only field that moves, and it is
excluded from the identity hash for exactly that reason.

Partial risk failure is handled here rather than in the Risk Review builder,
because it is a decision about what to SEND, not about what the risk position
is: if risk computation failed, the Portfolio Update still goes out (the
operational facts are unaffected) and the Risk Review goes out saying the
assessment was incomplete. What must never happen is the third option — a
Risk Review that reads as a clear state because the checks behind it did not
run.
"""

from __future__ import annotations

import datetime as _dt
import logging
from typing import Any, Dict, List, Optional

from . import portfolio_update, risk_review
from .config import load as load_config
from .contract import (
    MESSAGE_RISK_REVIEW, MessageItem, NotificationBatch, NotificationMessage,
    SEVERITY_ATTENTION, UPDATE_COMBINED, UPDATE_FUNDED, UPDATE_PIPELINE,
)
from .sources import GovernedInputs, unavailable_summary

logger = logging.getLogger("trakt_notifications.generate")

#: What a Risk Review says when its own construction failed. Deliberately not a
#: clear state and deliberately not silence.
RISK_INCOMPLETE_HEADLINE = "Risk Review — assessment incomplete"
RISK_INCOMPLETE_STATEMENT = (
    "The risk assessment could not be completed for this update, so no "
    "conclusion about material portfolio risk is offered. This is not a "
    "statement that no risks exist.")


def now_iso() -> str:
    return _dt.datetime.now(_dt.timezone.utc).replace(microsecond=0).isoformat()


def update_type_for(datasets: List[str]) -> str:
    """The update type implied by the datasets approved together."""
    normalised = {str(d).strip().lower() for d in datasets if d}
    has_pipeline = "pipeline" in normalised
    has_funded = "funded" in normalised
    if has_pipeline and has_funded:
        return UPDATE_COMBINED
    if has_funded:
        return UPDATE_FUNDED
    return UPDATE_PIPELINE


def build(inputs: GovernedInputs, *, update_type: str,
          approved_run_ids: Optional[List[str]] = None,
          corrects_batch_id: Optional[str] = None,
          generated_at: Optional[str] = None,
          eligibility: Optional[Dict[str, Any]] = None) -> NotificationBatch:
    """One batch, two messages, from already-resolved governed inputs."""
    conf = load_config()
    messages: List[NotificationMessage] = []

    if conf.get("update_message_enabled", True):
        messages.append(_safe_update(inputs, update_type, conf))
    if conf.get("risk_message_enabled", True):
        messages.append(_safe_risk(inputs, conf))

    run_ids = sorted({str(r) for r in
                      list(approved_run_ids or []) + list(inputs.run_ids) if r})

    return NotificationBatch(
        tenant_id=inputs.tenant_id,
        portfolio_id=inputs.portfolio_id,
        portfolio_context=inputs.portfolio_context,
        update_type=update_type,
        approved_run_ids=run_ids,
        source_dates=dict(inputs.source_dates),
        messages=messages,
        generated_at=generated_at or now_iso(),
        methodology_versions=dict(inputs.methodology_versions),
        provenance={
            "generated_from": ("governed MI outputs and the deterministic "
                               "Insight Engine; no metric recalculated for "
                               "delivery"),
            "unavailable_capabilities": unavailable_summary(inputs),
            "config_source": conf.get("source"),
        },
        delivery_eligibility=dict(eligibility or {}),
        corrects_batch_id=corrects_batch_id,
    )


def _safe_update(inputs: GovernedInputs, update_type: str,
                 conf: Dict[str, Any]) -> NotificationMessage:
    """The Portfolio Update, degrading to a stated-unavailable card on failure.

    A generator failure here costs the reader the detail, not the notification:
    they still learn that an approved update landed, and the deep link still
    takes them to the governed view.
    """
    try:
        return portfolio_update.build(
            inputs, update_type=update_type,
            max_items=int(conf.get("maximum_update_items") or 5))
    except Exception as exc:  # noqa: BLE001 - one message, not the batch
        logger.warning("notifications: portfolio update generation failed: %s", exc)
        from . import deep_links
        return NotificationMessage(
            message_type=portfolio_update.MESSAGE_PORTFOLIO_UPDATE,
            headline="Portfolio Update — detail unavailable",
            severity=SEVERITY_ATTENTION,
            items=[MessageItem(
                text=("An approved data update was published, but the update "
                      "detail could not be produced. The governed figures are "
                      "available in Trakt."),
                metric="generation_failed", unavailable=True)],
            deep_link=deep_links.for_update(
                update_type, portfolio_context=inputs.portfolio_context,
                pipeline_as_of=inputs.source_dates.get("pipeline_as_of"),
                funded_as_of=inputs.source_dates.get("funded_as_of")),
        )


def _safe_risk(inputs: GovernedInputs,
               conf: Dict[str, Any]) -> NotificationMessage:
    """The Risk Review, degrading to an explicit incomplete state on failure.

    The failure mode this guards is the dangerous one. A Risk Review that
    silently omitted its findings would read exactly like a clear week, so a
    failure produces a message that says, in terms, that no conclusion is being
    offered.
    """
    try:
        return risk_review.build(
            inputs, max_items=int(conf.get("maximum_risk_items") or 3))
    except Exception as exc:  # noqa: BLE001
        logger.warning("notifications: risk review generation failed: %s", exc)
        from . import deep_links
        unavailable = unavailable_summary(inputs)
        items = [MessageItem(text=RISK_INCOMPLETE_STATEMENT,
                             metric="risk_generation_failed", unavailable=True)]
        if unavailable:
            items.append(MessageItem(
                text="Checks unavailable for this update: "
                     + ", ".join(unavailable) + ".",
                metric="unavailable_checks", unavailable=True))
        return NotificationMessage(
            message_type=MESSAGE_RISK_REVIEW,
            headline=RISK_INCOMPLETE_HEADLINE,
            severity=SEVERITY_ATTENTION,
            items=items,
            deep_link=deep_links.for_risk(
                "current_breach", portfolio_context=inputs.portfolio_context,
                as_of=(inputs.source_dates.get("funded_as_of")
                       or inputs.source_dates.get("pipeline_as_of"))),
            unavailable_checks=unavailable or ["risk assessment"],
        )


def label_corrections(batch: NotificationBatch) -> NotificationBatch:
    """Relabel a correction batch's headlines in place.

    A correction is sent as a new, clearly labelled message rather than by
    editing the original card. Editing is optional in v1 and a relabelled
    message is unambiguous: the reader sees both, in order, and can tell which
    is current.
    """
    if not batch.is_correction:
        return batch
    for message in batch.messages:
        if not message.headline.startswith("Corrected "):
            message.headline = f"Corrected {message.headline}"
        message.items.append(MessageItem(
            text=("This replaces the earlier message for the same reporting "
                  "date, following a corrected data upload."),
            metric="correction"))
    return batch
