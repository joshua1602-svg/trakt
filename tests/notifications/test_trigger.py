"""The approval trigger: eligibility, deduplication, corrections, supersession."""

from __future__ import annotations

import pytest

from trakt_notifications import sources, trigger
from trakt_notifications.contract import MESSAGE_PORTFOLIO_UPDATE
from trakt_notifications.outbox import (
    Outbox, STATE_CORRECTED, STATE_SUPERSEDED,
)
from trakt_notifications.store import BatchStore

from .conftest import TENANT, funded_inputs, pipeline_inputs


@pytest.fixture
def governed(monkeypatch):
    """Replace MI resolution with a fixed governed result.

    The trigger's job is eligibility, identity and durable intent — resolving
    MI is :mod:`trakt_notifications.sources`' job and is exercised separately.
    """
    def _resolve(*, tenant_id, portfolio_id, portfolio_context="total",
                 want_pipeline=True, want_funded=True, **_kwargs):
        if want_funded and not want_pipeline:
            inputs = funded_inputs()
        else:
            inputs = pipeline_inputs()
            if want_funded:
                funded = funded_inputs()
                inputs.funded_movement = funded.funded_movement
                inputs.source_dates.update(funded.source_dates)
        inputs.tenant_id = tenant_id
        inputs.portfolio_id = portfolio_id
        inputs.portfolio_context = portfolio_context
        return inputs

    monkeypatch.setattr(sources, "resolve", _resolve)
    return _resolve


def _approve(storage, layout, *, datasets=("pipeline",), run_ids=("run-1",),
             status="published", context="total"):
    return trigger.on_publication_approved(
        tenant_id=TENANT, portfolio_id=TENANT, datasets=list(datasets),
        approved_run_ids=list(run_ids), portfolio_context=context,
        run_status=status, storage=storage, layout=layout)


# --------------------------------------------------------------------------- #
# Eligibility
# --------------------------------------------------------------------------- #
def test_an_approved_pipeline_update_enqueues_two_messages(storage, layout,
                                                           recipient, governed,
                                                           enabled_config):
    outcome = _approve(storage, layout)
    assert outcome.sent_to_outbox
    assert len(outcome.outbox_item_ids) == 2
    assert outcome.recipients == 1
    assert outcome.suppressed_reason is None


def test_disabled_notifications_suppress_everything(storage, layout, recipient,
                                                    governed, enabled_config,
                                                    monkeypatch):
    monkeypatch.setenv("TRAKT_TEAMS_NOTIFICATIONS", "false")
    outcome = _approve(storage, layout)
    assert outcome.suppressed_reason == trigger.SUPPRESS_DISABLED
    assert outcome.outbox_item_ids == []


def test_an_unapproved_run_never_notifies(storage, layout, recipient, governed,
                                          enabled_config):
    for status in ("awaiting_publication", "blocked", "failed", "needs_review"):
        outcome = _approve(storage, layout, status=status)
        assert outcome.suppressed_reason == trigger.SUPPRESS_NOT_APPROVED


def test_a_non_notifying_dataset_is_ignored(storage, layout, recipient,
                                            governed, enabled_config):
    outcome = _approve(storage, layout, datasets=["annex2"])
    assert outcome.suppressed_reason == trigger.SUPPRESS_UNKNOWN_DATASET


def test_no_authorised_recipient_still_records_the_batch(storage, layout,
                                                         governed,
                                                         enabled_config):
    """The generated content is the audit record of what would have been said,
    and a recipient authorised tomorrow must not re-notify yesterday."""
    outcome = _approve(storage, layout)
    assert outcome.suppressed_reason == trigger.SUPPRESS_NO_RECIPIENTS
    assert BatchStore(storage, layout).exists(TENANT,
                                              outcome.notification_batch_id)


def test_a_combined_approval_produces_one_integrated_update(storage, layout,
                                                            recipient, governed,
                                                            enabled_config):
    outcome = _approve(storage, layout, datasets=["pipeline", "funded"])
    batch = BatchStore(storage, layout).load(TENANT,
                                             outcome.notification_batch_id)
    assert batch.update_type == "COMBINED"
    update = batch.message(MESSAGE_PORTFOLIO_UPDATE)
    assert "pipeline 7 August" in update.headline
    assert "funded 31 July" in update.headline


# --------------------------------------------------------------------------- #
# Duplicates and corrections
# --------------------------------------------------------------------------- #
def test_a_duplicate_upload_does_not_notify_again(storage, layout, recipient,
                                                  governed, enabled_config):
    first = _approve(storage, layout)
    second = _approve(storage, layout)
    assert first.sent_to_outbox
    assert second.suppressed_reason == trigger.SUPPRESS_DUPLICATE
    assert second.notification_batch_id == first.notification_batch_id
    assert len(Outbox(storage, layout).list(TENANT)) == 2


def test_a_corrected_upload_supersedes_undelivered_originals(storage, layout,
                                                             recipient,
                                                             governed,
                                                             enabled_config):
    first = _approve(storage, layout, run_ids=["run-1"])
    second = _approve(storage, layout, run_ids=["run-2"])

    assert second.is_correction
    assert second.notification_batch_id != first.notification_batch_id
    assert second.reporting_key == first.reporting_key
    assert len(second.superseded) == 2

    states = {i.item_id: i.state for i in Outbox(storage, layout).list(TENANT)}
    assert all(states[i] == STATE_SUPERSEDED for i in first.outbox_item_ids)
    assert all(states[i] == "pending" for i in second.outbox_item_ids)


def test_a_correction_after_delivery_is_labelled_not_silent(storage, layout,
                                                            recipient, governed,
                                                            enabled_config):
    first = _approve(storage, layout, run_ids=["run-1"])
    outbox = Outbox(storage, layout)
    for item in outbox.list(TENANT):
        outbox.mark_sent(item, teams_message_id="msg")

    second = _approve(storage, layout, run_ids=["run-2"])
    assert second.is_correction
    assert len(second.corrected) == 2

    states = {i.item_id: i.state for i in outbox.list(TENANT)}
    assert all(states[i] == STATE_CORRECTED for i in first.outbox_item_ids)

    batch = BatchStore(storage, layout).load(TENANT,
                                             second.notification_batch_id)
    assert batch.corrects_batch_id == first.notification_batch_id
    assert batch.message(MESSAGE_PORTFOLIO_UPDATE).headline.startswith(
        "Corrected ")


def test_a_correction_of_a_correction_stays_traceable(storage, layout,
                                                      recipient, governed,
                                                      enabled_config):
    first = _approve(storage, layout, run_ids=["run-1"])
    second = _approve(storage, layout, run_ids=["run-2"])
    third = _approve(storage, layout, run_ids=["run-3"])

    pointer = BatchStore(storage, layout).reporting_pointer(
        TENANT, third.reporting_key)
    assert pointer["notification_batch_id"] == third.notification_batch_id
    assert first.notification_batch_id in pointer["history"]
    assert second.notification_batch_id in pointer["history"]


def test_a_different_reporting_date_is_a_new_update_not_a_correction(
        storage, layout, recipient, governed, enabled_config, monkeypatch):
    first = _approve(storage, layout, run_ids=["run-1"])

    def _later(*, tenant_id, portfolio_id, portfolio_context="total",
               want_pipeline=True, want_funded=True, **_kwargs):
        inputs = pipeline_inputs()
        inputs.source_dates = {"pipeline_as_of": "2026-08-14"}
        inputs.tenant_id, inputs.portfolio_id = tenant_id, portfolio_id
        inputs.portfolio_context = portfolio_context
        return inputs

    monkeypatch.setattr(sources, "resolve", _later)
    second = _approve(storage, layout, run_ids=["run-2"])
    assert not second.is_correction
    assert second.reporting_key != first.reporting_key
    assert second.sent_to_outbox


# --------------------------------------------------------------------------- #
# Failure containment
# --------------------------------------------------------------------------- #
def test_a_generation_failure_never_raises(storage, layout, recipient,
                                           enabled_config, monkeypatch):
    """A publication is a governed action; a notification defect must not turn
    it into an error."""
    def explode(**_kwargs):
        raise RuntimeError("MI resolution exploded")

    monkeypatch.setattr(sources, "resolve", explode)
    outcome = _approve(storage, layout)
    assert outcome.error == "MI resolution exploded"
    assert outcome.sent_to_outbox is False
    assert outcome.suppressed_reason


def test_refusals_are_recorded_for_the_operator(storage, layout, governed,
                                                enabled_config):
    from trakt_notifications.recipients import RecipientStore
    RecipientStore(storage, layout).capture_conversation(
        tenant_id=TENANT, microsoft_tenant_id="ms", entra_object_id="u2",
        conversation_id="a:3", service_url="https://smba.trafficmanager.net/e/",
        conversation_reference={})
    outcome = _approve(storage, layout)
    assert outcome.recipients == 0
    assert outcome.refusals[0]["code"] == "notifications_disabled"
