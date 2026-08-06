"""Recipients, outbox, delivery and the security boundaries around them."""

from __future__ import annotations

import json

import pytest

from trakt_notifications import cards, generate
from trakt_notifications.contract import (
    MESSAGE_PORTFOLIO_UPDATE, MESSAGE_RISK_REVIEW, UPDATE_PIPELINE,
)
from trakt_notifications.delivery import DeliveryWorker
from trakt_notifications.outbox import (
    Outbox, STATE_CORRECTED, STATE_PERMANENT, STATE_RETRYABLE, STATE_SENT,
    STATE_SUPERSEDED,
)
from trakt_notifications.recipients import (
    INSTALL_REMOVED, RecipientStore, REFUSE_DISABLED, REFUSE_NOT_ADDRESSABLE,
    REFUSE_NOT_INSTALLED, REFUSE_TENANT_MISMATCH,
    REFUSE_UNAUTHORISED_PORTFOLIO, select,
)
from trakt_notifications.store import BatchStore
from trakt_notifications.teams_client import TeamsClient, TeamsSendError

from .conftest import MS_TENANT, TENANT, pipeline_inputs


# --------------------------------------------------------------------------- #
# A fake Bot Framework transport
# --------------------------------------------------------------------------- #
class FakeClient(TeamsClient):
    """A TeamsClient with the network replaced, and a scripted failure list."""

    def __init__(self, *, failures=None, configured=True):
        super().__init__(app_id="app" if configured else None,
                         app_password="secret" if configured else None)
        self.sent = []
        self._failures = list(failures or [])

    @property
    def configured(self) -> bool:
        return bool(self._app_id and self._app_password)

    def send_card(self, *, service_url, conversation_id, attachment, summary):
        if self._failures:
            raise self._failures.pop(0)
        self.sent.append({"service_url": service_url,
                          "conversation_id": conversation_id,
                          "attachment": attachment, "summary": summary})
        from trakt_notifications.teams_client import SendResult
        return SendResult(teams_message_id=f"msg-{len(self.sent)}", status=201)


@pytest.fixture
def batch(enabled_config):
    return generate.build(pipeline_inputs(), update_type=UPDATE_PIPELINE,
                          approved_run_ids=["run-1"])


@pytest.fixture
def enqueued(storage, layout, batch, recipient):
    BatchStore(storage, layout).save(batch)
    items = Outbox(storage, layout).enqueue(batch, [recipient])
    return items


# --------------------------------------------------------------------------- #
# Recipient authorisation
# --------------------------------------------------------------------------- #
def test_capture_makes_a_user_addressable_but_not_authorised(storage, layout):
    """Installing the app must never, by itself, start a feed of portfolio data."""
    store = RecipientStore(storage, layout)
    captured = store.capture_conversation(
        tenant_id=TENANT, microsoft_tenant_id=MS_TENANT,
        entra_object_id="new-user", conversation_id="a:2",
        service_url="https://smba.trafficmanager.net/emea/",
        conversation_reference={}, display_name="Somebody")
    assert captured.addressable
    assert captured.notifications_enabled is False
    assert captured.portfolio_contexts == []

    eligible, refused = select(store, tenant_id=TENANT,
                               portfolio_context="total")
    assert eligible == []
    assert refused[0].code == REFUSE_DISABLED


def test_recipient_id_is_scoped_to_the_microsoft_tenant():
    from trakt_notifications.recipients import recipient_id
    assert recipient_id("tenant-a", "user-1") != recipient_id("tenant-b", "user-1")


def test_unauthorised_portfolio_context_is_refused(storage, layout, recipient):
    store = RecipientStore(storage, layout)
    eligible, refused = select(store, tenant_id=TENANT,
                               portfolio_context="direct_001")
    assert eligible == []
    assert refused[0].code == REFUSE_UNAUTHORISED_PORTFOLIO


def test_a_different_microsoft_tenant_is_refused_not_rerouted(storage, layout,
                                                              recipient):
    store = RecipientStore(storage, layout)
    eligible, refused = select(
        store, tenant_id=TENANT, portfolio_context="total",
        expected_microsoft_tenant="99999999-9999-9999-9999-999999999999")
    assert eligible == []
    assert refused[0].code == REFUSE_TENANT_MISMATCH


def test_an_uninstalled_app_stops_delivery_but_keeps_the_record(storage, layout,
                                                                recipient):
    store = RecipientStore(storage, layout)
    store.mark_uninstalled(TENANT, recipient.recipient_id)
    eligible, refused = select(store, tenant_id=TENANT,
                               portfolio_context="total")
    assert eligible == []
    assert refused[0].code in (REFUSE_NOT_INSTALLED, REFUSE_DISABLED)
    assert store.load(TENANT, recipient.recipient_id).installation_status \
        == INSTALL_REMOVED


def test_a_missing_conversation_reference_is_refused(storage, layout, recipient):
    store = RecipientStore(storage, layout)
    recipient.conversation_id = None
    recipient.service_url = None
    store.save(recipient)
    eligible, refused = select(store, tenant_id=TENANT,
                               portfolio_context="total")
    assert eligible == []
    assert refused[0].code == REFUSE_NOT_ADDRESSABLE


def test_one_tenants_listing_cannot_see_another(storage, layout, recipient):
    store = RecipientStore(storage, layout)
    store.capture_conversation(
        tenant_id="OTHER", microsoft_tenant_id="other-ms",
        entra_object_id="other-user", conversation_id="a:9",
        service_url="https://smba.trafficmanager.net/emea/",
        conversation_reference={})
    assert [r.recipient_id for r in store.list(TENANT)] == [recipient.recipient_id]
    assert len(store.list("OTHER")) == 1


def test_a_spoofed_tenant_claim_cannot_reach_an_existing_record(storage, layout,
                                                                recipient):
    """A claim to be the same user in a different Microsoft tenant addresses a
    different record entirely, because the record id derives from the tenant.

    It therefore cannot redirect an authorised recipient's messages, and the new
    record it does create is unauthorised like any other fresh capture.
    """
    store = RecipientStore(storage, layout)
    spoofed = store.capture_conversation(
        tenant_id=TENANT, microsoft_tenant_id="attacker-tenant",
        entra_object_id=recipient.entra_object_id,
        conversation_id="a:evil", service_url="https://evil.example/",
        conversation_reference={})

    assert spoofed.recipient_id != recipient.recipient_id
    assert spoofed.notifications_enabled is False
    assert spoofed.portfolio_contexts == []

    # The authorised recipient is untouched, and still the only deliverable one.
    original = store.load(TENANT, recipient.recipient_id)
    assert original.conversation_id == "a:1conversation"
    assert original.service_url == "https://smba.trafficmanager.net/emea/"
    eligible, _refused = select(store, tenant_id=TENANT,
                                portfolio_context="total")
    assert [r.recipient_id for r in eligible] == [recipient.recipient_id]


# --------------------------------------------------------------------------- #
# Outbox
# --------------------------------------------------------------------------- #
def test_enqueue_writes_one_item_per_message_per_recipient(enqueued):
    assert len(enqueued) == 2
    assert [i.message_type for i in enqueued] == [MESSAGE_PORTFOLIO_UPDATE,
                                                  MESSAGE_RISK_REVIEW]
    assert [i.sequence for i in enqueued] == [1, 2]


def test_a_duplicate_upload_does_not_resend(storage, layout, batch, recipient,
                                            enqueued):
    outbox = Outbox(storage, layout)
    for item in enqueued:
        outbox.mark_sent(item, teams_message_id="msg-1")

    # The same approved run regenerated → the same batch id → the same items.
    again = generate.build(pipeline_inputs(), update_type=UPDATE_PIPELINE,
                           approved_run_ids=["run-1"])
    assert again.notification_batch_id == batch.notification_batch_id
    assert outbox.enqueue(again, [recipient]) == []


def test_a_stored_batch_is_never_overwritten(storage, layout, batch):
    store = BatchStore(storage, layout)
    store.save(batch)
    tampered = generate.build(pipeline_inputs(), update_type=UPDATE_PIPELINE,
                              approved_run_ids=["run-1"])
    tampered.messages[0].headline = "Something else"
    store.save(tampered)
    assert store.load(TENANT, batch.notification_batch_id).messages[0].headline \
        == "Weekly Pipeline Update — 7 August"


def test_undelivered_items_are_superseded_by_a_correction(storage, layout,
                                                          batch, enqueued):
    outbox = Outbox(storage, layout)
    changed = outbox.supersede_undelivered(
        TENANT, batch.reporting_key, superseded_by="corrected-batch")
    assert len(changed) == 2
    assert all(i.state == STATE_SUPERSEDED for i in changed)
    assert outbox.due(TENANT) == []


def test_delivered_items_are_marked_corrected_not_superseded(storage, layout,
                                                             batch, enqueued):
    outbox = Outbox(storage, layout)
    for item in enqueued:
        outbox.mark_sent(item, teams_message_id="msg-1")
    changed = outbox.mark_corrected(TENANT, batch.reporting_key,
                                    corrected_by="corrected-batch")
    assert len(changed) == 2
    assert all(i.state == STATE_CORRECTED for i in changed)


def test_a_persisted_error_never_carries_a_credential(storage, layout, enqueued):
    outbox = Outbox(storage, layout)
    item = outbox.mark_permanent(
        enqueued[0],
        error="401 from https://x/?sig=SECRETTOKEN Authorization: Bearer abc123")
    assert "SECRETTOKEN" not in (item.last_error or "")
    assert "abc123" not in (item.last_error or "")
    assert "[redacted]" in (item.last_error or "")


# --------------------------------------------------------------------------- #
# Delivery
# --------------------------------------------------------------------------- #
def test_both_messages_are_delivered_in_order(storage, layout, enqueued):
    client = FakeClient()
    report = DeliveryWorker(storage=storage, layout=layout, client=client).run(
        TENANT)
    assert report.sent == 2
    assert [s["summary"] for s in client.sent] == [
        "Trakt: Weekly Pipeline Update — 7 August",
        "Trakt: Risk Review — no material risks identified"]


def test_a_second_pass_does_not_resend(storage, layout, enqueued):
    worker = DeliveryWorker(storage=storage, layout=layout, client=FakeClient())
    worker.run(TENANT)
    second = DeliveryWorker(storage=storage, layout=layout,
                            client=FakeClient()).run(TENANT)
    assert second.sent == 0
    assert second.attempted == 0


def test_a_transient_failure_is_retried_not_abandoned(storage, layout, enqueued):
    client = FakeClient(failures=[TeamsSendError("429", retryable=True,
                                                 status=429)])
    report = DeliveryWorker(storage=storage, layout=layout,
                            client=client).run(TENANT)
    assert report.retried == 1
    items = {i.message_type: i for i in Outbox(storage, layout).list(TENANT)}
    update = items[MESSAGE_PORTFOLIO_UPDATE]
    assert update.state == STATE_RETRYABLE
    assert update.attempt_count == 1
    assert update.next_attempt_at is not None
    # The Risk Review waits for the update rather than overtaking it.
    assert items[MESSAGE_RISK_REVIEW].state == "pending"
    assert report.blocked == 1


def test_a_permanent_failure_stops_retrying(storage, layout, enqueued):
    client = FakeClient(failures=[TeamsSendError("403 forbidden",
                                                 retryable=False, status=403)])
    report = DeliveryWorker(storage=storage, layout=layout,
                            client=client).run(TENANT)
    assert report.failed == 1
    items = {i.message_type: i for i in Outbox(storage, layout).list(TENANT)}
    assert items[MESSAGE_PORTFOLIO_UPDATE].state == STATE_PERMANENT


def test_a_permanently_failed_update_releases_the_risk_review(storage, layout,
                                                              enqueued):
    """The more important message must not be suppressed by the less important
    one failing — and it is released within the same pass, not left waiting."""
    client = FakeClient(failures=[TeamsSendError("403", retryable=False,
                                                 status=403)])
    report = DeliveryWorker(storage=storage, layout=layout,
                            client=client).run(TENANT)
    assert report.failed == 1
    assert report.sent == 1
    assert report.blocked == 0
    assert client.sent[0]["summary"].startswith("Trakt: Risk Review")


def test_retries_exhaust_into_a_permanent_failure(storage, layout, enqueued,
                                                  enabled_config):
    """max_attempts is 3 in the test configuration."""
    for _ in range(3):
        client = FakeClient(failures=[TeamsSendError("500", retryable=True,
                                                     status=500)])
        worker = DeliveryWorker(storage=storage, layout=layout, client=client)
        # Clear the backoff so the item is due again immediately.
        for item in Outbox(storage, layout).list(TENANT):
            item.next_attempt_at = None
            Outbox(storage, layout)._write(item)  # noqa: SLF001 - test setup
        worker.run(TENANT)
    update = next(i for i in Outbox(storage, layout).list(TENANT)
                  if i.message_type == MESSAGE_PORTFOLIO_UPDATE)
    assert update.state == STATE_PERMANENT
    assert "gave up after" in update.last_error


def test_an_unconfigured_bot_fails_closed_without_burning_attempts(storage,
                                                                   layout,
                                                                   enqueued):
    report = DeliveryWorker(storage=storage, layout=layout,
                            client=FakeClient(configured=False)).run(TENANT)
    assert report.sent == 0
    assert report.attempted == 0
    assert "bot identity is not configured" in report.errors[0]
    assert all(i.attempt_count == 0 for i in Outbox(storage, layout).list(TENANT))


def test_a_missing_payload_is_a_permanent_failure(storage, layout, batch,
                                                  recipient):
    """The batch was never stored, so there is nothing to render.

    Both messages fail permanently rather than being retried forever against a
    payload that will never appear.
    """
    Outbox(storage, layout).enqueue(batch, [recipient])
    report = DeliveryWorker(storage=storage, layout=layout,
                            client=FakeClient()).run(TENANT)
    assert report.failed == 2
    assert all(i.state == STATE_PERMANENT
               for i in Outbox(storage, layout).list(TENANT))


# --------------------------------------------------------------------------- #
# Transport safety
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("service_url", [
    "https://evil.example.com/",
    "http://smba.trafficmanager.net/emea/",   # not https
    "https://smba.trafficmanager.net.evil.io/",
])
def test_a_send_refuses_a_service_url_outside_bot_framework(service_url):
    client = TeamsClient(app_id="app", app_password="secret")
    with pytest.raises(TeamsSendError) as excinfo:
        client.send_card(service_url=service_url, conversation_id="a:1",
                         attachment={}, summary="x")
    assert excinfo.value.retryable is False


def test_an_unconfigured_client_refuses_without_a_network_call():
    client = TeamsClient(app_id=None, app_password=None)
    with pytest.raises(TeamsSendError) as excinfo:
        client.send_card(service_url="https://smba.trafficmanager.net/emea/",
                         conversation_id="a:1", attachment={}, summary="x")
    assert excinfo.value.retryable is False


# --------------------------------------------------------------------------- #
# Cards
# --------------------------------------------------------------------------- #
def test_cards_carry_no_loan_level_data_or_internal_identifiers(batch):
    blob = json.dumps([cards.render(batch, m) for m in batch.messages])
    for forbidden in ("case_id", "caseId", "loan_id", "run-1", "blob://",
                      "pipeline_row_id", "application_id"):
        assert forbidden not in blob


def test_a_card_declares_one_action_at_most(batch):
    for message in batch.messages:
        assert len(cards.render(batch, message)["actions"]) <= 1


def test_no_workspace_url_means_no_button(batch):
    for message in batch.messages:
        assert message.deep_link is None
        assert cards.render(batch, message)["actions"] == []


def test_severity_drives_the_card_style_only(batch, monkeypatch):
    risk = batch.message(MESSAGE_RISK_REVIEW)
    rendered = cards.render(batch, risk)
    container = next(b for b in rendered["body"] if b["type"] == "Container")
    assert container["style"] == "good"       # clear state
    risk.severity = "concern"
    assert next(b for b in cards.render(batch, risk)["body"]
                if b["type"] == "Container")["style"] == "attention"
