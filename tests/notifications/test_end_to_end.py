"""End-to-end: approved upload → MI → insights → outbox → Teams → deep link.

Runs the whole pathway with only two things replaced: the MI resolution (which
would need a full governed dataset on disk) and the Bot Framework network call.
Everything between them — generation, identity, recipient authorisation, the
outbox, card rendering, ordering, delivery and the audit trail — is the real
code.
"""

from __future__ import annotations

import json

import pytest

from trakt_notifications import cards, sources, trigger
from trakt_notifications.contract import (
    MESSAGE_PORTFOLIO_UPDATE, MESSAGE_RISK_REVIEW,
)
from trakt_notifications.delivery import DeliveryWorker
from trakt_notifications.outbox import Outbox, STATE_SENT
from trakt_notifications.store import BatchStore
from trakt_notifications.teams_client import SendResult, TeamsClient

def _json(value) -> str:
    r"""Serialise for substring assertions without escaping non-ASCII.

    ``json.dumps`` defaults to escaping every em dash and pound sign as a
    \uXXXX sequence, so an assertion about the text a reader actually sees
    would never match the serialised card.
    """
    return json.dumps(value, ensure_ascii=False)


from .conftest import (
    TENANT, concentration_risk, contributors_block, funded_inputs,
    pipeline_inputs,
)


class RecordingClient(TeamsClient):
    def __init__(self):
        super().__init__(app_id="app", app_password="secret")
        self.sent = []

    def send_card(self, *, service_url, conversation_id, attachment, summary):
        self.sent.append({"conversation_id": conversation_id,
                          "card": attachment["content"], "summary": summary})
        return SendResult(teams_message_id=f"msg-{len(self.sent)}", status=201)


@pytest.fixture
def workspace(monkeypatch):
    monkeypatch.setenv("TRAKT_COPILOT_WORKSPACE_BASE_URL",
                       "https://trakt.example.com/app")


def _run(storage, layout, *, datasets, run_ids):
    outcome = trigger.on_publication_approved(
        tenant_id=TENANT, portfolio_id=TENANT, datasets=list(datasets),
        approved_run_ids=list(run_ids), portfolio_context="total",
        run_status="published", storage=storage, layout=layout)
    client = RecordingClient()
    report = DeliveryWorker(storage=storage, layout=layout,
                            client=client).run(TENANT)
    return outcome, report, client


# --------------------------------------------------------------------------- #
# Pipeline
# --------------------------------------------------------------------------- #
def test_an_approved_pipeline_upload_reaches_teams(storage, layout, recipient,
                                                   enabled_config, workspace,
                                                   monkeypatch):
    inputs = pipeline_inputs()
    inputs.concentration = concentration_risk()
    inputs.contributors = {"region_sw": contributors_block()}
    monkeypatch.setattr(sources, "resolve", lambda **_k: inputs)

    outcome, report, client = _run(storage, layout, datasets=["pipeline"],
                                   run_ids=["orun-weekly-1"])

    # Two outbox records, two Teams messages, in the required order.
    assert len(outcome.outbox_item_ids) == 2
    assert report.sent == 2
    assert [s["summary"] for s in client.sent] == [
        "Trakt: Weekly Pipeline Update — 7 August",
        "Trakt: Risk Review — South West exposure"]

    # The update card carries the governed operational position.
    update_card = _json(client.sent[0]["card"])
    assert "Pipeline contains 812 cases totalling £503.4m." in update_card
    assert "Case count is 9% above the five-week average." in update_card
    assert "led by Broker A and the South East" in update_card

    # The risk card carries the governed finding, its evidence and a
    # consideration — not an instruction.
    risk_card = _json(client.sent[1]["card"])
    assert "expected to breach" in risk_card
    assert ("£8.2m of probability-weighted expected completions from Broker A "
            "in the South West") in risk_card
    assert "Consider reviewing completion timing" in risk_card
    assert "Stop Broker A" not in risk_card

    # Every item is recorded as delivered, with its Teams message id.
    items = Outbox(storage, layout).list(TENANT)
    assert all(i.state == STATE_SENT and i.teams_message_id for i in items)


def test_the_deep_links_open_the_right_react_context(storage, layout, recipient,
                                                     enabled_config, workspace,
                                                     monkeypatch):
    inputs = pipeline_inputs()
    inputs.concentration = concentration_risk()
    inputs.contributors = {"region_sw": contributors_block()}
    monkeypatch.setattr(sources, "resolve", lambda **_k: inputs)

    outcome, _report, client = _run(storage, layout, datasets=["pipeline"],
                                    run_ids=["orun-weekly-1"])
    batch = BatchStore(storage, layout).load(TENANT,
                                             outcome.notification_batch_id)

    update = batch.message(MESSAGE_PORTFOLIO_UPDATE)
    assert update.deep_link.startswith("https://trakt.example.com/app/mi/pipeline")
    assert "asOf=2026-08-07" in update.deep_link
    assert "portfolioContext=total" in update.deep_link

    risk = batch.message(MESSAGE_RISK_REVIEW)
    assert risk.deep_link.startswith(
        "https://trakt.example.com/app/mi/risk-limits")
    assert "testId=region_sw" in risk.deep_link
    assert risk.deep_link_label == "Open concentration analysis"

    # A link carries context, never authority — no tenant, no token.
    for link in (update.deep_link, risk.deep_link):
        assert "tenant" not in link.lower()
        assert "token" not in link.lower()

    # And the buttons on the cards are those links.
    assert client.sent[0]["card"]["actions"][0]["url"] == update.deep_link
    assert client.sent[1]["card"]["actions"][0]["url"] == risk.deep_link


# --------------------------------------------------------------------------- #
# Funded
# --------------------------------------------------------------------------- #
def test_an_approved_funded_upload_reaches_teams(storage, layout, recipient,
                                                 enabled_config, workspace,
                                                 monkeypatch):
    monkeypatch.setattr(sources, "resolve", lambda **_k: funded_inputs())

    outcome, report, client = _run(storage, layout, datasets=["funded"],
                                   run_ids=["orun-monthly-1"])
    assert report.sent == 2

    update_card = _json(client.sent[0]["card"])
    assert "Monthly Funded Update — 31 July" in update_card
    assert "Funded balance is £1.42bn, +£24.1m on the month." in update_card
    assert "Loan count is 4,820 (+59 on the month)." in update_card
    assert "Largest book contribution: Direct at +£18.4m." in update_card

    batch = BatchStore(storage, layout).load(TENANT,
                                             outcome.notification_batch_id)
    assert batch.message(MESSAGE_PORTFOLIO_UPDATE).deep_link.startswith(
        "https://trakt.example.com/app/mi/funded")


# --------------------------------------------------------------------------- #
# Whole-pathway invariants
# --------------------------------------------------------------------------- #
def test_no_card_ever_carries_loan_level_data_or_internal_identifiers(
        storage, layout, recipient, enabled_config, workspace, monkeypatch):
    inputs = pipeline_inputs()
    inputs.concentration = concentration_risk()
    inputs.contributors = {"region_sw": contributors_block()}
    monkeypatch.setattr(sources, "resolve", lambda **_k: inputs)

    _outcome, _report, client = _run(storage, layout, datasets=["pipeline"],
                                     run_ids=["orun-weekly-1"])
    blob = _json([s["card"] for s in client.sent])
    for forbidden in ("orun-weekly-1", "blob://", "trakt-state", "caseId",
                      "case_id", "loan_id", "pipeline_row_id"):
        assert forbidden not in blob


def test_a_replayed_approval_delivers_nothing_further(storage, layout,
                                                      recipient, enabled_config,
                                                      workspace, monkeypatch):
    """The duplicate-upload guarantee, end to end."""
    monkeypatch.setattr(sources, "resolve", lambda **_k: pipeline_inputs())
    _first, first_report, first_client = _run(storage, layout,
                                              datasets=["pipeline"],
                                              run_ids=["orun-weekly-1"])
    second, second_report, second_client = _run(storage, layout,
                                                datasets=["pipeline"],
                                                run_ids=["orun-weekly-1"])
    assert first_report.sent == 2
    assert second.suppressed_reason == trigger.SUPPRESS_DUPLICATE
    assert second_report.sent == 0
    assert second_client.sent == []


def test_a_corrected_upload_delivers_a_labelled_correction(storage, layout,
                                                           recipient,
                                                           enabled_config,
                                                           workspace,
                                                           monkeypatch):
    monkeypatch.setattr(sources, "resolve", lambda **_k: pipeline_inputs())
    _first, _report, _client = _run(storage, layout, datasets=["pipeline"],
                                    run_ids=["orun-weekly-1"])
    second, second_report, second_client = _run(storage, layout,
                                                datasets=["pipeline"],
                                                run_ids=["orun-weekly-1-fix"])
    assert second.is_correction
    assert second_report.sent == 2
    assert second_client.sent[0]["summary"].startswith(
        "Trakt: Corrected Weekly Pipeline Update")
    card = _json(second_client.sent[0]["card"])
    assert "following a corrected data upload" in card


def test_one_payload_is_rendered_for_every_recipient(storage, layout, recipient,
                                                     enabled_config, workspace,
                                                     monkeypatch):
    """Adding a recipient must cost a card render, not an MI run."""
    calls = []

    def _resolve(**kwargs):
        calls.append(kwargs)
        return pipeline_inputs()

    monkeypatch.setattr(sources, "resolve", _resolve)

    from trakt_notifications.recipients import RecipientStore
    store = RecipientStore(storage, layout)
    second = store.capture_conversation(
        tenant_id=TENANT, microsoft_tenant_id="11111111-1111-1111-1111-111111111111",
        entra_object_id="second-user", conversation_id="a:2conversation",
        service_url="https://smba.trafficmanager.net/emea/",
        conversation_reference={})
    store.authorise(TENANT, second.recipient_id, portfolio_contexts=["total"],
                    actor="operator")

    outcome, report, client = _run(storage, layout, datasets=["pipeline"],
                                   run_ids=["orun-weekly-1"])
    assert len(calls) == 1                      # MI resolved exactly once
    assert len(outcome.outbox_item_ids) == 4    # 2 messages x 2 recipients
    assert report.sent == 4
    assert {s["conversation_id"] for s in client.sent} == {"a:1conversation",
                                                           "a:2conversation"}
    # One stored payload, shared.
    assert BatchStore(storage, layout).exists(TENANT,
                                              outcome.notification_batch_id)
