"""The production invariant: enrichment may improve the briefing, never withhold it.

    A guaranteed governed portfolio briefing, enhanced by 2-4 autonomous MI
    insights where the agent can safely add value.

The whole of this file exists to make the second half unable to damage the
first. Every way the autonomous layer can fail gets its own test, and each one
asserts the same thing from a different direction: **the deterministic briefing
was still delivered.**

WHAT IS AND IS NOT REPLACED
---------------------------
Two seams, and only two, exactly as ``test_end_to_end`` uses them:

* **MI resolution** is supplied, because resolving it for real needs a full
  governed dataset on disk and this file is about delivery, not analytics. The
  ten-period deterministic suite covers the analytics against real canonical.
* **The Bot Framework HTTPS call** is recorded rather than made, by subclassing
  the real :class:`TeamsClient` and overriding its outermost method. Nothing
  else is stubbed.

Between those two seams every line is production code: the approval hook, the
generator, enrichment, contract validation, deduplication, correction,
recipient authorisation, the outbox, card rendering, the ordering rules, the
delivery worker, retry classification and the audit trail. A message that
reaches ``RecordingClient.send_card`` is a message Trakt would have put on the
wire.
"""

from __future__ import annotations

import json

import pytest

from trakt_notifications import enrichment, sources, trigger
from trakt_notifications.contract import MESSAGE_PORTFOLIO_UPDATE
from trakt_notifications.delivery import DeliveryWorker
from trakt_notifications.outbox import Outbox, STATE_SENT
from trakt_notifications.store import BatchStore
from trakt_notifications.teams_client import SendResult, TeamsClient

from .conftest import TENANT, funded_inputs


def _json(value) -> str:
    return json.dumps(value, ensure_ascii=False)


class RecordingClient(TeamsClient):
    """The real client with only the outbound HTTPS call replaced."""

    def __init__(self):
        super().__init__(app_id="app", app_password="secret")
        self.sent = []

    def send_card(self, *, service_url, conversation_id, attachment, summary):
        self.sent.append({"service_url": service_url,
                          "conversation_id": conversation_id,
                          "card": attachment["content"], "summary": summary})
        return SendResult(teams_message_id=f"msg-{len(self.sent)}", status=201)


# --------------------------------------------------------------------------- #
# Autonomous outcomes, in every state the controller can produce
# --------------------------------------------------------------------------- #
class _Outcome:
    """Duck-types ``portfolio_review.controller.ReviewOutcome``."""

    def __init__(self, *, card=None, gate_status="", dropped=(),
                 unsupported=(), steps=8, calls=14):
        self.card = card
        self.gate_status = gate_status
        self.steps = steps
        self.efficiency = {"total_calls": calls}
        self.dropped_findings = list(dropped)
        self.unsupported_claims = list(unsupported)
        self.out_of_mandate_calls = []


def _finding(title, observation, why):
    return {"title": title, "observation": observation,
            "why_it_matters": why, "severity": "medium"}


PUBLISHABLE_CARD = {
    "period_verdict": "MATERIAL_DEVELOPMENTS",
    "headline": "The acquired book drove the month.",
    "findings": [
        _finding("Acquisition dominates",
                 "funded_composition reports 93.4% of the movement is the "
                 "acquired book.",
                 "The underlying business must be read separately."),
        _finding("Regional mix shifted",
                 "The South East rose to 22.6% of balance.",
                 "Worth watching against the approved regional limit."),
    ],
}

BLOCKED_OUTCOME = _Outcome(card=None, gate_status="BLOCKED",
                           unsupported=[{"stated": "116%", "in": "headline"}])

DEGRADED_OUTCOME = _Outcome(
    card={**PUBLISHABLE_CARD, "findings": PUBLISHABLE_CARD["findings"][:1]},
    gate_status="DEGRADED",
    dropped=[{"finding": {"title": "Two loans above 70% LTV"},
              "reason": "1.88m is not a governed value"}],
    unsupported=[{"stated": "1.88m", "in": "findings[1].observation"}])


@pytest.fixture
def funded(monkeypatch):
    inputs = funded_inputs()
    monkeypatch.setattr(sources, "resolve", lambda **_k: inputs)
    return inputs


def _approve(storage, layout, *, reviewer=None, run_ids=("orun-funded-1",),
             tenant=TENANT):
    return trigger.on_publication_approved(
        tenant_id=tenant, portfolio_id=tenant, datasets=["funded"],
        approved_run_ids=list(run_ids), portfolio_context="total",
        run_status="published", storage=storage, layout=layout,
        reviewer=reviewer)


def _deliver(storage, layout, tenant=TENANT):
    client = RecordingClient()
    report = DeliveryWorker(storage=storage, layout=layout,
                            client=client).run(tenant)
    return report, client


def _update_card(client):
    return _json(next(s["card"] for s in client.sent
                      if "Funded" in s["summary"] or "Update" in s["summary"]))


# =========================================================================== #
# §7 — the four fallback states
# =========================================================================== #
def test_publishable_enrichment_adds_observations_to_the_deterministic_card(
        storage, layout, recipient, enabled_config, funded):
    """Deterministic core + autonomous observations, one message."""
    outcome = _approve(storage, layout,
                       reviewer=lambda: _Outcome(card=PUBLISHABLE_CARD,
                                                 gate_status="PUBLISHABLE"))
    report, client = _deliver(storage, layout)

    assert outcome.sent_to_outbox is True
    assert report.sent >= 1
    card = _update_card(client)
    # The governed baseline is present...
    assert "Funded balance is" in card
    # ...and so are the agent's observations.
    assert "93.4% of the movement is the acquired book" in card
    assert "South East rose to 22.6%" in card

    record = enrichment.record_of(
        BatchStore(storage, layout).load(TENANT, outcome.notification_batch_id))
    assert record["status"] == enrichment.ENRICHED
    assert record["added"] == 2


def test_degraded_enrichment_delivers_what_survived_and_never_names_the_rest(
        storage, layout, recipient, enabled_config, funded):
    """A dropped unsafe finding leaves no trace in the message.

    The reader must not be told that something was withheld: a briefing that
    apologises for its optional half is worse than one that simply does not
    have it. The operator is told, in the batch record.
    """
    outcome = _approve(storage, layout, reviewer=lambda: DEGRADED_OUTCOME)
    _report, client = _deliver(storage, layout)

    card = _update_card(client)
    assert "93.4% of the movement is the acquired book" in card
    for leak in ("1.88", "dropped", "withheld", "unsupported", "70% LTV"):
        assert leak not in card

    record = enrichment.record_of(
        BatchStore(storage, layout).load(TENANT, outcome.notification_batch_id))
    assert record["status"] == enrichment.ENRICHED
    assert record["added"] == 1
    assert record["dropped"][0]["title"] == "Two loans above 70% LTV"


def test_a_blocked_review_still_delivers_the_deterministic_briefing(
        storage, layout, recipient, enabled_config, funded):
    """The invariant, in the case that used to produce silence.

    `E_mixed` — the most complex period — was BLOCKED in both real-model
    passes. Before this sprint that meant the reader got nothing. Now the gate
    still blocks and the briefing still arrives.
    """
    outcome = _approve(storage, layout, reviewer=lambda: BLOCKED_OUTCOME)
    report, client = _deliver(storage, layout)

    assert outcome.sent_to_outbox is True
    assert report.sent >= 1
    card = _update_card(client)
    assert "Funded balance is" in card
    assert "116%" not in card

    record = enrichment.record_of(
        BatchStore(storage, layout).load(TENANT, outcome.notification_batch_id))
    assert record["status"] == enrichment.BLOCKED
    assert record["gate_status"] == "BLOCKED"
    assert record["added"] == 0


@pytest.mark.parametrize("failure", [
    RuntimeError("the model returned 500"),
    TimeoutError("the model did not respond"),
    ValueError("credit balance is too low to access the API"),
    KeyError("findings"),
])
def test_any_autonomous_runtime_failure_still_delivers_the_briefing(
        storage, layout, recipient, enabled_config, funded, failure):
    """§8: silent loss of the reporting message is not an acceptable failure.

    Parameterised over the failures actually seen — a model error, a timeout,
    an exhausted balance, a malformed payload — because the invariant is not
    "we handled the errors we thought of".
    """
    def _explode():
        raise failure

    outcome = _approve(storage, layout, reviewer=_explode)
    report, client = _deliver(storage, layout)

    assert outcome.sent_to_outbox is True
    assert report.sent >= 1
    assert "Funded balance is" in _update_card(client)

    record = enrichment.record_of(
        BatchStore(storage, layout).load(TENANT, outcome.notification_batch_id))
    assert record["status"] == enrichment.FAILED
    assert type(failure).__name__ in record["error"]
    # The reader never sees it.
    assert type(failure).__name__ not in _update_card(client)


def test_no_reviewer_configured_is_not_a_failure(
        storage, layout, recipient, enabled_config, funded):
    """Enrichment is optional. A deployment without it delivers as before."""
    outcome = _approve(storage, layout, reviewer=None)
    report, client = _deliver(storage, layout)

    assert report.sent >= 1
    assert "Funded balance is" in _update_card(client)
    record = enrichment.record_of(
        BatchStore(storage, layout).load(TENANT, outcome.notification_batch_id))
    assert record["status"] == enrichment.NOT_ATTEMPTED


def test_enrichment_never_removes_a_deterministic_item(
        storage, layout, recipient, enabled_config, funded, monkeypatch):
    """The additive property, asserted directly rather than inferred.

    Whatever the autonomous layer does, the deterministic items the generator
    produced are all still on the message.
    """
    from trakt_notifications import generate

    baseline = generate.build(funded, update_type="FUNDED")
    before = [i.text for i in baseline.message(MESSAGE_PORTFOLIO_UPDATE).items]

    for reviewer in (lambda: _Outcome(card=PUBLISHABLE_CARD,
                                      gate_status="PUBLISHABLE"),
                     lambda: BLOCKED_OUTCOME,
                     lambda: DEGRADED_OUTCOME,
                     None):
        batch = generate.build(funded, update_type="FUNDED")
        batch = enrichment.enrich(batch, reviewer=reviewer)
        after = [i.text for i in batch.message(MESSAGE_PORTFOLIO_UPDATE).items]
        assert before == after[:len(before)], "a deterministic item was lost"


def test_at_most_four_observations_reach_the_card(
        storage, layout, recipient, enabled_config, funded):
    """§1: approximately 2-4. Four is the ceiling; there is no floor."""
    from trakt_notifications import generate

    many = {**PUBLISHABLE_CARD,
            "findings": [_finding(f"F{i}", f"Observation {i}.", "Because.")
                         for i in range(9)]}
    batch = enrichment.enrich(
        generate.build(funded, update_type="FUNDED"),
        reviewer=lambda: _Outcome(card=many, gate_status="PUBLISHABLE"))

    observations = [i for i in batch.message(MESSAGE_PORTFOLIO_UPDATE).items
                    if i.metric == "autonomous_observation"]
    assert len(observations) == enrichment.MAX_INSIGHTS
    assert enrichment.record_of(batch)["withheld"] == 5


def test_a_card_with_no_findings_is_not_a_failure(
        storage, layout, recipient, enabled_config, funded):
    """A quiet period may legitimately yield nothing to add."""
    from trakt_notifications import generate

    batch = enrichment.enrich(
        generate.build(funded, update_type="FUNDED"),
        reviewer=lambda: _Outcome(card={"headline": "Quiet.", "findings": []},
                                  gate_status="PUBLISHABLE"))

    assert enrichment.record_of(batch)["status"] == enrichment.NOTHING_TO_ADD


# =========================================================================== #
# §18 — delivery
# =========================================================================== #
def test_the_bot_sends_without_the_user_having_messaged_it(
        storage, layout, recipient, enabled_config, funded):
    """Proactive initiation: the send is driven by an approval, not a reply.

    Nothing in this test posts an inbound activity. The conversation reference
    captured at authorisation is the whole basis for addressing the user.
    """
    _approve(storage, layout)
    _report, client = _deliver(storage, layout)

    assert client.sent
    assert client.sent[0]["conversation_id"] == "a:1conversation"
    assert client.sent[0]["service_url"].startswith("https://smba.")


def test_a_message_for_one_client_cannot_reach_another(
        storage, layout, recipient, enabled_config, funded):
    """Routing isolation, asserted from the other tenant's side.

    `recipient` is authorised for ERE only. Approving a run for a different
    tenant must deliver nothing, because there is no authorised recipient
    there — and the tenant id comes from the governed run, so no caller can
    name someone else's.
    """
    outcome = _approve(storage, layout, tenant="OTHER_LENDER",
                       run_ids=["orun-other-1"])
    report, client = _deliver(storage, layout, tenant="OTHER_LENDER")

    assert outcome.sent_to_outbox is False
    assert outcome.recipients == 0
    assert client.sent == []
    assert report.sent == 0
    # And ERE's own outbox was never touched by the other tenant's approval.
    assert Outbox(storage, layout).list(TENANT) == []


def test_client_bs_authorised_user_never_receives_client_as_briefing(
        storage, layout, recipient, enabled_config, funded):
    """The §18 requirement stated the way it matters.

    Both lenders have a real, authorised, addressable recipient. Client A's
    reporting run is approved. Client B's user must receive nothing — not
    because B has no recipient, but because the message is addressed from B's
    own tenant partition and A's approval never reaches it.
    """
    from trakt_notifications.recipients import RecipientStore

    store = RecipientStore(storage, layout)
    other = store.capture_conversation(
        tenant_id="OTHER_LENDER",
        microsoft_tenant_id="99999999-9999-9999-9999-999999999999",
        entra_object_id="88888888-8888-8888-8888-888888888888",
        conversation_id="a:otherconversation",
        service_url="https://smba.trafficmanager.net/emea/",
        conversation_reference={"conversation": {"id": "a:otherconversation"}},
        display_name="Other Lender User")
    store.authorise("OTHER_LENDER", other.recipient_id,
                    portfolio_contexts=["total"], actor="operator")

    _approve(storage, layout)                       # Client A's run only
    _report, client_a = _deliver(storage, layout)
    _report_b, client_b = _deliver(storage, layout, tenant="OTHER_LENDER")

    assert client_a.sent, "Client A's own briefing should have been delivered"
    assert all(s["conversation_id"] == "a:1conversation" for s in client_a.sent)
    assert client_b.sent == [], "Client B received Client A's reporting run"
    assert Outbox(storage, layout).list("OTHER_LENDER") == []


def test_the_same_reporting_period_is_not_delivered_twice(
        storage, layout, recipient, enabled_config, funded):
    """Duplicate protection, at the approval boundary."""
    first = _approve(storage, layout)
    report_one, client_one = _deliver(storage, layout)

    second = _approve(storage, layout)          # the same approved run again
    report_two, client_two = _deliver(storage, layout)

    assert first.sent_to_outbox is True
    assert report_one.sent >= 1
    assert second.sent_to_outbox is False
    assert second.suppressed_reason == trigger.SUPPRESS_DUPLICATE
    assert client_two.sent == []
    assert report_two.sent == 0


def test_a_second_delivery_pass_does_not_resend(
        storage, layout, recipient, enabled_config, funded):
    """Idempotency in the worker as well as the trigger."""
    _approve(storage, layout)
    _deliver(storage, layout)
    report, client = _deliver(storage, layout)

    assert client.sent == []
    assert report.sent == 0
    assert all(i.state == STATE_SENT for i in Outbox(storage, layout).list(TENANT))


def test_with_delivery_disabled_nothing_is_generated_or_sent(
        storage, layout, recipient, tmp_path, monkeypatch, funded):
    """§18: `enabled: false` means no external message, from the first step."""
    config = tmp_path / "off.yaml"
    config.write_text("teams_notifications:\n  enabled: false\n",
                      encoding="utf-8")
    monkeypatch.setenv("TRAKT_TEAMS_NOTIFICATIONS_CONFIG", str(config))
    monkeypatch.delenv("TRAKT_TEAMS_NOTIFICATIONS", raising=False)

    outcome = _approve(storage, layout)
    report, client = _deliver(storage, layout)

    assert outcome.sent_to_outbox is False
    assert outcome.suppressed_reason == trigger.SUPPRESS_DISABLED
    assert client.sent == []
    assert report.sent == 0


def test_an_operator_can_reconstruct_what_happened(
        storage, layout, recipient, enabled_config, funded):
    """§18 auditability: message, period, client, recipient, enrichment, result."""
    outcome = _approve(storage, layout, reviewer=lambda: DEGRADED_OUTCOME)
    _report, _client = _deliver(storage, layout)

    batch = BatchStore(storage, layout).load(
        TENANT, outcome.notification_batch_id)
    items = Outbox(storage, layout).list(TENANT)

    assert batch.tenant_id == TENANT                        # client
    assert batch.source_dates.get("funded_as_of")           # reporting period
    assert batch.approved_run_ids                           # governed run
    assert batch.message(MESSAGE_PORTFOLIO_UPDATE).items    # what was said
    record = enrichment.record_of(batch)                    # enrichment outcome
    assert record["status"] == enrichment.ENRICHED
    assert record["gate_status"] == "DEGRADED"
    assert record["dropped"]
    assert all(i.recipient_id for i in items)               # recipient
    assert all(i.state == STATE_SENT and i.teams_message_id
               for i in items)                             # delivery result


# =========================================================================== #
# §19 — shadow mode
# =========================================================================== #
def test_shadow_mode_stores_the_briefing_without_sending_it(
        storage, layout, enabled_config, funded):
    """With no recipient authorised, the batch is still generated and stored.

    This is the intended first activation step, and it is existing production
    behaviour rather than anything added here: the trigger saves the batch and
    records the reporting position BEFORE it checks whether anyone is eligible,
    so an operator can read exactly what would have been sent.

    Note there is no `recipient` fixture in this test — that is the point.
    """
    outcome = _approve(storage, layout,
                       reviewer=lambda: _Outcome(card=PUBLISHABLE_CARD,
                                                 gate_status="PUBLISHABLE"))
    report, client = _deliver(storage, layout)

    assert outcome.recipients == 0
    assert outcome.sent_to_outbox is False
    assert outcome.suppressed_reason == trigger.SUPPRESS_NO_RECIPIENTS
    assert client.sent == []
    assert report.sent == 0

    # ...and the whole briefing is on disk, enrichment included.
    batch = BatchStore(storage, layout).load(
        TENANT, outcome.notification_batch_id)
    assert batch is not None
    message = batch.message(MESSAGE_PORTFOLIO_UPDATE)
    assert any("93.4% of the movement" in i.text for i in message.items)
    assert enrichment.record_of(batch)["status"] == enrichment.ENRICHED


def test_a_shadow_period_is_not_re_notified_when_a_recipient_appears(
        storage, layout, enabled_config, funded, tmp_path):
    """Yesterday's shadow batch must not be delivered as though it were new."""
    from trakt_notifications.recipients import RecipientStore

    first = _approve(storage, layout)
    assert first.suppressed_reason == trigger.SUPPRESS_NO_RECIPIENTS

    store = RecipientStore(storage, layout)
    captured = store.capture_conversation(
        tenant_id=TENANT, microsoft_tenant_id="11111111-1111-1111-1111-111111111111",
        entra_object_id="33333333-3333-3333-3333-333333333333",
        conversation_id="a:2conversation",
        service_url="https://smba.trafficmanager.net/emea/",
        conversation_reference={"conversation": {"id": "a:2conversation"}},
        display_name="Late Arrival")
    store.authorise(TENANT, captured.recipient_id,
                    portfolio_contexts=["total"], actor="operator")

    second = _approve(storage, layout)
    _report, client = _deliver(storage, layout)

    assert second.sent_to_outbox is False
    assert second.suppressed_reason == trigger.SUPPRESS_DUPLICATE
    assert client.sent == []


# =========================================================================== #
# §5 — the verdict enum is internal, and stays internal
# =========================================================================== #
def test_the_period_verdict_never_reaches_the_teams_card(
        storage, layout, recipient, enabled_config, funded):
    """The uninformative enum cannot hold up deployment, because it is not sent.

    The agent kept returning INCOMPLETE_REVIEW on quiet periods — first because
    it read absent deployment capabilities as failed checks, and after the
    prompt was sharpened, because it judged that being unable to test high-LTV
    exposure against approved thresholds is itself material. The second
    reading is defensible, so it was not argued away.

    It does not need to be. `enrichment.attach` takes the card's FINDINGS and
    nothing else: the verdict is a monitoring signal for operators, never a
    line a manager reads. That is asserted here rather than left as a property
    of the current implementation, so a future change that starts rendering it
    fails loudly.
    """
    verdicts = ("INCOMPLETE_REVIEW", "ROUTINE_PERIOD", "ATTENTION_REQUIRED",
                "MATERIAL_DEVELOPMENTS")
    for verdict in verdicts:
        card = {**PUBLISHABLE_CARD, "period_verdict": verdict}
        outcome = _approve(
            storage, layout, run_ids=[f"orun-{verdict.lower()}"],
            reviewer=lambda c=card: _Outcome(card=c, gate_status="PUBLISHABLE"))
        _report, client = _deliver(storage, layout)
        rendered = " ".join(_json(s["card"]) for s in client.sent)
        assert verdict not in rendered, f"{verdict} leaked into the card"


def test_a_quiet_period_still_states_the_factual_baseline(
        storage, layout, recipient, enabled_config, funded, monkeypatch):
    """§3: a still month is a useful management outcome, not a failed review.

    Whatever the agent's internal verdict, the deterministic core carries the
    period's facts and says plainly that nothing material was found — and it
    says it about the MI that was reviewed, not about risk in general.
    """
    from trakt_notifications import generate

    quiet = funded_inputs()
    quiet.funded_brief = {"status": "success", "insights": [], "omitted": []}
    monkeypatch.setattr(sources, "resolve", lambda **_k: quiet)

    batch = generate.build(quiet, update_type="FUNDED")
    message = batch.message(MESSAGE_PORTFOLIO_UPDATE)
    text = " ".join(i.text for i in message.items)

    assert "Funded balance is" in message.summary or "Funded balance is" in text
    assert "No material developments were identified" in text
    # §4: the qualification must survive. This is a claim about the MI
    # reviewed, never a clearance of limits that were never tested.
    assert "all risk limits" not in text.lower()
    assert "within tolerance" not in text.lower()
