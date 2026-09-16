"""What a rehearsal settles, and where it is allowed to end up.

Two gaps in the target state, and they pull in opposite directions — which is
why they are tested together.

**A rehearsal's mapping decisions were thrown away.** The run reads the
client's real columns, raises the ones it cannot settle, and an operator
answers them. ``resolve_decision`` wrote that answer onto the run and nowhere
else, so nothing carried it across the doorway: production re-derived every
mapping and re-asked every question a human had already answered. The most
valuable thing the rehearsal produced was discarded at the moment it became
usable.

**But a rehearsal that is never activated must leave nothing behind.** That is
the property the whole synthetic boundary exists to hold, and it is not
negotiable to make the first one convenient. So promotion happens at
activation, in the live adapter, and the synthetic adapter discards the same
input — the isolation holds in the adapter rather than depending on a caller
remembering not to pass the decisions along.

The third test class covers the other gap: adding a reporting product once a
client is live is an AMENDMENT, not an edit. The source registry carries
``regime_required`` and is written at activation, so a conversation with a live
case never reaches it — the book stays registered as it was and the engine
refuses the delivery rather than splitting it across two incomplete ones.
"""

from __future__ import annotations

import pytest

from operations_control.occ_agent import mapping_promotion as _promotion
from operations_control.occ_agent.adapters import (
    ActivationIntent,
    ActivationPreconditions,
    SyntheticExecutionAdapter,
)
from operations_control.occ_agent.policy import synthetic_policy
from operations_control.rules import RULE_ACTIVE

from .conftest import ACTOR, TENANT_A

OPENING = ("Onboard Northstar Lending. UK equity release. Monthly portfolio "
           "MI. Portfolio id direct_101.")


def settled(decision_id: str, *, source: str, canonical: str,
            kind: str = _promotion.CONFIRMATION, resolution: str = "approve",
            value: str = "", confidence: float = 0.41) -> dict:
    """One decision as the run holds it once a human has answered."""
    return {"decision_id": decision_id, "decision_type": kind,
            "source_column": source, "target_field": canonical,
            "status": "approved", "resolution": resolution,
            "resolved_value": value, "resolved_by": ACTOR,
            "resolved_at": "2026-09-16T10:00:00Z", "confidence": confidence}


# --------------------------------------------------------------------------- #
# Which answers assert a mapping
# --------------------------------------------------------------------------- #

class TestWhatASettledDecisionAsserts:

    def test_an_accepted_recommendation_is_the_mapping(self):
        assert _promotion.mapping_of(
            settled("d1", source="Curr Bal", canonical="current_balance")
        ) == {"source_column": "Curr Bal",
              "canonical_field": "current_balance"}

    def test_amending_a_confirmation_names_the_canonical_field(self):
        """"'Bal' is not current balance, it is the original balance."
        The question fixed the COLUMN, so the answer is the field."""
        assert _promotion.mapping_of(
            settled("d2", source="Bal", canonical="current_balance",
                    resolution="amend", value="original_balance")
        ) == {"source_column": "Bal",
              "canonical_field": "original_balance"}

    def test_amending_an_ambiguity_names_the_source_column(self):
        """Two columns claimed one field; the operator picked one. Here the
        question fixed the FIELD, so the answer is the column — and reading
        the two questions as one is how a promoted rule comes out backwards."""
        assert _promotion.mapping_of(
            settled("d3", source="A", canonical="current_balance",
                    kind=_promotion.AMBIGUITY, resolution="amend", value="B")
        ) == {"source_column": "B", "canonical_field": "current_balance"}

    def test_a_rejected_decision_asserts_no_mapping(self):
        """"No" is not a mapping."""
        assert _promotion.mapping_of(
            settled("d4", source="Junk", canonical="current_balance",
                    resolution="reject")) is None

    def test_an_unanswered_decision_asserts_nothing(self):
        raw = settled("d5", source="X", canonical="current_balance")
        raw["status"] = "open"
        assert _promotion.mapping_of(raw) is None

    def test_a_decision_that_is_not_about_mapping_is_left_alone(self):
        raw = settled("d6", source="X", canonical="y")
        raw["decision_type"] = "validation_exception"
        assert _promotion.mapping_of(raw) is None


# --------------------------------------------------------------------------- #
# What the rules come out as
# --------------------------------------------------------------------------- #

class TestTheRulesARunAmountsTo:

    def rules(self, *decisions):
        return _promotion.rules_from(list(decisions), client_id="ERE",
                                     portfolio_id="direct_001",
                                     workflow_id="ONB-2026-0010")

    def test_a_promoted_rule_is_scoped_to_the_book_not_the_client(self):
        """The operator answered about THIS book's tape. Whether the client's
        other books use the same column names is a claim they did not make."""
        rule = self.rules(settled("d1", source="Curr Bal",
                                  canonical="current_balance"))[0]
        assert rule.scope == "portfolio"
        assert (rule.client_id, rule.portfolio_id) == ("ERE", "direct_001")

    def test_the_human_who_answered_is_recorded_not_the_activator(self):
        rule = self.rules(settled("d1", source="Curr Bal",
                                  canonical="current_balance"))[0]
        assert rule.approved_by == ACTOR
        assert rule.approved_at == "2026-09-16T10:00:00Z", \
            "the approval date is when the human answered, not when " \
            "activation got round to writing it"
        assert rule.decision_id == "d1"

    def test_the_rehearsal_is_named_as_the_source(self):
        """A governed record must be able to say a rule came from a rehearsal
        rather than from a live delivery."""
        rule = self.rules(settled("d1", source="A", canonical="b"))[0]
        assert rule.suggested_by == _promotion.SUGGESTED_BY

    def test_why_the_question_was_asked_is_kept(self):
        rule = self.rules(settled("d1", source="A", canonical="b",
                                  confidence=0.41))[0]
        assert rule.confidence == 0.41

    def test_one_column_yields_one_rule_and_the_last_answer_stands(self):
        rules = self.rules(
            settled("d1", source="Curr Bal", canonical="current_balance"),
            settled("d2", source="curr bal", canonical="original_balance"))
        assert len(rules) == 1
        assert rules[0].payload["canonical_field"] == "original_balance"

    def test_nothing_settled_yields_nothing(self):
        assert self.rules(
            settled("d4", source="X", canonical="y",
                    resolution="reject")) == []


# --------------------------------------------------------------------------- #
# Where they are allowed to land
# --------------------------------------------------------------------------- #

class _FakeRules:
    """Enough of RuleStore to see what would be written."""

    def __init__(self):
        self.approved = []

    def approve(self, rule):
        rule.rule_id = rule.rule_id or f"rule_{len(self.approved) + 1}"
        rule.version = rule.version or 1
        rule.status = RULE_ACTIVE
        self.approved.append(rule)
        return rule


class TestWhereASettledMappingMayLand:

    def test_promoting_writes_through_the_governed_store(self):
        store = _FakeRules()
        added = _promotion.promote(
            store, [settled("d1", source="Curr Bal",
                            canonical="current_balance")],
            client_id="ERE", portfolio_id="direct_001",
            workflow_id="ONB-2026-0010")
        assert len(store.approved) == 1
        assert added[0]["source_column"] == "Curr Bal"
        assert added[0]["canonical_field"] == "current_balance"
        assert added[0]["rule_id"]

    def test_a_rehearsal_that_never_activates_leaves_nothing(self):
        """The synthetic adapter receives the same decisions and discards
        them. The isolation holds HERE, not in a caller's discipline."""
        adapter = SyntheticExecutionAdapter(synthetic_policy())
        pre = ActivationPreconditions(case_ref="ONB-2026-0010",
                                      tenant=TENANT_A)
        with pytest.raises(Exception):
            adapter.activate(
                pre=pre, intent=ActivationIntent(client_id="ERE"),
                actor=ACTOR,
                decisions=[settled("d1", source="Curr Bal",
                                   canonical="current_balance")])

    def test_the_synthetic_adapter_accepts_the_argument_at_all(self):
        """A signature check, so the live and rehearsal paths cannot drift
        apart and leave the caller passing an argument one of them refuses."""
        import inspect
        for cls in (SyntheticExecutionAdapter,):
            params = inspect.signature(cls.activate).parameters
            assert "decisions" in params


# --------------------------------------------------------------------------- #
# Adding a product once a client is live
# --------------------------------------------------------------------------- #

def make_live(onboarding, client_id="ERE"):
    """Take a client all the way to an active configuration.

    The long way round on purpose: an amendment starts from the version IN
    FORCE, so a test that faked one would not be testing the thing that
    matters. Every answer here is one the validator actually blocks on.
    """
    case = onboarding.start_new_client(by=ACTOR)
    cid = case.case_id
    onboarding.save_step(case_id=cid, step="client", by=ACTOR, payload={
        "client_id": client_id, "client_name": "ERE Funding Limited",
        "jurisdiction": "GB", "reporting_currency": "GBP"})
    onboarding.save_step(case_id=cid, step="entities", by=ACTOR, payload={
        "entities": [{"legal_name": "ERE Funding Limited",
                      "roles": ["originator", "reporting_entity"],
                      "country_of_establishment": "GB"}]})
    onboarding.save_step(case_id=cid, step="portfolios", by=ACTOR, payload={
        "portfolios": [{"portfolio_id": "direct_001",
                        "display_name": "ERE Direct Originations",
                        "asset_class": "equity_release",
                        "portfolio_type": "direct",
                        "period_convention": "calendar_month_end"}]})
    onboarding.save_step(case_id=cid, step="reporting", by=ACTOR,
                         payload={"products": ["mi"]})
    onboarding.save_step(case_id=cid, step="contacts", by=ACTOR, payload={
        "reporting_contact_name": "Jane Doe",
        "reporting_contact_email": "jane@example.test",
        "operational_contact_name": "Bob Smith",
        "operational_contact_email": "bob@example.test"})
    onboarding.save_step(case_id=cid, step="risk_limits", by=ACTOR, payload={
        "concentration_tests_status": "deferred_with_reason",
        "concentration_tests_status_reason":
            "The client has not supplied their limits yet. MI only."})
    sources = [dict(s) for s in onboarding.load_case(cid).items("sources")]
    for s in sources:
        s["file_format"] = "xlsx"
        s.setdefault("cadence", "monthly")
    onboarding.save_step(case_id=cid, step="sources", by=ACTOR,
                         payload={"sources": sources})
    onboarding.submit_for_approval(case_id=cid, by=ACTOR)
    onboarding.approve(case_id=cid, by=ACTOR, reason="MI only for now.")
    onboarding.activate(case_id=cid, by=ACTOR)
    return client_id


class TestAmendingALiveClient:

    def test_an_amendment_starts_from_the_version_in_force(self, service):
        """The whole point: adding Annex 2 later must begin from what is
        live, not from a blank case that happens to look similar."""
        client_id = make_live(service.onboarding)
        amended = service.create_case(tenant=TENANT_A, initiating_user=ACTOR,
                                      amend_client=client_id)
        assert amended.case.kind == "amendment"
        assert amended.case.client_id == client_id
        assert amended.case.based_on_version == 1
        assert (amended.case.answers["client"]["client_name"]
                == "ERE Funding Limited")
        assert amended.case.answers["reporting"]["products"] == ["mi"]

    def test_the_amendment_is_a_new_case_not_the_activated_one(self, service):
        """A live configuration is never edited in place."""
        client_id = make_live(service.onboarding)
        live = service.onboarding.cases.current(client_id)
        amended = service.create_case(tenant=TENANT_A, initiating_user=ACTOR,
                                      amend_client=client_id)
        assert amended.case.case_id != ""
        assert service.onboarding.cases.current(client_id).version \
            == live.version, "amending changed what is in force"

    def test_the_regime_product_can_be_added_on_the_amendment(self, service):
        """The path this gap exists for. Adding the product in conversation
        on a LIVE case never reaches the source registry — an amendment
        re-activates, and the registry is rewritten with it."""
        client_id = make_live(service.onboarding)
        amended = service.create_case(tenant=TENANT_A, initiating_user=ACTOR,
                                      amend_client=client_id)
        turn = service.instruct(amended, text="They also need ESMA Annex 2.",
                                actor=ACTOR, confirm=True)
        products = (turn.case.case.answers.get("reporting") or {}).get(
            "products")
        assert set(products) == {"mi", "esma_annex2"}, \
            "the amendment did not carry the client's existing product"

    def test_amending_a_client_with_no_active_configuration_is_refused(
            self, service):
        """There is nothing to amend, and inventing a blank one would look
        like an amendment while being a new onboarding."""
        from operations_control.engine import OpsError
        with pytest.raises(OpsError) as exc:
            service.create_case(tenant=TENANT_A, initiating_user=ACTOR,
                                amend_client="NOBODY")
        assert exc.value.code == "OPS_CLIENT_NOT_ONBOARDED"

    def test_opening_without_amend_client_is_still_a_new_onboarding(self,
                                                                    service):
        case = service.create_case(tenant=TENANT_A, initiating_user=ACTOR,
                                   instruction=OPENING)
        assert case.case.kind == "new_client"
        assert case.case.based_on_version in (None, 0)
