"""Approval lifecycle: nothing activates without an explicit human approval,
history is immutable, and every decision lands on the audit chain."""

from __future__ import annotations

import pytest

from mi_agent.concentration_tests.models import (
    ActiveConfiguration,
    PROPOSAL_APPROVED,
    PROPOSAL_PENDING_APPROVAL,
)
from operations_control.concentration import ConcentrationError

CLAUSE = ("1.1 The aggregate Current Balance of Loans secured on Properties "
          "located in East Anglia must not exceed 25% of the Portfolio.")
NET_WAC = "5.1 The Net WAC of the Portfolio must be at least 3.75%."


def extract(service, text=CLAUSE, client="client_a"):
    return service.extract(client, actor="Alice",
                           source_reference="covenant.txt", text=text)


class TestApprovalGate:
    def test_a_proposal_cannot_activate_without_approval(self, service):
        extract(service)
        with pytest.raises(ConcentrationError, match="nothing to activate"):
            service.activate("client_a", actor="Alice")

    def test_an_ambiguous_proposal_cannot_be_approved(self, service):
        [p] = extract(service, NET_WAC)
        with pytest.raises(ConcentrationError, match="Unresolved concerns"):
            service.approve("client_a", p.proposal_id, actor="Alice")

    def test_a_rejected_proposal_remains_inactive(self, service):
        [p] = extract(service)
        service.reject("client_a", p.proposal_id, actor="Alice",
                       reason="not in the amended facility")
        with pytest.raises(ConcentrationError):
            service.activate("client_a", actor="Alice")

    def test_not_applicable_is_recorded_and_stays_inactive(self, service):
        [p] = extract(service)
        out = service.mark_not_applicable("client_a", p.proposal_id,
                                          actor="Alice",
                                          reason="facility repaid")
        assert out.status == "not_applicable"
        assert any("facility repaid" in n["note"] for n in out.operator_notes)

    def test_approval_records_operator_time_metric_and_parameters(self, service):
        [p] = extract(service)
        approved = service.approve("client_a", p.proposal_id, actor="Alice",
                                   effective_date="2026-01-31",
                                   comments="Matches clause 1.1")
        assert approved.status == PROPOSAL_APPROVED
        assert approved.approval["operator"] == "Alice"
        assert approved.approval["decided_at"]
        assert approved.approval["approved_metric_id"] == "geo_region_share"
        assert approved.approval["approved_parameters"] == \
            {"regions": ["East Anglia"]}
        assert approved.approval["effective_date"] == "2026-01-31"

    def test_operator_edits_are_versioned_and_recorded(self, service):
        [p] = extract(service, NET_WAC)
        service.answer_confirmation(
            "client_a", p.proposal_id, actor="Alice",
            question=p.confirmation_questions[0],
            answer="Servicing only, 50bp.")
        edited = service.update_proposal(
            "client_a", p.proposal_id, actor="Alice",
            parameters={"deduction_percent": 0.5,
                        "deduction_basis": "servicing_fee_only"},
            resolved_concerns=["net_wac_definition_uncertain"])
        assert edited.version == 2
        assert edited.status == PROPOSAL_PENDING_APPROVAL
        assert edited.confirmation_answers[0]["answered_by"] == "Alice"

    def test_invalid_parameter_edits_are_refused(self, service):
        [p] = extract(service)
        with pytest.raises(ConcentrationError, match="not declared"):
            service.update_proposal("client_a", p.proposal_id, actor="Alice",
                                    parameters={"regions": ["Wales"],
                                                "formula": "x"})

    def test_approved_proposal_is_immutable(self, service):
        [p] = extract(service)
        service.approve("client_a", p.proposal_id, actor="Alice")
        with pytest.raises(ConcentrationError, match="immutable"):
            service.update_proposal("client_a", p.proposal_id, actor="Alice",
                                    threshold=30.0)


class TestActivationAndHistory:
    def test_activation_is_versioned_immutable_and_idempotent(self, service):
        [p] = extract(service)
        service.approve("client_a", p.proposal_id, actor="Alice")
        v1 = service.activate("client_a", actor="Alice", reason="initial")
        assert v1.version == 1
        assert v1.tests[0].approval.operator == "Alice"
        again = service.activate("client_a", actor="Alice", reason="repeat")
        assert again.version == 1  # identical content cannot fork history

    def test_supersede_keeps_prior_versions_intact(self, service):
        [p] = extract(service)
        service.approve("client_a", p.proposal_id, actor="Alice")
        service.activate("client_a", actor="Alice")
        service.supersede("client_a", p.proposal_id, actor="Alice",
                          reason="amended")
        [p2] = extract(service, CLAUSE.replace("25%", "20%"))
        service.approve("client_a", p2.proposal_id, actor="Bob")
        v2 = service.activate("client_a", actor="Bob", reason="amendment")
        assert v2.version == 2
        assert v2.based_on_version == 1
        v1 = service.store.get_version("client_a", 1)
        assert v1.status == "superseded"
        assert v1.tests[0].threshold == 25.0  # history unchanged
        assert service.store.current_configuration("client_a").version == 2

    def test_effective_date_travels_to_the_active_test(self, service):
        [p] = extract(service)
        service.approve("client_a", p.proposal_id, actor="Alice",
                        effective_date="2026-06-30")
        config = service.activate("client_a", actor="Alice")
        assert config.tests[0].effective_date == "2026-06-30"

    def test_every_decision_lands_on_the_audit_chain(self, service, ops_store):
        [p] = extract(service)
        service.approve("client_a", p.proposal_id, actor="Alice")
        service.activate("client_a", actor="Alice")
        events = [e["event"] for e in ops_store.list_audit("client_a")]
        assert "concentration_tests_extracted" in events
        assert "concentration_proposal_approved" in events
        assert "concentration_configuration_activated" in events
        assert ops_store.verify_audit_chain("client_a")


class TestTenantIsolation:
    def test_proposals_and_versions_are_tenant_scoped(self, service):
        [p] = extract(service, client="client_a")
        service.approve("client_a", p.proposal_id, actor="Alice")
        service.activate("client_a", actor="Alice")
        assert service.store.list_proposals("client_b") == []
        assert service.store.current_configuration("client_b") is None

    def test_a_hostile_client_id_is_refused_at_the_store(self, service):
        with pytest.raises(ValueError):
            service.store.list_proposals("../../other")

    def test_configuration_hash_ignores_minted_identifiers(self):
        from mi_agent.concentration_tests.models import ActiveTest
        a = ActiveConfiguration(client_id="c", tests=[ActiveTest(
            metric_id="geo_region_share", threshold=25.0)])
        b = ActiveConfiguration(client_id="c", tests=[ActiveTest(
            metric_id="geo_region_share", threshold=25.0)])
        assert a.compute_content_hash() == b.compute_content_hash()
