"""Delivery classification: new client / new portfolio / recurring / backfill,
operator override, and audit persistence of the classification decision."""

from __future__ import annotations

from apps.blob_trigger_app.source_registry import SourceRecord, STATUS_ACTIVE

from operations_control.classification import classify_delivery
from operations_control.contracts import (
    WF_BACKFILL,
    WF_NEW_CLIENT,
    WF_NEW_PORTFOLIO,
    WF_RECURRING,
)

from .conftest import register_and_create


def _record(fingerprint="sha256:abc", last_period="2026-05-31") -> SourceRecord:
    return SourceRecord(
        client_id="client_a", source_portfolio_id="pf1", dataset="funded",
        frequency="monthly", approved_mapping_id="m1",
        expected_schema_fingerprint=fingerprint,
        last_successful_reporting_period=last_period, status=STATUS_ACTIVE)


def test_unknown_client_is_new_client(source_registry):
    c = classify_delivery(source_registry, client_id="client_a",
                          portfolio_id="pf1", fingerprint="sha256:abc")
    assert c.suggested == WF_NEW_CLIENT


def test_known_client_unknown_portfolio_is_new_portfolio(source_registry):
    source_registry.upsert(_record())
    c = classify_delivery(source_registry, client_id="client_a",
                          portfolio_id="pf2", fingerprint="sha256:zzz")
    assert c.suggested == WF_NEW_PORTFOLIO


def test_schema_drift_is_secondary_onboarding_not_recurring(source_registry):
    """A new schema for an already-approved client must be treated as a
    secondary onboarding, never recurring reporting."""
    source_registry.upsert(_record(fingerprint="sha256:old"))
    c = classify_delivery(source_registry, client_id="client_a",
                          portfolio_id="pf1", fingerprint="sha256:new",
                          reporting_period="2026-06-30")
    assert c.suggested == WF_NEW_PORTFOLIO
    assert "drift" in c.reason


def test_matching_fingerprint_is_recurring(source_registry):
    source_registry.upsert(_record(fingerprint="sha256:abc"))
    c = classify_delivery(source_registry, client_id="client_a",
                          portfolio_id="pf1", fingerprint="sha256:abc",
                          reporting_period="2026-06-30")
    assert c.suggested == WF_RECURRING


def test_earlier_period_is_backfill(source_registry):
    source_registry.upsert(_record(fingerprint="sha256:abc",
                                   last_period="2026-05-31"))
    c = classify_delivery(source_registry, client_id="client_a",
                          portfolio_id="pf1", fingerprint="sha256:abc",
                          reporting_period="2026-01-31")
    assert c.suggested == WF_BACKFILL


def test_operator_override_is_persisted_in_audit(engine, delivery_dir):
    run = register_and_create(engine, delivery_dir,
                              workflow_type=WF_BACKFILL)
    assert run.classification_suggested == WF_NEW_CLIENT
    assert run.workflow_type == WF_BACKFILL
    assert run.classification_overridden is True

    audit = engine.store.list_audit("client_a")
    created = [a for a in audit if a["event"] == "workflow_created"]
    assert created, "workflow_created audit record missing"
    detail = created[-1]["detail"]
    assert detail["classification_suggested"] == WF_NEW_CLIENT
    assert detail["classification_final"] == WF_BACKFILL
    assert detail["classification_overridden"] is True
    assert engine.store.verify_audit_chain("client_a")


def test_suggestion_accepted_when_no_override(engine, delivery_dir):
    run = register_and_create(engine, delivery_dir)
    assert run.workflow_type == WF_NEW_CLIENT
    assert run.classification_overridden is False
