"""Publication gating: never automatic, explicit approval invokes the EXISTING
promotion path, duplicate approvals rejected, onboarding publish promotes the
source record so the next delivery classifies as recurring."""

from __future__ import annotations

from pathlib import Path

import pytest

from operations_control.classification import classify_delivery
from operations_control.contracts import (
    RUN_AWAITING_PUBLICATION,
    RUN_HELD,
    RUN_PUBLISHED,
    WF_RECURRING,
)
from operations_control.engine import OpsError

from .conftest import make_engine, register_and_create, start_and_wait


def _to_awaiting(store, source_registry, delivery_dir):
    engine = make_engine(store, source_registry, "happy")
    run = register_and_create(engine, delivery_dir)
    final = start_and_wait(engine, run)
    assert final.status == RUN_AWAITING_PUBLICATION
    return engine, final


class TestGating:
    def test_agent_completion_never_publishes(self, store, source_registry,
                                              delivery_dir, ops_env):
        _to_awaiting(store, source_registry, delivery_dir)
        latest = (ops_env["blob_root"] / "processed-v2" / "platform"
                  / "client_a" / "latest" / "platform_canonical_typed.csv")
        assert not latest.exists(), "nothing may publish without approval"

    def test_approval_invokes_existing_promotion_path(self, store,
                                                      source_registry,
                                                      delivery_dir, ops_env):
        engine, run = _to_awaiting(store, source_registry, delivery_dir)
        pub = engine.approve_publication(client_id="client_a",
                                         workflow_id=run.workflow_id,
                                         actor="alice")
        assert pub["status"] == "published"
        # The EXISTING layout locations now hold the published canonical.
        latest = (ops_env["blob_root"] / "processed-v2" / "platform"
                  / "client_a" / "latest" / "platform_canonical_typed.csv")
        period = (ops_env["blob_root"] / "processed-v2" / "platform"
                  / "client_a" / "2026-06-30" / "platform_canonical_typed.csv")
        assert latest.exists() and period.exists()

        final = store.load_workflow("client_a", run.workflow_id)
        assert final.status == RUN_PUBLISHED
        # Publication references: workflow, rules, approvals, artefacts.
        assert pub["workflow_id"] == run.workflow_id
        assert "rule_versions" in pub
        assert pub["approved_by"] == "alice"
        assert pub["source_artefacts"]["orchestrator_run_id"]

    def test_duplicate_publication_is_rejected(self, store, source_registry,
                                               delivery_dir):
        engine, run = _to_awaiting(store, source_registry, delivery_dir)
        engine.approve_publication(client_id="client_a",
                                   workflow_id=run.workflow_id, actor="alice")
        with pytest.raises(OpsError) as e:
            engine.approve_publication(client_id="client_a",
                                       workflow_id=run.workflow_id,
                                       actor="alice")
        assert e.value.code in ("OPS_ALREADY_PUBLISHED",
                                "OPS_PUBLICATION_NOT_PREPARED")

    def test_premature_publication_is_rejected(self, store, source_registry,
                                               delivery_dir):
        engine = make_engine(store, source_registry, "mapping_halt")
        run = register_and_create(engine, delivery_dir)
        start_and_wait(engine, run)          # parked at needs_review
        with pytest.raises(OpsError) as e:
            engine.approve_publication(client_id="client_a",
                                       workflow_id=run.workflow_id,
                                       actor="alice")
        assert e.value.code == "OPS_PUBLICATION_NOT_PREPARED"

    def test_hold_requires_reason_and_parks_run(self, store, source_registry,
                                                delivery_dir):
        engine, run = _to_awaiting(store, source_registry, delivery_dir)
        with pytest.raises(OpsError):
            engine.reject_publication(client_id="client_a",
                                      workflow_id=run.workflow_id,
                                      actor="alice", reason="")
        engine.reject_publication(client_id="client_a",
                                  workflow_id=run.workflow_id,
                                  actor="alice", reason="waiting on sign-off")
        assert store.load_workflow("client_a",
                                   run.workflow_id).status == RUN_HELD


class TestSourcePromotion:
    def test_onboarding_publish_promotes_source_for_recurring_routing(
            self, store, source_registry, delivery_dir):
        engine, run = _to_awaiting(store, source_registry, delivery_dir)
        assert run.workflow_type == "new_client"
        engine.approve_publication(client_id="client_a",
                                   workflow_id=run.workflow_id, actor="alice")

        source_registry.load()
        rec = source_registry.lookup("client_a", "pf1", "funded", "monthly")
        assert rec is not None and rec.status == "active"
        assert rec.mapping_version == 1
        assert rec.expected_schema_fingerprint == \
            run.delivery["schema_fingerprint"]

        # The SAME delivery schema next month now classifies as recurring.
        c = classify_delivery(source_registry, client_id="client_a",
                              portfolio_id="pf1",
                              fingerprint=run.delivery["schema_fingerprint"],
                              reporting_period="2026-07-31")
        assert c.suggested == WF_RECURRING
