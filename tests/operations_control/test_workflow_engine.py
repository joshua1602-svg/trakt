"""Workflow state transitions, review gating, decision resolution, rerun,
duplicate-action protection and idempotent creation."""

from __future__ import annotations

import pytest

from operations_control.contracts import (
    DEC_SUPERSEDED,
    IllegalTransition,
    RUN_AWAITING_PUBLICATION,
    RUN_CANCELLED,
    RUN_NEEDS_REVIEW,
    RUN_PUBLISHED,
    RUN_RECEIVED,
    RUN_RUNNING,
    WorkflowRun,
    transition,
)
from operations_control.engine import OpsError

from .conftest import (
    make_engine,
    register_and_create,
    start_and_wait,
    wait_for,
)


def _bare_run(**kw) -> WorkflowRun:
    base = dict(workflow_id="wf_x", client_id="client_a", portfolio_id="pf1",
                outcome="mi", workflow_type="new_client", delivery={})
    base.update(kw)
    return WorkflowRun(**base)


class TestTransitions:
    def test_legal_paths(self):
        run = _bare_run()
        transition(run, RUN_RUNNING)
        transition(run, RUN_NEEDS_REVIEW)
        transition(run, RUN_RUNNING)
        transition(run, RUN_AWAITING_PUBLICATION)
        transition(run, RUN_PUBLISHED)
        assert run.status == RUN_PUBLISHED

    def test_illegal_transitions_rejected(self):
        run = _bare_run()
        with pytest.raises(IllegalTransition):
            transition(run, RUN_PUBLISHED)          # received -> published
        transition(run, RUN_RUNNING)
        transition(run, RUN_AWAITING_PUBLICATION)
        transition(run, RUN_PUBLISHED)
        with pytest.raises(IllegalTransition):
            transition(run, RUN_RUNNING)            # published is terminal


class TestHappyPath:
    def test_runs_to_awaiting_publication_and_never_publishes(
            self, store, source_registry, delivery_dir):
        engine = make_engine(store, source_registry, "happy")
        run = register_and_create(engine, delivery_dir)
        assert run.status == RUN_RECEIVED
        final = start_and_wait(engine, run)
        assert final.status == RUN_AWAITING_PUBLICATION
        # Publication is prepared but nothing has been published.
        pub = store.load_publication("client_a", "2026-06-30")
        assert pub is not None and pub["status"] == "prepared"
        assert pub["published_at"] is None

    def test_creation_is_idempotent(self, store, source_registry, delivery_dir):
        engine = make_engine(store, source_registry, "happy")
        d = engine.register_delivery(client_id="client_a", portfolio_id="pf1",
                                     input_path=str(delivery_dir),
                                     reporting_period="2026-06-30")
        r1 = engine.create_workflow(client_id="client_a",
                                    delivery_id=d["delivery_id"], outcome="mi")
        r2 = engine.create_workflow(client_id="client_a",
                                    delivery_id=d["delivery_id"], outcome="mi")
        assert r1.workflow_id == r2.workflow_id

    def test_start_twice_is_conflict(self, store, source_registry, delivery_dir):
        engine = make_engine(store, source_registry, "happy")
        run = register_and_create(engine, delivery_dir)
        final = start_and_wait(engine, run)
        # Wait for the executor thread to fully exit so the second start hits
        # the persisted-status guard, not the in-flight idempotency path.
        wait_for(lambda: not engine._is_executing(run.workflow_id))
        with pytest.raises(OpsError) as e:
            engine.start(final, actor="test")
        assert e.value.code == "OPS_ALREADY_RUNNING"


class TestReviewLoop:
    def test_mapping_halt_surfaces_decisions_and_approval_reruns(
            self, store, source_registry, delivery_dir):
        engine = make_engine(store, source_registry, "mapping_halt")
        run = register_and_create(engine, delivery_dir)
        parked = start_and_wait(engine, run)
        assert parked.status == RUN_NEEDS_REVIEW
        assert parked.stage_status("mapping") == "needs_review"

        open_decisions = store.open_decisions("client_a", run.workflow_id)
        mapping = [d for d in open_decisions if d["kind"] in
                   ("field_mapping", "transformation")]
        assert mapping, "expected a mapping decision"
        dec = mapping[0]
        assert dec["blocking"] is True

        result = engine.resolve_decision(
            client_id="client_a", decision_id=dec["decision_id"],
            action="approve", actor="alice", value="accept_mapping",
            scope="client")
        assert result["rule"] is not None
        assert result["rerun_scheduled"] is True

        final = wait_for(lambda: (
            (lambda r: r if r and r.status == RUN_AWAITING_PUBLICATION else None)(
                store.load_workflow("client_a", run.workflow_id))))
        assert final.stage_status("mapping") == "completed"
        # The approved decisions file was produced for the deterministic rerun.
        assert engine._approved_decisions_path(final) is not None

    def test_duplicate_resolution_is_rejected(self, store, source_registry,
                                              delivery_dir):
        engine = make_engine(store, source_registry, "mapping_halt")
        run = register_and_create(engine, delivery_dir)
        start_and_wait(engine, run)
        dec = store.open_decisions("client_a", run.workflow_id)[0]
        engine.resolve_decision(client_id="client_a",
                                decision_id=dec["decision_id"],
                                action="approve", actor="alice",
                                value="accept_mapping", scope="client")
        with pytest.raises(OpsError) as e:
            engine.resolve_decision(client_id="client_a",
                                    decision_id=dec["decision_id"],
                                    action="approve", actor="alice",
                                    value="accept_mapping", scope="client")
        assert e.value.code == "OPS_DECISION_ALREADY_RESOLVED"

    def test_reject_requires_reason_and_persists_no_rule(
            self, store, source_registry, delivery_dir):
        engine = make_engine(store, source_registry, "mapping_halt")
        run = register_and_create(engine, delivery_dir)
        start_and_wait(engine, run)
        dec = store.open_decisions("client_a", run.workflow_id)[0]
        with pytest.raises(OpsError) as e:
            engine.resolve_decision(client_id="client_a",
                                    decision_id=dec["decision_id"],
                                    action="reject", actor="alice")
        assert e.value.code == "OPS_REASON_REQUIRED"
        out = engine.resolve_decision(client_id="client_a",
                                      decision_id=dec["decision_id"],
                                      action="reject", actor="alice",
                                      reason="wrong file")
        assert out["rule"] is None
        assert not engine.rules.list_current("client_a")


class TestValidationException:
    def test_validation_halt_needs_approval_then_force_publishes(
            self, store, source_registry, delivery_dir):
        engine = make_engine(store, source_registry, "validation_halt")
        run = register_and_create(engine, delivery_dir,
                                  outcome="mi_annex2")
        parked = start_and_wait(engine, run)
        assert parked.status == RUN_NEEDS_REVIEW
        assert parked.stage_status("validation") == "needs_review"

        dec = [d for d in store.open_decisions("client_a", run.workflow_id)
               if d["kind"] == "validation_exception"][0]
        result = engine.resolve_decision(
            client_id="client_a", decision_id=dec["decision_id"],
            action="approve", actor="alice", value="proceed",
            reason="known data issue this month")
        assert result["rerun_scheduled"] is True
        final = wait_for(lambda: (
            (lambda r: r if r and r.status == RUN_AWAITING_PUBLICATION else None)(
                store.load_workflow("client_a", run.workflow_id))))
        assert final.status == RUN_AWAITING_PUBLICATION
        # The exception rule is file-scoped — it never generalises silently.
        rules = engine.rules.list_current("client_a")
        exc = [r for r in rules if r.kind == "validation_exception"]
        assert exc and exc[0].scope == "file"


class TestLlmAdvisory:
    def test_llm_suggestion_is_advisory_until_approved(
            self, store, source_registry, delivery_dir):
        """An LLM recommendation attached to a decision creates NO rule until
        the operator approves; the persisted rule records llm provenance."""
        engine = make_engine(store, source_registry, "mapping_halt")
        run = register_and_create(engine, delivery_dir)
        start_and_wait(engine, run)
        dec = store.open_decisions("client_a", run.workflow_id)[0]
        # Simulate an LLM-sourced recommendation on the persisted decision.
        dec["recommendation"] = {"source": "llm", "value": "accept_mapping",
                                 "confidence": 0.95, "checked": True}
        store.save_decision("client_a", dec)

        assert not engine.rules.list_current("client_a")   # advisory only
        result = engine.resolve_decision(
            client_id="client_a", decision_id=dec["decision_id"],
            action="approve", actor="alice", scope="portfolio")
        rule = result["rule"]
        assert rule is not None
        assert rule["suggested_by"] == "llm"
        assert rule["approved_by"] == "alice"


class TestCancellation:
    """A cancelled delivery is kept, but it stops asking and stops appearing."""

    def test_cancelling_closes_the_questions_it_was_still_asking(
            self, store, source_registry, delivery_dir):
        engine = make_engine(store, source_registry, "mapping_halt")
        run = register_and_create(engine, delivery_dir)
        parked = start_and_wait(engine, run)
        assert parked.status == RUN_NEEDS_REVIEW
        assert store.open_decisions("client_a", run.workflow_id)

        engine.cancel(parked, actor="alice", reason="Raised by mistake")

        # Nothing is left in the review queue for a workflow nobody can act on.
        assert store.open_decisions("client_a", run.workflow_id) == []
        assert store.open_decisions("client_a") == []

    def test_a_closed_question_is_superseded_not_answered(
            self, store, source_registry, delivery_dir):
        """Neither approve nor reject was chosen, so neither is recorded."""
        engine = make_engine(store, source_registry, "mapping_halt")
        run = register_and_create(engine, delivery_dir)
        parked = start_and_wait(engine, run)
        decision_id = store.open_decisions("client_a", run.workflow_id)[0][
            "decision_id"]

        engine.cancel(parked, actor="alice", reason="Raised by mistake")

        doc = store.load_decision("client_a", decision_id)
        assert doc["status"] == DEC_SUPERSEDED
        assert doc["resolved_by"] == "alice"
        assert doc["resolution_reason"] == "Raised by mistake"

    def test_the_record_and_the_audit_survive(self, store, source_registry,
                                              delivery_dir):
        engine = make_engine(store, source_registry, "mapping_halt")
        run = register_and_create(engine, delivery_dir)
        parked = start_and_wait(engine, run)
        engine.cancel(parked, actor="alice", reason="Raised by mistake")

        kept = store.load_workflow("client_a", run.workflow_id)
        assert kept is not None and kept.status == RUN_CANCELLED
        events = [e["event"] for e in store.list_audit("client_a")]
        assert "workflow_cancelled" in events
        assert "decision_superseded" in events
        assert store.verify_audit_chain("client_a") is True

    def test_a_published_delivery_cannot_be_cancelled(self):
        """Cancelling would imply the published report went away. It did not."""
        run = _bare_run()
        transition(run, RUN_RUNNING)
        transition(run, RUN_AWAITING_PUBLICATION)
        transition(run, RUN_PUBLISHED)
        with pytest.raises(IllegalTransition):
            transition(run, RUN_CANCELLED)
