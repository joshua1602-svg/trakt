"""Restart recovery + reconstruction: workflow state survives engine/API
restarts; interrupted executions are detected and resumable; everything is
reconstructable from blob-persisted state alone."""

from __future__ import annotations

from operations_control.contracts import (
    RUN_AWAITING_PUBLICATION,
    RUN_BLOCKED,
    RUN_NEEDS_REVIEW,
    RUN_RUNNING,
)
from operations_control.stores import OpsLayout, OpsStore

from .conftest import make_engine, register_and_create, start_and_wait, wait_for


def test_reconstruction_from_persisted_state(store, storage, source_registry,
                                             delivery_dir):
    """A brand-new store/engine (a 'restarted API') sees the same workflow,
    stages, results, decisions and audit — React state is never the record."""
    engine = make_engine(store, source_registry, "mapping_halt")
    run = register_and_create(engine, delivery_dir)
    start_and_wait(engine, run)

    fresh_store = OpsStore(storage, OpsLayout("operations-control"))
    reconstructed = fresh_store.load_workflow("client_a", run.workflow_id)
    assert reconstructed is not None
    assert reconstructed.status == RUN_NEEDS_REVIEW
    assert reconstructed.stage_status("mapping") == "needs_review"
    gar = fresh_store.load_result("client_a", run.workflow_id, "mapping")
    assert gar is not None and gar.status == "needs_review"
    assert fresh_store.open_decisions("client_a", run.workflow_id)
    assert fresh_store.list_events("client_a", run.workflow_id)
    assert fresh_store.verify_audit_chain("client_a")


def test_interrupted_running_workflow_recovers_and_resumes(
        store, storage, source_registry, delivery_dir):
    engine = make_engine(store, source_registry, "happy")
    run = register_and_create(engine, delivery_dir)
    # Simulate a crash: persisted as running, but no live executor thread.
    run.status = RUN_RUNNING
    store.save_workflow(run)
    store.write_lease(run)

    engine2 = make_engine(OpsStore(storage, OpsLayout("operations-control")),
                          source_registry, "happy")
    recovered = engine2.recover_on_startup()
    assert run.workflow_id in recovered

    parked = engine2.store.load_workflow("client_a", run.workflow_id)
    assert parked.status == RUN_BLOCKED
    assert parked.interrupted is True
    assert parked.blockers and "Run again" in parked.blockers[0]

    # The operator's 'Run again' resumes to completion.
    engine2.rerun(parked, actor="alice")
    final = wait_for(lambda: (
        (lambda r: r if r and r.status == RUN_AWAITING_PUBLICATION else None)(
            engine2.store.load_workflow("client_a", run.workflow_id))))
    assert final.status == RUN_AWAITING_PUBLICATION
    assert final.interrupted is False


def test_rerun_resumes_without_repeating_completed_steps(
        store, source_registry, delivery_dir):
    from .conftest import ScriptedAdapters
    adapters = ScriptedAdapters("mapping_halt")
    engine = make_engine(store, source_registry, adapters=adapters)
    run = register_and_create(engine, delivery_dir)
    start_and_wait(engine, run)
    first_onboards = adapters.calls.count("onboard")
    assert first_onboards == 1

    dec = store.open_decisions("client_a", run.workflow_id)[0]
    engine.resolve_decision(client_id="client_a",
                            decision_id=dec["decision_id"],
                            action="approve", actor="alice",
                            value="accept_mapping", scope="client")
    wait_for(lambda: (
        (lambda r: r if r and r.status == RUN_AWAITING_PUBLICATION else None)(
            store.load_workflow("client_a", run.workflow_id))))
    # Onboard re-ran once (the halted step); stamp/assemble ran once only.
    assert adapters.calls.count("onboard") == 2
    assert adapters.calls.count("assemble") == 1


def test_workflow_index_rebuild(store, source_registry, delivery_dir):
    engine = make_engine(store, source_registry, "happy")
    run = register_and_create(engine, delivery_dir)
    # Wipe the index, then rebuild from the workflow documents.
    store.storage.write_text(store.layout.workflow_index_uri("client_a"), "{}")
    assert store.list_workflows("client_a") == []
    n = store.rebuild_workflow_index("client_a")
    assert n == 1
    assert store.list_workflows("client_a")[0]["workflow_id"] == run.workflow_id
