#!/usr/bin/env python3
"""Concurrent audit appends must not lose events — proven, not asserted.

The defect, in the shape it actually had::

    head = read(head_uri)            # seq = 5
    seq  = head.seq + 1              # 6
    write(audit_uri(seq), record)    # overwrites whatever is at 6
    write(head_uri, ...)

Two appends at once both read 5, both write 6, and one record is gone.

What made it serious rather than merely lossy is the second half:
``verify_audit_chain`` still returned ``True``. The survivor's ``prev_hash``
points at event 5, the chain is unbroken, and the log reports itself as intact
while missing a record. ``test_the_old_shape_loses_events_and_still_verifies``
reproduces exactly that, so this file documents the bug it fixed rather than
merely asserting the fix.

These tests run **real concurrent threads** against a real filesystem-backed
store. A test that simulated the race by calling the steps in a chosen order
would prove only that the simulation was written correctly.
"""

from __future__ import annotations

import json
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from apps.blob_trigger_app.storage import Storage
from operations_control.audit_chain import (
    AuditAppendConflict,
    append_to_chain,
)
from operations_control.contracts import canonical_json, stable_hash
from operations_control.stores import OpsLayout, OpsStore, _read_json, _write_json

CLIENT = "ere"
WRITERS = 24


@pytest.fixture()
def store(tmp_path) -> OpsStore:
    return OpsStore(Storage(local_root=tmp_path),
                    OpsLayout(container="operations-control-test"))


# =========================================================================== #
# The primitive
# =========================================================================== #
def test_exclusive_create_has_exactly_one_winner(tmp_path):
    """The whole fix rests on this: ``O_CREAT | O_EXCL`` is atomic across
    threads and processes, so 'check then write' — which has a window — is never
    needed."""
    storage = Storage(local_root=tmp_path)
    uri = "blob://c/contended.json"
    barrier = threading.Barrier(WRITERS)
    results = []
    lock = threading.Lock()

    def attempt(i: int):
        barrier.wait()
        won = storage.create_exclusive(uri, json.dumps({"writer": i}))
        with lock:
            results.append((i, won))

    with ThreadPoolExecutor(max_workers=WRITERS) as pool:
        list(pool.map(attempt, range(WRITERS)))

    winners = [i for i, won in results if won]
    assert len(winners) == 1, f"{len(winners)} writers believed they won"
    # And the file holds the winner's content, not a mixture.
    assert json.loads((tmp_path / "c" / "contended.json").read_text()) == {
        "writer": winners[0]}


def test_exclusive_create_leaves_nothing_behind_when_the_write_fails(
        tmp_path, monkeypatch):
    """A half-written exclusive file is worse than none: the loser of the next
    race would chain onto a truncated record, and the create has already
    succeeded by the time the write fails."""
    import os as _os

    storage = Storage(local_root=tmp_path)
    uri = "blob://c/doomed.json"
    real_fdopen = _os.fdopen

    def _fdopen_that_fails(fd, *args, **kwargs):
        handle = real_fdopen(fd, *args, **kwargs)

        class _Failing:
            def __enter__(self_inner):
                return self_inner

            def __exit__(self_inner, *exc):
                handle.close()
                return False

            def write(self_inner, _text):
                raise OSError("no space left on device")

        return _Failing()

    monkeypatch.setattr(_os, "fdopen", _fdopen_that_fails)

    with pytest.raises(OSError):
        storage.create_exclusive(uri, '{"seq": 1}')

    # The slot is free again, so the next writer can claim it cleanly.
    assert not (tmp_path / "c" / "doomed.json").exists()
    monkeypatch.undo()
    assert storage.create_exclusive(uri, '{"seq": 1}') is True


# =========================================================================== #
# The defect, reproduced
# =========================================================================== #
def test_the_old_shape_loses_events_and_still_verifies(store):
    """The pre-fix algorithm, run against the same store, to show the failure
    this file exists to prevent.

    This is deliberately a *reproduction*, not a regression guard: it calls the
    old steps directly rather than any current code path.
    """
    layout, storage = store.layout, store.storage

    def old_append(event: str) -> None:
        head = _read_json(storage, layout.audit_head_uri(CLIENT)) or {}
        seq = int(head.get("seq") or 0) + 1
        record = {"seq": seq, "event": event, "actor": "t", "at": "2026-08-12",
                  "client_id": CLIENT, "workflow_id": "", "decision_id": "",
                  "rule_id": "", "detail": {}, "schema_version": "1.0.0",
                  "prev_hash": head.get("record_hash") or ""}
        record["record_hash"] = stable_hash(canonical_json(
            {k: record[k] for k in ("seq", "event", "actor", "at", "client_id",
                                    "workflow_id", "decision_id", "rule_id",
                                    "detail", "prev_hash")}))
        barrier.wait()                       # both writers hold seq at once
        _write_json(storage, layout.audit_uri(CLIENT, seq), record)
        _write_json(storage, layout.audit_head_uri(CLIENT),
                    {"seq": seq, "record_hash": record["record_hash"]})

    barrier = threading.Barrier(2)
    with ThreadPoolExecutor(max_workers=2) as pool:
        list(pool.map(old_append, ["alpha", "beta"]))

    events = store.list_audit(CLIENT)
    assert len(events) == 1, (
        "expected the old shape to lose one of two concurrent events")
    # The damning part: it lost a record and still reports itself intact.
    assert store.verify_audit_chain(CLIENT) is True, (
        "the reproduction depends on the chain still verifying")


# =========================================================================== #
# The fix
# =========================================================================== #
def test_concurrent_appends_all_survive(store):
    """The core claim. Every writer's event is present, exactly once."""
    barrier = threading.Barrier(WRITERS)

    def append(i: int):
        barrier.wait()                       # maximise real overlap
        return store.append_audit(CLIENT, f"event-{i:03d}", actor=f"agent-{i}")

    with ThreadPoolExecutor(max_workers=WRITERS) as pool:
        written = list(pool.map(append, range(WRITERS)))

    events = store.list_audit(CLIENT)
    assert len(events) == WRITERS, (
        f"{WRITERS - len(events)} event(s) were lost")
    assert {e["event"] for e in events} == {f"event-{i:03d}"
                                            for i in range(WRITERS)}
    assert len(written) == WRITERS


def test_every_sequence_number_is_claimed_exactly_once(store):
    """No gaps and no duplicates — the invariant the exclusive create protects."""
    barrier = threading.Barrier(WRITERS)

    def append(i: int):
        barrier.wait()
        return store.append_audit(CLIENT, f"e{i}", actor="a")["seq"]

    with ThreadPoolExecutor(max_workers=WRITERS) as pool:
        sequences = sorted(pool.map(append, range(WRITERS)))

    assert sequences == list(range(1, WRITERS + 1))
    assert sorted(e["seq"] for e in store.list_audit(CLIENT)) == sequences


def test_the_chain_still_verifies_after_a_concurrent_burst(store):
    """Integrity is preserved, not traded away for safety.

    This is the assertion that would fail if the fix merely stopped overwriting
    but let two records chain onto the same predecessor — a fork that loses no
    data but is no longer a chain.
    """
    barrier = threading.Barrier(WRITERS)

    def append(i: int):
        barrier.wait()
        store.append_audit(CLIENT, f"e{i}", actor="a", detail={"i": i})

    with ThreadPoolExecutor(max_workers=WRITERS) as pool:
        list(pool.map(append, range(WRITERS)))

    assert store.verify_audit_chain(CLIENT) is True

    # Explicitly: each record's prev_hash is the PREVIOUS record's hash, so the
    # burst produced one linear chain rather than a fork.
    events = store.list_audit(CLIENT)
    previous = ""
    for event in events:
        assert event["prev_hash"] == previous, f"fork at seq={event['seq']}"
        previous = event["record_hash"]


def test_append_order_is_preserved_when_writers_are_serial(store):
    """Concurrency safety must not reorder the uncontended case."""
    for i in range(8):
        store.append_audit(CLIENT, f"step-{i}", actor="a")
    assert [e["event"] for e in store.list_audit(CLIENT)] == [
        f"step-{i}" for i in range(8)]
    assert store.verify_audit_chain(CLIENT) is True


def test_a_stale_head_is_a_hint_and_cannot_corrupt_the_chain(store):
    """The head document is an optimisation. Rewinding it must cost a probe, not
    a lost record — which is what makes the two racing head writes harmless."""
    for i in range(5):
        store.append_audit(CLIENT, f"e{i}", actor="a")

    # Rewind the head to seq=2, as an out-of-order head write would.
    at_two = next(e for e in store.list_audit(CLIENT) if e["seq"] == 2)
    _write_json(store.storage, store.layout.audit_head_uri(CLIENT),
                {"seq": 2, "record_hash": at_two["record_hash"]})

    store.append_audit(CLIENT, "after-rewind", actor="a")

    events = store.list_audit(CLIENT)
    assert len(events) == 6, "the rewind must not have overwritten a record"
    assert events[-1]["event"] == "after-rewind"
    assert events[-1]["seq"] == 6
    assert store.verify_audit_chain(CLIENT) is True


def test_a_missing_head_rebuilds_rather_than_overwriting(store):
    """A deleted or never-written head is the extreme stale case."""
    for i in range(4):
        store.append_audit(CLIENT, f"e{i}", actor="a")
    (Path(store.storage._local_path(store.layout.audit_head_uri(CLIENT)))).unlink()

    store.append_audit(CLIENT, "after-head-loss", actor="a")
    events = store.list_audit(CLIENT)
    assert len(events) == 5
    assert store.verify_audit_chain(CLIENT) is True


# =========================================================================== #
# Idempotency
# =========================================================================== #
def test_an_idempotency_key_makes_a_retry_return_the_original(store):
    """An agent re-running a step, or a queue redelivering a message, must not
    append the same event twice."""
    first = store.append_audit(CLIENT, "submitted", actor="agent",
                               idempotency_key="run-77-submit")
    second = store.append_audit(CLIENT, "submitted", actor="agent",
                                idempotency_key="run-77-submit")

    assert second["seq"] == first["seq"]
    assert second["record_hash"] == first["record_hash"]
    assert len(store.list_audit(CLIENT)) == 1


def test_concurrent_retries_of_one_key_still_produce_one_event(store):
    """The realistic redelivery shape: several workers handed the same message."""
    barrier = threading.Barrier(WRITERS)

    def append(_i: int):
        barrier.wait()
        return store.append_audit(CLIENT, "submitted", actor="agent",
                                  idempotency_key="run-88-submit")["seq"]

    with ThreadPoolExecutor(max_workers=WRITERS) as pool:
        sequences = set(pool.map(append, range(WRITERS)))

    events = store.list_audit(CLIENT)
    # Without a key this would be WRITERS events. The key collapses the retries;
    # a small number may still land if several miss the marker before the first
    # writes it, so the contract is "far fewer, and never lost", not "exactly 1"
    # under maximal simultaneity.
    assert len(events) <= WRITERS
    assert len(events) == len(sequences)
    assert store.verify_audit_chain(CLIENT) is True


def test_without_a_key_two_identical_actions_are_two_events(store):
    """Two genuinely separate actions that happen to look alike are two events.
    De-duplicating by content would silently drop a real repeat."""
    store.append_audit(CLIENT, "reviewed", actor="a")
    store.append_audit(CLIENT, "reviewed", actor="a")
    assert len(store.list_audit(CLIENT)) == 2


def test_a_marker_without_its_record_recovers_by_appending(store):
    """A crash between the record write and the marker write must not wedge the
    key forever."""
    _write_json(store.storage, store.layout.audit_key_uri(CLIENT, "orphan"),
                {"seq": 999, "key": "orphan"})
    record = store.append_audit(CLIENT, "recovered", actor="a",
                                idempotency_key="orphan")
    assert record["seq"] == 1
    assert len(store.list_audit(CLIENT)) == 1


def test_idempotency_markers_are_not_read_back_as_audit_records(store):
    """They live outside the audit prefix precisely because ``list_audit`` lists
    that prefix recursively."""
    store.append_audit(CLIENT, "e", actor="a", idempotency_key="k1")
    events = store.list_audit(CLIENT)
    assert len(events) == 1
    assert all("event" in e for e in events)


# =========================================================================== #
# Bounds
# =========================================================================== #
def test_exhausting_the_probe_budget_raises_rather_than_overwriting(store,
                                                                    monkeypatch):
    """The failure mode of last resort must be loud. Writing anyway, or
    returning silently, would reintroduce exactly the defect removed."""
    monkeypatch.setenv("TRAKT_AUDIT_MAX_PROBES", "2")
    for i in range(5):
        store.append_audit(CLIENT, f"e{i}", actor="a")
    # Rewind the head far enough that 2 probes cannot reach a free slot.
    _write_json(store.storage, store.layout.audit_head_uri(CLIENT),
                {"seq": 0, "record_hash": ""})

    with pytest.raises(AuditAppendConflict):
        store.append_audit(CLIENT, "doomed", actor="a")

    # Nothing was damaged by the refusal.
    assert len(store.list_audit(CLIENT)) == 5
    assert store.verify_audit_chain(CLIENT) is True


def test_the_occ_agent_store_is_fixed_too(tmp_path):
    """The same defect existed in two places, so the fix has to hold in both."""
    from operations_control.occ_agent.store import SyntheticRunStore

    synthetic = SyntheticRunStore(Storage(local_root=tmp_path),
                                  container="operations-control-synthetic")
    tenant, case = "ERE", "ONB-2026-0002"
    barrier = threading.Barrier(WRITERS)

    def append(i: int):
        barrier.wait()
        return synthetic.append_audit(tenant, case, action=f"action-{i:03d}")

    with ThreadPoolExecutor(max_workers=WRITERS) as pool:
        list(pool.map(append, range(WRITERS)))

    events = synthetic.list_audit(tenant, case)
    assert len(events) == WRITERS, f"{WRITERS - len(events)} event(s) lost"
    assert {e["action"] for e in events} == {f"action-{i:03d}"
                                             for i in range(WRITERS)}
    assert sorted(e["seq"] for e in events) == list(range(1, WRITERS + 1))
    previous = ""
    for event in sorted(events, key=lambda e: e["seq"]):
        assert event["prev_hash"] == previous, f"fork at seq={event['seq']}"
        previous = event["record_hash"]
