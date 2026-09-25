"""A backfilled month's answers stay with that month.

An answered question is merged into the run's decisions and the result is
written, whole, over the source's standing mapping contract — the file that
lets next month run without asking again. ERE's older months are a different
layout: the interest rate is `Loan Interest Rate`, one month has no property
tape. Answering October's questions would have replaced the contract the
current months run on, and put August's questions back into September.

A backfill keeps its answers in its own record. Every other run still writes
the standing contract.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from operations_control.adapters import APPROVED_DECISIONS_FILE, DECISIONS_FILE
from operations_control.contracts import DEC_APPROVED, WF_BACKFILL, WF_RECURRING

from .conftest import make_client_config, make_engine, register_and_create


@pytest.fixture()
def engine(store, source_registry, tmp_path):
    return make_engine(store, source_registry, "happy",
                       client_config=make_client_config(tmp_path))


def _answered_run(engine, tmp_path, workflow_type, period):
    d = tmp_path / f"pack_{workflow_type}"
    d.mkdir()
    (d / "loans.csv").write_text("loan_ref,balance\nL1,1000\n", encoding="utf-8")
    run = register_and_create(engine, d, period=period,
                              workflow_type=workflow_type)
    staging = engine._staging_dir(run)
    staging.mkdir(parents=True, exist_ok=True)
    (staging / DECISIONS_FILE).write_text(yaml.safe_dump({"decisions": [
        {"decision_id": "D001", "target_field": "current_interest_rate",
         "status": "pending"}]}), encoding="utf-8")
    engine.store.save_decision(run.client_id, {
        "decision_id": f"{run.workflow_id}_D001", "workflow_id": run.workflow_id,
        "status": DEC_APPROVED, "resolution_action": "approve",
        "resolution_value": "provide_source_mapping",
        "resolution_source_column": "Loan Interest Rate",
        "subject": {"decision_id": "D001",
                    "target_field": "current_interest_rate"}})
    return run


def _contract_uri(engine, run):
    return engine.store.layout.approved_decisions_uri(
        run.client_id, run.portfolio_id, run.delivery.get("dataset", "funded"),
        APPROVED_DECISIONS_FILE)


def test_a_backfilled_month_does_not_replace_the_standing_contract(engine, tmp_path):
    run = _answered_run(engine, tmp_path, WF_BACKFILL, "2025-10-31")
    engine._write_approved_decisions_file(run)
    assert not engine.store.storage.exists(_contract_uri(engine, run))


def test_it_keeps_its_answer_in_its_own_run(engine, tmp_path):
    run = _answered_run(engine, tmp_path, WF_BACKFILL, "2025-10-31")
    engine._write_approved_decisions_file(run)
    own = Path(engine._staging_dir(run)) / APPROVED_DECISIONS_FILE
    doc = yaml.safe_load(own.read_text(encoding="utf-8"))
    entry = doc["decisions"][0]
    assert entry["status"] == "approved"
    assert entry["selected_source_column"] == "Loan Interest Rate"


def test_a_current_month_still_writes_the_standing_contract(engine, tmp_path):
    run = _answered_run(engine, tmp_path, WF_RECURRING, "2026-09-30")
    engine._write_approved_decisions_file(run)
    assert engine.store.storage.exists(_contract_uri(engine, run))
