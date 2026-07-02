"""Tests for artifact discovery/loading (acceptance criterion 11)."""

from __future__ import annotations

import json

from mi_agent_pptx.artifact_loader import load_run_artifacts


def test_loads_canonical_tape(run_dir):
    art = load_run_artifacts(run_dir)
    assert art.has_tape
    assert art.tape_kind == "platform_canonical"
    assert len(art.tape) == 6
    assert art.tape_path is not None


def test_loads_run_state(run_dir):
    art = load_run_artifacts(run_dir)
    assert art.run_state.get("run_id") == "orun_test"


def test_missing_run_dir_degrades(tmp_path):
    art = load_run_artifacts(tmp_path / "does_not_exist")
    assert not art.has_tape
    assert any("not found" in n.lower() for n in art.coverage_notes)


def test_empty_run_dir_degrades(empty_run_dir):
    art = load_run_artifacts(empty_run_dir)
    assert not art.has_tape
    assert not art.has_pipeline
    assert any("no canonical typed tape" in n.lower() for n in art.coverage_notes)


def test_optional_json_artifacts_discovered(run_dir):
    # Drop a risk-monitor artifact and confirm it's picked up.
    (run_dir / "mi_risk_monitor.json").write_text(json.dumps({
        "summary": {"total": 5, "breaches": 1}, "tests": [{"limitId": "x"}],
    }))
    art = load_run_artifacts(run_dir)
    assert art.has_artifact("risk_monitor")
    assert art.artifact("risk_monitor")["summary"]["breaches"] == 1
