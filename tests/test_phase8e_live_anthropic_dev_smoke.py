"""Phase 8E — guards for the dev-only live Anthropic smoke.

These tests prove SAFE behaviour only. They NEVER call Anthropic, never require
ANTHROPIC_API_KEY, and never import the anthropic SDK:

  * the smoke script refuses to run (exit 2) when the key is missing;
  * importing the script does not pull in the anthropic SDK or hit the network;
  * the smoke question set matches the controlled Phase 8A/8D set;
  * invalid/ambiguous LLM output still does not execute (via the bridge with
    fake clients — the same governed path the live smoke uses).

No live API calls, no API keys, no Azure/Streamlit/legacy-analytics imports.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts" / "phase8e_live_anthropic_smoke.py"


def _load_script():
    spec = importlib.util.spec_from_file_location("phase8e_smoke", SCRIPT_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# --------------------------------------------------------------------------- #
# 1. Missing API key fails gracefully
# --------------------------------------------------------------------------- #


def test_missing_api_key_exits_gracefully(monkeypatch, tmp_path):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    mod = _load_script()
    code = mod.run_smoke(out_path=tmp_path / "out.json")
    assert code == 2
    # Nothing should have been written, and no SDK should have been imported.
    assert not (tmp_path / "out.json").exists()
    assert "anthropic" not in sys.modules


def test_main_missing_key_returns_two(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    mod = _load_script()
    assert mod.main([]) == 2


# --------------------------------------------------------------------------- #
# 2. No anthropic SDK imported merely by importing the script
# --------------------------------------------------------------------------- #


def test_import_does_not_pull_anthropic_sdk():
    sys.modules.pop("anthropic", None)
    _load_script()
    assert "anthropic" not in sys.modules


def test_script_not_collected_as_tests():
    # The script lives under scripts/ and has no test_* functions, so pytest does
    # not collect it. Assert the latter property explicitly.
    mod = _load_script()
    assert not [n for n in dir(mod) if n.startswith("test_")]


# --------------------------------------------------------------------------- #
# 3. Controlled question set matches the documented Phase 8A/8D set
# --------------------------------------------------------------------------- #


def test_smoke_question_set():
    mod = _load_script()
    questions = [q for q, _ in mod.SMOKE_QUESTIONS]
    expected_execute = {
        "show total funded", "show total pipeline", "show forecast funded",
        "trend funded balance over the last three months",
        "compare funded balance to last month",
        "show funded balance by portfolio", "show funded balance by region",
        "show pipeline by stage", "show concentration by region",
        "show risk grade migration", "show IFRS stage migration",
        "show PD bucket migration",
    }
    expected_clarify = {"show risk", "show changes", "show stage",
                        "show portfolio", "show rate"}
    assert expected_execute | expected_clarify == set(questions)
    behaviours = dict(mod.SMOKE_QUESTIONS)
    for q in expected_execute:
        assert behaviours[q] == "execute"
    for q in expected_clarify:
        assert behaviours[q] == "clarify"


# --------------------------------------------------------------------------- #
# 4. Invalid / ambiguous LLM output still does not execute (fake clients only)
# --------------------------------------------------------------------------- #


class _FakeClient:
    def __init__(self, response):
        self.response = response

    def complete_mi_spec_json(self, prompt):
        return self.response


@pytest.fixture
def smoke_env():
    """A synthetic store + semantics built by the script's own helper."""
    import tempfile

    mod = _load_script()
    from mi_agent.interpreter import InterpreterContext
    from mi_agent.mi_query_validator import load_mi_semantics
    from mi_agent.risk_monitor import load_risk_monitor_config

    tmp = tempfile.TemporaryDirectory()
    store = mod._build_store(tmp.name)
    semantics = load_mi_semantics(
        REPO_ROOT / "mi_agent" / "mi_semantics_field_registry.yaml")
    risk_cfg = load_risk_monitor_config()
    ctx = InterpreterContext(snapshot_client_id=mod.CLIENT, route_id="mi",
                             as_of="2024-03-31", prev_period="2024-02-29",
                             range_start="2024-01-01")
    yield ctx, store, semantics, risk_cfg
    tmp.cleanup()


def test_malformed_output_not_executed(smoke_env):
    from mi_agent.interpreter import interpret_and_run_mi_query

    ctx, store, semantics, risk_cfg = smoke_env
    r = interpret_and_run_mi_query("show total funded", ctx,
                                   _FakeClient("not json {"), store,
                                   semantics=semantics, risk_config=risk_cfg)
    assert not r.executed and r.runtime_result is None


def test_invalid_enum_not_executed(smoke_env):
    from mi_agent.interpreter import interpret_and_run_mi_query

    ctx, store, semantics, risk_cfg = smoke_env
    raw = json.dumps({"execution_mode": "state", "state": "not_a_state",
                      "temporal_mode": "latest"})
    r = interpret_and_run_mi_query("show total funded", ctx, _FakeClient(raw),
                                   store, semantics=semantics,
                                   risk_config=risk_cfg)
    assert not r.executed
    assert "invalid_enum_value" in r.issue_codes()


def test_valid_fake_output_executes(smoke_env):
    # Proves the governed path the live smoke uses does run a valid spec — with a
    # FAKE client, so no Anthropic access is needed.
    from mi_agent.interpreter import interpret_and_run_mi_query

    ctx, store, semantics, risk_cfg = smoke_env
    raw = json.dumps({"execution_mode": "state", "state": "total_funded",
                      "temporal_mode": "latest"})
    r = interpret_and_run_mi_query("show total funded", ctx, _FakeClient(raw),
                                   store, semantics=semantics,
                                   risk_config=risk_cfg)
    assert r.executed and r.runtime_result.ok
    assert r.runtime_result.row_count == 3


# --------------------------------------------------------------------------- #
# 5. Evaluation/grading helper behaves correctly (no network)
# --------------------------------------------------------------------------- #


def test_grade_helper(smoke_env):
    from mi_agent.interpreter import interpret_and_run_mi_query

    mod = _load_script()
    ctx, store, semantics, risk_cfg = smoke_env

    good = json.dumps({"execution_mode": "state", "state": "total_funded",
                       "temporal_mode": "latest"})
    r_exec = interpret_and_run_mi_query("show total funded", ctx,
                                        _FakeClient(good), store,
                                        semantics=semantics, risk_config=risk_cfg)
    assert mod._grade("execute", r_exec) is True
    assert mod._grade("clarify", r_exec) is False

    clar = json.dumps({"clarification_required": True,
                       "clarification_question": "Which view?"})
    r_clar = interpret_and_run_mi_query("show risk", ctx, _FakeClient(clar),
                                        store, semantics=semantics,
                                        risk_config=risk_cfg)
    assert mod._grade("clarify", r_clar) is True
    assert mod._grade("execute", r_clar) is False
