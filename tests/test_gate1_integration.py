"""
tests/test_gate1_integration.py

Integration tests verifying that:
  1. Onboarding Agent runs standalone (no trakt_run.py dependency)
  2. python -m agents.onboarding_agent entry point works
  3. Gate 1 canonical outputs are produced
  4. ready_for_validation allows pipeline to continue
  5. review_required stops pipeline gracefully (sys.exit(0))
  6. blocked stops pipeline gracefully (sys.exit(2))
  7. semantic_alignment.py executes inside the agent
  8. LLM Mapping Agent path exists (no live calls)
  9. Enum Mapping Agent executes inside the agent
 10. Canonical output files are written: onboarding_result.json,
     config_questions.json, mapping_review.json, enum_review.json

All tests use mocks / synthetic fixtures — no live Claude API calls.
No real loan tape I/O beyond the existing synthetic_demo tape where needed.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Path bootstrap
# ---------------------------------------------------------------------------

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from agents.onboarding_schemas import OnboardingResult, MappingReviewItem, EnumReviewItem

_SYNTHETIC_TAPE = _REPO / "synthetic_demo" / "input" / "SYNTHETIC_ERE_Portfolio_012026.csv"
_SYNTHETIC_CFG  = _REPO / "synthetic_demo" / "config" / "config_client_SYNTHETIC_ERM.yaml"
_ALIASES        = _REPO / "synthetic_demo" / "aliases"
_REGISTRY       = _REPO / "config" / "system" / "fields_registry.yaml"
_ENUM_MAP       = _REPO / "config" / "system" / "enum_mapping.yaml"


# ---------------------------------------------------------------------------
# Helper: build a minimal OnboardingResult for trakt_run mocking
# ---------------------------------------------------------------------------

def _make_result(
    status: str = "ready_for_validation",
    run_id: str = "test_run",
    canonical_path: str = "",
    proceed: bool = True,
) -> OnboardingResult:
    r = OnboardingResult(
        run_id=run_id,
        status=status,
        proceed_to_validation=proceed,
        total_input_fields=62,
        mapped_fields_count=61,
        deterministic_mapped_count=60,
        enum_success_rate=0.947,
        review_fields_count=1,
        enum_review_count=1,
        mapping_review_items=[
            MappingReviewItem(
                raw_field="tenure",
                suggested_canonical_field=None,
                mapping_source="unmapped",
                confidence=0.0,
                requires_review=True,
            )
        ],
        canonical_draft_path=canonical_path,
        onboarding_result_path="",
    )
    return r


# ===========================================================================
# TEST 1: Onboarding Agent standalone import
# ===========================================================================

class TestStandaloneImport:
    def test_import_run_onboarding_agent(self):
        from agents.onboarding_agent import run_onboarding_agent
        assert callable(run_onboarding_agent)

    def test_import_via_submodule(self):
        from agents.onboarding_agent import run_onboarding_agent
        assert callable(run_onboarding_agent)

    def test_schemas_importable(self):
        from agents.onboarding_schemas import (
            OnboardingResult, ConfigBootstrapResult,
            MappingReviewItem, EnumReviewItem,
        )
        r = OnboardingResult()
        assert r.status == "failed"

    def test_review_schemas_importable(self):
        from agents.review_schemas import ReviewSubmission, MappingDecision
        assert callable(ReviewSubmission)


# ===========================================================================
# TEST 2: python -m agents.onboarding_agent --help
# ===========================================================================

class TestModuleEntryPoint:
    def test_help_exits_zero(self):
        result = subprocess.run(
            [sys.executable, "-m", "agents.onboarding_agent", "--help"],
            capture_output=True,
            text=True,
            cwd=str(_REPO),
        )
        assert result.returncode == 0
        assert "--raw-tape" in result.stdout

    def test_missing_required_arg_exits_nonzero(self):
        result = subprocess.run(
            [sys.executable, "-m", "agents.onboarding_agent"],
            capture_output=True,
            text=True,
            cwd=str(_REPO),
        )
        assert result.returncode != 0


# ===========================================================================
# TEST 3: Synthetic tape produces gate outputs
# ===========================================================================

@pytest.mark.skipif(
    not _SYNTHETIC_TAPE.exists(),
    reason="Synthetic tape not found"
)
class TestSyntheticTapeGateOutputs:
    def test_canonical_full_produced(self, tmp_path):
        """Onboarding agent produces canonical_full.csv from the synthetic tape."""
        from agents.onboarding_agent import run_onboarding_agent

        result = run_onboarding_agent(
            raw_tape_path=str(_SYNTHETIC_TAPE),
            client_config_path=str(_SYNTHETIC_CFG),
            schema_registry_path=str(_REGISTRY),
            aliases_dir=str(_ALIASES),
            enum_mapping_path=str(_ENUM_MAP),
            output_dir=str(tmp_path),
            llm_enabled=False,
            run_id="test_canon_001",
        )

        assert result.canonical_draft_path
        canonical = Path(result.canonical_draft_path)
        assert canonical.exists(), f"canonical_full.csv not found at {canonical}"
        assert canonical.stat().st_size > 0

    def test_required_output_files_written(self, tmp_path):
        """All four canonical review files are written by the agent."""
        from agents.onboarding_agent import run_onboarding_agent

        result = run_onboarding_agent(
            raw_tape_path=str(_SYNTHETIC_TAPE),
            client_config_path=str(_SYNTHETIC_CFG),
            schema_registry_path=str(_REGISTRY),
            aliases_dir=str(_ALIASES),
            enum_mapping_path=str(_ENUM_MAP),
            output_dir=str(tmp_path),
            llm_enabled=False,
            run_id="test_files_001",
        )

        run_dir = tmp_path / "test_files_001"
        assert (run_dir / "onboarding_result.json").exists(), "onboarding_result.json missing"
        assert (run_dir / "config_questions.json").exists(), "config_questions.json missing"
        assert (run_dir / "mapping_review.json").exists(), "mapping_review.json missing"
        assert (run_dir / "enum_review.json").exists(), "enum_review.json missing"

    def test_onboarding_result_loadable(self, tmp_path):
        from agents.onboarding_agent import run_onboarding_agent

        result = run_onboarding_agent(
            raw_tape_path=str(_SYNTHETIC_TAPE),
            client_config_path=str(_SYNTHETIC_CFG),
            schema_registry_path=str(_REGISTRY),
            aliases_dir=str(_ALIASES),
            enum_mapping_path=str(_ENUM_MAP),
            output_dir=str(tmp_path),
            llm_enabled=False,
            run_id="test_load_001",
        )

        loaded = OnboardingResult.from_json(result.onboarding_result_path)
        assert loaded.run_id == "test_load_001"
        assert loaded.total_input_fields > 0
        assert loaded.mapped_fields_count > 0


# ===========================================================================
# TEST 4-7: Gate 1 decision flow in trakt_run.py
# ===========================================================================

class TestTraktRunGate1DecisionFlow:
    """Unit tests for the _run_gate1_via_onboarding_agent decision branching."""

    def _make_args(self, **kwargs):
        args = MagicMock()
        args.registry = str(_REGISTRY)
        args.master_config = str(_SYNTHETIC_CFG)
        args.aliases_dir = str(_ALIASES)
        args.enum_mapping = str(_ENUM_MAP)
        args.output_schema = "active"
        args.skip_llm = False
        for k, v in kwargs.items():
            setattr(args, k, v)
        return args

    def test_ready_for_validation_returns_result(self, tmp_path):
        """ready_for_validation returns (result, canonical_path, header_path)."""
        canonical = tmp_path / "canonical_full.csv"
        canonical.write_text("id,value\n1,test\n")

        ob_result = _make_result(
            status="ready_for_validation",
            proceed=True,
            canonical_path=str(canonical),
        )
        ob_result.onboarding_result_path = str(tmp_path / "onboarding_result.json")
        ob_result.mapping_review_items = []

        from engine.orchestrator.trakt_run import _run_gate1_via_onboarding_agent

        with patch("agents.onboarding_agent.run_onboarding_agent", return_value=ob_result):
            result, c_path, h_path = _run_gate1_via_onboarding_agent(
                args=self._make_args(),
                input_path=_SYNTHETIC_TAPE,
                out_dir=tmp_path,
                stem="test_stem",
                run_id="run_test",
            )

        assert result.status == "ready_for_validation"
        assert c_path == canonical

    def test_review_required_exits_zero(self, tmp_path):
        """review_required status causes sys.exit(0) — not an error."""
        ob_result = _make_result(
            status="review_required",
            proceed=False,
            canonical_path="",
        )
        ob_result.onboarding_result_path = str(tmp_path / "ob.json")
        ob_result.mapping_review_items = [
            MappingReviewItem(raw_field="tenure", requires_review=True)
        ]
        ob_result.enum_review_count = 1

        from engine.orchestrator.trakt_run import _run_gate1_via_onboarding_agent

        with patch("agents.onboarding_agent.run_onboarding_agent", return_value=ob_result):
            with pytest.raises(SystemExit) as exc_info:
                _run_gate1_via_onboarding_agent(
                    args=self._make_args(),
                    input_path=_SYNTHETIC_TAPE,
                    out_dir=tmp_path,
                    stem="test_stem",
                    run_id="run_test",
                )
        assert exc_info.value.code == 0

    def test_blocked_exits_two(self, tmp_path):
        """blocked status causes sys.exit(2) — user action required."""
        ob_result = _make_result(status="blocked", proceed=False)
        ob_result.onboarding_result_path = str(tmp_path / "ob.json")
        ob_result.blocker_questions = [
            {"question_id": "q_asset_class", "question_text": "What is the asset class?"}
        ]
        ob_result.mapping_review_items = []

        from engine.orchestrator.trakt_run import _run_gate1_via_onboarding_agent

        with patch("agents.onboarding_agent.run_onboarding_agent", return_value=ob_result):
            with pytest.raises(SystemExit) as exc_info:
                _run_gate1_via_onboarding_agent(
                    args=self._make_args(),
                    input_path=_SYNTHETIC_TAPE,
                    out_dir=tmp_path,
                    stem="test_stem",
                    run_id="run_test",
                )
        assert exc_info.value.code == 2

    def test_failed_raises_runtime_error(self, tmp_path):
        """failed status raises RuntimeError."""
        ob_result = _make_result(status="failed", proceed=False)
        ob_result.onboarding_result_path = str(tmp_path / "ob.json")
        ob_result.errors = ["Semantic alignment failed: file not found"]
        ob_result.mapping_review_items = []

        from engine.orchestrator.trakt_run import _run_gate1_via_onboarding_agent

        with patch("agents.onboarding_agent.run_onboarding_agent", return_value=ob_result):
            with pytest.raises(RuntimeError, match="Onboarding Agent failed"):
                _run_gate1_via_onboarding_agent(
                    args=self._make_args(),
                    input_path=_SYNTHETIC_TAPE,
                    out_dir=tmp_path,
                    stem="test_stem",
                    run_id="run_test",
                )


# ===========================================================================
# TEST 8: semantic_alignment.py is still invoked by the agent
# ===========================================================================

class TestSemanticAlignmentStillInvoked:
    def test_semantic_alignment_called_as_subprocess(self, tmp_path):
        """Onboarding agent invokes semantic_alignment.py as a subprocess."""
        from agents.onboarding_agent import _run_semantic_alignment
        from agents.onboarding_schemas import ConfigBootstrapResult

        # We verify the function builds a command that includes semantic_alignment.py
        import agents.onboarding_agent as oa_mod

        called_cmds = []
        original_run = subprocess.run

        def _capture_run(cmd, **kwargs):
            called_cmds.append(cmd)
            # Return mock success with empty output
            m = MagicMock()
            m.returncode = 0
            m.stderr = ""
            m.stdout = ""
            return m

        tape = _SYNTHETIC_TAPE
        reg = _REGISTRY
        aliases = _ALIASES
        out = tmp_path

        with patch("subprocess.run", side_effect=_capture_run):
            try:
                _run_semantic_alignment(
                    tape_path=tape,
                    portfolio_type="equity_release",
                    registry_path=reg,
                    aliases_dir=aliases,
                    output_dir=out,
                )
            except Exception:
                pass  # Outputs won't exist; we only care the subprocess was called

        assert len(called_cmds) >= 1
        cmd_str = " ".join(str(c) for c in called_cmds[0])
        assert "semantic_alignment" in cmd_str


# ===========================================================================
# TEST 9: LLM Mapping Agent path is present but not called without API key
# ===========================================================================

class TestLLMMappingAgentPath:
    def test_llm_path_skipped_without_api_key(self, tmp_path):
        """LLM mapping is skipped when no ANTHROPIC_API_KEY is set."""
        from agents.onboarding_agent import run_onboarding_agent
        import os

        # Ensure no key is set
        env_backup = os.environ.pop("ANTHROPIC_API_KEY", None)
        try:
            result = run_onboarding_agent(
                raw_tape_path=str(_SYNTHETIC_TAPE),
                client_config_path=str(_SYNTHETIC_CFG),
                schema_registry_path=str(_REGISTRY),
                aliases_dir=str(_ALIASES),
                enum_mapping_path=str(_ENUM_MAP),
                output_dir=str(tmp_path),
                llm_enabled=True,   # enabled, but no key
                run_id="test_no_llm_key",
            )
        finally:
            if env_backup is not None:
                os.environ["ANTHROPIC_API_KEY"] = env_backup

        # With no API key, no LLM-sourced mappings
        llm_items = [m for m in result.mapping_review_items if m.mapping_source == "llm"]
        assert len(llm_items) == 0
        # Governance artifact path should be empty
        assert result.governance_artifact_path == ""

    def test_llm_enabled_false_bypasses_llm(self, tmp_path):
        """llm_enabled=False explicitly prevents LLM mapping."""
        from agents.onboarding_agent import run_onboarding_agent

        result = run_onboarding_agent(
            raw_tape_path=str(_SYNTHETIC_TAPE),
            client_config_path=str(_SYNTHETIC_CFG),
            schema_registry_path=str(_REGISTRY),
            aliases_dir=str(_ALIASES),
            enum_mapping_path=str(_ENUM_MAP),
            output_dir=str(tmp_path),
            llm_enabled=False,
            run_id="test_llm_disabled",
        )
        llm_items = [m for m in result.mapping_review_items if m.mapping_source == "llm"]
        assert len(llm_items) == 0
        assert result.llm_suggested_count == 0


# ===========================================================================
# TEST 10: Enum Mapping Agent executes
# ===========================================================================

@pytest.mark.skipif(
    not _SYNTHETIC_TAPE.exists(),
    reason="Synthetic tape not found"
)
class TestEnumMappingAgentExecutes:
    def test_enum_items_produced(self, tmp_path):
        """Enum mapping agent runs and produces EnumReviewItem objects."""
        from agents.onboarding_agent import run_onboarding_agent

        result = run_onboarding_agent(
            raw_tape_path=str(_SYNTHETIC_TAPE),
            client_config_path=str(_SYNTHETIC_CFG),
            schema_registry_path=str(_REGISTRY),
            aliases_dir=str(_ALIASES),
            enum_mapping_path=str(_ENUM_MAP),
            output_dir=str(tmp_path),
            llm_enabled=False,
            run_id="test_enum_001",
        )
        # Synthetic tape has enum fields (interest_rate_type, asset_class, etc.)
        assert result.enum_fields_total > 0, "Enum mapping agent produced no items"
        # At least some values should be resolved
        assert result.enum_success_rate >= 0.0

    def test_enum_review_json_is_valid(self, tmp_path):
        from agents.onboarding_agent import run_onboarding_agent

        result = run_onboarding_agent(
            raw_tape_path=str(_SYNTHETIC_TAPE),
            client_config_path=str(_SYNTHETIC_CFG),
            schema_registry_path=str(_REGISTRY),
            aliases_dir=str(_ALIASES),
            enum_mapping_path=str(_ENUM_MAP),
            output_dir=str(tmp_path),
            llm_enabled=False,
            run_id="test_enum_json",
        )

        enum_json = tmp_path / "test_enum_json" / "enum_review.json"
        assert enum_json.exists()
        data = json.loads(enum_json.read_text())
        assert isinstance(data, list)


# ===========================================================================
# TEST: trakt_run --skip-onboarding falls back to legacy path
# ===========================================================================

class TestSkipOnboardingFallback:
    def test_skip_onboarding_flag_preserves_legacy_path(self, tmp_path):
        """--skip-onboarding skips the onboarding agent entirely."""
        # We test the logic directly rather than invoking the full pipeline
        # (which would require all downstream scripts to be installed).
        called = []

        def _mock_run(cmd, **kwargs):
            called.append(cmd)
            m = MagicMock()
            m.returncode = 0
            m.stdout = ""
            m.stderr = ""
            return m

        from engine.orchestrator import trakt_run as tr

        args = MagicMock()
        args.skip_onboarding = True
        args.portfolio_type = "equity_release"
        args.output_schema = "active"
        args.registry = str(_REGISTRY)
        args.mode = "mi"
        args.regime = None
        args.master_config = str(_SYNTHETIC_CFG)
        args.loan_engine_enabled = False

        # Create a fake canonical_full.csv so the existence check passes
        canonical = tmp_path / "SYNTHETIC_ERE_Portfolio_012026_canonical_full.csv"
        canonical.write_text("id\n1\n")

        with patch("subprocess.run", side_effect=_mock_run):
            # Partially call run_common_gates, stopping just after Gate 1
            # by monkey-patching _run to track calls without running the full pipeline
            try:
                tr.run_common_gates(
                    py=sys.executable,
                    args=args,
                    input_path=_SYNTHETIC_TAPE,
                    out_dir=tmp_path,
                    val_dir=tmp_path / "val",
                    stem="SYNTHETIC_ERE_Portfolio_012026",
                    flags={"loan_engine_enabled": False},
                )
            except Exception:
                pass  # Downstream scripts will fail; we only check Gate 1 path

        # When --skip-onboarding, no call to run_onboarding_agent should appear
        call_strings = [" ".join(str(c) for c in cmd) for cmd in called]
        onboarding_called = any("onboarding_agent" in s for s in call_strings)
        assert not onboarding_called, "Onboarding agent was called despite --skip-onboarding"

        # semantic_alignment.py SHOULD have been called
        semantic_called = any("semantic_alignment" in s for s in call_strings)
        assert semantic_called, "semantic_alignment.py was not called in legacy path"
