"""
tests/test_review_workbench.py

Tests for the Review & Approval Workbench:
  - agents/review_schemas.py
  - agents/learning_persistence.py
  - cli/onboarding_review_cli.py (key functions only; no live I/O)

No live Claude API calls.  No Streamlit rendering.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Path bootstrapping
# ---------------------------------------------------------------------------

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from agents.review_schemas import (
    EnumDecision,
    MappingDecision,
    QuestionAnswer,
    ReviewSubmission,
    build_enum_overrides_json,
    build_mapping_overrides_json,
    build_questionnaire_answers_json,
)
from agents.learning_persistence import (
    _normalise,
    persist_enum_decisions,
    persist_mapping_decisions,
)
from agents.onboarding_schemas import (
    ConfigBootstrapResult,
    EnumReviewItem,
    MappingReviewItem,
    OnboardingResult,
)


# ===========================================================================
# Review schemas
# ===========================================================================

class TestQuestionAnswer:
    def test_defaults(self):
        qa = QuestionAnswer()
        assert qa.question_id == ""
        assert qa.approved is False

    def test_to_dict_round_trip(self):
        qa = QuestionAnswer(question_id="q_asset_class", answer="RMBS", approved=True)
        d = qa.to_dict()
        assert d["question_id"] == "q_asset_class"
        assert d["answer"] == "RMBS"
        assert d["approved"] is True

    def test_approved_false_by_default(self):
        qa = QuestionAnswer(question_id="q_x", answer="val")
        assert qa.approved is False


class TestMappingDecision:
    def test_approve_with_canonical(self):
        d = MappingDecision(raw_field="LN_ID", approved=True, selected_canonical_field="loan_id")
        assert d.approved is True
        assert d.selected_canonical_field == "loan_id"

    def test_skip_decision(self):
        d = MappingDecision(raw_field="UNKNOWN_COL", approved=False)
        assert d.selected_canonical_field is None

    def test_to_dict(self):
        d = MappingDecision(raw_field="x", approved=True, selected_canonical_field="y", comments="ok")
        raw = d.to_dict()
        assert raw["raw_field"] == "x"
        assert raw["selected_canonical_field"] == "y"


class TestEnumDecision:
    def test_approve(self):
        d = EnumDecision(field_name="asset_class", raw_value="HomeLoan", approved=True, selected_value="RMBS")
        assert d.approved is True
        assert d.selected_value == "RMBS"

    def test_skip(self):
        d = EnumDecision(field_name="asset_class", raw_value="unknown", approved=False)
        assert d.selected_value is None


class TestReviewSubmission:
    def _make_submission(self):
        return ReviewSubmission(
            run_id="run_test_001",
            question_answers=[
                QuestionAnswer(question_id="q_asset_class", answer="RMBS", approved=True),
                QuestionAnswer(question_id="q_regime", answer="", approved=False),
            ],
            mapping_decisions=[
                MappingDecision(raw_field="LN_ID", approved=True, selected_canonical_field="loan_id"),
                MappingDecision(raw_field="JUNK", approved=False),
            ],
            enum_decisions=[
                EnumDecision(field_name="asset_class", raw_value="HomeLoan", approved=True, selected_value="RMBS"),
                EnumDecision(field_name="asset_class", raw_value="???", approved=False),
            ],
        )

    def test_approved_counts(self):
        s = self._make_submission()
        assert s.approved_question_count == 1
        assert s.approved_mapping_count == 1
        assert s.approved_enum_count == 1

    def test_to_dict_has_all_keys(self):
        s = self._make_submission()
        d = s.to_dict()
        assert "run_id" in d
        assert "question_answers" in d
        assert "mapping_decisions" in d
        assert "enum_decisions" in d
        assert "submitted_at" in d

    def test_to_json_round_trip(self, tmp_path):
        s = self._make_submission()
        path = tmp_path / "submission.json"
        s.to_json(path)
        loaded = ReviewSubmission.from_json(path)
        assert loaded.run_id == s.run_id
        assert loaded.approved_question_count == 1
        assert loaded.approved_mapping_count == 1
        assert loaded.approved_enum_count == 1
        # Nested types reconstructed
        assert isinstance(loaded.question_answers[0], QuestionAnswer)
        assert isinstance(loaded.mapping_decisions[0], MappingDecision)
        assert isinstance(loaded.enum_decisions[0], EnumDecision)

    def test_submitted_at_auto_populated(self):
        s = ReviewSubmission(run_id="x")
        assert s.submitted_at != ""


class TestBuilders:
    def test_build_questionnaire_answers_json(self):
        answers = [
            QuestionAnswer(question_id="q_a", answer="RMBS", approved=True),
            QuestionAnswer(question_id="q_b", answer="", approved=False),
        ]
        result = build_questionnaire_answers_json(answers)
        assert len(result) == 1
        assert result[0]["question_id"] == "q_a"
        assert result[0]["answer"] == "RMBS"

    def test_build_mapping_overrides_json(self):
        decisions = [
            MappingDecision(raw_field="LN_ID", approved=True, selected_canonical_field="loan_id"),
            MappingDecision(raw_field="JUNK", approved=False),
        ]
        result = build_mapping_overrides_json(decisions)
        assert len(result) == 1
        assert result[0]["raw_field"] == "LN_ID"
        assert result[0]["canonical_field"] == "loan_id"
        assert result[0]["action"] == "confirmed"

    def test_build_enum_overrides_json(self):
        decisions = [
            EnumDecision(field_name="asset_class", raw_value="HomeLoan", approved=True, selected_value="RMBS"),
            EnumDecision(field_name="asset_class", raw_value="???", approved=False),
        ]
        result = build_enum_overrides_json(decisions)
        assert len(result) == 1
        assert result[0]["field_name"] == "asset_class"
        assert result[0]["raw_value"] == "HomeLoan"
        assert result[0]["canonical_value"] == "RMBS"


# ===========================================================================
# Learning persistence
# ===========================================================================

class TestPersistMappingDecisions:
    def _make_decisions(self, raw: str, canonical: str) -> List[MappingDecision]:
        return [MappingDecision(raw_field=raw, approved=True, selected_canonical_field=canonical)]

    def test_writes_alias_file(self, tmp_path):
        decisions = self._make_decisions("LN_ID", "loan_id")
        n = persist_mapping_decisions(decisions, tmp_path, session_id="test_session")
        assert n == 1
        alias_file = tmp_path / "aliases_llm_confirmed.yaml"
        assert alias_file.exists()
        content = alias_file.read_text()
        assert "loan_id" in content
        assert "LN_ID" in content

    def test_idempotent_no_duplicate(self, tmp_path):
        decisions = self._make_decisions("LN_ID", "loan_id")
        n1 = persist_mapping_decisions(decisions, tmp_path)
        n2 = persist_mapping_decisions(decisions, tmp_path)
        assert n1 == 1
        assert n2 == 0   # already exists

    def test_skips_unapproved(self, tmp_path):
        decisions = [MappingDecision(raw_field="x", approved=False, selected_canonical_field="y")]
        n = persist_mapping_decisions(decisions, tmp_path)
        assert n == 0
        alias_file = tmp_path / "aliases_llm_confirmed.yaml"
        assert not alias_file.exists()

    def test_skips_no_canonical(self, tmp_path):
        decisions = [MappingDecision(raw_field="x", approved=True, selected_canonical_field=None)]
        n = persist_mapping_decisions(decisions, tmp_path)
        assert n == 0

    def test_creates_dir_if_missing(self, tmp_path):
        new_dir = tmp_path / "aliases" / "nested"
        decisions = self._make_decisions("RATE", "interest_rate")
        n = persist_mapping_decisions(decisions, new_dir)
        assert n == 1
        assert (new_dir / "aliases_llm_confirmed.yaml").exists()

    def test_normalise_dedup(self, tmp_path):
        # "LN ID" and "ln id" should not both be written
        d1 = [MappingDecision(raw_field="LN ID", approved=True, selected_canonical_field="loan_id")]
        d2 = [MappingDecision(raw_field="ln id", approved=True, selected_canonical_field="loan_id")]
        n1 = persist_mapping_decisions(d1, tmp_path)
        n2 = persist_mapping_decisions(d2, tmp_path)
        assert n1 == 1
        assert n2 == 0


class TestPersistEnumDecisions:
    def _make_decisions(self) -> List[EnumDecision]:
        return [
            EnumDecision(field_name="asset_class", raw_value="HomeLoan", approved=True, selected_value="RMBS"),
            EnumDecision(field_name="asset_class", raw_value="Auto", approved=True, selected_value="ABS"),
        ]

    def test_writes_enum_file(self, tmp_path):
        out = tmp_path / "enum_synonyms_confirmed.yaml"
        n = persist_enum_decisions(self._make_decisions(), out, session_id="s1")
        assert n == 2
        assert out.exists()
        content = out.read_text()
        assert "HomeLoan" in content
        assert "RMBS" in content

    def test_idempotent(self, tmp_path):
        out = tmp_path / "enum_synonyms_confirmed.yaml"
        n1 = persist_enum_decisions(self._make_decisions(), out)
        n2 = persist_enum_decisions(self._make_decisions(), out)
        assert n1 == 2
        assert n2 == 0

    def test_skips_unapproved(self, tmp_path):
        out = tmp_path / "enum_synonyms_confirmed.yaml"
        decisions = [EnumDecision(field_name="x", raw_value="y", approved=False)]
        n = persist_enum_decisions(decisions, out)
        assert n == 0

    def test_namespace_in_yaml(self, tmp_path):
        out = tmp_path / "enum_synonyms_confirmed.yaml"
        d = [EnumDecision(field_name="f", raw_value="r", approved=True, selected_value="c")]
        persist_enum_decisions(d, out, namespace="esma_rmbs")
        import yaml
        data = yaml.safe_load(out.read_text())
        assert "esma_rmbs" in data
        assert "f" in data["esma_rmbs"]


class TestNormalise:
    def test_strips_whitespace(self):
        assert _normalise("  hello  ") == "hello"

    def test_lowercases(self):
        assert _normalise("LN_ID") == "ln_id"

    def test_collapses_spaces(self):
        assert _normalise("loan   id") == "loan id"


# ===========================================================================
# CLI — unit tests (no I/O)
# ===========================================================================

class TestCliHelpers:
    """Test CLI helper functions without any stdin/stdout interaction."""

    def test_write_decision_files(self, tmp_path):
        from cli.onboarding_review_cli import _write_decision_files

        answers = [QuestionAnswer(question_id="q_a", answer="RMBS", approved=True)]
        mappings = [MappingDecision(raw_field="LN_ID", approved=True, selected_canonical_field="loan_id")]
        enums = [EnumDecision(field_name="asset_class", raw_value="HomeLoan", approved=True, selected_value="RMBS")]

        submission = _write_decision_files(tmp_path, "run_001", answers, mappings, enums)

        assert (tmp_path / "questionnaire_answers.json").exists()
        assert (tmp_path / "mapping_overrides.json").exists()
        assert (tmp_path / "enum_overrides.json").exists()
        assert (tmp_path / "run_001_review_submission.json").exists()

        qa_data = json.loads((tmp_path / "questionnaire_answers.json").read_text())
        assert len(qa_data) == 1
        assert qa_data[0]["question_id"] == "q_a"

        mo_data = json.loads((tmp_path / "mapping_overrides.json").read_text())
        assert len(mo_data) == 1
        assert mo_data[0]["canonical_field"] == "loan_id"

        eo_data = json.loads((tmp_path / "enum_overrides.json").read_text())
        assert len(eo_data) == 1
        assert eo_data[0]["canonical_value"] == "RMBS"

        assert isinstance(submission, ReviewSubmission)
        assert submission.run_id == "run_001"

    def test_write_decision_files_empty(self, tmp_path):
        from cli.onboarding_review_cli import _write_decision_files

        submission = _write_decision_files(tmp_path, "run_empty", [], [], [])
        assert submission.approved_question_count == 0
        assert (tmp_path / "questionnaire_answers.json").exists()

    def test_run_persistence_no_crash_on_missing_paths(self, tmp_path):
        """persist functions should not raise when paths are empty strings."""
        from cli.onboarding_review_cli import _run_persistence

        result = OnboardingResult(run_id="r1", aliases_dir="", enum_mapping_path="")
        submission = ReviewSubmission(run_id="r1")
        # Should complete without raising
        _run_persistence(submission, result, tmp_path, session_id="s1")


class TestCliMainFlow:
    """Integration-style tests for main() using mocked I/O."""

    def _make_result(self, tmp_path: Path) -> Path:
        result = OnboardingResult(
            run_id="run_test_cli_001",
            status="review_required",
            total_input_fields=10,
            mapped_fields_count=8,
            review_fields_count=1,
            unmapped_mandatory_count=0,
            enum_success_rate=0.9,
            proceed_to_validation=False,
            narrative_summary="Test narrative.",
            blocker_questions=[
                {
                    "question_id": "q_asset_class",
                    "question_text": "What is the asset class?",
                    "suggested_answer": "RMBS",
                    "blocking": True,
                }
            ],
            mapping_review_items=[
                MappingReviewItem(
                    raw_field="LN_ID",
                    suggested_canonical_field="loan_id",
                    mapping_source="llm",
                    confidence=0.80,
                    requires_review=True,
                    blocker=False,
                )
            ],
            enum_review_items=[
                EnumReviewItem(
                    field_name="asset_class",
                    raw_value="HomeLoan",
                    suggested_value="RMBS",
                    mapping_source="llm",
                    confidence=0.78,
                    requires_review=True,
                    blocker=False,
                    sample_count=5,
                )
            ],
        )
        result_path = tmp_path / "run_test_cli_001_onboarding_result.json"
        result.to_json(result_path)
        return tmp_path

    def test_main_missing_run_dir(self, tmp_path):
        from cli.onboarding_review_cli import main

        rc = main(["--run-dir", str(tmp_path / "nonexistent")])
        assert rc == 1

    def test_main_no_result_file(self, tmp_path):
        from cli.onboarding_review_cli import main

        rc = main(["--run-dir", str(tmp_path)])
        assert rc == 1

    def test_main_auto_approve_all_no_rerun(self, tmp_path):
        """Simulate user approving all suggestions, skipping re-run."""
        run_dir = self._make_result(tmp_path)
        from cli.onboarding_review_cli import main

        # Inputs: approve blocker Q, approve mapping, approve enum, confirm Y
        inputs = ["A", "A", "A", "Y"]
        with patch("builtins.input", side_effect=inputs):
            rc = main(["--run-dir", str(run_dir), "--no-rerun"])

        # With proceed_to_validation=False on original result, exit code should be 2
        assert rc in (0, 2)
        # Decision files must be written
        assert (run_dir / "questionnaire_answers.json").exists()
        assert (run_dir / "mapping_overrides.json").exists()
        assert (run_dir / "enum_overrides.json").exists()

    def test_main_user_aborts(self, tmp_path):
        """Simulate user choosing N at submission prompt."""
        run_dir = self._make_result(tmp_path)
        from cli.onboarding_review_cli import main

        inputs = ["A", "A", "A", "N"]
        with patch("builtins.input", side_effect=inputs):
            rc = main(["--run-dir", str(run_dir), "--no-rerun"])

        assert rc == 0
        # No decision files written
        assert not (run_dir / "questionnaire_answers.json").exists()

    def test_main_with_rerun_succeeds(self, tmp_path):
        """Simulate a successful re-run that returns proceed_to_validation=True."""
        run_dir = self._make_result(tmp_path)
        from cli.onboarding_review_cli import main

        new_result = OnboardingResult(
            run_id="run_test_cli_002",
            status="ready_for_validation",
            proceed_to_validation=True,
            mapped_fields_count=10,
            total_input_fields=10,
            enum_success_rate=1.0,
        )
        new_result.onboarding_result_path = str(run_dir / "new_result.json")

        inputs = ["A", "A", "A", "Y"]
        with patch("builtins.input", side_effect=inputs):
            with patch("cli.onboarding_review_cli._rerun_agent", return_value=new_result):
                rc = main(["--run-dir", str(run_dir)])

        assert rc == 0  # proceed_to_validation is True → exit 0

    def test_main_skip_all(self, tmp_path):
        """Simulate user skipping all review items."""
        run_dir = self._make_result(tmp_path)
        from cli.onboarding_review_cli import main

        # S for all items, then Y to submit
        inputs = ["S", "S", "S", "Y"]
        with patch("builtins.input", side_effect=inputs):
            rc = main(["--run-dir", str(run_dir), "--no-rerun"])

        assert rc in (0, 2)
        qa = json.loads((run_dir / "questionnaire_answers.json").read_text())
        assert qa == []   # all skipped → nothing approved

    def test_main_already_approved_no_items(self, tmp_path):
        """If result has no review items and is already approved, exit 0."""
        result = OnboardingResult(
            run_id="run_clean",
            status="ready_for_validation",
            proceed_to_validation=True,
        )
        result.to_json(tmp_path / "run_clean_onboarding_result.json")
        from cli.onboarding_review_cli import main

        rc = main(["--run-dir", str(tmp_path), "--no-rerun"])
        assert rc == 0


# ===========================================================================
# OnboardingResult serialisation (regression — re-run metadata fields)
# ===========================================================================

class TestOnboardingResultRerunFields:
    def test_rerun_fields_persist(self, tmp_path):
        r = OnboardingResult(
            run_id="r1",
            raw_tape_path="/data/tape.csv",
            schema_registry_path="/schemas/registry.yaml",
            aliases_dir="/aliases",
            enum_mapping_path="/enums/mapping.yaml",
        )
        path = tmp_path / "result.json"
        r.to_json(path)
        loaded = OnboardingResult.from_json(path)
        assert loaded.raw_tape_path == "/data/tape.csv"
        assert loaded.schema_registry_path == "/schemas/registry.yaml"
        assert loaded.aliases_dir == "/aliases"
        assert loaded.enum_mapping_path == "/enums/mapping.yaml"
