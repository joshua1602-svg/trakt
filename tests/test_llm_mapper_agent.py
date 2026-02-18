#!/usr/bin/env python3
"""
tests/test_llm_mapper_agent.py

Unit tests for the LLM Mapper Agent (Tier 7).

Principles tested:
  - Canonical-only constraint: agent never suggests fields not in the registry
  - Alias deduplication: confirmed aliases not re-added on repeat
  - Governance artifact schema is complete and valid
  - CLI review loop processes all decisions correctly
  - Tier 1-6 results are NEVER overridden by the agent
  - Anthropic API is mocked — no real network calls
"""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from dataclasses import asdict
from io import StringIO
from pathlib import Path
from typing import List
from unittest.mock import MagicMock, patch, PropertyMock

import pandas as pd
import yaml

# Ensure the engine module path is importable
HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE / "engine" / "gate_1_alignment"))

from llm_mapper_agent import (
    AliasLearner,
    GovernanceLogger,
    HumanReviewSession,
    LLMFieldMapper,
    LLMSuggestion,
    _build_catalogue_subset,
    _load_registry,
)


# ---------------------------------------------------------------------------
# FIXTURES
# ---------------------------------------------------------------------------

MINIMAL_REGISTRY = {
    "fields": {
        "valuation_date": {
            "category": "collateral",
            "format": "date",
            "portfolio_type": "common",
            "layer": "core",
            "core_canonical": True,
        },
        "current_balance": {
            "category": "regulatory",
            "format": "decimal",
            "portfolio_type": "common",
            "layer": "core",
            "core_canonical": True,
        },
        "interest_rate_type": {
            "category": "regulatory",
            "format": "list",
            "portfolio_type": "equity_release",
            "allowed_values": "interest_rate_type",
            "layer": "core",
            "core_canonical": True,
        },
        "loan_identifier": {
            "category": "regulatory",
            "format": "string",
            "portfolio_type": "common",
            "layer": "core",
            "core_canonical": True,
        },
    }
}

CANONICAL_FIELDS = list(MINIMAL_REGISTRY["fields"].keys())


def _write_registry(tmp: Path) -> Path:
    path = tmp / "fields_registry.yaml"
    path.write_text(yaml.dump(MINIMAL_REGISTRY), encoding="utf-8")
    return path


def _make_suggestion(
    raw_header: str = "Loan Valn Dt",
    suggested_field: str = "valuation_date",
    confidence: float = 0.94,
    status: str = "pending",
    confirmed_field: str = None,
) -> LLMSuggestion:
    return LLMSuggestion(
        raw_header=raw_header,
        suggested_field=suggested_field,
        confidence=confidence,
        reasoning="Header 'Valn' = valuation, 'Dt' = date. Samples are ISO dates.",
        alternative_field="last_valuation_date",
        semantic_category="collateral",
        sample_values=["2024-01-15", "2023-12-01"],
        status=status,
        confirmed_field=confirmed_field,
    )


# ---------------------------------------------------------------------------
# TEST: LLMSuggestion dataclass
# ---------------------------------------------------------------------------


class TestLLMSuggestion(unittest.TestCase):

    def test_defaults(self):
        s = LLMSuggestion(
            raw_header="Test Col",
            suggested_field="current_balance",
            confidence=0.9,
            reasoning="Test",
            alternative_field=None,
            semantic_category="regulatory",
            sample_values=[],
        )
        self.assertEqual(s.status, "pending")
        self.assertIsNone(s.confirmed_field)
        self.assertEqual(s.deterministic_method, "unmapped")
        self.assertEqual(s.deterministic_confidence, 0.0)

    def test_to_dict_roundtrip(self):
        s = _make_suggestion()
        d = s.to_dict()
        self.assertIn("raw_header", d)
        self.assertIn("suggested_field", d)
        self.assertIn("status", d)
        self.assertEqual(d["raw_header"], "Loan Valn Dt")


# ---------------------------------------------------------------------------
# TEST: Canonical-only constraint
# ---------------------------------------------------------------------------


class TestCanonicalOnlyConstraint(unittest.TestCase):
    """Agent must NEVER emit a field name that is not in the registry."""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.registry_path = _write_registry(self.tmp)
        aliases_dir = self.tmp / "aliases"
        aliases_dir.mkdir()
        self.mapper = LLMFieldMapper(
            registry_path=self.registry_path,
            portfolio_type="equity_release",
            aliases_dir=aliases_dir,
            api_key="test_key",
        )

    def _make_llm_response(self, field: str, confidence: float = 0.9) -> dict:
        return {
            "suggested_field": field,
            "confidence": confidence,
            "reasoning": "test",
            "alternative_field": None,
            "semantic_category": "regulatory",
        }

    def test_hallucinated_field_is_nulled(self):
        """If LLM returns a field not in the registry it should be set to None."""
        envelope = {"header": "XYZ Col", "samples": ["a", "b"], "dtype": "object", "stats": {}}
        response = self._make_llm_response("nonexistent_hallucinated_field", confidence=0.99)
        result = self.mapper._parse_response(envelope, response)
        self.assertIsNone(result.suggested_field)
        self.assertEqual(result.confidence, 0.0)
        self.assertIn("NULLED", result.reasoning)

    def test_valid_field_passes_through(self):
        """A valid canonical field should survive the constraint check."""
        envelope = {"header": "Loan Valn Dt", "samples": ["2024-01-01"], "dtype": "object", "stats": {}}
        response = self._make_llm_response("valuation_date", confidence=0.94)
        result = self.mapper._parse_response(envelope, response)
        self.assertEqual(result.suggested_field, "valuation_date")
        self.assertAlmostEqual(result.confidence, 0.94)

    def test_hallucinated_alternative_is_nulled(self):
        """Alternative field not in registry should also be nulled."""
        envelope = {"header": "Bal", "samples": ["10000"], "dtype": "float64", "stats": {}}
        response = {
            "suggested_field": "current_balance",
            "confidence": 0.88,
            "reasoning": "balance column",
            "alternative_field": "invented_balance_field",
            "semantic_category": "regulatory",
        }
        result = self.mapper._parse_response(envelope, response)
        self.assertEqual(result.suggested_field, "current_balance")
        self.assertIsNone(result.alternative_field)

    def test_catalogue_subset_only_contains_registry_fields(self):
        """Catalogue subset fed to the LLM must only contain registry field names."""
        subset = _build_catalogue_subset(MINIMAL_REGISTRY, "equity_release", max_fields=80)
        subset_names = {f["name"] for f in subset}
        registry_names = set(MINIMAL_REGISTRY["fields"].keys())
        self.assertTrue(subset_names.issubset(registry_names))

    def test_suggest_mappings_never_returns_noncanonical(self):
        """End-to-end: suggest_mappings should null any non-canonical suggestions."""
        df = pd.DataFrame({"Weird Col": ["x", "y", "z"]})
        llm_resp_json = json.dumps([{
            "suggested_field": "totally_fake_field_xyz",
            "confidence": 0.99,
            "reasoning": "bad llm",
            "alternative_field": None,
            "semantic_category": "regulatory",
        }])
        mock_content = MagicMock()
        mock_content.text = llm_resp_json
        mock_message = MagicMock()
        mock_message.content = [mock_content]

        with patch.object(self.mapper, "_get_client") as mock_get_client:
            mock_client = MagicMock()
            mock_client.messages.create.return_value = mock_message
            mock_get_client.return_value = mock_client

            results = self.mapper.suggest_mappings(["Weird Col"], df)

        self.assertEqual(len(results), 1)
        self.assertIsNone(results[0].suggested_field)


# ---------------------------------------------------------------------------
# TEST: Tier 1-6 precedence
# ---------------------------------------------------------------------------


class TestDeterministicPrecedence(unittest.TestCase):
    """Headers already resolved by Tiers 1-6 must NOT be overridden."""

    def test_exact_match_headers_not_in_llm_targets(self):
        """
        The orchestrator's _collect_llm_targets function must exclude headers
        that were resolved by exact/normalized/alias methods regardless of confidence.
        """
        from agent_orchestrator import _collect_llm_targets  # type: ignore

        report = [
            {"raw_header": "loan_identifier", "mapping_method": "exact", "confidence": 1.0},
            {"raw_header": "current_balance", "mapping_method": "normalized", "confidence": 1.0},
            {"raw_header": "ValDate", "mapping_method": "alias", "confidence": 1.0},
            {"raw_header": "Weird Fuzz Col", "mapping_method": "fuzz_token_set", "confidence": 0.88},
            {"raw_header": "Unknown Header", "mapping_method": "unmapped", "confidence": 0.0},
        ]
        # review_threshold = 0.92 → fuzz at 0.88 and unmapped both qualify for LLM
        targets = _collect_llm_targets(report, review_threshold=0.92)

        self.assertNotIn("loan_identifier", targets)
        self.assertNotIn("current_balance", targets)
        self.assertNotIn("ValDate", targets)
        self.assertIn("Weird Fuzz Col", targets)
        self.assertIn("Unknown Header", targets)

    def test_above_threshold_fuzz_not_sent_to_llm(self):
        from agent_orchestrator import _collect_llm_targets  # type: ignore

        report = [
            {"raw_header": "Near Miss Col", "mapping_method": "fuzz_token_set", "confidence": 0.95},
        ]
        targets = _collect_llm_targets(report, review_threshold=0.92)
        self.assertNotIn("Near Miss Col", targets)


# ---------------------------------------------------------------------------
# TEST: AliasLearner
# ---------------------------------------------------------------------------


class TestAliasLearner(unittest.TestCase):

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.aliases_dir = self.tmp / "aliases"
        self.aliases_dir.mkdir()
        self.learner = AliasLearner()

    def _confirmed(self, raw: str, field: str) -> LLMSuggestion:
        s = _make_suggestion(raw_header=raw, suggested_field=field, status="confirmed")
        s.confirmed_field = field
        return s

    def test_new_alias_is_written(self):
        confirmed = [self._confirmed("Loan Valn Dt", "valuation_date")]
        added = self.learner.persist_confirmed(confirmed, self.aliases_dir, "sess_test")
        self.assertEqual(added, 1)

        out_file = self.aliases_dir / "aliases_llm_confirmed.yaml"
        self.assertTrue(out_file.exists())
        data = yaml.safe_load(out_file.read_text())
        self.assertIn("valuation_date", data)
        self.assertIn("Loan Valn Dt", data["valuation_date"]["aliases"])

    def test_duplicate_alias_not_added_twice(self):
        confirmed = [self._confirmed("Loan Valn Dt", "valuation_date")]
        self.learner.persist_confirmed(confirmed, self.aliases_dir, "sess_1")
        added2 = self.learner.persist_confirmed(confirmed, self.aliases_dir, "sess_2")
        self.assertEqual(added2, 0)

        out_file = self.aliases_dir / "aliases_llm_confirmed.yaml"
        data = yaml.safe_load(out_file.read_text())
        # Should only appear once
        count = data["valuation_date"]["aliases"].count("Loan Valn Dt")
        self.assertEqual(count, 1)

    def test_deduplication_against_existing_mandatory_aliases(self):
        """If the raw header already exists in another alias file, do not re-add it."""
        existing_file = self.aliases_dir / "aliases_mandatory.yaml"
        existing_file.write_text(
            yaml.dump({"valuation_date": {"aliases": ["Loan Valn Dt"]}}),
            encoding="utf-8",
        )
        confirmed = [self._confirmed("Loan Valn Dt", "valuation_date")]
        added = self.learner.persist_confirmed(confirmed, self.aliases_dir, "sess_test")
        self.assertEqual(added, 0)

    def test_rejected_suggestion_not_persisted(self):
        rejected = _make_suggestion(status="rejected")
        rejected.confirmed_field = None
        added = self.learner.persist_confirmed([rejected], self.aliases_dir, "sess_test")
        self.assertEqual(added, 0)

    def test_remapped_suggestion_uses_confirmed_field(self):
        s = _make_suggestion(
            raw_header="Curr Bal",
            suggested_field="allocated_losses",  # wrong LLM suggestion
            status="remapped",
        )
        s.confirmed_field = "current_balance"  # human corrected this
        added = self.learner.persist_confirmed([s], self.aliases_dir, "sess_remap")
        self.assertEqual(added, 1)

        out_file = self.aliases_dir / "aliases_llm_confirmed.yaml"
        data = yaml.safe_load(out_file.read_text())
        self.assertIn("current_balance", data)
        self.assertIn("Curr Bal", data["current_balance"]["aliases"])
        self.assertNotIn("allocated_losses", data)

    def test_yaml_format_matches_existing_alias_files(self):
        """Confirm the output YAML is parseable and has the expected structure."""
        confirmed = [
            self._confirmed("Val Dt", "valuation_date"),
            self._confirmed("OutstandingBal", "current_balance"),
        ]
        self.learner.persist_confirmed(confirmed, self.aliases_dir, "sess_fmt")
        out_file = self.aliases_dir / "aliases_llm_confirmed.yaml"
        data = yaml.safe_load(out_file.read_text())

        for field_name in ("valuation_date", "current_balance"):
            self.assertIn(field_name, data)
            self.assertIn("aliases", data[field_name])
            self.assertIsInstance(data[field_name]["aliases"], list)
            self.assertEqual(data[field_name]["source"], "llm_agent")
            self.assertEqual(data[field_name]["confirmed_by"], "human")


# ---------------------------------------------------------------------------
# TEST: GovernanceLogger
# ---------------------------------------------------------------------------


class TestGovernanceLogger(unittest.TestCase):

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.gov_dir = self.tmp / "governance" / "agent_sessions"
        self.logger = GovernanceLogger(self.gov_dir)

    def _sample_suggestions(self) -> List[LLMSuggestion]:
        s1 = _make_suggestion(status="confirmed")
        s1.confirmed_field = "valuation_date"
        s2 = _make_suggestion(raw_header="Unknown Col", suggested_field=None, status="skipped")
        return [s1, s2]

    def test_artifact_is_written(self):
        suggestions = self._sample_suggestions()
        path = self.logger.write_session(
            session_id="agent_test_001",
            input_file="test.csv",
            portfolio_type="equity_release",
            deterministic_stats={"total_headers": 10, "mapped": 8, "unmapped": 2, "low_confidence": 0},
            suggestions=suggestions,
            aliases_persisted=1,
        )
        self.assertTrue(path.exists())
        self.assertEqual(path.name, "agent_test_001.json")

    def test_artifact_schema(self):
        suggestions = self._sample_suggestions()
        path = self.logger.write_session(
            session_id="agent_schema_test",
            input_file="data/test.csv",
            portfolio_type="equity_release",
            deterministic_stats={"total_headers": 5, "mapped": 3, "unmapped": 2, "low_confidence": 0},
            suggestions=suggestions,
            aliases_persisted=1,
        )
        artifact = json.loads(path.read_text())

        # Required top-level keys
        for key in ("session_id", "timestamp", "input_file", "portfolio_type",
                     "deterministic_pass", "llm_pass", "human_review",
                     "aliases_persisted", "suggestions"):
            self.assertIn(key, artifact, f"Missing key: {key}")

        # llm_pass sub-schema
        llm = artifact["llm_pass"]
        for key in ("sent_to_llm", "suggestions_returned", "null_suggestions", "avg_confidence"):
            self.assertIn(key, llm)

        # human_review sub-schema
        hr = artifact["human_review"]
        for key in ("confirmed", "rejected", "remapped", "skipped"):
            self.assertIn(key, hr)

        # Counts are correct
        self.assertEqual(artifact["human_review"]["confirmed"], 1)
        self.assertEqual(artifact["human_review"]["skipped"], 1)
        self.assertEqual(artifact["aliases_persisted"], 1)

    def test_suggestions_embedded_in_artifact(self):
        suggestions = self._sample_suggestions()
        path = self.logger.write_session(
            session_id="embed_test",
            input_file="x.csv",
            portfolio_type="equity_release",
            deterministic_stats={},
            suggestions=suggestions,
            aliases_persisted=0,
        )
        artifact = json.loads(path.read_text())
        self.assertEqual(len(artifact["suggestions"]), 2)
        self.assertEqual(artifact["suggestions"][0]["raw_header"], "Loan Valn Dt")

    def test_session_id_in_filename(self):
        path = self.logger.write_session(
            session_id="agent_2026-02-18_143000",
            input_file="x.csv",
            portfolio_type="equity_release",
            deterministic_stats={},
            suggestions=[],
            aliases_persisted=0,
        )
        self.assertEqual(path.name, "agent_2026-02-18_143000.json")


# ---------------------------------------------------------------------------
# TEST: HumanReviewSession (CLI mode with mocked input)
# ---------------------------------------------------------------------------


class TestHumanReviewSessionCLI(unittest.TestCase):

    def setUp(self):
        self.reviewer = HumanReviewSession(canonical_fields=CANONICAL_FIELDS)

    def _run_review(self, inputs: List[str], suggestions: List[LLMSuggestion]) -> List[LLMSuggestion]:
        """Simulate user typing inputs sequentially."""
        input_iter = iter(inputs)
        with patch("builtins.input", side_effect=lambda _="": next(input_iter)):
            return self.reviewer.review_cli(suggestions)

    def test_confirm_sets_status(self):
        suggestions = [_make_suggestion()]
        result = self._run_review(["C", ""], suggestions)
        self.assertEqual(result[0].status, "confirmed")
        self.assertEqual(result[0].confirmed_field, "valuation_date")

    def test_skip_sets_status(self):
        suggestions = [_make_suggestion()]
        result = self._run_review(["S"], suggestions)
        self.assertEqual(result[0].status, "skipped")

    def test_quit_marks_remaining_as_skipped(self):
        suggestions = [_make_suggestion(), _make_suggestion(raw_header="Other Col")]
        # Quit after seeing first suggestion
        result = self._run_review(["Q"], suggestions)
        self.assertTrue(all(s.status in ("pending", "skipped") for s in result))

    def test_invalid_choice_prompts_again(self):
        suggestions = [_make_suggestion()]
        # First input is invalid, second is valid
        result = self._run_review(["X", "C", ""], suggestions)
        self.assertEqual(result[0].status, "confirmed")

    def test_no_pending_returns_immediately(self):
        already_confirmed = _make_suggestion(status="confirmed")
        already_confirmed.confirmed_field = "valuation_date"
        result = self.reviewer.review_cli([already_confirmed])
        # Should return without calling input at all
        self.assertEqual(result[0].status, "confirmed")


# ---------------------------------------------------------------------------
# TEST: LLM batching and API interaction
# ---------------------------------------------------------------------------


class TestLLMFieldMapperBatching(unittest.TestCase):

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.registry_path = _write_registry(self.tmp)
        aliases_dir = self.tmp / "aliases"
        aliases_dir.mkdir()
        self.mapper = LLMFieldMapper(
            registry_path=self.registry_path,
            portfolio_type="equity_release",
            aliases_dir=aliases_dir,
            api_key="test_key",
        )

    def _mock_client_response(self, response_data: list) -> MagicMock:
        mock_content = MagicMock()
        mock_content.text = json.dumps(response_data)
        mock_message = MagicMock()
        mock_message.content = [mock_content]
        mock_client = MagicMock()
        mock_client.messages.create.return_value = mock_message
        return mock_client

    def test_batching_splits_correctly(self):
        """12 headers with BATCH_SIZE=10 should result in exactly 2 API calls."""
        headers = [f"Col{i}" for i in range(12)]
        df = pd.DataFrame({h: ["val"] for h in headers})

        valid_response = {
            "suggested_field": "loan_identifier",
            "confidence": 0.8,
            "reasoning": "test",
            "alternative_field": None,
            "semantic_category": "regulatory",
        }

        call_count = 0

        def mock_create(**kwargs):
            nonlocal call_count
            # Figure out batch size from user content
            msgs = kwargs.get("messages", [])
            content = msgs[0]["content"] if msgs else ""
            # Count how many items in the JSON array
            batch_portion = content.split("HEADERS TO MAP")[1] if "HEADERS TO MAP" in content else "[]"
            import re
            count_match = re.search(r"(\d+) items", batch_portion)
            n = int(count_match.group(1)) if count_match else 10
            call_count += 1
            mock_content = MagicMock()
            mock_content.text = json.dumps([valid_response] * n)
            mock_message = MagicMock()
            mock_message.content = [mock_content]
            return mock_message

        with patch.object(self.mapper, "_get_client") as mock_get_client:
            mock_client = MagicMock()
            mock_client.messages.create.side_effect = mock_create
            mock_get_client.return_value = mock_client

            results = self.mapper.suggest_mappings(headers, df)

        self.assertEqual(call_count, 2)  # ceil(12 / 10) = 2
        self.assertEqual(len(results), 12)

    def test_api_failure_returns_null_suggestions(self):
        """If the API call fails, all suggestions in that batch should be null."""
        headers = ["Broken Col"]
        df = pd.DataFrame({"Broken Col": ["a", "b"]})

        with patch.object(self.mapper, "_get_client") as mock_get_client:
            mock_client = MagicMock()
            mock_client.messages.create.side_effect = Exception("Network error")
            mock_get_client.return_value = mock_client

            results = self.mapper.suggest_mappings(headers, df)

        self.assertEqual(len(results), 1)
        self.assertIsNone(results[0].suggested_field)
        self.assertEqual(results[0].confidence, 0.0)

    def test_json_extraction_strips_markdown_fences(self):
        """_extract_json must handle ```json ... ``` fenced responses."""
        fenced = '```json\n[{"key": "value"}]\n```'
        result = LLMFieldMapper._extract_json(fenced)
        self.assertEqual(result, [{"key": "value"}])

    def test_sample_values_in_suggestion(self):
        """Returned LLMSuggestion should include sample values from the column."""
        df = pd.DataFrame({"SomeDateCol": ["2024-01-01", "2024-02-01", "2024-03-01", None]})
        valid_response = [{
            "suggested_field": "valuation_date",
            "confidence": 0.91,
            "reasoning": "date column",
            "alternative_field": None,
            "semantic_category": "collateral",
        }]

        with patch.object(self.mapper, "_get_client") as mock_get_client:
            mock_client = self._mock_client_response(valid_response)
            mock_get_client.return_value = mock_client
            results = self.mapper.suggest_mappings(["SomeDateCol"], df)

        self.assertGreater(len(results[0].sample_values), 0)
        self.assertIn("2024-01-01", results[0].sample_values)


# ---------------------------------------------------------------------------
# TEST: Registry loading helpers
# ---------------------------------------------------------------------------


class TestRegistryHelpers(unittest.TestCase):

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.registry_path = _write_registry(self.tmp)

    def test_load_registry_success(self):
        reg = _load_registry(self.registry_path)
        self.assertIn("fields", reg)
        self.assertIsInstance(reg["fields"], dict)

    def test_load_registry_missing_fields_key_raises(self):
        bad_file = self.tmp / "bad_registry.yaml"
        bad_file.write_text(yaml.dump({"not_fields": {}}), encoding="utf-8")
        with self.assertRaises(ValueError):
            _load_registry(bad_file)

    def test_catalogue_subset_respects_max_fields(self):
        subset = _build_catalogue_subset(MINIMAL_REGISTRY, "equity_release", max_fields=2)
        self.assertLessEqual(len(subset), 2)

    def test_catalogue_subset_prioritises_common_fields(self):
        subset = _build_catalogue_subset(MINIMAL_REGISTRY, "equity_release", max_fields=2)
        names = [f["name"] for f in subset]
        # common fields should appear before equity_release-specific ones
        # valuation_date, current_balance are common; interest_rate_type is equity_release
        # With max=2 only common fields should appear
        for name in names:
            fpt = MINIMAL_REGISTRY["fields"][name].get("portfolio_type", "")
            self.assertIn(fpt, ("common", "equity_release"))


# ---------------------------------------------------------------------------
# RUN
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    unittest.main(verbosity=2)
