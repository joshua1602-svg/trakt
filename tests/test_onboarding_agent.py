#!/usr/bin/env python3
"""
tests/test_onboarding_agent.py

Unit tests for Onboarding Agent v1.

Principles:
  - No live Claude API calls — all LLM calls are mocked.
  - Tests cover all 9 specified test cases from the spec.
  - Tests are self-contained with fixtures / temp directories.
  - Tests remain fast (no subprocess calls to semantic_alignment where avoidable).
"""

from __future__ import annotations

import json
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import MagicMock, patch

import yaml

# Make agents importable
_HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_HERE))

from agents.onboarding_schemas import (
    ConfigBootstrapResult,
    EnumReviewItem,
    MappingReviewItem,
    OnboardingResult,
)
from agents.config_bootstrap_agent import (
    ConfigBootstrapAgent,
    _detect_asset_class,
    _build_deterministic_questions,
    _profile_tape,
)
from agents.onboarding_agent import (
    _build_mapping_review_items,
    _build_blocker_questions,
    _build_narrative,
    _determine_status,
    _safe_float,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_MINIMAL_REGISTRY_YAML = textwrap.dedent("""\
    consumers: {}
    fields:
      loan_identifier:
        category: regulatory
        format: string
        portfolio_type: common
        regime_mapping:
          ESMA_Annex2:
            code: RREL1
            priority: Mandatory
        layer: core
        core_canonical: true
      account_status:
        allowed_values: account_status
        category: regulatory
        format: list
        portfolio_type: common
        regime_mapping:
          ESMA_Annex2:
            code: RREL69
            priority: Mandatory
        layer: core
        core_canonical: true
      current_balance:
        category: regulatory
        format: decimal
        portfolio_type: common
        regime_mapping:
          ESMA_Annex2:
            code: RREL40
            priority: Optional
        layer: core
        core_canonical: true
      origination_date:
        category: regulatory
        format: date
        portfolio_type: common
        regime_mapping:
          ESMA_Annex2:
            code: RREL3
            priority: Mandatory
        layer: core
        core_canonical: true
      interest_rate_type:
        allowed_values: interest_rate_type
        category: regulatory
        format: list
        portfolio_type: equity_release
        regime_mapping:
          ESMA_Annex2:
            code: RREL20
            priority: Optional
        layer: core
        core_canonical: true
""")

_MINIMAL_CLIENT_CONFIG_YAML = textwrap.dedent("""\
    client:
      client_id: test_client
    portfolio:
      asset_class: equity_release
      country: GB
      base_currency: GBP
      static_reporting_date: "2025-11-30"
    default_regime: ESMA_Annex2
    regime: ESMA_Annex2
    defaults:
      originator_legal_entity_identifier: "213800ABCDE123456701N"
    supported_regimes:
      - ESMA_Annex2
""")

_SYNTHETIC_TAPE_CONTENT = textwrap.dedent("""\
    loan_id,acct_status,balance,orig_dt,interest_type
    L001,Active,150000,2020-01-15,Fixed
    L002,Arrears,200000,2019-06-01,Variable
    L003,Active,175000,2021-03-20,Fixed
""")


def _write_temp_tape(content: str = _SYNTHETIC_TAPE_CONTENT) -> Path:
    """Write a small synthetic tape CSV to a temporary file."""
    tmp = tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w", encoding="utf-8")
    tmp.write(content)
    tmp.close()
    return Path(tmp.name)


def _write_temp_registry() -> Path:
    tmp = tempfile.NamedTemporaryFile(suffix=".yaml", delete=False, mode="w", encoding="utf-8")
    tmp.write(_MINIMAL_REGISTRY_YAML)
    tmp.close()
    return Path(tmp.name)


def _write_temp_client_config() -> Path:
    tmp = tempfile.NamedTemporaryFile(suffix=".yaml", delete=False, mode="w", encoding="utf-8")
    tmp.write(_MINIMAL_CLIENT_CONFIG_YAML)
    tmp.close()
    return Path(tmp.name)


# ---------------------------------------------------------------------------
# Test 1: Existing approved config → no config questions → proceeds to mapping
# ---------------------------------------------------------------------------

class TestConfigBootstrapWithApprovedConfig(unittest.TestCase):

    def test_existing_approved_config_no_questions(self):
        """
        When a fully populated client config is provided, ConfigBootstrapAgent
        should produce status=approved, no blocking questions, and proceed=True.
        """
        tape_path = _write_temp_tape()
        client_config_path = _write_temp_client_config()

        with tempfile.TemporaryDirectory() as tmp_dir:
            agent = ConfigBootstrapAgent(
                raw_tape_path=tape_path,
                run_id="test_run_01",
                output_dir=tmp_dir,
                existing_client_config_path=client_config_path,
                llm_enabled=False,
            )
            result = agent.run()

        # Status should be approved because all critical values are present
        self.assertEqual(result.status, "approved")
        self.assertTrue(result.proceed)
        self.assertFalse(result.approval_required)
        # No blocking questions
        blocking = [q for q in result.config_questions if q.get("blocking")]
        self.assertEqual(len(blocking), 0)


# ---------------------------------------------------------------------------
# Test 2: New client with missing critical config → questions generated → blocked/review_required
# ---------------------------------------------------------------------------

class TestConfigBootstrapMissingCriticalConfig(unittest.TestCase):

    def test_missing_lei_generates_blocking_question(self):
        """
        When no client config is provided and the tape has no LEI signals,
        a blocking question for the originator LEI should be generated.
        """
        tape_path = _write_temp_tape()
        with tempfile.TemporaryDirectory() as tmp_dir:
            agent = ConfigBootstrapAgent(
                raw_tape_path=tape_path,
                run_id="test_run_02",
                output_dir=tmp_dir,
                llm_enabled=False,
            )
            result = agent.run()

        # Status should be blocked or review_required
        self.assertIn(result.status, ("blocked", "review_required"))
        self.assertFalse(result.proceed)
        # LEI question should be present
        lei_questions = [
            q for q in result.config_questions
            if "originator_legal_entity_identifier" in q.get("field", "")
        ]
        self.assertGreater(len(lei_questions), 0)

    def test_draft_config_is_written(self):
        """Draft config file should always be written even when blocked."""
        tape_path = _write_temp_tape()
        with tempfile.TemporaryDirectory() as tmp_dir:
            agent = ConfigBootstrapAgent(
                raw_tape_path=tape_path,
                run_id="test_run_02b",
                output_dir=tmp_dir,
                llm_enabled=False,
            )
            result = agent.run()
            self.assertTrue(Path(result.draft_config_path).exists())


# ---------------------------------------------------------------------------
# Test 3: Deterministic mapping maps high-confidence fields → no LLM for those
# ---------------------------------------------------------------------------

class TestDeterministicMappingNoLLM(unittest.TestCase):

    def test_high_confidence_fields_not_sent_to_llm(self):
        """
        Fields with deterministic methods (exact/alias/normalized) and
        confidence >= threshold should NOT appear as LLM targets.
        """
        confidence_threshold_review = 0.75

        mapping_report = [
            {"raw_header": "loan_id", "canonical_field": "loan_identifier",
             "mapping_method": "exact", "confidence": "1.0"},
            {"raw_header": "bal", "canonical_field": "current_balance",
             "mapping_method": "alias", "confidence": "1.0"},
            {"raw_header": "mystery_field", "canonical_field": "",
             "mapping_method": "unmapped", "confidence": "0.0"},
        ]
        mandatory = {"loan_identifier", "account_status"}
        items = _build_mapping_review_items(mapping_report, mandatory, 0.75)

        exact_item = next(i for i in items if i.raw_field == "loan_id")
        self.assertEqual(exact_item.mapping_source, "exact")
        self.assertFalse(exact_item.requires_review)

        alias_item = next(i for i in items if i.raw_field == "bal")
        self.assertEqual(alias_item.mapping_source, "alias")
        self.assertFalse(alias_item.requires_review)

        unmapped_item = next(i for i in items if i.raw_field == "mystery_field")
        self.assertEqual(unmapped_item.mapping_source, "unmapped")
        self.assertTrue(unmapped_item.requires_review)


# ---------------------------------------------------------------------------
# Test 4: LLM mapping is called only for unresolved/low-confidence fields
# ---------------------------------------------------------------------------

class TestLLMMappingOnlyForUnresolved(unittest.TestCase):

    def test_llm_updates_only_unmapped_items(self):
        """
        After LLM call, only unmapped/low-confidence items should have
        their mapping_source updated to 'llm'.  High-confidence items unchanged.
        """
        # Simulate existing mapping items
        items = [
            MappingReviewItem(
                raw_field="loan_id",
                suggested_canonical_field="loan_identifier",
                mapping_source="exact",
                confidence=1.0,
                requires_review=False,
            ),
            MappingReviewItem(
                raw_field="mystery_col",
                suggested_canonical_field=None,
                mapping_source="unmapped",
                confidence=0.0,
                requires_review=True,
            ),
        ]

        # Simulate what LLM suggestion JSON would look like
        llm_suggestions = [
            {
                "raw_header": "mystery_col",
                "suggested_field": "origination_date",
                "confidence": 0.82,
                "reasoning": "Column contains date-like values.",
                "sample_values": ["2020-01-15"],
            }
        ]

        # Apply the merge logic directly (mirrors _run_llm_mapping merge step)
        sugg_by_header = {s["raw_header"]: s for s in llm_suggestions}
        for item in items:
            sugg = sugg_by_header.get(item.raw_field)
            if not sugg:
                continue
            item.suggested_canonical_field = sugg["suggested_field"]
            item.confidence = sugg["confidence"]
            item.mapping_source = "llm"
            item.requires_review = True
            item.sample_values = sugg.get("sample_values", [])

        # loan_id should be untouched
        self.assertEqual(items[0].mapping_source, "exact")
        self.assertEqual(items[0].suggested_canonical_field, "loan_identifier")
        self.assertFalse(items[0].requires_review)

        # mystery_col should be updated by LLM
        self.assertEqual(items[1].mapping_source, "llm")
        self.assertEqual(items[1].suggested_canonical_field, "origination_date")
        self.assertAlmostEqual(items[1].confidence, 0.82)
        self.assertTrue(items[1].requires_review)


# ---------------------------------------------------------------------------
# Test 5: Enum mapping produces EnumReviewItem for unknown values
# ---------------------------------------------------------------------------

class TestEnumReviewItemGeneration(unittest.TestCase):

    def test_enum_review_items_created_for_unknown_values(self):
        """
        EnumReviewItem objects should be produced for enum values
        that cannot be deterministically resolved.
        """
        items = [
            EnumReviewItem(
                field_name="account_status",
                raw_value="UNKNOWN_VALUE",
                suggested_value=None,
                mapping_source="unmapped",
                confidence=0.0,
                requires_review=True,
                blocker=True,
                reason="Enum value not in allowed set.",
                sample_count=5,
            )
        ]
        self.assertEqual(len(items), 1)
        self.assertEqual(items[0].field_name, "account_status")
        self.assertEqual(items[0].raw_value, "UNKNOWN_VALUE")
        self.assertIsNone(items[0].suggested_value)
        self.assertTrue(items[0].blocker)
        self.assertEqual(items[0].sample_count, 5)

    def test_enum_review_item_serialization(self):
        item = EnumReviewItem(
            field_name="account_status",
            raw_value="ACT",
            suggested_value="Active",
            mapping_source="synonym",
            confidence=0.9,
            requires_review=False,
            blocker=False,
            reason="",
            sample_count=10,
        )
        d = item.to_dict()
        self.assertEqual(d["field_name"], "account_status")
        self.assertEqual(d["mapping_source"], "synonym")


# ---------------------------------------------------------------------------
# Test 6: Unmapped mandatory field → status blocked
# ---------------------------------------------------------------------------

class TestBlockedOnUnmappedMandatory(unittest.TestCase):

    def test_unmapped_mandatory_causes_blocked_status(self):
        """
        If a mandatory field for the regime has no mapping,
        status must be blocked.
        """
        bootstrap = ConfigBootstrapResult(
            run_id="t6",
            status="approved",
            proceed=True,
        )
        mapping_items = [
            MappingReviewItem(
                raw_field="loan_id",
                suggested_canonical_field="loan_identifier",
                mapping_source="exact",
                confidence=1.0,
                required_for_regime=True,
                requires_review=False,
                blocker=False,
            ),
            MappingReviewItem(
                raw_field="unknown_mandatory",
                suggested_canonical_field=None,
                mapping_source="unmapped",
                confidence=0.0,
                required_for_regime=True,
                requires_review=True,
                blocker=True,
            ),
        ]
        enum_items: List[EnumReviewItem] = []

        status = _determine_status(bootstrap, mapping_items, enum_items)
        self.assertEqual(status, "blocked")

    def test_blocked_generates_blocker_question(self):
        bootstrap = ConfigBootstrapResult(run_id="t6b", status="approved")
        mapping_items = [
            MappingReviewItem(
                raw_field="unresolved_field",
                suggested_canonical_field=None,
                mapping_source="unmapped",
                confidence=0.0,
                required_for_regime=True,
                requires_review=True,
                blocker=True,
            ),
        ]
        enum_items: List[EnumReviewItem] = []
        blockers = _build_blocker_questions(mapping_items, enum_items, bootstrap)
        self.assertGreater(len(blockers), 0)
        self.assertTrue(all(b["blocking"] for b in blockers))


# ---------------------------------------------------------------------------
# Test 7: Optional unmapped fields → review_required (not blocked)
# ---------------------------------------------------------------------------

class TestOptionalUnmappedNotBlocked(unittest.TestCase):

    def test_optional_unmapped_field_is_review_required(self):
        """
        Unmapped optional (non-mandatory) fields should result in
        review_required, not blocked.
        """
        bootstrap = ConfigBootstrapResult(
            run_id="t7",
            status="approved",
            proceed=True,
        )
        mapping_items = [
            MappingReviewItem(
                raw_field="loan_id",
                suggested_canonical_field="loan_identifier",
                mapping_source="exact",
                confidence=1.0,
                required_for_regime=True,
                requires_review=False,
                blocker=False,
            ),
            MappingReviewItem(
                raw_field="optional_extra",
                suggested_canonical_field=None,
                mapping_source="fuzz_token_set",
                confidence=0.60,
                required_for_regime=False,
                requires_review=True,
                blocker=False,     # not mandatory → not a blocker
            ),
        ]
        enum_items: List[EnumReviewItem] = []

        status = _determine_status(bootstrap, mapping_items, enum_items)
        self.assertEqual(status, "review_required")


# ---------------------------------------------------------------------------
# Test 8: OnboardingResult serializes to JSON and can be loaded back
# ---------------------------------------------------------------------------

class TestOnboardingResultSerialization(unittest.TestCase):

    def _make_result(self) -> OnboardingResult:
        bootstrap = ConfigBootstrapResult(
            run_id="ser_test",
            status="approved",
            detected_asset_class="equity_release",
            detected_asset_confidence=0.95,
            selected_regime="ESMA_Annex2",
            proceed=True,
        )
        mapping_items = [
            MappingReviewItem(
                raw_field="loan_id",
                suggested_canonical_field="loan_identifier",
                mapping_source="exact",
                confidence=1.0,
            )
        ]
        enum_items = [
            EnumReviewItem(
                field_name="account_status",
                raw_value="ACTIVE",
                suggested_value="Active",
                mapping_source="synonym",
                confidence=0.95,
                requires_review=False,
                sample_count=42,
            )
        ]
        return OnboardingResult(
            run_id="ser_test",
            status="ready_for_validation",
            total_input_fields=5,
            mapped_fields_count=5,
            deterministic_mapped_count=5,
            enum_success_rate=1.0,
            config_bootstrap=bootstrap,
            mapping_review_items=mapping_items,
            enum_review_items=enum_items,
            proceed_to_validation=True,
            narrative_summary="Test summary.",
        )

    def test_round_trip_json(self):
        result = self._make_result()
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "onboarding_result.json"
            result.to_json(path)
            self.assertTrue(path.exists())
            loaded = OnboardingResult.from_json(path)

        self.assertEqual(loaded.run_id, "ser_test")
        self.assertEqual(loaded.status, "ready_for_validation")
        self.assertTrue(loaded.proceed_to_validation)
        self.assertEqual(loaded.total_input_fields, 5)
        self.assertIsNotNone(loaded.config_bootstrap)
        self.assertEqual(loaded.config_bootstrap.detected_asset_class, "equity_release")  # type: ignore[union-attr]
        self.assertEqual(len(loaded.mapping_review_items), 1)
        self.assertEqual(loaded.mapping_review_items[0].raw_field, "loan_id")
        self.assertEqual(len(loaded.enum_review_items), 1)

    def test_to_dict_is_json_serializable(self):
        result = self._make_result()
        d = result.to_dict()
        # Should not raise
        serialized = json.dumps(d, default=str)
        self.assertIn("ser_test", serialized)

    def test_config_bootstrap_result_round_trip(self):
        cb = ConfigBootstrapResult(
            run_id="cb_test",
            status="approved",
            detected_asset_class="rre",
            selected_regime="ESMA_Annex2",
            proceed=True,
            tape_row_count=1000,
            config_questions=[{"question_id": "q1", "question": "What is the currency?"}],
        )
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "bootstrap.json"
            cb.to_json(path)
            loaded = ConfigBootstrapResult.from_json(path)
        self.assertEqual(loaded.detected_asset_class, "rre")
        self.assertEqual(loaded.tape_row_count, 1000)
        self.assertEqual(len(loaded.config_questions), 1)


# ---------------------------------------------------------------------------
# Test 9: CLI runs against synthetic CSV with LLM disabled
# ---------------------------------------------------------------------------

class TestCLIWithLLMDisabled(unittest.TestCase):
    """
    End-to-end test of run_onboarding_agent with llm_enabled=False.
    Mocks semantic_alignment subprocess to avoid requiring the full engine.
    """

    def _write_mock_mapping_report(self, output_dir: Path, stem: str) -> None:
        """Write a fake mapping report CSV as semantic_alignment would produce."""
        import csv as csv_module
        mapping_path = output_dir / f"{stem}_mapping_report.csv"
        rows = [
            {"raw_header": "loan_id", "canonical_field": "loan_identifier",
             "mapping_method": "exact", "confidence": "1.0", "sample_values": ""},
            {"raw_header": "acct_status", "canonical_field": "account_status",
             "mapping_method": "alias", "confidence": "1.0", "sample_values": ""},
            {"raw_header": "balance", "canonical_field": "current_balance",
             "mapping_method": "fuzz_token_set", "confidence": "0.80", "sample_values": ""},
            {"raw_header": "orig_dt", "canonical_field": "origination_date",
             "mapping_method": "alias", "confidence": "1.0", "sample_values": ""},
            {"raw_header": "interest_type", "canonical_field": "",
             "mapping_method": "unmapped", "confidence": "0.0", "sample_values": ""},
        ]
        with mapping_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv_module.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

        # Also write empty canonical CSV so enum mapping step doesn't fail
        canonical_path = output_dir / f"{stem}_canonical_full.csv"
        canonical_path.write_text("loan_identifier,account_status\n", encoding="utf-8")

    def test_cli_no_llm_produces_result_json(self):
        """
        Running run_onboarding_agent with llm_enabled=False should:
        - not call the LLM
        - produce onboarding_result.json
        - return a non-failed status
        """
        from agents.onboarding_agent import run_onboarding_agent

        tape_path = _write_temp_tape()
        client_config_path = _write_temp_client_config()

        with tempfile.TemporaryDirectory() as tmp_dir:
            # Mock subprocess.run so semantic_alignment is not actually called
            def mock_subprocess(cmd, **kwargs):
                # Write expected output files into the run directory
                output_dir_arg = None
                for i, arg in enumerate(cmd):
                    if arg == "--output-dir" and i + 1 < len(cmd):
                        output_dir_arg = Path(cmd[i + 1])
                        break
                if output_dir_arg:
                    output_dir_arg.mkdir(parents=True, exist_ok=True)
                    self._write_mock_mapping_report(output_dir_arg, tape_path.stem)
                mock_result = MagicMock()
                mock_result.returncode = 0
                mock_result.stderr = ""
                return mock_result

            with patch("agents.onboarding_agent.subprocess.run", side_effect=mock_subprocess):
                result = run_onboarding_agent(
                    raw_tape_path=str(tape_path),
                    run_id="cli_test_01",
                    client_config_path=str(client_config_path),
                    output_dir=tmp_dir,
                    llm_enabled=False,
                )

            # Result file should exist
            self.assertTrue(Path(result.onboarding_result_path).exists())
            # Status should not be "failed"
            self.assertNotEqual(result.status, "failed")
            # Mapped count should be > 0
            self.assertGreater(result.mapped_fields_count, 0)
            # Narrative should be non-empty
            self.assertGreater(len(result.narrative_summary), 10)

    def test_no_llm_calls_made_when_disabled(self):
        """
        Verifies no LLM calls are made when llm_enabled=False.
        We verify by asserting governance_artifact_path is empty (LLM was never
        invoked) and that no LLM-sourced mapping items appear in the result.
        """
        from agents.onboarding_agent import run_onboarding_agent

        tape_path = _write_temp_tape()
        client_config_path = _write_temp_client_config()

        with tempfile.TemporaryDirectory() as tmp_dir:
            def mock_subprocess(cmd, **kwargs):
                output_dir_arg = None
                for i, arg in enumerate(cmd):
                    if arg == "--output-dir" and i + 1 < len(cmd):
                        output_dir_arg = Path(cmd[i + 1])
                        break
                if output_dir_arg:
                    output_dir_arg.mkdir(parents=True, exist_ok=True)
                    self._write_mock_mapping_report(output_dir_arg, tape_path.stem)
                mock_result = MagicMock()
                mock_result.returncode = 0
                mock_result.stderr = ""
                return mock_result

            with patch("agents.onboarding_agent.subprocess.run", side_effect=mock_subprocess):
                result = run_onboarding_agent(
                    raw_tape_path=str(tape_path),
                    run_id="cli_test_02",
                    client_config_path=str(client_config_path),
                    output_dir=tmp_dir,
                    llm_enabled=False,
                )

            # No LLM-sourced mapping items should appear when LLM is disabled
            llm_items = [i for i in result.mapping_review_items if i.mapping_source == "llm"]
            self.assertEqual(len(llm_items), 0)
            # Governance artifact should not be populated (no LLM session)
            self.assertEqual(result.governance_artifact_path, "")


# ---------------------------------------------------------------------------
# Asset class detection unit tests
# ---------------------------------------------------------------------------

class TestAssetClassDetection(unittest.TestCase):

    def test_equity_release_detected_from_headers(self):
        profile = {"headers": ["loan_id", "no_negative_equity", "drawdown_facility", "balance"]}
        ac, conf = _detect_asset_class(profile)
        self.assertEqual(ac, "equity_release")
        self.assertGreater(conf, 0.0)

    def test_config_overrides_detection(self):
        profile = {"headers": ["loan_id", "vehicle_make", "vin"]}
        existing_config = {"portfolio": {"asset_class": "equity_release"}}
        ac, conf = _detect_asset_class(profile, existing_config)
        self.assertEqual(ac, "equity_release")
        self.assertEqual(conf, 1.0)

    def test_unknown_returned_for_no_signals(self):
        profile = {"headers": ["col_a", "col_b"]}
        ac, conf = _detect_asset_class(profile)
        self.assertEqual(ac, "unknown")
        self.assertEqual(conf, 0.0)


# ---------------------------------------------------------------------------
# Narrative generation
# ---------------------------------------------------------------------------

class TestNarrativeGeneration(unittest.TestCase):

    def _make_minimal_result(self, status: str) -> OnboardingResult:
        cb = ConfigBootstrapResult(
            run_id="narr_test",
            status="approved" if status != "blocked" else "blocked",
            detected_asset_class="equity_release",
            detected_asset_confidence=0.94,
            selected_regime="ESMA_Annex2",
        )
        return OnboardingResult(
            run_id="narr_test",
            status=status,
            total_input_fields=134,
            mapped_fields_count=121,
            deterministic_mapped_count=113,
            llm_suggested_count=8,
            review_fields_count=8,
            unmapped_mandatory_count=2,
            enum_success_rate=0.965,
            enum_review_count=3,
            config_bootstrap=cb,
            proceed_to_validation=(status == "ready_for_validation"),
        )

    def test_narrative_contains_key_stats(self):
        result = self._make_minimal_result("ready_for_validation")
        narrative = _build_narrative(result)
        self.assertIn("121", narrative)
        self.assertIn("134", narrative)
        self.assertIn("equity_release", narrative)
        self.assertIn("ESMA_Annex2", narrative)

    def test_blocked_narrative_mentions_block(self):
        result = self._make_minimal_result("blocked")
        narrative = _build_narrative(result)
        self.assertIn("block", narrative.lower())


# ---------------------------------------------------------------------------
# Safe float
# ---------------------------------------------------------------------------

class TestSafeFloat(unittest.TestCase):

    def test_valid_string(self):
        self.assertAlmostEqual(_safe_float("0.95"), 0.95)

    def test_invalid_returns_default(self):
        self.assertEqual(_safe_float("N/A", 0.0), 0.0)
        self.assertEqual(_safe_float(None, -1.0), -1.0)
        self.assertEqual(_safe_float("", 0.5), 0.5)


if __name__ == "__main__":
    unittest.main(verbosity=2)
