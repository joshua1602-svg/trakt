#!/usr/bin/env python3
"""
tests/test_onboarding_agent_v2.py

Tests for the Trakt Onboarding Agent v2 (engine/onboarding_agent).

Covers the PART 12 minimum test matrix:
  1. File classifier (cashflow / collateral / loan / pipeline / warehouse)
  2. Column profiler (dates, numeric balances, identifiers, <DATE> redaction)
  3. Source overlap detects duplicate balance fields across loan/cashflow
  4. Mapping candidates (balance, postcode/valuation, pipeline dates, cashflow)
  5. Config suggestions (reporting date, asset class, currency, warehouse)
  6. Gap questions (conflicting dates, enum, missing config, ambiguous source)
  7. Review pack HTML is generated
  8. End-to-end CLI run produces all numbered artefacts

No network / LLM calls. Uses the committed synthetic_onboarding_pack.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import pandas as pd

from engine.onboarding_agent import file_classifier, file_profiler
from engine.onboarding_agent.file_profiler import redact_value
from engine.onboarding_agent.onboarding_orchestrator import run_onboarding

PACK = _REPO_ROOT / "synthetic_onboarding_pack"
REGISTRY = _REPO_ROOT / "config" / "system" / "fields_registry.yaml"
ALIASES = _REPO_ROOT / "config" / "system"


# ---------------------------------------------------------------------------
# 1. File classifier
# ---------------------------------------------------------------------------


class TestFileClassifier(unittest.TestCase):
    def setUp(self):
        self.by_name = {
            i.file_name: i for i in file_classifier.classify_directory(PACK)
        }

    def test_cashflow_report(self):
        self.assertEqual(
            self.by_name["monthly_cashflow_report.csv"].classification, "cashflow_report"
        )

    def test_collateral_report(self):
        self.assertEqual(
            self.by_name["monthly_collateral_report.csv"].classification, "collateral_report"
        )

    def test_loan_report(self):
        self.assertEqual(
            self.by_name["monthly_loan_report.csv"].classification, "current_loan_report"
        )

    def test_pipeline_report(self):
        self.assertEqual(
            self.by_name["monthly_pipeline_report.csv"].classification, "pipeline_report"
        )

    def test_warehouse_agreement(self):
        self.assertEqual(
            self.by_name["monthly_loan_report.csv"].file_type, "csv"
        )
        self.assertEqual(
            self.by_name["warehouse_funding_agreement.md"].classification, "warehouse_agreement"
        )

    def test_row_and_column_counts(self):
        loan = self.by_name["monthly_loan_report.csv"]
        self.assertEqual(loan.row_count, 8)
        self.assertEqual(loan.column_count, 10)


# ---------------------------------------------------------------------------
# 2. Column profiler + redaction
# ---------------------------------------------------------------------------


class TestColumnProfiler(unittest.TestCase):
    def setUp(self):
        profiles = file_profiler.profile_file(PACK / "monthly_loan_report.csv")
        self.by_col = {p.source_column: p for p in profiles}

    def test_detects_date_column(self):
        self.assertEqual(self.by_col["origination_date"].inferred_type, "date")
        self.assertTrue(self.by_col["maturity_date"].date_min)

    def test_detects_numeric_balance(self):
        self.assertIn(self.by_col["current_balance"].inferred_type, ("decimal", "integer"))
        self.assertTrue(self.by_col["current_balance"].max_value)

    def test_detects_identifier(self):
        self.assertTrue(self.by_col["loan_id"].likely_identifier)
        self.assertEqual(self.by_col["loan_id"].inferred_type, "identifier")

    def test_detects_reporting_date(self):
        self.assertTrue(self.by_col["reporting_date"].likely_reporting_date)

    def test_dates_redacted_as_date_not_phone(self):
        # The headline fix: ISO dates must become <DATE>, never <PHONE>.
        self.assertEqual(redact_value("2026-01-31"), "<DATE>")
        self.assertEqual(redact_value("31/01/2026"), "<DATE>")
        self.assertNotIn("<PHONE>", redact_value("2026-01-31"))
        # Long identifiers must not be phones either.
        self.assertEqual(redact_value("77658601"), "<ID>")
        self.assertNotIn("<PHONE>", redact_value("77658601"))
        # Plain financial amounts stay visible.
        self.assertEqual(redact_value("148250.55"), "148250.55")
        # A genuinely phone-shaped value is still redacted.
        self.assertEqual(redact_value("020 7946 0123"), "<PHONE>")

    def test_profile_samples_use_date_token(self):
        samples = self.by_col["origination_date"].sample_values_redacted
        self.assertTrue(all(s == "<DATE>" for s in samples))
        self.assertFalse(any("<PHONE>" in s for s in samples))


# ---------------------------------------------------------------------------
# Shared end-to-end project for the remaining tests
# ---------------------------------------------------------------------------


class _ProjectFixture(unittest.TestCase):
    project = None

    @classmethod
    def setUpClass(cls):
        import tempfile

        cls._tmp = tempfile.mkdtemp(prefix="onboarding_v2_")
        cls.project = run_onboarding(
            input_dir=str(PACK),
            client_name="SYNTHETIC_ONBOARDING_TEST",
            output_dir=cls._tmp,
            registry_path=str(REGISTRY),
            aliases_dir=str(ALIASES),
        )
        cls.out = Path(cls._tmp)


# ---------------------------------------------------------------------------
# 3. Source overlap
# ---------------------------------------------------------------------------


class TestSourceOverlap(_ProjectFixture):
    def test_balance_overlap_detected(self):
        balances = [
            o for o in self.project.overlap_analysis
            if o.canonical_candidate == "current_principal_balance"
        ]
        self.assertTrue(balances, "expected a balance overlap finding")
        cols = {balances[0].source_column_a, balances[0].source_column_b}
        self.assertIn("principal_outstanding", cols)
        self.assertIn("current_balance", cols)

    def test_balance_overlap_high_match_rate(self):
        balances = [
            o for o in self.project.overlap_analysis
            if o.canonical_candidate == "current_principal_balance"
            and {"principal_outstanding", "current_balance"}
            == {o.source_column_a, o.source_column_b}
        ]
        self.assertEqual(balances[0].sample_match_rate, 1.0)


# ---------------------------------------------------------------------------
# 4. Mapping candidates
# ---------------------------------------------------------------------------


class TestMappingCandidates(_ProjectFixture):
    def _map(self, file_name, source_column):
        for m in self.project.mapping_candidates:
            if m.source_file == file_name and m.source_column == source_column:
                return m
        return None

    def test_current_principal_balance_from_loan_report(self):
        m = self._map("monthly_loan_report.csv", "current_balance")
        self.assertEqual(m.candidate_canonical_field, "current_principal_balance")

    def test_postcode_and_valuation_from_collateral_report(self):
        pc = self._map("monthly_collateral_report.csv", "property_post_code")
        val = self._map("monthly_collateral_report.csv", "valuation_amount")
        self.assertEqual(pc.candidate_canonical_field, "property_post_code")
        self.assertEqual(val.candidate_canonical_field, "current_valuation_amount")

    def test_pipeline_dates_present_as_candidates(self):
        m = self._map("monthly_pipeline_report.csv", "expected_completion_date")
        self.assertIsNotNone(m)
        f = self._map("monthly_pipeline_report.csv", "expected_funding_amount")
        self.assertIsNotNone(f)

    def test_cashflow_fields_present(self):
        m = self._map("monthly_cashflow_report.csv", "principal_outstanding")
        self.assertEqual(m.candidate_canonical_field, "current_principal_balance")
        self.assertIsNotNone(self._map("monthly_cashflow_report.csv", "total_cashflow"))

    def test_region_not_mapped_to_classification_year(self):
        m = self._map("monthly_collateral_report.csv", "collateral_region")
        self.assertNotEqual(m.candidate_canonical_field, "geographic_region_classification")


# ---------------------------------------------------------------------------
# 5. Config suggestions
# ---------------------------------------------------------------------------


class TestConfigSuggestions(_ProjectFixture):
    def _cfg(self, field):
        return {c.field: c for c in self.project.config_suggestions}.get(field)

    def test_reporting_date_candidate(self):
        self.assertIsNotNone(self._cfg("reporting_date"))

    def test_asset_class_candidate(self):
        self.assertEqual(self._cfg("asset_class").suggested_value, "equity_release")

    def test_currency_and_jurisdiction(self):
        self.assertEqual(self._cfg("currency").suggested_value, "GBP")
        self.assertEqual(self._cfg("jurisdiction").suggested_value, "GB")

    def test_warehouse_present(self):
        self.assertEqual(self._cfg("warehouse_facility_present").suggested_value, "true")
        self.assertIsNotNone(self._cfg("advance_rate"))


# ---------------------------------------------------------------------------
# 6. Gap questions
# ---------------------------------------------------------------------------


class TestGapQuestions(_ProjectFixture):
    def _categories(self):
        return {q.category for q in self.project.gap_questions}

    def test_conflicting_dates_blocking(self):
        date_qs = [q for q in self.project.gap_questions if q.category == "date"]
        self.assertTrue(date_qs)
        self.assertEqual(date_qs[0].severity, "blocking")

    def test_unresolved_enum(self):
        enum_qs = [q for q in self.project.gap_questions if q.category == "enum"]
        self.assertTrue(enum_qs)
        self.assertIn("manual", enum_qs[0].question)

    def test_manual_enum_default_is_requires_review_not_othr(self):
        enum_qs = [
            q for q in self.project.gap_questions
            if q.category == "enum" and q.subject_value == "manual"
        ]
        self.assertTrue(enum_qs)
        q = enum_qs[0]
        self.assertEqual(q.default_recommendation, "requires_review")
        self.assertNotEqual(q.default_recommendation, "OTHR")
        self.assertEqual(q.severity, "high")
        # Source evidence preserved.
        self.assertIn("employment_status", q.source_evidence)
        self.assertIn("treat_as_missing", q.candidate_answers)


    def test_ambiguous_source(self):
        self.assertIn("source_of_truth", self._categories())

    def test_geography_policy_question(self):
        self.assertIn("geography", self._categories())


class TestClassificationYear(_ProjectFixture):
    def _cfg(self, field):
        return {c.field: c for c in self.project.config_suggestions}.get(field)

    def test_classification_year_not_from_reporting_date(self):
        cy = self._cfg("classification_year")
        rd = self._cfg("reporting_date")
        self.assertIsNotNone(cy)
        # reporting date is 2026-0x; classification year must NOT be 2026.
        self.assertNotEqual(cy.suggested_value, "2026")
        if rd:
            self.assertNotEqual(cy.suggested_value, rd.suggested_value[:4])

    def test_classification_year_sourced_from_policy(self):
        cy = self._cfg("classification_year")
        self.assertEqual(cy.suggested_value, "2021")
        self.assertEqual(cy.review_status, "requires_review")
        self.assertIn("policy", (cy.source_column_or_document_reference + cy.evidence).lower())
        # Evidence must clarify it is NOT the reporting date (RREL12 semantics).
        self.assertIn("not derived from the reporting date", cy.evidence.lower())


# ---------------------------------------------------------------------------
# 7 & 8. Review pack + artefacts
# ---------------------------------------------------------------------------


class TestArtifacts(_ProjectFixture):
    def test_review_pack_html_generated(self):
        html = self.out / "08_onboarding_review_pack.html"
        self.assertTrue(html.exists())
        text = html.read_text(encoding="utf-8")
        self.assertIn("Onboarding Review Pack", text)
        self.assertIn("Executive summary", text)

    def test_all_numbered_artifacts_present(self):
        expected = [
            "01_file_inventory.csv", "01_file_inventory.json",
            "02_column_profiles.csv", "02_column_profiles.json",
            "03_candidate_keys.csv", "04_source_overlap_analysis.csv",
            "05_mapping_candidates.csv", "05_mapping_candidates.json",
            "06_config_suggestions.yaml", "06_config_suggestions.csv",
            "07_gap_questions.yaml", "07_gap_questions.csv",
            "08_onboarding_review_pack.html", "09_onboarding_run_summary.json",
        ]
        for name in expected:
            self.assertTrue((self.out / name).exists(), f"missing {name}")

    def test_review_status_blocked(self):
        # Conflicting dates make this run blocked.
        self.assertEqual(self.project.review_status, "blocked")


if __name__ == "__main__":
    unittest.main()
