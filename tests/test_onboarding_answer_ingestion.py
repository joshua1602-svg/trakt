#!/usr/bin/env python3
"""
tests/test_onboarding_answer_ingestion.py

Tests for the v1 answer-ingestion loop (engine/onboarding_agent/answer_ingestion).

Covers PART 8 of the spec:
  4. example answers file validates
  5. ingesting answers produces all approved artefacts (10..15)
  6. blocking question Q1 cleared after ingestion
  7. invalid source-of-truth answer fails validation
  8. approved config: reporting_date = answer; classification_year != reporting
     year; geography_policy ESMA_Annex2 uk_geography_mode = GBZZZ
  9. approved enum decision for employment_status.manual = treat_as_missing

Deterministic, no network / LLM, no Gates 1–5.
"""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import yaml

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from engine.onboarding_agent.answer_ingestion import (
    STATUS_INVALID,
    STATUS_READY,
    ingest_answers,
)
from engine.onboarding_agent.onboarding_orchestrator import run_onboarding

PACK = _REPO_ROOT / "synthetic_onboarding_pack"
REGISTRY = _REPO_ROOT / "config" / "system" / "fields_registry.yaml"
ALIASES = _REPO_ROOT / "config" / "system"


class _GeneratedPack(unittest.TestCase):
    """Generates a fresh onboarding pack in a temp dir for each test class."""

    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.mkdtemp(prefix="onboarding_ingest_")
        cls.out = Path(cls._tmp)
        cls.project = run_onboarding(
            input_dir=str(PACK),
            client_name="SYNTHETIC_ONBOARDING_TEST",
            output_dir=str(cls.out),
            registry_path=str(REGISTRY),
            aliases_dir=str(ALIASES),
        )
        cls.answers_file = cls.out / "example_answers.yaml"


class TestExampleAnswers(_GeneratedPack):
    def test_example_answers_generated_and_valid_yaml(self):
        self.assertTrue(self.answers_file.exists())
        data = yaml.safe_load(self.answers_file.read_text())
        self.assertIn("answers", data)
        # Every gap question has a pre-filled answer.
        q_ids = {q.question_id for q in self.project.gap_questions}
        self.assertTrue(q_ids.issubset(set(data["answers"].keys())))

    def test_example_answers_ingest_cleanly(self):
        report = ingest_answers(self.out, self.answers_file)
        self.assertEqual(report["answers_invalid"], 0)


class TestApprovedArtefacts(_GeneratedPack):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.report = ingest_answers(cls.out, cls.answers_file)

    def test_all_approved_artefacts_written(self):
        for name in [
            "10_approved_onboarding_project.yaml",
            "11_approved_config.yaml",
            "12_approved_mapping_overrides.yaml",
            "13_source_precedence_rules.yaml",
            "14_enum_review_decisions.yaml",
            "15_answer_ingestion_report.json",
        ]:
            self.assertTrue((self.out / name).exists(), f"missing {name}")

    def test_blocking_cleared_and_ready(self):
        self.assertEqual(self.report["blocking_answered"], self.report["blocking_total"])
        self.assertEqual(self.report["approval_status"], STATUS_READY)

    def test_q1_blocking_cleared(self):
        project = yaml.safe_load((self.out / "10_approved_onboarding_project.yaml").read_text())
        self.assertEqual(project["blocking_status"]["unanswered"], [])
        self.assertIn("Q1", project["answered_questions"])

    def test_approved_config_semantics(self):
        cfg = yaml.safe_load((self.out / "11_approved_config.yaml").read_text())
        # reporting_date came from the answer (a real date string)
        self.assertTrue(cfg["reporting_date"].startswith("2026-0"))
        # classification_year must NOT equal the reporting year
        self.assertNotEqual(str(cfg["classification_year"]), cfg["reporting_date"][:4])
        self.assertEqual(str(cfg["classification_year"]), "2021")
        # geography policy
        self.assertEqual(
            cfg["geography_policy"]["ESMA_Annex2"]["uk_geography_mode"], "GBZZZ"
        )

    def test_enum_decision_captured(self):
        enum = yaml.safe_load((self.out / "14_enum_review_decisions.yaml").read_text())
        self.assertEqual(
            enum["employment_status"]["manual"]["decision"], "treat_as_missing"
        )

    def test_source_precedence_captured(self):
        prec = yaml.safe_load((self.out / "13_source_precedence_rules.yaml").read_text())
        self.assertIn("current_principal_balance", prec)
        self.assertEqual(
            prec["current_principal_balance"]["primary_source_file"],
            "monthly_loan_report.csv",
        )
        self.assertEqual(
            prec["current_principal_balance"]["reconciliation_status"], "matched"
        )

    def test_report_json_shape(self):
        report = json.loads((self.out / "15_answer_ingestion_report.json").read_text())
        for key in [
            "questions_total", "blocking_total", "blocking_answered",
            "answers_invalid", "approval_status", "artefacts_written",
        ]:
            self.assertIn(key, report)

    def test_review_pack_shows_approval(self):
        html = (self.out / "08_onboarding_review_pack.html").read_text()
        self.assertIn('id="approval"', html)
        self.assertIn("READY FOR HANDOFF", html)


class TestValidation(_GeneratedPack):
    def _base_answers(self):
        return yaml.safe_load(self.answers_file.read_text())

    def test_invalid_source_of_truth_fails(self):
        answers = self._base_answers()
        # Q2 is a source-of-truth question; point it at a non-existent file.
        answers["answers"]["Q2"]["answer"] = "not_a_real_file.csv"
        bad = self.out / "bad_answers.yaml"
        bad.write_text(yaml.safe_dump(answers))
        report = ingest_answers(self.out, bad)
        self.assertGreaterEqual(report["answers_invalid"], 1)
        self.assertEqual(report["approval_status"], STATUS_INVALID)
        self.assertTrue(any(d["question_id"] == "Q2" for d in report["invalid_detail"]))

    def test_invalid_date_fails(self):
        answers = self._base_answers()
        answers["answers"]["Q1"]["answer"] = "not-a-date"
        bad = self.out / "bad_date_answers.yaml"
        bad.write_text(yaml.safe_dump(answers))
        report = ingest_answers(self.out, bad)
        self.assertGreaterEqual(report["answers_invalid"], 1)

    def test_missing_blocking_answer_is_not_ready(self):
        answers = self._base_answers()
        del answers["answers"]["Q1"]  # drop the blocking date question
        partial = self.out / "partial_answers.yaml"
        partial.write_text(yaml.safe_dump(answers))
        report = ingest_answers(self.out, partial)
        self.assertNotEqual(report["approval_status"], STATUS_READY)


if __name__ == "__main__":
    unittest.main()
