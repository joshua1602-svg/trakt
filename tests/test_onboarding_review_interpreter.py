#!/usr/bin/env python3
"""
tests/test_onboarding_review_interpreter.py

PART 9 tests 7-10 — LLM-assisted review interpreter (deterministic v1):
NL -> structured spec -> deterministic validation -> confirmation gate.
"""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import yaml

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from engine.onboarding_agent.answer_ingestion import (
    STATUS_NEEDS_CONFIRMATION,
    STATUS_READY,
    ProjectContext,
    ingest_answers,
)
from engine.onboarding_agent.onboarding_orchestrator import run_onboarding
from engine.onboarding_agent.review_interpreter import (
    interpret_answers,
    interpret_answers_to_file,
    validate_spec,
)

PACK = _REPO_ROOT / "synthetic_onboarding_pack"
REGISTRY = _REPO_ROOT / "config" / "system" / "fields_registry.yaml"
ALIASES = _REPO_ROOT / "config" / "system"

_NL = (
    "Use the loan report date of 2026-01-31 as reporting date, use the loan "
    "report for balance, use the collateral report for region, treat manual "
    "employment status as missing, and confirm GBZZZ for ESMA."
)


class _Fixture(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp(prefix="interp_"))
        run_onboarding(
            input_dir=str(PACK), client_name="TEST", output_dir=str(cls.tmp),
            registry_path=str(REGISTRY), aliases_dir=str(ALIASES), mode="regulatory_mi",
        )
        cls.ctx = ProjectContext(cls.tmp)


class TestInterpretation(_Fixture):
    def test_nl_to_structured_spec(self):
        spec = interpret_answers(_NL, self.ctx.questions)
        ans = {qid: a.answer for qid, a in spec.answers.items()}
        self.assertEqual(ans.get("Q1"), "2026-01-31")
        self.assertEqual(ans.get("Q2"), "monthly_loan_report.csv")
        self.assertEqual(ans.get("Q3"), "monthly_collateral_report.csv")
        self.assertEqual(ans.get("Q4"), "treat_as_missing")
        self.assertEqual(ans.get("Q5"), "GBZZZ")
        self.assertTrue(spec.requires_confirmation)

    def test_spec_requires_confirmation_flag(self):
        spec = interpret_answers(_NL, self.ctx.questions)
        self.assertTrue(spec.requires_confirmation)
        self.assertEqual(spec.interpreter, "deterministic")

    def test_validation_valid(self):
        spec = interpret_answers(_NL, self.ctx.questions)
        v = validate_spec(spec, self.ctx)
        # All interpreted answers are individually valid; status is "requires_review"
        # only because unanswered (missing-core-field) blocking gaps remain.
        self.assertEqual(v["invalid_answers"], [])
        self.assertIn(v["status"], ("valid", "requires_review"))
        self.assertIn("config/reporting_date", v["proposed_updates"])

    def test_invalid_interpreted_answer_fails_validation(self):
        # An LLM-style callable returning a bad source file.
        def bad_llm(text, questions):
            return {"Q2": "not_a_real_file.csv", "Q1": "2026-01-31"}

        spec = interpret_answers(_NL, self.ctx.questions, llm_enabled=True, llm_callable=bad_llm)
        v = validate_spec(spec, self.ctx)
        self.assertEqual(v["status"], "invalid")
        self.assertTrue(any(i["question_id"] == "Q2" for i in v["invalid_answers"]))

    def test_interpret_to_file_writes_only_spec(self):
        out = interpret_answers_to_file(self.tmp, _NL)
        self.assertTrue(out.exists())
        self.assertEqual(out.name, "16_interpreted_answer_spec.yaml")
        # Dry-run must NOT write approved artefacts.
        for n in ["10_approved_onboarding_project.yaml", "11_approved_config.yaml"]:
            self.assertFalse((self.tmp / n).exists())


class TestConfirmationGate(_Fixture):
    def test_no_write_without_confirmation(self):
        # The confirmation gate must never write approved artefacts when confirm
        # is False, regardless of remaining blocking gaps.
        out = interpret_answers_to_file(self.tmp, _NL)
        report = ingest_answers(self.tmp, out, confirm=False)
        self.assertFalse(report["approved_artefacts_written"])
        self.assertNotEqual(report["approval_status"], STATUS_READY)
        self.assertFalse((self.tmp / "11_approved_config.yaml").exists())

    def test_write_after_confirmation(self):
        # The NL answer resolves the judgment questions; the generated template
        # covers the remaining (missing-core-field) gaps so the pack can reach a
        # confirmable READY state and write approved artefacts.
        spec_path = interpret_answers_to_file(self.tmp, _NL)
        interpreted = yaml.safe_load(spec_path.read_text())["answers"]
        merged = yaml.safe_load((self.tmp / "example_answers.yaml").read_text())
        merged["answers"].update(interpreted)  # NL answers win on the judgment Qs
        combined = self.tmp / "combined_answers.yaml"
        combined.write_text(yaml.safe_dump(merged))

        report = ingest_answers(self.tmp, combined, confirm=True)
        self.assertEqual(report["approval_status"], STATUS_READY)
        self.assertTrue(report["approved_artefacts_written"])
        cfg = yaml.safe_load((self.tmp / "11_approved_config.yaml").read_text())
        self.assertEqual(cfg["reporting_date"], "2026-01-31")
        self.assertEqual(str(cfg["classification_year"]), "2021")
        self.assertEqual(cfg["geography_policy"]["ESMA_Annex2"]["uk_geography_mode"], "GBZZZ")


if __name__ == "__main__":
    unittest.main()
