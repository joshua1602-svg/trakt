#!/usr/bin/env python3
"""tests/test_gate4_mi_readiness_flow.py

Gate 4 target-first approval / readiness flow for MI-only onboarding.

Covers the focused fix:
  * the target advisor emits/parses strict machine-readable JSON, tolerating
    markdown fences / prose / a bare array (no wholesale parse_failed);
  * a genuine parse failure emits clear diagnostics (status, error, raw path) and
    accept-target-advice surfaces them instead of silently skipping;
  * non-blocking Gate 4 confirmations can be approved deterministically, writing a
    REAL approved 34 file (not a copied pending template);
  * blocking decisions are never auto-approved;
  * a copied pending YAML is not treated as an approval;
  * MI runtime readiness can be ready while governance readiness is blocked;
  * blocking decisions still prevent governance readiness;
  * funded / pipeline MI calculations are unchanged by this flow.
"""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import yaml

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from engine.onboarding_agent import accept_target_advice as ata
from engine.onboarding_agent import domain_coverage as dc
from engine.onboarding_agent import non_blocking_approval as nba
from engine.onboarding_agent import readiness as rd
from engine.onboarding_agent import target_first_decisions as tfd
from engine.onboarding_agent import target_first_llm_advisor as adv

_DECISIONS = [
    {"decision_id": "D1", "target_field": "youngest_borrower_age",
     "decision_type": "source_priority_confirmation", "blocking": False,
     "source_file": "borrowers.csv", "source_column": "AgeYoungest"},
    {"decision_id": "D2", "target_field": "origination_date",
     "decision_type": "nd_default_confirmation", "blocking": False},
    {"decision_id": "D3", "target_field": "broker_channel",
     "decision_type": "not_applicable_confirmation", "blocking": False},
    {"decision_id": "D4", "target_field": "current_loan_to_value",
     "decision_type": "missing_required_target", "blocking": True},
]
_COVERAGE = [
    {"target_field": "youngest_borrower_age", "selected_source_column": "AgeYoungest",
     "alternative_source_candidates": ""},
    {"target_field": "origination_date", "selected_source_column": "",
     "alternative_source_candidates": ""},
    {"target_field": "broker_channel", "selected_source_column": "Broker",
     "alternative_source_candidates": ""},
    {"target_field": "current_loan_to_value", "selected_source_column": "",
     "alternative_source_candidates": ""},
]


def _write_template(pdir: Path, rows=None):
    tpl = tfd.build_decision_template(rows or _DECISIONS, mode="regulatory_mi",
                                      client_id="client_001", run_id="mi_2025_11")
    tfd.write_decision_template(tpl, pdir)
    return tpl


# --------------------------------------------------------------------------- #
# 1. Strict advisor JSON parsing
# --------------------------------------------------------------------------- #
class TestStrictAdvisorParsing(unittest.TestCase):
    def test_parses_markdown_wrapped_json(self):
        def llm(_prompt):
            payload = {"recommendations": [
                {"decision_id": "D1", "recommended_action": "confirm_selected", "confidence": 0.9},
                {"decision_id": "D2", "recommended_action": "confirm_default_or_nd",
                 "default_confirmation": True, "confidence": 0.8},
            ]}
            return "Here is my analysis:\n```json\n" + json.dumps(payload) + "\n```\nDone."

        res = adv.run_target_advisor(_DECISIONS[:2], _COVERAGE[:2], llm_callable=llm)
        self.assertEqual(res["advice_status"], adv.ADVISED)
        statuses = {r["decision_id"]: r["llm_advice_status"] for r in res["recommendations"]}
        self.assertEqual(statuses["D1"], adv.ADVISED)
        self.assertEqual(statuses["D2"], adv.ADVISED)

    def test_parses_bare_top_level_array(self):
        def llm(_prompt):
            return json.dumps([
                {"decision_id": "D1", "recommended_action": "confirm_selected", "confidence": 0.7},
                {"decision_id": "D2", "recommended_action": "mark_not_applicable", "confidence": 0.6},
            ])

        res = adv.run_target_advisor(_DECISIONS[:2], _COVERAGE[:2], llm_callable=llm)
        self.assertEqual(res["advice_status"], adv.ADVISED)
        self.assertTrue(all(r["llm_advice_status"] == adv.ADVISED for r in res["recommendations"]))

    def test_parse_failure_emits_clear_diagnostics(self):
        def llm(_prompt):
            return "I'm sorry, I can't do that."

        res = adv.run_target_advisor(_DECISIONS[:2], _COVERAGE[:2], llm_callable=llm)
        self.assertEqual(res["advice_status"], adv.PARSE_FAILED)
        self.assertEqual(res["parse_status"], "parse_failed")
        self.assertTrue(res["parse_error"])
        self.assertEqual(res["usage"]["parse_status"], "parse_failed")
        # Every row carries an honest, non-"missing field" rationale.
        for r in res["recommendations"]:
            self.assertEqual(r["llm_advice_status"], adv.PARSE_FAILED)
            self.assertIn("could not be parsed", r["llm_rationale"])
            self.assertIn("may still be present", r["llm_rationale"])


# --------------------------------------------------------------------------- #
# 2. accept-target-advice surfaces parse diagnostics (no silent skip)
# --------------------------------------------------------------------------- #
class TestAcceptSurfacesParseDiagnostics(unittest.TestCase):
    def _project_with_parse_failure(self, pdir: Path):
        _write_template(pdir)

        def llm(_prompt):
            return "no json here"

        res = adv.run_target_advisor(_DECISIONS, _COVERAGE, llm_callable=llm)
        adv.write_advisor_artifacts(res, pdir)

    def test_accept_reports_parse_diagnostics_and_does_not_invent_approval(self):
        with tempfile.TemporaryDirectory() as d:
            pdir = Path(d)
            self._project_with_parse_failure(pdir)
            summary = ata.accept_target_advice(pdir, approved_by="ci")
            self.assertEqual(summary["approved"], 0)
            self.assertGreater(summary["parse_failed_skips"], 0)
            diag = summary["parse_diagnostics"]
            self.assertEqual(diag["parse_status"], "parse_failed")
            self.assertTrue(diag["raw_response_path"])
            self.assertTrue(Path(diag["raw_response_path"]).exists())
            text = ata.format_summary(summary)
            self.assertIn("could not be parsed", text)
            self.assertIn("may still be present", text)
            # The skip wording must not imply the field is missing from MI output.
            self.assertNotIn("missing from", text.lower())


# --------------------------------------------------------------------------- #
# 3. Deterministic non-blocking approval (real approved file)
# --------------------------------------------------------------------------- #
class TestNonBlockingApproval(unittest.TestCase):
    def test_non_blocking_decisions_approved_blocking_not(self):
        with tempfile.TemporaryDirectory() as d:
            pdir = Path(d)
            _write_template(pdir)
            summary = nba.approve_non_blocking_decisions(pdir, approved_by="ci")
            self.assertEqual(summary["approved"], 3)
            self.assertEqual(sorted(summary["approved_ids"]), ["D1", "D2", "D3"])
            self.assertEqual([s["decision_id"] for s in summary["skipped_blocking"]], ["D4"])

            out = yaml.safe_load(Path(summary["out_path"]).read_text())
            by_id = {x["decision_id"]: x for x in out["decisions"]}
            # Non-blocking confirmations are approved with a concrete action + audit.
            self.assertEqual(by_id["D1"]["status"], "approved")
            self.assertEqual(by_id["D1"]["selected_action"], "confirm_selected")
            self.assertTrue(by_id["D1"]["auto_approved"])
            self.assertEqual(by_id["D2"]["selected_action"], "confirm_default_or_nd")
            self.assertEqual(by_id["D3"]["selected_action"], "mark_not_applicable")
            # Blocking decision is never auto-approved.
            self.assertEqual(by_id["D4"]["status"], "pending")
            self.assertIsNone(by_id["D4"]["selected_action"])
            # Audit trail records what was approved and why.
            audit = out["non_blocking_approval"]
            self.assertEqual(audit["approved"], 3)
            self.assertEqual([s["decision_id"] for s in audit["skipped_blocking"]], ["D4"])

    def test_approved_file_applies_deterministically(self):
        with tempfile.TemporaryDirectory() as d:
            pdir = Path(d)
            _write_template(pdir)
            summary = nba.approve_non_blocking_decisions(pdir, approved_by="ci")
            out = yaml.safe_load(Path(summary["out_path"]).read_text())
            cov_rows = [{"target_field": r["target_field"]} for r in _DECISIONS]
            _cov, _remaining, log = tfd.apply_decisions(
                cov_rows, _DECISIONS, tfd.approved_decisions(out))
            applied = {e["decision_id"]: e["application_status"] for e in log}
            self.assertEqual(applied, {"D1": "applied", "D2": "applied", "D3": "applied"})

    def test_copied_pending_template_is_not_an_approval(self):
        tpl = tfd.build_decision_template(_DECISIONS, mode="regulatory_mi")
        # A straight copy of the pending template — every decision still pending.
        self.assertFalse(tfd.is_real_approval(tpl))
        self.assertEqual(tfd.approved_decisions(tpl), [])
        # After deterministic approval it IS a real approval.
        with tempfile.TemporaryDirectory() as d:
            pdir = Path(d)
            _write_template(pdir)
            summary = nba.approve_non_blocking_decisions(pdir)
            out = yaml.safe_load(Path(summary["out_path"]).read_text())
            self.assertTrue(tfd.is_real_approval(out))


# --------------------------------------------------------------------------- #
# 4. Separated readiness concepts
# --------------------------------------------------------------------------- #
def _coverage(missing_collateral: bool = False):
    status = dc.MISSING if missing_collateral else dc.COVERED
    return [SimpleNamespace(domain=dc.COLLATERAL, status=status, blocking=True)]


_TAPE_OK = {
    "central_lender_tape_created": True, "loan_count": 73,
    "central_pipeline_tape_created": True, "pipeline_count": 10,
    "conflict_count": 0, "lender_summary": {"gap_count": 0},
}


class TestSeparatedReadiness(unittest.TestCase):
    def test_mi_runtime_ready_while_governance_pending(self):
        with tempfile.TemporaryDirectory() as d:
            pdir = Path(d)
            _write_template(pdir, rows=_DECISIONS[:3])  # non-blocking pending only
            b = rd.compute_readiness_breakdown(pdir, _TAPE_OK, _coverage(),
                                               "regulatory_mi", False)
            self.assertTrue(b["mi_runtime_readiness"]["ready"])
            self.assertEqual(b["mi_runtime_readiness"]["status"], rd.READY)
            # Governance is not ready, but only because of non-blocking confirmations.
            gov = b["onboarding_governance_readiness"]
            self.assertFalse(gov["ready"])
            self.assertEqual(gov["status"], rd.PENDING_CONFIRMATIONS)
            self.assertEqual(gov["pending_blocking"], 0)
            self.assertFalse(b["xml_delivery_readiness"]["ready"])

    def test_blocking_decision_blocks_governance_not_mi(self):
        with tempfile.TemporaryDirectory() as d:
            pdir = Path(d)
            _write_template(pdir, rows=_DECISIONS)  # includes blocking D4
            b = rd.compute_readiness_breakdown(pdir, _TAPE_OK, _coverage(),
                                               "regulatory_mi", False)
            self.assertEqual(b["onboarding_governance_readiness"]["status"], rd.BLOCKED)
            self.assertGreaterEqual(b["onboarding_governance_readiness"]["pending_blocking"], 1)
            # MI runtime is unaffected by the blocking governance decision.
            self.assertTrue(b["mi_runtime_readiness"]["ready"])

    def test_governance_ready_after_all_approved(self):
        with tempfile.TemporaryDirectory() as d:
            pdir = Path(d)
            _write_template(pdir, rows=_DECISIONS[:3])
            nba.approve_non_blocking_decisions(pdir)  # writes approved file
            b = rd.compute_readiness_breakdown(pdir, _TAPE_OK, _coverage(),
                                               "regulatory_mi", False)
            self.assertEqual(b["onboarding_governance_readiness"]["status"], rd.READY)
            self.assertTrue(b["onboarding_governance_readiness"]["real_approval"])

    def test_empty_tape_blocks_mi_runtime(self):
        with tempfile.TemporaryDirectory() as d:
            pdir = Path(d)
            _write_template(pdir, rows=_DECISIONS[:3])
            tape = dict(_TAPE_OK, central_lender_tape_created=False, loan_count=0)
            b = rd.compute_readiness_breakdown(pdir, tape, _coverage(),
                                               "regulatory_mi", False)
            self.assertFalse(b["mi_runtime_readiness"]["ready"])
            self.assertTrue(b["mi_runtime_readiness"]["reasons"])

    def test_conflicts_block_governance_not_mi(self):
        with tempfile.TemporaryDirectory() as d:
            pdir = Path(d)
            _write_template(pdir, rows=[])  # no Gate 4 decisions
            tape = dict(_TAPE_OK, conflict_count=2)
            b = rd.compute_readiness_breakdown(pdir, tape, _coverage(),
                                               "regulatory_mi", False)
            self.assertEqual(b["onboarding_governance_readiness"]["status"], rd.BLOCKED)
            self.assertEqual(b["onboarding_governance_readiness"]["unresolved_conflicts"], 2)
            self.assertTrue(b["mi_runtime_readiness"]["ready"])


# --------------------------------------------------------------------------- #
# 5. Funded / pipeline MI calculations are unchanged by this flow
# --------------------------------------------------------------------------- #
class TestMiCalculationsUnchanged(unittest.TestCase):
    def test_pipeline_snapshot_independent_of_gate4_flow(self):
        # The Gate 4 approval / readiness flow must not touch MI computation. Verify
        # the pipeline snapshot still resolves the latest extract from the committed
        # fixture pack regardless of any Gate 4 state.
        from mi_agent_api import pipeline_contract as pc
        fixture = _REPO_ROOT / "tests" / "fixtures" / "client_001_mi_pack"
        scope = pc.resolve_pipeline_source(fixture, "client_001", "mi_2025_11")
        self.assertEqual(scope["pipeline_as_of_date"], "2025-12-01")


if __name__ == "__main__":
    unittest.main(verbosity=2)
