#!/usr/bin/env python3
"""tests/test_onboarding_accept_target_advice.py

`accept-target-advice`: accept the target-first LLM advisor's recommendations
(36) into an approved 34 decision file, safely.

Proves:
  * valid LLM source-mapping advice -> approved source-mapping decisions;
  * mark_not_applicable advice -> approved not_applicable decisions;
  * configure_static_value advice -> approved configured-value decisions;
  * invalid_response is not applied;
  * requires_operator_review is not auto-approved;
  * decisions without usable advice stay pending;
  * a recommended source column outside the 28a candidates is skipped;
  * the generated approved file is consumed by apply_decisions on rerun;
  * 28a/28c are never mutated by acceptance.
"""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import yaml

from engine.onboarding_agent import accept_target_advice as ata
from engine.onboarding_agent import target_first_decisions as tfd


def _decision(dec_id, field, **extra):
    d = {
        "decision_id": dec_id, "decision_type": "missing_required_target",
        "target_field": field, "status": "pending", "selected_action": None,
        "selected_source_file": None, "selected_source_column": None,
        "configured_value": None, "default_confirmed": None,
        "not_applicable_confirmed": None, "operator_note": None,
        "approved_by": None, "approved_at": None, "blocking": True,
    }
    d.update(extra)
    return d


def _rec(dec_id, field, action, status="advised", **extra):
    r = {
        "decision_id": dec_id, "target_field": field,
        "llm_recommended_action": action, "llm_advice_status": status,
        "llm_recommended_source_file": "", "llm_recommended_source_column": "",
        "llm_recommended_configured_value": "", "llm_recommended_default_confirmation": "",
        "llm_recommended_not_applicable": "", "llm_confidence": 0.9,
        "llm_rationale": f"advice for {field}",
    }
    r.update(extra)
    return r


def _write_project(decisions, recs, coverage_rows=None):
    d = Path(tempfile.mkdtemp(prefix="ata_"))
    (d / "34_target_first_decisions.yaml").write_text(
        yaml.safe_dump({"version": 1, "mode": "mi_only", "decisions": decisions},
                       sort_keys=False), encoding="utf-8")
    (d / "36_target_first_llm_recommendations.json").write_text(
        json.dumps({"summary": {}, "rows": recs}, indent=2), encoding="utf-8")
    (d / "28a_target_coverage_matrix.json").write_text(
        json.dumps({"rows": coverage_rows or []}, indent=2), encoding="utf-8")
    return d


def _approved_decision(out_path, dec_id):
    doc = yaml.safe_load(Path(out_path).read_text())
    for x in doc["decisions"]:
        if x["decision_id"] == dec_id:
            return x
    return None


class TestAcceptTargetAdvice(unittest.TestCase):
    def test_valid_source_mapping_approved(self):
        decs = [_decision("DQ-1", "loan_identifier")]
        recs = [_rec("DQ-1", "loan_identifier", "provide_source_mapping",
                     llm_recommended_source_file="loan.csv",
                     llm_recommended_source_column="Loan Ref")]
        cov = [{"target_field": "loan_identifier", "selected_source_column": "",
                "alternative_source_candidates": "loan.csv::Sheet1::Loan Ref (0.8)"}]
        d = _write_project(decs, recs, cov)
        s = ata.accept_target_advice(d, approved_by="Joshua")
        self.assertEqual(s["approved"], 1)
        a = _approved_decision(s["out_path"], "DQ-1")
        self.assertEqual(a["status"], "approved")
        self.assertEqual(a["selected_action"], "provide_source_mapping")
        self.assertEqual(a["selected_source_column"], "Loan Ref")
        self.assertEqual(a["approved_by"], "Joshua")
        self.assertTrue(a["approved_at"])

    def test_mark_not_applicable_approved(self):
        decs = [_decision("DQ-2", "maturity_date")]
        recs = [_rec("DQ-2", "maturity_date", "mark_not_applicable",
                     llm_recommended_not_applicable=True)]
        d = _write_project(decs, recs)
        s = ata.accept_target_advice(d)
        a = _approved_decision(s["out_path"], "DQ-2")
        self.assertEqual(a["status"], "approved")
        self.assertEqual(a["selected_action"], "mark_not_applicable")
        self.assertTrue(a["not_applicable_confirmed"])

    def test_configure_static_value_approved(self):
        decs = [_decision("DQ-3", "exposure_currency_denomination")]
        recs = [_rec("DQ-3", "exposure_currency_denomination", "configure_static_value",
                     llm_recommended_configured_value="GBP")]
        d = _write_project(decs, recs)
        s = ata.accept_target_advice(d)
        a = _approved_decision(s["out_path"], "DQ-3")
        self.assertEqual(a["status"], "approved")
        self.assertEqual(a["selected_action"], "configure_static_value")
        self.assertEqual(a["configured_value"], "GBP")

    def test_configure_static_value_missing_value_skipped(self):
        decs = [_decision("DQ-3b", "currency")]
        recs = [_rec("DQ-3b", "currency", "configure_static_value",
                     llm_recommended_configured_value="")]
        d = _write_project(decs, recs)
        s = ata.accept_target_advice(d)
        self.assertEqual(s["approved"], 0)
        self.assertEqual(_approved_decision(s["out_path"], "DQ-3b")["status"], "pending")

    def test_invalid_response_not_applied(self):
        decs = [_decision("DQ-4", "origination_date")]
        recs = [_rec("DQ-4", "origination_date", "mark_not_applicable",
                     status="invalid_response")]
        d = _write_project(decs, recs)
        s = ata.accept_target_advice(d)
        self.assertEqual(s["approved"], 0)
        self.assertEqual(_approved_decision(s["out_path"], "DQ-4")["status"], "pending")
        self.assertTrue(any("invalid_response" in x["reason"] for x in s["skipped"]))

    def test_requires_operator_review_not_auto_approved(self):
        decs = [_decision("DQ-5", "current_principal_balance")]
        recs = [_rec("DQ-5", "current_principal_balance", "requires_operator_review",
                     status="advised")]
        d = _write_project(decs, recs)
        s = ata.accept_target_advice(d)
        self.assertEqual(s["approved"], 0)
        a = _approved_decision(s["out_path"], "DQ-5")
        self.assertEqual(a["status"], "pending")
        self.assertTrue(any("requires_operator_review" in x["reason"] for x in s["skipped"]))

    def test_pending_decisions_remain_pending(self):
        decs = [_decision("DQ-6", "current_interest_rate")]  # no matching rec
        d = _write_project(decs, recs=[])
        s = ata.accept_target_advice(d)
        self.assertEqual(s["approved"], 0)
        self.assertEqual(s["pending"], 1)
        self.assertEqual(_approved_decision(s["out_path"], "DQ-6")["status"], "pending")
        self.assertTrue(any(x["reason"] == "no_recommendation" for x in s["skipped"]))

    def test_source_not_in_candidates_skipped(self):
        decs = [_decision("DQ-7", "loan_identifier")]
        recs = [_rec("DQ-7", "loan_identifier", "provide_source_mapping",
                     llm_recommended_source_column="Hallucinated Col")]
        cov = [{"target_field": "loan_identifier", "selected_source_column": "",
                "alternative_source_candidates": "loan.csv::S1::Loan Ref (0.8)"}]
        d = _write_project(decs, recs, cov)
        s = ata.accept_target_advice(d)
        self.assertEqual(s["approved"], 0)
        self.assertTrue(any("source_not_in_candidates" in x["reason"] for x in s["skipped"]))

    def test_allow_status_flag_opts_in(self):
        decs = [_decision("DQ-8", "maturity_date")]
        recs = [_rec("DQ-8", "maturity_date", "mark_not_applicable",
                     status="requires_operator_review")]
        d = _write_project(decs, recs)
        # Default: skipped.
        self.assertEqual(ata.accept_target_advice(d)["approved"], 0)
        # Explicitly allowed: approved.
        s = ata.accept_target_advice(d, allow_statuses=["requires_operator_review"])
        self.assertEqual(s["approved"], 1)

    def test_acceptance_does_not_mutate_28a_28c(self):
        decs = [_decision("DQ-9", "maturity_date")]
        recs = [_rec("DQ-9", "maturity_date", "mark_not_applicable")]
        d = _write_project(decs, recs)
        cov_before = (d / "28a_target_coverage_matrix.json").read_text()
        dec_before = (d / "34_target_first_decisions.yaml").read_text()
        ata.accept_target_advice(d)
        # The original 34 template and 28a are untouched; only the *approved* file is new.
        self.assertEqual((d / "28a_target_coverage_matrix.json").read_text(), cov_before)
        self.assertEqual((d / "34_target_first_decisions.yaml").read_text(), dec_before)
        self.assertTrue((d / "34_target_first_decisions_approved.yaml").exists())


class TestRerunConsumesApprovedFile(unittest.TestCase):
    """The generated approved file is applied by apply_decisions on rerun."""

    def test_apply_decisions_accepts_generated_file(self):
        decs = [
            _decision("DQ-1", "maturity_date"),
            _decision("DQ-2", "amortisation_type"),
        ]
        recs = [
            _rec("DQ-1", "maturity_date", "mark_not_applicable"),
            _rec("DQ-2", "amortisation_type", "configure_static_value",
                 llm_recommended_configured_value="OTHR"),
        ]
        d = _write_project(decs, recs)
        s = ata.accept_target_advice(d, approved_by="Joshua")
        self.assertEqual(s["approved"], 2)

        # Simulate the rerun: load approved -> apply to a coverage matrix + queue.
        doc = tfd.load_decisions(s["out_path"])
        approved = tfd.approved_decisions(doc)
        self.assertEqual(len(approved), 2)
        coverage_rows = [
            {"target_field": "maturity_date", "coverage_status": "missing_required",
             "blocking": True, "requires_user_decision": True},
            {"target_field": "amortisation_type", "coverage_status": "missing_required",
             "blocking": True, "requires_user_decision": True},
        ]
        decision_rows = [
            {"decision_id": "DQ-1", "target_field": "maturity_date"},
            {"decision_id": "DQ-2", "target_field": "amortisation_type"},
        ]
        cov, remaining, log = tfd.apply_decisions(coverage_rows, decision_rows, approved)
        applied = [e for e in log if e["application_status"] == tfd.APPLIED]
        self.assertEqual(len(applied), 2)
        self.assertEqual(remaining, [])  # both resolved -> dropped from 28c
        by_field = {r["target_field"]: r for r in cov}
        self.assertEqual(by_field["maturity_date"]["coverage_status"], "not_applicable")
        self.assertFalse(by_field["maturity_date"]["blocking"])
        self.assertEqual(by_field["amortisation_type"]["coverage_status"], "configured_static")
        self.assertFalse(by_field["amortisation_type"]["blocking"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
