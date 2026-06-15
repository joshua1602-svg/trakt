#!/usr/bin/env python3
"""tests/test_onboarding_target_first_decisions.py

Gate 4 decision capture + deterministic re-application for target-first
onboarding (34_target_first_decisions.yaml / 35_target_first_decision_application_log.*).
"""

from __future__ import annotations

import sys
import tempfile
import unittest
import warnings
from pathlib import Path

import yaml

_REPO_ROOT = Path(__file__).resolve().parent.parent
for _p in (str(_REPO_ROOT), str(_REPO_ROOT / "tests")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from engine.onboarding_agent import target_first_decisions as tfd

PACK = _REPO_ROOT / "synthetic_demo" / "input"
REGISTRY = _REPO_ROOT / "config" / "system" / "fields_registry.yaml"
ALIASES = _REPO_ROOT / "config" / "system"


def _cov(target_field, **kw):
    row = {
        "target_field": target_field, "target_domain": "core",
        "required_status": "required", "applicability_status": "applicable",
        "coverage_status": "source_mapped_with_alternatives", "coverage_basis": "name_synonym_match",
        "selected_source_file": "a.csv", "selected_source_column": "Col A",
        "selected_source_confidence": 0.7, "alternative_source_candidates": "a.csv::::Col B (0.65)",
        "value_compatibility_status": "compatible", "derivation_rule": "", "default_rule": "",
        "nd_rule_applied": "", "configured_value_source": "", "requires_user_decision": True,
        "blocking": False, "decision_reason": "two candidates", "operator_question": "which?",
    }
    row.update(kw)
    return row


def _dec(decision_id, target_field, decision_type, **kw):
    row = {
        "decision_id": decision_id, "decision_type": decision_type, "priority": "medium",
        "mode": "mi_only", "target_contract_id": "mi_semantics_field_registry",
        "target_field": target_field, "source_file": "a.csv", "source_column": "Col A",
        "issue": "issue", "recommendation": "rec", "options": ["confirm_selected"],
        "blocking": False, "operator_question": "q", "evidence_summary": "ev",
    }
    row.update(kw)
    return row


def _approved(decision_id, target_field, action, **kw):
    d = {"decision_id": decision_id, "target_field": target_field,
         "selected_action": action, "status": "approved"}
    d.update(kw)
    return d


# --------------------------------------------------------------------------- #
# 1. Template generation
# --------------------------------------------------------------------------- #
class TestTemplateGeneration(unittest.TestCase):
    def test_one_entry_per_decision_all_pending(self):
        rows = [_dec("DQ-1", "f1", "source_priority_confirmation"),
                _dec("DQ-2", "f2", "missing_required_target", blocking=True)]
        tpl = tfd.build_decision_template(rows, "mi_only", client_id="c", run_id="r")
        self.assertEqual(len(tpl["decisions"]), 2)
        self.assertTrue(all(d["status"] == "pending" for d in tpl["decisions"]))
        self.assertEqual(tpl["version"], 1)
        self.assertIn("confirm_selected", tpl["supported_actions"])
        # Source-priority pre-populates the deterministic recommended source.
        sp = tpl["decisions"][0]
        self.assertEqual(sp["selected_source_column"], "Col A")
        self.assertIsNone(sp["selected_action"])

    def test_context_echoed_from_28c(self):
        rows = [_dec("DQ-1", "f1", "source_priority_confirmation", priority="high")]
        d = tfd.build_decision_template(rows, "mi_only")["decisions"][0]
        for k in ("decision_id", "decision_type", "priority", "target_field",
                  "source_file", "source_column", "issue", "recommendation",
                  "options", "blocking", "operator_question", "evidence_summary"):
            self.assertIn(k, d)


# --------------------------------------------------------------------------- #
# 2-6. Apply individual actions
# --------------------------------------------------------------------------- #
class TestApplyActions(unittest.TestCase):
    def test_confirm_selected_resolves_28c_keeps_deterministic_source(self):
        cov = [_cov("f1")]
        dec = [_dec("DQ-1", "f1", "source_priority_confirmation")]
        cov, rem, log = tfd.apply_decisions(
            cov, dec, [_approved("DQ-1", "f1", "confirm_selected")])
        self.assertEqual(rem, [])  # 28c decision removed
        self.assertEqual(cov[0]["selected_source_column"], "Col A")  # deterministic kept
        self.assertFalse(cov[0]["requires_user_decision"])
        self.assertEqual(log[0]["application_status"], "applied")
        self.assertTrue(log[0]["applied_to_28a"] and log[0]["applied_to_28c"])

    def test_choose_alternative_updates_source(self):
        cov = [_cov("f1")]
        dec = [_dec("DQ-1", "f1", "source_priority_confirmation")]
        cov, rem, log = tfd.apply_decisions(cov, dec, [_approved(
            "DQ-1", "f1", "choose_alternative",
            selected_source_file="b.csv", selected_source_column="Col B")])
        self.assertEqual(cov[0]["selected_source_file"], "b.csv")
        self.assertEqual(cov[0]["selected_source_column"], "Col B")
        self.assertEqual(cov[0]["coverage_basis"], "operator_override_alternative_source")
        self.assertEqual(rem, [])
        self.assertEqual(log[0]["application_status"], "applied")

    def test_configure_static_value(self):
        cov = [_cov("f1", coverage_status="needs_confirmation")]
        dec = [_dec("DQ-1", "f1", "config_value_required")]
        cov, rem, log = tfd.apply_decisions(cov, dec, [_approved(
            "DQ-1", "f1", "configure_static_value", configured_value="UK")])
        self.assertEqual(cov[0]["coverage_status"], "configured_static")
        self.assertFalse(cov[0]["requires_user_decision"])
        self.assertEqual(rem, [])
        self.assertEqual(log[0]["application_status"], "applied")

    def test_confirm_default_or_nd(self):
        cov = [_cov("f1", coverage_status="defaulted_ND", default_rule="ND1")]
        dec = [_dec("DQ-1", "f1", "nd_default_confirmation")]
        cov, rem, log = tfd.apply_decisions(
            cov, dec, [_approved("DQ-1", "f1", "confirm_default_or_nd")])
        self.assertFalse(cov[0]["requires_user_decision"])
        self.assertIn("confirmed", cov[0]["default_rule"])
        self.assertEqual(rem, [])
        self.assertTrue(log[0]["default_confirmed"])

    def test_mark_not_applicable(self):
        cov = [_cov("f1")]
        dec = [_dec("DQ-1", "f1", "source_priority_confirmation")]
        cov, rem, log = tfd.apply_decisions(
            cov, dec, [_approved("DQ-1", "f1", "mark_not_applicable")])
        self.assertEqual(cov[0]["applicability_status"], "not_applicable")
        self.assertEqual(cov[0]["coverage_status"], "not_applicable")
        self.assertEqual(rem, [])
        self.assertTrue(log[0]["not_applicable_confirmed"])

    def test_provide_source_mapping_resolves_blocker(self):
        cov = [_cov("f1", coverage_status="missing_required", blocking=True,
                    selected_source_file="", selected_source_column="")]
        dec = [_dec("DQ-1", "f1", "missing_required_target", blocking=True)]
        cov, rem, log = tfd.apply_decisions(cov, dec, [_approved(
            "DQ-1", "f1", "provide_source_mapping",
            selected_source_file="s.csv", selected_source_column="Bal")])
        self.assertEqual(cov[0]["coverage_status"], "source_mapped")
        self.assertEqual(cov[0]["selected_source_column"], "Bal")
        self.assertFalse(cov[0]["blocking"])
        self.assertEqual(rem, [])


# --------------------------------------------------------------------------- #
# 7. Pending / deferred handling
# --------------------------------------------------------------------------- #
class TestPendingDeferred(unittest.TestCase):
    def test_only_approved_status_is_selected(self):
        doc = {"decisions": [
            {"decision_id": "DQ-1", "status": "pending", "selected_action": "confirm_selected"},
            {"decision_id": "DQ-2", "status": "approved", "selected_action": "confirm_selected"},
            {"decision_id": "DQ-3", "status": "rejected", "selected_action": "confirm_selected"},
            {"decision_id": "DQ-4", "status": "draft", "selected_action": "confirm_selected"},
        ]}
        appr = tfd.approved_decisions(doc)
        self.assertEqual([d["decision_id"] for d in appr], ["DQ-2"])

    def test_defer_keeps_decision_and_logs(self):
        cov = [_cov("f1")]
        dec = [_dec("DQ-1", "f1", "source_priority_confirmation")]
        cov, rem, log = tfd.apply_decisions(
            cov, dec, [_approved("DQ-1", "f1", "defer")])
        self.assertEqual(len(rem), 1)  # kept in 28c
        self.assertEqual(log[0]["application_status"], "ignored_deferred")

    def test_merge_or_reconcile_requires_review(self):
        cov = [_cov("f1")]
        dec = [_dec("DQ-1", "f1", "conflicting_source_candidates")]
        cov, rem, log = tfd.apply_decisions(
            cov, dec, [_approved("DQ-1", "f1", "merge_or_reconcile")])
        self.assertEqual(len(rem), 1)
        self.assertEqual(log[0]["application_status"], "requires_operator_review")


# --------------------------------------------------------------------------- #
# 8. Invalid decisions never crash
# --------------------------------------------------------------------------- #
class TestInvalidDecisions(unittest.TestCase):
    def test_unknown_decision_id(self):
        cov = [_cov("f1")]
        dec = [_dec("DQ-1", "f1", "source_priority_confirmation")]
        cov, rem, log = tfd.apply_decisions(
            cov, dec, [_approved("DQ-NOPE", "f1", "confirm_selected")])
        self.assertEqual(len(rem), 1)  # nothing resolved
        self.assertEqual(log[0]["application_status"], "decision_id_not_found")

    def test_unknown_target_field(self):
        cov = [_cov("f1")]
        dec = [_dec("DQ-1", "ghost", "source_priority_confirmation")]
        cov, rem, log = tfd.apply_decisions(
            cov, dec, [_approved("DQ-1", "ghost", "confirm_selected")])
        self.assertEqual(log[0]["application_status"], "target_field_not_found")

    def test_unknown_action(self):
        cov = [_cov("f1")]
        dec = [_dec("DQ-1", "f1", "source_priority_confirmation")]
        cov, rem, log = tfd.apply_decisions(
            cov, dec, [_approved("DQ-1", "f1", "do_something_weird")])
        self.assertEqual(log[0]["application_status"], "invalid")
        self.assertEqual(len(rem), 1)


# --------------------------------------------------------------------------- #
# 9. CLI / end-to-end loop
# --------------------------------------------------------------------------- #
class TestEndToEndLoop(unittest.TestCase):
    def test_external_decisions_applied_and_28c_shrinks(self):
        from engine.onboarding_agent.onboarding_orchestrator import run_onboarding
        import json
        warnings.simplefilter("ignore")
        # Run 1 — generate the template.
        out1 = Path(tempfile.mkdtemp(prefix="dec1_"))
        run_onboarding(input_dir=str(PACK), client_name="C", output_dir=str(out1),
                       registry_path=str(REGISTRY), aliases_dir=str(ALIASES),
                       mode="mi_only", client_id="client_001", run_id="r1",
                       enable_mapping_review=True)
        tpl_path = out1 / "34_target_first_decisions.yaml"
        self.assertTrue(tpl_path.exists())
        tpl = yaml.safe_load(tpl_path.read_text())
        dec1 = json.loads((out1 / "28c_human_decision_queue.json").read_text())
        self.assertEqual(len(tpl["decisions"]), len(dec1["rows"]))
        self.assertTrue(tpl["decisions"], "expected at least one decision to approve")
        self.assertTrue(all(d["status"] == "pending" for d in tpl["decisions"]))

        # Approve every decision as mark_not_applicable (resolves cleanly).
        for d in tpl["decisions"]:
            d["status"] = "approved"
            d["selected_action"] = "mark_not_applicable"
            d["approved_by"] = "test_operator"
        approved_path = out1 / "34_target_first_decisions_APPROVED.yaml"
        approved_path.write_text(yaml.safe_dump(tpl, sort_keys=False), encoding="utf-8")

        # Run 2 — supply the approved decisions via the CLI-style path argument.
        out2 = Path(tempfile.mkdtemp(prefix="dec2_"))
        run_onboarding(input_dir=str(PACK), client_name="C", output_dir=str(out2),
                       registry_path=str(REGISTRY), aliases_dir=str(ALIASES),
                       mode="mi_only", client_id="client_001", run_id="r2",
                       enable_mapping_review=True,
                       target_first_decisions_path=str(approved_path))
        dec2 = json.loads((out2 / "28c_human_decision_queue.json").read_text())
        log = json.loads((out2 / "35_target_first_decision_application_log.json").read_text())
        # The application log exists and records applied decisions.
        self.assertTrue((out2 / "35_target_first_decision_application_log.csv").exists())
        self.assertEqual(log["summary"]["applied"], len(tpl["decisions"]))
        # 28c shrinks by the number of resolved decisions.
        self.assertLess(len(dec2["rows"]), len(dec1["rows"]))
        # The review pack surfaces the application summary.
        html = (out2 / "08_onboarding_review_pack.html").read_text()
        self.assertIn("Applied target-first decisions", html)


if __name__ == "__main__":
    unittest.main()
