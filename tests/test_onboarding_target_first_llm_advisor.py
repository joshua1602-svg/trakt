#!/usr/bin/env python3
"""tests/test_onboarding_target_first_llm_advisor.py

Target-contract-first LLM ADVISOR (36_target_first_llm_*). Advisory only — it
operates on the 28c Gate 4 decisions + 28a/28b evidence and never mutates the
deterministic target-first state.
"""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
import warnings
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
for _p in (str(_REPO_ROOT), str(_REPO_ROOT / "tests")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from engine.onboarding_agent import target_first_llm_advisor as adv
from engine.onboarding_agent.llm_assisted_mapping import run_llm_assisted_mapping

ERE = str(_REPO_ROOT / "synthetic_demo" / "input" / "SYNTHETIC_ERE_Portfolio_012026.csv")


def _cov(target_field, **kw):
    row = {"target_field": target_field, "target_domain": "core",
           "required_status": "required", "coverage_status": "source_mapped_with_alternatives",
           "selected_source_file": "a.csv", "selected_source_column": "Col A",
           "alternative_source_candidates": "a.csv::::Col B (0.65)",
           "value_compatibility_status": "compatible"}
    row.update(kw)
    return row


def _dec(decision_id, target_field, decision_type, **kw):
    row = {"decision_id": decision_id, "decision_type": decision_type, "priority": "medium",
           "mode": "mi_only", "target_contract_id": "mi_semantics_field_registry",
           "target_field": target_field, "source_file": "a.csv", "source_column": "Col A",
           "issue": "issue", "recommendation": "confirm Col A", "options": ["confirm_selected"],
           "blocking": False, "operator_question": "which source?", "evidence_summary": "ev"}
    row.update(kw)
    return row


def _echo_llm(action="confirm_selected", **extra):
    def _call(prompt):
        decs = json.loads(prompt.split("DECISIONS = ", 1)[1])
        recs = []
        for d in decs:
            r = {"decision_id": d["decision_id"], "recommended_action": action,
                 "confidence": 0.8, "rationale": "advised", "requires_human_confirmation": True}
            r.update(extra)
            recs.append(r)
        return json.dumps({"recommendations": recs})
    return _call


# --------------------------------------------------------------------------- #
# 1. CLI flag
# --------------------------------------------------------------------------- #
class TestCliFlag(unittest.TestCase):
    def test_flag_accepted(self):
        from engine.onboarding_agent.cli import build_parser
        a = build_parser().parse_args(
            ["--input-dir", "x", "--client-name", "c", "--enable-llm-target-advisor"])
        self.assertTrue(a.enable_llm_target_advisor)

    def test_flag_independent_of_mapping_review(self):
        a = __import__("engine.onboarding_agent.cli", fromlist=["build_parser"]).build_parser()
        ns = a.parse_args(["--input-dir", "x", "--client-name", "c",
                           "--enable-llm-target-advisor"])
        self.assertFalse(ns.enable_llm_mapping_review)
        self.assertTrue(ns.enable_llm_target_advisor)


# --------------------------------------------------------------------------- #
# 2. No decisions
# --------------------------------------------------------------------------- #
class TestNoDecisions(unittest.TestCase):
    def test_skipped_no_decisions(self):
        res = adv.run_target_advisor([], [], llm_callable=_echo_llm())
        self.assertEqual(res["advice_status"], adv.SKIPPED_NO_DECISIONS)
        out = Path(tempfile.mkdtemp())
        adv.write_advisor_artifacts(res, out)
        u = json.loads((out / "36_target_first_llm_usage_summary.json").read_text())
        self.assertTrue(u["llm_target_advisor_enabled"])
        self.assertEqual(u["decision_rows_available"], 0)


# --------------------------------------------------------------------------- #
# 3. Advisory only — never mutates 28a/28c
# --------------------------------------------------------------------------- #
class TestAdvisoryOnly(unittest.TestCase):
    def test_does_not_mutate_coverage_or_decisions(self):
        cov = [_cov("f1")]
        dec = [_dec("DQ-1", "f1", "source_priority_confirmation")]
        import copy
        cov_before, dec_before = copy.deepcopy(cov), copy.deepcopy(dec)
        adv.run_target_advisor(dec, cov, llm_callable=_echo_llm("choose_alternative",
                               recommended_source_column="Col B"))
        self.assertEqual(cov, cov_before)  # 28a untouched
        self.assertEqual(dec, dec_before)  # 28c untouched (no rows removed)


# --------------------------------------------------------------------------- #
# 4. Recommendation schema
# --------------------------------------------------------------------------- #
class TestRecommendationSchema(unittest.TestCase):
    def test_one_row_per_decision(self):
        dec = [_dec("DQ-1", "f1", "source_priority_confirmation"),
               _dec("DQ-2", "f2", "source_priority_confirmation")]
        cov = [_cov("f1"), _cov("f2")]
        res = adv.run_target_advisor(dec, cov, llm_callable=_echo_llm())
        self.assertEqual(len(res["recommendations"]), 2)
        self.assertEqual(res["usage"]["decision_rows_advised"], 2)
        for r in res["recommendations"]:
            for col in adv._RECOMMENDATION_COLUMNS:
                self.assertIn(col, r)

    def test_invalid_json_parse_failed_writes_raw(self):
        dec = [_dec("DQ-1", "f1", "source_priority_confirmation")]
        cov = [_cov("f1")]
        res = adv.run_target_advisor(dec, cov, llm_callable=lambda p: "not json")
        self.assertEqual(res["recommendations"][0]["llm_advice_status"], adv.PARSE_FAILED)
        out = Path(tempfile.mkdtemp())
        adv.write_advisor_artifacts(res, out)
        raw = json.loads((out / "36_target_first_llm_raw_response.json").read_text())
        self.assertIn("not json", raw["raw_response"])


# --------------------------------------------------------------------------- #
# 5. Source containment — never accept invented sources
# --------------------------------------------------------------------------- #
class TestSourceContainment(unittest.TestCase):
    def test_invented_source_rejected(self):
        dec = [_dec("DQ-1", "f1", "source_priority_confirmation")]
        cov = [_cov("f1")]  # candidates: Col A, Col B
        res = adv.run_target_advisor(dec, cov, llm_callable=_echo_llm(
            "choose_alternative", recommended_source_column="GHOST COLUMN"))
        r = res["recommendations"][0]
        self.assertEqual(r["llm_advice_status"], adv.INVALID_RESPONSE)
        self.assertEqual(r["llm_recommended_action"], "requires_operator_review")

    def test_supplied_alternative_accepted(self):
        dec = [_dec("DQ-1", "f1", "source_priority_confirmation")]
        cov = [_cov("f1")]
        res = adv.run_target_advisor(dec, cov, llm_callable=_echo_llm(
            "choose_alternative", recommended_source_column="Col B"))
        r = res["recommendations"][0]
        self.assertEqual(r["llm_advice_status"], adv.ADVISED)
        self.assertEqual(r["llm_recommended_action"], "choose_alternative")

    def test_unknown_action_escalates(self):
        dec = [_dec("DQ-1", "f1", "source_priority_confirmation")]
        cov = [_cov("f1")]
        res = adv.run_target_advisor(dec, cov, llm_callable=_echo_llm("teleport_source"))
        self.assertEqual(res["recommendations"][0]["llm_recommended_action"],
                         "requires_operator_review")


# --------------------------------------------------------------------------- #
# 6. Review pack display
# --------------------------------------------------------------------------- #
class TestReviewPackDisplay(unittest.TestCase):
    def _write_min_target_first(self, d: Path, with_advisor: bool):
        (d / "28a_target_coverage_matrix.json").write_text(json.dumps({
            "summary": {"target_fields_total": 1,
                        "coverage_status_counts": {"source_mapped_with_alternatives": 1}},
            "rows": [{"target_field": "f1", "coverage_status": "source_mapped_with_alternatives",
                      "blocking": False, "required_status": "required", "target_domain": "core"}]}),
            encoding="utf-8")
        (d / "28c_human_decision_queue.json").write_text(json.dumps({
            "summary": {"human_decision_rows_total": 1, "blocking_decisions": 0,
                        "decision_type_counts": {"source_priority_confirmation": 1}},
            "rows": [{"decision_id": "DQ-1", "decision_type": "source_priority_confirmation",
                      "priority": "medium", "target_field": "f1", "blocking": False,
                      "operator_question": "which?", "recommendation": "confirm Col A",
                      "options": "", "evidence_summary": "ev"}]}), encoding="utf-8")
        if with_advisor:
            (d / "36_target_first_llm_recommendations.json").write_text(json.dumps({
                "summary": {"recommendations_total": 1, "advised": 1, "requires_operator_review": 0},
                "rows": [{"decision_id": "DQ-1", "target_field": "f1",
                          "decision_type": "source_priority_confirmation",
                          "llm_recommended_action": "confirm_selected",
                          "llm_recommended_source_column": "Col A", "llm_confidence": 0.8,
                          "llm_rationale": "looks right", "llm_advice_status": "advised"}]}),
                encoding="utf-8")
            (d / "36_target_first_llm_usage_summary.json").write_text(json.dumps({
                "llm_target_advisor_enabled": True, "decision_rows_reviewed": 1,
                "decision_rows_advised": 1}), encoding="utf-8")

    def _build(self, with_advisor):
        from engine.onboarding_agent.onboarding_models import OnboardingProject
        from engine.onboarding_agent.review_pack_builder import build_review_pack
        d = Path(tempfile.mkdtemp())
        self._write_min_target_first(d, with_advisor)
        proj = OnboardingProject(project_id="p", client_name="C", input_dir="i",
                                 output_dir=str(d), onboarding_mode="mi_only")
        build_review_pack(proj, d / "08_onboarding_review_pack.html")
        return (d / "08_onboarding_review_pack.html").read_text()

    def test_gate4_shows_llm_advisory_when_present(self):
        html = self._build(with_advisor=True)
        self.assertIn("Target-first LLM advisor", html)
        self.assertIn("LLM advisory", html)

    def test_unchanged_when_advisor_absent(self):
        html = self._build(with_advisor=False)
        self.assertNotIn("Target-first LLM advisor", html)
        self.assertNotIn("LLM advisory", html)


# --------------------------------------------------------------------------- #
# 7. Cost / budget guardrails
# --------------------------------------------------------------------------- #
class TestBudget(unittest.TestCase):
    def test_zero_calls_budget_skips(self):
        dec = [_dec("DQ-1", "f1", "source_priority_confirmation")]
        cov = [_cov("f1")]
        res = adv.run_target_advisor(dec, cov, llm_callable=_echo_llm(), max_calls=0)
        self.assertEqual(res["advice_status"], adv.SKIPPED_BUDGET)
        self.assertTrue(res["usage"]["budget_exhausted"])

    def test_cost_cap_skips(self):
        dec = [_dec("DQ-1", "f1", "source_priority_confirmation")]
        cov = [_cov("f1")]
        res = adv.run_target_advisor(dec, cov, llm_callable=_echo_llm(),
                                     cost_per_call_gbp=1.0, max_cost_gbp=0.01)
        self.assertEqual(res["advice_status"], adv.SKIPPED_BUDGET)


# --------------------------------------------------------------------------- #
# 8. Integration via run_llm_assisted_mapping
# --------------------------------------------------------------------------- #
class TestIntegration(unittest.TestCase):
    def test_advisor_run_matches_deterministic_state(self):
        warnings.simplefilter("ignore")
        base_out = Path(tempfile.mkdtemp(prefix="adv_base_"))
        run_llm_assisted_mapping(input_file=ERE, output_dir=str(base_out),
                                 mode="mi_only", client_id="c", run_id="b")
        adv_out = Path(tempfile.mkdtemp(prefix="adv_llm_"))
        run_llm_assisted_mapping(input_file=ERE, output_dir=str(adv_out),
                                 mode="mi_only", client_id="c", run_id="a",
                                 enable_llm_target_advisor=True,
                                 llm_target_advisor_callable=_echo_llm("confirm_selected"))
        b28a = json.loads((base_out / "28a_target_coverage_matrix.json").read_text())
        l28a = json.loads((adv_out / "28a_target_coverage_matrix.json").read_text())
        b28c = json.loads((base_out / "28c_human_decision_queue.json").read_text())
        l28c = json.loads((adv_out / "28c_human_decision_queue.json").read_text())
        # Deterministic state identical.
        self.assertEqual(len(b28a["rows"]), len(l28a["rows"]))
        self.assertEqual(len(b28c["rows"]), len(l28c["rows"]))
        # 36 only in the advisor run.
        self.assertTrue((adv_out / "36_target_first_llm_recommendations.csv").exists())
        self.assertFalse((base_out / "36_target_first_llm_recommendations.csv").exists())
        # 34 decisions remain pending (advisor never applies).
        import yaml
        tpl = yaml.safe_load((adv_out / "34_target_first_decisions.yaml").read_text())
        self.assertTrue(all(d["status"] == "pending" for d in tpl["decisions"]))


if __name__ == "__main__":
    unittest.main()
