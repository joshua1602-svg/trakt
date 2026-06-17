#!/usr/bin/env python3
"""tests/test_onboarding_handoff_contract.py

Onboarding Agent → Transformation & Validation handoff package (24–27).

Covers:
  * a formal handoff package is created after a regulatory_mi onboarding run;
  * the manifest identifies itself as a canonical_onboarding_package, not raw
    source, and forbids re-running Gate 1 on the central tape;
  * ready_for_transformation_validation can be true while ready_for_xml_delivery
    is false;
  * the central tape remains the generic, unchanged onboarding artefact;
  * asset defaults → default_downstream; ND defaults → nd_default_downstream;
  * current_outstanding_balance is NOT silently aliased to
    current_principal_balance (semantic_derivation_required);
  * pending regime rules are owned by Projection, not treated as onboarding
    failures;
  * LLM recommendations remain advisory-only and do not alter handoff state;
  * the MI workflow remains unchanged (no handoff package).
"""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
import warnings
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from engine.onboarding_agent import onboarding_handoff as oh
from engine.onboarding_agent import target_coverage as tcov
from engine.onboarding_agent import workflow as wf

PACK = str(_REPO_ROOT / "synthetic_demo" / "input")
REGISTRY = str(_REPO_ROOT / "config" / "system" / "fields_registry.yaml")
ALIASES = str(_REPO_ROOT / "config" / "system")


def _run(out: Path, advisor: bool = False, mode: str = "regulatory_mi"):
    warnings.simplefilter("ignore")
    return wf.run_operator_workflow(
        input_dir=PACK, client_name="CLIENT_001_TEST", client_id="client_001",
        run_id="handoff", project_dir=str(out), mode=mode,
        registry=REGISTRY, aliases_dir=ALIASES, enable_llm_target_advisor=advisor)


# --------------------------------------------------------------------------- #
# Unit-level classification (pure, no IO)
# --------------------------------------------------------------------------- #
class TestClassifyField(unittest.TestCase):
    def _row(self, **kw):
        base = {
            "target_field": "RREL_X", "esma_code": "RREL_X",
            "coverage_status": tcov.SOURCE_MAPPED, "required_status": "mandatory",
            "requires_user_decision": False, "blocking": False,
            "default_rule_source": "", "default_value": "",
            "projected_source_field": "", "selected_source_column": "",
            "selected_source_file": "", "selected_value": "",
        }
        base.update(kw)
        return base

    def test_asset_default_is_default_downstream(self):
        r = self._row(coverage_status=tcov.DEFAULTED_VALUE, default_value="GBP",
                      default_rule_source="asset_config", selected_source_file="")
        c = oh.classify_field(r)
        self.assertEqual(c["handoff_classification"], oh.HC_DEFAULT_DOWNSTREAM)
        self.assertEqual(c["downstream_owner"], oh.OWN_TRANSFORMATION)
        self.assertEqual(c["next_agent_action"], "materialise_default_from_asset_config")

    def test_nd_default_is_nd_default_downstream(self):
        r = self._row(coverage_status=tcov.DEFAULTED_ND, default_value="ND1")
        c = oh.classify_field(r)
        self.assertEqual(c["handoff_classification"], oh.HC_ND_DEFAULT_DOWNSTREAM)
        self.assertEqual(c["downstream_owner"], oh.OWN_TRANSFORMATION)
        self.assertEqual(c["next_agent_action"], "materialise_nd_default_if_still_unmapped")

    def test_pending_regime_rule_owned_by_projection(self):
        r = self._row(coverage_status=tcov.PENDING_REGIME_RULE, required_status="optional")
        c = oh.classify_field(r)
        self.assertEqual(c["handoff_classification"], oh.HC_PENDING_REGIME_RULE)
        self.assertEqual(c["downstream_owner"], oh.OWN_PROJECTION)
        self.assertEqual(c["next_agent_action"], "implement_or_defer_regime_rule")

    def test_outstanding_balance_not_silently_aliased(self):
        # current_outstanding_balance -> current_principal_balance is a semantic
        # derivation decision, never a silent alias.
        r = self._row(coverage_status=tcov.SOURCE_MAPPED,
                      projected_source_field="current_principal_balance",
                      selected_source_column="Current Outstanding Balance",
                      selected_source_file="loans.csv")
        c = oh.classify_field(r)
        self.assertEqual(c["handoff_classification"], oh.HC_SEMANTIC_DERIVATION_REQUIRED)
        self.assertEqual(c["downstream_owner"], oh.OWN_TRANSFORMATION)
        self.assertEqual(c["next_agent_action"],
                         "define_approved_ERM_balance_derivation_or_operator_decision")

    def test_clean_principal_balance_not_flagged(self):
        # A genuinely-named principal balance column is a normal source_mapped.
        r = self._row(coverage_status=tcov.SOURCE_MAPPED,
                      projected_source_field="current_principal_balance",
                      selected_source_column="Current Principal Balance GBP",
                      selected_source_file="loans.csv")
        c = oh.classify_field(r)
        self.assertEqual(c["handoff_classification"], oh.HC_SOURCE_MAPPED)

    def test_blocking_decision_is_operator_owned(self):
        r = self._row(coverage_status=tcov.MISSING_REQUIRED,
                      requires_user_decision=True, blocking=True)
        c = oh.classify_field(r)
        self.assertEqual(c["handoff_classification"], oh.HC_OPERATOR_DECISION_PENDING)
        self.assertEqual(c["downstream_owner"], oh.OWN_OPERATOR)

    def test_is_semantic_derivation_helper(self):
        self.assertTrue(oh.is_semantic_derivation(
            "current_principal_balance", "current_outstanding_balance"))
        self.assertFalse(oh.is_semantic_derivation(
            "current_principal_balance", "current_principal_balance"))
        self.assertFalse(oh.is_semantic_derivation("some_other_field", "outstanding"))


class TestReadinessLogic(unittest.TestCase):
    def _counts(self, **kw):
        c = {"pending_regime_rule_count": 0, "semantic_derivation_required_count": 0,
             "downstream_transformation_required_count": 0,
             "operator_decision_pending_count": 0}
        c.update(kw)
        return c

    def test_tv_ready_independent_of_xml(self):
        r = oh.compute_readiness(
            central_exists=True, coverage_present=True, target_universe_loaded=True,
            registry_gap_count=0, blocking_decision_count=0,
            counts=self._counts(downstream_transformation_required_count=40))
        self.assertTrue(r["ready_for_transformation_validation"])
        self.assertFalse(r["ready_for_xml_delivery"])

    def test_blocking_decisions_block_tv(self):
        r = oh.compute_readiness(
            central_exists=True, coverage_present=True, target_universe_loaded=True,
            registry_gap_count=0, blocking_decision_count=2, counts=self._counts())
        self.assertFalse(r["ready_for_transformation_validation"])

    def test_missing_central_tape_blocks_tv(self):
        r = oh.compute_readiness(
            central_exists=False, coverage_present=True, target_universe_loaded=True,
            registry_gap_count=0, blocking_decision_count=0, counts=self._counts())
        self.assertFalse(r["ready_for_transformation_validation"])

    def test_projection_never_ready_with_pending_rules(self):
        r = oh.compute_readiness(
            central_exists=True, coverage_present=True, target_universe_loaded=True,
            registry_gap_count=0, blocking_decision_count=0,
            counts=self._counts(pending_regime_rule_count=3))
        self.assertFalse(r["ready_for_projection"])


# --------------------------------------------------------------------------- #
# Full regulatory_mi workflow — handoff package produced
# --------------------------------------------------------------------------- #
class TestHandoffWorkflow(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.out = Path(tempfile.mkdtemp(prefix="handoff_wf_"))
        cls.summary = _run(cls.out)
        cls.hf = cls.out / "output" / "handoff"
        cls.manifest = json.loads(
            (cls.hf / "24_onboarding_handoff_manifest.json").read_text())
        cls.contract = json.loads(
            (cls.hf / "26_onboarding_handoff_field_contract.json").read_text())["rows"]

    def test_all_handoff_artefacts_written(self):
        for name in ("24_onboarding_handoff_manifest.json",
                     "24_onboarding_handoff_manifest.yaml",
                     "25_onboarding_handoff_readiness.json",
                     "25_onboarding_handoff_readiness.md",
                     "26_onboarding_handoff_field_contract.csv",
                     "26_onboarding_handoff_field_contract.json",
                     "27_onboarding_handoff_lineage.json"):
            self.assertTrue((self.hf / name).exists(), name)

    def test_manifest_identity_flags(self):
        m = self.manifest
        self.assertEqual(m["handoff_type"], "canonical_onboarding_package")
        self.assertEqual(m["handoff_stage"], "post_onboarding_pre_transformation_validation")
        self.assertEqual(m["next_agent"], "transformation_validation")
        self.assertTrue(m["not_raw_source"])
        self.assertTrue(m["do_not_rerun_gate1_on_central_tape"])

    def test_tv_ready_but_not_xml_ready(self):
        m = self.manifest
        self.assertTrue(m["ready_for_transformation_validation"])
        self.assertFalse(m["ready_for_xml_delivery"])
        self.assertTrue(m["not_xml_ready"])

    def test_manifest_references_consumable_artefacts(self):
        m = self.manifest
        self.assertTrue(m["central_tape_path"].endswith("18_central_lender_tape.csv"))
        self.assertGreater(m["central_tape_row_count"], 0)
        self.assertTrue(m["target_coverage_matrix_path"].endswith(
            "28a_target_coverage_matrix.csv"))
        self.assertTrue(Path(m["lineage_path"]).exists())

    def test_field_contract_covers_every_target_field(self):
        cov = json.loads(
            (self.out / "28a_target_coverage_matrix.json").read_text())
        self.assertEqual(len(self.contract), len(cov["rows"]))
        self.assertEqual(self.manifest["target_field_count"], len(cov["rows"]))

    def test_field_contract_controlled_vocabulary(self):
        allowed_cls = {
            oh.HC_SOURCE_MAPPED, oh.HC_SOURCE_MAPPED_ALT, oh.HC_OPERATOR_DECISION_PENDING,
            oh.HC_APPROVED_DECISION_APPLIED, oh.HC_CONFIGURED_STATIC,
            oh.HC_DEFAULT_DOWNSTREAM, oh.HC_ND_DEFAULT_DOWNSTREAM,
            oh.HC_PENDING_REGIME_RULE, oh.HC_SOURCE_ABSENT, oh.HC_ALIAS_MISMATCH,
            oh.HC_SEMANTIC_DERIVATION_REQUIRED, oh.HC_TRANSFORMATION_REQUIRED,
            oh.HC_PROJECTION_REQUIRED, oh.HC_DELIVERY_REQUIRED, oh.HC_NOT_APPLICABLE,
        }
        allowed_owner = {oh.OWN_ONBOARDING, oh.OWN_TRANSFORMATION, oh.OWN_PROJECTION,
                         oh.OWN_DELIVERY, oh.OWN_OPERATOR}
        for r in self.contract:
            self.assertIn(r["handoff_classification"], allowed_cls)
            self.assertIn(r["downstream_owner"], allowed_owner)

    def test_asset_defaults_not_treated_as_missing_client_data(self):
        # Every ND default is downstream-owned by transformation_validation, never
        # surfaced as missing client data.
        nd = [r for r in self.contract
              if r["coverage_status"] == tcov.DEFAULTED_ND]
        self.assertTrue(nd)
        for r in nd:
            # ND defaults are either downstream ND materialisation or an explicit
            # (non-silent) operator confirmation — never source_absent.
            self.assertIn(r["handoff_classification"],
                          {oh.HC_ND_DEFAULT_DOWNSTREAM, oh.HC_OPERATOR_DECISION_PENDING})
            self.assertNotEqual(r["handoff_classification"], oh.HC_SOURCE_ABSENT)

    def test_pending_regime_rules_owned_by_projection(self):
        pend = [r for r in self.contract
                if r["handoff_classification"] == oh.HC_PENDING_REGIME_RULE]
        self.assertGreater(len(pend), 0)
        for r in pend:
            self.assertEqual(r["downstream_owner"], oh.OWN_PROJECTION)
        # Pending rules do not block transformation/validation readiness.
        self.assertTrue(self.manifest["ready_for_transformation_validation"])

    def test_summary_carries_handoff_references(self):
        s = self.summary
        self.assertEqual(s["onboarding_handoff_type"], "canonical_onboarding_package")
        self.assertEqual(s["onboarding_handoff_next_agent"], "transformation_validation")
        self.assertTrue(s["ready_for_transformation_validation"])
        self.assertFalse(s["ready_for_xml_delivery"])
        self.assertTrue(Path(s["onboarding_handoff_manifest_json"]).exists())

    def test_review_pack_shows_handoff_section(self):
        html = (self.out / "08_onboarding_review_pack.html").read_text()
        self.assertIn("Onboarding handoff", html)
        self.assertIn("canonical onboarding package", html)
        self.assertIn("not raw source input", html)
        self.assertIn("transformation_validation", html)

    def test_central_tape_is_generic_and_unchanged(self):
        # The central tape carries no Annex 2 / ESMA RREL codes nor handoff
        # governance columns — it is the generic canonical lender tape.
        tape = (self.out / "output" / "central" / "18_central_lender_tape.csv")
        header = tape.read_text().splitlines()[0]
        self.assertNotIn("RREL", header)
        self.assertNotIn("handoff_classification", header)
        self.assertNotIn("esma_code", header)


# --------------------------------------------------------------------------- #
# LLM advisory remains advisory-only
# --------------------------------------------------------------------------- #
class TestHandoffLlmAdvisoryOnly(unittest.TestCase):
    def test_llm_does_not_alter_handoff_state(self):
        out_llm = Path(tempfile.mkdtemp(prefix="handoff_llm_"))
        out_no = Path(tempfile.mkdtemp(prefix="handoff_nollm_"))
        _run(out_llm, advisor=True)
        _run(out_no, advisor=False)
        m_llm = json.loads(
            (out_llm / "output" / "handoff" /
             "24_onboarding_handoff_manifest.json").read_text())
        m_no = json.loads(
            (out_no / "output" / "handoff" /
             "24_onboarding_handoff_manifest.json").read_text())
        self.assertTrue(m_llm["llm_recommendations_advisory_only"])
        for k in ("ready_for_transformation_validation", "ready_for_projection",
                  "ready_for_xml_delivery", "target_field_count",
                  "source_mapped_count", "defaulted_nd_count",
                  "pending_regime_rule_count", "blocking_decision_count"):
            self.assertEqual(m_llm[k], m_no[k], k)


# --------------------------------------------------------------------------- #
# MI workflow unchanged — no handoff package
# --------------------------------------------------------------------------- #
class TestMiUnchanged(unittest.TestCase):
    def test_mi_run_has_no_handoff_package(self):
        out = Path(tempfile.mkdtemp(prefix="handoff_mi_"))
        warnings.simplefilter("ignore")
        summary = wf.run_operator_workflow(
            input_dir=PACK, client_name="MI", client_id="mi", run_id="r",
            project_dir=str(out), mode="mi_only", registry=REGISTRY,
            aliases_dir=ALIASES)
        self.assertFalse((out / "output" / "handoff" /
                          "24_onboarding_handoff_manifest.json").exists())
        self.assertNotIn("onboarding_handoff_type", summary)


if __name__ == "__main__":
    unittest.main()
