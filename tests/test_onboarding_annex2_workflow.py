#!/usr/bin/env python3
"""tests/test_onboarding_annex2_workflow.py — ESMA Annex 2 target-first delivery.

Covers the acceptance behaviours for running Annex 2 delivery through the
existing target-first operator workflow with TWO config layers:

  1. Annex 2 target contract loading (ESMA codes; not the MI registry).
  2. Regime + asset config loading (both recorded in the 40 summary).
  3. ND / default application (regime + asset, within the regime envelope).
  4. Invalid default handling (surfaced in 42 + Gate 4, never silently applied).
  5. 28c decision-queue quality (genuine regulatory decisions only).
  6. Workflow summary (40) target_contract_id + Annex 2 counts.
  7. Review pack shows the Annex 2 target contract + coverage + Gate 4.
  8. The MI workflow remains unchanged (target contract + no 42 artefact).
  9. The optional LLM target advisor remains advisory only (no 28a/28c mutation).
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

from engine.onboarding_agent import target_coverage as tcov
from engine.onboarding_agent import workflow as wf

PACK = str(_REPO_ROOT / "synthetic_demo" / "input")
REGISTRY = str(_REPO_ROOT / "config" / "system" / "fields_registry.yaml")
ALIASES = str(_REPO_ROOT / "config" / "system")
REGIME = str(_REPO_ROOT / "config" / "regime" / "annex2_delivery_rules.yaml")
ASSET = str(_REPO_ROOT / "config" / "asset" / "product_defaults_ERM.yaml")


def _run_annex2(out: Path, advisor: bool = False):
    warnings.simplefilter("ignore")
    return wf.run_operator_workflow(
        input_dir=PACK, client_name="CLIENT_001_TEST", client_id="client_001",
        run_id="annex2", project_dir=str(out), mode="regulatory_mi",
        registry=REGISTRY, aliases_dir=ALIASES, enable_llm_target_advisor=advisor)


# --------------------------------------------------------------------------- #
# 1 — Annex 2 target loading (unit level)
# --------------------------------------------------------------------------- #
class TestAnnex2TargetLoading(unittest.TestCase):
    def test_regulatory_mode_loads_annex2_codes(self):
        cid, csrc, fields = tcov.load_target_contract("regulatory_mi", {})
        self.assertEqual(cid, "esma_annex_2")
        self.assertTrue(csrc.endswith("annex2_delivery_rules.yaml"))
        names = {f["target_field"] for f in fields}
        for code in ("RREL1", "RREL2", "RREL6", "RREC9", "RREL16", "RREL40",
                     "RREC8", "RREC15"):
            self.assertIn(code, names)

    def test_does_not_use_mi_registry_as_target_contract(self):
        _cid, csrc, fields = tcov.load_target_contract("regulatory_mi", {})
        self.assertNotIn("mi_semantics_field_registry", csrc)
        # No MI canonical field names leak into the Annex 2 target contract.
        names = {f["target_field"] for f in fields}
        self.assertNotIn("account_status", names)
        self.assertNotIn("current_interest_rate", names)


# --------------------------------------------------------------------------- #
# 3 (unit) — ND/default classification + explicit default policy
# --------------------------------------------------------------------------- #
class TestDefaultApplicationUnit(unittest.TestCase):
    def test_mandatory_no_source_no_default_is_blocking(self):
        # A mandatory/enforce_presence field with no source, derivation or valid
        # default is missing_required AND blocking.
        fields = [{
            "target_field": "RREX1", "esma_code": "RREX1",
            "projected_source_field": "made_up_field", "target_domain": "loan",
            "target_label": "x", "required_status": "mandatory",
            "enforce_presence": True, "applicability_status": "applicable",
            "match_field": "made_up_field", "synonyms": [], "derived": False,
            "derivation_rule": "", "default_rule": "", "default_value": "",
            "default_rule_source": "", "default_reason": "", "nd_allowed": [],
            "configured_value_source": "",
        }]
        rows, _ = tcov.build_target_coverage(
            "regulatory_mi", {}, "esma_annex_2", REGIME, fields,
            evidence_rows=[], resolved_rows=[])
        r = rows[0]
        self.assertEqual(r["coverage_status"], tcov.MISSING_REQUIRED)
        self.assertTrue(r["blocking"])

    def test_nd_default_and_value_default_distinct(self):
        cid, csrc, fields = tcov.load_target_contract("regulatory_mi", {})
        by = {f["target_field"]: f for f in fields}
        # RREL16 carries an ND default (ND1); RREC8 carries a non-ND value ("1").
        self.assertTrue(tcov._is_nd(by["RREL16"]["default_value"]))
        self.assertEqual(by["RREC8"]["default_value"], "1")
        self.assertFalse(tcov._is_nd(by["RREC8"]["default_value"]))


# --------------------------------------------------------------------------- #
# 4 (unit) — invalid asset default handling + 2-layer validation
# --------------------------------------------------------------------------- #
class TestConfigValidationUnit(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rows, cls.overlay, cls.asset_src = tcov.build_annex2_config_validation(
            REGIME, ASSET)
        cls.by_code = {r["esma_code"]: r for r in cls.rows if r["esma_code"]}

    def test_both_config_layers_consumed(self):
        self.assertTrue(self.asset_src.endswith("product_defaults_ERM.yaml"))
        # A known regime rule is present.
        self.assertIn("RREL16", self.by_code)
        self.assertIn("RREC8", self.by_code)

    def test_valid_asset_default_applied(self):
        # primary_income (RREL16) asset default ND1 is within nd_allowed -> valid
        # and applied as an asset_config-sourced default.
        self.assertEqual(self.by_code["RREL16"]["validation_status"], tcov.VS_VALID)
        self.assertEqual(self.overlay["RREL16"]["default_rule_source"], "asset_config")

    def test_invalid_asset_default_not_applied(self):
        # debt_to_income_ratio (RREL40) asset default ND1 is NOT in nd_allowed
        # [ND5] -> invalid, surfaced, and the regime default is kept.
        row = self.by_code["RREL40"]
        self.assertEqual(row["validation_status"], tcov.VS_INVALID)
        self.assertEqual(row["asset_default_value"], "ND1")
        # The overlay does NOT apply the invalid value.
        self.assertFalse(self.overlay["RREL40"].get("valid"))
        self.assertNotEqual(self.overlay["RREL40"].get("default_value"), "ND1")

    def test_unknown_and_missing_statuses_present(self):
        statuses = {r["validation_status"] for r in self.rows}
        self.assertIn(tcov.VS_UNKNOWN, statuses)
        self.assertIn(tcov.VS_MISSING_NOT_REQ, statuses)


# --------------------------------------------------------------------------- #
# 1/2/5/6 — full workflow first pass
# --------------------------------------------------------------------------- #
class TestAnnex2Workflow(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.out = Path(tempfile.mkdtemp(prefix="annex2_wf_"))
        cls.summary = _run_annex2(cls.out)
        cls.cov = json.loads(
            (cls.out / "28a_target_coverage_matrix.json").read_text())
        cls.dec = json.loads(
            (cls.out / "28c_human_decision_queue.json").read_text())

    def test_28a_contains_esma_codes(self):
        names = {r["target_field"] for r in self.cov["rows"]}
        self.assertEqual(self.cov["target_contract_id"], "esma_annex_2")
        for code in ("RREL1", "RREL16", "RREL40", "RREC8"):
            self.assertIn(code, names)

    def test_42_artefacts_written(self):
        for name in ("42_annex2_config_validation.csv",
                     "42_annex2_config_validation.json",
                     "42_annex2_config_validation_summary.md"):
            self.assertTrue((self.out / name).exists(), name)

    def test_40_summary_records_contract_and_config_paths(self):
        s = self.summary
        self.assertEqual(s["target_contract_id"], "esma_annex_2")
        self.assertTrue(s["regime_config_path"].endswith("annex2_delivery_rules.yaml"))
        self.assertTrue(s["asset_config_path"].endswith("product_defaults_ERM.yaml"))
        self.assertEqual(s["annex2_field_count"], len(self.cov["rows"]))
        self.assertGreaterEqual(s["annex2_invalid_default_count"], 1)

    def test_40_status_not_ready_when_universe_incomplete(self):
        # With pending_regime_rule codes in the authoritative universe, the
        # Annex 2 run must NOT be READY (config completeness gap).
        self.assertNotEqual(self.summary["status"], wf.READY)
        self.assertEqual(self.summary["status"], wf.NEEDS_CONFIGURATION)
        self.assertGreater(self.summary["annex2_pending_regime_rule_count"], 0)

    def test_explicitly_defaulted_fields_not_in_28c(self):
        # A field that is explicitly ND/value defaulted with no confirmation must
        # NOT appear as a Gate 4 decision.
        dec_fields = {d["target_field"] for d in self.dec["rows"]}
        cov_by = {r["target_field"]: r for r in self.cov["rows"]}
        self.assertEqual(cov_by["RREL16"]["coverage_status"], tcov.DEFAULTED_ND)
        self.assertNotIn("RREL16", dec_fields)  # clean ND default, no decision
        self.assertEqual(cov_by["RREC8"]["coverage_status"], tcov.DEFAULTED_VALUE)
        self.assertNotIn("RREC8", dec_fields)

    def test_invalid_default_appears_as_nonblocking_decision(self):
        inv = [d for d in self.dec["rows"]
               if d["decision_type"] == tcov.D_INVALID_DEFAULT]
        self.assertTrue(inv)
        self.assertIn("RREL40", {d["target_field"] for d in inv})
        for d in inv:
            self.assertFalse(d["blocking"])  # regime fallback exists -> non-blocking

    def test_queue_contains_only_genuine_regulatory_decisions(self):
        allowed = {tcov.D_MISSING, tcov.D_CONFLICT, tcov.D_PRIORITY, tcov.D_VALUE,
                   tcov.D_CONFIG, tcov.D_ND, tcov.D_INVALID_DEFAULT,
                   tcov.D_EXTENSION, tcov.D_PARSE}
        for d in self.dec["rows"]:
            self.assertIn(d["decision_type"], allowed)

    def test_review_pack_shows_annex2(self):
        html = (self.out / "08_onboarding_review_pack.html").read_text()
        self.assertIn("esma_annex_2", html)
        self.assertIn("ESMA Annex 2 delivery", html)
        self.assertIn("Annex 2 coverage by field family", html)
        self.assertIn("Annex 2 delivery readiness", html)

    def test_34_template_generated(self):
        self.assertTrue((self.out / "34_target_first_decisions.yaml").exists())


# --------------------------------------------------------------------------- #
# Field-universe completeness (28a == authoritative universe; 43 reconciliation)
# --------------------------------------------------------------------------- #
class TestAnnex2FieldUniverse(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.out = Path(tempfile.mkdtemp(prefix="annex2_univ_"))
        cls.summary = _run_annex2(cls.out)
        cls.cov = json.loads(
            (cls.out / "28a_target_coverage_matrix.json").read_text())
        cls.recon = json.loads(
            (cls.out / "43_annex2_field_universe_reconciliation.json").read_text())

    def test_universe_loader_unit(self):
        auth = tcov.load_annex2_authoritative_universe(REGISTRY)
        # Workbook registry carries the full Annex 2 code set (> the 68 regime
        # rules), e.g. RREL3 / RREL5 / RREL7 are present but unruled.
        self.assertGreater(len(auth), 68)
        for code in ("RREL3", "RREL5", "RREL7"):
            self.assertIn(code, auth)

    def test_workbook_universe_is_authoritative_107(self):
        wb, src = tcov.load_annex2_workbook_universe()
        # The workbook-derived universe is the authoritative ESMA Annex 2 set.
        self.assertEqual(len(wb), 107)
        self.assertTrue(src.endswith("annex2_field_universe.yaml"))
        # RREC1 is in the authoritative workbook universe (84 RREL + 23 RREC).
        self.assertIn("RREC1", wb)
        self.assertEqual(sum(1 for c in wb if c.startswith("RREL")), 84)
        self.assertEqual(sum(1 for c in wb if c.startswith("RREC")), 23)
        self.assertIn("nd5_allowed", wb["RREL1"])
        self.assertEqual(self.recon["summary"]["authoritative_field_count"], 107)

    def test_rrec1_present_in_28a(self):
        cov_codes = {r["target_field"] for r in self.cov["rows"]}
        self.assertIn("RREC1", cov_codes)

    def test_no_active_phantom_deferred_codes(self):
        # After alignment, NO active runtime code sits outside the authoritative
        # workbook universe: the phantom RREC24-39 codes were moved to the
        # audit-only list, so 43 reports zero not_in_authoritative_universe.
        phantom = [r for r in self.recon["rows"]
                   if r["reconciliation_status"] == "not_in_authoritative_universe"]
        self.assertEqual(phantom, [])
        self.assertEqual(
            self.recon["summary"].get("not_in_authoritative_universe_count", 0), 0)
        self.assertEqual(self.summary["annex2_active_phantom_deferred_count"], 0)
        # The previously-phantom codes are not in 28a either.
        cov_codes = {r["target_field"] for r in self.cov["rows"]}
        for code in ("RREC24", "RREC30", "RREC39"):
            self.assertNotIn(code, cov_codes)

    def test_rrec1_registry_mapped(self):
        # RREC1 (the known registry gap) is now mapped in fields_registry, so 43
        # records it as registry_mapped with zero registry gaps overall.
        by = {r["esma_code"]: r for r in self.recon["rows"]}
        self.assertEqual(by["RREC1"]["registry_mapping_status"], "registry_mapped")
        self.assertTrue(by["RREC1"]["in_registry_mapping"])
        self.assertEqual(self.recon["summary"]["registry_gap_count"], 0)
        self.assertEqual(self.summary["annex2_registry_gap_count"], 0)
        self.assertEqual(self.summary["annex2_registry_mapped_count"], 107)

    def test_28a_equals_authoritative_universe_count(self):
        recon_sum = self.recon["summary"]
        self.assertEqual(len(self.cov["rows"]),
                         recon_sum["authoritative_field_count"])
        self.assertEqual(self.summary["annex2_coverage_field_count"],
                         self.summary["annex2_authoritative_field_count"])

    def test_no_authoritative_code_missing_from_28a(self):
        self.assertEqual(self.recon["summary"]["missing_from_28a_count"], 0)
        missing = [r for r in self.recon["rows"]
                   if r["reconciliation_status"] == "missing_from_28a"]
        self.assertEqual(missing, [])
        # Every regime field rule and every workbook code is represented in 28a.
        cov_codes = {r["target_field"] for r in self.cov["rows"]}
        for r in self.recon["rows"]:
            if r["in_regime_field_rules"] or r["in_workbook_reconciliation"]:
                self.assertIn(r["esma_code"], cov_codes)

    def test_43_artefacts_written(self):
        for name in ("43_annex2_field_universe_reconciliation.csv",
                     "43_annex2_field_universe_reconciliation.json",
                     "43_annex2_field_universe_reconciliation_summary.md"):
            self.assertTrue((self.out / name).exists(), name)

    def test_deferred_fields_present_as_deferred(self):
        cov_by = {r["target_field"]: r for r in self.cov["rows"]}
        # RREC22 is a deferred reconciliation code: present in 28a, not dropped.
        self.assertIn("RREC22", cov_by)
        deferred_codes = [r["esma_code"] for r in self.recon["rows"]
                          if r["reconciliation_status"] == "deferred_in_regime"]
        self.assertGreater(len(deferred_codes), 0)
        # Deferred codes either carry deferred applicability or are source-mapped,
        # never silently omitted.
        for code in deferred_codes:
            self.assertIn(code, cov_by)

    def test_pending_codes_appear_as_pending_regime_rule(self):
        cov_by = {r["target_field"]: r for r in self.cov["rows"]}
        pending = [r["esma_code"] for r in self.recon["rows"]
                   if r["reconciliation_status"] == "missing_from_regime_rules"]
        self.assertGreater(len(pending), 0)
        # At least some pending (unruled, no source) carry the pending status.
        statuses = {cov_by[c]["coverage_status"] for c in pending if c in cov_by}
        self.assertIn(tcov.PENDING_REGIME_RULE, statuses)

    def test_40_reports_universe_counts(self):
        s = self.summary
        for k in ("annex2_authoritative_field_count", "annex2_coverage_field_count",
                  "annex2_regime_rule_count", "annex2_config_validation_count",
                  "annex2_missing_from_28a_count", "annex2_deferred_field_count",
                  "annex2_deliverable_field_count"):
            self.assertIn(k, s)
        self.assertEqual(s["annex2_regime_rule_count"], 68)
        self.assertGreater(s["annex2_authoritative_field_count"], 68)

    def test_review_pack_shows_universe_reconciliation(self):
        html = (self.out / "08_onboarding_review_pack.html").read_text()
        self.assertIn("Annex 2 field universe reconciliation", html)
        self.assertIn("Authoritative Annex 2 fields", html)


# --------------------------------------------------------------------------- #
# ND-eligibility reconciliation (44): regime nd_allowed vs workbook eligibility
# --------------------------------------------------------------------------- #
class TestAnnex2NdEligibility(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.out = Path(tempfile.mkdtemp(prefix="annex2_nd_"))
        cls.summary = _run_annex2(cls.out)
        cls.nd = json.loads(
            (cls.out / "44_annex2_nd_eligibility_reconciliation.json").read_text())

    def test_44_artefacts_written(self):
        for name in ("44_annex2_nd_eligibility_reconciliation.csv",
                     "44_annex2_nd_eligibility_reconciliation.json",
                     "44_annex2_nd_eligibility_reconciliation_summary.md"):
            self.assertTrue((self.out / name).exists(), name)

    def test_nd_reconciliation_unit(self):
        rows = tcov.build_annex2_nd_eligibility_reconciliation()
        by = {r["esma_code"]: r for r in rows}
        # RREL40: regime restricts to [ND5] but the workbook allows ND1-ND5 too,
        # so the regime is STRICTER than the authoritative eligibility.
        self.assertEqual(by["RREL40"]["nd_alignment_status"], "regime_stricter")
        # Statuses are drawn from the documented vocabulary.
        allowed = {"match", "regime_stricter", "regime_broader", "divergent",
                   "no_regime_rule", "not_in_workbook"}
        for r in rows:
            self.assertIn(r["nd_alignment_status"], allowed)

    def test_regime_broader_is_zero_after_tightening(self):
        s = self.nd["summary"]
        # The 5 regime_broader cases (RREL1/2/6/69/83) were tightened to the
        # workbook envelope, so NO regime rule is broader than the workbook.
        self.assertEqual(s["regime_broader"], 0)
        self.assertEqual(self.summary["annex2_nd_regime_broader_count"], 0)
        by = {r["esma_code"]: r for r in self.nd["rows"]}
        for code in ("RREL1", "RREL2", "RREL6", "RREL69", "RREL83"):
            self.assertEqual(by[code]["nd_alignment_status"], "match")

    def test_divergent_and_stricter_surfaced_not_widened(self):
        s = self.nd["summary"]
        # Divergent + stricter cases remain (surfaced for review / kept by
        # policy) — they are NOT auto-widened away.
        self.assertGreater(s["divergent"], 0)
        self.assertGreater(s["regime_stricter"], 0)
        self.assertEqual(self.summary["annex2_nd_divergent_count"], s["divergent"])
        self.assertEqual(self.summary["annex2_nd_regime_stricter_count"],
                         s["regime_stricter"])
        # Divergent cases are flagged for manual review in the workflow warnings.
        self.assertTrue(any("manual review" in w for w in self.summary["warnings"]))

    def test_regime_validation_behaviour_unchanged(self):
        # 42 config validation still uses the regime nd_allowed (RREL40 -> [ND5]),
        # i.e. the reconciliation is report-only and does NOT widen regime rules.
        val = json.loads(
            (self.out / "42_annex2_config_validation.json").read_text())
        rrel40 = next(r for r in val["rows"] if r["esma_code"] == "RREL40")
        self.assertEqual(rrel40["regime_nd_allowed"], "ND5")
        self.assertEqual(rrel40["validation_status"], tcov.VS_INVALID)

    def test_review_pack_shows_nd_reconciliation(self):
        html = (self.out / "08_onboarding_review_pack.html").read_text()
        self.assertIn("Annex 2 ND-eligibility reconciliation", html)


# --------------------------------------------------------------------------- #
# Enum-coverage reconciliation (46): regime enum_map vs workbook allowed codes
# --------------------------------------------------------------------------- #
class TestAnnex2EnumCoverage(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.out = Path(tempfile.mkdtemp(prefix="annex2_enum_"))
        cls.summary = _run_annex2(cls.out)
        cls.enum = json.loads(
            (cls.out / "46_annex2_enum_coverage_reconciliation.json").read_text())
        cls.by = {r["esma_code"]: r for r in cls.enum["rows"]}

    def test_46_artefacts_written(self):
        for name in ("46_annex2_enum_coverage_reconciliation.csv",
                     "46_annex2_enum_coverage_reconciliation.json",
                     "46_annex2_enum_coverage_reconciliation_summary.md"):
            self.assertTrue((self.out / name).exists(), name)

    def test_no_enum_exceeds_workbook(self):
        # No regime enum_map may map to a code the workbook forbids.
        self.assertEqual(self.enum["summary"]["targets_outside_workbook"], 0)
        self.assertEqual(self.summary["annex2_enum_targets_outside_workbook_count"], 0)

    def test_added_enum_maps_are_within_workbook(self):
        import yaml as _y
        wb = tcov.load_annex2_workbook_universe()[0]
        fr = _y.safe_load(open("config/regime/annex2_delivery_rules.yaml"))["field_rules"]
        for code in ("RREL19", "RREL56", "RREL57", "RREC10", "RREC18"):
            allowed = set(tcov._annex2_workbook_enum_codes(wb[code]["content"]))
            targets = set(fr[code]["transform"]["enum_map"].values())
            self.assertTrue(targets, code)
            self.assertTrue(targets <= allowed, f"{code}: {targets - allowed} outside workbook")
            self.assertEqual(self.by[code]["enum_coverage_status"],
                             "constrained_within_workbook")

    def test_semantic_mismatch_surfaced_not_constrained(self):
        # Codes whose regime rule points at the wrong field (vs the workbook) are
        # surfaced for review and NOT silently enum-constrained.
        for code in ("RREL17", "RREL70", "RREC23"):
            self.assertEqual(self.by[code]["enum_coverage_status"], "semantic_mismatch")
            self.assertTrue(self.by[code]["requires_manual_review"])
        self.assertGreater(self.enum["summary"]["semantic_mismatch"], 0)
        self.assertTrue(any("source field does not match" in w
                            for w in self.summary["warnings"]))

    def test_45_records_enum_constraint_actions(self):
        align = json.loads(
            (self.out / "45_annex2_config_alignment_review.json").read_text())
        constrained = [r for r in align["rows"]
                       if r["alignment_status"] == "enum_constrained_to_workbook"]
        self.assertEqual(len(constrained), 5)

    def test_review_pack_shows_enum_reconciliation(self):
        html = (self.out / "08_onboarding_review_pack.html").read_text()
        self.assertIn("Annex 2 enum-coverage reconciliation", html)


# --------------------------------------------------------------------------- #
# Config-alignment review (45): actions taken + manual-review items
# --------------------------------------------------------------------------- #
class TestAnnex2ConfigAlignment(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.out = Path(tempfile.mkdtemp(prefix="annex2_align_"))
        cls.summary = _run_annex2(cls.out)
        cls.align = json.loads(
            (cls.out / "45_annex2_config_alignment_review.json").read_text())
        cls.by = {}
        for r in cls.align["rows"]:
            cls.by.setdefault(r["esma_code"], []).append(r)

    def test_45_artefacts_written(self):
        for name in ("45_annex2_config_alignment_review.csv",
                     "45_annex2_config_alignment_review.json",
                     "45_annex2_config_alignment_review_summary.md"):
            self.assertTrue((self.out / name).exists(), name)

    def test_columns_match_spec(self):
        for col in ("esma_code", "workbook_field_name", "workbook_nd_allowed",
                    "regime_nd_allowed_before", "regime_nd_allowed_after",
                    "alignment_status", "action_taken", "requires_manual_review",
                    "message"):
            self.assertIn(col, self.align["rows"][0])

    def test_records_alignment_actions(self):
        s = self.align["summary"]
        self.assertEqual(s["tightened_to_workbook"], 5)
        self.assertEqual(s["phantom_deferred_removed"], 11)
        self.assertEqual(s["registry_mapping_added"], 1)
        self.assertEqual(s["registry_gap"], 0)
        # Tightened rows show before != after (broader -> workbook envelope).
        rrel1 = self.by["RREL1"][0]
        self.assertEqual(rrel1["alignment_status"], "tightened_to_workbook")
        self.assertEqual(rrel1["regime_nd_allowed_before"], "ND1; ND2; ND3")
        self.assertEqual(rrel1["regime_nd_allowed_after"], "")

    def test_interest_rate_type_resolved_via_enum_map(self):
        # interest_rate_type=Fixed now maps to ESMA FXRL (not OTHR), so RREL42 is
        # no longer an invalid asset default.
        val = json.loads(
            (self.out / "42_annex2_config_validation.json").read_text())
        rrel42 = next(r for r in val["rows"] if r["esma_code"] == "RREL42")
        self.assertEqual(rrel42["validation_status"], tcov.VS_VALID)
        ok, _ = tcov._validate_value_against_rule(
            "Fixed", {"transform": {"enum_map": {"Fixed": "FXRL"}}})
        self.assertTrue(ok)

    def test_remaining_asset_conflicts_require_manual_review(self):
        # Bullet amortisation and DTI=ND1 are NOT auto-resolved (no obvious/safe
        # enum map; regime stricter by policy) — surfaced as conflicts.
        conflicts = [r for r in self.align["rows"]
                     if r["alignment_status"] == "asset_default_conflict"]
        codes = {r["esma_code"] for r in conflicts}
        self.assertIn("RREL35", codes)  # amortisation_type = Bullet
        self.assertIn("RREL40", codes)  # debt_to_income_ratio = ND1
        for r in conflicts:
            self.assertTrue(r["requires_manual_review"])

    def test_40_reports_alignment_counts(self):
        s = self.summary
        self.assertEqual(s["annex2_alignment_tightened_count"], 5)
        self.assertEqual(s["annex2_alignment_phantom_removed_count"], 11)
        self.assertEqual(s["annex2_alignment_registry_added_count"], 1)
        self.assertEqual(s["annex2_asset_default_conflict_count"], 2)
        self.assertGreater(s["annex2_alignment_manual_review_count"], 0)

    def test_review_pack_shows_alignment_review(self):
        html = (self.out / "08_onboarding_review_pack.html").read_text()
        self.assertIn("Annex 2 config-alignment review", html)


# --------------------------------------------------------------------------- #
# 8 — MI workflow remains unchanged
# --------------------------------------------------------------------------- #
class TestMiUnchanged(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.out = Path(tempfile.mkdtemp(prefix="annex2_mi_"))
        warnings.simplefilter("ignore")
        cls.summary = wf.run_operator_workflow(
            input_dir=PACK, client_name="MI", client_id="mi", run_id="r",
            project_dir=str(cls.out), mode="mi_only", registry=REGISTRY,
            aliases_dir=ALIASES)

    def test_mi_uses_mi_registry_contract(self):
        cov = json.loads((self.out / "28a_target_coverage_matrix.json").read_text())
        self.assertEqual(cov["target_contract_id"], "mi_semantics_field_registry")

    def test_mi_has_no_annex2_artefacts_or_summary(self):
        for name in ("42_annex2_config_validation.csv",
                     "43_annex2_field_universe_reconciliation.csv",
                     "44_annex2_nd_eligibility_reconciliation.csv",
                     "45_annex2_config_alignment_review.csv",
                     "46_annex2_enum_coverage_reconciliation.csv"):
            self.assertFalse((self.out / name).exists(), name)
        self.assertNotIn("annex2_field_count", self.summary)
        self.assertNotIn("annex2_authoritative_field_count", self.summary)
        self.assertNotIn("annex2_registry_mapped_count", self.summary)
        self.assertEqual(self.summary.get("target_contract_id"),
                         "mi_semantics_field_registry")

    def test_mi_field_count_unchanged(self):
        cov = json.loads((self.out / "28a_target_coverage_matrix.json").read_text())
        self.assertEqual(cov["summary"]["target_fields_total"], 72)


# --------------------------------------------------------------------------- #
# 9 — LLM target advisor optional + advisory only
# --------------------------------------------------------------------------- #
class TestAnnex2LlmAdvisorOptional(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.out = Path(tempfile.mkdtemp(prefix="annex2_llm_"))
        cls.summary = _run_annex2(cls.out, advisor=True)

    def test_advisor_writes_36_artefacts(self):
        self.assertTrue((self.out / "36_target_first_llm_recommendations.csv").exists())
        self.assertTrue((self.out / "36_target_first_llm_usage_summary.json").exists())
        self.assertTrue(self.summary["llm_target_advisor_enabled"])

    def test_advisor_does_not_mutate_28a_28c(self):
        # Re-run the same first pass WITHOUT the advisor and compare the
        # deterministic target-first state.
        out2 = Path(tempfile.mkdtemp(prefix="annex2_nollm_"))
        _run_annex2(out2, advisor=False)
        cov_llm = json.loads((self.out / "28a_target_coverage_matrix.json").read_text())
        cov_no = json.loads((out2 / "28a_target_coverage_matrix.json").read_text())
        self.assertEqual(cov_llm["summary"]["coverage_status_counts"],
                         cov_no["summary"]["coverage_status_counts"])
        dec_llm = json.loads((self.out / "28c_human_decision_queue.json").read_text())
        dec_no = json.loads((out2 / "28c_human_decision_queue.json").read_text())
        self.assertEqual(dec_llm["summary"]["human_decision_rows_total"],
                         dec_no["summary"]["human_decision_rows_total"])


if __name__ == "__main__":
    unittest.main()
