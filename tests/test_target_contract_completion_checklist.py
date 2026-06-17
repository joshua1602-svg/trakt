#!/usr/bin/env python3
"""tests/test_target_contract_completion_checklist.py

Target Contract Completion Checklist / Target Field Disposition layer.

Onboarding owns the per-target-field disposition; downstream agents execute it.

Covers:
  * every target field receives a field_disposition (real Annex 2 configs);
  * RREL40 -> nd_policy_selected for ERM/client policy when configured;
  * RREL40 stays unresolved/config-required for a no-policy fixture;
  * RREL24 -> nd_policy_selected (maturity_date = ND5) for ERM;
  * RREL24 is NOT generically ND5 for a non-ERM / no-policy asset;
  * RREL1/RREL2 can be marked client_onboarding_required;
  * "ND allowed" is not treated as "ND selected";
  * unresolved rows surface to the review bench with the right categories;
  * handoff manifest references the checklist + the field contract carries
    disposition columns;
  * Transformation / Validation / Projection consume the disposition;
  * no XML / delivery artefacts are produced.
"""

from __future__ import annotations

import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from engine.onboarding_agent import target_contract_completion as tcc
from engine.onboarding_agent import onboarding_handoff as oh

REGISTRY = str(_REPO_ROOT / "config" / "system" / "fields_registry.yaml")
ASSET = str(_REPO_ROOT / "config" / "asset" / "product_defaults_ERM.yaml")
REGIME = str(_REPO_ROOT / "config" / "regime" / "annex2_delivery_rules.yaml")
UNIVERSE = str(_REPO_ROOT / "config" / "regime" / "annex2_field_universe.yaml")


def _real_configs():
    return tcc.load_target_contract_configs(
        regime_config_path=REGIME, field_universe_path=UNIVERSE,
        registry_path=REGISTRY, asset_config_path=ASSET)


def _checklist(asset_cfg=None, asset_class="equity_release", coverage_by_code=None,
               client_policy=None):
    cfgs = _real_configs()
    return tcc.build_completion_checklist(
        contract_id="ESMA_Annex2",
        field_universe=cfgs["field_universe"], registry_fields=cfgs["registry_fields"],
        regime_index=cfgs["regime_index"],
        asset_cfg=asset_cfg if asset_cfg is not None else cfgs["asset_cfg"],
        asset_class=asset_class, coverage_by_code=coverage_by_code or {},
        client_policy=client_policy or {})


# --------------------------------------------------------------------------- #
# Completeness + core dispositions (real configs)
# --------------------------------------------------------------------------- #
class TestChecklistCompleteness(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rows = _checklist()
        cls.by = {r["esma_code"]: r for r in cls.rows}

    def test_every_field_has_a_disposition(self):
        self.assertGreaterEqual(len(self.rows), 100)
        for r in self.rows:
            self.assertIn(r["field_disposition"], tcc.DISPOSITIONS, r["esma_code"])
            self.assertTrue(r["field_disposition"])  # never blank

    def test_one_disposition_per_field(self):
        codes = [r["esma_code"] for r in self.rows]
        self.assertEqual(len(codes), len(set(codes)))

    def test_rrel40_nd_policy_selected_for_erm(self):
        r = self.by["RREL40"]  # debt_to_income_ratio
        self.assertEqual(r["field_disposition"], tcc.D_ND_POLICY_SELECTED)
        self.assertEqual(r["configured_default_value"], "ND5")
        self.assertFalse(r["requires_operator_review"])
        self.assertFalse(r["blocking_for_projection"])

    def test_rrel24_nd_policy_selected_for_erm(self):
        r = self.by["RREL24"]  # maturity_date
        self.assertIn(r["field_disposition"],
                      (tcc.D_ND_POLICY_SELECTED, tcc.D_ASSET_DEFAULT))
        self.assertEqual(r["configured_default_value"], "ND5")
        self.assertFalse(r["blocking_for_projection"])

    def test_rrel1_rrel2_formal_identifier_policy_required(self):
        # Formal regulatory identifiers are NOT satisfied by ordinary loan IDs
        # unless an explicit approval policy exists.
        for code in ("RREL1", "RREL2"):
            r = self.by[code]
            self.assertEqual(r["field_disposition"], tcc.D_FORMAL_IDENTIFIER_POLICY_REQUIRED)
            self.assertTrue(r["requires_client_input"])
            self.assertTrue(r["requires_formal_identifier_policy"])
            self.assertTrue(r["blocking_for_validation"])
            self.assertTrue(r["blocking_for_projection"])
            self.assertEqual(r["owner"], tcc.OWN_CLIENT)

    def test_formal_identifier_source_approved_allows_source(self):
        # With an explicit approval policy, the ordinary source column IS accepted.
        cfgs = _real_configs()
        asset_cfg = dict(cfgs["asset_cfg"])
        rp = dict(asset_cfg.get("reporting_policy", {}))
        rp["formal_identifier_source_approved"] = ["unique_identifier"]
        asset_cfg["reporting_policy"] = rp
        cov = {"RREL1": {"esma_code": "RREL1", "coverage_status": "source_mapped",
                         "selected_source_confidence": 0.95}}
        rows = _checklist(asset_cfg=asset_cfg, coverage_by_code=cov)
        r = {x["esma_code"]: x for x in rows}["RREL1"]
        self.assertEqual(r["field_disposition"], tcc.D_SOURCE_SUPPLIED)

    def test_nd_allowed_is_not_nd_selected(self):
        # RREC9 / RREL27 permit ND but, with no source and no selected policy in
        # this fixture, must NOT be completed-by-ND — they need an asset/client
        # policy decision (ND allowed != ND selected).
        for code in ("RREC9", "RREL27"):
            r = self.by[code]
            self.assertTrue(r["nd_allowed"])              # ND permitted
            self.assertNotEqual(r["field_disposition"], tcc.D_ND_POLICY_SELECTED)
            self.assertEqual(r["field_disposition"], tcc.D_ASSET_POLICY_REQUIRED)
            self.assertTrue(r["requires_asset_policy"])


# --------------------------------------------------------------------------- #
# Asset-specificity: policy vs no-policy, ERM vs non-ERM
# --------------------------------------------------------------------------- #
class TestAssetSpecificity(unittest.TestCase):
    def test_rrel40_disposition_is_config_driven_not_engine_hardcoded(self):
        # ERM: the ASSET policy selects the ND (disposition_source = asset_config).
        erm = {x["esma_code"]: x for x in _checklist()}["RREL40"]
        self.assertEqual(erm["field_disposition"], tcc.D_ND_POLICY_SELECTED)
        self.assertEqual(erm["disposition_source"], "asset_config")
        # Bare asset config: it falls back to the REGIME-configured default (ND5
        # lives in annex2_delivery_rules, not in engine code) — different source.
        bare = {x["esma_code"]: x for x in _checklist(
            asset_cfg={"defaults": {}, "nd_defaults": {}},
            asset_class="residential_mortgage")}["RREL40"]
        self.assertEqual(bare["disposition_source"], "regime_config")

    def test_field_without_any_selection_needs_asset_policy(self):
        # RREL27 permits ND but has no regime default and no asset/client policy
        # here -> needs an asset/client policy decision, never silently ND.
        rows = _checklist(asset_cfg={"defaults": {}, "nd_defaults": {}},
                          asset_class="residential_mortgage")
        r = {x["esma_code"]: x for x in rows}["RREL27"]
        self.assertEqual(r["field_disposition"], tcc.D_ASSET_POLICY_REQUIRED)
        self.assertTrue(r["requires_config"])
        self.assertTrue(r["requires_asset_policy"])

    def test_rrel24_not_generically_nd5_for_non_erm(self):
        rows = _checklist(asset_cfg={"defaults": {}, "nd_defaults": {}},
                          asset_class="residential_mortgage")
        r = {x["esma_code"]: x for x in rows}["RREL24"]
        self.assertNotEqual(r["configured_default_value"], "ND5")
        self.assertNotEqual(r["field_disposition"], tcc.D_ND_POLICY_SELECTED)

    def test_rrel24_asset_config_nd5_beats_pending_regime_rule(self):
        # ERM maturity_date has an explicit asset-config ND5; it must win over a
        # pending_regime_rule coverage status (the bug: projection_rule_required).
        cov = {"RREL24": {"esma_code": "RREL24", "coverage_status": "pending_regime_rule"}}
        r = {x["esma_code"]: x for x in _checklist(coverage_by_code=cov)}["RREL24"]
        self.assertEqual(r["field_disposition"], tcc.D_ND_POLICY_SELECTED)
        self.assertEqual(r["disposition_source"], "asset_config")
        self.assertEqual(r["configured_default_value"], "ND5")
        self.assertFalse(r["requires_projection_rule"])
        self.assertFalse(r["blocking_for_projection"])

    def test_generic_regime_default_does_not_override_source_ambiguity(self):
        # RREC17 has a regime ND1 default AND ambiguous valuation sources. The
        # ambiguous source must win (operator review), never blind ND1.
        cov = {"RREC17": {"esma_code": "RREC17",
                          "coverage_status": "source_mapped_with_alternatives",
                          "selected_source_column": "Original Valuation",
                          "requires_user_decision": True}}
        r = {x["esma_code"]: x for x in _checklist(coverage_by_code=cov)}["RREC17"]
        self.assertEqual(r["field_disposition"], tcc.D_OPERATOR_REVIEW_REQUIRED)
        self.assertEqual(r["disposition_source"], "source_ambiguity")
        self.assertTrue(r["requires_operator_review"])
        self.assertNotEqual(r["configured_default_value"], "ND1")

    def test_explicit_asset_policy_overrides_source_ambiguity(self):
        # If the asset/client policy explicitly says the field is unavailable and
        # should be ND, that deliberate policy DOES override an ambiguous source.
        cfgs = _real_configs()
        asset_cfg = dict(cfgs["asset_cfg"])
        rp = dict(asset_cfg.get("reporting_policy", {}))
        rp["nd_policy"] = dict(rp.get("nd_policy", {}))
        rp["nd_policy"]["original_valuation_amount"] = "ND1"
        asset_cfg["reporting_policy"] = rp
        cov = {"RREC17": {"esma_code": "RREC17",
                          "coverage_status": "source_mapped_with_alternatives",
                          "requires_user_decision": True}}
        r = {x["esma_code"]: x for x in _checklist(
            asset_cfg=asset_cfg, coverage_by_code=cov)}["RREC17"]
        self.assertEqual(r["field_disposition"], tcc.D_ND_POLICY_SELECTED)
        self.assertEqual(r["disposition_source"], "asset_config")

    def test_rrel40_not_operator_review_when_policy_says_not_captured(self):
        # Even if the coverage flags an ambiguity, a deliberate asset ND policy
        # (DTI not captured) must win — never an operator mapping mystery.
        cov = {"RREL40": {"esma_code": "RREL40", "coverage_status": "defaulted_ND",
                          "selected_value_sample": "ND5", "asset_default_value": "ND1",
                          "requires_user_decision": True, "blocking": True}}
        r = {x["esma_code"]: x for x in _checklist(coverage_by_code=cov)}["RREL40"]
        self.assertEqual(r["field_disposition"], tcc.D_ND_POLICY_SELECTED)
        self.assertFalse(r["requires_operator_review"])
        self.assertFalse(r["blocking_for_projection"])

    def test_rrel40_traditional_uses_source_not_nd5(self):
        # A traditional lender that captures DTI -> use the source, not ND5.
        cov = {"RREL40": {"esma_code": "RREL40", "coverage_status": "source_mapped",
                          "selected_source_confidence": 0.95,
                          "selected_source_column": "DTI"}}
        rows = _checklist(asset_cfg={"defaults": {}, "nd_defaults": {}},
                          asset_class="residential_mortgage", coverage_by_code=cov)
        r = {x["esma_code"]: x for x in rows}["RREL40"]
        self.assertEqual(r["field_disposition"], tcc.D_SOURCE_SUPPLIED)
        self.assertNotEqual(r["configured_default_value"], "ND5")

    def test_source_candidate_alone_is_not_source_supplied(self):
        # An enum field mapped to a source column, with no confirmation that the
        # ESMA enum mapping is complete, is NOT source_supplied.
        cov = {"RREL27": {"esma_code": "RREL27", "coverage_status": "source_mapped",
                          "selected_source_confidence": 0.9}}  # no enum_coverage_status
        r = {x["esma_code"]: x for x in _checklist(coverage_by_code=cov)}["RREL27"]
        self.assertNotEqual(r["field_disposition"], tcc.D_SOURCE_SUPPLIED)
        self.assertEqual(r["field_disposition"], tcc.D_CONFIG_MAPPING_REQUIRED)
        self.assertTrue(r["requires_enum_mapping"])

    def test_client_policy_overrides_to_nd_policy(self):
        # A client policy can select an allowed ND even when the asset config is bare.
        rows = _checklist(
            asset_cfg={"defaults": {}, "nd_defaults": {}},
            asset_class="residential_mortgage",
            client_policy={"reporting_policy": {"nd_policy": {"debt_to_income_ratio": "ND5"}}})
        r = {x["esma_code"]: x for x in rows}["RREL40"]
        self.assertEqual(r["field_disposition"], tcc.D_ND_POLICY_SELECTED)
        self.assertEqual(r["disposition_source"], "client_policy")


# --------------------------------------------------------------------------- #
# Source / enum signals from the coverage matrix
# --------------------------------------------------------------------------- #
class TestSourceSignals(unittest.TestCase):
    def test_source_supplied_when_confidently_mapped(self):
        cov = {"RREC9": {"esma_code": "RREC9", "coverage_status": "source_mapped",
                         "selected_source_confidence": 0.97, "requires_user_decision": False,
                         "enum_coverage_status": "complete"}}
        r = {x["esma_code"]: x for x in _checklist(coverage_by_code=cov)}["RREC9"]
        self.assertEqual(r["field_disposition"], tcc.D_SOURCE_SUPPLIED)

    def test_config_mapping_required_when_enum_incomplete(self):
        cov = {"RREL27": {"esma_code": "RREL27", "coverage_status": "source_mapped",
                          "selected_source_confidence": 0.95, "requires_user_decision": False,
                          "enum_coverage_status": "incomplete"}}
        r = {x["esma_code"]: x for x in _checklist(coverage_by_code=cov)}["RREL27"]
        self.assertEqual(r["field_disposition"], tcc.D_CONFIG_MAPPING_REQUIRED)
        self.assertTrue(r["requires_config"])
        self.assertFalse(r["blocking_for_validation"])  # config blocker, not data failure

    def test_operator_review_when_multiple_candidates(self):
        # multiple plausible source columns -> source_mapped_alt -> operator review.
        cov = {"RREC9": {"esma_code": "RREC9", "coverage_status": "source_mapped_alt",
                         "selected_source_confidence": 0.6, "requires_user_decision": True}}
        r = {x["esma_code"]: x for x in _checklist(coverage_by_code=cov)}["RREC9"]
        self.assertEqual(r["field_disposition"], tcc.D_OPERATOR_REVIEW_REQUIRED)
        self.assertTrue(r["requires_operator_review"])

    def test_source_mapped_but_flagged_for_confirmation(self):
        cov = {"RREC9": {"esma_code": "RREC9", "coverage_status": "source_mapped",
                         "selected_source_confidence": 0.8, "requires_user_decision": True}}
        r = {x["esma_code"]: x for x in _checklist(coverage_by_code=cov)}["RREC9"]
        self.assertEqual(r["field_disposition"], tcc.D_SOURCE_MAPPED_REVIEW)


# --------------------------------------------------------------------------- #
# Review bench
# --------------------------------------------------------------------------- #
class TestReviewBench(unittest.TestCase):
    def test_unresolved_rows_surface_to_bench(self):
        rows = _checklist()
        bench = tcc.build_review_bench(rows)
        bench_codes = {b["esma_code"] for b in bench}
        # client onboarding identifiers must be on the bench as client input.
        self.assertIn("RREL1", bench_codes)
        cats = {b["esma_code"]: b["review_category"] for b in bench}
        self.assertEqual(cats["RREL1"], tcc.RB_CLIENT_INPUT)
        # completed dispositions never appear on the bench.
        completed = {r["esma_code"] for r in rows
                     if r["field_disposition"] in tcc._COMPLETED}
        self.assertFalse(completed & bench_codes)

    def test_bench_distinguishes_categories(self):
        rows = _checklist()
        bench = tcc.build_review_bench(rows)
        cats = {b["review_category"] for b in bench}
        # at least client input + a policy/config gap category present.
        self.assertIn(tcc.RB_CLIENT_INPUT, cats)
        self.assertTrue({tcc.RB_ASSET_POLICY, tcc.RB_CONFIG, tcc.RB_OPERATOR} & cats)


# --------------------------------------------------------------------------- #
# Artefact writing
# --------------------------------------------------------------------------- #
class TestArtefacts(unittest.TestCase):
    def test_writes_29_and_29a(self):
        rows = _checklist()
        bench = tcc.build_review_bench(rows)
        out = Path(tempfile.mkdtemp(prefix="tcc_"))
        paths = tcc.write_checklist_artefacts(
            out, rows, bench, contract_id="ESMA_Annex2",
            client_id="client_001", run_id="run_test")
        for name in ("29_target_contract_completion_checklist.csv",
                     "29_target_contract_completion_checklist.json",
                     "29_target_contract_completion_checklist.md",
                     "29a_target_contract_review_bench.csv"):
            self.assertTrue((out / name).exists(), name)
        with open(out / "29_target_contract_completion_checklist.csv", newline="",
                  encoding="utf-8") as fh:
            header = next(csv.reader(fh))
        for col in ("esma_code", "canonical_field", "field_disposition",
                    "requires_client_input", "blocking_for_projection",
                    "nd_allowed", "owner"):
            self.assertIn(col, header)
        self.assertGreater(paths["summary"]["target_field_count"], 100)

    def test_no_xml_or_delivery_artefacts(self):
        rows = _checklist()
        out = Path(tempfile.mkdtemp(prefix="tcc_noxml_"))
        tcc.write_checklist_artefacts(out, rows, tcc.build_review_bench(rows))
        self.assertEqual(list(out.rglob("*.xml")), [])
        self.assertFalse((out / "delivery").exists())
        self.assertFalse((out / "xml").exists())


# --------------------------------------------------------------------------- #
# Downstream consumption (execute the disposition)
# --------------------------------------------------------------------------- #
class TestDispositionExecution(unittest.TestCase):
    def test_transformation_action_mapping(self):
        self.assertEqual(
            tcc.transformation_action_for_disposition(tcc.D_ND_POLICY_SELECTED),
            "materialise_selected_nd")
        self.assertEqual(
            tcc.transformation_action_for_disposition(tcc.D_ASSET_DEFAULT),
            "materialise_asset_default")
        self.assertEqual(
            tcc.transformation_action_for_disposition(tcc.D_CLIENT_ONBOARDING_REQUIRED),
            "carry_forward_client_input_required")

    def test_validation_classification_mapping(self):
        self.assertEqual(
            tcc.validation_classification_for_disposition(tcc.D_CLIENT_ONBOARDING_REQUIRED),
            "client_onboarding_required")
        self.assertEqual(
            tcc.validation_classification_for_disposition(tcc.D_CONFIG_MAPPING_REQUIRED),
            "config_required")  # config blocker, not data failure
        self.assertEqual(
            tcc.validation_classification_for_disposition(tcc.D_ND_POLICY_SELECTED),
            "validation_pass")

    def test_projection_status_mapping(self):
        self.assertEqual(
            tcc.projection_status_for_disposition(tcc.D_SOURCE_SUPPLIED),
            "projected_from_transformed")
        self.assertEqual(
            tcc.projection_status_for_disposition(tcc.D_ND_POLICY_SELECTED),
            "projected_nd_default")
        self.assertEqual(
            tcc.projection_status_for_disposition(tcc.D_CLIENT_ONBOARDING_REQUIRED),
            "blocked_client_onboarding_dependency")


# --------------------------------------------------------------------------- #
# Handoff integration
# --------------------------------------------------------------------------- #
class TestHandoffIntegration(unittest.TestCase):
    def test_field_contract_columns_include_disposition(self):
        for col in ("field_disposition", "disposition_source", "requires_client_input",
                    "requires_operator_review", "requires_config",
                    "requires_projection_rule", "blocking_for_validation",
                    "blocking_for_projection"):
            self.assertIn(col, oh._FIELD_CONTRACT_COLUMNS)

    def test_build_and_apply_disposition_to_contract(self):
        project_dir = Path(tempfile.mkdtemp(prefix="tcc_handoff_"))
        coverage_rows = [
            {"esma_code": "RREC9", "target_field": "RREC9", "canonical_field": "property_type",
             "coverage_status": "source_mapped", "selected_source_confidence": 0.97,
             "enum_coverage_status": "complete"},
        ]
        completion = oh._build_completion_checklist(
            project_dir, coverage_rows, contract_id="ESMA_Annex2",
            client_id="client_001", run_id="run_test", registry=REGISTRY,
            regime_config_path=REGIME, asset_config_path=ASSET)
        # 29 + 29a written under the project dir.
        self.assertTrue((project_dir / "29_target_contract_completion_checklist.csv").exists())
        self.assertTrue((project_dir / "29a_target_contract_review_bench.csv").exists())
        disp = completion["disposition_by_code"]
        self.assertEqual(disp["RREL1"]["field_disposition"],
                         tcc.D_FORMAL_IDENTIFIER_POLICY_REQUIRED)
        self.assertEqual(disp["RREC9"]["field_disposition"], tcc.D_SOURCE_SUPPLIED)

        # the disposition columns are carried onto a handoff field-contract row.
        contract = [{"esma_code": "RREL1", "target_field": "RREL1",
                     "canonical_field": "unique_identifier"}]
        oh._apply_dispositions_to_contract(contract, disp)
        self.assertEqual(contract[0]["field_disposition"],
                         tcc.D_FORMAL_IDENTIFIER_POLICY_REQUIRED)
        self.assertTrue(contract[0]["requires_client_input"])

    def test_build_is_robust_to_bad_config(self):
        project_dir = Path(tempfile.mkdtemp(prefix="tcc_robust_"))
        completion = oh._build_completion_checklist(
            project_dir, [], contract_id="ESMA_Annex2", client_id="c", run_id="r",
            registry="/no/such/registry.yaml", regime_config_path="/no/such/regime.yaml",
            asset_config_path="/no/such/asset.yaml")
        # never raises; returns an (empty) result.
        self.assertIn("disposition_by_code", completion)


# --------------------------------------------------------------------------- #
# Projection executes the onboarding disposition end-to-end
# --------------------------------------------------------------------------- #
class TestProjectionExecutesDisposition(unittest.TestCase):
    def test_projection_carries_client_onboarding_dependency(self):
        # Reuse the projection test's validation-package builder, then add a
        # disposition column to the transformation field contract (32).
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        import test_projection_agent_workflow as tp
        from engine.projection_agent import projection_agent as pa

        root = Path(tempfile.mkdtemp(prefix="tcc_proj_"))
        mpath = tp._write_validation_package(root)
        tx = root / "output" / "transformation" / "32_transformation_field_contract.csv"
        with open(tx, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=[
                "esma_code", "target_field", "canonical_field", "field_disposition"])
            w.writeheader()
            w.writerow({"esma_code": "RREL2", "target_field": "RREL2",
                        "canonical_field": "original_underlying_exposure_identifier",
                        "field_disposition": "client_onboarding_required"})

        result = pa.build_projection_package(mpath)
        out = Path(result["projection_dir"])
        resolution = json.loads(
            (out / "56_projection_blocker_resolution.json").read_text())["rows"]
        issues = json.loads((out / "55_projection_issues.json").read_text())["rows"]

        r = next(x for x in resolution if x["validation_issue_id"] == "VAL-0005")
        self.assertEqual(r["onboarding_disposition"], "client_onboarding_required")
        self.assertEqual(r["projection_status"], pa.ST_BLOCKED_CLIENT)
        self.assertTrue(r["remaining_issue_id"])
        iss = next(i for i in issues if i["issue_id"] == r["remaining_issue_id"])
        self.assertEqual(iss["issue_type"], pa.IT_CLIENT_ONBOARDING)
        self.assertEqual(iss["downstream_owner"], "client_onboarding")
        # still no XML / XML readiness.
        self.assertFalse(result["manifest"]["ready_for_xml_delivery"])

    def test_config_mapping_disposition_is_config_not_source_gap(self):
        # A config_mapping_required field must NOT be reported as an
        # unresolved_source_mapping — that was the contradictory output.
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        import test_projection_agent_workflow as tp
        from engine.projection_agent import projection_agent as pa

        root = Path(tempfile.mkdtemp(prefix="tcc_cfg_"))
        mpath = tp._write_validation_package(root)
        tx = root / "output" / "transformation" / "32_transformation_field_contract.csv"
        with open(tx, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=[
                "esma_code", "canonical_field", "field_disposition"])
            w.writeheader()
            w.writerow({"esma_code": "RREC7", "canonical_field": "occupancy",
                        "field_disposition": "config_mapping_required"})
        result = pa.build_projection_package(mpath)
        out = Path(result["projection_dir"])
        resolution = json.loads(
            (out / "56_projection_blocker_resolution.json").read_text())["rows"]
        r = next(x for x in resolution if x["canonical_field"] == "occupancy")
        self.assertEqual(r["onboarding_disposition"], "config_mapping_required")
        self.assertEqual(r["projection_status"], pa.ST_BLOCKED_OP_CONFIG)
        self.assertNotEqual(r["projection_status"], pa.ST_UNRESOLVED_SOURCE)

    def test_operator_review_not_auto_resolved_by_nd_default(self):
        # A field flagged operator_review by onboarding must NOT be resolved by
        # applying a blind ND/default (the RREC17 risk) — it stays an operator
        # dependency. VAL-0002 (primary_income / RREL16) would normally resolve
        # via ND; the operator_review disposition must prevent that.
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        import test_projection_agent_workflow as tp
        from engine.projection_agent import projection_agent as pa

        root = Path(tempfile.mkdtemp(prefix="tcc_oprev_"))
        mpath = tp._write_validation_package(root)
        tx = root / "output" / "transformation" / "32_transformation_field_contract.csv"
        with open(tx, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=[
                "esma_code", "canonical_field", "field_disposition"])
            w.writeheader()
            w.writerow({"esma_code": "RREL16", "canonical_field": "primary_income",
                        "field_disposition": "operator_review_required"})
        result = pa.build_projection_package(mpath)
        out = Path(result["projection_dir"])
        resolution = json.loads(
            (out / "56_projection_blocker_resolution.json").read_text())["rows"]
        r = next(x for x in resolution if x["validation_issue_id"] == "VAL-0002")
        self.assertEqual(r["onboarding_disposition"], "operator_review_required")
        self.assertFalse(r["resolved"])  # NOT auto-resolved via ND
        self.assertEqual(r["projection_status"], pa.ST_BLOCKED_OP_CONFIG)

    def test_target_frame_not_nd_defaulted_under_operator_review(self):
        # The RREC17 bug: the blocker resolution correctly carried the field as an
        # operator dependency, but the projected TARGET FRAME still filled ND1.
        # The frame must now be blank/blocked, aligned with 56_resolution.
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        import test_projection_agent_workflow as tp
        from engine.projection_agent import projection_agent as pa

        root = Path(tempfile.mkdtemp(prefix="tcc_frame_"))
        mpath = tp._write_validation_package(root)
        tx = root / "output" / "transformation" / "32_transformation_field_contract.csv"
        with open(tx, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=["esma_code", "canonical_field", "field_disposition"])
            w.writeheader()
            # primary_income (RREL16) has an asset ND default that WOULD be filled.
            w.writerow({"esma_code": "RREL16", "canonical_field": "primary_income",
                        "field_disposition": "operator_review_required"})
        result = pa.build_projection_package(mpath)
        out = Path(result["projection_dir"])
        frame = json.loads(
            (out / "51_projected_annex2_target_frame.json").read_text())["rows"]
        cells = [c for c in frame if c["esma_code"] == "RREL16"]
        self.assertTrue(cells)
        for c in cells:
            self.assertEqual(c["projected_value"], "")          # NOT ND1
            self.assertFalse(c["nd_applied"])
            self.assertFalse(c["default_applied"])
            self.assertEqual(c["projection_status"], pa.ST_BLOCKED_OP_CONFIG)
        # frame status aligns with the blocker resolution.
        resolution = json.loads(
            (out / "56_projection_blocker_resolution.json").read_text())["rows"]
        r = next(x for x in resolution if x["validation_issue_id"] == "VAL-0002")
        self.assertEqual(r["projection_status"], cells[0]["projection_status"])
        # conservative readiness preserved.
        self.assertFalse(result["manifest"]["ready_for_delivery_normalisation"])
        self.assertFalse(result["manifest"]["ready_for_xml_delivery"])

    def test_frame_default_still_applied_without_suppressing_disposition(self):
        # Sanity: a field WITHOUT a suppressing disposition still gets its ND/default.
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        import test_projection_agent_workflow as tp
        from engine.projection_agent import projection_agent as pa

        root = Path(tempfile.mkdtemp(prefix="tcc_frame_ok_"))
        mpath = tp._write_validation_package(root)
        result = pa.build_projection_package(mpath)  # no disposition on 32
        out = Path(result["projection_dir"])
        frame = json.loads(
            (out / "51_projected_annex2_target_frame.json").read_text())["rows"]
        cells = [c for c in frame if c["esma_code"] == "RREL16"]
        self.assertTrue(any(c["projected_value"] == "ND1" and c["nd_applied"]
                            for c in cells))


if __name__ == "__main__":
    unittest.main()
