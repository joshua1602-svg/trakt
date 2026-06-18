#!/usr/bin/env python3
"""tests/test_projection_agent_workflow.py

Trakt Projection Agent v1 — consume the Validation Agent package and produce a
projected Annex 2 target frame package (artefacts 50..56). NOT XML.

Covers:
  * rejects a missing validation manifest;
  * rejects an invalid / non-validation manifest;
  * loads the transformed canonical tape + projection blocker diagnostics;
  * writes projection artefacts 50..56;
  * handles materialised_projection_pending by projecting transformed values;
  * handles nd_or_default_rule_pending using allowed ND/default metadata;
  * does not invent ND/defaults;
  * carries operator/config dependencies forward;
  * carries unresolved source mapping forward unless an explicit rule exists;
  * does not generate XML and does not create output/delivery or output/xml;
  * does not claim XML readiness;
  * preserves lineage;
  * readiness booleans remain conservative (distinct flags);
  * the blocker-resolution report both reduces and carries blockers forward.
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

import yaml

from engine.projection_agent import projection_agent as pa
from engine.projection_agent import gate4_adapter as g4
from engine.projection_agent.projection_agent import ValidationHandoffError

REGISTRY = str(_REPO_ROOT / "config" / "system" / "fields_registry.yaml")
ASSET = str(_REPO_ROOT / "config" / "asset" / "product_defaults_ERM.yaml")
REGIME = str(_REPO_ROOT / "config" / "regime" / "annex2_delivery_rules.yaml")

# --------------------------------------------------------------------------- #
# Synthetic validation package builder
# --------------------------------------------------------------------------- #

# property_type (RREC9) + data_cut_off_date (RREL6) are materialised; primary_income
# is absent (-> asset nd_default ND1); occupancy / original_underlying_exposure are
# absent (-> source-mapping / not-materialised blockers).
_TAPE_HEADER = [
    "unique_identifier", "loan_identifier", "current_principal_balance",
    "data_cut_off_date", "property_type",
]
_TAPE_ROWS = [
    ["LN0001", "LN0001", "177334.06", "2026-01-31", "RHOS"],
    ["LN0002", "LN0002", "98000", "2026-01-31", "RFLT"],
]

# transformation field contract (subset).
_TX_CONTRACT = [
    dict(esma_code="RREL1", target_field="RREL1", canonical_field="unique_identifier",
         transformation_status="copied", downstream_owner="validation"),
    dict(esma_code="RREC9", target_field="RREC9", canonical_field="property_type",
         transformation_status="enum_normalized", downstream_owner="validation"),
]

# 46 projection blocker diagnostics — one per subtype we want to exercise.
_BLOCKER_COLUMNS = [
    "issue_id", "canonical_field", "esma_code", "validation_classification",
    "issue_type", "projection_blocker_subtype", "projection_blocker_rationale",
    "has_materialised_value", "nd_or_default_allowed", "related_fields_in_tape",
    "recommended_action", "downstream_owner", "blocking_for_projection",
]
_BLOCKERS = [
    # materialised → should resolve from the transformed tape.
    dict(issue_id="VAL-0001", canonical_field="property_type", esma_code="RREC9",
         validation_classification="projection_required", issue_type="pending_projection_rule",
         projection_blocker_subtype="materialised_projection_pending",
         projection_blocker_rationale="field has non-blank values",
         has_materialised_value="True", nd_or_default_allowed="True",
         related_fields_in_tape="", recommended_action="implement projection rule",
         downstream_owner="projection", blocking_for_projection="True"),
    # nd/default → should resolve via asset nd_default (ND1) within nd_allowed.
    dict(issue_id="VAL-0002", canonical_field="primary_income", esma_code="RREL16",
         validation_classification="projection_required", issue_type="source_absent",
         projection_blocker_subtype="nd_or_default_rule_pending",
         projection_blocker_rationale="absent but nd/default allowed",
         has_materialised_value="False", nd_or_default_allowed="True",
         related_fields_in_tape="", recommended_action="apply nd/default",
         downstream_owner="config_policy", blocking_for_projection="True"),
    # operator dependency (formal client identifier) → must be carried forward.
    dict(issue_id="VAL-0003", canonical_field="formal_client_identifier", esma_code="",
         validation_classification="operator_required", issue_type="operator_decision_pending",
         projection_blocker_subtype="operator_or_config_dependency",
         projection_blocker_rationale="operator_required issue",
         has_materialised_value="False", nd_or_default_allowed="False",
         related_fields_in_tape="", recommended_action="resolve operator decision",
         downstream_owner="operator", blocking_for_projection="True"),
    # source mapping with no explicit rule → carried forward.
    dict(issue_id="VAL-0004", canonical_field="occupancy", esma_code="RREC7",
         validation_classification="projection_required", issue_type="source_absent",
         projection_blocker_subtype="source_mapping_pending",
         projection_blocker_rationale="related fields present",
         has_materialised_value="False", nd_or_default_allowed="False",
         related_fields_in_tape="property_type", recommended_action="derive",
         downstream_owner="projection", blocking_for_projection="True"),
    # not materialised, no allowed nd/default → carried forward.
    dict(issue_id="VAL-0005", canonical_field="original_underlying_exposure_identifier",
         esma_code="RREL2", validation_classification="projection_required",
         issue_type="source_absent",
         projection_blocker_subtype="not_materialised_projection_pending",
         projection_blocker_rationale="no nd/default, no source",
         has_materialised_value="False", nd_or_default_allowed="False",
         related_fields_in_tape="", recommended_action="supply source",
         downstream_owner="transformation_validation", blocking_for_projection="True"),
]


def _write_validation_package(root: Path, *, agent: str = "validation_agent",
                              xml_ready: bool = False, write_tape: bool = True,
                              consumes_tx: bool = True) -> Path:
    output = root / "output"
    val = output / "validation"
    tx = output / "transformation"
    handoff = output / "handoff"
    for d in (val, tx, handoff):
        d.mkdir(parents=True, exist_ok=True)

    (handoff / "24_onboarding_handoff_manifest.json").write_text(
        json.dumps({"handoff_type": "canonical_onboarding_package",
                    "client_id": "client_001"}), encoding="utf-8")

    if write_tape:
        with open(tx / "31_transformed_canonical_tape.csv", "w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(_TAPE_HEADER)
            w.writerows(_TAPE_ROWS)

    with open(tx / "32_transformation_field_contract.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(_TX_CONTRACT[0].keys()))
        w.writeheader()
        for r in _TX_CONTRACT:
            w.writerow(r)

    (tx / "34_transformation_lineage.json").write_text(json.dumps({
        "onboarding_lineage": [{"target_field": "RREL1"}],
        "transformation_lineage": [{"target_field": "RREC9", "transformation_applied": "enum"}],
    }), encoding="utf-8")

    # 43 validation issues (carried-forward, minimal).
    icols = ["issue_id", "source_issue_id", "severity", "field", "canonical_field",
             "esma_code", "validation_classification", "issue_type",
             "blocking_for_validation", "blocking_for_projection",
             "blocking_for_xml_delivery", "recommended_action", "downstream_owner"]
    with open(val / "43_validation_issues.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=icols)
        w.writeheader()
        for b in _BLOCKERS:
            w.writerow({
                "issue_id": b["issue_id"], "source_issue_id": "", "severity": "warn",
                "field": b["esma_code"], "canonical_field": b["canonical_field"],
                "esma_code": b["esma_code"],
                "validation_classification": b["validation_classification"],
                "issue_type": b["issue_type"], "blocking_for_validation": "False",
                "blocking_for_projection": "True", "blocking_for_xml_delivery": "True",
                "recommended_action": b["recommended_action"],
                "downstream_owner": b["downstream_owner"]})

    # 44 validation lineage.
    (val / "44_validation_lineage.json").write_text(json.dumps({
        "onboarding_lineage": [{"target_field": "RREL1"}],
        "transformation_lineage": [{"target_field": "RREC9"}],
        "validation_lineage": [{"validation_rule_id": "VR-property_type-enum",
                                "field": "property_type", "check_type": "enum",
                                "status": "pass", "issue_ids": []}],
    }), encoding="utf-8")

    # 46 projection blocker diagnostics.
    with open(val / "46_projection_blocker_diagnostics.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=_BLOCKER_COLUMNS)
        w.writeheader()
        for b in _BLOCKERS:
            w.writerow(b)

    # 42 validation readiness (referenced; not strictly required).
    (val / "42_validation_readiness.json").write_text(json.dumps({
        "ready_for_validation_complete": True, "ready_for_projection": False,
        "ready_for_xml_delivery": False}), encoding="utf-8")

    manifest = {
        "agent": agent, "agent_version": "1.0",
        "client_id": "client_001", "run_id": "run_test",
        "target_contract_id": "ESMA_Annex2",
        "consumes_transformation_package": consumes_tx,
        "not_raw_source": True, "did_not_rerun_gate1": True,
        "performed_projection": False, "performed_xml_delivery": False,
        "registry_path": REGISTRY, "regime_config_path": REGIME, "asset_config_path": ASSET,
        "ready_for_validation_complete": True,
        "ready_for_projection": False,
        "ready_for_xml_delivery": xml_ready,
    }
    mpath = val / "40_validation_manifest.json"
    mpath.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return mpath


# --------------------------------------------------------------------------- #
# Manifest validation
# --------------------------------------------------------------------------- #
class TestManifestValidation(unittest.TestCase):
    def test_missing_manifest_raises(self):
        with self.assertRaises(ValidationHandoffError):
            pa.build_projection_package("/no/such/40_validation_manifest.json")

    def test_non_validation_agent_raises(self):
        root = Path(tempfile.mkdtemp(prefix="pa_badagent_"))
        with self.assertRaises(ValidationHandoffError):
            pa.build_projection_package(
                _write_validation_package(root, agent="transformation_agent"))

    def test_xml_ready_true_raises(self):
        root = Path(tempfile.mkdtemp(prefix="pa_xmlready_"))
        with self.assertRaises(ValidationHandoffError):
            pa.build_projection_package(_write_validation_package(root, xml_ready=True))

    def test_missing_tape_raises(self):
        root = Path(tempfile.mkdtemp(prefix="pa_notape_"))
        with self.assertRaises(ValidationHandoffError):
            pa.build_projection_package(_write_validation_package(root, write_tape=False))


# --------------------------------------------------------------------------- #
# Full projection run
# --------------------------------------------------------------------------- #
class TestProjectionRun(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root = Path(tempfile.mkdtemp(prefix="pa_run_"))
        cls.mpath = _write_validation_package(cls.root)
        cls.result = pa.build_projection_package(cls.mpath)
        cls.out = cls.root / "output" / "projection"
        cls.manifest = json.loads((cls.out / "50_projection_manifest.json").read_text())
        cls.frame = json.loads((cls.out / "51_projected_annex2_target_frame.json").read_text())["rows"]
        cls.issues = json.loads((cls.out / "55_projection_issues.json").read_text())["rows"]
        cls.resolution = json.loads((cls.out / "56_projection_blocker_resolution.json").read_text())["rows"]

    def _frame_cells(self, esma_code):
        return [r for r in self.frame if r["esma_code"] == esma_code]

    def _resolution_for(self, val_issue_id):
        return next(r for r in self.resolution if r["validation_issue_id"] == val_issue_id)

    def test_all_artefacts_written(self):
        for name in ("50_projection_manifest.json", "50_projection_manifest.yaml",
                     "51_projected_annex2_target_frame.csv", "51_projected_annex2_target_frame.json",
                     "52_projection_field_contract.csv", "52_projection_field_contract.json",
                     "53_projection_readiness.json", "53_projection_readiness.md",
                     "54_projection_lineage.json",
                     "55_projection_issues.csv", "55_projection_issues.json",
                     "56_projection_blocker_resolution.csv", "56_projection_blocker_resolution.json"):
            self.assertTrue((self.out / name).exists(), name)

    def test_output_under_projection_dir(self):
        self.assertEqual(self.out.name, "projection")
        self.assertEqual(self.out.parent.name, "output")

    def test_tape_loaded(self):
        self.assertEqual(self.manifest["row_count"], 2)
        self.assertTrue(self.manifest["consumes_validation_package"])
        self.assertGreater(self.manifest["frame_row_count"], 0)

    def test_materialised_projected_from_transformed(self):
        cells = self._frame_cells("RREC9")  # property_type
        self.assertTrue(cells)
        self.assertTrue(all(c["projection_status"] == pa.ST_FROM_TRANSFORMED for c in cells))
        self.assertEqual({c["projected_value"] for c in cells}, {"RHOS", "RFLT"})
        self.assertFalse(any(c["blocking_for_delivery"] for c in cells))
        # blocker VAL-0001 resolved from the transformed tape.
        r = self._resolution_for("VAL-0001")
        self.assertTrue(r["resolved"])
        self.assertEqual(r["projection_status"], pa.ST_FROM_TRANSFORMED)
        self.assertEqual(r["resolution_source"], "transformed_tape")

    def test_nd_default_applied_within_allowed(self):
        cells = self._frame_cells("RREL16")  # primary_income -> asset nd_default ND1
        self.assertTrue(cells)
        self.assertTrue(all(c["projected_value"] == "ND1" for c in cells))
        self.assertTrue(all(c["nd_applied"] for c in cells))
        self.assertTrue(all(c["projection_status"] == pa.ST_ND_DEFAULT for c in cells))
        r = self._resolution_for("VAL-0002")
        self.assertTrue(r["resolved"])
        self.assertEqual(r["projection_status"], pa.ST_ND_DEFAULT)

    def test_does_not_invent_nd_or_default(self):
        # occupancy (RREC7): non-mandatory, no configured ND/default -> left blank,
        # never invented.
        cells = self._frame_cells("RREC7")
        self.assertTrue(cells)
        self.assertTrue(all(c["projected_value"] == "" for c in cells))
        self.assertTrue(all(not c["nd_applied"] and not c["default_applied"] for c in cells))

    def test_operator_dependency_carried_forward(self):
        r = self._resolution_for("VAL-0003")
        self.assertFalse(r["resolved"])
        self.assertEqual(r["projection_status"], pa.ST_BLOCKED_OP_CONFIG)
        self.assertTrue(r["remaining_issue_id"])
        iss = next(i for i in self.issues if i["issue_id"] == r["remaining_issue_id"])
        self.assertEqual(iss["issue_type"], pa.IT_OPERATOR)
        self.assertEqual(iss["downstream_owner"], pa.OWN_OPERATOR)
        self.assertTrue(iss["blocking_for_delivery"])

    def test_unresolved_source_mapping_carried_forward(self):
        r = self._resolution_for("VAL-0004")
        self.assertFalse(r["resolved"])
        self.assertEqual(r["projection_status"], pa.ST_UNRESOLVED_SOURCE)
        iss = next(i for i in self.issues if i["issue_id"] == r["remaining_issue_id"])
        self.assertEqual(iss["issue_type"], pa.IT_SOURCE_MAPPING)

    def test_not_materialised_carried_forward(self):
        r = self._resolution_for("VAL-0005")
        self.assertFalse(r["resolved"])
        self.assertEqual(r["projection_status"], pa.ST_UNRESOLVED_NOT_MATERIALISED)

    def test_blocker_report_reduces_and_carries(self):
        resolved = sum(1 for r in self.resolution if r["resolved"])
        remaining = sum(1 for r in self.resolution if not r["resolved"])
        self.assertGreaterEqual(resolved, 2)   # materialised + nd/default
        self.assertGreaterEqual(remaining, 3)  # operator + source-mapping + not-materialised
        self.assertEqual(self.manifest["blockers_resolved_count"], resolved)
        self.assertEqual(self.manifest["remaining_blocker_count"], remaining)

    def test_no_xml_generated(self):
        self.assertFalse(self.manifest["invoked_gate5_xml_builder"])
        self.assertFalse(self.manifest["performed_xml_delivery"])
        # no XML artefacts anywhere under the run output.
        xml_files = list((self.root / "output").rglob("*.xml"))
        self.assertEqual(xml_files, [])

    def test_no_delivery_or_xml_dirs(self):
        self.assertFalse((self.root / "output" / "delivery").exists())
        self.assertFalse((self.root / "output" / "xml").exists())

    def test_does_not_claim_xml_readiness(self):
        self.assertFalse(self.manifest["ready_for_xml_delivery"])

    def test_readiness_flags_conservative_and_distinct(self):
        self.assertTrue(self.manifest["projection_ran"])
        # unresolved operator/source/not-materialised blockers remain.
        self.assertFalse(self.manifest["projection_complete"])
        self.assertFalse(self.manifest["ready_for_delivery_normalisation"])
        self.assertFalse(self.manifest["ready_for_xml_delivery"])

    def test_delivery_deferred_issue_present(self):
        deferred = [i for i in self.issues if i["issue_type"] == pa.IT_DELIVERY_DEFERRED]
        self.assertEqual(len(deferred), 1)
        self.assertFalse(deferred[0]["blocking_for_delivery"])
        self.assertTrue(deferred[0]["blocking_for_xml_delivery"])

    def test_lineage_preserved_and_extended(self):
        lin = json.loads((self.out / "54_projection_lineage.json").read_text())
        self.assertTrue(lin["transformation_lineage"])
        self.assertTrue(lin["validation_lineage"])
        self.assertTrue(lin["projection_lineage"])
        sample = lin["projection_lineage"][0]
        for k in ("esma_code", "canonical_field", "record_group",
                  "field_projection_status", "input_artifact", "output_artifact"):
            self.assertIn(k, sample)

    def test_frame_columns_complete(self):
        with open(self.out / "51_projected_annex2_target_frame.csv", newline="", encoding="utf-8") as fh:
            header = next(csv.reader(fh))
        for col in ("row_id", "loan_identifier", "record_group", "esma_code",
                    "canonical_field", "projected_value", "projected_value_type",
                    "value_source", "projection_status", "nd_applied",
                    "default_applied", "source_field", "source_value_sample",
                    "rule_id", "blocking_for_delivery"):
            self.assertIn(col, header)

    def test_issue_columns_complete(self):
        with open(self.out / "55_projection_issues.csv", newline="", encoding="utf-8") as fh:
            header = next(csv.reader(fh))
        for col in ("issue_id", "source_issue_id", "esma_code", "canonical_field",
                    "record_group", "issue_type", "projection_status", "severity",
                    "blocking_for_delivery", "blocking_for_xml_delivery",
                    "recommended_action", "downstream_owner", "description"):
            self.assertIn(col, header)

    def test_resolution_columns_complete(self):
        with open(self.out / "56_projection_blocker_resolution.csv", newline="", encoding="utf-8") as fh:
            header = next(csv.reader(fh))
        for col in ("validation_issue_id", "esma_code", "canonical_field",
                    "diagnostic_subtype", "projection_action", "projection_status",
                    "resolved", "resolution_source", "projected_value_sample",
                    "remaining_issue_id", "notes"):
            self.assertIn(col, header)

    def test_field_contract_flags_blocking(self):
        contract = json.loads(
            (self.out / "52_projection_field_contract.json").read_text())["rows"]
        by_code = {r["esma_code"]: r for r in contract}
        # RREL2 is mandatory, no ND/default, absent -> unresolved + delivery-blocking.
        self.assertIn("RREL2", by_code)
        self.assertTrue(by_code["RREL2"]["blocking_for_delivery"])
        self.assertEqual(by_code["RREL2"]["field_projection_status"],
                         pa.ST_UNRESOLVED_NOT_MATERIALISED)

    def test_record_group_tagged(self):
        groups = {r["record_group"] for r in self.frame}
        self.assertIn("RREL", groups)
        self.assertIn("RREC", groups)

    def test_upstream_artefacts_not_mutated(self):
        tx = self.root / "output" / "transformation"
        names = sorted(p.name for p in tx.iterdir())
        self.assertEqual(names, sorted([
            "31_transformed_canonical_tape.csv",
            "32_transformation_field_contract.csv",
            "34_transformation_lineage.json"]))
        val = self.root / "output" / "validation"
        # projection wrote nothing into the validation dir.
        self.assertNotIn("50_projection_manifest.json",
                         [p.name for p in val.iterdir()])


class TestRrel35AmortisationEnum(unittest.TestCase):
    """RREL35 amortisation_type enum reconciliation + asset-policy override.

    The mapping is config-driven (regime enum_map + asset reporting_policy), never
    hard-coded in Python.
    """

    @classmethod
    def setUpClass(cls):
        cls.regime = yaml.safe_load(open(REGIME, encoding="utf-8"))
        cls.rule = g4.build_projection_index(cls.regime)["RREL35"]
        cls.asset_cfg = yaml.safe_load(open(ASSET, encoding="utf-8"))
        cls.overrides = g4.load_asset_enum_overrides(cls.asset_cfg)

    def test_authoritative_enum_set_present(self):
        # regime config carries the full authoritative workbook code list.
        values = set((self.regime["field_rules"]["RREL35"]["transform"]["enum_map"]).values())
        self.assertTrue({"FRXX", "DEXX", "FIXE", "BLLT", "OTHR"}.issubset(values))

    def test_workbook_nd_corrected(self):
        # workbook RREL35: ND1-4 allowed, ND5 not. Config must not allow ND5.
        nd = self.regime["field_rules"]["RREL35"]["nd_allowed"]
        self.assertNotIn("ND5", [str(x).upper() for x in nd])

    def test_generic_bullet_maps_to_bllt(self):
        # No asset overrides -> generic Annex meaning Bullet = BLLT.
        cell = pa._project_cell("Bullet", self.rule, {}, {}, {})
        self.assertEqual(cell["projected_value"], "BLLT")
        self.assertEqual(cell["value_source"], "enum_map")
        self.assertFalse(cell["blocking_for_delivery"])

    def test_erm_bullet_maps_to_othr_by_asset_policy(self):
        cell = pa._project_cell(
            "Bullet", self.rule, self.asset_cfg.get("defaults", {}),
            self.asset_cfg.get("nd_defaults", {}), self.overrides)
        self.assertEqual(cell["projected_value"], "OTHR")
        self.assertEqual(cell["value_source"], "asset_policy")
        self.assertFalse(cell["blocking_for_delivery"])

    def test_erm_asset_default_bullet_maps_to_othr(self):
        # blank source -> ERM asset default 'Bullet' is reported as OTHR by policy.
        cell = pa._project_cell(
            "", self.rule, self.asset_cfg.get("defaults", {}),
            self.asset_cfg.get("nd_defaults", {}), self.overrides)
        self.assertEqual(cell["projected_value"], "OTHR")
        self.assertEqual(cell["value_source"], "asset_policy")

    def test_override_is_config_driven_not_hardcoded(self):
        # Removing the configured override must change the result back to the
        # generic BLLT — proving the OTHR decision lives in config, not Python.
        cell = pa._project_cell("Bullet", self.rule, {}, {}, {})
        self.assertEqual(cell["projected_value"], "BLLT")
        # a config without reporting_policy.enum_overrides yields no overrides.
        self.assertEqual(g4.load_asset_enum_overrides({"defaults": {}}), {})
        self.assertEqual(
            g4.apply_asset_enum_override("amortisation_type", "Bullet", {}),
            ("Bullet", False))

    def test_other_labels_map_to_codes(self):
        for label, code in (("French", "FRXX"), ("German", "DEXX"),
                            ("Fixed amortisation schedule", "FIXE"), ("Other", "OTHR")):
            cell = pa._project_cell(label, self.rule, {}, {}, {})
            self.assertEqual(cell["projected_value"], code, label)


if __name__ == "__main__":
    unittest.main()
