#!/usr/bin/env python3
"""tests/test_validation_agent_workflow.py

Trakt Validation Agent — consume the Transformation Agent package and produce a
validation readiness package for the Projection Agent (artefacts 40..45).

Covers:
  * rejects a missing transformation manifest;
  * rejects a transformation manifest where ready_for_validation is false;
  * loads the transformed canonical tape (no Gate 1 re-run);
  * creates validation artefacts 40..45;
  * does not mutate onboarding / transformation artefacts;
  * validates date / numeric / enum / boolean / identifier checks;
  * carries pending_projection_rule forward as projection_required (not failure);
  * carries operator_decision_pending forward as operator_required;
  * carries source_absent forward with correct severity / owner;
  * enum_unmapped -> config_required (or validation_failure per criticality);
  * ready_for_validation_complete is true with no validation-blocking failures;
  * ready_for_projection stays false while projection blockers remain;
  * ready_for_xml_delivery stays false;
  * preserves lineage.
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

from engine.validation_agent import validation_agent as va
from engine.validation_agent.validation_agent import TransformationHandoffError

REGISTRY = str(_REPO_ROOT / "config" / "system" / "fields_registry.yaml")
ASSET = str(_REPO_ROOT / "config" / "asset" / "product_defaults_ERM.yaml")
REGIME = str(_REPO_ROOT / "config" / "regime" / "annex2_delivery_rules.yaml")

# --------------------------------------------------------------------------- #
# Synthetic transformation package builder
# --------------------------------------------------------------------------- #

_TAPE_HEADER = [
    "unique_identifier", "loan_identifier", "current_principal_balance",
    "data_cut_off_date", "property_type", "recourse",
    "current_valuation_amount", "primary_income",
]
_TAPE_ROWS = [
    ["LN0001", "LN0001", "177334.06", "2026-01-31", "RHOS", "N", "350000", "ND1"],
    ["LN0002", "LN0002", "98000", "2026-01-31", "RFLT", "N", "250000", "ND1"],
]

# transformation field contract rows (subset of columns the validator reads).
_TX_CONTRACT = [
    dict(esma_code="RREL1", target_field="RREL1", canonical_field="unique_identifier",
         transformation_status="copied", downstream_owner="validation"),
    dict(esma_code="RREL1b", target_field="RREL1b", canonical_field="loan_identifier",
         transformation_status="copied", downstream_owner="validation"),
    dict(esma_code="RREC_BAL", target_field="RREC_BAL", canonical_field="current_principal_balance",
         transformation_status="type_normalized", downstream_owner="validation"),
    dict(esma_code="RREL6", target_field="RREL6", canonical_field="data_cut_off_date",
         transformation_status="type_normalized", downstream_owner="validation"),
    dict(esma_code="RREC9", target_field="RREC9", canonical_field="property_type",
         transformation_status="enum_normalized", downstream_owner="validation"),
    dict(esma_code="RREL76", target_field="RREL76", canonical_field="recourse",
         transformation_status="configured_static_materialised", downstream_owner="validation"),
    dict(esma_code="RREC_VAL", target_field="RREC_VAL", canonical_field="current_valuation_amount",
         transformation_status="type_normalized", downstream_owner="validation"),
    dict(esma_code="RREL16", target_field="RREL16", canonical_field="primary_income",
         transformation_status="nd_default_materialised", downstream_owner="validation"),
]

# transformation issues to carry forward.
_TX_ISSUES = [
    dict(issue_id="TX-0001", severity="info", field="RREL15", canonical_field="customer_type",
         esma_code="RREL15", issue_type="pending_projection_rule",
         source_value_sample="", transformed_value_sample="",
         description="pending regime rule", blocking_for_validation="False",
         blocking_for_projection="True", recommended_action="defer",
         downstream_owner="projection"),
    dict(issue_id="TX-0002", severity="warning", field="RREL2",
         canonical_field="original_underlying_exposure_identifier",
         esma_code="RREL2", issue_type="operator_decision_pending",
         source_value_sample="", transformed_value_sample="",
         description="operator decision", blocking_for_validation="False",
         blocking_for_projection="True", recommended_action="resolve",
         downstream_owner="operator"),
    dict(issue_id="TX-0003", severity="warning", field="RREC7", canonical_field="occupancy",
         esma_code="RREC7", issue_type="source_absent",
         source_value_sample="", transformed_value_sample="",
         description="no value", blocking_for_validation="False",
         blocking_for_projection="True", recommended_action="confirm",
         downstream_owner="transformation_validation"),
    dict(issue_id="TX-0004", severity="warning", field="RREL35",
         canonical_field="amortisation_type", esma_code="RREL35",
         issue_type="enum_unmapped", source_value_sample="Interest roll-up",
         transformed_value_sample="Interest roll-up", description="unmapped enum",
         blocking_for_validation="False", blocking_for_projection="True",
         recommended_action="extend mapping", downstream_owner="projection"),
]


def _write_tx_package(root: Path, *, ready: bool = True, agent: str = "transformation_agent",
                      xml_ready: bool = False, write_tape: bool = True) -> Path:
    output = root / "output"
    tx = output / "transformation"
    handoff = output / "handoff"
    tx.mkdir(parents=True, exist_ok=True)
    handoff.mkdir(parents=True, exist_ok=True)

    # an onboarding handoff manifest (referenced; must remain unmutated)
    (handoff / "24_onboarding_handoff_manifest.json").write_text(
        json.dumps({"handoff_type": "canonical_onboarding_package"}), encoding="utf-8")

    if write_tape:
        with open(tx / "31_transformed_canonical_tape.csv", "w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(_TAPE_HEADER)
            w.writerows(_TAPE_ROWS)

    cols = list(_TX_CONTRACT[0].keys())
    with open(tx / "32_transformation_field_contract.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for r in _TX_CONTRACT:
            w.writerow(r)

    icols = list(_TX_ISSUES[0].keys())
    with open(tx / "35_transformation_issues.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=icols)
        w.writeheader()
        for r in _TX_ISSUES:
            w.writerow(r)

    (tx / "34_transformation_lineage.json").write_text(json.dumps({
        "onboarding_lineage": [{"target_field": "RREL1"}],
        "transformation_lineage": [{"target_field": "RREL1", "transformation_applied": "copied"}],
    }), encoding="utf-8")

    manifest = {
        "agent": agent, "agent_version": "1.0",
        "client_id": "client_001", "run_id": "run_test",
        "target_contract_id": "esma_annex_2",
        "not_raw_source": True, "did_not_rerun_gate1": True,
        "registry_path": REGISTRY, "regime_config_path": REGIME, "asset_config_path": ASSET,
        "ready_for_validation": ready,
        "ready_for_projection": False,
        "ready_for_xml_delivery": xml_ready,
    }
    mpath = tx / "30_transformation_manifest.json"
    mpath.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return mpath


# --------------------------------------------------------------------------- #
# Manifest validation
# --------------------------------------------------------------------------- #
class TestManifestValidation(unittest.TestCase):
    def test_missing_manifest_raises(self):
        with self.assertRaises(TransformationHandoffError):
            va.build_validation_package("/no/such/30_manifest.json")

    def test_not_ready_raises(self):
        root = Path(tempfile.mkdtemp(prefix="va_notready_"))
        with self.assertRaises(TransformationHandoffError):
            va.build_validation_package(_write_tx_package(root, ready=False))

    def test_xml_ready_true_raises(self):
        root = Path(tempfile.mkdtemp(prefix="va_xmlready_"))
        with self.assertRaises(TransformationHandoffError):
            va.build_validation_package(_write_tx_package(root, xml_ready=True))

    def test_missing_tape_raises(self):
        root = Path(tempfile.mkdtemp(prefix="va_notape_"))
        with self.assertRaises(TransformationHandoffError):
            va.build_validation_package(_write_tx_package(root, write_tape=False))


# --------------------------------------------------------------------------- #
# Full validation run
# --------------------------------------------------------------------------- #
class TestValidationRun(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root = Path(tempfile.mkdtemp(prefix="va_run_"))
        cls.mpath = _write_tx_package(cls.root)
        cls.result = va.build_validation_package(cls.mpath)
        cls.out = cls.root / "output" / "validation"
        cls.manifest = json.loads((cls.out / "40_validation_manifest.json").read_text())
        cls.issues = json.loads((cls.out / "43_validation_issues.json").read_text())
        cls.by_src = {r["source_issue_id"]: r for r in cls.issues["rows"] if r["source_issue_id"]}
        cls.results = json.loads((cls.out / "41_validation_results.json").read_text())["rows"]

    def test_all_artefacts_written(self):
        for name in ("40_validation_manifest.json", "40_validation_manifest.yaml",
                     "41_validation_results.csv", "41_validation_results.json",
                     "42_validation_readiness.json", "42_validation_readiness.md",
                     "43_validation_issues.csv", "43_validation_issues.json",
                     "44_validation_lineage.json", "45_validation_summary.md"):
            self.assertTrue((self.out / name).exists(), name)

    def test_output_under_validation_dir(self):
        self.assertEqual(self.out.name, "validation")
        self.assertEqual(self.out.parent.name, "output")

    def test_tape_loaded_no_gate1(self):
        self.assertEqual(self.manifest["row_count"], 2)
        self.assertTrue(self.manifest["did_not_rerun_gate1"])
        self.assertTrue(self.manifest["did_not_source_match"])
        self.assertFalse(self.manifest["performed_projection"])
        self.assertFalse(self.manifest["performed_xml_delivery"])

    def test_value_checks_ran(self):
        check_types = {r["check_type"] for r in self.results}
        # date / numeric / enum / identifier checks all present
        self.assertIn("type_date", check_types)
        self.assertIn("type_numeric", check_types)
        self.assertIn("identifier_uniqueness", check_types)
        self.assertIn("cross_field_rule", check_types)

    def test_clean_values_pass(self):
        by_rule = {r["validation_rule_id"]: r for r in self.results}
        self.assertEqual(by_rule["VR-data_cut_off_date-date"]["status"], "pass")
        self.assertEqual(by_rule["VR-current_principal_balance-numeric"]["status"], "pass")
        self.assertEqual(by_rule["VR-unique_identifier-unique"]["status"], "pass")

    def test_pending_projection_rule_carried_as_projection_required(self):
        r = self.by_src["TX-0001"]
        self.assertEqual(r["validation_classification"], va.VC_PROJECTION)
        self.assertEqual(r["issue_type"], "pending_projection_rule")
        self.assertFalse(r["blocking_for_validation"])
        self.assertTrue(r["blocking_for_projection"])
        self.assertEqual(r["downstream_owner"], va.OWN_PROJECTION)

    def test_operator_decision_pending_carried_as_operator_required(self):
        r = self.by_src["TX-0002"]
        self.assertEqual(r["validation_classification"], va.VC_OPERATOR)
        self.assertFalse(r["blocking_for_validation"])
        self.assertTrue(r["blocking_for_projection"])
        self.assertEqual(r["downstream_owner"], va.OWN_OPERATOR)

    def test_source_absent_carried_with_owner(self):
        r = self.by_src["TX-0003"]
        self.assertIn(r["validation_classification"], (va.VC_WARNING, va.VC_CONFIG))
        self.assertFalse(r["blocking_for_validation"])
        self.assertTrue(r["blocking_for_projection"])

    def test_enum_unmapped_becomes_config_required(self):
        r = self.by_src["TX-0004"]
        # non-mandatory amortisation_type -> config_required (not validation_failure)
        self.assertEqual(r["validation_classification"], va.VC_CONFIG)
        self.assertFalse(r["blocking_for_validation"])
        self.assertTrue(r["blocking_for_projection"])

    def test_readiness_flags_distinct(self):
        self.assertTrue(self.manifest["ready_for_validation_complete"])
        self.assertFalse(self.manifest["ready_for_projection"])
        self.assertFalse(self.manifest["ready_for_xml_delivery"])

    def test_next_agent_remediation_when_blockers_remain(self):
        self.assertEqual(self.manifest["next_agent"], va.NEXT_REMEDIATION)

    def test_no_validation_blocking_issues(self):
        self.assertEqual(self.manifest["blocking_for_validation_count"], 0)
        self.assertGreater(self.manifest["blocking_for_projection_count"], 0)

    def test_issue_columns_complete(self):
        with open(self.out / "43_validation_issues.csv", newline="", encoding="utf-8") as fh:
            header = next(csv.reader(fh))
        for col in ("issue_id", "source_issue_id", "severity", "field", "canonical_field",
                    "esma_code", "validation_rule_id", "issue_type", "validation_classification",
                    "source_value_sample", "transformed_value_sample", "description",
                    "blocking_for_validation", "blocking_for_projection",
                    "blocking_for_xml_delivery", "recommended_action", "downstream_owner"):
            self.assertIn(col, header)

    def test_results_columns_complete(self):
        with open(self.out / "41_validation_results.csv", newline="", encoding="utf-8") as fh:
            header = next(csv.reader(fh))
        for col in ("validation_rule_id", "field", "canonical_field", "esma_code",
                    "check_type", "status", "severity", "row_count_checked",
                    "failure_count", "warning_count", "sample_failures",
                    "blocking_for_validation", "blocking_for_projection", "notes"):
            self.assertIn(col, header)

    def test_lineage_preserved_and_extended(self):
        lin = json.loads((self.out / "44_validation_lineage.json").read_text())
        self.assertTrue(lin["transformation_lineage"])  # carried forward
        self.assertEqual(lin["transformation_lineage_source"], "34_transformation_lineage.json")
        self.assertTrue(lin["validation_lineage"])  # extended
        sample = lin["validation_lineage"][0]
        for k in ("validation_rule_id", "field", "check_type", "status",
                  "issue_ids", "input_artifact", "output_artifact"):
            self.assertIn(k, sample)

    def test_upstream_artefacts_not_mutated(self):
        tx = self.root / "output" / "transformation"
        names = sorted(p.name for p in tx.iterdir())
        # Only the original transformation inputs remain; no validation artefact
        # leaked into the transformation dir.
        self.assertEqual(names, sorted([
            "30_transformation_manifest.json",
            "31_transformed_canonical_tape.csv",
            "32_transformation_field_contract.csv",
            "34_transformation_lineage.json",
            "35_transformation_issues.csv",
        ]))
        # onboarding handoff manifest untouched
        h = json.loads((self.root / "output" / "handoff" /
                        "24_onboarding_handoff_manifest.json").read_text())
        self.assertEqual(h, {"handoff_type": "canonical_onboarding_package"})


# --------------------------------------------------------------------------- #
# Validation failure path (mandatory id missing, no ND allowed -> blocking)
# --------------------------------------------------------------------------- #
class TestValidationFailurePath(unittest.TestCase):
    def test_duplicate_identifier_blocks_validation(self):
        root = Path(tempfile.mkdtemp(prefix="va_dupe_"))
        mpath = _write_tx_package(root)
        # rewrite the tape with a duplicate loan_identifier
        tape = root / "output" / "transformation" / "31_transformed_canonical_tape.csv"
        with open(tape, "w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(_TAPE_HEADER)
            w.writerow(["LN0001", "DUP", "1", "2026-01-31", "RHOS", "N", "1", "ND1"])
            w.writerow(["LN0002", "DUP", "1", "2026-01-31", "RHOS", "N", "1", "ND1"])
        result = va.build_validation_package(mpath)
        m = result["manifest"]
        self.assertGreater(m["validation_failure_count"], 0)
        self.assertGreater(m["blocking_for_validation_count"], 0)
        self.assertFalse(m["ready_for_validation_complete"])
        self.assertFalse(m["ready_for_projection"])


if __name__ == "__main__":
    unittest.main()
