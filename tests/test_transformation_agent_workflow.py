#!/usr/bin/env python3
"""tests/test_transformation_agent_workflow.py

Trakt Transformation Agent — consume the Onboarding Agent handoff package and
produce a validation-ready transformed canonical package (artefacts 30..35).

Covers:
  * rejects a missing / invalid handoff manifest;
  * refuses a handoff where ready_for_transformation_validation is false;
  * loads the central canonical tape without re-running Gate 1;
  * creates the output/transformation artefacts (30..35);
  * materialises asset defaults and valid ND defaults;
  * normalizes dates / numeric / rate fields deterministically;
  * applies enum mappings (canonical normalisation, never guessing);
  * preserves onboarding lineage and extends it with transformation lineage;
  * surfaces source_absent rather than failing raw mapping;
  * surfaces semantic_derivation_required for
    current_outstanding_balance -> current_principal_balance;
  * marks pending regime/projection rules as pending_projection_rule, not a
    transformation failure;
  * ready_for_validation and ready_for_projection are distinct;
  * ready_for_xml_delivery remains false.
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

from engine.transformation_agent import transformation_agent as ta
from engine.transformation_agent.transformation_agent import HandoffValidationError

REGISTRY = str(_REPO_ROOT / "config" / "system" / "fields_registry.yaml")
ASSET = str(_REPO_ROOT / "config" / "asset" / "product_defaults_ERM.yaml")
REGIME = str(_REPO_ROOT / "config" / "regime" / "annex2_delivery_rules.yaml")


# --------------------------------------------------------------------------- #
# Synthetic handoff package builder
# --------------------------------------------------------------------------- #

_TAPE_HEADER = [
    "unique_identifier", "current_principal_balance", "current_outstanding_balance",
    "data_cut_off_date", "property_type", "interest_rate_type",
]
_TAPE_ROWS = [
    ["LN0001", "177,334.06", "180000.00", "31/01/2026", "Detached House", "Fixed"],
    ["LN0002", "98,000", "99000", "31/01/2026", "Flat", "Fixed"],
]

_CONTRACT = [
    # source_mapped — identifier (string copy)
    dict(target_field="RREL1", esma_code="RREL1", canonical_field="unique_identifier",
         domain="loan", coverage_status="source_mapped",
         selected_source_file="raw.csv", selected_source_column="Unique Loan Identifier",
         selected_value_sample="", handoff_classification="source_mapped",
         downstream_owner="transformation_validation", notes="", blocking_decision=False),
    # source_mapped — numeric normalisation
    dict(target_field="RREC_BAL", esma_code="RREC_BAL", canonical_field="current_principal_balance",
         domain="loan", coverage_status="source_mapped",
         selected_source_file="raw.csv", selected_source_column="Current Principal Balance",
         selected_value_sample="", handoff_classification="source_mapped",
         downstream_owner="transformation_validation", notes="", blocking_decision=False),
    # source_mapped — date normalisation
    dict(target_field="RREL6", esma_code="RREL6", canonical_field="data_cut_off_date",
         domain="loan", coverage_status="source_mapped",
         selected_source_file="raw.csv", selected_source_column="Data Cut-Off Date",
         selected_value_sample="", handoff_classification="source_mapped",
         downstream_owner="transformation_validation", notes="", blocking_decision=False),
    # source_mapped — enum normalisation (property_type)
    dict(target_field="RREC9", esma_code="RREC9", canonical_field="property_type",
         domain="collateral", coverage_status="source_mapped",
         selected_source_file="raw.csv", selected_source_column="Property Type",
         selected_value_sample="", handoff_classification="source_mapped",
         downstream_owner="transformation_validation", notes="", blocking_decision=False),
    # source_mapped — enum normalisation (interest_rate_type Fixed -> FXRL)
    dict(target_field="RREL42", esma_code="RREL42", canonical_field="interest_rate_type",
         domain="loan", coverage_status="source_mapped",
         selected_source_file="raw.csv", selected_source_column="Interest Rate Type",
         selected_value_sample="", handoff_classification="source_mapped",
         downstream_owner="transformation_validation", notes="", blocking_decision=False),
    # ND default — primary_income ND1
    dict(target_field="RREL16", esma_code="RREL16", canonical_field="primary_income",
         domain="loan", coverage_status="defaulted_ND",
         selected_source_file="", selected_source_column="",
         selected_value_sample="ND1", handoff_classification="nd_default_downstream",
         downstream_owner="transformation_validation", notes="", blocking_decision=False),
    # asset default — exposure_currency_denomination GBP (not in tape)
    dict(target_field="RREL_CCY", esma_code="RREL_CCY",
         canonical_field="exposure_currency_denomination",
         domain="loan", coverage_status="defaulted_value",
         selected_source_file="", selected_source_column="",
         selected_value_sample="", handoff_classification="default_downstream",
         downstream_owner="transformation_validation", notes="", blocking_decision=False),
    # default_downstream — lien 1 (regime default)
    dict(target_field="RREC8", esma_code="RREC8", canonical_field="lien",
         domain="collateral", coverage_status="defaulted_value",
         selected_source_file="", selected_source_column="",
         selected_value_sample="1", handoff_classification="default_downstream",
         downstream_owner="transformation_validation", notes="", blocking_decision=False),
    # configured_static — recourse N
    dict(target_field="RREL76", esma_code="RREL76", canonical_field="recourse",
         domain="loan", coverage_status="configured_static",
         selected_source_file="", selected_source_column="",
         selected_value_sample="N", handoff_classification="configured_static",
         downstream_owner="transformation_validation", notes="", blocking_decision=False),
    # configured_static with NO resolvable value -> source_absent
    dict(target_field="RREC7", esma_code="RREC7", canonical_field="occupancy",
         domain="collateral", coverage_status="configured_static",
         selected_source_file="", selected_source_column="",
         selected_value_sample="", handoff_classification="configured_static",
         downstream_owner="transformation_validation", notes="", blocking_decision=False),
    # pending_regime_rule -> pending_projection_rule
    dict(target_field="RREL15", esma_code="RREL15", canonical_field="customer_type",
         domain="loan", coverage_status="pending_regime_rule",
         selected_source_file="", selected_source_column="",
         selected_value_sample="", handoff_classification="pending_regime_rule",
         downstream_owner="projection", notes="no full regime field yet",
         blocking_decision=False),
    # semantic_derivation_required — outstanding -> principal (must not alias)
    dict(target_field="RREL_PB", esma_code="RREL_PB",
         canonical_field="current_principal_balance",
         domain="loan", coverage_status="source_mapped",
         selected_source_file="raw.csv", selected_source_column="current_outstanding_balance",
         selected_value_sample="", handoff_classification="semantic_derivation_required",
         downstream_owner="transformation_validation", notes="", blocking_decision=False),
    # operator_decision_pending (non-blocking)
    dict(target_field="RREL2", esma_code="RREL2",
         canonical_field="original_underlying_exposure_identifier",
         domain="loan", coverage_status="source_mapped_with_alternatives",
         selected_source_file="raw.csv", selected_source_column="Underlying Exposure Id",
         selected_value_sample="", handoff_classification="operator_decision_pending",
         downstream_owner="operator", notes="2 candidates", blocking_decision=False),
]


def _write_handoff(root: Path, *, ready: bool = True, handoff_type: str = "canonical_onboarding_package",
                   not_raw_source: bool = True, next_agent: str = "transformation_validation",
                   write_central: bool = True) -> Path:
    output = root / "output"
    handoff = output / "handoff"
    central = output / "central"
    handoff.mkdir(parents=True, exist_ok=True)
    central.mkdir(parents=True, exist_ok=True)

    if write_central:
        with open(central / "18_central_lender_tape.csv", "w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(_TAPE_HEADER)
            w.writerows(_TAPE_ROWS)

    # field contract (json + csv)
    (handoff / "26_onboarding_handoff_field_contract.json").write_text(
        json.dumps({"target_contract_id": "esma_annex_2", "rows": _CONTRACT}, indent=2),
        encoding="utf-8")
    cols = list(_CONTRACT[0].keys())
    with open(handoff / "26_onboarding_handoff_field_contract.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for r in _CONTRACT:
            w.writerow(r)

    # lineage
    (handoff / "27_onboarding_handoff_lineage.json").write_text(
        json.dumps({"target_contract_id": "esma_annex_2", "rows": [
            {"target_field": "RREL1", "canonical_field": "unique_identifier",
             "source_column": "Unique Loan Identifier", "classification": "source_mapped"},
        ]}, indent=2), encoding="utf-8")

    manifest = {
        "client_id": "client_001", "run_id": "run_test",
        "target_contract_id": "esma_annex_2",
        "handoff_type": handoff_type,
        "handoff_stage": "post_onboarding_pre_transformation_validation",
        "next_agent": next_agent,
        "not_raw_source": not_raw_source,
        "do_not_rerun_gate1_on_central_tape": True,
        "central_tape_path": "central/18_central_lender_tape.csv",
        "blocking_decision_count": 0,
        "asset_config_path": ASSET,
        "regime_config_path": REGIME,
        "registry_path": REGISTRY,
        "ready_for_transformation_validation": ready,
        "ready_for_projection": False,
        "ready_for_xml_delivery": False,
    }
    mpath = handoff / "24_onboarding_handoff_manifest.json"
    mpath.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return mpath


# --------------------------------------------------------------------------- #
# Manifest validation
# --------------------------------------------------------------------------- #
class TestHandoffValidation(unittest.TestCase):
    def test_missing_manifest_raises(self):
        with self.assertRaises(HandoffValidationError):
            ta.build_transformation_package("/no/such/24_manifest.json")

    def test_invalid_json_raises(self):
        d = Path(tempfile.mkdtemp(prefix="tx_badjson_"))
        bad = d / "24_onboarding_handoff_manifest.json"
        bad.write_text("{not json", encoding="utf-8")
        with self.assertRaises(HandoffValidationError):
            ta.build_transformation_package(bad)

    def test_refuses_not_ready_handoff(self):
        root = Path(tempfile.mkdtemp(prefix="tx_notready_"))
        mpath = _write_handoff(root, ready=False)
        with self.assertRaises(HandoffValidationError):
            ta.build_transformation_package(mpath)

    def test_refuses_wrong_handoff_type(self):
        root = Path(tempfile.mkdtemp(prefix="tx_wrongtype_"))
        mpath = _write_handoff(root, handoff_type="raw_source_pack")
        with self.assertRaises(HandoffValidationError):
            ta.build_transformation_package(mpath)

    def test_refuses_raw_source(self):
        root = Path(tempfile.mkdtemp(prefix="tx_raw_"))
        mpath = _write_handoff(root, not_raw_source=False)
        with self.assertRaises(HandoffValidationError):
            ta.build_transformation_package(mpath)

    def test_refuses_missing_central_tape(self):
        root = Path(tempfile.mkdtemp(prefix="tx_nocentral_"))
        mpath = _write_handoff(root, write_central=False)
        with self.assertRaises(HandoffValidationError):
            ta.build_transformation_package(mpath)


# --------------------------------------------------------------------------- #
# Full transformation run
# --------------------------------------------------------------------------- #
class TestTransformationRun(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root = Path(tempfile.mkdtemp(prefix="tx_run_"))
        cls.mpath = _write_handoff(cls.root)
        cls.result = ta.build_transformation_package(cls.mpath)
        cls.out = cls.root / "output" / "transformation"
        cls.manifest = json.loads((cls.out / "30_transformation_manifest.json").read_text())
        cls.tape = json.loads((cls.out / "31_transformed_canonical_tape.json").read_text())
        cls.contract = json.loads(
            (cls.out / "32_transformation_field_contract.json").read_text())["rows"]
        cls.by_field = {r["target_field"]: r for r in cls.contract}
        cls.issues = json.loads((cls.out / "35_transformation_issues.json").read_text())

    def test_all_artefacts_written(self):
        for name in ("30_transformation_manifest.json", "30_transformation_manifest.yaml",
                     "31_transformed_canonical_tape.csv", "31_transformed_canonical_tape.json",
                     "32_transformation_field_contract.csv", "32_transformation_field_contract.json",
                     "33_transformation_readiness.json", "33_transformation_readiness.md",
                     "34_transformation_lineage.json",
                     "35_transformation_issues.csv", "35_transformation_issues.json"):
            self.assertTrue((self.out / name).exists(), name)

    def test_output_under_transformation_dir(self):
        self.assertEqual(self.out.name, "transformation")
        self.assertEqual(self.out.parent.name, "output")

    def test_central_tape_loaded_not_recanonicalised(self):
        self.assertEqual(self.tape["row_count"], 2)
        self.assertTrue(self.manifest["did_not_rerun_gate1"])
        self.assertTrue(self.manifest["not_raw_source"])
        self.assertTrue(self.manifest["did_not_fuzzy_match_sources"])

    def test_numeric_normalised(self):
        row0 = self.tape["rows"][0]
        # "177,334.06" -> 177334.06
        self.assertEqual(float(row0["current_principal_balance"]), 177334.06)

    def test_date_normalised_deterministically(self):
        row0 = self.tape["rows"][0]
        self.assertEqual(row0["data_cut_off_date"], "2026-01-31")
        self.assertEqual(self.by_field["RREL6"]["transformation_status"], ta.TS_TYPE_NORMALIZED)

    def test_enum_mapping_applied(self):
        row0 = self.tape["rows"][0]
        self.assertEqual(row0["property_type"], "RHOS")
        self.assertEqual(row0["interest_rate_type"], "FXRL")
        self.assertEqual(self.by_field["RREC9"]["transformation_status"], ta.TS_ENUM_NORMALIZED)

    def test_asset_default_materialised(self):
        row0 = self.tape["rows"][0]
        self.assertEqual(row0["exposure_currency_denomination"], "GBP")
        self.assertEqual(self.by_field["RREL_CCY"]["transformation_status"], ta.TS_DEFAULT)
        self.assertEqual(self.by_field["RREL_CCY"]["value_source"], "asset_config_default")

    def test_nd_default_materialised(self):
        row0 = self.tape["rows"][0]
        self.assertEqual(row0["primary_income"], "ND1")
        self.assertEqual(self.by_field["RREL16"]["transformation_status"], ta.TS_ND_DEFAULT)

    def test_configured_static_materialised(self):
        row0 = self.tape["rows"][0]
        self.assertEqual(row0["recourse"], "N")
        self.assertEqual(self.by_field["RREL76"]["transformation_status"],
                         ta.TS_CONFIGURED_STATIC)

    def test_source_absent_surfaced_not_failed(self):
        r = self.by_field["RREC7"]
        self.assertEqual(r["transformation_status"], ta.TS_SOURCE_ABSENT)
        self.assertFalse(r["blocking_for_validation"])
        self.assertTrue(r["blocking_for_projection"])

    def test_semantic_derivation_required_not_aliased(self):
        r = self.by_field["RREL_PB"]
        self.assertEqual(r["transformation_status"], ta.TS_SEMANTIC_DERIVATION)
        self.assertFalse(r["blocking_for_validation"])
        self.assertTrue(r["blocking_for_projection"])

    def test_pending_projection_rule_not_failure(self):
        r = self.by_field["RREL15"]
        self.assertEqual(r["transformation_status"], ta.TS_PENDING_PROJECTION)
        self.assertEqual(r["downstream_owner"], ta.OWN_PROJECTION)
        self.assertFalse(r["blocking_for_validation"])

    def test_operator_decision_pending_carried(self):
        r = self.by_field["RREL2"]
        self.assertEqual(r["transformation_status"], ta.TS_OPERATOR_PENDING)
        self.assertFalse(r["blocking_for_validation"])

    def test_lineage_preserved_and_extended(self):
        lin = json.loads((self.out / "34_transformation_lineage.json").read_text())
        self.assertTrue(lin["onboarding_lineage"])  # carried forward
        self.assertEqual(lin["onboarding_lineage_source"],
                         "27_onboarding_handoff_lineage.json")
        self.assertTrue(lin["transformation_lineage"])  # extended
        sample = lin["transformation_lineage"][0]
        for k in ("transformed_field", "transformation_applied", "default_source",
                  "enum_map_used", "type_cast", "parse_rule", "issue_id"):
            self.assertIn(k, sample)

    def test_readiness_flags_distinct(self):
        self.assertTrue(self.manifest["ready_for_validation"])
        self.assertFalse(self.manifest["ready_for_projection"])
        self.assertFalse(self.manifest["ready_for_xml_delivery"])

    def test_no_projection_or_xml_performed(self):
        self.assertFalse(self.manifest["performed_projection"])
        self.assertFalse(self.manifest["performed_xml_delivery"])

    def test_issue_columns_complete(self):
        with open(self.out / "35_transformation_issues.csv", newline="", encoding="utf-8") as fh:
            header = next(csv.reader(fh))
        for col in ("issue_id", "severity", "field", "canonical_field", "esma_code",
                    "issue_type", "source_value_sample", "transformed_value_sample",
                    "description", "blocking_for_validation", "blocking_for_projection",
                    "recommended_action", "downstream_owner"):
            self.assertIn(col, header)

    def test_onboarding_artefacts_not_mutated(self):
        # The transformation run must only write under output/transformation.
        handoff = self.root / "output" / "handoff"
        names = sorted(p.name for p in handoff.iterdir())
        self.assertEqual(names, [
            "24_onboarding_handoff_manifest.json",
            "26_onboarding_handoff_field_contract.csv",
            "26_onboarding_handoff_field_contract.json",
            "27_onboarding_handoff_lineage.json",
        ])


if __name__ == "__main__":
    unittest.main()
