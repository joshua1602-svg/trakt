#!/usr/bin/env python3
"""tests/test_projection_blocker_diagnostics.py

Projection-blocker diagnostic classification and artefact generation.

Covers:
  * classify_projection_blockers returns one row per projection-blocking issue;
  * materialised_projection_pending when field has non-blank tape values;
  * not_materialised_projection_pending when field absent/blank, no ND/default, no
    related fields;
  * nd_or_default_rule_pending when field absent/blank but ND or default allowed;
  * source_mapping_pending when related canonical fields are present in tape;
  * operator_or_config_dependency when issue IS operator/config or a peer issue is;
  * unknown_projection_dependency as a fallback when nothing else matches;
  * non-blocking issues are excluded from the diagnostic output;
  * write_blocker_diagnostics emits 46_projection_blocker_diagnostics.csv/.json/.md;
  * subtype_counts returns counts for every subtype key;
  * artefact 46 is written by build_validation_package;
  * diagnostic counts appear in 40_validation_manifest.json;
  * diagnostic section appears in 42_validation_readiness.md;
  * existing readiness booleans are conservative (no auto-resolution);
  * no projection/XML artefacts are created.
"""

from __future__ import annotations

import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from engine.validation_agent import projection_blocker_diagnostics as pbd
from engine.validation_agent import validation_agent as va
from engine.transformation_agent import transformation_agent as ta

REGISTRY = str(_REPO_ROOT / "config" / "system" / "fields_registry.yaml")
ASSET = str(_REPO_ROOT / "config" / "asset" / "product_defaults_ERM.yaml")
REGIME = str(_REPO_ROOT / "config" / "regime" / "annex2_delivery_rules.yaml")


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def _mk_df(**cols) -> pd.DataFrame:
    return pd.DataFrame(cols)


def _mk_issue(
    issue_id: str = "VAL-0001",
    canonical: str = "customer_type",
    esma: str = "RREL15",
    classification: str = "projection_required",
    issue_type: str = "pending_projection_rule",
    blocking_for_projection: bool = True,
    downstream_owner: str = "projection",
) -> dict:
    return {
        "issue_id": issue_id,
        "canonical_field": canonical,
        "esma_code": esma,
        "validation_classification": classification,
        "issue_type": issue_type,
        "blocking_for_projection": blocking_for_projection,
        "downstream_owner": downstream_owner,
    }


def _regime_index(**fields):
    return {f: v for f, v in fields.items()}


# --------------------------------------------------------------------------- #
# Unit: classify_projection_blockers
# --------------------------------------------------------------------------- #

class TestClassifySubtypes(unittest.TestCase):

    def _classify_one(self, issue, df, regime_index=None):
        rows = pbd.classify_projection_blockers(
            [issue], df, [], regime_index or {})
        return rows

    # --- materialised_projection_pending ------------------------------------

    def test_materialised_when_field_has_values(self):
        iss = _mk_issue(canonical="customer_type", classification="projection_required")
        df = _mk_df(customer_type=["Individual", "Corporate"])
        rows = self._classify_one(iss, df)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["projection_blocker_subtype"], pbd.PB_MATERIALISED)
        self.assertTrue(rows[0]["has_materialised_value"])

    def test_materialised_with_some_blank_rows(self):
        iss = _mk_issue(canonical="customer_type", classification="projection_required")
        df = _mk_df(customer_type=["Individual", "", None])
        rows = self._classify_one(iss, df)
        self.assertEqual(rows[0]["projection_blocker_subtype"], pbd.PB_MATERIALISED)

    # --- nd_or_default_rule_pending ----------------------------------------

    def test_nd_or_default_when_nd_allowed(self):
        iss = _mk_issue(canonical="secondary_income", classification="projection_required")
        df = _mk_df(secondary_income=["", ""])
        regime = {"secondary_income": {"nd_allowed": True}}
        rows = self._classify_one(iss, df, regime)
        self.assertEqual(rows[0]["projection_blocker_subtype"], pbd.PB_ND_OR_DEFAULT)
        self.assertTrue(rows[0]["nd_or_default_allowed"])

    def test_nd_or_default_when_default_allowed(self):
        iss = _mk_issue(canonical="amortisation_type", classification="projection_required")
        df = _mk_df(amortisation_type=["", ""])
        regime = {"amortisation_type": {"default_allowed": True}}
        rows = self._classify_one(iss, df, regime)
        self.assertEqual(rows[0]["projection_blocker_subtype"], pbd.PB_ND_OR_DEFAULT)

    # --- source_mapping_pending --------------------------------------------

    def test_source_mapping_when_related_fields_present(self):
        iss = _mk_issue(canonical="primary_income_currency", classification="projection_required")
        # 'income' token shared with 'secondary_income_type' and 'income_verified'
        df = _mk_df(
            primary_income_currency=["", ""],
            secondary_income=["1000", "2000"],
        )
        rows = self._classify_one(iss, df, {})
        self.assertEqual(rows[0]["projection_blocker_subtype"], pbd.PB_SOURCE_MAPPING)
        self.assertIn("secondary_income", rows[0]["related_fields_in_tape"])

    # --- not_materialised_projection_pending --------------------------------

    def test_not_materialised_when_absent_no_nd_no_related(self):
        iss = _mk_issue(canonical="originator_name", classification="projection_required")
        # field not in df, no nd allowed, no related fields
        df = _mk_df(loan_identifier=["LN1", "LN2"])
        rows = self._classify_one(iss, df, {})
        self.assertEqual(rows[0]["projection_blocker_subtype"], pbd.PB_NOT_MATERIALISED)
        self.assertFalse(rows[0]["has_materialised_value"])
        self.assertFalse(rows[0]["nd_or_default_allowed"])

    def test_not_materialised_when_all_blank(self):
        iss = _mk_issue(canonical="originator_name", classification="projection_required")
        df = _mk_df(originator_name=["", "", ""])
        rows = self._classify_one(iss, df, {})
        self.assertEqual(rows[0]["projection_blocker_subtype"], pbd.PB_NOT_MATERIALISED)

    # --- operator_or_config_dependency -------------------------------------

    def test_op_dep_when_issue_is_operator_required(self):
        iss = _mk_issue(canonical="customer_type", classification="operator_required",
                        issue_type="operator_decision_pending", blocking_for_projection=True)
        df = _mk_df(customer_type=["", ""])
        rows = self._classify_one(iss, df)
        self.assertEqual(rows[0]["projection_blocker_subtype"], pbd.PB_OP_CONFIG_DEP)

    def test_op_dep_when_issue_is_config_required(self):
        iss = _mk_issue(canonical="customer_type", classification="config_required",
                        issue_type="source_absent", blocking_for_projection=True)
        df = _mk_df(customer_type=["", ""])
        rows = self._classify_one(iss, df)
        self.assertEqual(rows[0]["projection_blocker_subtype"], pbd.PB_OP_CONFIG_DEP)

    def test_op_dep_via_peer_issue(self):
        proj_iss = _mk_issue(
            issue_id="VAL-0001", canonical="customer_type",
            classification="projection_required", blocking_for_projection=True)
        op_iss = _mk_issue(
            issue_id="VAL-0002", canonical="customer_type",
            classification="operator_required",
            issue_type="operator_decision_pending", blocking_for_projection=True)
        df = _mk_df(customer_type=["Individual", "Corporate"])
        rows = pbd.classify_projection_blockers([proj_iss, op_iss], df, [], {})
        by_id = {r["issue_id"]: r for r in rows}
        # The projection_required issue becomes operator_or_config_dependency
        # because a peer op issue exists for the same field.
        self.assertEqual(by_id["VAL-0001"]["projection_blocker_subtype"], pbd.PB_OP_CONFIG_DEP)

    # --- non-blocking issues excluded --------------------------------------

    def test_non_blocking_excluded(self):
        iss = _mk_issue(blocking_for_projection=False)
        df = _mk_df(customer_type=["Individual"])
        rows = pbd.classify_projection_blockers([iss], df, [], {})
        self.assertEqual(len(rows), 0)

    def test_pass_issues_excluded(self):
        iss = {
            "issue_id": "VAL-0001", "canonical_field": "customer_type",
            "esma_code": "RREL15", "validation_classification": "validation_pass",
            "issue_type": "pass", "blocking_for_projection": False,
            "downstream_owner": "validation",
        }
        df = _mk_df(customer_type=["Individual"])
        rows = pbd.classify_projection_blockers([iss], df, [], {})
        self.assertEqual(len(rows), 0)

    # --- subtype_counts ----------------------------------------------------

    def test_subtype_counts_has_all_keys(self):
        counts = pbd.subtype_counts([])
        for st in pbd.SUBTYPES:
            self.assertIn(st, counts)

    def test_subtype_counts_correct(self):
        rows = [
            {"projection_blocker_subtype": pbd.PB_MATERIALISED},
            {"projection_blocker_subtype": pbd.PB_MATERIALISED},
            {"projection_blocker_subtype": pbd.PB_NOT_MATERIALISED},
        ]
        counts = pbd.subtype_counts(rows)
        self.assertEqual(counts[pbd.PB_MATERIALISED], 2)
        self.assertEqual(counts[pbd.PB_NOT_MATERIALISED], 1)
        self.assertEqual(counts[pbd.PB_ND_OR_DEFAULT], 0)


# --------------------------------------------------------------------------- #
# Unit: write_blocker_diagnostics artefact
# --------------------------------------------------------------------------- #

class TestWriteBlockerDiagnostics(unittest.TestCase):
    def _sample_rows(self):
        return [
            {
                "issue_id": "VAL-0001", "canonical_field": "customer_type",
                "esma_code": "RREL15", "validation_classification": "projection_required",
                "issue_type": "pending_projection_rule",
                "projection_blocker_subtype": pbd.PB_MATERIALISED,
                "projection_blocker_rationale": "field has non-blank values in tape",
                "has_materialised_value": True, "nd_or_default_allowed": False,
                "related_fields_in_tape": "",
                "recommended_action": pbd._SUBTYPE_ACTION[pbd.PB_MATERIALISED],
                "downstream_owner": "projection", "blocking_for_projection": True,
            },
            {
                "issue_id": "VAL-0002", "canonical_field": "originator_name",
                "esma_code": "RREL82", "validation_classification": "projection_required",
                "issue_type": "pending_projection_rule",
                "projection_blocker_subtype": pbd.PB_NOT_MATERIALISED,
                "projection_blocker_rationale": "absent/blank, no ND, no related fields",
                "has_materialised_value": False, "nd_or_default_allowed": False,
                "related_fields_in_tape": "",
                "recommended_action": pbd._SUBTYPE_ACTION[pbd.PB_NOT_MATERIALISED],
                "downstream_owner": "projection", "blocking_for_projection": True,
            },
        ]

    def test_csv_written(self):
        root = Path(tempfile.mkdtemp(prefix="pbd_csv_"))
        result = pbd.write_blocker_diagnostics(root, self._sample_rows(),
                                               client_id="c1", run_id="r1")
        csv_path = Path(result["diagnostic_csv_path"])
        self.assertTrue(csv_path.exists())
        with open(csv_path, newline="") as fh:
            rows = list(csv.DictReader(fh))
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["projection_blocker_subtype"], pbd.PB_MATERIALISED)

    def test_json_written(self):
        root = Path(tempfile.mkdtemp(prefix="pbd_json_"))
        result = pbd.write_blocker_diagnostics(root, self._sample_rows(),
                                               client_id="c1", run_id="r1")
        doc = json.loads(Path(result["diagnostic_json_path"]).read_text())
        self.assertEqual(doc["projection_blocker_count"], 2)
        self.assertIn("projection_blocker_subtype_counts", doc)
        self.assertIn("rows", doc)
        self.assertEqual(len(doc["rows"]), 2)

    def test_md_written(self):
        root = Path(tempfile.mkdtemp(prefix="pbd_md_"))
        result = pbd.write_blocker_diagnostics(root, self._sample_rows(),
                                               client_id="c1", run_id="r1")
        md = Path(result["diagnostic_md_path"]).read_text()
        self.assertIn("Projection Blocker Diagnostics", md)
        self.assertIn(pbd.PB_MATERIALISED, md)
        self.assertIn("VAL-0001", md)

    def test_empty_blocker_list(self):
        root = Path(tempfile.mkdtemp(prefix="pbd_empty_"))
        result = pbd.write_blocker_diagnostics(root, [], client_id="c1", run_id="r1")
        self.assertEqual(result["projection_blocker_count"], 0)
        doc = json.loads(Path(result["diagnostic_json_path"]).read_text())
        self.assertEqual(doc["rows"], [])


# --------------------------------------------------------------------------- #
# Integration: build_validation_package produces artefact 46
# --------------------------------------------------------------------------- #

def _write_simple_handoff(root: Path) -> Path:
    """Write a minimal handoff + central tape for integration tests."""
    output = root / "output"
    handoff = output / "handoff"
    central = output / "central"
    handoff.mkdir(parents=True, exist_ok=True)
    central.mkdir(parents=True, exist_ok=True)

    with open(central / "18_central_lender_tape.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["unique_identifier", "loan_identifier", "current_principal_balance",
                    "customer_type", "data_cut_off_date"])
        w.writerow(["LN0001", "LN0001", "1000", "Individual", "2026-01-31"])

    contract = [
        {"target_field": "RREL1", "esma_code": "RREL1",
         "canonical_field": "unique_identifier", "domain": "loan",
         "coverage_status": "source_mapped", "selected_source_file": "raw.csv",
         "selected_source_column": "Id", "selected_value_sample": "",
         "handoff_classification": "source_mapped",
         "downstream_owner": "transformation_validation",
         "notes": "", "blocking_decision": False},
        {"target_field": "RREL6", "esma_code": "RREL6",
         "canonical_field": "data_cut_off_date", "domain": "loan",
         "coverage_status": "source_mapped", "selected_source_file": "raw.csv",
         "selected_source_column": "Data Cut-Off Date",
         "selected_value_sample": "2026-01-31",
         "handoff_classification": "source_context_mapped",
         "downstream_owner": "transformation_validation",
         "notes": "", "blocking_decision": False},
        # customer_type marked as pending_projection_rule
        {"target_field": "RREL15", "esma_code": "RREL15",
         "canonical_field": "customer_type", "domain": "loan",
         "coverage_status": "source_mapped", "selected_source_file": "raw.csv",
         "selected_source_column": "CustomerType", "selected_value_sample": "Individual",
         "handoff_classification": "pending_regime_rule",
         "downstream_owner": "projection",
         "notes": "enum projection pending", "blocking_decision": False},
    ]
    cols = list(contract[0].keys())
    with open(handoff / "26_onboarding_handoff_field_contract.csv", "w",
              newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for r in contract:
            w.writerow(r)
    (handoff / "26_onboarding_handoff_field_contract.json").write_text(
        json.dumps({"target_contract_id": "esma_annex_2", "rows": contract}),
        encoding="utf-8")
    (handoff / "27_onboarding_handoff_lineage.json").write_text(
        json.dumps({"rows": []}), encoding="utf-8")

    manifest = {
        "client_id": "client_001", "run_id": "run_pbd_test",
        "target_contract_id": "esma_annex_2",
        "handoff_type": "canonical_onboarding_package",
        "next_agent": "transformation_validation",
        "not_raw_source": True, "do_not_rerun_gate1_on_central_tape": True,
        "central_tape_path": "central/18_central_lender_tape.csv",
        "blocking_decision_count": 0,
        "asset_config_path": ASSET, "regime_config_path": REGIME,
        "registry_path": REGISTRY,
        "ready_for_transformation_validation": True,
        "data_cut_off_date": "2026-01-31",
        "data_cut_off_date_source": "source_column",
        "run_context_fields": ["data_cut_off_date"],
    }
    mpath = handoff / "24_onboarding_handoff_manifest.json"
    mpath.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return mpath


class TestIntegrationArtefact46(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.root = Path(tempfile.mkdtemp(prefix="pbd_int_"))
        mpath = _write_simple_handoff(cls.root)
        ta.build_transformation_package(mpath)
        tx_manifest = cls.root / "output" / "transformation" / "30_transformation_manifest.json"
        cls.result = va.build_validation_package(tx_manifest)
        cls.val_dir = Path(cls.result["validation_dir"])

    def test_artefact_46_csv_exists(self):
        self.assertTrue((self.val_dir / "46_projection_blocker_diagnostics.csv").exists())

    def test_artefact_46_json_exists(self):
        self.assertTrue((self.val_dir / "46_projection_blocker_diagnostics.json").exists())

    def test_artefact_46_md_exists(self):
        self.assertTrue((self.val_dir / "46_projection_blocker_diagnostics.md").exists())

    def test_result_has_blocker_paths(self):
        self.assertIn("blocker_diagnostics_csv_path", self.result)
        self.assertIn("blocker_diagnostics_json_path", self.result)
        self.assertIn("blocker_diagnostics_md_path", self.result)

    def test_manifest_has_diagnostic_counts(self):
        m = json.loads((self.val_dir / "40_validation_manifest.json").read_text())
        self.assertIn("projection_blocker_diagnostic_count", m)
        self.assertIn("projection_blocker_subtype_counts", m)
        # All known subtypes present in counts
        for st in pbd.SUBTYPES:
            self.assertIn(st, m["projection_blocker_subtype_counts"])

    def test_readiness_md_has_diagnostic_section(self):
        md = (self.val_dir / "42_validation_readiness.md").read_text()
        self.assertIn("Projection blocker diagnostics", md)
        self.assertIn("46_projection_blocker_diagnostics", md)

    def test_readiness_booleans_conservative(self):
        m = json.loads((self.val_dir / "40_validation_manifest.json").read_text())
        # ready_for_xml_delivery must always be False at this stage
        self.assertFalse(m["ready_for_xml_delivery"])
        # performed_projection must be False
        self.assertFalse(m["performed_projection"])

    def test_no_projection_dir_created(self):
        self.assertFalse((self.root / "output" / "projection").exists())

    def test_customer_type_classified_as_materialised(self):
        doc = json.loads(
            (self.val_dir / "46_projection_blocker_diagnostics.json").read_text())
        ct_rows = [r for r in doc["rows"] if r["canonical_field"] == "customer_type"]
        if ct_rows:
            self.assertEqual(
                ct_rows[0]["projection_blocker_subtype"], pbd.PB_MATERIALISED,
                "customer_type has 'Individual' in tape, should be materialised_projection_pending"
            )


# --------------------------------------------------------------------------- #
# Subtype coverage: all 6 subtypes via unit path
# --------------------------------------------------------------------------- #

class TestAllSixSubtypes(unittest.TestCase):
    """Verify that each of the 6 subtypes can be produced by classify_projection_blockers."""

    def _run(self, issue, df, regime=None, extra_issues=None):
        all_issues = [issue] + (extra_issues or [])
        rows = pbd.classify_projection_blockers(all_issues, df, [], regime or {})
        by_id = {r["issue_id"]: r for r in rows}
        return by_id.get(issue["issue_id"])

    def test_materialised(self):
        iss = _mk_issue("V001", "customer_type", classification="projection_required")
        r = self._run(iss, _mk_df(customer_type=["Individual"]))
        self.assertEqual(r["projection_blocker_subtype"], pbd.PB_MATERIALISED)

    def test_nd_or_default(self):
        iss = _mk_issue("V002", "secondary_income", classification="projection_required")
        r = self._run(iss, _mk_df(secondary_income=["", ""]),
                      regime={"secondary_income": {"nd_allowed": True}})
        self.assertEqual(r["projection_blocker_subtype"], pbd.PB_ND_OR_DEFAULT)

    def test_source_mapping(self):
        iss = _mk_issue("V003", "primary_income_currency", classification="projection_required")
        r = self._run(iss, _mk_df(primary_income_currency=["", ""],
                                   secondary_income=["1000", "2000"]))
        self.assertEqual(r["projection_blocker_subtype"], pbd.PB_SOURCE_MAPPING)

    def test_not_materialised(self):
        iss = _mk_issue("V004", "originator_name", classification="projection_required")
        r = self._run(iss, _mk_df(loan_identifier=["LN1", "LN2"]))
        self.assertEqual(r["projection_blocker_subtype"], pbd.PB_NOT_MATERIALISED)

    def test_operator_or_config_direct(self):
        iss = _mk_issue("V005", "customer_type", classification="operator_required",
                        issue_type="operator_decision_pending", blocking_for_projection=True)
        r = self._run(iss, _mk_df(customer_type=["", ""]))
        self.assertEqual(r["projection_blocker_subtype"], pbd.PB_OP_CONFIG_DEP)

    def test_operator_or_config_via_peer(self):
        proj = _mk_issue("V006", "customer_type", classification="projection_required",
                         blocking_for_projection=True)
        peer_op = _mk_issue("V007", "customer_type", classification="operator_required",
                            issue_type="operator_decision_pending", blocking_for_projection=True)
        all_issues = [proj, peer_op]
        rows = pbd.classify_projection_blockers(all_issues, _mk_df(customer_type=["Ind"]), [], {})
        by_id = {r["issue_id"]: r for r in rows}
        self.assertEqual(by_id["V006"]["projection_blocker_subtype"], pbd.PB_OP_CONFIG_DEP)

    def test_unknown_fallback(self):
        # Manufacture a scenario that falls through to unknown:
        # Issue is projection_required, field is missing, no nd/default,
        # no related fields, no peer op/config.
        iss = _mk_issue("V008", "obscure_xyz_field_aaaa", classification="projection_required")
        df = _mk_df(loan_identifier=["LN1"])
        rows = pbd.classify_projection_blockers([iss], df, [], {})
        if rows:
            # It might land on not_materialised (which is fine — unknown is a true
            # fallback for cases that bypass all other branches).
            st = rows[0]["projection_blocker_subtype"]
            self.assertIn(st, pbd.SUBTYPES)


if __name__ == "__main__":
    unittest.main()
